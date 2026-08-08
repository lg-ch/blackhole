"""Long-running HTTP REST + /metrics server backed by libmangrove.so FFI.

Endpoints:
  POST /search   { "qvec": [...], "top_n": 500, "query_depth": 0,
                   "where": "primary_cat='cs.LG'" }     -> { "ids": [...] }
  GET  /health   -> { "status": "ok", "n_trees": ..., "n_docs": ... }
  GET  /metrics  -> Prometheus text format (mangrove_queries_total, latency)

Designed for one forest per process. For multi-shard, run one process per
shard and put a router (nginx/envoy) in front, or wire query_multi here.

Run:
  python3 scripts/serve.py --index /mnt/mangrove/indexes/sift100m \
      --n_trees 200 --dim 128 --sub_dim 16 --depth 25 \
      --n_docs 100000000 --gen 3 --port 8000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mangrove_ffi import Forest                                # noqa: E402
from telemetry import (Counter, Gauge, Histogram,              # noqa: E402
                       render_all)

try:
    from clickhouse_driver import Client
except ImportError:
    Client = None


# ---- metrics ----
QUERIES        = Counter('mangrove_queries_total', 'Queries served')
ERRORS         = Counter('mangrove_errors_total',  'Query errors')
LAT_MS         = Histogram('mangrove_query_latency_ms', 'Per-query latency ms',
                           buckets=(0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500))
DISTINCT_TOTAL = Counter('mangrove_n_distinct_total',
                         'Sum of K-way merge n_distinct (candidates visited)')
FOREST_LOADED  = Gauge('mangrove_forest_loaded',
                       '1 if forest is loaded and queryable')


_forest: Forest | None = None
_ch:     'Client | None' = None
_args:   argparse.Namespace | None = None
# Forest has a single io_uring ring + shared scratch buffers — not thread-safe.
# Serialize queries with this lock; ThreadingHTTPServer still handles
# connection accept + filter fetch (CH) concurrently.
_FOREST_LOCK = threading.Lock()


def _fetch_filter(where: str) -> bytes | None:
    if not where or _ch is None:
        return None
    sql = (f"SELECT cast(groupBitmapState(internal_id), 'String') "
           f"FROM mangrove.docs WHERE {where}")
    try:
        rows = _ch.execute(sql)
        if not rows or not rows[0]:
            return None
        return rows[0][0]
    except Exception as e:
        sys.stderr.write(f'[degrade] CH filter fetch failed: {e}\n')
        return None


class _Handler(BaseHTTPRequestHandler):
    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _text(self, code: int, body: str, ctype: str = 'text/plain') -> None:
        b = body.encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):  # noqa: N802
        path = urlparse(self.path).path
        if path == '/health':
            self._json(200, {
                'status':  'ok' if _forest is not None else 'no_forest',
                'n_trees': _forest.n_trees if _forest else 0,
                'n_docs':  _forest.n_docs  if _forest else 0,
                'depth':   _forest.depth   if _forest else 0,
                'dim':     _forest.dim     if _forest else 0,
            })
        elif path == '/metrics':
            self._text(200, render_all(),
                       ctype='text/plain; version=0.0.4')
        else:
            self._json(404, {'error': 'unknown path'})

    def do_POST(self):  # noqa: N802
        if urlparse(self.path).path != '/search':
            self._json(404, {'error': 'unknown path'}); return
        if _forest is None:
            ERRORS.inc()
            self._json(503, {'error': 'forest not loaded'}); return

        try:
            n = int(self.headers.get('Content-Length', '0'))
            body = json.loads(self.rfile.read(n).decode()) if n > 0 else {}
            qvec = np.asarray(body['qvec'], dtype=np.float32)
            top_n = int(body.get('top_n', 500))
            qd    = int(body.get('query_depth', 0))
            where = body.get('where', '') or ''
        except Exception as e:
            ERRORS.inc()
            self._json(400, {'error': f'bad request: {e}'}); return

        allowed = _fetch_filter(where)
        t0 = time.time()
        with _FOREST_LOCK:
            try:
                ids, votes, k = _forest.query(qvec, top_n=top_n, query_depth=qd,
                                              allowed_state=allowed)
            except Exception as e:
                ERRORS.inc()
                self._json(500, {'error': f'query failed: {e}'}); return
            dt_ms = (time.time() - t0) * 1000
            distinct = _forest.n_distinct()  # capture inside lock

        LAT_MS.observe(dt_ms)
        QUERIES.inc()
        DISTINCT_TOTAL.inc(distinct)
        self._json(200, {
            'ids':      ids[:k].tolist(),
            'votes':    votes[:k].tolist(),
            'n':        int(k),
            'latency_ms': round(dt_ms, 3),
            'n_distinct': distinct,
            'filtered': allowed is not None,
        })

    def log_message(self, *a, **kw):
        return


def main() -> None:
    global _forest, _ch, _args
    ap = argparse.ArgumentParser()
    ap.add_argument('--index',   required=True)
    ap.add_argument('--n_trees', type=int, required=True)
    ap.add_argument('--dim',     type=int, required=True)
    ap.add_argument('--sub_dim', type=int, default=0)
    ap.add_argument('--depth',   type=int, required=True)
    ap.add_argument('--n_docs',  type=int, required=True)
    ap.add_argument('--gen',     type=int, default=3)
    ap.add_argument('--ch_host', default='127.0.0.1')
    ap.add_argument('--port',    type=int, default=8000)
    _args = ap.parse_args()

    if Client is not None:
        try:
            _ch = Client(_args.ch_host, connect_timeout=2)
            _ch.execute('SELECT 1')
            sys.stderr.write(f'[serve] CH connected at {_args.ch_host}\n')
        except Exception as e:
            sys.stderr.write(f'[degrade] CH unreachable: {e}\n')
            _ch = None

    _forest = Forest(_args.index, n_trees=_args.n_trees, dim=_args.dim,
                     sub_dim=_args.sub_dim, depth=_args.depth,
                     n_docs=_args.n_docs, gen_version=_args.gen)
    FOREST_LOADED.set(1.0)
    sys.stderr.write(f'[serve] forest loaded from {_args.index}\n')

    srv = ThreadingHTTPServer(('0.0.0.0', _args.port), _Handler)
    sys.stderr.write(f'[serve] listening on :{_args.port}\n')
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        sys.stderr.write('\n[serve] shutting down\n')
    finally:
        FOREST_LOADED.set(0.0)
        _forest.close()


if __name__ == '__main__':
    main()
