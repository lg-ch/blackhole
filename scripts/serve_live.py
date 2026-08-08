"""Long-running HTTP server backed by LiveIndex.

Endpoints :
  POST /search   { "qvec": [...], "top_n": 500 } -> top-k ids
  POST /insert   { "vec": [...] }                 -> { "doc_id": N }
  POST /insert_batch { "vecs": [[...], ...] }     -> { "doc_ids": [...] }
  POST /delete   { "doc_id": N }
  POST /freeze   { }                              -> manually trigger freeze
  GET  /health
  GET  /stats                                     -> manifest stats
  GET  /metrics                                   -> Prometheus

Graceful shutdown on SIGTERM/SIGINT : freezes the pending active buffer
to disk so no docs are lost on restart, then closes Forests cleanly.

Usage :
  python3 scripts/serve_live.py --root /path/to/live_idx \
      [--dim 128 --sub_dim 16 --depth 14 --gen 3 --create]
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from live_index import LiveIndex, Backpressure
from telemetry import Counter, Gauge, Histogram, render_all


# ---- metrics ----
QUERIES        = Counter('mg_queries_total',  'Search queries served')
INSERTS        = Counter('mg_inserts_total',  'Docs inserted')
DELETES        = Counter('mg_deletes_total',  'Docs tombstoned')
FREEZES        = Counter('mg_freezes_total',  'Manual freezes triggered')
ERRORS         = Counter('mg_errors_total',   'Endpoint errors')
BP_REJECTS     = Counter('mg_backpressure_rejects_total',
                         'Inserts rejected due to full active buffer')
LAT_QUERY      = Histogram('mg_query_latency_ms',  'Query latency ms')
LAT_INSERT     = Histogram('mg_insert_latency_ms', 'Insert latency ms')
SEGS_ALIVE     = Gauge('mg_segments_alive', '# of frozen segments alive')
ACTIVE_SIZE    = Gauge('mg_active_buffer_size', '# of docs in active buffer')


_li:   LiveIndex | None = None
_lock = threading.Lock()
# AuthN : if non-empty, every request must carry X-API-Key matching one
# of these. Loaded from env or file at startup. Empty set = auth disabled
# (dev/single-tenant ; document that prod must enable).
_auth_keys: dict[str, str] = {}   # key → label (audit-friendly)
AUTH_REJECTS = Counter('mg_auth_rejects_total',
                       'Requests rejected for missing/invalid API key')


def _graceful_shutdown(signum=None, frame=None):
    global _li
    sys.stderr.write(f'\n[serve_live] graceful shutdown (sig={signum})\n')
    if _li is not None:
        try:
            with _lock:
                # Flush any pending active buffer so it survives restart.
                if _li.active_size() > 0:
                    sys.stderr.write(
                        f'[serve_live] freezing {_li.active_size()} pending docs\n')
                    _li.freeze()
                _li.close()
        except Exception as e:
            sys.stderr.write(f'[serve_live] shutdown error: {e}\n')
        _li = None
    sys.exit(0)


_PUBLIC_PATHS = {'/health', '/metrics'}


class _Handler(BaseHTTPRequestHandler):
    def _authed(self) -> bool:
        """Return True if the request passes AuthN. Health/metrics are
           always public so a Kubernetes liveness probe + Prometheus
           scrape don't need to carry the key. Others require X-API-Key
           when _auth_keys is non-empty.                                  */"""
        if not _auth_keys:
            return True
        path = urlparse(self.path).path
        if path in _PUBLIC_PATHS:
            return True
        key = self.headers.get('X-API-Key', '')
        return key in _auth_keys

    def _reject_auth(self):
        AUTH_REJECTS.inc()
        b = json.dumps({'error': 'unauthorized'}).encode()
        self.send_response(401)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(b)))
        self.send_header('WWW-Authenticate', 'API-Key')
        self.end_headers()
        self.wfile.write(b)

    def _json(self, code, payload):
        b = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _text(self, code, body):
        b = body.encode()
        self.send_response(code)
        self.send_header('Content-Type', 'text/plain')
        self.send_header('Content-Length', str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _body(self):
        n = int(self.headers.get('Content-Length', '0'))
        return json.loads(self.rfile.read(n).decode()) if n > 0 else {}

    def do_GET(self):
        if not self._authed():
            self._reject_auth(); return
        p = urlparse(self.path).path
        if p == '/health':
            self._json(200, {
                'status':       'ok' if _li else 'no_index',
                'next_doc_id':  _li.manifest['next_doc_id'] if _li else 0,
                'active_size':  _li.active_size() if _li else 0,
                'segments':     len(_li.manifest['segments']) if _li else 0,
            })
        elif p == '/stats':
            if _li is None:
                self._json(503, {'error': 'no index'}); return
            self._json(200, {
                'segments':     _li.manifest['segments'],
                'next_doc_id':  _li.manifest['next_doc_id'],
                'active_size':  _li.active_size(),
                'dim':          _li.manifest['dim'],
                'sub_dim':      _li.manifest['sub_dim'],
                'n_trees':      _li.manifest['n_trees'],
            })
        elif p == '/metrics':
            if _li:
                SEGS_ALIVE.set(len(_li.manifest['segments']))
                ACTIVE_SIZE.set(_li.active_size())
            self._text(200, render_all())
        else:
            self._json(404, {'error': 'unknown'})

    def do_POST(self):
        if not self._authed():
            self._reject_auth(); return
        p = urlparse(self.path).path
        if _li is None:
            ERRORS.inc(); self._json(503, {'error': 'no index'}); return
        try:
            body = self._body()
        except Exception as e:
            ERRORS.inc(); self._json(400, {'error': f'bad json: {e}'}); return

        if p == '/search':
            try:
                q = np.asarray(body['qvec'], dtype=np.float32)
                top_n = int(body.get('top_n', 4000))
                top_k = int(body.get('top_k', 10))
                t0 = time.time()
                with _lock:
                    ids = _li.query(q, top_n=top_n, top_k=top_k)
                LAT_QUERY.observe((time.time() - t0) * 1000)
                QUERIES.inc()
                self._json(200, {
                    'ids':        ids.tolist(),
                    'latency_ms': round((time.time() - t0) * 1000, 3),
                })
            except Exception as e:
                ERRORS.inc(); self._json(500, {'error': str(e)})
        elif p == '/insert':
            try:
                v = np.asarray(body['vec'], dtype=np.float32)
                t0 = time.time()
                with _lock:
                    doc_id = _li.insert(v, doc_id=body.get('doc_id'))
                LAT_INSERT.observe((time.time() - t0) * 1000)
                INSERTS.inc()
                self._json(200, {'doc_id': doc_id})
            except Backpressure as e:
                BP_REJECTS.inc()
                self.send_response(503)
                self.send_header('Retry-After', '1')
                self.send_header('Content-Type', 'application/json')
                payload = json.dumps({'error': 'backpressure', 'detail': str(e)}).encode()
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers(); self.wfile.write(payload)
            except Exception as e:
                ERRORS.inc(); self._json(500, {'error': str(e)})
        elif p == '/insert_batch':
            try:
                vecs = body['vecs']
                ids: list[int] = []
                with _lock:
                    for v in vecs:
                        ids.append(_li.insert(np.asarray(v, dtype=np.float32)))
                INSERTS.inc(len(ids))
                self._json(200, {'doc_ids': ids, 'count': len(ids)})
            except Backpressure as e:
                BP_REJECTS.inc()
                # Partial success: report how many made it through.
                self.send_response(503)
                self.send_header('Retry-After', '1')
                self.send_header('Content-Type', 'application/json')
                payload = json.dumps({
                    'error':     'backpressure',
                    'detail':    str(e),
                    'doc_ids':   ids,
                    'completed': len(ids),
                }).encode()
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers(); self.wfile.write(payload)
            except Exception as e:
                ERRORS.inc(); self._json(500, {'error': str(e)})
        elif p == '/delete':
            try:
                doc_id = int(body['doc_id'])
                with _lock:
                    _li.delete(doc_id)
                DELETES.inc()
                self._json(200, {'doc_id': doc_id, 'deleted': True})
            except Exception as e:
                ERRORS.inc(); self._json(500, {'error': str(e)})
        elif p == '/freeze':
            try:
                with _lock:
                    seg = _li.freeze()
                FREEZES.inc()
                self._json(200, {'segment': seg, 'active_after': _li.active_size()})
            except Exception as e:
                ERRORS.inc(); self._json(500, {'error': str(e)})
        else:
            self._json(404, {'error': 'unknown'})

    def log_message(self, *a, **kw):
        return


def _load_auth_keys(env_var: str, key_file: str | None) -> dict[str, str]:
    """Loaded sources, in order : --auth-keys-file (one 'label:key' per
       line) wins, else env MG_API_KEYS=label1:key1,label2:key2.
       Empty result = AuthN disabled.                                   */"""
    keys: dict[str, str] = {}
    if key_file and os.path.exists(key_file):
        with open(key_file) as f:
            for ln in f:
                ln = ln.strip()
                if not ln or ln.startswith('#'): continue
                label, _, k = ln.partition(':')
                if k: keys[k] = label
    else:
        env = os.environ.get(env_var, '')
        for tok in env.split(','):
            tok = tok.strip()
            if not tok: continue
            label, _, k = tok.partition(':')
            if k: keys[k] = label
    return keys


def main():
    global _li, _auth_keys
    ap = argparse.ArgumentParser()
    ap.add_argument('--root',  required=True)
    ap.add_argument('--port',  type=int, default=8000)
    ap.add_argument('--create', action='store_true',
                    help='create the index if it does not exist')
    ap.add_argument('--dim', type=int, default=128)
    ap.add_argument('--sub_dim', type=int, default=16)
    ap.add_argument('--depth',   type=int, default=14)
    ap.add_argument('--n_trees', type=int, default=200)
    ap.add_argument('--gen',     type=int, default=3)
    ap.add_argument('--base',    default=None,
                    help='external base.fvecs; omit for auto-store')
    ap.add_argument('--auth-keys-file', default=None,
                    help='Path to a file with "label:key" lines. If absent, '
                         'reads MG_API_KEYS env var. Empty = no auth.')
    args = ap.parse_args()

    _auth_keys = _load_auth_keys('MG_API_KEYS', args.auth_keys_file)
    if _auth_keys:
        sys.stderr.write(
            f'[serve_live] AuthN enabled, {len(_auth_keys)} key(s) loaded\n')
    else:
        sys.stderr.write(
            '[serve_live] WARNING: AuthN DISABLED (no keys). Front with a '
            'reverse proxy or set MG_API_KEYS for prod.\n')

    if args.create and not os.path.exists(os.path.join(args.root, 'manifest.json')):
        _li = LiveIndex.create(args.root, dim=args.dim, sub_dim=args.sub_dim,
                               n_trees=args.n_trees, depth=args.depth,
                               gen_version=args.gen, base_path=args.base)
        sys.stderr.write(f'[serve_live] created {args.root}\n')
    else:
        _li = LiveIndex.open(args.root)
        sys.stderr.write(
            f'[serve_live] opened {args.root}, '
            f'{len(_li.manifest["segments"])} segments, '
            f'next_doc_id={_li.manifest["next_doc_id"]}\n')

    # Graceful shutdown wiring.
    signal.signal(signal.SIGTERM, _graceful_shutdown)
    signal.signal(signal.SIGINT,  _graceful_shutdown)
    atexit.register(lambda: _graceful_shutdown(None, None) if _li else None)

    srv = ThreadingHTTPServer(('0.0.0.0', args.port), _Handler)
    sys.stderr.write(f'[serve_live] listening on :{args.port}\n')
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        _graceful_shutdown('SIGINT', None)


if __name__ == '__main__':
    main()
