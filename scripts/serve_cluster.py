"""HTTP server backed by MangroveCluster (multi-index registry).

Endpoints (all JSON unless noted) :

Cluster-level :
  GET  /health
  GET  /stats
  GET  /indexes?pattern=<glob>           — list index names
  POST /indexes  {name, dim, sub_dim, ...} — register a new index
  DELETE /indexes/<name>                 — drop
  POST /search   {pattern, qvec, top_n, top_k}   — cluster-wide search
  GET  /metrics                          — Prometheus

Per-index :
  GET  /indexes/<name>/health
  GET  /indexes/<name>/stats
  POST /indexes/<name>/insert         {vec, [doc_id]}
  POST /indexes/<name>/insert_batch   {vecs}
  POST /indexes/<name>/search         {qvec, top_n, top_k}
  POST /indexes/<name>/delete         {doc_id}
  POST /indexes/<name>/freeze         {}

Auth : X-API-Key header (same as serve_live.py), /health /metrics public.

Run :
  python3 scripts/serve_cluster.py --root /var/mangrove/cluster --port 8000
"""
from __future__ import annotations

import argparse, atexit, json, os, signal, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from live_index import Backpressure
from registry import MangroveCluster
from telemetry import Counter, Gauge, Histogram, render_all
from mangrove_ffi import set_external_leaves


# ---- metrics ----
QUERIES     = Counter('mg_queries_total',  'Search requests served')
INSERTS     = Counter('mg_inserts_total',  'Docs inserted (all indexes)')
DELETES     = Counter('mg_deletes_total',  'Docs tombstoned (all indexes)')
FREEZES     = Counter('mg_freezes_total',  'Manual freezes triggered')
ERRORS      = Counter('mg_errors_total',   'Endpoint errors')
BP_REJECTS  = Counter('mg_backpressure_rejects_total',
                      'Inserts rejected due to full active buffer')
AUTH_REJECTS = Counter('mg_auth_rejects_total',
                       'Requests rejected for missing/invalid API key')
LAT_QUERY   = Histogram('mg_query_latency_ms',  'Query latency ms')
LAT_INSERT  = Histogram('mg_insert_latency_ms', 'Insert latency ms')
INDEXES_ALIVE = Gauge('mg_indexes_alive', 'Number of registered indexes')


_cluster: MangroveCluster | None = None
_lock = threading.Lock()


def _exhaustive_stats_one(name: str) -> dict:
    """Detailed stats for one index : params, segments per tier with
       depth + n_docs, total disk size, WAL size, active buffer size."""
    li = _cluster.get(name)
    segs = li.manifest['segments']
    from collections import Counter
    by_tier = Counter()
    seg_details = []
    total_docs = li.active_size()
    for s in segs:
        by_tier[s.get('tier', 0)] += 1
        total_docs += s['n_docs']
        seg_details.append({
            'name':       s['name'],
            'tier':       s.get('tier', 0),
            'depth':      s['depth'],
            'n_docs':     s['n_docs'],
            'doc_offset': s['doc_offset'],
        })
    # Disk usage of the index dir (best-effort, fast cached du if available)
    idx_dir = os.path.join(_cluster.root, name)
    disk_bytes = 0
    try:
        for root, _dirs, files in os.walk(idx_dir):
            for f in files:
                try: disk_bytes += os.path.getsize(os.path.join(root, f))
                except OSError: pass
    except OSError: pass
    # WAL size (tracked separately)
    wal_path = os.path.join(idx_dir, 'wal.bin')
    wal_bytes = os.path.getsize(wal_path) if os.path.exists(wal_path) else 0
    return {
        'name':          name,
        'dim':           li.manifest['dim'],
        'sub_dim':       li.manifest['sub_dim'],
        'n_trees':       li.manifest['n_trees'],
        'gen_version':   li.manifest['gen_version'],
        'max_active':    li.manifest.get('max_active', 0),
        'next_doc_id':   li.manifest['next_doc_id'],
        'active_size':   li.active_size(),
        'total_docs':    total_docs,
        'tier_counts':   dict(by_tier),
        'segments':      seg_details,
        'n_segments':    len(segs),
        'disk_bytes':    disk_bytes,
        'wal_bytes':     wal_bytes,
        'mode':          li.mode,
    }


def _exhaustive_cluster_stats(pattern: str | None) -> dict:
    """Cluster-wide stats with per-index detail + aggregates.
       If pattern is given, restrict to matching index names."""
    names = _cluster.list_indexes(pattern)
    per_index = []
    total_docs = 0
    total_segments = 0
    total_disk = 0
    total_wal = 0
    for name in names:
        try:
            st = _exhaustive_stats_one(name)
            per_index.append(st)
            total_docs     += st['total_docs']
            total_segments += st['n_segments']
            total_disk     += st['disk_bytes']
            total_wal      += st['wal_bytes']
        except Exception as e:
            per_index.append({'name': name, 'error': str(e)})
    return {
        'pattern':         pattern,
        'matched_indexes': names,
        'indexes':         per_index,
        'aggregate': {
            'n_indexes':      len(names),
            'total_docs':     total_docs,
            'total_segments': total_segments,
            'total_disk_bytes': total_disk,
            'total_wal_bytes':  total_wal,
        },
    }
_PUBLIC_PATHS = {'/health', '/metrics', '/', '/ui',
                 '/ui/indexes', '/ui/cluster', '/ui/metrics'}

_UI_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>mangrove-search dashboard</title>
  <script src="https://unpkg.com/htmx.org@1.9.10"></script>
  <style>
    :root { color-scheme: dark; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
           Inter, system-ui, sans-serif;
           background: #0f1419; color: #d1d4dc; margin: 0; padding: 0; }
    header { padding: 1em 2em; border-bottom: 1px solid #222;
             display: flex; align-items: baseline; justify-content: space-between; }
    h1 { margin: 0; color: #6cf; font-weight: 500; font-size: 1.4em; }
    main { padding: 1em 2em; max-width: 1400px; margin: 0 auto; }
    section { margin-bottom: 2em; }
    h2 { color: #888; font-weight: 500; font-size: 1em;
         text-transform: uppercase; letter-spacing: 0.1em;
         border-bottom: 1px solid #222; padding-bottom: 0.4em; }
    .cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
             gap: 1em; }
    .card { background: #1a2028; padding: 1em; border-radius: 4px;
            border-left: 3px solid #6cf; }
    .card .label { color: #888; font-size: 0.85em; text-transform: uppercase; }
    .card .value { font-size: 1.6em; font-weight: 500; margin-top: 0.3em;
                   color: #fff; font-variant-numeric: tabular-nums; }
    table { border-collapse: collapse; width: 100%; }
    th, td { padding: 0.4em 0.8em; border-bottom: 1px solid #1a2028;
             text-align: left; font-variant-numeric: tabular-nums; }
    th { color: #888; font-weight: 500; font-size: 0.85em;
         text-transform: uppercase; }
    tr:hover { background: #1a2028; }
    .tier-0 { color: #d1d4dc; }
    .tier-1 { color: #6cf; }
    .tier-2 { color: #f9c74f; }
    .tier-3 { color: #f4a261; }
    .tier-4 { color: #e63946; }
    .pill { display: inline-block; padding: 0.1em 0.6em; border-radius: 10px;
            background: #2a3340; font-size: 0.85em; }
    .pill.green { background: #1b4332; color: #74c69d; }
    .pill.red   { background: #4a1c24; color: #ef5350; }
    a { color: #6cf; text-decoration: none; }
    a:hover { text-decoration: underline; }
    footer { padding: 1em 2em; border-top: 1px solid #222; color: #666;
             font-size: 0.85em; }
  </style>
</head>
<body>
<header>
  <h1>mangrove-search</h1>
  <span style="color:#888">auto-refresh every 5 s</span>
</header>
<main>

  <section>
    <h2>Cluster</h2>
    <div hx-get="/ui/cluster" hx-trigger="load, every 5s" class="cards">
      loading...
    </div>
  </section>

  <section>
    <h2>Indexes</h2>
    <table>
      <thead>
        <tr>
          <th>name</th>
          <th style="text-align:right">segments</th>
          <th style="text-align:right">total docs</th>
          <th style="text-align:right">active buffer</th>
          <th>tier breakdown</th>
          <th>params</th>
        </tr>
      </thead>
      <tbody hx-get="/ui/indexes" hx-trigger="load, every 5s">
        <tr><td colspan="6">loading...</td></tr>
      </tbody>
    </table>
  </section>

  <section>
    <h2>Recent metrics</h2>
    <div hx-get="/ui/metrics" hx-trigger="load, every 5s" class="cards">
      loading...
    </div>
  </section>

</main>
<footer>
  <a href="/metrics">/metrics</a> &middot;
  <a href="/health">/health</a> &middot;
  <a href="https://gitlab.com/leo_chartier/mangrove-search">docs</a>
</footer>
</body>
</html>
"""

# AuthN + AuthZ + rate-limit per key. _auth_keys[key] = ScopedKey :
#   label       : human-readable, for audit
#   patterns    : list of index-name globs this key may touch
#   perms       : 'r' (search/health/stats), 'rw' (+ insert/delete/freeze),
#                 'admin' (+ create/drop)
#   qps         : token-bucket rate (queries/sec); 0 = unlimited
import fnmatch
from dataclasses import dataclass, field
@dataclass
class ScopedKey:
    label:    str
    patterns: list[str] = field(default_factory=lambda: ['*'])
    perms:    str = 'admin'
    qps:      float = 0.0
    # token-bucket state
    _tokens:  float = 0.0
    _last:    float = 0.0

    def allows(self, index_name: str, required: str) -> bool:
        if required == 'admin' and self.perms != 'admin':
            return False
        if required == 'rw' and self.perms not in ('rw', 'admin'):
            return False
        # required == 'r' is fine for any
        return any(fnmatch.fnmatch(index_name, p) for p in self.patterns)

    def consume(self) -> bool:
        """Token bucket : returns True if a token was available, False
           if the caller should be rate-limited."""
        if self.qps <= 0:
            return True
        now = time.monotonic()
        elapsed = now - self._last
        self._last = now
        # refill : qps tokens per second, capped at qps (1-second burst)
        self._tokens = min(self.qps, self._tokens + elapsed * self.qps)
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False

_auth_keys: dict[str, ScopedKey] = {}


def _graceful_shutdown(signum=None, frame=None):
    global _cluster
    sys.stderr.write(f'\n[serve_cluster] graceful shutdown (sig={signum})\n')
    if _cluster:
        try:
            with _lock:
                for name in _cluster.list_indexes():
                    li = _cluster.get(name)
                    if li.active_size() > 0:
                        sys.stderr.write(f'  freezing {li.active_size()} '
                                         f'pending docs in {name}\n')
                        li.freeze()
                _cluster.close()
        except Exception as e:
            sys.stderr.write(f'  shutdown error: {e}\n')
        _cluster = None
    sys.exit(0)


def _parse_key_line(ln: str) -> tuple[str, ScopedKey] | None:
    """Parse 'label:key:patterns:perms:qps' (any trailing field optional).
       Patterns are comma-separated globs.                              """
    ln = ln.strip()
    if not ln or ln.startswith('#'): return None
    parts = ln.split(':')
    if len(parts) < 2: return None
    label, k = parts[0], parts[1]
    patterns = (parts[2].split(',') if len(parts) > 2 and parts[2] else ['*'])
    perms    = parts[3] if len(parts) > 3 and parts[3] else 'admin'
    qps      = float(parts[4]) if len(parts) > 4 and parts[4] else 0.0
    return k, ScopedKey(label=label, patterns=patterns, perms=perms, qps=qps)


def _load_auth_keys(env_var: str, key_file: str | None) -> dict[str, ScopedKey]:
    keys: dict[str, ScopedKey] = {}
    if key_file and os.path.exists(key_file):
        with open(key_file) as f:
            for ln in f:
                kv = _parse_key_line(ln)
                if kv: keys[kv[0]] = kv[1]
    else:
        for tok in os.environ.get(env_var, '').split(','):
            kv = _parse_key_line(tok)
            if kv: keys[kv[0]] = kv[1]
    return keys


RATE_LIMITED = Counter('mg_rate_limited_total',
                       'Requests denied by per-key rate limit')
# Per-stage breakdowns : caller can read individual histograms in /metrics
LAT_FOREST   = Histogram('mg_stage_forest_ms',  'Per-stage: forest traversal')
LAT_MERGE    = Histogram('mg_stage_merge_ms',   'Per-stage: K-way merge')
LAT_RERANK   = Histogram('mg_stage_rerank_ms',  'Per-stage: L2/cosine rerank')

# Slow query log — emit a structured line for queries above this threshold.
SLOW_QUERY_MS = float(os.environ.get('MG_SLOW_QUERY_MS', '500'))
SLOW_QUERIES = Counter('mg_slow_queries_total',
                       'Queries that exceeded the slow-query threshold')


class _Handler(BaseHTTPRequestHandler):

    def _authed_key(self) -> ScopedKey | None:
        """Return matched ScopedKey or None if no auth required.
           Caller must check separately if the key is missing (401)."""
        if not _auth_keys:
            return ScopedKey(label='_anon', patterns=['*'], perms='admin', qps=0)
        k = self.headers.get('X-API-Key', '')
        return _auth_keys.get(k)

    def _check(self, index_name: str | None, required: str) -> tuple[bool, str]:
        """Authn (key) + Authz (scope+perm) + rate-limit. Returns
           (allowed, reason)."""
        path = urlparse(self.path).path
        if path in _PUBLIC_PATHS:
            return True, ''
        if not _auth_keys:
            return True, ''
        k = self.headers.get('X-API-Key', '')
        sk = _auth_keys.get(k)
        if sk is None:
            return False, 'unauthorized'
        # AuthZ : when index_name is given, check both perm AND scope ; when
        # None (cluster-level op like POST /indexes, /search), check perm
        # alone — the route knows what perm it needs.
        if required == 'admin' and sk.perms != 'admin':
            return False, 'forbidden'
        if required == 'rw' and sk.perms not in ('rw', 'admin'):
            return False, 'forbidden'
        if index_name and not sk.allows(index_name, required):
            return False, 'forbidden'
        # Rate limit (last check so unauthorized rejections don't burn tokens)
        if not sk.consume():
            return False, 'rate_limited'
        return True, ''

    def _reject(self, reason: str):
        if reason == 'rate_limited':
            RATE_LIMITED.inc()
            code, body = 429, {'error': 'rate_limited'}
            extra_headers = [('Retry-After', '1')]
        elif reason == 'forbidden':
            AUTH_REJECTS.inc()
            code, body = 403, {'error': 'forbidden'}
            extra_headers = []
        else:
            AUTH_REJECTS.inc()
            code, body = 401, {'error': 'unauthorized'}
            extra_headers = [('WWW-Authenticate', 'API-Key')]
        b = json.dumps(body).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        for h, v in extra_headers: self.send_header(h, v)
        self.send_header('Content-Length', str(len(b)))
        self.end_headers(); self.wfile.write(b)

    def _safe_write(self, data: bytes) -> None:
        """Wrap wfile.write to swallow BrokenPipeError when the client
           disconnected before we could respond (common with long-running
           ops whose client timed out). Single-line warn, no stack trace.
        """
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            sys.stderr.write(
                f'[warn] client disconnected before reply ({self.path})\n')

    def _json(self, code, payload):
        b = json.dumps(payload).encode()
        try:
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(b)))
            self.end_headers()
        except (BrokenPipeError, ConnectionResetError):
            return  # client gone before headers complete
        self._safe_write(b)

    def _text(self, code, body):
        b = body.encode()
        try:
            self.send_response(code)
            self.send_header('Content-Type', 'text/plain')
            self.send_header('Content-Length', str(len(b)))
            self.end_headers()
        except (BrokenPipeError, ConnectionResetError):
            return
        self._safe_write(b)

    def _body(self):
        n = int(self.headers.get('Content-Length', '0'))
        return json.loads(self.rfile.read(n).decode()) if n > 0 else {}

    def log_message(self, *a, **kw):
        return

    # ------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------

    def do_GET(self):
        u = urlparse(self.path)
        p = u.path
        qs = parse_qs(u.query or '')

        # Determine per-route index scope + required perm
        idx_name, perm = None, 'r'
        if p.startswith('/indexes/'):
            parts = p.split('/')
            if len(parts) >= 3:
                idx_name = parts[2]
        ok, reason = self._check(idx_name, perm)
        if not ok: self._reject(reason); return

        if p == '/' or p == '/ui':
            # Minimal HTMX management page.
            self._text(200, _UI_HTML); return
        if p == '/ui/indexes':
            # HTMX fragment refreshed every few seconds — richer with tier breakdown
            stats = _cluster.stats() if _cluster else {}
            rows = []
            for name, st in sorted(stats.items()):
                if 'error' in st:
                    rows.append(
                        f'<tr><td>{name}</td><td colspan="5">err: {st["error"]}</td></tr>')
                    continue
                # Pull richer info via direct registry access (cheap, in-memory)
                try:
                    li = _cluster.get(name)
                    segs = li.manifest['segments']
                    tier_counts = {}
                    for s in segs:
                        t = s.get('tier', 0)
                        tier_counts[t] = tier_counts.get(t, 0) + 1
                    tier_html = ' '.join(
                        f'<span class="pill tier-{t}">t{t}×{n}</span>'
                        for t, n in sorted(tier_counts.items()))
                    total_docs = sum(s['n_docs'] for s in segs) + li.active_size()
                    params = (f'dim {li.manifest["dim"]} · sub {li.manifest["sub_dim"]} '
                              f'· {li.manifest["n_trees"]}t · v{li.manifest["gen_version"]}')
                    rows.append(
                        f'<tr>'
                        f'<td><b>{name}</b></td>'
                        f'<td style="text-align:right">{len(segs)}</td>'
                        f'<td style="text-align:right">{total_docs:,}</td>'
                        f'<td style="text-align:right">{li.active_size():,}</td>'
                        f'<td>{tier_html or "<i>—</i>"}</td>'
                        f'<td style="color:#888;font-size:0.85em">{params}</td>'
                        f'</tr>')
                except Exception as e:
                    rows.append(
                        f'<tr><td>{name}</td><td colspan="5">err: {e}</td></tr>')
            if not rows:
                rows = ['<tr><td colspan="6"><i>no indexes — POST /indexes to create one</i></td></tr>']
            self._text(200, '\n'.join(rows)); return
        if p == '/ui/cluster':
            n_idx = len(_cluster.list_indexes()) if _cluster else 0
            total_docs = 0
            total_segs = 0
            if _cluster:
                for name in _cluster.list_indexes():
                    try:
                        li = _cluster.get(name)
                        total_docs += sum(s['n_docs'] for s in li.manifest['segments'])
                        total_segs += len(li.manifest['segments'])
                    except Exception:
                        pass
            cards = [
                ('Indexes', f'{n_idx}'),
                ('Total docs', f'{total_docs:,}'),
                ('Total segments', f'{total_segs}'),
                ('Status', '<span class="pill green">ok</span>'),
            ]
            html = '\n'.join(
                f'<div class="card"><div class="label">{l}</div>'
                f'<div class="value">{v}</div></div>'
                for l, v in cards)
            self._text(200, html); return
        if p == '/ui/metrics':
            cards = [
                ('Queries (cumulative)', f'{QUERIES.value:,.0f}'),
                ('Inserts', f'{INSERTS.value:,.0f}'),
                ('Deletes', f'{DELETES.value:,.0f}'),
                ('Errors', f'{ERRORS.value:,.0f}'),
                ('Backpressure 503', f'{BP_REJECTS.value:,.0f}'),
                ('Slow queries (>{0:.0f}ms)'.format(SLOW_QUERY_MS),
                 f'{SLOW_QUERIES.value:,.0f}'),
                ('Rate-limited 429', f'{RATE_LIMITED.value:,.0f}'),
                ('Auth rejects', f'{AUTH_REJECTS.value:,.0f}'),
            ]
            html = '\n'.join(
                f'<div class="card"><div class="label">{l}</div>'
                f'<div class="value">{v}</div></div>'
                for l, v in cards)
            self._text(200, html); return
        if p == '/health':
            self._json(200, {
                'status':  'ok' if _cluster else 'no_cluster',
                'indexes': len(_cluster.list_indexes()) if _cluster else 0,
            })
        elif p == '/stats':
            if _cluster is None:
                self._json(503, {'error': 'no cluster'}); return
            pat = (qs.get('pattern') or [None])[0]
            self._json(200, _exhaustive_cluster_stats(pat))
        elif p == '/metrics':
            if _cluster:
                INDEXES_ALIVE.set(len(_cluster.list_indexes()))
            self._text(200, render_all())
        elif p == '/indexes':
            pat = (qs.get('pattern') or [None])[0]
            self._json(200, {'names': _cluster.list_indexes(pat)})
        elif p.startswith('/indexes/'):
            parts = p.split('/')
            if len(parts) >= 4 and parts[3] == 'health':
                name = parts[2]
                try:
                    li = _cluster.get(name)
                    self._json(200, {
                        'name':         name,
                        'segments':     len(li.manifest['segments']),
                        'active_size':  li.active_size(),
                        'next_doc_id':  li.manifest['next_doc_id'],
                    })
                except KeyError:
                    self._json(404, {'error': f'no such index {name!r}'})
            elif len(parts) >= 4 and parts[3] == 'stats':
                name = parts[2]
                try:
                    self._json(200, _exhaustive_stats_one(name))
                except KeyError:
                    self._json(404, {'error': f'no such index {name!r}'})
            else:
                self._json(404, {'error': 'unknown'})
        else:
            self._json(404, {'error': 'unknown'})

    def do_DELETE(self):
        p = urlparse(self.path).path
        if not p.startswith('/indexes/'):
            self._json(404, {'error': 'unknown'}); return
        name = p.split('/')[2]
        ok, reason = self._check(name, 'admin')
        if not ok: self._reject(reason); return
        if _cluster is None:
            ERRORS.inc(); self._json(503, {'error': 'no cluster'}); return
        if p.startswith('/indexes/'):
            try:
                with _lock:
                    _cluster.drop_index(name)
                self._json(200, {'dropped': name})
            except KeyError:
                self._json(404, {'error': f'no such index {name!r}'})
            except Exception as e:
                ERRORS.inc(); self._json(500, {'error': str(e)})
        else:
            self._json(404, {'error': 'unknown'})

    def do_POST(self):
        p = urlparse(self.path).path
        # Compute required perm + scoped index name per route
        idx_name, perm = None, 'rw'
        if p == '/indexes':
            perm = 'admin'  # creation needs admin (no specific index yet)
        elif p == '/search':
            perm = 'r'
        elif p.startswith('/indexes/'):
            parts = p.split('/')
            if len(parts) >= 4:
                idx_name = parts[2]
                op = parts[3]
                perm = 'r' if op in ('search', 'search_by_leaves') else 'rw'
        ok, reason = self._check(idx_name, perm)
        if not ok: self._reject(reason); return

        if _cluster is None:
            ERRORS.inc(); self._json(503, {'error': 'no cluster'}); return
        try:
            body = self._body()
        except Exception as e:
            ERRORS.inc(); self._json(400, {'error': f'bad json: {e}'}); return

        if p == '/indexes':
            try:
                # Defaults are tuned for "set it and forget it" UX. The user
                # only has to specify `dim`. depth=14 starts small ; LSM
                # auto-compaction grows depth +2 per tier as docs accumulate
                # (tier 0=14, tier 1=16, ..., tier 8=30 covers up to ~3B docs).
                with _lock:
                    _cluster.create_index(
                        body['name'], dim=int(body['dim']),
                        sub_dim=int(body.get('sub_dim', 16)),
                        n_trees=int(body.get('n_trees', 1000)),
                        depth=int(body.get('depth', 14)),
                        gen_version=int(body.get('gen_version', 3)),
                        max_active=int(body.get('max_active', 100_000)))
                self._json(200, {'created': body['name']})
            except ValueError as e:
                self._json(409, {'error': str(e)})
            except Exception as e:
                ERRORS.inc(); self._json(500, {'error': str(e)})

        elif p == '/search':
            try:
                pattern = body['pattern']
                q = np.asarray(body['qvec'], dtype=np.float32)
                t0 = time.time()
                with _lock:
                    res = _cluster.search(pattern, q,
                                          top_n=int(body.get('top_n', 4000)),
                                          top_k=int(body.get('top_k', 10)))
                LAT_QUERY.observe((time.time() - t0) * 1000)
                QUERIES.inc()
                res['latency_ms'] = round((time.time() - t0) * 1000, 3)
                self._json(200, res)
            except Exception as e:
                ERRORS.inc(); self._json(500, {'error': str(e)})

        elif p.startswith('/indexes/'):
            parts = p.split('/')
            if len(parts) < 4:
                self._json(404, {'error': 'unknown'}); return
            name, op = parts[2], parts[3]
            try:
                li = _cluster.get(name)
            except KeyError:
                self._json(404, {'error': f'no such index {name!r}'}); return

            if op == 'insert':
                try:
                    t0 = time.time()
                    with _lock:
                        doc_id = li.insert(
                            np.asarray(body['vec'], dtype=np.float32),
                            doc_id=body.get('doc_id'))
                    LAT_INSERT.observe((time.time() - t0) * 1000)
                    INSERTS.inc()
                    self._json(200, {'doc_id': doc_id})
                except Backpressure as e:
                    BP_REJECTS.inc()
                    self.send_response(503)
                    self.send_header('Retry-After', '1')
                    self.send_header('Content-Type', 'application/json')
                    pl = json.dumps({'error': 'backpressure',
                                     'detail': str(e)}).encode()
                    self.send_header('Content-Length', str(len(pl)))
                    self.end_headers(); self.wfile.write(pl)
                except Exception as e:
                    ERRORS.inc(); self._json(500, {'error': str(e)})

            elif op == 'insert_batch':
                ids: list[int] = []
                try:
                    with _lock:
                        for v in body['vecs']:
                            ids.append(li.insert(np.asarray(v, dtype=np.float32)))
                    INSERTS.inc(len(ids))
                    self._json(200, {'doc_ids': ids, 'count': len(ids)})
                except Backpressure as e:
                    BP_REJECTS.inc()
                    self.send_response(503)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Retry-After', '1')
                    pl = json.dumps({'error': 'backpressure',
                                     'completed': len(ids),
                                     'doc_ids': ids}).encode()
                    self.send_header('Content-Length', str(len(pl)))
                    self.end_headers(); self.wfile.write(pl)
                except Exception as e:
                    ERRORS.inc(); self._json(500, {'error': str(e)})

            elif op == 'search':
                try:
                    q = np.asarray(body['qvec'], dtype=np.float32)
                    top_k = int(body.get('top_k', 5))
                    metric = body.get('metric', 'l2')
                    # Adaptive top_n : if client didn't specify, derive from
                    # index params. Low-dim corpora (≤256) need a smaller
                    # candidate pool ; high-dim (≥384) need a wider pool to
                    # compensate for sub_dim sampling sparsity. See memory
                    # feedback_target_ratio.
                    dim = li.manifest['dim']
                    n_docs_total = max(
                        li.manifest['next_doc_id'],
                        sum(s['n_docs'] for s in li.manifest['segments']),
                        1,
                    )
                    if 'top_n' in body:
                        top_n = int(body['top_n'])
                    else:
                        # Adaptive top_n :
                        #   - small corpora (<1M)  : larger ratio (0.02) to
                        #     not starve recall at small N
                        #   - large corpora (≥1M)  : lower ratio scaled by dim
                        #     (0.001 for dim≤256, 0.05 for dim≥384)
                        if n_docs_total < 1_000_000:
                            ratio = 0.02
                        elif dim >= 384:
                            ratio = 0.05
                        else:
                            ratio = 0.001
                        top_n = int(n_docs_total * ratio)
                        top_n = max(top_n, 10 * top_k, 2000)
                        top_n = min(top_n, 50_000)
                    allowed_ids = body.get('allowed_ids')
                    if allowed_ids is not None:
                        allowed_ids = np.asarray(allowed_ids, dtype=np.int32)
                    # Bitmap path : raw bytes from ClickHouse groupBitmapState
                    # arrive base64-encoded over JSON ; decode once here.
                    allowed_bitmap = body.get('allowed_bitmap_b64')
                    if allowed_bitmap is not None:
                        import base64 as _b64
                        allowed_bitmap = _b64.b64decode(allowed_bitmap)
                    # Pagination via cursor : {'after': last_doc_id_seen}.
                    # MVP — we re-run the same query but skip cursor's doc_id
                    # and earlier. Stable for monotonic-rerank metrics on a
                    # fixed corpus snapshot. Future: stateful cursor server-side.
                    cursor_after = body.get('cursor_after')
                    query_depth = int(body.get('query_depth', 0))
                    # Adaptive n_probes : caller can override via body, else
                    # we default to 5 — the measured Pareto sweet spot across
                    # SIFT (dim 128), arxiv (dim 768), Cohere (dim 1024).
                    # Recall stays ≥ 0.95 at 5 probes with the truly-fused
                    # path ; below 5 the fused vote dedup loses ground.
                    n_probes = int(body.get('n_probes', 5))
                    # Adaptive probe_depth : opt-in via body. When n_probes > 0
                    # and the corpus has high dim (>= 384), the recall jumps
                    # if we descend a few levels below native ; the sub_dim/dim
                    # ratio becomes too small for native splits to discriminate
                    # alone (cf BENCH.md arxiv / cohere sections).
                    if 'probe_depth' in body:
                        probe_depth = int(body['probe_depth'])
                    elif n_probes > 0 and dim >= 384:
                        # depth − 6 puts arxiv (depth 20) at qd=14 (recall 0.994)
                        # and cohere (depth 22) at qd=16 (recall 0.98) — the
                        # sweet-spot Pareto we measured. Override per-query if
                        # you need higher recall (lower qd) or lower latency.
                        build_depth = int(li.manifest.get('depth', 0))
                        probe_depth = max(0, build_depth - 6) if build_depth > 0 else 0
                    else:
                        probe_depth = 0   # native
                    max_leaf_bytes = int(body.get('max_leaf_bytes', 0))
                    t0 = time.time()
                    with _lock:
                        ids = li.query(q, top_n=top_n,
                                       top_k=top_k * 4 if cursor_after else top_k,
                                       metric=metric,
                                       allowed_ids=allowed_ids,
                                       allowed_bitmap=allowed_bitmap,
                                       query_depth=query_depth,
                                       n_probes=n_probes,
                                       max_leaf_bytes=max_leaf_bytes)
                    if cursor_after is not None:
                        ids_l = ids.tolist()
                        if cursor_after in ids_l:
                            ids_l = ids_l[ids_l.index(cursor_after) + 1:]
                        ids_l = ids_l[:top_k]
                    else:
                        ids_l = ids.tolist()[:top_k]
                    dt_ms = (time.time() - t0) * 1000
                    LAT_QUERY.observe(dt_ms)
                    QUERIES.inc()
                    if dt_ms > SLOW_QUERY_MS:
                        SLOW_QUERIES.inc()
                        sys.stderr.write(
                            f'[slow] index={name} t={dt_ms:.1f}ms '
                            f'top_n={top_n} top_k={top_k} metric={metric}\n')
                    self._json(200, {
                        'ids': ids_l,
                        'next_cursor': ids_l[-1] if ids_l else None,
                        'latency_ms': round(dt_ms, 3),
                    })
                except Exception as e:
                    ERRORS.inc(); self._json(500, {'error': str(e)})

            elif op == 'search_by_leaves':
                # Privacy mode : client-computed leaves, no qvec on the wire.
                try:
                    leaves = np.asarray(body['leaves'], dtype=np.int32)
                    top_n = int(body.get('top_n', 4000))
                    top_k = int(body.get('top_k', 10))
                    t0 = time.time()
                    set_external_leaves(leaves)
                    try:
                        with _lock:
                            # qvec arg is required by Forest.query but ignored
                            # by the C path when external leaves are armed.
                            dummy_q = np.zeros(li.manifest['dim'], dtype=np.float32)
                            forests = list(li._forests.values())
                            vote_acc: dict[int, int] = {}
                            for f in forests:
                                ids, votes, n = f.query(dummy_q, top_n=top_n)
                                for j in range(n):
                                    vote_acc[int(ids[j])] = vote_acc.get(
                                        int(ids[j]), 0) + int(votes[j])
                    finally:
                        set_external_leaves(None)
                    ranked = sorted(vote_acc.items(),
                                    key=lambda kv: -kv[1])[:top_k]
                    dt_ms = (time.time() - t0) * 1000
                    LAT_QUERY.observe(dt_ms)
                    QUERIES.inc()
                    self._json(200, {
                        'results': [{'doc_id': d, 'votes': v}
                                    for d, v in ranked],
                        'latency_ms': round(dt_ms, 3),
                        'privacy_mode': True,
                    })
                except Exception as e:
                    ERRORS.inc(); self._json(500, {'error': str(e)})

            elif op == 'delete':
                try:
                    with _lock:
                        li.delete(int(body['doc_id']))
                    DELETES.inc()
                    self._json(200, {'doc_id': int(body['doc_id']),
                                     'deleted': True})
                except Exception as e:
                    ERRORS.inc(); self._json(500, {'error': str(e)})

            elif op == 'freeze':
                try:
                    with _lock:
                        seg = li.freeze()
                    FREEZES.inc()
                    self._json(200, {'segment': seg,
                                     'active_after': li.active_size()})
                except Exception as e:
                    ERRORS.inc(); self._json(500, {'error': str(e)})

            else:
                self._json(404, {'error': f'unknown op {op}'})
        else:
            self._json(404, {'error': 'unknown'})


def main():
    global _cluster, _auth_keys
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True,
                    help='cluster root dir (each index = subdir)')
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--auth-keys-file', default=None)
    args = ap.parse_args()

    _cluster = MangroveCluster(args.root)
    sys.stderr.write(
        f'[serve_cluster] root={args.root} '
        f'indexes={len(_cluster.list_indexes())}\n')

    _auth_keys = _load_auth_keys('MG_API_KEYS', args.auth_keys_file)
    if _auth_keys:
        sys.stderr.write(f'  AuthN ON ({len(_auth_keys)} key(s))\n')
    else:
        sys.stderr.write('  WARN: AuthN DISABLED (front with reverse proxy)\n')

    signal.signal(signal.SIGTERM, _graceful_shutdown)
    signal.signal(signal.SIGINT,  _graceful_shutdown)
    atexit.register(lambda: _graceful_shutdown(None, None) if _cluster else None)

    srv = ThreadingHTTPServer(('0.0.0.0', args.port), _Handler)
    sys.stderr.write(f'[serve_cluster] listening on :{args.port}\n')
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        _graceful_shutdown('SIGINT', None)


if __name__ == '__main__':
    main()
