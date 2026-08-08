"""Tiny Prometheus exporter for mangrove-search.

Pure Python (no prometheus_client dep). Designed to be embedded in a long-
lived Python orchestrator and serve /metrics via the stdlib http.server.

Usage:
    from telemetry import Counter, Histogram, start_http_server

    queries  = Counter('mangrove_queries_total', 'Queries served')
    qlat_ms  = Histogram('mangrove_query_latency_ms', 'Query latency (ms)',
                         buckets=(1, 2, 5, 10, 20, 50, 100, 200, 500, 1000))
    distinct = Counter('mangrove_n_distinct_total',
                       'Sum of K-way merge n_distinct over queries')

    start_http_server(8000)

    # in your query path:
    t0 = time.time()
    ids, votes, n = forest.query(qvec, top_n=500)
    qlat_ms.observe((time.time() - t0) * 1000)
    queries.inc()
    distinct.inc(forest.n_distinct())
"""
from __future__ import annotations

import threading
import time
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer


_REGISTRY: list = []
_LOCK = threading.Lock()


class Counter:
    def __init__(self, name: str, help: str = '') -> None:
        self.name  = name
        self.help  = help
        self.value = 0.0
        _REGISTRY.append(self)

    def inc(self, n: float = 1.0) -> None:
        with _LOCK:
            self.value += n

    def render(self) -> str:
        lines = []
        if self.help:
            lines.append(f'# HELP {self.name} {self.help}')
        lines.append(f'# TYPE {self.name} counter')
        lines.append(f'{self.name} {self.value:g}')
        return '\n'.join(lines)


class Gauge:
    def __init__(self, name: str, help: str = '') -> None:
        self.name  = name
        self.help  = help
        self.value = 0.0
        _REGISTRY.append(self)

    def set(self, v: float) -> None:
        with _LOCK:
            self.value = v

    def render(self) -> str:
        lines = []
        if self.help:
            lines.append(f'# HELP {self.name} {self.help}')
        lines.append(f'# TYPE {self.name} gauge')
        lines.append(f'{self.name} {self.value:g}')
        return '\n'.join(lines)


class Histogram:
    def __init__(self, name: str, help: str = '',
                 buckets: tuple[float, ...] = (1, 2, 5, 10, 20, 50, 100,
                                                200, 500, 1000)) -> None:
        self.name    = name
        self.help    = help
        self.buckets = list(buckets) + [float('inf')]
        self.counts  = [0] * len(self.buckets)
        self.sum_v   = 0.0
        self.count   = 0
        _REGISTRY.append(self)

    def observe(self, v: float) -> None:
        with _LOCK:
            for i, b in enumerate(self.buckets):
                if v <= b:
                    self.counts[i] += 1
                    # cumulative — Prometheus expects each bucket to be a
                    # running tally, so we update all higher-le buckets too.
                    for j in range(i + 1, len(self.buckets)):
                        self.counts[j] += 1
                    break
            else:
                self.counts[-1] += 1
            self.sum_v += v
            self.count += 1

    def render(self) -> str:
        lines = []
        if self.help:
            lines.append(f'# HELP {self.name} {self.help}')
        lines.append(f'# TYPE {self.name} histogram')
        for b, c in zip(self.buckets, self.counts):
            le = '+Inf' if b == float('inf') else f'{b:g}'
            lines.append(f'{self.name}_bucket{{le="{le}"}} {c}')
        lines.append(f'{self.name}_sum {self.sum_v:g}')
        lines.append(f'{self.name}_count {self.count}')
        return '\n'.join(lines)


def render_all() -> str:
    with _LOCK:
        return '\n'.join(m.render() for m in _REGISTRY) + '\n'


class _MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 — stdlib API
        if self.path != '/metrics':
            self.send_response(404); self.end_headers(); return
        body = render_all().encode()
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain; version=0.0.4')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args, **kw):  # silence per-request log spam
        return


def start_http_server(port: int = 8000) -> threading.Thread:
    """Start a background HTTP server serving /metrics on the given port."""
    server = HTTPServer(('0.0.0.0', port), _MetricsHandler)
    t = threading.Thread(target=server.serve_forever, name='telemetry',
                         daemon=True)
    t.start()
    sys.stderr.write(f'[telemetry] /metrics serving on :{port}\n')
    return t


if __name__ == '__main__':
    # quick demo
    queries = Counter('demo_queries', 'demo')
    lat     = Histogram('demo_latency_ms', 'demo')
    start_http_server(8000)
    for i in range(10):
        queries.inc()
        lat.observe(i * 1.7)
    print(render_all())
