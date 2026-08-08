"""Stress test: concurrent queries against serve.py.

Launches N parallel workers, each posting M queries from sift_query.fvecs
to the /search endpoint. Measures:
  - throughput (queries/sec)
  - latency percentiles (p50, p95, p99, p999)
  - result consistency (same query → same top-k across runs)

Usage:
  # in one terminal:
  python3 scripts/serve.py --index /tmp/srt3_test --n_trees 20 \
      --dim 128 --depth 12 --n_docs 1000000 --gen 0 --port 8765
  # in another:
  python3 scripts/stress_concurrency.py --url http://127.0.0.1:8765 \
      --concurrency 10 --queries 1000
"""
from __future__ import annotations

import argparse
import json
import statistics
import struct
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def read_fvecs(path: str, n: int) -> list[list[float]]:
    out = []
    with open(path, 'rb') as f:
        for _ in range(n):
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack('<i', dim_bytes)[0]
            data = f.read(dim * 4)
            out.append(list(struct.unpack(f'<{dim}f', data)))
    return out


def post_search(url: str, qvec: list[float], top_n: int) -> tuple[float, list[int]]:
    body = json.dumps({'qvec': qvec, 'top_n': top_n}).encode()
    req = urllib.request.Request(url + '/search', data=body,
                                 headers={'Content-Type': 'application/json'},
                                 method='POST')
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=10) as r:
        payload = json.loads(r.read().decode())
    dt_ms = (time.time() - t0) * 1000
    return dt_ms, payload.get('ids', [])


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--url',         default='http://127.0.0.1:8765')
    ap.add_argument('--qvecs',       default='sift/sift_query.fvecs')
    ap.add_argument('--concurrency', type=int, default=10)
    ap.add_argument('--queries',     type=int, default=500)
    ap.add_argument('--top_n',       type=int, default=500)
    args = ap.parse_args()

    queries = read_fvecs(args.qvecs, args.queries)
    if not queries:
        sys.exit('no queries loaded')
    print(f'Loaded {len(queries)} queries, concurrency={args.concurrency}')

    # warmup
    for q in queries[:5]:
        post_search(args.url, q, args.top_n)

    seen_first: dict[int, tuple[int, ...]] = {}  # qid -> first-seen top-10
    mismatches = 0
    latencies: list[float] = []

    def worker(qid: int) -> tuple[int, float, tuple[int, ...]]:
        dt, ids = post_search(args.url, queries[qid], args.top_n)
        return qid, dt, tuple(ids[:10])

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(worker, i % len(queries))
                   for i in range(args.queries)]
        for fut in as_completed(futures):
            try:
                qid, dt, top10 = fut.result()
            except Exception as e:
                sys.stderr.write(f'error: {e}\n')
                continue
            latencies.append(dt)
            if qid in seen_first:
                if seen_first[qid] != top10:
                    mismatches += 1
            else:
                seen_first[qid] = top10
    dt_total = time.time() - t0

    print(f'\n--- {args.queries} queries via {args.concurrency} workers in {dt_total:.2f}s ---')
    print(f'throughput : {args.queries / dt_total:.1f} q/s')
    print(f'mean       : {statistics.mean(latencies):.2f} ms')
    print(f'p50        : {percentile(latencies, 50):.2f} ms')
    print(f'p95        : {percentile(latencies, 95):.2f} ms')
    print(f'p99        : {percentile(latencies, 99):.2f} ms')
    print(f'p999       : {percentile(latencies, 99.9):.2f} ms')
    print(f'mismatches : {mismatches} (same-query top-10 inconsistent across threads)')
    if mismatches > 0:
        print('  ⚠  Concurrent queries produced different results — thread-safety bug.')


if __name__ == '__main__':
    main()
