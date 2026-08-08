"""Stress test: query the forest with synthetic filters of varying density.

For each density in [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5], generate a random
sample of `density * n_docs` allowed doc_ids, then run N queries through the
forest with that filter. Reports recall vs unfiltered baseline (proxy),
latency percentiles, n_distinct.

Goal: verify the K-way merge filter-aware skip works well across the density
spectrum, especially the sparse end where pre-filter should win.

Usage:
    python3 scripts/stress_filter_density.py \
        --index /mnt/mangrove/indexes/sift100m \
        --queries .../queries.u8bin \
        --n_trees 200 --depth 25 --sub_dim 16 --n_docs 100000000 \
        --n_queries 100
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mangrove_ffi import Forest  # noqa: E402


def percentile(values, p):
    s = sorted(values)
    if not s:
        return 0.0
    k = (len(s) - 1) * p / 100.0
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def read_u8bin(path: str, n: int, dim: int) -> np.ndarray:
    with open(path, 'rb') as f:
        f.read(8)  # skip header
        buf = f.read(n * dim)
        return np.frombuffer(buf, dtype=np.uint8).reshape(n, dim).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--index',     required=True)
    ap.add_argument('--queries',   required=True)
    ap.add_argument('--n_trees',   type=int, required=True)
    ap.add_argument('--depth',     type=int, required=True)
    ap.add_argument('--sub_dim',   type=int, default=0)
    ap.add_argument('--n_docs',    type=int, required=True)
    ap.add_argument('--dim',       type=int, default=128)
    ap.add_argument('--gen',       type=int, default=3)
    ap.add_argument('--n_queries', type=int, default=100)
    ap.add_argument('--top_n',     type=int, default=500)
    ap.add_argument('--top_k',     type=int, default=10)
    ap.add_argument('--densities', default='0.0001,0.001,0.01,0.05,0.1,0.5')
    args = ap.parse_args()
    densities = [float(d) for d in args.densities.split(',')]

    queries = read_u8bin(args.queries, args.n_queries, args.dim)

    f = Forest(args.index, n_trees=args.n_trees, dim=args.dim,
               sub_dim=args.sub_dim, depth=args.depth,
               n_docs=args.n_docs, gen_version=args.gen)
    print(f'forest opened: n_trees={f.n_trees} depth={f.depth} '
          f'sub_dim={f.sub_dim} n_docs={f.n_docs}')

    # Baseline (no filter) for recall proxy
    print(f'\n--- baseline (no filter, {args.n_queries} queries) ---')
    baseline_top: list[set] = []
    base_lat: list[float] = []
    for q in queries:
        t0 = time.time()
        ids, _, _ = f.query(q, top_n=args.top_n)
        base_lat.append((time.time() - t0) * 1000)
        baseline_top.append(set(int(x) for x in ids[:args.top_k]))
    print(f'  baseline p50={percentile(base_lat, 50):.2f} ms '
          f'p99={percentile(base_lat, 99):.2f} ms')

    print(f'\n{"density":>10}  {"|filter|":>10}  {"p50_ms":>7}  {"p99_ms":>7}  '
          f'{"avg_dist":>10}  recall_vs_base')
    print('-' * 70)

    rng = np.random.default_rng(42)
    for d in densities:
        card = max(1, int(d * args.n_docs))
        allowed = rng.choice(args.n_docs, size=card, replace=False).astype(np.int32)
        allowed.sort()
        lat = []
        recalls = []
        distinct_sum = 0
        for i, q in enumerate(queries):
            t0 = time.time()
            ids, _, _ = f.query_with_ids(q, allowed, top_n=args.top_n)
            lat.append((time.time() - t0) * 1000)
            topk = set(int(x) for x in ids[:args.top_k])
            base_filtered = baseline_top[i] & set(int(x) for x in allowed)
            if base_filtered:
                recalls.append(len(topk & base_filtered) / len(base_filtered))
            distinct_sum += f.n_distinct()
        rcl = np.mean(recalls) if recalls else float('nan')
        print(f'{d:>10.4%}  {card:>10}  '
              f'{percentile(lat, 50):>7.2f}  {percentile(lat, 99):>7.2f}  '
              f'{distinct_sum / args.n_queries:>10.0f}  {rcl:.4f}')

    f.close()


if __name__ == '__main__':
    main()
