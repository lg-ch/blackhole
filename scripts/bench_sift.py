"""Benchmark a SIFT index via FFI: recall@k, latency percentiles, peak RSS.

Reads queries from a .u8bin or .bvecs file, GT (top-K nearest) from .ivecs,
runs N queries against the forest, computes recall@k and latency stats.

Usage:
    python3 scripts/bench_sift.py \
        --index /mnt/mangrove/indexes/sift100m \
        --queries /home/chatelet/mangrove-search/datasets/sift100M/queries.u8bin \
        --gt /home/chatelet/Documents/bigann_gnd/gnd/idx_100M.ivecs \
        --n_trees 200 --depth 25 --sub_dim 16 --n_docs 100000000 \
        --n_queries 1000 --top_k 10 --top_n 500
"""
from __future__ import annotations

import argparse
import os
import resource
import struct
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mangrove_ffi import Forest  # noqa: E402


def read_u8bin_queries(path: str, n: int, dim: int) -> np.ndarray:
    with open(path, 'rb') as f:
        # u8bin header: int32 n_rows, int32 dim
        n_rows, file_dim = struct.unpack('<ii', f.read(8))
        if file_dim != dim:
            raise ValueError(f'queries dim {file_dim} != expected {dim}')
        m = min(n, n_rows)
        buf = f.read(m * dim)
        return np.frombuffer(buf, dtype=np.uint8).reshape(m, dim).astype(np.float32)


def read_bvecs_queries(path: str, n: int, dim: int) -> np.ndarray:
    out = np.empty((n, dim), dtype=np.float32)
    with open(path, 'rb') as f:
        for i in range(n):
            dim_bytes = f.read(4)
            if not dim_bytes:
                return out[:i]
            d = struct.unpack('<i', dim_bytes)[0]
            if d != dim:
                raise ValueError(f'bvecs dim {d} != expected {dim}')
            buf = f.read(d)
            out[i] = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    return out


def read_fvecs_queries(path: str, n: int, dim: int) -> np.ndarray:
    out = np.empty((n, dim), dtype=np.float32)
    with open(path, 'rb') as f:
        for i in range(n):
            dim_bytes = f.read(4)
            if not dim_bytes:
                return out[:i]
            d = struct.unpack('<i', dim_bytes)[0]
            if d != dim:
                raise ValueError(f'fvecs dim {d} != expected {dim}')
            out[i] = np.frombuffer(f.read(d * 4), dtype=np.float32)
    return out


def read_ivecs_gt(path: str, n: int, top_k: int) -> np.ndarray:
    """Returns (n, top_k) array of GT internal_ids."""
    out = np.empty((n, top_k), dtype=np.int32)
    with open(path, 'rb') as f:
        # First row: read k from header
        head = f.read(4)
        k = struct.unpack('<i', head)[0]
        row_bytes = 4 + k * 4
        f.seek(0)
        for i in range(n):
            f.read(4)  # skip k
            row = f.read(k * 4)
            arr = np.frombuffer(row, dtype=np.int32)
            out[i] = arr[:top_k]
    return out


def percentile(values, p):
    s = sorted(values)
    if not s:
        return 0.0
    k = (len(s) - 1) * p / 100.0
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--index',     required=True)
    ap.add_argument('--queries',   required=True)
    ap.add_argument('--gt',        required=True)
    ap.add_argument('--n_trees',   type=int, required=True)
    ap.add_argument('--depth',     type=int, required=True)
    ap.add_argument('--sub_dim',   type=int, default=0)
    ap.add_argument('--n_docs',    type=int, required=True)
    ap.add_argument('--dim',       type=int, default=128)
    ap.add_argument('--gen',       type=int, default=3)
    ap.add_argument('--n_queries', type=int, default=1000)
    ap.add_argument('--top_k',     type=int, default=10)
    ap.add_argument('--top_n',     type=int, default=4000)
    ap.add_argument('--query_depth', type=int, default=0)
    ap.add_argument('--auto_qd_v2', action='store_true',
                    help='2-probe auto-calibrate qd from observed n_distinct')
    ap.add_argument('--target_ratio', type=float, default=0.0,
                    help='auto_qd_v2 target = ratio × n_pool. '
                         '0 = auto: 0.001 for dim ≤ 256, 0.05 for dim ≥ 384 '
                         '(higher dims need wider scope to compensate for '
                         'sub_dim/dim noisy splits).')
    ap.add_argument('--base', default=None,
                    help='base file for L2 rerank (omit to skip rerank '
                         '— recall will reflect raw forest votes)')
    args = ap.parse_args()

    # Read queries (auto-detected by extension)
    if args.queries.endswith('.u8bin'):
        queries = read_u8bin_queries(args.queries, args.n_queries, args.dim)
    elif args.queries.endswith('.bvecs'):
        queries = read_bvecs_queries(args.queries, args.n_queries, args.dim)
    else:
        queries = read_fvecs_queries(args.queries, args.n_queries, args.dim)
    n_q = len(queries)

    # Read GT
    gt = read_ivecs_gt(args.gt, n_q, args.top_k)
    print(f'queries: {n_q} × dim {args.dim} | GT top_k={args.top_k}')

    f = Forest(args.index, n_trees=args.n_trees, dim=args.dim,
               sub_dim=args.sub_dim, depth=args.depth,
               n_docs=args.n_docs, gen_version=args.gen)
    print(f'forest opened: n_trees={f.n_trees} depth={f.depth} '
          f'sub_dim={f.sub_dim} n_docs={f.n_docs}')

    # Resolve query_depth: explicit > auto_qd_v2 > default
    qd_use = args.query_depth
    if args.auto_qd_v2:
        tr = args.target_ratio
        if tr == 0.0:
            tr = 0.05 if args.dim >= 384 else 0.001
        qd_use = f.auto_qd_v2(queries[0], top_n=args.top_n,
                              target_ratio=tr,
                              n_pool=args.n_docs)
        print(f'auto_qd_v2 picked qd = {qd_use} (build_depth = {f.depth}, '
              f'target_ratio = {tr})')

    # Warmup
    for q in queries[: min(5, n_q)]:
        f.query(q, top_n=args.top_n, query_depth=qd_use)

    latencies = []
    recalls = []
    distinct_total = 0
    t0 = time.time()
    for i in range(n_q):
        q = queries[i]
        t_q = time.time()
        ids, votes, k = f.query(q, top_n=args.top_n, query_depth=qd_use)
        if args.base:
            topk = f.rerank_l2(args.base, q, ids[:k], top_k=args.top_k)
        else:
            topk = ids[:args.top_k]
        dt_ms = (time.time() - t_q) * 1000
        latencies.append(dt_ms)
        gt_set = set(int(x) for x in gt[i])
        hits = sum(1 for x in topk if int(x) in gt_set)
        recalls.append(hits / args.top_k)
        distinct_total += f.n_distinct()
    dt_total = time.time() - t0
    rss_peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    print(f'\n=== RESULTS ({n_q} queries in {dt_total:.2f}s) ===')
    print(f'recall@{args.top_k}     : {np.mean(recalls):.4f}')
    print(f'latency mean   : {np.mean(latencies):.2f} ms')
    print(f'latency p50    : {percentile(latencies, 50):.2f} ms')
    print(f'latency p95    : {percentile(latencies, 95):.2f} ms')
    print(f'latency p99    : {percentile(latencies, 99):.2f} ms')
    print(f'latency p999   : {percentile(latencies, 99.9):.2f} ms')
    print(f'avg n_distinct : {distinct_total / n_q:.0f}')
    print(f'peak RSS       : {rss_peak_kb / 1024:.1f} MB')
    print(f'throughput     : {n_q / dt_total:.1f} q/s (single-thread)')

    f.close()


if __name__ == '__main__':
    main()
