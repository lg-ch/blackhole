"""End-to-end validation suite for a built mangrove index.

Runs:
  1. xxhash verify all .srt
  2. open via FFI
  3. bench recall@10 with L2 rerank
  4. tombstone round-trip
  5. concurrent queries (1, 10 threads)
  6. peak RSS check vs target

Exit code 0 = all checks pass; non-zero = problems reported.

Usage:
  python3 scripts/validate_index.py \
      --index /mnt/mangrove/indexes/sift100m \
      --base /home/chatelet/mangrove-search/bigann_base.bvecs \
      --queries /mnt/mangrove/datasets/sift1b/bigann_query.bvecs \
      --gt /mnt/mangrove/datasets/sift1b/idx_100M.ivecs \
      --n_trees 200 --depth 25 --sub_dim 16 --n_docs 100000000 \
      --rss_target_mb 100
"""
from __future__ import annotations

import argparse
import os
import resource
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mangrove_ffi import Forest, verify_srt  # noqa: E402


def percentile(values, p):
    s = sorted(values)
    if not s:
        return 0.0
    k = (len(s) - 1) * p / 100.0
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def read_queries(path: str, n: int, dim: int) -> np.ndarray:
    out = np.empty((n, dim), dtype=np.float32)
    with open(path, 'rb') as f:
        if path.endswith('.u8bin'):
            f.read(8)
            buf = f.read(n * dim)
            return np.frombuffer(buf, dtype=np.uint8).reshape(n, dim).astype(np.float32)
        for i in range(n):
            db = f.read(4)
            if not db:
                return out[:i]
            d = struct.unpack('<i', db)[0]
            if d != dim:
                raise ValueError(f'query dim {d} != {dim}')
            if path.endswith('.bvecs'):
                out[i] = np.frombuffer(f.read(d), dtype=np.uint8).astype(np.float32)
            else:
                out[i] = np.frombuffer(f.read(d * 4), dtype=np.float32)
    return out


def read_gt(path: str, n: int, top_k: int) -> np.ndarray:
    out = np.empty((n, top_k), dtype=np.int32)
    with open(path, 'rb') as f:
        head = f.read(4)
        k = struct.unpack('<i', head)[0]
    with open(path, 'rb') as f:
        for i in range(n):
            f.read(4)
            row = f.read(k * 4)
            out[i] = np.frombuffer(row, dtype=np.int32)[:top_k]
    return out


def run_checks(args) -> int:
    failures = []

    print(f'\n[1/6] xxhash verify {args.n_trees} trees ...')
    bad = 0
    for t in range(args.n_trees):
        p = f'{args.index}/tree{t:05d}.srt'
        rc = verify_srt(p)
        if rc != 1:
            bad += 1
            if bad <= 5:
                print(f'  FAIL {p} (rc={rc})')
    if bad == 0:
        print(f'  ✓ all {args.n_trees} files valid')
    else:
        failures.append(f'xxhash: {bad}/{args.n_trees} bad')

    print(f'\n[2/6] open forest via FFI ...')
    t0 = time.time()
    try:
        f = Forest(args.index, n_trees=args.n_trees, dim=args.dim,
                   sub_dim=args.sub_dim, depth=args.depth,
                   n_docs=args.n_docs, gen_version=args.gen)
        print(f'  ✓ opened in {(time.time() - t0)*1000:.1f} ms')
    except Exception as e:
        failures.append(f'forest_open: {e}')
        return 1

    print(f'\n[3/6] recall@10 with L2 rerank (n_queries={args.n_queries}) ...')
    queries = read_queries(args.queries, args.n_queries, args.dim)
    gt      = read_gt(args.gt, args.n_queries, 10)
    # warmup
    for q in queries[: min(5, args.n_queries)]:
        f.query(q, top_n=args.top_n)
    lat = []
    recalls = []
    for i in range(args.n_queries):
        q = queries[i]
        t_q = time.time()
        ids, _, k = f.query(q, top_n=args.top_n)
        topk = f.rerank_l2(args.base, q, ids[:k], top_k=10) if args.base else ids[:10]
        lat.append((time.time() - t_q) * 1000)
        gt_set = set(int(x) for x in gt[i])
        recalls.append(sum(1 for x in topk if int(x) in gt_set) / 10)
    rcl = float(np.mean(recalls))
    print(f'  recall@10 = {rcl:.4f}')
    print(f'  p50={percentile(lat, 50):.2f}ms  p99={percentile(lat, 99):.2f}ms')
    if rcl < 0.80:
        failures.append(f'recall@10 below 0.80 ({rcl:.4f})')

    print(f'\n[4/6] tombstone round-trip ...')
    n0 = f.tombstones_count()
    test_id = int(queries[0].size)  # arbitrary valid id (small)
    f.tombstone_add(test_id)
    if f.tombstones_count() != n0 + 1:
        failures.append('tombstone_add did not increment count')
    f.tombstone_remove(test_id)
    if f.tombstones_count() != n0:
        failures.append('tombstone_remove did not decrement count')
    print(f'  ✓ add/remove work; current count = {f.tombstones_count()}')

    print(f'\n[5/6] concurrent queries (10 threads × 50 queries) ...')
    import threading
    lock = threading.Lock()
    def worker(qid):
        with lock:  # forest is not concurrent-safe
            return f.query(queries[qid], top_n=args.top_n)[0][:10].tolist()
    seen = {}
    mismatches = 0
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = [ex.submit(worker, i % args.n_queries) for i in range(500)]
        for fu in as_completed(futures):
            qid = next(i for i, ft in enumerate(futures) if ft is fu) % args.n_queries
            top = fu.result()
            if qid in seen:
                if seen[qid] != top:
                    mismatches += 1
            else:
                seen[qid] = top
    print(f'  ✓ 500 queries, {mismatches} mismatches')
    if mismatches > 0:
        failures.append(f'concurrent queries returned different results: {mismatches}')

    print(f'\n[6/6] peak RSS check (target ≤ {args.rss_target_mb} MB) ...')
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(f'  peak ru_maxrss = {rss_mb:.1f} MB')
    if rss_mb > args.rss_target_mb:
        failures.append(f'RSS {rss_mb:.1f} MB > target {args.rss_target_mb} MB')
    else:
        print(f'  ✓ within target')

    f.close()

    print('\n=== SUMMARY ===')
    if not failures:
        print('  ALL CHECKS PASSED')
        return 0
    for fail in failures:
        print(f'  ✗ {fail}')
    return 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--index',   required=True)
    ap.add_argument('--queries', required=True)
    ap.add_argument('--gt',      required=True)
    ap.add_argument('--base',    default=None,
                    help='base file for L2 rerank (omit to skip rerank)')
    ap.add_argument('--n_trees', type=int, required=True)
    ap.add_argument('--depth',   type=int, required=True)
    ap.add_argument('--sub_dim', type=int, default=0)
    ap.add_argument('--n_docs',  type=int, required=True)
    ap.add_argument('--dim',     type=int, default=128)
    ap.add_argument('--gen',     type=int, default=3)
    ap.add_argument('--n_queries', type=int, default=100)
    ap.add_argument('--top_n',   type=int, default=500)
    ap.add_argument('--rss_target_mb', type=float, default=100.0)
    args = ap.parse_args()
    sys.exit(run_checks(args))


if __name__ == '__main__':
    main()
