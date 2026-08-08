"""LSM segments PoC on SIFT 1M.

Loads N segments × 100k docs each (built separately as sub-indexes), queries
across them via mg_forest_query_multi (pairwise-seed multi-index), measures
recall vs the single-forest baseline.

Usage:
    python3 scripts/test_lsm.py \
        --segments_dir /tmp/sift1m_lsm \
        --n_segments 10 --docs_per_seg 100000 \
        --n_trees 1000 --depth 15 --sub_dim 16 \
        --queries sift/sift_query.fvecs \
        --gt sift/sift_groundtruth.ivecs \
        --base sift/sift_base.fvecs \
        --n_queries 100 --top_k 10 --top_n 4000
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
from mangrove_ffi import Forest, query_multi, set_gen_version  # noqa: E402


def percentile(values, p):
    s = sorted(values)
    if not s:
        return 0.0
    k = (len(s) - 1) * p / 100.0
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def read_fvecs(path, n, dim):
    out = np.empty((n, dim), dtype=np.float32)
    with open(path, 'rb') as f:
        for i in range(n):
            f.read(4)
            out[i] = np.frombuffer(f.read(dim * 4), dtype=np.float32)
    return out


def read_gt(path, n, top_k):
    with open(path, 'rb') as f:
        k = struct.unpack('<i', f.read(4))[0]
    out = np.empty((n, top_k), dtype=np.int32)
    with open(path, 'rb') as f:
        for i in range(n):
            f.read(4)
            out[i] = np.frombuffer(f.read(k * 4), dtype=np.int32)[:top_k]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--segments_dir', required=True)
    ap.add_argument('--n_segments', type=int, required=True)
    ap.add_argument('--docs_per_seg', type=int, required=True)
    ap.add_argument('--n_trees', type=int, required=True)
    ap.add_argument('--depth', type=int, required=True)
    ap.add_argument('--sub_dim', type=int, default=16)
    ap.add_argument('--gen', type=int, default=3)
    ap.add_argument('--queries', required=True)
    ap.add_argument('--gt', required=True)
    ap.add_argument('--base', required=True)
    ap.add_argument('--n_queries', type=int, default=100)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--top_n', type=int, default=4000)
    ap.add_argument('--dim', type=int, default=128)
    args = ap.parse_args()

    queries = read_fvecs(args.queries, args.n_queries, args.dim)
    gt      = read_gt(args.gt, args.n_queries, args.top_k)

    # Open all segments. Each has n_docs = docs_per_seg, with global doc_ids
    # offset = i * docs_per_seg (so the index reports global ids, not slice-local).
    set_gen_version(args.gen)
    forests = []
    t_open = time.time()
    for i in range(args.n_segments):
        path = os.path.join(args.segments_dir, f'seg{i}')
        # The build wrote meta with n_docs = doc_offset + n_vecs. For multi-index
        # we pass the END doc_id range (i.e. (i+1) * docs_per_seg).
        n_docs_meta = (i + 1) * args.docs_per_seg
        f = Forest(path, n_trees=args.n_trees, dim=args.dim,
                   sub_dim=args.sub_dim, depth=args.depth,
                   n_docs=n_docs_meta, gen_version=args.gen)
        forests.append(f)
    print(f'opened {len(forests)} segments in {(time.time()-t_open)*1000:.0f} ms')

    # Warm up
    for q in queries[:5]:
        query_multi(forests, q, top_n=args.top_n)

    # Bench multi-query
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    latencies = []
    recalls = []
    for i in range(args.n_queries):
        q = queries[i]
        t0 = time.time()
        ids, votes, k = query_multi(forests, q, top_n=args.top_n)
        # L2 rerank using first segment's Forest (any segment has same base_path)
        topk = forests[0].rerank_l2(args.base, q, ids[:k], top_k=args.top_k)
        latencies.append((time.time() - t0) * 1000)
        gt_set = set(int(x) for x in gt[i])
        hits = sum(1 for x in topk if int(x) in gt_set)
        recalls.append(hits / args.top_k)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    print(f'\n=== LSM {args.n_segments} segments × {args.docs_per_seg} docs ===')
    print(f'recall@{args.top_k} : {np.mean(recalls):.4f}')
    print(f'latency mean   : {np.mean(latencies):.2f} ms')
    print(f'latency p50    : {percentile(latencies, 50):.2f} ms')
    print(f'latency p99    : {percentile(latencies, 99):.2f} ms')
    print(f'peak RSS       : {rss / 1024:.1f} MB')
    # Disk
    seg_size_mb = 0
    for i in range(args.n_segments):
        seg_dir = os.path.join(args.segments_dir, f'seg{i}')
        for root, _, files in os.walk(seg_dir):
            for f in files:
                seg_size_mb += os.path.getsize(os.path.join(root, f))
    print(f'total disk     : {seg_size_mb / 1024**2:.1f} MB')

    for f in forests:
        f.close()


if __name__ == '__main__':
    main()
