"""Bench SIFT 1B multi-index : 10 segments × 100M docs each, depth 30.

Reports recall@10, latency p50/p95/p99, peak RSS.
Uses the per-segment query path (each forest at its own depth) +
L2 rerank.
"""
from __future__ import annotations
import os, struct, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mangrove_ffi import Forest, set_gen_version, set_shared_scratch_pool


IDX_ROOT  = '/mnt/mangrove/indexes/sift1b'
BASE      = '/home/chatelet/mangrove-search/bigann_base.bvecs'
QUERIES   = '/mnt/mangrove/datasets/sift1b/bigann_query.bvecs'
GT        = '/mnt/mangrove/datasets/sift1b/idx_1000M.ivecs'

DIM       = 128
SUB_DIM   = 16
N_TREES   = 1000
DEPTH     = 30
N_SEGS    = 10
SEG_SIZE  = 100_000_000
N_DOC     = 1_000_000_000

# Tune these to taste
N_QUERIES = 100
TOP_N     = 4000
TOP_K     = 10


def read_bvecs(path, n, d=DIM):
    """SIFT 1B base + queries are uint8 bvecs : per row [int32 dim][dim × uint8]."""
    out = np.empty((n, d), dtype=np.float32)
    row_bytes = 4 + d
    with open(path, 'rb') as f:
        for i in range(n):
            f.read(4)
            out[i] = np.frombuffer(f.read(d), dtype=np.uint8).astype(np.float32)
    return out


def read_ivecs(path, n, k=10):
    """Ground truth : per row [int32 k_total][k × int32 doc_id], we keep top-k."""
    out = np.empty((n, k), dtype=np.int32)
    with open(path, 'rb') as f:
        first_k = struct.unpack('<i', f.read(4))[0]
        f.seek(0)
        for i in range(n):
            f.read(4)
            out[i] = np.frombuffer(f.read(k * 4), dtype=np.int32)[:k]
            f.read(4 * (first_k - k))   # skip rest of row
    return out


def rss_mb() -> float:
    with open('/proc/self/status') as f:
        for ln in f:
            if ln.startswith('VmRSS:'):
                return float(ln.split()[1]) / 1024.0
    return -1


def main():
    print(f'=== SIFT 1B bench : {N_SEGS} segments × {SEG_SIZE:,} docs × '
          f'{N_TREES} trees × depth {DEPTH} ===\n')
    print('opening forests ...')
    set_gen_version(3)
    set_shared_scratch_pool(True)  # one bytes_buf/docs_buf for all forests
    forests = []
    t_open = time.time()
    for i in range(N_SEGS):
        sdir = f'{IDX_ROOT}/seg{i}'
        f = Forest(sdir, n_trees=N_TREES, dim=DIM, sub_dim=SUB_DIM,
                   depth=DEPTH, n_docs=N_DOC, gen_version=3)
        forests.append(f)
    print(f'  {N_SEGS} forests opened in {time.time() - t_open:.1f}s')
    print(f'  RSS after open : {rss_mb():.1f} MB')

    queries = read_bvecs(QUERIES, N_QUERIES)
    gt      = read_ivecs(GT, N_QUERIES, k=TOP_K)
    base    = None   # don't load 128 GB of base into RAM, rerank uses Forest's

    # Warm cache : 10 throw-away queries
    print('warming cache (10 queries) ...')
    for q in queries[:10]:
        for f in forests: f.query(q, top_n=TOP_N)

    print(f'\n=== bench : {N_QUERIES} queries × top_n={TOP_N} × top_k={TOP_K} ===')
    lats = []
    recalls = []
    rss_peak = rss_mb()
    for qi in range(N_QUERIES):
        q = queries[qi]
        t0 = time.time()
        vote_acc: dict[int, int] = {}
        for f in forests:
            ids, votes, n = f.query(q, top_n=TOP_N)
            for j in range(n):
                vote_acc[int(ids[j])] = vote_acc.get(int(ids[j]), 0) + int(votes[j])
        items = sorted(vote_acc.items(), key=lambda kv: -kv[1])[:TOP_N]
        cand_ids = np.array([k for k, _ in items], dtype=np.int32)
        # L2 rerank via the first forest (any will do, all share base_path)
        top10 = forests[0].rerank_l2(BASE, q, cand_ids, top_k=TOP_K)
        lats.append((time.time() - t0) * 1000)
        s = set(int(x) for x in gt[qi])
        recalls.append(sum(1 for x in top10 if int(x) in s) / TOP_K)
        rss_peak = max(rss_peak, rss_mb())
        if (qi + 1) % 10 == 0:
            sys.stderr.write(
                f'  [{qi+1}/{N_QUERIES}] recall_so_far={np.mean(recalls):.4f} '
                f'lat_p50={np.percentile(lats, 50):.0f}ms '
                f'rss={rss_mb():.0f} MB\n')

    print(f'\n=== RESULTS ===')
    print(f'  queries        : {N_QUERIES}')
    print(f'  recall@{TOP_K}      : {np.mean(recalls):.4f}')
    print(f'  recall p10     : {np.percentile(recalls, 10):.4f}')
    print(f'  latency p50    : {np.percentile(lats, 50):.1f} ms')
    print(f'  latency p95    : {np.percentile(lats, 95):.1f} ms')
    print(f'  latency p99    : {np.percentile(lats, 99):.1f} ms')
    print(f'  latency p999   : {np.percentile(lats, 99.9):.1f} ms')
    print(f'  peak RSS       : {rss_peak:.1f} MB')

    for f in forests: f.close()


if __name__ == '__main__':
    main()
