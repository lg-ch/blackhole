#!/usr/bin/env python3
"""
Bruteforce L2 fallback for small filters.

When `filter_card × dim × 4 < THRESHOLD_BYTES`, reading the subset and
running a direct L2 is cheaper (and recall = 100%) than driving the forest
through low-qd / high-top_n. Threshold tuned per dim — for dim=768 the
elbow is ~50-100k docs (≈ 150-300 MB to read).
"""
import argparse, os, struct, sys, time
import numpy as np
from clickhouse_driver import Client

ROOT  = '/home/chatelet/mangrove-search'
BASE  = f'{ROOT}/datasets/arxiv/arxiv_base.fvecs'
QVECS = f'{ROOT}/datasets/arxiv/bench_q.fvecs'
GT    = f'{ROOT}/datasets/arxiv/bench_gt.ivecs'
N_DOCS = 2_058_751
DIM    = 768

ch = Client('127.0.0.1')


def fetch_allowed(where: str) -> np.ndarray:
    """Sorted np.int64 array of internal_ids matching the WHERE."""
    rows = ch.execute(f'SELECT internal_id FROM mangrove.docs WHERE {where}')
    a = np.fromiter((r[0] for r in rows), dtype=np.int64)
    a.sort()
    return a


def bruteforce_topk(qvec: np.ndarray, allowed: np.ndarray,
                    base_mmap: np.memmap, top_k: int) -> list[int]:
    if len(allowed) == 0:
        return []
    sub = base_mmap[allowed, 1:]                  # (n_allowed, DIM)
    sq  = np.einsum('ij,ij->i', sub, sub)         # ||v||²
    qq  = float(np.dot(qvec, qvec))
    d   = sq - 2.0 * (sub @ qvec) + qq            # L2² up to constant qq
    k   = min(top_k, len(allowed))
    idx = np.argpartition(d, k - 1)[:k]
    order = idx[np.argsort(d[idx])]
    return [int(allowed[i]) for i in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--where', required=True)
    ap.add_argument('--n_q',   type=int, default=100)
    ap.add_argument('--top_k', type=int, default=10)
    args = ap.parse_args()

    allowed = fetch_allowed(args.where)
    bytes_read = len(allowed) * DIM * 4
    print(f'filter card={len(allowed)}  est. read={bytes_read/1024**2:.0f} MB')

    # Load queries
    qsize = os.path.getsize(QVECS) // (4 + DIM * 4)
    qvecs = np.fromfile(QVECS, dtype='float32').reshape(qsize, DIM + 1)[:, 1:]
    n_q = min(args.n_q, qsize)

    # GT for recall check (filter-aware)
    # Build minimal GT on-the-fly for this filter
    print('build filter-aware GT bruteforce ...', flush=True)
    base_mmap = np.memmap(BASE, dtype='float32', mode='r').reshape(N_DOCS, DIM + 1)
    sub_for_gt = base_mmap[allowed, 1:].astype(np.float32)
    sq_sub = np.einsum('ij,ij->i', sub_for_gt, sub_for_gt)
    gt_top100 = np.zeros((n_q, 100), dtype=np.int64)
    for q in range(n_q):
        d = sq_sub - 2 * (sub_for_gt @ qvecs[q]) + float(np.dot(qvecs[q], qvecs[q]))
        k = min(100, len(allowed))
        idx = np.argpartition(d, k - 1)[:k]
        order = idx[np.argsort(d[idx])]
        gt_top100[q, :k] = allowed[order]

    # Bruteforce search (single forward pass; same numbers as GT but timed)
    print('bruteforce timing ...', flush=True)
    per_q_ms = []
    for q in range(n_q):
        t0 = time.time()
        # We could cache sub_for_gt / sq_sub above for the timing, but the prod
        # path would re-read on each query (different filter per query).
        # Mimic that by reading fresh.
        sub2 = base_mmap[allowed, 1:]
        sq2  = np.einsum('ij,ij->i', sub2, sub2)
        d    = sq2 - 2.0 * (sub2 @ qvecs[q]) + float(np.dot(qvecs[q], qvecs[q]))
        k    = min(args.top_k, len(allowed))
        idx  = np.argpartition(d, k - 1)[:k]
        order = idx[np.argsort(d[idx])]
        _    = [int(allowed[i]) for i in order]
        per_q_ms.append((time.time() - t0) * 1000)

    a = np.asarray(per_q_ms)
    print(f'\nbruteforce {n_q} queries, filter={len(allowed)} docs:')
    print(f'  p50={np.percentile(a,50):6.1f} ms')
    print(f'  p95={np.percentile(a,95):6.1f} ms')
    print(f'  p99={np.percentile(a,99):6.1f} ms')
    print(f'  max={a.max():6.1f} ms')
    print(f'  recall@10 = 1.0000 (exact by construction)')


if __name__ == '__main__':
    main()
