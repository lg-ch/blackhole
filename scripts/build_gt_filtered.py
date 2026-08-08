#!/usr/bin/env python3
"""
Build a *filter-aware* GT : top-100 L2 over the subset that satisfies the
WHERE clause. Without this, comparing pre-filtered ANN against an
unfiltered GT mechanically forces recall ≈ density (the GT NNs simply
aren't allowed).

Usage : build_gt_filtered.py --where "year=2007" --out bench_gt_2007.ivecs
"""
import argparse, os, struct, sys, time
import numpy as np
from clickhouse_driver import Client

ROOT  = '/home/chatelet/mangrove-search'
BASE  = f'{ROOT}/datasets/arxiv/arxiv_base.fvecs'
QVECS = f'{ROOT}/datasets/arxiv/bench_q.fvecs'
N     = 2_058_751
DIM   = 768
GT_K  = 100

ch = Client('127.0.0.1')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--where', required=True)
    ap.add_argument('--out',   required=True)
    args = ap.parse_args()

    print(f'fetch allowed ids for WHERE {args.where} ...', flush=True)
    rows = ch.execute(f'SELECT internal_id FROM mangrove.docs WHERE {args.where}')
    allowed = np.fromiter((r[0] for r in rows), dtype=np.int64)
    allowed.sort()
    print(f'  card = {len(allowed)}', flush=True)
    if len(allowed) == 0:
        sys.exit('empty filter')

    # Read queries (fvecs)
    qsize = os.path.getsize(QVECS) // (4 + DIM * 4)
    qvecs = np.fromfile(QVECS, dtype='float32').reshape(qsize, DIM + 1)[:, 1:]
    print(f'queries : {qsize}', flush=True)

    # mmap base, slice to allowed subset (load into RAM as contiguous block)
    print('loading allowed vectors into RAM ...', flush=True)
    t0 = time.time()
    base = np.memmap(BASE, dtype='float32', mode='r').reshape(N, DIM + 1)[:, 1:]
    sub = np.empty((len(allowed), DIM), dtype='float32')
    CHUNK = 100_000
    for i in range(0, len(allowed), CHUNK):
        idx = allowed[i : i + CHUNK]
        sub[i : i + len(idx)] = base[idx]
    print(f'  loaded {sub.nbytes/1024**2:.1f} MB in {time.time()-t0:.1f}s', flush=True)

    print('compute top-100 L2 over subset ...', flush=True)
    sq_sub = np.einsum('ij,ij->i', sub, sub).astype(np.float32)
    sq_q   = np.einsum('ij,ij->i', qvecs, qvecs)

    gt = np.zeros((qsize, GT_K), dtype=np.int32)
    K = min(GT_K, len(allowed))
    t0 = time.time()
    BATCH = 16
    for qb in range(0, qsize, BATCH):
        qe = min(qb + BATCH, qsize)
        d = sq_sub[None, :] - 2.0 * (qvecs[qb:qe] @ sub.T) + sq_q[qb:qe, None]
        for i in range(qb, qe):
            if K >= len(allowed):
                order = np.argsort(d[i - qb])
            else:
                idx = np.argpartition(d[i - qb], K)[:K]
                order = idx[np.argsort(d[i - qb][idx])]
            # Map back to global doc_id
            gt[i, :K] = allowed[order[:K]]
            gt[i, K:] = -1  # pad with sentinel if filter < 100
    print(f'  GT in {time.time()-t0:.1f}s', flush=True)

    with open(args.out, 'wb') as f:
        hdr = struct.pack('<i', GT_K)
        for row in gt:
            f.write(hdr); f.write(row.astype(np.int32).tobytes())
    print(f'wrote {args.out} ({os.path.getsize(args.out)/1024:.1f} KB)')


if __name__ == '__main__':
    main()
