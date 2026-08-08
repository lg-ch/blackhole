#!/usr/bin/env python3
"""
Build a bruteforce GT (top-100 L2) for N queries vs the arxiv 2M corpus.

Outputs (in datasets/arxiv/):
  bench_q.fvecs   - the query vectors (N × dim, fvecs)
  bench_gt.ivecs  - top-100 L2 doc_ids per query (ivecs: [K=100][int32×100])

Reproducible: query ids = numpy.default_rng(seed).integers(0, N, n_q).
"""
import argparse, os, struct, sys, time
import numpy as np

ROOT = '/home/chatelet/mangrove-search'
BASE = f'{ROOT}/datasets/arxiv/arxiv_base.fvecs'
N    = 2_058_751
DIM  = 768
GT_K = 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_q',  type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out',  default=f'{ROOT}/datasets/arxiv')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    qids = rng.integers(0, N, args.n_q).astype(np.int64)

    # mmap the fvecs as (N, DIM+1) float32 — first float per row is the dim
    # header reinterpreted; we slice it off with [:, 1:].
    print(f'mmap {BASE} ...', flush=True)
    arr = np.memmap(BASE, dtype='float32', mode='r').reshape(N, DIM + 1)
    base = arr[:, 1:]  # (N, DIM)

    print(f'precompute ||base||² ...', flush=True)
    t0 = time.time()
    sq_base = np.einsum('ij,ij->i', base, base).astype(np.float32)
    print(f'  done in {time.time()-t0:.1f}s', flush=True)

    qvecs = base[qids].copy()                     # (n_q, DIM)
    sq_q  = np.einsum('ij,ij->i', qvecs, qvecs)   # (n_q,)

    print(f'compute distances + top-{GT_K} for {args.n_q} queries ...', flush=True)
    gt = np.zeros((args.n_q, GT_K), dtype=np.int32)
    t0 = time.time()
    BATCH = 16  # process queries in batches to keep memory bounded
    for qb in range(0, args.n_q, BATCH):
        qe = min(qb + BATCH, args.n_q)
        # (b, N) = ||base||² - 2·base·qᵀ + ||q||²
        d = sq_base[None, :] - 2.0 * (qvecs[qb:qe] @ base.T) + sq_q[qb:qe, None]
        for i in range(qb, qe):
            # argpartition is O(N), then sort the K winners
            idx = np.argpartition(d[i - qb], GT_K)[:GT_K]
            order = idx[np.argsort(d[i - qb][idx])]
            gt[i] = order.astype(np.int32)
        print(f'  q {qe}/{args.n_q}  ({(qe)/(time.time()-t0):.1f} q/s)', flush=True)

    # Write qvecs (fvecs)
    qpath = f'{args.out}/bench_q.fvecs'
    with open(qpath, 'wb') as f:
        hdr = struct.pack('<i', DIM)
        for v in qvecs:
            f.write(hdr); f.write(v.astype(np.float32).tobytes())
    print(f'wrote {qpath} ({os.path.getsize(qpath)/1024:.1f} KB)')

    # Write GT (ivecs)
    gpath = f'{args.out}/bench_gt.ivecs'
    with open(gpath, 'wb') as f:
        hdr = struct.pack('<i', GT_K)
        for row in gt:
            f.write(hdr); f.write(row.astype(np.int32).tobytes())
    print(f'wrote {gpath} ({os.path.getsize(gpath)/1024:.1f} KB)')

    # Also dump qids for later filter checks
    ipath = f'{args.out}/bench_qids.npy'
    np.save(ipath, qids)
    print(f'wrote {ipath}')


if __name__ == '__main__':
    main()
