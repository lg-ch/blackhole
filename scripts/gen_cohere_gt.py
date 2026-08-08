"""Brute-force ground-truth generator for the Cohere wiki-en 41.5M corpus.

Picks N_Q query vectors (sampled randomly from the corpus itself by
default, or supplied as an external .fvecs), then computes exact L2
top-K against all docs by streaming chunks of the base file.

Output : an .ivecs file with one row per query, K int32 doc_ids each.
"""
from __future__ import annotations
import argparse, os, struct, sys, time
import numpy as np


def read_fvecs_rows(path: str, rows: list[int], dim: int) -> np.ndarray:
    """Read the given row indices from an .fvecs file."""
    row_bytes = 4 + dim * 4
    out = np.empty((len(rows), dim), dtype=np.float32)
    with open(path, 'rb') as f:
        for i, r in enumerate(sorted(set(rows))):
            f.seek(r * row_bytes + 4)  # skip the per-row dim header
            out[i] = np.frombuffer(f.read(dim * 4), dtype=np.float32)
    return out


def stream_chunks(path: str, dim: int, chunk: int, total: int):
    """Yield (start_id, vecs) chunks of size up to `chunk` from .fvecs."""
    row_bytes = 4 + dim * 4
    with open(path, 'rb') as f:
        # The header byte pattern is per-row [dim u32][dim*float32]. We strip
        # the dim header per row in batch by reshape + slice.
        i = 0
        while i < total:
            n = min(chunk, total - i)
            raw = f.read(n * row_bytes)
            if len(raw) < n * row_bytes:
                n = len(raw) // row_bytes
                if n == 0: break
            arr = np.frombuffer(raw[: n * row_bytes],
                                dtype=np.uint8).reshape(n, row_bytes)
            # bytes 4..end of each row = the float32 vector
            vecs = np.frombuffer(arr[:, 4:].tobytes(),
                                 dtype=np.float32).reshape(n, dim)
            yield i, vecs
            i += n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base',   required=True, help='cohere_en_35m.fvecs')
    ap.add_argument('--out',    required=True, help='output .ivecs')
    ap.add_argument('--n_docs', type=int, default=41488110)
    ap.add_argument('--dim',    type=int, default=1024)
    ap.add_argument('--n_q',    type=int, default=1000)
    ap.add_argument('--top_k',  type=int, default=100)
    ap.add_argument('--chunk',  type=int, default=200_000,
                    help='How many docs per matmul step (RAM × speed tradeoff)')
    ap.add_argument('--queries', default=None,
                    help='Optional path to external query fvecs '
                         '(default: sample n_q docs from the corpus itself)')
    ap.add_argument('--seed',   type=int, default=42)
    args = ap.parse_args()

    # 1. Build the query matrix.
    if args.queries:
        # Read full external query file
        sys.stderr.write(f'reading external queries from {args.queries} ...\n')
        with open(args.queries, 'rb') as f:
            blob = f.read()
        row_bytes = 4 + args.dim * 4
        n_q = len(blob) // row_bytes
        arr = np.frombuffer(blob, dtype=np.uint8).reshape(n_q, row_bytes)
        Q = np.frombuffer(arr[:, 4:].tobytes(),
                          dtype=np.float32).reshape(n_q, args.dim)
        q_ids = np.arange(n_q)
    else:
        rng = np.random.default_rng(args.seed)
        q_ids = rng.choice(args.n_docs, size=args.n_q, replace=False)
        q_ids.sort()
        sys.stderr.write(f'sampling {args.n_q} query docs from corpus ...\n')
        Q = read_fvecs_rows(args.base, q_ids.tolist(), args.dim)

    sys.stderr.write(
        f'  Q.shape = {Q.shape}, top_k = {args.top_k}, '
        f'chunk = {args.chunk}, n_docs = {args.n_docs}\n')

    n_q = Q.shape[0]
    # Pre-compute |q|² (broadcasted in L2 trick).
    q_norm2 = (Q * Q).sum(axis=1)               # (n_q,)

    # Top-K state : keep best K (d2, doc_id) per query, as parallel arrays.
    # Use a "max-heap by negative distance" semantically — we just track
    # the current worst-of-best per query and only insert if better.
    best_d2  = np.full((n_q, args.top_k), np.inf, dtype=np.float32)
    best_ids = np.full((n_q, args.top_k), -1,     dtype=np.int32)

    t0 = time.time()
    n_done = 0
    for start, chunk in stream_chunks(args.base, args.dim,
                                      args.chunk, args.n_docs):
        # L2² = |q|² + |x|² - 2 q.x  (matrix form)
        x_norm2 = (chunk * chunk).sum(axis=1)            # (chunk_n,)
        dots = Q @ chunk.T                                # (n_q, chunk_n)
        d2 = q_norm2[:, None] + x_norm2[None, :] - 2 * dots   # (n_q, chunk_n)
        # For each query, merge this chunk's distances with the running top_k.
        # Combine into a (n_q, top_k + chunk_n) array, then partial-sort.
        combined_d2  = np.concatenate([best_d2,
                                       d2.astype(np.float32)], axis=1)
        combined_ids = np.concatenate([best_ids,
                                       np.full((n_q, chunk.shape[0]),
                                               0, dtype=np.int32)], axis=1)
        combined_ids[:, args.top_k:] = (
            np.arange(chunk.shape[0], dtype=np.int32)[None, :] + start)
        idx = np.argpartition(combined_d2, args.top_k - 1, axis=1)[:, :args.top_k]
        rows = np.arange(n_q)[:, None]
        best_d2  = combined_d2[rows, idx]
        best_ids = combined_ids[rows, idx]
        n_done += chunk.shape[0]
        dt = time.time() - t0
        sys.stderr.write(
            f'  [{n_done/1e6:6.2f} / {args.n_docs/1e6:.2f}M docs] '
            f'{dt/60:5.1f} min ({n_done/dt:,.0f} docs/s)\n')

    # Sort each query's top_k by d2 ascending.
    order = np.argsort(best_d2, axis=1)
    rows = np.arange(n_q)[:, None]
    final_ids = best_ids[rows, order]

    # Write .ivecs : per row [int32 K][K * int32 doc_id]
    sys.stderr.write(f'writing {args.out} ...\n')
    with open(args.out, 'wb') as f:
        for row in final_ids:
            f.write(struct.pack('<i', args.top_k))
            f.write(row.astype(np.int32).tobytes())

    # Also save the query vectors for repeatability (so future bench can
    # use the same queries without re-sampling).
    qpath = args.out.rsplit('.', 1)[0] + '_queries.fvecs'
    sys.stderr.write(f'writing {qpath} ...\n')
    with open(qpath, 'wb') as f:
        for row in Q:
            f.write(struct.pack('<i', args.dim))
            f.write(row.tobytes())

    sys.stderr.write(
        f'DONE in {(time.time() - t0)/60:.1f} min, '
        f'{n_q} queries × top_{args.top_k} → {args.out}\n')


if __name__ == '__main__':
    main()
