"""Read-only access to the SIFT 1B forest built via the bench flow.

The 1B index at /mnt/mangrove/indexes/sift1b is NOT in LiveIndex /
MangroveCluster format — it was built with the Forest FFI directly
(see bench_sift1b.py), so serve_cluster cannot host it as-is.

This helper opens the 10 segments × 1000 trees × depth 30 forests
and gives you :
  - `open_sift1b()` → list of Forest objects (one per segment)
  - `search(forests, qvec, top_k=10, top_n=4000)` → top-k doc_ids
     via vote merge + L2 rerank against the bvecs base file

Run as CLI :
    python3 scripts/query_sift1b.py --n_queries 5 --top_k 10

Or import from your own script :
    from query_sift1b import open_sift1b, search
    fs = open_sift1b()
    ids = search(fs, qvec, top_k=10)
"""
from __future__ import annotations
import argparse, os, struct, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mangrove_ffi import Forest, set_gen_version, set_shared_scratch_pool


# Layout from bench_sift1b.py — kept in sync there
IDX_ROOT = '/mnt/mangrove/indexes/sift1b'
BASE     = '/home/chatelet/mangrove-search/bigann_base.bvecs'
QUERIES  = '/mnt/mangrove/datasets/sift1b/bigann_query.bvecs'

DIM      = 128
SUB_DIM  = 16
N_TREES  = 1000
DEPTH    = 30
N_SEGS   = 10
SEG_SIZE = 100_000_000          # docs per segment
N_DOC    = 1_000_000_000


def open_sift1b(idx_root: str = IDX_ROOT,
                shared_pool: bool = True) -> list[Forest]:
    """Open all 10 segments. ~5 s warm-up.
       `shared_pool=True` shares the big scratch buffer across forests
       (935 MB RSS for the full 1B instead of ~3.7 GB without)."""
    set_gen_version(3)
    set_shared_scratch_pool(shared_pool)
    forests = []
    for i in range(N_SEGS):
        sdir = f'{idx_root}/seg{i}'
        f = Forest(sdir, n_trees=N_TREES, dim=DIM, sub_dim=SUB_DIM,
                   depth=DEPTH, n_docs=N_DOC, gen_version=3)
        forests.append(f)
    return forests


def read_bvecs(path: str, n: int, d: int = DIM,
               offset_rows: int = 0) -> np.ndarray:
    """SIFT 1B base + queries are uint8 bvecs : per row [int32 dim][dim × uint8]."""
    out  = np.empty((n, d), dtype=np.float32)
    rb   = 4 + d
    with open(path, 'rb') as f:
        f.seek(offset_rows * rb)
        for i in range(n):
            f.read(4)
            out[i] = np.frombuffer(f.read(d), dtype=np.uint8).astype(np.float32)
    return out


def _l2_rerank(qvec: np.ndarray, doc_ids: list[int],
               base_path: str, top_k: int) -> list[int]:
    """Read each doc_id's vector from the base bvecs file, compute L2,
       return top_k doc_ids in ascending distance order."""
    rb     = 4 + DIM                       # bvecs row : int32 + uint8 × dim
    scored = []
    qv     = qvec.astype(np.float32, copy=False)
    with open(base_path, 'rb') as bf:
        for d in doc_ids:
            bf.seek(d * rb + 4)
            vec = np.frombuffer(bf.read(DIM), dtype=np.uint8).astype(np.float32)
            if len(vec) != DIM:
                continue
            scored.append((float(((vec - qv) ** 2).sum()), d))
    scored.sort()
    return [d for _, d in scored[:top_k]]


def search(forests: list[Forest], qvec: np.ndarray,
           top_k: int = 10, top_n: int = 4000,
           query_depth: int = 0,
           base_path: str = BASE,
           rerank: bool = True) -> list[int] | dict:
    """Search the 1B forest. Aggregates votes across all 10 segments,
       optionally L2-reranks the candidate pool against the base file.

       Returns the list of top_k doc_ids if `rerank=True`, otherwise a
       dict {doc_id: total_vote} for the pre-rerank candidate set."""
    if qvec.dtype != np.float32:
        qvec = qvec.astype(np.float32, copy=False)
    vote_acc: dict[int, int] = {}
    for f in forests:
        ids, votes, k = f.query(qvec, top_n=top_n, query_depth=query_depth)
        for j in range(k):
            d = int(ids[j])
            vote_acc[d] = vote_acc.get(d, 0) + int(votes[j])
    if not rerank:
        return vote_acc
    # Top candidates by vote, then L2 rerank
    top_cands = [d for d, _ in sorted(vote_acc.items(), key=lambda kv: -kv[1])[:top_n]]
    return _l2_rerank(qvec, top_cands, base_path, top_k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--idx_root',   default=IDX_ROOT)
    ap.add_argument('--queries',    default=QUERIES)
    ap.add_argument('--base',       default=BASE)
    ap.add_argument('--n_queries',  type=int, default=5)
    ap.add_argument('--top_k',      type=int, default=10)
    ap.add_argument('--top_n',      type=int, default=4000)
    ap.add_argument('--query_depth',type=int, default=0,
                    help='0 = native (build depth 30) ; lower → higher recall but slower')
    args = ap.parse_args()

    print(f'opening {args.idx_root} ...')
    t0 = time.time()
    forests = open_sift1b(args.idx_root)
    print(f'  10 segments opened in {time.time() - t0:.1f}s')

    print(f'reading {args.n_queries} queries from {args.queries} ...')
    qs = read_bvecs(args.queries, args.n_queries)

    print(f'\nrunning {args.n_queries} queries '
          f'(top_k={args.top_k} top_n={args.top_n} qd={args.query_depth}):')
    for i, q in enumerate(qs):
        t0 = time.time()
        ids = search(forests, q, top_k=args.top_k, top_n=args.top_n,
                     query_depth=args.query_depth, base_path=args.base)
        dt = (time.time() - t0) * 1000
        print(f'  q{i} : {dt:6.1f} ms  ids={ids[:5]}{"..." if len(ids) > 5 else ""}')


if __name__ == '__main__':
    main()
