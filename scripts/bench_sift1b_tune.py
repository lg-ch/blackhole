"""Sweep latency tuning knobs on SIFT 1B (10 segments) :
   - top_n : 4000 → 2000 → 1000 → 500
   - max_stable_rejects : 0 (off) → 100 → 50 → 25
   Reports recall@10 + p50/p95/p99 for each combo.
"""
from __future__ import annotations
import os, struct, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mangrove_ffi import (Forest, set_gen_version, set_shared_scratch_pool,
                          _lib)
from ctypes import c_int


def set_max_stable_rejects(n: int) -> None:
    _lib.mg_set_max_stable_rejects(n)


DIM, SUB_DIM, N_TREES, DEPTH, N_SEGS = 128, 16, 1000, 30, 10
N_DOC = 1_000_000_000
IDX_ROOT = '/mnt/mangrove/indexes/sift1b'
BASE     = '/home/chatelet/mangrove-search/bigann_base.bvecs'
QUERIES  = '/mnt/mangrove/datasets/sift1b/bigann_query.bvecs'
GT       = '/mnt/mangrove/datasets/sift1b/idx_1000M.ivecs'
N_Q      = 50    # smaller sweep for speed


def read_bvecs(path, n, d=DIM):
    out = np.empty((n, d), dtype=np.float32)
    with open(path, 'rb') as f:
        for i in range(n):
            f.read(4)
            out[i] = np.frombuffer(f.read(d), dtype=np.uint8).astype(np.float32)
    return out


def read_ivecs(path, n, k=10):
    out = np.empty((n, k), dtype=np.int32)
    with open(path, 'rb') as f:
        first_k = struct.unpack('<i', f.read(4))[0]
        f.seek(0)
        for i in range(n):
            f.read(4)
            out[i] = np.frombuffer(f.read(k * 4), dtype=np.int32)[:k]
            f.read(4 * (first_k - k))
    return out


def main():
    set_gen_version(3)
    set_shared_scratch_pool(True)
    print('opening 10 forests ...')
    forests = [Forest(f'{IDX_ROOT}/seg{i}', n_trees=N_TREES, dim=DIM,
                      sub_dim=SUB_DIM, depth=DEPTH, n_docs=N_DOC,
                      gen_version=3) for i in range(N_SEGS)]
    queries = read_bvecs(QUERIES, N_Q)
    gt      = read_ivecs(GT, N_Q, k=10)

    # Warm cache
    set_max_stable_rejects(0)
    for q in queries[:5]:
        for f in forests: f.query(q, top_n=4000)

    print(f'\n{"top_n":>6} {"stable_rej":>10} {"recall":>8} {"p50ms":>8} {"p95ms":>8} {"p99ms":>8}')
    print('-' * 56)

    configs = [
        (4000,    0),    # baseline
        (4000,  100),
        (4000,   50),
        (4000,   25),
        (2000,    0),
        (2000,   50),
        (1000,    0),
        (1000,   50),
        ( 500,    0),
    ]
    for top_n, stable_rej in configs:
        set_max_stable_rejects(stable_rej)
        lats, recalls = [], []
        for qi in range(N_Q):
            q = queries[qi]
            t0 = time.time()
            vote_acc: dict[int, int] = {}
            for f in forests:
                ids, votes, n = f.query(q, top_n=top_n)
                for j in range(n):
                    vote_acc[int(ids[j])] = vote_acc.get(int(ids[j]), 0) + int(votes[j])
            items = sorted(vote_acc.items(), key=lambda kv: -kv[1])[:top_n]
            cand = np.array([k for k, _ in items], dtype=np.int32)
            top10 = forests[0].rerank_l2(BASE, q, cand, top_k=10)
            lats.append((time.time() - t0) * 1000)
            s = set(int(x) for x in gt[qi])
            recalls.append(sum(1 for x in top10 if int(x) in s) / 10)
        print(f'{top_n:>6} {stable_rej:>10} {np.mean(recalls):>8.4f} '
              f'{np.percentile(lats, 50):>8.0f} {np.percentile(lats, 95):>8.0f} '
              f'{np.percentile(lats, 99):>8.0f}')

    for f in forests: f.close()


if __name__ == '__main__':
    main()
