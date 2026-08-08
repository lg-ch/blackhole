"""End-to-end test : client-side leaf computation + server-side merge.

Setup :
  - Build a forest from SIFT 1M
  - For each query :
     A) baseline : server-side full traversal (qvec → leaves → merge)
     B) client-side : compute leaves locally, send to server via
        set_external_leaves(), call query, server skips traversal
  - Both paths must return IDENTICAL candidate sets (the K-way merge
    is deterministic given the same leaves).
"""
from __future__ import annotations
import os, shutil, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mangrove.traversal import compute_leaves
from mangrove_ffi import Forest, set_gen_version, set_external_leaves


DIM, SUB_DIM, DEPTH, N_TREES = 128, 16, 14, 50
N_DOC = 100_000
IDX = '/tmp/clientside_idx'


def build():
    if not os.path.exists(IDX):
        subprocess.check_call([
            './rpforest', '--dim', str(DIM), '--sub_dim', str(SUB_DIM),
            '--gen', 'v3', '--doc_offset', '0', '--doc_count', str(N_DOC),
            'build', 'sift/sift_base.fvecs', IDX,
            str(N_TREES), str(DEPTH),
        ], stdout=subprocess.DEVNULL)


def read_fvecs(path, n, d=DIM):
    out = np.empty((n, d), dtype=np.float32)
    with open(path, 'rb') as f:
        for i in range(n):
            f.read(4); out[i] = np.frombuffer(f.read(d * 4), dtype=np.float32)
    return out


def main():
    build()
    set_gen_version(3)
    f = Forest(IDX, n_trees=N_TREES, dim=DIM, sub_dim=SUB_DIM,
               depth=DEPTH, n_docs=N_DOC, gen_version=3)
    queries = read_fvecs('sift/sift_query.fvecs', 50)

    diffs = 0
    for qi in range(50):
        q = queries[qi]

        # A) server-side traversal (baseline)
        set_external_leaves(None)
        ids_a, votes_a, n_a = f.query(q, top_n=500)
        topa = set(int(x) for x in ids_a[:n_a])

        # B) client-side leaves
        leaves = compute_leaves(q, n_trees=N_TREES, depth=DEPTH,
                                sub_dim=SUB_DIM, dim=DIM, gen_version=3)
        set_external_leaves(leaves)
        ids_b, votes_b, n_b = f.query(q, top_n=500)
        topb = set(int(x) for x in ids_b[:n_b])

        # Reset
        set_external_leaves(None)

        if topa != topb:
            diffs += 1
            if diffs <= 3:
                print(f'  q{qi} diff: a-only={len(topa - topb)} b-only={len(topb - topa)}')

    print(f'\n50 queries : {diffs} differ (must be 0)')
    print('PASS' if diffs == 0 else 'FAIL')
    f.close()


if __name__ == '__main__':
    main()
