"""E2E : live-insert routing on a MEDIAN-built index.

The stateless FFI traversal helpers (mg_traverse_sub & co) are the live
insert routing path : SDK computes (tree, leaf) per doc, then hot_append.
On a median-built index they used to route sign-split (θ=0) → inserted
docs stranded in leaves the query never visits.

This test builds a tiny median index, then checks doc-vs-build parity :
the leaf where `rpforest build` actually stored each doc (ground truth,
read back via mg_leaf_docs) must equal the leaf mg_traverse_sub routes
that doc's vector to.

  1. WITHOUT load_live_medians : mismatch expected (demonstrates the bug)
  2. WITH    load_live_medians : exact parity  (validates the fix)
  3. probes_scored main path == traverse_sub  (insert ↔ query coherent)
  4. FULL live chain on the median index : route (medians armed) →
     hot_append → query_pathrank finds the doc. Negative control : docs
     routed sign-split (medians cleared) into the same overlay stay
     invisible to the query — the exact production failure mode.
  5. classic index (no medians.bin), helpers unarmed : exact parity
     (guards the default sign-split path against regression)

Cheap on purpose : 20k docs, dim 64, 8 trees, depth 10 — runs in seconds,
negligible next to any production build running on the box.
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mangrove_ffi as mf                                   # noqa: E402
from mangrove_ffi import Forest, set_gen_version            # noqa: E402
from ctypes import c_char_p, c_int                          # noqa: E402

RPFOREST = os.path.join(os.path.dirname(HERE), 'rpforest')

N_DOCS    = 20_000
DIM       = 64
SUB_DIM   = 16
N_TREES   = 8
DEPTH     = 10
MED_DEPTH = 6
GEN_V     = 3
SAMPLE_N  = 5_000
N_CHECK   = 400          # docs sampled for the parity checks

# test-only bindings (not in mangrove_ffi's public surface)
from ctypes import c_void_p, c_uint32                       # noqa: E402
mf._lib.mg_calibrate_medians.argtypes = [c_char_p, c_char_p, c_int, c_int,
                                         c_int, c_int, c_int]
mf._lib.mg_calibrate_medians.restype = c_int
mf._lib.mg_hot_init.argtypes   = [c_int, c_int, c_char_p]
mf._lib.mg_hot_init.restype    = c_void_p
mf._lib.mg_hot_free.argtypes   = [c_void_p]
mf._lib.mg_hot_free.restype    = None
mf._lib.mg_hot_append.argtypes = [c_void_p, c_int, c_uint32, c_uint32]
mf._lib.mg_hot_append.restype  = c_int
mf._lib.mg_forest_set_hot_overlay.argtypes = [c_void_p]
mf._lib.mg_forest_set_hot_overlay.restype  = None
mf._lib.mg_hot_compact_all.argtypes = [c_void_p, c_void_p, c_char_p]
mf._lib.mg_hot_compact_all.restype  = c_int


def make_vecs(n, dim, seed):
    """Clustered, non-centered data → node medians clearly ≠ 0, so
    sign-split and median routing genuinely disagree."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((20, dim)).astype(np.float32)
    which = rng.integers(0, 20, size=n)
    v = centers[which] + 0.15 * rng.standard_normal((n, dim)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return np.ascontiguousarray(v, dtype=np.float32)


def write_fvecs(path, arr):
    n, dim = arr.shape
    out = np.empty((n, dim + 1), dtype=np.float32)
    out[:, 0].view(np.int32)[:] = dim
    out[:, 1:] = arr
    out.tofile(path)


def build_index(base_path, idir, with_medians):
    os.makedirs(idir, exist_ok=True)
    if with_medians:
        set_gen_version(GEN_V)   # calibration MUST match build gen_version
        rc = mf._lib.mg_calibrate_medians(base_path.encode(), idir.encode(),
                                          N_TREES, DIM, SUB_DIM,
                                          MED_DEPTH, SAMPLE_N)
        assert rc == 0, f'mg_calibrate_medians rc={rc}'
        assert os.path.exists(os.path.join(idir, 'medians.bin'))
    subprocess.run([RPFOREST, 'build', base_path, idir,
                    str(N_TREES), str(DEPTH),
                    '--sub_dim', str(SUB_DIM), '--gen', f'v{GEN_V}',
                    '--dim', str(DIM)],
                   check=True, capture_output=True)


def built_leaf_of_docs(f, tree, check_ids):
    """Ground truth : leaf where the build stored each doc, by scanning
    every leaf of `tree` once."""
    want = set(int(d) for d in check_ids)
    got = {}
    for leaf in range(1 << DEPTH):
        for d in mf.leaf_docs(f._h, tree, leaf):
            if int(d) in want:
                got[int(d)] = leaf
    assert len(got) == len(want), f'{len(want)-len(got)} docs not found in tree {tree}'
    return got


def insert_leaf(vec, tree):
    node = mf.traverse_sub(vec, SUB_DIM, DEPTH, tree)
    return node - ((1 << DEPTH) - 1)


def parity(f, vecs, check_ids, trees=(0, N_TREES - 1)):
    """% of (doc, tree) pairs where insert routing ≠ build placement."""
    mism = tot = 0
    for t in trees:
        truth = built_leaf_of_docs(f, t, check_ids)
        for d in check_ids:
            tot += 1
            if insert_leaf(vecs[d], t) != truth[int(d)]:
                mism += 1
    return 100.0 * mism / tot


def main():
    tmp = tempfile.mkdtemp(prefix='mangrove_live_med_')
    try:
        vecs = make_vecs(N_DOCS, DIM, seed=42)
        base = os.path.join(tmp, 'base.fvecs')
        write_fvecs(base, vecs)
        rng = np.random.default_rng(7)
        check_ids = rng.choice(N_DOCS, size=N_CHECK, replace=False)

        # ---------- median index ----------
        idir = os.path.join(tmp, 'idx_med')
        build_index(base, idir, with_medians=True)
        set_gen_version(GEN_V)
        f = Forest(idir, n_trees=N_TREES, dim=DIM, sub_dim=SUB_DIM,
                   depth=DEPTH, n_docs=N_DOCS, gen_version=GEN_V)

        mf.clear_live_medians()
        pct_off = parity(f, vecs, check_ids)
        print(f'[1] median index, live medians OFF : {pct_off:5.1f}% mismatch '
              f'(inserted docs stranded)')

        md = mf.load_live_medians(idir)
        assert md == MED_DEPTH, f'load_live_medians returned {md}'
        pct_on = parity(f, vecs, check_ids)
        print(f'[2] median index, live medians ON  : {pct_on:5.1f}% mismatch')

        new = make_vecs(200, DIM, seed=1234)
        # NB : query_probes_scored already returns leaf ids (C subtracts lbase)
        bad = sum(1 for i in range(len(new)) for t in range(N_TREES)
                  if mf.query_probes_scored(new[i], DIM, SUB_DIM, DEPTH, t, 3
                                            )[0][0]
                  != insert_leaf(new[i], t))
        print(f'[3] insert path vs query main path : {bad} / '
              f'{len(new) * N_TREES} disagreements')

        # ---------- full live chain : hot insert → query ----------
        hotdir = os.path.join(tmp, 'hot')
        os.makedirs(hotdir, exist_ok=True)
        hot = mf._lib.mg_hot_init(N_TREES, DEPTH, hotdir.encode())
        assert hot, 'mg_hot_init failed'
        mf._lib.mg_forest_set_hot_overlay(hot)

        def hot_insert(vec, doc_id):
            for t in range(N_TREES):
                rc = mf._lib.mg_hot_append(hot, t, insert_leaf(vec, t), doc_id)
                assert rc == 0

        def doc_votes(vec, doc_id):
            """Votes for doc_id on the main-path-only query (n_probes=0) :
            8/8 = every tree's query leaf holds the doc, i.e. perfectly
            reachable ; low votes = mostly stranded (at scale, low-vote
            candidates don't survive the top_n cut)."""
            ids, votes, n = f.query_pathrank(vec, 0, N_TREES, 500,
                                             query_depth=DEPTH)
            for j in range(n):
                if int(ids[j]) == doc_id:
                    return int(votes[j])
            return 0

        live = make_vecs(100, DIM, seed=999)
        assert mf.live_medians_depth() == MED_DEPTH
        for i in range(100):
            hot_insert(live[i], N_DOCS + i)          # routed WITH medians
        v_live = [doc_votes(live[i], N_DOCS + i) for i in range(100)]
        n_full = sum(1 for v in v_live if v == N_TREES)
        print(f'[4] hot insert → query, medians armed : mean votes '
              f'{np.mean(v_live):.1f}/{N_TREES}, {n_full}/100 docs at '
              f'{N_TREES}/{N_TREES}, found {sum(1 for v in v_live if v)}/100')

        ghost = make_vecs(100, DIM, seed=31337)
        mf.clear_live_medians()                      # simulate the old bug
        for i in range(100):
            hot_insert(ghost[i], N_DOCS + 1000 + i)  # routed sign-split
        mf.load_live_medians(idir)
        v_ghost = [doc_votes(ghost[i], N_DOCS + 1000 + i) for i in range(100)]
        print(f'[4b] same, routed sign-split (old bug): mean votes '
              f'{np.mean(v_ghost):.1f}/{N_TREES} (stranded)')

        # compaction : drain HOT into the main SRTs, requery from MAIN
        rc = mf._lib.mg_hot_compact_all(f._h, hot, idir.encode())
        assert rc == 0, f'mg_hot_compact_all rc={rc}'
        mf._lib.mg_forest_set_hot_overlay(None)      # HOT drained + detached
        v_post = [doc_votes(live[i], N_DOCS + i) for i in range(100)]
        print(f'[4c] after compaction (HOT drained)  : mean votes '
              f'{np.mean(v_post):.1f}/{N_TREES}, found '
              f'{sum(1 for v in v_post if v)}/100 from MAIN')

        mf._lib.mg_hot_free(hot)
        f.close()

        # ---------- classic index (regression guard) ----------
        idir2 = os.path.join(tmp, 'idx_classic')
        build_index(base, idir2, with_medians=False)
        mf.clear_live_medians()
        assert mf.load_live_medians(idir2) == 0   # no medians.bin → no-op
        f2 = Forest(idir2, n_trees=N_TREES, dim=DIM, sub_dim=SUB_DIM,
                    depth=DEPTH, n_docs=N_DOCS, gen_version=GEN_V)
        pct_classic = parity(f2, vecs, check_ids)
        print(f'[5] classic index, unarmed         : {pct_classic:5.1f}% mismatch')
        f2.close()

        ok = (pct_off > 5.0 and pct_on < 0.5 and bad == 0
              and np.mean(v_live) > N_TREES - 0.5 and np.mean(v_ghost) < 4.0
              and np.mean(v_post) > N_TREES - 0.5
              and pct_classic < 0.5)
        print('PASS' if ok else 'FAIL')
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
