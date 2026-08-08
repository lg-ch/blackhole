"""Parity test : pure-Python compute_leaves vs C-side traverse_sub.

[1] Classic sign-split : 100 random queries × 50 trees × depth 14 ×
    sub_dim=16 × dim=128 — every leaf_id must match bit-exact.

[2][3] Median index (same tiny build as test_live_medians : 20k docs,
    dim 64, sub_dim 16, 8 trees, depth 10, med_depth 6, gen v3) :
      [2] both sides UNARMED  → parity (non-regression of sign splits)
      [3] both sides ARMED    → parity (Python load_medians/compute_leaves
          vs C mg_traverse_sub with mg_live_medians_load), 100 queries ×
          8 trees, and a sanity check that θ actually changes some routes.
"""
from __future__ import annotations
import os, sys
import tempfile, shutil
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mangrove.traversal import compute_leaves, load_medians, traverse_sub_one, tree_seed
import mangrove_ffi as mf
from mangrove_ffi import _lib, set_gen_version
from ctypes import POINTER, c_int, c_float, c_int32, c_uint64


def c_traverse_sub(qvec, full_dim, sub_dim, depth, tree_idx):
    """Call mg_traverse_sub via FFI. The C wrapper applies tree_seed()
       internally, so we pass tree_idx directly."""
    qbuf = qvec.astype(np.float32, copy=False)
    fn = _lib.mg_traverse_sub
    fn.argtypes = [POINTER(c_float), c_int, c_int, c_int, c_int]
    fn.restype = c_int
    return int(fn(qbuf.ctypes.data_as(POINTER(c_float)), full_dim, sub_dim,
                  depth, tree_idx))


def run_parity(dim, sub_dim, depth, n_trees, n_q, seed,
               medians=None, med_depth=0, tag=''):
    """Python compute_leaves vs C mg_traverse_sub over n_q × n_trees.
       The C side uses whatever live-median state is currently armed —
       the caller keeps both sides consistent."""
    rng = np.random.default_rng(seed)
    lbase = (1 << depth) - 1
    mismatches = total = 0
    for qi in range(n_q):
        q = rng.standard_normal(dim).astype(np.float32)
        py_leaves = compute_leaves(q, n_trees=n_trees, depth=depth,
                                   sub_dim=sub_dim, dim=dim, gen_version=3,
                                   medians=medians, med_depth=med_depth)
        for t in range(n_trees):
            py_node = py_leaves[t] + lbase
            c_node = c_traverse_sub(q, dim, sub_dim, depth, t)
            total += 1
            if py_node != c_node:
                mismatches += 1
                if mismatches <= 3:
                    print(f'  MISMATCH{tag} q{qi} t{t}: py={py_node} c={c_node}')
    return mismatches, total


def main():
    set_gen_version(3)
    ok = True

    # ---------- [1] classic sign-split parity (original test) ----------
    mf.clear_live_medians()          # helpers must be unarmed here
    mm, tot = run_parity(dim=128, sub_dim=16, depth=14, n_trees=50,
                         n_q=100, seed=42, tag='[1]')
    print(f'[1] sign-split          : {tot} traversals, {mm} mismatch')
    ok &= (mm == 0)

    # ---------- median index (built exactly like test_live_medians) ----------
    import test_live_medians as tlm
    tmp = tempfile.mkdtemp(prefix='mangrove_client_med_')
    try:
        vecs = tlm.make_vecs(tlm.N_DOCS, tlm.DIM, seed=42)
        base = os.path.join(tmp, 'base.fvecs')
        tlm.write_fvecs(base, vecs)
        idir = os.path.join(tmp, 'idx_med')
        tlm.build_index(base, idir, with_medians=True)

        loaded = load_medians(idir)
        assert loaded is not None, 'load_medians returned None on a median index'
        med_tab, med_depth = loaded
        assert med_depth == tlm.MED_DEPTH, f'med_depth {med_depth}'
        assert med_tab.shape == (tlm.N_TREES, (1 << med_depth) - 1)

        # [2] both sides unarmed on the median index (non-regression).
        mf.clear_live_medians()
        mm2, tot2 = run_parity(dim=tlm.DIM, sub_dim=tlm.SUB_DIM,
                               depth=tlm.DEPTH, n_trees=tlm.N_TREES,
                               n_q=100, seed=7, tag='[2]')
        print(f'[2] median idx, UNARMED : {tot2} traversals, {mm2} mismatch')
        ok &= (mm2 == 0)

        # [3] both sides armed : Python medians vs C live medians.
        md = mf.load_live_medians(idir)
        assert md == tlm.MED_DEPTH, f'load_live_medians returned {md}'
        mm3, tot3 = run_parity(dim=tlm.DIM, sub_dim=tlm.SUB_DIM,
                               depth=tlm.DEPTH, n_trees=tlm.N_TREES,
                               n_q=100, seed=7,
                               medians=med_tab, med_depth=med_depth,
                               tag='[3]')
        print(f'[3] median idx, ARMED   : {tot3} traversals, {mm3} mismatch')
        ok &= (mm3 == 0)

        # Sanity : θ must actually reroute part of the traffic (otherwise
        # [3] could pass with medians silently ignored on both sides).
        rng = np.random.default_rng(7)
        diff = 0
        for _ in range(20):
            q = rng.standard_normal(tlm.DIM).astype(np.float32)
            a = compute_leaves(q, n_trees=tlm.N_TREES, depth=tlm.DEPTH,
                               sub_dim=tlm.SUB_DIM, dim=tlm.DIM,
                               medians=med_tab, med_depth=med_depth)
            b = compute_leaves(q, n_trees=tlm.N_TREES, depth=tlm.DEPTH,
                               sub_dim=tlm.SUB_DIM, dim=tlm.DIM)
            diff += sum(1 for x, y in zip(a, b) if x != y)
        print(f'[3b] θ reroutes {diff} / {20 * tlm.N_TREES} (must be > 0)')
        ok &= (diff > 0)
    finally:
        mf.clear_live_medians()      # leave process-global state clean
        shutil.rmtree(tmp, ignore_errors=True)

    print('PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
