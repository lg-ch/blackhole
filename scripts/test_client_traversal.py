"""Parity test : pure-Python compute_leaves vs C-side traverse_sub.

For 100 random queries × 50 trees × depth 14 × sub_dim=16 × dim=128 :
  - Compute leaves Python-side using mangrove.traversal
  - Compute leaves C-side via FFI mg_traverse_sub
  - Every leaf_id must match bit-exact.
"""
from __future__ import annotations
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mangrove.traversal import compute_leaves, traverse_sub_one, tree_seed
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


def main():
    set_gen_version(3)
    rng = np.random.default_rng(42)
    dim, sub_dim, depth, n_trees = 128, 16, 14, 50
    n_q = 100
    mismatches = 0
    total = 0
    for qi in range(n_q):
        q = rng.standard_normal(dim).astype(np.float32)
        py_leaves = compute_leaves(q, n_trees=n_trees, depth=depth,
                                   sub_dim=sub_dim, dim=dim, gen_version=3)
        for t in range(n_trees):
            # Python (raw node id including leaf_base)
            leaf_base = (1 << depth) - 1
            py_node = py_leaves[t] + leaf_base
            # C — wrapper applies tree_seed() internally
            c_node = c_traverse_sub(q, dim, sub_dim, depth, t)
            total += 1
            if py_node != c_node:
                mismatches += 1
                if mismatches <= 3:
                    print(f'  MISMATCH q{qi} t{t}: py={py_node} c={c_node}')
    print(f'\n{total} traversals, {mismatches} mismatch')
    print(f'PASS' if mismatches == 0 else 'FAIL')


if __name__ == '__main__':
    main()
