"""Pure-Python port of src/traversal.c for CLIENT-SIDE leaf computation.

Why : when a client computes its own 1000 leaf_ids locally and sends only
those to the server, the server never sees the query vector. The server
still does the K-way merge over posting lists. See PRIVACY.md for the
threat-model analysis.

Math compatibility : these routines reproduce the exact bit-equivalent
result of the C-side traversal — we use the same xorshift sequence,
the same hyperplane normalization, and the same node-id arithmetic.
A unit test in tests/ verifies parity against the C-side via FFI.

Usage :
    from mangrove.traversal import compute_leaves
    leaves = compute_leaves(qvec, n_trees=1000, depth=20,
                            sub_dim=16, dim=128, gen_version=3)
    # send `leaves` (list of u32) to server
"""
from __future__ import annotations
import math
import numpy as np


# --- seed derivation (mirrors traversal.c) ---

def tree_seed(tree_idx: int) -> int:
    return (tree_idx * 99991 + 7) & 0xFFFFFFFFFFFFFFFF


def node_seed(ts: int, bfs_idx: int) -> int:
    h = (ts * 2654435761) & 0xFFFFFFFFFFFFFFFF
    h ^= ((bfs_idx * 40503) & 0xFFFFFFFFFFFFFFFF)
    return h if h else 1


def pick_dims(ts: int, node_id: int, full_dim: int, sub_dim: int) -> list[int]:
    h = (ts * 2654435761) & 0xFFFFFFFFFFFFFFFF
    h ^= ((node_id * 3 * 40503) & 0xFFFFFFFFFFFFFFFF)
    out = [0] * sub_dim
    for i in range(sub_dim):
        h = (h * 2654435761 + i * 40503) & 0xFFFFFFFFFFFFFFFF
        out[i] = int(h % full_dim)
    return out


# --- hyperplane generation (mirrors gen_vec_v3) ---

def gen_vec_v3(seed: int, dim: int) -> np.ndarray:
    """Reproduce the C-side gen_vec_v3 xorshift sequence bit-for-bit."""
    st = seed if seed else 1
    st &= 0xFFFFFFFFFFFFFFFF
    out = np.empty(dim, dtype=np.float32)
    ns = 0.0
    i = 0
    while i + 1 < dim:
        st ^= (st << 13) & 0xFFFFFFFFFFFFFFFF
        st ^= (st >> 7)
        st ^= (st << 17) & 0xFFFFFFFFFFFFFFFF
        a = float(st & 0xFFFFFF) / 0xFFFFFF - 0.5
        b = float((st >> 24) & 0xFFFFFF) / 0xFFFFFF - 0.5
        out[i] = a
        out[i + 1] = b
        ns += a * a + b * b
        i += 2
    if i < dim:
        st ^= (st << 13) & 0xFFFFFFFFFFFFFFFF
        st ^= (st >> 7)
        st ^= (st << 17) & 0xFFFFFFFFFFFFFFFF
        a = float(st & 0xFFFFFF) / 0xFFFFFF - 0.5
        out[i] = a
        ns += a * a
    inv = 1.0 / math.sqrt(ns + 1e-10)
    out *= inv
    return out


# --- traversal ---

def traverse_sub_one(qvec: np.ndarray, full_dim: int, sub_dim: int,
                     depth: int, ts: int) -> int:
    """Compute the leaf_id for a single tree."""
    node = 0
    for level in range(depth):
        dims = pick_dims(ts, node, full_dim, sub_dim)
        v0 = gen_vec_v3(node_seed(ts, node * 2),     sub_dim)
        v1 = gen_vec_v3(node_seed(ts, node * 2 + 1), sub_dim)
        q_sub = qvec[dims]
        c0 = float(np.dot(q_sub, v0))
        c1 = float(np.dot(q_sub, v1))
        node = (2 * node + 2) if c1 > c0 else (2 * node + 1)
    return node


def traverse_full_one(qvec: np.ndarray, dim: int, depth: int, ts: int) -> int:
    """Full-dim (no sub_dim) traverse — mirrors C traverse()."""
    # Normalize qvec like the C side does in forest_collect_topn.
    qn = qvec / max(1e-10, float(np.linalg.norm(qvec)))
    node = 0
    for level in range(depth):
        v0 = gen_vec_v3(node_seed(ts, node * 2),     dim)
        v1 = gen_vec_v3(node_seed(ts, node * 2 + 1), dim)
        c0 = float(np.dot(qn, v0))
        c1 = float(np.dot(qn, v1))
        node = (2 * node + 2) if c1 > c0 else (2 * node + 1)
    return node


def compute_leaves(qvec: np.ndarray, n_trees: int, depth: int,
                   sub_dim: int, dim: int,
                   gen_version: int = 3) -> list[int]:
    """Client-side: compute the leaf_id for every tree in a forest.
       Returns a list of `n_trees` integers, each in [0, 2^depth).         */"""
    if gen_version != 3:
        raise ValueError(f'only gen_version=3 supported, got {gen_version}')
    if qvec.dtype != np.float32:
        qvec = qvec.astype(np.float32, copy=False)
    if qvec.ndim != 1 or qvec.shape[0] != dim:
        raise ValueError(f'qvec must be 1-D length {dim}, got shape {qvec.shape}')
    use_sub = (sub_dim > 0 and sub_dim < dim)
    # Sub-dim mode reads qvec by indexed dims (no global normalize required).
    # Full-dim mode normalizes qvec like forest_collect_topn does.
    if use_sub:
        traverse_fn = lambda ts: traverse_sub_one(qvec, dim, sub_dim, depth, ts)
    else:
        traverse_fn = lambda ts: traverse_full_one(qvec, dim, depth, ts)
    leaf_base = (1 << depth) - 1
    out: list[int] = []
    for t in range(n_trees):
        ts = tree_seed(t)
        node = traverse_fn(ts)
        out.append(node - leaf_base)
    return out
