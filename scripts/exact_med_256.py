"""Medianes EXACTES (int8-snappees) pour 256 arbres sur DEEP 10M, ecrites
au format MED1 (f32 aux valeurs int8-representables) — multiprocess."""
import os
import struct
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, "/root/mangrove-campaign/mangrove-search/scripts")
from mangrove.traversal import tree_seed, node_seed, pick_dims, gen_vec_v3

B = "/root/deep10m/deep10m.fbin"
OUT = "/root/deep10m/medians_exact.bin"
N = 10_000_000
DEPTH = 17
DIM = 96
MED_MAGIC = 0x3144454D

ids_all = np.arange(N, dtype=np.uint64)
PAR = (((ids_all * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(37))
       & np.uint64(1)).astype(bool)


def one_tree(t):
    X = np.memmap(B, dtype=np.float32, mode="r", offset=8, shape=(N, DIM))
    ts = tree_seed(t)
    cache = {}

    def planes(node):
        hit = cache.get(node)
        if hit is None:
            dims = np.asarray(pick_dims(ts, node, DIM, 16), dtype=np.int64)
            v0 = gen_vec_v3(node_seed(ts, node * 2), 16)
            v1 = gen_vec_v3(node_seed(ts, node * 2 + 1), 16)
            hit = (dims, v1 - v0)
            cache[node] = hit
        return hit

    node0 = np.zeros(N, dtype=np.int64)
    theta = np.zeros((1 << DEPTH) - 1, dtype=np.float32)
    for level in range(DEPTH):
        order = np.argsort(node0, kind="stable")
        sorted_nodes = node0[order]
        uniq = np.unique(sorted_nodes)
        bounds = np.append(np.searchsorted(sorted_nodes, uniq), len(order))
        raw = np.empty(len(uniq), dtype=np.float32)
        projs = [None] * len(uniq)
        sels = [None] * len(uniq)
        for k in range(len(uniq)):
            nd = int(uniq[k])
            sel = order[bounds[k]:bounds[k + 1]]
            dims, w = planes(nd)
            proj = X[sel][:, dims] @ w
            raw[k] = np.median(proj)
            projs[k] = proj
            sels[k] = sel
        # snap int8 par niveau (grille affine), route avec le snap
        lo, hi = float(raw.min()), float(raw.max())
        snap = raw if hi <= lo else \
            (np.round((raw - lo) / (hi - lo) * 254.0) / 254.0
             * (hi - lo) + lo).astype(np.float32)
        for k in range(len(uniq)):
            nd = int(uniq[k])
            th = float(snap[k])
            theta[nd] = th
            proj, sel = projs[k], sels[k]
            right = proj > th
            tie = proj == th
            if tie.any():
                right = right | (tie & PAR[sel])
            node0[sel] = 2 * nd + 1 + right.astype(np.int64)
    return t, theta


if __name__ == "__main__":
    t0 = time.time()
    per = (1 << DEPTH) - 1
    out = np.zeros((256, per), dtype=np.float32)
    with Pool(20) as pool:
        for t, theta in pool.imap_unordered(one_tree, range(256)):
            out[t] = theta
            done = int((out != 0).any(axis=1).sum())
            if done % 32 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] {done}/256 arbres "
                      f"({time.time()-t0:.0f}s)", flush=True)
    with open(OUT, "wb") as fp:
        fp.write(struct.pack("<IIII", MED_MAGIC, 256, DEPTH, 0))
        fp.write(out.tobytes())
    print(f"MEDEXACT OK en {time.time()-t0:.0f}s "
          f"({os.path.getsize(OUT)/1e6:.0f} Mo)", flush=True)
