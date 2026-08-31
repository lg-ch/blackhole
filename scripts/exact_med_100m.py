"""Medianes EXACTES int8, DEEP 100M depth 20, 256 arbres -> MED1.
SANS mmap (regle projet : la RAM doit etre explicite) : base chargee UNE
fois en shared_memory (38 Go, visibles dans free), workers attaches sans
copie. Projections par tranches (<1 Go par worker). A lancer sous cgroup
MemoryMax : un bug memoire tue le job, jamais la machine (OOM du 29/08)."""
import os
import struct
import sys
import time
from multiprocessing import Pool, shared_memory

import numpy as np

sys.path.insert(0, "/root/mangrove-campaign/mangrove-search/scripts")
from mangrove.traversal import tree_seed, node_seed, pick_dims, gen_vec_v3

B = "/root/deep100m/deep100m.fbin"
OUT = "/root/deep100m/medians_exact.bin"
N = 100_000_000
DEPTH = 20
DIM = 96
MED_MAGIC = 0x3144454D
SHM_NAME = "deep100m_base"

_ids = np.arange(N, dtype=np.uint64)
PAR = (((_ids * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(37))
       & np.uint64(1)).astype(bool)
del _ids


def get_X():
    shm = shared_memory.SharedMemory(name=SHM_NAME)
    return np.ndarray((N, DIM), dtype=np.float32, buffer=shm.buf), shm


def one_tree(t):
    X, shm = get_X()
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

    node0 = np.zeros(N, dtype=np.int32)
    theta = np.zeros((1 << DEPTH) - 1, dtype=np.float32)
    proj = np.empty(N, dtype=np.float32)      # reutilise a chaque niveau
    for level in range(DEPTH):
        order = np.argsort(node0, kind="stable").astype(np.int64)
        sorted_nodes = node0[order]
        uniq = np.unique(sorted_nodes)
        bounds = np.append(np.searchsorted(sorted_nodes, uniq), N)
        raw = np.empty(len(uniq), dtype=np.float32)
        for k in range(len(uniq)):
            sel = order[bounds[k]:bounds[k + 1]]
            dims, w = planes(int(uniq[k]))
            for c0 in range(0, len(sel), 2_000_000):
                sl = sel[c0:c0 + 2_000_000]
                proj[c0 + bounds[k]:c0 + bounds[k] + len(sl)] = \
                    X[sl][:, dims] @ w
            raw[k] = np.median(proj[bounds[k]:bounds[k + 1]])
        lo, hi = float(raw.min()), float(raw.max())
        snap = raw if hi <= lo else \
            (np.round((raw - lo) / (hi - lo) * 254.0) / 254.0
             * (hi - lo) + lo).astype(np.float32)
        for k in range(len(uniq)):
            nd = int(uniq[k])
            th = float(snap[k])
            theta[nd] = th
            sel = order[bounds[k]:bounds[k + 1]]
            pr = proj[bounds[k]:bounds[k + 1]]
            right = pr > th
            tie = pr == th
            if tie.any():
                right = right | (tie & PAR[sel])
            node0[sel] = (2 * nd + 1 + right.astype(np.int32)).astype(
                np.int32)
    shm.close()
    return t, theta


if __name__ == "__main__":
    t0 = time.time()
    # base -> shared memory (RAM explicite, une seule fois)
    try:
        old = shared_memory.SharedMemory(name=SHM_NAME)
        old.close(); old.unlink()
    except FileNotFoundError:
        pass
    shm = shared_memory.SharedMemory(name=SHM_NAME, create=True,
                                     size=N * DIM * 4)
    Xs = np.ndarray((N, DIM), dtype=np.float32, buffer=shm.buf)
    with open(B, "rb") as fh:
        fh.seek(8)
        CH = 2_000_000
        for off in range(0, N, CH):
            n = min(CH, N - off)
            Xs[off:off + n] = np.frombuffer(fh.read(n * DIM * 4),
                                            dtype=np.float32
                                            ).reshape(n, DIM)
    print(f"[{time.strftime('%H:%M:%S')}] base en shm : 38,4 Go "
          f"({time.time()-t0:.0f}s)", flush=True)

    per = (1 << DEPTH) - 1
    out = np.lib.format.open_memmap(OUT + ".npy", mode="w+",
                                    dtype=np.float32, shape=(256, per))
    done = 0
    with Pool(8) as pool:
        for t, theta in pool.imap_unordered(one_tree, range(256)):
            out[t] = theta
            done += 1
            if done % 8 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] {done}/256 arbres "
                      f"({time.time()-t0:.0f}s)", flush=True)
    out.flush()
    with open(OUT, "wb") as fp:
        fp.write(struct.pack("<IIII", MED_MAGIC, 256, DEPTH, 0))
        fp.write(np.ascontiguousarray(out).tobytes())
    os.remove(OUT + ".npy")
    shm.close(); shm.unlink()
    print(f"MEDEXACT100 OK en {time.time()-t0:.0f}s "
          f"({os.path.getsize(OUT)/1e9:.2f} Go)", flush=True)
