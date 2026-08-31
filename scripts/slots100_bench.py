"""Slots 512 o pour DEEP 100M depth 20 + bench A/B v1/v2 warm+cold."""
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, "/root/mangrove-campaign/mangrove-search/scripts")
import mangrove_ffi as mf
from mangrove_ffi import Forest, slots_v2_open, query_v2

D = "/root/deep100m"
B = D + "/deep100m.fbin"
SL = D + "/slots20"
N = 100_000_000
DEPTH = 20
NT = 256
SLOT = 512
CAP = (SLOT - 4) // 4

log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

if not os.path.exists(f"{SL}/tree00255.slt"):
    os.makedirs(SL, exist_ok=True)
    mf.set_gen_version(3)
    assert mf.load_live_medians(D + "/idx_exact20") == DEPTH
    X = np.memmap(B, dtype=np.float32, mode="r", offset=8, shape=(N, 96))
    words = SLOT // 4
    trunc_tot = 0
    t0 = time.time()
    # par PAQUETS d arbres (8) pour tenir la RAM : leaf_of 8 x 100M i32
    for tg in range(0, NT, 8):
        leaf_of = np.empty((8, N), dtype=np.int32)
        for off in range(0, N, 1_000_000):
            lv = mf.traverse_batch(
                np.ascontiguousarray(X[off:off + 1_000_000]), 16, DEPTH,
                tg + 8)
            leaf_of[:, off:off + 1_000_000] = lv[:, tg:tg + 8].T
        for j in range(8):
            t = tg + j
            lv = leaf_of[j].astype(np.int64)
            order = np.argsort(lv, kind="stable")
            ls = lv[order]
            counts = np.bincount(ls, minlength=1 << DEPTH)
            if t == 0:
                p99 = int(np.percentile(counts, 99))
                print(f"garde-fou arbre 0 : p99 {p99} (cap {CAP})",
                      flush=True)
                assert p99 < CAP, "routage desequilibre"
            starts = np.zeros(1 << DEPTH, dtype=np.int64)
            starts[1:] = np.cumsum(counts)[:-1]
            pos = np.arange(N, dtype=np.int64) - starts[ls]
            valid = pos < CAP
            trunc_tot += int((~valid).sum())
            arr = np.zeros(((1 << DEPTH), words), dtype=np.uint32)
            arr[:, 0] = np.minimum(counts, CAP).astype(np.uint32)
            flat = ls[valid] * words + 1 + pos[valid]
            arr.reshape(-1)[flat] = order[valid].astype(np.uint32)
            arr.tofile(f"{SL}/tree{t:05d}.slt")
        log(f"slots arbres {tg}..{tg+7} ({time.time()-t0:.0f}s)")
    log(f"BUILD SLOTS100 OK ({time.time()-t0:.0f}s, tronques "
        f"{trunc_tot:,} = {trunc_tot/(N*NT):.4%})")

f = Forest(D + "/idx_exact20", n_trees=NT, dim=96, sub_dim=16,
           depth=DEPTH, n_docs=N)
sh = slots_v2_open(SL, NT, 1 << DEPTH, SLOT)
Q = np.load("/root/mangrove-campaign/data/queries.npy").astype(np.float32)
gt = np.load(D + "/gt_100m.npy")


def bench(mode, qd, np_, tp, tn, nq, cold=False):
    recs, lat = [], []
    for qi in range(nq):
        if cold:
            subprocess.run("sync; echo 3 > /proc/sys/vm/drop_caches",
                           shell=True)
        t0 = time.time()
        if mode == "v1":
            ids, votes, n = f.query_pathrank(Q[qi], np_, tp, tn, qd)
        else:
            ids, votes, n = query_v2(f, sh, Q[qi], np_, tp, tn, qd)
        top = f.rerank_l2(B, Q[qi], ids[:n]) if n else []
        lat.append((time.time() - t0) * 1000)
        recs.append(len(set(np.asarray(top).tolist())
                        & set(gt[qi].tolist())) / 10.0)
    tag = "COLD" if cold else "warm"
    print(f"{tag} {mode} qd{qd} NP{np_} tp{tp}: recall "
          f"{np.mean(recs):.3f} p50 {np.percentile(lat, 50):.1f} "
          f"p99 {np.percentile(lat, 99):.1f} ms", flush=True)


for qd, np_, tp, tn in ((18, 3, 1024, 6000), (17, 3, 2048, 20000),
                        (17, 7, 4096, 20000)):
    bench("v1", qd, np_, tp, tn, 200)
    bench("v2", qd, np_, tp, tn, 200)
for qd, np_, tp, tn in ((18, 3, 1024, 6000), (17, 7, 4096, 20000)):
    bench("v1", qd, np_, tp, tn, 40, cold=True)
    bench("v2", qd, np_, tp, tn, 40, cold=True)
print("SLOTS100 BENCH OK", flush=True)
