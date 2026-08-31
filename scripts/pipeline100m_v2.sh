#!/bin/bash
# Nuit 100M v2 — SOUS CGROUP (MemoryMax 100G, swap 0 : le swap-death est
# impossible ; un bug memoire tue le job, jamais la machine).
# build2 (medianes exactes + slots natifs) -> GT -> bench v2 (warm+cold).
exec >> /root/deep100m/pipeline2.log 2>&1
set -e
D=/root/deep100m
B=$D/deep100m.fbin
cd /root/mangrove-campaign/mangrove-search
export OMP_NUM_THREADS=16

echo "== BUILD2 100M d20 $(date +%H:%M:%S)"
mkdir -p $D/idx_v2
if [ ! -s "$D/idx_v2/medians.bin" ]; then
  ./rpforest build2 $B $D/idx_v2 256 20 --sub_dim 16 --dim 96 \
      --slot_bytes 512 --group 32
fi
echo "== build2 OK $(date +%H:%M:%S)"

if [ ! -f $D/gt_100m.npy ]; then
  echo "== GT 100M $(date +%H:%M:%S)"
  python3 - <<'PY'
import time
import numpy as np
D = "/root/deep100m"
N = 100_000_000
Q = np.load("/root/mangrove-campaign/data/queries.npy").astype(np.float32)
NQ = len(Q)
bd = np.full((NQ, 10), np.inf, dtype=np.float32)
bi = np.zeros((NQ, 10), dtype=np.int64)
CH = 2_000_000
t0 = time.time()
with open(D + "/deep100m.fbin", "rb") as fh:
    fh.seek(8)
    done = 0
    while done < N:
        n = min(CH, N - done)
        blk = np.frombuffer(fh.read(n * 384), dtype=np.float32
                            ).reshape(n, 96)
        x2 = np.einsum("ij,ij->i", blk, blk)
        d2 = x2[:, None] - 2.0 * (blk @ Q.T)
        part = np.argpartition(d2, 9, axis=0)[:10]
        cd = np.take_along_axis(d2, part, axis=0)
        alld = np.concatenate([bd, cd.T], axis=1)
        alli = np.concatenate([bi, (part + done).T], axis=1)
        order = np.argsort(alld, axis=1)[:, :10]
        bd = np.take_along_axis(alld, order, axis=1)
        bi = np.take_along_axis(alli, order, axis=1)
        done += n
        if done % 20_000_000 < CH:
            print(f"  GT {done/1e6:.0f}M ({time.time()-t0:.0f}s)",
                  flush=True)
np.save(D + "/gt_100m.npy", bi)
print(f"GT100 OK en {time.time()-t0:.0f}s", flush=True)
PY
fi

echo "== montage traversee (symlinks srt factices non requis : v2 pur)"
# forest_open exige des .srt : on pointe sur les srt du 10M (JAMAIS lus
# par query_v2 — seuls f->medians et les seeds servent a la traversee),
# avec les medians de build2. n_docs/depth passes cote python.
mkdir -p $D/idx_v2t
for f in /root/deep10m/idx_exact17/tree*.srt; do ln -sf $f $D/idx_v2t/; done
cp $D/idx_v2/medians.bin $D/idx_v2t/

echo "== bench v2 $(date +%H:%M:%S)"
python3 - <<'PY'
import subprocess
import sys
import time
import numpy as np
sys.path.insert(0, "/root/mangrove-campaign/mangrove-search/scripts")
from mangrove_ffi import Forest, slots_v2_open, query_v2
D = "/root/deep100m"
N = 100_000_000
f = Forest(D + "/idx_v2t", n_trees=256, dim=96, sub_dim=16, depth=20,
           n_docs=N)
sh = slots_v2_open(D + "/idx_v2", 256, 1 << 20, 512)
Q = np.load("/root/mangrove-campaign/data/queries.npy").astype(np.float32)
gt = np.load(D + "/gt_100m.npy")


def bench(qd, np_, tp, tn, nq, cold=False):
    recs, lat = [], []
    for qi in range(nq):
        if cold:
            subprocess.run("sync; echo 3 > /proc/sys/vm/drop_caches",
                           shell=True)
        t0 = time.time()
        ids, votes, n = query_v2(f, sh, Q[qi], np_, tp, tn, qd)
        top = f.rerank_l2(D + "/deep100m.fbin", Q[qi], ids[:n]) \
            if n else []
        lat.append((time.time() - t0) * 1000)
        recs.append(len(set(np.asarray(top).tolist())
                        & set(gt[qi].tolist())) / 10.0)
    tag = "COLD" if cold else "warm"
    print(f"{tag} v2 qd{qd} NP{np_} tp{tp} tn{tn}: recall "
          f"{np.mean(recs):.3f} p50 {np.percentile(lat, 50):.1f} "
          f"p99 {np.percentile(lat, 99):.1f} ms", flush=True)


for cfg in ((18, 3, 1024, 6000), (17, 3, 2048, 20000),
            (17, 7, 4096, 20000), (16, 7, 4096, 100000)):
    bench(*cfg, 200)
for cfg in ((18, 3, 1024, 6000), (17, 7, 4096, 20000)):
    bench(*cfg, 40, cold=True)
print("BENCH100 V2 OK", flush=True)
PY
echo "== PIPELINE100M V2 OK $(date +%H:%M:%S)"
