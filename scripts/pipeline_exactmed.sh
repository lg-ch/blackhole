#!/bin/bash
# A/B recall DEEP 10M : medianes EXACTES int8 depth17 vs calibration
# echantillonnee med13. Deux builds reels + meme bench vs gt_10m.
exec >> /root/deep10m/pipeline.log 2>&1
set -e
D=/root/deep10m
B=$D/deep10m.fbin
cd /root/mangrove-campaign/mangrove-search
export OMP_NUM_THREADS=20

# 0. base 10M extraite de deep50m
if [ ! -f $B ]; then
  echo "== extraction deep10m $(date +%H:%M:%S)"
  python3 - <<PY
import struct
import numpy as np
src = "/root/mangrove-campaign/data/deep50m.fbin"
with open(src, "rb") as s, open("$B", "wb") as d:
    d.write(struct.pack("<II", 10_000_000, 96))
    s.seek(8)
    left = 10_000_000 * 96 * 4
    while left:
        c = s.read(min(1 << 24, left))
        d.write(c)
        left -= len(c)
print("extraction OK", flush=True)
PY
fi

# 1. medianes exactes 256 arbres (multiproc)
if [ ! -f $D/medians_exact.bin ]; then
  echo "== medianes exactes $(date +%H:%M:%S)"
  python3 -u $D/exact_med_256.py
fi

# 2. build EXACT : depth 17 = niveau median (feuilles ~76 docs)
IDXE=$D/idx_exact17
mkdir -p $IDXE
if [ ! -s "$IDXE/tree00255.srt" ]; then
  cp $D/medians_exact.bin $IDXE/medians.bin
  echo "== build exact17 $(date +%H:%M:%S)"
  ./rpforest build $B $IDXE 256 17 --total_trees 256 --sub_dim 16 \
      --gen v3 --dim 96 --fast 2>&1 | grep -E "convert:|DONE" | tail -2
fi

# 3. build REFERENCE : calibration echantillonnee med13, meme depth 17
IDXR=$D/idx_ref13
mkdir -p $IDXR
if [ ! -f $IDXR/medians.bin ]; then
  echo "== calibrate ref med13 $(date +%H:%M:%S)"
  python3 - <<PY
import sys
sys.path.insert(0, "scripts")
import mangrove_ffi as mf
from ctypes import c_char_p, c_int
mf.set_gen_version(3)
lib = mf._lib
lib.mg_calibrate_medians.argtypes = [c_char_p, c_char_p] + [c_int] * 5
lib.mg_calibrate_medians.restype = c_int
rc = lib.mg_calibrate_medians(b"$B", b"$IDXR", 256, 96, 16, 13, 1_500_000)
assert rc == 0, rc
print("medians ref OK", flush=True)
PY
fi
if [ ! -s "$IDXR/tree00255.srt" ]; then
  echo "== build ref13 $(date +%H:%M:%S)"
  ./rpforest build $B $IDXR 256 17 --total_trees 256 --sub_dim 16 \
      --gen v3 --dim 96 --fast 2>&1 | grep -E "convert:|DONE" | tail -2
fi

# 4. bench A/B contre la GT campagne (requetes hors base)
echo "== bench $(date +%H:%M:%S)"
python3 - <<'PY'
import sys, time, json
import numpy as np
sys.path.insert(0, "/root/mangrove-campaign/mangrove-search/scripts")
import mangrove_ffi as mf
from mangrove_ffi import Forest
D = "/root/deep10m"
B = D + "/deep10m.fbin"
Q = np.load("/root/mangrove-campaign/data/queries.npy").astype(np.float32)
gt = np.load("/root/mangrove-campaign/data/gt_10m.npy")
out = {}
for name, idx in (("exact17", D + "/idx_exact17"),
                  ("ref13", D + "/idx_ref13")):
    f = Forest(idx, n_trees=256, dim=96, sub_dim=16, depth=17,
               n_docs=10_000_000)
    for qd, np_, tp, tn, mfl in ((17, 3, 1024, 6000, False),
                                 (15, 3, 1024, 6000, False),
                                 (14, 3, 2048, 20000, False),
                                 (14, 7, 4096, 20000, False),
                                 (13, 15, 4096, 100000, True)):
        mf.set_probe_multiflip(mfl)
        recs, lat = [], []
        for qi in range(200):
            t0 = time.time()
            ids, votes, n = f.query_pathrank(Q[qi], np_, tp, tn, qd)
            top = f.rerank_l2(B, Q[qi], ids[:n]) if n else []
            lat.append((time.time() - t0) * 1000)
            recs.append(len(set(np.asarray(top).tolist())
                            & set(gt[qi].tolist())) / 10.0)
        row = {"recall": round(float(np.mean(recs)), 4),
               "p50": round(float(np.percentile(lat, 50)), 1),
               "p99": round(float(np.percentile(lat, 99)), 1)}
        out.setdefault(name, []).append(dict(qd=qd, np=np_, tp=tp,
                                             mf=int(mfl), **row))
        print(f"{name:8s} qd{qd} NP{np_:2d} tp{tp}"
              f"{' mf' if mfl else '   '}: recall {row['recall']:.3f} "
              f"p50 {row['p50']:.0f} p99 {row['p99']:.0f} ms", flush=True)
    mf.set_probe_multiflip(False)
    json.dump(out, open(D + "/ab_exactmed.json", "w"), indent=1)
print("BENCH AB OK", flush=True)
PY
echo "== PIPELINE EXACTMED OK $(date +%H:%M:%S)"
