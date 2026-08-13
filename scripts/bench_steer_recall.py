#!/usr/bin/env python3
"""Prototype veto-steering : mesure le RECALL réel (pas seulement les probes
simulées) du pilotage des probes par comptages d'éligibles, via
set_external_leaves — aucun changement moteur.

Pour chaque prédicat (cor1, cor02, ind1-contrôle) :
  - sidecar comptages (arbre, bucket med) rebâti en routant les éligibles ;
  - probes candidates à probe_depth = med (ordre de marge géométrique) ;
  - mode BLIND  : rows k = k-ième probe par arbre (ordre marge pur) ;
  - mode STEER  : par arbre, on ne garde que les probes à comptage > 0
    (ordre de marge préservé — le comptage est un VETO, pas un tri) ;
  - pool = union des candidats filtrés des R rows -> rerank L2 exact ;
  - recall@10 vs GT filtrée brute-force, latence wall par requête.

Le filtre passe par allowed_state (roaring portable multi-containers
enveloppé au format ch_state) — chemin curseurs, fixé commit fe9815a.
"""
import json
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mangrove_ffi as mf
from mangrove_ffi import Forest, set_external_leaves

ROOT = os.environ.get('SEL_ROOT', '/root/mangrove-campaign')
IDX = f'{ROOT}/run/idx_live'
BASE = f'{ROOT}/data/deep50m.fbin'
STORE = f'{ROOT}/selstore'

NT, DIM, SD, DEPTH, NDOCS = 256, 96, 16, 22, 50_000_000
MED = 13
NQ = 100
NPROBE = 63          # 64 probes candidates par arbre
TOPN_ROW = 100_000   # assez grand pour ne jamais tronquer les éligibles


def log(m):
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def roaring_ch_state(ids):
    """Roaring portable MULTI-containers (array<=4096 / bitset au-delà),
    enveloppé [0x01][varint][portable] (format ClickHouse groupBitmap)."""
    ids = np.unique(np.asarray(ids, dtype=np.uint32))
    hi = (ids >> np.uint32(16)).astype(np.uint32)
    lo = (ids & np.uint32(0xFFFF)).astype(np.uint16)
    keys, starts = np.unique(hi, return_index=True)
    n = len(keys)
    bounds = np.append(starts, len(ids))
    desc = b''
    bodies = []
    for i, k in enumerate(keys):
        c = lo[bounds[i]:bounds[i + 1]]
        card = len(c)
        desc += struct.pack('<HH', int(k), card - 1)
        if card > 4096:
            bits = np.zeros(1024, dtype=np.uint64)
            c64 = c.astype(np.uint64)
            np.bitwise_or.at(bits, (c64 >> np.uint64(6)).astype(np.int64),
                             np.uint64(1) << (c64 & np.uint64(63)))
            bodies.append(bits.tobytes())
        else:
            bodies.append(c.astype('<u2').tobytes())
    header = struct.pack('<II', 12346, n) + desc
    off = len(header) + 4 * n
    offsets = b''
    for b in bodies:
        offsets += struct.pack('<I', off)
        off += len(b)
    body = header + offsets + b''.join(bodies)
    nb = len(body)
    varint = bytearray()
    while True:
        b7 = nb & 0x7F
        nb >>= 7
        varint.append(b7 | (0x80 if nb else 0))
        if not nb:
            break
    return b'\x01' + bytes(varint) + body


def main():
    f = Forest(IDX, n_trees=NT, dim=DIM, sub_dim=SD, depth=DEPTH,
               n_docs=NDOCS)
    assert mf.load_live_medians(IDX) == MED
    Q = np.load(f'{ROOT}/data/queries.npy')[:NQ].astype(np.float32)
    X = np.memmap(BASE, dtype=np.float32, mode='r', offset=8,
                  shape=(NDOCS, DIM))
    out = {}
    for name in ('cor1', 'cor02', 'ind1'):
        eids = np.load(f'{STORE}/ids_{name}.npy')
        gt = np.load(f'{STORE}/gt_{name}.npy')
        elig = set(eids.tolist())
        ch = roaring_ch_state(eids)
        log(f'--- {name} : |B|={len(eids):,}, ch_state {len(ch):,} o ---')

        counts = np.zeros((NT, 1 << MED), dtype=np.int32)
        t0 = time.time()
        for off in range(0, len(eids), 200_000):
            xv = np.ascontiguousarray(X[eids[off:off + 200_000]])
            bk = mf.traverse_batch(xv, SD, MED, NT)
            for t in range(NT):
                np.add.at(counts[t], bk[:, t], 1)
        log(f'  sidecar : {time.time() - t0:.0f}s')

        res = {}
        for mode in ('blind', 'steer'):
            for R in ((2, 4, 8, 16) if mode == 'blind' else (1, 2, 4, 8)):
                recs, lats, pools = [], [], []
                for qi in range(NQ):
                    t0 = time.time()
                    probes = f.probe_leaves(Q[qi], NPROBE, probe_depth=MED)
                    if mode == 'blind':
                        rows = probes[:R]
                    else:
                        cnt = counts[np.arange(NT)[None, :], probes]
                        rows = np.empty((R, NT), dtype=np.int32)
                        for t in range(NT):
                            nz = probes[cnt[:, t] > 0, t]
                            if len(nz) == 0:
                                nz = probes[:1, t]
                            take = np.minimum(np.arange(R), len(nz) - 1)
                            rows[:, t] = nz[take]
                    pool = set()
                    for j in range(len(rows)):
                        set_external_leaves(rows[j])
                        ids, votes, n = f.query(Q[qi], top_n=TOPN_ROW,
                                                query_depth=MED,
                                                allowed_state=ch)
                        pool.update(ids[:n].tolist())
                    set_external_leaves(None)
                    cand = np.fromiter(pool, dtype=np.int32)
                    top = f.rerank_l2(BASE, Q[qi], cand) if len(cand) else []
                    lats.append((time.time() - t0) * 1000)
                    recs.append(len(set(np.asarray(top).tolist())
                                    & set(gt[qi].tolist())) / 10.0)
                    pools.append(len(pool))
                row = {'recall': round(float(np.mean(recs)), 4),
                       'p50_ms': round(float(np.percentile(lats, 50)), 1),
                       'pool_mean': int(np.mean(pools))}
                res[f'{mode}_R{R}'] = row
                log(f'  {mode:5s} R={R:2d} recall {row["recall"]:.3f} '
                    f'p50 {row["p50_ms"]:7.1f} ms pool {row["pool_mean"]:,}')
        out[name] = res
        with open(f'{STORE}/steer_recall.json', 'w') as fp:
            json.dump(out, fp, indent=1)
    log('STEER-RECALL OK')


if __name__ == '__main__':
    main()
