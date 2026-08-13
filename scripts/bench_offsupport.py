#!/usr/bin/env python3
"""Cas difficile : requêtes HORS SUPPORT du filtre (sujet sémantique disjoint
des métadonnées demandées — "le sujet date de la semaine dernière").

Sélection des requêtes : les 50 (sur 200) de plus FAIBLE projection sur la
direction w du demi-espace cor* — garanties loin du support (filtre = top
1 % / 0,2 % de cette même projection). GT filtrées déjà calculées (setup).

Mesure aveugle vs descente guidée, R jusqu'à 12, plus le taux d'échec
"pool < 10 candidats" (le mode de défaillance pgvector : résultats vides).
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mangrove_ffi as mf
from mangrove_ffi import Forest, set_external_leaves
from mangrove.traversal import load_medians
from bench_steer_recall import roaring_ch_state
from bench_guided_descent import guided_rows
from bench_selectivity import cor_dir

ROOT = os.environ.get('SEL_ROOT', '/root/mangrove-campaign')
IDX = f'{ROOT}/run/idx_live'
BASE = f'{ROOT}/data/deep50m.fbin'
STORE = f'{ROOT}/selstore'

NT, DIM, SD, DEPTH, NDOCS = 256, 96, 16, 22, 50_000_000
MED = 13
N_HARD = 50
TOPN_ROW = 100_000
RS = (1, 2, 4, 8, 10, 12)


def log(m):
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def main():
    f = Forest(IDX, n_trees=NT, dim=DIM, sub_dim=SD, depth=DEPTH,
               n_docs=NDOCS)
    assert mf.load_live_medians(IDX) == MED
    med_tab, md = load_medians(IDX)
    assert md == MED
    Q = np.load(f'{ROOT}/data/queries.npy')[:200].astype(np.float32)
    X = np.memmap(BASE, dtype=np.float32, mode='r', offset=8,
                  shape=(NDOCS, DIM))

    w = cor_dir()
    proj = Q @ w
    hard = np.argsort(proj)[:N_HARD]
    log(f'requêtes hors-support : proj ∈ [{proj[hard].min():.3f}, '
        f'{proj[hard].max():.3f}] (médiane globale {np.median(proj):.3f})')

    Qn = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-9)
    Qn = Qn.astype(np.float32)

    out = {'hard_query_idx': hard.tolist()}
    for name in ('cor1', 'cor02'):
        eids = np.load(f'{STORE}/ids_{name}.npy')
        gt = np.load(f'{STORE}/gt_{name}.npy')
        ch = roaring_ch_state(eids)
        log(f'--- {name} : |B|={len(eids):,} ---')

        counts = np.zeros((NT, 1 << MED), dtype=np.int32)
        for off in range(0, len(eids), 200_000):
            xv = np.ascontiguousarray(X[eids[off:off + 200_000]])
            bk = mf.traverse_batch(xv, SD, MED, NT)
            for t in range(NT):
                np.add.at(counts[t], bk[:, t], 1)
        alive_per_tree = []
        for t in range(NT):
            lv = [None] * (MED + 1)
            lv[MED] = counts[t] > 0
            for l in range(MED - 1, -1, -1):
                lv[l] = lv[l + 1].reshape(-1, 2).any(axis=1)
            alive_per_tree.append(lv)

        res = {}
        for mode, masks in (('cblind', [None] * NT),
                            ('guided', alive_per_tree)):
            for R in RS:
                recs, lats, pools, starved = [], [], [], 0
                for qi in hard:
                    t0 = time.time()
                    rows = guided_rows(Qn[qi], med_tab, masks, R)
                    pool = set()
                    for j in range(R):
                        set_external_leaves(rows[j])
                        ids, votes, n = f.query(Q[qi], top_n=TOPN_ROW,
                                                query_depth=MED,
                                                allowed_state=ch)
                        pool.update(ids[:n].tolist())
                    set_external_leaves(None)
                    if len(pool) < 10:
                        starved += 1
                    cand = np.fromiter(pool, dtype=np.int32) \
                        if pool else np.empty(0, dtype=np.int32)
                    top = f.rerank_l2(BASE, Q[qi], cand) if len(cand) else []
                    lats.append((time.time() - t0) * 1000)
                    recs.append(len(set(np.asarray(top).tolist())
                                    & set(gt[qi].tolist())) / 10.0)
                    pools.append(len(pool))
                row = {'recall': round(float(np.mean(recs)), 4),
                       'p50_ms': round(float(np.percentile(lats, 50)), 1),
                       'pool_p10': int(np.percentile(pools, 10)),
                       'pool_mean': int(np.mean(pools)),
                       'starved_frac': round(starved / len(hard), 3)}
                res[f'{mode}_R{R}'] = row
                log(f'  {mode:6s} R={R:2d} recall {row["recall"]:.3f} '
                    f'p50 {row["p50_ms"]:7.1f} ms pool_p10 '
                    f'{row["pool_p10"]:>6,} affamées {row["starved_frac"]:.0%}')
        out[name] = res
        with open(f'{STORE}/offsupport.json', 'w') as fp:
            json.dump(out, fp, indent=1)
    log('OFFSUPPORT OK')


if __name__ == '__main__':
    main()
