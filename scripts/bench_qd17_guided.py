#!/usr/bin/env python3
"""qd17 vs qd13 : le guidage par corrélation permet-il de rester à la
profondeur optimale ? Une row qd17 lit 2^5 feuilles/arbre (vs 2^9 à qd13)
= 16x moins d'I/O. On compare aveugle/guidé à qd17, R ∈ {8, 16}, contre
les points qd13 de guided.json (mêmes 100 requêtes, mêmes GT).

Descente : niveaux 0..12 = médianes + veto de vivacité (comptages) ;
niveaux 13..16 = sign-split géométrique (pas de comptages sous med13 —
la garantie jamais-mort ne vaut qu'à la granularité bucket).

Optimisation : la marche Python complète ne sert que là où elle diffère du
C — le walk de base stocke les états par niveau, et chaque row-variante
(flip au j-ième niveau de plus petite marge) ne recalcule que les niveaux
sous le flip.
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
from mangrove.traversal import tree_seed, load_medians
from bench_steer_recall import roaring_ch_state
from bench_guided_descent import node_planes

ROOT = os.environ.get('SEL_ROOT', '/root/mangrove-campaign')
IDX = f'{ROOT}/run/idx_live'
BASE = f'{ROOT}/data/deep50m.fbin'
STORE = f'{ROOT}/selstore'

NT, DIM, SD, DEPTH, NDOCS = 256, 96, 16, 22, 50_000_000
MED = 13
QD = int(os.environ.get('SEL_QD', '17'))
NQ = 100
TOPN_ROW = 100_000
RS = tuple(int(x) for x in os.environ.get('SEL_RS', '8,16').split(','))


def log(m):
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def walk_full(qvec, ts, med_row, alive):
    """Marche 0..QD avec veto (si alive) sur les niveaux < MED.
    Retourne (leaf, states, margins) ; states[l] = (node, pos) AVANT le
    niveau l — permet une continuation depuis n'importe quel niveau."""
    node = 0
    pos = 0
    states = [(0, 0)] * QD
    margins = np.empty(QD, dtype=np.float32)
    for level in range(QD):
        states[level] = (node, pos)
        dims, v0, v1 = node_planes(ts, node)
        q_sub = qvec[dims]
        c0 = np.float32(np.dot(q_sub, v0))
        c1 = np.float32(np.dot(q_sub, v1))
        th = np.float32(med_row[node]) if node < len(med_row) else np.float32(0)
        diff = np.float32(c1 - c0) - th
        margins[level] = abs(diff)
        bit = 1 if diff > 0 else 0
        if alive is not None and level < MED:
            cand = 2 * pos + bit
            if not alive[level + 1][cand]:
                bit ^= 1
        pos = 2 * pos + bit
        node = 2 * node + 1 + bit
    return pos, states, margins


def walk_from(qvec, ts, med_row, alive, state, level0, flip):
    """Continuation depuis states[level0] avec flip forcé au niveau level0."""
    node, pos = state
    for level in range(level0, QD):
        dims, v0, v1 = node_planes(ts, node)
        q_sub = qvec[dims]
        c0 = np.float32(np.dot(q_sub, v0))
        c1 = np.float32(np.dot(q_sub, v1))
        th = np.float32(med_row[node]) if node < len(med_row) else np.float32(0)
        bit = 1 if (np.float32(c1 - c0) - th) > 0 else 0
        if level == level0 and flip:
            bit ^= 1
        if alive is not None and level < MED:
            cand = 2 * pos + bit
            if not alive[level + 1][cand]:
                bit ^= 1
        pos = 2 * pos + bit
        node = 2 * node + 1 + bit
    return pos


def make_rows(qvec, med_tab, masks, R):
    rows = np.empty((R, NT), dtype=np.int32)
    for t in range(NT):
        ts = tree_seed(t)
        alive = masks[t]
        leaf0, states, margins = walk_full(qvec, ts, med_tab[t], alive)
        rows[0, t] = leaf0
        if R > 1:
            order = np.argsort(margins)
            j = 0
            for r in range(1, R):
                leaf = leaf0
                while j < QD:
                    lv = int(order[j])
                    j += 1
                    cand = walk_from(qvec, ts, med_tab[t], alive,
                                     states[lv], lv, True)
                    if cand != leaf0:
                        leaf = cand
                        break
                rows[r, t] = leaf
    return rows


def main():
    f = Forest(IDX, n_trees=NT, dim=DIM, sub_dim=SD, depth=DEPTH,
               n_docs=NDOCS)
    assert mf.load_live_medians(IDX) == MED
    med_tab, md = load_medians(IDX)
    assert md == MED
    Q = np.load(f'{ROOT}/data/queries.npy')[:NQ].astype(np.float32)
    X = np.memmap(BASE, dtype=np.float32, mode='r', offset=8,
                  shape=(NDOCS, DIM))
    Qn = (Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-9)
          ).astype(np.float32)

    # parité walk qd17 sans veto vs traverse_batch à depth 17
    tb = mf.traverse_batch(np.ascontiguousarray(Qn[:5]), SD, QD, NT)
    ok = sum(int(walk_full(Qn[qi], tree_seed(t), med_tab[t], None)[0]
                 == tb[qi, t])
             for qi in range(5) for t in range(NT))
    log(f'parité walk qd17 vs traverse_batch : {ok}/1280')
    assert ok >= 1275

    out = {}
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
                recs, lats, walks, pools = [], [], [], []
                for qi in range(NQ):
                    t0 = time.time()
                    rows = make_rows(Qn[qi], med_tab, masks, R)
                    t1 = time.time()
                    pool = set()
                    for j in range(R):
                        set_external_leaves(rows[j])
                        ids, votes, n = f.query(Q[qi], top_n=TOPN_ROW,
                                                query_depth=QD,
                                                allowed_state=ch)
                        pool.update(ids[:n].tolist())
                    set_external_leaves(None)
                    cand = np.fromiter(pool, dtype=np.int32) \
                        if pool else np.empty(0, dtype=np.int32)
                    top = f.rerank_l2(BASE, Q[qi], cand) if len(cand) else []
                    lats.append((time.time() - t0) * 1000)
                    walks.append((t1 - t0) * 1000)
                    recs.append(len(set(np.asarray(top).tolist())
                                    & set(gt[qi].tolist())) / 10.0)
                    pools.append(len(pool))
                row = {'recall': round(float(np.mean(recs)), 4),
                       'p50_ms': round(float(np.percentile(lats, 50)), 1),
                       'walk_p50_ms': round(float(np.percentile(walks, 50)), 1),
                       'pool_mean': int(np.mean(pools))}
                res[f'{mode}_R{R}'] = row
                log(f'  {mode:6s} R={R:2d} recall {row["recall"]:.3f} '
                    f'p50 {row["p50_ms"]:7.1f} ms (dont walk py '
                    f'{row["walk_p50_ms"]:.0f}) pool {row["pool_mean"]:,}')
        out[name] = res
        with open(f'{STORE}/qd{QD}.json', 'w') as fp:
            json.dump(out, fp, indent=1)
    log(f'QD{QD} OK')


if __name__ == '__main__':
    main()
