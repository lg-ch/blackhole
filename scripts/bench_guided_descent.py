#!/usr/bin/env python3
"""Prototype DESCENTE GUIDÉE : à chaque nœud de la traversée, si le
sous-arbre choisi par la géométrie ne contient AUCUN éligible (masque de
vivacité issu du sidecar), on prend l'autre enfant. Propriété : la racine
contient |B| > 0 et vivacité(nœud) = OR(enfants) → chaque arbre atteint
TOUJOURS un bucket contenant des éligibles, même requête hors-support.

Rows k>0 : variantes multi-probe = flip forcé au k-ième niveau de plus
petite marge |c1-c0-θ| dont l'enfant flippé est vivant, puis descente
guidée en dessous (analogue client-side des probes C, avec veto).

Même harnais que bench_steer_recall (mêmes 100 requêtes, ch_state,
union+rerank, GT filtrées) -> comparaison directe blind/steer/guided.

Parité : row 0 sans veto DOIT égaler probe_leaves()[0] du C (médianes
armées) — vérifiée sur 5 requêtes avant la mesure.
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
from mangrove.traversal import (tree_seed, node_seed, pick_dims, gen_vec_v3,
                                load_medians)
from bench_steer_recall import roaring_ch_state

ROOT = os.environ.get('SEL_ROOT', '/root/mangrove-campaign')
IDX = f'{ROOT}/run/idx_live'
BASE = f'{ROOT}/data/deep50m.fbin'
STORE = f'{ROOT}/selstore'

NT, DIM, SD, DEPTH, NDOCS = 256, 96, 16, 22, 50_000_000
MED = 13
NQ = 100
TOPN_ROW = 100_000


def log(m):
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


_node_cache = {}


def node_planes(ts, node):
    """(dims, v0, v1) du nœud — cache global (hyperplans par (arbre,nœud),
    identiques pour toutes les requêtes)."""
    key = (ts, node)
    hit = _node_cache.get(key)
    if hit is None:
        dims = np.asarray(pick_dims(ts, node, DIM, SD), dtype=np.int64)
        v0 = gen_vec_v3(node_seed(ts, node * 2), SD)
        v1 = gen_vec_v3(node_seed(ts, node * 2 + 1), SD)
        hit = (dims, v0, v1)
        _node_cache[key] = hit
    return hit


def guided_walk(qvec, ts, med_row, alive, flip_level=-1):
    """Descente 0..MED avec veto de vivacité. alive[l] = bool (2^l,) ou None
    (pas de veto). flip_level >= 0 : flip forcé à ce niveau (si vivant).
    Retourne (leaf_pos, margins[MED]) — margins = |c1-c0-θ| par niveau."""
    node = 0        # id heap
    pos = 0         # position dans le niveau
    margins = np.empty(MED, dtype=np.float32)
    for level in range(MED):
        dims, v0, v1 = node_planes(ts, node)
        q_sub = qvec[dims]
        c0 = np.float32(np.dot(q_sub, v0))
        c1 = np.float32(np.dot(q_sub, v1))
        th = np.float32(med_row[node])
        diff = np.float32(c1 - c0) - th
        margins[level] = abs(diff)
        bit = 1 if diff > 0 else 0
        if level == flip_level:
            bit ^= 1
        if alive is not None:
            cand = 2 * pos + bit
            if not alive[level + 1][cand]:
                bit ^= 1
        pos = 2 * pos + bit
        node = 2 * node + 1 + bit
    return pos, margins


def guided_rows(qvec, med_tab, alive_per_tree, R):
    """R rows (R, NT) : row 0 = descente guidée ; rows suivantes = flip au
    j-ième niveau de plus petite marge (enfant flippé vivant exigé via le
    veto de la descente elle-même)."""
    rows = np.empty((R, NT), dtype=np.int32)
    for t in range(NT):
        ts = tree_seed(t)
        alive = alive_per_tree[t]
        leaf0, margins = guided_walk(qvec, ts, med_tab[t], alive)
        rows[0, t] = leaf0
        if R > 1:
            order = np.argsort(margins)
            j = 0
            for r in range(1, R):
                # cherche le prochain flip qui change réellement la feuille
                leaf = leaf0
                while j < MED:
                    lv = int(order[j])
                    j += 1
                    cand, _ = guided_walk(qvec, ts, med_tab[t], alive,
                                          flip_level=lv)
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

    # Le C normalise la requête avant la traversée (θ calibrés sur vecteurs
    # unitaires — le test de seuil n'est PAS invariant d'échelle).
    Qn = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-9)
    Qn = Qn.astype(np.float32)

    # --- parité : descente SANS veto == traverse_batch (le routeur C qui a
    # construit le sidecar ET qui route l ingest — la référence vraie ;
    # probe_leaves est sign-split, ne PAS l utiliser ici). ---
    tb = mf.traverse_batch(np.ascontiguousarray(Qn[:5]), SD, MED, NT)
    ok = tot = 0
    for qi in range(5):
        for t in range(NT):
            leaf, _ = guided_walk(Qn[qi], tree_seed(t), med_tab[t], None)
            ok += int(leaf == tb[qi, t])
            tot += 1
    log(f'parité descente Python vs traverse_batch : {ok}/{tot}')
    assert ok >= tot - 5, 'descente client non conforme au C'

    out = {}
    for name in ('cor1', 'cor02', 'ind1'):
        eids = np.load(f'{STORE}/ids_{name}.npy')
        gt = np.load(f'{STORE}/gt_{name}.npy')
        ch = roaring_ch_state(eids)
        log(f'--- {name} : |B|={len(eids):,} ---')

        counts = np.zeros((NT, 1 << MED), dtype=np.int32)
        t0 = time.time()
        for off in range(0, len(eids), 200_000):
            xv = np.ascontiguousarray(X[eids[off:off + 200_000]])
            bk = mf.traverse_batch(xv, SD, MED, NT)
            for t in range(NT):
                np.add.at(counts[t], bk[:, t], 1)
        # masques de vivacité hiérarchiques (l'objet "1 bit/nœud" du design)
        alive_per_tree = []
        for t in range(NT):
            lv = [None] * (MED + 1)
            lv[MED] = counts[t] > 0
            for l in range(MED - 1, -1, -1):
                lv[l] = lv[l + 1].reshape(-1, 2).any(axis=1)
            assert lv[0][0], f'arbre {t} : racine morte ?!'
            alive_per_tree.append(lv)
        log(f'  sidecar+masques : {time.time() - t0:.0f}s')

        res = {}
        for mode, masks in (('cblind', [None] * NT), ('guided', alive_per_tree)):
          for R in (1, 2, 4, 8):
            recs, lats, pools = [], [], []
            for qi in range(NQ):
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
            log(f'  {mode:6s} R={R:2d} recall {row["recall"]:.3f} '
                f'p50 {row["p50_ms"]:7.1f} ms pool {row["pool_mean"]:,}')
        out[name] = res
        with open(f'{STORE}/guided.json', 'w') as fp:
            json.dump(out, fp, indent=1)
    log('GUIDED OK')


if __name__ == '__main__':
    main()
