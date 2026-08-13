#!/usr/bin/env python3
"""Bench sélectivité : trace la courbe recall/latence des requêtes filtrées
sur toute la plage de sélectivité (0,01 % -> 50 %), pour trois familles de
prédicats :

  ind*  independant : membership par hash de l'id (décorrélé de la géométrie)
  cor*  corrélé     : demi-espace géométrique (proj > quantile) — simule une
                      métadonnée alignée sur le contenu (lang, sujet...)
  ten*  tenant      : intervalle d'ids contigu (SaaS multi-tenant)

Phases (argv[1]) :
  setup  une passe sur la base : GT filtrées brute-force par prédicat
         + insertion des memberships dans un MetaStore natif + cartes.
  sweep  courbe qd -> (recall@10 e2e, p50/mean pathrank et total) par
         prédicat, bitmap natif in-process (query_pathrank_meta).
  steer  sidecar de comptages (arbre, bucket med14) construit par routage
         des éligibles ; mesure Gini de concentration, probes gaspillées,
         gain d'ordonnancement par comptage vs marge géométrique.

Chemins surchargables par env SEL_* (défauts = campagne GB10).
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mangrove_ffi as mf
from mangrove_ffi import Forest, MetaStore

ROOT = os.environ.get('SEL_ROOT', '/root/mangrove-campaign')
IDX = os.environ.get('SEL_IDX', f'{ROOT}/run/idx_live')
BASE = os.environ.get('SEL_BASE', f'{ROOT}/data/deep50m.fbin')
QUERIES = os.environ.get('SEL_QUERIES', f'{ROOT}/data/queries.npy')
STORE = os.environ.get('SEL_STORE', f'{ROOT}/selstore')

NT, DIM, SD, DEPTH, NDOCS = 256, 96, 16, 22, 50_000_000
MED_DEPTH = 13   # idx_live 50M : medians.bin = seuils des niveaux 0..12
CHUNK = 2_000_000
NQ = 200
TOPN = 6000
SAVE_IDS_MAX = 1_200_000     # sauvegarde des ids éligibles si carte <= ça

# (nom, famille, sélectivité cible)
PREDS = [
    ('ind50', 'ind', 0.5),    ('ind20', 'ind', 0.2),
    ('ind5', 'ind', 0.05),    ('ind1', 'ind', 0.01),
    ('ind02', 'ind', 0.002),  ('ind005', 'ind', 5e-4),
    ('ind001', 'ind', 1e-4),
    ('cor5', 'cor', 0.05),    ('cor1', 'cor', 0.01),
    ('cor02', 'cor', 0.002),
    ('ten1', 'ten', 0.01),    ('ten01', 'ten', 0.001),
]


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def base_mmap():
    hdr = np.fromfile(BASE, dtype=np.uint32, count=2)
    assert hdr[0] == NDOCS and hdr[1] == DIM, f'header fbin inattendu: {hdr}'
    return np.memmap(BASE, dtype=np.float32, mode='r',
                     offset=8, shape=(NDOCS, DIM))


def hash_u(ids):
    """Uniforme [0,1) déterministe par id (hash multiplicatif 64 bits)."""
    h = ids.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)
    return (h >> np.uint64(40)).astype(np.float64) / float(1 << 24)


def cor_dir():
    return np.random.default_rng(42).standard_normal(DIM).astype(np.float32)


def memberships(ids, proj, thresholds):
    """dict nom -> masque bool pour un chunk (ids int64, proj float32)."""
    u = hash_u(ids)
    out = {}
    for name, fam, s in PREDS:
        if fam == 'ind':
            out[name] = u < s
        elif fam == 'cor':
            out[name] = proj >= thresholds[name]
        else:  # ten : intervalle contigu, décalé pour ten01 (pas de nesting)
            lo = 0 if name == 'ten1' else 10_000_000
            hi = lo + int(s * NDOCS)
            out[name] = (ids >= lo) & (ids < hi)
    return out


# ---------------------------------------------------------------- setup
def phase_setup():
    os.makedirs(STORE, exist_ok=True)
    X = base_mmap()
    Q = np.load(QUERIES)[:NQ].astype(np.float32)
    w = cor_dir()

    log('seuils cor* par quantiles sur échantillon 2M...')
    sample = np.asarray(X[:2_000_000])
    ps = sample @ w
    thresholds = {name: float(np.quantile(ps, 1.0 - s))
                  for name, fam, s in PREDS if fam == 'cor'}
    del sample, ps

    bd = {n: np.full((NQ, 10), np.inf, dtype=np.float32) for n, _, _ in PREDS}
    bi = {n: np.zeros((NQ, 10), dtype=np.int64) for n, _, _ in PREDS}
    cards = {n: 0 for n, _, _ in PREDS}
    saved_ids = {n: [] for n, _, _ in PREDS}
    ms = MetaStore(STORE)

    t0 = time.time()
    for off in range(0, NDOCS, CHUNK):
        n = min(CHUNK, NDOCS - off)
        xb = np.asarray(X[off:off + n])
        ids = np.arange(off, off + n, dtype=np.int64)
        proj = xb @ w
        x2 = np.einsum('ij,ij->i', xb, xb)
        d2 = x2[:, None] - 2.0 * (xb @ Q.T)          # (n, NQ), ordre L2 exact
        mem = memberships(ids, proj, thresholds)
        for name, fam, s in PREDS:
            m = mem[name]
            cnt = int(m.sum())
            if cnt == 0:
                continue
            cards[name] += cnt
            eids = ids[m]
            ms.add('sel', name, eids.astype(np.uint32))
            saved_ids[name].append(eids)
            dm = d2[m]                               # (cnt, NQ)
            k = min(10, cnt)
            part = np.argpartition(dm, k - 1, axis=0)[:k]     # (k, NQ)
            cd = np.take_along_axis(dm, part, axis=0)
            ci = eids[part]
            alld = np.concatenate([bd[name], cd.T], axis=1)   # (NQ, 10+k)
            alli = np.concatenate([bi[name], ci.T], axis=1)
            order = np.argsort(alld, axis=1)[:, :10]
            bd[name] = np.take_along_axis(alld, order, axis=1)
            bi[name] = np.take_along_axis(alli, order, axis=1)
        log(f'setup {off + n:>9,}/{NDOCS:,}  ({time.time() - t0:.0f}s)')

    n_frozen = ms.compact()
    log(f'meta compact : {n_frozen} clés gelées')
    for name, fam, s in PREDS:
        np.save(f'{STORE}/gt_{name}.npy', bi[name])
        if cards[name] <= SAVE_IDS_MAX:
            allids = np.concatenate(saved_ids[name]).astype(np.uint32)
            np.save(f'{STORE}/ids_{name}.npy', allids)
    state = {'cards': cards, 'thresholds': thresholds,
             'preds': [[n, f, s] for n, f, s in PREDS],
             'setup_s': round(time.time() - t0, 1)}
    with open(f'{STORE}/state.json', 'w') as fp:
        json.dump(state, fp, indent=1)
    for name, fam, s in PREDS:
        log(f'  {name:7s} cible {s:8.4%}  réel {cards[name] / NDOCS:8.4%} '
            f'({cards[name]:,} docs)')
    log(f'SETUP OK en {time.time() - t0:.0f}s')


# ---------------------------------------------------------------- sweep
def open_forest():
    f = Forest(IDX, n_trees=NT, dim=DIM, sub_dim=SD,
               depth=DEPTH, n_docs=NDOCS)
    md = mf.load_live_medians(IDX)
    assert md == MED_DEPTH, f'med_depth {md} != {MED_DEPTH}'
    return f


def phase_sweep():
    state = json.load(open(f'{STORE}/state.json'))
    f = open_forest()
    ms = MetaStore(STORE)
    Q = np.load(QUERIES)[:NQ].astype(np.float32)
    X = base_mmap()

    log('préchauffage du cache (base + feuilles, lecture 1 query/qd)...')
    for qd in (18, 15, 12):
        bmp0 = ms.filter({'sel': 'ind5'})
        f.query_pathrank_meta(Q[0], 3, 1024, bmp0, TOPN, qd)
        MetaStore.filter_free(bmp0)

    results = {}
    qds = [22, 20, 18, 17, 16, 15, 14, 13, 12]
    for name, fam, s in PREDS:
        gt = np.load(f'{STORE}/gt_{name}.npy')
        bmp = ms.filter({'sel': name})
        card = MetaStore.filter_card(bmp)
        log(f'--- {name} (|B|={card:,}, {card / NDOCS:.4%}) ---')
        curve = []
        for qd in qds:
            tp_list, tt_list, rec = [], [], []
            aborted = False
            t_qd = time.time()
            for qi in range(NQ):
                t0 = time.time()
                ids, votes, n = f.query_pathrank_meta(
                    Q[qi], 3, 1024, bmp, TOPN, qd)
                t1 = time.time()
                top10 = f.rerank_l2(BASE, Q[qi], ids[:n]) if n > 0 else []
                t2 = time.time()
                tp_list.append((t1 - t0) * 1000)
                tt_list.append((t2 - t0) * 1000)
                rec.append(len(set(np.asarray(top10).tolist())
                               & set(gt[qi].tolist())) / 10.0)
                if time.time() - t_qd > 240:        # garde-fou runtime
                    aborted = True
                    break
            row = {'qd': qd, 'n_q': len(rec),
                   'recall': round(float(np.mean(rec)), 4),
                   'path_p50_ms': round(float(np.percentile(tp_list, 50)), 1),
                   'tot_p50_ms': round(float(np.percentile(tt_list, 50)), 1),
                   'tot_mean_ms': round(float(np.mean(tt_list)), 1),
                   'aborted': aborted}
            curve.append(row)
            log(f'  qd{qd:2d} recall {row["recall"]:.3f} '
                f'p50 {row["tot_p50_ms"]:6.1f} ms'
                + (' [ABORT 240s]' if aborted else ''))
            if aborted and qd <= 16:
                log('  qd plus bas: coût explosif, on coupe')
                break
        MetaStore.filter_free(bmp)

        entry = {'card': card, 'sel': card / NDOCS, 'family': fam,
                 'curve': curve}
        # référence brute-force filtrée (gather RAM) pour les petits |B|
        idpath = f'{STORE}/ids_{name}.npy'
        if os.path.exists(idpath):
            eids = np.load(idpath)
            xe = np.asarray(X[eids])                # gather une fois
            x2e = np.einsum('ij,ij->i', xe, xe)
            bts, brec = [], []
            for qi in range(NQ):
                t0 = time.time()
                d2 = x2e - 2.0 * (xe @ Q[qi])
                k = min(10, len(eids))
                top = eids[np.argpartition(d2, k - 1)[:k]]
                bts.append((time.time() - t0) * 1000)
                brec.append(len(set(top.tolist())
                                & set(gt[qi].tolist())) / 10.0)
            entry['brute'] = {
                'p50_ms': round(float(np.percentile(bts, 50)), 1),
                'recall': round(float(np.mean(brec)), 4)}
            log(f'  brute |B|={len(eids):,} p50 '
                f'{entry["brute"]["p50_ms"]} ms '
                f'recall {entry["brute"]["recall"]:.3f}')
            del xe
        results[name] = entry
        with open(f'{STORE}/sweep.json', 'w') as fp:
            json.dump(results, fp, indent=1)
    log('SWEEP OK')


# ---------------------------------------------------------------- steer
def gini(counts):
    """Gini de concentration d'un vecteur de comptages (0=plat, 1=concentré)."""
    c = np.sort(counts.astype(np.float64))
    n = len(c)
    tot = c.sum()
    if tot == 0:
        return 0.0
    lorenz = np.cumsum(c) / tot
    return float(1.0 - 2.0 * np.trapz(lorenz, dx=1.0 / n))


def phase_steer():
    f = open_forest()
    Q = np.load(QUERIES)[:NQ].astype(np.float32)
    X = base_mmap()
    out = {}
    for name in ('cor1', 'ind1', 'cor02', 'ten1'):
        idpath = f'{STORE}/ids_{name}.npy'
        if not os.path.exists(idpath):
            log(f'{name}: pas d ids sauvés, skip')
            continue
        eids = np.load(idpath)
        log(f'--- {name} : sidecar comptages ({len(eids):,} éligibles) ---')
        counts = np.zeros((NT, 1 << MED_DEPTH), dtype=np.int32)
        t0 = time.time()
        for off in range(0, len(eids), 200_000):
            batch = eids[off:off + 200_000]
            xv = np.ascontiguousarray(X[batch])
            buckets = mf.traverse_batch(xv, SD, MED_DEPTH, NT)  # (n, NT)
            for t in range(NT):
                np.add.at(counts[t], buckets[:, t], 1)
        assert counts.sum() == len(eids) * NT
        log(f'  routage éligibles : {time.time() - t0:.0f}s')
        g_per_tree = [gini(counts[t]) for t in range(NT)]
        empty_frac = float((counts == 0).mean())

        # probes à qd=med14 : gaspillage + gain d'ordonnancement
        NPROBE = 31
        POOL = 2000
        zero_hit, n_marge, n_count = [], [], []
        for qi in range(NQ):
            probes = f.probe_leaves(Q[qi], NPROBE, probe_depth=MED_DEPTH)
            cnt = counts[np.arange(NT)[None, :], probes]      # (32, NT)
            flat = cnt.ravel()
            zero_hit.append(float((flat == 0).mean()))
            # ordre naturel (rang de marge) vs tri décroissant par comptage
            cum_m = np.cumsum(flat)
            cum_c = np.cumsum(np.sort(flat)[::-1])
            n_marge.append(int(np.searchsorted(cum_m, POOL) + 1))
            n_count.append(int(np.searchsorted(cum_c, POOL) + 1))
        out[name] = {
            'n_eligible': int(len(eids)),
            'gini_mean': round(float(np.mean(g_per_tree)), 4),
            'bucket_empty_frac': round(empty_frac, 4),
            'probe_zero_frac': round(float(np.mean(zero_hit)), 4),
            'probes_pool2000_margin_order_p50':
                int(np.percentile(n_marge, 50)),
            'probes_pool2000_count_order_p50':
                int(np.percentile(n_count, 50)),
        }
        log(f'  gini {out[name]["gini_mean"]:.3f}  '
            f'buckets vides {empty_frac:.1%}  '
            f'probes gaspillées {out[name]["probe_zero_frac"]:.1%}  '
            f'probes pour pool 2000 : marge '
            f'{out[name]["probes_pool2000_margin_order_p50"]} vs comptage '
            f'{out[name]["probes_pool2000_count_order_p50"]}')
        with open(f'{STORE}/steer.json', 'w') as fp:
            json.dump(out, fp, indent=1)
    log('STEER OK')


if __name__ == '__main__':
    phase = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if phase in ('setup', 'all'):
        phase_setup()
    if phase in ('sweep', 'all'):
        phase_sweep()
    if phase in ('steer', 'all'):
        phase_steer()
