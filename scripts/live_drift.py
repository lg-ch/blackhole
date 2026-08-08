"""live_drift — instrumentation de dérive + health-check d'injection live
sur un index médianes RÉEL, sans le modifier (HOT overlay en dossier temp,
queries en lecture seule).

Deux fonctions :

1. DÉRIVE : route des échantillons sous les θ gelés et compare l'occupation
   des buckets de niveau med_depth (ceux que les médianes équilibrent) :
     - baseline   : échantillon stridé du corpus (la distribution calibrée)
     - stream     : échantillon disjoint (simule le flux entrant)
     - shifted    : flux artificiellement décalé (prouve que la métrique
                    détecte une vraie dérive)
   Métriques par arbre : max/mean (balance) et coefficient de variation.
   Règle pratique : stream ≈ baseline → seuils toujours bons ; stream qui
   décroche (ratio ×2+) → planifier recalibration/delta-forêt.

2. E2E LIVE : insère N docs via le chemin réel (mg_traverse_sub médianes
   armées → mg_hot_append_batch), puis vérifie que chaque doc est retrouvé
   par la query avec des votes pleins. Mesure débit d'insert et RSS.

Usage :
  python3 live_drift.py --index ~/deep10m/idx_med --base ~/deep10m/base.fbin
"""
import argparse
import os
import shutil
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mangrove_ffi as mf                                   # noqa: E402
from mangrove_ffi import Forest, set_gen_version            # noqa: E402
from ctypes import c_void_p, c_char_p, c_int, c_uint32, POINTER  # noqa: E402

mf._lib.mg_hot_init.argtypes   = [c_int, c_int, c_char_p]
mf._lib.mg_hot_init.restype    = c_void_p
mf._lib.mg_hot_free.argtypes   = [c_void_p]
mf._lib.mg_hot_free.restype    = None
mf._lib.mg_hot_append_batch.argtypes = [c_void_p, POINTER(c_int),
                                        POINTER(c_uint32), POINTER(c_uint32),
                                        c_int]
mf._lib.mg_hot_append_batch.restype  = c_int
mf._lib.mg_forest_set_hot_overlay.argtypes = [c_void_p]
mf._lib.mg_forest_set_hot_overlay.restype  = None


def read_meta(index_dir):
    meta = {}
    with open(os.path.join(index_dir, 'meta.txt')) as fh:
        for line in fh:
            p = line.split()
            if len(p) == 2:
                meta[p[0]] = int(p[1]) if p[1].lstrip('-').isdigit() else p[1]
    return meta


def load_fbin_sample(path, n, dim, offset_rows=0, stride=None):
    """Échantillon stridé d'un .fbin (header 8 B : n_total, dim)."""
    hdr = np.fromfile(path, dtype=np.int32, count=2)
    n_total, d = int(hdr[0]), int(hdr[1])
    assert d == dim, f'dim fbin {d} != meta {dim}'
    if stride is None:
        stride = max(1, (n_total - offset_rows) // n)
    rows = (offset_rows + np.arange(n) * stride) % n_total
    out = np.empty((n, dim), dtype=np.float32)
    row_bytes = dim * 4
    with open(path, 'rb') as fh:
        for i, r in enumerate(rows):
            fh.seek(8 + int(r) * row_bytes)
            out[i] = np.frombuffer(fh.read(row_bytes), dtype=np.float32)
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
    return out


def bucket_metrics(vecs, trees, depth, med_depth, sub_dim):
    """Route vecs (médianes live armées) ; occupation des buckets de niveau
    med_depth par arbre → (mean max/mean, mean CV)."""
    shift = depth - med_depth
    ratios, cvs = [], []
    for t in trees:
        counts = np.zeros(1 << med_depth, dtype=np.int64)
        for v in vecs:
            node = mf.traverse_sub(v, sub_dim, depth, t)
            leaf = node - ((1 << depth) - 1)
            counts[leaf >> shift] += 1
        nz_mean = counts.mean()
        ratios.append(counts.max() / max(nz_mean, 1e-9))
        cvs.append(counts.std() / max(nz_mean, 1e-9))
    return float(np.mean(ratios)), float(np.mean(cvs))


def rss_mb():
    with open('/proc/self/status') as fh:
        for line in fh:
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) / 1024.0
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', required=True)
    ap.add_argument('--base', required=True)
    ap.add_argument('--n-sample', type=int, default=3000,
                    help='taille des échantillons de dérive')
    ap.add_argument('--n-insert', type=int, default=300)
    ap.add_argument('--drift-trees', type=int, default=4,
                    help="nb d'arbres échantillonnés pour l'histogramme")
    args = ap.parse_args()
    idir = os.path.expanduser(args.index)
    base = os.path.expanduser(args.base)

    meta = read_meta(idir)
    nt, depth = meta['n_trees'], meta['depth']
    dim, sd   = meta['dim'], meta['sub_dim']
    n_docs    = meta['n_docs']
    gen       = meta.get('gen_version', 3)
    set_gen_version(gen)
    md = mf.load_live_medians(idir)
    assert md > 0, 'index sans medians.bin — rien à tester ici'
    print(f'index: {nt} arbres, depth {depth}, dim {dim}, med_depth {md}, '
          f'{n_docs/1e6:.0f}M docs')

    trees = list(range(args.drift_trees))
    t0 = time.time()
    baseline = load_fbin_sample(base, args.n_sample, dim, offset_rows=0)
    stream   = load_fbin_sample(base, args.n_sample, dim, offset_rows=1)
    rng = np.random.default_rng(4242)
    delta = rng.standard_normal(dim).astype(np.float32)
    delta /= np.linalg.norm(delta)
    shifted = stream + 0.35 * delta
    shifted /= np.linalg.norm(shifted, axis=1, keepdims=True)
    print(f'échantillons chargés ({time.time()-t0:.0f}s)')

    for name, s in (('baseline', baseline), ('stream  ', stream),
                    ('shifted ', shifted)):
        t0 = time.time()
        ratio, cv = bucket_metrics(s, trees, depth, md, sd)
        print(f'[dérive] {name} : max/mean {ratio:5.2f}, CV {cv:.3f} '
              f'({time.time()-t0:.0f}s, {args.drift_trees} arbres, '
              f'{args.n_sample} vecs)')

    # ---------- E2E live sur l'index réel (lecture seule + HOT temp) ------
    f = Forest(idir, n_trees=nt, dim=dim, sub_dim=sd, depth=depth,
               n_docs=n_docs, gen_version=gen)
    hotdir = tempfile.mkdtemp(prefix='drift_hot_')
    hot = mf._lib.mg_hot_init(nt, depth, hotdir.encode())
    assert hot, 'mg_hot_init failed'
    mf._lib.mg_forest_set_hot_overlay(hot)

    ins = load_fbin_sample(base, args.n_insert, dim, offset_rows=2)
    lbase = (1 << depth) - 1
    tree_ids = np.arange(nt, dtype=np.int32)
    t0 = time.time()
    for i in range(args.n_insert):
        leaves = np.empty(nt, dtype=np.uint32)
        for t in range(nt):
            leaves[t] = mf.traverse_sub(ins[i], sd, depth, t) - lbase
        doc_ids = np.full(nt, n_docs + i, dtype=np.uint32)
        rc = mf._lib.mg_hot_append_batch(
            hot, tree_ids.ctypes.data_as(POINTER(c_int)),
            leaves.ctypes.data_as(POINTER(c_uint32)),
            doc_ids.ctypes.data_as(POINTER(c_uint32)), nt)
        assert rc == 0
    dt = time.time() - t0
    print(f'[insert] {args.n_insert} docs × {nt} arbres en {dt:.1f}s '
          f'→ {args.n_insert/dt:.0f} vec/s (routage+append, mono-thread)')

    votes_all, found = [], 0
    t0 = time.time()
    for i in range(args.n_insert):
        ids, votes, n = f.query_pathrank(ins[i], 0, nt, 6000, query_depth=depth)
        v = 0
        for j in range(n):
            if int(ids[j]) == n_docs + i:
                v = int(votes[j]); break
        votes_all.append(v)
        found += (v > 0)
    dt = time.time() - t0
    print(f'[query ] {found}/{args.n_insert} docs live retrouvés, votes '
          f'moyens {np.mean(votes_all):.0f}/{nt} '
          f'({dt/args.n_insert*1000:.0f} ms/query sous charge)')
    print(f'[ram   ] RSS process : {rss_mb():.0f} MB '
          f'(médianes {nt*((1<<md)-1)*4/1e6:.0f} MB incluses)')

    mf._lib.mg_forest_set_hot_overlay(None)
    mf._lib.mg_hot_free(hot)
    f.close()
    shutil.rmtree(hotdir, ignore_errors=True)

    ok = (found == args.n_insert and np.mean(votes_all) > nt * 0.97)
    print('PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
