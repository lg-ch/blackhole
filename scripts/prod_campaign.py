"""prod_campaign — répétition générale de la prod : croissance par
injection live 1M → 10M → 20M → 50M avec checkpoints benchmarkés.

Scénario (DEEP, préfixes de deep100m/base.fbin, doc_id = numéro de ligne) :

  S0    bulk build 1M médianes (d17, md10) + calibration
  S1    injection live 1M→10M  : routage batché médianes (traverse_all_trees)
        → hot_append_batch, compaction background pendant l'ingest,
        drain complet à la fin
  CP10  checkpoint "vieilli" (index d17, θ_1M, corpus 10M) :
          drift metric, qd_calibrate, bench recall/latence/RSS
        puis recalibration + rebuild "frais" (d20, md12) : re-bench
        → la différence vieilli/frais = la valeur de la recalibration
  S2    injection 10M→20M sur l'index frais, CP20 (rebuild d21, md12)
  S3    injection 20M→50M, CP50 (rebuild d22, md13)

Résumable : chaque étape écrit son résultat dans journal.json et est
sautée si déjà marquée done. Tout vit sous ~/campaign/.

Usage : python3 prod_campaign.py [--until 10|20|50]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mangrove_ffi as mf                                   # noqa: E402
from mangrove_ffi import Forest, set_gen_version            # noqa: E402
from mangrove_calibrate import read_meta                    # noqa: E402
from qd_calibrate import bench_qd                           # noqa: E402
from live_drift import bucket_metrics, load_fbin_sample     # noqa: E402
from ctypes import c_void_p, c_char_p, c_int, c_uint32, POINTER  # noqa: E402

# Chemins surchargables par env (campagne locale WSL vs serveur) :
#   CAMPAIGN_BASE, CAMPAIGN_QUERIES, CAMPAIGN_ROOT, CAMPAIGN_GT_DIR
BASE = os.path.expanduser(os.environ.get('CAMPAIGN_BASE',
                                         '~/deep100m/base.fbin'))
QUERIES = os.path.expanduser(os.environ.get('CAMPAIGN_QUERIES',
                                            '~/deep1m/queries.npy'))
ROOT = os.path.expanduser(os.environ.get('CAMPAIGN_ROOT', '~/campaign'))
RPF = os.path.join(os.path.dirname(HERE), 'rpforest')
DIM, SD, NT, GEN = 96, 16, 256, 3

_GTD = os.environ.get('CAMPAIGN_GT_DIR', '')
if _GTD:
    GT = {n: os.path.join(_GTD, f'gt_{n}m.npy') for n in (10, 20, 50)}
else:
    GT = {10: os.path.expanduser('~/deep10m/gt_top10.npy'),
          20: os.path.expanduser('~/deep100m/gt_20m.npy'),
          50: os.path.expanduser('~/deep100m/gt_50m.npy')}

# échelle → (depth build frais, med_depth)
LADDER = {1: (17, 10), 10: (20, 12), 20: (21, 12), 50: (22, 13)}

for name, argt in (('mg_hot_init', [c_int, c_int, c_char_p]),
                   ('mg_hot_free', [c_void_p]),
                   ('mg_forest_set_hot_overlay', [c_void_p]),
                   ('mg_hot_compact_all', [c_void_p, c_void_p, c_char_p]),
                   ('mg_hot_compact_bg_stop', [c_void_p]),
                   ('mg_calibrate_medians',
                    [c_char_p, c_char_p, c_int, c_int, c_int, c_int, c_int])):
    fn = getattr(mf._lib, name)
    fn.argtypes = argt
mf._lib.mg_hot_init.restype = c_void_p
mf._lib.mg_hot_free.restype = None
mf._lib.mg_forest_set_hot_overlay.restype = None
mf._lib.mg_hot_compact_all.restype = c_int
mf._lib.mg_hot_compact_bg_stop.restype = None
mf._lib.mg_calibrate_medians.restype = c_int
mf._lib.mg_hot_compact_bg_start.argtypes = [c_void_p, c_void_p, c_char_p,
                                            c_int, c_int, c_int]
mf._lib.mg_hot_compact_bg_start.restype = c_void_p
mf._lib.mg_hot_append_batch.argtypes = [c_void_p, POINTER(c_int),
                                        POINTER(c_uint32), POINTER(c_uint32),
                                        c_int]
mf._lib.mg_hot_append_batch.restype = c_int

JOURNAL = os.path.join(ROOT, 'journal.json')


def jload():
    if os.path.exists(JOURNAL):
        return json.load(open(JOURNAL))
    return {}


def jsave(j):
    tmp = JOURNAL + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(j, fh, indent=2)
    os.replace(tmp, JOURNAL)


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def rss_mb():
    with open('/proc/self/status') as fh:
        for line in fh:
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) / 1024.0
    return 0.0


def write_meta_ndocs(idir, n_docs):
    """Après un leg d'injection le corpus a grossi : meta.txt doit refléter
    le n_docs courant pour les benchs."""
    meta = read_meta(idir)
    meta['n_docs'] = n_docs
    with open(os.path.join(idir, 'meta.txt'), 'w') as fh:
        for k, v in meta.items():
            fh.write(f'{k} {v}\n')


def build_fresh(idir, n_docs, depth, med_depth):
    """Calibration + bulk build médianes sur les n_docs premiers vecteurs."""
    os.makedirs(idir, exist_ok=True)
    set_gen_version(GEN)
    if not os.path.exists(os.path.join(idir, 'medians.bin')):
        log(f'  calibration médianes md={med_depth}...')
        rc = mf._lib.mg_calibrate_medians(
            BASE.encode(), idir.encode(), NT, DIM, SD, med_depth,
            min(1_500_000, n_docs))
        assert rc == 0, f'calibrate rc={rc}'
    if not os.path.exists(os.path.join(idir, f'tree{NT-1:05d}.srt')):
        log(f'  bulk build d={depth} sur {n_docs//1_000_000}M docs...')
        t0 = time.time()
        # PAS de --fast ici : il garde les paires en RAM (32 GB de RSS sur
        # le serveur 64 GB) — la VM WSL locale n'y survit pas. Le chemin
        # batché sur disque tient dans quelques GB quelle que soit l'échelle.
        subprocess.run([RPF, 'build', BASE, idir, str(NT), str(depth),
                        '--sub_dim', str(SD), '--gen', f'v{GEN}',
                        '--dim', str(DIM), '--doc_count', str(n_docs)],
                       check=True, capture_output=True)
        log(f'  build fini en {time.time()-t0:.0f}s')


def inject_leg(idir, from_doc, to_doc):
    """Injection live [from_doc, to_doc) : routage batché médianes +
    hot_append_batch + compaction background, drain final."""
    meta = read_meta(idir)
    depth = int(meta['depth'])
    set_gen_version(GEN)
    mf.clear_live_medians()
    assert mf.load_live_medians(idir) > 0
    f = Forest(idir, n_trees=NT, dim=DIM, sub_dim=SD, depth=depth,
               n_docs=from_doc, gen_version=GEN)
    hotdir = os.path.join(idir, 'hot')
    hot = mf._lib.mg_hot_init(NT, depth, hotdir.encode())
    assert hot
    bg = mf._lib.mg_hot_compact_bg_start(f._h, hot, idir.encode(),
                                         50_000, 200, 3)
    tree_ids = np.arange(NT, dtype=np.int32)
    row_bytes = DIM * 4
    CH = 20_000
    t0 = time.time()
    n_total = to_doc - from_doc
    with open(BASE, 'rb') as fh:
        done = 0
        while done < n_total:
            n = min(CH, n_total - done)
            fh.seek(8 + (from_doc + done) * row_bytes)
            chunk = np.frombuffer(fh.read(n * row_bytes),
                                  dtype=np.float32).reshape(n, DIM).copy()
            chunk /= np.maximum(
                np.linalg.norm(chunk, axis=1, keepdims=True), 1e-12)
            for i in range(n):
                leaves = mf.traverse_all_trees(chunk[i], SD, depth, NT)
                doc = np.full(NT, from_doc + done + i, dtype=np.uint32)
                rc = mf._lib.mg_hot_append_batch(
                    hot, tree_ids.ctypes.data_as(POINTER(c_int)),
                    leaves.astype(np.uint32).ctypes.data_as(POINTER(c_uint32)),
                    doc.ctypes.data_as(POINTER(c_uint32)), NT)
                assert rc == 0
            done += n
            if done % 500_000 < CH:
                rate = done / (time.time() - t0)
                log(f'  inject {done/1e6:.1f}M/{n_total/1e6:.1f}M '
                    f'({rate:.0f} vec/s, RSS {rss_mb():.0f} MB)')
    rate = n_total / (time.time() - t0)
    log(f'  injection finie : {rate:.0f} vec/s moyen')
    log('  drain final (compact all)...')
    mf._lib.mg_hot_compact_bg_stop(bg)
    t0 = time.time()
    rc = mf._lib.mg_hot_compact_all(f._h, hot, idir.encode())
    assert rc == 0, f'compact_all rc={rc}'
    log(f'  drain fini en {time.time()-t0:.0f}s')
    mf._lib.mg_forest_set_hot_overlay(None)
    mf._lib.mg_hot_free(hot)
    f.close()
    shutil.rmtree(hotdir, ignore_errors=True)
    write_meta_ndocs(idir, to_doc)
    return {'rate_vec_s': round(rate), 'n_injected': n_total}


def checkpoint(idir, scale_m, tag):
    """Drift + qd + bench sur l'index tel quel. Retourne le dict résultats."""
    res = {}
    meta = read_meta(idir)
    depth = int(meta['depth'])
    set_gen_version(GEN)
    mf.clear_live_medians()
    md = mf.load_live_medians(idir)
    baseline = load_fbin_sample(BASE, 2000, DIM, offset_rows=0)
    stream = load_fbin_sample(BASE, 2000, DIM,
                              offset_rows=scale_m * 1_000_000 - 2_000_001)
    r_b, _ = bucket_metrics(baseline, range(4), depth, md, SD)
    r_s, _ = bucket_metrics(stream, range(4), depth, md, SD)
    res['drift'] = {'baseline_ratio': round(r_b, 2),
                    'stream_ratio': round(r_s, 2)}
    log(f'  [drift] baseline {r_b:.2f} / stream {r_s:.2f}')

    qd_cmd = [sys.executable, os.path.join(HERE, 'qd_calibrate.py'),
              '--index', idir, '--base', BASE,
              '--queries', QUERIES, '--gt', GT[scale_m],
              '--n-q', '40', '--span', '6']
    out = subprocess.run(qd_cmd, capture_output=True, text=True)
    print(out.stdout, flush=True)
    rec = json.load(open(os.path.join(idir, 'recommended_qd.json')))
    qd = rec['recommended_qd']
    res['qd'] = rec['sweep']
    res['recommended_qd'] = qd

    Q = np.load(QUERIES)
    gt = np.load(GT[scale_m])[:, :10]
    b = bench_qd(idir, read_meta(idir), Q, gt, BASE, 1024, 300_000, 6000,
                 60, qd)
    res['bench'] = {k: round(v, 3) for k, v in b.items()}
    log(f'  [{tag}] recall {b["recall"]:.3f}  p50 {b["p50_ms"]:.0f} ms  '
        f'RSS {b["peak_mb"]:.0f} MB  (qd={qd})')
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--until', type=int, default=50, choices=(10, 20, 50))
    args = ap.parse_args()
    os.makedirs(ROOT, exist_ok=True)
    j = jload()

    legs = [(1, 10), (10, 20), (20, 50)]
    # S0
    if 'bulk_1m' not in j:
        log('=== S0 : bulk 1M médianes ===')
        d, md = LADDER[1]
        idir = os.path.join(ROOT, 'idx_live')
        build_fresh(idir, 1_000_000, d, md)
        j['bulk_1m'] = {'idir': idir, 'depth': d}
        jsave(j)

    for frm, to in legs:
        if to > args.until:
            break
        leg_key = f'leg_{frm}m_{to}m'
        cp_key = f'cp_{to}m'
        idir = os.path.join(ROOT, 'idx_live')
        if leg_key not in j:
            log(f'=== S : injection live {frm}M → {to}M ===')
            j[leg_key] = inject_leg(idir, frm * 1_000_000, to * 1_000_000)
            jsave(j)
        if cp_key not in j:
            log(f'=== CP{to}M : index vieilli ===')
            aged = checkpoint(idir, to, f'aged@{to}M')
            log(f'=== CP{to}M : recalibration + rebuild frais ===')
            d, md = LADDER[to]
            fresh_dir = os.path.join(ROOT, f'idx_fresh_{to}m')
            build_fresh(fresh_dir, to * 1_000_000, d, md)
            fresh = checkpoint(fresh_dir, to, f'fresh@{to}M')
            j[cp_key] = {'aged': aged, 'fresh': fresh}
            jsave(j)
            # l'index frais devient la base du leg suivant (swap prod)
            log(f'  swap : idx_live ← idx_fresh_{to}m')
            shutil.rmtree(idir, ignore_errors=True)
            os.rename(fresh_dir, idir)
            jsave(j)

    log('=== CAMPAGNE TERMINÉE ===')
    log(json.dumps({k: v for k, v in j.items() if k.startswith('cp')},
                   indent=2)[:2000])
    return 0


if __name__ == '__main__':
    sys.exit(main())
