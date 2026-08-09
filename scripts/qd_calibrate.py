"""qd_calibrate — mesure le query_depth idéal d'un index.

Le qd contrôle la profondeur de descente des queries : plus qd est COURT,
plus les plages lues sont larges → pool riche → recall en hausse mais
latence/RAM en hausse. Ce script balaye qd de `depth` à `depth - span`,
mesure recall@10 / latence / RSS à config fixe, et retient le PLUS GRAND
qd qui atteint l'objectif `--target` (descente la plus fine qui tient le
recall → latence minimale).

Bench complet du pipeline prod : query_pathrank → rerank L2 contre la
base → recall@10 vs GT. (Le tri par votes seul ne suffit pas : le pool
top-N contient les voisins mais seul le rerank les fait remonter.)
Latence mesurée = pathrank + rerank, bout en bout.

NB : il existe aussi mg_auto_qd_v2 (auto-sélection online par ratio de
pool, sans GT) — ce script est la mesure OFFLINE de référence avec GT.

Usage :
  python3 qd_calibrate.py --index ~/deep10m/idx_med --base ~/deep10m/base.fbin \
      --queries ~/deep1m/queries.npy --gt ~/deep10m/gt_top10.npy
Écrit <index>/recommended_qd.json et affiche le tableau.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mangrove_ffi as mf                                  # noqa: E402
from mangrove_ffi import Forest, set_gen_version           # noqa: E402
from mangrove_calibrate import read_meta                   # noqa: E402


def bench_qd(idir, meta, Q, GT, base, tp, mlb, top_n, n_q, qd, n_probes=3):
    """recall@10 + latence bout-en-bout (pathrank + rerank L2) à qd fixé."""
    set_gen_version(int(meta.get('gen_version', 3)))
    mf.set_max_leaf_bytes(mlb)
    f = Forest(idir, n_trees=int(meta['n_trees']), dim=int(meta['dim']),
               sub_dim=int(meta.get('sub_dim', 0)), depth=int(meta['depth']),
               n_docs=int(meta['n_docs']),
               gen_version=int(meta.get('gen_version', 3)))
    _ = f.query_pathrank(Q[0], n_probes, tp, top_n, query_depth=qd)  # warmup
    lats, recalls = [], []
    for qi in range(min(n_q, len(Q))):
        ref = set(int(x) for x in GT[qi])
        t0 = time.time()
        ids, votes, n = f.query_pathrank(Q[qi], n_probes, tp, top_n,
                                         query_depth=qd)
        top10 = f.rerank_l2(base, Q[qi], ids[:n], 10) if n else []
        lats.append((time.time() - t0) * 1000)
        recalls.append(sum(1 for x in top10 if int(x) in ref) / 10.0)
    rss = 0
    with open('/proc/self/status') as fh:
        for line in fh:
            if line.startswith('VmRSS:'):
                rss = int(line.split()[1]) // 1024
    f.close()
    return {'recall': float(np.mean(recalls)),
            'mean_ms': float(np.mean(lats)),
            'p50_ms': float(np.percentile(lats, 50)),
            'peak_mb': float(rss)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', required=True)
    ap.add_argument('--base', required=True,
                    help='base .fbin/.fvecs pour le rerank L2')
    ap.add_argument('--queries', required=True)
    ap.add_argument('--gt', required=True)
    ap.add_argument('--tp', type=int, default=1024)
    ap.add_argument('--mlb', type=int, default=300_000)
    ap.add_argument('--top-n', type=int, default=6000)
    ap.add_argument('--n-q', type=int, default=50)
    ap.add_argument('--span', type=int, default=8,
                    help='balaye qd de depth a depth-span')
    ap.add_argument('--target', type=float, default=0.95,
                    help='objectif de recall@10')
    args = ap.parse_args()

    idir = os.path.expanduser(args.index)
    meta = read_meta(idir)
    depth = int(meta['depth'])
    Q = np.load(os.path.expanduser(args.queries))
    GT = np.load(os.path.expanduser(args.gt))[:, :10]

    base = os.path.expanduser(args.base)
    rows = []
    for qd in range(depth, depth - args.span - 1, -1):
        r = bench_qd(idir, meta, Q, GT, base, args.tp, args.mlb, args.top_n,
                     args.n_q, qd)
        r['qd'] = qd
        rows.append(r)
        print(f'  qd={qd:2d} → recall {r["recall"]:.3f}  '
              f'p50 {r["p50_ms"]:6.1f} ms  mean {r["mean_ms"]:6.1f} ms  '
              f'rss {r["peak_mb"]:5.0f} MB', flush=True)

    # Le recall MONTE quand qd descend (pool plus riche) mais la latence
    # explose. L'idéal = le plus GRAND qd (descente la plus profonde,
    # leaves les plus fines) qui atteint l'objectif de recall.
    ok = [r for r in rows if r['recall'] >= args.target]
    if ok:
        pick = max(ok, key=lambda r: r['qd'])
    else:  # objectif hors d'atteinte : moins pire = meilleur recall
        pick = max(rows, key=lambda r: (r['recall'], r['qd']))
    print(f'\nqd idéal : {pick["qd"]} '
          f'(recall {pick["recall"]:.3f}, objectif {args.target}, '
          f'p50 {pick["p50_ms"]:.1f} ms)')

    out = {
        'index': os.path.basename(idir.rstrip('/')),
        'depth': depth,
        'config': {'tp': args.tp, 'mlb': args.mlb, 'top_n': args.top_n,
                   'n_q': args.n_q},
        'sweep': [{'qd': r['qd'], 'recall': round(r['recall'], 4),
                   'p50_ms': round(r['p50_ms'], 1),
                   'mean_ms': round(r['mean_ms'], 1)} for r in rows],
        'recommended_qd': pick['qd'],
    }
    tmp = os.path.join(idir, 'recommended_qd.json.tmp')
    dst = os.path.join(idir, 'recommended_qd.json')
    with open(tmp, 'w') as fh:
        json.dump(out, fh, indent=2)
    os.replace(tmp, dst)
    print(f'écrit : {dst}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
