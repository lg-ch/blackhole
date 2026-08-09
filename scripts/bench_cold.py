"""bench_cold — latence/recall/RSS à FROID, protocole de référence.

Réplique fidèle du protocole validé le 2026-08-02 (bench_cold_cgroup) :
  - OMP_NUM_THREADS=1, affinité 8 cœurs
  - sync + drop_caches AVANT CHAQUE requête (cold par requête)
  - latence "pathrank" = descente + collect, HORS rerank (comme les
    mesures de référence) ; latence "e2e" = avec rerank L2, en bonus
  - recall@10 après rerank L2 contre la base

Pour la borne mémoire, lancer sous cgroup :
  systemd-run --scope -p MemoryMax=200M --uid=chatelet \
      python3 bench_cold.py --index ... [args]

Usage :
  python3 bench_cold.py --index ~/deep10m/idx_med --base ~/deep10m/base.fbin \
      --queries ~/deep1m/queries.npy --gt ~/deep10m/gt_top10.npy \
      --np 3 --tp 1024 --qd 18 [--n-q 50] [--json out.json]
"""
import argparse
import json
import os
import subprocess
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
try:
    os.sched_setaffinity(0, set(range(8)))
except OSError:
    pass

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mangrove_ffi as mf                                  # noqa: E402
from mangrove_ffi import Forest, set_gen_version           # noqa: E402
from mangrove_calibrate import read_meta                   # noqa: E402


def drop_caches():
    subprocess.run(['sync'], check=True)
    # root : écriture directe ; sinon sudo (NOPASSWD requis, comme le 02-08)
    try:
        with open('/proc/sys/vm/drop_caches', 'w') as fh:
            fh.write('3')
    except PermissionError:
        subprocess.run(['sudo', 'tee', '/proc/sys/vm/drop_caches'],
                       input=b'3\n', check=False,
                       stdout=subprocess.DEVNULL)


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
    ap.add_argument('--queries', required=True)
    ap.add_argument('--gt', required=True)
    ap.add_argument('--np', type=int, default=3, dest='n_probes')
    ap.add_argument('--tp', type=int, default=1024)
    ap.add_argument('--qd', type=int, default=0,
                    help='0 = recommended_qd.json si présent, sinon depth-2')
    ap.add_argument('--top-n', type=int, default=4000)
    ap.add_argument('--mlb', type=int, default=200_000)
    ap.add_argument('--n-q', type=int, default=50)
    ap.add_argument('--json', default='')
    args = ap.parse_args()

    idir = os.path.expanduser(args.index)
    base = os.path.expanduser(args.base)
    meta = read_meta(idir)
    depth = int(meta['depth'])
    qd = args.qd
    if qd <= 0:
        rec = os.path.join(idir, 'recommended_qd.json')
        qd = (json.load(open(rec))['recommended_qd']
              if os.path.exists(rec) else depth - 2)

    Q = np.load(os.path.expanduser(args.queries))
    GT = np.load(os.path.expanduser(args.gt))[:, :10]
    set_gen_version(int(meta.get('gen_version', 3)))
    mf.set_max_leaf_bytes(args.mlb)
    f = Forest(idir, n_trees=int(meta['n_trees']), dim=int(meta['dim']),
               sub_dim=int(meta.get('sub_dim', 0)), depth=depth,
               n_docs=int(meta['n_docs']),
               gen_version=int(meta.get('gen_version', 3)))

    drop_caches()
    _ = f.query_pathrank(Q[0], args.n_probes, args.tp, args.top_n,
                         query_depth=qd)          # warmup structurel
    lat_pr, lat_e2e, recalls = [], [], []
    peak = rss_mb()
    for qi in range(min(args.n_q, len(Q))):
        drop_caches()
        t0 = time.time()
        ids, votes, n = f.query_pathrank(Q[qi], args.n_probes, args.tp,
                                         args.top_n, query_depth=qd)
        t1 = time.time()
        lat_pr.append((t1 - t0) * 1000)
        top10 = f.rerank_l2(base, Q[qi], ids[:n], top_k=10) if n else []
        lat_e2e.append((time.time() - t0) * 1000)
        peak = max(peak, rss_mb())
        ref = set(int(x) for x in GT[qi])
        recalls.append(sum(1 for x in top10 if int(x) in ref) / 10.0)
    f.close()

    lp = np.array(lat_pr)
    le = np.array(lat_e2e)
    out = {
        'index': idir, 'np': args.n_probes, 'tp': args.tp, 'qd': qd,
        'n_q': len(lp), 'recall_at_10': round(float(np.mean(recalls)), 4),
        'pathrank_ms': {'p50': round(float(np.median(lp)), 1),
                        'p95': round(float(np.percentile(lp, 95)), 1),
                        'p99': round(float(np.percentile(lp, 99)), 1)},
        'e2e_ms':      {'p50': round(float(np.median(le)), 1),
                        'p99': round(float(np.percentile(le, 99)), 1)},
        'peak_rss_mb': round(peak),
    }
    print(f'COLD np={args.n_probes} tp={args.tp} qd={qd} : '
          f'recall {out["recall_at_10"]:.3f}  '
          f'pathrank p50 {out["pathrank_ms"]["p50"]} / '
          f'p99 {out["pathrank_ms"]["p99"]} ms  '
          f'e2e p50 {out["e2e_ms"]["p50"]} ms  '
          f'RSS {out["peak_rss_mb"]} MB')
    if args.json:
        with open(args.json, 'w') as fh:
            json.dump(out, fh, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
