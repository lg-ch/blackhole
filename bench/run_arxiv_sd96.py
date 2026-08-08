"""arxiv 2M — sub_dim=96 variant.

Tests the hypothesis : keep the ratio sub_dim/dim ≈ 16/128 (= 12.5 %)
that works on SIFT and DEEP. For arxiv dim 768 → sub_dim 96.
Compare against the baseline (sub_dim=16) at the same operating points.

Build:
  ./rpforest build datasets/arxiv/arxiv_base.fvecs \
      /mnt/mangrove/indexes/arxiv_sd96 256 20 \
      --sub_dim 96 --gen v3 --dim 768
"""
import os, sys, time, json, subprocess, resource

os.environ['OMP_NUM_THREADS'] = '1'
os.sched_setaffinity(0, {0})

import numpy as np
sys.path.insert(0, '/home/chatelet/mangrove-search/scripts')
import mangrove_ffi as mf
from mangrove_ffi import Forest

DATA  = '/home/chatelet/mangrove-search/datasets/arxiv'
IDX   = '/mnt/mangrove/indexes/arxiv_sd96'
BASE  = f'{DATA}/arxiv_base.fvecs'
TQ1   = f'{DATA}/arxiv.tq1'
QP    = f'{DATA}/bench_q.fvecs'
GP    = f'{DATA}/bench_gt.ivecs'
DIM, N_TREES, DEPTH, SUB, N_DOCS = 768, 256, 20, 96, 2058751


def drop_caches():
    os.sync(); open('/proc/sys/vm/drop_caches', 'w').write('3')

def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

def disk_size_mb(path):
    if not os.path.exists(path): return 0.0
    out = subprocess.run(['du', '-sb', path], capture_output=True, text=True)
    return int(out.stdout.split()[0]) / (1024.0 * 1024.0)


def time_single(f, Q, GT, np_, qd, tn, kp,
                n_warmup=5, n_warm=100):
    drop_caches()
    for i in range(n_warmup):
        q = Q[i % len(Q)]
        ids, _, n = f.query_probes(q, np_, top_n=tn, probe_depth=qd)
        cand = np.asarray(ids[:n], dtype=np.int32); cand = cand[cand >= 0]
        f.rerank_tq1(TQ1, BASE, q, cand, kprime=kp, top_k=10)
    recs, lats = [], []
    for qi in range(n_warmup, n_warmup + n_warm):
        q = Q[qi % len(Q)]
        ref = set(int(x) for x in GT[qi % len(GT)])
        t0 = time.time()
        ids, _, n = f.query_probes(q, np_, top_n=tn, probe_depth=qd)
        cand = np.asarray(ids[:n], dtype=np.int32); cand = cand[cand >= 0]
        top = f.rerank_tq1(TQ1, BASE, q, cand, kprime=kp, top_k=10)
        lats.append((time.time() - t0) * 1000.0)
        recs.append(sum(1 for x in top if int(x) in ref) / 10.0)
    return float(np.mean(recs)), float(np.percentile(lats, 50)), \
           float(np.percentile(lats, 95))


def main():
    qraw = np.fromfile(QP, dtype=np.int32)
    nq = qraw.size // (1 + DIM)
    Q = qraw.reshape(nq, 1 + DIM)[:, 1:].view(np.float32).copy()
    graw = np.fromfile(GP, dtype=np.int32)
    gk = graw.size // nq - 1
    GT = graw.reshape(nq, 1 + gk)[:, 1:11].copy()

    idx_mb = disk_size_mb(IDX)
    tq1_mb = disk_size_mb(TQ1)

    print('==========================================================')
    print(f' arxiv 2M — sub_dim={SUB} (ratio sub_dim/dim = {SUB/DIM:.1%})')
    print('==========================================================')
    print(f'  Vectors      : float32, dim {DIM}, n_docs {N_DOCS:,}')
    print(f'  Source       : Cohere v3 arxiv abstracts')
    print(f'  Hypothesis   : sub_dim/dim = 96/768 = 12.5 % matches '
          f'SIFT 16/128 = 12.5 %')
    print(f'  Technique    : RP-forest ({N_TREES} trees, depth {DEPTH}, '
          f'sub_dim {SUB}, gen v3) + TQ1 sidecar + exact L2 rerank')
    print(f'  Disk         : index {idx_mb:.1f} MB  +  tq1 {tq1_mb:.1f} MB '
          f'= {idx_mb + tq1_mb:.1f} MB')
    print(f'  Baseline     : sub_dim=16 → recall 0.998 / p50 229 ms / '
          f'index 2552 MB')
    print(f'  CPU policy   : pinned to core 0, OMP_NUM_THREADS=1')
    print()

    mf.set_shared_scratch_pool(True)
    f = Forest(IDX, n_trees=N_TREES, dim=DIM, sub_dim=SUB, depth=DEPTH,
               n_docs=N_DOCS, gen_version=3)

    SWEEP = []
    for np_ in (5, 10, 20):
        for qd in (18, 16, 14):
            for tn in (16_000, 32_000, 64_000):
                SWEEP.append((np_, qd, tn))

    print(f'  Sweep matrix : NP × QD × top_n  (TQ1 K\' = max(500, top_n/16))')
    print(f'  Single CPU, 100 warm queries per row\n')
    print(f'  {"NP":>3} {"QD":>3} {"top_n":>7} {"K\'":>6} | '
          f'{"recall":>7} | {"p50w":>7} {"p95w":>7}')
    print('  ' + '-' * 60)

    results = []
    for np_, qd, tn in SWEEP:
        kp = max(500, tn // 16)
        rec, p50, p95 = time_single(f, Q, GT, np_, qd, tn, kp)
        results.append({
            'n_probes': np_, 'probe_depth': qd, 'top_n': tn, 'kprime': kp,
            'recall': rec, 'p50_warm_ms': p50, 'p95_warm_ms': p95,
        })
        mark = ' *' if rec >= 0.999 else ('  ' if rec < 0.99 else ' .')
        print(f'  {np_:>3} {qd:>3} {tn:>7} {kp:>6} | '
              f'{rec:>7.4f} | {p50:>5.1f}ms {p95:>5.1f}ms{mark}', flush=True)

    best = sorted(results, key=lambda r: (-r['recall'], r['p50_warm_ms']))[0]
    print()
    print(f'  Sweet spot   : recall {best["recall"]:.4f} / '
          f'p50 warm {best["p50_warm_ms"]:.1f} ms')
    print(f'                 (NP={best["n_probes"]} QD={best["probe_depth"]} '
          f'top_n={best["top_n"]} K\'={best["kprime"]})')
    print(f'  Query peak RSS : {peak_rss_mb():.1f} MB')
    print('==========================================================')

    os.makedirs('bench/results', exist_ok=True)
    with open('bench/results/arxiv_sd96.json', 'w') as fh:
        json.dump({
            'dataset': 'arxiv_sd96',
            'dim': DIM, 'sub_dim': SUB, 'ratio': SUB / DIM,
            'n_docs': N_DOCS, 'n_trees': N_TREES, 'depth': DEPTH,
            'gen_version': 3,
            'disk_index_mb': idx_mb,
            'sweep': results,
            'sweet_spot': best,
            'query_peak_rss_mb': peak_rss_mb(),
        }, fh, indent=2)
    print('  → bench/results/arxiv_sd96.json')


if __name__ == '__main__':
    main()
