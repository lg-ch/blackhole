"""GIST 1M variant: 1000 trees, depth 22, NP=1 (single probe).

Same TQ1 sidecar / base / queries / GT as the 256-tree run.
Build:
  ./rpforest build /mnt/mangrove/datasets/gist/gist/gist_base.fvecs \
      /mnt/mangrove/indexes/gist_1m_1000t_d22 1000 22 \
      --sub_dim 16 --gen v3 --dim 960
"""
import os, sys, time, json, struct, subprocess, resource, multiprocessing as mp

os.environ['OMP_NUM_THREADS'] = '1'
os.sched_setaffinity(0, {0})

import numpy as np
sys.path.insert(0, '/home/chatelet/mangrove-search/scripts')
import mangrove_ffi as mf
from mangrove_ffi import Forest

DATA = '/mnt/mangrove/datasets/gist/gist'
IDX  = '/mnt/mangrove/indexes/gist_1m_1000t_d22'
BASE = f'{DATA}/gist_base.fvecs'
TQ1  = '/mnt/mangrove/datasets/gist/gist.tq1'
QP   = f'{DATA}/gist_query.fvecs'
GP   = f'{DATA}/gist_groundtruth.ivecs'
DIM, N_TREES, DEPTH, N_DOCS = 960, 1000, 22, 1_000_000
NP = 1


def drop_caches():
    os.sync(); open('/proc/sys/vm/drop_caches', 'w').write('3')


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def disk_size_mb(path):
    if not os.path.exists(path): return 0.0
    out = subprocess.run(['du', '-sb', path], capture_output=True, text=True)
    return int(out.stdout.split()[0]) / (1024.0 * 1024.0)


def time_single(forest, queries, gt, qd, tn, kp):
    drop_caches()
    for i in range(5):
        q = queries[i % len(queries)]
        ids, _, n = forest.query_probes(q, NP, top_n=tn, probe_depth=qd)
        cand = np.asarray(ids[:n], dtype=np.int32); cand = cand[cand >= 0]
        forest.rerank_tq1(TQ1, BASE, q, cand, kprime=kp, top_k=10)
    recs, lats = [], []
    for qi in range(5, 5 + 100):
        q = queries[qi % len(queries)]
        ref = set(int(x) for x in gt[qi % len(gt)])
        t0 = time.time()
        ids, _, n = forest.query_probes(q, NP, top_n=tn, probe_depth=qd)
        cand = np.asarray(ids[:n], dtype=np.int32); cand = cand[cand >= 0]
        top = forest.rerank_tq1(TQ1, BASE, q, cand, kprime=kp, top_k=10)
        lats.append((time.time() - t0) * 1000.0)
        recs.append(sum(1 for x in top if int(x) in ref) / 10.0)
    return float(np.mean(recs)), float(np.percentile(lats, 50)), \
           float(np.percentile(lats, 95))


def _worker(args):
    core, n_q, qd, tn, kp, qbytes = args
    os.environ['OMP_NUM_THREADS'] = '1'
    os.sched_setaffinity(0, {core})
    import numpy as _np
    from mangrove_ffi import Forest as _Forest
    import mangrove_ffi as _mf
    _mf.set_shared_scratch_pool(True)
    f = _Forest(IDX, n_trees=N_TREES, dim=DIM, sub_dim=16, depth=DEPTH,
                n_docs=N_DOCS, gen_version=3)
    qs = _np.frombuffer(qbytes, dtype=_np.float32).reshape(-1, DIM)
    for i in range(5):
        q = qs[i % len(qs)]
        ids, _, n = f.query_probes(q, NP, top_n=tn, probe_depth=qd)
        cand = _np.asarray(ids[:n], dtype=_np.int32); cand = cand[cand >= 0]
        f.rerank_tq1(TQ1, BASE, q, cand, kprime=kp, top_k=10)
    t0 = time.time()
    for i in range(n_q):
        q = qs[i % len(qs)]
        ids, _, n = f.query_probes(q, NP, top_n=tn, probe_depth=qd)
        cand = _np.asarray(ids[:n], dtype=_np.int32); cand = cand[cand >= 0]
        f.rerank_tq1(TQ1, BASE, q, cand, kprime=kp, top_k=10)
    dt = time.time() - t0
    f.close()
    return n_q / dt


def main():
    qraw = np.fromfile(QP, dtype=np.int32)
    nq = qraw.size // (1 + DIM)
    Q  = qraw.reshape(nq, 1 + DIM)[:, 1:].view(np.float32).copy()[:200]
    graw = np.fromfile(GP, dtype=np.int32)
    gk = graw.size // nq - 1
    GT = graw.reshape(nq, 1 + gk)[:, 1:11].copy()[:200]

    cpus_allowed = sorted(os.sched_getaffinity(0))
    idx_mb = disk_size_mb(IDX)
    tq1_mb = disk_size_mb(TQ1)

    print('==========================================================')
    print(' GIST 1M VARIANT — 1000 trees, depth 22, NP=1 (single probe)')
    print('==========================================================')
    print(f'  Vectors        : float32, dim {DIM}, n_docs {N_DOCS:,}')
    print(f'  Technique      : RP-forest ({N_TREES} trees, depth {DEPTH}, '
          f'sub_dim 16, gen v3) + TQ1 + exact L2 rerank')
    print(f'  Disk           : index {idx_mb:.1f} MB  +  tq1 {tq1_mb:.1f} MB '
          f'= {idx_mb + tq1_mb:.1f} MB')
    print(f'  CPU policy     : pinned to core(s) {cpus_allowed}, '
          f'OMP_NUM_THREADS=1')
    print(f'  Single probe   : NP=1  (no multi-probe; depth=22 native fused)')
    print()

    mf.set_shared_scratch_pool(True)
    f = Forest(IDX, n_trees=N_TREES, dim=DIM, sub_dim=16, depth=DEPTH,
               n_docs=N_DOCS, gen_version=3)

    print(f'  top_n sweep at QD=fused (k_shift=0), single CPU\n')
    print(f'  {"top_n":>8} {"K\'":>6} | {"recall":>7} {"p50w":>7} {"p95w":>7}')
    print('  ' + '-' * 50)

    sweep = []
    for tn in (16_000, 32_000, 64_000, 128_000, 256_000, 500_000):
        kp = max(500, tn // 16)
        rec, p50, p95 = time_single(f, Q, GT, 0, tn, kp)  # qd_eff=0 = fused
        sweep.append({'top_n': tn, 'kprime': kp, 'recall': rec,
                      'p50_warm_ms': p50, 'p95_warm_ms': p95})
        mark = ' *' if rec >= 0.999 else ('  ' if rec < 0.99 else ' .')
        print(f'  {tn:>8} {kp:>6} | {rec:>7.4f} {p50:>5.1f}ms {p95:>5.1f}ms{mark}',
              flush=True)

    best = sorted(sweep, key=lambda r: (-r['recall'], r['p50_warm_ms']))[0]
    print()
    print(f'  Sweet spot     : recall {best["recall"]:.4f}, '
          f'p50 warm {best["p50_warm_ms"]:.1f} ms')
    print(f'                   (top_n={best["top_n"]} K\'={best["kprime"]})')
    print(f'  Query peak RSS : {peak_rss_mb():.1f} MB')
    print()

    print('  Parallel scaling at sweet spot (N worker processes):')
    print()
    print(f'  {"workers":>7} {"qps total":>10} {"speedup":>8} {"per-q":>9}')
    print('  ' + '-' * 40)

    n_per = 200
    qbytes = Q.astype(np.float32).tobytes()
    base_qps = None
    parallel = []
    for n_workers in (1, 2, 4, 8):
        cores = list(range(n_workers))
        args = [(c, n_per, 0, best['top_n'], best['kprime'], qbytes)
                for c in cores]
        t0 = time.time()
        with mp.Pool(n_workers) as pool:
            _ = pool.map(_worker, args)
        wall = time.time() - t0
        total_q = n_per * n_workers
        agg = total_q / wall
        if base_qps is None: base_qps = agg
        per_q_ms = 1000.0 / agg
        parallel.append({'workers': n_workers, 'qps_total': agg,
                         'speedup': agg / base_qps,
                         'per_query_ms': per_q_ms, 'wall_s': wall})
        print(f'  {n_workers:>7} {agg:>10.1f} {agg/base_qps:>7.2f}× '
              f'{per_q_ms:>7.1f}ms', flush=True)

    print()
    print(f'  Query peak RSS (after parallel) : {peak_rss_mb():.1f} MB')
    print('==========================================================')

    os.makedirs('bench/results', exist_ok=True)
    with open('bench/results/gist_1m_1000t.json', 'w') as fh:
        json.dump({
            'dataset': 'gist_1m_1000t',
            'dim': DIM, 'n_docs': N_DOCS, 'n_trees': N_TREES, 'depth': DEPTH,
            'sub_dim': 16, 'gen_version': 3,
            'n_probes': NP,
            'disk_index_mb': idx_mb,
            'disk_tq1_mb': tq1_mb,
            'cpu_affinity': cpus_allowed,
            'sweep': sweep,
            'sweet_spot': best,
            'parallel': parallel,
            'query_peak_rss_mb': peak_rss_mb(),
        }, fh, indent=2)
    print('  → bench/results/gist_1m_1000t.json')


if __name__ == '__main__':
    main()
