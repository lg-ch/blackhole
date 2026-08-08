"""GIST 1M (dim 960) — full benchmark, same protocol as run_arxiv_2m.py.

Build:
  ./rpforest build /mnt/mangrove/datasets/gist/gist/gist_base.fvecs \
      /mnt/mangrove/indexes/gist_1m 256 18 \
      --sub_dim 16 --gen v3 --dim 960
TQ1 sidecar:
  ./rpforest tquant1 /mnt/mangrove/datasets/gist/gist/gist_base.fvecs \
      /mnt/mangrove/datasets/gist/gist.tq1 --dim 960

Run:  python3 bench/run_gist_1m.py
"""

import os, sys, time, json, struct, subprocess, resource, multiprocessing as mp

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.sched_setaffinity(0, {0})

import numpy as np
sys.path.insert(0, '/home/chatelet/mangrove-search/scripts')
import mangrove_ffi as mf
from mangrove_ffi import Forest

DATA = '/mnt/mangrove/datasets/gist/gist'
IDX  = '/mnt/mangrove/indexes/gist_1m'
BASE = f'{DATA}/gist_base.fvecs'
TQ1  = '/mnt/mangrove/datasets/gist/gist.tq1'
QP   = f'{DATA}/gist_query.fvecs'
GP   = f'{DATA}/gist_groundtruth.ivecs'
DIM, N_TREES, DEPTH, N_DOCS = 960, 256, 18, 1_000_000

BUILD_TIME_S = 117.5          # observed on the 2026-06-18 build
BUILD_PEAK_RSS_MB = 20.7      # idem


def drop_caches():
    os.sync()
    open('/proc/sys/vm/drop_caches', 'w').write('3')


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def disk_size_mb(path):
    if not os.path.exists(path): return 0.0
    out = subprocess.run(['du', '-sb', path], capture_output=True, text=True)
    return int(out.stdout.split()[0]) / (1024.0 * 1024.0)


def cpu_model():
    try:
        out = subprocess.run(['lscpu'], env={**os.environ, 'LANG': 'C'},
                             capture_output=True, text=True).stdout
        for line in out.splitlines():
            if line.lower().startswith('model name'):
                return line.split(':', 1)[1].strip()
    except Exception:
        pass
    return 'unknown'


def time_single_cpu(forest, queries, gt, np_, qd, tn, kp, use_tq1,
                    n_warmup=5, n_warm=100):
    drop_caches()
    for i in range(n_warmup):
        q = queries[i % len(queries)]
        ids, _, n = forest.query_probes(q, np_, top_n=tn, probe_depth=qd)
        cand = np.asarray(ids[:n], dtype=np.int32); cand = cand[cand >= 0]
        if use_tq1: forest.rerank_tq1(TQ1, BASE, q, cand, kprime=kp, top_k=10)
        else:       forest.rerank_l2 (BASE,        q, cand, top_k=10)

    recs, lats = [], []
    for qi in range(n_warmup, n_warmup + n_warm):
        q = queries[qi % len(queries)]; ref = set(int(x) for x in gt[qi % len(gt)])
        t0 = time.time()
        ids, _, n = forest.query_probes(q, np_, top_n=tn, probe_depth=qd)
        cand = np.asarray(ids[:n], dtype=np.int32); cand = cand[cand >= 0]
        if use_tq1: top = forest.rerank_tq1(TQ1, BASE, q, cand, kprime=kp, top_k=10)
        else:       top = forest.rerank_l2 (BASE,        q, cand, top_k=10)
        lats.append((time.time() - t0) * 1000.0)
        recs.append(sum(1 for x in top if int(x) in ref) / 10.0)

    drop_caches()
    qcold = queries[n_warmup % len(queries)]
    t0 = time.time()
    ids, _, n = forest.query_probes(qcold, np_, top_n=tn, probe_depth=qd)
    cand = np.asarray(ids[:n], dtype=np.int32); cand = cand[cand >= 0]
    if use_tq1: forest.rerank_tq1(TQ1, BASE, qcold, cand, kprime=kp, top_k=10)
    else:       forest.rerank_l2 (BASE,        qcold, cand, top_k=10)
    cold = (time.time() - t0) * 1000.0

    return float(np.mean(recs)), float(np.percentile(lats, 50)), \
           float(np.percentile(lats, 95)), float(cold)


def _worker(args):
    core, n_q, np_, qd, tn, kp, use_tq1, qbytes = args
    os.environ['OMP_NUM_THREADS'] = '1'
    os.sched_setaffinity(0, {core})
    import numpy as _np
    from mangrove_ffi import Forest as _Forest
    import mangrove_ffi as _mf
    _mf.set_shared_scratch_pool(True)
    _mf.set_tree_sub(0); _mf.set_tree_sub_groups(0); _mf.set_node_perm(0)
    f = _Forest(IDX, n_trees=N_TREES, dim=DIM, sub_dim=16, depth=DEPTH,
                n_docs=N_DOCS, gen_version=3)
    qs = _np.frombuffer(qbytes, dtype=_np.float32).reshape(-1, DIM)
    for i in range(5):
        q = qs[i % len(qs)]
        ids, _, n = f.query_probes(q, np_, top_n=tn, probe_depth=qd)
        cand = _np.asarray(ids[:n], dtype=_np.int32); cand = cand[cand >= 0]
        if use_tq1: f.rerank_tq1(TQ1, BASE, q, cand, kprime=kp, top_k=10)
        else:       f.rerank_l2 (BASE,        q, cand, top_k=10)
    t0 = time.time()
    for i in range(n_q):
        q = qs[i % len(qs)]
        ids, _, n = f.query_probes(q, np_, top_n=tn, probe_depth=qd)
        cand = _np.asarray(ids[:n], dtype=_np.int32); cand = cand[cand >= 0]
        if use_tq1: f.rerank_tq1(TQ1, BASE, q, cand, kprime=kp, top_k=10)
        else:       f.rerank_l2 (BASE,        q, cand, top_k=10)
    dt = time.time() - t0
    f.close()
    return n_q / dt


def main():
    qraw = np.fromfile(QP, dtype=np.int32)
    nq = qraw.size // (1 + DIM)
    Q  = qraw.reshape(nq, 1 + DIM)[:, 1:].view(np.float32).copy()[:1000]
    graw = np.fromfile(GP, dtype=np.int32)
    gk = graw.size // nq - 1
    GT = graw.reshape(nq, 1 + gk)[:, 1:11].copy()[:1000]

    cpus_allowed = sorted(os.sched_getaffinity(0))
    cpu = cpu_model()
    idx_mb = disk_size_mb(IDX)
    tq1_mb = disk_size_mb(TQ1)

    print('==========================================================')
    print(' GIST 1M — mangrove-search benchmark')
    print('==========================================================')
    print(f'  Vectors        : float32, dim {DIM}, n_docs {N_DOCS:,}')
    print(f'  Source         : corpus-texmex.irisa.fr — GIST descriptor')
    print(f'  Queries / GT   : {min(1000, nq)} held-out / GT@10 by exhaustive L2')
    print(f'  Technique      : RP-forest ({N_TREES} trees, depth {DEPTH}, '
          f'sub_dim 16, gen v3)')
    print(f'                   + TQ1 1-bit sidecar  (128 B/code)')
    print(f'                   + exact L2 rerank on K\' survivors')
    if BUILD_TIME_S is not None:
        print(f'  Build time     : {BUILD_TIME_S:.1f} s (24 OMP threads)')
        print(f'  Build peak RSS : {BUILD_PEAK_RSS_MB:.1f} MB')
    print(f'  Disk           : index {idx_mb:.1f} MB  +  tq1 {tq1_mb:.1f} MB '
          f'= {idx_mb + tq1_mb:.1f} MB')
    print(f'  CPU policy     : pinned to core(s) {cpus_allowed}, '
          f'OMP_NUM_THREADS=1')
    print(f'  CPU model      : {cpu}')
    print()

    mf.set_shared_scratch_pool(True)
    mf.set_tree_sub(0); mf.set_tree_sub_groups(0); mf.set_node_perm(0)
    f = Forest(IDX, n_trees=N_TREES, dim=DIM, sub_dim=16, depth=DEPTH,
               n_docs=N_DOCS, gen_version=3)

    SWEEP = []
    for np_ in (5, 10, 20):
        for qd in (DEPTH - 2, DEPTH - 4, DEPTH - 6):  # 16, 14, 12
            for tn in (16_000, 32_000, 64_000):
                kp = max(500, tn // 16)
                SWEEP.append((np_, qd, tn, kp, True))

    print(f'  Sweep matrix : NP × QD × top_n  (TQ1 K\' = max(500, top_n/16))')
    print(f'  Single CPU, 100 warm queries + 1 cold per row\n')
    print(f'  {"NP":>3} {"QD":>3} {"top_n":>7} {"K\'":>6} | '
          f'{"recall":>7} | {"p50w":>7} {"p95w":>7} {"cold":>7}')
    print('  ' + '-' * 67)

    results = []
    for np_, qd, tn, kp, use_tq1 in SWEEP:
        rec, p50, p95, cold = time_single_cpu(f, Q, GT, np_, qd, tn, kp, use_tq1)
        results.append({
            'n_probes': np_, 'probe_depth': qd, 'top_n': tn,
            'kprime': kp, 'use_tq1': use_tq1,
            'recall': rec, 'p50_warm_ms': p50, 'p95_warm_ms': p95,
            'cold_q0_ms': cold,
        })
        mark = ' *' if rec >= 0.999 else ('  ' if rec < 0.99 else ' .')
        print(f'  {np_:>3} {qd:>3} {tn:>7} {kp:>6} | '
              f'{rec:>7.4f} | {p50:>5.1f}ms {p95:>5.1f}ms {cold:>5.1f}ms{mark}',
              flush=True)

    best = sorted(results, key=lambda r: (-r['recall'], r['p50_warm_ms']))[0]
    print()
    print(f'  Sweet spot     : recall {best["recall"]:.4f}, '
          f'p50 warm {best["p50_warm_ms"]:.1f} ms')
    print(f'                   (NP={best["n_probes"]} QD={best["probe_depth"]} '
          f'top_n={best["top_n"]} K\'={best["kprime"]})')
    print(f'  Query peak RSS : {peak_rss_mb():.1f} MB')
    print()

    print('  Parallel scaling at sweet spot (N worker processes,')
    print('  each pinned to its own core, each running 200 queries):')
    print()
    print(f'  {"workers":>7} {"qps total":>10} {"speedup":>8}')
    print('  ' + '-' * 30)

    n_total = 200
    qbytes = Q.astype(np.float32).tobytes()
    base_qps = None
    parallel = []
    for n_workers in (1, 2, 4, 8):
        cores = list(range(n_workers))
        args = [(c, n_total, best['n_probes'], best['probe_depth'],
                 best['top_n'], best['kprime'], True, qbytes) for c in cores]
        t0 = time.time()
        with mp.Pool(n_workers) as pool:
            worker_qps = pool.map(_worker, args)
        wall = time.time() - t0
        total_q = n_total * n_workers
        agg = total_q / wall
        if base_qps is None: base_qps = agg
        parallel.append({'workers': n_workers, 'qps_total': agg,
                         'speedup': agg / base_qps, 'wall_s': wall})
        print(f'  {n_workers:>7} {agg:>10.1f} {agg/base_qps:>7.2f}×', flush=True)

    print()
    print(f'  Query peak RSS (after parallel) : {peak_rss_mb():.1f} MB')
    print('==========================================================')

    os.makedirs('bench/results', exist_ok=True)
    with open('bench/results/gist_1m.json', 'w') as fh:
        json.dump({
            'dataset': 'gist_1m',
            'dim': DIM, 'n_docs': N_DOCS, 'n_trees': N_TREES, 'depth': DEPTH,
            'sub_dim': 16, 'gen_version': 3, 'tree_sub': 0,
            'build_time_s': BUILD_TIME_S,
            'build_peak_rss_mb': BUILD_PEAK_RSS_MB,
            'disk_index_mb': idx_mb,
            'disk_tq1_mb': tq1_mb,
            'cpu_affinity': cpus_allowed,
            'cpu_model': cpu,
            'sweep': results,
            'sweet_spot': best,
            'parallel': parallel,
            'query_peak_rss_mb': peak_rss_mb(),
        }, fh, indent=2)
    print('  → bench/results/gist_1m.json')


if __name__ == '__main__':
    main()
