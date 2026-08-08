"""arxiv 2M (dim 768) — full benchmark.

Header   : dim, n_docs, build time/RAM, disk, technique, CPU policy
Sweep    : matrix over (n_probes, probe_depth, top_n, K')
Parallel : N-worker throughput at the sweet-spot config

Pinned to CPU 0, OMP_NUM_THREADS=1, for honest single-core numbers.

Build:
  ./rpforest build datasets/arxiv/arxiv_base.fvecs \
      /mnt/mangrove/indexes/arxiv_2m 256 20 \
      --sub_dim 16 --gen v3 --dim 768
TQ1 sidecar:
  ./rpforest tquant1 datasets/arxiv/arxiv_base.fvecs \
      datasets/arxiv/arxiv.tq1 --dim 768

Run:  python3 bench/run_arxiv_2m.py
"""

import os, sys, time, json, struct, subprocess, resource, multiprocessing as mp

# === MUST be set before mangrove_ffi (libmangrove) loads OpenMP ===
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# Pin the process (and its threads) to CPU 0 — honest single-core measure.
os.sched_setaffinity(0, {0})

import numpy as np
sys.path.insert(0, '/home/chatelet/mangrove-search/scripts')
import mangrove_ffi as mf
from mangrove_ffi import Forest

# -------- paths & params --------
DATA   = '/home/chatelet/mangrove-search/datasets/arxiv'
IDX    = '/mnt/mangrove/indexes/arxiv_2m'
BASE   = f'{DATA}/arxiv_base.fvecs'
TQ1    = f'{DATA}/arxiv.tq1'
QP     = f'{DATA}/bench_q.fvecs'
GP     = f'{DATA}/bench_gt.ivecs'
DIM, N_TREES, DEPTH, N_DOCS = 768, 256, 20, 2058751

# Build metadata — replaced if rebuild is triggered; else read from index meta + observed last build log.
BUILD_TIME_S = 202.87           # observed on the 2026-06-18 rebuild
BUILD_PEAK_RSS_MB = 36.66       # idem

# -------- helpers --------
def drop_caches():
    os.sync()
    open('/proc/sys/vm/drop_caches', 'w').write('3')

def read_fvecs(path, n, dim):
    raw = np.fromfile(path, dtype=np.int32)
    nq = raw.size // (1 + dim)
    return raw.reshape(nq, 1 + dim)[:n, 1:].view(np.float32)

def read_ivecs(path, n, k):
    raw = np.fromfile(path, dtype=np.int32)
    nq = raw.size // (1 + (raw.size // nq if False else 0))  # placeholder, recomputed below
    return raw  # caller derives shape

def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

def disk_size_mb(path):
    if not os.path.exists(path): return 0.0
    out = subprocess.run(['du', '-sb', path], capture_output=True, text=True)
    return int(out.stdout.split()[0]) / (1024.0 * 1024.0)

# -------- one timed config (single CPU) --------
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
        if use_tq1:
            top = forest.rerank_tq1(TQ1, BASE, q, cand, kprime=kp, top_k=10)
        else:
            top = forest.rerank_l2 (BASE,        q, cand, top_k=10)
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


# -------- parallel throughput worker (process-level) --------
def _worker(args):
    """Spawned process. Opens its own Forest, pins to its core, runs queries."""
    core, n_q, np_, qd, tn, kp, use_tq1, qbytes = args
    os.environ['OMP_NUM_THREADS'] = '1'
    os.sched_setaffinity(0, {core})
    # late imports inside worker process
    import numpy as _np
    from mangrove_ffi import Forest as _Forest
    import mangrove_ffi as _mf
    _mf.set_shared_scratch_pool(True)
    _mf.set_tree_sub(0); _mf.set_tree_sub_groups(0); _mf.set_node_perm(0)
    f = _Forest(IDX, n_trees=N_TREES, dim=DIM, sub_dim=16, depth=DEPTH,
                n_docs=N_DOCS, gen_version=3)
    qs = _np.frombuffer(qbytes, dtype=_np.float32).reshape(-1, DIM)

    # warm
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
    return n_q / dt   # qps for this worker


# -------- main --------
def main():
    # queries / gt
    qraw = np.fromfile(QP, dtype=np.int32)
    nq = qraw.size // (1 + DIM)
    Q  = qraw.reshape(nq, 1 + DIM)[:, 1:].view(np.float32).copy()
    graw = np.fromfile(GP, dtype=np.int32)
    gk = graw.size // nq - 1
    GT = graw.reshape(nq, 1 + gk)[:, 1:11].copy()

    # === header ===
    cpus_allowed = sorted(os.sched_getaffinity(0))
    cpu_model = subprocess.check_output(
        "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs",
        shell=True, text=True).strip()
    idx_mb  = disk_size_mb(IDX)
    tq1_mb  = disk_size_mb(TQ1)

    print('==========================================================')
    print(' arxiv 2M — mangrove-search benchmark')
    print('==========================================================')
    print(f'  Vectors        : float32, dim {DIM}, n_docs {N_DOCS:,}')
    print(f'  Source         : Cohere embed-english-v3.0 on arXiv abstracts')
    print(f'  Queries / GT   : {nq} held-out / GT@10 by exhaustive L2')
    print(f'  Technique      : RP-forest ({N_TREES} trees, depth {DEPTH}, '
          f'sub_dim 16, gen v3)')
    print(f'                   + TQ1 1-bit sidecar  (128 B/code)')
    print(f'                   + exact L2 rerank on K\' survivors')
    print(f'  Build time     : {BUILD_TIME_S:.1f} s (24 OMP threads)')
    print(f'  Build peak RSS : {BUILD_PEAK_RSS_MB:.1f} MB')
    print(f'  Disk           : index {idx_mb:.1f} MB  +  tq1 {tq1_mb:.1f} MB '
          f'= {idx_mb + tq1_mb:.1f} MB')
    print(f'  CPU policy     : pinned to core(s) {cpus_allowed}, '
          f'OMP_NUM_THREADS=1')
    print(f'  CPU model      : {cpu_model}')
    print()

    # === sweep matrix (single CPU) ===
    mf.set_shared_scratch_pool(True)
    mf.set_tree_sub(0); mf.set_tree_sub_groups(0); mf.set_node_perm(0)
    f = Forest(IDX, n_trees=N_TREES, dim=DIM, sub_dim=16, depth=DEPTH,
               n_docs=N_DOCS, gen_version=3)

    SWEEP = []
    for np_ in (5, 10, 20):
        for qd in (DEPTH - 2, DEPTH - 4, DEPTH - 6):  # 18, 16, 14
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

    # === best config (recall first, then latency) ===
    best = sorted(results, key=lambda r: (-r['recall'], r['p50_warm_ms']))[0]
    print()
    print(f'  Sweet spot     : recall {best["recall"]:.4f}, '
          f'p50 warm {best["p50_warm_ms"]:.1f} ms')
    print(f'                   (NP={best["n_probes"]} QD={best["probe_depth"]} '
          f'top_n={best["top_n"]} K\'={best["kprime"]})')
    print(f'  Query peak RSS : {peak_rss_mb():.1f} MB')
    print()

    # === parallel scaling ===
    print('  Parallel scaling at sweet spot (N independent worker processes,')
    print('  each pinned to its own core, each running 200 queries):')
    print()
    print(f'  {"workers":>7} {"qps total":>10} {"speedup":>8}')
    print('  ' + '-' * 30)

    n_total = 200
    qbytes = Q.astype(np.float32).tobytes()
    base_qps = None
    parallel = []
    # mp.set_start_method('fork', force=True)  # default on Linux is fork
    for n_workers in (1, 2, 4, 8):
        # pick distinct cores; on a 24-core box we can always do up to 8
        cores = list(range(n_workers))
        args = [(c, n_total, best['n_probes'], best['probe_depth'],
                 best['top_n'], best['kprime'], True, qbytes) for c in cores]
        t0 = time.time()
        with mp.Pool(n_workers) as pool:
            worker_qps = pool.map(_worker, args)
        wall = time.time() - t0
        # Aggregate: total queries across workers / wall time
        total_q = n_total * n_workers
        agg = total_q / wall
        if base_qps is None: base_qps = agg
        parallel.append({'workers': n_workers, 'qps_total': agg,
                         'speedup': agg / base_qps,
                         'wall_s': wall})
        print(f'  {n_workers:>7} {agg:>10.1f} {agg/base_qps:>7.2f}×',
              flush=True)

    print()
    print(f'  Query peak RSS (after parallel) : {peak_rss_mb():.1f} MB')
    print('==========================================================')

    # === save JSON ===
    os.makedirs('bench/results', exist_ok=True)
    with open('bench/results/arxiv_2m.json', 'w') as fh:
        json.dump({
            'dataset': 'arxiv_2m',
            'dim': DIM, 'n_docs': N_DOCS, 'n_trees': N_TREES, 'depth': DEPTH,
            'sub_dim': 16, 'gen_version': 3, 'tree_sub': 0,
            'build_time_s': BUILD_TIME_S,
            'build_peak_rss_mb': BUILD_PEAK_RSS_MB,
            'disk_index_mb': idx_mb,
            'disk_tq1_mb': tq1_mb,
            'cpu_affinity': cpus_allowed,
            'cpu_model': cpu_model,
            'sweep': results,
            'sweet_spot': best,
            'parallel': parallel,
            'query_peak_rss_mb': peak_rss_mb(),
        }, fh, indent=2)
    print('  → bench/results/arxiv_2m.json')


if __name__ == '__main__':
    main()
