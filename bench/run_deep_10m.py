"""DEEP 10M (dim 96) — full benchmark, same protocol as run_arxiv_2m.py.

Source: DEEP1B big-ann-benchmarks (first 10M of base.1B.fbin), dim 96 float32.
Queries: query.10K.fbin (10K queries); GT@10 from gt.10M.bin.

Build:
  ./rpforest build /mnt/mangrove/datasets/deep1b/base.1B.fbin \
      /mnt/mangrove/indexes/deep_10m 256 22 \
      --sub_dim 16 --gen v3 --dim 96 --doc_count 10000000

TQ sidecar: we re-use the existing deep_10m.tq4 (covers 1B at 64 GB; only
the first 10M codes are needed at query time). For a TQ1 (1-bit) variant
matching arxiv/GIST harness, build it on a subset of 10M.
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

DATA = '/mnt/mangrove/datasets/deep1b'
IDX  = '/mnt/mangrove/indexes/deep_10m'
BASE = f'{DATA}/base.1B.fbin'
TQ4  = f'{DATA}/deep_10m.tq4'       # existing 64 GB sidecar (covers 1B)
TQ1  = f'{DATA}/deep_10m.tq1'       # built on full 1B file but only first 10M
                                    # codes are touched at query time (doc_ids ≤ 10M)
QP   = f'{DATA}/query.10K.fbin'
GP   = f'{DATA}/gt.10M.bin'
DIM, N_TREES, DEPTH, N_DOCS = 96, 256, 22, 10_000_000

BUILD_TIME_S = 1587.0           # observed (under shared CPU with SIFT 1B build)
BUILD_PEAK_RSS_MB = 198.6       # convert phase peak (shared with concurrent build)


def drop_caches():
    os.sync(); open('/proc/sys/vm/drop_caches', 'w').write('3')


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def disk_size_mb(path):
    if not os.path.exists(path): return 0.0
    out = subprocess.run(['du', '-sb', path], capture_output=True, text=True)
    return int(out.stdout.split()[0]) / (1024.0 * 1024.0)


def read_fbin_floats(path, nmax):
    with open(path, 'rb') as f:
        n, d = struct.unpack('<II', f.read(8))
        n = min(n, nmax)
        return np.frombuffer(f.read(n*d*4), np.float32).reshape(n, d).copy()


def read_fbin_ids(path, nmax, k):
    with open(path, 'rb') as f:
        n, kf = struct.unpack('<II', f.read(8))
        n = min(n, nmax)
        return np.frombuffer(f.read(n*kf*4), np.int32).reshape(n, kf)[:, :k].copy()


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


def time_single(forest, queries, gt, np_, qd, tn, kp, tq_path, use_tq1):
    drop_caches()
    rerank_fn = forest.rerank_tq1 if use_tq1 else forest.rerank_tq
    for i in range(5):
        q = queries[i % len(queries)]
        ids, _, n = forest.query_probes(q, np_, top_n=tn, probe_depth=qd)
        cand = np.asarray(ids[:n], dtype=np.int32); cand = cand[cand >= 0]
        rerank_fn(tq_path, BASE, q, cand, kprime=kp, top_k=10)
    recs, lats = [], []
    for qi in range(5, 5 + 100):
        q = queries[qi % len(queries)]
        ref = set(int(x) for x in gt[qi % len(gt)])
        t0 = time.time()
        ids, _, n = forest.query_probes(q, np_, top_n=tn, probe_depth=qd)
        cand = np.asarray(ids[:n], dtype=np.int32); cand = cand[cand >= 0]
        top = rerank_fn(tq_path, BASE, q, cand, kprime=kp, top_k=10)
        lats.append((time.time() - t0) * 1000.0)
        recs.append(sum(1 for x in top if int(x) in ref) / 10.0)

    drop_caches()
    qcold = queries[5 % len(queries)]
    t0 = time.time()
    ids, _, n = forest.query_probes(qcold, np_, top_n=tn, probe_depth=qd)
    cand = np.asarray(ids[:n], dtype=np.int32); cand = cand[cand >= 0]
    rerank_fn(tq_path, BASE, qcold, cand, kprime=kp, top_k=10)
    cold = (time.time() - t0) * 1000.0

    return float(np.mean(recs)), float(np.percentile(lats, 50)), \
           float(np.percentile(lats, 95)), float(cold)


def _worker(args):
    core, n_q, np_, qd, tn, kp, tq_path, use_tq1, qbytes = args
    os.environ['OMP_NUM_THREADS'] = '1'
    os.sched_setaffinity(0, {core})
    import numpy as _np
    from mangrove_ffi import Forest as _Forest
    import mangrove_ffi as _mf
    _mf.set_shared_scratch_pool(True)
    f = _Forest(IDX, n_trees=N_TREES, dim=DIM, sub_dim=16, depth=DEPTH,
                n_docs=N_DOCS, gen_version=3)
    qs = _np.frombuffer(qbytes, dtype=_np.float32).reshape(-1, DIM)
    rerank_fn = f.rerank_tq1 if use_tq1 else f.rerank_tq
    for i in range(5):
        q = qs[i % len(qs)]
        ids, _, n = f.query_probes(q, np_, top_n=tn, probe_depth=qd)
        cand = _np.asarray(ids[:n], dtype=_np.int32); cand = cand[cand >= 0]
        rerank_fn(tq_path, BASE, q, cand, kprime=kp, top_k=10)
    t0 = time.time()
    for i in range(n_q):
        q = qs[i % len(qs)]
        ids, _, n = f.query_probes(q, np_, top_n=tn, probe_depth=qd)
        cand = _np.asarray(ids[:n], dtype=_np.int32); cand = cand[cand >= 0]
        rerank_fn(tq_path, BASE, q, cand, kprime=kp, top_k=10)
    dt = time.time() - t0
    f.close()
    return n_q / dt


def main():
    Q  = read_fbin_floats(QP, 200)
    GT = read_fbin_ids(GP, 200, 10)
    nq = len(Q)

    # pick sidecar : TQ1 if built, else TQ4
    use_tq1 = os.path.exists(TQ1)
    tq_path = TQ1 if use_tq1 else TQ4
    tq_label = 'TQ1 (1-bit)' if use_tq1 else 'TQ4 (4-bit)'
    tq_mb = disk_size_mb(tq_path)

    cpus_allowed = sorted(os.sched_getaffinity(0))
    cpu = cpu_model()
    idx_mb = disk_size_mb(IDX)

    print('==========================================================')
    print(' DEEP 10M — mangrove-search benchmark')
    print('==========================================================')
    print(f'  Vectors        : float32, dim {DIM}, n_docs {N_DOCS:,}')
    print(f'  Source         : DEEP1B (big-ann-benchmarks), first 10M')
    print(f'  Queries / GT   : {nq} held-out / GT@10 from gt.10M.bin')
    print(f'  Technique      : RP-forest ({N_TREES} trees, depth {DEPTH}, '
          f'sub_dim 16, gen v3)')
    print(f'                   + {tq_label} sidecar')
    print(f'                   + exact L2 rerank on K\' survivors')
    if BUILD_TIME_S:
        print(f'  Build time     : {BUILD_TIME_S:.1f} s (24 OMP threads)')
        print(f'  Build peak RSS : {BUILD_PEAK_RSS_MB:.1f} MB')
    print(f'  Disk           : index {idx_mb:.1f} MB  +  {tq_label.split()[0]} '
          f'{tq_mb:.1f} MB')
    print(f'  CPU policy     : pinned to core(s) {cpus_allowed}, '
          f'OMP_NUM_THREADS=1')
    print(f'  CPU model      : {cpu}')
    print()

    mf.set_shared_scratch_pool(True)
    f = Forest(IDX, n_trees=N_TREES, dim=DIM, sub_dim=16, depth=DEPTH,
               n_docs=N_DOCS, gen_version=3)

    SWEEP = []
    for np_ in (5, 10, 20):
        for qd in (DEPTH - 2, DEPTH - 4, DEPTH - 6):  # 20, 18, 16
            for tn in (16_000, 32_000, 64_000):
                kp = max(100, tn // 16) if not use_tq1 else max(500, tn // 16)
                SWEEP.append((np_, qd, tn, kp))

    print(f'  Sweep matrix : NP × QD × top_n')
    print(f'  Single CPU, 100 warm queries + 1 cold per row\n')
    print(f'  {"NP":>3} {"QD":>3} {"top_n":>7} {"K\'":>6} | '
          f'{"recall":>7} | {"p50w":>7} {"p95w":>7} {"cold":>7}')
    print('  ' + '-' * 67)

    results = []
    for np_, qd, tn, kp in SWEEP:
        rec, p50, p95, cold = time_single(f, Q, GT, np_, qd, tn, kp,
                                          tq_path, use_tq1)
        results.append({
            'n_probes': np_, 'probe_depth': qd, 'top_n': tn, 'kprime': kp,
            'use_tq1': use_tq1,
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
        args = [(c, n_per, best['n_probes'], best['probe_depth'],
                 best['top_n'], best['kprime'], tq_path, use_tq1, qbytes)
                for c in cores]
        t0 = time.time()
        with mp.Pool(n_workers) as pool:
            _ = pool.map(_worker, args)
        wall = time.time() - t0
        total_q = n_per * n_workers
        agg = total_q / wall
        if base_qps is None: base_qps = agg
        per_q = 1000.0 / agg
        parallel.append({'workers': n_workers, 'qps_total': agg,
                         'speedup': agg / base_qps, 'per_query_ms': per_q,
                         'wall_s': wall})
        print(f'  {n_workers:>7} {agg:>10.1f} {agg/base_qps:>7.2f}× '
              f'{per_q:>7.1f}ms', flush=True)

    print()
    print(f'  Query peak RSS (after parallel) : {peak_rss_mb():.1f} MB')
    print('==========================================================')

    os.makedirs('bench/results', exist_ok=True)
    with open('bench/results/deep_10m.json', 'w') as fh:
        json.dump({
            'dataset': 'deep_10m',
            'dim': DIM, 'n_docs': N_DOCS, 'n_trees': N_TREES, 'depth': DEPTH,
            'sub_dim': 16, 'gen_version': 3,
            'tq_sidecar': tq_label, 'tq_disk_mb': tq_mb,
            'build_time_s': BUILD_TIME_S,
            'build_peak_rss_mb': BUILD_PEAK_RSS_MB,
            'disk_index_mb': idx_mb,
            'cpu_affinity': cpus_allowed,
            'cpu_model': cpu,
            'sweep': results,
            'sweet_spot': best,
            'parallel': parallel,
            'query_peak_rss_mb': peak_rss_mb(),
        }, fh, indent=2)
    print('  → bench/results/deep_10m.json')


if __name__ == '__main__':
    main()
