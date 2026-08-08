"""GIST 1M focused parallel scaling test: NP=10 at QD=14 and QD=12,
each measured single-CPU and 8-CPU parallel for a few top_n values."""

import os, sys, time, multiprocessing as mp
os.environ['OMP_NUM_THREADS'] = '1'
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


def drop_caches():
    os.sync(); open('/proc/sys/vm/drop_caches', 'w').write('3')


def time_single(forest, queries, gt, np_, qd, tn, kp):
    drop_caches()
    for i in range(5):
        q = queries[i % len(queries)]
        ids, _, n = forest.query_probes(q, np_, top_n=tn, probe_depth=qd)
        cand = np.asarray(ids[:n], dtype=np.int32); cand = cand[cand >= 0]
        forest.rerank_tq1(TQ1, BASE, q, cand, kprime=kp, top_k=10)
    recs, lats = [], []
    for qi in range(5, 5 + 100):
        q = queries[qi % len(queries)]
        ref = set(int(x) for x in gt[qi % len(gt)])
        t0 = time.time()
        ids, _, n = forest.query_probes(q, np_, top_n=tn, probe_depth=qd)
        cand = np.asarray(ids[:n], dtype=np.int32); cand = cand[cand >= 0]
        top = forest.rerank_tq1(TQ1, BASE, q, cand, kprime=kp, top_k=10)
        lats.append((time.time() - t0) * 1000.0)
        recs.append(sum(1 for x in top if int(x) in ref) / 10.0)
    return float(np.mean(recs)), float(np.percentile(lats, 50)), \
           float(np.percentile(lats, 95))


def _worker(args):
    core, n_q, np_, qd, tn, kp, qbytes = args
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
        ids, _, n = f.query_probes(q, np_, top_n=tn, probe_depth=qd)
        cand = _np.asarray(ids[:n], dtype=_np.int32); cand = cand[cand >= 0]
        f.rerank_tq1(TQ1, BASE, q, cand, kprime=kp, top_k=10)
    t0 = time.time()
    for i in range(n_q):
        q = qs[i % len(qs)]
        ids, _, n = f.query_probes(q, np_, top_n=tn, probe_depth=qd)
        cand = _np.asarray(ids[:n], dtype=_np.int32); cand = cand[cand >= 0]
        f.rerank_tq1(TQ1, BASE, q, cand, kprime=kp, top_k=10)
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

    mf.set_shared_scratch_pool(True)
    f = Forest(IDX, n_trees=N_TREES, dim=DIM, sub_dim=16, depth=DEPTH,
               n_docs=N_DOCS, gen_version=3)

    qbytes = Q.astype(np.float32).tobytes()
    # NP=5 sweep at fused path (depth=18 → qd_eff=0)
    CONFIGS_NP = [
        (5,  0, 16_000, 1000),
        (5,  0, 32_000, 2000),
        (5,  0, 64_000, 4000),
        (10, 0, 64_000, 4000),  # ref point from earlier
    ]
    # Repack into the loop variable name
    CONFIGS = [(qd, tn, kp) for (NP_loop, qd, tn, kp) in CONFIGS_NP]
    NP_PER_CONFIG = [cfg[0] for cfg in CONFIGS_NP]

    print(f'GIST 1M : NP sweep, single CPU vs 8 workers throughput\n')
    print(f'{"NP":>3} {"QD":>3} {"top_n":>7} {"K\'":>6} | '
          f'{"recall":>7} {"p50 1cpu":>9} | '
          f'{"qps 1cpu":>9} {"qps 8cpu":>9} {"speedup":>8} {"throughput":>11}')
    print('-' * 93)

    for cfg_i, (qd, tn, kp) in enumerate(CONFIGS):
        NP = NP_PER_CONFIG[cfg_i]
        # single CPU
        rec, p50, p95 = time_single(f, Q, GT, NP, qd, tn, kp)
        qps_1 = 1000.0 / p50  # rough qps from p50

        # parallel 8 workers
        cores = list(range(8))
        n_per = 100
        args = [(c, n_per, NP, qd, tn, kp, qbytes) for c in cores]
        t0 = time.time()
        with mp.Pool(8) as pool:
            _ = pool.map(_worker, args)
        wall = time.time() - t0
        total_q = n_per * 8
        qps_8 = total_q / wall
        speedup = qps_8 / qps_1
        p50_eff = 1000.0 / qps_8  # effective per-query throughput latency

        print(f'{NP:>3} {qd:>3} {tn:>7} {kp:>6} | '
              f'{rec:>7.4f} {p50:>7.0f}ms | '
              f'{qps_1:>7.1f}/s {qps_8:>7.1f}/s {speedup:>6.2f}× '
              f'{p50_eff:>9.1f}ms',
              flush=True)


if __name__ == '__main__':
    main()
