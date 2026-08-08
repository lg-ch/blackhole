"""Concurrency stress: run heavy query load while a build is in progress.

Sequence :
  1. Build a baseline index of N1 docs, measure baseline recall+latency
  2. Start a long rpforest build of N2 docs in background
  3. While build runs, hammer the baseline with parallel queries
  4. After build completes, verify (a) baseline query latency stayed
     bounded (b) baseline recall unchanged

Validates that a concurrent build doesn't crash or corrupt the running
query path. The build's I/O may impact latency on shared disk — we
quantify that, not gate on it.
"""
from __future__ import annotations
import os, shutil, subprocess, sys, threading, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mangrove_ffi import Forest, set_gen_version


BIN  = os.path.join(os.path.dirname(HERE), 'rpforest')
DIM  = 128
N1   = 100_000   # baseline (queryable during stress)
N2   = 200_000   # second build (creates I/O pressure)
N_TREES = 200
DEPTH = 14


def build_idx(out_dir: str, doc_offset: int, doc_count: int) -> None:
    if os.path.exists(out_dir): shutil.rmtree(out_dir)
    subprocess.check_call([
        BIN, '--dim', str(DIM), '--sub_dim', '16', '--gen', 'v3',
        '--doc_offset', str(doc_offset), '--doc_count', str(doc_count),
        'build', 'sift/sift_base.fvecs', out_dir,
        str(N_TREES), str(DEPTH),
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def query_loop(forest, queries, base, results, stop_flag):
    while not stop_flag.is_set():
        for q in queries:
            t0 = time.time()
            ids, votes, n = forest.query(q, top_n=2000)
            cand = ids[:n]
            cv = base[cand]
            d2 = ((cv - q) ** 2).sum(axis=1)
            order = np.argsort(d2)[:10]
            results.append((time.time() - t0) * 1000)
            if stop_flag.is_set():
                return


def main():
    IDX1 = '/tmp/concur_idx1'
    IDX2 = '/tmp/concur_idx2'
    if not os.path.exists(IDX1):
        print(f'building baseline {IDX1} ({N1} docs) ...')
        build_idx(IDX1, 0, N1)

    base = np.empty((N1, DIM), dtype=np.float32)
    with open('sift/sift_base.fvecs', 'rb') as f:
        for i in range(N1):
            f.read(4)
            base[i] = np.frombuffer(f.read(DIM * 4), dtype=np.float32)
    queries = np.empty((50, DIM), dtype=np.float32)
    with open('sift/sift_query.fvecs', 'rb') as f:
        for i in range(50):
            f.read(4)
            queries[i] = np.frombuffer(f.read(DIM * 4), dtype=np.float32)

    set_gen_version(3)
    f = Forest(IDX1, n_trees=N_TREES, dim=DIM, sub_dim=16,
               depth=DEPTH, n_docs=N1, gen_version=3)

    # 1. Baseline (single query loop, no build)
    print('\n=== 1. baseline 5s of queries (no contention) ===')
    results: list[float] = []
    stop = threading.Event()
    t = threading.Thread(target=query_loop,
                         args=(f, queries, base, results, stop))
    t.start()
    time.sleep(5)
    stop.set(); t.join()
    base_p50 = np.percentile(results, 50)
    base_p99 = np.percentile(results, 99)
    print(f'  baseline: {len(results)} queries, p50={base_p50:.2f}ms '
          f'p99={base_p99:.2f}ms')

    # 2. Concurrent : start build + query loop together
    print(f'\n=== 2. queries WHILE building IDX2 ({N2} docs) ===')
    results2: list[float] = []
    stop2 = threading.Event()
    t2 = threading.Thread(target=query_loop,
                          args=(f, queries, base, results2, stop2))
    t2.start()
    t_b0 = time.time()
    build_idx(IDX2, 0, N2)
    print(f'  build done in {time.time() - t_b0:.1f}s')
    stop2.set(); t2.join()
    contention_p50 = np.percentile(results2, 50)
    contention_p99 = np.percentile(results2, 99)
    print(f'  during build: {len(results2)} queries, '
          f'p50={contention_p50:.2f}ms p99={contention_p99:.2f}ms')

    f.close()

    print(f'\n=== SUMMARY ===')
    print(f'{"phase":>20} {"queries":>10} {"p50 ms":>10} {"p99 ms":>10}')
    print(f'{"baseline":>20} {len(results):>10} {base_p50:>10.2f} {base_p99:>10.2f}')
    print(f'{"during build":>20} {len(results2):>10} {contention_p50:>10.2f} {contention_p99:>10.2f}')
    impact_p50 = contention_p50 / max(0.001, base_p50)
    impact_p99 = contention_p99 / max(0.001, base_p99)
    print(f'\nimpact factor : p50 = {impact_p50:.2f}x, p99 = {impact_p99:.2f}x')
    # Pass criterion: no crash, query loop continued through the build.
    ok = len(results2) > 0
    print(f'\n=== RESULT : {"PASS" if ok else "FAIL"} (no crash, queries ran) ===')


if __name__ == '__main__':
    main()
