"""Bench : server-side traversal vs client-side traversal.

Two modes for the same query :
  A) server-side : client sends qvec, server does traversal + merge + rerank
  B) client-side : client computes leaves locally (Python), sends them,
                   server does merge only, client reranks locally

We measure :
  - SERVER time only (process time on the server side)
  - CLIENT round-trip time (what the user actually waits)
  - CLIENT compute time (the cost of moving work to the client)

Run on a real index (SIFT 1M @ 200 trees x depth 18, sub_dim=16).
"""
from __future__ import annotations
import os, shutil, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mangrove.traversal import compute_leaves
from mangrove_ffi import Forest, set_gen_version, set_external_leaves


DIM, SUB_DIM, DEPTH, N_TREES = 128, 16, 18, 200
N_DOC = 1_000_000
TOP_N = 4000
TOP_K = 10
IDX = '/tmp/cs_bench_idx'


def build_if_needed():
    if not os.path.exists(IDX + '/meta.txt'):
        subprocess.check_call([
            './rpforest', '--dim', str(DIM), '--sub_dim', str(SUB_DIM),
            '--gen', 'v3', '--doc_offset', '0', '--doc_count', str(N_DOC),
            'build', 'sift/sift_base.fvecs', IDX,
            str(N_TREES), str(DEPTH),
        ], stdout=subprocess.DEVNULL)


def read_fvecs(path, n, d=DIM):
    out = np.empty((n, d), dtype=np.float32)
    with open(path, 'rb') as f:
        for i in range(n):
            f.read(4); out[i] = np.frombuffer(f.read(d * 4), dtype=np.float32)
    return out


def main():
    build_if_needed()
    set_gen_version(3)
    f = Forest(IDX, n_trees=N_TREES, dim=DIM, sub_dim=SUB_DIM,
               depth=DEPTH, n_docs=N_DOC, gen_version=3)
    base = read_fvecs('sift/sift_base.fvecs', N_DOC)
    queries = read_fvecs('sift/sift_query.fvecs', 100)

    # Warm up cache (cold first query is unfair to compare)
    for q in queries[:10]:
        set_external_leaves(None)
        _ = f.query(q, top_n=TOP_N)

    # ---- A) server-side path ----
    a_server = []   # the part the server actually does
    a_total  = []   # client perspective (= server + client rerank which is via FFI rerank_l2)
    for q in queries:
        set_external_leaves(None)
        t0 = time.time()
        ids, votes, n = f.query(q, top_n=TOP_N)
        t_query = time.time()
        # server-side rerank (FFI rerank_l2 reads from base_path)
        top10 = f.rerank_l2('sift/sift_base.fvecs', q, ids[:n], top_k=TOP_K)
        t_rerank = time.time()
        a_server.append((t_query - t0) * 1000)
        a_total.append((t_rerank - t0) * 1000)

    # ---- B) client-side path ----
    b_client_traversal = []   # Python compute_leaves
    b_server = []             # K-way merge only (no traversal, no rerank)
    b_client_rerank = []      # Python L2 rerank on candidates
    b_total = []              # full round-trip from client perspective
    for q in queries:
        t0 = time.time()
        leaves = compute_leaves(q, n_trees=N_TREES, depth=DEPTH,
                                sub_dim=SUB_DIM, dim=DIM, gen_version=3)
        t_trav = time.time()
        set_external_leaves(leaves)
        ids, votes, n = f.query(q, top_n=TOP_N)
        set_external_leaves(None)
        t_merge = time.time()
        # client-side rerank in numpy
        cand_vecs = base[ids[:n]]
        d2 = ((cand_vecs - q) ** 2).sum(axis=1)
        order = np.argsort(d2)[:TOP_K]
        top10 = ids[order]
        t_rerank = time.time()
        b_client_traversal.append((t_trav - t0) * 1000)
        b_server.append((t_merge - t_trav) * 1000)
        b_client_rerank.append((t_rerank - t_merge) * 1000)
        b_total.append((t_rerank - t0) * 1000)

    f.close()

    def stat(name, arr):
        return (f'{name:>22} : p50={np.percentile(arr, 50):>7.2f}ms '
                f'p95={np.percentile(arr, 95):>7.2f}ms '
                f'p99={np.percentile(arr, 99):>7.2f}ms')

    print(f'\nBench SIFT 1M : {N_TREES} trees × d{DEPTH} × sub_dim={SUB_DIM} · '
          f'top_n={TOP_N} top_k={TOP_K} · 100 queries\n')

    print('--- A) server-side path (current default) ---')
    print(stat('server query', a_server))
    print(stat('server query+rerank', a_total))
    print()

    print('--- B) client-side path (privacy mode) ---')
    print(stat('client traverse (py)', b_client_traversal))
    print(stat('server merge (no trav)', b_server))
    print(stat('client rerank (np)',  b_client_rerank))
    print(stat('CLIENT TOTAL',         b_total))
    print()

    server_a_p50 = np.percentile(a_total, 50)
    server_b_p50 = np.percentile(b_server, 50)
    print(f'=== SERVER load comparison (per-query p50) ===')
    print(f'  server-side path : {server_a_p50:.2f} ms')
    print(f'  client-side path : {server_b_p50:.2f} ms')
    print(f'  server save factor : {server_a_p50/server_b_p50:.2f}x  '
          f'({(1 - server_b_p50/server_a_p50)*100:.0f}% less server CPU)')

    total_a_p50 = np.percentile(a_total, 50)
    total_b_p50 = np.percentile(b_total, 50)
    print(f'\n=== END-TO-END comparison (client perspective, p50) ===')
    print(f'  server-side total : {total_a_p50:.2f} ms')
    print(f'  client-side total : {total_b_p50:.2f} ms (incl Python traverse)')
    print(f'  client overhead   : {total_b_p50 - total_a_p50:+.2f} ms')


if __name__ == '__main__':
    main()
