"""Validate pre-filter recall claim : with a random bitmap filter at
   density d, recall@10 of forest_with_filter(qvec, filter)
   must stay close to recall@10 of brute-force L2(qvec ∩ filter).

Setup :
   1. Build SIFT 1M monolithic
   2. Generate random filter bitmaps at densities 1%, 10%, 30%, 70%
   3. For each (query, density) :
        - forest with filter (pre-filter via CRoaring)
        - brute-force L2 restricted to the filter set → ground truth
   4. recall@10 = | forest_topk ∩ bf_topk | / 10

We expect recall ≥ 0.95 across all densities. Low-density (1%) is the
edge case : few candidates pass the filter, K-way merge may exhaust the
budget before finding the actual NNs.
"""
from __future__ import annotations
import os, shutil, struct, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mangrove_ffi import Forest, set_gen_version


SIFT_BASE  = 'sift/sift_base.fvecs'
SIFT_QUERY = 'sift/sift_query.fvecs'
DIM        = 128
N_DOC      = 1_000_000
N_TREES    = 200
SUB_DIM    = 16
DEPTH      = 18
TOP_N      = 4000


def read_fvecs(path, n, d=DIM):
    out = np.empty((n, d), dtype=np.float32)
    with open(path, 'rb') as f:
        for i in range(n):
            f.read(4)
            out[i] = np.frombuffer(f.read(d * 4), dtype=np.float32)
    return out


def build_index(out_dir):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    cmd = [
        os.path.join(os.path.dirname(HERE), 'rpforest'),
        '--dim', str(DIM), '--sub_dim', str(SUB_DIM), '--gen', 'v3',
        'build', SIFT_BASE, out_dir,
        str(N_TREES), str(DEPTH),
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL)


def brute_force_topk(query, base_arr, allowed_ids, k=10):
    """Return top-k doc_ids ∈ allowed_ids by L2 to query."""
    sub = base_arr[allowed_ids]
    d2 = ((sub - query) ** 2).sum(axis=1)
    top = np.argpartition(d2, min(k - 1, len(d2) - 1))[:k]
    top = top[np.argsort(d2[top])]
    return [int(allowed_ids[i]) for i in top]


def main():
    print(f'SIFT 1M, n_trees={N_TREES}, depth={DEPTH}, top_n={TOP_N}\n')

    out_dir = '/tmp/filter_bench_idx'
    if not os.path.exists(out_dir + '/meta.txt'):
        print('  building index ...')
        build_index(out_dir)

    base_arr = read_fvecs(SIFT_BASE, N_DOC)
    queries  = read_fvecs(SIFT_QUERY, 100)

    set_gen_version(3)
    f = Forest(out_dir, n_trees=N_TREES, dim=DIM, sub_dim=SUB_DIM,
               depth=DEPTH, n_docs=N_DOC, gen_version=3)

    densities = [0.01, 0.05, 0.10, 0.30, 0.70]
    rng = np.random.default_rng(42)
    rows = []

    for dens in densities:
        n_allow = int(N_DOC * dens)
        allowed = rng.choice(N_DOC, size=n_allow, replace=False)
        allowed.sort()
        allowed_i32 = allowed.astype(np.int32)

        lats = []
        recalls = []
        for qi in range(100):
            q = queries[qi]

            # 1. ground truth : brute-force L2 over allowed
            bf_top10 = set(brute_force_topk(q, base_arr, allowed, k=10))

            # 2. forest with pre-filter via the doc-ids API
            t0 = time.time()
            ids, votes, n = f.query_with_ids(q, allowed_i32, top_n=TOP_N)
            cand = ids[:n]
            # Re-rank cands by L2
            cv = base_arr[cand]
            d2 = ((cv - q) ** 2).sum(axis=1)
            order = np.argsort(d2)[:10]
            top10 = set(int(cand[i]) for i in order)
            lats.append((time.time() - t0) * 1000)

            recalls.append(len(top10 & bf_top10) / 10)

        rows.append({
            'dens':     dens,
            'n_allow':  n_allow,
            'recall':   float(np.mean(recalls)),
            'recall_p10': float(np.percentile(recalls, 10)),
            'lat_p50':  float(np.percentile(lats, 50)),
            'lat_p95':  float(np.percentile(lats, 95)),
        })

    f.close()

    print(f'{"density":>10} {"|allowed|":>12} {"recall@10":>10} '
          f'{"r@10 p10":>10} {"p50 ms":>8} {"p95 ms":>8}')
    for r in rows:
        flag = '✓' if r['recall'] >= 0.95 else '✗'
        print(f'{r["dens"]*100:>9.1f}% {r["n_allow"]:>12,} '
              f'{r["recall"]:>10.4f} {r["recall_p10"]:>10.4f} '
              f'{r["lat_p50"]:>8.2f} {r["lat_p95"]:>8.2f}  {flag}')

    pass_all = all(r['recall'] >= 0.95 for r in rows)
    print(f'\nVERDICT : {"PASS" if pass_all else "FAIL (recall < 0.95 at some density)"}')


if __name__ == '__main__':
    main()
