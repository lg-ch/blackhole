"""Validate the multi-index recall-parity claim :
   recall@10 of 1×N segment should == recall@10 of k×(N/k) segments
   (same n_trees, depths chosen as log2(N_seg)-2 each).

Why this should work : we use the pairwise-seed scheme (each tree's
hyperplanes derive from a deterministic tree_seed(t)), so all segments
share the SAME splitting structure across docs. The per-segment merge in
LiveIndex.query computes votes per doc, summing across segments — the
mathematics is identical to a single bigger index with the same trees.
"""
from __future__ import annotations
import os, shutil, struct, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from live_index import LiveIndex
from mangrove_ffi import Forest, set_gen_version


SIFT_BASE  = 'sift/sift_base.fvecs'
SIFT_QUERY = 'sift/sift_query.fvecs'
SIFT_GT    = 'sift/sift_groundtruth.ivecs'
DIM        = 128
N_DOC      = 1_000_000
N_TREES    = 200
SUB_DIM    = 16


def read_fvecs(path, n, d=DIM):
    out = np.empty((n, d), dtype=np.float32)
    with open(path, 'rb') as f:
        for i in range(n):
            f.read(4)
            out[i] = np.frombuffer(f.read(d * 4), dtype=np.float32)
    return out


def read_gt(path, n, k=100):
    out = np.empty((n, k), dtype=np.int32)
    with open(path, 'rb') as f:
        for i in range(n):
            f.read(4)
            out[i] = np.frombuffer(f.read(k * 4), dtype=np.int32)
    return out


def build_segment(out_dir, doc_offset, n_vecs, depth):
    """Build one rpforest segment from a slice of sift_base.fvecs."""
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    cmd = [
        os.path.join(os.path.dirname(HERE), 'rpforest'),
        '--dim', str(DIM), '--sub_dim', str(SUB_DIM), '--gen', 'v3',
        '--doc_offset', str(doc_offset),
        '--doc_count', str(n_vecs),
        '--doc_id_base', str(doc_offset),
        'build', SIFT_BASE, out_dir,
        str(N_TREES), str(depth),
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL)


def bench_config(name, splits: list[tuple[int, int]], queries, gt,
                 top_n=4000, n_q=100, max_query_depth: int | None = None):
    """splits = [(doc_offset, n_vecs), ...]. depth chosen per segment.
       Returns (mean_recall, mean_p50_ms).                              */"""
    seg_dirs = []
    for i, (off, n) in enumerate(splits):
        d = max(8, int(np.log2(max(2, n))) - 2)
        sdir = f'/tmp/parity_{name}_seg{i}'
        build_segment(sdir, off, n, d)
        seg_dirs.append((sdir, d, off, n))

    # Open each as Forest, query, merge votes in Python (same as LiveIndex.query)
    set_gen_version(3)
    forests = []
    for sdir, d, off, n in seg_dirs:
        n_docs_sentinel = N_DOC  # global sentinel >= any doc_id
        forests.append(Forest(sdir, n_trees=N_TREES, dim=DIM, sub_dim=SUB_DIM,
                              depth=d, n_docs=n_docs_sentinel, gen_version=3))

    rss_start = os.popen(f'grep VmRSS /proc/{os.getpid()}/status').read().split()[1]

    base_arr = read_fvecs(SIFT_BASE, N_DOC)  # for rerank
    lats = []
    recalls = []
    for qi in range(n_q):
        q = queries[qi]
        t0 = time.time()
        vote_acc: dict[int, int] = {}
        for f in forests:
            ids, votes, n = f.query(q, top_n=top_n)
            for j in range(n):
                vote_acc[int(ids[j])] = vote_acc.get(int(ids[j]), 0) + int(votes[j])
        items = sorted(vote_acc.items(), key=lambda kv: -kv[1])[:top_n]
        cand_ids = np.array([k for k, _ in items], dtype=np.int32)
        # L2 rerank in numpy
        cand_vecs = base_arr[cand_ids]
        d2 = ((cand_vecs - q) ** 2).sum(axis=1)
        order = np.argsort(d2)[:10]
        topk = cand_ids[order]
        lats.append((time.time() - t0) * 1000)
        s = set(int(x) for x in gt[qi, :10])
        recalls.append(sum(1 for x in topk if int(x) in s) / 10)

    for f in forests: f.close()
    rss_end = os.popen(f'grep VmRSS /proc/{os.getpid()}/status').read().split()[1]
    return {
        'name':    name,
        'n_segs':  len(splits),
        'depths':  [d for _, d, _, _ in seg_dirs],
        'recall':  float(np.mean(recalls)),
        'lat_p50': float(np.percentile(lats, 50)),
        'lat_p95': float(np.percentile(lats, 95)),
        'rss_kb':  int(rss_end) - int(rss_start),
    }


def main():
    print(f'SIFT 1M, n_trees={N_TREES}, sub_dim={SUB_DIM}, gen=v3, top_n=4000\n')
    queries = read_fvecs(SIFT_QUERY, 100)
    gt = read_gt(SIFT_GT, 100, 100)

    configs = [
        ('mono',   [(0, N_DOC)]),
        ('split2', [(i * N_DOC // 2, N_DOC // 2) for i in range(2)]),
        ('split4', [(i * N_DOC // 4, N_DOC // 4) for i in range(4)]),
        ('split8', [(i * N_DOC // 8, N_DOC // 8) for i in range(8)]),
    ]

    results = []
    for name, splits in configs:
        print(f'=== {name} : {len(splits)} segments ===')
        r = bench_config(name, splits, queries, gt)
        results.append(r)
        print(f'  depths {r["depths"]}, recall={r["recall"]:.4f}, '
              f'p50={r["lat_p50"]:.2f}ms p95={r["lat_p95"]:.2f}ms\n')

    print('\n=== SUMMARY ===')
    print(f'{"config":>10} {"n_segs":>7} {"recall":>8} {"p50 ms":>8} {"p95 ms":>8}')
    for r in results:
        print(f'{r["name"]:>10} {r["n_segs"]:>7} {r["recall"]:>8.4f} '
              f'{r["lat_p50"]:>8.2f} {r["lat_p95"]:>8.2f}')

    # Parity assertion: split recalls should be within 2% of mono recall
    mono_r = results[0]['recall']
    print(f'\nMono recall : {mono_r:.4f}')
    max_drift = 0.0
    for r in results[1:]:
        d = mono_r - r['recall']
        max_drift = max(max_drift, abs(d))
        flag = '✓' if abs(d) <= 0.02 else '✗'
        print(f'  {r["name"]} drift = {d:+.4f} {flag}')
    print(f'Max drift : {max_drift:.4f}')
    print(f'\nVERDICT : {"PASS (parity within 2%)" if max_drift <= 0.02 else "FAIL"}')


if __name__ == '__main__':
    main()
