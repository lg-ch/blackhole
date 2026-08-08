"""Side-by-side bench: mangrove-search vs FAISS-IVF+PQ vs hnswlib.

For each library, measures on the same query set:
  - recall@10 (vs ground truth)
  - latency p50 / p99 (ms)
  - peak RSS (MB)
  - on-disk index size (MB)

Usage:
    python3 scripts/bench_competitive.py \
        --base sift/sift_base.fvecs \
        --queries sift/sift_query.fvecs \
        --gt sift/sift_groundtruth.ivecs \
        --n_docs 1000000 --dim 128 \
        --mangrove_index /tmp/sift1m_competitive \
        --n_trees 200 --depth 20 --sub_dim 16

Build phases are slow (FAISS train + add ~minutes, hnswlib add ~minutes).
Indexes are cached on disk and reused on subsequent runs.
"""
from __future__ import annotations

import argparse
import os
import resource
import struct
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def read_fvecs(path: str, n: int, dim: int) -> np.ndarray:
    out = np.empty((n, dim), dtype=np.float32)
    with open(path, 'rb') as f:
        for i in range(n):
            d = struct.unpack('<i', f.read(4))[0]
            assert d == dim
            out[i] = np.frombuffer(f.read(d * 4), dtype=np.float32)
    return out


def read_ivecs_gt(path: str, n: int, top_k: int) -> np.ndarray:
    with open(path, 'rb') as f:
        head = struct.unpack('<i', f.read(4))[0]
        k = head
    out = np.empty((n, top_k), dtype=np.int32)
    row_bytes = 4 + k * 4
    with open(path, 'rb') as f:
        for i in range(n):
            f.read(4)
            row = f.read(k * 4)
            out[i] = np.frombuffer(row, dtype=np.int32)[:top_k]
    return out


def percentile(values, p):
    s = sorted(values)
    if not s:
        return 0.0
    k = (len(s) - 1) * p / 100.0
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def measure(fn, queries):
    lat = []
    out = []
    for q in queries:
        t0 = time.time()
        r = fn(q)
        lat.append((time.time() - t0) * 1000)
        out.append(r)
    return out, lat


def compute_recall(results, gt, top_k):
    rs = []
    for i, r in enumerate(results):
        gt_set = set(int(x) for x in gt[i][:top_k])
        hit = sum(1 for x in r[:top_k] if int(x) in gt_set)
        rs.append(hit / top_k)
    return float(np.mean(rs))


def bench_mangrove(args, queries, gt):
    """Build mangrove forest if missing, then query via FFI."""
    from mangrove_ffi import Forest
    if not os.path.exists(os.path.join(args.mangrove_index, 'meta.txt')):
        os.makedirs(args.mangrove_index, exist_ok=True)
        cmd = [
            './rpforest', '--dim', str(args.dim), '--sub_dim', str(args.sub_dim),
            '--gen', f'v{args.gen}',
            'build', args.base, args.mangrove_index,
            str(args.n_trees), str(args.depth),
        ]
        sys.stderr.write(f'  building: {" ".join(cmd)}\n')
        subprocess.check_call(cmd)
    t_open = time.time()
    f = Forest(args.mangrove_index, n_trees=args.n_trees, dim=args.dim,
               sub_dim=args.sub_dim, depth=args.depth,
               n_docs=args.n_docs, gen_version=args.gen)
    # warmup
    for q in queries[:5]:
        f.query(q, top_n=args.top_n)
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Forest returns vote-based candidates, then L2 rerank to top_k. This is
    # the realistic prod path — forest narrows to ~500 cands, rerank picks
    # exact NN. FAISS-IVF+PQ and hnswlib return L2-sorted by construction.
    def _q(q):
        ids, _, n = f.query(q, top_n=args.top_n)
        return f.rerank_l2(args.base, q, ids[:n], top_k=args.top_k)
    res, lat = measure(_q, queries)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rcl = compute_recall(res, gt, args.top_k)
    size_mb = _dir_size_mb(args.mangrove_index)
    f.close()
    return dict(name='mangrove', recall=rcl,
                p50=percentile(lat, 50), p99=percentile(lat, 99),
                rss_mb=rss / 1024, size_mb=size_mb,
                open_ms=(time.time() - t_open) * 1000)


def bench_faiss(args, queries, gt):
    import faiss
    cache = os.path.join(args.cache_dir, 'faiss.ivfpq')
    base = read_fvecs(args.base, args.n_docs, args.dim)
    nlist = int(np.sqrt(args.n_docs))
    if not os.path.exists(cache):
        quantizer = faiss.IndexFlatL2(args.dim)
        index = faiss.IndexIVFPQ(quantizer, args.dim, nlist, 8, 8)
        train_n = min(args.n_docs, 100_000)
        sys.stderr.write(f'  faiss train ({train_n}) ...\n')
        index.train(base[:train_n])
        sys.stderr.write(f'  faiss add ({args.n_docs}) ...\n')
        index.add(base)
        faiss.write_index(index, cache)
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    index = faiss.read_index(cache)
    index.nprobe = 16
    # warmup
    index.search(queries[:5], args.top_k)
    res = []
    lat = []
    for q in queries:
        t0 = time.time()
        _, I = index.search(q.reshape(1, -1), args.top_k)
        lat.append((time.time() - t0) * 1000)
        res.append(I[0])
    rcl = compute_recall(res, gt, args.top_k)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(name='faiss-ivfpq', recall=rcl,
                p50=percentile(lat, 50), p99=percentile(lat, 99),
                rss_mb=rss / 1024, size_mb=os.path.getsize(cache) / 1024**2,
                open_ms=0.0)


def bench_hnswlib(args, queries, gt):
    import hnswlib
    cache = os.path.join(args.cache_dir, 'hnsw.bin')
    if not os.path.exists(cache):
        base = read_fvecs(args.base, args.n_docs, args.dim)
        idx = hnswlib.Index(space='l2', dim=args.dim)
        idx.init_index(max_elements=args.n_docs, M=16, ef_construction=200)
        sys.stderr.write(f'  hnswlib add ({args.n_docs}) ...\n')
        idx.add_items(base, np.arange(args.n_docs))
        idx.save_index(cache)
    idx = hnswlib.Index(space='l2', dim=args.dim)
    idx.load_index(cache)
    idx.set_ef(50)
    # warmup
    idx.knn_query(queries[:5], k=args.top_k)
    res = []
    lat = []
    for q in queries:
        t0 = time.time()
        labels, _ = idx.knn_query(q, k=args.top_k)
        lat.append((time.time() - t0) * 1000)
        res.append(labels[0])
    rcl = compute_recall(res, gt, args.top_k)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(name='hnswlib', recall=rcl,
                p50=percentile(lat, 50), p99=percentile(lat, 99),
                rss_mb=rss / 1024, size_mb=os.path.getsize(cache) / 1024**2,
                open_ms=0.0)


def _dir_size_mb(d: str) -> float:
    total = 0
    for root, _, files in os.walk(d):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(root, fn))
            except OSError:
                pass
    return total / 1024**2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--base',    required=True)
    ap.add_argument('--queries', required=True)
    ap.add_argument('--gt',      required=True)
    ap.add_argument('--n_docs',  type=int, required=True)
    ap.add_argument('--dim',     type=int, required=True)
    ap.add_argument('--mangrove_index', required=True)
    ap.add_argument('--cache_dir', default='/tmp/competitive_cache')
    ap.add_argument('--n_trees', type=int, default=200)
    ap.add_argument('--depth',   type=int, default=20)
    ap.add_argument('--sub_dim', type=int, default=16)
    ap.add_argument('--gen',     type=int, default=3)
    ap.add_argument('--n_queries', type=int, default=200)
    ap.add_argument('--top_k',   type=int, default=10)
    ap.add_argument('--top_n',   type=int, default=4000)
    ap.add_argument('--engines', default='mangrove,faiss,hnswlib')
    args = ap.parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)

    queries = read_fvecs(args.queries, args.n_queries, args.dim)
    gt      = read_ivecs_gt(args.gt, args.n_queries, args.top_k)
    engines = args.engines.split(',')

    results = []
    for eng in engines:
        eng = eng.strip()
        sys.stderr.write(f'\n=== {eng} ===\n')
        try:
            if eng == 'mangrove':
                results.append(bench_mangrove(args, queries, gt))
            elif eng == 'faiss':
                results.append(bench_faiss(args, queries, gt))
            elif eng == 'hnswlib':
                results.append(bench_hnswlib(args, queries, gt))
        except Exception as e:
            sys.stderr.write(f'  {eng} failed: {e}\n')

    print(f'\n{"engine":>15}  {"recall@k":>9}  {"p50_ms":>7}  {"p99_ms":>7}  '
          f'{"RSS_MB":>8}  {"size_MB":>9}')
    print('-' * 70)
    for r in results:
        print(f'{r["name"]:>15}  {r["recall"]:>9.4f}  '
              f'{r["p50"]:>7.2f}  {r["p99"]:>7.2f}  '
              f'{r["rss_mb"]:>8.1f}  {r["size_mb"]:>9.1f}')


if __name__ == '__main__':
    main()
