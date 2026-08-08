"""Compare filter_mode='pre' vs filter_mode='post' on a live server.

Uses `allowed_bitmap` (no ClickHouse required). The bitmap is synthesized
locally with a target density to mimic a real metadata filter.

Reports per-mode :
  - p50 / p95 / p99 latency
  - QPS
  - top_k overlap between pre and post (sanity : same vec → same results)

Usage :
    python3 scripts/bench_filter_mode.py \\
        --url http://localhost:8000 --name sift1m \\
        --queries-file sift/sift_query.fvecs --dim 128 \\
        --densities 0.005 0.02 0.05 0.20 --n_queries 200
"""
from __future__ import annotations
import argparse, struct, sys, time, random
import numpy as np
import mangrove as mg


def read_fvecs(path, n, d):
    out = np.empty((n, d), dtype=np.float32)
    with open(path, 'rb') as f:
        for i in range(n):
            f.read(4)
            out[i] = np.frombuffer(f.read(d * 4), dtype=np.float32)
    return out


def make_bitmap(n_docs: int, density: float, seed: int = 0) -> bytes:
    """Random subset of [0..n_docs) at the requested density, serialized
       as portable CRoaring bytes."""
    from pyroaring import BitMap
    rng = np.random.default_rng(seed)
    keep = rng.random(n_docs) < density
    ids = np.flatnonzero(keep).tolist()
    return BitMap(ids).serialize()


def measure(client, name, queries, bitmap, mode, top_k):
    lats, results = [], []
    for q in queries:
        t0 = time.time()
        r = client.search(q.tolist(), name=name, top_k=top_k,
                          allowed_bitmap=bitmap, filter_mode=mode)
        lats.append((time.time() - t0) * 1000)
        results.append(tuple(r['ids']))
    return lats, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://localhost:8000')
    ap.add_argument('--name', required=True)
    ap.add_argument('--queries-file', required=True)
    ap.add_argument('--dim', type=int, required=True)
    ap.add_argument('--n_queries', type=int, default=100)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--densities', type=float, nargs='+',
                    default=[0.005, 0.02, 0.05, 0.20])
    ap.add_argument('--api_key', default=None)
    args = ap.parse_args()

    client = mg.Client(args.url, api_key=args.api_key)
    try:
        st = client.stats(name=args.name)
    except Exception as e:
        print(f'ERROR : cannot reach {args.url} / index {args.name} : {e}',
              file=sys.stderr)
        sys.exit(2)
    n_docs = st.get('total_docs') or st.get('next_doc_id') or 1_000_000
    print(f'Server OK : index={args.name}  n_docs={n_docs:,}  '
          f'dim={args.dim}  top_k={args.top_k}')

    queries = read_fvecs(args.queries_file, args.n_queries, args.dim)
    # warm
    for q in queries[:3]:
        client.search(q.tolist(), name=args.name, top_k=args.top_k)

    print(f'\n{"density":>8} {"mode":>5} {"qps":>7} '
          f'{"p50":>7} {"p95":>7} {"p99":>7} {"overlap":>8}')
    print('-' * 60)
    for d in args.densities:
        bm = make_bitmap(n_docs, d, seed=42)
        t0 = time.time()
        lats_pre,  ids_pre  = measure(client, args.name, queries, bm, 'pre',  args.top_k)
        elapsed_pre = time.time() - t0
        t0 = time.time()
        lats_post, ids_post = measure(client, args.name, queries, bm, 'post', args.top_k)
        elapsed_post = time.time() - t0

        # set-overlap of returned ids at each query (averaged)
        overlap = np.mean([
            len(set(a) & set(b)) / max(1, len(set(a) | set(b)))
            for a, b in zip(ids_pre, ids_post)
        ])

        print(f'{d:>8.3f} {"pre":>5} {len(queries)/elapsed_pre:>7.1f} '
              f'{np.percentile(lats_pre, 50):>7.1f} '
              f'{np.percentile(lats_pre, 95):>7.1f} '
              f'{np.percentile(lats_pre, 99):>7.1f} '
              f'{overlap:>8.3f}')
        print(f'{d:>8.3f} {"post":>5} {len(queries)/elapsed_post:>7.1f} '
              f'{np.percentile(lats_post, 50):>7.1f} '
              f'{np.percentile(lats_post, 95):>7.1f} '
              f'{np.percentile(lats_post, 99):>7.1f} '
              f'{"":>8}')


if __name__ == '__main__':
    main()
