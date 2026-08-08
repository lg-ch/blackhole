"""Concurrency stress test : measure QPS with N parallel HTTP queries
against serve_cluster. Identifies the throughput ceiling per pod.

Usage : (serve_cluster must be running on --port with an index ready)

  python3 scripts/bench_concurrency.py \
      --url http://localhost:8000 --name sift1m \
      --dim 128 --queries-file sift/sift_query.fvecs \
      --workers 1 2 4 8 16 32 --duration 30
"""
from __future__ import annotations
import argparse, struct, sys, time
import threading, queue
import numpy as np
import mangrove as mg


def read_fvecs(path, n, d):
    out = np.empty((n, d), dtype=np.float32)
    with open(path, 'rb') as f:
        for i in range(n):
            f.read(4)
            out[i] = np.frombuffer(f.read(d * 4), dtype=np.float32)
    return out


def run_workers(client: mg.Client, name: str, queries: np.ndarray,
                n_workers: int, duration: float) -> tuple[int, list[float]]:
    """Run n_workers threads issuing queries for `duration` seconds.
       Returns (total_queries, latencies_ms)."""
    stop = threading.Event()
    lats: list[float] = []
    lats_lock = threading.Lock()
    n_q = len(queries)

    def worker(wid: int) -> None:
        local_lats = []
        i = wid
        while not stop.is_set():
            q = queries[i % n_q]
            t0 = time.time()
            try:
                client.search(q.tolist(), name=name, top_k=10)
                local_lats.append((time.time() - t0) * 1000)
            except Exception:
                pass
            i += n_workers
        with lats_lock:
            lats.extend(local_lats)

    threads = [threading.Thread(target=worker, args=(w,), daemon=True)
               for w in range(n_workers)]
    t0 = time.time()
    for t in threads: t.start()
    time.sleep(duration)
    stop.set()
    for t in threads: t.join(timeout=30)
    elapsed = time.time() - t0
    return len(lats), lats, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url',           default='http://localhost:8000')
    ap.add_argument('--name',          required=True)
    ap.add_argument('--queries-file',  required=True)
    ap.add_argument('--dim',           type=int, required=True)
    ap.add_argument('--n_queries',     type=int, default=200)
    ap.add_argument('--workers',       type=int, nargs='+',
                    default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument('--duration',      type=float, default=15.0)
    ap.add_argument('--api_key',       default=None)
    args = ap.parse_args()

    client = mg.Client(args.url, api_key=args.api_key, pool_size=128)
    queries = read_fvecs(args.queries_file, args.n_queries, args.dim)

    # warm cache
    for q in queries[:5]:
        client.search(q.tolist(), name=args.name, top_k=10)

    print(f'\n{"workers":>8} {"queries":>10} {"qps":>10} {"p50ms":>8} '
          f'{"p95ms":>8} {"p99ms":>8} {"avg ms":>8}')
    print('-' * 64)
    for w in args.workers:
        n, lats, elapsed = run_workers(client, args.name, queries, w, args.duration)
        qps = n / elapsed
        if not lats:
            print(f'{w:>8} {n:>10} {qps:>10.1f}    (no successful queries)')
            continue
        print(f'{w:>8} {n:>10} {qps:>10.1f} '
              f'{np.percentile(lats, 50):>8.1f} {np.percentile(lats, 95):>8.1f} '
              f'{np.percentile(lats, 99):>8.1f} {np.mean(lats):>8.1f}')


if __name__ == '__main__':
    main()
