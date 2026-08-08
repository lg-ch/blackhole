"""Shared benchmark utilities for mangrove-search.

Single source of truth for the measurement protocol used to populate
BENCHMARKS.md. All per-dataset scripts (bench/run_*.py) call into this
to ensure cold/warm, drop_caches, metrics are computed identically.
"""
import os, struct, sys, time, json, subprocess, resource
import numpy as np

sys.path.insert(0, '/home/chatelet/mangrove-search/scripts')
import mangrove_ffi as mf
from mangrove_ffi import Forest


def drop_caches() -> None:
    """Drop OS page cache + dentries + inodes. Requires root."""
    os.sync()
    with open('/proc/sys/vm/drop_caches', 'w') as fh:
        fh.write('3')


def read_fvecs(path: str, n: int, dim: int) -> np.ndarray:
    """Per-row [int32 dim][dim × float32]. Reads up to n rows."""
    out = np.empty((n, dim), np.float32)
    with open(path, 'rb') as fp:
        for i in range(n):
            fp.read(4)  # skip per-row dim header
            out[i] = np.frombuffer(fp.read(dim * 4), np.float32)
    return out


def read_ivecs(path: str, n: int, k: int) -> np.ndarray:
    """Per-row [int32 k][k × int32]. Truncate to first k columns."""
    out = np.empty((n, k), np.int32)
    with open(path, 'rb') as fp:
        first_k = struct.unpack('<i', fp.read(4))[0]
        fp.seek(0)
        for i in range(n):
            fp.read(4)
            out[i] = np.frombuffer(fp.read(k * 4), np.int32)[:k]
            fp.read(4 * (first_k - k))
    return out


def read_bvecs_to_f32(path: str, n: int, dim: int) -> np.ndarray:
    """SIFT-style uint8 per-row [int32 dim][dim × uint8] → float32."""
    out = np.empty((n, dim), np.float32)
    with open(path, 'rb') as fp:
        for i in range(n):
            fp.read(4)
            out[i] = np.frombuffer(fp.read(dim), np.uint8).astype(np.float32)
    return out


def read_fbin_floats(path: str, n: int) -> np.ndarray:
    """big-ann-benchmarks fbin: 8 B [uint32 n][uint32 dim] header + raw f32."""
    with open(path, 'rb') as fp:
        ntot, d = struct.unpack('<II', fp.read(8))
        n = min(n, ntot)
        return np.frombuffer(fp.read(n * d * 4), np.float32).reshape(n, d).copy()


def read_fbin_ids(path: str, n: int, k: int) -> np.ndarray:
    """big-ann-benchmarks GT bin: 8 B header + per-query top-k int32."""
    with open(path, 'rb') as fp:
        ntot, kf = struct.unpack('<II', fp.read(8))
        n = min(n, ntot)
        return np.frombuffer(fp.read(n * kf * 4), np.int32).reshape(n, kf)[:, :k].copy()


def peak_rss_mb() -> float:
    """Self peak RSS in MB (resource.RUSAGE_SELF, KB on Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def disk_size_mb(path: str) -> float:
    out = subprocess.run(['du', '-sb', path], capture_output=True, text=True)
    return int(out.stdout.split()[0]) / (1024.0 * 1024.0)


def bench_config(forest, base_path, tq1_path,
                 queries, gt, np_, qd, tn, kprime,
                 n_warmup=5, n_warm=100):
    """Measure (recall@10, p50_warm, p95_warm, cold_q0) for one config.

    Protocol:
      1. drop_caches
      2. n_warmup queries to populate page cache
      3. n_warm timed queries for warm percentiles
      4. drop_caches again
      5. 1 cold query for cold-q0 latency
    """
    nq = min(len(queries), n_warm + n_warmup)
    drop_caches()
    for i in range(n_warmup):
        q = queries[i % len(queries)]
        ids, _, n = forest.query_probes(q, np_, top_n=tn, probe_depth=qd)
        cand = np.asarray(ids[:n], dtype=np.int32); cand = cand[cand >= 0]
        if tq1_path:
            forest.rerank_tq1(tq1_path, base_path, q, cand,
                              kprime=kprime, top_k=10)
        else:
            forest.rerank_l2(base_path, q, cand, top_k=10)

    recs, lats = [], []
    base_qi = n_warmup
    for qi in range(base_qi, base_qi + n_warm):
        q = queries[qi % len(queries)]; ref = set(int(x) for x in gt[qi % len(gt)])
        t0 = time.time()
        ids, _, n = forest.query_probes(q, np_, top_n=tn, probe_depth=qd)
        cand = np.asarray(ids[:n], dtype=np.int32); cand = cand[cand >= 0]
        if tq1_path:
            top = forest.rerank_tq1(tq1_path, base_path, q, cand,
                                    kprime=kprime, top_k=10)
        else:
            top = forest.rerank_l2(base_path, q, cand, top_k=10)
        lats.append((time.time() - t0) * 1000.0)
        recs.append(sum(1 for x in top if int(x) in ref) / 10.0)

    drop_caches()
    qcold = queries[base_qi % len(queries)]
    t0 = time.time()
    ids, _, n = forest.query_probes(qcold, np_, top_n=tn, probe_depth=qd)
    cand = np.asarray(ids[:n], dtype=np.int32); cand = cand[cand >= 0]
    if tq1_path:
        forest.rerank_tq1(tq1_path, base_path, qcold, cand,
                          kprime=kprime, top_k=10)
    else:
        forest.rerank_l2(base_path, qcold, cand, top_k=10)
    cold_q0 = (time.time() - t0) * 1000.0

    return {
        'recall': float(np.mean(recs)),
        'p50_warm_ms': float(np.percentile(lats, 50)),
        'p95_warm_ms': float(np.percentile(lats, 95)),
        'cold_q0_ms':  float(cold_q0),
        'n_warm':      n_warm,
    }


def print_row(label, res, extra=''):
    print(f'  {label:>20} | rec {res["recall"]:.4f} | '
          f'p50 warm {res["p50_warm_ms"]:6.1f} ms | '
          f'p95 warm {res["p95_warm_ms"]:6.1f} ms | '
          f'cold q0 {res["cold_q0_ms"]:6.1f} ms {extra}', flush=True)


def save_json(out_path, payload):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as fh:
        json.dump(payload, fh, indent=2)
    print(f'  → results saved to {out_path}', flush=True)
