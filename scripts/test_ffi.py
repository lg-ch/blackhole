"""Quick FFI smoke test: open SIFT 1M index, run a few queries, compare timings."""
import os, struct, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mangrove_ffi import Forest


def read_fvecs(path: str, n: int = 10) -> list[np.ndarray]:
    out = []
    with open(path, 'rb') as f:
        for _ in range(n):
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack('<i', dim_bytes)[0]
            vec = np.frombuffer(f.read(dim * 4), dtype=np.float32).copy()
            out.append(vec)
    return out


def main() -> None:
    INDEX = '/tmp/srt3_test'
    N_TREES = 20
    DEPTH = 12
    N_DOCS = 1_000_000

    if not os.path.exists(INDEX):
        print(f'index not found: {INDEX} (run small build first)')
        return

    queries = read_fvecs('sift/sift_query.fvecs', 50)
    print(f'loaded {len(queries)} query vectors')

    f = Forest(INDEX, n_trees=N_TREES, dim=128, sub_dim=0,
               depth=DEPTH, n_docs=N_DOCS, gen_version=0)
    print(f'opened forest: n_trees={f.n_trees} dim={f.dim} depth={f.depth}')

    # Warm-up
    for q in queries[:5]:
        f.query(q, top_n=500)

    t0 = time.time()
    n_calls = 0
    for q in queries:
        ids, votes, n = f.query(q, top_n=500)
        n_calls += 1
    dt = time.time() - t0
    print(f'FFI: {n_calls} queries in {dt*1000:.1f} ms '
          f'({dt*1000/n_calls:.2f} ms/q, n_distinct last={f.n_distinct()})')

    # Subprocess baseline (one full run)
    import subprocess
    t0 = time.time()
    r = subprocess.run(
        ['./rpforest', 'topn', 'sift/sift_query.fvecs',
         'sift/sift_groundtruth.ivecs', 'sift/sift_base.fvecs',
         INDEX, str(N_TREES), str(DEPTH), str(N_DOCS),
         '10', '500', str(len(queries))],
        capture_output=True, text=True
    )
    sub_dt = time.time() - t0
    print(f'subprocess: {sub_dt*1000:.1f} ms wall ({sub_dt*1000/len(queries):.2f} ms/q amortized)')
    f.close()


if __name__ == '__main__':
    main()
