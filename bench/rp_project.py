"""Random projection of an fvecs file from `dim_in` to `dim_out`.

Uses a deterministic seed-derived sparse Achlioptas matrix
(3 levels {-1, 0, +1} with probabilities {1/6, 2/3, 1/6}) so the
projection is reproducible and stateless (no codebook to ship — just
the seed).

Usage:
    python3 bench/rp_project.py <input.fvecs> <output.fvecs> \
        <dim_in> <dim_out> <seed>

Writes the projection matrix bytes to <output>.matrix.fvecs (one fvec
row per output dim, for later projection of queries).
"""
import os, sys, struct
import numpy as np


def achlioptas_matrix(dim_in: int, dim_out: int, seed: int) -> np.ndarray:
    """Sparse Achlioptas projection matrix, shape (dim_in, dim_out).

    Each entry is independently:
      +sqrt(3) with prob 1/6
      0         with prob 2/3
      -sqrt(3)  with prob 1/6
    Variance is 1, so the projection preserves squared L2 norm in
    expectation. Stored as float32 for fast matmul.
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 1.0, size=(dim_in, dim_out))
    M = np.zeros((dim_in, dim_out), dtype=np.float32)
    s = np.float32(np.sqrt(3.0))
    M[u < 1.0 / 6.0]  =  s
    M[u > 5.0 / 6.0]  = -s
    # scale by 1/sqrt(dim_out) so projected L2 distances are comparable
    return M / np.float32(np.sqrt(dim_out))


def project_fvecs(in_path: str, out_path: str,
                  dim_in: int, dim_out: int, M: np.ndarray,
                  chunk: int = 8192) -> int:
    """Stream-read in_path (.fvecs), project rows through M, stream-write out_path."""
    n_rows = os.path.getsize(in_path) // (4 + dim_in * 4)
    print(f'  input  : {in_path}  ({n_rows:,} rows × {dim_in} dim)', flush=True)
    print(f'  output : {out_path} ({n_rows:,} rows × {dim_out} dim)', flush=True)

    # Pre-cast row dim header (the prefix int32 dim before every fvec row)
    row_hdr_out = np.int32(dim_out).tobytes()

    with open(in_path, 'rb') as fin, open(out_path, 'wb') as fout:
        done = 0
        while done < n_rows:
            n = min(chunk, n_rows - done)
            # Read chunk : each row = 4 B dim header + dim_in × 4 B floats
            raw = fin.read(n * (4 + dim_in * 4))
            arr = np.frombuffer(raw, dtype=np.float32).reshape(n, 1 + dim_in)
            vecs = arr[:, 1:]                          # strip per-row dim header
            proj = (vecs @ M).astype(np.float32)       # (n, dim_out)
            # Pack with per-row dim header
            for i in range(n):
                fout.write(row_hdr_out)
                fout.write(proj[i].tobytes())
            done += n
            if done % 200_000 < chunk:
                print(f'    projected {done:>9} / {n_rows}', flush=True)
    return n_rows


def main():
    if len(sys.argv) != 6:
        print(__doc__); sys.exit(1)
    in_path, out_path = sys.argv[1], sys.argv[2]
    dim_in, dim_out = int(sys.argv[3]), int(sys.argv[4])
    seed = int(sys.argv[5])

    print(f'Achlioptas RP : dim {dim_in} → {dim_out}  (seed={seed})')
    M = achlioptas_matrix(dim_in, dim_out, seed)
    nz = int(np.count_nonzero(M))
    print(f'  matrix    : ({dim_in} × {dim_out}), nonzeros = {nz:,} '
          f'({nz/(dim_in*dim_out)*100:.1f} %)')

    # Save the matrix so queries can be projected reproducibly later.
    M_path = out_path + '.matrix.npy'
    np.save(M_path, M)
    print(f'  matrix → {M_path}  ({os.path.getsize(M_path)/1e6:.1f} MB)')

    import time
    t0 = time.time()
    n = project_fvecs(in_path, out_path, dim_in, dim_out, M)
    dt = time.time() - t0
    print(f'  done {n:,} rows in {dt:.1f} s '
          f'({n/dt/1000:.1f}k vec/s).  out = '
          f'{os.path.getsize(out_path)/1e9:.2f} GB')


if __name__ == '__main__':
    main()
