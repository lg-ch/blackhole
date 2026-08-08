"""Download Cohere wikipedia-2023-11-embed-multilingual-v3 (English subset)
and convert to .fvecs for mangrove build.

41.5M docs × dim 1024 from 415 parquet files (~86 GB compressed).
"""
from __future__ import annotations

import argparse, os, struct, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq


REPO = 'Cohere/wikipedia-2023-11-embed-multilingual-v3'


def download_one(idx: int, cache_dir: str) -> tuple[int, str, float]:
    fname = f'en/{idx:04d}.parquet'
    t0 = time.time()
    path = hf_hub_download(repo_id=REPO, filename=fname, repo_type='dataset',
                           cache_dir=cache_dir)
    return idx, path, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True,
                    help='Output .fvecs file (will be created)')
    ap.add_argument('--cache_dir',
                    default='/mnt/mangrove/datasets/cohere_cache')
    ap.add_argument('--n_files', type=int, default=415,
                    help='How many English parquet files to fetch (max 415)')
    ap.add_argument('--workers', type=int, default=6,
                    help='Parallel download workers')
    args = ap.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # 1. Download files in parallel.
    print(f'== downloading {args.n_files} English files (workers={args.workers}) ==')
    paths: dict[int, str] = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download_one, i, args.cache_dir): i
                for i in range(args.n_files)}
        done = 0
        for fut in as_completed(futs):
            try:
                idx, path, dt = fut.result()
                paths[idx] = path
                done += 1
                if done % 10 == 0 or done == args.n_files:
                    sys.stderr.write(
                        f'  [dl] {done}/{args.n_files} files, '
                        f'{(time.time()-t0)/60:.1f} min elapsed\n')
            except Exception as e:
                sys.stderr.write(f'  ERR file {futs[fut]}: {e}\n')

    print(f'\n== download done in {(time.time()-t0)/60:.1f} min ==')

    # 2. Convert parquet → fvecs (vectorized, no per-row Python loop).
    print('== converting parquet -> fvecs ==')
    import numpy as np
    t0 = time.time()
    n_total = 0
    with open(args.out, 'wb') as out:
        for idx in sorted(paths.keys()):
            pf = pq.ParquetFile(paths[idx])
            for batch in pf.iter_batches(batch_size=20000, columns=['emb']):
                emb_col = batch.column('emb')
                # Fast path: pyarrow ListArray of FixedSizeList<float> has
                # contiguous float values in `flat_values()`. Reshape directly.
                values = emb_col.values.to_numpy(zero_copy_only=False)
                n = len(emb_col)
                dim = len(values) // n
                arr = values.reshape(n, dim).astype(np.float32, copy=False)
                # fvecs: per row [int32 dim][dim float32], packed via uint32
                # view since dim header (4B) fits where one float would.
                out_arr = np.empty((n, dim + 1), dtype=np.uint32)
                out_arr[:, 0] = dim
                out_arr[:, 1:] = arr.view(np.uint32).reshape(n, dim)
                out.write(out_arr.tobytes())
                n_total += n
            if (idx + 1) % 10 == 0 or idx == args.n_files - 1:
                sys.stderr.write(
                    f'  [conv] file {idx}: {n_total:>10,} rows, '
                    f'{(time.time()-t0)/60:.1f} min\n')
    print(f'\n== converted {n_total} rows to {args.out} in '
          f'{(time.time()-t0)/60:.1f} min ==')
    print(f'   file size: {os.path.getsize(args.out)/1024**3:.2f} GB')


if __name__ == '__main__':
    main()
