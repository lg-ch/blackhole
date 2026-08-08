#!/usr/bin/env python3
"""
Stream les parquets arxiv → :
  - arxiv_base.fvecs (768f par doc, ~5.9 GB)
  - metadata directement insérée dans mangrove.docs via clickhouse-driver
  - pré-agrégation finale par year / primary_cat / top_cat
internal_id = position séquentielle (0..N-1) dans l'ordre 2007..2024 puis row order intra-parquet.
"""
import os, sys, time, struct, glob
import numpy as np
import pyarrow.parquet as pq
from clickhouse_driver import Client

ROOT = '/home/chatelet/mangrove-search'
PARQUET_DIR = f'{ROOT}/datasets/arxiv'
FVECS_OUT = f'{ROOT}/datasets/arxiv/arxiv_base.fvecs'
DIM = 768
BATCH = 50_000

ch = Client('127.0.0.1', settings={'max_block_size': 200_000})

def parse_year(y):
    if not y: return 0
    try:
        v = int(y)
        return 2000 + v if v < 100 else v
    except Exception:
        return 0

def primary_and_top(cats):
    if not cats: return ('', '')
    p = cats.split()[0]
    t = p.split('.')[0] if '.' in p else p
    return (p, t)

def main():
    files = sorted(glob.glob(f'{PARQUET_DIR}/*.parquet'))
    print(f'[prepare] {len(files)} parquets → {FVECS_OUT}', flush=True)

    fout = open(FVECS_OUT, 'wb')
    dim_hdr = struct.pack('<i', DIM)
    internal_id = 0
    rows_buf = []
    t0 = time.time()

    def flush_meta():
        if not rows_buf: return
        ch.execute(
            'INSERT INTO mangrove.docs (internal_id,arxiv_id,year,primary_cat,top_cat) VALUES',
            rows_buf, types_check=False
        )
        rows_buf.clear()

    for path in files:
        pf = pq.ParquetFile(path)
        n = pf.metadata.num_rows
        print(f'[{os.path.basename(path)}] {n} rows', flush=True)
        cols = ['id', 'year', 'categories', 'abstract_embedding']
        for batch in pf.iter_batches(batch_size=10_000, columns=cols):
            ids   = batch.column('id').to_pylist()
            years = batch.column('year').to_pylist()
            cats  = batch.column('categories').to_pylist()
            embs  = batch.column('abstract_embedding')  # ListArray
            # Convert each embedding to float32 → fvecs entry
            for i in range(len(ids)):
                v = embs[i].as_py()
                if v is None or len(v) != DIM:
                    continue
                arr = np.asarray(v, dtype=np.float32)
                fout.write(dim_hdr)
                fout.write(arr.tobytes())
                yr = parse_year(years[i])
                pc, tc = primary_and_top(cats[i])
                rows_buf.append((internal_id, ids[i] or '', yr, pc, tc))
                internal_id += 1
                if len(rows_buf) >= BATCH:
                    flush_meta()
        elapsed = time.time() - t0
        print(f'  → total internal_id={internal_id} elapsed={elapsed:.1f}s '
              f'({internal_id/max(1,elapsed):.0f} rows/s)', flush=True)

    flush_meta()
    fout.close()
    print(f'[prepare] DONE n_docs={internal_id} time={time.time()-t0:.1f}s', flush=True)

    # Pré-agrégation des bitmaps
    print('[preagg] building bitmap pre-aggs …', flush=True)
    for col in ['year', 'primary_cat', 'top_cat']:
        ch.execute(f'TRUNCATE TABLE mangrove.bm_by_{col}')
        ch.execute(
            f'INSERT INTO mangrove.bm_by_{col} '
            f'SELECT {col}, groupBitmapState(internal_id) FROM mangrove.docs GROUP BY {col}'
        )
        n = ch.execute(f'SELECT count() FROM mangrove.bm_by_{col}')[0][0]
        print(f'  bm_by_{col}: {n} groups', flush=True)

if __name__ == '__main__':
    main()
