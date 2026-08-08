"""Bulk-import vectors from external sources into a mangrove index.

Supported sources :
   --from fvecs   <path>    : standard .fvecs (int32 dim header + floats)
   --from npy     <path>    : numpy .npy of shape (N, dim) float32
   --from parquet <path>    : Parquet column 'emb' of fixed-size float list
   --from faiss   <path>    : FAISS IndexFlatL2/IndexFlatIP/IndexHNSW serialized
   --from s3      <bucket/prefix> : S3 of .npy or .fvecs (uses boto3)

Streams batches of `--batch_size` vectors into LiveIndex.insert(), then
freezes at the end so all docs end up in segments.
"""
from __future__ import annotations
import argparse, os, struct, sys, time
from typing import Iterable

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from live_index import LiveIndex


# ---- source readers -------------------------------------------------

def from_fvecs(path: str, dim: int) -> Iterable[np.ndarray]:
    row_bytes = 4 + dim * 4
    with open(path, 'rb') as f:
        while True:
            hdr = f.read(4)
            if len(hdr) < 4: return
            d = struct.unpack('<i', hdr)[0]
            buf = f.read(d * 4)
            if len(buf) < d * 4: return
            yield np.frombuffer(buf, dtype=np.float32)


def from_npy(path: str, dim: int) -> Iterable[np.ndarray]:
    arr = np.load(path, mmap_mode='r')
    assert arr.shape[1] == dim, f'npy dim {arr.shape[1]} != expected {dim}'
    for v in arr:
        yield np.asarray(v, dtype=np.float32)


def from_parquet(path: str, dim: int) -> Iterable[np.ndarray]:
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=20000, columns=['emb']):
        col = batch.column('emb')
        values = col.values.to_numpy(zero_copy_only=False)
        n = len(col)
        arr = values.reshape(n, -1).astype(np.float32, copy=False)
        for row in arr:
            yield row


def from_faiss(path: str, dim: int) -> Iterable[np.ndarray]:
    import faiss
    idx = faiss.read_index(path)
    n = idx.ntotal
    sys.stderr.write(f'  faiss : {n} vectors\n')
    for i in range(n):
        # reconstruct each (only Flat/HNSW expose this)
        v = idx.reconstruct(int(i))
        yield np.asarray(v, dtype=np.float32)


def from_s3(uri: str, dim: int) -> Iterable[np.ndarray]:
    """s3://bucket/prefix.npy or .fvecs — downloads to tempfile then streams."""
    import boto3, tempfile
    if not uri.startswith('s3://'):
        sys.exit('expected s3://bucket/key')
    bucket, key = uri[5:].split('/', 1)
    s3 = boto3.client('s3')
    suffix = os.path.splitext(key)[1] or '.bin'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as t:
        s3.download_fileobj(bucket, key, t)
        local = t.name
    if local.endswith('.npy'):
        yield from from_npy(local, dim)
    elif local.endswith('.fvecs') or local.endswith('.bin'):
        yield from from_fvecs(local, dim)
    else:
        sys.exit(f'unsupported s3 key suffix : {local}')


READERS = {
    'fvecs':   from_fvecs,
    'npy':     from_npy,
    'parquet': from_parquet,
    'faiss':   from_faiss,
    's3':      from_s3,
}


# ---- main -----------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root',  required=True, help='LiveIndex dir')
    ap.add_argument('--from',  required=True, dest='source',
                    choices=list(READERS.keys()))
    ap.add_argument('--path',  required=True, help='source file/URI')
    ap.add_argument('--dim',   type=int, required=True)
    ap.add_argument('--sub_dim', type=int, default=16)
    ap.add_argument('--n_trees', type=int, default=1000)
    ap.add_argument('--depth',   type=int, default=20)
    ap.add_argument('--gen',     type=int, default=3)
    ap.add_argument('--batch_size', type=int, default=10_000,
                    help='Freeze after this many vectors (governs segment size)')
    ap.add_argument('--no_freeze_end', action='store_true',
                    help="Don't freeze pending active at the end")
    args = ap.parse_args()

    if os.path.exists(os.path.join(args.root, 'manifest.json')):
        sys.stderr.write(f'  opening existing index at {args.root}\n')
        li = LiveIndex.open(args.root)
    else:
        sys.stderr.write(f'  creating index at {args.root}\n')
        li = LiveIndex.create(args.root, dim=args.dim, sub_dim=args.sub_dim,
                              n_trees=args.n_trees, depth=args.depth,
                              gen_version=args.gen)

    reader = READERS[args.source]
    t0 = time.time()
    n_total = 0
    for v in reader(args.path, args.dim):
        li.insert(v)
        n_total += 1
        if n_total % args.batch_size == 0:
            sys.stderr.write(
                f'  [{n_total:>10}] freezing batch in '
                f'{(time.time() - t0):.1f}s ...\n')
            li.freeze()

    if not args.no_freeze_end and li.active_size() > 0:
        sys.stderr.write(f'  final freeze ({li.active_size()} pending)\n')
        li.freeze()

    sys.stderr.write(
        f'DONE — {n_total:,} docs in {(time.time() - t0)/60:.1f} min, '
        f'{len(li.manifest["segments"])} segments\n')
    li.close()


if __name__ == '__main__':
    main()
