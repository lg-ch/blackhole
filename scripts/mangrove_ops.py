"""Operational CLI : export an index's vec store, copy an index.

These are heavyweight ops (read all of vecs.fvecs) intended for ops /
data-migration, not user-facing query paths.

Usage :
    python3 scripts/mangrove_ops.py export <root> <name> <out.fvecs>
    python3 scripts/mangrove_ops.py copy   <root> <src> <dst>
"""
from __future__ import annotations
import argparse, os, shutil, struct, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from live_index import LiveIndex
from registry import MangroveCluster


def cmd_export(args):
    cl = MangroveCluster(args.cluster_root)
    li = cl.get(args.name)
    src = li._base_path_abs
    if not os.path.exists(src):
        sys.exit(f'no vecs.fvecs at {src} — index has no auto-store')
    print(f'  copying {src} → {args.out} ({os.path.getsize(src)/1e9:.2f} GB)')
    shutil.copyfile(src, args.out)
    print(f'  done. {li.manifest["next_doc_id"]} doc rows, dim {li.manifest["dim"]}')


def cmd_copy(args):
    """Clone an index by re-ingesting its vectors under a new name.
       Uses the source's vecs.fvecs as the input. Server-side LSM
       rebuild — this is slow."""
    cl = MangroveCluster(args.cluster_root)
    src = cl.get(args.src)
    if args.dst in cl.list_indexes():
        sys.exit(f'destination {args.dst!r} already exists')

    src_vecs = src._base_path_abs
    if not os.path.exists(src_vecs):
        sys.exit('source index has no auto-store ; cannot copy')

    print(f'  source : {args.src}  dim={src.manifest["dim"]} '
          f'n_docs={src.manifest["next_doc_id"]:,}')
    print(f'  destination : {args.dst}')

    dst = cl.create_index(args.dst,
                          dim         = src.manifest['dim'],
                          sub_dim     = src.manifest['sub_dim'],
                          n_trees     = src.manifest['n_trees'],
                          depth       = src.manifest['depth'],
                          gen_version = src.manifest['gen_version'])

    import numpy as np
    dim = src.manifest['dim']
    row_bytes = 4 + dim * 4
    n_done = 0
    batch = []
    with open(src_vecs, 'rb') as f:
        while True:
            buf = f.read(row_bytes)
            if len(buf) < row_bytes: break
            v = np.frombuffer(buf[4:], dtype=np.float32)
            batch.append(v)
            if len(batch) >= 5000:
                for v in batch:
                    dst.insert(v)
                n_done += len(batch); batch.clear()
                if n_done % 50_000 == 0:
                    dst.freeze()
                    print(f'    {n_done:,} docs copied, freeze checkpoint')
    for v in batch:
        dst.insert(v)
    if dst.active_size() > 0:
        dst.freeze()
    print(f'  done. {n_done + len(batch):,} docs in {args.dst}')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('export', help='Export vecs.fvecs to a path')
    p.add_argument('cluster_root')
    p.add_argument('name')
    p.add_argument('out')
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser('copy', help='Clone an index by re-ingesting')
    p.add_argument('cluster_root')
    p.add_argument('src')
    p.add_argument('dst')
    p.set_defaults(fn=cmd_copy)

    args = ap.parse_args()
    args.fn(args)


if __name__ == '__main__':
    main()
