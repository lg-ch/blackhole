"""Atomic backup + restore of a LiveIndex.

A backup is a tar.gz of the entire <root> dir AFTER a flush:
  1. open the LiveIndex (replays WAL into vecs.fvecs if needed)
  2. freeze any pending active buffer (no docs lost)
  3. close (releases fds)
  4. tar.gz to the target path

Restore extracts a backup tarball to a fresh dir; the LiveIndex is then
openable from that dir without any further setup.

Usage :
  python3 scripts/backup_restore.py backup  /path/to/live_idx /path/to/snap.tar.gz
  python3 scripts/backup_restore.py restore /path/to/snap.tar.gz /path/to/new_dir
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from live_index import LiveIndex


def cmd_backup(args):
    li = LiveIndex.open(args.root)
    if li.active_size() > 0:
        sys.stderr.write(f'  freezing {li.active_size()} pending docs ...\n')
        li.freeze()
    li.close()
    sys.stderr.write(f'  taring {args.root} -> {args.out} ...\n')
    t0 = time.time()
    # Use streaming tar with gzip for portability; for size-critical
    # production prefer rsync --link-dest or a dedicated snapshot tool.
    tmp = args.out + '.tmp'
    with tarfile.open(tmp, 'w:gz', compresslevel=1) as t:
        t.add(args.root, arcname=os.path.basename(args.root.rstrip('/')))
    os.rename(tmp, args.out)
    sz = os.path.getsize(args.out)
    sys.stderr.write(f'  done in {time.time() - t0:.1f}s, archive {sz/1024**2:.1f} MB\n')


def cmd_restore(args):
    if os.path.exists(args.dest):
        if not args.force:
            sys.stderr.write(f'  dest {args.dest} exists; use --force to overwrite\n')
            sys.exit(1)
        subprocess.check_call(['rm', '-rf', args.dest])
    parent = os.path.dirname(os.path.abspath(args.dest)) or '.'
    os.makedirs(parent, exist_ok=True)
    sys.stderr.write(f'  extracting {args.archive} -> {args.dest} ...\n')
    t0 = time.time()
    with tarfile.open(args.archive, 'r:gz') as t:
        t.extractall(path=parent)
        # The archive's root entry is the basename of the original index;
        # rename to whatever the user requested.
        first = t.getnames()[0].split('/')[0]
        src = os.path.join(parent, first)
        if src != args.dest:
            os.rename(src, args.dest)
    sys.stderr.write(f'  done in {time.time() - t0:.1f}s\n')
    # Sanity check : try opening
    li = LiveIndex.open(args.dest)
    sys.stderr.write(
        f'  restored : {len(li.manifest["segments"])} segments, '
        f'next_doc_id={li.manifest["next_doc_id"]}, '
        f'active={li.active_size()}\n')
    li.close()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('backup')
    p.add_argument('root')
    p.add_argument('out')
    p.set_defaults(fn=cmd_backup)

    p = sub.add_parser('restore')
    p.add_argument('archive')
    p.add_argument('dest')
    p.add_argument('--force', action='store_true')
    p.set_defaults(fn=cmd_restore)

    args = ap.parse_args()
    args.fn(args)


if __name__ == '__main__':
    main()
