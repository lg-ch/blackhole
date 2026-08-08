"""Index lifecycle helper : drop indexes past their TTL.

Each index can carry a `ttl_days` field in its manifest. A simple
periodic job calls expire() to drop expired indexes. Commonly used
with the `name-YYYYMMDD` daily-roll convention.

CLI usage :
    # one-shot
    python3 scripts/lifecycle.py expire /var/mangrove/cluster

    # cron-style loop (e.g. via systemd timer or k8s CronJob)
    python3 scripts/lifecycle.py expire /var/mangrove/cluster \
        --loop --interval 3600
"""
from __future__ import annotations

import argparse, os, sys, time
import datetime as dt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from registry import MangroveCluster


def index_age_days(idx_dir: str) -> float:
    """Approximate age via manifest mtime. Future v2 could store a
       creation_date field in the manifest itself.                      */"""
    mfp = os.path.join(idx_dir, 'manifest.json')
    if not os.path.exists(mfp):
        return -1.0
    return (time.time() - os.path.getmtime(mfp)) / 86400.0


def parse_date_from_name(name: str) -> dt.date | None:
    """If the index name ends in -YYYYMMDD or -YYYY-MM-DD, parse it.
       Falls back to None if no date is encoded.                        */"""
    for fmt in ('%Y%m%d', '%Y-%m-%d'):
        tail = name.split('-')[-1] if fmt == '%Y%m%d' else name[-10:]
        try:
            return dt.datetime.strptime(tail, fmt).date()
        except ValueError:
            continue
    return None


def expire(cluster_root: str, default_ttl_days: int | None,
           dry_run: bool = False) -> int:
    """Drop indexes whose age > ttl_days. Returns count dropped."""
    cl = MangroveCluster(cluster_root)
    today = dt.date.today()
    dropped = 0
    for name in cl.list_indexes():
        idx_dir = os.path.join(cl.root, name)
        # Prefer date encoded in name (deterministic). Fall back to mtime.
        d = parse_date_from_name(name)
        if d is not None:
            age_days = (today - d).days
        else:
            age_days = index_age_days(idx_dir)
        ttl = default_ttl_days
        if ttl is None:
            continue   # nothing to do when no ttl configured
        if age_days < ttl:
            continue
        print(f'  [{"DRY " if dry_run else ""}DROP] {name:30} '
              f'age={age_days:.0f}d ttl={ttl}d')
        if not dry_run:
            cl.drop_index(name)
        dropped += 1
    cl.close()
    return dropped


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('expire')
    p.add_argument('root')
    p.add_argument('--ttl-days', type=int, required=True,
                   help='Drop indexes whose age (by name date or mtime) > N days')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--loop', action='store_true',
                   help='Run continuously, sleeping between passes')
    p.add_argument('--interval', type=int, default=3600,
                   help='Loop sleep seconds (default 1h)')
    args = ap.parse_args()

    if args.cmd == 'expire':
        if args.loop:
            while True:
                n = expire(args.root, args.ttl_days, args.dry_run)
                print(f'  pass complete, dropped {n} indexes ; '
                      f'sleeping {args.interval}s')
                time.sleep(args.interval)
        else:
            n = expire(args.root, args.ttl_days, args.dry_run)
            print(f'done, dropped {n} index(es)')


if __name__ == '__main__':
    main()
