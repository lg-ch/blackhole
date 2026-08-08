"""Forward-migration tool for LiveIndex manifest schemas.

Usage:
    python3 scripts/migrate_manifest.py /path/to/index_dir

For now, the only documented version is v1. Future v2 changes will
extend this script with migration steps (e.g. v1 → v2 might rename
fields, split base_path into a list, etc.).
"""
from __future__ import annotations
import argparse, json, os, sys


def migrate(root: str) -> None:
    mfp = os.path.join(root, 'manifest.json')
    if not os.path.exists(mfp):
        sys.exit(f'no manifest at {mfp}')
    with open(mfp) as f:
        m = json.load(f)
    v = m.get('version', 1)
    print(f'  current version : {v}')

    # No migrations currently needed. The framework is in place for
    # future bumps.
    changed = False
    if v < 1:
        # Hypothetical v0 → v1 step would go here.
        pass

    if changed:
        tmp = mfp + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(m, f, indent=2)
            f.flush(); os.fsync(f.fileno())
        os.rename(tmp, mfp)
        print(f'  migrated to v{m["version"]}')
    else:
        print('  no migration needed')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    args = ap.parse_args()
    migrate(args.root)


if __name__ == '__main__':
    main()
