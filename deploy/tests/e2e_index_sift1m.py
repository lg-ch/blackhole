"""Deployment e2e — Phase: SIFT 1M indexing via the SDK against the kube pod.

Reads sift_base.fvecs (1M × 128), creates an index over the deployed
mangrove-search service (reached via `kubectl port-forward`), and streams the
vectors in through the SDK's insert_batch. Server assigns doc_ids; we assert
they come back as a contiguous 0..N-1 range so that doc_id == groundtruth row
(needed by the search/recall test).

Run after:  kubectl port-forward svc/mg-mangrove-search 8000:8000
Usage:      python3 deploy/tests/e2e_index_sift1m.py [N]
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import mangrove as mg
from _sift import read_fvecs

URL    = os.environ.get('MG_URL', 'http://localhost:8000')
BASE   = os.environ.get('SIFT_BASE', 'sift/sift_base.fvecs')
NAME   = os.environ.get('MG_INDEX', 'sift1m')
N      = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
BATCH  = int(os.environ.get('MG_BATCH', '25000'))


def main():
    print(f'[index] reading {BASE} (limit={N}) ...', flush=True)
    vecs = read_fvecs(BASE, limit=N)
    n, dim = vecs.shape
    print(f'[index] {n} vectors, dim={dim}', flush=True)

    client = mg.Client(URL, timeout=600.0)
    print(f'[index] health: {client.health()}', flush=True)

    if client.exists(NAME):
        print(f'[index] dropping pre-existing index {NAME!r}', flush=True)
        client.drop(NAME)

    max_active = os.environ.get('MG_MAX_ACTIVE')
    kw = {'max_active': int(max_active)} if max_active else {}
    client.create(NAME, dim=dim, **kw)      # server defaults (sub_dim=16, n_trees=1000, depth=14)
    print(f'[index] created {NAME!r} (defaults, max_active={max_active or "100k"})', flush=True)

    t0 = time.time()
    next_expected = 0
    for off in range(0, n, BATCH):
        chunk = vecs[off:off + BATCH]
        tb = time.time()
        ids = client.insert_batch(NAME, chunk)
        dt = time.time() - tb
        # Contiguity check: doc_ids must be next_expected .. next_expected+len-1
        if ids[0] != next_expected or ids[-1] != next_expected + len(ids) - 1:
            raise SystemExit(
                f'[index] FAIL non-contiguous doc_ids at off={off}: '
                f'got [{ids[0]}..{ids[-1]}], expected start {next_expected}')
        next_expected += len(ids)
        done = off + len(chunk)
        rate = done / (time.time() - t0)
        print(f'[index] {done:>8}/{n}  (+{len(ids)} in {dt:5.1f}s)  '
              f'avg {rate:7.0f} vec/s', flush=True)

    print('[index] final freeze ...', flush=True)
    tf = time.time()
    seg = client.freeze(NAME)
    print(f'[index] freeze -> {seg} in {time.time()-tf:.1f}s', flush=True)

    total = time.time() - t0
    st = client.stats(name=NAME)
    print('\n=== SIFT 1M INDEXING RESULT ===', flush=True)
    print(f'  inserted      : {next_expected}', flush=True)
    print(f'  total_docs    : {st.get("total_docs")}', flush=True)
    print(f'  total_segments: {st.get("total_segments")}', flush=True)
    print(f'  disk_bytes    : {st.get("total_disk_bytes")}', flush=True)
    print(f'  wall time     : {total:.1f}s  ({next_expected/total:.0f} vec/s)', flush=True)
    assert next_expected == n, 'inserted count mismatch'
    print('  STATUS        : OK', flush=True)


if __name__ == '__main__':
    main()
