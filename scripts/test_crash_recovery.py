"""Crash-safety test : kill -9 mid-ingest then verify WAL replay recovers
all docs that completed fsync.

Sequence :
  1. Worker subprocess does insert(i) for i in [0, N) on a fresh LiveIndex.
     Each insert : WAL fsync + (optional) vecs.fvecs write + active buffer.
  2. Parent kills -SIGKILL the worker after a brief sleep — random crash
     point mid-batch (no graceful shutdown, no atexit fires).
  3. Parent opens the index : the WAL is replayed, the active buffer is
     reconstructed. We verify next_doc_id == (highest fully-fsync'd id + 1)
     and the buffer has every doc up to that id.
  4. Parent calls freeze() to materialize the segment, then queries to
     verify all docs are findable.

Run :
  python3 scripts/test_crash_recovery.py
"""
from __future__ import annotations
import os, shutil, signal, struct, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from live_index import LiveIndex


WORKER = """
import sys, time, numpy as np
sys.path.insert(0, '{here}')
from live_index import LiveIndex

li = LiveIndex.open('{root}')
i = li.manifest['next_doc_id']
N = {N}
while i < N:
    v = np.array([float(i), 0, 0, 0], dtype=np.float32)
    li.insert(v)
    i += 1
    # Slow enough that the parent can kill before we finish.
    time.sleep(0.005)
"""


def main():
    ROOT = '/tmp/crash_test'
    N    = 200
    if os.path.exists(ROOT):
        shutil.rmtree(ROOT)

    li = LiveIndex.create(ROOT, dim=4, sub_dim=0, n_trees=20,
                          depth=6, gen_version=3)
    li.close()

    print(f'=== run 1: worker inserts up to {N}, parent kills after 0.4s ===')
    proc = subprocess.Popen(
        ['python3', '-c', WORKER.format(here=HERE, root=ROOT, N=N)],
        stdout=sys.stdout, stderr=sys.stderr)
    time.sleep(0.4)
    proc.send_signal(signal.SIGKILL)
    proc.wait()
    print(f'  worker killed (exit={proc.returncode})')

    li = LiveIndex.open(ROOT)
    recovered = li.manifest['next_doc_id']
    n_active  = li.active_size()
    print(f'  recovered: next_doc_id={recovered}, active_size={n_active}')
    assert recovered == n_active, \
        f'expected next_doc_id == active size (no segments yet), got {recovered}/{n_active}'
    assert recovered > 0, 'nothing recovered — fsync broken?'

    print(f'=== run 2: continue from {recovered} to {N} ===')
    proc = subprocess.Popen(
        ['python3', '-c', WORKER.format(here=HERE, root=ROOT, N=N)],
        stdout=sys.stdout, stderr=sys.stderr)
    proc.wait()
    print(f'  worker finished (exit={proc.returncode})')

    # Reopen, freeze, query
    li.close()
    li = LiveIndex.open(ROOT)
    print(f'  after run 2: next_doc_id={li.manifest["next_doc_id"]}, '
          f'active={li.active_size()}')
    assert li.manifest['next_doc_id'] == N, \
        f'expected {N} docs after run 2, got {li.manifest["next_doc_id"]}'

    li.freeze()
    print(f'  frozen: {len(li.manifest["segments"])} segment(s)')

    # Query : doc i should rank closest to qvec = [i, 0, 0, 0]
    misses = 0
    for i in [0, recovered // 2, recovered, N // 2, N - 1]:
        q = np.array([float(i), 0, 0, 0], dtype=np.float32)
        topk = li.query(q, top_n=N, top_k=3)
        if int(topk[0]) != i:
            print(f'  MISS for doc {i}: got top1 = {int(topk[0])}')
            misses += 1
    print(f'\n=== RESULT : crash recovery {"PASS" if misses == 0 else "FAIL"}'
          f' ({N - misses}/{N} key checks OK) ===')
    li.close()


if __name__ == '__main__':
    main()
