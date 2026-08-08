"""Test LiveIndex auto-store mode (no external base_path).

Streaming-ingest 200k SIFT 1M vectors, freeze every 50k, verify:
  - vecs.fvecs grows correctly
  - compactions still work via the auto-managed store
  - recall is comparable to managed-base mode
"""
from __future__ import annotations
import os, struct, sys, shutil
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from live_index import LiveIndex


def main():
    ROOT  = '/tmp/sift1m_autostore'
    SRC   = 'sift/sift_base.fvecs'
    DIM   = 128
    N     = 200_000   # only 200k for a quick test (compactions still trigger)
    BATCH = 50_000

    if os.path.exists(ROOT):
        shutil.rmtree(ROOT)

    print('=== CREATE (no base_path → auto-store) ===')
    li = LiveIndex.create(ROOT, dim=DIM, sub_dim=16, n_trees=200,
                          depth=14, gen_version=3)
    print(f'  base_path in manifest = {li.manifest["base_path"]}')
    print(f'  auto_store = {li.manifest["auto_store"]}')

    print(f'=== INGEST {N} docs ({N//BATCH} batches × {BATCH}) ===')
    with open(SRC, 'rb') as f:
        for i in range(N):
            f.read(4)
            v = np.frombuffer(f.read(DIM * 4), dtype=np.float32)
            li.insert(v, doc_id=i)
            if (i + 1) % BATCH == 0:
                li.freeze()

    print(f'\n=== STATE ===')
    print(f'  segments: {len(li.manifest["segments"])}')
    for s in li.manifest['segments']:
        print(f'    {s["name"]:25} tier={s.get("tier",0)} depth={s["depth"]} n={s["n_docs"]}')
    print(f'  managed vecs.fvecs size: {os.path.getsize(li._base_path_abs):,} bytes')

    print('\n=== QUERY recall@10 vs SIFT GT (first 200k subset) ===')
    def read_fvecs(p, n, d=128):
        out = np.empty((n, d), dtype=np.float32)
        with open(p, 'rb') as f:
            for i in range(n):
                f.read(4)
                out[i] = np.frombuffer(f.read(d * 4), dtype=np.float32)
        return out
    def read_gt(p, n):
        out = np.empty((n, 10), dtype=np.int32)
        with open(p, 'rb') as f: k = struct.unpack('<i', f.read(4))[0]
        with open(p, 'rb') as f:
            for i in range(n):
                f.read(4); out[i] = np.frombuffer(f.read(k * 4), dtype=np.int32)[:10]
        return out
    qs = read_fvecs('sift/sift_query.fvecs', 100)
    gt = read_gt('sift/sift_groundtruth.ivecs', 100)
    rs = []
    for i, q in enumerate(qs):
        topk = li.query(q, top_n=4000, top_k=10)
        # Filter GT to only docs in [0, N)
        s = set(int(x) for x in gt[i] if int(x) < N)
        if not s:
            continue
        rs.append(sum(1 for x in topk if int(x) in s) / max(1, len(s)))
    print(f'  recall@10 (vs in-corpus GT) = {np.mean(rs):.4f}')

    li.close()


if __name__ == '__main__':
    main()
