"""End-to-end test of LiveIndex auto-compaction on SIFT 1M.

Ingest 1M docs in 20 batches of 50k. Auto-compaction triggers after each
freeze. Final manifest should show ~2-3 segments instead of 20.
"""
from __future__ import annotations
import os, struct, sys, shutil, time
from collections import Counter

import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from live_index import LiveIndex


def show_manifest(li, label):
    by_tier = Counter()
    total = 0
    print(f'\n--- after {label} ---')
    for s in li.manifest['segments']:
        by_tier[s.get('tier', 0)] += 1
        total += s['n_docs']
        print(f'  {s["name"]:25} tier={s.get("tier",0)} '
              f'depth={s["depth"]:2} n_docs={s["n_docs"]:>7} '
              f'offset={s["doc_offset"]}')
    print(f'  totals : {dict(by_tier)} | sum n_docs = {total}')


def main():
    ROOT  = '/tmp/sift1m_live_compact'
    BASE  = 'sift/sift_base.fvecs'
    DIM   = 128
    N_DOC = 1_000_000
    BATCH = 50_000

    if os.path.exists(ROOT):
        shutil.rmtree(ROOT)

    print('=== 1. CREATE ===')
    li = LiveIndex.create(ROOT, dim=DIM, sub_dim=16, n_trees=200,
                          depth=14, gen_version=3, base_path=BASE)

    print(f'=== 2. STREAM-INGEST {N_DOC} docs ({N_DOC//BATCH} batches × {BATCH}) ===')
    t0 = time.time()
    with open(BASE, 'rb') as f:
        for i in range(N_DOC):
            f.read(4)
            v = np.frombuffer(f.read(DIM * 4), dtype=np.float32)
            li.insert(v, doc_id=i)
            if (i + 1) % BATCH == 0:
                wt = time.time()
                li.freeze()
                show_manifest(li, f'batch {(i+1)//BATCH:2}/{N_DOC//BATCH} '
                                  f'(freeze+compact {time.time()-wt:.1f}s)')
    print(f'\n=== total time : {(time.time()-t0)/60:.1f} min ===')

    print('\n=== 3. FINAL STATE ===')
    show_manifest(li, 'all done')
    print(f'\nfinal segment count: {len(li.manifest["segments"])}')

    # Recall sanity check
    print('\n=== 4. QUERY recall vs SIFT GT (top_k=10) ===')
    def read_fvecs(path, n):
        out = np.empty((n, DIM), dtype=np.float32)
        with open(path, 'rb') as f:
            for i in range(n):
                f.read(4)
                out[i] = np.frombuffer(f.read(DIM * 4), dtype=np.float32)
        return out
    def read_gt(path, n):
        out = np.empty((n, 10), dtype=np.int32)
        with open(path, 'rb') as f:
            k = struct.unpack('<i', f.read(4))[0]
        with open(path, 'rb') as f:
            for i in range(n):
                f.read(4)
                out[i] = np.frombuffer(f.read(k * 4), dtype=np.int32)[:10]
        return out
    queries = read_fvecs('sift/sift_query.fvecs', 100)
    gt      = read_gt('sift/sift_groundtruth.ivecs', 100)
    recalls = []
    for i, q in enumerate(queries):
        topk = li.query(q, top_n=4000, top_k=10)
        s = set(int(x) for x in gt[i])
        recalls.append(sum(1 for x in topk if int(x) in s) / 10)
    print(f'  recall@10 = {np.mean(recalls):.4f}')

    li.close()


if __name__ == '__main__':
    main()
