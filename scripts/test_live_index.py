"""End-to-end test of LiveIndex: stream-ingest SIFT 1M, verify recall+WAL crash."""
from __future__ import annotations

import os, struct, sys, time, shutil
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from live_index import LiveIndex


def read_fvecs(path, n, dim):
    out = np.empty((n, dim), dtype=np.float32)
    with open(path, 'rb') as f:
        for i in range(n):
            f.read(4)
            out[i] = np.frombuffer(f.read(dim * 4), dtype=np.float32)
    return out


def read_gt(path, n, top_k):
    out = np.empty((n, top_k), dtype=np.int32)
    with open(path, 'rb') as f:
        k = struct.unpack('<i', f.read(4))[0]
    with open(path, 'rb') as f:
        for i in range(n):
            f.read(4)
            out[i] = np.frombuffer(f.read(k * 4), dtype=np.int32)[:top_k]
    return out


def main():
    ROOT  = '/tmp/sift1m_live'
    BASE  = 'sift/sift_base.fvecs'
    QF    = 'sift/sift_query.fvecs'
    GTF   = 'sift/sift_groundtruth.ivecs'
    DIM   = 128
    N_DOC = 1_000_000
    BATCH = 100_000  # 10 batches × 100k = 1M

    # Fresh start
    if os.path.exists(ROOT):
        shutil.rmtree(ROOT)

    print('=== 1. CREATE ===')
    li = LiveIndex.create(ROOT, dim=DIM, sub_dim=16, n_trees=1000,
                          depth=15, gen_version=3, base_path=BASE)

    print('=== 2. STREAM-INGEST 1M docs (10 × 100k batches) ===')
    t0 = time.time()
    with open(BASE, 'rb') as f:
        for i in range(N_DOC):
            f.read(4)
            v = np.frombuffer(f.read(DIM * 4), dtype=np.float32)
            li.insert(v, doc_id=i)
            if (i + 1) % BATCH == 0:
                wt = time.time()
                seg = li.freeze()
                print(f'  batch {(i+1)//BATCH:2}/10  doc {i+1:7}  '
                      f'freeze={time.time()-wt:5.1f}s  -> {seg}')
    total = time.time() - t0
    print(f'  total ingest+freeze time: {total:.1f}s')

    print('\n=== 3. STATS ===')
    print(f'  segments: {len(li.manifest["segments"])}')
    print(f'  next_doc_id: {li.manifest["next_doc_id"]}')

    print('\n=== 4. QUERY recall vs SIFT GT ===')
    queries = read_fvecs(QF, 100, DIM)
    gt      = read_gt(GTF, 100, 10)
    lat, recalls = [], []
    for i, q in enumerate(queries):
        t0 = time.time()
        topk = li.query(q, top_n=4000, top_k=10)
        lat.append((time.time() - t0) * 1000)
        gt_set = set(int(x) for x in gt[i])
        recalls.append(sum(1 for x in topk if int(x) in gt_set) / 10)
    print(f'  recall@10 = {np.mean(recalls):.4f}')
    print(f'  p50={sorted(lat)[50]:.1f}ms  p99={sorted(lat)[99]:.1f}ms')

    print('\n=== 5. WAL CRASH-RECOVERY TEST ===')
    print('  Inserting 1000 more docs WITHOUT freeze ...')
    for i in range(1000):
        f_open = open(BASE, 'rb')
        f_open.seek(4)
        v = np.frombuffer(f_open.read(DIM * 4), dtype=np.float32)
        f_open.close()
        li.insert(v.copy(), doc_id=N_DOC + i)
    print(f'  active size BEFORE close: {li.active_size()}')
    li.close()

    print('  Re-opening (simulates after-crash restart) ...')
    li2 = LiveIndex.open(ROOT)
    print(f'  active size AFTER re-open (replayed from WAL): {li2.active_size()}')
    assert li2.active_size() == 1000, 'WAL replay failed!'
    print('  ✓ WAL replay restored the 1000 buffered docs')
    li2.close()


if __name__ == '__main__':
    main()
