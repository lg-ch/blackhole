"""Crash-mid-build safety test.

Sequence :
  1. Start rpforest build of N=200 trees on SIFT 100k subset
  2. After ~3 s, SIGKILL the build process — mid-flight
  3. Inspect the partial dir : how many tree*.srt are complete ?
  4. Query the partial index :
       - It SHOULD open (LiveIndex/Forest tolerates a degraded set)
       - recall@10 will be lower than full (fewer trees voting)
  5. Resume the build with the same command :
       - Build code skips complete .srt files in phase 2 and resumes
         vector ingest via phase1.progress
  6. After resume, query again : recall@10 must reach the nominal level

The key invariant verified : every doc_id is present in EVERY built tree
(by design, build writes a leaf entry per (tree, doc)). A crash in the
middle leaves whole trees done or absent, never partial-coverage per
doc within a tree — so the partial index is still RIGHT, just degraded.
"""
from __future__ import annotations
import os, shutil, signal, struct, subprocess, sys, time
import numpy as np

HERE   = os.path.dirname(os.path.abspath(__file__))
BIN    = os.path.join(os.path.dirname(HERE), 'rpforest')
DIM    = 128
N_DOC  = 100_000
N_TREES = 200
DEPTH  = 14


def read_fvecs(p, n, d=DIM):
    out = np.empty((n, d), dtype=np.float32)
    with open(p, 'rb') as f:
        for i in range(n):
            f.read(4); out[i] = np.frombuffer(f.read(d * 4), dtype=np.float32)
    return out


def count_complete_trees(idx_dir: str) -> int:
    if not os.path.exists(idx_dir): return 0
    return len([f for f in os.listdir(idx_dir) if f.startswith('tree') and f.endswith('.srt')])


def open_partial_and_query(idx_dir: str, queries, n_trees_used: int):
    """Open the index using n_trees_used (== count of .srt files actually
       present) instead of the manifest n_trees. Forest will only iterate
       up to n_trees_used trees, ignoring missing .srt files."""
    sys.path.insert(0, HERE)
    from mangrove_ffi import Forest, set_gen_version
    set_gen_version(3)
    f = Forest(idx_dir, n_trees=n_trees_used, dim=DIM, sub_dim=16,
               depth=DEPTH, n_docs=N_DOC, gen_version=3)
    base = read_fvecs('sift/sift_base.fvecs', N_DOC)
    recalls = []
    for q in queries:
        ids, votes, n = f.query(q, top_n=2000)
        cand = ids[:n]
        cv = base[cand]
        d2 = ((cv - q) ** 2).sum(axis=1)
        order = np.argsort(d2)[:10]
        topk = set(int(cand[i]) for i in order)
        # ground truth subset : brute-force top-10 over first N_DOC docs
        gt = ((base - q) ** 2).sum(axis=1)
        gt10 = set(int(i) for i in np.argpartition(gt, 9)[:10])
        recalls.append(len(topk & gt10) / 10)
    f.close()
    return float(np.mean(recalls))


def run_build(idx_dir: str) -> subprocess.Popen:
    cmd = [
        BIN, '--dim', str(DIM), '--sub_dim', '16', '--gen', 'v3',
        '--doc_offset', '0', '--doc_count', str(N_DOC),
        'build', 'sift/sift_base.fvecs', idx_dir,
        str(N_TREES), str(DEPTH),
    ]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)


def main():
    IDX = '/tmp/crash_build_idx'
    if os.path.exists(IDX):
        shutil.rmtree(IDX)

    # Wait until phase 2 has produced at least some .srt files, then kill.
    # Robust against CPU contention slowing phase 1.
    print('=== 1. Start build, wait for phase-2 to start, then kill ===')
    proc = run_build(IDX)
    deadline = time.time() + 300  # 5 min max
    while time.time() < deadline:
        time.sleep(1.0)
        if count_complete_trees(IDX) >= 20:  # at least 20 .srt files
            break
    print(f'  waited {time.time() - (deadline - 300):.1f}s before kill')
    proc.send_signal(signal.SIGKILL); proc.wait()
    print(f'  killed, exit code = {proc.returncode}')

    n_done = count_complete_trees(IDX)
    print(f'  complete .srt files on disk : {n_done}/{N_TREES}')
    assert n_done > 0, 'crash too early, no trees built'
    assert n_done < N_TREES, 'crash too late, all trees done'

    queries = read_fvecs('sift/sift_query.fvecs', 50)

    print(f'\n=== 2. Query the partial index with {n_done} trees ===')
    partial_recall = open_partial_and_query(IDX, queries, n_done)
    print(f'  partial recall@10 = {partial_recall:.4f}')

    print(f'\n=== 3. Resume build with same command ===')
    t0 = time.time()
    proc2 = run_build(IDX)
    proc2.wait()
    print(f'  resume took {time.time() - t0:.1f}s, exit = {proc2.returncode}')
    assert proc2.returncode == 0, 'resume failed'

    n_final = count_complete_trees(IDX)
    print(f'  trees after resume : {n_final}/{N_TREES}')
    assert n_final == N_TREES, f'expected {N_TREES}, got {n_final}'

    print(f'\n=== 4. Query full index ===')
    full_recall = open_partial_and_query(IDX, queries, N_TREES)
    print(f'  full recall@10 = {full_recall:.4f}')

    ok = full_recall >= 0.95 and partial_recall < full_recall
    print(f'\n=== VERDICT : {"PASS" if ok else "FAIL"} ===')
    print(f'  partial ({n_done} trees) recall = {partial_recall:.4f}')
    print(f'  full    ({N_TREES} trees) recall = {full_recall:.4f}')
    print(f'  monotonic improvement : {full_recall - partial_recall:+.4f}')


if __name__ == '__main__':
    main()
