"""arxiv 2M — random-projection variant.

Bench an arxiv 2M forest built on a low-dim random projection (RP128 or RP64),
comparing two rerank modes:
  - "stage2 = projected"  : exact L2 on the projected base file (dim k)
  - "stage2 = original"   : exact L2 on the original base file (dim 768)
Both report recall against the canonical GT (computed on the original dim 768).

Run:  python3 bench/run_arxiv_rp.py <128|64>
"""
import os, sys, time, json, struct, subprocess, resource, multiprocessing as mp

os.environ['OMP_NUM_THREADS'] = '1'
os.sched_setaffinity(0, {0})

import numpy as np
sys.path.insert(0, '/home/chatelet/mangrove-search/scripts')
import mangrove_ffi as mf
from mangrove_ffi import Forest

K = int(sys.argv[1]) if len(sys.argv) > 1 else 128
DATA = '/home/chatelet/mangrove-search/datasets/arxiv'
BASE_ORIG = f'{DATA}/arxiv_base.fvecs'                       # dim 768
BASE_PROJ = f'{DATA}/arxiv_rp{K}.fvecs'                      # dim K
MATRIX    = f'{DATA}/arxiv_rp{K}.fvecs.matrix.npy'           # 768 × K
IDX       = f'/mnt/mangrove/indexes/arxiv_rp{K}'
QP        = f'{DATA}/bench_q.fvecs'
GP        = f'{DATA}/bench_gt.ivecs'
DIM_ORIG, DIM_PROJ = 768, K
N_TREES, DEPTH, N_DOCS = 256, 20, 2058751


def drop_caches():
    os.sync(); open('/proc/sys/vm/drop_caches', 'w').write('3')

def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

def disk_size_mb(path):
    if not os.path.exists(path): return 0.0
    out = subprocess.run(['du', '-sb', path], capture_output=True, text=True)
    return int(out.stdout.split()[0]) / (1024.0 * 1024.0)


def time_run(forest, Q_proj, Q_orig, GT, np_, qd, tn,
             stage2_base, stage2_q, label, mode_b_dim=None, mode_b_n=0,
             n_warmup=5, n_warm=100):
    """If mode_b_dim is set, stage 2 reads `mode_b_dim`-sized rows from
    `stage2_base` (np.memmap, robust to oob ids)."""
    drop_caches()
    if mode_b_dim:
        row_bytes_b = 4 + mode_b_dim * 4
        # fvecs: per-row [int32 dim][dim × float32] — view directly as records
        rec_dt = np.dtype([('hdr', np.int32), ('v', np.float32, mode_b_dim)])
        mm = np.memmap(stage2_base, dtype=rec_dt, mode='r', shape=(mode_b_n,))

    def do_query(qi):
        q_proj = Q_proj[qi % len(Q_proj)]
        q_orig = stage2_q[qi % len(stage2_q)]
        ids, _, n = forest.query_probes(q_proj, np_, top_n=tn, probe_depth=qd)
        cand = np.asarray(ids[:n], dtype=np.int32); cand = cand[cand >= 0]
        cand = cand[(cand < mode_b_n) if mode_b_dim else (cand < 2**31)]
        if mode_b_dim is None:
            return forest.rerank_l2(stage2_base, q_orig, cand, top_k=10)
        if len(cand) == 0:
            return np.array([], dtype=np.int32)
        vecs = mm['v'][cand]                     # (n_cand, dim) view
        diff = vecs - q_orig
        d2 = np.einsum('ij,ij->i', diff, diff)
        order = np.argsort(d2)[:10]
        return cand[order]

    for i in range(n_warmup):
        do_query(i)
    recs, lats = [], []
    for qi in range(n_warmup, n_warmup + n_warm):
        ref = set(int(x) for x in GT[qi % len(GT)])
        t0 = time.time()
        top = do_query(qi)
        lats.append((time.time() - t0) * 1000.0)
        recs.append(sum(1 for x in top if int(x) in ref) / 10.0)
    return label, float(np.mean(recs)), \
        float(np.percentile(lats, 50)), float(np.percentile(lats, 95))


def main():
    # --- queries + GT (original ground truth) ---
    qraw = np.fromfile(QP, dtype=np.int32)
    nq = qraw.size // (1 + DIM_ORIG)
    Q_orig = qraw.reshape(nq, 1 + DIM_ORIG)[:, 1:].view(np.float32).copy()
    graw = np.fromfile(GP, dtype=np.int32)
    gk = graw.size // nq - 1
    GT = graw.reshape(nq, 1 + gk)[:, 1:11].copy()

    # --- project queries with the SAME matrix used at index time ---
    M = np.load(MATRIX)           # (768, K), float32
    Q_proj = (Q_orig @ M).astype(np.float32, copy=False)
    assert Q_proj.shape == (nq, DIM_PROJ)

    idx_mb  = disk_size_mb(IDX)
    proj_mb = disk_size_mb(BASE_PROJ)
    orig_mb = disk_size_mb(BASE_ORIG)

    print('==========================================================')
    print(f' arxiv 2M — Random Projection 768 → {DIM_PROJ}')
    print('==========================================================')
    print(f'  Vectors        : float32, dim {DIM_ORIG} → {DIM_PROJ} (Achlioptas)')
    print(f'  Source         : Cohere v3 arXiv abstracts, projected via '
          f'sparse Achlioptas')
    print(f'  Queries / GT   : {nq} held-out / GT@10 by exhaustive L2 on dim 768')
    print(f'  Technique      : RP {DIM_ORIG}→{DIM_PROJ} '
          f'+ RP-forest ({N_TREES} trees, depth {DEPTH}, sub_dim 16, gen v3)')
    print(f'                   + exact L2 rerank (two variants compared)')
    print(f'  Disk           : index {idx_mb:.1f} MB, projected base '
          f'{proj_mb:.1f} MB, original base {orig_mb:.1f} MB')
    print(f'  CPU            : pinned to core 0, OMP_NUM_THREADS=1')
    print()

    mf.set_shared_scratch_pool(True)
    f = Forest(IDX, n_trees=N_TREES, dim=DIM_PROJ, sub_dim=16, depth=DEPTH,
               n_docs=N_DOCS, gen_version=3)

    SWEEP = []
    for np_ in (5, 10, 20):
        for qd in (DEPTH - 2, DEPTH - 4, DEPTH - 6):  # 18, 16, 14
            for tn in (16_000, 32_000, 64_000):
                SWEEP.append((np_, qd, tn))

    print('  Mode A — exact L2 on PROJECTED base (dim {})  fast but lossy'.format(DIM_PROJ))
    print(f'  Single CPU, 100 warm queries per row\n')
    print(f'  {"NP":>3} {"QD":>3} {"top_n":>7} | '
          f'{"recall@10":>9} {"p50w":>7} {"p95w":>7}')
    print('  ' + '-' * 50)

    results = {'A_projected_L2': [], 'B_original_L2': []}
    for np_, qd, tn in SWEEP:
        # Mode A: rerank L2 on projected base, with projected query
        lbl, rec, p50, p95 = time_run(f, Q_proj, Q_orig, GT, np_, qd, tn,
                                       BASE_PROJ, Q_proj, 'A')
        results['A_projected_L2'].append({
            'n_probes': np_, 'probe_depth': qd, 'top_n': tn,
            'recall': rec, 'p50_warm_ms': p50, 'p95_warm_ms': p95,
        })
        mark = ' *' if rec >= 0.999 else ('  ' if rec < 0.99 else ' .')
        print(f'  {np_:>3} {qd:>3} {tn:>7} | '
              f'{rec:>9.4f} {p50:>5.1f}ms {p95:>5.1f}ms{mark}', flush=True)

    print()
    print('  Mode B — exact L2 on ORIGINAL base (dim 768)  slower but accurate')
    print()
    print(f'  {"NP":>3} {"QD":>3} {"top_n":>7} | '
          f'{"recall@10":>9} {"p50w":>7} {"p95w":>7}')
    print('  ' + '-' * 50)

    for np_, qd, tn in SWEEP:
        # Mode B: rerank L2 on ORIGINAL base, with ORIGINAL query, Python-side
        lbl, rec, p50, p95 = time_run(f, Q_proj, Q_orig, GT, np_, qd, tn,
                                       BASE_ORIG, Q_orig, 'B',
                                       mode_b_dim=DIM_ORIG,
                                       mode_b_n=N_DOCS)
        results['B_original_L2'].append({
            'n_probes': np_, 'probe_depth': qd, 'top_n': tn,
            'recall': rec, 'p50_warm_ms': p50, 'p95_warm_ms': p95,
        })
        mark = ' *' if rec >= 0.999 else ('  ' if rec < 0.99 else ' .')
        print(f'  {np_:>3} {qd:>3} {tn:>7} | '
              f'{rec:>9.4f} {p50:>5.1f}ms {p95:>5.1f}ms{mark}', flush=True)

    print()
    best_A = sorted(results['A_projected_L2'],
                    key=lambda r: (-r['recall'], r['p50_warm_ms']))[0]
    best_B = sorted(results['B_original_L2'],
                    key=lambda r: (-r['recall'], r['p50_warm_ms']))[0]
    print(f'  Best Mode A : recall {best_A["recall"]:.4f} / '
          f'p50 {best_A["p50_warm_ms"]:.1f} ms  '
          f'(NP={best_A["n_probes"]} QD={best_A["probe_depth"]} '
          f'top_n={best_A["top_n"]})')
    print(f'  Best Mode B : recall {best_B["recall"]:.4f} / '
          f'p50 {best_B["p50_warm_ms"]:.1f} ms  '
          f'(NP={best_B["n_probes"]} QD={best_B["probe_depth"]} '
          f'top_n={best_B["top_n"]})')
    print(f'  Query peak RSS : {peak_rss_mb():.1f} MB')
    print('==========================================================')

    os.makedirs('bench/results', exist_ok=True)
    with open(f'bench/results/arxiv_rp{K}.json', 'w') as fh:
        json.dump({
            'dataset': f'arxiv_rp{K}',
            'dim_orig': DIM_ORIG, 'dim_proj': DIM_PROJ,
            'n_docs': N_DOCS, 'n_trees': N_TREES, 'depth': DEPTH,
            'sub_dim': 16, 'gen_version': 3,
            'disk_index_mb': idx_mb,
            'disk_proj_mb': proj_mb,
            'disk_orig_mb': orig_mb,
            'sweep': results,
            'best_A_projected': best_A,
            'best_B_original': best_B,
            'query_peak_rss_mb': peak_rss_mb(),
        }, fh, indent=2)
    print(f'  → bench/results/arxiv_rp{K}.json')


if __name__ == '__main__':
    main()
