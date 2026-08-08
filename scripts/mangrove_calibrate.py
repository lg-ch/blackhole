"""mangrove_calibrate : appelé pendant le build pour recalibrer la config.

Design (cf. project_autotune_doubling) : lu par le C build à chaque
doublement de taille (100k → 200k → 400k → ...). Ce script :
  1. Lit meta.txt de l'index partiel
  2. Lit calibration_{queries,gt}.bin (snapshots par set_build_calib)
  3. Run un grid sweep + top_n sweep (auto-tune version courte)
  4. Écrit recommended_config.json dans le dossier index
  5. Exit — le build C reprend

Version courte : N_Q=15 + grid réduit + early-stop plateau → ~1-2 min pour
un index à 100k, jusqu'à ~5 min pour 1B.
"""
import argparse, os, sys, time, subprocess, json, struct
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import mangrove_ffi as mf
from mangrove_ffi import Forest


def read_meta(index_dir):
    """meta.txt format is `key value` (space-separated, one per line)."""
    meta = {}
    with open(f'{index_dir}/meta.txt') as fh:
        for line in fh:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                meta[parts[0]] = parts[1]
    return meta


def read_calib_bin(index_dir, dim, top_k=10):
    """Format calibration.c :
       queries.bin : 2 × uint32 header (n_q, dim) + n_q × dim × float32 + n_q × int32 doc_ids
       gt.bin      : 2 × uint32 header (n_q, top_k) + n_q × top_k × int32 doc_ids"""
    qp = f'{index_dir}/calibration_queries.bin'
    gp = f'{index_dir}/calibration_gt.bin'
    if not (os.path.exists(qp) and os.path.exists(gp)):
        return None, None
    with open(qp, 'rb') as fh: qraw = fh.read()
    n_q_hdr, dim_hdr = struct.unpack('<II', qraw[:8])
    q_bytes = n_q_hdr * dim_hdr * 4
    Q = np.frombuffer(qraw, dtype=np.float32, count=n_q_hdr*dim_hdr,
                      offset=8).reshape(n_q_hdr, dim_hdr).copy()
    with open(gp, 'rb') as fh: graw = fh.read()
    n_q_gt, k_hdr = struct.unpack('<II', graw[:8])
    GT = np.frombuffer(graw, dtype=np.int32, count=n_q_gt*k_hdr,
                       offset=8).reshape(n_q_gt, k_hdr).copy()
    return Q, GT[:, :top_k]


def bench_config(index_dir, meta, Q, GT, tp, mlb, top_n, deadline_ms=1800,
                 n_q=15, qd=None):
    dim   = int(meta['dim'])
    sd    = int(meta.get('sub_dim', 0))
    depth = int(meta['depth'])
    n_docs = int(meta['n_docs'])
    nt    = int(meta['n_trees'])
    gen_v = int(meta.get('gen_version', 3))
    qd = qd if qd is not None else depth
    subprocess.run(['sync'], check=True)
    try:
        with open('/proc/sys/vm/drop_caches','w') as fh: fh.write('3')
    except PermissionError:
        pass  # non-root — skip drop_caches
    mf.set_max_leaf_bytes(mlb)
    f = Forest(index_dir, n_trees=nt, dim=dim, sub_dim=sd, depth=depth,
               n_docs=n_docs, gen_version=gen_v)
    mf.set_query_deadline_ms(deadline_ms)
    _ = f.query_pathrank(Q[0], 3, tp, top_n, query_depth=qd)  # warmup

    lats, recalls = [], []
    for qi in range(min(n_q, len(Q))):
        ref = set(int(x) for x in GT[qi])
        mf.set_query_deadline_ms(deadline_ms)
        t0 = time.time()
        ids,_,_ = f.query_pathrank(Q[qi], 3, tp, top_n, query_depth=qd)
        lats.append((time.time()-t0)*1000)
        recalls.append(sum(1 for x in ids[:10] if int(x) in ref) / 10.0)
    peak_kb = 0
    try:
        with open('/proc/self/status') as fh:
            for line in fh:
                if line.startswith('VmRSS:'): peak_kb = int(line.split()[1])
    except FileNotFoundError: pass
    f.close()
    return {'recall': float(np.mean(recalls)),
            'mean_ms': float(np.mean(lats)),
            'p50_ms': float(np.percentile(lats, 50)),
            'peak_mb': peak_kb / 1024.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', required=True)
    ap.add_argument('--n-docs', type=int, default=0)
    args = ap.parse_args()
    index_dir = args.index

    print(f'[calibrate] index={index_dir} n_docs_indexed={args.n_docs}', flush=True)
    if not os.path.exists(f'{index_dir}/meta.txt'):
        print('[calibrate] meta.txt not yet written — build not started, skip', flush=True)
        return 0
    meta = read_meta(index_dir)
    dim = int(meta['dim'])
    # Trees queryable = at least tree00000.srt exists (Phase 2 done).
    # During Phase 1 the trees are in .bin pair-file format (not queryable).
    if not os.path.exists(f'{index_dir}/tree00000.srt'):
        print('[calibrate] no .srt files yet (Phase 1 in progress) — skip', flush=True)
        return 0
    Q, GT = read_calib_bin(index_dir, dim)
    if Q is None or len(Q) < 5:
        print('[calibrate] no calibration data (queries+gt), skip', flush=True)
        return 0

    print(f'[calibrate] loaded {len(Q)} calib queries, dim={dim}', flush=True)
    t_start = time.time()

    # Grid réduit : tp ∈ {768, 1024}, mlb ∈ {200k, 300k, 400k}
    tps  = [768, 1024]
    mlbs = [200_000, 300_000, 400_000]
    STOP_DELTA = 0.005

    results = []
    for tp in tps:
        prev_recall = 0.0
        for mlb in mlbs:
            r = bench_config(index_dir, meta, Q, GT, tp, mlb, top_n=6000)
            r['tp'] = tp; r['mlb'] = mlb
            print(f'  tp={tp} mlb={mlb//1000}k → recall={r["recall"]:.3f} '
                  f'lat={r["mean_ms"]:.0f}ms peak={r["peak_mb"]:.0f}MB', flush=True)
            results.append(r)
            if r['recall'] - prev_recall < STOP_DELTA and prev_recall > 0:
                print(f'    ↳ plateau ({r["recall"]-prev_recall:+.4f}), stop tp={tp}', flush=True)
                break
            prev_recall = r['recall']

    # Config optimale = pareto (max recall, min lat)
    best = max(results, key=lambda x: (x['recall'], -x['mean_ms']))
    print(f'[calibrate] best : tp={best["tp"]} mlb={best["mlb"]//1000}k '
          f'recall={best["recall"]:.3f} lat={best["mean_ms"]:.0f}ms', flush=True)

    # Top-n sweep sur config optimale
    for top_n in (8000,):
        r = bench_config(index_dir, meta, Q, GT, best['tp'], best['mlb'], top_n)
        r['tp'] = best['tp']; r['mlb'] = best['mlb']; r['top_n'] = top_n
        print(f'  top_n={top_n} → recall={r["recall"]:.3f} lat={r["mean_ms"]:.0f}ms', flush=True)
        if r['recall'] > best['recall'] + STOP_DELTA:
            best = r
    best.setdefault('top_n', 6000)

    # Write recommended_config.json
    out = {
        'index': os.path.basename(index_dir.rstrip('/')),
        'n_docs_indexed': args.n_docs,
        'ts_unix': int(time.time()),
        'wall_calibrate_s': time.time() - t_start,
        'recommended': {
            'top_paths': int(best['tp']),
            'max_leaf_bytes': int(best['mlb']),
            'top_n': int(best.get('top_n', 6000)),
            'recall_at_10': round(best['recall'], 4),
            'mean_ms': round(best['mean_ms'], 1),
            'p50_ms': round(best['p50_ms'], 1),
            'peak_rss_mb': round(best['peak_mb'], 1),
        },
    }
    tmp = f'{index_dir}/recommended_config.json.tmp'
    dst = f'{index_dir}/recommended_config.json'
    with open(tmp, 'w') as fh:
        json.dump(out, fh, indent=2)
    os.replace(tmp, dst)   # atomic
    print(f'[calibrate] wrote {dst} in {time.time()-t_start:.1f}s', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
