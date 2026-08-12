"""Smoke test VECFMT_F16BIN : build médianes + query + rerank depuis un
fichier float16, parité de recall avec le même corpus en float32.

  [1] count/build/query : l'index construit depuis .f16bin donne le même
      recall@10 (±1 pt) que l'index construit depuis .fvecs f32
  [2] rerank_l2 sur .f16bin : la requête = un vecteur du corpus → son id
      doit sortir top-1 (lecture + conversion f16 correctes au rerank)
  [3] calibration médianes lit le .f16bin (échantillonnage stridé)
"""
import os
import shutil
import struct
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mangrove_ffi as mf                                   # noqa: E402
from mangrove_ffi import Forest, set_gen_version            # noqa: E402
import test_live_medians as T                               # noqa: E402

RPFOREST = os.path.join(os.path.dirname(HERE), 'rpforest')


def write_f16bin(path, arr):
    n, dim = arr.shape
    with open(path, 'wb') as fh:
        fh.write(struct.pack('<II', n, dim))
        fh.write(arr.astype(np.float16).tobytes())


def build(base_path, idir):
    os.makedirs(idir, exist_ok=True)
    set_gen_version(T.GEN_V)
    rc = mf._lib.mg_calibrate_medians(base_path.encode(), idir.encode(),
                                      T.N_TREES, T.DIM, T.SUB_DIM,
                                      T.MED_DEPTH, T.SAMPLE_N)
    assert rc == 0, f'calibrate rc={rc} ({base_path})'
    subprocess.run([RPFOREST, 'build', base_path, idir,
                    str(T.N_TREES), str(T.DEPTH),
                    '--sub_dim', str(T.SUB_DIM), '--gen', f'v{T.GEN_V}',
                    '--dim', str(T.DIM)], check=True, capture_output=True)


def bench_recall(idir, base_path, vecs16, Q, GT):
    set_gen_version(T.GEN_V)
    mf.clear_live_medians()
    mf.load_live_medians(idir)
    f = Forest(idir, n_trees=T.N_TREES, dim=T.DIM, sub_dim=T.SUB_DIM,
               depth=T.DEPTH, n_docs=T.N_DOCS, gen_version=T.GEN_V)
    rec = 0.0
    for qi in range(len(Q)):
        ids, votes, n = f.query_pathrank(Q[qi], 3, 64, 2000,
                                         query_depth=T.DEPTH)
        top10 = f.rerank_l2(base_path, Q[qi], ids[:n], 10)
        rec += len(set(int(x) for x in top10) &
                   set(int(x) for x in GT[qi])) / 10.0
    f.close()
    return rec / len(Q)


def main():
    tmp = tempfile.mkdtemp(prefix='mangrove_f16_')
    try:
        vecs = T.make_vecs(T.N_DOCS, T.DIM, seed=42)
        vecs16 = vecs.astype(np.float16).astype(np.float32)
        f32_path = os.path.join(tmp, 'base.fvecs')
        f16_path = os.path.join(tmp, 'base.f16bin')
        T.write_fvecs(f32_path, vecs)
        write_f16bin(f16_path, vecs)

        # GT brute force sur les valeurs RÉELLEMENT stockées (arrondi f16)
        rng = np.random.default_rng(7)
        qidx = rng.choice(T.N_DOCS, 40, replace=False)
        Q = vecs[qidx]
        d2 = ((vecs16[None, :, :] - Q[:, None, :]) ** 2).sum(-1) \
            if False else None
        GT = np.empty((40, 10), dtype=np.int64)
        for i, q in enumerate(Q):
            d = ((vecs16 - q) ** 2).sum(1)
            GT[i] = np.argsort(d)[:10]

        # [3] + [1] : build f16 et build f32, recalls comparés
        i16 = os.path.join(tmp, 'idx16')
        i32 = os.path.join(tmp, 'idx32')
        build(f16_path, i16)
        print('[3] calibration + build depuis f16bin : OK')
        build(f32_path, i32)
        r16 = bench_recall(i16, f16_path, vecs16, Q, GT)
        r32 = bench_recall(i32, f32_path, vecs16, Q, GT)
        print(f'[1] recall@10 : f16 {r16:.3f} vs f32 {r32:.3f} '
              f'(écart {abs(r16-r32):.3f})')

        # [2] rerank : requête = vecteur du corpus → top-1 = lui-même
        set_gen_version(T.GEN_V)
        mf.clear_live_medians()
        mf.load_live_medians(i16)
        f = Forest(i16, n_trees=T.N_TREES, dim=T.DIM, sub_dim=T.SUB_DIM,
                   depth=T.DEPTH, n_docs=T.N_DOCS, gen_version=T.GEN_V)
        self_top1 = 0
        for did in (17, 4242, 19999):
            ids, votes, n = f.query_pathrank(vecs[did], 3, 64, 2000,
                                             query_depth=T.DEPTH)
            top = f.rerank_l2(f16_path, vecs[did], ids[:n], 3)
            self_top1 += (len(top) > 0 and int(top[0]) == did)
        f.close()
        print(f'[2] rerank f16 self-top1 : {self_top1}/3')

        ok = abs(r16 - r32) < 0.02 and r16 > 0.9 and self_top1 == 3
        print('PASS' if ok else 'FAIL')
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
