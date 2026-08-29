"""EMPREINTE STRUCTURELLE (idee : L. Chartier) — le code d'appartenance
aux feuilles comme quantizer gratuit.

Chaque doc possede un code = sa feuille dans chacun des T arbres
(17-20 bits x T). Similarite candidat/requete = somme sur les arbres du
prefixe commun entre le code du doc et le chemin de la requete
(XOR + position du bit de poids fort) — aucune donnee apprise, aucun
centroide : l'index EST le quantizer.

Mesure DEEP 10M (pool reel query_v2 300k, rerank exact apres selection) :
  garde  500 : votes binaires 0.925  |  CODE PLEIN 0.992
  garde 1000 : votes 0.937          |  CODE PLEIN 0.995 (plafond pool)
  garde 3000 : votes 0.957          |  CODE PLEIN 0.995
=> rerank de 500 suffit : vague de rerank /12. Deploiement : table codes
64 arbres x 20 bits = 160 o/doc ; pipeline votes -> codes -> rerank 500.
NB : la variante sans stockage (prefixe intra-range, arbres visites
seulement) ne vaut que +0.9 pt — le signal vient des arbres NON visites.
"""
import sys
import time

import numpy as np

sys.path.insert(0, "/root/mangrove-campaign/mangrove-search/scripts")
import mangrove_ffi as mf
from mangrove_ffi import Forest, slots_v2_open, query_v2

D = "/root/deep10m"
BN = D + "/deep10m.fbin"
N = 10_000_000
DEPTH = 17

f = Forest(D + "/idx_exact17", n_trees=256, dim=96, sub_dim=16,
           depth=DEPTH, n_docs=N)
assert mf.load_live_medians(D + "/idx_exact17") == DEPTH
sh = slots_v2_open(D + "/slots17b", 256, 1 << DEPTH, 512)
Q = np.load("/root/mangrove-campaign/data/queries.npy").astype(np.float32)
gt = np.load("/root/mangrove-campaign/data/gt_10m.npy")
X = np.memmap(BN, dtype=np.float32, mode="r", offset=8, shape=(N, 96))

print("codes 10M x 256 (leaf par arbre)...", flush=True)
t0 = time.time()
codes = np.empty((N, 256), dtype=np.int32)
for off in range(0, N, 1_000_000):
    codes[off:off + 1_000_000] = mf.traverse_batch(
        np.ascontiguousarray(X[off:off + 1_000_000]), 16, DEPTH, 256)
print(f"codes en {time.time()-t0:.0f}s", flush=True)


def prefix_score(pool, qpath):
    x = (codes[pool].astype(np.int64)
         ^ qpath.astype(np.int64)[None, :]).astype(np.uint32)
    msb = np.zeros(x.shape, dtype=np.int32)
    nz = x > 0
    msb[nz] = np.floor(np.log2(x[nz])).astype(np.int32) + 1
    return (DEPTH - msb).sum(axis=1)


for keep in (500, 1000, 3000):
    rv, rp = [], []
    for qi in range(60):
        qn = (Q[qi] / np.linalg.norm(Q[qi])).astype(np.float32)
        qpath = mf.traverse_batch(np.ascontiguousarray(qn[None, :]),
                                  16, DEPTH, 256)[0]
        ids, votes, n = query_v2(f, sh, Q[qi], 3, 1024, 300000, 15)
        pool = ids[:n].astype(np.int64)
        selv = pool[np.argsort(-votes[:n].astype(np.int64),
                               kind="stable")[:keep]]
        ps = prefix_score(pool, qpath)
        selp = pool[np.argsort(-ps, kind="stable")[:keep]]
        for sel, acc in ((selv, rv), (selp, rp)):
            v = X[sel].astype(np.float32)
            d2 = np.einsum("ij,ij->i", v, v) - 2.0 * (v @ Q[qi])
            top = sel[np.argsort(d2)[:10]]
            acc.append(len(set(top.tolist())
                           & set(gt[qi].tolist())) / 10.0)
    print(f"garde {keep:5d} : votes {np.mean(rv):.3f}  |  "
          f"CODE PLEIN {np.mean(rp):.3f}", flush=True)
print("FULLCODE OK", flush=True)
