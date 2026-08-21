"""Etape 1 v2 : medianes EXACTES niveau-synchrone sur DEEP 10M (96d),
depth 17, avec ablation de QUANTIZATION des seuils :
  f32 (4 o/noeud) vs f16 (2 o) vs int8 (1 o, grille affine par niveau).
Le build ROUTE avec le seuil quantifie (comme la requete et le live le
feront) — l'impact de la quantization est donc mesure de bout en bout.
Build exact 8M + injection 2M sur seuils geles. Arbres 0 et 1."""
import sys
import time

import numpy as np

sys.path.insert(0, "/root/mangrove-campaign/mangrove-search/scripts")
from mangrove.traversal import tree_seed, node_seed, pick_dims, gen_vec_v3

B = "/root/mangrove-campaign/data/deep50m.fbin"
N = 10_000_000
NB = 8_000_000
DEPTH = 17
DIM = 96

log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

hdr = np.fromfile(B, dtype=np.uint32, count=2)
assert hdr[1] == DIM, hdr
log("chargement DEEP 10M...")
X = np.memmap(B, dtype=np.float32, mode="r", offset=8,
              shape=(int(hdr[0]), DIM))[:N]
X = np.asarray(X)

ids_all = np.arange(N, dtype=np.uint64)
par = (((ids_all * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(37))
       & np.uint64(1)).astype(bool)


def quantize(thetas_level, mode):
    """Snap les medianes d'un niveau au format de stockage choisi."""
    th = np.asarray(thetas_level, dtype=np.float32)
    if mode == "f32":
        return th
    if mode == "f16":
        return th.astype(np.float16).astype(np.float32)
    # int8 : grille affine par niveau
    lo, hi = float(th.min()), float(th.max())
    if hi <= lo:
        return th
    q = np.round((th - lo) / (hi - lo) * 254.0)
    return (q / 254.0 * (hi - lo) + lo).astype(np.float32)


def build_exact(doc_ids, ts, mode):
    planes_cache = {}

    def planes(node):
        hit = planes_cache.get(node)
        if hit is None:
            dims = np.asarray(pick_dims(ts, node, DIM, 16), dtype=np.int64)
            v0 = gen_vec_v3(node_seed(ts, node * 2), 16)
            v1 = gen_vec_v3(node_seed(ts, node * 2 + 1), 16)
            hit = (dims, v1 - v0)
            planes_cache[node] = hit
        return hit

    node0 = np.zeros(len(doc_ids), dtype=np.int64)
    thetas = {}
    for level in range(DEPTH):
        order = np.argsort(node0, kind="stable")
        sorted_nodes = node0[order]
        uniq = np.unique(sorted_nodes)
        bounds = np.append(np.searchsorted(sorted_nodes, uniq), len(order))
        # 1er passage : medianes exactes du niveau
        raw_th = np.empty(len(uniq), dtype=np.float32)
        projs = [None] * len(uniq)
        sels = [None] * len(uniq)
        for k in range(len(uniq)):
            nd = int(uniq[k])
            sel = order[bounds[k]:bounds[k + 1]]
            dims, w = planes(nd)
            proj = X[doc_ids[sel]][:, dims] @ w
            raw_th[k] = np.median(proj)
            projs[k] = proj
            sels[k] = sel
        # 2e passage : snap + routage AVEC le seuil quantifie
        snap = quantize(raw_th, mode)
        for k in range(len(uniq)):
            nd = int(uniq[k])
            th = float(snap[k])
            thetas[nd] = th
            proj, sel = projs[k], sels[k]
            right = proj > th
            tie = proj == th
            if tie.any():
                right = right | (tie & par[doc_ids[sel]])
            node0[sel] = 2 * nd + 1 + right.astype(np.int64)
    return node0 - ((1 << DEPTH) - 1), thetas, planes


def route_frozen(doc_ids, thetas, planes):
    node0 = np.zeros(len(doc_ids), dtype=np.int64)
    for level in range(DEPTH):
        order = np.argsort(node0, kind="stable")
        sorted_nodes = node0[order]
        uniq = np.unique(sorted_nodes)
        bounds = np.append(np.searchsorted(sorted_nodes, uniq), len(order))
        for k in range(len(uniq)):
            nd = int(uniq[k])
            sel = order[bounds[k]:bounds[k + 1]]
            dims, w = planes(nd)
            proj = X[doc_ids[sel]][:, dims] @ w
            th = thetas.get(nd, 0.0)
            right = proj > th
            tie = proj == th
            if tie.any():
                right = right | (tie & par[doc_ids[sel]])
            node0[sel] = 2 * nd + 1 + right.astype(np.int64)
    return node0 - ((1 << DEPTH) - 1)


def report(label, counts, n):
    mean = n / (1 << DEPTH)
    ov = []
    for mult in (1.25, 1.5, 2.0):
        slot = int(np.ceil(mean * mult))
        ov.append(f"x{mult}: {np.maximum(counts - slot, 0).sum() / n:.2%}")
    print(f"{label}: p50 {int(np.percentile(counts, 50))} "
          f"p99 {int(np.percentile(counts, 99))} max {counts.max()} "
          f"vides {(counts == 0).mean():.2%} | debord "
          + " | ".join(ov), flush=True)


build_ids = np.arange(NB, dtype=np.int64)
live_ids = np.arange(NB, N, dtype=np.int64)
for t in (0, 1):
    ts = tree_seed(t)
    for mode in ("f32", "f16", "int8"):
        t0 = time.time()
        leaf_b, thetas, planes = build_exact(build_ids, ts, mode)
        cb = np.bincount(leaf_b, minlength=1 << DEPTH)
        report(f"arbre {t} {mode:4s} BUILD 8M (moy 61)", cb, NB)
        leaf_l = route_frozen(live_ids, thetas, planes)
        cl = cb + np.bincount(leaf_l, minlength=1 << DEPTH)
        report(f"arbre {t} {mode:4s} +2M live (moy 76)", cl, N)
        log(f"  ({time.time()-t0:.0f}s)")
print("DEEPMEDQ OK", flush=True)
