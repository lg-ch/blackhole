"""Simulation du build v2 : medianes EXACTES niveau-synchrone + tie-break
par hash d'id, un arbre, LAION-10M. Build exact sur 8M puis injection live
des 2M restants par seuils geles — equilibre avant/apres + promotions de
classes. A comparer au mesure actuel (med17 echantillonne : p50=4/moy 76,
max 21 609, >50 % de debordement a slot 3x)."""
import sys
import time

import numpy as np

sys.path.insert(0, "/root/mangrove-campaign/mangrove-search/scripts")
from mangrove.traversal import tree_seed, node_seed, pick_dims, gen_vec_v3

D = "/root/laion10m"
B = D + "/laion10m.f16bin"
N = 10_000_000
NB = 8_000_000          # build
DEPTH = 17
TS = tree_seed(0)

log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

log("chargement base...")
X = np.fromfile(B, dtype=np.float16, offset=8,
                count=N * 512).reshape(N, 512).astype(np.float32)

# hash parite deterministe par doc (tie-break)
ids_all = np.arange(N, dtype=np.uint64)
par = (((ids_all * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(37))
       & np.uint64(1)).astype(bool)

node_planes = {}


def planes(node):
    hit = node_planes.get(node)
    if hit is None:
        dims = np.asarray(pick_dims(TS, node, 512, 16), dtype=np.int64)
        v0 = gen_vec_v3(node_seed(TS, node * 2), 16)
        v1 = gen_vec_v3(node_seed(TS, node * 2 + 1), 16)
        hit = (dims, v1 - v0)          # proj = x[dims] @ (v1 - v0)
        node_planes[node] = hit
    return hit


def build_exact(doc_ids):
    """Retourne (assign niveau DEPTH, thetas dict node->theta)."""
    pos = np.zeros(len(doc_ids), dtype=np.int64)   # position dans le niveau
    node0 = np.zeros(len(doc_ids), dtype=np.int64)  # id heap du noeud courant
    thetas = {}
    order = np.argsort(pos, kind="stable")
    for level in range(DEPTH):
        t0 = time.time()
        # groupe par noeud courant
        order = np.argsort(node0, kind="stable")
        sorted_nodes = node0[order]
        bounds = np.searchsorted(sorted_nodes,
                                 np.unique(sorted_nodes))
        uniq = np.unique(sorted_nodes)
        bounds = np.append(np.searchsorted(sorted_nodes, uniq), len(order))
        for k in range(len(uniq)):
            nd = int(uniq[k])
            sel = order[bounds[k]:bounds[k + 1]]
            dims, w = planes(nd)
            proj = X[doc_ids[sel]][:, dims] @ w
            th = float(np.median(proj))
            thetas[nd] = th
            right = proj > th
            tie = proj == th
            if tie.any():
                right = right | (tie & par[doc_ids[sel]])
            node0[sel] = 2 * nd + 1 + right.astype(np.int64)
        if level % 4 == 0:
            log(f"  build niveau {level} ({len(uniq)} noeuds, "
                f"{time.time()-t0:.0f}s)")
    base = (1 << DEPTH) - 1
    return node0 - base, thetas


def route_frozen(doc_ids, thetas):
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
            th = thetas.get(nd)
            if th is None:      # noeud jamais vu au build : sign-split
                th = 0.0
            right = proj > th
            tie = proj == th
            if tie.any():
                right = right | (tie & par[doc_ids[sel]])
            node0[sel] = 2 * nd + 1 + right.astype(np.int64)
    return node0 - ((1 << DEPTH) - 1)


def report(label, counts, n):
    mean = n / (1 << DEPTH)
    line = (f"{label}: moy {mean:.1f}  p50 {int(np.percentile(counts, 50))}  "
            f"p99 {int(np.percentile(counts, 99))}  max {counts.max()}  "
            f"vides {(counts == 0).mean():.1%}")
    ov = []
    for mult in (1.5, 2.0, 3.0):
        slot = int(np.ceil(mean * mult))
        ov.append(f"x{mult}: {np.maximum(counts - slot, 0).sum() / n:.2%}")
    print(line + "\n    debordement slots " + " | ".join(ov), flush=True)


build_ids = np.arange(NB, dtype=np.int64)
live_ids = np.arange(NB, N, dtype=np.int64)

t0 = time.time()
leaf_b, thetas = build_exact(build_ids)
log(f"build exact fini en {time.time()-t0:.0f}s "
    f"({len(thetas):,} seuils)")
cb = np.bincount(leaf_b, minlength=1 << DEPTH)
report("BUILD 8M (medianes exactes + tie-break)", cb, NB)

t0 = time.time()
leaf_l = route_frozen(live_ids, thetas)
log(f"injection 2M routee en {time.time()-t0:.0f}s")
cl = cb + np.bincount(leaf_l, minlength=1 << DEPTH)
report("APRES +2M live (seuils geles)", cl, N)

# promotions de classes (anneaux 32/128/512/2048 docs)
classes = np.array([32, 128, 512, 2048, 1 << 30])
cls_b = np.searchsorted(classes, cb, side="left")
cls_l = np.searchsorted(classes, cl, side="left")
promo = (cls_l > cls_b).sum()
print(f"promotions de classe pendant l injection : {promo:,} feuilles "
      f"sur {1 << DEPTH:,} ({promo / (1 << DEPTH):.2%})", flush=True)
print("SIM OK", flush=True)
