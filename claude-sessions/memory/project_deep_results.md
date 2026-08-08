---
name: deep-results
description: DEEP 10M recall 1.000 (exact qd=18) / 0.995 TQ — fbin natif intégré
metadata: 
  node_type: memory
  type: project
  originSessionId: 0e757620-83dd-4e01-864c-56db0c985e00
---

DEEP 1B téléchargé (384 GB), build subset 10M (256t/depth22/sd16) + tq4 sidecar 60 GB (couvre tout le 1B). Bench 10K queries held-out + GT_10M officiel : **recall@10 = 1.000 exact à qd=18/10p/16k (627 ms cold)**, 0.995 TQ à qd=18/10p/16k (341 ms cold, -46%). Sur 10M docs dim 96.

Stack ajouté : VECFMT_FBIN dans `vec_format.h` (header 8 B + raw float32, format big-ann-benchmarks). build_tree, recall, tquant gèrent fbin nativement. `count_vecs` accepte fbins partiels (utile pour subsets pendant download).

**Why:** DEEP est le benchmark billion-scale "moderne" demandé (vs SIFT old-school). Datasets prep : `/mnt/mangrove/datasets/deep1b/base.1B.fbin` + queries.10K.fbin + gt.10K.bin (1B GT) + gt.10M.bin (subset GT).

**How to apply:**
- Pour DEEP 1B full bench (à venir) : build avec --doc_count 1000000000 (~5-6 h estimé) + bench avec gt.10K.bin et le tq4 existant.
- Bench script : `R&D/bench_deep_10m.py`. Réutilisable en swappant n_docs.
- Bug latent : touche header sans make → recall=0 silencieux. Voir [[makefile-header-deps]].
