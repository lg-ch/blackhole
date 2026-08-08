---
name: Leaf-coverage exploration — sub-dim sweep on SIFT 1M
description: In-progress exploration of the "pick K trees with smallest leaves per query" strategy in hd_ann_test.cpp, varying sub-dim / n_trees.
type: project
originSessionId: 6e797379-f7e5-4dc5-9845-9f6b9a13da79
---
Axe d'expérimentation ouvert : pour chaque query, prendre les K arbres dont la feuille touchée est la plus petite (ou la plus profonde) → union des feuilles = candidats. Objectif : atteindre 0.98 recall@10 avec K ≤ 500 arbres "utiles" (et ensuite un total d'arbres raisonnable).

**Modifs déjà faites dans `hd_ann_test.cpp` :**
- `analyze_leaf_coverage` : tri des arbres changé de `(depth desc, size asc)` → `(size asc, depth desc)`.
- `K_list` dans le main hardcodée à `{100, 200, 300, 400, 500}` (au lieu de `{1,2,5,...,1024}`).
- Commentaires et printf "[leaf-cov]" mis à jour pour refléter le nouveau tri.

**Runs faits :**
- 2026-04-22 run #1 : SIFT 1M, 2048 trees, sub-dim=8, **max_leaf=32**, depth=25, rademacher, split=median (défaut). Build 99.6s.
  - K=100→0.629 | 200→0.810 | 300→0.893 | 400→0.937 | 500→0.959
  - Feuilles saturées à max_leaf=32 (mean 30.5, p99=31) → tri "plus petite feuille" presque random. Plafonne à 0.959 à K=500.
- 2026-04-22 run #2 : SIFT 1M, 2048 trees, sub-dim=8, **max_leaf=0** (pas de cap), depth=16, rademacher, **split=sign**. Build 110s.
  - K=100→0.460 | 200→0.769 | 300→0.909 | 400→0.969 | **500→0.989** ✓ objectif 98% atteint
  - Feuilles très variables (mean 3932, p50 1148, p99 38k, max 138k) → tri "smallest leaf" vraiment discriminant.
  - 0.34% empty leaves. Trade-off: 4× plus de candidats à rescorer que config max_leaf=32 (54k vs 13k à K=500) pour +3 pts recall.

**Idée validée** : build 2048 arbres "cheap" (sign-split = pas de nth_element), puis pour chaque query sélectionner K=500 arbres avec les plus petites feuilles → union des candidats → 0.989 recall@10.

**Piège noté** : `n_pts <= max_leaf` ⇒ si on passe `max_leaf=1000000` sur N=1M le root devient feuille et aucun split ne se fait (recall 1.0 trivial car tous les candidats). Pour désactiver le cap : `--max-leaf 0`.

**Sweep depth (2026-04-22, 4096 trees sd8 sign) :**
- d22 : K=1400→recall=0.989, 29.6k unique, 44k sum_leaf (RSS ~65 GB)
- d24 : K=1600→recall=0.990, 24.4k unique, 36k sum_leaf (RSS ~85 GB, build 267s)
- d26+ : **OOM sur machine 119 GB** (d26 nécessite ~122 GB). Extrapolation : 65→85→122, progression à peu près exponentielle post-saturation.
- d24 Pareto-bat d22 : −17 % sur unique_cand ET −18 % sur sum_leaf à iso-recall 0.989, au prix de +200 K (1400→1600 arbres traversés).
- Pour explorer d26+ à 4096 trees : soit réduire à 2048 trees, soit activer le build streaming disque (existe dans `hd_ann_test.cpp`).

**Pistes pour la suite (non faites) :**
1. Comparer au build dim=16 équivalent (1024 trees sub-dim=16) — point de référence mentionné par l'user, paramètres exacts inconnus.
2. Faire leaf-coverage en mode `--load-from` pour futurs builds 10M/100M (actuellement uniquement in-RAM). `base.u8bin` (100M) est déjà disponible dans le working dir.
3. Mesurer le coût du rescoring avec med_ml8 en QPS réel (pas encore fait — leaf-coverage ne donne que recall + sum_leaf_sizes, pas la latence).
4. Intégrer le "pick K trees by smallest-leaf" dans `search_forest` (mode query réel) — actuellement c'est juste un outil d'analyse.

**Sweep 2026-04-22 (`leaf_coverage_sweep.cpp`) :**
- Pareto optimal trouvé : med_ml8 K≈1800 → 0.98 recall @ 13.5k sum (4-6× mieux que sign-split ~60-80k).
- sign-split plafonne à ~63k sum pour 0.98, indépendamment de sub_dim/depth/n_trees.
- Confirme : l'hypothèse "plancher 50k" ne tient que pour sign-split, pas pour l'architecture médian+small-leaf.
- Le tri "plus petite feuille" n'est qu'un pis-aller pour compenser la variance de sign-split. En median le tri devient sans objet (feuilles uniformes) et on gagne quand même.
- sub_dim est quasi-neutre au-delà de 8 (sd8=sd16=sd32 à iso-sum pour sign).
- **sub_dim=4 est strictement dominé par sub_dim=8** (run 2026-04-22, 4096 trees, sign, d22, K∈{100..2048}). À iso-recall 0.989 : sd4 demande K=1400 / 44k candidats vs sd8 K=900 / 31k. Moins de discrimination par arbre ⇒ union plus large. Plancher pratique : sub_dim=8.
- **GT cosine vs GT L2 réelle (SIFT 1M, 2026-04-22)** : `recall(cos_top10 vs real_top10) = 0.9920` en absolu, mais `recall_cos ≈ recall_real` à chaque K de leaf-coverage (delta ≤ 0.0005). La sélection par leaf-size est metric-agnostic : l'union de candidats couvre les top-k des deux métriques. À K=2000, recall_real=0.9996 dépasse le plafond cos-only (0.9920), preuve que l'union n'est pas limitée par la topologie cosine. Impact pratique : bencher contre la GT cos en dev est safe, rescorer en L2 à la fin si la métrique cible est L2. Fichier GT officiel : `sift/sift_groundtruth.ivecs` (10k queries × 100 voisins L2). Le sweep lit ivecs via `read_ivecs_topk` dans `leaf_coverage_sweep.cpp`. Commande : `/tmp/leaf_sweep sift/sift_base.fvecs sift/sift_query.fvecs 1000 sift/sift_groundtruth.ivecs`.
- **sub_dim=16 vs sub_dim=8 : rendements décroissants, pas de trade-off** (replay clean 2026-04-22, 4096 trees, sign, d22, sum_leaf ET unique tabulés). À K=1400 : sd8 = 44.1k sum / 29.6k uniq / recall 0.990 ; sd16 = 41.6k / 27.4k / 0.991 → sd16 gagne uniformément −6% sum et −7% unique. À comparer aux gains sd4→sd8 (~−32% sur les deux axes). Build sd16 = 343s (+35% vs sd8). Verdict : sd8 reste le sweet spot par Pareto-efficiency ; sd16 paie +35% build pour seulement −7% candidats. **Correction importante** : une version antérieure de cette mémoire indiquait un trade-off asymétrique sd16 vs sd8 — c'était une erreur de lecture (sum_leaf vs unique confondus dans le roadmap d'avant le replay).
- Outputs complets : `/tmp/leaf_sweep.log`. Fichier source : `leaf_coverage_sweep.cpp` (utilise `#define HD_ANN_NO_MAIN` + `#include "hd_ann_test.cpp"`).

**Contexte bench :**
- `base.u8bin` + `queries.u8bin` (BigANN 100M × 128 u8) disponibles dans le working dir mais on reste sur SIFT 1M pour l'exploration.
- `sift_base.fvecs` (1M × 128 f32), `sift_query.fvecs` (10k × 128 f32).

**Why:** L'user explore si un "routage" par qualité de feuille (plus petite = plus sélective) permet d'atteindre 0.98 recall avec peu d'arbres exploités par query, en buildant un gros pool (2048) et en sélectionnant dynamiquement.

**How to apply:** Quand on reprend ce fil, aller voir le roadmap.md (journal + table bench) pour les derniers runs ; relancer avec max_leaf bas d'abord (piste 1 ci-dessus) avant tout autre changement. Commande type :
`/tmp/hd_ann_test sift_base.fvecs sift_query.fvecs --nq 1000 --k 10 --n-trees 2048 --sub-dim 8 --gen rademacher --max-leaf 8 --depth 25 --leaf-coverage`
