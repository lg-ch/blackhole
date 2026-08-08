---
name: feedback-recall-levers
description: "Pour gagner du recall : augmenter top_n d'abord (cheap), baisser query_depth en dernier recours (cher)"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 0e757620-83dd-4e01-864c-56db0c985e00
---

**Règle opérationnelle : `top_n >> query_depth` comme levier recall ↔ latence.**

Quand on veut plus de recall sur un index RP-forest avec votes :

1. **D'ABORD : augmenter `top_n`** (nombre de candidats à reranker)
   - Coût marginal par doc supplémentaire : ~2-3 µs (1 random 4-KB read + 1 L2 NEON)
   - Phase 1+2 io_uring INCHANGÉE (mêmes posting lists lues)
   - K-way merge et qsort INCHANGÉS (mêmes docs)
   - Seule la rerank L2 grossit linéairement avec top_n

2. **EN DERNIER RECOURS : baisser `query_depth`** (lookup en range parent)
   - Coût marginal par baisse d'1 niveau : ~50-150 ms (~30× plus cher que top_n × 2)
   - Phase 2 IO grossit (~2× payload utile par baisse), qsort grossit, K-way merge grossit
   - Beaucoup de docs lus/triés/mergés finissent par ÊTRE REJETÉS par le top-N voting → IO et CPU gaspillés
   - Mais inévitable quand le `top_n` n'atteint plus le recall cible (les votes saturent → le rerank ne discrimine plus parmi les top candidats)

**Mesures validées 2026-05-16 (SIFT 10M multi-index 4 sous-corpora à d=22) :**

| Pour atteindre recall... | Path A (qd ↓) | Path B (top_n ↑) | Ratio gain |
|---|---|---|---|
| 98.08 % | qd=22, top_n=500 → 145 ms | qd=22, top_n=500 → 145 ms | — |
| 99.0 %+ | qd=20, top_n=500 → 286 ms | qd=22, top_n=1000 → 148 ms | **×1.9 plus rapide** |
| 99.4 %+ | qd=19, top_n=500 → 398 ms | qd=22, top_n=2000 → 150 ms | **×2.6 plus rapide** |
| 99.5 %+ | qd ≤ 18, latence > 500 ms | qd=22, top_n=4000 → 154 ms | **×3 plus rapide** |

→ Bumper top_n de 500 à 4000 = +9 ms et +1.7 pt recall. Baisser qd de 1 niveau = +60-250 ms pour +0.5-1.4 pt.

**Mécanisme :**
- Les votes à qd natif (sweet spot 2-4 docs/leaf) sont **déjà bien discriminants** : vrais NN à 300-500 votes/1024, bruit à 3-15.
- Le top-N par vote en sélectionne déjà les meilleurs candidats. Augmenter top_n laisse simplement le rerank départager plus de candidats déjà bien-classés.
- Baisser qd "dilue" les votes (chaque feuille a plus de docs, mais avec des votes plus faibles). On AJOUTE BEAUCOUP DE BRUIT et on doit alourdir le pipeline pour le filtrer.

**Quand top_n n'est plus suffisant :**
- Si recall plafonne (par exemple ~99 % à 1.5 docs/leaf), bumper top_n n'aide plus parce que le **plancher est inhérent à la sparseness** (votes uniforme-1 sur le bruit, top-N indissociable).
- Dans ce cas, soit baisser qd (Path A), soit utiliser un build à depth plus modéré (sub-qd target 2-4 docs/leaf).

**Découverte 2026-05-16 — Native query à depth EXTREME = fast lane viable.**

À SIFT 1M / d=30 (0.03 % de leaves non-vides au lieu de 10 % à d=18) :

| Query | Recall@10 | Latence | Notes |
|---|---|---|---|
| Native qd=30, top_n=500  | 93.12 % | **8.5 ms** | sub_dim concentration crée des "hot leaves" qui sauvent le pool |
| Native qd=30, top_n=4000 | 97.64 % | **11 ms**  | top_n bump pour 9× moins cher que d=18 |
| Artificial qd=18 sur d=30 | 99.76 % | 126 ms | matche native d=18 exactement |

**Le native query à high depth ne s'effondre PAS à 0** comme on aurait pu craindre — la concentration sub_dim crée des leaves "hot" qui agrègent les docs voisins. À native d=30 / 1M, recall plafonne ~98 % avec top_n élevé, mais la latence est **dramatiquement plus basse** que d=18 native parce que chaque leaf est minuscule (peu de candidats à traiter).

**Pattern d'usage "deux lanes" sur le même index :**
- **Fast lane** : native query au depth de build (élevé) + top_n agressif → ~10 ms / 97-98 %
- **Premium lane** : artificial query_depth lower + top_n moderate → ~100-150 ms / 99.5-99.8 %

L'utilisateur choisit la lane à la query, **même index, deux régimes**. Particulièrement utile pour des produits où la majorité des queries acceptent 97 % en 10 ms, et où certaines queries premium veulent 99.5 % en plus de latence.

**Limite int32 : max depth = 30** avec le code actuel (`Pair.leaf_id` et `traverse` retour sont int32). Pour d > 30, refactor int64 (~50-100 lignes).

**⚠ Important : la règle `top_n >> query_depth` est SIFT-spécifique. À HAUTE DIMENSION elle s'inverse.**

**Signal prédictif AVANT de sweeper : `sub_dim / dim`.** Validation arxiv vs SIFT (mêmes params build) :

| | sub_dim/dim | R∞ à qd native (top_n=128k) | qd cible 0.99 |
|---|---:|---:|---:|
| SIFT 1M dim=128 sd16 | 12.5% | 0.998 | native (build-0) |
| arxiv 2M dim=768 sd16 | 2.1%  | 0.836 | build-6 |

- **`sub_dim/dim ≥ 10%`** → forest "comprend" la géométrie → R∞ ≥ 0.99 → qd=native suffit, descendre qd est gaspillage.
- **`sub_dim/dim < 5%`** → hyperplanes quasi-random → plafond < 0.99 → on doit structurellement baisser qd à query time (build − 4 à − 8).

C'est observable avant tout build : pour un dataset dim X et un budget RAM, choisir sub_dim ≥ 0.1×dim si recall@native est la priorité.

**Formule prédictive du qd_target via `n_distinct_cands` (observable au query time, sans GT) :**

Le K-way merge expose `n_distinct` = nombre de doc_ids distincts visités. Mesure ajoutée dans `forest_get_last_n_distinct()` (query_tree.c) et `eval_recall_topn` la moyenne.

Validation arxiv (N=2.06M) vs SIFT (N=1M), sd=16 :

| qd | arxiv n_distinct | %corpus | recall | SIFT n_distinct | %corpus | recall |
|---:|---:|---:|---:|---:|---:|---:|
| build native | 13 648 | 0.66% | 0.667 | 434 380 | 43.4% | 0.998 |
| build-2 | 41 157 | 2.0% | 0.830 | – | – | – |
| build-4 | 123 598 | 6.0% | 0.917 | – | – | – |
| build-6 | 354 945 | 17.2% | 0.973 | – | – | – |
| build-8 | 897 830 | 43.6% | 0.994 | – | – | – |

**Règle observée** : pour recall ≥ 0.99 il faut `n_distinct ≈ 0.4 × N_corpus` quel que soit le dataset.

**Formule** :
```
qd_target = build_depth − log2( 0.4 × N_corpus / n_distinct_native )
```

Vérif arxiv : log2(0.4 × 2.06M / 13 648) = log2(60) ≈ 5.9 → qd=14 → recall 0.973 ✓ (proche cible).

**Protocole nouveau dataset** :
1. 5 queries au qd=build_depth → moyenne `n_distinct_native` (~100 ms, no GT).
2. Calculer qd_target.
3. Confirmer avec GT bruteforce sur 50 queries (~1 min).

Sub_dim/dim détermine `n_distinct_native` : plus le ratio est haut, plus les NN sont co-routés → cands "hot" (lots de duplicates entre trees) → distinct count élevé au native.

**Implémentation `--auto_qd_v2` (2026-05-18, main.c) :**

Calibration 2-probes pour mesurer le **vrai facteur de doublement** par level (théorique 2×, en pratique 1.5-1.7× sur arxiv à cause de la saturation par déduplication inter-trees) :

```
probe0 = avg_distinct(qd = build_depth, 5 queries)
probe2 = avg_distinct(qd = build_depth - 2, 5 queries)
factor_per_level = (probe2 / probe0)^(1/2)
qd_target = build_depth − ceil(log(target / probe0) / log(factor_per_level))
target = target_ratio × N_corpus   (filter-AGNOSTIC, voir ci-dessous)
```

CLI :  `--auto_qd_v2 [--target_ratio 0.4] [--probe_n 5]`

Validation arxiv + SIFT (un seul flag, cross-dataset) :

| | factor/lvl mesuré | qd choisi | Recall |
|---|---:|---:|---:|
| SIFT 1M sd16 | 1.10 (~saturé) | qd=18 native | 0.998 |
| arxiv 2M sd16 no filter | 1.74 | qd=12 (build-8) | 0.994 |
| arxiv + filter year=2007 (2%) | 1.74 | qd=12 | 0.945 |
| arxiv + filter cs.LG (0.8%) | 1.74 | qd=12 | 0.766 (top_n cap) |

**Target (corrigé après filter-aware merge)** : `target = ratio × filter_card` quand filtré, `ratio × N_corpus` sinon. **Important** : depuis que le K-way merge est filter-aware (cursor_seek_allowed dans query_tree.c), `n_distinct` ne compte QUE les docs allowed visités. Le target doit donc être en unité filter_card. Avant filter-aware merge, n_distinct était filter-agnostic donc target=N_corpus était correct — les deux changes doivent évoluer ensemble.

**Plafond du filter ultra-sparse** : pour cs.LG (16k docs) avec qd=12 (44% du filter visité), recall plafonne à 0.77. C'est intrinsèque : 23% des vrais NN sur cs.LG sont dans des leaves natives qu'aucun tree ne visite à qd=12 pour cette query (curse of dim × sub_dim/dim faible). Lever : qd plus bas (cher) ou bruteforce direct si filter_card × dim × 4 < ~50 MB.

**Compensation sub-corpus dans auto_qd_v2 (2026-05-18) :**

Quand le filter est sparse, viser `ratio × filter_card` cands visités donne le bon effort budget MAIS le forest est sub-optimal sur sub-corpora arbitraires (ses hyperplanes optimisent la géométrie globale). Levé en pénalisant qd d'extra niveaux :

```
extra_levels = max(0, floor(log2(N_corpus / filter_card)) - 1)
qd_final = qd_base - extra_levels
```

Validation arxiv 2M dim=768 :

| Filter | density | extra | qd_final | Recall (sans → avec) |
|---|---:|---:|---:|---:|
| year=2007 | 2.06% | 4 | 8 | 0.945 → 0.999 |
| cs.LG | 0.79% | 5 | 7 | 0.766 → 0.988 |

CLI : actif par défaut avec `--auto_qd_v2 + filter`. Désactivable via `--no_filter_compensate`.

**Mode 2 alternatif (`--filter_comp_mode 2`) : `extra = floor(log2(N/F)/2)`** — formule plus douce, ~moitié des levels.

Comparaison arxiv 2M (100 queries) :

| Filter | Mode 1 (−1) | Mode 2 (/2) | BF |
|---|---|---|---|
| year=2007 | 0.999 / 607 ms | **0.991 / 163 ms** | 0.999 / 77 ms |
| cs.LG | 0.988 / 1198 ms | 0.933 / 325 ms | 1.000 / 34 ms |

Mode 2 = bon trade-off pour filter ~2 % (year-like) : −0.008 recall pour /3.7 latence. Mais pénalise les filter ultra-sparse (cs.LG : −0.055 recall pour /3.7 latence). Sur arxiv 2M, BF reste optimal partout.

**Trade-off latence** : ×12 à ×24 sur arxiv 2M (612 ms / 1206 ms). Sur ce calibre, **bruteforce reste meilleur** (77 ms / 34 ms à recall 1.0). La compensation est essentielle à plus grande échelle où le bruteforce n'est plus viable : à arxiv 100M, filter 2% = 6 GB → BF impossible, forest + compensation est la seule voie.

**Routing recommandé (3 régimes)** :
- subset < 50 MB → bruteforce (recall 1.0, <80 ms)
- 50 MB - 5 GB → forest + compensation (recall ~0.99)
- > 5 GB → forest sans compensation (best effort)

Mesures arxiv 2M / dim=768 / sub_dim=16 / depth=20 / 1000 trees (2026-05-18) :

| Levier | top_n=500 | top_n=8000 | qd=18 / tn=2000 | qd=16 / tn=2000 | qd=14 / tn=2000 |
|---|---|---|---|---|---|
| Recall@10 | 0.634 | 0.760 | 0.830 | 0.917 | 0.973 |
| ms/q | 6 | 18 | 9 | 14 | 30 |

→ Sur dim=768, **qd 20→16 fait gagner +28 pts** pour 14 ms vs top_n 500→8000 qui gagne +13 pts pour 18 ms. **Query_depth domine top_n** en haute dim.

**Mécanisme** : à dim=768, sub_dim=16 = seulement 2% des dims utilisées par hyperplane. Les splits sont peu discriminants, donc à depth native les vrais NN se retrouvent souvent dans des leaves voisines (pas dans la leaf du query). Élargir le scope (qd ↓) capture beaucoup plus de NN. À dim=128 + sub_dim=8 (6% des dims), les splits restent assez bons pour que top_n native suffise.

**Pratique high-dim** : sweet spot `qd = build_depth - 4 to -6`, top_n=2000.

**Multi-index `multitopn_auto` (2026-05-18, main.c) :**

Extension naturelle aux multi-index : chaque forest a son propre `qd_i` calculé par probe sur ses propres stats `n_distinct_native`. Pertinent quand les sous-corpora ont des profils différents (factor_per_level, intrinsic dimensionality). Implémenté en voie B (N appels séquentiels à `forest_collect_topn`, perd l'efficience pairwise-seed). Voie A (traversée partagée + cursors hétérogènes à qd_i différents) à faire pour la prod.

Cas d'usage : multi-modal (texte + images), tailles très inégales (10M + 100k docs), ou filter qui force compensation locale par forest.

Lien : [[feedback-depth-vote-discrimination]] sur le rôle de la depth dans la qualité des votes. [[project-rpforest-sift]] pour le contexte SIFT. [[project-arxiv-2m-clickhouse]] pour le contexte dim=768.
