---
name: project-arxiv-2m-clickhouse
description: "Stack arxiv 2M dim=768 + ClickHouse + CRoaring : setup pré-agg, bench bitmap-pré-aggrégé, p99 forest 7 ms, filter coute zéro hot path"
metadata: 
  node_type: memory
  type: project
  originSessionId: 0e757620-83dd-4e01-864c-56db0c985e00
---

# arXiv 2M / dim 768 + ClickHouse + CRoaring (état 2026-05-18)

## Fait (J1-J3, session 2026-05-18)

**Données** :
- 2_058_751 docs arxiv (2007-2024) depuis `bluuebunny/arxiv_embeddings_Alibaba-NLP_gte-base-en-v1.5`
- `datasets/arxiv/arxiv_base.fvecs` 6.33 GB (dim=768 fp32)
- ClickHouse `mangrove.docs` (internal_id, arxiv_id, year, primary_cat, top_cat)
- Pré-agg bitmaps : `mangrove.bm_by_{year, primary_cat, top_cat}` via `AggregatingMergeTree` + `groupBitmapState`
- Forest : `index_arxiv/` 1000 trees × depth 20 × sub_dim 16 × gen v3, 12 GB SRT, build 12 min peak RSS 36 MB

**Code livré** :
- `src/croaring_io.{h,c}` : décode CH state `[0x01][varint][CRoaring portable]` → `roaring_bitmap_t*`. Format validé externally via pyroaring (match count SQL).
- `src/query_tree.{h,c}` : API forest passée à `roaring_bitmap_t*` partout, hot path = `roaring_bitmap_contains`. Dense bitmap supprimé.
- `src/main.c` : nouvelle commande `rpforest search` (single-shot, sans GT), nouveaux flags `--filter_ch` et `--dim`.
- `scripts/prepare_arxiv.py` : stream parquets → fvecs + insert metadata + pré-agg.
- `scripts/search_arxiv.py` : orchestrator Python avec décision pre/post auto (densité 3%).
- `scripts/bench_arxiv.py` : bench 5 scénarios × N queries, percentiles p50/p95/p99/max.

## Bench 1K queries (2026-05-18)

| Scénario | Density | Forest p99 | Post p99 |
|---|---:|---:|---:|
| no_filter | 100% | 7.4 ms | – |
| pre year=2007 | 2.1% | 6.1 ms | – |
| pre year∈[23,24] | 14% | 7.1 ms | – |
| pre 2023 ∧ cs.LG | 0.8% | 6.2 ms | – |
| post top_cat='cs' | 24% | 6.3 ms | 8.8 ms |

State CH : 17 B (year=2007 RLE), 77 B (year∈2-3), 27 KB (combo on-the-fly).

**Conclusion bench** : filter au hot path = quasi-gratuit. p99 < 8 ms partout.

## Contraintes user respectées

- Pas de HTTP : tout via clickhouse-driver natif TCP/9000.
- Pas d'array lourd : bitmap state ≤ KB, jamais MB.
- Pré-agg côté serveur : `bm_by_X` via `AggregatingMergeTree`.
- Forest reste uint32 / SRT2 / depth ≤ 30.

## Pas encore fait

- FFI in-process (ctypes) — `rpforest search` est encore subprocess (~10ms overhead fork+exec).
- Vrai embedder gte-base-en-v1.5 (pour générer des queries hors-distribution).
- Test concurrence (10 queries parallèles).
- Combos pre+post mixtes.

## Recall + RAM mesurés (2026-05-18)

GT bruteforce sur 100 queries (random seed=42) + GT filter-aware (top-100 L2 restreint au filter).

**Recall@10 (top_n=2000, qd=16) :**

| Scénario | Density | Recall | ms/q |
|---|---:|---:|---:|
| no_filter | 100% | 0.917 | 15.1 |
| pre year=2007 | 2.1% | 0.717 | 13.2 |
| pre year∈[23,24] | 14% | 0.810 | 13.9 |
| pre 2023∧cs.LG | 0.8% | 0.423 | 13.4 |
| post top_cat=cs | 24% | 0.162 | 14.0 |

Avec `--auto_qd` (qd descend selon densité) : year=2007 → 0.845 (qd=14, 30 ms) ; 2023∧cs.LG → 0.681 (qd=13, 52 ms).

**Recall pre-filter sparse < 1** : le forest retourne top_n=2000 global, le ∩ avec filter sparse contient peu d'éléments NN du sous-corpus. Lever : augmenter top_n ∝ 1/density OU baisser qd plus agressivement.

**RAM (par run) :**
- Forest : anon 13-19 MB, mapped 2 MB cold (12 GB de SRT mmap, working set tiny), peak ru_maxrss 59 MB.
- ClickHouse idle : 334 MB anon + 424 MB file. Pendant 5 scénarios : +15 MB anon.

Voir [[feedback-recall-levers]] : règle SIFT (top_n>>qd) NE TIENT PAS à dim=768. Sur arxiv qd domine top_n.

Liens : [[feedback-filter-strategy]] pour la décision pre/post. [[feedback-recall-levers]] sur top_n. [[project-rpforest-sift]] pour l'historique forest (SIFT).
