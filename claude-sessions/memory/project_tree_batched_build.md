---
name: project_tree_batched_build
description: "Build en lots d'arbres pour borner le pic disque (fichiers pair non-compressés) — permet 1B sur disque < 2×"
metadata: 
  node_type: memory
  type: project
  originSessionId: 0e757620-83dd-4e01-864c-56db0c985e00
  modified: 2026-08-06T21:36:34.393Z
---

Le build écrit **tous les N fichiers `pair` (leaf_id,doc_id = 8 o/doc) en même temps** pendant un seul passage sur base.fbin, PUIS compresse (varbyte) arbre par arbre à la conversion. Le pic disque n'est donc PAS l'index final (~640 GB à 1B) mais **N_trees × n_docs × 8 o** de fichiers pair non-compressés = **~2 TB à 1B/256 arbres**, avant conversion. C'est pourquoi le X10 8 TB était nécessaire.

**Patch tree-batch** (livré 2026-08-06) : `--tree_offset K --total_trees T` builde seulement les arbres [K, K+n_trees) dans le même index_dir. Les fichiers `tree%05d.{bin,srt}` sont nommés par `t + tree_offset`, les médianes indexées par id GLOBAL `(t+offset)`, `medians.bin` validé contre `total_trees`, `phase1_tKKKKK.progress` keyé par offset, meta.txt écrit `total_trees`. Driver = boucle de lots ; pic pair = `batch × n_docs × 8`. Batch=16 à 1B → pic ~128 GB pair (+ base 384 + srt accumulé) ≈ 1.1 TB, tient sur 1.28 TB.

**Prouvé byte-identique** : DEEP 1M 64 arbres mono vs 4×16 batché → les 64 .srt matchent sha256 exactement. Le batching ne change rien au résultat (mêmes seeds, mêmes médianes, mêmes docs → routing identique). Coût = re-lecture de base.fbin par lot (négligeable NVMe).

`--fast` (hyperplanes cachés, `traverse_sub_cached`) honore bien `med_th()` → compatible médianes. Voir [[project_median_toplevels]] [[feedback_ram_1gb_hard]].
