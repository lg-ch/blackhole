---
name: mangrove vs mangrove-search
description: Deux projets distincts — mangrove implémente déjà l'archi pairwise-seed compacte qu'on discutait, mangrove-search est l'ancienne archi data-dépendante.
type: reference
originSessionId: e91b1dca-1dcd-4c5a-8a66-7e6b9afc1390
---
Deux projets distincts dans `/home/chatelet/` :

- **`mangrove-search/`** (le cwd ici) : héberge **les deux archis** :
  - `hd_ann_test.cpp` + `leaf_coverage_sweep.cpp` — ancien prototype data-dépendant (Nodes 24 B avec threshold+left+right+first+size), pas scalable à 1B.
  - `pairwise_test.cpp` (28 avril) — **implé pairwise-seed standalone** (mêmes helpers que mangrove : `node_seed`, `gen_vec`, traverse `cos(q,v1) > cos(q,v0)`). C'est ici que le travail récent se fait. Builds : `forests/pw_10m_d24_sd16_t1024[_sorted].bin`, `forests/pw_100m_d28_sd16_t1024.bin` (125 GB).

- **`/home/chatelet/mangrove/`** : production. Implémente déjà **l'archi pairwise-seed** (v0, v1 générés depuis `hash(tree_seed, bfs_idx)`, route par `cos(v,v1) > cos(v,v0)`). **Rien de data-dépendant stocké côté structure d'arbre**. Format disque :
  - `unique_leaves[]` int64 × n_leaves (paths non-vides uniquement, encodés comme BFS-index)
  - `offsets[]` int32 × (n_leaves+1)
  - `vec_ids[]` int32 × n_vectors (triés par leaf)
  
  Storage par arbre ≈ 12 B × n_leaves + 4 B × N → à N=1B et d=30, ~4 GB/tree (dominé par vec_ids), soit ~4 TB à 1024 trees. Exactement le minimum théorique qu'on avait calculé.

**Quand l'user parle de ses idées "pairwise-seed, pas de médiane, pas de threshold"**, il parle de l'archi `mangrove`, pas de `mangrove-search`. Ne pas refaire l'erreur de raisonner sur l'archi mangrove-search (qui stocke les Nodes et a des thresholds) quand le sujet est la vraie implé.

Code-clés :
- `mangrove/mangrove/io/format.py` — `IndexFile`, `write_tree_file`, `read_tree_file` (format actuel : unique_leaves + offsets + vec_ids)
- `mangrove/mangrove/native/src/pairwise_beam.cpp` — `build_pairwise_single`, search beam-width>1, `gen_node_vec`, `node_seed`
- `mangrove/CLAUDE(1).md` — baselines SIFT 100k : 30 trees d=18 → 77% @ K=1k, 95% @ K=9k ; extrapolation 500 trees → 90% @ K=0.2%

**Doc architectural le plus complet : `mangrove-search/CLAUDE(2).md`** (aussi copie dans Documents/). Décrit :
- Vision "zéro stockage nodes" (seuls les seeds et posting lists + vecteurs bruts)
- **Design slot-fixe** encore plus compact que le format actuel : `offset = tree × N_SLOTS × SLOT_SIZE + leaf_rank × SLOT_SIZE`, zéro index en RAM (pas encore implémenté dans mangrove, le format actuel a encore unique_leaves + offsets).
- Top-K smallest leaves : passer de 330k à 38k candidats à recall ≈ en gardant les 500 arbres à plus petite feuille (= ce qu'on a re-découvert en leaf-coverage).
- Sub-dim (16 dims / 128) : 8× FLOPS réduction, −6.7 pt recall.
- Scoring E (`n_trees × Σ 1/(1+dist_feuille)`) vs Scoring G (pondère par dist 2D cosines).
- Cibles bench : 10M @ 99% / 200 MB RAM, 100M @ 99% / 50 MB RAM, SIFT 1M target 200 trees d=20.
- I/O idéal : 500 lectures // avec io_uring + MADV_WILLNEED → 2 ms sur NVMe.

Sur SIFT 1M, target bench mentionnée : 200 arbres depth 20.
