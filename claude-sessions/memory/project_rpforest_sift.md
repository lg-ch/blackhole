---
name: project-rpforest-sift
description: "SIFT 1M RP forest — Phase 5 sparse counting-sort layout : 70 ms/q recall 99.77% sous cgroup 100M (×15 vs blocks)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 0e757620-83dd-4e01-864c-56db0c985e00
---

Built 2026-05-13 : forêt 1024 arbres × depth 21 sur SIFT 1M.

**Résultats validés :**
- Build streaming + counting-sort : 33 min, peak RSS **27 MB** (objectif < 100 MB tenu)
- Index disque : 7.7 GB (1024 × 7.7 MB)
- Query : peak RSS **3.7 MB**
- Recall@10 = **99.39% avec top-N=500 candidats** (33 ms/query, latence dominée par la traversée pas par le rerank)
- Voir `/home/chatelet/mangrove-search/journal_2026-05-13.md` pour les chiffres complets

**Apprentissage non-obvious : top-N-par-vote >> filtre-par-seuil.**
- **Why :** Les vrais NN saturent à 500+ votes sur 1024, le bruit a 3-15 votes. Filtrer par threshold garde des milliers de candidats à vote=8-12 qui sont presque tous du bruit. Trier par votes desc et capper à 500 supprime 95%+ des candidats inutiles à recall équivalent.
- **How to apply :** par défaut utiliser `forest_collect_topn`/CLI `topn`, pas `forest_collect`/CLI `recall`. Le param `threshold` peut être laissé à 1-2 (filtre grossier des zéros), la vraie sélection se fait par le top-N.

**Bug rencontré (corrigé) :** la version initiale de `forest_collect` scannait `d=0..n_docs` et coupait à `max_out` → perdait silencieusement les NN à id élevé. `forest_collect_topn` (min-heap sur votes) corrige.

**Hypothèse à valider :** depth 21 est probablement surdimensionné pour 1M docs (2M leaves, 0.5 doc/leaf moyenne → beaucoup de bruit de co-occurrence). Tester depth 18-19. Lien : [[project-leaf-coverage-experiment]] confirmait déjà cette intuition sur d'autres expés.

**Optis validées 2026-05-13 :**
- `gen_vec_v3` (1 xorshift / 2 dims, uniforme) **strictement meilleur** que v0 quasi-gaussien : +0.4 pt recall + 1.8× query plus rapide à config égale. Toujours préférer v3.
- `sub_dim=16` (chaque nœud projette sur 16 dims aléatoires depuis seed) perd ~1.6 pt recall isolément MAIS coûte 8× moins de FLOPS → compense avec plus d'arbres.
- **Config gagnante SIFT 1M : 1000 trees + `--gen v3 --sub_dim 16` → recall@10 = 99.37%, build 4.5 min (vs 33 min ref), query 16 ms (vs 33 ms ref), disque ≡.**
- CLI : `--sub_dim N` et `--gen v0|v3` parsés par `parse_flags` dans main.c. Build persiste params dans `<index>/meta.txt`. Query/topn/recall l'utilisent pour forcer la consistance gen_version + sub_dim (impératif sinon recall s'effondre).

**Phase 2 — chained blocks (2026-05-13) :**
- Storage layer remplacée : `block_store.{h,c}` + `convert_tree_to_blocks` + `vote_leaf` iter chaîne. Plus de sort.
- Format : Block 32 bytes = `{count, doc_id[6], next}`. Fichier `.bks` = sentinel + n_leaves racines + overflow. Chaîne via `next` (0 = fin).
- Depth 18 préféré à depth 21 (mieux pour 1M : 99.77% vs 99.37%, leaves moins sparse).
- Build 1000t × d18 × sub16 v3 : convert phase RAM peak = 13 MB (vs 20 MB sort). Avec mmap en read : **RssAnon 0.14 MB**, RssFile (cache OS) variable. Recall@10 = 99.77%, 54 ms/q.
- Trade-off : sort gagne 3× sur ms/q à 1M scale ; blocks scale au 1B (sort exploserait).
- **RssAnon = vrai compteur "RAM privée"**. RssFile = page cache OS, partagé, évincible. Reporter les deux dans le code (`read_proc_rss` dans main.c).

**Phase 3 — K-way merge (2026-05-13) :**
- `forest_collect_topn` réécrit streaming sur les chaînes triées (les doc_ids dans chaque chaîne sont croissants car build streaming en ordre doc_id).
- Remplace `uint16_t votes[n_docs]++` (2 MB à 1M, 2 GB à 1B) par K curseurs + min-heap sur `doc` + top-N min-heap par `vote`.
- Param `votes_scratch` supprimé. eval_recall_topn dans recall.c ne malloc plus 2 MB de votes.
- Résultat : peak RssAnon **228 KB** au query (vs ~2 MB avant), **indépendant de n_docs**. Recall@10 identique 99.77%. Query 125 ms (2× plus lent que vote array à cause du heap, mais c'est le prix pour scaler).
- Le RssFile (page cache OS pour mmap'd files) reste ~6 GB sur cette machine — c'est OS-managed, pas dans le budget privé.

**Phase 4 — io_uring (2026-05-13) :**
- `Forest.ring` créé une fois (`forest_open`) ; Forest heap-alloué dans main.c pour stabilité d'adresse (le ring sur stack se corrompt entre fonctions).
- `rerank_l2_uring` : batch les 500 reads de SIFT vecs ; `forest_collect_topn` réécrit en prefetch-blocks-then-merge (batches de 1-2 phases via io_uring).
- io_uring lit via `fd`, **pas via mmap** → pages OS-cached ne sont plus mappées dans le process. `RssFile` chute de 5 GB à 2 MB.
- **Test sous `sudo systemd-run --scope -p MemoryMax=100M -p MemorySwapMax=0` validé :** recall@10 = 99.80%, **1048 ms/q**, peak RSS = **47 MB** (sous le cap 100M). Speedup **15×** vs sync mmap (15700 ms/q sous 200M cap).
- Pourquoi sync mmap était si lent sous cap : 1500 random page faults séquentielles, SSD queue depth = 1. io_uring submit tout en batch, SSD queue depth ~64+, parallélisme massif.

**Code organization clé :** Forest doit être heap-alloué dans le caller (pas sur stack) sinon le ring se corrompt entre les appels eval_recall_topn/rerank_l2_uring. Si tu vois un segfault dans `_io_uring_get_sqe` sur `sq->khead`, c'est ça.

**Objectif global atteint :** RAG quasi-zéro RAM mesurée honnêtement sous cgroup réel (47 MB sur 100 MB cap, 99.80% recall, ~1 s/q). Pipeline scale au 1B sans changement de design.

**Phase 5 — sparse counting-sort layout (2026-05-14) :**
- Pivot après cadrage scaling 1B : blocks chaînés pré-allouent 32 TB de root blocks à 2^30 leaves (63% vides), impossible. Counting sort dense (`count[2^depth]` = 4 GB) impossible aussi à 1B.
- **Décision : sparse layout via counting sort externe, pas de pré-allocation de feuille vide.**
- Format `.srt` par arbre : `[16B header][offsets: (n_leaves+1)×uint32][data: total_docs×uint32 triés par leaf]`. Voir `src/sorted_store.h`.
- Build : 2 passes sur pair-file (count, puis write à `pos[leaf]++`). Query : 2 phases io_uring (offsets, puis posting lists), K-way merge inchangé.
- **Résultats SIFT 1M, 1000 trees × d18 × sub16 v3, cgroup 100M : 70.3 ms/q (×15 vs blocks 1048 ms), recall 99.77%, peak RSS 53 MB, disque 4.8 GB (-36%).**
- Build wall-clock identique (4:30 min), peak build RSS 10 MB (-63% vs blocks 27 MB).
- RssAnon query monte à 24 MB (vs 3 MB blocks) car K-way alloue les posting lists en RAM — toujours honnête, scale en n_trees pas en n_docs.
- Latence plate cgroup 100M ↔ 300M (70.3 vs 70.8 ms) : SSD-bound en 2 phases, plus du tout RAM-bound.
- **Code : `src/sorted_store.{h,c}` nouveau. `block_store.{h,c}` retiré du Makefile (gardé en repo pour historique).**
- **Index actuel : `index_sort/` (1000 trees, d18, 4.8 GB).**

**Scaling 1B projeté avec Phase 5 (1024 trees × depth 30, 300 MB RAM, 8 TB disque) :**
- Build : counting in-RAM impossible (4 GB count). External merge sort : run buffer ~80 MB, 100 runs/tree, K-way merge sur runs. ~24 GB IO/tree → **~5-6 h wall-clock** pour 1024 trees à 1-3 GB/s sustained.
- Query : ~2048 random reads en 2 phases, estimé **~150-300 ms/q** selon le drive.
- Disque : ~5-7 TB avec varint deltas sur les doc_ids (sous budget 8 TB).
- RAM : ~80 MB build peak (1 run buffer), ~24 MB query (constant en n_docs).

**Validation SIFT 10M (2026-05-15) :**
- Dataset : HuggingFace `sift10m-6filter-6a.hdf5` (10.7 GB), 9 991 000 docs, dim 128. GT du HDF5 **filtré** donc inutilisable pour ANN pur → recompute brute-force non-filtré sur 250 queries (158 s en numpy matmul, GT_K=100 pour compat `recall.c`).
- Pipeline conversion : `datasets/sift10M/prepare.py` (idempotent, 3 étapes : base.fvecs, query.fvecs, groundtruth.ivecs).
- Config build : 1024 trees × depth 23 (= ⌈log₂(N)⌉) × sub_dim 16 × gen v3.
- **Build** : 45:38 wall, peak RSS **106 MB** (count 32 + offsets 32 + data 40 MB), disque **71 GB**, 1.19 docs/leaf moyenne.
- **Query cgroup 300M, top_n 500** : recall@10 = **97.72 %**, **130 ms/q**, peak RSS **77.7 MB**.
- Sweep top_n : 500→97.72%, 1000→98.56%, 2000→98.96% — plafonne ~99% à cause sparseness.
- Inflation disque par doc : ×14.8 vs 1M (vs ×10 attendu) car offset_table grossit en `2^depth`. À d23, offset_table = 44% des bytes du file (vs 20% à d18).
- **Latence super-linéaire** vs 1M (×1.86 au lieu de constant) : offset_table totale 32 GB ne tient pas dans page cache sous cap 300M → plus de cold reads.
- **Plancher recall lié à sparseness** : 1.19 docs/leaf empêche la co-occurrence d'être robuste. Pour recall publiable >99%, baisser à d22 (2.4 docs/leaf) — à tester.
- **Index actuel** : `index_sift10m/` (1024 trees, d23, 71 GB).

**Apprentissage scaling clé : ne pas appliquer depth = log₂(N) aveuglément.**
- Why : à 1B/d30, offsets = 2^30 × 4 B = 4 GB/tree (impossible). À 10M/d23 déjà 32 MB/tree gros.
- Trade-off : depth élevé = feuilles fines = compaction par leaf mais offsets énormes ET recall plafonné par sparseness.
- Sweet spot empirique : **~2-4 docs/leaf** (donc depth = log₂(N) - 1 ou -2). À 1B, viser d27-d28 plutôt que d30.
