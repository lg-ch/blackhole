---
name: feedback-depth-vote-discrimination
description: "Choix de depth dans RP-forest : c'est la discriminance des votes qui gouverne, pas juste la flexibilité multi-index"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 0e757620-83dd-4e01-864c-56db0c985e00
---

Le bon depth dans une forêt RP n'est pas "le plus profond possible pour la flexibilité" — c'est celui qui rend les **votes discriminants** au K-way merge.

**Why :**
- À depth trop élevée (ex: native d=40 sur 1M docs) : chaque feuille a 0-1 docs → ~1024 candidats avec UN SEUL VOTE chacun. Le top-N par votes ne discrimine plus rien, le rerank L2 fait 100% du travail — recall s'effondre malgré le supplément de candidats.
- À depth modérée (ex: d=18 sur 1M, d=25 sur 1B) : chaque feuille a 2-10 docs → après K-way merge, les vrais NN ont des votes très élevés (300-500/1024) et le bruit reste à 3-15. Le top-N par votes a un signal réel.
- C'est la raison sous-jacente de `top-N par vote >> filtre par seuil` (voir [[project-rpforest-sift]]) : ça marche PARCE QUE les votes sont discriminants. Detune la depth et le pivot lui-même se brise.

**How to apply :**
- Pour un corpus seul de N docs : `depth ≈ log₂(N) - 1` ou -2 (sweet spot 2-4 docs/leaf, validé empiriquement avec sub_dim 16 v3).
- Pour une famille multi-index avec target N_combined : `depth_build = log₂(N_combined) - 1`. Permet de requêter la combinaison à depth_build natif, et un sous-corpus de N_sub à `query_depth = log₂(N_sub) - 1` artificiellement (gratuit via le code multi-index).
- **Décision projet (2026-05-16) : depth=25 fixé** pour les futurs builds dans la famille multi-index. Couvre 1B docs combinés (~30 docs/leaf), 100M (~3 docs/leaf, sweet spot), sous-corpora plus petits via query_depth abaissé.
- NE PAS aller à depth=40 "pour la flexibilité". Le coût build (1.4× CPU) ET la perte de recall en query natif d=40 (vote uniforme) ne sont pas compensés par "plus de granularité".

**Mécanisme du code (déjà en place) :**
- `forest_collect_topn(f, qvec, top_n, query_depth, ...)` : si `query_depth < f->depth`, traverse stop à query_depth et lookup en RANGE [parent_id << k, (parent_id+1) << k) où k = depth_build - query_depth.
- CLI : `--query_depth N` dans `cmd_topn` (déjà branché dans main.c via `g_cli_query_depth`).
- Même seeds (`tree_seed(t)`, `node_seed`) → arbres aux niveaux 0..query_depth strictement identiques entre tous les corpora d'une famille.

**Validation 2026-05-16 — SIFT 1M, d=25 build, qd=18 artificial :**
- Recall@10 = 99.77 % IDENTIQUE au build natif d=18 (99.77 %).
- Latence 111 ms vs natif 97 ms (+14 ms : 2 reads window par tree + qsort des docs par tree).
- Le multi-index code est validé fonctionnellement.

**Validation 2026-05-16 — SIFT 1M+10M, d=40 build (int64), qd=22/18 artificial :**
- Recall artif qd=22 sur d=40 multi-index = 98.08/99.08/99.40 % aux top_n {500,1000,2000} : **identique au native d=22**. Confirme la stricte équivalence math.
- Coût permanent vs native d=22 : latence +75 % (262 vs 149 ms), peak RSS +91 % (244 vs 128 MB), disque +170 % (117 vs 43 GB), build time +126 %.
- **Pourquoi disque ×2.7 et non ×1.5 (uint64 simple) :** à d=40 quasi tous les docs ont un leaf unique → ×6 plus d'entrées sparse_index par tree (1.14 B → 9.6 B per doc en overhead sparse).
- **Décision projet : rester à depth ≤ 30** (uint32 suffit, int64 inutile). À 1B, marge multi-index reste large.
- Quand top_n bumping ne suffit pas et qu'on plafonne (1B+) : rebuild à d=33+ avec int64 ré-activé. ~6h par shard.
- **Int64 refactor reverted** : codebase back to uint32 / SRT2 / max depth 30. Cleaner code. Git history a l'historique int64 si on doit y revenir.

**Décision 2026-05-16 — Sparse delta encoding skipped.**
- Gain : -25 à -40 % disque sur sparse_index (-3.5 TB à 1B docs / d=30).
- Coût query : +5 ms par query (varint decode dans le window). Pas d'économie IO (NVMe 4 KB granularity).
- Coût implémentation : ~150 lignes nouvelle, format SRT4, rebuild indexes existants.
- **Pas le bon levier au target dim=1024 fp32** : la vraie économie disque vient de la quantization int8 des vectors (-3 TB à 1B avec recall -0.5 à -1.5 pt).
- Sparse delta réservé pour plus tard si n_trees grossit ou si on cible une compression agressive.

**Context dim 1024 fp32 (target embedding) :**
- Vecteur brut = 4 KB/doc vs 128-512 B pour SIFT. Ratio index/vecs passe de 8-50× à ~1×.
- Disque 1B = ~13-14 TB sans optimisation (4 TB vecteurs + 4 TB index data + 5.6 TB sparse).
- Avec int8 quant vecteurs + d=30 SRT2 actuel : ~10 TB total. Tient sur drive 12 TB.
- Avec int8 quant + sparse delta : ~6.5 TB. Tient sur 8 TB.

**Piège résolu (qsort obligatoire quand k_shift > 0) :**
- Les docs concaténés sur plusieurs build-depth leaves NE SONT PAS monotones globalement (sorted par leaf, pas par doc_id).
- Le K-way merge sur les TCursor exige des séquences monotones → qsort de chaque tree's slice quand `k_shift > 0`.
- Native query (k_shift = 0) : pas de sort nécessaire, docs d'un même leaf déjà triés par build streaming.

**Validation multi-index 4 sous-familles (2026-05-16) :**
- SIFT 10M split en 4 sous-index disjoints (5M + 3M + 1.5M + 0.5M, doc_offset shifté) à d=22 sub16 v3.
- Code `forest_collect_topn_multi` + CLI `multitopn idx_a,idx_b,...`.
- **Recall@10 = 98.08 % vs single 10M/d23 97.72 %** (+0.36 pt) à 152 ms/q (vs 130 ms).
- Peak RSS 128 MB sous cgroup 800M, disque 43 GB (vs single 71 GB, -40 %).
- Le multi-index donne un RECALL ÉGAL (ou meilleur) à un single-index unifié — confirmé empiriquement.

**Pièges io_uring résolus :**
- Plusieurs ring (1 par forest) : reap par-ring avec un compteur `per_ring_subs[fi]`. `io_uring_wait_cqe` BLOQUE si on attend sur un ring déjà drainé.
- FD limit : raise à `n_forests × n_trees + 64` AVANT le premier `forest_open` (par défaut 1024, à 4×1000=4032 il faut bouger).

Lien : [[project-rpforest-sift]] pour le contexte global, [[feedback-n-trees-sweep]] pour l'autre levier déjà épuisé.
