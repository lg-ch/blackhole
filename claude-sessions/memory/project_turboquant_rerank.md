---
name: turboquant-rerank
description: "TurboQuant 4-bit LIVRÉ — recall identique, latence -29/-43%, IOPS-bound : prochain déblocage = codes interleavés dans les leaves (avec SRT4)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 0e757620-83dd-4e01-864c-56db0c985e00
---

LIVRÉ (tâche #107) : rerank deux-étages TurboQuant 4-bit en C. Stack : `src/tquant.{c,h}` (rotation (HD)³ seedée splitmix64 + FWHT, Lloyd-Max codebook partagé dans le header .tq4, scoring LUT nibble 64 KB, io_uring batch), CLI `rpforest tquant <fvecs> <out.tq4> --dim N`, FFI `mg_rerank_tq` / Python `Forest.rerank_tq(tq4, base, q, cands, kprime, top_k)`.

Mesures C (en2 20M dim 1024, drop_caches, 5 cold + 10 warm) : recall IDENTIQUE à l'exact partout ; 32k : 1.31→0.94 s (-29 %) ; 64k : 2.38→1.50 s (-37 %) ; 128k : 4.21→2.38 s cold, 1.64 s warm (-43/-58 %). K'=100 suffit (K'=200 identique). Sidecar 20M docs = 9.6 GB (÷8.5 vs fvecs), build 6 min single-thread. Warm profite enfin du cache (codes 9.6 GB cachables vs 82 GB).

**Why:** idée utilisateur (TurboQuant arXiv 2504.19874, ICLR 2026). Data-oblivious, rotation seedée → aligné philosophie pairwise-seed. Pas le ÷8 espéré car IOPS-bound : 64k lectures aléatoires restent 64k IOPS quelle que soit leur taille.

**How to apply:**
- Utiliser `rerank_tq` plutôt que `rerank_l2` dès que top_n ≥ 32k ; K'=100 par défaut.
- Bugs à connaître : `vec_row_bytes()` = payload SEUL (sans header 4 B fvecs) — ne pas l'utiliser comme stride ; premier build sidecar avait frame-shift à cause de ça.
- Prochain déblocage latence = interleaver les codes 4-bit DANS les leaves .srt (lus avec les posting lists → zéro IOPS pour l'étage 1) — à concevoir avec SRT4/PForDelta (tâche #101). Gain estimé : élimine 64-128k IOPS/query, resterait K' lectures exactes + posting lists.
- 2-bit possible (÷16 disque) mais n'aidera pas l'IOPS-bound ; utile seulement pour l'empreinte.
- `--tree_offset` (src/build_tree.c) : étendre une forêt arbre par arbre, validé byte-identique.
