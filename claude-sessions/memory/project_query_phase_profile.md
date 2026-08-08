---
name: query-phase-profile
description: "Profil par phase DEEP 1B (laptop 7520U) : Phase1 29%, Phase2 30%, RADIX 25% (surprise). Leviers chiffrés."
metadata: 
  node_type: memory
  type: project
  originSessionId: 0e757620-83dd-4e01-864c-56db0c985e00
  modified: 2026-07-19T17:49:28.907Z
---

## Instrumentation

`forest_collect_topn_probes` a des timers par phase (thread-local `g_phase_ms[8]`,
FFI `mg_get_phase_ms(i)`). Indices : 0=phase1_io, 1=resolve, 2=phase2_io,
3=decode, 4=pack(+HOT), 5=radix, 6=scan, 7=total_C. Overhead ~0.

## Mesure 2026-07-19 — DEEP 1B qd=18 NP=3 tp=1024 top_n=4000 mlb=200k, laptop Ryzen 7520U, cgroup 1G

| phase | COLD | WARM |
|---|---:|---:|
| phase1_io (windows 33 MB) | 338 ms (30%) | 134 ms |
| phase2_io (leaf data ~15 MB / 1024 reads) | 348 ms (30%) | **518 ms** (!) |
| decode varbyte (~3.9 M docs) | 68 ms | 69 ms |
| **radix 3 passes sur 3.9 M pairs** | **291 ms (25%)** | 279 ms |
| scan+pick | 77 ms | 52 ms |
| TOTAL C | 1147 ms | 1069 ms |
| traversal+pathrank (hors C) | ~3 ms | — |

## Findings clés

1. **Radix = 25 % du temps** (291 ms) — bien plus que estimé (~50 ms). Sur 7520U
   (512 KB L2, DDR lente) le scatter 3 passes × 62 MB de traffic mémoire coûte cher.
   → parallel radix OMP (per-thread histograms) = levier -200 ms sur ce CPU.
2. **Phase 2 WARM > COLD (518 vs 348)** : sous cgroup 1G le page cache n'a que
   ~300-400 MB ; reclaim pressure ralentit les reads. Le "warm" sous cgroup serré
   n'existe presque pas pour les leaf data. Attention aux benchs warm sous cgroup.
3. Phase 1 = 33 MB de windows (stride 4096 × 8 B × 1024 probes) → samples denser
   (stride 64) divise par 64 → levier -300 ms cold.
4. Traversal = ~3 ms même single-thread : négligeable, ne pas optimiser.
5. Decode 68 ms seulement (pas 150 comme estimé).

## Plan optim chiffré (laptop → réf CPU)

| levier | gain laptop | note |
|---|---:|---|
| samples denses stride 64 | -200/-300 ms | RAM +8 MB, tue phase1 |
| parallel radix OMP | -200 ms | classic per-thread hist |
| TQ1 rerank (hors collect) | -100/-220 ms | selon top_n |
| coalescing probes phase2 | -50 ms | |
| rebuild d=20 (÷4 docs/leaf) | ÷2-3 sur decode+radix+phase2 | 30 h build |

Cible réaliste post-optim : ~400-500 ms laptop, ~200-250 ms réf CPU.
100 ms exige en plus le rebuild plus profond.

## n_nonempty réel SIFT 1B d=28 (mesuré 2026-07-19, headers .srt)

- **8.4-10.5 M leaves non-vides / arbre** (occupation 3.5 % de 2^28)
- **~105 docs / leaf non-vide** — clustering massif, PAS 3.7 (qui divisait par 2^28 total)
- Sparse index : 76 MB/arbre disque
- **Samples denses VIABLES à d=28** : stride 128 → 76 MB RAM total, stride 64 → 152 MB
  (l'estimation "4 GB, infaisable" supposait 100-260 M nonempty — fausse ×25)
- Sidecar `.smp` = levier n°1 même en régime build unifié d=28
- Clustering → explique max_vote faibles (7-11/256) : hyperplans coupent mal DANS les clusters
  → renforce piste sub_dim↑ / tree_sub pour discriminance

## Incident disque 2026-07-19

Crucial X10 (USB/WSL2) : tempête d'erreurs hv_storvsc, process en D-state.
Fix SANS reboot : `umount -l` + `wsl.exe --unmount \\.\PHYSICALDRIVE1` (interop) +
réattach élevé via `Start-Process wsl.exe -Verb RunAs` (UAC user) + mount UUID.
FS clean, aucune perte (workload read-only). Le détach débloque les D-state.

## Profil sift_1b_d28 qd=26 NP=3 tp=1024 top_n=6000 mlb=200k (laptop, cgroup 1G)

| phase | COLD (917 ms) | WARM (683 ms) |
|---|---:|---:|
| phase1_io | **341 ms (37 %)** | 153 ms (23 %) |
| phase2_io | 279 ms (30 %) | 247 ms (36 %) |
| radix | 197 ms (22 %) | 189 ms (28 %) |
| decode+pack+scan | 99 ms | 93 ms |

À qd=26 (k_shift=2) : 2 fenêtres/probe × 1024 × 32 KB = **65 MB Phase 1 par query**.
Verdict régime d=28 : Phase 1 domine cold → **sidecar .smp = chantier n°1**
(projection : cold 917→~470 ms, warm 683→~400 ms avec .smp stride 128 + radix OMP).
Macroblocs = long-terme (fusion P1+P2, unité de compaction HOT).

## Diagnostic I/O 2026-07-19 : le disque est IOPS-bound DRAM-less, pas bytes-bound

Micro-benchs io_uring (probe C, 1024 reads batch, cold) sur Crucial X10 via WSL2 :
- **1 fichier (span 2.7 GB) : 28k IOPS** ; **256 fichiers (span 708 GB) : 5k IOPS**
- Taille de read indifférente (1 KB = 32 KB, même IOPS) → SSD DRAM-less :
  mapping FTL fetch en NAND sur wide-span random. Hardware, pas fixable.
- FADV_RANDOM : aucun effet. IOSQE_ASYNC : aide le burst froid (metadata) seulement.
- fio QD32 single-file 31k IOPS confirmé — le "QD=1 apparent" de nos queries
  était en réalité le plafond wide-span du SSD.

**Conséquences** :
1. `.smp` (samples denses) : AUCUN gain sur ce hardware (bytes gratuits).
   Gardé quand même : 75 MB sidecar, zéro régression, utile sur stacks bytes-bound
   (NVMe DC avec DRAM), et fenêtres plus petites = moins de RAM churn.
2. **Le vrai levier sur ce hardware = réduire le NOMBRE de reads** →
   macroblocs (fusion Phase1+Phase2 : 2048 → 1024 reads = ÷2 sur l'I/O).
3. Sur serveur NVMe avec DRAM (500k+ IOPS wide-span), les phases I/O fondent
   et le CPU (radix) redevient dominant → radix OMP.
4. drop_caches per query paie ~200 ms de metadata ext4 (extent trees 256 fichiers)
   au premier burst — les benchs "cold" sur-estiment vs prod (server up).

## Fix bug subtree straddle (livré 2026-07-19)

Refactor Phase 1 de forest_collect_topn_probes : UNE fenêtre span par probe
(pré-passe RAM calcule bucket_lo..bucket_hi), remplace les 2 fenêtres low/high.
Corrige une troncature silencieuse préexistante : les subtrees chevauchant une
frontière de bucket perdaient leur queue au decode (proba ∝ 1/stride).
Validé : résultats identiques avec/sans .smp à tous qd (28→14) sur arxiv d=28.
Bonus : -50 % de reads Phase 1 à qd<depth (2→1 par probe).

## Expérience macroblocs .mbk (2026-07-19) — implémentée, mesurée, REVERTÉE

Design : blocs à plage FIXE de leaf_ids (R = 1<<range_bits, alignement puissance
de 2 ⇒ 1 read contigu par probe quel que soit qd), mini-index colocalisé avec
la data, table d'offsets u64[n_blocks+1] en RAM (134 MB @ 1B d=28), blocs vides
= 0 byte = zéro I/O. Payload copié verbatim du SRT V3. Correctness validée
IDENTIQUE au SRT sur tous les qd (arxiv d=28), disque iso (+0 %).

**Verdict A/B** : perd ×6-10 sur stockage byte-bound (mesuré via host cache),
gagnerait seulement ~×2 sur X10 cold IOPS-bound (2048→1024 reads). Le domaine
de victoire = exactement le hardware qu'on déconseille → reverté. Code
supprimé (design ré-implémentable ~1h depuis cette note si cible edge un jour).

## Piège de bench découvert : CACHE HOST WINDOWS (3e niveau)

Sur WSL2, `drop_caches` ne vide que le cache GUEST. Le host Windows cache les
blocs du disque passthrough : tout corpus récemment écrit/lu (arxiv 6 GB…)
est servi depuis la RAM host à ~250 MB/s via VMBus. Conséquences :
- Les benchs "cold" de PETITS corpus sur cette machine mesurent le host cache,
  pas le disque. Seul le 1B (708 GB >> RAM host) est un vrai cold.
- Protocole cold/warm à 3 niveaux désormais : guest cache / host cache / disque.
  Pour un vrai cold small-corpus : rebooter WSL ou lire un gros volume d'abord.
