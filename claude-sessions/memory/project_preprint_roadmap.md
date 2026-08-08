---
name: preprint-roadmap
description: "Roadmap fin d'itération pour préprint arXiv \"prod-ready ANN billion-scale sous 1 GB RAM strict\""
metadata: 
  node_type: memory
  type: project
  originSessionId: 0e757620-83dd-4e01-864c-56db0c985e00
---

Décidée 2026-07-15 après itérations SIFT/DEEP 1B. Focus « ANN pour RAG sur indexs milliardaires », pas gestion metadata (deferred).

## Ordre de travail

**Semaine 1 : Auto-tune + parallel query**
1. **Auto-tune finish** (2-3 jours dev, EN COURS)
   - [x] Depth calibrator (marche, valide sur SIFT/DEEP)
   - [x] Distribution analyzer (heuristics à recalibrer)
   - [x] Streaming GT build pendant phase 1 (`--calib-queries N`)
   - [ ] Joint grid sweep (top_paths × mlb) avec **budget guard 90 %**
   - [ ] `mangrove calibrate` CLI wrapper → écrit `recommended_config.json`
   - [ ] Server startup lit auto-config si présent

2. **Parallel query multi-core** (3-5 jours dev, GROS IMPACT)
   - Traversal parallèle 256 trees split
   - Radix sort parallel per-tree merge (LSM-style)
   - io_uring rerank_L2 parallel split
   - Cible : cold 100-150 ms sur SIFT 1B (vs 441 ms single core actuel)

**Semaine 2 : Corpus high-dim milliardaire**
3. **LAION-400M CLIP 768d** (téléchargement ~600 GB + build ~1 semaine)
   - Vraie donnée production haute dim
   - Data point différenciant

**Semaine 3 : RAG SDK + robustesse**
4. **RAG chunks SDK** (1-2 jours)
   - Zero C code, SDK Python `RagIndex(mangrove) + KVStore(sqlite)`
   - Demo end-to-end MSMarco/Wikipedia → query → top-K passages

5. **Recovery/robustness demo** (1 jour)
   - Kill mid-query → restart identique
   - Kill mid-build → resume via progress.txt
   - Corrupt bytes → graceful error
   - Read-only mount → queries OK

**Rédaction préprint** en parallèle semaine 3.

## Ce qui est explicitement DEFERRED (post-préprint)

- Vraie compaction LSM segments background
- Adaptive depth per leaf
- Metadata management natif (users use external KV)
- Fully async writes / hot RAM segment
- Multi-thread build (déjà là via OMP mais peut être optim)

## Baselines finaux à publier

| corpus | recall | latence cold | RAM | note |
|--------|--------|--------------|-----|------|
| DEEP 1B d=18 | 0.983 | 630 ms 1-core | 684 MB | Pareto Pareto |
| SIFT 1B d=25 | 0.987 | 623 ms 1-core | 796 MB | max recall |
| SIFT 1B d=25 | 0.967 | 441 ms 1-core | 523 MB | fast |
| LAION-400M ? | ? | ? | ? | à build |
| Parallel query | ? | 100-200 ms 4-8 cores | 1 GB | key data point |

Tous sous cgroup 1G strict + drop_caches per query (mode « prod-worst-case »).
