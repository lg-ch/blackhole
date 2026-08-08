---
name: deep1b-results
description: DEEP 1B d=18 pathrank pipeline atteint 0.970 recall / 530 ms cold sous 1 GB RSS strict — nouveau baseline projet
metadata: 
  node_type: memory
  type: project
  originSessionId: 0e757620-83dd-4e01-864c-56db0c985e00
---

Build 2026-07-08/09 : DEEP 1B d=18 avec --fast --batch 8192 (RAM build 8.4 GB, disque final 639 GB, durée 30h).

**Baseline pathrank pipeline sous cgroup 1G strict, DEEP 1B (30 queries, cold) :**

| config | recall | mean cold | peak RSS |
|--------|--------|-----------|----------|
| NP=3 1024×2000 mlb=200k | 0.953 | 487 ms | 682 MB |
| NP=3 1024×4000 mlb=200k | 0.970 | 540 ms | 682 MB |
| NP=3 1024×6000 mlb=200k | 0.977 | 582 ms | 684 MB |
| **NP=3 1024×8000 mlb=200k** ★ | **0.983** | **630 ms** | **684 MB** |
| NP=3 1024×10000 mlb=200k | 0.983 (plateau) | 678 ms | 684 MB |

**Comparaison vs baseline SIFT 1B (d=25 heap top-N) :**
- Latence : 560 ms vs 722 ms (-22 %)
- RAM : 684 MB vs 935 MB (-27 %)
- Disque : 639 GB vs 850 GB (-25 %)
- Recall : 0.970 vs 0.993 (-0.023 mais corpus différent, DEEP + dur query-to-doc)

**Leviers utilisés :**
- Depth 18 (avg 3800 docs/leaf per tree)
- max_leaf_bytes=200_000 (cf [[project_max_leaf_bytes_lever]])
- Pathrank cross-tree by margin (mg_query_pathrank)
- L2 rerank exact top-10 (io_uring)

**NP=7 avec top_paths 1500+ OOM sous 1G strict** : besoin de rerank plus léger (TQ1) pour aller au-delà de 0.97 recall à ce budget.

Voir aussi [[project_sift1b_perf]] pour la baseline heap top-N précédente.
