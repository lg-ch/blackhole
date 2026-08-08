---
name: project-sift1b-perf
description: SIFT 1B perf baseline after radix sort path + 5p/16k config sweep (session 2026-06-07). Single CPU 722 ms p50 recall 0.993.
metadata: 
  node_type: memory
  type: project
  originSessionId: 0e757620-83dd-4e01-864c-56db0c985e00
---

## Current SIFT 1B baseline (recall 0.993)

| metric | single core | parallel ×10 |
|---|---|---|
| p50 | **722 ms** | **130 ms** |
| p95 | 3472 ms | 413 ms |
| p99 | 4046 ms | n/a |
| RAM total | 935 MB (10 segments combined) | idem |
| Disk | 0.85 TB | idem |

Config : `Forest(n_trees=256)` × 10 segs × native depth 30 × `n_probes=5` × `top_n=16000` × fused C path (`query_probes`).

## Algo path (forest_collect_topn_probes, no-filter branch)

```
gather pairs (tree_id<<32 | doc_id) → 3-pass LSB radix sort (11+11+10 bits,
4-way parallel sub-histograms, pooled buffers) → linear scan with per-tree
vote dedup → top-N heap push.
```

## Timeline des optims (single CPU)

| variante | p50 ms |
|---|---|
| heap K-way merge baseline | 3089 |
| Loser tree | 3254 |
| Pre-merge intra-tree | 3434 |
| Radix sort + scan (8-bit, malloc) | 1328 |
| + pool buffers | 1051 |
| + 11/11/10 radix | 1013 |
| + 4-way parallel sub-hist | 997 |
| + 5p / 16k config | **722** |

## Pourquoi 5p/16k bat 10p/8k

Mécanisme : moins de probes → moins d'entrées dans le radix sort
(48k → 32k env), donc 30 % moins de travail merge. Mais top_n × 2 garde
le pool de candidats large pour que le rerank L2 retrouve le top-10.
Pas un trade-off recall : 0.993 dans les deux configs (sample 15 q).

## p95/p99 sources (à voir si on veut < 1 sec p99)

Bad queries qui tombent dans des feuilles de plusieurs centaines de
milliers de docs (cf [[project_filter_mode_crossover]] et l'analyse
"tree 800" de la même session) → travail × 10-20 sur une seule query.

## Sources fichiers

- `scripts/query_sift1b.py` (helper read-only)
- `R&D/bench_1b_probes_topn_sweep.py` (gitignored — le sweep)
- `BENCH.md` section SIFT 1B (publié dans le repo)
- `src/query_tree.c` `forest_collect_topn_probes` (radix path)
