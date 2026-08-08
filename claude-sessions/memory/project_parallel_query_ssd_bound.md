---
name: parallel-query-ssd-bound
description: Multi-ring io_uring ne gagne rien sur SIFT 1B cold — SSD-bound. Affinity fait tout.
metadata: 
  node_type: memory
  type: project
  originSessionId: 0e757620-83dd-4e01-864c-56db0c985e00
---

## Fait 2026-07-15

Tenté : multi-ring io_uring (1 ring/thread OMP), pour paralléliser Phase 1 + Phase 2 de `forest_collect_topn_probes`. Revert car aucun gain mesurable.

**Résultats sur SIFT 1B tp=1024 mlb=400k top_n=8000 cold (recall 0.973)** :

| OMP | affinity | mean latence |
|-----|----------|--------------|
| 1 | 1 | 614 ms (baseline single-core) |
| 1 | 4 | 619 ms (neutre) |
| **1** | **8** | **500 ms (-116 ms, -19 %)** ← gain "gratuit" affinity |
| 4 | 4 | 616 ms |
| 8 | 8 | 516 ms (LEGEREMENT pire que OMP=1 aff=8) |

**Diagnostic** : Phase 2 lit ~200 MB de leaves par query (1024 leaves × ~200 KB), SSD Crucial X10 fait ~400-500 MB/s en random. **Latence ~500 ms = SSD-throughput-bound**, pas CPU-bound.

Le -116 ms entre aff=1 et aff=8 vient des kernel io_wq workers (spawned proportionnellement au CPU count) qui saturent la bande passante SSD. À aff=1, ils sont starved.

Multi-ring OMP ajoute overhead (~12 ms/query init + reduce) sans bénéfice car SSD est déjà saturé.

## Where parallel WOULD help

1. **QPS scaling** — queries CONCURRENTES : chaque query a son propre ring, pas de contention sur le ring unique du Forest.
2. **Warm cache** — Phase 2 hit RAM → CPU-bound → parallel décode/l2sq utiles.
3. **High-dim corpora** (arxiv/LAION dim 768) — l2sq × 8000 cands = ~1 s CPU sériel → parallélisable.

## What's kept

- `#pragma omp parallel` traversal dans `mg_query_pathrank` (ffi.c) — small change, neutre sur SIFT, aide un peu sur high-dim.
- `#pragma omp parallel for` rerank_L2 CPU dans `rerank_l2_uring` (recall.c) — perte CPU/IO overlap (~5 ms sur SIFT), gros gain sur high-dim.
- Multi-ring io_uring : reverté, code multi-ring supprimé de query_tree.c.

## Why:

Vraie leçon : **profiler avant de paralléliser**. La parallélisation n'aide QUE si le hotspot est CPU. Sur cold IO-bound single query, le SSD est le mur.

## How to apply:

- Pour préprint SIFT 1B latence : mentionner "500 ms cold sous 1 core with OS-level io_uring workers on 8 cores, SSD-bound at 400 MB/s Phase 2 throughput".
- Pour QPS bench : le multi-ring pourrait être ressorti dans un thread-per-query pattern (chaque query a son propre thread + ring).
- Pour arxiv/LAION (high-dim) : le parallel rerank_l2 devrait payer bien plus, à vérifier une fois LAION buildé.
