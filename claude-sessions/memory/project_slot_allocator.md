---
name: slot-allocator
description: "Format .slt slot allocator — writer C livré, query reader à faire"
metadata: 
  node_type: memory
  type: project
  originSessionId: 0e757620-83dd-4e01-864c-56db0c985e00
---

## Contexte

Design streaming smart : mlb=300k comme constante hard système + slot allocator power-of-2 classes → bounded max leaf structural + reads SSD-aligned + enables MAIN+HOT streaming.

## État 2026-07-16

### Livré

- **`src/slot_store.h`** : format spec, 16 classes power-of-2 [16 B → 384 KB], sparse packed (class 4b + slot 28b)
- **`src/slot_store.c`** :
  - `slt_open_rdonly(SltStore*, path)` — open + load samples RAM
  - `slt_close`, `slt_sample_bucket`
  - **`slt_convert_from_srt_v2(srt_path, slt_path, &n_overflow)`** — converter fonctionnel
- Compilé (juste warnings pwrite return unused, non-critique)

### Mesures converter

- SIFT 10M tree 0 : `.srt 47.5 MB → .slt 70.3 MB` (+48%)
- 0 overflow sur 10M (max leaf < 384 KB)
- Distribution : 79% class 0 (16 B), tail 20% spread across classes 1-13

### Livré aussi

- **`slt_convert_all.py`** : convert 64 trees SIFT 10M en 97 s
- **Bench cold pread** SRT vs SLT (single thread aff=8) :

  | format | mean | p50 | p95 | p99 | bytes lus |
  |--------|------|-----|-----|-----|-----------|
  | SRT | 51 ms | 49 | 54 | 77 | 5 MB |
  | SLT | 46 ms | 46 | 50 | **50** | 7 MB |

  - **-10% mean, -35% p99** — tail latence lissée
  - +40% bytes lus (padding) mais moins de wall
  - Tighter distribution : p95 ≈ p99 (vs SRT écart notable)

### C query path livré (avec radix + histogram top-N)

- `src/slot_query.c` : `SltForest`, `slt_forest_open`, `slt_query_pathrank`
- FFI : `mg_slt_forest_open`, `mg_slt_forest_close`, `mg_slt_query_pathrank`
- Pipeline complet : traversal (OMP) + pathrank margin sort + Phase 1 io_uring + Phase 2 io_uring + radix sort 2-3 passes + histogram top-N
- Bench sur SIFT 10M cold qd=depth tp=1024 (30 queries) :

  | format | mean | p50 | p75 | p95 | p99 |
  |--------|------|-----|-----|-----|-----|
  | SRT | 23 | 23 | 27 | 29 | 32 |
  | **SLT** | **20** | **20** | **21** | 32 | 35 |

  - **-13% mean, -13% p50, -22% p75** (slot-aligned reads)
  - Tail comparable (+2-3 ms p99 dans le bruit)
  - Gain limité par SIFT 10M sparse (median leaf 13 B, class 0 slot 16 B → minimal padding gain)

- Pread bench précédent avait montré p99 -35% ; c'était probablement avec un ensemble différent de leaves.

### CRITICAL FINDING (2026-07-17)

Bench qd=20 avec subtree expansion :

| format | mean | p50 | p95 | p99 |
|--------|------|-----|-----|-----|
| SRT | 47 ms | 46 | 74 | 80 |
| SLT | 102 ms | 96 | 171 | 186 |

**SLT est 2.17× PLUS LENT à qd<depth**. Cause fondamentale :
- SRT stocke les leaves d'un subtree ADJACENTES sur disque (sorted par leaf_id) → 1 range read = tout le subtree
- SLT chaque leaf dans son slot par classe → N reads séparés (un par storage leaf non-vide)
- À qd=20 : subtree = 2^8 = 256 storage leaves, ~20-100 non-vides → 20-100k reads au lieu de 1024

**SLT casse la locality intra-subtree**. OK pour native depth, catastrophique pour qd<depth.

### Design révisé MAIN+HOT bimodal

- **MAIN** (frozen, immutable, huge) → **SRT** — locality subtree préservée pour qd<depth
- **HOT** (small, streaming, mutable) → **SLT** — slots dynamiques, mutable, peu de leaves per subtree
- Query = SRT(main) + SLT(hot) merge cross-format

Le vrai gain de SLT = **mutable streaming**, PAS remplacer SRT.

### MVP MAIN(SRT) + HOT(in-RAM) LIVRÉ 2026-07-17

- `src/hot_store.{h,c}` : per-tree sorted `HotLeaf[]` (leaf_id, docs, cap) + binary search + append + range lookup pour subtree expansion
- `query_tree.c` : `g_hot_overlay` thread-local, injecté dans le pack loop de `forest_collect_topn_probes` (no-filter path). `tree_seen[]` dedupe MAIN+HOT par tree, 0 double-count.
- FFI : `mg_hot_init/free/append/append_batch/ram_bytes/total_docs` + `mg_forest_set_hot_overlay`

**Bench SIFT 10M qd=20 tp=1024 top_n=6000 sous cgroup 1G cold** :

| config | mean | p50 | p95 | p99 |
|--------|------|-----|-----|-----|
| MAIN only | 49 | 48 | 74 | 81 |
| MAIN + HOT 50k  (~0.5%) | 46 | 46 | 67 | 74 |
| MAIN + HOT 250k (~2.5%) | 47 | 47 | 69 | 74 |

**Zéro régression** — overhead HOT ≈ bruit. 250k HOT docs = 10 MB RAM.

**Correctness validée** : fake doc injecté dans tous les 64 build-leaves visités par une query → vote=64 dans le résultat (== n_trees, deduplication cross-format OK).

### À FAIRE next session

1. HOT persistence disque (SLT-style dynamic slot alloc + in-RAM sparse index chargé au boot)
2. WAL append-only pour recovery HOT
3. Compaction MAIN+HOT → nouveau MAIN.srt (threshold RAM / periodic)
4. Filter path (bitmap allowed) : injecter HOT dans le K-way merge aussi

### Snapshot design précédent

Snapshot pre-slot dans `/mnt/mangrove/snapshots/design_20260716_2046/` (src+scripts+binaries).
Rollback trivial si le refactor slot échoue.

### Why

- Bounded max leaf = query working set prévisible (mlb × top_paths deterministic)
- Slot-aligned = SSD reads plus efficaces (4KB boundaries)
- Sparse index compact (4B class+slot au lieu de 4B offset seul) permet HOT sparse en RAM
- Streaming avec un seul HOT overlay au lieu de N-segments LSM
