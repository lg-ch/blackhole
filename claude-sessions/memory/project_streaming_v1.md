---
name: streaming-v1
description: "Streaming ingest MVP livré : HOT thread-safe + compaction per-tree + swap dir. Validé E2E SIFT 10M 2026-07-17."
metadata: 
  node_type: memory
  type: project
  originSessionId: 0e757620-83dd-4e01-864c-56db0c985e00
---

## Livré 2026-07-17

### Piece 1 : HOT thread-safe
- `pthread_mutex_t` par HotTree
- `hot_tree_lock/unlock` autour de `hot_range/hot_lookup` dans query path
- `hot_snapshot_and_clear` : deep-copy → drain, sous lock, pour compaction
- Overhead lock query : ~40 ns par probe × 256 probes = **10 µs par query, invisible**

### Piece 2 : Per-tree HOT → SRT compaction
- `src/hot_compact.{h,c}` — `hot_compact_tree_v2(main_srt, hot_leaves, out_srt)`
- Streaming leaf-by-leaf, RAM bornée (sparse index + petit buffer)
- Handle input V2 ou V3 (varbyte decode), sortie V2 (raw uint32, +3× disque temporaire)
- Union sort + dedup par leaf
- `forest_reopen_tree(tree_id)` swap fd Forest après rename atomique
- FFI : `mg_hot_compact_tree`, `mg_hot_compact_all`

### Piece 3 : Bulk swap
- `mg_forest_swap_dir(forest, new_dir)` — reopen tous les fd depuis nouveau dir
- Assume schema identique (n_trees, dim, depth, sub_dim)
- Combine avec build CLI externe pour construire shadow dir → swap

## E2E test validé

`/tmp/test_hot_compaction.py` :

1. Copy SIFT 10M mono to scratch dir
2. Open Forest + HOT
3. Inject fake_doc_id dans 64 build-leaves (query probe path)
4. Query : ✓ fake trouvé vote=64
5. `mg_hot_compact_all(64 trees)` en 60 s wall (V3 decode dominant)
6. Query après compact (HOT drained) : ✓ fake toujours vote=64

**Merge MAIN+HOT sémantiquement correct.**

## Limitations MVP

1. **Output SRT V2 uniquement** — 3× disque temporaire vs V3. Ajouter V3 encode = ~50 lignes varbyte.
2. **60 s / 64 trees = 940 ms/tree** — dominé par V3 decode + qsort per-leaf. Optimiser :
   - Skip decode si input est V2
   - K-way merge sorted (docs sorted au build) au lieu de qsort
3. **Pas de background thread** — compaction actuellement synchrone via FFI. Wrap dans threading Python en attendant thread C.
4. **Bulk swap : pas de rollback auto** si new_dir corrupt post-swap. Faut protéger avec `.old` backup + verify.
5. **HOT total_docs stat pas décrémenté** au snapshot — cosmétique, pas correctness.

## Next

1. V3 output encoding (~50 lignes) → compaction ×3 disk-friendly
2. Background compaction thread C (~200 lignes) → cadence continue transparente
3. Rate-limit SDK 1k vec/s côté client
4. Bench compaction concurrent avec queries (thread stress test)

## Complété 2026-07-17 (session suite)

### V3 output encode LIVRÉ
- `hot_compact_tree(..., out_format)` : 2 = V2 raw, 3 = V3 varbyte delta
- FFI `mg_hot_compact_tree_ex(..., out_format)` + default V3 via `mg_hot_compact_tree`
- Output V3 = varbyte encoding standard mangrove-search (compatibilité prod)

### Background compaction thread LIVRÉ
- `src/hot_compact_bg.{h,c}` : pthread round-robin per-tree, throttled sleep_ms
- Trigger : HOT tree > threshold_docs
- `pthread_rwlock_t` sur forest_reopen_tree pour synchro fd swap ↔ query
- Atomics (`stdatomic.h`) pour stop flag + stats
- FFI : `mg_hot_compact_bg_start/stop/n_compactions/n_docs_merged`

### SDK rate limiter LIVRÉ
- `client.py::Client` : new params `ingest_rate_limit=1000.0` (default), `ingest_rate_burst=1024`
- Token bucket thread-safe (mutex), block dans `insert()` et `insert_batch()`
- Défault 1k vec/s aligné avec design HOT + compaction

### Stress bench validé
Config : SIFT 10M mono, qd=20, tp=1024, top_n=6000, 500 queries
- Baseline (idle) : mean 12 ms / p99 34 ms
- Stress (1k vec/s ingest + bg compact + queries) : mean 13 ms / p99 31 ms
- Delta : **+6.6 % mean / -8 % p99** = dans le bruit
- 7 compactions déclenchées pendant le run, 25 k docs merged live
- Pipeline complet validé sous charge concurrente réaliste
