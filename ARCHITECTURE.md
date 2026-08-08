# Architecture

Detailed walkthrough of how mangrove-search keeps process RSS bounded while
serving filtered ANN queries over billion-doc corpora.

## Storage layout

A forest is `n_trees` independent random-projection trees, each persisted to
`<index_dir>/tree%05d.srt`. The tree topology is **deterministic from a
seed**: tree `t` uses `tree_seed(t) = t * 99991 + 7`. At each internal node
`n`, two hyperplanes are derived from `node_seed(ts, 2n)` and
`node_seed(ts, 2n+1)`. Query traversal recomputes those hyperplanes on the
fly — no per-tree state is stored in RAM.

### .srt file format

```
[ 24 B header        ]   magic | depth | n_nonempty | total_docs |
                         sample_stride | data_bytes
[ samples × 4 B      ]   first leaf_id of every sample_stride'th sparse entry
[ sparse_index × 8 B ]   sorted by leaf_id, each = (uint32 leaf_id,
                         uint32 byte_offset_into_data_block)
[ sentinel × 8 B     ]   (0xFFFFFFFF, data_bytes)
[ data block         ]   per leaf: u32 first_doc, then VarByte(delta) per
                         subsequent doc. Doc_ids sorted asc within a leaf.
[ xxhash64 trailer   ]   8 B XXH64 of everything above
```

There are two on-disk versions — **SRT2** (legacy uint32 doc_ids, offsets in
doc indices) and **SRT3** (delta VarByte, offsets in bytes). The reader
auto-detects via the magic field. Build always emits SRT3.

### Why VarByte deltas

For a 2-doc leaf, raw uint32 takes 8 B; delta+VarByte takes 5–6 B. The
savings compound at 1B scale: the data block on a dense filter scan is the
main I/O cost, and delta encoding shrinks it 1.3–2×. The encoding/decoding
is per-leaf only; the K-way merge sees decoded `uint32_t*` arrays exactly
as in SRT2.

### Atomic write + xxhash

Every `.srt` is written to `.srt.tmp`, fsynced, then renamed. The xxhash64
trailer is computed after the rename and appended. The `verify` CLI command
streams the whole file to recompute the hash and compares with the
trailer — guards against bit-flip corruption and partial writes.

## Query flow (single forest)

```
                        ┌─────────────────────────┐
qvec ─▶ normalize ─▶    │  traverse_sub per tree  │
                        │  pick_dims + gen_vec    │
                        └────────┬────────────────┘
                                 │ leaf_id at qd
                  ┌──────────────┴────────────────┐
                  ▼                                ▼
       phase 1: 2× io_uring             (per-tree, parallel)
       read sparse_index window
       around (low_leaf, high_leaf)
                  │
                  ▼
       phase 2: io_uring read raw bytes
       [byte_start, byte_end) for tree t
                  │
                  ▼
       VarByte decode → uint32[] cursors
                  │
                  ▼
       K-way merge with filter-aware skip
       (cursor_seek_allowed iterator)
                  │
                  ▼
       top-N by votes (min-heap)
                  │
                  ▼
       L2 rerank top-N → final top-K
       (io_uring batched read on base file)
```

Two `io_uring` submission rounds per query: one for sparse-index windows,
one for posting-list data. Both use **`O_RDONLY` + `io_uring_prep_read`** —
no mmap on the hot path. The kernel page cache absorbs repeated reads
(visible in `/proc/meminfo:Cached`), but **process RSS stays bounded**:
that's the central design rule.

## Filter-aware merge

The filter is a CRoaring bitmap. Two construction paths:

1. **ClickHouse**: `groupBitmapState(internal_id)` aggregate, serialized to
   wire bytes via `cast(state, 'String')`. Decoded with `croaring_io`.
2. **Raw**: int32 doc_id array (`mg_forest_query_ids`).

In the K-way merge, `cursor_seek_allowed` advances each tree's cursor to
the next doc that the filter accepts, using the bitmap's iterator
(forward-monotonic) plus a binary search inside the cursor's sorted list.
Skip is O(log K + log L) per disallowed run instead of O(L).

Tombstones (soft-deleted doc_ids) are AND-NOT'd with the user filter at
query start, producing a composed effective filter for that query.

## Pairwise-seed multi-index

Multiple forests built on disjoint corpus slices share the same `tree_seed`
schedule, so tree `t` of forest A and tree `t` of forest B navigate
**identical hyperplanes**. At query time, the leaf reached at depth qd is
the same across all forests — only the data block differs (each shard
contributes its own posting list). The K-way merge concatenates them per
tree, qsort if qd < build_depth, then merges across trees as usual.

This enables true horizontal scaling: build N shards in parallel on
disjoint slices, hot-swap any shard for compaction without affecting the
others.

## RSS targets

| Corpus      | Cible RSS | Pourquoi                                       |
| :---------- | --------: | :--------------------------------------------- |
| < 100M docs |  ≤ 100 MB | sparse_index sample table + scratch buffers    |
| 100M – 1B   |  ≤ 800 MB | + tombstones bitmap if used + larger samples   |
| > 1B        |  ≤ 1.6 GB | + composed filter for AND-NOT                  |

The kernel page cache grows with the working set on hot queries — but the
process accounting (`RssAnon` + `RssFile`) never charges that memory. On
edge / VPS deployments with constrained RSS, this is the differentiator.

## What's deliberately NOT in the design

- **No mmap on hot path.** Would inflate `RssFile` on RAM-rich machines,
  hiding the differentiator. See `memory/feedback_no_mmap.md`.
- **No per-leaf min/max ranges.** Our pairwise-seed traversal spreads
  doc_ids randomly across leaves; per-leaf bounds collapse to
  `[0, n_docs)` and add no skip benefit (see `ROADMAP_10D.md` A1 abandon).
- **No HNSW-style dynamic insertion.** The intended ingest path is
  LSM-style: active shard (small, writable, fast-rebuild) + frozen tiers.
- **No write-side locking.** Forest is single-threaded for queries (one
  io_uring ring); higher concurrency = multiple processes or future
  ring-pool refactor.

## Mutations (P1)

- **Soft delete** (implemented): persistent `tombstones.roaring`, applied
  at query time. `rpforest delete <idx> <id...>` or FFI
  `Forest.tombstone_add` + `tombstones_flush`.
- **Insert incremental** (planned): LSM active shard, periodic merge.
- **WAL durability** (planned): append-only log for crash-safe deletes
  between flushes.
- **Compaction** (planned): rewrite frozen .srt without tombstones when
  `tombstones / total > 15 %`.
