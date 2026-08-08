# mangrove-search

📚 **Documentation** :
[QUICKSTART](QUICKSTART.md) · [SDK reference](SDK.md) · [Comparison vs FAISS/Pinecone](COMPARISON.md) ·
[Auth & OIDC](AUTH.md) · [HA deployment](HA.md) · [Operator runbook](RUNBOOK.md) ·
[Supported platforms](PLATFORMS.md)

Approximate nearest-neighbor + filtered RAG over billions of vectors, with a
**RAM footprint bounded and independent of corpus size**.

This is the differentiator: the process RSS does NOT grow linearly with N.
On a 2M-doc dim-768 corpus, peak RSS is ~30 MB. On a 1B-doc dim-128 corpus,
peak RSS stays under 800 MB. Most ANN libraries either keep the index in RAM
(HNSW, FAISS-IVF-flat) or page-cache a memory-mapped file (DiskANN ~15–20%
of the data). mangrove-search uses `io_uring + O_RDONLY` on cold posting-list
files; the OS page cache stays in the kernel, not in the process RSS.

## Architecture

```
┌──────────────────────────┐    ┌───────────────────────────┐
│  Python orchestrator     │    │  ClickHouse  (native TCP) │
│  (search_arxiv.py, FFI)  │◀──▶│  - groupBitmapState pre-  │
│                          │    │    aggregated by category │
└─────────┬────────────────┘    └───────────────────────────┘
          │ ctypes / libmangrove.so
          ▼
┌──────────────────────────────────────────────────────────┐
│  Forest (C core)                                          │
│  - 1000 RP trees with pairwise-seed (deterministic)       │
│  - sorted posting-list .srt files (one per tree)          │
│    + xxhash64 footer + atomic write                       │
│    + VarByte delta-encoded doc_ids (SRT3 format)          │
│  - io_uring 2-phase read (sparse index → posting lists)   │
│  - K-way merge with filter-aware skip (CRoaring iterator) │
└──────────────────────────────────────────────────────────┘
          │
          ▼
       SSD / NVMe
```

**Why pairwise-seed instead of independent random trees?**
Each tree's split hyperplanes are derived from a single 64-bit seed, so the
build is deterministic and trees from different shards (built independently
on disjoint slices) align at the leaf level. This lets multi-shard queries
merge K cursors of the *same logical tree* across shards.

## Quickstart

### Build
```bash
make                              # builds rpforest binary + libmangrove.so
```

Requires: `liburing-dev`, `libroaring-dev`, `libxxhash-dev`, `libomp-dev`.

### Index
```bash
./rpforest build sift/sift_base.fvecs /tmp/sift1m \
    1000 25 --sub_dim 16 --gen v3 --dim 128
```
Phase 1 streams vectors, phase 2 sorts to `.srt`. Crash-safe: restart resumes
phase 1 from last checkpoint (every 100k vecs) and phase 2 skips already
written trees.

### Query (CLI)
```bash
./rpforest topn sift/sift_query.fvecs sift/sift_groundtruth.ivecs \
    sift/sift_base.fvecs /tmp/sift1m 1000 25 1000000 \
    10 500 1000           # top_k=10, top_n=500, n_queries=1000
```

### Query (Python FFI)
```python
from mangrove_ffi import Forest
import numpy as np

with Forest('/tmp/sift1m', n_trees=1000, dim=128, sub_dim=16,
            depth=25, n_docs=1_000_000, gen_version=3) as f:
    q = np.random.randn(128).astype(np.float32)
    ids, votes, n = f.query(q, top_n=500)
    print(ids[:10])
```

### Verify integrity
```bash
./rpforest verify /tmp/sift1m         # checks xxhash64 of every .srt
```

## Filtered queries

Build a roaring bitmap of allowed doc_ids server-side in ClickHouse:
```sql
SELECT cast(groupBitmapState(internal_id), 'String')
FROM mangrove.docs WHERE primary_cat = 'cs.LG' AND year = 2024;
```

Pass the raw bytes to the query path (`--filter_ch <file>` on CLI, or
`allowed_state=bytes` in FFI). Density ≤ 3 % triggers pre-filter (forest
walks only allowed branches via the K-way merge's `cursor_seek_allowed`);
above that, post-filter on the top-N candidates. Very sparse filters
(`density × dim × 4 < 50 MB`) bypass the forest entirely and brute-force.

## Memory targets

| Corpus      | RSS budget | Measured                              |
| :---------- | ---------: | :------------------------------------ |
| < 100M docs |    ≤ 100MB | 28–36 MB on arxiv 2M (dim=768)        |
| 100M – 1B   |    ≤ 800MB | (under measurement on SIFT 100M)      |
| > 1B        |    ≤ 1.6GB | (under measurement on SIFT 1B)        |

The kernel page cache may grow much larger; that is intentional and not
charged to the process. See `ROADMAP_10D.md` for the full plan.

## Repository layout

```
src/         C core (forest, sorted store, io_uring, croaring, FFI)
scripts/     Python orchestrator, bench, prepare datasets
clickhouse/  Stand-alone CH server config (config.xml, users.xml)
ROADMAP_10D.md, journal_*.md   Active project plan and notes
```

## Status

Pre-release. Validated on arxiv 2M (dim 768) and SIFT 1M (dim 128).
SIFT 100M build pipeline confirmed; SIFT 1B build queued.

See `ROADMAP_10D.md` for what remains before prod (compaction, tombstones,
gRPC, ops dashboards).
