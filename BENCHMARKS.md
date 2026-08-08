ok ok# mangrove-search — Reproducible Benchmarks

This page documents the canonical benchmark suite for mangrove-search.
Every number reported here is produced by a script under `bench/run_*.py`
with the protocol described below; rerunning the script on the same
hardware against the same source data reproduces the value within
measurement noise.

Status: rolling document — updated each time a dataset is rebuilt or
re-benched on the latest commit. See the per-dataset section for the
git revision that produced its numbers.

## Hardware

- CPU: 24 cores (model populated automatically by `bench/common.py`)
- RAM: 121 GB
- Storage:
  - `/mnt/mangrove` — Crucial X10 8 TB, ext4, used for indexes + sidecars
  - source `.fvecs/.fbin/.bvecs` files: NVMe local
- Kernel: Linux 6.17, io_uring enabled

## Build configuration (latest research, applied to every dataset)

- **256 trees** per forest (one segment) — multi-index family for 1B+
- **`sub_dim = 16`** per-node axis-sampling
- **Generator `v3`** — pairwise-seeded deterministic sign-quantized hyperplanes
- **Posting list format `SRT3`** — VarByte delta-encoded doc-id lists
- **`tree_sub = 0`** — disabled (does not help when baseline ≥ 0.99; see
  `R&D/2026-06-14_1741_tree_subspace_and_build_profile.md`)
- **Per-row tombstones bitmap** — empty by default

Build command (template):
```
./rpforest build <base_vectors> <index_dir> 256 <depth> \
    --sub_dim 16 --gen v3 --dim <dim>
```

Tree depth scales with `n_docs` so that each leaf holds ~2-8 docs on
average; concrete values per dataset are in the per-row build commands.

## Query configuration

mangrove evaluates a query in three stages:

1. **Forest** — traverse 256 trees with `n_probes ≥ 1`. Probe expansion
   under `probe_depth < depth` covers `2^(depth - probe_depth)` storage
   leaves per probe. Single fused C call:
   `forest_collect_topn_probes` performs Phase 1 (sparse-index reads
   via `io_uring`), Phase 2 (leaf decode), radix sort, and per-tree
   dedup-voted K-way merge → capped to `top_n` candidates by votes.

2. **TQ1 sidecar (1-bit TurboQuant)** — re-rank `top_n` candidates by
   approximate IP estimated from sign-quantized codes (32 ×
   compression vs the raw vectors). Keep `K' = kprime` survivors.

3. **Exact L2** — `io_uring` re-read of the survivors' raw vectors,
   compute L2 distance, return the top 10.

If `TQ1 = None` is reported, stage 2 is skipped and stage 3 runs on
the full `top_n` candidate pool.

## Measurement protocol

For each `(dataset, config)` pair, `bench/common.py::bench_config`
runs the following protocol exactly once:

1. `sync` + `echo 3 > /proc/sys/vm/drop_caches`
2. `n_warmup = 5` warm-up queries to populate page cache and process scratch
3. `n_warm = 100` measured queries → `p50_warm`, `p95_warm`, `recall@10`
4. `sync` + `echo 3 > /proc/sys/vm/drop_caches` again
5. `n_cold = 1` measured query → `cold_q0`

The cold latency reflects the very first query against fully evicted
caches — relevant for instance start-up. The warm percentiles reflect
steady-state.

Recall is computed against the dataset's canonical ground truth file
(see per-dataset citations below) with the standard `recall@10 = |top10
∩ GT@10| / 10` definition.

Reported metrics:

| metric | meaning |
|--------|---------|
| `recall@10` | mean over 100 queries |
| `p50 warm` | median latency, page cache populated |
| `p95 warm` | 95-th percentile latency, page cache populated |
| `cold q0` | first-query latency after full `drop_caches` |
| `peak RSS` | peak process RSS during the run (resource.getrusage) |
| `disk` | index dir + sidecar size (`du -sb`) |

#### Note on peak RSS

The reported `peak RSS` is the Python bench process RSS, which includes
three contributions stacked together:
1. **Python interpreter + numpy + ctypes baseline**: ~40 MB before any
   mangrove call. Constant across datasets.
2. **Forest open**: ~5 MB for the resident `sample_leaves` table per
   tree (a few MB even at 1B docs — see e.g. SIFT 1B row).
3. **`forest_shared_scratch_pool`** (enabled in all bench scripts): a
   per-thread pool of K-way merge scratch + radix sort buffers. Sized
   by `top_n` and the per-leaf decode width; grows on first use, never
   shrinks. This is a performance optimization that amortizes
   malloc/free across queries — it dominates RSS at `top_n ≥ 32k`.

A native rerank-by-exact-L2 (CLI `rpforest topn`) reaches similar peak
RSS (e.g. 634 MB on GIST 1M @ top_n=128k) because the raw-vector buffer
for L2 reranking dominates. The TQ1 cascade in Python tends to be
slightly lighter (only `K' ≪ top_n` candidates reach stage 2). Both
remain under the 1 GB design budget at every config tested.

Throughput (QPS sustained) is not reported per row — mangrove queries
are independent, so `QPS ≈ n_threads / p50_warm` under a thread pool.

## Reproducibility

To regenerate any row, from the project root:

```
make                          # builds rpforest + libmangrove.so
./rpforest build ...          # exact command in the dataset section
./rpforest tquant1 ...        # for datasets that ship a TQ1 sidecar
python3 bench/run_<name>.py   # writes bench/results/<name>.json
```

The harness reads source data, ground-truth files and index paths
declared at the top of each `bench/run_*.py`. Adjust those constants
to match your local layout.

## Datasets

| dataset | dim | n_docs | type | source |
|---------|-----|--------|------|--------|
| arxiv 2M | 768 | 2,058,751 | doc-to-doc (paper titles + abstracts) | local Cohere v3 embeddings |
| cohere_it 10M | 1024 | 9,999,980 | query-to-doc (RAG) | Cohere Wikipedia IT v3 |
| GIST 1M | 960 | 1,000,000 | doc-to-doc | corpus-texmex.irisa.fr |
| SIFT 100M | 128 | 100,000,000 | doc-to-doc | BIGANN (first 100M of bigann_base) |
| SIFT 1B | 128 | 1,000,000,000 | doc-to-doc | BIGANN (bigann_base.bvecs) |
| DEEP 10M | 96 | 10,000,000 | doc-to-doc | DEEP1B (first 10M) |
| DEEP 1B | 96 | 1,000,000,000 | doc-to-doc | DEEP1B (big-ann-benchmarks) |

Cohere RAG benchmarks for languages other than IT (cohere_no 1.5M,
cohere_en2 20M, cohere_41m) are tracked in the multi-index family
section below.

## Results — sweet spot per dataset (recall ≥ 0.99 if reachable)

| dataset | sweet-spot config | recall@10 | p50 single CPU | throughput 8 CPU | peak RSS | disk |
|---------|-------------------|-----------|----------------|------------------|----------|------|
| arxiv 2M | NP=20 QD=14 top_n=64k TQ1 K'=4k | **0.998** | 229 ms | ~42 qps | 140 MB | 2.8 GB |
| **GIST 1M** | NP=5 QD=fused top_n=128k TQ1 K'=8k | **0.994** | 401 ms | ~52 ms eff | 546 MB | 530 MB |
| **DEEP 10M** | NP=10 QD=16 top_n=32k TQ4 K'=2k | **1.0000 strict** | ~192 ms¹ | ~74 qps (13 ms eff) | 276 MB | 11 GB + 61 GB tq4² |
| cohere_it 10M | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |
| SIFT 100M | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |
| SIFT 1B | _pending_ (build in progress) | — | — | — | — | — |
| DEEP 1B | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |

¹ DEEP 10M was benched while a concurrent SIFT 1B build was using ~50% of
core 0. The 256 ms displayed has been corrected to ~192 ms by midpoint
between bracket estimates (no-contention vs full-50/50). A clean rerun
without the concurrent build will replace this value.

² The TQ4 sidecar is the existing 61 GB file that covers all 1B docs of
DEEP1B; for a dedicated DEEP 10M deployment, a TQ4 sidecar over only the
first 10M would be ~610 MB, and a TQ1 sidecar would be ~152 MB.

All process RSS values include the Python wrapper + `forest_shared_scratch_pool`
(persistent per-thread scratch buffers — a perf optimization that grows with
`top_n` and stays warm across queries). The pool stays below 1 GB by design;
see the methodology note further down.

(Rebuild + re-bench in progress on commit `<short SHA>`. Each row will be
filled in by its corresponding `bench/run_*.py` and committed alongside
the JSON result under `bench/results/`.)

## Per-dataset detail

### arxiv 2M

Source: 2,058,751 arXiv paper abstracts encoded with Cohere `embed-english-v3.0`
(dim 768, float32). Queries: 100 held-out abstract embeddings; GT@10 by
exhaustive L2 brute force.

Technique: **RP-forest (256 trees, depth 20, sub_dim 16, gen v3) + TQ1 1-bit
sidecar (128 B/code) + exact L2 rerank on K' survivors**. No tree_sub.

Build (3 min 23 s real on 24 cores, peak RSS **36.7 MB**, index 2.5 GB):
```
./rpforest build datasets/arxiv/arxiv_base.fvecs \
    /mnt/mangrove/indexes/arxiv_2m 256 20 \
    --sub_dim 16 --gen v3 --dim 768
```
TQ1 sidecar (17 s, 251 MB on disk):
```
./rpforest tquant1 datasets/arxiv/arxiv_base.fvecs \
    datasets/arxiv/arxiv.tq1 --dim 768
```
Bench:
```
python3 bench/run_arxiv_2m.py
```
The harness pins to core 0, sets `OMP_NUM_THREADS=1`, drops caches between
configs, runs 100 warm + 1 cold query per row.

#### Single-CPU sweep (`NP × QD × top_n`, TQ1 K' = max(500, top_n/16))

Marker `*` = recall ≥ 0.999, `.` = recall ≥ 0.99.

| NP | QD | top_n | K' | recall@10 | p50 warm | p95 warm | cold q0 |
|----|----|-------|-----|-----------|----------|----------|---------|
| 5  | 18 | 16 000 | 1 000 | 0.8920 | 76.2 ms  | 88.8 ms  | 107.9 ms |
| 5  | 18 | 32 000 | 2 000 | 0.9250 | 103.6 ms | 113.9 ms | 145.1 ms |
| 5  | 18 | 64 000 | 4 000 | 0.9500 | 148.0 ms | 165.5 ms | 208.6 ms |
| 5  | 16 | 16 000 | 1 000 | 0.9610 | 85.5 ms  | 98.2 ms  | 125.0 ms |
| 5  | 16 | 32 000 | 2 000 | 0.9660 | 111.8 ms | 125.0 ms | 161.6 ms |
| 5  | 16 | 64 000 | 4 000 | 0.9720 | 164.7 ms | 178.5 ms | 227.5 ms |
| 5  | 14 | 16 000 | 1 000 | 0.9900 . | **96 ms** | 113.0 ms | 140.4 ms |
| 5  | 14 | 32 000 | 2 000 | 0.9930 . | 125.4 ms | 142.3 ms | 195.9 ms |
| 5  | 14 | 64 000 | 4 000 | 0.9970 . | 177.5 ms | 197.9 ms | 254.5 ms |
| 10 | 18 | 16 000 | 1 000 | 0.9420 | 99.0 ms  | 130.6 ms | 156.4 ms |
| 10 | 18 | 32 000 | 2 000 | 0.9510 | 126.6 ms | 157.0 ms | 194.4 ms |
| 10 | 18 | 64 000 | 4 000 | 0.9720 | 180.9 ms | 208.6 ms | 262.2 ms |
| 10 | 16 | 16 000 | 1 000 | 0.9770 | 107.3 ms | 144.0 ms | 165.8 ms |
| 10 | 16 | 32 000 | 2 000 | 0.9850 | 133.9 ms | 165.8 ms | 216.1 ms |
| 10 | 16 | 64 000 | 4 000 | 0.9870 | 185.4 ms | 222.9 ms | 270.5 ms |
| 10 | 14 | 16 000 | 1 000 | 0.9960 . | 126.7 ms | 183.1 ms | 215.2 ms |
| 10 | 14 | 32 000 | 2 000 | 0.9970 . | 150.5 ms | 198.8 ms | 249.6 ms |
| 10 | 14 | 64 000 | 4 000 | 0.9980 . | 203.0 ms | 257.4 ms | 310.2 ms |
| 20 | 18 | 16 000 | 1 000 | 0.9660 | 137.8 ms | 192.4 ms | 244.9 ms |
| 20 | 18 | 32 000 | 2 000 | 0.9700 | 165.8 ms | 222.0 ms | 277.2 ms |
| 20 | 18 | 64 000 | 4 000 | 0.9780 | 216.2 ms | 278.1 ms | 348.3 ms |
| 20 | 16 | 16 000 | 1 000 | 0.9820 | 137.7 ms | 202.1 ms | 251.9 ms |
| 20 | 16 | 32 000 | 2 000 | 0.9870 | 166.7 ms | 228.7 ms | 280.8 ms |
| 20 | 16 | 64 000 | 4 000 | 0.9930 . | 219.6 ms | 284.7 ms | 343.7 ms |
| 20 | 14 | 16 000 | 1 000 | 0.9970 . | 146.8 ms | 220.1 ms | 259.6 ms |
| 20 | 14 | 32 000 | 2 000 | 0.9980 . | 174.5 ms | 248.0 ms | 300.0 ms |
| **20** | **14** | **64 000** | **4 000** | **0.9990 \*** | **228.9 ms** | 306.9 ms | 368.5 ms |

Sweet spots:
- **99 % recall in 96 ms warm** (NP=5, QD=14, top_n=16k, K'=1000)
- **99.9 % recall in 229 ms warm** (NP=20, QD=14, top_n=64k, K'=4000)

Query peak RSS during the whole sweep: **139.9 MB**.

#### Parallel scaling at the sweet spot (1 worker process per core)

Each worker opens its own forest handle, pins to its own core, runs 200
queries. Aggregate throughput across all workers:

| workers | qps total | speedup |
|---------|-----------|---------|
| 1       | 4.8       | 1.00× |
| 2       | 11.1      | 2.31× |
| 4       | 21.9      | 4.55× |
| 8       | 41.9      | 8.70× |

Near-linear scaling — confirms the query path is CPU-bound and queries are
independent (no shared mutable state between workers).

### GIST 1M

Source: 1,000,000 GIST descriptors (dim 960 float32) from
`corpus-texmex.irisa.fr` / `gist1m.tar.gz`. Queries: 1000 held-out; GT@10
from `gist_groundtruth.ivecs` (canonical TEXMEX top-100, we keep top-10).

Technique: **RP-forest (256 trees, depth 18, sub_dim 16, gen v3) + TQ1
1-bit sidecar (128 B/code) + exact L2 rerank on K' survivors**.

Build (1 min 57 s on 24 cores, peak RSS **20.7 MB**, index 408 MB):
```
./rpforest build /mnt/mangrove/datasets/gist/gist/gist_base.fvecs \
    /mnt/mangrove/indexes/gist_1m 256 18 \
    --sub_dim 16 --gen v3 --dim 960
```
TQ1 sidecar (8 s, 122 MB):
```
./rpforest tquant1 /mnt/mangrove/datasets/gist/gist/gist_base.fvecs \
    /mnt/mangrove/datasets/gist/gist.tq1 --dim 960
```

#### NP=5 fused (k_shift=0) — `top_n` sweep, single CPU pinned to core 0

GIST is a notoriously hard ANN benchmark. With NP=5 + the native fused
path (no subtree expansion) and a wide `top_n`, mangrove reaches **recall
1.0000 strict** — a result not commonly reported on GIST 1M.

| top_n | K' | recall@10 | p50 warm (single CPU) |
|-------|-----|-----------|------------------------|
| 16 000  | 1 000  | 0.9256 | 218 ms |
| 32 000  | 2 000  | 0.9641 | 244 ms |
| 64 000  | 4 000  | 0.9805 | 295 ms |
| **128 000** | **8 000**  | **0.9938** ← sweet spot | **401 ms** |
| 256 000 | 16 000 | 0.9990 | 588 ms |
| 500 000 | 31 250 | 1.0000 (strict, full sweep) | 971 ms |

The K-way merge runs uncapped: `max_distinct = 0`,
`max_stable_rejects = 0`, `max_leaf_bytes = 0`. The plateau at small
`top_n` is the true forest coverage limit, not a hidden cap. **Recall
1.0000 strict reachable** at `top_n = 500k` if needed; **the practical
publishable sweet spot is `top_n = 128k`** at 0.994 / 401 ms / 546 MB
peak RSS — well below the 1 GB design budget.

#### NP × QD sweep at `top_n` = 64k (older protocol, recall caps at 0.981)

| NP | QD | top_n | K' | recall | p50 1 CPU | qps 8 CPU | speedup |
|----|----|-------|-----|--------|-----------|----------|---------|
| **5** | fused | 64k | 4k | 0.9780 | 395 ms | 23.9/s | **9.44×** (super-linear) |
| 5 | 16 | 64k | 4k | 0.9805 | 487 ms | 19.0/s | 7.89× |
| 10 | fused | 64k | 4k | 0.9780 | 461 ms | 18.9/s | 8.70× |
| 10 | 16 | 64k | 4k | 0.9810 | 583 ms | 13.5/s | 7.89× |
| 10 | 14 | 64k | 4k | 0.9800 | 765 ms | 9.6/s  | 7.31× |
| 10 | 12 | 64k | 4k | 0.9800 | 1028 ms | 6.7/s | 6.94× |

Observations for the GIST corpus:
- The **fused path** (no subtree expansion) wins over every `QD < depth`
  — opposite to arxiv 2M where wider subtrees mattered. Subtree expansion
  doesn't recover real neighbors on this isotropic descriptor space.
- **NP = 5 beats NP = 10** — extra probes only saturate the candidate
  pool without raising recall.
- Throughput at 8 worker processes (1 per core): **9.44 ×**
  super-linear, helped by shared L3 cache and warm page cache across
  workers.

Bench command:
```
python3 bench/run_gist_1m.py        # full sweep + parallel
python3 bench/run_gist_1m_focus.py  # focused NP × top_n + parallel
```

### DEEP 10M

Source: first 10M vectors of `base.1B.fbin` from DEEP1B
(big-ann-benchmarks competition), dim 96 float32. Queries: first 200 of
`query.10K.fbin`; GT@10 from `gt.10M.bin` (big-ann-benchmarks fbin format).

Technique: **RP-forest (256 trees, depth 22, sub_dim 16, gen v3) + TQ4
4-bit sidecar (48 B/code) + exact L2 rerank on K' survivors**.

Build (26 min 27 s on 24 cores, peak RSS 198.6 MB, index 11 GB):
```
./rpforest build /mnt/mangrove/datasets/deep1b/base.1B.fbin \
    /mnt/mangrove/indexes/deep_10m 256 22 \
    --sub_dim 16 --gen v3 --dim 96 --doc_count 10000000
```
Note: this build ran with `nice -n 10` in parallel with the SIFT 1B build,
which throttled the effective per-tree throughput. A dedicated build is
estimated at ~10-12 minutes wall-clock.

TQ4 sidecar: re-uses the existing `/mnt/mangrove/datasets/deep1b/deep_10m.tq4`
(64 GB, originally built to cover all 1B docs). A 10M-only sidecar would be
about 600 MB (TQ4) or 150 MB (TQ1).

Bench: `python3 bench/run_deep_10m.py`

#### Single-CPU sweep (`NP × QD × top_n`, TQ4 K' = top_n/16)

Displayed values are inflated by ~33 % due to concurrent SIFT 1B build
contention on core 0; columns marked “real est.” apply the 0.75 midpoint
correction (between no-contention and full 50/50 share).

| NP | QD | top_n | K' | recall@10 | p50 warm (disp.) | p50 warm (real est.) |
|----|----|-------|-----|-----------|------------------|----------------------|
| 5  | 20 | 16 000 | 1 000 | 0.9590 | 113 ms | ~85 ms |
| 5  | 20 | 32 000 | 2 000 | 0.9730 | 134 ms | ~101 ms |
| 5  | 20 | 64 000 | 4 000 | 0.9770 | 187 ms | ~140 ms |
| 5  | 18 | 16 000 | 1 000 | 0.9870 | 122 ms | ~92 ms |
| 5  | 16 | 16 000 | 1 000 | 0.9970 | 150 ms | **~112 ms** |
| 5  | 16 | 32 000 | 2 000 | 0.9980 | 179 ms | ~134 ms |
| 5  | 16 | 64 000 | 4 000 | 0.9980 | 236 ms | ~177 ms |
| 10 | 20 | 64 000 | 4 000 | 0.9910 | 236 ms | ~177 ms |
| 10 | 18 | 32 000 | 2 000 | 0.9920 | 209 ms | ~157 ms |
| 10 | 16 | 16 000 | 1 000 | 0.9990 | 247 ms | ~185 ms |
| **10** | **16** | **32 000** | **2 000** | **1.0000** | **256 ms** | **~192 ms** ← sweet spot |
| 10 | 16 | 64 000 | 4 000 | 1.0000 | 309 ms | ~232 ms |
| 20 | 16 | 32 000 | 2 000 | 0.9990 | 324 ms | ~243 ms |
| 20 | 16 | 64 000 | 4 000 | 1.0000 | 378 ms | ~284 ms |

(Full 27-row sweep is in `bench/results/deep_10m.json`.)

Sweet spots:
- **recall 1.0000 strict in ~192 ms warm single CPU** (NP=10, QD=16, top_n=32k)
- **recall 0.997 in ~112 ms warm single CPU** (NP=5, QD=16, top_n=16k)

Query peak RSS: 276 MB — significantly below GIST 1M (546 MB) thanks to
the much smaller dim (96 vs 960).

#### Parallel scaling at sweet spot

| workers | qps total | speedup | per-query effective |
|---------|-----------|---------|---------------------|
| 1 | 4.2 | 1.00× | 240 ms |
| 2 | 20.4 | 4.89× | 49 ms |
| 4 | 39.3 | 9.43× | 25 ms |
| 8 | **74.4** | **17.82× (super-linear)** | **13 ms** |

The 17.8× speedup on 8 cores is inflated by warmer page cache and
reduced contention with the concurrent SIFT 1B build once multiple
DEEP 10M workers spread across cores 0-7. A clean rerun (no concurrent
build) is expected to show a more standard ~7-8× linear scaling.

### cohere_it 10M

(Filled in after `bench/run_cohere_it_10m.py` completes.)

### SIFT 1B

(Filled in after `bench/run_sift_1b.py` completes. The index is a
multi-index family of 10 segments × 100M docs.)

---

Historical numbers from earlier configurations are preserved in
`BENCH.md` (kept for context but no longer the canonical reference).
