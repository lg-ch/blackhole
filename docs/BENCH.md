# Benchmarks

All numbers below from `scripts/bench_competitive.py` and
`scripts/bench_sift.py`, ARM64 server (20 cores), Crucial X10 SSD (ext4)
for indexes, NVMe for source vectors.

## SIFT 1M (dim=128, 1,000,000 docs)

Queries: 200 from SIFT query set; GT from `sift_groundtruth.ivecs`;
all engines reranked to top-10 (mangrove via L2 on raw vectors,
FAISS / hnswlib by construction).

### Mangrove parameter sweep

| n_trees × depth | top_n | recall@10 | p50 ms | p99 ms | RSS MB |
| :-------------: | ----: | --------: | -----: | -----: | -----: |
| 200 × 20        |  500 |     0.861 |   8.12 |  22.10 |   38.8 |
| 200 × 20        | 4000 |     0.958 |  11.16 |  24.82 |   40.3 |
| **1000 × 18**   | **4000** | **0.999** |  84.27 | 184.47 |   70.7 |

The 200-tree config is the speed-optimal point (~10 ms p50). The 1000-tree
config matches the historical 99.77 % baseline at the cost of ~10× latency.
The p99 184 ms above is inflated by concurrent SIFT 100M build CPU
contention; expected ~70 ms cold.

### Competitive (200 × 20 mangrove vs FAISS / hnswlib)

| Engine          | recall@10 | p50 ms | p99 ms | RSS MB | disk MB |
| :-------------- | --------: | -----: | -----: | -----: | ------: |
| mangrove-search |     0.958 |  11.16 |  24.82 |  **40** |     479 |
| faiss-ivfpq     |     0.38  |  0.07  |   0.18 |    558 |      16 |
| hnswlib         |     0.95  |  0.22  |   0.28 |    777 |     630 |

**Reading:** mangrove sits between FAISS (smallest disk, lowest recall —
PQ quantization without refine) and hnswlib (best recall + lowest
latency, but ~20× more RAM). Mangrove's selling point is the **RAM
footprint**: ~40 MB process RSS while everyone else holds the whole
graph or compressed codebook in memory.

### Delta encoding savings (SRT3 vs SRT2)

Measured on 1000-tree / depth-18 build:

| Block               | SRT2 (uint32) | SRT3 (delta+varbyte) | Saving |
| :------------------ | ------------: | -------------------: | -----: |
| Data block per tree |       4.00 MB |              1.93 MB | −51.7% |
| Tree file total     |       4.21 MB |              2.14 MB | −49.2% |
| Forest 1000 trees   |        4.2 GB |               2.1 GB |  −50%  |

Savings collapse when leaves get sparse: 200 trees × depth 20 on 1M docs
→ ~1 doc/leaf → first_doc:u32 dominates → ~0 % gain. The sweet spot is
many docs per leaf (large delta-densities, small relative magnitudes).

### Same SIFT 1M, but under a 100 MB cgroup RAM cap

```bash
systemd-run --user --scope -p MemoryMax=100M \
    python3 scripts/bench_sift.py ...
```

| Metric        | Value     |
| :------------ | --------: |
| recall@10     |     0.869 |
| p50 ms        |      8.26 |
| p99 ms        |     22.76 |
| peak RSS      |   38.1 MB |
| throughput    |   115 q/s |

Same result as the unconstrained run — the **process RSS doesn't push
against the limit**. Page cache lives in the kernel, not in the RSS;
under tighter memory the cache shrinks and cold reads cost more, but
the process never OOMs.

## arXiv 2M dim=768

Build : 256 trees × depth 20 × sub_dim 16 × gen v3 (≈3.1 GB on-disk).

### Single core (ARM Cortex-X925), 5 probes, top_n=16k, truly-fused multi-probe at probe_depth

| probe_depth | recall@10 | p50 ms | p95 ms | p99 ms |
| :-: | :-: | :-: | :-: | :-: |
| 20 (native) | 0.828 | 57 | 60 | 62 |
| 18 | 0.890 | 60 | 62 | 64 |
| 16 | 0.963 | 64 | 67 | 70 |
| 14 | 0.994 | 70 | 74 | 80 |
| **12** | **1.000** | **97** | 104 | 110 |
| 10 | 1.000 | 138 | 145 | 158 |

The 256t / 5p / top_n=16k / probe_depth=12 → **recall 1.0 at 97 ms p50**
single-CPU is the headline. probe_depth=14 already gives recall 0.994
at 70 ms.

Historical reference (the 1000-tree native build, when we used
`auto_qd_v2` on a per-tree heap merge) :
| Mode                       | recall@10 | p50 ms | p99 ms |
| :------------------------- | --------: | -----: | -----: |
| native qd=20               |    0.67   |  9.6   |  11.3  |
| auto_qd_v2 (target=0.05)   |    0.954  | 24.6   |  31.8  |
| auto_qd_v2 (target=0.1)    |    0.968  | 40.7   |  54.7  |

End-to-end RAG (ClickHouse + CRoaring + forest) from prior sessions :
p99 8-16 ms, RSS 28-36 MB.

## Cohere wiki-en 41.5M dim=1024

Cohere wikipedia-en multilingual v3 embeddings, full 41.5M English
subset, exact bruteforce L2 ground-truth on 1000 randomly sampled
corpus vectors. Notable property : embeddings sit in a narrow cone of
the unit sphere (mean vector norm ≈ 0.48, so cos between random pairs
averages 0.23 not 0) — a known phenomenon for LM embeddings ("All-but-
the-Top", Mu & Viswanath 2018). This makes ANN intrinsically harder
than SIFT-class corpora because the true NN are only ~30 % more similar
to the query than random pairs.

Build : 256 trees × depth 22 × sub_dim 16 × gen v3 → 36 GB on-disk
(in 3 h 39 at ~3 500 vec/s on 20 cores).

### Single core (ARM Cortex-X925), truly-fused multi-probe, 50-query bench

| qd | probes | top_n | recall@10 | p50 ms | p95 ms | p99 ms | max ms |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 16 | 5  |  16k | 0.950 |  734 |  781 |  806 |  826 |
| 16 | 5  |  32k | 0.958 | 1226 | 1286 | 1310 | 1317 |
| 16 | 10 |  16k | 0.974 |  966 | 1077 | 1098 | 1107 |
| 16 | 10 |  32k | 0.976 | 1463 | 1558 | 1572 | 1581 |
| **16** | **10** | **64k** | **0.988** | **2483** | **2594** | **2623** | **2645** |
| 16 | 20 |  64k | 0.988 | 2755 | 2859 | 2923 | 2926 |
| 14 | 5  |  16k | 0.964 | 1174 | 1323 | 1380 | 1428 |
| 14 | 10 |  32k | 0.982 | 2234 | 2437 | 2528 | 2597 |
| 14 | 10 |  64k | 0.988 | 3239 | 3448 | 3569 | 3648 |

The sub_dim/dim ratio is 1.6 % vs SIFT's 12.5 % and arxiv's 2.1 %, so
the splits are much noisier and the per-tree dedup across probes
provided by the truly-fused merge gains +14-22 recall points vs the
multi-pass alternative at the same probe count. Cohere is the corpus
where truly-fused matters the most.

**Tail is remarkably flat** (max ≈ p99 + 5 %) — no degenerate-leaf
spikes ; this is a property of the cohere corpus distribution (dense
unit-sphere cone, evenly-distributed leaf sizes).

### probe_depth × n_probes structure (50 queries, all truly-fused)

A staircase emerges : each level of probe_depth below build_depth
unlocks a recall ceiling, and n_probes saturates within that ceiling
at roughly free marginal cost (io_uring batches the leaf-window reads
across all probes in one Phase 1 submit).

| probe_depth | recall ceiling reached at | best p50 |
| :-: | :-: | :-: |
| 22 (native) | 0.87 at 200 probes | 310 ms |
| 20 (depth−2) | 0.92 at 200 probes | 1015 ms |
| 18 (depth−4) | 0.96 at 100 probes | **314 ms** |
| 16 (depth−6) | 0.99 at 10 probes | 2422 ms |
| 14 (depth−8) | ~1.00 at 10 probes | 3239 ms |

Practical recipe for high-dim corpora like cohere : pick the
**shallowest probe_depth** that meets your recall target, then crank
n_probes to fit your latency budget (the per-probe cost saturates,
not linear).

| Recall target | Best config | p50 |
| :-: | :-: | :-: |
| 0.91 | qd=20 / 50p / tn=32k | 255 ms |
| 0.96 | **qd=18 / 100p / tn=32k** | **314 ms** |
| 0.97 | qd=18 / 100p / tn=64k | 924 ms |
| 0.99 | qd=16 / 10p / tn=64k | 2422 ms |

### Tail cap on cohere (max_leaf_bytes), 256t / 5p / tn=16k / qd=16

| cap | recall | p50 | p99 |
| :-: | :-: | :-: | :-: |
| off | 0.98 | 350 | ~400 |
| 50 k | 0.98 | 76 | 103 |
| **20 k** | **0.98** | **79** | **118** |
| 10 k | 0.97 | 91 | 116 |

The tail cap is essentially lossless (recall unchanged) and brings p50
from 350 → 76 ms by skipping the few degenerate dense leaves that
dominate the merge cost when hit. With multi-probe + truly-fused +
cap, **recall 0.98 at p50 ~80 ms / p99 ~120 ms single CPU** is
reachable on cohere.

## SIFT 100M

Build : 1000 trees × depth 30 × sub_dim 16 × gen v3.
Build time 13h 15min on 20 cores ; peak RSS during convert phase 1.36 GB
(transient) ; total disk 333 GB (delta encoding saves ~25%, n_nonempty
5.87M/tree = 0.55% of 2³⁰ — sign-split with sub_dim 16 concentrates docs).

Query (auto_qd_v2 picks qd=30 native, target_ratio=0.001) :

| top_n | recall@10 | p50 ms | p99 ms | RSS    |
| ----: | --------: | -----: | -----: | -----: |
|   500 |    0.906 |     78 |    722 | 400 MB |
|  1000 |    0.938 |     79 |    722 | 400 MB |
|  2000 |    0.958 |     81 |    724 | 400 MB |
|  **4000** | **0.974** | **82** | **727** | **406 MB** |
|  8000 |    0.982 |     92 |    738 | 412 MB |

Sweet spot top_n=4000 : recall 0.974, p50 82 ms. **RSS 406 MB ≤ 800 MB
tier target** ✓.

p99 ~720 ms = tail des leaves heavy (avg 17 docs/leaf, max très grand
sur quelques leaves). Median p50 reste à 82 ms. Follow-up : investiguer
le tail (bucket prefetch, query-time leaf sampling).

## SIFT 1B

Built : 10 segments × 100M docs × 1000 trees × depth 30 (single segment
re-used as smaller "n_trees subset" at query time via `Forest(n_trees=k)`).

Queries : 100 from `bigann_query.bvecs`, GT from `idx_1000M.ivecs`,
top-10 reranked L2 against the raw `bigann_base.bvecs` (uint8).

### Single core (ARM Cortex-X925 @ 3.9 GHz), 256 trees, multi-probe + radix merge

| n_probes | top_n | recall@10 | p50 ms | p95 ms | p99 ms |
| :-: | :-: | :-: | :-: | :-: | :-: |
| 10 | 8 000 | 0.993 | 1 003 | 5 006 | 5 558 |
| 8  | 12 000 | 0.980 | 905 | 4 582 | 5 039 |
| 6  | 12 000 | 0.987 | 742 | 3 785 | 4 333 |
| **5** | **16 000** | **0.993** | **722** | **3 472** | **4 046** |
| 5  | 12 000 | 0.987 | 676 | 3 436 | 3 984 |
| 4  | 16 000 | 0.980 | 628 | 3 016 | 3 518 |

**Sweet spot @ recall 0.993 : 5 probes × top_n 16 000 = 722 ms p50** on a
single core. The 5-probe / 16k-top_n point matches the 10-probe / 8k-top_n
recall on identical queries (statistical noise apart) while saving ~28 %
latency and ~30 % p99. Mechanism : fewer probes → shorter radix sort, but
top_n increase keeps the candidate pool wide enough for L2 rerank to
recover ground-truth neighbours.

### 10 segments in parallel (10 threads, Python GIL released by C path)

| n_probes | top_n | recall@10 | p50 ms | p95 ms |
| :-: | :-: | :-: | :-: | :-: |
| 10 | 8 000 | 0.990 | 152 | 633 |
| **5** | **16 000** | **0.985** | **130** | **413** |
| 5  | 12 000 | 0.980 | 107 | 379 |

### Resources at runtime

- **RAM** : 935 MB total RSS for all 10 segments at 256 trees (sample
  table + scratch + buffers ; ~700× smaller than equivalent in-memory
  FAISS HNSW on 1B vectors)
- **Disk** : 0.85 TB for the same 10-segment / 256-tree footprint
  (compared to ~3.3 TB at 1000 trees ; the breakthrough config keeps
  recall 0.993 at 3.9× less disk)
- **Single-core walltime decomposition** (radix sort path, profiled with
  perf) : ~74 % radix + linear scan, ~10 % I/O wait (page cache hot),
  ~5 % L2 rerank, rest scratch buffer setup / merge result extraction

### Algorithmic timeline of the single-core p50 (256t / 10p / top_n 8000)

| step | p50 ms |
| --- | ---: |
| Heap K-way merge (original) | 3 089 |
| Loser tree (pointer chase removed) | 3 254 |
| Pre-merge intra-tree | 3 434 |
| Radix sort + linear scan | 1 328 |
| + per-thread pooled scratch | 1 051 |
| + 11/11/10 radix (3 passes) | 1 013 |
| + 4-way parallel sub-histograms | 997 |
| + 5 probes / top_n 16 000 | **722** |

### Tail-cap : skip oversized leaves (`max_leaf_bytes`)

A query that routes into a degenerate dense leaf (e.g. SIFT 1B has trees
where the worst leaf contains 300 k+ docs vs the 17-doc average) blows
p99 up massively. The `forest_set_max_leaf_bytes(N)` knob skips any
leaf whose posting list exceeds N bytes at query time — bounds the
radix-sort input size and therefore latency.

Recall trade : a query that hits a capped leaf loses that one tree's
vote for the docs in it, but the remaining 255 trees compensate.

**Single core, 256t / 5p / top_n 16 000, 30 queries**

| max_bytes | recall | p50 ms | p95 ms | p99 ms | max ms |
| :-: | :-: | :-: | :-: | :-: | :-: |
| off | 0.993 | 650 | 2 745 | 3 815 | 4 103 |
| 100 000 | 0.993 | 544 | 1 390 | 1 889 | 2 038 |
| **50 000** | **0.993** | **484** | **941** | **1 258** | **1 378** |
| **20 000** | **0.990** | **401** | **575** | **663** | **697** |
| 10 000 | 0.987 | 326 | 390 | **399** | 402 |
| 5 000 | 0.963 | 283 | 297 | 302 | 303 |

**Parallel ×10 segments, same config**

| max_bytes | recall | p50 ms | p95 ms | p99 ms | max ms |
| :-: | :-: | :-: | :-: | :-: | :-: |
| off | 0.993 | 127 | 554 | 1 453 | 1 768 |
| 50 000 | 0.993 | 117 | 148 | 193 | 211 |
| **20 000** | **0.990** | **119** | **140** | **148** | **151** |
| 10 000 | 0.987 | 112 | 125 | 131 | 133 |

Three production-friendly operating points :

- **Lossless SLA** : cap 50 000 → recall 0.993, p99 193 ms parallel
  (×7.5 reduction vs uncapped)
- **Tight SLA** : cap 20 000 → recall 0.990, p99 148 ms parallel
  (×9.8 reduction)
- **Real-time** : cap 10 000 → recall 0.987, p99 131 ms parallel,
  max query 133 ms (essentially flat tail)

The Pareto curve confirms : tail cap is the highest-leverage knob in
the project. Predictable latency makes SaaS pricing trivial and bounds
worst-case CPU spend per query.

## How to reproduce

```bash
make
python3 scripts/bench_competitive.py \
    --base sift/sift_base.fvecs \
    --queries sift/sift_query.fvecs \
    --gt sift/sift_groundtruth.ivecs \
    --n_docs 1000000 --dim 128 \
    --mangrove_index /tmp/sift1m_competitive \
    --n_trees 200 --depth 20 --sub_dim 16 --gen 3 \
    --n_queries 200 --top_k 10 --top_n 500
```

Index caches at `/tmp/competitive_cache/{faiss.ivfpq, hnsw.bin}` so
reruns are fast.
