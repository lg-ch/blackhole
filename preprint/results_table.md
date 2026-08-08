# Consolidated results — DEEP corpora (SIFT 1B to complete after build)

## Table 1 — Main results, under strict cgroup 1 GB RSS + cold cache

All measurements: 30 queries, `drop_caches` between each query, page cache
enforced to 0 pre-query, systemd-run `--scope -p MemoryMax=1G
-p MemorySwapMax=0`. Rerank stage: exact L2 top-10 via io_uring.

| dataset | pipeline config | recall@10 | mean cold | p50 cold | p95 cold | peak RSS | disk index |
|---------|-----------------|-----------|-----------|----------|----------|----------|-----------|
| DEEP 10M (d=12) | NP=3 600×1000 mlb=200k L2 | 0.990 | 137 ms | 118 ms | 233 ms | ≤ 400 MB | 4.7 GB |
| DEEP 100M (d=16) | NP=3 1024×2000 mlb=200k L2 | 0.973 | 296 ms | 261 ms | 508 ms | 672 MB | 59 GB |
| DEEP 100M (d=16) | **NP=3 1024×4000 mlb=200k L2** ★ | **0.987** | **357 ms** | **319 ms** | **578 ms** | **647 MB** | **59 GB** |
| DEEP 100M (d=16) | NP=7 1200×4000 mlb=200k L2 | 0.990 | 393 ms | 352 ms | 643 ms | 730 MB | 59 GB |
| DEEP 1B (d=18) | NP=3 1024×4000 mlb=200k L2 | 0.970 | 540 ms | 565 ms | 692 ms | 682 MB | 639 GB |
| DEEP 1B (d=18) | NP=3 1024×6000 mlb=200k L2 | 0.977 | 582 ms | 615 ms | 703 ms | 684 MB | 639 GB |
| DEEP 1B (d=18) | **NP=3 1024×8000 mlb=200k L2** ★ | **0.983** | **630 ms** | **659 ms** | **758 ms** | **684 MB** | **639 GB** |
| SIFT 1B (d=18) | _pending_ | — | — | — | — | — | ~640 GB |

**Legend :**
- NP = multi-probe count (paths per tree = NP+1)
- `600×1000` = 600 paths kept by margin ranking × 1000 top_n cap by vote count
- `mlb=200k` = `max_leaf_bytes` = per-leaf byte cap (skips leaves > 200 KB varbyte)
- `L2` = exact L2 rerank stage (top-10)

## Table 2 — WARM ≈ COLD property under cgroup 1G

Design invariant: under the target RAM budget, WARM and COLD have
identical latency because the OS page cache cannot hold the multi-GB
index within the cgroup memory limit.

| dataset | config | WARM p50 | COLD p50 | Δ |
|---------|--------|----------|----------|---|
| DEEP 100M (d=16) | NP=3 1024×4000 mlb=200k | 319 ms | 319 ms | 0 % |
| DEEP 1B (d=18)   | NP=3 1024×4000 mlb=200k | (measure) | 565 ms | ~0 % expected |

This is a property, not a limitation: mangrove delivers predictable
latency independent of cache-warmup state.

## Table 3 — Build stats

| dataset | depth | threads | classic rate | `--fast` rate | build wall | peak RSS build | final disk |
|---------|-------|---------|--------------|---------------|------------|----------------|------------|
| DEEP 100M (d=14) | 14 | 20 | ~15 k v/s | — | 158 min | 500 MB | 52 GB |
| DEEP 100M (d=16) | 16 | 20 | 15.5 k v/s | 20.5 k v/s | 3 h 0 min | 500 MB / 2.5 GB (--fast) | 59 GB |
| DEEP 1B (d=18) | 18 | 20 | (extrapolated 13 k v/s) | 16-18 k v/s | 30 h (--fast) | 8.4 GB (--fast) | 639 GB |
| SIFT 1B (d=18) | 18 | 20 | — | ~45 k v/s (bvecs light I/O) | ETA ~19 h | 8.4 GB (--fast) | ~640 GB |

## Notes on hardware

- Machine : 20 physical cores, 128 GB RAM, Crucial X10 8 TB NVMe SSD (~1.5 GB/s read seq, ~500k IOPS random 4 KB)
- OS : Linux 6.17, glibc 2.39
- Query pinned to 1 core (`sched_setaffinity`, `OMP_NUM_THREADS=1`)
- Bench script : `/tmp/deep_1b_topn.py` and equivalents (published in artifact repo)

## Reproducibility snippet

```bash
# Build
./rpforest build /path/to/base.1B.fbin /path/to/index 256 18 \
    --sub_dim 16 --gen v3 --dim 96 --doc_count 1000000000 \
    --fast --batch 8192

# Query bench under strict RAM budget
sync && echo 3 > /proc/sys/vm/drop_caches
systemd-run --scope --unit=mangrove -p MemoryMax=1G -p MemorySwapMax=0 \
    python3 bench/run_deep_1b.py --config "NP=3 1024x8000 mlb=200k"
```
