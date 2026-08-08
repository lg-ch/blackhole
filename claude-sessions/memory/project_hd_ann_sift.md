---
name: hd_ann_test.cpp on SIFT 1M — tuning context
description: Context for hd_ann_test.cpp benchmarking on SIFT 1M (d=128): learnings about good configs, RAM-constrained operation, I/O bottlenecks.
type: project
originSessionId: b722493c-5024-46d5-a669-8ab2348ac60d
---
Fact: hd_ann_test.cpp (standalone RP-forest ANN tool) is being tuned for SIFT 1M on disk-constrained / low-RAM deployments. Relevant learnings so far:

- SIFT 1M defaults (nl=50, pool=3000, 200 trees mt19937) are heavily over-dimensioned for this dataset. Recall saturates much earlier.
- With 500 trees + rademacher, nl=1 pool=500 gives recall ~0.98 and works well under cgroup MemoryMax=150M (12.6 QPS vs 7.2 at nl=2 pool=3000).
- Reordering base.f32 by layout-tree leaf clustering gives only ~10% QPS improvement — not worth it here. User decided to skip that optimization.
- The main lever under RAM constraint is reducing `top_k_pool` (kills random rescoring I/O), not reducing `nl` (which affects forest traversal).

Known bugs worth tracking:
1. `--gen` generator kind (mt19937/xorshift/rademacher) is a global runtime flag, NOT persisted in `forest.bin`. Load-from MUST pass the same `--gen` as build or queries silently return wrong results (recall ~= 0). See task #8.
2. **Cosmetic only**: log `using N threads, sub_dim=X, gen_v0` in `build_tree.c:105` hardcodes the `0` (`(int)0 /* gen version filled by caller via set_gen_version */`). The actual gen_version is set via `set_gen_version()` from `main.c:236`, so the build IS using the requested version — the log is just lying. Don't be misled when log says `gen_v0` even though CLI was `--gen v3`.

**Why:** Captured during a tuning session where user wanted < 100 MB RAM build for SIFT 1M, then explored cgroup-bounded bench configs.

**How to apply:** When advising on this codebase's defaults or new benchmarks, recommend smaller nl/pool than defaults for this dataset. Always remind to pass `--gen` at query time matching build.
