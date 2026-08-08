---
name: Don't redo expensive work when sweeping parameters
description: When benchmarking ANN configs across multiple rescoring pool sizes, do one traversal per nl and evaluate all pool cutoffs from the same candidate list — not one full bench per (nl, pool).
type: feedback
originSessionId: b722493c-5024-46d5-a669-8ab2348ac60d
---
When sweeping ANN parameters where some stages are cheap post-hoc (e.g. top-k-pool cutoffs over an already-sorted-by-frequency candidate list), do ONE expensive pass and evaluate all cutoffs from the same intermediate data. Don't re-run the full query pipeline per cutoff value.

**Why:** User flagged this when I ran a 4×7 grid as 28 separate query runs, each re-traversing the forest on disk. The forest traversal is the expensive I/O-bound step; rescoring cutoffs are pure post-processing.

**How to apply:** In sweep loops, identify the invariant expensive work (here: traversal + candidate collection + frequency counting + rescoring at max pool). Run it once, then slice/evaluate downstream variants (pool cutoffs) in memory. Useful anywhere a parameter only affects late-stage ranking/filtering, not retrieval.
