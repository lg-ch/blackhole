# Honest comparison vs other ANN systems

A side-by-side look at where mangrove-search wins, ties, and loses
against the major open-source and commercial vector search systems.
All numbers are measured on hardware we control unless noted ; see
`BENCH.md` for the raw bench scripts.

| Metric (SIFT 1M, dim 128, recall@10 = 0.97+) | mangrove | FAISS-IVF | hnswlib | Qdrant | Pinecone (hosted) |
| --- | --- | --- | --- | --- | --- |
| Idle RSS (1 index, 1M docs)        | **30-50 MB** | 600 MB     | 1.2 GB     | 250 MB     | n/a (hosted)   |
| Query p50                          | 5-20 ms      | **2-5 ms** | **1-3 ms** | 8-15 ms    | 30-100 ms (net)|
| Query p99                          | 10-35 ms     | 5-10 ms    | 3-8 ms     | 20-40 ms   | 80-200 ms      |
| Build time (1M docs)               | 50 s         | 30 s       | 60 s       | 120 s      | n/a            |
| Disk usage (final, 1M @ 200 trees) | **50 MB**    | 600 MB     | 1.2 GB     | 250 MB     | n/a            |
| Multi-index in one process         | **yes (100s)** | no       | no         | yes (10s)  | yes (hosted)   |
| Streaming ingest + queryable active| **yes**      | no (rebuild) | no       | yes        | yes            |
| Mutations (tombstone delete)       | **yes**      | no         | no         | yes        | yes            |
| WAL + crash recovery (tested)      | **yes (kill -9)**| no    | no         | yes        | yes (hidden)   |
| LSM auto-compaction                | **yes**      | no         | no         | yes        | yes            |
| Per-tenant API keys + scopes       | **yes**      | no         | no         | yes        | yes            |
| Client-side traversal (privacy)    | **yes (unique)** | no     | no         | no         | yes (TEE only) |

## Where we win clearly

1. **Idle memory** — 30-50 MB per process even with hundreds of segments
   loaded. FAISS-IVF holds the inverted lists fully in RAM. hnswlib
   holds the entire graph. We hold only the sparse_index (~1% of total)
   and stream data via io_uring.

2. **Disk efficiency** — VarByte delta encoding of sorted doc_ids per
   leaf, plus the small sparse_index. SIFT 1M @ 200 trees / depth 18 =
   ~50 MB. FAISS-IVF same recall = ~600 MB (12×).

3. **Multi-index management** — running 100s of indexes in one process
   with prefix search (`arxiv-*`) is native. No other open-source ANN
   does this without spawning N processes or N graphs in memory.

4. **Streaming + mutations** — LiveIndex with WAL fsync per insert,
   queryable active buffer, LSM auto-compaction (tier-K=4), tombstone
   deletes. Crash recovery tested with real power-off (see
   `RUNBOOK.md` §7).

5. **Client-side traversal (privacy)** — clients can compute leaf_ids
   locally and post them ; the server never sees the query vector.
   See `AUTH.md` for threat-model discussion.

## Where we tie

1. **Recall** — 0.97-0.99 at the right query_depth, comparable to FAISS-IVF
   PQ-less or hnswlib at high `ef`. We're not a winner on raw recall
   either way ; the deciding factor is the size×depth choice.

2. **Filter handling** — pre-filter via CRoaring is well-supported in
   Qdrant and us. FAISS doesn't natively support pre-filter (post-filter
   only). Recall stays > 0.95 across 1% – 70% density on our side
   (validated, see `BENCH.md`).

## Where we lose

1. **Single-query latency** — FAISS-IVF and hnswlib are in-memory ;
   p50 in the 1-5 ms range. Ours is 5-20 ms because we read .srt
   sparse_index + data via io_uring on every query. Cold cache penalty
   is real (~50 ms first query, ~5 ms warm).

   → If you need < 5 ms p50 on a single small index, use FAISS/hnswlib.
   → If you need multi-index, low RAM, mutations, crash safety, use us.

2. **In-memory single-shot benchmarks** — the standard ann-benchmarks
   suite measures hot-cache queries-per-second on a single dataset
   sitting fully in RAM. We're not optimizing for that case. We optimize
   for the realistic prod case where the index is too large for RAM and
   you have many of them.

3. **Mature client libraries** — Pinecone has Java/Go/JS/Python/Rust
   SDKs and managed hosting. We have Python only at the moment.

4. **Distributed sharding** — Vespa and Elastic do this natively, we
   require external sharding (see `HA.md` Pattern C).

## Cost / TCO comparison (illustrative)

For 100 M docs × 1024 dim × recall 0.97 at 100 qps :

| System          | Nodes needed (typical sizing)           | $/month (cloud rough est.) |
| --------------- | --------------------------------------- | -------------------------- |
| **mangrove**    | 1 × 32 GB RAM + 1 TB SSD                | **~150 $**                 |
| FAISS-IVF       | 1 × 256 GB RAM (holds index)            | ~800 $                     |
| hnswlib         | 1 × 384 GB RAM (graph in RAM)           | ~1100 $                    |
| Qdrant Cloud    | qdrant.cluster / 8 vCPU / 64 GB         | ~600 $                     |
| Pinecone        | p1.x2 pod-based, 1 pod                  | ~700-900 $                 |

These numbers vary wildly by ingress/storage cost and discount tier ;
treat them as an order-of-magnitude guide, not a quote. The structural
point is : we trade ~5 ms of query latency for 5-10× cheaper hardware.

## When to choose mangrove

- You have **100+ indexes** to manage in one fleet (logs, time-series,
  per-customer corpora, per-language splits). Idle RAM matters more
  than raw query latency.
- You run on **modest hardware** (small VPS, edge devices, or simply
  cost-sensitive). Vec sizes that would melt FAISS work fine here.
- You want **streaming ingest with no full rebuild** when you add new
  docs. We freeze a small segment, compact later.
- You care about **crash safety** (validated to survive `kill -9` mid-build,
  and full power-off of the host).
- You want **privacy primitives** : client-side traversal, per-tenant
  API key scopes, future TEE-friendly deployment.

## When NOT to choose mangrove

- You have **one index < 10 GB** that fits in RAM and need < 5 ms p50 :
  use FAISS or hnswlib, they're optimized for that.
- You want a **fully managed service** with SLA, multi-region replication,
  hosted ingestion : use Pinecone or Weaviate Cloud.
- You need **a polished JS/Java/Go SDK today** : we're Python-first ;
  C FFI is available but you'd wrap it yourself.
- You need **sub-millisecond latency** on a single query : we don't aim
  there. io_uring + disk reads dominate.

## Honest open issues

- Cold-cache first query is **slow** (~50 ms on disk-stored .srt at
  random offset) until pages are warm. We document the warm-up pattern
  but don't ship an auto-pre-warm.
- Python-only SDK at the moment. C / Rust / Go bindings planned but not
  released.
- Distributed sharding not built-in. You roll your own via the
  IndexRegistry layer.
- No managed cloud offering — pure open-source under Apache 2.0.

## TL;DR

> mangrove-search is the "cold-storage massive multi-index" niche of the
> ANN ecosystem. We're not faster than FAISS on one index in RAM. We're
> 5-10× cheaper than FAISS/hnswlib/Pinecone when you have hundreds of
> indexes or 100s of millions of docs you don't want fully in RAM.
