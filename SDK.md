# mangrove-search Python SDK reference

Flat API : every operation hangs off `Client`. For one-index ops pass
`name=`, for multi-index ops pass `pattern=`.

For a 10-min walkthrough see [QUICKSTART.md](QUICKSTART.md). For server
config see [RUNBOOK.md](RUNBOOK.md) and [AUTH.md](AUTH.md).

---

## Install

```bash
pip install mangrove-search           # PyPI distribution name
```

**Note** : the PyPI distribution is `mangrove-search` but the import
name is `mangrove`. Same split as scikit-learn / sklearn.

```python
import mangrove as mg
```

---

## `Client(url, ...)`

Construct one Client per endpoint, share it across threads. Holds a
urllib3 connection pool.

```python
client = mg.Client(
    url           = 'http://localhost:8000',
    api_key       = 'optional-secret',     # X-API-Key header
    timeout       = 10.0,                  # default read timeout (s)
    retries       = 3,                     # 502/503/504/429 retries
    pool_size     = 10,                    # concurrent connections
    metadata_sink = None,                  # OPTIONAL — see below
)
```

### `metadata_sink=` — optional, default `None`

A metadata sink is a connection to an external store (typically
ClickHouse) that holds **per-doc filter metadata** : the columns you want
to filter on with `where="..."` at query time.

**Without a sink** (default) — you can still do everything except
metadata-backed filtering :

```python
client = mg.Client('http://localhost:8000')
client.insert('docs', vec)
client.search(qvec, name='docs', top_k=10)
```

**With a sink** — `insert(metadata=...)` and `search(where='SQL...')`
become available :

```python
sink   = mg.ClickHouseSink(url='http://ch:8123', table='docs_metadata',
                           schema={'category': 'LowCardinality(String)'})
client = mg.Client('http://localhost:8000', metadata_sink=sink)

client.insert('docs', vec, metadata={'category': 'finance'})
client.search(qvec, name='docs', where="category='finance'")
```

The sink is owned by the application — mangrove never reads your
metadata for ranking, only for membership testing during filtering.
See [ClickHouseSink](#clickhousesink) for details.

---

## `client.health()`

Cluster-wide liveness check.

```python
>>> client.health()
{'status': 'ok', 'indexes': 3}
```

Always public — no API key required (matches K8s liveness probes).

---

## `client.list(pattern=None)`

List index names, optionally filtered by a glob pattern.

```python
>>> client.list()
['arxiv-2025', 'arxiv-2026', 'docs-2026', 'media-2026']

>>> client.list(pattern='arxiv-*')
['arxiv-2025', 'arxiv-2026']
```

---

## `client.exists(name)`

```python
>>> client.exists('arxiv-2026')
True
```

---

## `client.create(name, dim=..., ...)`

Create an index. **Only `name` and `dim` are required.** Server picks
sensible defaults for everything else.

```python
client.create('arxiv-2026', dim=768)
```

Override defaults only if you have a reason :

```python
client.create('big-index',
    dim         = 768,
    sub_dim     = 16,        # default 16, works dim 64-4096
    n_trees     = 1000,      # default 1000, recall sweet spot
    depth       = 14,        # default 14, LSM grows it +2 per tier
    max_active  = 100_000,   # active buffer cap → backpressure 503
    gen_version = 3,         # default 3, internal
)
```

Raises `MangroveError(409)` if `name` already exists.

---

## `client.drop(name=None, pattern=None)`

Delete one index by `name`, or every index matching a glob `pattern`.
Returns the list of names actually dropped.

```python
client.drop('arxiv-2025')              # → ['arxiv-2025']
client.drop(pattern='tmp-*')           # → ['tmp-foo', 'tmp-bar']
client.drop(pattern='arxiv-202[34]*')  # → ['arxiv-2023-q1', 'arxiv-2024-q3']
```

Pattern form is **destructive at scale** — `client.drop(pattern='*')`
deletes everything. List first with `client.list(pattern=...)` if unsure.

Raises `MangroveError(404)` if `name=` doesn't exist. Pattern form
silently returns `[]` if nothing matches.

---

## `client.stats(name=None, pattern=None)`

Exhaustive stats. Pass `name=` for one index, `pattern=` for many,
nothing for the whole cluster.

```python
>>> client.stats(name='arxiv-2026')
{
    'name': 'arxiv-2026',
    'dim': 768, 'sub_dim': 16, 'n_trees': 1000, 'gen_version': 3,
    'next_doc_id': 1_234_567,
    'active_size': 412,
    'total_docs': 1_234_979,
    'tier_counts': {0: 2, 1: 1, 2: 1},
    'segments': [
        {'name': 'seg_t2_5', 'tier': 2, 'depth': 18, 'n_docs': 800_000, ...},
        ...
    ],
    'n_segments': 4,
    'disk_bytes': 2_345_678_901,
    'wal_bytes': 12_345,
    'mode': 'primary',
}
```

```python
>>> client.stats(pattern='docs-*')['aggregate']
{'n_indexes': 3, 'total_docs': 4_512_000,
 'total_segments': 7, 'total_disk_bytes': 9_876_543_210, ...}
```

---

## `client.insert(name, vec, doc_id=None, metadata=None)`

Insert one vector. Returns the assigned `doc_id`.

```python
doc_id = client.insert('arxiv-2026', vec)
```

If `metadata=` is provided AND a `metadata_sink` is configured on the
client, the metadata is also pushed to the sink keyed by `doc_id` (with
auto-injected `ts`) :

```python
doc_id = client.insert('arxiv-2026', vec,
                       metadata={'category': 'finance', 'region': 'eu'})
```

Raises `MangroveError(503)` if the active buffer is full (backpressure).
The caller can wait, call `freeze`, then retry.

---

## `client.insert_batch(name, vecs, metadatas=None, ...)`

Bulk insert. Returns a parallel list of `doc_id`s.

```python
ids = client.insert_batch('arxiv-2026', vecs)
```

With metadata :

```python
ids = client.insert_batch('arxiv-2026', vecs,
                          metadatas=[{'category': 'finance'}, ...])
```

Auto-handles backpressure : on `503`, calls `freeze()` and retries the
batch once. Disable with `auto_freeze_on_full=False` if you want the
error to bubble up.

Timeout scales with batch size (~10 ms / vec, 30 s floor). Override
with `timeout=` if you have a slow link.

---

## `client.delete(name|pattern=..., doc_id|doc_ids|where=...)`

Tombstone documents. Tombstones are persistent and survive compaction.

**Single document** (legacy form, returns `None`) :

```python
client.delete('arxiv-2026', 42)
```

**Bulk by id list** :

```python
client.delete('arxiv-2026', doc_ids=[1, 7, 42])
# → {'arxiv-2026': 3}
```

**By metadata clause** — requires `Client(metadata_sink=...)`. The sink
is queried for all matching `doc_id`s, then each is tombstoned in the
target index(es). The sink rows are also deleted (so future `where=`
queries don't see them) :

```python
client.delete('arxiv-2026', where="lang='ja' AND ts < '2024-01-01'")
# → {'arxiv-2026': 1247}
```

**Across many indexes by pattern** — any of the above modes combines
with `pattern=` :

```python
client.delete(pattern='arxiv-*', doc_id=42)
# → {'arxiv-2023': 1, 'arxiv-2024': 1, 'arxiv-2025': 1, 'arxiv-2026': 1}

client.delete(pattern='arxiv-*', where="category='retracted'")
# → {'arxiv-2023': 12, 'arxiv-2024': 8, 'arxiv-2025': 5, 'arxiv-2026': 2}
```

Returns a `{index_name: count}` dict for every form except the legacy
single-id positional call. Per-index 404s on individual ids are
swallowed (mangrove tombstones are idempotent).

**Caveat on pattern + metadata** : the default sink schema doesn't carry
an `index_name` column, so the same `doc_id` value may exist in
multiple indexes. The current implementation tombstones the matching
`doc_id` in *every* targeted index. If you need per-index disambiguation,
add an `index_name LowCardinality(String)` column to your sink schema
and include it in your `where` clause.

---

## `client.search(qvec, name=..., ...)`

Search one index.

```python
>>> client.search(qvec, name='arxiv-2026', top_k=10)
{'ids': [doc_id, ...], 'latency_ms': 12.3, 'next_cursor': last_id}
```

Optional knobs :

- `top_k=5` — number of results to return (default 5)
- `top_n=None` — K-way merge candidate cap (default : adaptive). See
  *Advanced > Tuning search*
- `n_probes=None` — multi-probe routing (default 0 = single-probe).
  Recommended `5` for high-recall workloads. See *Advanced > Tuning search*
- `max_leaf_bytes=None` — skip oversized leaves at query time to bound
  p99 latency. Recommended `20000` for SLA-bound deployments. See
  *Advanced > Tuning search*
- `query_depth=None` — runtime tree-walk depth override (default `None`
  = use build depth). See *Advanced > Tuning search*
- `metric='l2'` — `'l2'` | `'cosine'` | `'ip'`
- `where='SQL'` — filter via metadata sink (see [ClickHouseSink](#clickhousesink))
- `filter_mode='pre'` — `'pre'` (default) | `'post'`. Controls when the
  filter is applied. Pre-filter is fast when the clause is selective
  (matches <~10 % of docs) ; post-filter is faster for non-selective
  clauses. See [Pre- vs post-filter](#pre--vs-post-filter).
- `allowed_ids=[...]` — pre-filter by doc-id list
- `allowed_bitmap=bytes` — pre-filter by raw CRoaring bitmap
- `cursor_after=N` — pagination
- `client_side=True` — privacy mode (computes leaves locally)

---

## `client.search(qvec, pattern=..., ...)`

Search across many indexes matching a glob pattern.

```python
>>> client.search(qvec, pattern='arxiv-*', top_k=10)
{
    'results': [
        {'index': 'arxiv-2026', 'doc_id': 42, 'l2': 0.123},
        {'index': 'arxiv-2025', 'doc_id': 7,  'l2': 0.234},
        ...
    ],
    'matched_indexes': ['arxiv-2025', 'arxiv-2026'],
    'latency_ms': 18.7,
}
```

Same knobs as single-index search, **except** `client_side` and
`cursor_after` aren't supported across patterns yet.

---

## `client.search(qvec, name=..., where='SQL')`

Filter results by an SQL `WHERE` clause evaluated against the configured
metadata sink. The SDK runs
`SELECT groupBitmapState(doc_id) FROM <table> WHERE <your-clause>`,
gets the bitmap bytes, and passes them to mangrove — zero rebuild.

```python
client.search(qvec, name='arxiv-2026',
              where="category='finance' AND region IN ('eu','us')")
```

Combines with `pattern=` too. Requires `Client(metadata_sink=...)`.

### Pre- vs post-filter

The default is **pre-filter** (`filter_mode='pre'`) : the doc-id bitmap
is pushed into the K-way merge so candidates outside the filter are
skipped before scoring. This is what you almost always want.

Pass `filter_mode='post'` for the opposite strategy : run the ANN search
without the filter, over-fetch candidates, then keep only those that
satisfy the clause. Slower in the common case but faster when the filter
is so non-selective that pre-filtering wastes work scanning the bitmap.

Rule of thumb :

| Filter selectivity (match rate) | Use `filter_mode=` |
| :---                            | :---               |
| < 10 % (e.g. `region='fr'`)     | `'pre'` (default)  |
| > 10 % (e.g. `lang IN (...)`)   | `'post'`           |
| unsure                          | `'pre'` — almost never wrong, occasionally slower |

```python
client.search(qvec, name='docs',
              where="lang IN ('fr','en','es')",
              filter_mode='post')
```

Pre-filter is the project's signature optimization (zero-copy bitmap
into the merge). Post-filter is a safety valve for edge cases.

The SDK estimates density automatically (via `count(*)` on the sink for
`where=`, bitmap cardinality for `allowed_bitmap=`, or list length for
`allowed_ids=`) and sizes the over-fetch so that ≈ `top_k` results
survive the filter. Post-filter therefore returns the same top-k as
pre-filter at all densities ; only the latency curve differs.

---

## `mg.ClickHouseSink`

Bridge between mangrove (vectors) and ClickHouse (filter metadata).
The SDK handles the wire format automatically.

```python
sink = mg.ClickHouseSink(
    url    = 'http://clickhouse:8123',
    table  = 'docs_metadata',         # auto-created if create_table=True
    schema = {                        # filter columns + types
        'category': 'LowCardinality(String)',
        'region':   'LowCardinality(String)',
    },
)
client = mg.Client(url, metadata_sink=sink)
```

The sink table always has `doc_id UInt32 + ts DateTime64(3)` columns
on top of the user schema. `ts` is auto-injected at insert time
(`now64()` server-side if not in `metadata`).

---

## Defaults

| Param        | Default  | Notes                                                |
| ------------ | -------- | ---------------------------------------------------- |
| `sub_dim`    | 16       | Dims sampled per tree split. Works for dim 64-4096.  |
| `n_trees`    | 1000     | Recall sweet spot.                                   |
| `depth`      | 14       | Initial tier-0 depth (build-time). LSM compaction    |
|              |          | grows it +2 per tier as docs accumulate.             |
| `max_active` | 100 000  | Active buffer cap → backpressure 503 when full.      |
| `gen_version`| 3        | Internal RNG version. Don't touch.                   |
| `top_k`      | 5        | Default search result count.                         |
| `top_n`      | adaptive | server picks 0.02 (small corpus), 0.001-0.05 (big). |
| `query_depth`| `0`      | Runtime override of tree walk depth. `0` = use build depth. See *Advanced > Tuning search*. |

---

## Authentication

When the server has `MG_API_KEYS=...` set, non-public endpoints need
`X-API-Key` :

```python
client = mg.Client('https://api.example.com', api_key='my-token')
```

`/health` and `/metrics` are always public (K8s liveness, Prometheus
scrape). See [AUTH.md](AUTH.md) for key format + the OIDC pattern.

---

# Advanced

Below are escape hatches and tuning knobs. Most users never need them.
The five blocks are organised by topic — pick the one that matches what
you're doing :

- **Tuning search** — when default recall/latency isn't right
- **Filtering deep dive** — alternatives to `where=`
- **Privacy mode** — client-side traversal
- **Pagination**
- **Async + auto-batching** — high-throughput ingest patterns
- **Operational tooling** — manual freeze, export / copy CLI, legacy API

---

## Tuning search

The two recall/latency knobs you can flip at query time without
touching the index are `top_n` and `query_depth`. They act on
different stages of the pipeline.

### `top_n=` — candidate pool size

```python
client.search(qvec, name='docs', top_n=8000)
```

`top_n` caps how many candidates the K-way merge keeps before the L2
rerank. The default is *adaptive* (server picks based on corpus size
and `dim`, see Defaults table) — typically 2 000-50 000.

- **Raising** `top_n` → more candidates considered → **higher recall**,
  but **slower** (more bytes scanned, more L2 distances computed).
  Diminishing returns past ~10× `top_k`.
- **Lowering** `top_n` → faster but recall drops sharply once the true
  neighbours fall out of the candidate pool.

Rule of thumb : start with the default. If recall is below target, try
`2× default`. If you're past 10 000 and still missing, the bottleneck
is elsewhere (try `query_depth` first, then `n_trees` at index build
time).

### `query_depth=` — runtime tree-walk depth

```python
client.search(qvec, name='docs', query_depth=18)
```

Each tree was built to a fixed `depth` (see `create()`). `query_depth`
lets you query *shallower* than the build depth without rebuilding —
each tree traversal stops at `query_depth`, and the entire subtree
under that node is treated as one leaf.

- **Lowering** `query_depth` (e.g. 25 → 18) → leaves are **larger** →
  **more candidates per tree** → **higher recall**, but slower (more
  vectors to score per tree).
- **Raising** it (up to build depth) → smaller leaves → faster but more
  fragile recall when the query lands near a partition boundary.
- `query_depth=0` or `query_depth=build_depth` → native (= build)
  depth.

This is the dominant recall lever for **high-dim corpora**
(`dim ≥ 384`) where each tree split is computed on a small random
subspace (`sub_dim` of `dim`), making individual splits weakly
discriminative. Lowering `query_depth` reaggregates evidence across
more vectors.

Empirical : on dim 768 we observed `query_depth 20 → 16` recovering
**+28 pts recall for +14 ms p50**.

On low-dim corpora (dim 128 SIFT), `top_n` matters more than
`query_depth`.

### `n_probes=` — multi-probe routing

```python
client.search(qvec, name='docs', n_probes=5)
```

Each tree is traversed to its canonical leaf, then re-traversed to
`n_probes` additional "neighbour" leaves (the next-most-likely
candidates at small-margin splits). The merge sees the union with
per-tree vote dedup.

- **Raising** `n_probes` → wider coverage per tree → **higher recall**,
  cost is proportional (each probe adds ~one leaf's worth of merge
  work). Diminishing returns past ~10.
- **`n_probes=0`** (default) reverts to classic single-probe routing.

Measured on SIFT 1B (dim 128) : moving from `n_probes=10, top_n=8000`
to `n_probes=5, top_n=16000` keeps recall 0.99 and saves ~30 % latency
— widening `top_n` compensates fewer probes. Start at `n_probes=5`
with `top_n` ≈ `2 × top_k * n_probes` and adjust.

### `max_leaf_bytes=` — tail cap (SLA knob)

```python
client.search(qvec, name='docs', max_leaf_bytes=20_000)
```

Skip leaves whose posting list exceeds `max_leaf_bytes` at query time
(measured in SRT3 bytes, ≈ doc-count × 4 for typical encodings).
Caps p99 latency by dropping degenerate dense clusters that
disproportionately blow worst-case work.

- `0` (default) : no cap, no recall sacrifice
- `50_000` : lossless on SIFT 1B, ÷3 p99 latency
- `20_000` : -0.3 pt recall, ÷5-10× p99 (production SLA point)
- `10_000` : -0.5 pt recall, flat tail (p99 ≈ p50 × 3)

Trade : a query that routes into a capped leaf loses ONE tree's vote
for the docs in it ; the other 255 trees compensate. Useful when you
need predictable latency more than perfect recall (SaaS pricing,
real-time UI).

### Tradeoffs cheat-sheet

| Symptom                          | First lever to try         |
| -------------------------------- | -------------------------- |
| Low recall, latency budget OK    | `top_n` × 2, `n_probes` ↑  |
| Low recall, dim ≥ 384            | `query_depth` − 2          |
| Latency too high, recall slack   | `top_n` ÷ 2, `n_probes` ↓  |
| Latency too high, recall tight   | `query_depth` + 2          |
| p99 too high, p50 fine           | **`max_leaf_bytes` = 20 k**|
| Both bad → rebuild with more `n_trees` or different `sub_dim` (build-time, see `create()`). |

---

## Filtering deep dive

Three ways to constrain search to a subset of doc_ids — pick by where
your filter list comes from.

### `allowed_ids=` — small explicit list

```python
client.search(qvec, name='docs', allowed_ids=[1, 5, 42, 100])
```

Cheapest path when you have a Python list already. OK up to ~1 k ids.
For larger filters, use `where=` or `allowed_bitmap=` (CRoaring is
much denser).

### `allowed_bitmap=` — raw CRoaring bytes

```python
client.search(qvec, name='docs', allowed_bitmap=raw_bytes)
```

When you've already computed a `doc_id` bitmap elsewhere (typically
from ClickHouse), pass the raw bytes directly — zero rebuild on the
mangrove side.

### Obtaining the bitmap

**Option A — let the sink do it** (one-liner) :

```python
sink   = mg.ClickHouseSink(url='http://ch:8123', table='docs_metadata')
client = mg.Client(url, metadata_sink=sink)

bitmap = sink.filter_bitmap("category='finance' AND ts > today() - 7")
client.search(qvec, name='docs', allowed_bitmap=bitmap)
```

(In most cases you'd just use `client.search(qvec, name='docs',
where="category='finance' AND ...")` and skip the explicit bitmap
step — the sink builds + passes it under the hood.)

**Option B — your own ClickHouse client** (if CH lives elsewhere in
your stack) :

```python
import clickhouse_connect
ch = clickhouse_connect.get_client(host='ch.internal')
raw = ch.raw_query(
    "SELECT groupBitmapState(doc_id) FROM docs_metadata "
    "WHERE category='finance' FORMAT RowBinary"
)
# raw includes a varuint length prefix from the RowBinary format ;
# strip it (or use ClickHouseSink.filter_bitmap which handles this).
client.search(qvec, name='docs', allowed_bitmap=raw)
```

**Option C — bitmap built in your app** (when filters don't come from
SQL — e.g. an ACL service returns a list of allowed doc_ids) :

```python
from pyroaring import BitMap
bm = BitMap(allowed_ids_list)
client.search(qvec, name='docs', allowed_bitmap=bm.serialize())
```

The SDK auto-detects the byte format : bytes already starting with `0x01`
(ClickHouse AggregateFunction envelope) pass through ; raw
portable-serialize bytes get wrapped in the envelope.

### `filter_mode='pre' | 'post'` — deep dive

`'pre'` (default) pushes the bitmap into the K-way merge so candidates
outside the filter are skipped before scoring. `'post'` runs the ANN
search without the filter, over-fetches by ~`top_k/density`, then
drops non-matching ids.

The SDK estimates density itself (sink `count()` for `where=`, bitmap
cardinality, or list length) and sizes the over-fetch so ≈ `top_k`
results survive. Post is therefore *sound* at every density — only
the latency curve differs.

Empirical crossover (SIFT 100k, `top_k=10`) : pre wins below ~10 %
selectivity, post wins above. See main `search(where=)` section above
for the rule-of-thumb table.

---

## Privacy mode : compute leaves client-side

```python
client.search(qvec, name='docs', client_side=True)
```

The SDK computes the per-tree leaf-ids locally (pure-Python
`traverse_sub`) and posts only those to the server. The server never
sees `qvec`. Single-index only. See [AUTH.md](AUTH.md) for the
threat-model analysis.

---

## Pagination via cursor

```python
page1 = client.search(qvec, name='docs', top_k=100)
page2 = client.search(qvec, name='docs', top_k=100,
                      cursor_after=page1['next_cursor'])
```

Cursor is the last `doc_id` of the previous page. Stable across pages
on a fixed snapshot — concurrent writes can shift later pages.

---

## Async + auto-batching

### `mg.AsyncClient` — parallel ingest / query

httpx-based async mirror of `Client`. Same surface, awaited methods.
`pip install httpx` required.

```python
import asyncio, mangrove as mg

async def main():
    async with mg.AsyncClient('http://localhost:8000') as c:
        await c.create('docs', dim=128)
        await asyncio.gather(*[
            c.insert_batch('docs', batch) for batch in iter_batches()
        ])

asyncio.run(main())
```

### `mg.BatchedInserter` — auto-buffer + flush

Wraps a `Client`. Buffer inserts in RAM, flush by size or by interval.

```python
with mg.BatchedInserter(client, 'docs', batch_size=2000) as bi:
    for vec, meta in stream:
        bi.insert(vec, metadata=meta)
    # auto-flush on context exit
```

Async mirror : `mg.AsyncBatchedInserter`.

---

## Operational tooling

### `client.freeze(name, timeout=600)` — manual segment build

Force-build a segment from the in-memory active buffer. Synchronous,
30 s to several minutes.

**You usually don't need this.** `insert_batch` already calls `freeze`
automatically on backpressure (503 / active buffer full), so streaming
ingest workloads never hit it. Reach for it only when you want to query
freshly-inserted vectors *right now* without waiting for the active
buffer to fill, or when migrating data and you want explicit control of
segment boundaries.

```python
seg_name = client.freeze('docs')
# → 'seg5' or None if the active buffer was empty
```

Not retried automatically (re-freezing an empty buffer returns `None`).
LSM compaction runs in the background and does NOT block this call.

### Export + copy CLI

```bash
python3 scripts/mangrove_ops.py export <cluster_root> <name> <out.fvecs>
python3 scripts/mangrove_ops.py copy   <cluster_root> <src>  <dst>
```

### `IndexHandle` — legacy chained API

Pre-v0.2 idiom :

```python
idx = client.index('docs-2026')
idx.insert(vec)
idx.search(qvec, top_k=10)
```

Functionally equivalent to the flat API. **Prefer the flat form.**

---

# Errors

## `mg.MangroveError`

Every non-2xx HTTP response (after the SDK's retry budget is spent)
surfaces as `mg.MangroveError`. Catch it once at your call site, then
branch on `e.code`.

```python
try:
    client.insert('arxiv-2026', vec)
except mg.MangroveError as e:
    print(e.code, e.body)
```

| code | meaning                                              |
| ---: | :--------------------------------------------------- |
|  400 | malformed JSON body                                  |
|  401 | missing or invalid X-API-Key                         |
|  403 | API key lacks scope or permission                    |
|  404 | index does not exist                                 |
|  409 | index name already taken (on `create`)               |
|  429 | rate-limited (Retry-After honored automatically)     |
|  500 | server-side exception (see server logs)              |
|  503 | backpressure — active buffer full ; retry or freeze  |
|  504 | server slow → SDK read-timeout                       |
|  599 | exhausted retries / network unreachable (synthetic)  |

The SDK automatically retries 429/502/503/504 up to `Client(retries=)`
times (default 3) with exponential backoff, respecting any
`Retry-After` header. By the time you see one of these codes raised,
the retry budget is exhausted — treat as a real failure.

---

## See also

- [QUICKSTART.md](QUICKSTART.md) — 10-min hands-on
- [AUTH.md](AUTH.md) — API key + OIDC patterns
- [COMPARISON.md](COMPARISON.md) — vs FAISS, hnswlib, Qdrant, Pinecone
- [RUNBOOK.md](RUNBOOK.md) — server-side ops
- [HA.md](HA.md) — multi-instance deployment
