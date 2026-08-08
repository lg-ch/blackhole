# mangrove-search — Operations Runbook

This document is for operators running mangrove-search in production.
For architecture, see `ARCHITECTURE.md`. For benchmarks, see `BENCH.md`.

---

## 1. Component map

```
┌─────────────────────┐
│ serve_live.py       │ HTTP service (1 process)
│  - LiveIndex        │   port 8000 by default
│  - WAL durability   │
│  - LSM compaction   │
└──────────┬──────────┘
           │ reads/writes
           ▼
┌─────────────────────┐
│ <root>/             │ Index directory on local SSD
│   manifest.json     │   - segments list (immutable history)
│   wal.bin           │   - append-only WAL
│   vecs.fvecs        │   - managed vec store (auto-store mode)
│   tombstones.roaring│   - global deletes
│   seg*/             │   - frozen segments (one dir each)
└─────────────────────┘
```

A single process owns one index dir. Don't run two processes against
the same `<root>` — file locking would conflict.

---

## 2. Day-1 operations

### Start the service

```bash
python3 scripts/serve_live.py \
    --root /mnt/mangrove/indexes/main \
    --port 8000 \
    --dim 768 --sub_dim 16 \
    --n_trees 1000 --depth 18 --gen 3 \
    --create     # only on first start
```

For Docker:

```bash
docker run -d --name mangrove \
    -p 8000:8000 \
    -v /mnt/mangrove:/data \
    mangrove-search:latest \
    serve_live.py --root /data/indexes/main --port 8000 --create
```

### Verify health

```bash
curl -s http://localhost:8000/health | jq
# {"status": "ok", "next_doc_id": 0, "active_size": 0, "segments": 0}
```

### Ingest a vector

```bash
curl -s -X POST http://localhost:8000/insert \
    -H 'Content-Type: application/json' \
    -d '{"vec": [0.1, 0.2, ...]}'
# {"doc_id": 0}
```

### Search

```bash
curl -s -X POST http://localhost:8000/search \
    -H 'Content-Type: application/json' \
    -d '{"qvec": [0.1, ...], "top_n": 4000, "top_k": 10}'
```

---

## 3. Maintenance procedures

### Backup the index (atomic)

```bash
# 1. Freeze any pending active buffer first (via HTTP if service running)
curl -s -X POST http://localhost:8000/freeze -d '{}'

# 2. Either : run the backup CLI (auto-freezes, then tar.gz)
python3 scripts/backup_restore.py backup \
    /mnt/mangrove/indexes/main \
    /backups/main-$(date +%Y%m%d).tar.gz

# OR : rsync (works while service is running but pending docs may be
# lost; use freeze + a brief read-only window for true atomicity)
rsync -av /mnt/mangrove/indexes/main/ backup-host:/backups/main/
```

A backup is consistent if `wal.bin` is empty at backup time (i.e. you
called `/freeze` first).

### Restore from backup

```bash
# 1. Stop the service
systemctl stop mangrove   # or: docker stop mangrove

# 2. Extract
python3 scripts/backup_restore.py restore \
    /backups/main-20260522.tar.gz \
    /mnt/mangrove/indexes/main \
    --force

# 3. Verify integrity
./rpforest verify /mnt/mangrove/indexes/main/seg0 1000  # repeat per segment

# 4. Restart
systemctl start mangrove
```

### Manual compaction

LSM compaction triggers automatically every K=4 segments per tier. To
force a compact (rarely needed) :

```bash
# Inspect current segments
curl -s http://localhost:8000/stats | jq '.segments'

# Drop into Python REPL with library
python3 -c "
import sys; sys.path.insert(0, 'scripts')
from live_index import LiveIndex
li = LiveIndex.open('/mnt/mangrove/indexes/main')
# Force compact all tier-0 segments into one tier-1
tier_0 = [s for s in li.manifest['segments'] if s.get('tier', 0) == 0]
if len(tier_0) >= 2:
    li._compact_tier(0, tier_0[:4])
li.close()
"
```

### Graceful shutdown

```bash
# Sends SIGTERM, LiveIndex freezes pending active, closes Forests
systemctl stop mangrove
# OR
kill -TERM <pid>
```

Active buffer is flushed to a segment, no docs lost. WAL replay on next
start covers the gap between last freeze and SIGTERM if any.

---

## 4. Troubleshooting

### Problem : queries return 503 "no_index"

LiveIndex didn't open. Check :

1. `<root>/manifest.json` exists and is readable
2. No corruption : `head <root>/manifest.json` shows valid JSON
3. Permissions : the process user can read/write `<root>`
4. `serve_live.log` for the actual exception traceback

If manifest is corrupt :

```bash
# Manifest is the only mutable file ; restore from .tmp if present
ls -la <root>/manifest.json*
# .tmp left over = atomic write failed mid-way ; check both, pick the
# valid one and rename
```

### Problem : queries return empty results but index has segments

Usually a `n_docs` sentinel mismatch. The K-way merge needs an upper
bound on doc_ids. Check `manifest['next_doc_id']` vs max
`doc_offset + n_docs` across segments :

```bash
python3 -c "
import sys, json; sys.path.insert(0, 'scripts')
m = json.load(open('<root>/manifest.json'))
print('next_doc_id =', m['next_doc_id'])
print('seg max doc =', max(s['doc_offset'] + s['n_docs'] for s in m['segments']))
"
```

If they disagree, the auto-fix happens on next LiveIndex.open() (we
derive next_doc_id from segments + active).

### Problem : insert returns 503 "backpressure"

Active buffer is full. Three options :

1. **Wait + retry** (`Retry-After: 1` header indicates 1 s)
2. **Call /freeze manually** to drain
3. **Tune `--max_active`** at create time : higher cap = more RAM tolerance

The default cap is 100k docs ≈ 70 MB RAM at dim 128. Scale linearly
with dim : at dim 768, set `max_active=20000` for ~70 MB.

### Problem : segment .srt file appears corrupted

Use the verifier :

```bash
./rpforest verify <root>/seg0 1000
# OK or "checksum mismatch on tree NNNN"
```

If a single segment is corrupt :

1. **Best** : restore from backup
2. **Lossy** : remove the segment from manifest, restart. Documents in
   that segment are gone but the rest survives.
3. **Recovery** : if the source vectors are still accessible (e.g. you
   have the `vecs.fvecs` and the segment's `doc_offset`), re-build
   the segment from the slice :
   ```bash
   ./rpforest --doc_offset=<offset> --doc_count=<count> \
       build vecs.fvecs <root>/seg_new 1000 <depth>
   ```
   Then patch `manifest.json` segments[] to point at the new dir.

### Problem : crash mid-freeze leaves a `seg*.tmp` directory

These are partial segment builds. On next open, LiveIndex doesn't pick
them up (it only iterates manifest['segments']). Safe to delete :

```bash
rm -rf <root>/seg*.tmp
```

### Problem : WAL grows unbounded

WAL is truncated on every freeze. If WAL is large (> 1 GB), freeze hasn't
fired in a while. Options :

```bash
# Force a freeze via HTTP
curl -X POST http://localhost:8000/freeze -d '{}'
```

If the service is down and WAL is too big to replay quickly, you can
inspect it :

```bash
python3 scripts/inspect_wal.py <root>/wal.bin    # TODO: helper not yet written
```

### Problem : high p99 latency suddenly

Probable causes, in order :

1. **Compaction running** : check `/stats`, look for `*.tmp` dirs
2. **Cold cache** : after restart, first queries hit cold .srt pages
3. **Heavy query corpus** : some queries hit popular leaves (n_distinct
   spike). Set `mg_max_distinct` or `mg_max_stable_rejects` via FFI to
   cap, but at recall cost
4. **Wrong query_depth** : if the corpus changed dimensions/distribution,
   re-tune `target_ratio` (see `feedback_target_ratio` memory)

---

## 5. Capacity planning

### Disk

| Component             | Per 1M docs (dim 768, depth 23) |
| --------------------- | --------------------------------: |
| vecs.fvecs            | ≈ 3 GB                            |
| segments (sum)        | ≈ 5 GB                            |
| pair files transient  | ≈ 8 GB (during phase 1 build only)|
| WAL                   | ≤ size of one freeze interval     |
| tombstones.roaring    | < 1 MB per million deleted        |

For 1B docs : provision **≥ 12 TB** if you'll do monolithic builds.
For multi-segment LSM at 100M each : peak 800 GB transient per
build + 3.3 TB final.

### Memory (RSS)

| Workload                          | Typical RSS |
| --------------------------------- | ----------: |
| Service idle                      | 30–50 MB    |
| 100 concurrent queries (1000 trees, depth 30) | 200–400 MB |
| Heavy mixed insert + search       | 50–80 MB    |
| Build-in-progress (separate proc) | < 1 GB      |

Set pod memory limit at **2× expected peak** to absorb spikes.

### Network

- **Build** : reads `vecs.fvecs` sequentially + writes pair files
  (1000 files in parallel = many small ops). Local NVMe preferred ;
  NFS-style PVC adds 2–5× overhead.
- **Query** : reads .srt pages via io_uring. Sequential prefetch,
  network-storage-friendly when RTT < 1 ms.

---

## 6. Monitoring (Prometheus)

`/metrics` exposes :

```
mg_queries_total                cumulative search requests
mg_inserts_total                cumulative inserts
mg_deletes_total                cumulative tombstones
mg_freezes_total                manual freezes triggered
mg_errors_total                 endpoint errors (5xx)
mg_backpressure_rejects_total   inserts refused (cap hit)
mg_query_latency_ms             histogram of /search ms
mg_insert_latency_ms            histogram of /insert ms
mg_segments_alive               current frozen segment count
mg_active_buffer_size           current active docs (in-mem)
```

Recommended alerts :

```yaml
- alert: MangroveHighLatency
  expr: histogram_quantile(0.99, mg_query_latency_ms) > 500
  for:  2m

- alert: MangroveBackpressure
  expr: rate(mg_backpressure_rejects_total[1m]) > 1
  for:  1m

- alert: MangroveSegmentRunaway
  expr: mg_segments_alive > 20
  for:  5m
  annotations:
    summary: LSM compaction not keeping up

- alert: MangroveActiveBufferGrowing
  expr: mg_active_buffer_size > 80000     # 80% of default cap
  for:  5m
```

---

## 7. Disaster recovery scenarios

### Scenario A : node dies, disk OK

1. Boot replacement node, mount the same disk (`/mnt/mangrove`)
2. `systemctl start mangrove` — LiveIndex.open() replays WAL, opens segments
3. Verify health, resume traffic

### Scenario B : disk lost, backup exists

1. Provision new disk
2. Restore from latest backup (§3.2)
3. Documents inserted SINCE the backup are lost — clients must replay
   them from upstream source

### Scenario C : single segment corruption

See §4.4. If unrecoverable, accept the data loss for that segment range
and continue.

### Scenario D : whole index inconsistent (rare)

Symptoms : manifest references segments that don't exist, segments
fail `verify`, queries return garbage.

```bash
# 1. Snapshot the broken state for forensics
tar czf /tmp/broken-index.tar.gz <root>

# 2. Restore from backup
python3 scripts/backup_restore.py restore <backup> <root> --force

# 3. Replay any post-backup inserts from upstream (manual ; depends on
#    your upstream system)
```

---

## 8. Useful one-liners

```bash
# Show segment layout
curl -s http://localhost:8000/stats | jq '.segments[] | {name, tier, depth, n_docs}'

# Sum docs across segments
curl -s http://localhost:8000/stats | jq '[.segments[].n_docs] | add'

# Watch active size grow
watch -n 2 'curl -s http://localhost:8000/health | jq .active_size'

# Count tombstones in seg0
python3 -c "
import sys; sys.path.insert(0, 'scripts')
from mangrove_ffi import Forest, set_gen_version
set_gen_version(3)
f = Forest('<root>/seg0', n_trees=1000, dim=768, sub_dim=16, depth=23,
           n_docs=10**9, gen_version=3)
print(f.tombstones_count())
f.close()
"

# Drain the active buffer immediately (e.g. before backup)
curl -s -X POST http://localhost:8000/freeze -d '{}'

# Tail metrics live
watch -n 5 'curl -s http://localhost:8000/metrics | grep -E "mg_(queries|inserts|backpressure|segments)" | tail -20'
```

---

## 9. Known limitations & roadmap

- **Single-process / single-node** : no horizontal sharding yet. Up to
  ~1 B docs feasible on a single beefy node ; beyond requires a router
  layer (planned).
- **AuthN/TLS** : not built into serve_live. Front with a reverse proxy
  (nginx, Caddy) for both.
- **Compaction is sequential** : only one at a time. Cascade compactions
  through tiers can block writes briefly (LSM bursts).
- **WAL replay is single-threaded** : large WALs slow restart (~5 MB/s).
  Workaround : freeze regularly (auto on /freeze and on cap hit).
