# High Availability deployment patterns

mangrove-search is a single-process service by design. Replication and
failover are NOT built in to the core — we follow the unix philosophy
of doing one thing well and letting standard tooling handle distribution.
This doc shows the patterns we recommend.

## Pattern A — Read replicas + shared storage (simplest)

```
                    ┌─ mangrove primary (RW) ─┐
                    │                         │
           ┌────────┤    /data (shared)       ├────────┐
           │        │      manifest.json      │        │
           │        │      wal.bin            │        │
           │        │      vecs.fvecs         │        │
           │        │      seg*/             │        │
           │        └────────────┬────────────┘        │
           │                     │                     │
           ▼                     ▼                     ▼
   mangrove replica 1   mangrove replica 2     mangrove replica 3
        (RO)                 (RO)                   (RO)
           │                  │                     │
           └──────────────┬───┴─────────────────────┘
                          ▼
                    Load Balancer
                    (HAProxy/nginx)
                          ▼
                       clients
```

**How it works** :

- Shared storage : NFS, CephFS, AWS EFS, or block-storage with multi-attach
- Primary runs in `--mode primary` : full read+write, owns WAL fsync,
  triggers compactions
- Replicas run in `--mode replica` : poll `manifest.json` for changes,
  open new segments, close compacted-away ones. **Never write**.
- Load balancer round-robins reads to all instances (primary + replicas)
  but routes writes ONLY to the primary

**Failover** :
- If a replica dies → LB removes it from rotation. Service continues.
- If primary dies → manual or automated promotion of one replica to
  primary (k8s leader election, etcd lease, or a human op). Window of
  no-writes during promotion (~seconds).

**Configuration** (k8s example) :

```yaml
# Primary
- name: mangrove-primary
  args: ["--root", "/data", "--mode", "primary"]
  volumeMounts:
    - name: shared
      mountPath: /data

# Replicas
- name: mangrove-replica
  args: ["--root", "/data", "--mode", "replica"]
  replicas: 3
  volumeMounts:
    - name: shared
      mountPath: /data
      readOnly: true   # belt-and-suspenders, replica mode also refuses to write
```

## Pattern B — Independent instances + ingest fan-out

```
                            ┌─ instance A ─┐
                ingest      │  /data-a     │      query (RR)
                 ─►─────────►              ──────────► LB ──► clients
                 fan-out    │  full RW     │
                            └──────────────┘
                            ┌─ instance B ─┐
                            │  /data-b     │
                 ─►─────────►              ──────────► LB
                            │  full RW     │
                            └──────────────┘
```

**When to use** :
- No shared storage available (each node has local NVMe)
- Ingest pipeline can fan-out writes to all instances (idempotent insert)
- Higher write throughput needed (each instance writes independently)

**Trade-offs vs Pattern A** :
- (+) No shared-storage dependency = simpler, cheaper, lower latency
- (+) Higher write throughput (no single primary bottleneck)
- (−) Storage cost = N × full index
- (−) Ingest fan-out requires idempotent doc_ids (caller-supplied, not auto)
- (−) Eventual consistency : a query right after an insert may hit an
  instance that hasn't applied the insert yet

## Pattern C — Sharded multi-node (large scale)

```
        Client query (q)
              │
              ▼
       ┌──────────────────┐
       │ Query router     │   (knows shard map : doc_id → shard)
       └──────┬───────────┘
              │ fan-out
       ┌──────┼──────┬──────┐
       ▼      ▼      ▼      ▼
   shard-0  shard-1  shard-2  shard-3
   (0-1B)   (1-2B)   (2-3B)   (3-4B)
       │      │      │      │
       └──────┴──────┴──────┘
              │
              ▼ merge & rerank
            top_k
```

**When to use** :
- Single-node disk can't hold the corpus (~5 TB+)
- Need to scale beyond ~1 B docs

**Status** : not natively supported by mangrove-search yet. Roll your
own router : shard by `doc_id % N` or by index name (e.g. by date).
Issue a single `client.search(pattern=...)` per shard, merge results
client-side.

## Choosing a pattern

| Scenario                            | Pattern |
| ----------------------------------- | ------- |
| 1-100M docs, want HA                | A       |
| 100M-1B docs, want HA               | A       |
| 1B-5B docs, single-tenant          | A or B  |
| 100s of indexes, mostly RO         | A       |
| Need every doc visible immediately | B       |
| 5B+ docs                           | C       |

## Replica mode implementation

`LiveIndex.open(root, mode='replica')` :

1. Reads manifest like primary
2. Refuses all write ops (`insert`/`delete`/`freeze`/`compact`)
3. Spawns a background thread that polls `manifest.json` mtime every N
   seconds (configurable, default 5 s)
4. On change : diff segments list, open new ones, close removed ones
5. Active buffer is NOT replayed (only the primary has WAL writes)

Quick test :

```bash
# Terminal 1 : primary
python3 scripts/serve_cluster.py --root /shared/data --port 8000 --mode primary

# Terminal 2 : replica reading the same /shared/data
python3 scripts/serve_cluster.py --root /shared/data --port 8001 --mode replica

# Terminal 3 : insert via primary
curl -X POST :8000/indexes/test/insert -d '{"vec":[1,0,0,0]}' ...

# After ≤5 s the replica sees it :
curl :8001/indexes/test/search -d '{"qvec":[1,0,0,0]}'
```

## Failover automation

For Pattern A, automate primary election with one of :

- **K8s leader election** : annotate the StatefulSet with
  `kubernetes.io/lease-name` or use the `coordination.k8s.io/Lease` API
- **etcd lease** : each instance acquires `/mangrove/primary` lease,
  loses it → drops to replica mode
- **Consul session** : same pattern with Consul's KV+session

mangrove doesn't ship one — the choice depends on your infra. We
recommend k8s leases when running in k8s, etcd otherwise.
