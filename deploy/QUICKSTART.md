# Deploy quickstart — Kubernetes (kind) + Helm

Reproduces a full, filter-capable mangrove-search stack on a local Kubernetes
cluster and runs the SIFT 1M e2e suite (indexing → search → filtering) against
it through the Python SDK.

For the docker-compose path see `deploy/docker-compose.yml`. For production
notes see `../RUNBOOK.md` and `../HA.md`.

## What the chart deploys

| Object | Kind | Notes |
|--------|------|-------|
| `mg-mangrove-search` | StatefulSet (1) + Service | `serve_cluster.py`, PVC `data` (100Gi) at `/data`, probes on `/health` |
| `mg-mangrove-search-clickhouse` | StatefulSet (1) + Service | optional (`clickhouse.enabled`), PVC `ch-data` (20Gi), init SQL `docs_metadata` |

**Architecture note.** ClickHouse is a *client-side* dependency. The mangrove
server never connects to it: the SDK's `ClickHouseSink` turns a `where=` clause
into a doc_id bitmap (`groupBitmapState`, HTTP 8123) and ships it to the search
node as `allowed_bitmap`. We deploy it in-chart so a single `helm install`
yields a self-contained, filter-capable stack; for production-scale metadata,
set `clickhouse.enabled=false` and point the SDK at a managed ClickHouse.

## Prerequisites

```bash
docker --version          # 24+
kind --version            # 0.24+   (https://kind.sigs.k8s.io/dl/)
kubectl version --client  # 1.30+
helm version --short      # 3.16+
```

## 1. Cluster + image

```bash
kind create cluster --name mangrove --wait 90s
docker build -t mangrove-search:dev .
kind load docker-image mangrove-search:dev --name mangrove
# clickhouse/clickhouse-server:25.3 is pulled by the node from Docker Hub.
```

## 2. Install

```bash
helm lint deploy/helm/
helm install mg deploy/helm/ \
  --set image.repository=mangrove-search \
  --set image.tag=dev
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=mangrove-search --timeout=180s
kubectl get pods,pvc
```

Both pods should be `Running 1/1`, both PVCs `Bound`.

## 3. Port-forward

```bash
kubectl port-forward svc/mg-mangrove-search 8000:8000 &
kubectl port-forward svc/mg-mangrove-search-clickhouse 8123:8123 &
curl -s localhost:8000/health                     # {"status":"ok","indexes":0}
curl -s 'localhost:8123/?query=SELECT 1'          # 1
```

## 4. e2e tests (SDK, SIFT 1M)

Needs `sift/sift_base.fvecs` etc. (the public SIFT1M set) and host deps
`numpy clickhouse-connect pyroaring` plus `scripts/` on `PYTHONPATH`.

```bash
# Indexing: create + stream 1M vectors via insert_batch (doc_ids contiguous 0..N-1).
MG_MAX_ACTIVE=250000 python3 deploy/tests/e2e_index_sift1m.py 1000000

# Search + recall@10 vs groundtruth.
python3 deploy/tests/e2e_search_sift1m.py 1000 10

# Metadata filtering (seeds docs_metadata, pre + post filter_mode, brute-force check).
python3 deploy/tests/e2e_filter_sift1m.py 200
```

### Measured results (kind, 1 control-plane node, arm64)

| Phase | Result |
|-------|--------|
| Indexing | 1,000,000 docs, 4 LSM segments, ~500 vec/s over the SDK HTTP path (insert + 4× freeze of 250k×1000 trees), wall ~35 min |
| Search | **recall@10 = 0.999**; server latency p50 ≈ 195 ms, p99 ≈ 375 ms |
| Filtering | `category='c3'` (density 0.10): **0 predicate violations** in both `pre` and `post`; **filtered recall 1.000** vs brute force |

> **Latency note.** p50 ≈ 195 ms (vs the ~70 ms single-forest reference) is the
> cost of a freshly-streamed, **un-compacted** index: 4 LSM segments × 1000
> trees each are traversed and merged per query. A compaction merges them and
> drops latency. Per-pod throughput is also capped (~9 QPS) by the global query
> lock in `serve_cluster` — expected; scale out with more pods/indexes.

## 5. Persistence (StatefulSet + PVC)

```bash
kubectl delete pod mg-mangrove-search-0          # restart the search pod
kubectl wait --for=condition=ready pod/mg-mangrove-search-0 --timeout=180s
# re-establish port-forward, then:
python3 -c "import sys;sys.path.insert(0,'scripts');import mangrove as mg;\
print(mg.Client('http://localhost:8000').stats(name='sift1m')['total_docs'])"   # 1000000
```

Verified: the 1M-doc index survives a pod restart (PVC `data` reattaches and
`serve_cluster` auto-discovers the orphan index at boot); ClickHouse metadata
survives a clean pod restart on PVC `ch-data`.

> A full StatefulSet *replacement* (`kubectl delete statefulset … && helm
> upgrade`) is an unclean shutdown and can lose un-fsync'd ClickHouse parts —
> use `kubectl rollout restart` / pod delete for restarts, not STS delete.

## 6. Teardown

```bash
helm uninstall mg
kubectl delete pvc -l app.kubernetes.io/name=mangrove-search   # drops persistent data
kind delete cluster --name mangrove
```

## Auth, ingress, monitoring

```bash
# Enable per-key auth (401 without X-API-Key):
helm upgrade mg deploy/helm/ --reuse-values \
  --set auth.enabled=true \
  --set auth.keys='ops:secret::admin:0'
```

`values.yaml` also wires `ingress.*` and `prometheus.serviceMonitor.*`. Pods
carry `prometheus.io/scrape` annotations exposing `/metrics`.
