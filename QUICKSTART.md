# Quickstart — your first 1M-doc index in 10 minutes

## 1. Install (Debian/Ubuntu)

```bash
apt-get install -y --no-install-recommends \
    gcc make pkg-config \
    libroaring-dev liburing-dev libxxhash-dev libomp-dev \
    python3 python3-numpy

git clone https://gitlab.com/leo_chartier/mangrove-search.git
cd mangrove-search
make                       # builds ./rpforest + ./libmangrove.so
```

For other distros see `PLATFORMS.md`.

## 2. Get some sample data

The SIFT 1M corpus is a standard benchmark — 1 M vectors of dim 128.

```bash
mkdir sift && cd sift
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar xf sift.tar.gz --strip-components=1
cd ..
```

You should see `sift/sift_base.fvecs` (~520 MB), `sift_query.fvecs`,
`sift_groundtruth.ivecs`.

## 3. Build an index (CLI)

```bash
./rpforest --dim 128 --sub_dim 16 --gen v3 \
    build sift/sift_base.fvecs /tmp/sift_idx 200 18
#         ^ source     ^ output      ^trees ^depth
```

This takes ~3 min on a modern CPU. Output is in `/tmp/sift_idx/` :
`meta.txt` + 200 `tree*.srt` files (~50 MB total).

## 4. Run queries (CLI)

```bash
./rpforest query sift/sift_query.fvecs /tmp/sift_idx 200 18 1000000 10 2 100
# 100 queries × top_10 expected
```

Look for the `recall@10` and `p50/p99 latency` lines in the output.

## 5. Same thing via the HTTP service

Start the server :

```bash
python3 scripts/serve_cluster.py --root /var/mangrove/cluster --port 8000
```

Register the index :

```bash
curl -X POST http://localhost:8000/indexes \
     -H 'Content-Type: application/json' \
     -d '{"name":"sift1m","dim":128,"sub_dim":16,"n_trees":200,"depth":18}'
```

Search :

```bash
curl -X POST http://localhost:8000/indexes/sift1m/search \
     -H 'Content-Type: application/json' \
     -d '{"qvec":[...your vector...], "top_n":4000, "top_k":10}'
```

Health + management UI :

- http://localhost:8000/health
- http://localhost:8000/         ← HTMX dashboard
- http://localhost:8000/metrics  ← Prometheus

## 6. Same thing via the Python SDK

Install :
```bash
pip install mangrove-search                  # PyPI distribution name
# or from the wheel built from source :
pip install dist/mangrove_search-0.1.0-py3-none-any.whl
```

Note : you `pip install mangrove-search` but `import mangrove` (same
pattern as scikit-learn / sklearn). All exports live under `mangrove.*`.

```python
import mangrove as mg
import numpy as np

client = mg.Client('http://localhost:8000')
client.indexes.create('sift1m', dim=128, sub_dim=16, n_trees=200, depth=18)

idx = client.indexes['sift1m']
# Stream-ingest your data
for vec in your_iter_of_np_arrays():
    idx.insert(vec)
idx.freeze()

# Search
result = idx.search(query_vec, top_n=4000, top_k=10)
print(result['ids'])
```

## 7. Multiple indexes + prefix search

```python
# Many indexes, one per day
for date in ['2026-05-22', '2026-05-23', '2026-05-24']:
    client.indexes.create(f'docs-{date}', dim=128, sub_dim=16, n_trees=200, depth=18)

# Search across all docs-*
result = client.search(pattern='docs-*', qvec=query_vec, top_n=4000, top_k=10)
print(result['matched_indexes'])
print(result['results'])
```

This is our main differentiator vs FAISS/Pinecone : you can run many
indexes on one process, each with its own segments, and search across
them by pattern. RAM stays small (~30 MB idle even with 100s of indexes).

## 8. Production checklist

- **Disk** : provision SSD with at least 5 × your-vec-count GB free
  (peak phase 1 transient + final segments — see `RUNBOOK.md` for the
  exact formula).
- **Memory** : the service idle is 30-80 MB. Each concurrent query
  adds a few MB.
- **Auth** : set `MG_API_KEYS=label:secret:patterns:perms:qps` env or
  use `--auth-keys-file`. See `AUTH.md`.
- **Backup** : `python3 scripts/backup_restore.py backup <root> <out.tar.gz>`.
- **K8s** : Helm chart in `deploy/helm/`. See `RUNBOOK.md` for capacity
  planning.

## 9. Common follow-ups

- "How do I delete docs ?" → `POST /indexes/<n>/delete {doc_id}` (tombstone)
- "How do I monitor ?" → `/metrics` Prometheus + `RUNBOOK.md` alert recipes
- "Multi-region HA ?" → see `HA.md` (deploy 2+ instances + LB pattern)
- "How does it compare to FAISS / Pinecone ?" → `COMPARISON.md`
- "Can I run the query client-side for privacy ?" → yes,
  `idx.search(qvec, client_side=True)` ; see `AUTH.md` for the threat model

## 10. Common errors

| Error                                       | Fix                                                |
| ------------------------------------------- | -------------------------------------------------- |
| `mg_forest_query failed`                    | check `n_docs` sentinel > all your doc_ids         |
| `503 backpressure` on insert                | call `/freeze` manually or raise `--max_active`    |
| `403 forbidden` from SDK                    | API key scope doesn't cover this index name        |
| `incompatible manifest version`             | run `scripts/migrate_manifest.py <root>`           |
| `recall too low`                            | try `--query_depth` lower than build_depth         |
