# Setup

Step-by-step to get from a fresh clone to a queryable index.

## Prerequisites

Linux 5.10+ (for io_uring SQE_USES_RING_GROUP_QUEUE) and these packages
(Debian / Ubuntu):

```bash
sudo apt install -y \
    gcc make pkg-config \
    liburing-dev libroaring-dev libxxhash-dev libomp-dev \
    python3 python3-pip
```

Optional, for the competitive bench and orchestrator:
```bash
pip install --user --break-system-packages \
    numpy faiss-cpu hnswlib clickhouse-driver
```

ClickHouse only needed if you use the `groupBitmap` filter path. For pure
ANN with raw int32 filters or no filter, skip CH.

## Build

```bash
make                  # produces rpforest + libmangrove.so
./rpforest            # prints usage
```

## A first index (SIFT 1M, ~3 min)

Downloads not handled here — fetch `sift_base.fvecs`, `sift_query.fvecs`,
`sift_groundtruth.ivecs` into `sift/` from
http://corpus-texmex.irisa.fr/ (or any mirror).

```bash
./rpforest --dim 128 --sub_dim 16 --gen v3 \
    build sift/sift_base.fvecs /tmp/sift1m 200 20

./rpforest verify /tmp/sift1m 200
```

Bench (FFI, with L2 rerank):
```bash
python3 scripts/bench_sift.py \
    --index /tmp/sift1m \
    --queries sift/sift_query.fvecs \
    --gt sift/sift_groundtruth.ivecs \
    --base sift/sift_base.fvecs \
    --n_trees 200 --depth 20 --sub_dim 16 --n_docs 1000000 \
    --gen 3 --dim 128 --n_queries 1000 --top_k 10 --top_n 500
```

Expected: recall@10 ≈ 0.85, p99 latency ≈ 20 ms, peak RSS ≈ 40 MB.

## SIFT 100M (medium)

Source dataset is `bigann_base.bvecs` from
ftp://ftp.irisa.fr/local/texmex/corpus/sift1B/. Slice the first 100M rows:

```bash
./rpforest --dim 128 --sub_dim 16 --gen v3 --doc_count 100000000 \
    build bigann_base.bvecs /mnt/mangrove/indexes/sift100m 200 25
```

Build time ~3–5 h on 20 cores. Crash-safe — restart the same command and
phase 1 resumes from the last 100k-vec checkpoint; phase 2 skips already
hashed `.srt` files.

GT for first 100M: `idx_100M.ivecs` from
ftp://ftp.irisa.fr/local/texmex/corpus/sift1B/gnd/.

```bash
python3 scripts/validate_index.py \
    --index /mnt/mangrove/indexes/sift100m \
    --queries /path/to/bigann_query.bvecs \
    --gt /path/to/idx_100M.ivecs \
    --base /path/to/bigann_base.bvecs \
    --n_trees 200 --depth 25 --sub_dim 16 --n_docs 100000000 \
    --rss_target_mb 800 --n_queries 200
```

## Long-running server

```bash
python3 scripts/serve.py \
    --index /mnt/mangrove/indexes/sift100m \
    --n_trees 200 --dim 128 --sub_dim 16 --depth 25 \
    --n_docs 100000000 --gen 3 --port 8000

# in another shell:
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/metrics
curl -s -X POST http://127.0.0.1:8000/search \
    -H 'Content-Type: application/json' \
    -d '{"qvec": [0.0, ...], "top_n": 500}'
```

## Concurrent / load test

```bash
python3 scripts/stress_concurrency.py \
    --url http://127.0.0.1:8000 \
    --queries sift/sift_query.fvecs \
    --concurrency 10 --queries 1000
```

## Soft delete (GDPR)

```bash
./rpforest delete /mnt/mangrove/indexes/sift100m 42 1337 99999
# results in /mnt/mangrove/indexes/sift100m/tombstones.roaring being updated.
# Restart serve.py — the deleted ids never appear in any query.
```

## Docker

```bash
docker compose up -d clickhouse prometheus grafana
docker build -t mangrove-search:dev .
docker run --rm -v $PWD/indexes:/indexes \
    mangrove-search:dev verify /indexes/sift1m 200
```
