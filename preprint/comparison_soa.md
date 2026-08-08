# Comparison with published state-of-the-art on DEEP-1B

## Source: Simhadri et al. 2022 — Results of the NeurIPS'21 Challenge on Billion-Scale Approximate Nearest Neighbor Search
Proceedings of Machine Learning Research 176:177-189 (2022)

The Big-ANN Challenge partitioned billion-scale ANN into three tracks with
distinct hardware constraints. All numbers below are recall@10 on the
Yandex DEEP-1B dataset (1B × 96-dim float vectors, doc-to-doc queries).

## Table 1 — Published DEEP-1B results at controlled QPS targets

| Track | Hardware constraint | Best algorithm | recall@10 | QPS target |
|-------|--------------------|-----------------|-----------|-----------|
| **T1** — index must fit in RAM | 32 vCPU, **64 GB RAM** (Azure F32s_v2) | puck-t1 (Baidu) | **0.7226** | 10,000 |
| T1 baseline | same | FAISS-CPU | 0.6503 | 10,000 |
| **T2** — index on SSD, RAM budget | 8 vCPU, **64 GB RAM + 1 TB SSD** (Azure L8s_v2) | DiskANN baseline | **0.9371** | 1,500 |
| **T3** — any hardware | custom (Optane, GPU, PIM) | OptaNNE GraphNN (Intel) | 0.99882 | 2,000 |
| T3 | 700 GB RAM + V100 GPU | FAISS-GPU | 0.94275 | 2,000 |
| T3 | GPU + RAM | CUANNS IVFPQ (NVIDIA) | 0.99543 | 2,000 |

**Notes :**
- T1 (in-RAM) results plateau around 0.72 recall at 10k QPS with 64 GB — quantization losses dominate. Even with unlimited time, the winning entries submitted after the deadline (puck-t1 starred entries) reached ~0.79 recall on other datasets.
- T2 (SSD-based, DiskANN-style) reaches 0.937 recall with 64 GB RAM + 1 TB SSD.
- T3 (unrestricted hardware) reaches 0.99+ recall with either 1 TB Optane
  memory (OptaNNE), or NVIDIA V100 GPU + 700 GB RAM (FAISS-GPU).

## Independent published number — SPANN (Chen et al. NeurIPS 2021)

- SPANN on DEEP-1B configured with **60 GB memory** (paper reports this
  baseline configuration comparing against DiskANN).
- Reported recall 0.9 achieved at ~1 ms latency (parallel, warm cache).
- Real-world SPANN deployments have been described with 128 GB RAM
  configurations depending on the QPS target and desired recall.

## HNSW at 1B scale

HNSW is a purely in-RAM graph structure. Reported memory overhead is
~2-3× the raw vector storage. For DEEP-1B (raw: 384 GB), a full HNSW
index requires ~750 GB–1.2 TB of RAM. Practical HNSW deployments at
1B scale typically **run out of memory** without heavy quantization
or distributed sharding.

## Our comparison point

| system | RAM | recall@10 | throughput | mode |
|--------|-----|-----------|-----------|------|
| FAISS-CPU T1 baseline | 64 GB | 0.6503 | 10k QPS | warm, 32 cores |
| puck-t1 (T1 winner) | 64 GB | 0.7226 | 10k QPS | warm, 32 cores |
| DiskANN T2 baseline | 64 GB + 1 TB SSD | 0.9371 | 1.5k QPS | warm, 8 cores |
| CUANNS IVFPQ (T3) | GPU + large RAM | 0.99543 | 2k QPS | warm, GPU |
| SPANN (paper) | ~60 GB | ~0.9 | ~1k QPS (1 ms) | warm |
| **mangrove (this work)** | **1 GB strict RSS** | **0.983** | **~1.5 QPS single core / ~30 QPS on 20 cores (est.)** | **cold, cgroup-enforced** |

## Discussion

**mangrove operates in a regime not addressed by prior published systems.**

The Big-ANN Challenge itself acknowledged this gap: T1 was designed for
in-RAM index at 64 GB (deliberately generous by 2021 standards), and no
track was defined for <10 GB RAM billion-scale search. Our result —
recall@10 = 0.983 under a hard 1 GB RSS budget with strict cold-cache
measurement — is neither directly comparable to T1 (much lower recall
ceiling because 64× more memory) nor to T2/T3 (higher throughput
targets but with 60-1000× more memory).

**We propose a new operating point** rather than competing directly with
the existing points. The use case is not "highest QPS on beefy machine"
but "billion-scale ANN available in memory-constrained environments"
(shared containers, edge servers, cheap tiers, systems where the ANN
service is a subsystem sharing RAM with the primary application).

At recall parity (≥ 0.98), no system in the published literature
delivers billion-scale search under 1 GB RAM. mangrove closes this gap.

**Latency mode disclosure.** All mangrove numbers reported are:
- Single core (`sched_setaffinity`, `OMP_NUM_THREADS=1`)
- **Cold cache** (`sync && echo 3 > /proc/sys/vm/drop_caches` before every query)
- Under `systemd-run --scope -p MemoryMax=1G -p MemorySwapMax=0`

Track T1/T2/T3 numbers cited above are from warm-cache measurements at
their target QPS on cloud hardware, per the challenge protocol.
