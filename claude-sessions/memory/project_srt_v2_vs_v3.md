---
name: srt-v2-vs-v3
description: "SIFT 10M d=20 — V2 (raw uint32) coûte 1.75× disque, 0 diff latence, RAM +10%, recall exact vs V3"
metadata: 
  node_type: memory
  type: project
  originSessionId: 0e757620-83dd-4e01-864c-56db0c985e00
---

## Mesures 2026-07-15 (SIFT 10M d=20, 64 trees, tp=512 top_n=4000)

Config test isolée, cold, cgroup 4 GB, single-thread.

| version | disk | mean lat | p50 | p95 | peak RSS | recall vs V3 |
|---------|------|----------|-----|-----|----------|--------------|
| V3 (varbyte) | 1437 MB | 74 ms | 70 | 108 | 117 MB | ref |
| V2 (raw uint32) | 2512 MB | 76 ms | 73 | 104 | 129 MB | **1.000 exact** |

## Interprétation

- **Idempotence V2 ↔ V3** : recall_vs_V3 = 1.000 exact sur 30 queries. Même index logique, seule la sérialisation diffère. Preuve que V2 encode/décode correctement.
- **Disk V2 / V3 = 1.75×** : cohérent avec estimation "varbyte gagne ~40% sur les deltas sorted". Pas 4×.
- **Latence identique** : le décode varbyte n'est PAS un hotspot mesurable ; les deux paths sont IO-dominated.
- **RAM query +10% V2** : le pool des candidats stocke raw uint32 (4 B/doc) au lieu de varbyte packed (~2.3 B/doc effectif).

## Extrapolation SIFT 1B

- V3 = 0.85 TB (mesuré)
- V2 = ~1.5 TB (extrapolation ×1.75)
- Crucial X10 = 8 TB → largement dans le budget

## Why:

Le prototype préprint privilégie la simplicité de décode et l'absence de phase 2 lourde de compression. Le coût disque est acceptable, la latence identique, la RAM presque identique.

## How to apply:

- Prototype/streaming/LSM → build direct en V2 (`--no-varbyte`)
- Prod optimisée disque long-terme → V3 après compaction
- Le code query supporte les deux via `is_v3 = (f->srt_version == 3)`. Aucun switch runtime nécessaire.

## Flag CLI

`./rpforest build ... --no-varbyte` → produit `.srt` en format V2 (magic `SRT_MAGIC_V2 = 0x53525432`)
