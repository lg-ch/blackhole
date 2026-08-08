---
name: project-tq1
description: "TurboQuant 1-bit sidecar (.tq1) livré. Recall identique à TQ4 avec K' × 5. Disque 4×, cold cache -17%, ouvre voie 1B."
metadata: 
  node_type: memory
  type: project
  originSessionId: 0e757620-83dd-4e01-864c-56db0c985e00
---

**Livré 2026-06-16** : nouveau sidecar `.tq1` (1 bit/coord) en parallèle de
`.tq4`. Code dans `src/tq1.{c,h}`. CLI `rpforest tquant1`, FFI
`mg_rerank_tq1`, Python `Forest.rerank_tq1`. Magic `0x31515400`.

**Format** : (HD)³ rotation seed-derived (idem TQ4), puis sign quantization
(2-level Lloyd-Max → {c_neg, c_pos}). Header 32 B + `pad_dim / 8` B par code.

**Bench arxiv 2M (dim 768, champion forest g=16 ts=128, 100 q) :**

| top_n  | K'   | recall (TQ4=TQ1) | warm p50 | cold q0 |
|--------|------|------------------|----------|---------|
| 16 000 | 100 (TQ4) | 0.992  | 48.5 ms  | 86 ms   |
| 16 000 | 500 (TQ1) | 0.992  | 47.2 ms  | **71 ms** (-17%) |

→ TQ1 K'=500 ≡ TQ4 K'=100 sur recall (mathématiquement sain : 5× plus de
survivants au stage 2 compense la précision/code réduite).

**Recall ceiling sur arxiv 2M** (sweep NP × QD × top_n) :

| NP | QD | top_n | recall | p50 warm |
|----|------|--------|--------|----------|
| 10 | 14 | 32 000 | 0.999 | 73 ms |
| 10 | 14 | **64 000** | **1.000** | **110 ms** ← sweet spot 100% |
| 20 | 12 | 16 000 | 1.000 | 119 ms |

→ **Recall 1.000 strict atteint** à NP=10 QD=14 top_n=64k / 110 ms warm.
QD est le levier dominant pour le dernier % sur dim=768 (cf [[feedback-recall-levers]]).
NP > 20 = forest saturé, plus aucun gain marginal.

**Footprint disque (arxiv 2M)** : tq4 = 1.05 GB, tq1 = 263 MB (4× moins).
Lloyd-Max converge c_neg=-0.46 c_pos=0.61.

**Process RAM** : INCHANGÉ vs TQ4 — pipeline io_uring + O_RDONLY identique
(cf [[feedback-no-mmap]]). Buffer transient ~3.5 MB par requête, libéré
après. Le sidecar TQ4 et TQ1 ne sont JAMAIS mappés.

**Implication 1B** : TQ4 sidecar = 64 GB, TQ1 = 16 GB — gain disque direct
(stockage + bandwidth I/O groupé via sorted reads). À mesurer sur DEEP 1B
quand SSD `/mnt/mangrove` remonté.

**Caveat CPU** : boucle de scoring TQ1 = pad_dim itérations bit-à-bit (1024
sur dim=768) vs pad_dim/2 (512) pour TQ4 → CPU ~2×. Compensé par I/O cold,
mais SIMD/popcount-based scoring serait un suivi naturel pour faire mordre
TQ1 en warm aussi.

Voir aussi [[project-turboquant-rerank]] (TQ4 livré antérieurement).
