---
name: storage-and-subdim-verdicts
description: "Verdicts mesurés 2026-07-19 : X10 vs NVMe ×7-9 end-to-end, sub_dim=16 optimal (full-dim n'apporte RIEN), projection 1B NVMe ~100-160 ms"
metadata:
  type: project
---

## A/B stockage — même index, même test (DEEP 1M d=14 sd=16, 256 trees)

| config | X10 froid | NVMe interne warm | ratio | recall |
|---|---:|---:|---:|---:|
| NP=0 tp=256 | 37.9 ms | 4.1 ms | ×9.2 | 0.92 |
| NP=3 tp=1024 | 94.9 ms | 13.6 ms | ×7.0 | 0.99 |

Le ×16 I/O brut → ×7-9 end-to-end (le CPU dilue). Recall identique.
Protocole X10 froid : purge host (lire 24 GB d'autres fichiers) + drop guest/query.
DEEP 1M : base+GT+indexes dans /home/chatelet/deep1m/ (NVMe), copie X10 dans
/mnt/mangrove/indexes/deep1m_d14_sd16.

## Projection 1B sur machine réf + NVMe

Réf 500 ms query (X10) ≈ 410 I/O + 90-130 CPU → NVMe : ~30 I/O + CPU
= **~120-160 ms query, ~100 ms avec radix OMP**. Rerank 140→~20 ms.
End-to-end 1B ≈ 130-180 ms @ 0.97. NON validé empiriquement (708 GB > nos
NVMe) — à valider sur machine avec NVMe 1 TB.

## Verdict sub_dim (DEEP 1M d=14, sd=16 vs sd=96 full-dim identity)

max_vote IDENTIQUE partout (17.9/17.6, 40.4/39.8, 54.2/53.9), recall identique,
sd=96 = +40 % latence traversal et ×2.2 build. **Full-dim n'apporte RIEN** :
un hyperplan aléatoire sur 16 dims reste un hyperplan aléatoire — le bruit de
routing n'est pas un bruit d'échantillonnage. sd=16 = optimum, pas un compromis
(dim ≤ 128 ; le régime dim=768 ratio 1/48 reste à tester un jour).
Gates `use_sub` passés à `sub <= dim` (picks identity) — ⚠️ casse la compat de
l'ancien index deep_10m_sd96_d23 (buildé sur le chemin classic full-dim).

## Hypothèses d'optim tranchées par la mesure (soirée 2026-07-19)

✗ sub_dim↑ / full-dim   ✗ samples denses .smp (sur X10)   ✗ macroblocs .mbk
✗ size-based pathrank   ✗ liveness filter
✓ span-window refactor (bug fix + ÷2 reads P1 à qd<depth) — LIVRÉ
Restants : radix OMP (multi-core), TQ1 rerank 1B, NVMe (levier n°1).
