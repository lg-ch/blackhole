---
name: autotune-doubling
description: Auto-tune schedule = doubling depuis 100k + drift/recall fallback triggers
metadata: 
  node_type: memory
  type: project
  originSessionId: 0e757620-83dd-4e01-864c-56db0c985e00
---

## Décision 2026-07-15

Auto-tune (grid tp × mlb + top_n sweep + recommended_config.json) doit tourner **pendant** le build streaming (prod-ready), pas seulement post-build.

Trigger principal : **doublement de taille indexée**, à partir de 100k docs.
- 100k → 200k → 400k → 800k → 1.6M → 3.2M → ... → 1B
- Pour 1B docs : **~14 checkpoints**, coût 14 × 5 min = **~70 min total** (single-core)

Triggers backup (invalident la config avant le prochain doublement) :
- **Distribution drift** : `p95(leaf_size)_current / p95_last_calib > 1.5` OU `< 0.66` — capture les changements non liés à la taille (batch de docs très différents, distribution multimodale)
- **Recall live regression** : entre 2 calibs, réévaluer la config actuelle sur les queries de calibration existantes (< 1 s), force recalib si drop > 0.02

## Why:

- **En prod, la taille finale est inconnue** — impossible de dire "calibre à 25 % du corpus". Le doubling est log(N) checkpoints, adaptatif par construction.
- Naturellement le pas s'agrandit avec le corpus (comme les couches d'un LSM), les calibs les plus chères tombent quand elles comptent le plus.
- Coût 1-3 % du wall build. Négligeable.

## How to apply:

Implémentation à faire (après parallel query multi-core) :
- Dans build_tree.c : callback `after_insert(n_docs)` qui triggerait auto-tune quand `n_docs >= next_calib_threshold`.
- L'auto-tune écrit `recommended_config.json` que le serveur relit à chaque N secondes (ou signal SIGHUP).
- `recommended_config.json` déjà défini pour SIFT 1B — cf. [[deadline-and-fallback]].
