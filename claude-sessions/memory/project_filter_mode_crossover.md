---
name: project-filter-mode-crossover
description: Empirical pre vs post-filter crossover measured on SIFT 100k after adaptive-oversample fix
metadata: 
  node_type: memory
  type: project
  originSessionId: 0e757620-83dd-4e01-864c-56db0c985e00
---

Bench `scripts/bench_filter_mode.py` sur SIFT 100k (1 segment, top_k=10, 100 queries, allowed_bitmap synthétique).

**Avant adaptive oversample (oversample=10× fixe)** :

| density | pre p50 | post p50 | overlap |
|---:|---:|---:|---:|
| 0.5 % | 16 | 32 | 0.06 (post sous-fetch) |
| 2 %   | 25 | 35 | 0.22 |
| 5 %   | 32 | 34 | 0.52 |
| 20 %  | 57 | 34 | 1.00 |

**Après adaptive oversample (`max(10k, 2×ceil(top_k/density))`)** :

| density | pre p50 | post p50 | overlap |
|---:|---:|---:|---:|
| 0.5 % | 16 | 52 | 1.00 |
| 2 %   | 26 | 40 | 0.99 |
| 5 %   | 29 | 37 | 1.00 |
| 20 %  | 56 | 34 | 1.00 |

**Conclusion** : post est maintenant sound à toute densité. Le crossover s'est déplacé de ~3-5 % à ~10-15 % parce que post coûte plus cher (over-fetch correct). Doc SDK.md mise à jour : pre <10 %, post >10 %.

**Why:** valider que filter_mode='post' n'est pas qu'une option cosmétique mais retourne effectivement le top-k correct, et qualifier le crossover de latence post-fix.

**How to apply:** rule "use pre at low density" reste valable mais le seuil empirique est plus haut (~10 %), pas ~3 %. Voir [[feedback_filter_strategy]].
