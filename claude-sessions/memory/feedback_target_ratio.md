---
name: feedback-target-ratio
description: "auto_qd_v2 target_ratio est corpus-spécifique, scale avec dim/sub_dim. SIFT 0.001, arxiv 0.05"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 0e757620-83dd-4e01-864c-56db0c985e00
---

**Règle (mesurée 2026-05-21)** : le default `target_ratio=0.001` de `auto_qd_v2` est tuné pour SIFT (dim=128). Pour les corpus dim≥384 où sub_dim/dim ratio est faible, il faut un target_ratio plus haut pour compenser les splits noisy.

**Why** : sub_dim détermine la discriminance par split.
- SIFT 128/sub16 = 12.5% dims/split → splits assez discriminants → vote concentration OK à native qd → target 0.001 = 0.1% du corpus suffit
- arxiv 768/sub16 = 2.1% dims/split → splits noisy → NN dispersés → besoin de wider scope (lower qd) pour compenser → target ~0.05 = 5% du corpus

Si on ne descend pas qd assez, recall plafonne très bas. Exemple arxiv 2M :
- target_ratio=0.001 → qd=20 native → recall 0.67
- target_ratio=0.05 → qd=15 → recall 0.95
- target_ratio=0.1 → qd=14 → recall 0.97

**How to apply** :
- Pour dim ≤ 128 (SIFT-like) : `target_ratio=0.001` (default)
- Pour dim ≥ 384 (texte embeddings, sub_dim généralement 16) : `target_ratio=0.05`
- Heuristique linéaire approximative : `target_ratio ≈ 50 × (sub_dim / dim)^2 / dim_factor`. Mieux : faire un sweep une fois au début sur le corpus, mémoriser.
- À long terme : auto-tune via auto_qd_v2 elle-même en mesurant probe_native + observed quality.

Lien : [[feedback-recall-levers]] (qd domine sur dim 768, top_n domine sur dim 128).
