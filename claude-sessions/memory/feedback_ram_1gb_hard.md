---
name: feedback-ram-1gb-hard
description: "Hard rule — RAM < 1 GB always, build inclus. Pas d'optimisations qui poussent au-dessus, même temporairement."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 0e757620-83dd-4e01-864c-56db0c985e00
---

Règle dure projet : **RAM < 1 GB tout le temps**, build INCLUS (pas seulement query).

**Why:** différenciation produit de mangrove vs. tout le reste de l'écosystème ANN.
"Tout le temps" exclut explicitement les pics temporaires — un build qui consomme
1.6 GB pendant 10 min puis relâche reste hors-règle. Le user l'a réaffirmé en
2026-06-15 après lecture d'une proposition d'optimisation R&D
([[project-tree-sub-groups-arxiv]]) qui suggérait de mémoïser les sorties RNG
des nœuds peu profonds du build pour gagner 1.5-2× — RAM build estimée 200 MB
(L=12) à 1.6 GB (L=15). Rejeté.

**How to apply:**
- Avant toute optim qui touche au build : estimer le peak RAM. Si > 1 GB → NON,
  cherche une variante streaming/incremental.
- Une mémoïsation par-arbre peut rester < 1 GB si on cap la profondeur cachée
  ET on traite les arbres en série (cache d'un seul arbre à la fois).
- Voir aussi [[feedback-no-mmap]] : même esprit, RAM-négligeable est la
  différenciation à protéger.
