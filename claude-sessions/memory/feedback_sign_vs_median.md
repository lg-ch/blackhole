---
name: Sign-split > median-split — la RAM/structure est le critère, pas le sum_leaf
description: Pour ce projet, sign-split bat median-split parce que la cible est zéro RAM data-dépendante. Ne pas reproposer median sur la base du Pareto recall/sum_leaf.
type: feedback
originSessionId: a891e619-8d9b-4b24-979f-0397e1e2abf6
---
Sign-split est **le bon choix** pour cette archi, malgré son Pareto recall/sum_leaf moins bon que med_ml8.

**Why:** L'objectif structurel du projet (cf. mangrove pairwise-seed, CLAUDE(2).md, scaling 1B / 50 MB RAM @ 100M) est **zéro stockage data-dépendant côté arbre** : seuls les seeds + posting lists + vecteurs bruts. Sign-split = on régénère le hyperplan à partir du seed à chaque traversée, rien à persister par nœud. Median-split exige un threshold float par nœud interne → à 1024 trees × ~2N nœuds, plusieurs GB de structure RAM/disque, casse le budget.

Le Pareto recall/sum_leaf optimise le **coût de scoring** (candidats à rescorer après traversée). Le critère réel à optimiser ici est le **coût de structure** (RAM/disque pour décrire l'arbre lui-même). Ce sont deux axes différents — sign gagne sur le second par construction.

**How to apply:**
- Quand on compare des stratégies de split, toujours mentionner le coût RAM/disque de structure en plus du recall et du sum_leaf.
- Ne jamais reproposer median (med_ml8 etc.) comme "meilleur" sur ce projet sans qualifier le surcoût RAM.
- Le Pareto sign-split "63k sum_leaf @ 0.98 recall" est le baseline à battre **avec d'autres leviers** (n_trees, depth, sub_dim, leaf-selection), pas en revenant sur median.
- La mémoire `project_leaf_coverage_experiment.md` insistait sur "med_ml8 écrase sign à iso-cost" — c'était myope sur l'axe coût.
