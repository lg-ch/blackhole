---
name: en-corpus-difficulty
description: "L'anomalie cohere_en (plafond 0.97) expliquée — Wikipedia EN a des voisins GT intrinsèquement plus distants, pas de corruption"
metadata: 
  node_type: memory
  type: project
  originSessionId: 0e757620-83dd-4e01-864c-56db0c985e00
---

L'écart de recall cohere_en (plafond ~0.97-0.98) vs cohere_it/no (1.000 facile) est une propriété du corpus, pas un bug : voisins GT plus distants en anglais (sim top1 0.761 vs 0.809, sim top10 0.670 vs 0.733). Angle query-voisin plus grand → hyperplans aléatoires les séparent plus souvent → moins de votes. Vérifié par re-download frais 20M + GT neuf (seed 123) + depth 23 : même dégradation → corruption exclue.

**Why:** l'utilisateur soupçonnait corruption/coupure pendant la prep originale ; investigation complète (intégrité fvecs OK, 256/256 trees OK, GT propre, 0 near-duplicates dans les deux GT) a tout écarté sauf la difficulté intrinsèque.

**How to apply:**
- Ne pas chasser le 1.000 sur EN comme si c'était un bug — le point d'équilibre est ~0.98 @ 64k top_n.
- Leviers mesurés sur en2 20M depth 23 (warm) : top_n 32k→64k = +2 pts gratuits (0.96→0.98, p50 inchangé ~373 ms) ; qd serré (19) plafonne à 0.85 quel que soit probes ; probes↑ à qd=17 DÉGRADE (bruit dans le tri par votes — optimum de probes, pas monotone).
- La difficulté d'un corpus est prévisible a priori via sim top10 du GT — argument publiable (métrique de difficulté).
- Datasets propres dispo : cohere_no 1.5M, cohere_it 10M, cohere_en2 20M sous /mnt/mangrove/datasets/ avec GT 20 queries seed 42/123.
