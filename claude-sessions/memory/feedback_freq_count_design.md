---
name: Freq-count design — K-way merge sur posting lists triées
description: Le bon design freq-count pour scaling 1B est un K-way merge streaming sur posting lists (qui sont triées par construction). Pas un CMS, pas un dense array.
type: feedback
originSessionId: e91b1dca-1dcd-4c5a-8a66-7e6b9afc1390
---
Pour l'étape "freq-count puis top-pool" dans le RP-forest, ne pas partir sur CMS / Misra-Gries / dense array sans avoir réfléchi à K-way merge.

**Why:** L'user a réfléchi au problème en parallèle et écrit `COUNT_FREQ.md` (à la racine de mangrove-search) qui résume 3 méthodes A/B/C avec leurs trade-offs RAM/temps :
- A) Hashmap (~28 MB pour 500k cands, gourmand)
- B) Sort flat numpy (~2 MB pour 500k, ~3 ms, simple)
- **C) K-way merge streaming sur posting lists triées (~4 KB, ~5-10 ms)** ← le bon choix à grande échelle

Insight clé : les `vec_ids` des feuilles **sont naturellement triés** parce que les doc_ids sont assignés séquentiellement (doc 0, 1, 2…) et chaque doc est appendé à la fin de sa posting list. Donc K-way merge avec un min-heap de K cursors (un par feuille sélectionnée) donne dedup+count en streaming, RAM = O(K_leaves), exact, scalable à 1B.

**How to apply:**
- Quand l'user parle de freq-count à grande échelle (≥100M), proposer K-way merge sur posting lists triées **en premier**, avant CMS / dense / hashmap.
- Vérifier que les posting lists soient effectivement triées par doc_id à l'intérieur de chaque feuille. Dans `pairwise_test.cpp` actuel, `build_tree` fait un partition BFS qui mélange l'ordre via swap — il faut ajouter un `std::sort` sur les vec_ids de chaque leaf au build pour activer K-way merge. Coût négligeable (~20 IDs/leaf en moyenne à 10M).
- Le bench bench cms_compare.cpp que j'avais commencé à écrire est obsolète — refaire avec K-way merge à la place de CMS.
- Référence : `COUNT_FREQ.md` à la racine de `mangrove-search/`.
