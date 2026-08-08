---
name: makefile-header-deps
description: "Le Makefile ignore les deps headers — toucher un .h ne déclenche PAS de rebuild des .o l'utilisant"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 0e757620-83dd-4e01-864c-56db0c985e00
---

Notre Makefile actuel n'a pas de dépendances headers (pas de -MMD/-MP). Toucher `src/vec_format.h` (ou tout autre header inline-only) NE TRIGGERE PAS un rebuild des .o qui l'incluent. Résultat : changements silencieusement ignorés, bugs runtime mystérieux.

**Why:** A coûté 30 min de debug sur DEEP 10M : ajout de VECFMT_FBIN dans vec_format.h compilé OK dans tquant.c (touché) et build_tree.c (touché), mais recall.c n'a pas vu le case FBIN, est tombé dans default → offsets fvecs au lieu de fbin → mangrove rerank slot X lisait des bytes différents → recall 0.

**How to apply:**
- Workflow safe : `touch src/<modified.h> src/*.c && make` quand on modifie un header
- Patch propre à terme : ajouter `CFLAGS += -MMD -MP` + `-include $(OBJ:.o=.d)` dans Makefile (auto-deps)
- Symptôme à reconnaître : recall très bas / scores L2 absurdes / bench qui tournait avant qui tombe à 0 → recompile FULL d'abord, debug ensuite.
- Voir aussi [[check-meta-depth]] qui a un symptôme similaire pour une autre cause.
