---
name: leaf-docs-max-n-trap
description: leaf_docs(max_n=8192) tronque silencieusement les grosses leaves — piège critique à shallow depth
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 0e757620-83dd-4e01-864c-56db0c985e00
---

`mangrove_ffi.leaf_docs(handle, tree, leaf_id, max_n=N)` **tronque silencieusement** si la leaf contient plus de N docs. Pas de warning.

**Piège spécifique** : à shallow depth (d=12), les leaves natives peuvent contenir **jusqu'à 118 655 docs** (avg 2441, max ~50× l'avg). Avec `max_n=8192` par défaut, **16% des docs par tree sont invisibles**.

**Symptôme observé (2026-07-05)** : DEEP 10M d=12 natif, POC avec `max_n=8192` → cap 2000 recall = 0.79 (vs 0.96 sur d=22+QD=12 équivalent). Chase de 3 h à investiguer la structure alors que le bug était trivial.

**Why:** À d=22 les leaves ont 2-3 docs → tronçage jamais déclenché → j'avais oublié le cap. Le cap est un vestige du design deep-tree où les leaves sont petites.

**How to apply:**
- Toujours passer `max_n >= total_docs / min_expected_leaves` — pour un index d=12 native sur 10M : `max_n=200_000` (safe).
- Sanity check en début de POC : énumérer toutes les leaves de tree0 et vérifier `sum == total_docs` du header.
- Idéalement, patcher le default ou lever une erreur si troncature.

Voir [[project_shallow_depth_poc]] pour le contexte du build d=12.
