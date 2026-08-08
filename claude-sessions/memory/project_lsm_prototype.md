---
name: lsm-prototype
description: "LSM segments statique SIFT 10M d=28 — recall préservé exact, latence 1.68× mono"
metadata: 
  node_type: memory
  type: project
  originSessionId: 0e757620-83dd-4e01-864c-56db0c985e00
---

## Livré 2026-07-15

Prototype LSM static (5 segments × 2M docs) validé sur SIFT 10M d=28 vs monolithique 10M.

### Design

- Chaque segment = index complet (64 trees, d=28) avec **MÊMES seeds** (`tree_offset=0` partout)
- Docs disjoints entre segments via `--doc_offset` / `--doc_id_base`
- Trees identiques → même partition hyperplane → même leaf_id pour même query
- Query = trigger même leaves dans tous les segments → chaque segment tire ses docs

### `mg_query_pathrank_multi` en C (livré)

- Signature : `(handles[], n_handles, qvec, n_probes, top_paths, top_n, qd, ...)`
- 1 seule traversal (mêmes seeds partagés) → 1 seul FFI round-trip
- Per-segment : `forest_collect_topn_probes` avec le MÊME array `leaves`
- Merge final : concat (top_n × n_handles) → qsort par vote desc → keep top_n
- Docs disjoints entre segments → pas de somme de votes (juste concat + tri)

### Mesures SIFT 10M d=28 tp=512 mlb=200k top_n=4000, cgroup 1G cold, vraie GT

| config | recall | mean lat | p95 | peak RSS |
|--------|--------|----------|-----|----------|
| **mono 10M** | **0.767** | **53 ms** | 65 | 62 MB |
| LSM Py 5×2M tp/seg | 0.763 | 122 ms | 132 | 100 MB |
| LSM Py 5×2M tp/N | **0.573** ❌ | 88 ms | 94 | 97 MB |
| **LSM C multi tp/seg** | **0.767** ✓ | **89 ms** | 101 | 100 MB |

**Findings clés** :

1. **Recall exactly preserved** : LSM C = mono (0.767 identique), équivalence algo prouvée.

2. **`tp/N` (divisé) casse recall** : perd 25 % (0.767 → 0.573). Le vote count opère par segment, chaque segment a besoin de tp complet pour signal fort. **Fair = tp/seg = tp/mono** (même budget par segment).

3. **C multi vs Py** : -27 % latence (89 vs 122 ms) grâce à traversal partagée + 1 FFI.

4. **Coût LSM final = 1.68× mono en latence** + 60 % RAM (5 forests ouverts).

### Why

Base algorithmique solide pour ajouter :
- Compaction background (level-0 → level-1)
- Live segment dynamique (in-RAM ou append-only)
- Multi-depth segments (level-0 shallow, master deep)

### How to apply

Prod LSM streaming :
```python
# Ingest : nouveau segment (small, level-0) créé every N docs
# Query : mf.query_pathrank_multi([f_master, f_seg0, f_seg1, ...], ...)
```

Le C multi peut remplacer directement query_pathrank sur des indexes de 1 segment (n_handles=1) — backward compatible.

### Suite

- Query parallel segment collect via multi-ring io_uring (potentiel -30 % lat)
- Support depth mixte (segments L0 à d=14, master à d=28)
- Live segment queryable (in-RAM ou small sealed)
