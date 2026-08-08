---
name: feedback-no-mmap
description: "Règle de design : pas de mmap dans le hot path. io_uring + O_RDONLY uniquement. mmap fausse les métriques RAM."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 0e757620-83dd-4e01-864c-56db0c985e00
---

**Règle (décidée par l'user 2026-05-19) : ne PAS introduire mmap dans le hot path du forest.**

**Why** : 
- La différenciation principale de mangrove-search est la RAM process **bornée et non-linéaire en N** (vs DiskANN ~15 % de la data en RAM, vs HNSW in-memory). Cibles concrètes :
  - corpus < 100M : RSS ≤ 100 MB
  - corpus 100M-1B : RSS ≤ 800 MB
  - corpus > 1B : RSS ≤ 1.6 GB
- mmap mappe le fichier dans l'address space du process → `RssFile` apparent peut grossir énormément, surtout sur machine RAM-rich où l'OS keep tout résident.
- Sur la machine de dev (RAM-riche), mmap donne l'illusion d'in-memory ANN → on perd le différenciateur quand on déploie sur edge / VPS budget.
- Le page cache OS s'applique de la même façon avec io_uring + O_RDONLY (sans O_DIRECT) que avec mmap, donc on garde les bénéfices de cache sans la pollution du RSS.

**How to apply** :
- **Reads de leaves / posting lists** : io_uring_prep_read sur fd ouvert en `open(O_RDONLY)`. C'est le pattern actuel de `sorted_store`. Ne pas régresser.
- **Reads de raw vectors (rerank)** : pread ou io_uring sur `base.fvecs`. Pas mmap.
- **Filter bitmaps** : roaring lib en RAM (allocé explicitement, free explicit). Pas mmap d'un fichier filter.
- **`block_store.c`** existant utilise mmap (legacy `.bin` format) — à terme à dépréccier ou bien marquer "legacy, ne pas utiliser pour nouveau code".
- **Nouveaux formats** (delta encoding, secondary indexes, etc) doivent suivre la même règle : io_uring + O_RDONLY.

**Note technique** : sans O_DIRECT, les reads io_uring passent quand même par le kernel page cache. Le bénéfice cache reste. La différence vs mmap est uniquement la comptabilité : avec io_uring, le cache vit côté kernel (visible dans `/proc/meminfo` champ `Cached:`), pas dans le RSS du process.

Lien : [[project-arxiv-2m-clickhouse]] pour les mesures qui valident l'approche actuelle (28-36 MB peak RSS sur 2M dim=768).
