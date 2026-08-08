---
name: reference_vultr_build_server
description: "Serveur Vultr cdg (Paris) pour builds 1B — accès SSH, specs, quirks de toolchain"
metadata: 
  node_type: memory
  type: reference
  originSessionId: 0e757620-83dd-4e01-864c-56db0c985e00
  modified: 2026-08-06T22:00:13.375Z
---

**Serveur build 1B** : `ssh root@199.247.15.211` (Vultr VOC, Paris cdg). 8 CPU AMD, 60 GB RAM, disque NVMe `/dev/vda2` 1.2 TB (~780 GB libres avec base 1B présente). Instance id `7f9b60a5-de31-4421-ab7d-6869ac169f0c`. Clé API Vultr dans `/root/.vultr_api_key` (accès API exige d'ajouter l'IP appelante dans Account→API→Access Control).

**DEEP 1B** : `/root/deep1b/base.fbin` (384 GB, n=1e9 dim=96, vérifié intègre). Queries held-out `/root/deep1b/queries.npy` (200×96). Download depuis `storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.1B.fbin` à ~1.4 GB/s (16 connexions), ~13 min.

**Block storage** : à Paris cdg SEUL le type `storage_opt` (HDD) est dispo — inutilisable pour servir des queries (random read ~150 IOPS vs NVMe 80k). NVMe high_perf pas à Paris. Upgrade disque = plan voc-s-1920s ($620/mo) mais ne suffit PAS au pic pair 1B (2.4 TB). → utiliser [[project_tree_batched_build]] pour tenir sur 1.2 TB.

**Quirks toolchain (gcc 15 / Ubuntu resolute)** — le repo build sur gcc 13 local mais pas sur gcc 15 serveur :
1. Déclarations implicites = ERREUR (pas warning). Corrigé : `traversal.c` manquait `<stdlib.h>`/`<string.h>` ; `ffi.c` manquait `#include "tombstones.h"`/`"build_tree.h"` ; 3 `forest_*` non déclarés → ajoutés à `query_tree.h`. (Ces fixes sont dans le repo, bénéfiques partout.)
2. CRoaring apt = 4.6.1, API iterator renommée (`roaring_init_iterator`→`roaring_iterator_init`). Solution : amalgamation CRoaring **v2.0.4** compilée en `src/roaring_amalg.o`, header dans `/usr/local/include/roaring/`, Makefile serveur `-lroaring`→objet.
3. liburing apt = 2.14 (header réf `BLOCK_URING_CMD_DISCARD` absent des UAPI). Solution : liburing **2.5** depuis sources dans `/usr/local`, lié statique `/usr/local/lib/liburing.a`, `CFLAGS += -I/usr/local/include`.

Code repo copié dans `/root/mangrove-search` (rsync src/ scripts/ Makefile). Le Makefile serveur est patché localement (roaring/uring) — ne pas re-rsync le Makefile sans re-appliquer.
