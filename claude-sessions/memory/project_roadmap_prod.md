---
name: project-roadmap-prod
description: "Roadmap restante avant 1B scaling et avant SDK prod : filter-aware I/O, voie A multi-index, telemetry, FFI, crash safety, delta encoding"
metadata: 
  node_type: memory
  type: project
  originSessionId: 0e757620-83dd-4e01-864c-56db0c985e00
---

# Roadmap mangrove-search avant prod (état 2026-05-19)

État actuel : arxiv 2M dim=768 end-to-end fonctionnel. ClickHouse natif + CRoaring + auto_qd_v2 + filter-aware merge + bruteforce fallback + compensation sub-corpus + multi-index avec qd auto par forest (voie B). Voir [[project-arxiv-2m-clickhouse]] et journal_2026-05-18.

## A — Bloqueurs pour scaling 1B (critique)

### A1. Filter-aware I/O
**Pourquoi** : le filter-aware merge skip les vote-events au merge mais on lit toujours TOUTES les super-leaves du disque (io_uring), peu importe le filter. À 1B, qd=12, 256 leaves × 1000 trees = 256k leaves lues même sur filter 1% → plafond résiduel.
**Comment** : index secondaire par .srt qui mappe leaf_id → range de doc_ids. Au query time, intersect avec le roaring filter avant de submitter les io_uring reads. Skip les leaves dont le range ne contient aucun doc allowed.
**Impact attendu** : −90% I/O sur filter sparse à 1B scale.

### A2. Voie A pairwise multi-index
**Pourquoi** : `multitopn_auto` actuel (voie B) fait N traversées séparées + merge O(N²). À 10 shards × 100M, c'est 10× la traversée du single forest.
**Comment** : refactor `forest_collect_topn_multi` pour accepter `qd_per_forest[]`. Traversée jusqu'à `max(qd_i)`, dérivation des leaf_ids par bit-shift, cursors hétérogènes au K-way merge.
**Impact attendu** : recover l'efficience pairwise-seed à multi-shard.

### A3. Build pipeline testé à 100M+
**Pourquoi** : streaming peak RSS 36 MB sur 2M = bonne signature, mais pas testé sur ≥100M en single-shot. Possibles surprises sur la phase de tri (sort des n_trees × ~N posting lists).
**Comment** : build complet sur un corpus 100M (eg sift100M ou repli arxiv en 100M synth). Mesurer RAM, latence, intégrité.

### A4. Delta encoding des doc_ids (suggestion user 2026-05-19)
**Pourquoi** : .srt files stockent doc_ids en uint32 = 4 B/doc. À 1B avec 1000 trees × ~2 docs/leaf = ~8 TB de .srt. Delta + varint sur doc_ids triés divise par 2-3× (deltas typiques < 16 = 1 byte each).
**Comment** : encoder delta = doc_ids[i+1] - doc_ids[i] en VarByte/Group VarInt à la build, décoder inline au merge. Décodage léger (~1 ns par doc).
**Impact attendu** : −50 à −66% disque/RAM mapped. Pas de gain latence direct mais permet de cacher plus en RAM.

## B — Production readiness

### B5. Telemetry intégrée
**Manque** :
- Counters cumulatifs (queries served, errors, recall samples)
- Histogrammes latence par étape (CH lookup, traversal, merge, rerank, total)
- Export Prometheus / OpenTelemetry
- Quality metrics live (self-recall sample 100 queries périodique)
**Actuellement** : tout en stderr, pas structuré.

### B6. FFI / SDK Python
**Pourquoi** : subprocess overhead ~10 ms par query, inacceptable pour latence p99 cible <20 ms.
**Comment** : compile rpforest en `.so` avec ABI stable (forest_open, forest_collect_topn, forest_close). Wrapper ctypes Python. Garder le CLI pour debug.

### B7. Validation intégrité index
**Pourquoi** : corruption silencieuse de .srt (bit-flip disque, crash mid-write) → forest renvoie du garbage sans erreur visible.
**Comment** : xxhash en footer de chaque .srt, check au `forest_open`. Optionnel : signature digest dans meta.txt couvrant tous les .srt.

## C — Crash recovery / robustness

### C8. Atomic write des .srt
**Pourquoi** : crash mid-build laisse un .srt partiel, lecture donne du garbage.
**Comment** : write to `tree_NNNNN.srt.tmp` puis `rename()` (atomic sur même FS). Standard pattern.

### C9. Build resume partial
**Pourquoi** : 1B builds prendront 8h+. Crash à 90% perd tout.
**Comment** : checkpoint `progress.json` après chaque tree finalisé. Au start, skip les trees déjà présents.

### C10. Graceful degradation
**Pourquoi** : si forest fail à l'open OU CH down pendant query → orchestrator doit fallback (bruteforce direct) ou refuse-clean (erreur structurée).
**Comment** : try/except autour des stages, decision matrix.

## D — DX / observabilité

### D11. README + setup guide
**Pourquoi** : onboarding nouveau dev = ~0 doc actuellement.
**Comment** : commandes d'install, structure repo, exemples de bench.

### D12. Bench reproductible
**Pourquoi** : tests CI / regression. Actuellement, scripts ad-hoc.
**Comment** : fix seeds, fix expected metrics (recall ±0.01), assert dans les scripts.

### D13. Embedder live (gte-base)
**Pourquoi** : tests sémantiques actuels utilisent vecs du corpus (self-match). Queries texte natives manquent.
**Comment** : load Alibaba-NLP gte-base-en-v1.5 dans le orchestrator Python, expose `search_text(text) → ids`.

### D14. Concurrence test
**Pourquoi** : `forest_collect_topn` thread-safe ? io_uring state per-forest peut conflicter si plusieurs threads.
**Comment** : bench 10 queries parallèles via OpenMP/threads, vérifier intégrité résultats et latence p99.

## Priorisation suggérée

**Avant 1B** : A1 (filter-aware I/O) → A4 (delta encoding) → A2 (voie A multi) → A3 (test 100M).
**Avant SDK prod** : B6 (FFI) → B5 (telemetry) → C8 (atomic write) → C10 (graceful degradation) → B7 (integrity).
**Nice-to-have** : reste.

## Bugs notés à corriger

- **`meta.txt` stocke `n_docs = doc_offset + n_vecs`** (au lieu de `n_vecs` réel). Bug visible quand forest buildé avec `--doc_offset != 0` : auto_qd_v2 sur-estime le pool. Fix : ajouter `doc_offset` au meta + utiliser `n_vecs_effective`. Voir journal_2026-05-18 J9.
- **Log build `gen_v0` hardcoded** dans `build_tree.c:105`. Cosmétique. Voir [[project-hd-ann-sift]].
