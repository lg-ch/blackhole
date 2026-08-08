---
name: feedback-filter-strategy
description: "Pre-filter vs post-filter dans le RAG : sémantiquement différents, pas juste deux optims de la même chose"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 0e757620-83dd-4e01-864c-56db0c985e00
---

**Règle conceptuelle (décidée par l'user 2026-05-17) :**

- **Pre-filter** = vocation **discriminante**. Le filtre est SUPPOSÉ être sélectif. Il PILOTE la recherche. "Cherche les NN seulement dans ces docs spécifiques." Forest reçoit le bitmap, son K-way merge skip les non-allowed via bitmap_contains(O(1)).

- **Post-filter** = vocation **non-discriminante**. Le filtre est juste une contrainte/policy ("pas japonais", "publié après 2020"). Il VALIDE le résultat à la fin, ne PILOTE pas la recherche. App over-fetch (~3× top_k), récupère metadata par ES mget/Postgres point-query, filtre, truncate.

**Implication :**

Cette distinction est sémantique, pas juste de perf. Elle décide quel chemin prend la query. L'app décide via density check (e.g. `count(filter) / N_total < 0.03` → pre-filter, sinon post-filter), mais le BUT est différent.

**Implementation (état 2026-05-18) :**

- **Pre-filter implémenté en CRoaring** : signature `roaring_bitmap_t*` dans `forest_collect_topn`/`_multi`. Hot path = `roaring_bitmap_contains`. Wire format compatible direct avec ClickHouse `AggregateFunction(groupBitmap, UInt32)` (prefix `[0x01][varint][CRoaring portable]`). Décodeur dans `src/croaring_io.c`. CLI `--filter <int32_ids>` (dev) + `--filter_ch <ch_state>` (prod).
- **Post-filter** : pure logique app-side, SQL `WHERE internal_id IN (...) AND <conditions>`. Pas de code dans notre forest.

**Bench arxiv 2M dim=768 (2026-05-18, 1000 queries) :**
- Forest p99 stable ~7 ms quel que soit le filter (pre, post, none). Filter au hot path = quasi-gratuit.
- Pre-filter state : 17 B pour 42k docs (year pré-agg RLE), 27 KB pour 16k docs (combo on-the-fly). Toujours ≪ MB même à grande échelle.
- Post-filter ajoute p99 ~9 ms (1 SQL CH par query). Confirme : pre quand discriminant.
- Total RAG p99 : ~8 ms (no/pre) à ~16 ms (post) sur 2M docs.

**Filter-aware K-way merge (2026-05-18, query_tree.c) :**

Initial impl : `roaring_bitmap_contains` après pop du heap = check par doc, mais le merge itère TOUS les vote-events lus du disque indépendamment du filter. À 1B docs scale, coûteux : qd=12 ⇒ 256M vote-events itérés peu importe la cardinalité du filter.

Fix : cursor-level skip via `roaring_uint32_iterator` (1 iter persistent/tree) + binary search dans c->docs. Le merge skip aux ranges allowed → le filter check devient gratuit.

Mesures :
- filter 2% à qd=12 : 91 → 48 ms (-47%), recall identique
- filter 0.8% à qd=12 : 91 → 50 ms (-45%)
- filter 0.8% à qd=8 : 1060 → 622 ms (-41%)
- avg_distinct au merge : 898k → 18.6k (= taille effective du filter visited)
- no_filter : 0% regression

Plafond résiduel = **I/O des leaves** (io_uring lit toutes les super-leaves même si elles sont vides du filter). Pour finir le scaling 1B, faut aussi un filter-aware I/O qui skip aux leaves non-vides du filter.

**Bruteforce fallback (scripts/bruteforce_arxiv.py + search_arxiv.py) :**

Pour filter < ~20k docs à dim=768 : lire le subset (48 MB) et compute L2 direct est plus rapide que forest, avec recall 1.0 garanti. Mesures :
- filter 16k (48 MB) : 34 ms / recall 1.000
- filter 42k (124 MB) : 77 ms / recall 1.000

**Routing automatique** dans `search_arxiv.py` (depuis 2026-05-18) :
- `filter_card × dim × 4 ≤ 50 MB` → bruteforce direct (recall 1.0 garanti, contourne le plafond intrinsèque du forest sur sub-corpora)
- `density ≤ 3 %` → pre-filter forest (filter-aware merge)
- `density > 3 %` → post-filter app-side

**Justification théorique du bruteforce pour filter sparse** : le forest a été buildé sur le corpus complet — ses hyperplanes optimisent la séparation des NN globaux, pas sub-corpus arbitraires. Quand on filtre à 0.8 %, viser "40 % du pool visité" (formule auto_qd_v2) donne le bon effort mathématique MAIS pas le bon recall absolu : 60 % des NN sub-corpus restent invisibles structurellement. Bruteforce résout en explorant tout le sub-corpus exactement.

**Reco architecture pour 1B+ (mise à jour 2026-05-18) :**

1. **Phase 100M-2M (maintenant, FAIT)** : ClickHouse natif (TCP/9000) + CRoaring dans forest. Pré-agg `AggregateFunction(groupBitmap, UInt32)` par dimension low-card (year, primary_cat, top_cat). Décodage direct du state CH en roaring côté C. App-side switch pre/post selon `count(filter)`. Seuil 3% validé. Voir [[project-arxiv-2m-clickhouse]].
2. **Phase 1B-10B** : même stack avec sharding (10 × ~200M par shard). Le format wire roaring est déjà compact (KB max), pas de refactor à prévoir.
3. **Phase 10B+** : architecture distribuée standard (broadcast query, per-shard search, global merge).

**Ce qui ne change PAS dans le forest :**
- Le multi-forest est déjà conçu pour scaler en N forests.
- L'API roaring bitmap remplacera le bitmap dense (drop-in upgrade plus tard).

**Antipatterns à éviter :**
- Pre-filter avec filter dense (50%+) : transfer payant pour rien, recall pas amélioré (vrais NN déjà dans le filter naturellement).
- Post-filter avec filter ultra-sparse (< 0.1%) : sur 1B avec 100K filter à post-filter, over-fetch nécessaire est ~10000× top_k → trop d'ES lookups.

Lien : [[feedback-recall-levers]] sur le rôle de top_n. [[project-rpforest-sift]] pour le contexte global.
