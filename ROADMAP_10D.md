/mode# Roadmap 10 jours — mangrove-search

État départ : arxiv 2M dim=768 end-to-end. Stack C + ClickHouse + CRoaring. Voir `journal_2026-05-18.md` pour l'historique et `memory/project_arxiv_2m_clickhouse.md` pour les chiffres.

## Règle de design centrale (non négociable)

**claude --resume f911cb8e-cc4c-4c87-9368-166aadbc9aa5**
**Pas de mmap dans le hot path.** io_uring + `open(O_RDONLY)` partout. Le différenciateur principal du projet = **RAM process qui ne grossit pas linéairement avec N**, et reste constante à corpus fixé. mmap pollue le `RssFile` du process et donne l'illusion d'in-memory ANN sur machines RAM-rich → on perd le différenciateur quand on déploie sur edge / VPS budget. Le page cache OS s'applique de la même manière avec io_uring + O_RDONLY (vivant côté kernel, pas dans le RSS process). Toute nouvelle feature doit suivre cette règle. `block_store.c` legacy est en mmap, à dépréccier.

**Cibles RAM process (RSS anon + peak ru_maxrss)** :

| Corpus | Cible RSS | État actuel |
|---|---:|---|
| < 100M docs | **≤ 100 MB** | 28-33 MB sur 2M dim=768 (mesuré) |
| 100M - 1B | **≤ 800 MB** | À mesurer |
| > 1B | **≤ 1.6 GB** | À mesurer |

Ces cibles couvrent scratch buffers per-query + roaring filter + io_uring state + ClickHouse client. Au-delà = on regarde où ça grossit anormalement (probably bitmap, ou trop d'iterators).

Chaque item ci-dessous est conditionné à : **préserver RAM process bornée selon la table, multi-index pairwise, vrai pre-filtering**.

---

## P0 — Validation à scale (J1-J3)

### Build & test 1B
- [ ] Setup nouveau SSD (mount, format, paths). À faire à livraison.
- [ ] Télécharger SIFT 1B (`base.u8bin` ~128 GB, queries, GT). big-ann-benchmarks repo.
- [ ] **Implémenter delta encoding doc_ids** AVANT le build 1B (sinon rebuild = 50 h gâchées). VarByte sur deltas dans les .srt. Décodage inline au merge. Pas de mmap, lecture via io_uring.
- [ ] Build SIFT 1B forest (1000 trees × depth 30 × sd 16 × gen v3). ETA ~50 h en background.
- [ ] Bench SIFT 1B : recall@10 (vs GT bruteforce 100 queries), p50/p95/p99/p999 latence, peak RSS process, RAM kernel cache utilisée.
- [ ] Build 100M dim=768 (synth gaussien ou re-embed wiki) pour valider dim cible à scale intermédiaire.
- [ ] Bench 100M dim=768 idem.

### Multi-index à scale
- [ ] Construire multi-index 1B + arxiv 2M (tailles très inégales). Bench multi_auto vs single.
- [ ] **Voie A pairwise** : refactor `forest_collect_topn_multi` pour qd_per_forest avec traversée partagée + cursors hétérogènes. Bench gain vs voie B actuelle.
- [ ] Cohere Wikipedia (35M chunks dim=768, HF dataset). Multi-corpus arxiv+wiki+sift heterogène.

### Bench compétitif (le plus important pour la narrative)
- [ ] vs **FAISS-IVF + PQ** sur SIFT 1B. Compare recall@10, p99, RAM, build time.
- [ ] vs **hnswlib** sur SIFT 100M (1B probably OOM). Idem.
- [ ] vs **DiskANN** (Microsoft, clone repo + build). Idem.
- [ ] Table comparative finale : RAM, disque, recall, p99, build time. Inclure dans README.

### Stress tests
- [ ] **Centaines de filters densités variées** : générer 100 filters arxiv 0.01 % → 50 % density, mesurer recall + latence par bucket density. Identifier les zones de plafond.
- [ ] **Concurrence** : 10/50/100 queries parallèles. Vérifier thread-safety `forest_collect_topn` (io_uring ring per-forest = sérialisation ? Tester refactor ring pool si besoin).
- [ ] Stress 1M queries successives : monitorer drift mémoire, dégradation latence (cache pollution ?).
- [ ] **Tail latency** p99.9, p99.99 (jamais mesuré jusqu'ici).
- [ ] Behavior dim variés : 128 / 384 / 768 / 1024 / 1536 sur 1M docs synth. Vérifier que la stack est uniforme.

---

## P0 — Resilience & correctness (J3-J5)

### Crash safety
- [ ] **Atomic write .srt** : write `.srt.tmp` + `rename()`. Bloque toute lecture partielle.
- [ ] **Build resume partial** : checkpoint `progress.json` après chaque tree finalisé. Au restart, skip les trees déjà persistés.
- [ ] **Crash tests indexation** : kill -9 mid-build à 10 / 50 / 90 %. Redémarrer, vérifier intégrité.
- [ ] **Integrity verifier** (`rpforest verify <index>`) :
  - Vérifie n_trees fichiers .srt présents.
  - Pour chaque tree, vérifie cardinality = n_docs (chaque vec_id présent dans chaque tree).
  - **Détecte trou** : si certain vec_id manque dans certains trees (e.g. crash entre 2 builds incrémentaux).
- [ ] **Hash intégrité** : xxhash en footer de chaque .srt, check au `forest_open`. Détecte corruption silencieuse SSD (bit-flip).
- [ ] **Graceful degradation** : si forest fail open OU CH down, fallback bruteforce direct OU refuse-clean avec erreur structurée.

### Correctness tests
- [ ] **Regression suite déterministe** : fix seeds, fix datasets, fix params, assert recall ±0.005 et latence ±10 %. CI bloquant.
- [ ] **Valgrind + AddressSanitizer** : fix tout warning.
- [ ] **Test correctness vs bruteforce** : sample 1000 queries random, comparer notre top-10 vs BF top-10, assertion overlap ≥ recall cible.
- [ ] Queries hors-distribution : queries non issues du corpus (vrais textes externes encodés). Mesure recall réaliste.

---

## P0 — Performance optimizations (J5-J6)

### Storage / I/O
- [ ] **Delta encoding doc_ids** (déjà listé P0 J1-J3 car prérequis au build 1B).
- [ ] **Filter-aware I/O** : index secondaire par .srt mapping leaf_id → range doc_ids. Intersection avec roaring filter avant les io_uring reads → skip super-leaves vides du filter. Gros gain à 1B sur filter sparse.
- [ ] **io_uring tuning** : queue depth, SQPOLL kernel thread, sweet spot batch size.
- [ ] **posix_fadvise(WILLNEED)** sur le rerank L2 (top_n=2000 vecs random) pour prefetcher avant fread.

### Build optimizations
- [ ] sub_dim sweep (sd16 vs sd64 vs sd128) sur 100M dim=768. Trade-off build time / recall native / disque.
- [ ] **Parallel sort phase** : actuellement single-thread, paralléliser sur n_trees.
- [ ] PGO (profile-guided optimization) : compile avec profile bench data → +5-15 %.

### Algorithm
- [ ] **Voie A pairwise multi-index** (déjà P0 multi-index).

---

## P0 — Production wrappers (J7-J8)

### SDK / API
- [ ] **FFI Python via ctypes** : compile `.so` avec ABI stable. Exporter `forest_open`, `forest_collect_topn`, `forest_close`, `forest_get_last_n_distinct`, `roaring_from_ch_state`. Bench overhead vs subprocess (espéré 0 ms).
- [ ] **HTTP REST API** style Pinecone : `POST /search`, `POST /index`, `GET /health`, `GET /metrics`. FastAPI + FFI ou Go binding.
- [ ] **OpenAPI spec** auto-générée.
- [ ] CLI tooling : `mangrove build`, `mangrove search`, `mangrove verify`, `mangrove stats`, `mangrove health`.

### Embedder pipeline
- [ ] gte-base-en-v1.5 wired dans orchestrator (text → vec dim=768).
- [ ] Cohere embed-multilingual-v3 (optional).
- [ ] Batch embed script `texts.jsonl` → `embeddings.fvecs` streaming.
- [ ] Embedder service HTTP endpoint (vec from text).

### Telemetry
- [ ] **Prometheus exporter** : counters cumulés, histograms latence par étape (CH lookup, traversal, merge, rerank, total).
- [ ] Structured logging (JSON, fluent-bit compat).
- [ ] **Quality metric live** : self-recall périodique (sample 100 queries / heure, comparer vs BF, alert si drift).
- [ ] Healthcheck endpoint : `/health` retourne forest loaded, CH up, latency p99 sane.

---

## P0 — Ops & deployment (J9)

### Containerization
- [ ] **Dockerfile multi-stage** : build C → runtime distroless. Image ~50 MB.
- [ ] **docker-compose stack** complète : `mangrove-search`, `clickhouse`, `grafana`, `prometheus`. Up en 1 commande.
- [ ] **systemd unit** : auto-restart, dependency CH, journald logs.
- [ ] Multi-arch CI : x86_64 + arm64.

### Backup / DR
- [ ] Snapshot procedure : tarball des .srt (read-only, deduplicable). Restore testé.
- [ ] CH backup : `BACKUP TABLE mangrove.*`.
- [ ] Disaster recovery runbook documenté.

### Separation builder / query nodes
- [ ] Mode `builder` : construit forest, push artifacts (S3 ou local).
- [ ] Mode `query` : read-only forest, sync via rsync ou S3 pull.
- [ ] Hot reload : nouveau forest version → atomic swap pointer → close old. Sans interruption queries.

---

## P0 — Documentation (J10)

- [ ] **README hero** : ce que c'est, pourquoi (vs DiskANN/Qdrant/FAISS), quick demo gif.
- [ ] **Quick start** : 5 commandes pour avoir RAG sur 1M docs en <30 min.
- [ ] **Architecture deep dive** : RP forest + sorted store + CRoaring + filter-aware merge. Schémas.
- [ ] **API reference** : endpoints HTTP + flags CLI + Python SDK.
- [ ] **Operations runbook** : backup, restart, scaling, troubleshooting.
- [ ] **Comparison page** : table chiffrée vs DiskANN/FAISS/Qdrant/Pinecone (chiffres du bench compétitif).
- [ ] **Cost calculator** : N docs × dim → RAM/disk/cost estimate.

---

## P1 — Index management & UI

- [ ] `mangrove stats <index>` : taille, n_docs, age, last query, sample latency/recall.
- [ ] `mangrove health` : status global stack (forest OK, CH up, disk free, RAM stable).
- [ ] **Admin UI minimal** : HTML statique + JSON API. List/create/delete indexes, query test interactif.
- [ ] **Grafana dashboard** preset (metrics Prometheus).
- [ ] ClickHouse UI : embed Tabix ou plugin Grafana-CH dans le docker-compose.

---

## P1 — Mutations (insert / update / delete)

Indispensable prod (GDPR delete, docs qui changent, ingestion continue). Design qui exploite multi-index existant — pas de rebuild full requis.

### Soft delete (tombstones)
- [ ] **Roaring bitmap `deleted_ids`** persisté par index (`deleted.roaring`), chargé en RAM au `forest_open`. Coût RAM borné (~120 MB worst-case à 1B, typiquement <1 MB).
- [ ] Intégré au `cursor_seek_allowed` : `allowed = user_filter AND NOT deleted`. Coût query = 1 roaring AND-NOT (negligeable).
- [ ] API `DELETE /docs/<id>` : add au bitmap, atomic write tmp+rename, fsync, ack.
- [ ] Coté CH : `ALTER TABLE filters DELETE WHERE doc_id = ?` ou marquage `is_deleted` colonne (selon perf mesurée).

### Update metadata (filter change, pas le vecteur)
- [ ] Vecteur reste, doc_id reste, forest non touché.
- [ ] Mise à jour CH uniquement (ReplacingMergeTree + version, ou DELETE+INSERT sur les bitmaps).
- [ ] Doit re-pré-agréger les `groupBitmapState` après update. Tester latence d'un rebuild de l'agrégat sur 1B.

### Update vecteur (ré-embedding)
- [ ] = soft-delete old `doc_id` + insert new `doc_id` avec nouveau vecteur dans active shard.
- [ ] Mapping `external_id → current_internal_id` côté CH (table `id_map`), pour préserver l'identité logique.

### Insert incrémental (LSM-style multi-shard)
Exploite notre multi-index : pas une nouvelle archi, juste un usage discipliné.
- [ ] **Active shard** writable : forest petit (<100k docs), rebuild rapide (<30 s). Garde un buffer RAM des nouveaux vecteurs en attente.
- [ ] **Frozen shards** immutables : .srt read-only, hot reload sans interruption.
- [ ] Query : `multitopn_auto` traverse active + frozens, merge.
- [ ] Build active déclenché à : `n_pending >= 10k` OU `time >= 60s` (selon flag).
- [ ] **WAL** : append-only log `wal.bin` (doc_id, vec, op_type) fsync avant ack. Replay au restart pour reconstruire le pending buffer.

### Compaction (merge active → frozen)
- [ ] Trigger : quand `active.n_docs > seuil` OU `tombstones / total > 15 %` sur un frozen.
- [ ] **Strategy** :
  - Tier 0 : active (10-100k docs, rebuild 30s).
  - Tier 1 : merged shards (1-10M docs, rebuild ~1 h).
  - Tier 2 : main frozen (>10M, rebuild rare, plusieurs h).
- [ ] Compaction = rebuild un new shard fusionnant `active + frozens[tier]` en filtrant les tombstones → swap atomique → delete old.
- [ ] Pendant la compaction : queries continuent sur l'ancien set, RAM peak = 2× durant la transition.
- [ ] **Crash mid-compaction** : nouveau shard pas activé tant que swap pas atomique. Cleanup `*.compact.tmp` au restart.

### Mutation API & garanties
- [ ] `POST /docs` (insert batch), `DELETE /docs/<id>`, `PUT /docs/<id>` (update metadata ou vecteur).
- [ ] Ack après WAL fsync + bitmap update (durabilité avant retour).
- [ ] **Read-your-writes** : pending buffer + active shard visibles immédiatement après ack.
- [ ] CLI `mangrove compact <index>` (force compaction manuelle), `mangrove tombstones <index>` (stats).

### Validation
- [ ] Test : insert 1M docs incrémentaux par batches de 10k sur un index 100M existant. Vérifier recall stable, RAM bornée, latence queries pendant l'ingestion.
- [ ] Test : delete 10 % aléatoire d'un 100M, mesurer recall avant/après tombstones puis après compaction.
- [ ] Stress GDPR : 1000 DELETE successifs, vérifier durabilité après kill -9.

---

## P1 — Multi-tenancy

- [ ] Isolation tenant : `tenants/<id>/{index/, ch_db/}`. No cross-leak.
- [ ] Quota par tenant : rate limit queries/s, max storage.
- [ ] API keys (auth basique).

---

## P2 — Nice-to-have

- [ ] gRPC API (perf vs HTTP/JSON).
- [ ] SDKs Node.js, Rust, Go.
- [ ] Streaming ingest (Kafka consumer).
- [ ] Bulk import (parquet/jsonl).
- [ ] Multiple read-only replicas.
- [ ] TLS (mTLS optional).
- [ ] Audit log.
- [ ] Encryption at rest.
- [ ] Kubernetes Helm chart.
- [ ] CI/CD GitHub Actions.
- [ ] PyPI package.

---

## Bugs notés à corriger

- [ ] `meta.txt` stocke `n_docs = doc_offset + n_vecs` au lieu de `n_vecs` réel. Visible quand `--doc_offset != 0`. Fix : ajouter `doc_offset` au meta + utiliser `n_vecs_effective` partout (auto_qd_v2, etc.).
- [ ] Log build affiche `gen_v0` en dur dans `build_tree.c:105` (cosmétique, gen_version réelle est ailleurs).

---

## Risques surveillés

- **Concurrence io_uring** : un seul ring per forest. Plusieurs queries parallèles = sérialisation des reads ? À tester / refactor si besoin (ring pool).
- **Petites machines RAM (4-8 GB)** : peu de cache kernel disponible → plus de cache misses io_uring → p99 dégradée vs serveurs RAM-rich. À mesurer sur cgroup pour caractériser le trade-off.
- **Tail dans le ClickHouse** : pré-agg bitmaps query peut p99 spike si CH compacte au mauvais moment.
- **Rerank L2 sur SSD froid** : top_n=2000 vecs × dim 768 × 4 = 6 MB random reads. Si pages froides = 200 ms juste pour rerank. Lever : posix_fadvise WILLNEED, ou batch via io_uring (déjà partiel dans rerank_l2_uring).

---

## Validation finale (avant claim "prod-ready")

Métrique perso : **est-ce qu'un dev qui découvre le repo peut `git clone`, `make`, charger un corpus, et obtenir du RAG en <30 min, avec une RAM process bornée selon la table des cibles ci-dessus ?**

Cibles concrètes :
- corpus 1M-100M dim=768 → RSS ≤ 100 MB
- corpus 100M-1B → RSS ≤ 800 MB
- corpus > 1B → RSS ≤ 1.6 GB

Tant que la RAM **reste bornée et ne grossit pas linéairement avec N**, le différenciateur est préservé.
