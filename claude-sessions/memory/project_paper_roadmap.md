---
name: project-paper-roadmap
description: "Roadmap publication arXiv : bench panel mangrove + HNSW + FAISS-IVF. Datasets décidés, points ouverts notés."
metadata: 
  node_type: memory
  type: project
  originSessionId: 0e757620-83dd-4e01-864c-56db0c985e00
---

**Décidé 2026-06-18** : roadmap papier mangrove = panel comparatif mangrove
vs HNSW vs FAISS-IVF sur 8-13 benchmarks, narrative "RAM-bounded ANN scalant
à 1B avec recall SOTA-comparable".

**Bench panel décidé (Phase 2 image) :**
- SIFT 100M, SIFT 1B, DEEP 1B, arxiv 2M
- Queries SIFT 1B : déjà sur la machine
- DEEP 1B : à finir build (était 38 %) OU rebuild propre

**Bench panel Phase 3 texte (RAG) — POINT OUVERT à vérifier :**

Question à résoudre : **existe-t-il un Cohere v3 ≥ 100M passages avec
embeddings + queries + GT publics ?** Hypothèses :
- (A) Cohere Wikipedia multilingual v3 sur HF — pas sûr de la taille
  cumulée (50M? 100M? 300M?). À vérifier via WebSearch.
- (B) On a déjà ~72M Cohere v3 multilingue chez nous (cohere_no 1.5M +
  cohere_it 10M + cohere_en2 20M + cohere_41m). Avec multi-index family
  (`forest_collect_topn_multi`) on peut bencher comme un seul corpus
  72M sans réencoder. Already-done.
- (C) Fallback : **MS MARCO V2 138M + BGE-M3** est devenu le standard
  open-source 2024-2025. Embeddings probablement dispo sur HF.

**Action différée** : WebSearch pour confirmer dispo HF de :
1. Cohere multilingual Wikipedia v3 — taille agrégée ?
2. MS MARCO V2 138M + BGE-M3 embeddings prêts ?
3. Tout autre corpus texte ≥ 100M avec embeddings v3-class.

**Critère décision** : si A donne ≥ 100M direct → option A. Sinon C
(MS MARCO V2 + BGE-M3). B reste fallback minimum (72M déjà chez nous).

**Métriques par bench** (à mesurer pour tous) :
recall@10, p50 warm, p95 warm, p50 cold, QPS sustained parallel, RAM RSS,
disk index+sidecar, build time, build RAM.

Voir [[project-sift1b-perf]], [[project-tq1]], [[project-arxiv-2m-clickhouse]],
[[project-nq-beir-validation]] pour le contexte numérique déjà existant.
