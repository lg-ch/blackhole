---
name: project_kube_helm_deploy
description: "Kube/Helm deployment validated on kind — SIFT 1M e2e passed, chart ClickHouse added + selector cross-match bug fixed"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2ab56d61-0a76-4916-b15c-ae347ed0fd0a
---

Session 2026-05-30 : déploiement kube validé (kind + Helm) avec suite e2e SIFT 1M.

**Stack** : `helm install mg deploy/helm/` → 2 StatefulSets (mangrove + ClickHouse), PVCs Bound. ClickHouse ajouté au chart (`deploy/helm/templates/clickhouse.yaml`, gated `clickhouse.enabled`, init SQL `docs_metadata`). Architecture : CH est dépendance *côté client* (le SDK `ClickHouseSink` construit le bitmap, le serveur ne parle jamais à CH).

**Résultats e2e** (kind arm64, scripts `deploy/tests/e2e_*_sift1m.py`) :
- Indexation 1M via SDK `insert_batch` : 4 segments LSM, ~500 vec/s, doc_ids contigus 0..N-1, wall ~35 min (insert + 4× freeze 250k×1000 arbres).
- Search : **recall@10 = 0.999**, p50 ~195 ms (élevé = 4 segments non-compactés × 1000 arbres ; une compaction fait tomber la latence).
- Filtering : pre + post = **0 violation de prédicat**, recall filtré **1.000** vs brute-force, densité 0.10.
- Persistance : index 1M survit `delete pod` (auto-discover au boot) ; CH survit restart propre.

**BUG du chart corrigé (important)** : le `Service` mangrove sélectionne par `selectorLabels` = {name, instance}. Mes pods ClickHouse portaient ces MÊMES labels + `component`, donc le selector mangrove (sans component) matchait AUSSI le pod CH → le Service mangrove avait 2 endpoints et round-robin le trafic search vers ClickHouse (50%). Masqué pendant l'ingestion par le keep-alive urllib3 du SDK (connexion épinglée au pod mangrove) ; révélé après restart. Fix : identité de selector indépendante pour CH (`name: <base>-clickhouse`). Leçon : un Service selector qui est un SOUS-ENSEMBLE des labels d'un autre pod le capture.

**Piège ops** : `delete statefulset + helm upgrade` = arrêt non-propre → peut perdre les parts CH non-fsync (1M lignes perdues ainsi en session). Utiliser `kubectl delete pod` / `rollout restart`, jamais delete STS, pour un restart.

Doc reproductible : `deploy/QUICKSTART.md`. Voir [[project_arxiv_2m_clickhouse]] (stack live CH+bitmap), [[reference_mangrove_project]].
