---
name: bulk-streaming-api
description: "Design API deux modes ingestion : bulk_import (rebuild+swap) + insert (streaming HOT). Décidé 2026-07-18."
metadata: 
  node_type: memory
  type: project
  originSessionId: 0e757620-83dd-4e01-864c-56db0c985e00
  modified: 2026-07-19T14:53:33.047Z
---

## Deux modes ingestion à exposer côté API SDK

**`bulk_import(name, **build_params)` context manager** :
- Chargement initial ou rebuild complet
- Débit ~15-40k vec/s selon dim
- Queries servies sur old MAIN pendant, atomic swap à la fin
- Cas d'usage : migration HNSW → mangrove, rebuild périodique

**`insert(vec)` / `insert_batch(vecs)`** :
- Ingest continu streaming
- 1k vec/s cap SDK par défaut (`ingest_rate_limit=1000.0`)
- Overhead query +2-6 %
- Cas d'usage : RAG, agent memory, streaming Kafka

## Décision architecturale : bulk_import refuse si HOT non-vide

Si l'index a déjà du HOT streaming en cours, `bulk_import` **refuse** avec erreur claire :

```
{"error": "hot_not_empty",
 "detail": "index has N pending HOT writes, call client.freeze() then client.compact() before bulk_import"}
```

Raisons options écartées :
- ~~Merger HOT dans new build~~ : complexe, casse l'atomicity du swap
- ~~Discard HOT silently~~ : perte de données, footgun majeur
- **Refuse + force user à drainer** : explicite, safe

**Workflow attendu** :
```python
client.freeze('index')       # bloque nouveaux inserts
client.compact('index')      # drain HOT → MAIN via bg thread
with client.bulk_import('index', ...) as bulk:   # HOT vide OK
    ...
```

## À implémenter côté serveur (~200 lignes)

- `POST /indexes/{name}/bulk/start` → shadow dir + writer, retourne bulk_id
- `POST /indexes/{name}/bulk/{id}/write` → streaming fvecs append
- `POST /indexes/{name}/bulk/{id}/commit` → invoke build + swap
- `POST /indexes/{name}/bulk/{id}/status` → progress %
- **Check HOT non-vide au /start → 409 Conflict** si oui

## À implémenter côté SDK (~100 lignes)

- `client.bulk_import(name, **params)` retourne context manager
- `bulk.write_batch(vecs, doc_ids)` multipart streaming
- `bulk.checkpoint()` pour resume-safety
- Optionnel : progress bar via `tqdm`

## Documentation README

| Volume | Fréquence | Mode | Débit |
|---|---|---|---:|
| Millions à milliards | one-shot | `bulk_import` | 15-40k vec/s |
| Milliers/sec continu | permanent | `insert_batch` | 1k vec/s |
| < 100/sec | continu | `insert` | 1k vec/s |

Règle mnémotechnique : si tu peux attendre le build (heures), utilise `bulk_import`. Sinon `insert`.
