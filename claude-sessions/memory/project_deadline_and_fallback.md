---
name: deadline-and-fallback
description: deadline_ns C + query_pathrank_with_fallback SDK — LIVRÉ 2026-07-15
metadata: 
  node_type: memory
  type: project
  originSessionId: 0e757620-83dd-4e01-864c-56db0c985e00
---

## Livré

Bloqueur "query fat-tail bloque le serveur" du préprint résolu.

### C-side : `deadline_ns` dans `forest_collect_topn_probes`

- Nouveau thread-local `g_query_deadline_ns` (ns absolus CLOCK_MONOTONIC).
- Checks aux boundaries **Phase 1 / Phase 2** (avant les gros io_uring reads) et dans la boucle merge (tous les 65 k pairs).
- Sur trip : return early avec top-N partiel, `g_last_query_partial=1`.
- FFI : `mg_set_query_deadline_ns(int64_t)`, `mg_now_ns()`, `mg_last_query_partial()`.
- Python : `set_query_deadline_ms(ms)`, `last_query_partial()`.

**Limitation reste** : les SQE en vol de Phase 2 ne sont pas cancellables mid-batch — si le deadline arrive PENDANT la Phase 2 (pas au boundary), on paie le reste des reads déjà submits. Overshoot mesuré ~50-150 ms (Phase 1 = 1024 sparse-index reads) ou plusieurs 100 ms si Phase 2 bien lancée. Fix propre = `io_uring_wait_cqe_timeout` + drain, à faire si les demos exigent une abort granularity plus tight.

### SDK-side : `query_pathrank_with_fallback`

- `total_deadline_ms` : budget wall total (utilisateur voit AU PIRE ce délai).
- `mlb_ladder` : liste décroissante de mlb (ex. `(400_000, 300_000, 200_000, 150_000)`).
- Budget par attempt = `total_deadline_ms // len(mlb_ladder)` (répartition équitable).
- Sur `partial=True` : passe au mlb suivant. Sur succès : retourne. Sur épuisement : retourne le meilleur partial vu.
- Retourne `(ids, votes, n, meta)` avec `meta['attempts'], meta['mlb_used'], meta['partial']`.

## Validation (SIFT 1B d=28 qd=26 tp=768, cold)

Total_deadline_ms=1800, ladder (400k, 300k, 200k, 150k) :

| query | wall | attempts | mlb used | outcome |
|-------|------|----------|----------|---------|
| q0 | 849 ms | 400k(partial 573) → 300k(275) | 300k | ✓ rescued |
| q1 | 435 ms | 400k(434 ok) | 400k | ✓ healthy |
| q2 | 855 ms | 400k(partial 568) → 300k(286) | 300k | ✓ rescued |

## Why:

Story préprint : "billion-scale sous 1 GB avec graceful degradation sur query pathologique". Sans ça, une seule fat-tail query fige un serveur multi-tenant.

## How to apply:

- Prod SDK : appeler `query_pathrank_with_fallback` par défaut avec `total_deadline_ms = SLA_p99_max`.
- Auto-tune : run le grid sweep avec `budget_gb` paramétrable ([[preprint-roadmap]] Semaine 1).
- Sweep 2 GB SIFT 1B : débloque 0.977 recall (tp=1024 mlb=500k / 677 ms / 1395 MB) vs 0.963 sous 1 GB.

## Bench data — SIFT 1B d=28 qd=26 pareto

**Budget 1 GB** (cold, cgroup strict) :
- tp=1024 mlb=200k → 0.953 / 403 ms / 490 MB (sweet spot)
- tp=1024 mlb=300k → 0.963 / 508 ms / 761 MB (max recall sous 1 GB)

**Budget 2 GB** (cold, cgroup strict) :
- tp=1024 mlb=200k → 0.953 / 403 ms / 490 MB
- tp=1024 mlb=300k → 0.963 / 508 ms / 761 MB
- tp=1024 mlb=400k → 0.967 / 598 ms / 1091 MB
- tp=1024 mlb=500k → **0.977** / 677 ms / 1395 MB (max recall)

Toutes configs 0 partial sur 360 queries avec deadline 1800 ms.
