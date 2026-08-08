---
name: query-deadline-gap
description: "mg_query_pathrank n'a pas de deadline_ns — bloqueur prod-ready pour préprint"
metadata: 
  node_type: memory
  type: project
  originSessionId: 0e757620-83dd-4e01-864c-56db0c985e00
---

## Fait

`mg_query_pathrank` (C) n'observe aucun signal ni deadline. Une fois lancée, elle tire toutes ses leaves via io_uring + décode + rerank sans checker si un budget wall-clock a été dépassé. SIGALRM depuis Python arrive dans le kernel mais aucune fonction C n'y répond → la query continue jusqu'à la fin naturelle.

Découvert 2026-07-15 pendant grid sweep SIFT 1B tp=768 mlb=400k : queries fat-tail bloquaient 5+ min chacune, guard latence Python inefficace.

## Why:

Pour un moteur qui prétend "prod-ready billion-scale sous 1 GB RAM strict", **impossible d'accepter qu'une query pathologique fige le serveur**. C'est exactement le genre de robustness que le préprint doit démontrer (cf. section "recovery/robustness demo" de [[preprint-roadmap]]).

## How to apply:

Ajouter à la roadmap préprint **avant** le "recovery demo" :

1. Passer un `int64_t deadline_ns` à `mg_query_pathrank` (0 = pas de deadline).
2. Checker dans les boucles principales :
   - traversal : entre chaque sous-arbre du fanout
   - io_uring batch reap : après chaque `io_uring_wait_cqe_nr` (idéalement)
   - rerank_l2 : tous les N docs
3. Sur dépassement : retourner **early** avec ce qu'on a (pas d'erreur), avec un flag `partial=true` dans le résultat.
4. `cancel_pending_sqe()` pour libérer les I/O en vol (io_uring_prep_cancel + reap).

**Contournement bench** : subprocess-per-config avec `subprocess.run(..., timeout=N)` — SIGKILL garantit l'abort. Voir `/tmp/sift_grid_driver.py`.

## Bloque

- Fin du grid auto-tune SIFT 1B (contourné via subprocess)
- Recovery demo (semaine 3 roadmap) — la story "kill mid-query → restart" ne marche pas si on ne peut pas kill proprement une query
