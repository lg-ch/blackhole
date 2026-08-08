---
name: cold-warm-bench-protocol
description: "Toujours séparer cold/warm dans les benchs latence — l'utilisateur a attrapé deux fois des chiffres warm présentés comme représentatifs"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 0e757620-83dd-4e01-864c-56db0c985e00
---

Règle : tout tableau de latence doit préciser cold ou warm, et les chiffres publiables exigent le protocole complet : `drop_caches` avant chaque config, mesure cold (1ère query) ET steady-state warm (après warmup explicite) en deux colonnes.

**Why:** deux incidents — (1) 918 ms vs 321 ms pour la même config cohere 41M (ordre des configs warmait des régions différentes du sparse_index), (2) sweep en2 leviers où les configs s'enchaînent dans le même process donc les dernières profitent du cache des premières. L'utilisateur a flagué les deux.

**How to apply:** dans tout script de bench multi-config, soit dropper les caches entre configs, soit annoncer explicitement "warm steady-state" dans le tableau. Pour mangrove le cold est un argument de vente (pas de chargement RAM massif au démarrage vs HNSW) — le mesurer proprement, pas le cacher. Voir pattern dans `R&D/bench_cohere_qd18_clean.py` (phases cold/warmup/warm).
