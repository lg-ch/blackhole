---
name: median-toplevels
description: "Médianes top-levels VALIDÉES (2026-08-02) : p99 ÷2.5-3, pool ÷2.2-3.3, queue leaves ÷4.4, build -35%, 10M@0.93 en 16ms/68MB RSS. Candidat défaut v1."
metadata:
  type: project
---

## Idée (utilisateur) et implémentation

Seuils médians sur les niveaux 0..med_depth-1 (θ[node] sur c1-c0), random en
dessous. RAM : (2^md - 1) × 4 B × n_trees (1-4 MB). Zéro coût query.

- `traversal.c` : table thread-local `traversal_set_medians(table, md)` ;
  test `node < 2^md - 1` ⟺ level < md (heap). 11 sites patchés, margins inclus.
- `traversal_calibrate_medians()` : médianes exactes niveau par niveau sur un
  échantillon STRIDÉ (600k), = limite convergée d'un estimateur online.
- `build_tree.c` : `calibrate_and_save_medians()` → `medians.bin`
  (MED1, n_trees, med_depth) ; build_forest_ex le charge automatiquement.
- `forest_open` charge medians.bin ; mg_query_pathrank set par arbre.
- FFI `mg_calibrate_medians`. Flow : mkdir → calibrate → build → query.

⚠️ PIÈGES : (1) calibration DOIT avoir le même gen_version que le build
(bug initial : v0 vs v3 → seuils aléatoires, balance DÉGRADÉE 3.7×) ;
(2) échantillon stridé sur tout le fichier (biais d'ordre) ;
(3) gates `use_sub` passés à `sub <= dim` — casse deep_10m_sd96_d23 (ancien).
(4) Cohérence streaming : seuils FIGÉS post-calibration (dérive = docs échoués),
refresh au bulk rebuild uniquement.

## Résultats DEEP (1M d=17 md=10 ; 10M d=20 md=12, NVMe interne)

Balance niveau md : 2.1-3.0× (classique) → **1.00-1.01×** (parfait).
Queue des leaves (1M) : max ÷4.4 (6580→1481 B), size-biased mean ÷2.2 (161→73 B).
Build 10M : **-35 % de temps** (3191 vs 4890 s — external sort équilibré).

10M qd=18 (régime prod) :
| config | classique p50/p99/recall/pool | médianes p50/p99/recall/pool |
| NP3 tp512 | 13.9/45.2/0.914/200k | 11.5/16.7/0.862/60k |
| NP3 tp1024 | 22.2/73.0/0.966/324k | 16.0/23.9/0.934/108k |
| NP7 tp2048 | 55.8/148.2/0.982/522k | 52.9/109.2/0.978/192k |

**Iso-recall** : pool ÷2.2, p50 ≈ voire mieux, **p99 ÷2.3-3**. Le gain CROÎT
quand qd descend (qd=16 : p50 ÷1.9, p99 ÷2.7 iso-recall) — régime bytes-bound.
Recall à config égale : -3 à -8 pts (médiane coupe au plus dense → voisins
séparés) — racheté par ~+40 % tp, déjà compté dans l'iso-recall.
Très haut recall (0.98+) : avantage p50 s'estompe ; reste pool/bytes/p99.

## RAM (le claim)

**DEEP 10M, recall 0.934, p50 16 ms sous cgroup 200 MB, RSS process 68 MB.**
Pool ÷2.2 → buffers dominants ÷2.2 → à 1B le "sous 1 GB" pourrait devenir
"sous ~512 MB" (à valider, build 1B médianes ~30h).

## Restes ouverts

- Anomalie qd=20/NP7 médianes p99 275 ms (artefact cache guest probable) — revalider.
- Plafond très-haut-recall (0.99) à confirmer avec tp élargi.
- mlb probablement relaxable (queue bornée) → recall floor mieux.
- Recommandation : **défaut ON dans v1** (med_depth=12 pour ≥10M docs).

## COLD sous cgroup 200 MB (le test décisif, 2026-08-02)

drop_caches/query + MemoryMax=200M, qd=18, DEEP 10M :

| config | classique p50/p99/recall/RSS | médianes p50/p99/recall/RSS |
| NP3 tp512  | 95/207/0.897/84MB  | 30/45/0.852/56MB  (p50 ÷3.1, p99 ÷4.6) |
| NP3 tp1024 | 112/226/0.955/110MB | 50/72/0.920/68MB  (p50 ÷2.2, p99 ÷3.1) |

**Iso-recall ~0.92 : p50 ÷2, p99 ÷3, RSS -35 %.** Le warm masquait l'effet
(bytes gratuits) ; à froid chaque byte du pool se paie → le ÷2.2 pool + ÷4.4
queue frappent en plein. Le régime froid = positionnement produit de mangrove.
Claim : **10M @ 0.92 recall, 50 ms p50 / 72 p99 COLD sous 200 MB total.**

Échelle cold 200MB médianes complète (10M qd=18) :
tp512 → 0.852/30ms ; tp1024 → 0.920/50ms ; NP7 tp2048 → **0.976/89ms p50/131 p99/RSS 93MB**.
p99 ≈ 1.5× p50 partout (variance domptée). Médianes COLD ≈ classique WARM (55.8/148).
