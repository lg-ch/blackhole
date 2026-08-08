# Mangrove-search — méthode

## Principe

Mangrove est un moteur ANN (approximate nearest neighbor) pour vecteurs
denses, conçu pour scaler au milliard de documents **sous 1 GB de RAM
process**. L'index vit sur SSD, la RAM ne garde que des échantillons
d'accès (samples) O(log N) par arbre.

Le socle est un **Random Projection Forest** avec deux propriétés fortes :

- **Zéro RAM data-dépendante** : les hyperplans de routing sont dérivés
  d'une seed déterministe, pas des données. Aucune table de centroïdes,
  quantifieurs entraînés, ou centroids résidents — l'arbre est
  reconstructible à partir de son seed seul.
- **Read O(1) probes indépendants** : chaque tree est un fichier `.srt`
  standalone, on lit une leaf en 1-2 io_uring ops.

## Arbres

### Construction

Chaque arbre est un BST binaire à profondeur fixe `depth` (typiquement
25-28) partitionnant l'espace des vecteurs. Chaque nœud interne encode un
**hyperplan sign-split** défini uniquement par :

- `seed(t)` : seed de l'arbre (déterministe, dérivée du tree_index)
- `path_bits` : le chemin binaire du root au nœud

On génère un vecteur pseudo-aléatoire de dimension `sub_dim` sur ces
seed + path_bits. Le split est `sign(⟨v, hyperplane⟩)` → 0 va à gauche,
1 à droite. Aucun apprentissage, aucune donnée nécessaire à ce stade.

### Storage

Les vecteurs (données) descendent chaque arbre selon ses hyperplans jusqu'à
une leaf. Chaque leaf accumule les `doc_id` qui y aboutissent. Le stockage
est **compact** : `.srt` V3 varbyte delta encoding, sparse index sur
`leaf_id → offset`, indexé par une petite table de samples résidente RAM
(stride 4096 → ~5 KB par arbre).

Multi-arbre = redondance stochastique. Un query descend indépendamment
chaque arbre → n_trees candidates leaves. La probabilité qu'un vrai
voisin soit trouvé dans la même leaf que le query par au moins un arbre
tend vers 1 avec n_trees.

### Traversée query

Pour une query `q` :
1. Chaque arbre `t` traverse depth niveaux → aboutit à une leaf `L_t`
2. Multi-probe : à chaque niveau on peut aussi explorer le "flip" (côté
   opposé de l'hyperplan). `n_probes` détermine combien de flips explorer.
3. À `n_probes=5` sur SIFT 1B on obtient 6 sets × 256 trees = **1536
   probe paths** candidats.

### Query depth

Un paramètre `query_depth` (qd) permet d'arrêter la traversée avant
`depth` build. À `qd < depth` :
- Chaque probe couvre `2^(depth-qd)` leaves consécutives dans son subtree
- Coverage plus large → recall plus élevé
- Trade-off : lecture disque plus grosse

Levier principal recall : baisser qd augmente le rayon de recherche
autour de chaque chemin natif.

## Votes

Le scoring cross-tree agrège par doc_id :

```
vote(doc) = # trees dans lesquels doc apparaît dans un probe leaf
```

Un vrai NN a une probabilité élevée d'apparaître dans plusieurs trees
(structure géométrique préservée à travers les hyperplans aléatoires).
Un doc bruit apparaît dans peu de trees.

Le top-N final = les `top_n` doc_ids avec les plus hauts votes.

### Path-rank (variante fusionnée)

Pour éviter de lire toutes les probe leaves, on rank les paths par leur
**min-margin score** = distance minimale du query aux hyperplans traversés.
Un score élevé indique un routing "clair" (peu d'ambiguïté). Path-rank
garde les `top_paths` meilleures paths cross-tree, les autres sont
droppées avant lecture disque.

## Pipeline query (SRT V3, path-rank fusionné)

1. **Traversal** : parallèle par arbre via OMP, quelques ms CPU
2. **Path-rank sort** : keep `top_paths` par margin
3. **Phase 1 io_uring** : batch read des sparse index windows autour de
   chaque probe leaf (~4 KB par read)
4. **Phase 2 io_uring** : batch read des données varbyte de chaque leaf
   (typiquement <100 B par leaf à sub-4-docs/leaf)
5. **Décode + radix sort** : `(tree, doc)` packés en uint64, LSB radix sort
   3 passes × 11+11+10 bits, séquentiel cache-friendly
6. **Vote linear scan** : dedup par tree via `tree_seen[]`, histogram
   `[vote_count → n_docs]` pour top-N pick sans heap

## I/O économique

Règles design dures :
- **Pas de mmap** dans le hot path. Uniquement `io_uring + O_RDONLY`.
  Préserve la caractéristique RAM-négligeable (mmap polluerait RssFile).
- **Samples in RAM, sparse index on disk** : par arbre ~5 KB RAM, le reste
  streaming via io_uring.
- **Radix sort > heap** : sur ~1 M paires (tree, doc) par query, radix
  est ~5× plus rapide que heap et cache-friendly.
- **max_leaf_bytes** : cap on-the-fly des mega-leaves lors du Phase 2
  (skip si > seuil) → borne le p99 sans dégrader le p50.

## Streaming (livré 2026-07-17)

- **HOT overlay in-RAM** : per-tree sorted sparse leaf_id → docs, thread-safe
  via mutex par tree
- **Query merge MAIN + HOT** : injection dans le pack loop du radix, dedup
  cross-format via `tree_seen[]`
- **Compaction background per-tree** : pthread round-robin, throttled, output
  SRT V3 varbyte, atomic rename + fd swap sous rwlock
- **Bulk mode** : atomic dir swap pour rebuild offline
- SDK rate-limit 1k vec/s (86 M docs/jour)

## Résultats

Latences **cold cache, cgroup 1 GB RAM, single CPU 8 cores affinity**, sauf
mention. Corpus billion-scale mesurés en production :

| corpus       | dim | n_trees | qd  | tp   | recall@10 | p50 lat | RAM   |
|--------------|----:|--------:|----:|-----:|----------:|--------:|------:|
| SIFT 1B      | 128 |     256 |  25 | 4000 |   **0.993** |  722 ms | 935 MB |
| SIFT 1B ×10  | 128 |     256 |  25 | 4000 |     0.993 |  130 ms | 935 MB |
| DEEP 1B      |  96 |     256 |  18 | 4000 |   **0.970** |  530 ms | ~700 MB |
| SIFT 10M     | 128 |      64 |  20 | 1024 |     >0.98 |   50 ms | ~200 MB |
| NQ Cohere v3 | 768 |      64 |  16 | 8000 |     0.81  |  197 ms | ~500 MB |
| arxiv 2M     | 768 |      64 |  14 | 2048 |   **1.000** |  110 ms | ~200 MB |

**Régime nominal 1B : recall 0.97-0.99 @ 500-600 ms cold single-thread,
< 1 GB RAM.** En parallèle ×10 cores → ~130 ms p50.

Warm cache (page cache réactif sur queries répétées) : latence typique
3-5× plus basse. En prod SaaS multi-tenant on tourne autour de **30-100 ms
p50 warm** sur nos corpus arxiv (2M) et cohere (20M).

## Streaming en régime stress

Bench SIFT 10M mono / 64 trees / 1 k vec/s ingest streaming + background
compaction + queries concurrentes :

| config              | mean | p50 | p95 | p99 |
|---------------------|-----:|----:|----:|----:|
| idle baseline       |   12 |  11 |  26 |  34 |
| stress live streaming |   13 |  12 |  27 |  31 |

**+6.6 % mean / -8 % p99** — dans le bruit. Pipeline complet validé sous
charge concurrente réaliste.

Extrapolation SIFT 1B (256 trees, compaction per-tree ~1.1 s vs 70 ms) :
overhead attendu **plus élevé** (~+20-40 % p99 selon throttle),
nécessite tuning du sleep_ms et io_uring priorities kernel. Bench 1B stress
à valider empiriquement.

## Comparaison

Face à HNSW et FAISS-IVF :
- **RAM** : mangrove ~500 MB / 1B ; HNSW 40-80 GB ; FAISS-IVF quantifié
  15-40 GB
- **Recall** : mangrove 0.99+ ; HNSW 0.99+ ; FAISS-IVF 0.90-0.98
- **Latency cold** : mangrove 500-700 ms ; HNSW <10 ms (tout en RAM) ;
  FAISS-IVF 100-300 ms
- **Streaming** : mangrove overlay + compaction background ; HNSW
  ajouts individuels O(log n) mais RAM permanente ; FAISS-IVF batch
  re-train obligatoire

Mangrove trade off latency contre RAM. Cible : déploiements avec
contrainte forte RAM (edge, on-prem à budget matériel serré,
multi-tenant à haute densité par machine). Sur AWS r7iz ou équivalent
"beaucoup de RAM", HNSW reste plus rapide.
