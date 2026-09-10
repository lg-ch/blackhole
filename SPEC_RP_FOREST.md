# Mangrove — forêt RP : sign-split, médianes échantillonnées, médianes exactes (v2)

Spécification des designs antérieurs à l'index « ancres »
(`SPEC_ANCHOR_INDEX.md`), avec les mesures obtenues. Tout a été implémenté en
C (`src/traversal.c`, `build_tree.c`, `query_tree.c`, `build2.c`,
`slots_v2.c`, `hot_*.c`) et mesuré ; c'est la forêt v1 qui a produit la preuve
à 1B (< 1 Go de RAM).

---

## 1. Principe commun : arbres de projections aléatoires recomputables

* **Forêt** de T arbres (T = 256 en production, 32-64 en sous-forêt), chaque
  arbre = arbre binaire complet de profondeur D (D = 17 à 10M, 20 à 100M, 25
  à 1B ; ≈ log2(N/500)). Numérotation en tas : racine 0, enfants 2n+1, 2n+2 ;
  feuille = nœud de profondeur D, id de feuille = nœud − (2^D − 1).
* **Rien n'est stocké pour la structure** : l'hyperplan de chaque nœud est
  regénéré à la volée depuis `(seed_arbre, nœud)`.
  ```
  ts      = tree_seed(t)                       # PRNG seedé par l'indice d'arbre
  dims    = pick_dims(ts, node, dim, sub_dim)  # sub_dim (16) coordonnées choisies par nœud
  v0, v1  = gen_vec(node_seed(ts, 2·node)), gen_vec(node_seed(ts, 2·node+1))   # ±1 / gaussiens seedés
  c0, c1  = Σ_{i∈dims} x[i]·v0[i], Σ x[i]·v1[i]
  branche = (c1 − c0 > θ[node]) ? droite : gauche
  ```
  `θ[node] = 0` en **sign-split** ; `θ` = médiane du nœud en mode **médianes**.
  La *marge* d'un niveau = `|c1 − c0 − θ|` : petite marge = doc/requête près de
  la frontière.
* **Version de génération** : `gen_version = 3` (v3) partout — un helper
  autonome qui n'arme pas la version ou la table de médianes route en silence
  dans un autre index (piège vécu trois fois : toujours armer l'état global
  avant `traverse_batch`/probes).

## 2. Forêt v1 : stores triés, deux vagues, vote

### 2.1 Build (streaming, ~50 Mo de RSS)
Les seuils sont **gelés avant** le build (θ = 0 ou médianes calibrées), donc
chaque vecteur se route indépendamment : lecture séquentielle de la base par
lots, descente dans les T arbres (OMP par arbre), append `(doc_id, feuille)`
dans un fichier par arbre, tri final par feuille → **store trié**
`tree%05d.srt` : postings de doc_ids **varbyte** (~2,5 o/doc/arbre) précédés
d'un **index creux** (une entrée toutes les `sample_stride` feuilles :
feuille → offset). Build par lots d'arbres résumable (1B : 256 arbres en lots
de 8, ~2 h 12/lot, 32 Go RSS par lot sur Vultr 8c).

### 2.2 Requête
1. Descente de la requête dans les T arbres à profondeur `qd ≤ D` (qd < D =
   plage contiguë de 2^(D−qd) feuilles ; « qd » pilote la largeur).
2. **Multi-probe** par arbre : trace des marges le long du chemin ;
   * *single-flip* : une probe = le chemin avec **un** niveau inversé, par
     ordre de marge croissante — sature à ~D probes par arbre ;
   * *multi-flip* (`mg_set_probe_multiflip`) : tas d'ensembles de perturbations
     (extension/décalage, cf. Lv et al.), re-descente exacte à partir du
     niveau inversé → nombre de probes illimité, ordonnées par somme des
     marges. Global : `tp` = probes gardées en tout (sélection par score) —
     réfuté à qd natif, l'allocation uniforme par arbre gagne.
3. **Phase 1** (vague io_uring) : lecture des fenêtres de l'index creux qui
   encadrent chaque (arbre, plage) → offsets exacts des postings.
   **Phase 2** (vague) : lecture des postings varbyte.
4. **Vote** : paires (arbre | doc) → **radix 11/11/10** par doc → un doc
   compte 1 vote par arbre où il apparaît (dédup `tree_seen`) →
   histogramme de votes → seuil pour tenir `tn` candidats, **émission en deux
   passes** (tous les votes > seuil, puis == seuil ; une passe premier-arrivé
   perdait les gros votes à id élevé).
5. **Rerank exact** des `tn` candidats (lecture des vecteurs par vague
   io_uring fenêtrée de 2048, L2/cos f32) → top-k.
   Requête membre de la base : rerank `top_k = k+1` et filtrer le self.
6. Filtres : bitmap d'éligibles (roaring) appliqué avant le rerank ; branche
   « curseurs » (sans overlay) ou « radix » (avec overlay HOT) — la
   première exigeait un qsort des tranches à k_shift > 0 (bug corrigé).

### 2.3 Ingestion live (HOT overlay)
Insert : descente (mêmes seeds, **table de médianes live** armée), append
dans `tree%05d.c%d.hot` (un fichier par arbre et **classe de taille** de
feuille, promotions de classe), WAL, backpressure ; lecture : union
MAIN + HOT ; compaction HOT → MAIN. Débit 2,5k → 0,7k vec/s de 1M à 50M
(mono-flux). Mesuré : requêtes pendant injection +6 ms, RSS 176 Mo.

## 3. Médianes échantillonnées (« idx_med »)

* **Calibration** avant le build : `n_sample` vecteurs (1,5-8M) tirés par
  pas constant dans la base ; pour les `med_depth` premiers niveaux, niveau
  par niveau, θ[nœud] = médiane des `c1 − c0` de l'échantillon tombé dans ce
  nœud. Fichier `medians.bin` MED1 : `[u32 'MED1'][n_trees][med_depth][rsvd]
  [n_trees × (2^med_depth − 1) f32]`.
* **Règle** : `med_depth ≈ log2(n_sample / 2000)` (~2000 échantillons par
  seuil ; plus profond = bruit : recall ET équilibre se dégradent).
* Sous-forêt : ouvrir T' < T arbres utilise le préfixe de la table
  (accepter `n_trees(table) ≥ T'`, sinon stranding massif).
* Table live process-globale pour les helpers sans handle de forêt
  (`mg_live_medians_load`) ; sans elle, 73 % des docs live étaient perdus.

## 4. v2 : médianes exactes niveau-synchrones + slots fixes

### 4.1 Build (`build2`, C, base en RAM explicite)
Pour chaque arbre, **niveau par niveau sur toute la population** :
```
tri par comptage des docs par chemin courant (2^niveau nœuds)
pour chaque nœud : projections c1−c0 de ses docs, médiane EXACTE (quickselect)
grille int8 du niveau : lo = min des médianes du niveau, span = max − lo,
   code = round((θ−lo)/span·254), θ_snappé = code/254·span + lo   (bit-exact au décodage)
routage avec θ_snappé ; ex æquo (proj == θ) : bit = hash(doc)&1   → équilibre p99 = moyenne
```
Résultat : feuilles de taille quasi constante (10M/d17 : p50 = p99 = 61-62,
0 vide, 0,08 % de débordement à 1,5×). Croissance sur seuils gelés : uniforme
(+20 % → p99 88). Coût : 100M × 256 arbres d20 = 1 h 26 sur 20 cœurs,
74 Go RSS (base 38 Go + chemins par groupe de 32 arbres).
Format **MED2** : `[u32 'MED2'][nt][md][rsvd][scales f32 nt×2×md : lo[md] puis
span[md] par arbre][codes u8 nt×(2^md−1)]` ; décodage à la volée
`(code/254)·span[niv] + lo[niv]` ; 268 Mo à 256 arbres d20 (f32 : 1,07 Go).

### 4.2 Store à slots fixes (`slots_v2`)
Une feuille = un **slot de taille fixe** (512 o : `[u32 count][u32 ids…]`,
cap 127 ≥ p99 + croissance ; règle slot ≈ moyenne × 1,25-1,5), offset =
`feuille × slot_bytes` : **plus d'index creux, plus de phase 1**, une seule
vague io_uring ; à qd < D une probe = 2^(D−qd) slots contigus. Troncature
0,01 % absorbée par la redondance de forêt. Disque ≈ 2,7× le varbyte.

### 4.3 Requête v2
Descente + multi-flip **à qd natif** (une probe = un slot), `tp = (NP+1)·T`
(pas de sélection globale), lecture en une vague, **vote pool qd natif** : à
qd = D un doc apparaît au plus une fois par arbre → dédup structurelle → ids
u32 nus, radix 11/11/10, comptage de runs (le hash open-addressing était plus
lent : insertions aléatoires < passes séquentielles), top-n deux passes,
rerank `tn = 8000` (un `tn` de 60-100k mangeait le gain de la phase 1).

---

## 5. Mesures (chronologiques ; cold = drop_caches par requête, cgroup)

### 5.1 Forêt v1
| corpus | config | recall@10 | latence | RAM |
|---|---|---|---|---|
| DEEP 1B, sign-split | 256 arbres d25 | ~0,98 | ~300 ms | < 1 Go |
| DEEP 1B, médianes md14 | NP3 tp1024 qd20 | 0,946 | 185 ms p50 e2e | 497 Mo |
| DEEP 1B, médianes | NP7 tp2048 | 0,968 | 272 ms p50 | 815 Mo |
| DEEP 50M (index vécu ×50 par injection live) | NP3 tp1024 qd18 | 0,956 | 34 ms p50 | 108 Mo |
| DEEP 50M | NP7 tp2048 | 0,980 | p99 72,7 ms | 154 Mo |
| DEEP 10M, médianes md13 | qd17 (calibré offline) | 0,968 | 188 ms warm | |
| campagne live 1M→50M | checkpoints vieilli/recalibré | 10M 0,972/0,957 · 20M 0,948/0,948 · 50M 0,958/0,955 | injection 2495→718 vec/s | |
| RAG complet 50M (métadonnées, filtres, injection simultanée) | | 49/50 pur, 47/50 composé, 49/50 sous injection | 55-61 ms | 121-176 Mo |
| wiki Cohere-v3 10M×1024d | qd13 NP7 tp4096 | **0,981** (HNSW plafond 0,965 / 41 Go) | 205 ms warm | |
| LAION CLIP 10M×512d | qd15 NP7 tp4096 / qd14 tn100k | 0,945 / 0,977 (HNSW 0,975 @ 4 ms / 22 Go) | 32 / 118 ms | |
| LAION 407M | qd21-24 | 0,850 / 0,929 / 0,941 | 144 ms / 644 ms / 1,08 s warm 8c | |
| CLIP 10M, multi-flip | 64 arbres / 256 arbres qd14 NP15 | 0,964 / **0,990** | 209 / 205 ms | |

Filtres (50M, warm, NP3) : 0,95 filtré coûte ×3 à 50 % de sélectivité
(66 ms), ×9 à 1 % (191 ms), mur à 0,01 % ; famille corrélée plafonne à
0,825 en ordre-marge → union + rerank (0,953 @ 806 ms) ; brute force
21 ms @ 500k warm, 2-4 ms @ 50-100k.

### 5.2 Médianes exactes et v2
| corpus | mesure | résultat |
|---|---|---|
| DEEP 10M | médianes exactes d17 vs échantillonnées md13, ancien pipeline | recall parité (0,994-0,9995 à qd14-15), **latence −10 %** p50 et p99 |
| DEEP 10M | slots fixes 512 o vs varbyte + phase 1, mêmes arbres | recall identique (5 configs), **−25 à −40 %** warm et cold |
| DEEP 100M, v2 | qd18 NP3 / qd17 NP7, tn 20k, cold | 0,932 @ 58 ms (p99 89) / 0,988 @ 120 ms |
| DEEP 100M, v2 | **qd natif 20**, NP15 / NP63 mf, tn 8000, cold | **0,980 @ 66-67 ms** (p99 88) / 0,990 @ 110 ms |
| idem warm (préchauffé) | NP15 / NP63 / NP127 | 0,953 @ 34 / 0,985 @ 56 / 0,992 @ 123 ms |
| DEEP 100M | RSS bench, MED1 f32 → MED2 int8 | 1,07-1,13 Go → **0,32 Go**, recall bit-identique |
| DEEP 100M | qd-only (plages profondes NP0) | 0,948 @ 203 ms cold — réfuté vs multi-probe |
| DEEP 100M | sélection top-tp vs uniforme | uniforme +3-4 pts à lectures égales |
| loi d'échelle 10M→100M | | ×2,5-3,3 cold ≈ asymptote pure-cold (lectures/req constantes) |

### 5.3 Limites qui ont motivé l'index ancres
* 16 000 lectures aléatoires par requête (16k IOPS) : parfait sur NVMe,
  **impossible sur S3**.
* Le vote ne trie plus en dessous de ~64 arbres (16 arbres : codes de
  feuilles 0,45, votes 0,63 @ 500).
* Le build v2 exige la population en RAM (médianes exactes globales) ; le
  build v1 est frugal mais sur seuils échantillonnés.
* Sur texte 1024d, la forêt exigeait tp4096/tn100k (205 ms warm à 10M) là où
  les ancres font 0,962 @ 44 ms cold à 40M.
