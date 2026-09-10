# Mangrove — index « ancres » : spécification algorithmique

Objectif : permettre une réimplémentation indépendante (autre langage, autre
assistant) de l'index de production de mangrove-search et de son protocole de
mesure. Tout ce qui suit a été implémenté en C (`src/anchor.c`) et mesuré ;
les chiffres de référence sont en §9.

Propriétés visées : RAM ≈ 0 (quelques centaines de Mo à 40M-1B vecteurs),
index sur NVMe ou stockage objet (S3), aucune phase d'entraînement,
ingestion live par simple append, recall@10 ≥ 0,95 sur des embeddings texte
1024d en 10-50 ms (NVMe) / ~120 ms (S3 à 40 ms de RTT).

---

## 1. Vue d'ensemble

```
BUILD
  base (N vecteurs, dim d, f16) ──► normalisation cos
  K ancres = K docs tirés au hasard (seed) ──► A[K][d] (f32, RAM)
  chaque doc → ses M=2 ancres les plus proches (spill par rang)
  code TQ par doc (rotation FWHT seedée + quantization b bits/dim)
  blocs par cellule : entrées [id u32 | code]  ──► blocks.bin + offs.bin

QUERY
  q ──► normalisation ──► scores q·A (K dots) ──► nprobe meilleures cellules
  ──► 1 vague de lectures (io_uring ou range-GET S3) des nprobe blocs
  ──► scoring TQ asymétrique de toutes les entrées (SDOT int8)
        (progressif : préfixe 256 dims → ~16k survivants → score complet)
  ──► top-R par score ──► rerank exact : lecture des R vecteurs f16 de la base
  ──► top-k
```

Il n'y a **ni arbre, ni graphe, ni médianes, ni k-means**. L'état à charger au
démarrage : `anchors.bin` (K·d·4 octets, ou K·d en int8), `offs.bin`
((K+1)·8 octets), `scale.bin` (d·4), `meta.txt`.

---

## 2. Conventions et primitives

* **Vecteurs** : base en f16 (`f16bin` : `[u32 n][u32 dim]` puis n·dim f16).
  Mode cosinus : tout vecteur (doc, ancre, requête) est normalisé L2 en f32
  avant usage. Score = produit scalaire.
* **PRNG** : splitmix64, état `s` :
  `s += 0x9E3779B97F4A7C15 ; z = s ; z = (z ^ (z>>30)) * 0xBF58476D1CE4E5B9 ;
   z = (z ^ (z>>27)) * 0x94D049BB133111EB ; return z ^ (z>>31)`.
* **FWHT** (dim = puissance de 2) : transformée de Walsh-Hadamard rapide en
  place, normalisée par 1/√d :
  ```
  for h in 1,2,4,…,d/2 :
    for i in 0..d step 2h : for j in i..i+h : (x[j], x[j+h]) = (x[j]+x[j+h], x[j]-x[j+h])
  x *= 1/sqrt(d)
  ```
* **Rotation seedée** `rot(x)` : `y[i] = x[i] * sgn[i]` avec
  `sgn[i] = ±1` tiré du PRNG (seed ^ 0x51CA), puis `y = FWHT(y)`.
  Recomputable, orthogonale, étale l'énergie sur toutes les coordonnées.
* **f16→f32** : conversion IEEE standard (`__fp16` en aarch64).

---

## 3. Build

### 3.1 Ancres
```
taken = bitset(N) ; s = seed
for k in 0..K : repeat id = PRNG(s) mod N until !taken[id] ; taken[id]=1 ; aids[k]=id
A[k] = normalize(base[aids[k]])          # f32, K x d
A8[k] = round(A[k] * 127 / max|A|)       # int8 pour l'assignation
```
Choix de K : **K ≈ 0,12 % de N** (cellules de ~1000-2000 entrées, blocs de
0,3-1,2 Mo). Mesuré : plus fin gagne en recall par octet mais perd sur
l'efficacité de lecture NVMe (petits blocs) ; plus grossier perd en couverture.

### 3.2 Assignation directe (N·K produits scalaires)
Pour chaque doc x (streamé par chunks de 200k, base jamais mmap-ée) :
```
v = normalize(x) ; v8 = round(v * 127 / max|v|)
pour k in 0..K : s = SDOT(v8, A8[k])      # int8, argmax invariant par échelle
garder les M=2 meilleurs (top-M par insertion) → topm[doc][0..M), topd[doc][0..M)
```
Coût : 40M×40k×1024 ≈ 3 h sur 20 cœurs ARM (borné par la bande passante :
A8 = 41 Mo défilent par doc). Cache `assign.bin` = `[i64 n][i64 M][topm n×M i32]
[topd n×M f32][sigmean d f32]` réutilisable pour rebâtir avec un autre code.

### 3.3 Assignation hiérarchique (`--coarse`, O(N·~96))
Depuis un index existant d'ancres `Ao` (K_o cellules) et son `assign.bin` :
```
pour chaque ancienne ancre a_o : nbr[a_o] = 64 nouvelles ancres les plus proches (SDOT)
pour chaque doc : candidats = ∪ nbr[topm_o[doc][j]] pour j<M_o  (≤128, dédupliqués)
                  top-M parmi les candidats (SDOT)
```
K=80-160k construit en ~5 min au lieu de 6-12 h. C'est le chemin de production
et le chemin à 1B (K_o = 5k direct, puis K = 320k-1M en coarse).

### 3.4 Spill (réplication frontière)
`--m 2 --eps 999` : chaque doc rejoint **ses 2 meilleures cellules** (rang).
Le spill par *marge* (`d ≤ (1+ε)·d_min`) marche sur DEEP (×2,5 à ε=0,2) mais
pas sur le texte (×1,09) : utiliser le rang. Mesuré : +5 à +8 pts de couverture.

### 3.5 Scale de quantization
Pendant l'assignation, un doc sur 64 est tourné ; `sigmean[j] = moyenne |rot(v)[j]|`.
Puis `scale[j] = qlevels / (3 · 1,2533 · sigmean[j])` avec
`qlevels = 2^(b-1) - 1 + 0,5` (b = bits/dim). (1,2533 = √(π/2), demi-normale.)

### 3.6 Codes TQ (passe 2, base re-streamée)
`r = rot(normalize(x))`, puis par dimension `q_j = round(r_j · scale_j)` :
* **TQ4** (b=4) : clip [-8,7], 2 dims/octet : `byte = (q0 & 15) | ((q1 & 15) << 4)`.
* **TQ2** (b=2) : clip [-2,1], 4 dims/octet : `byte |= ((q+2)&3) << (2·pos)`.
* **TQ1** (b=1) : bit de signe : `bit_j = (r_j ≥ 0)`, 8 dims/octet (bit j dans
  `code[j>>3]`, position `j&7`).
Taille du code : d·b/8 octets (1024d : 512 / 256 / 128 o). Une option `cdim`
tronque aux `cdim` premières dims tournées (réfuté : garder toutes les dims,
baisser b — « toutes les dims en grossier > un quart des dims en fin »).

### 3.7 Écriture des blocs
```
cnt[cell] = nb d'entrées ; offs[0]=0 ; offs[c+1] = offs[c] + cnt[c]·ent_b   (ent_b = 4 + code_b)
cur = copie de offs ; pour chaque doc, pour chacune de ses cellules c :
    pwrite(blocks.bin, [id u32 | code], cur[c]) ; cur[c] += ent_b
```
`blocks.bin` = concaténation contiguë des cellules ; `offs.bin` = (K+1)·u64 ;
`meta.txt` = `K dim M tqbits eps n seed cdim`.
Disque : N·M·ent_b (TQ1 1024d, M=2 : 272 o/doc).

---

## 4. Requête

### 4.1 Descente ancres
`s[k] = q · A[k]` pour tout k (f32, OMP), garder les `nprobe` meilleurs (top-n
par insertion, fusion par thread). K ≤ ~1M : 1-8 ms. (À 1M ancres 1024d,
A en int8 = 1 Go de RAM — SDOT ; ou graphe HNSW sur A.)

### 4.2 Vague de lecture
Pour chaque cellule c : lire `blocks.bin[offs[c], offs[c+1])` dans un buffer
contigu `blk` (offset cumulé `boff[c]`). NVMe : io_uring, toutes les SQE
soumises puis attente des nprobe CQE (ring ≥ nprobe, soumettre par lots si
`get_sqe` rend NULL). S3 : `nprobe` range-GETs HTTP parallèles sur l'objet
`blocks.bin` (libcurl multi, SigV4, connexions keep-alive réutilisées).
**Buffers alloués une fois et pré-touchés** (le malloc/mmap par requête coûtait
~90 ms de page-faults en mono-thread).

### 4.3 Requête quantizée et scoring
```
qr = rot(normalize(q)) ; q8 = round(qr / max|qr| · 127)      # int8
TQ4 : score = Σ_j q8[j]·q_j    via nibbles sign-étendus (shl4/shr4, shr4) + SDOT
      contre q8 réordonné en 2 flux (dims paires / impaires)
TQ2 : 4 flux (dims ≡ 0,1,2,3 mod 4), champs 2 bits (and 3, −2) + SDOT
TQ1 : 8 flux (dims ≡ b mod 8), (byte >> b) & 1 + SDOT   [= Σ_{bit=1} q8[j], offset constant]
```
Les scores sont asymétriques (requête en int8 complet, doc quantizé) : le
rang est bien plus fidèle qu'un Hamming. **Vérifier que SDOT est réellement
compilé** (`gcc -march=… -dM -E | grep ARM_FEATURE_DOTPROD` ; sur GB10
`-march=native` ne l'active pas, il faut `-march=armv8.2-a+dotprod+fp16`).

**Scoring progressif** : passe A sur les `DIM_PRE = 256` premières dims tournées
(le préfixe FWHT porte ¼ de la variance), présélection de ~16 384 survivants
au total (min-tas **binaire** par thread — l'insertion décalée est O(n) et
tue le mono-thread), passe B score complet des survivants, top-R local par
thread, fusion. Coût : ~3-4 CPU-ms pour 120-300k entrées.

### 4.4 Rerank exact
Dédupliquer les R meilleurs ids (les entrées spill apparaissent 2×), lire les
R vecteurs f16 de la base **en une vague** (io_uring ou range-GETs sur
`base.f16bin`, offset `8 + id·d·2`), normaliser, produit scalaire f32, trier,
rendre top-k. R = 100 pour TQ4, **300 pour TQ1/TQ2** (absorbe la perte de
quantization : TQ1 r300 = TQ4 r100 − 0,4 pt).

### 4.5 Paramètres de production (1024d, texte)
| | valeur | effet |
|---|---|---|
| K | 0,12 % de N | granularité (cf. §3.1) |
| M | 2, spill rang | couverture frontière |
| code | TQ1 (132 o/entrée) ou TQ2 (260) | octets lus ÷4 ou ÷2 vs TQ4 |
| nprobe | 64 / 128 / 256 | 0,94 / 0,96 / 0,97 (anglais 40M) |
| R | 300 | rerank |
| threads | 1 suffit (io-bound) | |

---

## 5. Mode S3

Aucun changement de format : `blocks.bin` et `base.f16bin` sont **un objet
chacun**, adressés par `Range: bytes=a-b` avec les offsets de `offs.bin`.
Client : libcurl `curl_multi`, `CURLOPT_RANGE`, `CURLOPT_AWS_SIGV4`
(`aws:amz:<region>:s3`), identifiants uniquement via `AWS_ACCESS_KEY_ID` /
`AWS_SECRET_ACCESS_KEY` (env), `CURLMOPT_MAX_HOST_CONNECTIONS ≥ nprobe`,
`net.ipv4.tcp_slow_start_after_idle = 0`. Deux vagues séquentielles (blocs,
puis rerank). Physique mesurée : vague blocs ≈ 2 RTT (requête + une fenêtre
TCP de ~200 Ko), rerank ≈ 1 RTT. Hedging (doublon des GETs en vol après un
délai, sur connexion chaude) : implémenté, à valider sur bucket réel (une
émulation netem mono-file ne peut pas le montrer). Simulation locale :
MinIO docker + `tc qdisc add dev docker0 root netem limit 200000 delay 20ms 5ms`.

---

## 6. Ingestion live

Un nouveau doc : normaliser, descendre (§4.1, M=2 cellules), calculer son
code (§3.6), **append** `[id|code]` en queue des 2 blocs (ou dans un segment
delta par cellule si les blocs sont immuables, ex. S3 : segment `cell.delta`
lu avec le bloc, fusionné à la compaction). Aucune structure à réparer, aucune
recalibration : les ancres sont gelées. Ré-échantillonner les ancres et
rebâtir (§3.3, minutes) après quelques doublements de N.

---

## 7. Filtrage par métadonnées

Le pool de candidats est connu **avant** le scoring (ids dans les blocs) : un
filtre (bitmap roaring par valeur, ou plage de lignes par langue — cf.
`manifest.json` de wikiall) s'applique à 4 octets par candidat ; le rerank ne
lit que les survivants. Sélectivité < ~50k éligibles → force brute sur les
éligibles (mesuré plus rapide).

---

## 8. Protocole de mesure (obligatoire pour comparer)

* **GT exacte** : top-10 cos brute-force par chunks explicites (jamais mmap)
  sur 200 requêtes tirées de la base ; si la requête est membre de la base,
  **exclure son propre id de la GT et rendre top-11 puis filtrer le self**
  (sinon plafond 0,90).
* **Cold** : `sync; echo 3 > /proc/sys/vm/drop_caches` **avant chaque
  requête** ; p50/p99 par phase (ancres / io / score / rerank) chronométrées
  dans le processus C. Warm = préchauffage explicite de 150 requêtes.
* **Mémoire** : RSS et `memory.current` sous `systemd-run --scope -p
  MemoryMax=… -p MemorySwapMax=0` ; aucun `mmap` dans les outils (fausse le RSS
  et a provoqué un OOM à 12 procs).
* **Recall@10** et **Δcos** (regret de similarité : cos moyen des GT − cos
  moyen des retrouvés ; à recall 0,95 on mesure ~0,002, p90 ≤ 0,0055).
* Mono-thread ET multi-thread (`OMP_NUM_THREADS=1`) : le mono révèle les
  coûts CPU masqués.

---

## 9. Résultats de référence (cold, GT exacte)

| corpus | config | recall@10 | p50 | p99 | RAM |
|---|---|---|---|---|---|
| wiki-it 7,4M×1024d | K=10k, TQ1, np64, r300, **1 thread** | 0,942 | **11,5 ms** | 15 | ~50 Mo |
| idem | 20 threads | 0,942 | 9,3 ms | 13 | |
| idem, TQ2 | 1 thread | 0,946 | 15,1 ms | 20 | |
| wiki-en 40M×1024d | K=40k, TQ1, np128/256/512, r300 | 0,940 / 0,962 / 0,974 | 28 / 44 / 76 ms | 39 / 65 / 104 | 170 Mo |
| idem, TQ4 r100 | | 0,946 / 0,968 / 0,984 | 55 / 95 / 178 ms | | |
| wiki-it, **S3 simulé 40 ms RTT** | TQ1 np128 r300 | 0,960 | 120 ms | ~245 | |
| DEEP 10M×96d (sim) | K=10k, spill ε0,2, np64, TQ4, exact100 | 0,988 | — | — | |

Repères externes sur la même GT : HNSW (hnswlib) wiki-10M plafonne à 0,965
avec 41 Go de RAM ; DiskANN 1B ≈ 64 Go de RAM, lectures séquentielles
(incompatible S3).

---

## 10. Résultats négatifs à ne pas refaire

* Sous-espace de dims (préfixe 256 dims, esquisses int8 16-48 dims) : 0,14-0,86.
* Buckets virtuels seedés dans la cellule (argmax de projections aléatoires,
  16×16) : non uniformes sur des résidus réels (p50 3, p99 124, 40-50 % vides),
  0,52-0,56 ; directions-voisins +3-11 pts seulement.
* Sous-ancres échantillonnées / adaptatives : ×1,7-3 d'octets dans la zone
  0,85-0,90 seulement ; à 0,95+ la courbe rejoint les cellules entières.
  **Constante du corpus : ~1 % de N à examiner pour 0,95, quel que soit le
  partitionnement** ; le levier restant = octets par doc examiné.
* Esquisse deux-vagues (sketch 256 bits + pages 4 Ko) : 0,945 @ 24 Mo, ne bat
  pas TQ1 mono-vague (0,944 @ ~18 Mo).
* K grossier (cellules 4-8k) : couverture inférieure à octets égaux.
* Hachage des candidats : plus lent que le radix sur ARM (insertions aléatoires).
* Codes de feuilles d'une petite forêt (16-64 arbres) comme quantizer : 0,45.

---

## Annexe — la forêt RP (design précédent, prouvé à 1B ; spécification complète et mesures dans `SPEC_RP_FOREST.md`)

256 arbres d'hyperplans seedés (sign-split ou médianes échantillonnées
`med_depth ≈ log2(N/2000)`), feuilles varbyte triées, multi-probe par
marges (single-flip puis multi-flip par ensembles de perturbations), deux
vagues io_uring, **vote** = multiplicité d'un id entre arbres, rerank des
top-N votés. DEEP 1B : 0,968 @ 272 ms, 815 Mo ; 50M : 0,980, p99 73 ms,
154 Mo ; ingestion live 1M→50M à 176 Mo RSS. Variante v2 (médianes exactes
niveau-synchrones int8, slots fixes, une vague) : 100M 0,980 @ 66 ms, 0,32 Go.
Limite structurelle : 16 000 lectures aléatoires par requête (NVMe ok,
S3 impossible), et le vote ne trie plus en dessous de ~64 arbres.
