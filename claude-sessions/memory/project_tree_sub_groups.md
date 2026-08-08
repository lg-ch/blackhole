---
name: project-tree-sub-groups
description: "Per-tree input subspace (tree_sub) + groupes partagés (tree_sub_groups) sur arxiv 2M. Sweet spot g=16, ts ∈ {64,128}. Patch pick_dims identity au passage."
metadata: 
  node_type: memory
  type: project
  originSessionId: 0e757620-83dd-4e01-864c-56db0c985e00
---

**Livré 2026-06-14/15/16** : ensemble "random subspace par arbre" + groupes
partagés. Tous câblés (build CLI `--tree_sub N --tree_sub_groups G`, persistés
dans `meta.txt`, auto-appliqués en query via `meta_read` pour éviter le
silent-KO type [[feedback-check-meta-depth]]).

**Two-knob design :**
- `tree_sub N` : chaque arbre voit un sous-espace fixe de N coords sur les
  full_dim (gather seed-derived) AVANT le node-level sub_dim sampling. N=0
  → off, byte-identical legacy.
- `tree_sub_groups G` : G < n_trees → les arbres partagent G sous-espaces
  distincts (chacun par ~n_trees/G arbres). G=0 → 1 sous-espace par arbre
  (= comportement précédent).

**Sweep arxiv 2M, dim 768, 256 trees, depth 20, NP=10 QD ∈ {16,20} :**

| variant     | QD=20 rec | QD=16 rec | latence QD=16 |
|-------------|-----------|-----------|---------------|
| ctrl        | 0.828     | 0.970     | ~30 ms        |
| ts128 g=16  | **0.901** | **0.989** | 32 ms         |
| ts128 g=32  | 0.873     | 0.985     | 31 ms         |
| ts64  g=16  | 0.902     | 0.988     | 33 ms         |
| ts64  g=32  | 0.896     | 0.982     | 32 ms         |
| ts32  g=16  | 0.905     | 0.974     | 51 ms (!)     |
| ts16  g=16 (id) | 0.884 | 0.967     | 41 ms         |

**Sweet spot consolidé : g=16, ts ∈ {64, 128}.** g=16 = 16 sous-espaces
renforcés > 256 indépendants bruités. Plancher empirique `ts ≥ 4×sub_dim`
— en dessous, splits trop peu discriminants, feuilles gonflent, latence
explose.

**Patch latent corrigé : `pick_dims` identity quand `full_dim ≤ sub_dim`.**
L'ancien sampling-avec-remise produisait ~38 % d'indices dupliqués à
full_dim=sub_dim=16 → splits collapse silencieusement. La branche identity
dans `src/traversal.c:40` est mathématiquement correcte (seul échantillonnage
sans remise possible quand pop ≤ taille échantillon). Test : ts=16 (id) gagne
+1.5 pt vs ts=16 (random) mais reste -2 pt sous champion ts=64 → l'identity
NE remplace PAS la diversité node-level apportée par pick_dims quand ts > sub_dim.

**Build : neutre à léger (~5-15 %)** — le gain gather est noyé par le profil
RNG-bound (78 %, cf [[feedback-build-rng-bound]] — pas créé encore).
Mémoïsation top-L=12 niveaux possible (~200 MB peak) pour ~1.5-2× build —
PAS prototypé (cf [[feedback-ram-1gb-hard]]).

**Caveat protocole** : 100 queries seulement, écarts <0.5 pt à QD=16 = bruit.
Signal QD=20 (+7 pt g16 vs ctrl) est robuste et reproductible (3 runs).

**Cohere_it 10M (2026-06-17) — NÉGATIF.** Rebuild dédié `cohere_it_10m_ts128`
avec ts=128 g=16, mêmes paramètres. Résultat : à QD=16 top_n=32k, ctrl =
**1.000 / 152 ms** vs ts128 = 0.993 / 163 ms (PERD 0.7 pt, COÛTE 11 ms).
Cause : baseline ctrl déjà au plafond 1.000 → aucune marge à récupérer ;
le gather des 128 dims/arbre ajoute juste du coût sans bénéfice.

**Règle généralisée : tree_sub n'aide QUE si baseline < ~0.99.** Sur arxiv
(ctrl=0.97) → +1.9 pt. Sur cohere_it (ctrl=1.000) → -0.7 pt. À considérer
comme "rescuer recall" pas comme "speedup universel". Sur SIFT et DEEP
probable même verdict (corpora "faciles" déjà saturés au baseline).

**Indexes test (working tree)** : `R&D/idx_arxiv_{ctrl,g6,g16,g32,g64,
ts128,g16_ts64,g32_ts64,g16_ts32,g16_ts16}`. Build ~3 min chacun.
