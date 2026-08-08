---
name: project-srt4-disqualified
description: SRT4 (inline TQ codes per-leaf) disqualifié — duplication ×n_trees inacceptable au-delà de 10M docs.
metadata: 
  node_type: memory
  type: project
  originSessionId: 0e757620-83dd-4e01-864c-56db0c985e00
---

**Décision 2026-06-17 : ne PAS implémenter SRT4 inline codes.**

L'idée : co-loquer les codes TQ avec les doc_ids dans chaque leaf de chaque
tree*.srt, pour scorer pendant le K-way merge et éliminer le round I/O sidecar
TQ1.

**Le problème de design fatal** : chaque doc apparaît dans UNE feuille par
arbre, donc dans 256 leaves différentes (une par arbre). Co-locating les codes
dans chaque arbre = **×n_trees duplication sur disque**.

| corpus | sidecar TQ1 | × 256 trees |
|--------|-------------|--------------|
| arxiv 2M (dim 768) | 263 MB | 67 GB |
| cohere_it 10M (dim 1024) | 1.3 GB | **332 GB** |
| DEEP 1B (dim 96) | 16 GB | **4 TB** |

→ Sur SSD 7 TB, DEEP 1B SRT4 prendrait 57 % du disque juste pour les codes.
Inacceptable pour la cible 1B.

**Pourquoi la duplication est inévitable** : les arbres ont des partitionnements
de feuilles distincts par construction (c'est tout l'intérêt du RP-forest). Il
n'existe AUCUN layout disque qui satisfasse les 256 groupements simultanément.

**Gain effectif limité de toute façon** : TQ1 stage 1 actuel à 32 ms warm est
CPU-bound (pas I/O-bound). SRT4 économiserait ~5-10 ms warm (qsort cands +
buffer alloc), pas 32 ms. ROI catastrophique vs coût disque.

**Pivot recommandé pour accélérer la query** (sans toucher disque ni RAM) :
1. SIMD le scoring TQ1 (~20-25 ms gain, attaque les 32 ms stage 1 CPU)
2. Multi-thread per-tree leaf decode dans `forest_collect_topn_probes`
   (~40-60 ms gain, attaque les 80 ms K-way merge)
3. Pondération votes par profondeur (~10-30 ms, attaque les 7/8 GT topn_pruned
   identifiés dans le diag arxiv 2M).

Cumulé visé : 152 ms cohere_it 10M warm → ~50-60 ms warm.

Voir [[project-tq1]] et [[project-turboquant-rerank]] pour le contexte TQ.
