---
name: feedback-n-trees-sweep
description: "Réduire n_trees comme levier perf : ne pas proposer, déjà testé hors session, recall chute sous 1000 trees"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 0e757620-83dd-4e01-864c-56db0c985e00
---

**Réduire `n_trees` sous 1000 fait chuter le recall** — vérifié à 1M ET à 100M.

**Why :** 
- SIFT 1M : balayé hors session n_trees ∈ {256..1024}, config gagnante 1000×d18×sub16 v3 = 0.9977 recall.
- SIFT 100M (2026-05-20, cette session) : tenté 200×d25×sub16 = 0.78 recall avec votes dilués à 3/200. Vote concentration impossible quand on a peu d'arbres + leaves heavy-tail.
- Pas un effet 1M-spécifique. Levier de votes/discrimination, indépendant de l'échelle.

**How to apply :** 
- Toujours partir de n_trees ≥ 1000 pour atteindre recall production. Garder 1024 comme rond de feuille.
- Ne pas proposer "moins d'arbres" comme levier latence. Leviers query restants : compression varbyte (déjà fait SRT3), drive plus rapide, batch cross-query si autorisé.
- Si build trop long avec 1000 trees, le bon trade-off est de réduire `depth` (vote concentration directe via leaves plus larges) pas `n_trees`.

**Coût build mesuré (2026-05-20)** : SIFT 100M / 200×d25 ≈ 3.6h, donc 1000×d22 ≈ 16h. SIFT 1B / 1000×d22 ≈ 7 jours linéaire — au-dessus du budget 10j si on veut aussi du bench et de la marge. Multi-index N×100M plus pratique.

**MAJ 2026-05-30 (session compression, sweep réel `R&D/idx1m_d18` vs `idx1m_d26`)** : confirmé avec chiffres disque. Pousser la build-depth PLUS profonde est strictement PIRE sur le front disque/recall (depth26 coûte 2×/arbre ET plafonne à 0.93 vs depth18 atteint 0.98 à 561 MB) — cohérent avec la ligne "réduire depth, pas l'augmenter" ci-dessus. Réduire n_trees EST possible à recall comparable MAIS uniquement via query_depth↓ (scope élargi) : 128t depth18 qd10 = 0.932 @ 280 MB vs 256t qd15 = 0.972 @ 561 MB → ~2× moins disque, 2.8× plus lent. C'est un troc disque↔latence non-scalable à 1B (qd↓ explose les candidats ∝ N/2^qd). Donc "moins d'arbres" reste un mauvais levier à 1B, mais le mécanisme est la latence, pas une impossibilité de recall. Détail : [[project_compression_reorder_ceiling]], `R&D/compression_findings.md` Fait 5.

Lien : [[project-rpforest-sift]], [[feedback-depth-vote-discrimination]].
