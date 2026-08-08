---
name: nq-beir-validation
description: "NQ BEIR sur Lightning — recall@10=0.81 nDCG@10=0.61 / 197 ms p50 = SOTA-comparable, valide l'hypothèse query-to-doc"
metadata: 
  node_type: memory
  type: project
  originSessionId: 0e757620-83dd-4e01-864c-56db0c985e00
---

Premier vrai bench query-to-doc avec qrels humains : NQ (Natural Questions, BEIR) sur Cohere v3 embeddings, 2.7M passages dim 1024, 3452 queries, qd=16/10p/32k/K'=100 single CPU sur Lightning AI (8 cœurs Xeon, 15 GB RAM). **Recall@10 = 0.8092, nDCG@10 = 0.6115, p50 = 197 ms.**

Cohere v3 lui-même rapporte nDCG@10 = 0.62 sur NQ exact → mangrove est **quasi-lossless vs exact** à scale 2.7M dim 1024.

**Why:** intuition utilisateur sur le doc-to-doc artificiellement dur (plafond 0.97 sur en2 inexplicable autrement) : "et si on était quasi 100% sur les vrais bench query-to-doc ?" → confirmé. Le 0.97 doc-to-doc n'est pas un problème pratique, c'est juste le stress test mathématique du voisinage exact.

**How to apply:**
- Pour publication : NQ est un des 3 BEIR-musts. Ajouter MS MARCO (8.8M) et HotpotQA (5.2M) idem facile sur Lightning.
- Datasets prêts : `/teamspace/studios/.../datasets/nq/` (11 GB corpus.fvecs + queries.fvecs + corpus_id_map.tsv + raw/qrels.parquet). Index `/teamspace/.../indexes/nq` (256t/depth20/sd16).
- `R&D/run_nq.py` = pipeline complet (prep + bench avec qrels + Recall@k/nDCG@k).
- TQ sidecar non encore appliqué sur NQ — gain attendu : 197→~120 ms si la même règle que cohere_it tient.
- **Bug bourde** : `Forest(IDX, depth=22)` ouvert au lieu de depth=20 du build → mangrove renvoie des slots aléatoires (~3% recall) silencieusement, AUCUNE erreur déclenchée. À fix en C (vérifier meta.txt depth contre l'arg).
