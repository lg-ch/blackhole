---
name: max-leaf-bytes-lever
description: max_leaf_bytes est LE lever pour la RAM-bounded scaling à shallow depth — cap le pool ET améliore le recall
metadata: 
  node_type: memory
  type: project
  originSessionId: 0e757620-83dd-4e01-864c-56db0c985e00
---

Breakthrough 2026-07-08 pendant debug DEEP 100M d=14 : le "leak" apparent sous cgroup 1G n'était pas un leak au sens strict mais **le pool grow-only qui suivait la query worst-case**. Certaines queries à shallow depth (d=14 sur 100M) touchent des méga-leaves (30k+ docs varbyte 500 KB) faisant exploser radix_pairs + radix_scratch + topn_h à 900 MB+.

**Fix** : `set_max_leaf_bytes(200_000)` — skip les leaves > 200 KB en varbyte encoding pendant Phase 2.

**Effet triple** :
1. **Cap peak RSS** : DEEP 100M 400 paths passe de 878 MB → 811 MB (fit <1G)
2. **Améliore le recall** : 0.977 → 0.980 sur config 600×2000 (skip les hubs géométriques qui polluent le vote)
3. **Latence -33 à -40%** : 252 ms → 169 ms

**Why:** Les mega-leaves ne sont PAS les vrais voisins — ce sont des régions denses de l'espace (hubs) où beaucoup de docs se rassemblent par géométrie mais pas par similarité. Les rejeter améliore le signal de vote.

**How to apply** :
- Pour shallow-depth (d=12-14) sur 10M-1B : cap à 200 KB par défaut
- Pour deeper (d>18) : moins critique car leaves plus petites naturellement
- Sweet spot empirique DEEP 100M d=14 : mlb=200_000
- Attention : trop bas (mlb<50k) : recall chute (0.977 → 0.83)

**Baseline établi DEEP 100M sous 1 GB RSS strict** :
- 400×4000 mlb=200k : recall 0.960 / 280 ms COLD
- 400×2000 mlb=200k : recall 0.937 / 223 ms COLD

Voir aussi [[feedback_ram_1gb_hard]] pour la règle projet.
