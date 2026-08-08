---
name: check-meta-depth
description: "TOUJOURS vérifier que depth/dim/sub_dim passés à Forest() correspondent au meta.txt du build, sinon recall silencieusement KO"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 0e757620-83dd-4e01-864c-56db0c985e00
---

Si on ouvre `Forest(IDX, depth=X, ...)` avec X ≠ depth du build, mangrove ne crash pas mais renvoie des slot indices aléatoires (~3% recall). AUCUN warning. A bouffé 2 reruns + 30 min de debug sur NQ Lightning.

**Why:** la lib lit les fichiers .srt en supposant que les bits d'offset/index correspondent à la depth fournie. Mismatch silencieux car les fichiers ont la bonne taille apparente.

**How to apply:**
- Toute Forest() doit récupérer depth/sub_dim/n_docs depuis `meta.txt` du index_dir, pas hardcoder.
- Patch C recommandé : Forest_open doit lire meta.txt et erreur si mismatch.
- En attendant : avant chaque bench, `cat /mnt/.../indexes/X/meta.txt` et copier exact dans le script.
- Symptôme : recall < 0.05 sur n'importe quel bench → vérifier depth en premier.
