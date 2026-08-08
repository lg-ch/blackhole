---
name: omp-subprocess-bug
description: Ne jamais set OMP_NUM_THREADS=1 globalement dans un driver Python qui spawn rpforest — bug bête récurrent
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 0e757620-83dd-4e01-864c-56db0c985e00
---

## Règle

Dans un driver Python qui spawn `rpforest build` via subprocess, **NE PAS** faire :
```python
os.environ['OMP_NUM_THREADS'] = '1'    # au TOP du script
```

Cette variable est **héritée par le subprocess** → rpforest build tourne single-thread → **5-10× plus lent** que multi-core.

## Why:

Le build C utilise OMP par tree → normalement 20 threads → 6 min pour 10M docs à d=28. Avec OMP=1 hérité → 30 min. Silent overhead, pas d'erreur.

3 fois répétée dans une session (2026-07-15) : SIFT 10M V2 vs V3 build, LSM test build, LSM v2 test build. Chaque fois j'ai perdu ~20-60 min à cause de ça.

## How to apply:

Pour un driver Python qui BUILD + QUERY sur SIFT/DEEP :
```python
# NE PAS faire ceci au top :
# os.environ['OMP_NUM_THREADS'] = '1'

# À la place, set OMP=1 UNIQUEMENT dans la fonction bench :
def bench(...):
    os.environ['OMP_NUM_THREADS'] = '1'
    os.sched_setaffinity(0, {0})
    # ... query ...
```

Le build subprocess hérite du parent env AU MOMENT du spawn. Si OMP=1 pas set à ce moment → build parallèle. Puis on set OMP=1 avant le query pour comparaison fair single-thread.

## Alternative

Passer explicitement l'env au subprocess :
```python
env = os.environ.copy()
env.pop('OMP_NUM_THREADS', None)   # ensure multi-thread
subprocess.run([...], env=env)
```
