---
name: feedback-decide-dont-ask
description: "User n'aime pas les multi-questions en série quand le contexte est suffisant. Décider et avancer."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 0e757620-83dd-4e01-864c-56db0c985e00
---

**Règle (corrigée par l'user 2026-05-20)** : ne pas empiler les `AskUserQuestion` quand le contexte permet déjà de décider.

**Why** : L'user a interrompu une suite de 2 questions sur "next step" et "dataset source" alors que le contexte (ROADMAP_10D dit "delta encoding AVANT build 1B", il a déjà un `bigann_base.bvecs` 132 GB local) suffisait à décider. Il a répondu par "ok c'est bon pour le SSD tu as tout ce qu'il faut ?" = "arrête de demander, valide ce que tu as et continue."

**How to apply** :
- Une question = OK quand vraiment ambigu (action destructive, choix de format avec impact long terme, source externe inconnue).
- 2+ questions empilées = red flag : check si tu peux pas répondre toi-même avec ce que tu sais déjà du projet (memory + ROADMAP + ce que l'user a déjà dit).
- Sur les actions à blast radius local et réversible (où mettre un script, quel ordre d'étapes, etc.) : décide, fais, montre, l'user corrige si besoin.
- Sur les actions visibles externes / destructives / coûteuses en temps machine : confirmer reste OK.

Lien : [[user-preferences]] (I/O-economic thinker, déteste les sweeps redondants — corollaire : déteste aussi les questions redondantes).
