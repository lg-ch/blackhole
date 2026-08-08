---
name: wal-streaming-demo
description: WAL producer + consumer streaming build LIVRÉ 2026-07-15 — pattern pause+recalib validé end-to-end
metadata: 
  node_type: memory
  type: project
  originSessionId: 0e757620-83dd-4e01-864c-56db0c985e00
---

## Livré 2026-07-15

Pattern "producer → WAL disk queue → consumer streaming build avec pause/calib" validé end-to-end.

### Composants

**C — `--tail` flag pour `rpforest build`** (`build_tree.c:set_build_tail`) :
- Consumer lit le fichier séquentiellement
- Sur `fread` short (WAL pas encore rempli) : poll 500 ms, retry jusqu'à 5 min
- Compatible avec tous les formats (bvecs, fbin, u8bin, fvecs)

**C — Meta écrit early** (main.c) : `meta.txt` produit au tout début du build, avant Phase 1, pour que les calibrate subprocess puissent lire dim/depth/etc.

**C — Post-Phase-2 final calibrate** (main.c) : si `--recalib-doubling`, appel final au calibrator après conversion .srt.

**Python — `/tmp/wal_producer.py`** : append à un fichier bvecs au rate contrôlé (docs/sec), jamais bloqué par le consumer.

**Python — `scripts/mangrove_calibrate.py`** :
- Skip gracieux si meta.txt absent
- Skip gracieux si pas de `.srt` (Phase 1 en cours)
- Lecture correcte du format calib (header 8 B + queries + doc_ids)
- Écrit `recommended_config.json` atomique (tmp+rename)

### Demo mesuré (250k docs SIFT bvecs, 16 trees, depth 12)

- Producer 8k docs/s → 250k en 31.1 s (jamais bloqué)
- Consumer streaming --tail synchronisé, 31.4 s wall
- Recalib fires à 100k, 200k → skip gracieux (rc=0, "no .srt yet, Phase 1 in progress")
- Post-Phase-2 : final calibrate en 1.0 s écrit `recommended_config.json`

### Limitation restante (documentée)

Les recalib pendant Phase 1 restent des no-op (skip) car l'index n'est pas queryable tant que la Phase 2 (conversion `.srt`) n'a pas tourné. **Ça deviendra utile en LSM-segments** où chaque segment fait sa Phase 1 + Phase 2 rapidement.

### Why:

Story préprint "prod-ready streaming ANN" :
- Producer & consumer **découplés** via WAL disk queue
- Consumer peut pauser (pour calib, GC, backup) sans bloquer l'ingestion
- Consumer catch-up automatique sur reprise
- Config auto-tunée écrite à la fin du build, le serveur la lit au startup

### How to apply:

Prod déploiement :
```bash
# Producer (Kafka consumer, Kinesis, etc → WAL)
python3 /tmp/wal_producer.py source /mnt/mangrove/wal/live.bvecs --rate 10000

# Consumer (build + serve)
./rpforest build /mnt/mangrove/wal/live.bvecs /mnt/mangrove/indexes/live \
  256 28 --dim 128 --sub_dim 16 --gen v3 --no-varbyte \
  --tail --calib-queries 100 --recalib-doubling \
  --doc_count 1000000000
```

### Suite

- LSM segments (post-préprint) → recalib pendant Phase 1 devient utile
- WAL rotation (segments de N GB, delete des consommés) → pas critique tant que la RSS n'explose pas
