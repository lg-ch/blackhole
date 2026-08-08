---
name: reference-ssd-layout
description: Crucial X10 8 TB ext4 monté sur /mnt/mangrove. Layout datasets/indexes/wal/scratch. UUID + benchmarks.
metadata: 
  node_type: memory
  type: reference
  originSessionId: 0e757620-83dd-4e01-864c-56db0c985e00
---

**SSD prod** : Crucial X10 USB, 8 TB physiques (7.3 TB ext4 utilisables).

**Mount** : `/mnt/mangrove`
- Device : `/dev/sdb2`
- UUID : `d8240adf-acae-4705-b789-faf07b91cd6b`
- Label : `mangrove`
- Filesystem : ext4 (formaté 2026-05-20 par mangrove-search)
- Owner : `chatelet:chatelet`

**Layout** :
- `/mnt/mangrove/datasets/` — fichiers bruts (SIFT 1B `base.u8bin`, queries, GT, embeddings text)
- `/mnt/mangrove/indexes/` — forests buildés (`.srt`, `meta.txt`)
- `/mnt/mangrove/wal/` — WAL des mutations P1 (insert/delete)
- `/mnt/mangrove/scratch/` — temporaires (delta encoding, compaction, bench buffers)

**Perfs mesurées (2026-05-20)** :
- Write direct (O_DIRECT) : **1.1 GB/s**
- Read froid : **1.5 GB/s**
- Read cache chaud : 7.5 GB/s (cache page kernel)

**fstab** : pas encore ajouté. Si besoin de mount auto au reboot :
```
UUID=d8240adf-acae-4705-b789-faf07b91cd6b /mnt/mangrove ext4 defaults,nofail 0 2
```

**Capacité prévue** :
- SIFT 1B `base.u8bin` : 128 GB
- Indexes .srt 1B avec delta encoding (~2-3× compression) : 1.5-3 TB
- Marge confortable : 4-5 TB libres après build complet.

Lien : [[project-roadmap-prod]] (build 1B prévu), [[feedback-no-mmap]] (io_uring sur ce SSD, pas mmap).
