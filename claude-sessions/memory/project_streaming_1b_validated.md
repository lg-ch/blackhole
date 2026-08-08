---
name: streaming-1b-validated
description: "Streaming SIFT 1B 256 trees + disk-HOT per-leaf : +3.3% mean / -1.3% p99 mesuré 2026-07-17"
metadata: 
  node_type: memory
  type: project
  originSessionId: 0e757620-83dd-4e01-864c-56db0c985e00
---

## Résultat validé

**SIFT 1B / 256 trees / 1B docs / disk-backed HOT per-leaf + 1k vec/s ingest streaming concurrent** :

| config | mean | p50 | p75 | p95 | p99 |
|--------|-----:|----:|----:|----:|----:|
| baseline idle | 428 | 420 | 460 | 506 | 527 |
| stress live | 443 | 444 | 474 | 492 | 520 |

**Delta : +3.3 % mean, -1.3 % p99** = dans le bruit total.

HOT steady state après ~14 s :
- 3.6 M docs
- **50 MB RAM index**, **57 MB disk slots**

Params : sweet_spot_1gb calibré (top_paths=1024, mlb=200 000, top_n=6000, qd=26)

## Pourquoi le disk-HOT scale mieux à grand corpus

Overhead HOT = quasi-fixe en absolu (~5-10 ms pour preads + mutex).
Ratio relatif : `overhead / baseline`.
- Sur 1B baseline 428 ms → 10 ms / 428 = **2 %**
- Sur 10M baseline 43 ms → 4 ms / 43 = **9 %**

**Plus l'index MAIN est gros, plus le disk-HOT est indolore.** Inverse de l'intuition initiale.

## Détails architecture livrée

- `src/hot_store.{h,c}` : disk-backed per-leaf slot allocator (4 classes power-of-2 : 4/16/64/256 docs par slot)
- Fichier par tree : `<hot_dir>/treeXXXXX.hot`
- RAM index : sorted array de `HotEntry {leaf_id, packed(class+slot), n_docs}` — 12 B par leaf
- Query merge : mutex tree lookup → snapshot `(disk_off, n_docs)` → lock-free `pread` → pack dans radix
- Append : mutex → alloc slot (promotion class si grow) → pwrite → update RAM index
- Compatible avec `hot_compact_tree_v3` (SnapEntry → SRT V3 varbyte), atomic rename + fd swap

## À faire pour vraie prod

1. ~~Recovery du RAM index depuis .hot file au reboot~~ **LIVRÉ 2026-07-17** : slot headers 8 B + tombstones + `hot_recover_tree` au reopen
2. ~~Bg compaction thread stress-testé avec disk HOT~~ **LIVRÉ** : bench 3 min = 238 compactions, HOT drain continu
3. ~~WAL append-only pour crash durability~~ **LIVRÉ** : `hot_wal.{h,c}` buffered + background fsync 100 ms + replay validé (256 records après crash simulé)
4. ~~Test extended pour valider fragmentation slots + memory leak absence~~ **LIVRÉ 120s** : RAM oscille 74-102 MB en steady state, RSS 114→182 MB, query stable 16-19 ms

**Checkpoint restant : WAL truncation** après compaction globale (WAL grow linéaire dans MVP, pas critique tant que compactions rendent le disk .hot). À implémenter avec un compteur "records depuis dernier compact" au niveau global overlay.

## HOT reads batched io_uring (livré 2026-07-17)

Le MVP initial utilisait `pread` bloquant pour lire chaque HOT slot depuis disque dans le pack loop. À shallow qd (14-18) avec HOT peuplé, la math séquentielle donnait 500 ms+ d'overhead — non-viable.

**Refactor** : query_tree.c pack loop restructuré en 3 phases :
1. MAIN pack (inchangé)
2. HOT collect ops sous lock (snapshot disk_off, n_docs, fd par entry)
3. HOT submit batché via `f->ring` io_uring + reap + pack dans pairs[]

**Mesure arxiv 2M 256 trees, 4M HOT docs, cold** :

| qd | idle | + HOT | delta |
|----|-----:|------:|------:|
| 18 | 36 ms | 37 ms | +3 % |
| 16 | 42 ms | 43 ms | +2 % |
| 14 | 48 ms | 51 ms | +6 % |

Le batch io_uring exploite le queue depth NVMe (~128-256) → wall time ≈ `N_reads / QD × ~200 µs`, pas `N_reads × 75 µs`. Ratio ~25× vs séquentiel.

**Overhead reste borné même à shallow qd + HOT peuplé.** Le design tient face au scénario "build unifié depth=28 pour tous les corpora".

## Bench arxiv d=28 256 trees — régime "build unifié"

Build : arxiv 2M en depth=28, 256 trees, dim=768, sub_dim=16, gen v3. Wall 4 min, disk 5.6 GB, peak RSS build 37 MB.

Query cold (drop_caches per query, cgroup 1G) :

| qd | k_shift | subtree | idle | + 500k HOT | + 4M HOT |
|---:|---:|---:|---:|---:|---:|
| 20 | 8 | 256 | 28 ms | 28 ms | 28 ms |
| 16 | 12 | 4096 | 38 ms | 39 ms | 40 ms |
| 14 | 14 | 16384 | 47 ms | 48 ms | 50 ms |

**Décision architecture : build unifié depth=28 pour tous corpora** est prod-safe. À qd=14 sur small corpus + HOT streaming réaliste (500k), overhead +2 % mean / +4 % p99. Sous stress heavy (4M HOT = 200 % du MAIN), +6 % / +10 %. Rendu possible par le batch io_uring HOT reads livré aujourd'hui.

Le design est **prêt pour prod streaming 1B sous 1 GB RAM**.
