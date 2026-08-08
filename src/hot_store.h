#ifndef HOT_STORE_H
#define HOT_STORE_H

/* HOT overlay disk-backed per-leaf.
 *
 * Design :
 *   - Per tree : one .hot file (append-only + slot-alloc, growable)
 *   - Per tree : RAM sparse index sorted by leaf_id (~12 B per delta leaf)
 *   - Query    : RAM lookup (few ns) → if hit, pread slot data (~75 µs cold)
 *
 * Slot allocation : 4 power-of-2 classes (4/16/64/256 docs = 16/64/256/1024 B).
 * When a leaf's doc count exceeds current class, promote to next class and
 * abandon old slot (compaction reclaims).
 *
 * Thread-safety : per-tree mutex protects the RAM index + file writes.
 * Query readers take lock ONLY to snapshot (leaf_id, disk_off, n_docs),
 * then pread lock-free — append-only means old slot data stays valid.
 *
 * Persistence : the RAM index is rebuilt from the .hot file on open (linear
 * scan of leaf headers). File is durable ; index is derived.
 *
 * Compaction (existing) : hot_snapshot_and_clear moves the delta into a
 * fresh SRT via hot_compact. .hot file is truncated after successful merge.
 */

#include <stdint.h>
#include <stddef.h>
#include <pthread.h>

#define HOT_N_CLASSES 4
#define HOT_MAGIC 0x31544f48u    /* "HOT1" — v1 slot header (leaf_id + n_docs) */
#define HOT_SLOT_HEADER_BYTES 8u  /* [4 B leaf_id][4 B n_docs] prefix per slot */
#define HOT_TOMBSTONE_LEAF    0xFFFFFFFFu

/* Class capacities in DOCS (payload). Slot bytes on disk = docs*4 + 8 header. */
static const uint32_t HOT_CLASS_DOCS[HOT_N_CLASSES] = { 4, 16, 64, 256 };

/* Per-leaf entry in RAM (12 B packed). */
typedef struct __attribute__((packed)) {
    uint32_t leaf_id;
    uint32_t n_docs;       /* actual docs in slot                */
    uint32_t packed;       /* upper 4 bits = class, lower 28 = slot_id */
} HotEntry;

#define HOT_CLASS(packed)   (((packed) >> 28) & 0xFu)
#define HOT_SLOT(packed)    ((packed) & 0x0FFFFFFFu)
#define HOT_PACK(cls, slot) ((((uint32_t)(cls) & 0xFu) << 28) | ((uint32_t)(slot) & 0x0FFFFFFFu))

typedef struct {
    int             n_entries;
    int             cap;
    HotEntry*       entries;          /* sorted asc by leaf_id */
    pthread_mutex_t mu;
    int             hot_fd;           /* .hot file, O_RDWR         */
    uint32_t        class_n_slots[HOT_N_CLASSES];  /* next slot id per class */
    uint64_t        class_data_off[HOT_N_CLASSES]; /* start of class data on disk */
    uint64_t        file_size;         /* current end-of-file offset */
} HotTree;

typedef struct {
    int       n_trees;
    int       depth;
    HotTree*  trees;
    uint64_t  total_docs;   /* atomic stat */
    char      dir[512];     /* HOT files live here */
} HotOverlay;

/* Init : creates one .hot per tree at dir/treeXXXXX.hot (or reopens & rebuilds
 * RAM index if file already exists). On reopen : full recovery scan (reads
 * slot headers to reconstruct the sorted RAM index). Tombstoned slots
 * (leaf_id = HOT_TOMBSTONE_LEAF) are skipped. Returns total docs recovered. */
int  hot_init  (HotOverlay* h, int n_trees, int depth, const char* dir);
void hot_free  (HotOverlay* h);

/* Explicit recovery of one tree from disk. Called internally by hot_init on
 * reopen ; exposed for tests. */
int  hot_recover_tree(HotOverlay* h, int tree_id);

/* Append doc_id under (tree_id, leaf_id). Takes tree mutex, writes to disk,
 * updates RAM index. Promotes slot class if leaf grows past current cap. */
int  hot_append(HotOverlay* h, int tree_id, uint32_t leaf_id, uint32_t doc_id);

/* Read the docs for a (tree_id, leaf_id). Returns 1 if HOT has data, 0 else.
 * Copies up to *out_n docs into out_buf. Lock-free after snapshot. */
int  hot_read  (const HotOverlay* h, int tree_id, uint32_t leaf_id,
                uint32_t* out_buf, uint32_t buf_cap, uint32_t* out_n);

/* Range lookup for subtree expansion. Returns index range [i_lo, i_hi) into
 * the tree's entries array. Caller MUST hold tree lock while iterating +
 * calling hot_read_entry. */
void hot_range (const HotOverlay* h, int tree_id,
                uint32_t low_leaf, uint32_t high_leaf,
                int* out_i_lo, int* out_i_hi);

/* Read the docs for entry index i (from prior hot_range). Snapshot then
 * lock-free pread. */
int  hot_read_entry(const HotOverlay* h, int tree_id, int entry_idx,
                    uint32_t* out_buf, uint32_t buf_cap, uint32_t* out_n);

/* Snapshot (disk_off, n_docs, fd) for entry i under tree lock. Used by
 * batched query paths that submit io_uring reads themselves.
 * Returns 0 on success, -1 if idx OOB. Also returns the tree's file
 * descriptor for use with io_uring. */
int  hot_snapshot_entry(const HotOverlay* h, int tree_id, int entry_idx,
                        uint64_t* out_off, uint32_t* out_n_docs, int* out_fd);

void hot_tree_lock  (const HotOverlay* h, int tree_id);
void hot_tree_unlock(const HotOverlay* h, int tree_id);

uint64_t hot_ram_bytes (const HotOverlay* h);
uint64_t hot_disk_bytes(const HotOverlay* h);

/* Snapshot per-tree entries + drain (used by compaction). Returns array of
 * (leaf_id, doc_ids...) fetched from disk into RAM. Post-snapshot, tree HOT
 * is EMPTY (both RAM index and disk file truncated to header only). */
typedef struct {
    uint32_t leaf_id;
    uint32_t n_docs;
    uint32_t* docs;   /* malloc'd copy */
} HotSnapEntry;

int  hot_snapshot_and_clear(HotOverlay* h, int tree_id,
                            HotSnapEntry** out_snap, int* out_n);
void hot_snapshot_free    (HotSnapEntry* snap, int n);

#endif
