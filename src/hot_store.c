/* HOT overlay disk-backed per-leaf slot allocator. */
#define _POSIX_C_SOURCE 200809L
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "hot_store.h"

/* File layout : ONE file per (tree, class) — tree%05d.c%d.hot :
 *   [4 B magic]
 *   [4 B depth]
 *   [4 B n_classes]
 *   [4 B reserved]
 *   [HOT_N_CLASSES × 4 B class_docs_cap]  (docs per slot per class)
 *   [HOT_N_CLASSES × 4 B class_n_slots]   (only entry [c] meaningful here)
 *   [HOT_N_CLASSES × 8 B legacy zeros]
 *   [slot 0 | slot 1 | ...]               (this class's slots, contiguous)
 *
 * Slot k of class c lives at HOT_HDR_TOTAL + k × class_bytes(c) in ITS OWN
 * file — contiguous by construction. (The previous single-file layout
 * computed offsets as class_data_off[c] + k × class_bytes(c), which is only
 * valid while a class's region is contiguous ; as soon as a promotion
 * interleaved allocations across classes, a class 0 slot silently
 * overwrote the start of the class 1 region. One file per class removes
 * the collision by design.)
 *
 * The RAM index is a sorted array of HotEntry, one per non-empty leaf.
 * Rebuilt from disk on hot_init if the files already exist (linear scan
 * of allocated slots — we don't store a persistent index yet; MVP relies
 * on WAL replay from the source of truth in production).
 *
 * Post-snapshot-and-clear, each class file is truncated to HEADER_SIZE
 * (dropping all slots, class_n_slots reset to 0).
 */

#define HOT_HEADER_BYTES  16
/* header : [magic, depth, n_classes, reserved]  = 16 B
 * classes:
 *   [class_docs_cap × HOT_N_CLASSES × 4]      = 4 * n_c B
 *   [class_n_slots  × HOT_N_CLASSES × 4]      = 4 * n_c B
 *   [class_data_off × HOT_N_CLASSES × 8]      = 8 * n_c B
 */
#define HOT_HDR_TOTAL     (HOT_HEADER_BYTES + HOT_N_CLASSES * 4 \
                                            + HOT_N_CLASSES * 4 \
                                            + HOT_N_CLASSES * 8)
/* Byte offset of class_data_off array in the file header block. */
#define HOT_CLASS_OFF_ARRAY_OFF (HOT_HEADER_BYTES + HOT_N_CLASSES * 4 + HOT_N_CLASSES * 4)
#define HOT_CLASS_N_SLOTS_OFF   (HOT_HEADER_BYTES + HOT_N_CLASSES * 4)

/* Total bytes per slot on disk : 8 B header + payload docs*4. */
static uint32_t class_bytes(int c) { return HOT_SLOT_HEADER_BYTES + HOT_CLASS_DOCS[c] * 4u; }

static int write_header(int fd, uint32_t depth) {
    uint32_t hdr[4] = { HOT_MAGIC, depth, HOT_N_CLASSES, 0 };
    if (pwrite(fd, hdr, sizeof(hdr), 0) != (ssize_t)sizeof(hdr)) return -1;
    uint32_t cap_arr[HOT_N_CLASSES];
    for (int c = 0; c < HOT_N_CLASSES; c++) cap_arr[c] = HOT_CLASS_DOCS[c];
    if (pwrite(fd, cap_arr, sizeof(cap_arr), HOT_HEADER_BYTES)
        != (ssize_t)sizeof(cap_arr)) return -1;
    /* class_n_slots = zeros initially */
    uint32_t zero[HOT_N_CLASSES] = {0};
    if (pwrite(fd, zero, sizeof(zero), HOT_HEADER_BYTES + HOT_N_CLASSES * 4)
        != (ssize_t)sizeof(zero)) return -1;
    return 0;
}

int hot_init(HotOverlay* h, int n_trees, int depth, const char* dir) {
    if (!h || n_trees <= 0 || !dir) return -1;
    memset(h, 0, sizeof(*h));
    h->n_trees = n_trees;
    h->depth = depth;
    strncpy(h->dir, dir, sizeof(h->dir) - 1);
    h->trees = (HotTree*)calloc((size_t)n_trees, sizeof(HotTree));
    if (!h->trees) return -1;
    /* Pre-mark every fd invalid so hot_free on a partial init is safe. */
    for (int t = 0; t < n_trees; t++)
        for (int c = 0; c < HOT_N_CLASSES; c++) h->trees[t].fds[c] = -1;
    mkdir(dir, 0755);
    /* Legacy single-file layout : not migrated (HOT is transient — WAL is
       the source of truth). Warn once so a stale overlay isn't silently
       ignored. */
    {
        char legacy[600];
        snprintf(legacy, sizeof(legacy), "%s/tree%05d.hot", dir, 0);
        struct stat lst;
        if (stat(legacy, &lst) == 0 && lst.st_size > HOT_HDR_TOTAL) {
            fprintf(stderr, "hot_init: legacy single-file HOT found in %s — "
                            "ignored (replay the WAL to repopulate)\n", dir);
        }
    }
    for (int t = 0; t < n_trees; t++) {
        HotTree* ht = &h->trees[t];
        pthread_mutex_init(&ht->mu, NULL);
        int any_existing = 0;
        for (int c = 0; c < HOT_N_CLASSES; c++) {
            char path[600];
            snprintf(path, sizeof(path), "%s/tree%05d.c%d.hot", dir, t, c);
            struct stat st;
            int exists = (stat(path, &st) == 0 && st.st_size >= HOT_HDR_TOTAL);
            ht->fds[c] = open(path, O_RDWR | O_CREAT, 0644);
            if (ht->fds[c] < 0) {
                fprintf(stderr, "hot_init: open %s failed\n", path);
                hot_free(h); return -1;
            }
            if (!exists) {
                if (ftruncate(ht->fds[c], HOT_HDR_TOTAL) != 0 ||
                    write_header(ht->fds[c], (uint32_t)depth) != 0) {
                    hot_free(h); return -1;
                }
                ht->file_size[c] = HOT_HDR_TOTAL;
            } else {
                uint32_t hdr[4];
                if (pread(ht->fds[c], hdr, sizeof(hdr), 0) != (ssize_t)sizeof(hdr)
                    || hdr[0] != HOT_MAGIC) {
                    fprintf(stderr, "hot_init: bad magic in %s (treating as empty)\n", path);
                }
                uint32_t ns[HOT_N_CLASSES];
                if (pread(ht->fds[c], ns, HOT_N_CLASSES * 4, HOT_CLASS_N_SLOTS_OFF)
                    == (ssize_t)(HOT_N_CLASSES * 4)) {
                    ht->class_n_slots[c] = ns[c];
                }
                ht->file_size[c] = (uint64_t)st.st_size;
                any_existing = 1;
            }
        }
        if (any_existing && hot_recover_tree(h, t) < 0) {
            fprintf(stderr, "hot_init: recover tree %d failed\n", t);
            hot_free(h); return -1;
        }
    }
    return 0;
}

void hot_free(HotOverlay* h) {
    if (!h || !h->trees) return;
    for (int t = 0; t < h->n_trees; t++) {
        HotTree* ht = &h->trees[t];
        if (ht->entries) free(ht->entries);
        for (int c = 0; c < HOT_N_CLASSES; c++) {
            if (ht->fds[c] >= 0) close(ht->fds[c]);
        }
        pthread_mutex_destroy(&ht->mu);
    }
    free(h->trees);
    h->trees = NULL;
    h->n_trees = 0;
}

void hot_tree_lock(const HotOverlay* h, int tree_id) {
    if (!h || tree_id < 0 || tree_id >= h->n_trees) return;
    pthread_mutex_lock(&((HotOverlay*)h)->trees[tree_id].mu);
}
void hot_tree_unlock(const HotOverlay* h, int tree_id) {
    if (!h || tree_id < 0 || tree_id >= h->n_trees) return;
    pthread_mutex_unlock(&((HotOverlay*)h)->trees[tree_id].mu);
}

static int hot_lower_bound(const HotTree* ht, uint32_t leaf_id) {
    int lo = 0, hi = ht->n_entries;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (ht->entries[mid].leaf_id < leaf_id) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

/* Byte offset of a slot inside ITS CLASS FILE — contiguous by design. */
static uint64_t slot_offset(const HotTree* ht, int cls, uint32_t slot) {
    (void)ht;
    return HOT_HDR_TOTAL + (uint64_t)slot * class_bytes(cls);
}

/* Persist one class's slot count in that class's file. */
static void persist_class_n_slots(HotTree* ht, int cls) {
    pwrite(ht->fds[cls], &ht->class_n_slots[cls], 4,
           (off_t)(HOT_CLASS_N_SLOTS_OFF + cls * 4));
}

/* Allocate new slot in class c : grows that class's own file. */
static uint32_t alloc_slot(HotTree* ht, int cls) {
    uint32_t slot = ht->class_n_slots[cls];
    ht->class_n_slots[cls]++;
    persist_class_n_slots(ht, cls);
    uint64_t new_end = HOT_HDR_TOTAL
                     + (uint64_t)ht->class_n_slots[cls] * class_bytes(cls);
    if (new_end > ht->file_size[cls]) {
        if (ftruncate(ht->fds[cls], new_end) != 0) { /* soft-fail */ }
        ht->file_size[cls] = new_end;
    }
    return slot;
}

/* Write slot header [leaf_id, n_docs] at slot start. */
static int write_slot_header(int fd, uint64_t slot_off, uint32_t leaf_id, uint32_t n_docs) {
    uint32_t hdr[2] = { leaf_id, n_docs };
    return (pwrite(fd, hdr, 8, (off_t)slot_off) == 8) ? 0 : -1;
}

/* Byte offset of docs payload within a slot (skip 8 B header). */
static uint64_t slot_payload_off(const HotTree* ht, int cls, uint32_t slot) {
    return slot_offset(ht, cls, slot) + HOT_SLOT_HEADER_BYTES;
}

int hot_append(HotOverlay* h, int tree_id, uint32_t leaf_id, uint32_t doc_id) {
    if (!h || tree_id < 0 || tree_id >= h->n_trees) return -1;
    HotTree* ht = &h->trees[tree_id];
    pthread_mutex_lock(&ht->mu);

    int rc = -1;
    int i = hot_lower_bound(ht, leaf_id);
    if (i < ht->n_entries && ht->entries[i].leaf_id == leaf_id) {
        HotEntry* e = &ht->entries[i];
        int cur_cls = (int)HOT_CLASS(e->packed);
        uint32_t cur_slot = HOT_SLOT(e->packed);
        uint32_t cur_cap = HOT_CLASS_DOCS[cur_cls];
        if (e->n_docs < cur_cap) {
            /* Room in current slot : append doc at position n_docs. */
            uint64_t doc_off = slot_payload_off(ht, cur_cls, cur_slot)
                             + (uint64_t)e->n_docs * 4u;
            if (pwrite(ht->fds[cur_cls], &doc_id, 4, (off_t)doc_off) != 4) goto out;
            /* Update n_docs in slot header. */
            uint32_t new_n = e->n_docs + 1;
            if (pwrite(ht->fds[cur_cls], &new_n, 4,
                       (off_t)(slot_offset(ht, cur_cls, cur_slot) + 4)) != 4) goto out;
            e->n_docs = new_n;
            rc = 0;
        } else if (cur_cls + 1 < HOT_N_CLASSES) {
            /* Promote to next class : allocate new slot, copy old + new. */
            int new_cls = cur_cls + 1;
            uint32_t new_slot = alloc_slot(ht, new_cls);
            uint64_t new_off  = slot_offset(ht, new_cls, new_slot);
            uint64_t old_off  = slot_offset(ht, cur_cls, cur_slot);
            uint32_t buf[256 + 2];   /* header + up to 256 docs */
            uint32_t copy_bytes = e->n_docs * 4u;
            if (pread(ht->fds[cur_cls], buf, copy_bytes,
                      (off_t)(old_off + HOT_SLOT_HEADER_BYTES))
                != (ssize_t)copy_bytes) goto out;
            buf[e->n_docs] = doc_id;
            uint32_t new_n = e->n_docs + 1;
            uint32_t hdr[2] = { leaf_id, new_n };
            if (pwrite(ht->fds[new_cls], hdr, 8, (off_t)new_off) != 8) goto out;
            if (pwrite(ht->fds[new_cls], buf, copy_bytes + 4,
                       (off_t)(new_off + HOT_SLOT_HEADER_BYTES))
                != (ssize_t)(copy_bytes + 4)) goto out;
            /* Mark old slot as tombstone (leaf_id=0xFFFFFFFF, n_docs=0). */
            write_slot_header(ht->fds[cur_cls], old_off, HOT_TOMBSTONE_LEAF, 0);
            e->packed = HOT_PACK(new_cls, new_slot);
            e->n_docs = new_n;
            rc = 0;
        } else {
            rc = -2;   /* max class ; caller must trigger compaction */
        }
    } else {
        /* New leaf : alloc slot in class 0 + insert entry. */
        if (ht->n_entries == ht->cap) {
            int new_cap = ht->cap ? ht->cap * 2 : 16;
            HotEntry* ne = (HotEntry*)realloc(ht->entries, (size_t)new_cap * sizeof(HotEntry));
            if (!ne) goto out;
            ht->entries = ne;
            ht->cap = new_cap;
        }
        uint32_t slot = alloc_slot(ht, 0);
        uint64_t off = slot_offset(ht, 0, slot);
        uint32_t hdr[2] = { leaf_id, 1u };
        if (pwrite(ht->fds[0], hdr, 8, (off_t)off) != 8) goto out;
        if (pwrite(ht->fds[0], &doc_id, 4, (off_t)(off + HOT_SLOT_HEADER_BYTES)) != 4) goto out;
        if (i < ht->n_entries) {
            memmove(&ht->entries[i+1], &ht->entries[i],
                    (size_t)(ht->n_entries - i) * sizeof(HotEntry));
        }
        ht->entries[i].leaf_id = leaf_id;
        ht->entries[i].n_docs = 1;
        ht->entries[i].packed = HOT_PACK(0, slot);
        ht->n_entries++;
        rc = 0;
    }
    if (rc == 0) h->total_docs++;
out:
    pthread_mutex_unlock(&ht->mu);
    return rc;
}

/* Sort entries by leaf_id after recovery scan (recovery reads slots in
 * class + slot_id order, not leaf_id order). qsort simple ; done once at boot. */
static int entry_leaf_cmp(const void* a, const void* b) {
    uint32_t la = ((const HotEntry*)a)->leaf_id;
    uint32_t lb = ((const HotEntry*)b)->leaf_id;
    return (la > lb) - (la < lb);
}

/* Rebuild RAM index for a tree by scanning slot headers on disk. */
int hot_recover_tree(HotOverlay* h, int tree_id) {
    if (!h || tree_id < 0 || tree_id >= h->n_trees) return -1;
    HotTree* ht = &h->trees[tree_id];
    pthread_mutex_lock(&ht->mu);
    /* class_n_slots already loaded from disk by hot_init. */

    uint32_t total_slots = 0;
    for (int c = 0; c < HOT_N_CLASSES; c++) total_slots += ht->class_n_slots[c];

    if (total_slots == 0) {
        pthread_mutex_unlock(&ht->mu);
        return 0;
    }

    HotEntry* entries = (HotEntry*)malloc((size_t)total_slots * sizeof(HotEntry));
    if (!entries) { pthread_mutex_unlock(&ht->mu); return -1; }

    int recovered = 0;
    uint64_t docs_recovered = 0;
    for (int c = 0; c < HOT_N_CLASSES; c++) {
        for (uint32_t s = 0; s < ht->class_n_slots[c]; s++) {
            uint64_t off = slot_offset(ht, c, s);
            uint32_t hdr[2];
            if (pread(ht->fds[c], hdr, 8, (off_t)off) != 8) continue;
            uint32_t lid = hdr[0], nd = hdr[1];
            if (lid == HOT_TOMBSTONE_LEAF) continue;   /* abandoned */
            if (nd == 0 || nd > HOT_CLASS_DOCS[c])   continue;  /* corrupt */
            entries[recovered].leaf_id = lid;
            entries[recovered].n_docs  = nd;
            entries[recovered].packed  = HOT_PACK(c, s);
            recovered++;
            docs_recovered += nd;
        }
    }
    /* Sort by leaf_id. Dedup keeping HIGHEST n_docs entry (latest promotion). */
    qsort(entries, (size_t)recovered, sizeof(HotEntry), entry_leaf_cmp);
    int uniq = 0;
    for (int i = 0; i < recovered; i++) {
        if (uniq > 0 && entries[uniq-1].leaf_id == entries[i].leaf_id) {
            /* Duplicate leaf_id (old + new after promotion) : keep the one
             * with more n_docs (= the promoted one). Should not happen if
             * promotions properly tombstoned, but robust to skipped fsyncs. */
            if (entries[i].n_docs > entries[uniq-1].n_docs) {
                entries[uniq-1] = entries[i];
            }
        } else {
            entries[uniq++] = entries[i];
        }
    }
    ht->entries = entries;
    ht->n_entries = uniq;
    ht->cap = recovered;
    h->total_docs += docs_recovered;
    pthread_mutex_unlock(&ht->mu);
    return recovered;
}

/* Caller holds tree lock. */
void hot_range(const HotOverlay* h, int tree_id,
               uint32_t low_leaf, uint32_t high_leaf,
               int* out_i_lo, int* out_i_hi) {
    if (!h || tree_id < 0 || tree_id >= h->n_trees) {
        *out_i_lo = 0; *out_i_hi = 0; return;
    }
    const HotTree* ht = &h->trees[tree_id];
    *out_i_lo = hot_lower_bound(ht, low_leaf);
    *out_i_hi = hot_lower_bound(ht, high_leaf);
}

/* Read entry i's docs from disk. Caller must have SNAPSHOTTED (leaf_id,
 * packed, n_docs) under lock before releasing ; the snapshot arguments are
 * passed to avoid re-reading a possibly-changed entry post-unlock.       */
/* Read docs from slot payload, skipping the 8 B header. */
static int read_slot(int fd, uint64_t off, uint32_t n_docs,
                     uint32_t* out_buf, uint32_t buf_cap, uint32_t* out_n) {
    uint32_t n = (n_docs > buf_cap) ? buf_cap : n_docs;
    if (n == 0) { *out_n = 0; return 0; }
    ssize_t r = pread(fd, out_buf, (size_t)n * 4u,
                      (off_t)(off + HOT_SLOT_HEADER_BYTES));
    if (r != (ssize_t)(n * 4u)) return -1;
    *out_n = n;
    return 0;
}

int hot_snapshot_entry(const HotOverlay* h, int tree_id, int entry_idx,
                       uint64_t* out_off, uint32_t* out_n_docs, int* out_fd) {
    if (!h || tree_id < 0 || tree_id >= h->n_trees) return -1;
    HotTree* ht = &((HotOverlay*)h)->trees[tree_id];
    pthread_mutex_lock(&ht->mu);
    if (entry_idx < 0 || entry_idx >= ht->n_entries) {
        pthread_mutex_unlock(&ht->mu);
        return -1;
    }
    HotEntry e = ht->entries[entry_idx];
    int cls = (int)HOT_CLASS(e.packed);
    uint32_t slot = HOT_SLOT(e.packed);
    /* Payload offset : skip the 8 B slot header. */
    *out_off = slot_offset(ht, cls, slot) + HOT_SLOT_HEADER_BYTES;
    *out_n_docs = e.n_docs;
    *out_fd = ht->fds[cls];
    pthread_mutex_unlock(&ht->mu);
    return 0;
}

int hot_read_entry(const HotOverlay* h, int tree_id, int entry_idx,
                   uint32_t* out_buf, uint32_t buf_cap, uint32_t* out_n) {
    if (!h || tree_id < 0 || tree_id >= h->n_trees) return -1;
    HotTree* ht = &((HotOverlay*)h)->trees[tree_id];
    pthread_mutex_lock(&ht->mu);
    if (entry_idx < 0 || entry_idx >= ht->n_entries) {
        pthread_mutex_unlock(&ht->mu);
        *out_n = 0;
        return 0;
    }
    HotEntry e = ht->entries[entry_idx];   /* snapshot */
    int cls = (int)HOT_CLASS(e.packed);
    uint32_t slot = HOT_SLOT(e.packed);
    uint64_t off = slot_offset(ht, cls, slot);
    int fd = ht->fds[cls];
    pthread_mutex_unlock(&ht->mu);
    return read_slot(fd, off, e.n_docs, out_buf, buf_cap, out_n);
}

int hot_read(const HotOverlay* h, int tree_id, uint32_t leaf_id,
             uint32_t* out_buf, uint32_t buf_cap, uint32_t* out_n) {
    if (!h || tree_id < 0 || tree_id >= h->n_trees) { *out_n = 0; return 0; }
    HotTree* ht = &((HotOverlay*)h)->trees[tree_id];
    pthread_mutex_lock(&ht->mu);
    int i = hot_lower_bound(ht, leaf_id);
    if (i >= ht->n_entries || ht->entries[i].leaf_id != leaf_id) {
        pthread_mutex_unlock(&ht->mu);
        *out_n = 0;
        return 0;
    }
    HotEntry e = ht->entries[i];
    int cls = (int)HOT_CLASS(e.packed);
    uint32_t slot = HOT_SLOT(e.packed);
    uint64_t off = slot_offset(ht, cls, slot);
    int fd = ht->fds[cls];
    pthread_mutex_unlock(&ht->mu);
    return (read_slot(fd, off, e.n_docs, out_buf, buf_cap, out_n) == 0) ? 1 : -1;
}

uint64_t hot_ram_bytes(const HotOverlay* h) {
    if (!h || !h->trees) return 0;
    uint64_t b = sizeof(HotOverlay) + (uint64_t)h->n_trees * sizeof(HotTree);
    for (int t = 0; t < h->n_trees; t++) {
        b += (uint64_t)h->trees[t].cap * sizeof(HotEntry);
    }
    return b;
}

uint64_t hot_disk_bytes(const HotOverlay* h) {
    if (!h || !h->trees) return 0;
    uint64_t b = 0;
    for (int t = 0; t < h->n_trees; t++)
        for (int c = 0; c < HOT_N_CLASSES; c++)
            b += h->trees[t].file_size[c];
    return b;
}

int hot_snapshot_and_clear(HotOverlay* h, int tree_id,
                           HotSnapEntry** out_snap, int* out_n) {
    if (!h || tree_id < 0 || tree_id >= h->n_trees) return -1;
    HotTree* ht = &h->trees[tree_id];
    pthread_mutex_lock(&ht->mu);
    int n = ht->n_entries;
    if (n == 0) {
        *out_snap = NULL; *out_n = 0;
        pthread_mutex_unlock(&ht->mu);
        return 0;
    }
    HotSnapEntry* snap = (HotSnapEntry*)malloc((size_t)n * sizeof(HotSnapEntry));
    if (!snap) { pthread_mutex_unlock(&ht->mu); return -1; }
    for (int i = 0; i < n; i++) {
        HotEntry e = ht->entries[i];
        int cls = (int)HOT_CLASS(e.packed);
        uint32_t slot = HOT_SLOT(e.packed);
        uint64_t off = slot_offset(ht, cls, slot);
        snap[i].leaf_id = e.leaf_id;
        snap[i].n_docs = e.n_docs;
        snap[i].docs = (uint32_t*)malloc((size_t)e.n_docs * sizeof(uint32_t));
        if (!snap[i].docs) {
            for (int j = 0; j < i; j++) free(snap[j].docs);
            free(snap);
            pthread_mutex_unlock(&ht->mu);
            return -1;
        }
        if (pread(ht->fds[cls], snap[i].docs, (size_t)e.n_docs * 4u,
                  (off_t)(off + HOT_SLOT_HEADER_BYTES))
            != (ssize_t)(e.n_docs * 4u)) {
            /* leave zeros; not fatal for MVP */
        }
    }
    /* Clear index + truncate every class file to header. */
    free(ht->entries);
    ht->entries = NULL;
    ht->n_entries = 0;
    ht->cap = 0;
    memset(ht->class_n_slots, 0, sizeof(ht->class_n_slots));
    for (int c = 0; c < HOT_N_CLASSES; c++) {
        ftruncate(ht->fds[c], HOT_HDR_TOTAL);
        ht->file_size[c] = HOT_HDR_TOTAL;
        persist_class_n_slots(ht, c);
    }
    pthread_mutex_unlock(&ht->mu);
    *out_snap = snap;
    *out_n = n;
    return 0;
}

void hot_snapshot_free(HotSnapEntry* snap, int n) {
    if (!snap) return;
    for (int i = 0; i < n; i++) free(snap[i].docs);
    free(snap);
}
