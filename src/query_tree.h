#ifndef QUERY_TREE_H
#define QUERY_TREE_H

#include <stdint.h>
#include <liburing.h>
#include "sorted_store.h"

/* Forward decl so callers don't have to pull <roaring/roaring.h>. */
struct roaring_bitmap_s;
typedef struct roaring_bitmap_s roaring_bitmap_t;

typedef struct {
    int    n_trees;
    int    dim;
    int    sub_dim;      /* 0 = full dim */
    int    depth;
    int    n_docs;       /* total indexed docs (used by legacy votes path) */
    SortedStore* stores; /* per-tree sorted posting-list file */
    struct io_uring ring;
    int    ring_ok;

    /* Persistent per-query scratch (allocated in forest_open).
       Avoids 32 MB malloc/free traffic per query.                       */
    uint32_t       max_window_n;     /* max(stride+1) across all trees   */
    void*          wbuf;             /* SparseEntry[n_trees × 2 × max_window_n] (low, high slots) */
    uint32_t*      wlen_buf;
    uint64_t*      woff_buf;
    int32_t*       leaves_buf;       /* node_id at query_depth (max depth 30) */
    uint32_t*      lens_buf;
    uint64_t*      data_off_buf;
    uint32_t*      buf_pos_buf;
    void*          cursors_buf;
    void**         heap_buf;
    void*          topn_h_buf;
    int            topn_h_cap;       /* current capacity of topn_h_buf   */
    uint32_t*      docs_buf;
    uint32_t       docs_cap;
    uint8_t*       bytes_buf;        /* SRT3 raw-encoded scratch (per query) */
    uint32_t       bytes_cap;
    uint32_t*      byte_pos_buf;     /* SRT3: per-tree start in bytes_buf    */
    int            srt_version;      /* 2 = SRT2, 3 = SRT3                   */

    /* Tombstones: soft-deleted doc_ids, loaded at forest_open from
       <index_dir>/tombstones.roaring. NULL = none. Applied at query time
       as AND-NOT with the caller's filter (see forest_collect_topn).
       In-memory bitmap; persist with forest_save_tombstones().              */
    roaring_bitmap_t* tombstones;
    char           index_dir[512];   /* kept for save()                      */

    /* Top-level median thresholds (medians.bin, optional). NULL = classic
       sign splits. Table = n_trees × (2^med_depth - 1) floats.            */
    float*         medians;
    int            med_depth;
    /* Variante MED2 int8 (exclusive de `medians`) : codes u8 par noeud +
       echelles par (arbre, niveau). 4x moins de RAM — voir medians8_load. */
    uint8_t*       medians8;
    float*         med_scales;

    /* io_uring registered buffer for the TQ1 stage 1 read path.
       Allocated at forest_open, registered with io_uring_register_buffers,
       used by tq1_select via io_uring_prep_read_fixed → eliminates the
       kernel→userspace copy_to_user (~18 % of total query time in profiles).
       Size = 8 MB, covers top_n × code_bytes up to ~64k codes at 128 B.
       If allocation or registration fails, fields stay NULL/0 and the
       code paths fall back transparently to the unregistered prep_read.     */
    void*          fixed_buf;
    size_t         fixed_buf_size;
    int            fixed_buf_registered;   /* 0 = OFF, 1 = ring has it       */
} Forest;

typedef struct {
    int32_t doc_id;
    int32_t votes;
} Result;

int  forest_open (Forest* f, const char* index_dir,
                  int n_trees, int dim, int sub_dim,
                  int depth, int n_docs);
void forest_close(Forest* f);

/* Reopen one tree's SRT (used after compaction/atomic-swap replacement).
 * Caller must synchronize against concurrent queries. */
int  forest_reopen_tree(Forest* f, int tree_id);

/* Legacy votes-based search (votes_scratch = uint16_t[n_docs]). */
int  forest_search(const Forest* f, const float* qvec,
                   int top_k, int threshold,
                   Result* out, uint16_t* votes_scratch);

int  forest_collect(const Forest* f, const float* qvec,
                    int threshold, int max_out,
                    int32_t* out_ids, int32_t* out_votes,
                    uint16_t* votes_scratch);

/* Primary path: top-N by votes via K-way merge on sorted posting lists.
   2-phase io_uring (offsets, then data). RAM O(n_trees + sum_leaf_docs).

   `query_depth` lets you query at any depth ≤ build_depth (the tree's depth).
   At query_depth < build_depth, traverse stops at query_depth and the
   lookup pulls all docs whose build-depth leaf_id falls in the range
   [parent_id << k, (parent_id+1) << k) where k = build_depth - query_depth.
   Pass query_depth = 0 (or build_depth) for native-depth queries.       */
/* `allowed` (NULL = disabled) is a CRoaring bitmap over uint32 doc_ids :
   doc_id i ∈ bitmap ⇒ i is allowed. roaring_bitmap_contains() is the
   membership probe in the K-way merge loop. Wire format matches ClickHouse's
   AggregateFunction(groupBitmap, UInt32) state (see croaring_io.h).      */
int  forest_collect_topn(const Forest* f, const float* qvec,
                         int top_n, int query_depth,
                         const roaring_bitmap_t* allowed,
                         int32_t* out_ids, int32_t* out_votes);

int  forest_collect_topn_multi(Forest** forests, int n_forests,
                               const float* qvec,
                               int top_n, int query_depth,
                               const roaring_bitmap_t* allowed,
                               int32_t* out_ids, int32_t* out_votes);

/* Fused multi-probe collect for ONE forest: single io_uring pass + single
   merge over all nt × n_sets probe leaves. `leaves` is int32[nt*n_sets],
   leaves[set*nt + t] = (node - leaf_base) leaf of tree t in probe-set `set`.
   Vote = # distinct trees containing the doc (deduped across a tree's
   probe leaves and across probe sets). Returns topn_size or -1.

   `probe_depth`: 0 (or = f->depth) → native depth (each leaf = 1 storage
   leaf). 0 < probe_depth < f->depth → subtree-aware path : each (set, tree)
   leaf covers 2^(f->depth - probe_depth) storage leaves at build depth ;
   the docs from all those storage sub-leaves are merged in the same
   fused pipeline (Phase 1 reads 2 sparse-index windows per (set, tree),
   Phase 2 reads the byte range covering all sub-leaves at once, decode
   walks the sub-leaf boundaries to reset each one's first_doc baseline). */
int  forest_collect_topn_probes(const Forest* f,
                                const int32_t* leaves, int n_sets,
                                int top_n, int probe_depth,
                                const roaring_bitmap_t* allowed,
                                int32_t* out_ids, int32_t* out_votes);

/* Privacy mode : the caller has already computed the per-tree leaf_ids
   client-side (via compute_leaves() in the Python SDK) and provides them
   via this setter BEFORE calling forest_collect_topn[_multi]. The server
   never traverses, never sees qvec. Final L2 rerank is the client's job
   (it has qvec, server doesn't). Pass NULL to reset to normal mode.       */
void forest_set_external_leaves(const int32_t* leaves, int n);

/* Per-query wall-clock deadline (thread-local, MUST set before query call).
   abs_ns = CLOCK_MONOTONIC absolute time in ns after which the query should
   stop and return what it has. 0 = disabled. When tripped, the query returns
   the partial top-N built so far; forest_get_last_query_partial() reports 1. */
void  forest_set_query_deadline_ns(int64_t abs_ns);
int   forest_get_last_query_partial(void);

/* HOT overlay opt-in (thread-local). When non-NULL, forest_collect_topn_probes
   merges HOT docs alongside SRT MAIN in the no-filter fast path. NULL disables.
   Filter path (allowed != NULL) does NOT consult HOT yet — TODO.               */
#include "hot_store.h"
void  forest_set_hot_overlay(const HotOverlay* h);
const HotOverlay* forest_get_hot_overlay(void);

/* Shared scratch pool : when enabled, all Forests in this thread share
   the same thread-local bytes_buf/docs_buf (the two largest per-query
   buffers, ~300-500 MB at SIFT 1B segment scale). Queries are already
   serial per LiveIndex, so sharing has zero correctness cost.
   Returns the current pool size in bytes. */
void   forest_set_shared_scratch_pool(int enable);
size_t forest_shared_scratch_bytes(void);

/* Number of distinct doc_ids the K-way merge visited during the last
   `forest_collect_topn[_multi]` call, before capping to top_n.            */
int  forest_get_last_n_distinct(void);

/* Total (non-distinct) doc_ids visited during the last collect. */
int  forest_get_last_n_total(void);

/* Cap mega-leaves at shallow depth : leaves whose encoded data exceeds this
   many bytes are truncated. 0 = no cap. See project memory max_leaf_bytes. */
void     forest_set_max_leaf_bytes(uint32_t n);
uint32_t forest_get_max_leaf_bytes(void);

/* Cap the K-way merge: once `n` distinct candidates have been fully
   counted, return immediately. 0 (default) = no cap. Thread-local.
   Use this to hard-bound p99 on heavy queries.                            */
void forest_set_max_distinct(int n);
int  forest_get_max_distinct(void);

/* Smarter cap: exit after `n` consecutive pushes failed to enter the
   (full) top_n heap. 0 = disabled. Default for production tail-trimming. */
void forest_set_max_stable_rejects(int n);
int  forest_get_max_stable_rejects(void);

/* Mutation API — soft delete (tombstones). Caller is responsible for
   serializing concurrent adds against in-flight queries (a forest is not
   thread-safe for queries; same applies here). Save = atomic write.       */
int  forest_add_tombstone   (Forest* f, uint32_t doc_id);
int  forest_remove_tombstone(Forest* f, uint32_t doc_id);
int  forest_save_tombstones (Forest* f);
int  forest_tombstone_count (const Forest* f);

#endif
