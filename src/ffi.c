/* Public C-ABI surface for Python ctypes / other bindings.
   All entry points start with `mg_` and use opaque void* handles. */

#define _POSIX_C_SOURCE 200809L
#include "query_tree.h"
#include "sorted_store.h"
#include "tombstones.h"   /* tombstones_load — returns a pointer, must not be implicit */
#include "build_tree.h"   /* calibrate_and_save_medians */
#include "varbyte.h"
#include "traversal.h"
#include "croaring_io.h"
#include "gen_vec.h"
#include "srt_hash.h"
#include "recall.h"
#include "vec_format.h"
#include "tquant.h"
#include "tq1.h"

#include <roaring/roaring.h>
#include <fcntl.h>
#include <unistd.h>

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "slot_store.h"

/* Forward decls from slot_query.c */
typedef struct {
    int         n_trees, dim, sub_dim, depth, n_docs;
    SltStore*   stores;
    struct io_uring ring;
    int         ring_ok;
} SltForest;
int  slt_forest_open(SltForest* f, const char* index_dir, int n_trees, int dim,
                     int sub_dim, int depth, int n_docs);
void slt_forest_close(SltForest* f);
int  slt_query_pathrank(SltForest* f, const float* qvec,
                        int n_probes, int top_paths, int top_n,
                        int query_depth,
                        int32_t* out_ids, int32_t* out_votes);

/* Convert .srt V2 to .slt (from slot_store.c) */
int mg_slt_convert(const char* srt_path, const char* slt_path) {
    int n_ovf = 0;
    return slt_convert_from_srt_v2(srt_path, slt_path, &n_ovf);
}

void* mg_slt_forest_open(const char* index_dir, int n_trees, int dim,
                         int sub_dim, int depth, int n_docs) {
    SltForest* f = (SltForest*)calloc(1, sizeof(SltForest));
    if (!f) return NULL;
    if (slt_forest_open(f, index_dir, n_trees, dim, sub_dim, depth, n_docs) != 0) {
        free(f); return NULL;
    }
    return f;
}
void mg_slt_forest_close(void* h) {
    if (!h) return;
    SltForest* f = (SltForest*)h;
    slt_forest_close(f);
    free(f);
}
int mg_slt_query_pathrank(void* h, const float* qvec,
                          int n_probes, int top_paths, int top_n,
                          int query_depth,
                          int32_t* out_ids, int32_t* out_votes) {
    if (!h) return -1;
    return slt_query_pathrank((SltForest*)h, qvec, n_probes, top_paths,
                              top_n, query_depth, out_ids, out_votes);
}

/* ---------- HOT overlay (in-RAM streaming appends) ---------- */

void* mg_hot_init(int n_trees, int depth, const char* dir) {
    HotOverlay* h = (HotOverlay*)calloc(1, sizeof(HotOverlay));
    if (!h) return NULL;
    if (hot_init(h, n_trees, depth, dir) != 0) { free(h); return NULL; }
    return h;
}

int64_t mg_hot_disk_bytes(void* h) {
    if (!h) return 0;
    return (int64_t)hot_disk_bytes((HotOverlay*)h);
}

void mg_hot_free(void* h) {
    if (!h) return;
    hot_free((HotOverlay*)h);
    free(h);
}

int mg_hot_append(void* h, int tree_id, uint32_t leaf_id, uint32_t doc_id) {
    if (!h) return -1;
    return hot_append((HotOverlay*)h, tree_id, leaf_id, doc_id);
}

/* Batch append : for each (tree_id, leaf_id) pair, append doc_id. Arrays same length. */
int mg_hot_append_batch(void* h, const int* tree_ids, const uint32_t* leaf_ids,
                        const uint32_t* doc_ids, int n) {
    if (!h) return -1;
    HotOverlay* ho = (HotOverlay*)h;
    for (int i = 0; i < n; i++) {
        if (hot_append(ho, tree_ids[i], leaf_ids[i], doc_ids[i]) != 0) return -1;
    }
    return 0;
}

int64_t mg_hot_ram_bytes(void* h) {
    if (!h) return 0;
    return (int64_t)hot_ram_bytes((HotOverlay*)h);
}

int64_t mg_hot_total_docs(void* h) {
    if (!h) return 0;
    return (int64_t)((HotOverlay*)h)->total_docs;
}

void mg_forest_set_hot_overlay(void* h) {
    forest_set_hot_overlay((const HotOverlay*)h);
}

/* Direct call into forest_collect_topn_probes with a pre-computed leaves
 * array. Used by experiments that bypass traversal (e.g. Python-side
 * liveness filter). leaves layout : int32[nt * n_sets], leaves[s*nt+t].
 * Returns topn count or -1. */
int mg_forest_query_leaves(void* h, const int32_t* leaves, int n_sets,
                           int top_n, int probe_depth,
                           int32_t* out_ids, int32_t* out_votes) {
    if (!h || !leaves) return -1;
    Forest* f = (Forest*)h;
    return forest_collect_topn_probes(f, leaves, n_sets, top_n, probe_depth,
                                       NULL, out_ids, out_votes);
}

/* ---------- HOT compaction ---------- */

#include "hot_compact.h"
#include <sys/stat.h>

/* Compact one tree : snapshot HOT for tree_id (drains it), merge with
 * MAIN SRT at <index_dir>/tree{tree_id}.srt, write to .tmp, atomic rename,
 * reopen forest fd.
 * out_format = 2 (SRT V2 raw uint32) or 3 (SRT V3 varbyte delta).
 * Returns 0 on success, -1 on error. */
int mg_hot_compact_tree_ex(void* forest_h, void* hot_h,
                           int tree_id, const char* index_dir,
                           int out_format) {
    if (!forest_h || !hot_h || !index_dir) return -1;
    Forest* f = (Forest*)forest_h;
    HotOverlay* h = (HotOverlay*)hot_h;
    if (tree_id < 0 || tree_id >= f->n_trees) return -1;

    /* Snapshot + clear HOT for this tree. */
    HotSnapEntry* snap = NULL;
    int snap_n = 0;
    if (hot_snapshot_and_clear(h, tree_id, &snap, &snap_n) != 0) return -1;
    if (snap_n == 0) return 0;   /* nothing to do */

    char main_path[512], tmp_path[520];
    snprintf(main_path, sizeof(main_path), "%s/tree%05d.srt", index_dir, tree_id);
    snprintf(tmp_path,  sizeof(tmp_path),  "%s/tree%05d.srt.tmp", index_dir, tree_id);

    int new_n = 0;
    uint64_t new_total = 0;
    int rc = hot_compact_tree(main_path, snap, snap_n, tmp_path,
                              out_format, &new_n, &new_total);
    hot_snapshot_free(snap, snap_n);
    if (rc != 0) { unlink(tmp_path); return -1; }

    if (rename(tmp_path, main_path) != 0) { perror("rename"); return -1; }
    return forest_reopen_tree(f, tree_id);
}

/* Default : V3 output (matches production SRT V3 format). */
int mg_hot_compact_tree(void* forest_h, void* hot_h,
                        int tree_id, const char* index_dir) {
    return mg_hot_compact_tree_ex(forest_h, hot_h, tree_id, index_dir, 3);
}

/* Compact all trees serially. Caller can throttle by sleeping between iters
 * if invoked from Python thread. */
int mg_hot_compact_all(void* forest_h, void* hot_h, const char* index_dir) {
    if (!forest_h || !hot_h || !index_dir) return -1;
    Forest* f = (Forest*)forest_h;
    for (int t = 0; t < f->n_trees; t++) {
        int rc = mg_hot_compact_tree(forest_h, hot_h, t, index_dir);
        if (rc != 0) return rc;
    }
    return 0;
}

/* Per-phase profiling of the last collect_topn_probes call (see query_tree.c).
   Index 0-7 : phase1, resolve, phase2, decode, pack, radix, scan, total. */
double forest_get_phase_ms(int i);
double mg_get_phase_ms(int i) { return forest_get_phase_ms(i); }

/* Build the dense-samples sidecar <srt_path>.smp (stride 0 = default 128). */
int mg_build_smp(const char* srt_path, unsigned int stride) {
    return srt_build_smp(srt_path, stride);
}

/* Macroblock (.mbk) experiment : implemented + benched 2026-07-19, then
   REVERTED by decision — read amplification loses on byte-bound storage
   (host cache, NVMe) and only wins ~×2 on strictly IOPS-bound cold USB.
   Design + results in memory `project_query_phase_profile.md`. */

/* ---------- HOT WAL ---------- */

#include "hot_wal.h"

void* mg_hot_wal_open(const char* path, int fsync_interval_ms) {
    return hot_wal_open(path, fsync_interval_ms);
}
void mg_hot_wal_close(void* w) { hot_wal_close((HotWal*)w); }
int  mg_hot_wal_append(void* w, uint32_t tree_id, uint32_t leaf_id, uint32_t doc_id) {
    return hot_wal_append((HotWal*)w, tree_id, leaf_id, doc_id);
}
int  mg_hot_wal_flush   (void* w) { return hot_wal_flush   ((HotWal*)w); }
int  mg_hot_wal_truncate(void* w) { return hot_wal_truncate((HotWal*)w); }
int64_t mg_hot_wal_size_bytes(void* w) { return (int64_t)hot_wal_size_bytes((HotWal*)w); }
int64_t mg_hot_wal_n_records (void* w) { return (int64_t)hot_wal_n_records ((HotWal*)w); }

/* Replay WAL into the given overlay via hot_append. Returns records replayed. */
struct hot_wal_replay_ctx { HotOverlay* h; };
static int hot_wal_replay_cb(void* ctx, uint32_t tid, uint32_t lid, uint32_t did) {
    return hot_append(((struct hot_wal_replay_ctx*)ctx)->h,
                      (int)tid, lid, did);
}
int mg_hot_wal_replay_into(const char* path, void* hot_h) {
    if (!path || !hot_h) return -1;
    struct hot_wal_replay_ctx c = { .h = (HotOverlay*)hot_h };
    return hot_wal_replay(path, hot_wal_replay_cb, &c);
}

/* ---------- Background compaction ---------- */

#include "hot_compact_bg.h"

void* mg_hot_compact_bg_start(void* forest_h, void* hot_h,
                              const char* index_dir,
                              int threshold_docs, int sleep_ms,
                              int out_format) {
    return hot_compact_bg_start((Forest*)forest_h, (HotOverlay*)hot_h,
                                 index_dir, threshold_docs, sleep_ms,
                                 out_format);
}
void mg_hot_compact_bg_stop(void* bg_h) { hot_compact_bg_stop((HotCompactBg*)bg_h); }
int64_t mg_hot_compact_bg_n_compactions(void* bg_h) {
    return (int64_t)hot_compact_bg_n_compactions((HotCompactBg*)bg_h);
}
int64_t mg_hot_compact_bg_n_docs_merged(void* bg_h) {
    return (int64_t)hot_compact_bg_n_docs_merged((HotCompactBg*)bg_h);
}

/* ---------- Bulk mode : atomic directory swap ---------- */

/* Swap forest to point at a new index_dir. Caller MUST have serialized
 * queries against this forest before calling (no in-flight queries).
 *
 * Assumes new_dir contains a fully built index with the SAME schema
 * (n_trees, dim, sub_dim, depth). If not, subsequent queries corrupt.
 *
 * MVP semantics : reopen each tree fd from new_dir, update index_dir.
 * Old fd's are closed → any queries mid-flight would see EBADF (caller
 * responsibility to prevent).                                              */
int mg_forest_swap_dir(void* forest_h, const char* new_dir) {
    if (!forest_h || !new_dir) return -1;
    Forest* f = (Forest*)forest_h;

    /* Snap new index_dir first (used by forest_reopen_tree). */
    strncpy(f->index_dir, new_dir, sizeof(f->index_dir) - 1);
    f->index_dir[sizeof(f->index_dir) - 1] = '\0';

    for (int t = 0; t < f->n_trees; t++) {
        if (forest_reopen_tree(f, t) != 0) {
            fprintf(stderr, "swap_dir: reopen tree %d failed\n", t);
            return -1;
        }
    }
    /* Reload tombstones if the file exists in new_dir. */
    if (f->tombstones) {
        roaring_bitmap_free(f->tombstones);
        f->tombstones = NULL;
    }
    f->tombstones = tombstones_load(new_dir);
    return 0;
}

/* ---------- Forest lifecycle ---------- */

void* mg_forest_open(const char* index_dir, int n_trees, int dim,
                     int sub_dim, int depth, int n_docs) {
    Forest* f = (Forest*)calloc(1, sizeof(Forest));
    if (!f) return NULL;
    if (forest_open(f, index_dir, n_trees, dim, sub_dim, depth, n_docs) != 0) {
        free(f);
        return NULL;
    }
    return (void*)f;
}

void mg_forest_close(void* h) {
    if (!h) return;
    Forest* f = (Forest*)h;
    forest_close(f);
    free(f);
}

/* ---------- Query (single + multi) ----------
   `allowed_state` is the raw ClickHouse groupBitmap wire format
   (see croaring_io.h). Pass NULL/0 to disable filtering.
   `out_ids` / `out_votes` are int32 buffers of size ≥ top_n.
   Returns the number of results actually filled.                          */

int mg_forest_query(void* h, const float* qvec, int top_n, int query_depth,
                    const uint8_t* allowed_state, int allowed_state_len,
                    int* out_ids, int* out_votes) {
    if (!h) return -1;
    Forest* f = (Forest*)h;
    roaring_bitmap_t* allowed = NULL;
    if (allowed_state && allowed_state_len > 0) {
        int card = 0;
        allowed = roaring_from_ch_state(allowed_state, (size_t)allowed_state_len, &card);
        (void)card;
        if (!allowed) return -1;
    }
    int n = forest_collect_topn(f, qvec, top_n, query_depth,
                                allowed,
                                (int32_t*)out_ids, (int32_t*)out_votes);
    if (allowed) roaring_bitmap_free(allowed);
    return n;
}

/* Multi-probe leaf computation (see traverse_sub_probes in traversal.c).
   Fills out_leaves[(n_probes+1) * n_trees] in pass-major order:
   out_leaves[p*n_trees + t] = probe-p leaf of tree t, in (node - leaf_base)
   form that the external-leaves query path expects. The caller runs n_probes+1
   query passes (arming mg_set_external_leaves with each pass-slice) and merges
   the votes. Returns n_trees, or -1 on failure. */
int mg_probe_leaves(void* h, const float* qvec, int n_probes,
                    int probe_depth, int probe_span, int32_t* out_leaves) {
    if (!h) return -1;
    Forest* f = (Forest*)h;
    int nt = f->n_trees, dim = f->dim, sub = f->sub_dim;
    int depth = (probe_depth > 0 && probe_depth < f->depth) ? probe_depth : f->depth;
    int use_sub = (sub > 0 && sub <= dim);
    if (n_probes < 0) n_probes = 0;
    /* probe_span = # of deepest levels eligible to flip. 0 = any level (legacy). */
    int min_flip_level = (probe_span > 0 && probe_span < depth) ? depth - probe_span : 0;

    float* qn   = (float*)malloc((size_t)dim * sizeof(float));
    float* v0   = (float*)malloc((size_t)dim * sizeof(float));
    float* v1   = (float*)malloc((size_t)dim * sizeof(float));
    int*   dims = (int*)malloc((size_t)dim * sizeof(int));
    int32_t* nodes = (int32_t*)malloc((size_t)(n_probes + 1) * sizeof(int32_t));
    if (!qn || !v0 || !v1 || !dims || !nodes) {
        free(qn); free(v0); free(v1); free(dims); free(nodes); return -1;
    }
    float nrm = 0.0f;
    for (int i = 0; i < dim; i++) nrm += qvec[i] * qvec[i];
    nrm = (nrm > 0.0f) ? 1.0f / sqrtf(nrm) : 1.0f;
    for (int i = 0; i < dim; i++) qn[i] = qvec[i] * nrm;

    int32_t base = leaf_base(depth);
    for (int t = 0; t < nt; t++) {
        int cnt;
        if (use_sub)
            cnt = traverse_sub_probes(qn, dim, sub, depth, tree_seed(t),
                                      v0, v1, dims, n_probes, min_flip_level, nodes);
        else { nodes[0] = traverse(qn, dim, depth, tree_seed(t), v0, v1); cnt = 1; }
        for (int p = 0; p <= n_probes; p++) {
            int32_t node = (p < cnt) ? nodes[p] : nodes[0];
            out_leaves[(size_t)p * nt + t] = node - base;
        }
    }
    free(qn); free(v0); free(v1); free(dims); free(nodes);
    return nt;
}

/* Fused multi-probe query: compute probe leaves AND run the single-pass collect
   in one C call (no per-pass FFI round-trips, no Python vote merge). Native
   depth only. allowed_state = ClickHouse groupBitmap wire bytes or NULL.
   out_ids/out_votes sized >= top_n. Returns topn_size or -1. */
int mg_query_probes(void* h, const float* qvec, int n_probes, int probe_span,
                    int probe_depth,
                    int top_n,
                    const uint8_t* allowed_state, int allowed_state_len,
                    int* out_ids, int* out_votes) {
    if (!h) return -1;
    Forest* f = (Forest*)h;
    int nt = f->n_trees, dim = f->dim, sub = f->sub_dim, depth = f->depth;
    int use_sub = (sub > 0 && sub <= dim);
    if (n_probes < 0) n_probes = 0;
    int n_sets = n_probes + 1;
    /* probe_depth ≤ 0 or ≥ depth → native (fused path). Else multi-pass at qd. */
    int qd_eff = (probe_depth > 0 && probe_depth < depth) ? probe_depth : depth;
    int use_fused = (qd_eff == depth);
    int min_flip_level = (probe_span > 0 && probe_span < qd_eff) ? qd_eff - probe_span : 0;

    float* qn   = (float*)malloc((size_t)dim * sizeof(float));
    float* v0   = (float*)malloc((size_t)dim * sizeof(float));
    float* v1   = (float*)malloc((size_t)dim * sizeof(float));
    int*   dims = (int*)malloc((size_t)dim * sizeof(int));
    int32_t* nodes  = (int32_t*)malloc((size_t)n_sets * sizeof(int32_t));
    int32_t* leaves = (int32_t*)malloc((size_t)nt * n_sets * sizeof(int32_t));
    if (!qn || !v0 || !v1 || !dims || !nodes || !leaves) {
        free(qn); free(v0); free(v1); free(dims); free(nodes); free(leaves);
        return -1;
    }
    float nrm = 0.0f;
    for (int i = 0; i < dim; i++) nrm += qvec[i] * qvec[i];
    nrm = (nrm > 0.0f) ? 1.0f / sqrtf(nrm) : 1.0f;
    for (int i = 0; i < dim; i++) qn[i] = qvec[i] * nrm;

    int32_t base = leaf_base(qd_eff);
    for (int t = 0; t < nt; t++) {
        int cnt;
        if (use_sub)
            cnt = traverse_sub_probes(qn, dim, sub, qd_eff, tree_seed(t),
                                      v0, v1, dims, n_probes, min_flip_level, nodes);
        else { nodes[0] = traverse(qn, dim, qd_eff, tree_seed(t), v0, v1); cnt = 1; }
        for (int s = 0; s < n_sets; s++) {
            int32_t node = (s < cnt) ? nodes[s] : -1;   /* -1 = no such probe → skip */
            leaves[(size_t)s * nt + t] = (node >= 0) ? node - base : -1;
        }
    }
    free(qn); free(v0); free(v1); free(dims); free(nodes);

    roaring_bitmap_t* allowed = NULL;
    if (allowed_state && allowed_state_len > 0) {
        int card = 0;
        allowed = roaring_from_ch_state(allowed_state, (size_t)allowed_state_len, &card);
        (void)card;
        if (!allowed) { free(leaves); return -1; }
    }

    /* Truly fused: single call handles k_shift=0 (native) AND k_shift>0
       (subtree expansion at probe_depth). The function does Phase 1 + 2 +
       multi-leaf decode + radix sort + per-tree dedup vote in one pass. */
    int n = forest_collect_topn_probes(f, leaves, n_sets, top_n, probe_depth,
                                       allowed,
                                       (int32_t*)out_ids, (int32_t*)out_votes);

    if (allowed) roaring_bitmap_free(allowed);
    free(leaves);
    return n;
}

/* ---------- Path-rank query : cross-tree top-K paths by margin ----------
   Traverses all trees with multi-probe (n_probes+1 paths/tree), then keeps
   the globally best `top_paths` by min-margin score across all trees ×
   probes. Non-selected paths are dropped (leaves[..]=-1). Then runs the
   standard vote-dedup + top_n cap via forest_collect_topn_probes.        */
typedef struct { float score; int32_t li; } PathRankEntry;
static int pathrank_cmp_desc(const void* a, const void* b) {
    float sa = ((const PathRankEntry*)a)->score;
    float sb = ((const PathRankEntry*)b)->score;
    if (sa > sb) return -1;
    if (sa < sb) return  1;
    return 0;
}

int mg_query_pathrank(void* h, const float* qvec,
                      int n_probes, int top_paths, int top_n,
                      int query_depth,
                      const uint8_t* allowed_state, int allowed_state_len,
                      int* out_ids, int* out_votes) {
    if (!h) return -1;
    Forest* f = (Forest*)h;
    int nt = f->n_trees, dim = f->dim, sub = f->sub_dim, depth = f->depth;
    int qd_eff = (query_depth > 0 && query_depth < depth) ? query_depth : depth;
    int use_sub = (sub > 0 && sub <= dim);
    if (n_probes < 0) n_probes = 0;
    int n_sets = n_probes + 1;
    size_t nL = (size_t)nt * (size_t)n_sets;
    if (top_paths <= 0 || (size_t)top_paths > nL) top_paths = (int)nL;

    float* qn   = (float*)malloc((size_t)dim * sizeof(float));
    int32_t*       leaves = (int32_t*)malloc(nL * sizeof(int32_t));
    PathRankEntry* rank   = (PathRankEntry*)malloc(nL * sizeof(PathRankEntry));
    if (!qn || !leaves || !rank) {
        free(qn); free(leaves); free(rank);
        return -1;
    }

    float nrm = 0.0f;
    for (int i = 0; i < dim; i++) nrm += qvec[i] * qvec[i];
    nrm = (nrm > 0.0f) ? 1.0f / sqrtf(nrm) : 1.0f;
    for (int i = 0; i < dim; i++) qn[i] = qvec[i] * nrm;

    int32_t base = leaf_base(qd_eff);
    for (size_t i = 0; i < nL; i++) { leaves[i] = -1; rank[i].score = -1.0f; rank[i].li = (int32_t)i; }

    /* Parallel traversal : trees are independent. Per-thread scratch on the
       heap (dim may be up to a few thousand → stack-allocation unsafe). */
    #pragma omp parallel
    {
        float*   _v0     = (float*)malloc((size_t)dim * sizeof(float));
        float*   _v1     = (float*)malloc((size_t)dim * sizeof(float));
        int*     _dims   = (int*)  malloc((size_t)dim * sizeof(int));
        int32_t* _nodes  = (int32_t*)malloc((size_t)n_sets * sizeof(int32_t));
        float*   _scores = (float*)  malloc((size_t)n_sets * sizeof(float));
        if (_v0 && _v1 && _dims && _nodes && _scores) {
            #pragma omp for schedule(static)
            for (int t = 0; t < nt; t++) {
                int cnt;
                traversal_set_medians(
                    f->medians ? f->medians + (size_t)t * ((1 << f->med_depth) - 1)
                               : NULL,
                    f->med_depth);
                if (use_sub) {
                    cnt = traverse_sub_probes_scored(qn, dim, sub, qd_eff, tree_seed(t),
                                                     _v0, _v1, _dims,
                                                     n_probes, 0,
                                                     _nodes, _scores);
                } else {
                    _nodes[0]  = traverse(qn, dim, qd_eff, tree_seed(t), _v0, _v1);
                    _scores[0] = 0.0f;
                    cnt = 1;
                }
                for (int s = 0; s < n_sets; s++) {
                    size_t li = (size_t)s * nt + t;
                    if (s < cnt) {
                        leaves[li]      = _nodes[s] - base;
                        rank[li].score  = _scores[s];
                        rank[li].li     = (int32_t)li;
                    }
                }
            }
        }
        free(_v0); free(_v1); free(_dims); free(_nodes); free(_scores);
    }

    /* Global sort by margin desc, keep top_paths ; mark the rest with leaves[..] = -1. */
    qsort(rank, nL, sizeof(PathRankEntry), pathrank_cmp_desc);
    for (size_t i = (size_t)top_paths; i < nL; i++) {
        leaves[rank[i].li] = -1;
    }
    free(rank);

    roaring_bitmap_t* allowed = NULL;
    if (allowed_state && allowed_state_len > 0) {
        int card = 0;
        allowed = roaring_from_ch_state(allowed_state, (size_t)allowed_state_len, &card);
        (void)card;
        if (!allowed) { free(leaves); return -1; }
    }

    int n = forest_collect_topn_probes(f, leaves, n_sets, top_n, qd_eff,
                                       allowed,
                                       (int32_t*)out_ids, (int32_t*)out_votes);
    if (allowed) roaring_bitmap_free(allowed);
    free(leaves);
    return n;
}

/* ---------- Pathrank by LEAF SIZE variant (experimental) ----------
   Same as mg_query_pathrank but selects top_paths by BYTE SIZE of leaves
   (bigger = more candidate docs per read) instead of margin. Motivation :
   NVMe cost dominated by IOPS ; a probe that yields many docs per op gives
   better docs-per-latency. The mlb (max_leaf_bytes) cap already excludes
   mega-leaves.

   Cost : one extra batched Phase 1 pass over ALL nL probes to read sparse
   windows and compute sizes. forest_collect_topn_probes re-does Phase 1 on
   the surviving probes → 2× Phase 1 total for the redundant path. Fine for
   experimentation ; production impl would fuse both into one Phase 1.       */
static uint32_t ffi_lb_in_window(const SparseEntry* window, uint32_t n,
                                 uint32_t target) {
    uint32_t lo = 0, hi = n;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (window[mid].leaf_id < target) lo = mid + 1;
        else                              hi = mid;
    }
    return lo;
}

int mg_query_pathrank_by_size(void* h, const float* qvec,
                              int n_probes, int top_paths, int top_n,
                              int query_depth,
                              int32_t* out_ids, int32_t* out_votes) {
    if (!h) return -1;
    Forest* f = (Forest*)h;
    int nt = f->n_trees, dim = f->dim, sub = f->sub_dim, depth = f->depth;
    int qd_eff = (query_depth > 0 && query_depth < depth) ? query_depth : depth;
    int use_sub = (sub > 0 && sub <= dim);
    int is_v3 = (f->srt_version == 3);
    int k_shift = (int)depth - qd_eff;
    if (n_probes < 0) n_probes = 0;
    int n_sets = n_probes + 1;
    size_t nL = (size_t)nt * (size_t)n_sets;
    if (top_paths <= 0 || (size_t)top_paths > nL) top_paths = (int)nL;

    float* qn   = (float*)malloc((size_t)dim * sizeof(float));
    int32_t*       leaves = (int32_t*)malloc(nL * sizeof(int32_t));
    PathRankEntry* rank   = (PathRankEntry*)malloc(nL * sizeof(PathRankEntry));
    if (!qn || !leaves || !rank) {
        free(qn); free(leaves); free(rank); return -1;
    }
    float nrm = 0.0f;
    for (int i = 0; i < dim; i++) nrm += qvec[i] * qvec[i];
    nrm = (nrm > 0.0f) ? 1.0f / sqrtf(nrm) : 1.0f;
    for (int i = 0; i < dim; i++) qn[i] = qvec[i] * nrm;

    int32_t base = leaf_base(qd_eff);
    for (size_t i = 0; i < nL; i++) { leaves[i] = -1; rank[i].score = -1.0f; rank[i].li = (int32_t)i; }

    /* Traversal (same as margin variant). */
    #pragma omp parallel
    {
        float*   _v0     = (float*)malloc((size_t)dim * sizeof(float));
        float*   _v1     = (float*)malloc((size_t)dim * sizeof(float));
        int*     _dims   = (int*)  malloc((size_t)dim * sizeof(int));
        int32_t* _nodes  = (int32_t*)malloc((size_t)n_sets * sizeof(int32_t));
        float*   _scores = (float*)  malloc((size_t)n_sets * sizeof(float));
        if (_v0 && _v1 && _dims && _nodes && _scores) {
            #pragma omp for schedule(static)
            for (int t = 0; t < nt; t++) {
                int cnt;
                if (use_sub) {
                    cnt = traverse_sub_probes_scored(qn, dim, sub, qd_eff, tree_seed(t),
                                                     _v0, _v1, _dims,
                                                     n_probes, 0, _nodes, _scores);
                } else {
                    _nodes[0] = traverse(qn, dim, qd_eff, tree_seed(t), _v0, _v1);
                    _scores[0] = 0.0f; cnt = 1;
                }
                for (int s = 0; s < n_sets; s++) {
                    size_t li = (size_t)s * nt + t;
                    if (s < cnt) leaves[li] = _nodes[s] - base;
                }
            }
        }
        free(_v0); free(_v1); free(_dims); free(_nodes); free(_scores);
    }

    /* Phase 1 batched : read sparse windows for ALL nL probes to get sizes. */
    struct io_uring* ring = (struct io_uring*)&f->ring;
    uint32_t window_n = f->max_window_n;
    int slots_to_read = (k_shift > 0) ? 2 : 1;
    SparseEntry* wbuf = (SparseEntry*)malloc(nL * 2 * window_n * sizeof(SparseEntry));
    uint32_t*    wlen = (uint32_t*)calloc(nL * 2, sizeof(uint32_t));
    if (!wbuf || !wlen) {
        free(wbuf); free(wlen); free(qn); free(leaves); free(rank); return -1;
    }
    uint32_t mlb = forest_get_max_leaf_bytes();
    int submitted = 0;
    for (size_t li = 0; li < nL; li++) {
        int t = (int)(li % nt);
        int32_t lf = leaves[li];
        if (lf < 0) continue;
        const SortedStore* st = &f->stores[t];
        if (st->n_nonempty == 0) continue;
        uint32_t low_leaf  = (k_shift > 0) ? ((uint32_t)lf << k_shift) : (uint32_t)lf;
        uint32_t high_leaf = (k_shift > 0) ? (((uint32_t)lf + 1u) << k_shift) : ((uint32_t)lf + 1u);
        uint32_t stride = st->sample_stride;
        for (int slot = 0; slot < slots_to_read; slot++) {
            uint32_t target = (slot == 0) ? low_leaf : high_leaf;
            uint32_t bucket = sorted_sample_bucket(st, target);
            uint32_t start_pos = bucket * stride;
            if (start_pos >= st->n_nonempty) continue;
            uint32_t take = stride + 1;
            if (start_pos + take > st->n_nonempty + 1) take = st->n_nonempty + 1 - start_pos;
            wlen[li * 2 + slot] = take;
            uint64_t off = st->index_base + (uint64_t)start_pos * SRT_INDEX_ENTRY;
            struct io_uring_sqe* sqe = io_uring_get_sqe(ring);
            if (!sqe) {
                io_uring_submit(ring);
                for (int r = 0; r < submitted; r++) {
                    struct io_uring_cqe* cqe;
                    if (io_uring_wait_cqe(ring, &cqe) == 0) io_uring_cqe_seen(ring, cqe);
                }
                submitted = 0;
                sqe = io_uring_get_sqe(ring);
            }
            io_uring_prep_read(sqe, st->fd,
                               wbuf + (li * 2 + slot) * window_n,
                               (unsigned)(take * SRT_INDEX_ENTRY), off);
            submitted++;
        }
    }
    io_uring_submit(ring);
    for (int r = 0; r < submitted; r++) {
        struct io_uring_cqe* cqe;
        if (io_uring_wait_cqe(ring, &cqe) == 0) io_uring_cqe_seen(ring, cqe);
    }

    /* Compute size per probe = (end_offset - start_offset). */
    for (size_t li = 0; li < nL; li++) {
        int t = (int)(li % nt);
        const SortedStore* st = &f->stores[t];
        int32_t lf = leaves[li];
        if (lf < 0 || wlen[li * 2] == 0) { rank[li].score = -1.0f; continue; }
        uint32_t low_leaf  = (k_shift > 0) ? ((uint32_t)lf << k_shift) : (uint32_t)lf;
        uint32_t high_leaf = (k_shift > 0) ? (((uint32_t)lf + 1u) << k_shift) : ((uint32_t)lf + 1u);
        const SparseEntry* low_w = wbuf + (li * 2 + 0) * window_n;
        uint32_t low_len = wlen[li * 2 + 0];
        uint32_t i_lo = ffi_lb_in_window(low_w, low_len, low_leaf);
        if (i_lo >= low_len) { rank[li].score = -1.0f; continue; }
        uint32_t start, end;
        if (k_shift == 0) {
            if (low_w[i_lo].leaf_id != low_leaf) { rank[li].score = -1.0f; continue; }
            start = low_w[i_lo].offset;
            end = (i_lo + 1 < low_len) ? low_w[i_lo + 1].offset
                                       : (is_v3 ? st->data_bytes : st->total_docs);
        } else {
            const SparseEntry* high_w = wbuf + (li * 2 + 1) * window_n;
            uint32_t high_len = wlen[li * 2 + 1];
            if (high_len == 0) { rank[li].score = -1.0f; continue; }
            uint32_t i_hi = ffi_lb_in_window(high_w, high_len, high_leaf);
            start = low_w[i_lo].offset;
            end   = (i_hi < high_len) ? high_w[i_hi].offset
                                       : (is_v3 ? st->data_bytes : st->total_docs);
        }
        if (end <= start) { rank[li].score = -1.0f; continue; }
        uint32_t size_bytes = is_v3 ? (end - start) : (end - start) * 4u;
        /* Exclude leaves above mlb, they'd be skipped by forest_collect_topn_probes anyway. */
        if (mlb != 0 && size_bytes > mlb) { rank[li].score = -1.0f; continue; }
        rank[li].score = (float)size_bytes;
    }
    free(wbuf); free(wlen);

    /* Sort DESC by size, mark leaves outside top_paths as -1. */
    qsort(rank, nL, sizeof(PathRankEntry), pathrank_cmp_desc);
    for (size_t i = (size_t)top_paths; i < nL; i++) leaves[rank[i].li] = -1;
    /* Also filter empty/mlb-exceeded (score < 0) even within top_paths. */
    for (size_t i = 0; i < (size_t)top_paths; i++) {
        if (rank[i].score < 0) leaves[rank[i].li] = -1;
    }
    free(rank);

    int n = forest_collect_topn_probes(f, leaves, n_sets, top_n, qd_eff,
                                        NULL, out_ids, out_votes);
    free(leaves); free(qn);
    return n;
}

/* Pathrank query across MULTIPLE segments sharing the same tree seeds.
   Design LSM : chaque segment = même partition trees (mêmes hyperplanes),
   docs disjoints. Traversal fait UNE FOIS, leaves array partagé.
   Chaque segment collecte ses docs via forest_collect_topn_probes.
   Merge final = concat + top-N par vote (docs disjoints → pas de somme). */
int mg_query_pathrank_multi(void** handles, int n_handles, const float* qvec,
                            int n_probes, int top_paths, int top_n,
                            int query_depth,
                            const uint8_t* allowed_state, int allowed_state_len,
                            int* out_ids, int* out_votes) {
    if (!handles || n_handles <= 0) return -1;
    Forest* f0 = (Forest*)handles[0];
    if (!f0) return -1;
    int nt = f0->n_trees, dim = f0->dim, sub = f0->sub_dim, depth = f0->depth;
    /* Sanity : all segments must share tree structure. */
    for (int i = 1; i < n_handles; i++) {
        Forest* fi = (Forest*)handles[i];
        if (!fi || fi->n_trees != nt || fi->dim != dim
                || fi->sub_dim != sub || fi->depth != depth) return -1;
    }
    int qd_eff = (query_depth > 0 && query_depth < depth) ? query_depth : depth;
    int use_sub = (sub > 0 && sub <= dim);
    if (n_probes < 0) n_probes = 0;
    int n_sets = n_probes + 1;
    size_t nL = (size_t)nt * (size_t)n_sets;
    if (top_paths <= 0 || (size_t)top_paths > nL) top_paths = (int)nL;

    float* qn   = (float*)malloc((size_t)dim * sizeof(float));
    int32_t*       leaves = (int32_t*)malloc(nL * sizeof(int32_t));
    PathRankEntry* rank   = (PathRankEntry*)malloc(nL * sizeof(PathRankEntry));
    if (!qn || !leaves || !rank) {
        free(qn); free(leaves); free(rank); return -1;
    }
    float nrm = 0.0f;
    for (int i = 0; i < dim; i++) nrm += qvec[i] * qvec[i];
    nrm = (nrm > 0.0f) ? 1.0f / sqrtf(nrm) : 1.0f;
    for (int i = 0; i < dim; i++) qn[i] = qvec[i] * nrm;
    int32_t base = leaf_base(qd_eff);
    for (size_t i = 0; i < nL; i++) { leaves[i] = -1; rank[i].score = -1.0f; rank[i].li = (int32_t)i; }

    /* TRAVERSAL PARTAGEE : identical across segments (mêmes seeds). Exécutée
       une seule fois. Contraste avec appel N × mg_query_pathrank qui refait
       traversal N fois. */
    #pragma omp parallel
    {
        float*   _v0     = (float*)malloc((size_t)dim * sizeof(float));
        float*   _v1     = (float*)malloc((size_t)dim * sizeof(float));
        int*     _dims   = (int*)  malloc((size_t)dim * sizeof(int));
        int32_t* _nodes  = (int32_t*)malloc((size_t)n_sets * sizeof(int32_t));
        float*   _scores = (float*)  malloc((size_t)n_sets * sizeof(float));
        if (_v0 && _v1 && _dims && _nodes && _scores) {
            #pragma omp for schedule(static)
            for (int t = 0; t < nt; t++) {
                int cnt;
                if (use_sub) {
                    cnt = traverse_sub_probes_scored(qn, dim, sub, qd_eff, tree_seed(t),
                                                     _v0, _v1, _dims, n_probes, 0,
                                                     _nodes, _scores);
                } else {
                    _nodes[0]  = traverse(qn, dim, qd_eff, tree_seed(t), _v0, _v1);
                    _scores[0] = 0.0f;
                    cnt = 1;
                }
                for (int s = 0; s < n_sets; s++) {
                    size_t li = (size_t)s * nt + t;
                    if (s < cnt) {
                        leaves[li]      = _nodes[s] - base;
                        rank[li].score  = _scores[s];
                        rank[li].li     = (int32_t)li;
                    }
                }
            }
        }
        free(_v0); free(_v1); free(_dims); free(_nodes); free(_scores);
    }
    qsort(rank, nL, sizeof(PathRankEntry), pathrank_cmp_desc);
    for (size_t i = (size_t)top_paths; i < nL; i++) leaves[rank[i].li] = -1;
    free(rank); free(qn);

    /* Compose filter (once, réutilisé par tous segments). */
    roaring_bitmap_t* allowed = NULL;
    if (allowed_state && allowed_state_len > 0) {
        int card = 0;
        allowed = roaring_from_ch_state(allowed_state, (size_t)allowed_state_len, &card);
        (void)card;
        if (!allowed) { free(leaves); return -1; }
    }

    /* COLLECT par segment. Chaque segment retourne au plus top_n candidats.
       On alloue top_n × n_handles buffers scratch pour concat. */
    int32_t* all_ids   = (int32_t*)malloc((size_t)top_n * n_handles * sizeof(int32_t));
    int32_t* all_votes = (int32_t*)malloc((size_t)top_n * n_handles * sizeof(int32_t));
    if (!all_ids || !all_votes) {
        free(all_ids); free(all_votes); free(leaves);
        if (allowed) roaring_bitmap_free(allowed);
        return -1;
    }
    /* PARALLEL collects : chaque segment sur son propre ring (per-Forest),
       aucune contention SQE. Kernel workers io_wq répartissent naturellement.
       Chaque segment écrit dans son slot fixe (top_n × hi) pour éviter les
       races sur l'offset. */
    int* per_seg_n = (int*)calloc((size_t)n_handles, sizeof(int));
    if (!per_seg_n) {
        free(all_ids); free(all_votes); free(leaves);
        if (allowed) roaring_bitmap_free(allowed);
        return -1;
    }
    #pragma omp parallel for schedule(static)
    for (int hi = 0; hi < n_handles; hi++) {
        Forest* fi = (Forest*)handles[hi];
        int n = forest_collect_topn_probes(fi, leaves, n_sets, top_n, qd_eff,
                                           allowed,
                                           all_ids + (size_t)hi * top_n,
                                           all_votes + (size_t)hi * top_n);
        per_seg_n[hi] = (n > 0) ? n : 0;
    }
    if (allowed) roaring_bitmap_free(allowed);
    free(leaves);

    /* Compact non-contiguous slot writes into contiguous total. */
    int total = 0;
    for (int hi = 0; hi < n_handles; hi++) {
        int n = per_seg_n[hi];
        if (n > 0 && hi > 0) {
            memmove(all_ids   + total, all_ids   + (size_t)hi * top_n, (size_t)n * sizeof(int32_t));
            memmove(all_votes + total, all_votes + (size_t)hi * top_n, (size_t)n * sizeof(int32_t));
        }
        total += n;
    }
    free(per_seg_n);

    /* Merge : docs disjoints (chaque doc dans exactement 1 segment), pas de
       somme de votes. Partial sort par vote desc → top top_n. */
    int keep = (total < top_n) ? total : top_n;
    if (keep > 0 && total > keep) {
        /* Build (vote, idx) pairs then sort by vote desc (custom cmp). */
        typedef struct { int32_t vote; int32_t idx; } VoteIdx;
        VoteIdx* vi = (VoteIdx*)malloc((size_t)total * sizeof(VoteIdx));
        if (!vi) { free(all_ids); free(all_votes); return -1; }
        for (int i = 0; i < total; i++) { vi[i].vote = all_votes[i]; vi[i].idx = i; }
        /* qsort with cmp func */
        int cmp_desc(const void* a, const void* b) {
            int32_t va = ((VoteIdx*)a)->vote, vb = ((VoteIdx*)b)->vote;
            return (va < vb) - (va > vb);   /* desc */
        }
        qsort(vi, (size_t)total, sizeof(VoteIdx), cmp_desc);
        for (int i = 0; i < keep; i++) {
            out_ids[i]   = all_ids[vi[i].idx];
            out_votes[i] = vi[i].vote;
        }
        free(vi);
    } else {
        for (int i = 0; i < keep; i++) {
            out_ids[i]   = all_ids[i];
            out_votes[i] = all_votes[i];
        }
    }
    free(all_ids); free(all_votes);
    return keep;
}

int mg_forest_query_multi(void** handles, int n_handles, const float* qvec,
                          int top_n, int query_depth,
                          const uint8_t* allowed_state, int allowed_state_len,
                          int* out_ids, int* out_votes) {
    if (!handles || n_handles <= 0) return -1;
    Forest** fs = (Forest**)handles;
    roaring_bitmap_t* allowed = NULL;
    if (allowed_state && allowed_state_len > 0) {
        int card = 0;
        allowed = roaring_from_ch_state(allowed_state, (size_t)allowed_state_len, &card);
        (void)card;
        if (!allowed) return -1;
    }
    int n = forest_collect_topn_multi(fs, n_handles, qvec, top_n, query_depth,
                                      allowed,
                                      (int32_t*)out_ids, (int32_t*)out_votes);
    if (allowed) roaring_bitmap_free(allowed);
    return n;
}

/* ---------- Telemetry (per-query, thread-local) ---------- */

int mg_last_n_distinct(void) {
    return forest_get_last_n_distinct();
}

int mg_last_n_total(void) {
    return forest_get_last_n_total();
}

/* Per-query wall-clock deadline.
   Pass an ABSOLUTE ns from CLOCK_MONOTONIC (see mg_now_ns()), or 0 to disable.
   Set BEFORE calling any mg_forest_query* / mg_query_pathrank* function.
   When the deadline fires, the query returns EARLY with a partial top-N ;
   mg_last_query_partial() then returns 1. Implemented via checks between
   io_uring phases and inside the merge loop (~every 65k pairs).
   Limitation : in-flight io_uring reads cannot be cancelled mid-batch (would
   require io_uring_prep_cancel + drain) so the effective abort granularity
   is at Phase 1 / Phase 2 boundaries + merge inner loop. */
int64_t mg_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + (int64_t)ts.tv_nsec;
}
void mg_set_query_deadline_ns(int64_t abs_ns) {
    forest_set_query_deadline_ns(abs_ns);
}
int mg_last_query_partial(void) {
    return forest_get_last_query_partial();
}

/* ---------- Forest metadata (read-only getters) ---------- */

int mg_n_trees (void* h) { return h ? ((Forest*)h)->n_trees  : -1; }
int mg_dim     (void* h) { return h ? ((Forest*)h)->dim      : -1; }
int mg_sub_dim (void* h) { return h ? ((Forest*)h)->sub_dim  :  0; }
int mg_depth   (void* h) { return h ? ((Forest*)h)->depth    : -1; }
int mg_n_docs  (void* h) { return h ? ((Forest*)h)->n_docs   : -1; }
int mg_srt_version(void* h) { return h ? ((Forest*)h)->srt_version : -1; }

/* ---------- Generator (random-projection vector) selector ----------
   Must be called before forest_open if the build used a non-default v.   */

void mg_set_gen_version(int v) { set_gen_version(v); }
int  mg_get_gen_version(void) { return get_gen_version(); }

/* Per-tree input subspace (random-subspace ensemble). Must match the value
   the index was built with — read it from meta.txt's `tree_sub` and set it
   before querying, or recall collapses. 0 = disabled (legacy). */
void mg_set_tree_sub(int k) { set_tree_sub(k); }
int  mg_get_tree_sub(void) { return get_tree_sub(); }

/* # distinct per-tree subspaces (0 = per-tree). Match meta.txt's
   tree_sub_groups before querying. */
void mg_set_tree_sub_groups(int g) { set_tree_sub_groups(g); }
int  mg_get_tree_sub_groups(void) { return get_tree_sub_groups(); }

/* Path-permuted dim selection (see traversal.h::set_node_perm). meta_read sets
   this automatically at Forest open; setter exposed so callers building forests
   in-process can opt-in before fan-out. */
void mg_set_node_perm(int on) { set_node_perm(on); }
int  mg_get_node_perm(void)   { return get_node_perm(); }

/* K-way merge tail cap (per-thread). 0 = no cap (default). */
void mg_set_max_distinct(int n) { forest_set_max_distinct(n); }
void mg_set_max_leaf_bytes(uint32_t n) { forest_set_max_leaf_bytes(n); }
int  mg_get_max_distinct(void)  { return forest_get_max_distinct(); }

/* Smarter tail cap: stop after n consecutive non-inserting pushes
   (= heap is stable). 0 = no cap. */
void mg_set_max_stable_rejects(int n) { forest_set_max_stable_rejects(n); }
int  mg_get_max_stable_rejects(void)  { return forest_get_max_stable_rejects(); }

/* Privacy mode: arm thread-local override with client-computed leaves.
   Call mg_forest_query[_multi] AFTER this — qvec is passed but ignored.
   Call mg_clear_external_leaves() to revert to normal mode.              */
void mg_set_external_leaves(const int32_t* leaves, int n) {
    forest_set_external_leaves(leaves, n);
}
void mg_clear_external_leaves(void) {
    forest_set_external_leaves(NULL, 0);
}

/* Enable / disable the shared scratch pool. When enabled, all forests
   in this thread share thread-local bytes_buf and docs_buf — drops
   memory ~10× on multi-segment clusters with no latency cost. */
void mg_set_shared_scratch_pool(int enable) {
    forest_set_shared_scratch_pool(enable);
}
size_t mg_shared_scratch_bytes(void) {
    return forest_shared_scratch_bytes();
}

/* ---------- Test-only exports (varbyte, traverse) ---------- */

#include "varbyte.h"
#include "traversal.h"

/* Encode `v` to `out` (≤5 bytes). Returns bytes written. */
int mg_varbyte_encode(unsigned int v, unsigned char* out) {
    size_t pos = 0;
    varbyte_encode_u32(out, &pos, (uint32_t)v);
    return (int)pos;
}

/* Decode starting at *pos; advances *pos. Returns decoded value. */
unsigned int mg_varbyte_decode(const unsigned char* in, size_t* pos) {
    return (unsigned int)varbyte_decode_u32(in, pos);
}

/* ---- Live-ingest medians (process-global, load-once) ----
   The stateless traversal helpers below run without a Forest handle, so
   they cannot reach f->medians. On a median-built index they would route
   in pure sign-split (θ=0) → a live-inserted doc lands in a leaf the
   query never visits (stranded). The live table gives them the same
   frozen thresholds as build and query. Not loaded = classic sign splits.
   Load once before ingest traffic; an old table is intentionally leaked
   on reload because in-flight traversals on other threads may still read
   it (bounded: one table per reload, ~4-16 MB). */
static const float* g_live_med       = NULL;
static int          g_live_med_depth = 0;
static int          g_live_med_trees = 0;

int mg_live_medians_load(const char* path) {
    int nt = 0, md = 0;
    float* tab = medians_load(path, &nt, &md);
    if (!tab) return -1;
    g_live_med_depth = md;
    g_live_med_trees = nt;
    g_live_med       = tab;
    return md;
}

void mg_live_medians_clear(void) {
    g_live_med = NULL;
    g_live_med_depth = 0;
    g_live_med_trees = 0;
}

int mg_live_medians_depth(void) { return g_live_med ? g_live_med_depth : 0; }

/* Arm the thread-local median table for `tree_idx` — or explicitly reset
   it to NULL. The reset matters even without a live table : the previous
   caller on this thread (e.g. a query) may have left another tree's
   thresholds armed. */
static inline void live_set_medians(int tree_idx) {
    const float* t = g_live_med;
    if (t && tree_idx >= 0 && tree_idx < g_live_med_trees) {
        traversal_set_medians(
            t + (size_t)tree_idx * ((1u << g_live_med_depth) - 1),
            g_live_med_depth);
    } else {
        traversal_set_medians(NULL, 0);
    }
}

/* Run traverse_sub(qvec, dim, sub_dim, depth, tree_seed(tree_idx)).
   Returns the global node id at `depth`. */
int mg_traverse_sub(const float* qvec, int dim, int sub_dim,
                    int depth, int tree_idx) {
    float v0[1024], v1[1024];
    int dims[256];
    if (sub_dim > 256 || dim > 1024) return -1;
    live_set_medians(tree_idx);
    return (int)traverse_sub(qvec, dim, sub_dim, depth, tree_seed(tree_idx),
                              v0, v1, dims);
}

/* Trace path margins : returns the leaf id AND writes the per-level
   |c1 - c0| margins into out_margins (caller-allocated, size depth).
   Pure CPU work (no disk reads) — meant for per-tree quality scoring. */
int mg_trace_margins(const float* qvec, int dim, int sub_dim, int depth,
                     int tree_idx, float* out_margins) {
    float v0[1024], v1[1024];
    int dims[256];
    if (sub_dim > 256 || dim > 1024 || depth > 64) return -1;
    live_set_medians(tree_idx);
    return (int)traverse_sub_trace(qvec, dim, sub_dim, depth,
                                    tree_seed(tree_idx),
                                    v0, v1, dims, out_margins);
}

/* Multi-probe traversal with per-path scores. For (query, tree_idx), runs
   the regular multi-probe (main + n_probes probes) but additionally reports
   per-path "min margin" score (larger = clearer routing → denser leaf).
   out_leaves and out_scores must hold at least n_probes+1 entries each.
   Returns the actual number of paths written (1 + #probes). */
int mg_query_probes_scored(const float* qvec, int dim, int sub_dim, int depth,
                           int tree_idx, int n_probes,
                           int* out_leaves, float* out_scores) {
    float v0[1024], v1[1024];
    int dims[256];
    if (sub_dim > 256 || dim > 1024 || depth > 64) return -1;
    live_set_medians(tree_idx);
    int32_t nodes[64];
    int cnt = traverse_sub_probes_scored(qvec, dim, sub_dim, depth,
                                          tree_seed(tree_idx),
                                          v0, v1, dims,
                                          n_probes, 0,
                                          nodes, out_scores);
    int32_t lbase = (1 << depth) - 1;
    for (int i = 0; i < cnt; i++) out_leaves[i] = (int)(nodes[i] - lbase);
    return cnt;
}

/* Read the doc_ids stored in leaf `leaf_id` of tree `tree_idx`.
   Writes up to `max_n` doc_ids into out_doc_ids, returns count.
   Synchronous pread (single random read into the leaf's data range
   then varbyte-decode if SRT3) — used by the per-tree analysis to
   check which GT neighbours each tree's path-target leaf contains. */
int mg_leaf_docs(void* h, int tree_idx, int leaf_id,
                 int* out_doc_ids, int max_n) {
    if (!h) return -1;
    Forest* f = (Forest*)h;
    if (tree_idx < 0 || tree_idx >= f->n_trees) return -1;
    SortedStore* s = &f->stores[tree_idx];
    if (s->n_nonempty == 0 || leaf_id < 0) return 0;

    /* Locate the leaf via 2-step sparse_index walk : (1) bsearch in the
       resident sample_leaves to find the bucket, (2) read that bucket's
       window of sparse_index entries and find leaf_id inside.            */
    uint32_t bucket = sorted_sample_bucket(s, (uint32_t)leaf_id);
    uint32_t start  = bucket * s->sample_stride;
    if (start >= s->n_nonempty) return 0;
    uint32_t take = s->sample_stride + 1;
    if (start + take > s->n_nonempty + 1) take = s->n_nonempty + 1 - start;
    SparseEntry* win = (SparseEntry*)malloc((size_t)take * SRT_INDEX_ENTRY);
    if (!win) return -1;
    if (pread(s->fd, win, (size_t)take * SRT_INDEX_ENTRY,
              (off_t)(s->index_base + (uint64_t)start * SRT_INDEX_ENTRY))
        != (ssize_t)((size_t)take * SRT_INDEX_ENTRY)) {
        free(win); return -1;
    }
    /* Linear scan for leaf_id (windows are small). */
    int found = -1;
    for (uint32_t i = 0; i < take && win[i].leaf_id != 0xFFFFFFFFu; i++) {
        if (win[i].leaf_id == (uint32_t)leaf_id) { found = (int)i; break; }
        if (win[i].leaf_id > (uint32_t)leaf_id) break;
    }
    if (found < 0) { free(win); return 0; }

    /* Compute the byte/doc range for this leaf. */
    uint32_t off_start = win[found].offset;
    uint32_t off_end;
    if (found + 1 < (int)take && win[found+1].leaf_id != 0xFFFFFFFFu) {
        off_end = win[found+1].offset;
    } else {
        off_end = sorted_is_delta(s) ? s->data_bytes : s->total_docs;
    }
    free(win);
    if (off_end <= off_start) return 0;

    if (!sorted_is_delta(s)) {
        /* SRT2 : raw uint32 doc_ids */
        uint32_t n = off_end - off_start;
        if ((int)n > max_n) n = (uint32_t)max_n;
        if (pread(s->fd, out_doc_ids, (size_t)n * 4,
                  (off_t)(s->data_base + (uint64_t)off_start * 4))
            != (ssize_t)((size_t)n * 4)) return -1;
        return (int)n;
    }

    /* SRT3 : varbyte-delta encoded. Slice = [u32 first_doc][varbyte deltas]. */
    uint32_t slice_len = off_end - off_start;
    uint8_t* slice = (uint8_t*)malloc(slice_len);
    if (!slice) return -1;
    if (pread(s->fd, slice, slice_len,
              (off_t)(s->data_base + (uint64_t)off_start))
        != (ssize_t)slice_len) {
        free(slice); return -1;
    }
    int n = 0;
    if (slice_len < 4) { free(slice); return 0; }
    uint32_t first; memcpy(&first, slice, 4);
    out_doc_ids[n++] = (int)first;
    uint32_t prev = first;
    size_t p = 4;
    while (p < slice_len && n < max_n) {
        uint32_t d = varbyte_decode_u32(slice, &p);
        prev += d;
        out_doc_ids[n++] = (int)prev;
    }
    free(slice);
    return n;
}

int mg_traverse_sub_continue(const float* qvec, int dim, int sub_dim,
                             int start_depth, int n_extra,
                             int start_node, int tree_idx) {
    float v0[1024], v1[1024];
    int dims[256];
    if (sub_dim > 256 || dim > 1024) return -1;
    live_set_medians(tree_idx);
    return (int)traverse_sub_continue(qvec, dim, sub_dim,
                                       start_depth, n_extra,
                                       (int32_t)start_node,
                                       tree_seed(tree_idx),
                                       v0, v1, dims);
}

/* ---------- Integrity verify (full file scan + xxhash check) ----------
   Returns 1 OK, 0 bad hash, -1 IO error.                                 */

int mg_verify_srt(const char* path) {
    return srt_verify_hash(path);
}

/* ---------- Query with raw int32 doc_id filter (no CH state needed) ----
   Convenience for tests / stress benches that build filters in Python.
   `allowed_ids` is a sorted-or-unsorted array of length `n_allowed`.       */

int mg_forest_query_ids(void* h, const float* qvec, int top_n, int query_depth,
                        const int* allowed_ids, int n_allowed,
                        int* out_ids, int* out_votes) {
    if (!h) return -1;
    Forest* f = (Forest*)h;
    roaring_bitmap_t* allowed = NULL;
    if (allowed_ids && n_allowed > 0) {
        allowed = roaring_bitmap_create();
        if (!allowed) return -1;
        for (int i = 0; i < n_allowed; i++) {
            if (allowed_ids[i] >= 0)
                roaring_bitmap_add(allowed, (uint32_t)allowed_ids[i]);
        }
    }
    int n = forest_collect_topn(f, qvec, top_n, query_depth, allowed,
                                (int32_t*)out_ids, (int32_t*)out_votes);
    if (allowed) roaring_bitmap_free(allowed);
    return n;
}

/* ---------- auto_qd_v2 — probe-based query_depth picker ----------
   Two-probe calibration via the live forest. `probe_qvec` is a single
   query vector used to drive both probes. Returns the suggested qd.
   Caller picks `target_ratio` (typical 0.001 = 0.1% of corpus visited).  */

int mg_auto_qd_v2(void* h, const float* probe_qvec,
                  int top_n, double target_ratio,
                  int n_pool, int filter_card) {
    if (!h) return -1;
    Forest* f = (Forest*)h;
    int build_depth = f->depth;

    /* Probe at build_depth (native). */
    int32_t* ids   = (int32_t*)malloc((size_t)top_n * sizeof(int32_t));
    int32_t* votes = (int32_t*)malloc((size_t)top_n * sizeof(int32_t));
    if (!ids || !votes) { free(ids); free(votes); return -1; }
    forest_collect_topn(f, probe_qvec, top_n, build_depth, NULL, ids, votes);
    double probe0 = (double)forest_get_last_n_distinct();

    /* Probe at build_depth - 2. */
    int probe_qd2 = build_depth - 2; if (probe_qd2 < 1) probe_qd2 = 1;
    forest_collect_topn(f, probe_qvec, top_n, probe_qd2, NULL, ids, votes);
    double probe2 = (double)forest_get_last_n_distinct();
    free(ids); free(votes);

    double pool = (filter_card > 0) ? (double)filter_card : (double)n_pool;
    double target = target_ratio * pool;
    if (probe0 <= 0.0 || target <= 0.0) return build_depth;

    double factor = 1.7;
    if (probe2 > probe0 && probe_qd2 < build_depth) {
        double ex = 1.0 / (double)(build_depth - probe_qd2);
        factor = pow(probe2 / probe0, ex);
        if (factor < 1.05) factor = 1.05;
    }
    int levels = (int)ceil(log(target / probe0) / log(factor));
    if (levels < 0) levels = 0;

    int extra = 0;
    if (filter_card > 0 && filter_card < n_pool) {
        double r = (double)n_pool / (double)filter_card;
        extra = (int)floor(log2(r)) - 1;
        if (extra < 0) extra = 0;
    }
    int qd = build_depth - levels - extra;
    if (qd < 1) qd = 1;
    if (qd > build_depth) qd = build_depth;
    return qd;
}

/* ---------- L2 rerank on raw base vectors ----------
   After forest_collect_topn returns candidates, rerank them by exact L2
   distance against the base file. Caller provides the open base fd and
   format (fvecs / bvecs / u8bin auto-detected from path).
   Returns the top_k doc_ids sorted by ascending L2.                       */

int mg_rerank_l2(void* h, const char* base_path,
                 const float* qvec, const int* cand_ids, int n_cands,
                 int top_k, int* out_ids) {
    if (!h) return -1;
    Forest* f = (Forest*)h;
    int bfd = open(base_path, O_RDONLY);
    if (bfd < 0) return -1;
    VecFmt bfmt = vec_fmt_from_path(base_path);
    int n;
    if (f->ring_ok) {
        n = rerank_l2_uring((struct io_uring*)&f->ring, bfd, bfmt, f->dim,
                            qvec, cand_ids, n_cands, top_k,
                            (int32_t*)out_ids);
    } else {
        n = rerank_l2(bfd, bfmt, f->dim, qvec, cand_ids, n_cands, top_k,
                      (int32_t*)out_ids);
    }
    close(bfd);
    return n;
}

/* ---------- Two-stage TurboQuant rerank ----------
   Stage 1: approximate IP on packed 4-bit codes (.tq4 sidecar), keep
   the kprime best candidates. Stage 2: exact L2 rerank of survivors
   against the raw base file. The TqReader is cached per handle on
   first use (one sidecar per forest).                                 */

int mg_rerank_tq(void* h, const char* tq4_path, const char* base_path,
                 const float* qvec, const int* cand_ids, int n_cands,
                 int kprime, int top_k, int* out_ids) {
    if (!h) return -1;
    Forest* f = (Forest*)h;
    if (!f->ring_ok) return -1;          /* io_uring required */

    static TqReader* cached = NULL;       /* process-wide single sidecar */
    static char cached_path[512];
    if (!cached || strncmp(cached_path, tq4_path, sizeof(cached_path)) != 0) {
        if (cached) tq_close(cached);
        cached = tq_open(tq4_path);
        if (!cached) return -1;
        snprintf(cached_path, sizeof(cached_path), "%s", tq4_path);
    }

    int32_t* surv = (int32_t*)malloc((size_t)kprime * sizeof(int32_t));
    if (!surv) return -1;
    int n_surv = tq_select(cached, (struct io_uring*)&f->ring, qvec,
                           (const int32_t*)cand_ids, n_cands, kprime, surv);
    if (n_surv <= 0) { free(surv); return n_surv; }

    int bfd = open(base_path, O_RDONLY);
    if (bfd < 0) { free(surv); return -1; }
    VecFmt bfmt = vec_fmt_from_path(base_path);
    int n = rerank_l2_uring((struct io_uring*)&f->ring, bfd, bfmt, f->dim,
                            qvec, surv, n_surv, top_k, (int32_t*)out_ids);
    close(bfd);
    free(surv);
    return n;
}

/* ---------- Two-stage TQ1 (1-bit) rerank, mirrors mg_rerank_tq. ---------- */
int mg_rerank_tq1(void* h, const char* tq1_path, const char* base_path,
                  const float* qvec, const int* cand_ids, int n_cands,
                  int kprime, int top_k, int* out_ids) {
    if (!h) return -1;
    Forest* f = (Forest*)h;
    if (!f->ring_ok) return -1;

    static Tq1Reader* cached = NULL;
    static char cached_path[512];
    if (!cached || strncmp(cached_path, tq1_path, sizeof(cached_path)) != 0) {
        if (cached) tq1_close(cached);
        cached = tq1_open(tq1_path);
        if (!cached) return -1;
        snprintf(cached_path, sizeof(cached_path), "%s", tq1_path);
    }

    int32_t* surv = (int32_t*)malloc((size_t)kprime * sizeof(int32_t));
    if (!surv) return -1;
    /* Pass the Forest's io_uring-registered fixed buffer to tq1_select_fixed.
     * If the buffer is large enough (n_cands × code_bytes ≤ fixed_buf_size)
     * the read path uses io_uring_prep_read_fixed, eliminating the kernel
     * copy_to_user step (~18 % of total query time in profiles).
     * If the buffer is too small or registration failed at open time, the
     * function transparently falls back to the unregistered malloc path. */
    void* fb_buf    = f->fixed_buf_registered ? f->fixed_buf : NULL;
    size_t fb_size  = f->fixed_buf_registered ? f->fixed_buf_size : 0;
    int n_surv = tq1_select_fixed(cached, (struct io_uring*)&f->ring,
                                  fb_buf, fb_size, 0,
                                  qvec, (const int32_t*)cand_ids, n_cands,
                                  kprime, surv);
    if (n_surv <= 0) { free(surv); return n_surv; }

    int bfd = open(base_path, O_RDONLY);
    if (bfd < 0) { free(surv); return -1; }
    VecFmt bfmt = vec_fmt_from_path(base_path);
    int n = rerank_l2_uring((struct io_uring*)&f->ring, bfd, bfmt, f->dim,
                            qvec, surv, n_surv, top_k, (int32_t*)out_ids);
    close(bfd);
    free(surv);
    return n;
}

/* ---------- Mutations: tombstones ----------
   Caller serializes with respect to queries (forest is not thread-safe).
   `flush` writes the in-memory bitmap to <index_dir>/tombstones.roaring
   atomically. Without flush, deletes are still applied at query time but
   not persisted across restart.                                          */

int mg_tombstone_add   (void* h, unsigned int doc_id) {
    return h ? forest_add_tombstone((Forest*)h, doc_id) : -1;
}
int mg_tombstone_remove(void* h, unsigned int doc_id) {
    return h ? forest_remove_tombstone((Forest*)h, doc_id) : -1;
}
int mg_tombstones_flush(void* h) {
    return h ? forest_save_tombstones((Forest*)h) : -1;
}
int mg_tombstones_count(void* h) {
    return h ? forest_tombstone_count((Forest*)h) : -1;
}

/* Calibrate + write <index_dir>/medians.bin (top-level median thresholds).
   Run BEFORE the build ; build routing and forest_open pick it up.        */
int mg_calibrate_medians(const char* vecs_path, const char* index_dir,
                         int n_trees, int dim, int sub_dim,
                         int med_depth, int sample_n) {
    return calibrate_and_save_medians(vecs_path, index_dir, n_trees, dim,
                                      sub_dim, med_depth, sample_n);
}
