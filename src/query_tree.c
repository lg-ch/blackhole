#define _POSIX_C_SOURCE 200809L
#include "query_tree.h"
#include "liburing_compat.h"
#include "sorted_store.h"
#include "build_tree.h"
#include "traversal.h"
#include "gen_vec.h"
#include "varbyte.h"
#include "tombstones.h"

#include <roaring/roaring.h>

#include <stdio.h>
#include <stdlib.h>
#include <sys/uio.h>           /* struct iovec for io_uring_register_buffers */
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/resource.h>

static int raise_fd_limit(int target) {
    struct rlimit rl;
    if (getrlimit(RLIMIT_NOFILE, &rl) != 0) return -1;
    if ((long)rl.rlim_cur >= (long)target) return 0;
    rl.rlim_cur = ((rlim_t)target < rl.rlim_max) ? (rlim_t)target : rl.rlim_max;
    return setrlimit(RLIMIT_NOFILE, &rl);
}

typedef struct {
    const uint32_t* docs;
    uint32_t        len;
    uint32_t        pos;
    uint32_t        doc;
    roaring_uint32_iterator_t* iter;  /* NULL when no filter */
} TCursor;

/* Advance cursor `c` to the next position whose doc ∈ filter, skipping
   non-allowed ids in O(log K + log L) instead of iterating every event.
   Uses the cursor's persistent roaring iterator (forward-monotonic) plus a
   binary search inside the cursor's sorted doc list.
   Returns 1 on success (c->doc holds an allowed id), 0 if exhausted.    */
static inline int cursor_seek_allowed(TCursor* c, const roaring_bitmap_t* filter) {
    if (!filter) {
        if (c->pos >= c->len) return 0;
        c->doc = c->docs[c->pos];
        return 1;
    }
    roaring_uint32_iterator_t* it = c->iter;
    while (c->pos < c->len) {
        uint32_t doc = c->docs[c->pos];
        if (!it->has_value || it->current_value < doc) {
            roaring_move_uint32_iterator_equalorlarger(it, doc);
        }
        if (!it->has_value) return 0;
        uint32_t a = it->current_value;
        if (a == doc) { c->doc = doc; return 1; }
        /* binary search forward to first c->docs[i] >= a   */
        size_t lo = c->pos + 1, hi = c->len;
        while (lo < hi) {
            size_t mid = (lo + hi) >> 1;
            if (c->docs[mid] < a) lo = mid + 1;
            else hi = mid;
        }
        c->pos = lo;
    }
    return 0;
}

typedef struct { int32_t id; int32_t vote; } IdVote;

static int cmp_uint32(const void* a, const void* b) {
    uint32_t x = *(const uint32_t*)a, y = *(const uint32_t*)b;
    return (x < y) ? -1 : (x > y);
}

/* CRoaring membership probe wrapped so signedness checks live in one place. */
static inline int allowed_contains(const roaring_bitmap_t* rb, int32_t doc) {
    if (doc < 0) return 0;
    return roaring_bitmap_contains(rb, (uint32_t)doc);
}

/* Per-query telemetry: number of *distinct* candidates the K-way merge
   visited before capping to top_n. High distinct count vs trees ⇒ NNs are
   dispersed across leaves (low concentration ⇒ likely needs lower qd).
   Thread-local so multitopn/parallel runs stay isolated.                 */
static __thread int g_last_n_distinct = 0;
int forest_get_last_n_distinct(void) { return g_last_n_distinct; }

/* Per-query : total candidates read (sum of lens[li] across probe leaves,
   BEFORE dedup by tree). Diff with n_distinct = redundancy across trees. */
static __thread int g_last_n_total = 0;
int forest_get_last_n_total(void) { return g_last_n_total; }

/* Per-query tail-cap: stop the K-way merge once we've fully counted
   `g_max_distinct` distinct candidates. 0 = unlimited (default).
   Hard-caps p99 latency on heavy queries at the cost of recall on the
   tail of candidates that would otherwise rank in top_n. Thread-local.   */
static __thread int g_max_distinct = 0;
void forest_set_max_distinct(int n) { g_max_distinct = (n > 0) ? n : 0; }
int  forest_get_max_distinct(void) { return g_max_distinct; }

/* Per-query tail cap: leaves whose posting list exceeds this many bytes
   (after sparse_index lookup, before Phase-2 read) are skipped entirely.
   Bounds p99 latency at the cost of recall on queries that route into
   degenerate dense leaves. 0 = no cap (default). Bytes, not docs —
   we have the byte length cheap from the sparse offsets diff. */
static __thread uint32_t g_max_leaf_bytes = 0;
void forest_set_max_leaf_bytes(uint32_t n) { g_max_leaf_bytes = n; }
uint32_t forest_get_max_leaf_bytes(void) { return g_max_leaf_bytes; }

/* Per-query wall-clock deadline. Set to CLOCK_MONOTONIC absolute ns before
   calling forest_collect_topn_probes. When the check helper trips, the
   collect loop returns EARLY with whatever's already accumulated in the
   top-N heap (marked partial). 0 = disabled (no deadline). */
#include <time.h>
static __thread int64_t g_query_deadline_ns = 0;
static __thread int     g_last_query_partial = 0;
void  forest_set_query_deadline_ns(int64_t abs_ns) { g_query_deadline_ns = abs_ns; }
int   forest_get_last_query_partial(void) { return g_last_query_partial; }

/* HOT overlay pointer (thread-local, opt-in). When non-NULL, forest_collect_topn_probes
   merges in docs from HOT for every visited (tree, storage_leaf) alongside the
   SRT MAIN docs, then radix + top-N picks over the union. NULL = MAIN-only. */
#include "hot_store.h"
static __thread const HotOverlay* g_hot_overlay = NULL;
void  forest_set_hot_overlay(const HotOverlay* h) { g_hot_overlay = h; }
const HotOverlay* forest_get_hot_overlay(void) { return g_hot_overlay; }

/* Per-phase wall-clock profiling of the last forest_collect_topn_probes call.
   Thread-local; overwritten each query. Indices :
     0 = Phase 1 (sparse window reads)        1 = resolve offsets
     2 = Phase 2 (leaf data reads)            3 = varbyte decode
     4 = radix pack (incl. HOT merge)         5 = radix sort passes
     6 = linear scan + top-N pick             7 = total in-function     */
#define QT_N_PHASES 8
static __thread double g_phase_ms[QT_N_PHASES];
double forest_get_phase_ms(int i) {
    return (i >= 0 && i < QT_N_PHASES) ? g_phase_ms[i] : -1.0;
}
static inline double phase_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

static inline int64_t deadline_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + (int64_t)ts.tv_nsec;
}
static inline int deadline_hit(void) {
    if (g_query_deadline_ns == 0) return 0;
    return deadline_now_ns() >= g_query_deadline_ns ? 1 : 0;
}

/* Smarter cap: stop after `g_max_stable_rejects` CONSECUTIVE pushes that
   didn't enter the (full) top_n heap. Once true NNs are all in the heap
   (their final vote count beats heap[0]), subsequent pushes are all
   below-threshold — exit as soon as the run is long enough. Preserves
   recall much better than the raw n_distinct cap. 0 = disabled.        */
static __thread int g_max_stable_rejects = 0;

/* Privacy mode : if set, forest_collect_topn skips traversal and uses
   these caller-supplied per-tree leaf_ids. NULL = traverse from qvec as usual.
   Thread-local so concurrent clients can independently opt in.            */
static __thread const int32_t* g_external_leaves = NULL;
static __thread int g_external_leaves_count = 0;
void forest_set_external_leaves(const int32_t* leaves, int n) {
    g_external_leaves = leaves;
    g_external_leaves_count = n;
}

/* Shared per-thread scratch pool for the large per-query buffers
   (bytes_buf and docs_buf). When multiple Forests are queried serially
   in the same thread (typical multi-segment cluster), we want to share
   these buffers instead of every Forest holding its own copy — they're
   the biggest RAM consumer (~300-500 MB at SIFT 1B segment scale).
   Allocation policy: grow on demand, never shrink. Freed once at
   program exit (or never; pool is 1-2 chunks per thread). */
static __thread uint8_t*  g_shared_bytes_buf  = NULL;
static __thread uint32_t  g_shared_bytes_cap  = 0;
static __thread uint32_t* g_shared_docs_buf   = NULL;
static __thread uint32_t  g_shared_docs_cap   = 0;
static __thread int       g_use_shared_pool   = 0;

/* Radix-sort scratch pool : pairs[] and scratch[] (each uint64) ; cleared
   via realloc-on-grow. Sized by N (total entries in the merge), typically
   30k-100k uint64 per query. Pooled per-thread to avoid the malloc/free
   round-trip + kernel page_clear on every query. */
static __thread uint64_t* g_radix_pairs_buf   = NULL;
static __thread uint64_t* g_radix_scratch_buf = NULL;
static __thread size_t    g_radix_cap         = 0;
static int radix_ensure_cap(size_t needed) {
    if (g_radix_cap >= needed) return 0;
    size_t new_cap = g_radix_cap ? g_radix_cap : 4096;
    while (new_cap < needed) new_cap <<= 1;
    uint64_t* p = (uint64_t*)realloc(g_radix_pairs_buf,   new_cap * sizeof(uint64_t));
    if (!p) return -1;
    g_radix_pairs_buf = p;
    uint64_t* s = (uint64_t*)realloc(g_radix_scratch_buf, new_cap * sizeof(uint64_t));
    if (!s) return -1;
    g_radix_scratch_buf = s;
    g_radix_cap = new_cap;
    return 0;
}

/* Probe-collect decode scratch : docs (uint32) + bytes (varbyte).
   Thread-local, grow-only. Avoids the malloc/free churn each query at
   shallow-depth workloads where total_units is large (100M+ docs). */
static __thread uint32_t* g_probes_docs_buf   = NULL;
static __thread size_t    g_probes_docs_cap   = 0;
static __thread uint8_t*  g_probes_bytes_buf  = NULL;
static __thread size_t    g_probes_bytes_cap  = 0;
static int probes_docs_ensure_cap(size_t needed) {
    if (g_probes_docs_cap >= needed) return 0;
    size_t new_cap = g_probes_docs_cap ? g_probes_docs_cap : 4096;
    while (new_cap < needed) new_cap <<= 1;
    uint32_t* p = (uint32_t*)realloc(g_probes_docs_buf, new_cap * sizeof(uint32_t));
    if (!p) return -1;
    g_probes_docs_buf = p; g_probes_docs_cap = new_cap;
    return 0;
}
static int probes_bytes_ensure_cap(size_t needed) {
    if (g_probes_bytes_cap >= needed) return 0;
    size_t new_cap = g_probes_bytes_cap ? g_probes_bytes_cap : 4096;
    while (new_cap < needed) new_cap <<= 1;
    uint8_t* p = (uint8_t*)realloc(g_probes_bytes_buf, new_cap);
    if (!p) return -1;
    g_probes_bytes_buf = p; g_probes_bytes_cap = new_cap;
    return 0;
}

/* tree_seen[] scratch (one int32 per tree). Reset to -1 between queries. */
static __thread int32_t* g_tree_seen_buf = NULL;
static __thread int      g_tree_seen_cap = 0;
static int tree_seen_ensure_cap(int nt) {
    if (g_tree_seen_cap >= nt) return 0;
    int new_cap = g_tree_seen_cap ? g_tree_seen_cap : 256;
    while (new_cap < nt) new_cap <<= 1;
    int32_t* p = (int32_t*)realloc(g_tree_seen_buf, (size_t)new_cap * sizeof(int32_t));
    if (!p) return -1;
    g_tree_seen_buf = p;
    g_tree_seen_cap = new_cap;
    return 0;
}
void forest_set_shared_scratch_pool(int enable) {
    g_use_shared_pool = enable ? 1 : 0;
}
size_t forest_shared_scratch_bytes(void) {
    return (size_t)g_shared_bytes_cap +
           (size_t)g_shared_docs_cap * sizeof(uint32_t);
}
void forest_set_max_stable_rejects(int n) {
    g_max_stable_rejects = (n > 0) ? n : 0;
}
int  forest_get_max_stable_rejects(void) { return g_max_stable_rejects; }

int forest_open(Forest* f, const char* index_dir,
                int n_trees, int dim, int sub_dim,
                int depth, int n_docs) {
    raise_fd_limit(n_trees + 16);

    f->n_trees = n_trees;
    f->dim     = dim;
    f->sub_dim = sub_dim;
    f->depth   = depth;
    f->n_docs  = n_docs;
    f->stores  = (SortedStore*)calloc((size_t)n_trees, sizeof(SortedStore));
    if (!f->stores) return -1;

    /* Size the ring for n_trees × 3 ops (2 phase-1 windows + 1 phase-2 data).
       Add 25% headroom, clamp to [256, 32768] (kernel max default).        */
    unsigned ring_n = (unsigned)(n_trees * 3 + n_trees);
    if (ring_n < 256)   ring_n = 256;
    if (ring_n > 32768) ring_n = 32768;
    int rc = io_uring_queue_init(ring_n, &f->ring, 0);
    f->ring_ok = (rc == 0);
    if (!f->ring_ok) fprintf(stderr, "warning: io_uring init failed (rc=%d)\n", rc);

    char path[512];
    uint32_t max_stride = 0;
    uint32_t first_magic = 0;
    for (int t = 0; t < n_trees; t++) {
        snprintf(path, sizeof(path), "%s/tree%05d.srt", index_dir, t);
        if (sorted_store_open_rdonly(&f->stores[t], path) != 0) {
            fprintf(stderr, "open %s: ", path); perror("");
            for (int j = 0; j < t; j++) sorted_store_close(&f->stores[j]);
            free(f->stores);
            return -1;
        }
        if (t == 0) first_magic = f->stores[t].magic;
        else if (f->stores[t].magic != first_magic) {
            fprintf(stderr, "forest_open: mixed SRT versions in %s "
                    "(tree 0 = 0x%08x, tree %d = 0x%08x)\n",
                    index_dir, first_magic, t, f->stores[t].magic);
            for (int j = 0; j <= t; j++) sorted_store_close(&f->stores[j]);
            free(f->stores);
            return -1;
        }
        if (f->stores[t].sample_stride > max_stride)
            max_stride = f->stores[t].sample_stride;
    }
    f->srt_version  = (first_magic == SRT_MAGIC_V3) ? 3 : 2;
    f->max_window_n = max_stride + 1;

    /* Tombstones: try to load (may be absent). */
    f->tombstones = tombstones_load(index_dir);
    {   /* Optional top-level medians (must match the build's). */
        char mpath[600];
        snprintf(mpath, sizeof(mpath), "%s/medians.bin", index_dir);
        int mnt = 0, mmd = 0;
        float* tab = medians_load(mpath, &mnt, &mmd);
        if (tab && mnt == f->n_trees) { f->medians = tab; f->med_depth = mmd; }
        else { free(tab); f->medians = NULL; f->med_depth = 0; }
    }
    strncpy(f->index_dir, index_dir, sizeof(f->index_dir) - 1);
    f->index_dir[sizeof(f->index_dir) - 1] = '\0';

    /* Pre-allocate per-query scratch. 2 windows per tree (low/high). */
    size_t nt = (size_t)n_trees;
    f->wbuf         = malloc(nt * 2 * f->max_window_n * sizeof(SparseEntry));
    f->wlen_buf     = (uint32_t*)malloc(nt * 2 * sizeof(uint32_t));
    f->woff_buf     = (uint64_t*)malloc(nt * 2 * sizeof(uint64_t));
    f->leaves_buf   = (int32_t*) malloc(nt * sizeof(int32_t));
    f->lens_buf     = (uint32_t*)malloc(nt * sizeof(uint32_t));
    f->data_off_buf = (uint64_t*)malloc(nt * sizeof(uint64_t));
    f->buf_pos_buf  = (uint32_t*)malloc(nt * sizeof(uint32_t));
    f->cursors_buf  = calloc(nt, sizeof(TCursor));
    f->heap_buf     = (void**)malloc(nt * sizeof(void*));
    f->topn_h_buf   = NULL;
    f->topn_h_cap   = 0;
    f->docs_buf     = NULL;
    f->docs_cap     = 0;
    f->bytes_buf    = NULL;
    f->bytes_cap    = 0;
    f->byte_pos_buf = (uint32_t*)malloc(nt * sizeof(uint32_t));

    if (!f->wbuf || !f->wlen_buf || !f->woff_buf || !f->leaves_buf
        || !f->lens_buf || !f->data_off_buf || !f->buf_pos_buf
        || !f->cursors_buf || !f->heap_buf) {
        fprintf(stderr, "forest_open: scratch alloc failed\n");
        forest_close(f); return -1;
    }

    /* io_uring registered buffer for TQ1 stage 1.
       8 MB covers top_n × code_bytes up to ~64k codes at 128 B/code.
       Page-aligned (4 KB) for the kernel pin path. */
    f->fixed_buf            = NULL;
    f->fixed_buf_size       = 0;
    f->fixed_buf_registered = 0;
    if (f->ring_ok) {
        const size_t FIXED = (size_t)8 * 1024 * 1024;
        void* fb = NULL;
        if (posix_memalign(&fb, 4096, FIXED) == 0 && fb) {
            struct iovec iov = { .iov_base = fb, .iov_len = FIXED };
            int rrc = io_uring_register_buffers(&f->ring, &iov, 1);
            if (rrc == 0) {
                f->fixed_buf            = fb;
                f->fixed_buf_size       = FIXED;
                f->fixed_buf_registered = 1;
            } else {
                free(fb);
                fprintf(stderr, "warning: io_uring_register_buffers failed "
                                "(rc=%d), fallback to unregistered reads\n", rrc);
            }
        }
    }
    return 0;
}

/* Reopen a single tree's SRT after external replacement (compaction).
 * Caller must serialize against concurrent queries on this forest. */
int forest_reopen_tree(Forest* f, int tree_id) {
    if (!f || tree_id < 0 || tree_id >= f->n_trees) return -1;
    sorted_store_close(&f->stores[tree_id]);
    char path[512];
    snprintf(path, sizeof(path), "%s/tree%05d.srt", f->index_dir, tree_id);
    return sorted_store_open_rdonly(&f->stores[tree_id], path);
}

void forest_close(Forest* f) {
    if (f->ring_ok) {
        if (f->fixed_buf_registered) {
            io_uring_unregister_buffers(&f->ring);
            f->fixed_buf_registered = 0;
        }
        io_uring_queue_exit(&f->ring); f->ring_ok = 0;
    }
    if (f->fixed_buf) {
        free(f->fixed_buf);
        f->fixed_buf = NULL; f->fixed_buf_size = 0;
    }
    if (f->stores) {
        for (int t = 0; t < f->n_trees; t++) sorted_store_close(&f->stores[t]);
        free(f->stores); f->stores = NULL;
    }
    free(f->wbuf);         f->wbuf         = NULL;
    free(f->wlen_buf);     f->wlen_buf     = NULL;
    free(f->woff_buf);     f->woff_buf     = NULL;
    free(f->leaves_buf);   f->leaves_buf   = NULL;
    free(f->lens_buf);     f->lens_buf     = NULL;
    free(f->data_off_buf); f->data_off_buf = NULL;
    free(f->buf_pos_buf);  f->buf_pos_buf  = NULL;
    free(f->cursors_buf);  f->cursors_buf  = NULL;
    free(f->heap_buf);     f->heap_buf     = NULL;
    free(f->topn_h_buf);    f->topn_h_buf    = NULL;
    free(f->docs_buf);      f->docs_buf      = NULL;
    free(f->bytes_buf);     f->bytes_buf     = NULL;
    free(f->byte_pos_buf);  f->byte_pos_buf  = NULL;
    free(f->medians); f->medians = NULL;
    if (f->tombstones) {
        roaring_bitmap_free(f->tombstones);
        f->tombstones = NULL;
    }
}

/* ---------- Mutation API: tombstones ---------- */

int forest_add_tombstone(Forest* f, uint32_t doc_id) {
    if (!f) return -1;
    if (!f->tombstones) f->tombstones = roaring_bitmap_create();
    if (!f->tombstones) return -1;
    roaring_bitmap_add(f->tombstones, doc_id);
    return 0;
}

int forest_remove_tombstone(Forest* f, uint32_t doc_id) {
    if (!f || !f->tombstones) return 0;
    roaring_bitmap_remove(f->tombstones, doc_id);
    return 0;
}

int forest_save_tombstones(Forest* f) {
    if (!f || !f->tombstones) return 0;
    return tombstones_save(f->index_dir, f->tombstones);
}

int forest_tombstone_count(const Forest* f) {
    if (!f || !f->tombstones) return 0;
    return (int)roaring_bitmap_get_cardinality(f->tombstones);
}

/* Lower-bound binary search: returns the smallest position `i` such that
   window[i].leaf_id >= target, or `n` if no such entry exists.            */
static uint32_t lb_in_window(const SparseEntry* window, uint32_t n,
                             uint32_t target) {
    uint32_t lo = 0, hi = n;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (window[mid].leaf_id < target) lo = mid + 1;
        else                              hi = mid;
    }
    return lo;
}

/* Legacy votes-based search left as stub: only the topn path is used. */
int forest_search(const Forest* f, const float* qvec,
                  int top_k, int threshold,
                  Result* out, uint16_t* votes) {
    (void)f; (void)qvec; (void)top_k; (void)threshold; (void)out; (void)votes;
    fprintf(stderr, "forest_search: legacy path disabled with sparse layout\n");
    return 0;
}

int forest_collect(const Forest* f, const float* qvec,
                   int threshold, int max_out,
                   int32_t* out_ids, int32_t* out_votes,
                   uint16_t* votes) {
    (void)f; (void)qvec; (void)threshold; (void)max_out;
    (void)out_ids; (void)out_votes; (void)votes;
    fprintf(stderr, "forest_collect: legacy path disabled with sparse layout\n");
    return 0;
}

/* ---------- 2-phase io_uring + K-way merge (sparse offset table) ---------- */

static void cheap_sift_down(TCursor** h, int n, int i) {
    while (1) {
        int l = 2*i + 1, r = 2*i + 2, s = i;
        if (l < n && h[l]->doc < h[s]->doc) s = l;
        if (r < n && h[r]->doc < h[s]->doc) s = r;
        if (s == i) break;
        TCursor* t = h[i]; h[i] = h[s]; h[s] = t;
        i = s;
    }
}

/* ---------- Loser tree (Knuth, TAOCP vol 3) for cache-friendly K-way merge.

   Replaces the pointer-heap (cheap_sift_down) which dominated single-CPU
   profile time (~71%) due to pointer chasing across ~28 k cursors. The tree
   stores at each interior node the loser of the comparison between its
   child subtrees ; the overall winner is tracked separately. Pop = O(log K)
   comparisons against a CONTIGUOUS doc_cache[] array (accessed by integer
   index), eliminating per-cursor pointer dereferences.

   Conventions:
     K_pad           = nearest power of 2 ≥ (K real cursors)
     doc_cache[0..K) = real cursors' current doc value
     doc_cache[K..K_pad] = sentinel UINT32_MAX (so the tree terminates
                          naturally when all real cursors are exhausted)
     losers[1..K_pad-1] = internal nodes (1-indexed for clean parent math)
     winner          = current overall winner cursor index                */
typedef struct {
    int K_pad;
    int winner;
    int* losers;
    uint32_t* doc_cache;
} LoserTree;

static int lt_build_subtree(LoserTree* lt, int node) {
    /* Post-order build: returns the winner cursor of the subtree rooted at
       `node`. Leaves are encoded as nodes ≥ K_pad ; their winner is the
       corresponding cursor index. Internal nodes record the loser of the
       comparison between their children's winners. */
    if (node >= lt->K_pad) return node - lt->K_pad;
    int lw = lt_build_subtree(lt, 2 * node);
    int rw = lt_build_subtree(lt, 2 * node + 1);
    if (lt->doc_cache[lw] <= lt->doc_cache[rw]) {
        lt->losers[node] = rw;
        return lw;
    }
    lt->losers[node] = lw;
    return rw;
}

static void lt_init(LoserTree* lt) {
    lt->winner = lt_build_subtree(lt, 1);
}

static inline int lt_pop_advance(LoserTree* lt) {
    /* doc_cache[lt->winner] must already be updated to the cursor's NEXT
       doc value (or UINT32_MAX if exhausted). Walks from the (former)
       winner's leaf up to root, swapping with stored losers as needed. */
    int w = lt->winner;
    int pos = (w + lt->K_pad) >> 1;
    while (pos >= 1) {
        int loser = lt->losers[pos];
        if (lt->doc_cache[loser] < lt->doc_cache[w]) {
            lt->losers[pos] = w;
            w = loser;
        }
        pos >>= 1;
    }
    lt->winner = w;
    return w;
}

static void th_sift_down(IdVote* h, int n, int i) {
    while (1) {
        int l = 2*i + 1, r = 2*i + 2, s = i;
        if (l < n && h[l].vote < h[s].vote) s = l;
        if (r < n && h[r].vote < h[s].vote) s = r;
        if (s == i) break;
        IdVote t = h[i]; h[i] = h[s]; h[s] = t;
        i = s;
    }
}
static void th_sift_up(IdVote* h, int i) {
    while (i > 0) {
        int p = (i - 1) / 2;
        if (h[p].vote <= h[i].vote) break;
        IdVote t = h[p]; h[p] = h[i]; h[i] = t;
        i = p;
    }
}
static void topn_push(IdVote* heap, int* size, int cap,
                      int32_t id, int32_t vote) {
    if (*size < cap) {
        heap[*size] = (IdVote){id, vote};
        th_sift_up(heap, *size);
        (*size)++;
    } else if (vote > heap[0].vote) {
        heap[0] = (IdVote){id, vote};
        th_sift_down(heap, *size, 0);
    }
}

/* ---------------------------------------------------------------------------
 * Histogram-based top-n selector — alternative to the binary-heap topn_push.
 *
 * Idea : votes are small integers (≤ n_trees, typically 256). Append every
 * distinct (doc, vote) into a flat buffer while bumping a vote histogram.
 * After the scan, walk the histogram from the highest vote downward to find
 * the threshold T such that #(vote > T) ≤ cap ≤ #(vote ≥ T), then a single
 * pass over the buffer emits the top-n. Two linear passes total, no heap
 * sifts — typically 5-8× cheaper than the heap path when distinct_docs is
 * in the 100k–500k range and cap is 16k–64k.
 *
 * `pairs[0..n)`   : (id, vote) for every distinct doc seen, in scan order.
 * `hist[0..n_trees]` : count of distinct docs at each vote level.
 * `out_ids`, `out_votes` : caller-owned arrays of length ≥ cap.
 * Returns the number of pairs emitted (≤ cap).                              */
static int topn_pick_hist(const IdVote* pairs, size_t n,
                          const uint32_t* hist, int max_vote,
                          int cap, int32_t* out_ids, int32_t* out_votes) {
    if (cap <= 0 || n == 0) return 0;
    /* 1) Find threshold T : highest vote level s.t. cum count crosses cap. */
    uint32_t cum = 0;
    int T = max_vote;
    while (T > 0 && cum + hist[T] <= (uint32_t)cap) {
        cum += hist[T];
        T--;
    }
    /* cum = # docs at vote > T (all kept).  cap - cum = # docs at vote == T
       to keep (any first-in-buffer; tie-broken by scan order, matches the
       heap variant's arbitrary tie-break).                                */
    int remaining_at_T = (int)((uint32_t)cap - cum);
    int out = 0;
    for (size_t i = 0; i < n && out < cap; i++) {
        int32_t v = pairs[i].vote;
        if (v > T) {
            out_ids[out] = pairs[i].id; out_votes[out] = v; out++;
        } else if (v == T && remaining_at_T > 0) {
            out_ids[out] = pairs[i].id; out_votes[out] = v; out++;
            remaining_at_T--;
        }
    }
    return out;
}

int forest_collect_topn(const Forest* f, const float* qvec,
                        int top_n, int query_depth,
                        const roaring_bitmap_t* allowed,
                        int32_t* out_ids, int32_t* out_votes) {
    if (top_n <= 0) return 0;
    int nt = f->n_trees;
    int use_sub = (f->sub_dim > 0 && f->sub_dim <= f->dim);
    int sd = use_sub ? f->sub_dim : f->dim;

    /* Compose effective filter: (allowed AND NOT tombstones).
       If no tombstones, effective == allowed; if no caller filter but we
       have tombstones, effective = NOT tombstones (flipped over corpus).  */
    const roaring_bitmap_t* effective = allowed;
    roaring_bitmap_t* composed = NULL;
    if (f->tombstones && roaring_bitmap_get_cardinality(f->tombstones) > 0) {
        if (allowed) {
            composed = roaring_bitmap_andnot(allowed, f->tombstones);
        } else {
            composed = roaring_bitmap_flip(f->tombstones, 0,
                                           (uint64_t)f->n_docs);
        }
        effective = composed;
    }
    allowed = effective;

    /* Default query_depth to build depth (native query). */
    int qd = (query_depth > 0 && query_depth <= f->depth) ? query_depth : f->depth;
    int k_shift = f->depth - qd;   /* range size at build depth = 1 << k_shift */

    /* Small per-query scratch (cheap). */
    float v0[1024], v1[1024];
    float qn[1024];
    int   dims[256];
    if (sd > 1024 || f->dim > 1024 || (use_sub && f->sub_dim > 256)) {
        fprintf(stderr, "dim too large for stack scratch\n");
        return -1;
    }
    memcpy(qn, qvec, (size_t)f->dim * sizeof(float));
    normalize(qn, f->dim);

    /* Persistent scratch from Forest. */
    Forest* fm = (Forest*)f;
    SparseEntry* wbuf      = (SparseEntry*)fm->wbuf;
    uint32_t*    wlen      = fm->wlen_buf;     /* size 2*nt: slot 0=low, 1=high */
    uint64_t*    woff      = fm->woff_buf;
    int32_t*     leaves    = fm->leaves_buf;
    uint32_t*    lens      = fm->lens_buf;
    uint64_t*    data_off  = fm->data_off_buf;
    uint32_t*    buf_pos   = fm->buf_pos_buf;
    TCursor*     cursors   = (TCursor*) fm->cursors_buf;
    TCursor**    heap      = (TCursor**)fm->heap_buf;
    uint32_t     window_n  = fm->max_window_n;

    if (fm->topn_h_cap < top_n) {
        free(fm->topn_h_buf);
        fm->topn_h_buf = malloc((size_t)top_n * sizeof(IdVote));
        fm->topn_h_cap = top_n;
    }
    IdVote* topn_h = (IdVote*)fm->topn_h_buf;

    /* Traverse each tree to query_depth — unless the caller provided
       pre-computed leaves (privacy mode : client-side traversal). */
    int32_t base_q = leaf_base(qd);
    if (g_external_leaves != NULL && g_external_leaves_count >= nt) {
        for (int t = 0; t < nt; t++) leaves[t] = g_external_leaves[t];
    } else {
        for (int t = 0; t < nt; t++) {
            int32_t leaf = use_sub
                ? traverse_sub(qn, f->dim, f->sub_dim, qd, tree_seed(t),
                               v0, v1, dims)
                : traverse(qn, f->dim, qd, tree_seed(t), v0, v1);
            leaves[t] = leaf - base_q;
        }
    }

    /* Phase 1: per tree, read two windows of sparse_index — one covering
       the bucket for low_leaf, one for high_leaf. When buckets coincide
       we'd be reading the same range twice — accepted overhead.          */
    struct io_uring* ring = (struct io_uring*)&f->ring;
    int phase1_submits = 0;
    for (int t = 0; t < nt; t++) {
        const SortedStore* s = &f->stores[t];
        uint32_t low_leaf  = (uint32_t)leaves[t] << k_shift;
        uint32_t high_leaf = ((uint32_t)leaves[t] + 1u) << k_shift;
        uint32_t stride = s->sample_stride;

        for (int slot = 0; slot < 2; slot++) {
            uint32_t target  = (slot == 0) ? low_leaf : high_leaf;
            size_t   idx     = (size_t)t * 2 + slot;
            wlen[idx] = 0; woff[idx] = 0;
            if (s->n_nonempty == 0) continue;
            uint32_t bucket = sorted_sample_bucket(s, target);
            uint32_t start_pos = bucket * stride;
            if (start_pos >= s->n_nonempty) continue;
            uint32_t take = stride + 1;
            if (start_pos + take > s->n_nonempty + 1)
                take = s->n_nonempty + 1 - start_pos;
            wlen[idx] = take;
            woff[idx] = s->index_base + (uint64_t)start_pos * SRT_INDEX_ENTRY;

            if (f->ring_ok) {
                struct io_uring_sqe* sqe = io_uring_get_sqe(ring);
                if (!sqe) { io_uring_submit(ring); sqe = io_uring_get_sqe(ring); }
                io_uring_prep_read(sqe, s->fd,
                                   wbuf + idx * window_n,
                                   (unsigned)(take * SRT_INDEX_ENTRY),
                                   (uint64_t)woff[idx]);
                io_uring_sqe_set_data(sqe, (void*)(uintptr_t)idx);
                phase1_submits++;
            } else {
                ssize_t pr1 = pread(s->fd, wbuf + idx * window_n,
                                    take * SRT_INDEX_ENTRY, (off_t)woff[idx]);
                (void)pr1;
            }
        }
    }
    if (f->ring_ok) {
        io_uring_submit(ring);
        for (int r = 0; r < phase1_submits; r++) {
            struct io_uring_cqe* cqe;
            if (io_uring_wait_cqe(ring, &cqe) < 0) continue;
            io_uring_cqe_seen(ring, cqe);
        }
    }

    /* Resolve (start, end) offsets per tree via two lower-bound searches.
       SRT3 (delta-VarByte): offsets in `data_off` and lengths in `lens`
       refer to BYTES during phase 2; `lens` becomes decoded doc count after.
       SRT2 (legacy uint32): offsets/lengths in DOCS; data_off pre-multiplied. */
    int is_v3 = (f->srt_version == 3);
    uint64_t total_bytes = 0;     /* SRT3 only */
    uint64_t total_docs_unit = 0; /* SRT2: docs total; SRT3: same as bytes (upper bound) */
    for (int t = 0; t < nt; t++) {
        lens[t] = 0; data_off[t] = 0;
        buf_pos[t] = is_v3 ? (uint32_t)total_bytes : (uint32_t)total_docs_unit;
        const SortedStore* s = &f->stores[t];
        if (s->n_nonempty == 0) continue;
        uint32_t low_leaf  = (uint32_t)leaves[t] << k_shift;
        uint32_t high_leaf = ((uint32_t)leaves[t] + 1u) << k_shift;

        const SparseEntry* low_w  = wbuf + ((size_t)t * 2 + 0) * window_n;
        const SparseEntry* high_w = wbuf + ((size_t)t * 2 + 1) * window_n;
        uint32_t low_len  = wlen[(size_t)t * 2 + 0];
        uint32_t high_len = wlen[(size_t)t * 2 + 1];
        if (low_len == 0 || high_len == 0) continue;

        uint32_t i_lo = lb_in_window(low_w,  low_len,  low_leaf);
        uint32_t i_hi = lb_in_window(high_w, high_len, high_leaf);
        if (i_lo >= low_len)  continue;
        uint32_t start = low_w[i_lo].offset;
        uint32_t end   = (i_hi < high_len) ? high_w[i_hi].offset
                                           : (is_v3 ? s->data_bytes : s->total_docs);
        if (end <= start) continue;
        lens[t]     = end - start;   /* bytes for v3, docs for v2 */
        data_off[t] = s->data_base + (uint64_t)start * (is_v3 ? 1u : 4u);
        if (is_v3) total_bytes      += lens[t];
        else       total_docs_unit  += lens[t];
    }

    /* Allocate read buffer:
       - SRT2: docs_buf sized to total_docs_unit
       - SRT3: bytes_buf sized to total_bytes; docs_buf sized to upper bound

       Buffer storage selection : when g_use_shared_pool is on, all Forests
       in this thread share thread-local growable buffers — drops RAM 10×
       on multi-segment clusters since queries are serial anyway.          */
    uint8_t**  bytes_buf_ptr = g_use_shared_pool ? &g_shared_bytes_buf : &fm->bytes_buf;
    uint32_t*  bytes_cap_ptr = g_use_shared_pool ? &g_shared_bytes_cap : &fm->bytes_cap;
    uint32_t** docs_buf_ptr  = g_use_shared_pool ? &g_shared_docs_buf  : &fm->docs_buf;
    uint32_t*  docs_cap_ptr  = g_use_shared_pool ? &g_shared_docs_cap  : &fm->docs_cap;
    if (is_v3) {
        if ((uint64_t)*bytes_cap_ptr < total_bytes) {
            free(*bytes_buf_ptr);
            uint32_t new_cap = (uint32_t)(total_bytes + (total_bytes >> 1));
            if (new_cap < 1024) new_cap = 1024;
            *bytes_buf_ptr = (uint8_t*)malloc((size_t)new_cap);
            *bytes_cap_ptr = new_cap;
        }
        /* Each VarByte byte decodes ≤ 1 doc; first u32 contributes 1 doc per
           leaf. Upper bound on decoded docs = total_bytes (safe).            */
        if ((uint64_t)*docs_cap_ptr < total_bytes) {
            free(*docs_buf_ptr);
            uint32_t new_cap = (uint32_t)(total_bytes + (total_bytes >> 1));
            if (new_cap < 1024) new_cap = 1024;
            *docs_buf_ptr = (uint32_t*)malloc((size_t)new_cap * sizeof(uint32_t));
            *docs_cap_ptr = new_cap;
        }
    } else {
        if ((uint64_t)*docs_cap_ptr < total_docs_unit) {
            free(*docs_buf_ptr);
            uint32_t new_cap = (uint32_t)(total_docs_unit + (total_docs_unit >> 1));
            if (new_cap < 1024) new_cap = 1024;
            *docs_buf_ptr = (uint32_t*)malloc((size_t)new_cap * sizeof(uint32_t));
            *docs_cap_ptr = new_cap;
        }
    }
    uint32_t* docs  = *docs_buf_ptr;
    uint8_t*  bytes = *bytes_buf_ptr;

    /* Phase 2: read posting lists (raw bytes for v3, uint32 array for v2). */
    int phase2_submits = 0;
    for (int t = 0; t < nt; t++) {
        if (lens[t] == 0) continue;
        void*    dst       = is_v3 ? (void*)(bytes + buf_pos[t])
                                   : (void*)(docs  + buf_pos[t]);
        unsigned nbytes    = is_v3 ? (unsigned)lens[t]
                                   : (unsigned)(lens[t] * 4u);
        if (f->ring_ok) {
            struct io_uring_sqe* sqe = io_uring_get_sqe(ring);
            if (!sqe) { io_uring_submit(ring); sqe = io_uring_get_sqe(ring); }
            io_uring_prep_read(sqe, f->stores[t].fd, dst, nbytes,
                               (uint64_t)data_off[t]);
            io_uring_sqe_set_data(sqe, (void*)(uintptr_t)t);
            phase2_submits++;
        } else {
            ssize_t pr2 = pread(f->stores[t].fd, dst, nbytes, (off_t)data_off[t]);
            (void)pr2;
        }
    }
    if (f->ring_ok) {
        io_uring_submit(ring);
        for (int r = 0; r < phase2_submits; r++) {
            struct io_uring_cqe* cqe;
            if (io_uring_wait_cqe(ring, &cqe) < 0) continue;
            io_uring_cqe_seen(ring, cqe);
        }
    }

    /* SRT3: decode VarByte → uint32 docs. Iterates leaves in [i_lo, ...)
       until reaching high_leaf or window end. Updates buf_pos and lens to
       reflect decoded doc layout in docs[]. Falls through to legacy K-way
       merge with docs[buf_pos[t]..buf_pos[t]+lens[t]] sorted ascending.    */
    if (is_v3) {
        uint32_t cumulative = 0;
        for (int t = 0; t < nt; t++) {
            uint32_t byte_start_in_buf = buf_pos[t];
            uint32_t byte_len          = lens[t];
            buf_pos[t] = cumulative;
            lens[t]    = 0;
            if (byte_len == 0) continue;

            const SortedStore* s = &f->stores[t];
            uint32_t low_leaf  = (uint32_t)leaves[t] << k_shift;
            uint32_t high_leaf = ((uint32_t)leaves[t] + 1u) << k_shift;
            const SparseEntry* low_w  = wbuf + ((size_t)t * 2 + 0) * window_n;
            const SparseEntry* high_w = wbuf + ((size_t)t * 2 + 1) * window_n;
            uint32_t low_len  = wlen[(size_t)t * 2 + 0];
            uint32_t high_len = wlen[(size_t)t * 2 + 1];
            uint32_t i_lo = lb_in_window(low_w,  low_len,  low_leaf);
            uint32_t i_hi = lb_in_window(high_w, high_len, high_leaf);
            if (i_lo >= low_len) continue;
            uint32_t slice_byte_start = low_w[i_lo].offset;
            uint32_t slice_byte_end   = (i_hi < high_len) ? high_w[i_hi].offset : s->data_bytes;

            uint32_t k = i_lo;
            uint32_t out_n = 0;
            while (k < low_len) {
                uint32_t lstart_g = low_w[k].offset;
                if (lstart_g >= slice_byte_end) break;
                if (low_w[k].leaf_id >= high_leaf) break;
                uint32_t lend_g;
                if (k + 1 < low_len)
                    lend_g = low_w[k + 1].offset;
                else if (i_hi < high_len)
                    lend_g = high_w[i_hi].offset;
                else
                    lend_g = s->data_bytes;
                if (lend_g > slice_byte_end) lend_g = slice_byte_end;
                if (lend_g <= lstart_g) { k++; continue; }

                uint32_t rel_start = lstart_g - slice_byte_start;
                uint32_t rel_end   = lend_g   - slice_byte_start;
                if (rel_start + 4 > rel_end) break;
                const uint8_t* slice = bytes + byte_start_in_buf;
                uint32_t first_doc;
                memcpy(&first_doc, slice + rel_start, 4);
                docs[cumulative + out_n++] = first_doc;
                uint32_t prev = first_doc;
                size_t bpos = rel_start + 4;
                while (bpos < rel_end) {
                    uint32_t delta = varbyte_decode_u32(slice, &bpos);
                    prev += delta;
                    docs[cumulative + out_n++] = prev;
                }
                k++;
            }
            lens[t]     = out_n;
            cumulative += out_n;
        }
    }

    /* === No-filter fast path : radix sort + linear scan ===
       Replaces the per-tree qsort + K-way heap merge that follows for the
       case where allowed == NULL. Same semantics as the heap path: emit
       (doc, vote) where vote = number of trees containing the doc in this
       call. Critical for the C-multipass code path used by qd↓ queries —
       calling this function N_sets times with heap-merge was the bottleneck;
       radix collapses the K-way merge to O(N) sequential passes.            */
    if (allowed == NULL) {
        size_t N_total = 0;
        for (int t = 0; t < nt; t++) N_total += lens[t];
        if (N_total == 0) {
            g_last_n_distinct = 0;
            if (composed) roaring_bitmap_free(composed);
            return 0;
        }
        if (radix_ensure_cap(N_total) != 0) {
            if (composed) roaring_bitmap_free(composed);
            return -1;
        }
        uint64_t* pairs   = g_radix_pairs_buf;
        uint64_t* scratch = g_radix_scratch_buf;
        size_t cum = 0;
        for (int t = 0; t < nt; t++) {
            if (lens[t] == 0) continue;
            uint64_t tag = (uint64_t)t << 32;
            const uint32_t* src_t = docs + buf_pos[t];
            for (uint32_t i = 0; i < lens[t]; i++) {
                pairs[cum++] = tag | (uint64_t)src_t[i];
            }
        }
        /* 3-pass 11/11/10 radix sort on the low 32 bits (= the doc field).
           Mirrors the radix path of forest_collect_topn_probes. */
        uint64_t* src = pairs;
        uint64_t* dst = scratch;
        const int    BITS[3]   = {11, 11, 10};
        const int    SHIFTS[3] = {0, 11, 22};
        const uint32_t MASKS[3] = {0x7FFu, 0x7FFu, 0x3FFu};
        for (int pass = 0; pass < 3; pass++) {
            int n_buckets = 1 << BITS[pass];
            int shift     = SHIFTS[pass];
            uint32_t mask = MASKS[pass];
            uint32_t hist [2048] = {0};
            uint32_t hist1[2048] = {0};
            uint32_t hist2[2048] = {0};
            uint32_t hist3[2048] = {0};
            size_t i = 0;
            for (; i + 4 <= N_total; i += 4) {
                hist [(src[i+0] >> shift) & mask]++;
                hist1[(src[i+1] >> shift) & mask]++;
                hist2[(src[i+2] >> shift) & mask]++;
                hist3[(src[i+3] >> shift) & mask]++;
            }
            for (; i < N_total; i++) hist[(src[i] >> shift) & mask]++;
            for (int b = 0; b < n_buckets; b++)
                hist[b] += hist1[b] + hist2[b] + hist3[b];
            uint32_t acc = 0;
            for (int b = 0; b < n_buckets; b++) {
                uint32_t c = hist[b];
                hist[b] = acc;
                acc += c;
            }
            for (size_t k = 0; k < N_total; k++) {
                uint32_t b = (src[k] >> shift) & mask;
                dst[hist[b]++] = src[k];
            }
            uint64_t* tmp = src; src = dst; dst = tmp;
        }
        /* Linear scan : consecutive same-doc entries come from distinct
           trees (each tree emits each doc at most once), so cnt = vote. */
        if (fm->topn_h_cap < top_n) {
            free(fm->topn_h_buf);
            fm->topn_h_buf = malloc((size_t)top_n * sizeof(IdVote));
            fm->topn_h_cap = top_n;
        }
        IdVote* topn_h_r = (IdVote*)fm->topn_h_buf;
        int topn_size = 0;
        int32_t prev_doc = -1, cnt = 0;
        int n_distinct = 0;
        int max_d = g_max_distinct;
        int max_sr = g_max_stable_rejects;
        int consec_rejects = 0;
        for (size_t k = 0; k < N_total; k++) {
            int32_t doc = (int32_t)(src[k] & 0xffffffffu);
            if (doc == prev_doc) { cnt++; continue; }
            if (prev_doc >= 0) {
                n_distinct++;
                int min_vote_before = (topn_size == top_n) ? topn_h_r[0].vote : -1;
                topn_push(topn_h_r, &topn_size, top_n, prev_doc, cnt);
                if (max_sr > 0 && topn_size == top_n) {
                    if (cnt <= min_vote_before) {
                        if (++consec_rejects >= max_sr) { prev_doc = -1; break; }
                    } else { consec_rejects = 0; }
                }
                if (max_d > 0 && n_distinct >= max_d) { prev_doc = -1; break; }
            }
            prev_doc = doc; cnt = 1;
        }
        if (prev_doc >= 0) {
            n_distinct++;
            topn_push(topn_h_r, &topn_size, top_n, prev_doc, cnt);
        }
        g_last_n_distinct = n_distinct;
        for (int i = 0; i < topn_size; i++) {
            out_ids[i]   = topn_h_r[i].id;
            out_votes[i] = topn_h_r[i].vote;
        }
        if (composed) roaring_bitmap_free(composed);
        return topn_size;
    }

    /* When the query depth is shallower than the build depth, a tree's
       posting list spans several build-depth leaves — sorted within each leaf,
       but not globally. Re-sort the slice so the K-way merge sees a monotonic
       sequence per tree. (Applies to both v2 and v3.)                       */
    if (k_shift > 0) {
        for (int t = 0; t < nt; t++) {
            if (lens[t] <= 1) continue;
            qsort(docs + buf_pos[t], lens[t], sizeof(uint32_t), cmp_uint32);
        }
    }

    /* Per-tree roaring iterator (one each so cursors advance independently).
       The iterator is forward-monotonic in practice since c->docs is sorted
       ASC and we only advance to ever-larger doc ids.                     */
    roaring_uint32_iterator_t* iters = NULL;
    if (allowed) {
        iters = (roaring_uint32_iterator_t*)malloc((size_t)nt * sizeof(*iters));
        for (int t = 0; t < nt; t++) roaring_init_iterator(allowed, &iters[t]);
    }

    /* K-way merge over the per-tree posting lists. */
    int heap_n = 0;
    for (int t = 0; t < nt; t++) {
        if (lens[t] == 0) continue;
        cursors[t].docs = docs + buf_pos[t];
        cursors[t].len  = lens[t];
        cursors[t].pos  = 0;
        cursors[t].iter = iters ? &iters[t] : NULL;
        if (!cursor_seek_allowed(&cursors[t], allowed)) continue;
        heap[heap_n++]  = &cursors[t];
    }
    for (int i = heap_n / 2 - 1; i >= 0; i--)
        cheap_sift_down(heap, heap_n, i);

    int topn_size = 0;
    int32_t prev_doc = -1, cnt = 0;
    int n_distinct = 0;
    int max_d = g_max_distinct;
    int max_sr = g_max_stable_rejects;
    int consec_rejects = 0;
    while (heap_n > 0) {
        TCursor* c = heap[0];
        int32_t doc = (int32_t)c->doc;
        if (doc == prev_doc) cnt++;
        else {
            if (prev_doc >= 0) {
                n_distinct++;
                int min_vote_before = (topn_size == top_n) ? ((IdVote*)topn_h)[0].vote : -1;
                topn_push(topn_h, &topn_size, top_n, prev_doc, cnt);
                if (max_sr > 0 && topn_size == top_n) {
                    if (cnt <= min_vote_before) {
                        if (++consec_rejects >= max_sr) {
                            prev_doc = -1; break;
                        }
                    } else {
                        consec_rejects = 0;
                    }
                }
                if (max_d > 0 && n_distinct >= max_d) {
                    prev_doc = -1;
                    break;
                }
            }
            prev_doc = doc; cnt = 1;
        }
        c->pos++;
        if (cursor_seek_allowed(c, allowed)) {
            cheap_sift_down(heap, heap_n, 0);
        } else {
            heap[0] = heap[--heap_n];
            cheap_sift_down(heap, heap_n, 0);
        }
    }
    if (prev_doc >= 0) {
        n_distinct++;
        topn_push(topn_h, &topn_size, top_n, prev_doc, cnt);
    }
    g_last_n_distinct = n_distinct;

    for (int i = 0; i < topn_size; i++) {
        out_ids[i]   = topn_h[i].id;
        out_votes[i] = topn_h[i].vote;
    }

    free(iters);
    if (composed) roaring_bitmap_free(composed);
    return topn_size;
}

/* ---------- Fused multi-probe collect (single pass, all probe leaves) ----------

   Replaces the (n_probes+1) separate forest_collect_topn calls + Python vote
   merge with ONE C pass over all nt × (n_probes+1) probe leaves:
     - one io_uring phase-1 (windows) + phase-2 (posting bytes) over every leaf
     - one K-way merge across all leaf-cursors
   A doc may appear in several probe leaves of the SAME tree; it must vote ONCE
   per tree. We dedupe by tracking, per merged doc, which trees already voted
   (small per-tree seen-stamp, reset when the doc id changes). Vote = number of
   DISTINCT trees that contain the doc across any of its probe leaves.

   leaves: caller-provided int32[nt * n_sets], leaves[set*nt + t] = (node-base)
   leaf for tree t in probe-set `set` (set 0 = primary). Native depth only
   (k_shift == 0): each leaf is a single posting list, already sorted ASC.
   Returns topn_size, or -1 on error. tree_stamp must be int[nt] caller scratch
   (or NULL → allocated here). */
int forest_collect_topn_probes(const Forest* f,
                               const int32_t* leaves, int n_sets,
                               int top_n, int probe_depth,
                               const roaring_bitmap_t* allowed_in,
                               int32_t* out_ids, int32_t* out_votes) {
    if (top_n <= 0 || n_sets <= 0) return 0;
    g_last_query_partial = 0;
    for (int pi = 0; pi < QT_N_PHASES; pi++) g_phase_ms[pi] = 0.0;
    double t_prof = phase_now_ms();
    const double t_func_start = t_prof;
    int nt = f->n_trees;
    int is_v3 = (f->srt_version == 3);
    /* probe_depth ≤ 0 or ≥ depth → native (k_shift=0, fast path).
       Else each probe-leaf at probe_depth covers 2^k_shift storage leaves. */
    int qd_eff = (probe_depth > 0 && probe_depth < (int)f->depth)
                 ? probe_depth : (int)f->depth;
    int k_shift = (int)f->depth - qd_eff;

    /* Compose effective filter with tombstones (same policy as collect_topn). */
    const roaring_bitmap_t* allowed = allowed_in;
    roaring_bitmap_t* composed = NULL;
    if (f->tombstones && roaring_bitmap_get_cardinality(f->tombstones) > 0) {
        composed = allowed_in
                 ? roaring_bitmap_andnot(allowed_in, f->tombstones)
                 : roaring_bitmap_flip(f->tombstones, 0, (uint64_t)f->n_docs);
        allowed = composed;
    }

    size_t nL = (size_t)nt * (size_t)n_sets;   /* total leaf-cursors */
    Forest* fm = (Forest*)f;
    struct io_uring* ring = (struct io_uring*)&f->ring;

    /* Per-leaf scratch (allocated per call; nL ~ a few thousand, cheap).
       wbuf is sized AFTER the RAM-only pre-pass below : span windows have
       a data-dependent width.                                             */
    uint32_t* wlen     = (uint32_t*)malloc(nL * sizeof(uint32_t));
    uint32_t* wstart   = (uint32_t*)malloc(nL * sizeof(uint32_t));
    uint64_t* doff     = (uint64_t*)malloc(nL * sizeof(uint64_t));
    uint32_t* lens     = (uint32_t*)malloc(nL * sizeof(uint32_t));
    uint32_t* bpos     = (uint32_t*)malloc(nL * sizeof(uint32_t));
    int32_t*  leaf_tree= (int32_t*) malloc(nL * sizeof(int32_t));
    if (!wlen || !wstart || !doff || !lens || !bpos || !leaf_tree) {
        free(wlen); free(wstart); free(doff); free(lens); free(bpos); free(leaf_tree);
        if (composed) roaring_bitmap_free(composed);
        return -1;
    }

    /* Phase 1 pre-pass (RAM samples only, no I/O) : ONE window per probe,
       spanning every sample bucket of the probe's storage-leaf range
       [low_leaf, high_leaf). This guarantees ALL sparse entries of a
       subtree land in the window whatever the stride — the former
       2-window scheme silently dropped the tail of subtrees straddling a
       bucket boundary (probability ∝ 1/stride).                          */
    uint32_t row_n = 1;
    for (int s = 0; s < n_sets; s++) {
        for (int t = 0; t < nt; t++) {
            size_t li = (size_t)s * nt + t;
            leaf_tree[li] = t;
            wlen[li] = 0; wstart[li] = 0;
            const SortedStore* st = &f->stores[t];
            int32_t lf = leaves[li];
            if (lf < 0 || st->n_nonempty == 0) continue;
            uint32_t low_leaf  = (k_shift > 0) ? ((uint32_t)lf << k_shift) : (uint32_t)lf;
            uint32_t high_leaf = (k_shift > 0) ? (((uint32_t)lf + 1u) << k_shift) : ((uint32_t)lf + 1u);
            uint32_t stride = st->sample_stride;
            uint32_t b_lo = sorted_sample_bucket(st, low_leaf);
            uint32_t b_hi = (k_shift > 0) ? sorted_sample_bucket(st, high_leaf) : b_lo;
            uint32_t start_pos = b_lo * stride;
            if (start_pos >= st->n_nonempty) continue;
            uint32_t take = (b_hi - b_lo + 1u) * stride + 1u;
            if (start_pos + take > st->n_nonempty + 1)
                take = st->n_nonempty + 1 - start_pos;
            wlen[li]   = take;
            wstart[li] = start_pos;
            if (take > row_n) row_n = take;
        }
    }
    SparseEntry* wbuf = (SparseEntry*)malloc(nL * (size_t)row_n * sizeof(SparseEntry));
    if (!wbuf) {
        free(wlen); free(wstart); free(doff); free(lens); free(bpos); free(leaf_tree);
        if (composed) roaring_bitmap_free(composed);
        return -1;
    }

    /* Phase 1 : batched window reads. */
    int p1 = 0;
    for (size_t li = 0; li < nL; li++) {
        if (wlen[li] == 0) continue;
        const SortedStore* st = &f->stores[leaf_tree[li]];
        uint64_t off = st->index_base + (uint64_t)wstart[li] * SRT_INDEX_ENTRY;
        if (f->ring_ok) {
            struct io_uring_sqe* sqe = io_uring_get_sqe(ring);
            if (!sqe) { io_uring_submit(ring); sqe = io_uring_get_sqe(ring); }
            io_uring_prep_read(sqe, st->fd, wbuf + li * (size_t)row_n,
                               (unsigned)(wlen[li] * SRT_INDEX_ENTRY), off);
            io_uring_sqe_set_data(sqe, (void*)(uintptr_t)li);
            p1++;
        } else {
            ssize_t r = pread(st->fd, wbuf + li * (size_t)row_n,
                              wlen[li] * SRT_INDEX_ENTRY, (off_t)off);
            (void)r;
        }
    }
    if (f->ring_ok) {
        io_uring_submit(ring);
        for (int r = 0; r < p1; r++) {
            struct io_uring_cqe* cqe;
            if (io_uring_wait_cqe(ring, &cqe) == 0) io_uring_cqe_seen(ring, cqe);
        }
    }
    free(wstart);   /* only needed for the reads above */
    { double t = phase_now_ms(); g_phase_ms[0] = t - t_prof; t_prof = t; }

    /* Resolve (start,end) byte/doc range per (s, t).
       k_shift==0 : single-leaf range [w[i].offset, w[i+1].offset).
       k_shift>0  : subtree range [low_w[i_lo].offset, high_w[i_hi].offset)
                    covering all non-empty storage leaves in the subtree. */
    uint64_t total_units = 0;
    for (size_t li = 0; li < nL; li++) {
        lens[li] = 0; doff[li] = 0;
        bpos[li] = (uint32_t)total_units;
        int t = leaf_tree[li];
        const SortedStore* st = &f->stores[t];
        int32_t lf = leaves[li];
        if (lf < 0) continue;
        uint32_t low_leaf  = (k_shift > 0) ? ((uint32_t)lf << k_shift) : (uint32_t)lf;
        uint32_t high_leaf = (k_shift > 0) ? (((uint32_t)lf + 1u) << k_shift) : ((uint32_t)lf + 1u);
        const SparseEntry* w = wbuf + li * (size_t)row_n;
        uint32_t len = wlen[li];
        if (len == 0) continue;
        uint32_t i_lo = lb_in_window(w, len, low_leaf);
        if (i_lo >= len) continue;
        uint32_t start, end;
        if (k_shift == 0) {
            /* Single leaf: must match exactly at i_lo, end from next entry. */
            if (w[i_lo].leaf_id != low_leaf) continue;
            start = w[i_lo].offset;
            if (i_lo + 1 < len) end = w[i_lo + 1].offset;
            else                end = is_v3 ? st->data_bytes : st->total_docs;
        } else {
            /* Subtree: [i_lo, i_hi) — both bounds inside the SAME window
               by construction of the span pre-pass. */
            uint32_t i_hi = lb_in_window(w, len, high_leaf);
            start = w[i_lo].offset;
            end   = (i_hi < len) ? w[i_hi].offset
                                 : (is_v3 ? st->data_bytes : st->total_docs);
        }
        if (end <= start) continue;
        uint32_t leaf_size = end - start;
        if (g_max_leaf_bytes != 0 && leaf_size > g_max_leaf_bytes) continue;
        lens[li]  = leaf_size;
        doff[li]  = st->data_base + (uint64_t)start * (is_v3 ? 1u : 4u);
        total_units += lens[li];
    }

    /* Decode buffers from thread-local grow-only pool (docs+bytes). */
    size_t need = total_units ? total_units : 1;
    if (probes_docs_ensure_cap(need) != 0 || (is_v3 && probes_bytes_ensure_cap(need) != 0)) {
        free(wbuf); free(wlen); free(doff); free(lens); free(bpos); free(leaf_tree);
        if (composed) roaring_bitmap_free(composed);
        return -1;
    }
    uint32_t* docs  = g_probes_docs_buf;
    uint8_t*  bytes = is_v3 ? g_probes_bytes_buf : NULL;
    { double t = phase_now_ms(); g_phase_ms[1] = t - t_prof; t_prof = t; }

    /* DEADLINE CHECK #1 — before Phase 2 bulk reads. Skipping the big reads
       is the biggest bang-for-buck : Phase 2 is the io_uring hotspot on
       fat-tail queries (mlb=400k+ SIFT 1B). */
    if (deadline_hit()) {
        g_last_query_partial = 1;
        free(wbuf); free(wlen); free(doff); free(lens); free(bpos); free(leaf_tree);
        if (composed) roaring_bitmap_free(composed);
        return 0;
    }

    /* Phase 2: read posting bytes/docs for every leaf. */
    int p2 = 0;
    for (size_t li = 0; li < nL; li++) {
        if (lens[li] == 0) continue;
        int t = leaf_tree[li];
        void* dst = is_v3 ? (void*)(bytes + bpos[li]) : (void*)(docs + bpos[li]);
        unsigned nb = is_v3 ? (unsigned)lens[li] : (unsigned)(lens[li] * 4u);
        if (f->ring_ok) {
            struct io_uring_sqe* sqe = io_uring_get_sqe(ring);
            if (!sqe) { io_uring_submit(ring); sqe = io_uring_get_sqe(ring); }
            io_uring_prep_read(sqe, f->stores[t].fd, dst, nb, (uint64_t)doff[li]);
            io_uring_sqe_set_data(sqe, (void*)(uintptr_t)li);
            p2++;
        } else {
            ssize_t r = pread(f->stores[t].fd, dst, nb, (off_t)doff[li]);
            (void)r;
        }
    }
    if (f->ring_ok) {
        io_uring_submit(ring);
        for (int r = 0; r < p2; r++) {
            struct io_uring_cqe* cqe;
            if (io_uring_wait_cqe(ring, &cqe) == 0) io_uring_cqe_seen(ring, cqe);
        }
    }
    { double t = phase_now_ms(); g_phase_ms[2] = t - t_prof; t_prof = t; }

    /* DEADLINE CHECK #2 — Phase 2 reads done, before CPU-heavy decode.
       At this point ring is clean, safe to bail early. */
    if (deadline_hit()) {
        g_last_query_partial = 1;
        free(wbuf); free(wlen); free(doff); free(lens); free(bpos); free(leaf_tree);
        if (composed) roaring_bitmap_free(composed);
        return 0;
    }

    /* Decode v3 VarByte leaves into docs[].
       k_shift==0 : single-leaf — one [first_doc + deltas] sequence per slice.
       k_shift>0  : multi-leaf — walk the sparse_index entries to find each
                    sub-leaf's boundaries within the chunk and reset
                    prev_doc at each sub-leaf's first_doc. */
    if (is_v3) {
        uint32_t cum = 0;
        for (size_t li = 0; li < nL; li++) {
            uint32_t blen = lens[li];
            uint32_t bstart = bpos[li];
            bpos[li] = cum; lens[li] = 0;
            if (blen == 0) continue;
            const uint8_t* slice = bytes + bstart;
            uint32_t outn = 0;
            if (k_shift == 0) {
                uint32_t first; memcpy(&first, slice, 4);
                docs[cum] = first;
                uint32_t prev = first; outn = 1;
                size_t p = 4;
                while (p < blen) {
                    uint32_t d = varbyte_decode_u32(slice, &p);
                    prev += d;
                    docs[cum + outn++] = prev;
                }
            } else {
                const SortedStore* st = &f->stores[leaf_tree[li]];
                int32_t lf = leaves[li];
                uint32_t low_leaf  = (uint32_t)lf << k_shift;
                uint32_t high_leaf = ((uint32_t)lf + 1u) << k_shift;
                /* Single span window : every entry of [i_lo, i_hi) is
                   inside it by construction of the Phase-1 pre-pass. */
                const SparseEntry* w = wbuf + li * (size_t)row_n;
                uint32_t len = wlen[li];
                uint32_t i_lo = lb_in_window(w, len, low_leaf);
                uint32_t slice_byte_start = w[i_lo].offset;
                uint32_t slice_byte_end   = slice_byte_start + blen;
                uint32_t k = i_lo;
                while (k < len) {
                    uint32_t lstart_g = w[k].offset;
                    if (lstart_g >= slice_byte_end) break;
                    if (w[k].leaf_id >= high_leaf) break;
                    uint32_t lend_g = (k + 1 < len) ? w[k + 1].offset
                                                    : st->data_bytes;
                    if (lend_g > slice_byte_end) lend_g = slice_byte_end;
                    if (lend_g <= lstart_g) { k++; continue; }
                    uint32_t rel_start = lstart_g - slice_byte_start;
                    uint32_t rel_end   = lend_g   - slice_byte_start;
                    if (rel_start + 4 > rel_end) break;
                    uint32_t first;
                    memcpy(&first, slice + rel_start, 4);
                    docs[cum + outn++] = first;
                    uint32_t prev = first;
                    size_t p = rel_start + 4;
                    while (p < rel_end) {
                        uint32_t d = varbyte_decode_u32(slice, &p);
                        prev += d;
                        docs[cum + outn++] = prev;
                    }
                    k++;
                }
            }
            lens[li] = outn;
            cum += outn;
        }
    }

    { double t = phase_now_ms(); g_phase_ms[3] = t - t_prof; t_prof = t; }

    if (fm->topn_h_cap < top_n) {
        free(fm->topn_h_buf);
        fm->topn_h_buf = malloc((size_t)top_n * sizeof(IdVote));
        fm->topn_h_cap = top_n;
    }
    IdVote* topn_h = (IdVote*)fm->topn_h_buf;
    int topn_size = 0;

    if (allowed == NULL) {
        /* RADIX-SORT + LINEAR-SCAN PATH (no filter case).
           Instead of a K-way merge with log K dependent loads per advance,
           gather all (doc, tree) pairs across all leaves into one packed
           uint64 array, radix-sort by doc (4 passes × 256 buckets — fully
           sequential memory access, no dependent loads), then scan
           linearly with per-tree vote dedup. O(N × 4) vs O(N × log K)
           ops, and the SIMD/prefetcher sees a friendly pattern. */

        size_t N_srt = 0;
        for (size_t li = 0; li < nL; li++) N_srt += lens[li];

        /* HOT overlay contribution to N. Per-probe lock to serialize against
           concurrent writes. Under a per-tree mutex the read of leaves[] +
           docs[] pointers is consistent (no realloc mid-scan). Contention
           minimal at streaming rates ≤ few k/s. */
        size_t N_hot = 0;
        if (g_hot_overlay) {
            for (size_t li = 0; li < nL; li++) {
                int32_t lf = leaves[li];
                if (lf < 0) continue;
                int t = leaf_tree[li];
                uint32_t low_leaf  = (k_shift > 0) ? ((uint32_t)lf << k_shift) : (uint32_t)lf;
                uint32_t high_leaf = (k_shift > 0) ? (((uint32_t)lf + 1u) << k_shift) : ((uint32_t)lf + 1u);
                hot_tree_lock(g_hot_overlay, t);
                int i_lo, i_hi;
                hot_range(g_hot_overlay, t, low_leaf, high_leaf, &i_lo, &i_hi);
                for (int i = i_lo; i < i_hi; i++) {
                    N_hot += g_hot_overlay->trees[t].entries[i].n_docs;
                }
                hot_tree_unlock(g_hot_overlay, t);
            }
            /* Extra slack in case writers race between count and pack. */
            N_hot += 1024;
        }
        size_t N = N_srt + N_hot;
        g_last_n_total = (N > 2147483647UL) ? 2147483647 : (int)N;
        if (N == 0) {
            /* docs/bytes pooled - no free */
            free(wbuf); free(wlen); free(doff); free(lens); free(bpos); free(leaf_tree);
            if (composed) roaring_bitmap_free(composed);
            return 0;
        }

        /* Pack (tree:32 | doc:32) → uint64. Doc in the LOW 32 bits so the
           LSB radix sorts by doc. Pooled per-thread (g_radix_*_buf) so we
           amortize the allocations across queries. */
        if (radix_ensure_cap(N) != 0) {
            /* docs/bytes pooled - no free */
            free(wbuf); free(wlen); free(doff); free(lens); free(bpos); free(leaf_tree);
            if (composed) roaring_bitmap_free(composed);
            return -1;
        }
        uint64_t* pairs   = g_radix_pairs_buf;
        uint64_t* scratch = g_radix_scratch_buf;
        size_t cum = 0;
        /* --- Phase MAIN pack : pack SRT docs into pairs[] --- */
        for (size_t li = 0; li < nL; li++) {
            uint32_t ll = lens[li];
            if (ll == 0) continue;
            uint64_t tag = ((uint64_t)leaf_tree[li]) << 32;
            const uint32_t* src = docs + bpos[li];
            for (uint32_t i = 0; i < ll; i++) {
                pairs[cum++] = tag | (uint64_t)src[i];
            }
        }

        /* --- Phase HOT : batched io_uring reads if any HOT entry hit ---
           1) Under lock : hot_range per probe, snapshot (off, n_docs, fd) per entry
           2) Submit ALL reads to shared io_uring ring, single submit
           3) Reap all completions
           4) Pack docs from hot_data buffer into pairs[]
           Reduces N × blocking preads to a single pipelined io_uring pass —
           critical at shallow qd where N HOT hits per query can be thousands. */
        if (g_hot_overlay && N_hot > 0) {
            typedef struct {
                int      tree;
                int      fd;
                uint64_t disk_off;
                uint32_t n_docs;
                uint32_t buf_off;   /* offset in hot_data[] where docs go */
            } HotOp;
            /* Upper bound on ops : N_hot / avg n_docs ≤ N_hot, so N_hot ops max
               (per-entry). Use a growing scratch. */
            uint32_t hop_cap = 256;
            HotOp* hops = (HotOp*)malloc(hop_cap * sizeof(HotOp));
            uint32_t n_hops = 0;
            uint32_t hot_buf_words = 0;
            for (size_t li = 0; li < nL; li++) {
                int32_t lf = leaves[li];
                if (lf < 0) continue;
                int t = leaf_tree[li];
                uint32_t low_leaf  = (k_shift > 0) ? ((uint32_t)lf << k_shift) : (uint32_t)lf;
                uint32_t high_leaf = (k_shift > 0) ? (((uint32_t)lf + 1u) << k_shift) : ((uint32_t)lf + 1u);
                hot_tree_lock(g_hot_overlay, t);
                int i_lo, i_hi;
                hot_range(g_hot_overlay, t, low_leaf, high_leaf, &i_lo, &i_hi);
                hot_tree_unlock(g_hot_overlay, t);
                for (int i = i_lo; i < i_hi; i++) {
                    uint64_t off; uint32_t nd; int fd;
                    if (hot_snapshot_entry(g_hot_overlay, t, i, &off, &nd, &fd) != 0) continue;
                    if (nd == 0) continue;
                    if (n_hops == hop_cap) {
                        hop_cap *= 2;
                        HotOp* nh = (HotOp*)realloc(hops, hop_cap * sizeof(HotOp));
                        if (!nh) { free(hops); hops = NULL; break; }
                        hops = nh;
                    }
                    hops[n_hops].tree = t;
                    hops[n_hops].fd = fd;
                    hops[n_hops].disk_off = off;
                    hops[n_hops].n_docs = nd;
                    hops[n_hops].buf_off = hot_buf_words;
                    hot_buf_words += nd;
                    n_hops++;
                }
                if (!hops) break;
            }
            if (hops && n_hops > 0) {
                uint32_t* hot_data = (uint32_t*)malloc((size_t)hot_buf_words * sizeof(uint32_t));
                if (hot_data) {
                    /* Submit all io_uring reads. */
                    struct io_uring* ring_h = (struct io_uring*)&f->ring;
                    if (f->ring_ok) {
                        int submitted = 0;
                        for (uint32_t k = 0; k < n_hops; k++) {
                            struct io_uring_sqe* sqe = io_uring_get_sqe(ring_h);
                            if (!sqe) {
                                io_uring_submit(ring_h);
                                for (int r = 0; r < submitted; r++) {
                                    struct io_uring_cqe* cqe;
                                    if (io_uring_wait_cqe(ring_h, &cqe) == 0)
                                        io_uring_cqe_seen(ring_h, cqe);
                                }
                                submitted = 0;
                                sqe = io_uring_get_sqe(ring_h);
                            }
                            io_uring_prep_read(sqe, hops[k].fd,
                                               hot_data + hops[k].buf_off,
                                               hops[k].n_docs * 4u,
                                               hops[k].disk_off);
                            io_uring_sqe_set_data(sqe, (void*)(uintptr_t)k);
                            submitted++;
                        }
                        io_uring_submit(ring_h);
                        for (int r = 0; r < submitted; r++) {
                            struct io_uring_cqe* cqe;
                            if (io_uring_wait_cqe(ring_h, &cqe) == 0)
                                io_uring_cqe_seen(ring_h, cqe);
                        }
                    } else {
                        /* Fallback : blocking pread if ring unavailable. */
                        for (uint32_t k = 0; k < n_hops; k++) {
                            pread(hops[k].fd, hot_data + hops[k].buf_off,
                                  hops[k].n_docs * 4u, (off_t)hops[k].disk_off);
                        }
                    }
                    /* Pack HOT docs into pairs[]. */
                    for (uint32_t k = 0; k < n_hops; k++) {
                        uint64_t tag = ((uint64_t)hops[k].tree) << 32;
                        size_t remaining = (cum < N) ? (N - cum) : 0;
                        uint32_t take = (hops[k].n_docs > (uint32_t)remaining)
                                        ? (uint32_t)remaining : hops[k].n_docs;
                        const uint32_t* src = hot_data + hops[k].buf_off;
                        for (uint32_t di = 0; di < take; di++) {
                            pairs[cum++] = tag | (uint64_t)src[di];
                        }
                    }
                    free(hot_data);
                }
            }
            free(hops);
        }
        /* Actual packed count. Radix sorts over `cum` entries only.        */
        N = cum;
        { double t = phase_now_ms(); g_phase_ms[4] = t - t_prof; t_prof = t; }
        /* LSB radix sort on the doc field (low 32 bits). Number of passes
           is chosen at runtime from f->n_docs:
              ≤ 2^22 docs (4 M) →  2 passes of 11+11 bits
              otherwise        →  3 passes of 11+11+10 bits (covers 1B)
           For small corpora (arxiv 2M, GIST 1M, …), this skips the third
           pass entirely — ~33 % faster radix.                            */
        uint32_t max_doc = (uint32_t)(f->n_docs > 0 ? f->n_docs - 1 : 0);
        int doc_bits = max_doc ? 32 - __builtin_clz(max_doc) : 1;
        int n_passes;
        int BITS[3]; int SHIFTS[3]; uint32_t MASKS[3];
        if (doc_bits <= 22) {
            n_passes = 2;
            BITS  [0] = 11;     BITS  [1] = 11;
            SHIFTS[0] =  0;     SHIFTS[1] = 11;
            MASKS [0] = 0x7FFu; MASKS [1] = 0x7FFu;
        } else {
            n_passes = 3;
            BITS  [0] = 11;     BITS  [1] = 11;     BITS  [2] = 10;
            SHIFTS[0] =  0;     SHIFTS[1] = 11;     SHIFTS[2] = 22;
            MASKS [0] = 0x7FFu; MASKS [1] = 0x7FFu; MASKS [2] = 0x3FFu;
        }
        uint64_t* src = pairs;
        uint64_t* dst = scratch;
        for (int pass = 0; pass < n_passes; pass++) {
            int n_buckets = 1 << BITS[pass];
            int shift     = SHIFTS[pass];
            uint32_t mask = MASKS[pass];
            uint32_t hist [2048] = {0};
            uint32_t hist1[2048] = {0};
            uint32_t hist2[2048] = {0};
            uint32_t hist3[2048] = {0};
            size_t i = 0;
            for (; i + 4 <= N; i += 4) {
                uint32_t b0 = (uint32_t)((src[i+0] >> shift) & mask);
                uint32_t b1 = (uint32_t)((src[i+1] >> shift) & mask);
                uint32_t b2 = (uint32_t)((src[i+2] >> shift) & mask);
                uint32_t b3 = (uint32_t)((src[i+3] >> shift) & mask);
                hist [b0]++;
                hist1[b1]++;
                hist2[b2]++;
                hist3[b3]++;
            }
            for (; i < N; i++) hist[(src[i] >> shift) & mask]++;
            /* Merge 4 sub-histograms. Vectorizes trivially. */
            for (int b = 0; b < n_buckets; b++)
                hist[b] += hist1[b] + hist2[b] + hist3[b];
            uint32_t acc = 0;
            for (int b = 0; b < n_buckets; b++) {
                uint32_t c = hist[b];
                hist[b] = acc;
                acc += c;
            }
            for (size_t i = 0; i < N; i++) {
                uint32_t b = (src[i] >> shift) & mask;
                dst[hist[b]++] = src[i];
            }
            uint64_t* t = src; src = dst; dst = t;
        }
        /* After n_passes swaps, sorted result lives in `src`:
           n_passes odd  → src now points to `scratch`
           n_passes even → src now points back to `pairs`.
           Either way the linear scan below reads from `src`.            */
        { double t = phase_now_ms(); g_phase_ms[5] = t - t_prof; t_prof = t; }

        /* Linear scan with per-tree vote dedup. Pooled tree_seen[] to
           avoid per-query alloc. */
        if (tree_seen_ensure_cap(nt) != 0) {
            /* docs/bytes pooled - no free */
            free(wbuf); free(wlen); free(doff); free(lens); free(bpos); free(leaf_tree);
            if (composed) roaring_bitmap_free(composed);
            return -1;
        }
        int32_t* tree_seen = g_tree_seen_buf;
        for (int t = 0; t < nt; t++) tree_seen[t] = -1;

        /* Histogram-based top-n pick : one buffer of (doc,vote) + vote
           histogram during the scan, then a single threshold lookup to emit
           the top `top_n`. Replaces the binary-heap topn_push (5 % of total
           query time). Buffer is the same `topn_h` slot, reused for storage
           of all distinct docs (sized by topn_h_cap; will realloc if
           needed by topn_ensure_distinct_cap-like logic below).            */
        size_t n_distinct = 0;
        /* Upper bound on distinct docs : N (every pair is distinct in the
           limit). Reuse topn_h_buf if big enough, else realloc.            */
        if (fm->topn_h_cap < (int)N) {
            free(fm->topn_h_buf);
            fm->topn_h_buf = malloc(N * sizeof(IdVote));
            fm->topn_h_cap = (int)N;
            topn_h = (IdVote*)fm->topn_h_buf;
        }
        uint32_t vote_hist[257] = {0};   /* votes ∈ [0, n_trees ≤ 256] */
        const int MAX_VOTE = (nt < 256) ? nt : 256;
        int32_t prev_doc = -1, vote = 0;
        for (size_t i = 0; i < N; i++) {
            /* DEADLINE CHECK — every 65536 pairs. Overhead ~15 ns/64k = négligeable. */
            if ((i & 0xFFFFu) == 0 && deadline_hit()) {
                g_last_query_partial = 1;
                if (prev_doc >= 0) {
                    int v = (vote > MAX_VOTE) ? MAX_VOTE : vote;
                    topn_h[n_distinct++] = (IdVote){prev_doc, v};
                    vote_hist[v]++;
                }
                break;
            }
            uint64_t p = src[i];
            int32_t doc = (int32_t)(p & 0xffffffffu);
            int32_t tree = (int32_t)(p >> 32);
            if (doc != prev_doc) {
                if (prev_doc >= 0) {
                    int v = (vote > MAX_VOTE) ? MAX_VOTE : vote;
                    topn_h[n_distinct++] = (IdVote){prev_doc, v};
                    vote_hist[v]++;
                }
                prev_doc = doc; vote = 0;
            }
            if (tree_seen[tree] != doc) {
                tree_seen[tree] = doc;
                vote++;
            }
        }
        if (prev_doc >= 0) {
            int v = (vote > MAX_VOTE) ? MAX_VOTE : vote;
            topn_h[n_distinct++] = (IdVote){prev_doc, v};
            vote_hist[v]++;
        }
        g_last_n_distinct = (int)n_distinct;
        topn_size = topn_pick_hist(topn_h, n_distinct, vote_hist, MAX_VOTE,
                                   top_n, out_ids, out_votes);
        { double t = phase_now_ms();
          g_phase_ms[6] = t - t_prof;
          g_phase_ms[7] = t - t_func_start; }

        /* pairs, scratch, tree_seen all pooled — do not free. */
        /* docs/bytes pooled - no free */
        free(wbuf); free(wlen); free(doff); free(lens); free(bpos); free(leaf_tree);
        if (composed) roaring_bitmap_free(composed);
        return topn_size;
    }

    /* FILTER PATH (allowed != NULL): per-leaf cursors with roaring iterator
       per cursor. Kept as-is so the bitmap-pre-filter semantics are preserved
       exactly. The pre-merge path above is only enabled when there is no
       caller-side filter to apply during the merge. */
    roaring_uint32_iterator_t* iters = (roaring_uint32_iterator_t*)malloc(nL * sizeof(*iters));
    for (size_t li = 0; li < nL; li++) roaring_init_iterator(allowed, &iters[li]);

    TCursor* cursors    = (TCursor*)malloc(nL * sizeof(TCursor));
    int32_t* cur_tree   = (int32_t*)malloc(nL * sizeof(int32_t));
    if (!cursors || !cur_tree || !iters) {
        free(cursors); free(cur_tree); free(iters);
        /* docs/bytes pooled - no free */
        free(wbuf); free(wlen); free(doff); free(lens); free(bpos); free(leaf_tree);
        if (composed) roaring_bitmap_free(composed);
        return -1;
    }
    int K = 0;
    for (size_t li = 0; li < nL; li++) {
        if (lens[li] == 0) continue;
        TCursor* c = &cursors[K];
        c->docs = docs + bpos[li];
        c->len  = lens[li];
        c->pos  = 0;
        c->iter = &iters[li];
        if (!cursor_seek_allowed(c, allowed)) continue;
        cur_tree[K] = leaf_tree[li];
        K++;
    }
    int K_pad = 2;
    while (K_pad < (K + 1)) K_pad <<= 1;
    uint32_t* doc_cache = (uint32_t*)malloc((size_t)(K_pad + 1) * sizeof(uint32_t));
    int*      losers    = (int*)     malloc((size_t)K_pad * sizeof(int));
    if (!doc_cache || !losers) {
        free(cursors); free(cur_tree); free(doc_cache); free(losers); free(iters);
        /* docs/bytes pooled - no free */
        free(wbuf); free(wlen); free(doff); free(lens); free(bpos); free(leaf_tree);
        if (composed) roaring_bitmap_free(composed);
        return -1;
    }
    for (int i = 0; i < K; i++) doc_cache[i] = cursors[i].doc;
    for (int i = K; i <= K_pad; i++) doc_cache[i] = UINT32_MAX;
    LoserTree lt = { K_pad, 0, losers, doc_cache };
    lt_init(&lt);

    int32_t* tree_seen = (int32_t*)malloc((size_t)nt * sizeof(int32_t));
    for (int t = 0; t < nt; t++) tree_seen[t] = -1;

    int32_t prev_doc = -1, vote = 0;
    uint64_t merge_ticks = 0;
    while (doc_cache[lt.winner] != UINT32_MAX) {
        /* DEADLINE CHECK — every 65536 K-way merge steps. */
        if ((merge_ticks++ & 0xFFFFu) == 0 && deadline_hit()) {
            g_last_query_partial = 1;
            break;
        }
        int w = lt.winner;
        int tree = cur_tree[w];
        int32_t doc = (int32_t)doc_cache[w];
        if (doc != prev_doc) {
            if (prev_doc >= 0) topn_push(topn_h, &topn_size, top_n, prev_doc, vote);
            prev_doc = doc; vote = 0;
        }
        if (tree_seen[tree] != doc) {
            tree_seen[tree] = doc;
            vote++;
        }
        TCursor* c = &cursors[w];
        c->pos++;
        if (cursor_seek_allowed(c, allowed)) {
            doc_cache[w] = c->doc;
        } else {
            doc_cache[w] = UINT32_MAX;
        }
        lt_pop_advance(&lt);
    }
    if (prev_doc >= 0) topn_push(topn_h, &topn_size, top_n, prev_doc, vote);

    for (int i = 0; i < topn_size; i++) {
        out_ids[i]   = topn_h[i].id;
        out_votes[i] = topn_h[i].vote;
    }

    free(tree_seen); free(cur_tree); free(cursors);
    free(doc_cache); free(losers); free(iters);
    /* docs/bytes pooled - no free */
    free(wbuf); free(wlen); free(doff); free(lens); free(bpos); free(leaf_tree);
    if (composed) roaring_bitmap_free(composed);
    return topn_size;
}

/* ---------- Multi-index query across forests sharing seeds ---------- */

int forest_collect_topn_multi(Forest** forests, int n_forests,
                              const float* qvec,
                              int top_n, int query_depth,
                              const roaring_bitmap_t* allowed,
                              int32_t* out_ids, int32_t* out_votes) {
    if (top_n <= 0 || n_forests <= 0) return 0;
    Forest* f0 = forests[0];
    int nt = f0->n_trees;
    int dim = f0->dim;
    int sub_dim = f0->sub_dim;
    int build_depth = f0->depth;
    int use_sub = (sub_dim > 0 && sub_dim <= dim);

    int qd = (query_depth > 0 && query_depth <= build_depth) ? query_depth : build_depth;
    int k_shift = build_depth - qd;

    /* Stack scratch for traverse. */
    float v0[1024], v1[1024], qn[1024];
    int dims[256];
    if (dim > 1024 || (use_sub && sub_dim > 256)) {
        fprintf(stderr, "dim too large for stack scratch\n");
        return -1;
    }
    memcpy(qn, qvec, (size_t)dim * sizeof(float));
    normalize(qn, dim);

    /* Per-tree shared leaf (same seeds across forests). */
    int32_t base_q = leaf_base(qd);
    int32_t* leaves = (int32_t*)malloc((size_t)nt * sizeof(int32_t));
    for (int t = 0; t < nt; t++) {
        int32_t leaf = use_sub
            ? traverse_sub(qn, dim, sub_dim, qd, tree_seed(t),
                           v0, v1, dims)
            : traverse(qn, dim, qd, tree_seed(t), v0, v1);
        leaves[t] = leaf - base_q;
    }

    /* Find max window across all forests. */
    uint32_t max_window_n = 0;
    for (int fi = 0; fi < n_forests; fi++)
        if (forests[fi]->max_window_n > max_window_n)
            max_window_n = forests[fi]->max_window_n;

    /* Allocate per-call scratch. Indexed by (tree, forest, slot) where
       slot ∈ {0=low, 1=high}. Linear index = (t*n_forests + fi)*2 + slot. */
    size_t n_pairs = (size_t)nt * (size_t)n_forests;
    SparseEntry* wbuf  = (SparseEntry*)malloc(n_pairs * 2 * max_window_n * sizeof(SparseEntry));
    uint32_t* wlen     = (uint32_t*)malloc(n_pairs * 2 * sizeof(uint32_t));
    uint32_t* lens     = (uint32_t*)malloc(n_pairs * sizeof(uint32_t));
    uint64_t* data_off = (uint64_t*)malloc(n_pairs * sizeof(uint64_t));
    uint32_t* buf_pos  = (uint32_t*)malloc(n_pairs * sizeof(uint32_t));
    if (!wbuf || !wlen || !lens || !data_off || !buf_pos) {
        free(wbuf); free(wlen); free(lens); free(data_off); free(buf_pos);
        free(leaves); return -1;
    }

    /* Phase 1: 2 window reads per (tree, forest). Per-ring submission count
       so we reap exactly that many CQEs from each ring.                  */
    int* per_ring_subs = (int*)calloc((size_t)n_forests, sizeof(int));
    for (int t = 0; t < nt; t++) {
        uint32_t low_leaf  = (uint32_t)leaves[t] << k_shift;
        uint32_t high_leaf = ((uint32_t)leaves[t] + 1u) << k_shift;
        for (int fi = 0; fi < n_forests; fi++) {
            const SortedStore* s = &forests[fi]->stores[t];
            uint32_t stride = s->sample_stride;
            for (int slot = 0; slot < 2; slot++) {
                size_t wi = ((size_t)t * n_forests + fi) * 2 + slot;
                wlen[wi] = 0;
                if (s->n_nonempty == 0) continue;
                uint32_t target = (slot == 0) ? low_leaf : high_leaf;
                uint32_t bucket = sorted_sample_bucket(s, target);
                uint32_t start_pos = bucket * stride;
                if (start_pos >= s->n_nonempty) continue;
                uint32_t take = stride + 1;
                if (start_pos + take > s->n_nonempty + 1)
                    take = s->n_nonempty + 1 - start_pos;
                wlen[wi] = take;
                uint64_t off = s->index_base + (uint64_t)start_pos * SRT_INDEX_ENTRY;

                struct io_uring* ring = (struct io_uring*)&forests[fi]->ring;
                struct io_uring_sqe* sqe = io_uring_get_sqe(ring);
                if (!sqe) { io_uring_submit(ring); sqe = io_uring_get_sqe(ring); }
                io_uring_prep_read(sqe, s->fd,
                                   wbuf + wi * max_window_n,
                                   (unsigned)(take * SRT_INDEX_ENTRY),
                                   off);
                io_uring_sqe_set_data(sqe, (void*)(uintptr_t)wi);
                per_ring_subs[fi]++;
            }
        }
    }
    /* Submit all rings, then reap per-ring exact count. */
    for (int fi = 0; fi < n_forests; fi++) {
        struct io_uring* ring = (struct io_uring*)&forests[fi]->ring;
        (void)io_uring_submit(ring);
    }
    for (int fi = 0; fi < n_forests; fi++) {
        struct io_uring* ring = (struct io_uring*)&forests[fi]->ring;
        for (int r = 0; r < per_ring_subs[fi]; r++) {
            struct io_uring_cqe* cqe;
            if (io_uring_wait_cqe(ring, &cqe) == 0)
                io_uring_cqe_seen(ring, cqe);
        }
    }

    /* Resolve (start, end) per (t, fi). For SRT3 forests, units are BYTES;
       for SRT2 forests, units are DOCS (multiplied by 4 on disk).
       buf_pos[pi] holds the cumulative unit count up to pi.                  */
    uint64_t total_units = 0;
    int any_v3 = 0;
    for (int fi = 0; fi < n_forests; fi++)
        if (forests[fi]->srt_version == 3) { any_v3 = 1; break; }

    for (int t = 0; t < nt; t++) {
        uint32_t low_leaf  = (uint32_t)leaves[t] << k_shift;
        uint32_t high_leaf = ((uint32_t)leaves[t] + 1u) << k_shift;
        for (int fi = 0; fi < n_forests; fi++) {
            size_t pi = (size_t)t * n_forests + fi;
            size_t wi_lo = pi * 2 + 0;
            size_t wi_hi = pi * 2 + 1;
            lens[pi] = 0; data_off[pi] = 0; buf_pos[pi] = (uint32_t)total_units;
            const SortedStore* s = &forests[fi]->stores[t];
            if (s->n_nonempty == 0) continue;
            uint32_t low_len  = wlen[wi_lo];
            uint32_t high_len = wlen[wi_hi];
            if (low_len == 0 || high_len == 0) continue;
            const SparseEntry* low_w  = wbuf + wi_lo * max_window_n;
            const SparseEntry* high_w = wbuf + wi_hi * max_window_n;
            uint32_t i_lo = lb_in_window(low_w,  low_len,  low_leaf);
            uint32_t i_hi = lb_in_window(high_w, high_len, high_leaf);
            if (i_lo >= low_len) continue;
            int is_v3 = (forests[fi]->srt_version == 3);
            uint32_t start = low_w[i_lo].offset;
            uint32_t end   = (i_hi < high_len) ? high_w[i_hi].offset
                                               : (is_v3 ? s->data_bytes : s->total_docs);
            if (end <= start) continue;
            lens[pi]     = end - start;          /* bytes for v3, docs for v2 */
            data_off[pi] = s->data_base + (uint64_t)start * (is_v3 ? 1u : 4u);
            total_units += lens[pi];
        }
    }

    /* Phase 2: read raw bytes for v3 forests, uint32 for v2.
       We use one buffer typed as bytes; v2 reads write uint32 into it which
       is also valid (alignment 8). Upper bound for decoded doc count =
       total_units (each byte ≥ 1 doc; v2 has 4 bytes/doc exactly).          */
    uint8_t* bytes_buf = (uint8_t*)malloc((size_t)(total_units ? total_units : 1));
    uint32_t* docs = (uint32_t*)malloc((size_t)(total_units ? total_units : 1) * sizeof(uint32_t));
    if (!bytes_buf || !docs) {
        free(bytes_buf); free(docs);
        free(wbuf); free(wlen); free(lens); free(data_off); free(buf_pos);
        free(per_ring_subs); free(leaves); return -1;
    }
    memset(per_ring_subs, 0, (size_t)n_forests * sizeof(int));
    for (int t = 0; t < nt; t++) {
        for (int fi = 0; fi < n_forests; fi++) {
            size_t pi = (size_t)t * n_forests + fi;
            if (lens[pi] == 0) continue;
            int is_v3 = (forests[fi]->srt_version == 3);
            unsigned nbytes = is_v3 ? (unsigned)lens[pi] : (unsigned)(lens[pi] * 4u);
            void*    dst    = bytes_buf + buf_pos[pi];
            struct io_uring* ring = (struct io_uring*)&forests[fi]->ring;
            struct io_uring_sqe* sqe = io_uring_get_sqe(ring);
            if (!sqe) { io_uring_submit(ring); sqe = io_uring_get_sqe(ring); }
            io_uring_prep_read(sqe, forests[fi]->stores[t].fd, dst, nbytes,
                               data_off[pi]);
            io_uring_sqe_set_data(sqe, (void*)(uintptr_t)pi);
            per_ring_subs[fi]++;
        }
    }
    for (int fi = 0; fi < n_forests; fi++) {
        struct io_uring* ring = (struct io_uring*)&forests[fi]->ring;
        (void)io_uring_submit(ring);
    }
    for (int fi = 0; fi < n_forests; fi++) {
        struct io_uring* ring = (struct io_uring*)&forests[fi]->ring;
        for (int r = 0; r < per_ring_subs[fi]; r++) {
            struct io_uring_cqe* cqe;
            if (io_uring_wait_cqe(ring, &cqe) == 0)
                io_uring_cqe_seen(ring, cqe);
        }
    }
    free(per_ring_subs);

    /* Decode v3 forests in-place into docs[]; copy v2 raw uint32 into docs[]. */
    {
        uint32_t cumulative = 0;
        for (int t = 0; t < nt; t++) {
            uint32_t low_leaf  = (uint32_t)leaves[t] << k_shift;
            uint32_t high_leaf = ((uint32_t)leaves[t] + 1u) << k_shift;
            for (int fi = 0; fi < n_forests; fi++) {
                size_t pi = (size_t)t * n_forests + fi;
                uint32_t unit_count_in_buf = lens[pi];
                uint32_t orig_buf_pos      = buf_pos[pi];
                buf_pos[pi] = cumulative;
                lens[pi]    = 0;
                if (unit_count_in_buf == 0) continue;
                int is_v3 = (forests[fi]->srt_version == 3);
                if (!is_v3) {
                    /* v2: copy 4-byte uint32 from bytes_buf into docs */
                    uint32_t n_docs = unit_count_in_buf;
                    memcpy(docs + cumulative, bytes_buf + orig_buf_pos,
                           (size_t)n_docs * 4u);
                    lens[pi]    = n_docs;
                    cumulative += n_docs;
                    continue;
                }
                /* v3 decode */
                const SortedStore* s = &forests[fi]->stores[t];
                size_t wi_lo = pi * 2 + 0;
                size_t wi_hi = pi * 2 + 1;
                uint32_t low_len  = wlen[wi_lo];
                uint32_t high_len = wlen[wi_hi];
                const SparseEntry* low_w  = wbuf + wi_lo * max_window_n;
                const SparseEntry* high_w = wbuf + wi_hi * max_window_n;
                uint32_t i_lo = lb_in_window(low_w,  low_len,  low_leaf);
                uint32_t i_hi = lb_in_window(high_w, high_len, high_leaf);
                if (i_lo >= low_len) continue;
                uint32_t slice_byte_start = low_w[i_lo].offset;
                uint32_t slice_byte_end   = (i_hi < high_len) ? high_w[i_hi].offset : s->data_bytes;
                const uint8_t* slice = bytes_buf + orig_buf_pos;
                uint32_t out_n = 0;
                uint32_t k = i_lo;
                while (k < low_len) {
                    uint32_t lstart_g = low_w[k].offset;
                    if (lstart_g >= slice_byte_end) break;
                    if (low_w[k].leaf_id >= high_leaf) break;
                    uint32_t lend_g;
                    if (k + 1 < low_len)             lend_g = low_w[k + 1].offset;
                    else if (i_hi < high_len)         lend_g = high_w[i_hi].offset;
                    else                              lend_g = s->data_bytes;
                    if (lend_g > slice_byte_end) lend_g = slice_byte_end;
                    if (lend_g <= lstart_g) { k++; continue; }
                    uint32_t rel_start = lstart_g - slice_byte_start;
                    uint32_t rel_end   = lend_g   - slice_byte_start;
                    if (rel_start + 4 > rel_end) break;
                    uint32_t first_doc;
                    memcpy(&first_doc, slice + rel_start, 4);
                    docs[cumulative + out_n++] = first_doc;
                    uint32_t prev = first_doc;
                    size_t bpos = rel_start + 4;
                    while (bpos < rel_end) {
                        uint32_t delta = varbyte_decode_u32(slice, &bpos);
                        prev += delta;
                        docs[cumulative + out_n++] = prev;
                    }
                    k++;
                }
                lens[pi]    = out_n;
                cumulative += out_n;
            }
        }
    }
    free(bytes_buf);
    (void)any_v3;

    /* Per tree: collect total range of (lens across forests) and sort.
       buf_pos[t*n_forests..t*n_forests+n_forests) cover this tree's slice. */
    uint32_t* tree_starts = (uint32_t*)malloc((size_t)nt * sizeof(uint32_t));
    uint32_t* tree_lens   = (uint32_t*)malloc((size_t)nt * sizeof(uint32_t));
    for (int t = 0; t < nt; t++) {
        size_t pi0 = (size_t)t * n_forests;
        tree_starts[t] = buf_pos[pi0];
        uint32_t tl = 0;
        for (int fi = 0; fi < n_forests; fi++) tl += lens[pi0 + fi];
        tree_lens[t] = tl;
        if (tl > 1)
            qsort(docs + tree_starts[t], tl, sizeof(uint32_t), cmp_uint32);
    }

    /* K-way merge across trees + top-N by votes. */
    TCursor*  cursors = (TCursor*) calloc((size_t)nt, sizeof(TCursor));
    TCursor** heap    = (TCursor**)malloc((size_t)nt * sizeof(TCursor*));
    IdVote*   topn_h  = (IdVote*)  malloc((size_t)top_n * sizeof(IdVote));

    roaring_uint32_iterator_t* iters_m = NULL;
    if (allowed) {
        iters_m = (roaring_uint32_iterator_t*)malloc((size_t)nt * sizeof(*iters_m));
        for (int t = 0; t < nt; t++) roaring_init_iterator(allowed, &iters_m[t]);
    }

    int heap_n = 0;
    for (int t = 0; t < nt; t++) {
        if (tree_lens[t] == 0) continue;
        cursors[t].docs = docs + tree_starts[t];
        cursors[t].len  = tree_lens[t];
        cursors[t].pos  = 0;
        cursors[t].iter = iters_m ? &iters_m[t] : NULL;
        if (!cursor_seek_allowed(&cursors[t], allowed)) continue;
        heap[heap_n++]  = &cursors[t];
    }
    for (int i = heap_n / 2 - 1; i >= 0; i--)
        cheap_sift_down(heap, heap_n, i);

    int topn_size = 0;
    int32_t prev_doc = -1, cnt = 0;
    int n_distinct = 0;
    int max_d = g_max_distinct;
    while (heap_n > 0) {
        TCursor* c = heap[0];
        int32_t doc = (int32_t)c->doc;
        if (doc == prev_doc) cnt++;
        else {
            if (prev_doc >= 0) {
                n_distinct++;
                topn_push(topn_h, &topn_size, top_n, prev_doc, cnt);
                if (max_d > 0 && n_distinct >= max_d) {
                    prev_doc = -1;
                    break;
                }
            }
            prev_doc = doc; cnt = 1;
        }
        c->pos++;
        if (cursor_seek_allowed(c, allowed)) {
            cheap_sift_down(heap, heap_n, 0);
        } else {
            heap[0] = heap[--heap_n];
            cheap_sift_down(heap, heap_n, 0);
        }
    }
    if (prev_doc >= 0) {
        n_distinct++;
        topn_push(topn_h, &topn_size, top_n, prev_doc, cnt);
    }
    free(iters_m);
    g_last_n_distinct = n_distinct;

    for (int i = 0; i < topn_size; i++) {
        out_ids[i]   = topn_h[i].id;
        out_votes[i] = topn_h[i].vote;
    }

    free(wbuf); free(wlen); free(lens); free(data_off); free(buf_pos);
    free(leaves); free(docs); free(tree_starts); free(tree_lens);
    free(cursors); free(heap); free(topn_h);
    (void)f0;
    return topn_size;
}
