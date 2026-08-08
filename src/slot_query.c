/* SLT query path : reads from .slt slot-allocated per-tree files.
 * Mirror of forest_collect_topn_probes for SRT, adapted for slot layout.
 * MVP : no filter (roaring), no partial deadline. Just pathrank + radix + top_n. */
#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L
#include <sys/uio.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <linux/openat2.h>   /* struct open_how — must precede liburing.h */
#include "liburing_compat.h"
#include "slot_store.h"
#include "traversal.h"
#include "gen_vec.h"

typedef struct {
    int         n_trees, dim, sub_dim, depth, n_docs;
    SltStore*   stores;
    struct io_uring ring;
    int         ring_ok;
} SltForest;

typedef struct { float score; int32_t li; } SltPathRankEntry;
static int slt_pathrank_cmp_desc(const void* a, const void* b) {
    float sa = ((const SltPathRankEntry*)a)->score;
    float sb = ((const SltPathRankEntry*)b)->score;
    return (sa < sb) - (sa > sb);
}

int slt_forest_open(SltForest* f, const char* index_dir, int n_trees, int dim,
                    int sub_dim, int depth, int n_docs) {
    f->n_trees = n_trees; f->dim = dim; f->sub_dim = sub_dim;
    f->depth = depth; f->n_docs = n_docs;
    f->stores = (SltStore*)calloc((size_t)n_trees, sizeof(SltStore));
    if (!f->stores) return -1;
    char path[512];
    for (int t = 0; t < n_trees; t++) {
        snprintf(path, sizeof(path), "%s/tree%05d.slt", index_dir, t);
        if (slt_open_rdonly(&f->stores[t], path) != 0) {
            fprintf(stderr, "slt_forest_open: failed to open %s\n", path);
            for (int j = 0; j < t; j++) slt_close(&f->stores[j]);
            free(f->stores); return -1;
        }
    }
    /* io_uring : sized for max_probes × 2 (Phase 1 + Phase 2) */
    unsigned ring_n = (unsigned)(n_trees * 4);
    if (ring_n < 256) ring_n = 256;
    if (ring_n > 32768) ring_n = 32768;
    int rc = io_uring_queue_init(ring_n, &f->ring, 0);
    f->ring_ok = (rc == 0);
    return 0;
}

void slt_forest_close(SltForest* f) {
    if (!f) return;
    if (f->ring_ok) { io_uring_queue_exit(&f->ring); f->ring_ok = 0; }
    if (f->stores) {
        for (int t = 0; t < f->n_trees; t++) slt_close(&f->stores[t]);
        free(f->stores); f->stores = NULL;
    }
}


/* Query pathrank for SLT : traversal + pathrank + phase 1 + phase 2 + radix + top_n.
 * Simpler than SRT version : no varbyte, no roaring filter, no partial deadline. */
int slt_query_pathrank(SltForest* f, const float* qvec,
                       int n_probes, int top_paths, int top_n,
                       int query_depth,
                       int32_t* out_ids, int32_t* out_votes) {
    int nt = f->n_trees, dim = f->dim, sub = f->sub_dim, depth = f->depth;
    int qd_eff = (query_depth > 0 && query_depth < depth) ? query_depth : depth;
    int k_shift = depth - qd_eff;
    int use_sub = (sub > 0 && sub <= dim);
    if (n_probes < 0) n_probes = 0;
    int n_sets = n_probes + 1;
    size_t nL = (size_t)nt * (size_t)n_sets;
    if (top_paths <= 0 || (size_t)top_paths > nL) top_paths = (int)nL;

    /* Normalize query */
    float* qn = (float*)malloc((size_t)dim * sizeof(float));
    if (!qn) return -1;
    float nrm = 0.0f;
    for (int i = 0; i < dim; i++) nrm += qvec[i] * qvec[i];
    nrm = (nrm > 0.0f) ? 1.0f / sqrtf(nrm) : 1.0f;
    for (int i = 0; i < dim; i++) qn[i] = qvec[i] * nrm;

    /* Traversal + pathrank : same code as SRT. */
    int32_t* leaves = (int32_t*)malloc(nL * sizeof(int32_t));
    SltPathRankEntry* rank = (SltPathRankEntry*)malloc(nL * sizeof(SltPathRankEntry));
    if (!leaves || !rank) { free(qn); free(leaves); free(rank); return -1; }
    int32_t base = leaf_base(qd_eff);
    for (size_t i = 0; i < nL; i++) {
        leaves[i] = -1; rank[i].score = -1.0f; rank[i].li = (int32_t)i;
    }
    #pragma omp parallel
    {
        float* v0 = (float*)malloc((size_t)dim * sizeof(float));
        float* v1 = (float*)malloc((size_t)dim * sizeof(float));
        int*   dims = (int*)malloc((size_t)dim * sizeof(int));
        int32_t* nodes = (int32_t*)malloc((size_t)n_sets * sizeof(int32_t));
        float*   scores = (float*)malloc((size_t)n_sets * sizeof(float));
        if (v0 && v1 && dims && nodes && scores) {
            #pragma omp for schedule(static)
            for (int t = 0; t < nt; t++) {
                int cnt;
                if (use_sub) {
                    cnt = traverse_sub_probes_scored(qn, dim, sub, qd_eff, tree_seed(t),
                                                     v0, v1, dims, n_probes, 0, nodes, scores);
                } else {
                    nodes[0] = traverse(qn, dim, qd_eff, tree_seed(t), v0, v1);
                    scores[0] = 0.0f; cnt = 1;
                }
                for (int s = 0; s < n_sets; s++) {
                    size_t li = (size_t)s * nt + t;
                    if (s < cnt) {
                        leaves[li] = nodes[s] - base;
                        rank[li].score = scores[s];
                        rank[li].li = (int32_t)li;
                    }
                }
            }
        }
        free(v0); free(v1); free(dims); free(nodes); free(scores);
    }
    qsort(rank, nL, sizeof(SltPathRankEntry), slt_pathrank_cmp_desc);
    for (size_t i = (size_t)top_paths; i < nL; i++) leaves[rank[i].li] = -1;
    free(rank); free(qn);

    /* Per-leaf scratch (nL entries) */
    int32_t* leaf_tree = (int32_t*)malloc(nL * sizeof(int32_t));
    uint64_t* doff = (uint64_t*)malloc(nL * sizeof(uint64_t));
    uint32_t* lens = (uint32_t*)malloc(nL * sizeof(uint32_t));
    /* Sparse windows : ~ stride entries × 8B per slot. Reuse SRT concept. */
    /* For SLT, we need to look up (leaf_id → packed). Same window approach as SRT. */
    uint32_t max_stride = 0;
    for (int t = 0; t < nt; t++) if (f->stores[t].sample_stride > max_stride) max_stride = f->stores[t].sample_stride;
    uint32_t window_n = max_stride + 1;
    SltSparseEntry* wbuf = (SltSparseEntry*)malloc(nL * window_n * sizeof(SltSparseEntry));
    uint32_t* wlen = (uint32_t*)calloc(nL, sizeof(uint32_t));
    if (!leaf_tree || !doff || !lens || !wbuf || !wlen) {
        free(leaves); free(leaf_tree); free(doff); free(lens); free(wbuf); free(wlen);
        return -1;
    }

    /* Phase 1 : read sparse windows for each probe leaf.
       k_shift = 0 : single storage leaf per probe (native depth).
       k_shift > 0 : subtree expansion — one probe → range of storage leaves
                     in [lf << k_shift, (lf+1) << k_shift). Read window
                     around low_leaf, walk it to find all matching entries. */
    int p1 = 0;
    struct io_uring* ring = &f->ring;
    for (size_t li = 0; li < nL; li++) {
        int t = (int)(li % nt);
        leaf_tree[li] = t;
        int32_t lf = leaves[li];
        if (lf < 0) continue;
        SltStore* st = &f->stores[t];
        if (st->n_nonempty == 0) continue;
        uint32_t low_leaf = (k_shift > 0)
            ? ((uint32_t)lf << k_shift)
            : (uint32_t)lf;
        uint32_t bucket = slt_sample_bucket(st, low_leaf);
        uint32_t start = bucket * st->sample_stride;
        if (start >= st->n_nonempty) continue;
        /* For k_shift > 0 : read a bigger window to catch multiple storage
           leaves in the subtree range. Cap at 2×stride for MVP. */
        uint32_t take = st->sample_stride + 1;
        if (k_shift > 0) take *= 2;
        if (take > window_n) take = window_n;
        if (start + take > st->n_nonempty) take = st->n_nonempty - start;
        wlen[li] = take;
        uint64_t off = st->index_base + (uint64_t)start * sizeof(SltSparseEntry);
        if (f->ring_ok) {
            struct io_uring_sqe* sqe = io_uring_get_sqe(ring);
            if (!sqe) { io_uring_submit(ring); sqe = io_uring_get_sqe(ring); }
            io_uring_prep_read(sqe, st->fd, wbuf + li * window_n,
                               (unsigned)(take * sizeof(SltSparseEntry)), off);
            io_uring_sqe_set_data(sqe, (void*)(uintptr_t)li);
            p1++;
        } else {
            pread(st->fd, wbuf + li * window_n, take * sizeof(SltSparseEntry), (off_t)off);
        }
    }
    if (f->ring_ok) {
        io_uring_submit(ring);
        for (int r = 0; r < p1; r++) {
            struct io_uring_cqe* cqe;
            if (io_uring_wait_cqe(ring, &cqe) == 0) io_uring_cqe_seen(ring, cqe);
        }
    }

    /* COALESCING pass : within a subtree, class-C leaves have consecutive
       slot_ids (invariant : converter walks leaves in leaf_id order and
       increments slot_id per class). So group entries by class per probe,
       emit ONE range read per (tree, probe, class) covering all consecutive
       slots. Cuts reads at qd<depth from ~20k to ~1k on SIFT 10M.

       SltClassRun : one io_uring read. n_leaves consecutive slot_size-aligned
       slots starting at byte_off. Actual per-leaf doc count in leaves_ndocs[]. */
    typedef struct {
        uint32_t tree;
        uint32_t n_leaves;
        uint32_t slot_size;    /* class_sizes[cls] */
        uint64_t byte_off;
        uint32_t buf_off;      /* start pos in docs[] buffer */
        uint32_t ndocs_off;    /* start pos in leaves_ndocs[] */
    } SltClassRun;

    /* Upper bounds : max runs = nL * SLT_N_CLASSES.
       max leaves total = nL * window_n (each sparse entry = 1 leaf max). */
    uint64_t max_runs = nL * (uint64_t)SLT_N_CLASSES;
    SltClassRun* runs = (SltClassRun*)malloc(max_runs * sizeof(SltClassRun));
    uint32_t* leaves_ndocs = (uint32_t*)malloc((size_t)nL * window_n * sizeof(uint32_t));
    /* Per-probe scratch : per-class first_slot + count + per-leaf n_docs list. */
    if (!runs || !leaves_ndocs) {
        free(runs); free(leaves_ndocs);
        free(leaves); free(leaf_tree); free(doff); free(lens); free(wbuf); free(wlen);
        return -1;
    }

    uint64_t n_runs = 0;
    uint64_t total_bytes_read = 0;
    uint32_t flat_buf_pos = 0;
    uint32_t flat_ndocs_pos = 0;

    /* Per-class scratch reused across probes. */
    uint32_t cls_first_slot[SLT_N_CLASSES];
    uint32_t cls_count[SLT_N_CLASSES];
    /* Per-class temp ndocs (max 256 leaves per class per subtree at qd=20). */
    uint32_t cls_tmp_ndocs[SLT_N_CLASSES][512];

    for (size_t li = 0; li < nL; li++) {
        int32_t lf = leaves[li];
        if (lf < 0 || wlen[li] == 0) continue;
        int t = leaf_tree[li];
        SltStore* st = &f->stores[t];
        uint32_t low_leaf  = (k_shift > 0) ? ((uint32_t)lf << k_shift) : (uint32_t)lf;
        uint32_t high_leaf = (k_shift > 0) ? (((uint32_t)lf + 1u) << k_shift) : ((uint32_t)lf + 1u);
        SltSparseEntry* w = wbuf + li * window_n;

        for (int c = 0; c < SLT_N_CLASSES; c++) cls_count[c] = 0;
        for (uint32_t i = 0; i < wlen[li]; i++) {
            uint32_t leaf_id = w[i].leaf_id;
            if (leaf_id < low_leaf)  continue;
            if (leaf_id >= high_leaf) break;
            uint32_t cls  = SLT_CLASS(w[i].packed);
            uint32_t slot = SLT_SLOT(w[i].packed);
            uint32_t nd   = w[i].n_docs;
            if (cls_count[cls] == 0) cls_first_slot[cls] = slot;
            if (cls_count[cls] < 512) cls_tmp_ndocs[cls][cls_count[cls]] = nd;
            cls_count[cls]++;
        }
        /* Emit one run per non-empty class. */
        for (int c = 0; c < SLT_N_CLASSES; c++) {
            if (cls_count[c] == 0) continue;
            uint32_t n_leaves = cls_count[c];
            if (n_leaves > 512) n_leaves = 512;  /* safety : capped scratch */
            uint32_t slot_size = st->class_sizes[c];
            runs[n_runs].tree      = (uint32_t)t;
            runs[n_runs].n_leaves  = n_leaves;
            runs[n_runs].slot_size = slot_size;
            runs[n_runs].byte_off  = st->data_base + st->class_data_offset[c]
                                     + (uint64_t)cls_first_slot[c] * slot_size;
            runs[n_runs].buf_off   = flat_buf_pos;
            runs[n_runs].ndocs_off = flat_ndocs_pos;
            /* Copy per-leaf n_docs. */
            for (uint32_t i = 0; i < n_leaves; i++)
                leaves_ndocs[flat_ndocs_pos + i] = cls_tmp_ndocs[c][i];
            flat_ndocs_pos += n_leaves;
            flat_buf_pos += n_leaves * (slot_size / 4u);
            total_bytes_read += (uint64_t)n_leaves * slot_size;
            n_runs++;
        }
    }
    uint64_t total_docs_read = flat_buf_pos;  /* words = uint32_t count of docs buffer */

    /* Phase 2 : issue coalesced reads — one io_uring op per class-run. */
    uint32_t* docs = (uint32_t*)malloc((total_docs_read ? total_docs_read : 1) * sizeof(uint32_t));
    if (!docs) {
        free(runs); free(leaves_ndocs);
        free(leaves); free(leaf_tree); free(doff); free(lens); free(wbuf); free(wlen);
        return -1;
    }
    for (uint64_t k = 0; k < n_runs; k++) {
        uint32_t nb = runs[k].n_leaves * runs[k].slot_size;
        if (f->ring_ok) {
            struct io_uring_sqe* sqe = io_uring_get_sqe(ring);
            if (!sqe) { io_uring_submit(ring); sqe = io_uring_get_sqe(ring); }
            io_uring_prep_read(sqe, f->stores[runs[k].tree].fd,
                               docs + runs[k].buf_off, nb, runs[k].byte_off);
            io_uring_sqe_set_data(sqe, (void*)(uintptr_t)k);
        } else {
            pread(f->stores[runs[k].tree].fd, docs + runs[k].buf_off, nb,
                  (off_t)runs[k].byte_off);
        }
    }
    if (f->ring_ok) {
        io_uring_submit(ring);
        for (uint64_t r = 0; r < n_runs; r++) {
            struct io_uring_cqe* cqe;
            if (io_uring_wait_cqe(ring, &cqe) == 0) io_uring_cqe_seen(ring, cqe);
        }
    }
    (void)total_bytes_read;

    /* RADIX-SORT path : pack (tree:32 | doc:32) into uint64 with doc in LOW
       bits so LSB radix sorts by doc. Then linear scan with per-tree vote
       dedup + histogram-based top-N pick.
       Actual N = sum of leaves_ndocs[]. Padding zeros are SKIPPED, so radix
       cost scales with real docs not slot capacity.                        */
    uint64_t N = 0;
    for (uint32_t i = 0; i < flat_ndocs_pos; i++) N += leaves_ndocs[i];
    if (N == 0) {
        free(runs); free(leaves_ndocs);
        free(leaves); free(leaf_tree); free(doff); free(lens); free(wbuf); free(wlen); free(docs);
        return 0;
    }
    uint64_t* pairs   = (uint64_t*)malloc(N * sizeof(uint64_t));
    uint64_t* scratch = (uint64_t*)malloc(N * sizeof(uint64_t));
    if (!pairs || !scratch) {
        free(pairs); free(scratch);
        free(runs); free(leaves_ndocs);
        free(leaves); free(leaf_tree); free(doff); free(lens); free(wbuf); free(wlen); free(docs);
        return -1;
    }
    /* Pack : walk runs[], and for each run walk n_leaves slots — pack only
       leaves_ndocs[j] docs per slot (skip padding zeros).                 */
    {
        uint64_t wr = 0;
        for (uint64_t k = 0; k < n_runs; k++) {
            uint64_t tag = ((uint64_t)runs[k].tree) << 32;
            uint32_t docs_per_slot = runs[k].slot_size / 4u;
            uint32_t nd_off = runs[k].ndocs_off;
            uint32_t buf_off = runs[k].buf_off;
            for (uint32_t li_r = 0; li_r < runs[k].n_leaves; li_r++) {
                uint32_t nd = leaves_ndocs[nd_off + li_r];
                uint32_t base = buf_off + li_r * docs_per_slot;
                for (uint32_t i = 0; i < nd; i++) {
                    pairs[wr++] = tag | (uint64_t)docs[base + i];
                }
            }
        }
    }

    /* LSB radix sort on doc (low 32 bits). Passes tuned for n_docs.        */
    uint32_t max_doc = (uint32_t)(f->n_docs > 0 ? f->n_docs - 1 : 0);
    int doc_bits = max_doc ? 32 - __builtin_clz(max_doc) : 1;
    int n_passes;
    int BITS[3]; int SHIFTS[3]; uint32_t MASKS[3];
    if (doc_bits <= 22) {
        n_passes = 2;
        BITS[0] = 11; BITS[1] = 11;
        SHIFTS[0] = 0; SHIFTS[1] = 11;
        MASKS[0] = 0x7FFu; MASKS[1] = 0x7FFu;
    } else {
        n_passes = 3;
        BITS[0] = 11; BITS[1] = 11; BITS[2] = 10;
        SHIFTS[0] = 0; SHIFTS[1] = 11; SHIFTS[2] = 22;
        MASKS[0] = 0x7FFu; MASKS[1] = 0x7FFu; MASKS[2] = 0x3FFu;
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
            hist [(uint32_t)((src[i+0] >> shift) & mask)]++;
            hist1[(uint32_t)((src[i+1] >> shift) & mask)]++;
            hist2[(uint32_t)((src[i+2] >> shift) & mask)]++;
            hist3[(uint32_t)((src[i+3] >> shift) & mask)]++;
        }
        for (; i < N; i++) hist[(src[i] >> shift) & mask]++;
        for (int b = 0; b < n_buckets; b++)
            hist[b] += hist1[b] + hist2[b] + hist3[b];
        uint32_t acc = 0;
        for (int b = 0; b < n_buckets; b++) {
            uint32_t c = hist[b];
            hist[b] = acc;
            acc += c;
        }
        for (size_t ii = 0; ii < N; ii++) {
            uint32_t b = (src[ii] >> shift) & mask;
            dst[hist[b]++] = src[ii];
        }
        uint64_t* t = src; src = dst; dst = t;
    }

    /* Linear scan : per-tree vote dedup + histogram-based top-N (O(N)). */
    int32_t* tree_seen = (int32_t*)malloc((size_t)nt * sizeof(int32_t));
    if (!tree_seen) {
        free(pairs); free(scratch);
        free(leaves); free(leaf_tree); free(doff); free(lens); free(wbuf); free(wlen); free(docs);
        return -1;
    }
    for (int t = 0; t < nt; t++) tree_seen[t] = -1;
    typedef struct { int32_t doc; int32_t vote; } IdVoteSlt;
    IdVoteSlt* topn_h = (IdVoteSlt*)malloc(N * sizeof(IdVoteSlt));
    if (!topn_h) {
        free(tree_seen); free(pairs); free(scratch);
        free(leaves); free(leaf_tree); free(doff); free(lens); free(wbuf); free(wlen); free(docs);
        return -1;
    }
    uint32_t vote_hist[257] = {0};
    const int MAX_VOTE = (nt < 256) ? nt : 256;
    int32_t prev_doc = -1, vote = 0;
    size_t n_distinct = 0;
    for (size_t i = 0; i < N; i++) {
        uint64_t p = src[i];
        int32_t doc = (int32_t)(p & 0xffffffffu);
        int32_t tree = (int32_t)(p >> 32);
        if (doc != prev_doc) {
            if (prev_doc >= 0) {
                int v = (vote > MAX_VOTE) ? MAX_VOTE : vote;
                topn_h[n_distinct].doc = prev_doc;
                topn_h[n_distinct].vote = v;
                n_distinct++;
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
        topn_h[n_distinct].doc = prev_doc;
        topn_h[n_distinct].vote = v;
        n_distinct++;
        vote_hist[v]++;
    }

    /* Histogram-based top-N pick : find threshold vote. */
    int keep = 0;
    int threshold = -1;
    uint32_t vcum = 0;
    for (int v = MAX_VOTE; v >= 0; v--) {
        vcum += vote_hist[v];
        if (vcum >= (uint32_t)top_n) { threshold = v; break; }
    }
    if (threshold < 0) threshold = 0;
    /* Emit docs with vote > threshold + partial docs at threshold up to top_n. */
    int at_threshold_needed = top_n;
    for (int v = MAX_VOTE; v > threshold; v--) at_threshold_needed -= vote_hist[v];
    if (at_threshold_needed < 0) at_threshold_needed = 0;
    for (size_t i = 0; i < n_distinct && keep < top_n; i++) {
        int v = topn_h[i].vote;
        if (v > threshold) {
            out_ids[keep] = topn_h[i].doc;
            out_votes[keep] = v;
            keep++;
        } else if (v == threshold && at_threshold_needed > 0) {
            out_ids[keep] = topn_h[i].doc;
            out_votes[keep] = v;
            keep++;
            at_threshold_needed--;
        }
    }

    free(topn_h); free(tree_seen); free(pairs); free(scratch);
    free(runs); free(leaves_ndocs);
    free(leaves); free(leaf_tree); free(doff); free(lens); free(wbuf); free(wlen);
    if (docs) free(docs);
    return keep;
}
