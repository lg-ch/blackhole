#define _POSIX_C_SOURCE 200809L
#include "recall.h"
#include "liburing_compat.h"
#include "query_tree.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <liburing.h>

static long peak_rss_kb(void) {
    struct rusage ru; getrusage(RUSAGE_SELF, &ru); return ru.ru_maxrss;
}
static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int read_fvec(int fd, int idx, int dim, float* out) {
    off_t row = 4 + (off_t)dim * 4;
    off_t off = (off_t)idx * row + 4;   /* skip the 4-byte dim header */
    ssize_t n = pread(fd, out, (size_t)dim * 4, off);
    return (n == (ssize_t)(dim * 4)) ? 0 : -1;
}

int read_vec_at(int fd, VecFmt fmt, int idx, int dim, float* out) {
    off_t off = vec_row_offset(fmt, idx, dim);
    if (fmt == VECFMT_U8BIN || fmt == VECFMT_BVECS) {
        uint8_t buf[1024];  /* dim is typically 128 */
        ssize_t n = pread(fd, buf, (size_t)dim, off);
        if (n != (ssize_t)dim) return -1;
        vec_u8_to_f32(buf, out, dim);
        return 0;
    }
    if (fmt == VECFMT_F16BIN) {
        uint16_t buf16[1024];
        if (dim > 1024) return -1;
        ssize_t n = pread(fd, buf16, (size_t)dim * 2, off);
        if (n != (ssize_t)(dim * 2)) return -1;
        vec_f16_to_f32(buf16, out, dim);
        return 0;
    }
    ssize_t n = pread(fd, out, (size_t)dim * 4, off);
    return (n == (ssize_t)(dim * 4)) ? 0 : -1;
}

int read_ivec(int fd, int idx, int len, int* out) {
    off_t row = 4 + (off_t)len * 4;
    off_t off = (off_t)idx * row + 4;
    ssize_t n = pread(fd, out, (size_t)len * 4, off);
    return (n == (ssize_t)(len * 4)) ? 0 : -1;
}

#ifdef __ARM_NEON
#include <arm_neon.h>
float l2sq(const float* a, const float* b, int n) {
    float32x4_t s = vdupq_n_f32(0);
    int i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t d = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
        s = vfmaq_f32(s, d, d);
    }
    float r = vaddvq_f32(s);
    for (; i < n; i++) { float d = a[i] - b[i]; r += d * d; }
    return r;
}
#else
float l2sq(const float* a, const float* b, int n) {
    float r = 0;
    for (int i = 0; i < n; i++) { float d = a[i] - b[i]; r += d * d; }
    return r;
}
#endif

/* Pair (distance, id) for partial sort. */
typedef struct { float d; int32_t id; } DistId;

/* Min-heap (root = largest distance) so we can keep top-K smallest.
   When a new candidate beats the root, replace and sift-down.          */
static void heap_sift_down(DistId* h, int n, int i) {
    while (1) {
        int l = 2*i + 1, r = 2*i + 2, biggest = i;
        if (l < n && h[l].d > h[biggest].d) biggest = l;
        if (r < n && h[r].d > h[biggest].d) biggest = r;
        if (biggest == i) break;
        DistId tmp = h[i]; h[i] = h[biggest]; h[biggest] = tmp;
        i = biggest;
    }
}

static int cmp_distid(const void* a, const void* b) {
    float da = ((const DistId*)a)->d, db = ((const DistId*)b)->d;
    return (da < db) ? -1 : (da > db);
}

int rerank_l2_uring(struct io_uring* ring,
                    int base_fd, VecFmt base_fmt, int dim, const float* qvec,
                    const int32_t* cand_ids, int n_cands,
                    int top_k, int32_t* out_ids) {
    if (n_cands <= 0) return 0;
    int k = (top_k < n_cands) ? top_k : n_cands;

    size_t row_bytes = vec_row_bytes(base_fmt, dim);
    /* For u8bin we read into a uint8 buffer; for fvecs into a float buffer.
       Both addressed via raw uint8_t* of n_cands × row_bytes.            */
    uint8_t* raw = (uint8_t*)malloc((size_t)n_cands * row_bytes);
    if (!raw) return -1;

    for (int i = 0; i < n_cands; i++) {
        struct io_uring_sqe* sqe = io_uring_get_sqe(ring);
        if (!sqe) {
            io_uring_submit(ring);
            sqe = io_uring_get_sqe(ring);
            if (!sqe) { free(raw); return -1; }
        }
        off_t off = vec_row_offset(base_fmt, cand_ids[i], dim);
        io_uring_prep_read(sqe, base_fd, raw + (size_t)i * row_bytes,
                           (unsigned)row_bytes, (uint64_t)off);
        io_uring_sqe_set_data(sqe, (void*)(uintptr_t)i);
    }
    int submitted = io_uring_submit(ring);
    if (submitted < 0) { free(raw); return -1; }

    /* Phase 1 : drain toutes les CQE. On perd le CPU/IO overlap mais on
       débloque le pass CPU en parallèle sur tous les cands d'un coup. */
    uint8_t* valid = (uint8_t*)calloc((size_t)n_cands, 1);
    if (!valid) { free(raw); return -1; }
    for (int reaped = 0; reaped < n_cands; reaped++) {
        struct io_uring_cqe* cqe;
        if (io_uring_wait_cqe(ring, &cqe) < 0) continue;
        int idx = (int)(uintptr_t)io_uring_cqe_get_data(cqe);
        int res = cqe->res;
        io_uring_cqe_seen(ring, cqe);
        if (res == (int)row_bytes) valid[idx] = 1;
    }

    /* Phase 2 : l2sq parallèle. Distance INFINITY pour les reads invalides
       → ignorées naturellement par la réduction. */
    float* dists = (float*)malloc((size_t)n_cands * sizeof(float));
    if (!dists) { free(valid); free(raw); return -1; }
    int is_u8  = (base_fmt == VECFMT_U8BIN || base_fmt == VECFMT_BVECS);
    int is_f16 = (base_fmt == VECFMT_F16BIN);
    #pragma omp parallel
    {
        float tmpf[1024];
        #pragma omp for schedule(static)
        for (int i = 0; i < n_cands; i++) {
            if (!valid[i]) { dists[i] = INFINITY; continue; }
            const float* fv;
            if (is_u8) {
                vec_u8_to_f32(raw + (size_t)i * row_bytes, tmpf, dim);
                fv = tmpf;
            } else if (is_f16) {
                vec_f16_to_f32((const uint16_t*)(raw + (size_t)i * row_bytes),
                               tmpf, dim);
                fv = tmpf;
            } else {
                fv = (const float*)(raw + (size_t)i * row_bytes);
            }
            dists[i] = l2sq(qvec, fv, dim);
        }
    }
    free(valid);

    /* Phase 3 : réduction sérielle heap (k typiquement 10-100 → très cheap). */
    DistId* heap = (DistId*)malloc((size_t)k * sizeof(DistId));
    if (!heap) { free(dists); free(raw); return -1; }
    int heap_n = 0;
    for (int i = 0; i < n_cands; i++) {
        float d = dists[i];
        if (d == INFINITY) continue;
        int32_t id = cand_ids[i];
        if (heap_n < k) {
            heap[heap_n].d = d; heap[heap_n].id = id;
            heap_n++;
            if (heap_n == k)
                for (int j = k/2 - 1; j >= 0; j--) heap_sift_down(heap, k, j);
        } else if (d < heap[0].d) {
            heap[0].d = d; heap[0].id = id;
            heap_sift_down(heap, k, 0);
        }
    }

    qsort(heap, (size_t)k, sizeof(DistId), cmp_distid);
    for (int i = 0; i < k; i++) out_ids[i] = heap[i].id;

    free(heap); free(dists); free(raw);
    return k;
}

int rerank_l2(int base_fd, VecFmt base_fmt, int dim, const float* qvec,
              const int32_t* cand_ids, int n_cands,
              int top_k, int32_t* out_ids) {
    if (n_cands <= 0) return 0;
    int k = (top_k < n_cands) ? top_k : n_cands;

    DistId* heap = (DistId*)malloc((size_t)k * sizeof(DistId));
    float*  cv   = (float*) malloc((size_t)dim * sizeof(float));
    if (!heap || !cv) { free(heap); free(cv); return -1; }

    for (int i = 0; i < k; i++) {
        if (read_vec_at(base_fd, base_fmt, cand_ids[i], dim, cv) != 0) {
            heap[i].d = INFINITY; heap[i].id = cand_ids[i]; continue;
        }
        heap[i].d  = l2sq(qvec, cv, dim);
        heap[i].id = cand_ids[i];
    }
    for (int i = k/2 - 1; i >= 0; i--) heap_sift_down(heap, k, i);

    for (int i = k; i < n_cands; i++) {
        if (read_vec_at(base_fd, base_fmt, cand_ids[i], dim, cv) != 0) continue;
        float d = l2sq(qvec, cv, dim);
        if (d < heap[0].d) {
            heap[0].d  = d;
            heap[0].id = cand_ids[i];
            heap_sift_down(heap, k, 0);
        }
    }

    qsort(heap, (size_t)k, sizeof(DistId), cmp_distid);
    for (int i = 0; i < k; i++) out_ids[i] = heap[i].id;

    free(heap); free(cv);
    return k;
}

int eval_recall(Forest* f,
                const char* qvecs_path, const char* gt_path,
                const char* base_path,
                int n_queries, int top_k, int threshold,
                int max_cands,
                double* out_recall, double* out_avg_cands,
                double* out_ms_per_query) {
    int dim = f->dim;
    VecFmt qfmt = vec_fmt_from_path(qvecs_path);
    VecFmt bfmt = vec_fmt_from_path(base_path);

    int qfd = open(qvecs_path,   O_RDONLY);
    int gfd = open(gt_path,      O_RDONLY);
    int bfd = open(base_path,    O_RDONLY);
    if (qfd < 0 || gfd < 0 || bfd < 0) {
        perror("open"); return -1;
    }
    /* GT_K from the ivecs file header (first int32). Default 100 if read fails. */
    int GT_K = 100;
    {
        int hdr;
        if (pread(gfd, &hdr, 4, 0) == 4 && hdr > 0 && hdr <= 10000) GT_K = hdr;
    }

    /* Buffers reused across queries. */
    float*    qvec      = (float*)   malloc((size_t)dim * 4);
    uint16_t* votes     = (uint16_t*)malloc((size_t)f->n_docs * sizeof(uint16_t));
    int32_t*  cand_ids  = (int32_t*) malloc((size_t)max_cands * 4);
    int32_t*  cand_vot  = (int32_t*) malloc((size_t)max_cands * 4);
    int32_t*  top_ids   = (int32_t*) malloc((size_t)top_k * 4);
    int*      gt        = (int*)     malloc((size_t)GT_K * 4);

    if (!qvec || !votes || !cand_ids || !cand_vot || !top_ids || !gt) {
        fprintf(stderr, "alloc failed\n"); return -1;
    }

    long long hits = 0, total_cands = 0;
    double t0 = now_s();

    for (int q = 0; q < n_queries; q++) {
        if (read_vec_at(qfd, qfmt, q, dim, qvec) != 0) { n_queries = q; break; }
        if (read_ivec(gfd, q, GT_K, gt) != 0)          { n_queries = q; break; }

        int n_c = forest_collect(f, qvec, threshold, max_cands,
                                 cand_ids, cand_vot, votes);
        total_cands += n_c;

        int n_top = rerank_l2(bfd, bfmt, dim, qvec, cand_ids, n_c, top_k, top_ids);

        /* Count how many of our top_k are in groundtruth's top_k. */
        for (int i = 0; i < n_top; i++) {
            int id = top_ids[i];
            for (int j = 0; j < top_k && j < GT_K; j++)
                if (gt[j] == id) { hits++; break; }
        }

        if ((q + 1) % 100 == 0 || q + 1 == n_queries) {
            double dt = now_s() - t0;
            fprintf(stderr,
                "  q=%4d/%d  recall@%d=%.4f  avg_cands=%.0f  %.1f ms/q  rss=%ld KB\n",
                q + 1, n_queries, top_k,
                (double)hits / ((q + 1) * (double)top_k),
                (double)total_cands / (q + 1),
                dt * 1000.0 / (q + 1),
                peak_rss_kb());
        }
    }

    double dt = now_s() - t0;
    *out_recall       = (double)hits / ((double)n_queries * top_k);
    *out_avg_cands    = (double)total_cands / n_queries;
    *out_ms_per_query = dt * 1000.0 / n_queries;

    free(qvec); free(votes); free(cand_ids); free(cand_vot);
    free(top_ids); free(gt);
    close(qfd); close(gfd); close(bfd);
    return 0;
}

int eval_recall_topn(Forest* f,
                     const char* qvecs_path, const char* gt_path,
                     const char* base_path,
                     int n_queries, int top_k, int top_n_cands,
                     int query_depth,
                     const roaring_bitmap_t* allowed,
                     double* out_recall, double* out_avg_cands,
                     double* out_ms_per_query) {
    int dim = f->dim;
    const int GT_K = 100;
    VecFmt qfmt = vec_fmt_from_path(qvecs_path);
    VecFmt bfmt = vec_fmt_from_path(base_path);

    int qfd = open(qvecs_path, O_RDONLY);
    int gfd = open(gt_path,    O_RDONLY);
    int bfd = open(base_path,  O_RDONLY);
    if (qfd < 0 || gfd < 0 || bfd < 0) { perror("open"); return -1; }

    float*    qvec     = (float*)   malloc((size_t)dim * 4);
    int32_t*  cand_ids = (int32_t*) malloc((size_t)top_n_cands * 4);
    int32_t*  cand_vot = (int32_t*) malloc((size_t)top_n_cands * 4);
    int32_t*  top_ids  = (int32_t*) malloc((size_t)top_k * 4);
    int*      gt       = (int*)     malloc((size_t)GT_K * 4);
    if (!qvec || !cand_ids || !cand_vot || !top_ids || !gt) {
        fprintf(stderr, "alloc failed\n"); return -1;
    }

    long long hits = 0, total_cands = 0, total_distinct = 0;
    double t0 = now_s();

    for (int q = 0; q < n_queries; q++) {
        if (read_vec_at(qfd, qfmt, q, dim, qvec) != 0) { n_queries = q; break; }
        if (read_ivec(gfd, q, GT_K, gt) != 0)          { n_queries = q; break; }

        int n_c = forest_collect_topn(f, qvec, top_n_cands, query_depth,
                                      allowed,
                                      cand_ids, cand_vot);
        total_cands    += n_c;
        total_distinct += forest_get_last_n_distinct();

        int n_top = f->ring_ok
            ? rerank_l2_uring(&f->ring, bfd, bfmt, dim, qvec, cand_ids, n_c, top_k, top_ids)
            : rerank_l2      (        bfd, bfmt, dim, qvec, cand_ids, n_c, top_k, top_ids);

        for (int i = 0; i < n_top; i++) {
            int id = top_ids[i];
            for (int j = 0; j < top_k && j < GT_K; j++)
                if (gt[j] == id) { hits++; break; }
        }

        if ((q + 1) % 100 == 0 || q + 1 == n_queries) {
            double dt = now_s() - t0;
            fprintf(stderr,
                "  q=%4d/%d  recall@%d=%.4f  avg_cands=%.0f  avg_distinct=%.0f  %.2f ms/q  rss=%ld KB\n",
                q + 1, n_queries, top_k,
                (double)hits / ((q + 1) * (double)top_k),
                (double)total_cands / (q + 1),
                (double)total_distinct / (q + 1),
                dt * 1000.0 / (q + 1),
                peak_rss_kb());
        }
    }

    double dt = now_s() - t0;
    *out_recall       = (double)hits / ((double)n_queries * top_k);
    *out_avg_cands    = (double)total_cands / n_queries;
    *out_ms_per_query = dt * 1000.0 / n_queries;
    fprintf(stderr, "telemetry: avg_distinct=%.0f (total cands visited by K-way merge)\n",
            (double)total_distinct / n_queries);

    free(qvec); free(cand_ids); free(cand_vot);
    free(top_ids); free(gt);
    close(qfd); close(gfd); close(bfd);
    return 0;
}

