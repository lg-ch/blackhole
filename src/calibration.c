#include "calibration.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
static float l2sq(const float* a, const float* b, int dim) {
    float32x4_t s = vdupq_n_f32(0);
    int i = 0;
    for (; i + 3 < dim; i += 4) {
        float32x4_t d = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
        s = vfmaq_f32(s, d, d);
    }
    float r = vaddvq_f32(s);
    for (; i < dim; i++) { float d = a[i] - b[i]; r += d * d; }
    return r;
}
#else
static float l2sq(const float* a, const float* b, int dim) {
    float s = 0;
    for (int i = 0; i < dim; i++) { float d = a[i] - b[i]; s += d * d; }
    return s;
}
#endif

typedef struct { float dist; int32_t doc_id; } DistDoc;

static struct {
    int    n_queries;
    int    top_k;
    int    dim;
    int    q_filled;
    float* queries;         /* n_queries × dim */
    int32_t* q_doc_ids;     /* n_queries */
    DistDoc* heaps;         /* n_queries × top_k, max-heap by dist */
} g;

int calib_is_enabled(void) { return g.n_queries > 0; }

int calib_init(int n_queries, int top_k, int dim) {
    if (g.n_queries > 0) return 0;   /* already inited */
    if (n_queries <= 0 || top_k <= 0 || dim <= 0) return -1;
    g.n_queries = n_queries;
    g.top_k     = top_k;
    g.dim       = dim;
    g.q_filled  = 0;
    g.queries   = (float*)calloc((size_t)n_queries * dim, sizeof(float));
    g.q_doc_ids = (int32_t*)malloc((size_t)n_queries * sizeof(int32_t));
    g.heaps     = (DistDoc*)malloc((size_t)n_queries * top_k * sizeof(DistDoc));
    if (!g.queries || !g.q_doc_ids || !g.heaps) { calib_free(); return -1; }
    for (int i = 0; i < n_queries * top_k; i++) {
        g.heaps[i].dist   = INFINITY;
        g.heaps[i].doc_id = -1;
    }
    fprintf(stderr, "  calib enabled : n_queries=%d top_k=%d dim=%d\n",
            n_queries, top_k, dim);
    return 0;
}

/* Max-heap sift-down. Root = biggest dist ; replace if new dist smaller. */
static void heap_push(DistDoc* h, int k, float dist, int32_t doc_id) {
    if (dist >= h[0].dist) return;
    h[0].dist = dist;
    h[0].doc_id = doc_id;
    int i = 0;
    while (1) {
        int l = 2*i + 1, r = 2*i + 2, m = i;
        if (l < k && h[l].dist > h[m].dist) m = l;
        if (r < k && h[r].dist > h[m].dist) m = r;
        if (m == i) break;
        DistDoc tmp = h[i]; h[i] = h[m]; h[m] = tmp;
        i = m;
    }
}

void calib_update(const float* vec_batch, const int32_t* doc_ids, int batch_n) {
    if (g.n_queries == 0) return;
    int i = 0;
    /* Fill query pool with first N docs */
    while (g.q_filled < g.n_queries && i < batch_n) {
        int slot = g.q_filled++;
        memcpy(&g.queries[(size_t)slot * g.dim],
               &vec_batch[(size_t)i * g.dim],
               (size_t)g.dim * sizeof(float));
        g.q_doc_ids[slot] = doc_ids[i];
        i++;
    }
    /* Update all query heaps with remaining docs of batch */
    for (; i < batch_n; i++) {
        const float* v = &vec_batch[(size_t)i * g.dim];
        int32_t doc = doc_ids[i];
        for (int q = 0; q < g.n_queries; q++) {
            if (g.q_doc_ids[q] == doc) continue;   /* self-exclude */
            float d = l2sq(&g.queries[(size_t)q * g.dim], v, g.dim);
            heap_push(&g.heaps[(size_t)q * g.top_k], g.top_k, d, doc);
        }
    }
}

static int cmp_distdoc(const void* a, const void* b) {
    float da = ((const DistDoc*)a)->dist;
    float db = ((const DistDoc*)b)->dist;
    return (da > db) - (da < db);
}

int calib_snapshot(const char* index_dir) {
    if (g.n_queries == 0) return -1;
    char path[1024];

    /* Write queries: [u32 n_queries][u32 dim][float queries][int32 doc_ids] */
    snprintf(path, sizeof(path), "%s/calibration_queries.bin", index_dir);
    FILE* fq = fopen(path, "wb");
    if (!fq) return -1;
    uint32_t hdr[2] = { (uint32_t)g.n_queries, (uint32_t)g.dim };
    fwrite(hdr, 4, 2, fq);
    fwrite(g.queries, sizeof(float), (size_t)g.n_queries * g.dim, fq);
    fwrite(g.q_doc_ids, sizeof(int32_t), (size_t)g.n_queries, fq);
    fclose(fq);

    /* Write GT: [u32 n_queries][u32 top_k][int32 doc_ids (sorted asc by dist)] */
    snprintf(path, sizeof(path), "%s/calibration_gt.bin", index_dir);
    FILE* fg = fopen(path, "wb");
    if (!fg) return -1;
    hdr[0] = (uint32_t)g.n_queries;
    hdr[1] = (uint32_t)g.top_k;
    fwrite(hdr, 4, 2, fg);
    /* Sort each heap ascending by dist before write */
    DistDoc* tmp = (DistDoc*)malloc((size_t)g.top_k * sizeof(DistDoc));
    for (int q = 0; q < g.n_queries; q++) {
        memcpy(tmp, &g.heaps[(size_t)q * g.top_k], (size_t)g.top_k * sizeof(DistDoc));
        qsort(tmp, (size_t)g.top_k, sizeof(DistDoc), cmp_distdoc);
        for (int i = 0; i < g.top_k; i++) {
            fwrite(&tmp[i].doc_id, 4, 1, fg);
        }
    }
    free(tmp);
    fclose(fg);
    return 0;
}

void calib_free(void) {
    free(g.queries); g.queries = NULL;
    free(g.q_doc_ids); g.q_doc_ids = NULL;
    free(g.heaps); g.heaps = NULL;
    g.n_queries = 0;
    g.q_filled = 0;
}
