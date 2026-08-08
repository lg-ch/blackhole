#define _POSIX_C_SOURCE 200809L
#include "tquant.h"
#include "vec_format.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>

/* ---------- seeded PRNG (splitmix64) ---------- */
static uint64_t sm64(uint64_t* st) {
    uint64_t z = (*st += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* ---------- fast Walsh-Hadamard, d = power of two ---------- */
static void fwht(float* x, int d) {
    for (int h = 1; h < d; h <<= 1)
        for (int i = 0; i < d; i += h << 1)
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j] = a + b; x[j + h] = a - b;
            }
    float s = 1.0f / sqrtf((float)d);
    for (int i = 0; i < d; i++) x[i] *= s;
}

/* Generate TQ_ROUNDS sign vectors (±1) for pad_dim coords from seed. */
static void gen_signs(uint64_t seed, int pad_dim, float* signs /*ROUNDS×pad*/) {
    uint64_t st = seed ? seed : 1;
    for (int r = 0; r < TQ_ROUNDS; r++)
        for (int i = 0; i < pad_dim; i++)
            signs[r * pad_dim + i] = (sm64(&st) & 1) ? 1.0f : -1.0f;
}

/* In-place (HD)^ROUNDS rotation of a pad_dim vector. */
static void rotate(float* x, int pad_dim, const float* signs) {
    for (int r = 0; r < TQ_ROUNDS; r++) {
        const float* s = signs + r * pad_dim;
        for (int i = 0; i < pad_dim; i++) x[i] *= s[i];
        fwht(x, pad_dim);
    }
}

static int next_pow2(int v) {
    int p = 1;
    while (p < v) p <<= 1;
    return p;
}

static inline uint8_t qcode(float v, const float* bounds) {
    /* branchy binary search over 15 bounds — 4 steps */
    int lo = 0, hi = TQ_LEVELS - 1;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (v > bounds[mid]) lo = mid + 1; else hi = mid;
    }
    return (uint8_t)lo;
}

/* ---------- 1-D Lloyd-Max on a sample ---------- */
static void lloyd_max(const float* s, size_t n, float* centers, float* bounds) {
    /* init: spread between approx quantiles via min/max clamp */
    float lo = INFINITY, hi = -INFINITY;
    for (size_t i = 0; i < n; i++) {
        if (s[i] < lo) lo = s[i];
        if (s[i] > hi) hi = s[i];
    }
    /* soft-clip extremes: shrink by 1% to approximate 0.1/99.9 quantiles */
    float span = hi - lo;
    lo += 0.01f * span; hi -= 0.01f * span;
    for (int k = 0; k < TQ_LEVELS; k++)
        centers[k] = lo + (hi - lo) * ((float)k / (TQ_LEVELS - 1));

    double sum[TQ_LEVELS]; size_t cnt[TQ_LEVELS];
    for (int it = 0; it < 40; it++) {
        for (int k = 0; k < TQ_LEVELS - 1; k++)
            bounds[k] = 0.5f * (centers[k] + centers[k + 1]);
        memset(sum, 0, sizeof(sum)); memset(cnt, 0, sizeof(cnt));
        for (size_t i = 0; i < n; i++) {
            uint8_t c = qcode(s[i], bounds);
            sum[c] += s[i]; cnt[c]++;
        }
        for (int k = 0; k < TQ_LEVELS; k++)
            if (cnt[k]) centers[k] = (float)(sum[k] / cnt[k]);
    }
    for (int k = 0; k < TQ_LEVELS - 1; k++)
        bounds[k] = 0.5f * (centers[k] + centers[k + 1]);
}

/* ---------- build ---------- */
int tq_build(const char* fvecs_path, const char* out_path,
             int dim, uint64_t seed, int calib_rows) {
    VecFmt fmt = vec_fmt_from_path(fvecs_path);
    if (fmt != VECFMT_FVECS && fmt != VECFMT_FBIN) {
        fprintf(stderr, "tq_build: only fvecs/fbin float inputs supported\n");
        return -1;
    }
    int fd = open(fvecs_path, O_RDONLY);
    if (fd < 0) { perror("tq_build open"); return -1; }

    struct stat st;
    fstat(fd, &st);
    /* fvecs: per-row [4-byte dim][dim floats]
       fbin : single 8-byte header [n,dim] then raw float32 rows  */
    int has_row_hdr = (fmt == VECFMT_FVECS);
    off_t file_hdr  = (fmt == VECFMT_FBIN)  ? 8 : 0;
    size_t row_bytes = (has_row_hdr ? 4 : 0) + (size_t)dim * 4;
    uint64_t n_docs = (uint64_t)((st.st_size - file_hdr) / (off_t)row_bytes);

    int pad = next_pow2(dim);
    float* signs = (float*)malloc((size_t)TQ_ROUNDS * pad * sizeof(float));
    gen_signs(seed, pad, signs);

    /* --- calibration: stride-sample calib_rows vectors --- */
    if (calib_rows <= 0) calib_rows = 20000;
    if ((uint64_t)calib_rows > n_docs) calib_rows = (int)n_docs;
    uint64_t stride = n_docs / (uint64_t)calib_rows;
    if (stride == 0) stride = 1;

    /* subsample coords to ~2M floats for Lloyd */
    size_t want = 2 * 1000 * 1000;
    size_t per_row = want / (size_t)calib_rows + 1;
    if (per_row > (size_t)pad) per_row = (size_t)pad;
    float* sample = (float*)malloc((want + (size_t)pad) * sizeof(float));
    float* rowbuf = (float*)calloc((size_t)pad, sizeof(float));
    uint8_t* rawrow = (uint8_t*)malloc(row_bytes);
    size_t n_s = 0;

    size_t hdr_skip = has_row_hdr ? 4 : 0;  /* per-row header bytes */
    fprintf(stderr, "  tq calibrating: %d rows (stride %llu) ...\n",
            calib_rows, (unsigned long long)stride);
    for (int i = 0; i < calib_rows; i++) {
        off_t off = file_hdr +
                    (off_t)((uint64_t)i * stride) * (off_t)row_bytes;
        if (pread(fd, rawrow, row_bytes, off) != (ssize_t)row_bytes) break;
        memset(rowbuf, 0, (size_t)pad * sizeof(float));
        memcpy(rowbuf, rawrow + hdr_skip, (size_t)dim * sizeof(float));
        rotate(rowbuf, pad, signs);
        for (size_t j = 0; j < per_row && n_s < want; j++)
            sample[n_s++] = rowbuf[(j * 911) % (size_t)pad];
    }

    TqHeader h;
    memset(&h, 0, sizeof(h));
    h.magic = TQ_MAGIC; h.version = 1;
    h.dim = (uint32_t)dim; h.pad_dim = (uint32_t)pad;
    h.seed = seed; h.n_docs = n_docs;
    lloyd_max(sample, n_s, h.centers, h.bounds);
    free(sample);
    fprintf(stderr, "  tq codebook: [%.4f .. %.4f]\n",
            h.centers[0], h.centers[TQ_LEVELS-1]);

    /* --- streaming quantization --- */
    FILE* out = fopen(out_path, "wb");
    if (!out) { perror("tq_build fopen out"); free(signs); close(fd); return -1; }
    fwrite(&h, sizeof(h), 1, out);

    size_t code_bytes = (size_t)pad / 2;
    uint8_t* codes = (uint8_t*)malloc(code_bytes);

    /* sequential read of the whole base in big chunks */
    const int CHUNK = 8192;
    uint8_t* big = (uint8_t*)malloc((size_t)CHUNK * row_bytes);
    uint64_t done = 0;
    lseek(fd, file_hdr, SEEK_SET);    /* skip global header if any */
    while (done < n_docs) {
        int n = (int)((n_docs - done) < CHUNK ? (n_docs - done) : CHUNK);
        ssize_t got = read(fd, big, (size_t)n * row_bytes);
        if (got != (ssize_t)((size_t)n * row_bytes)) {
            fprintf(stderr, "tq_build short read at doc %llu\n",
                    (unsigned long long)done);
            break;
        }
        for (int i = 0; i < n; i++) {
            memset(rowbuf, 0, (size_t)pad * sizeof(float));
            memcpy(rowbuf, big + (size_t)i * row_bytes + hdr_skip,
                   (size_t)dim * sizeof(float));
            rotate(rowbuf, pad, signs);
            for (int j = 0; j < pad; j += 2) {
                uint8_t c0 = qcode(rowbuf[j],     h.bounds);
                uint8_t c1 = qcode(rowbuf[j + 1], h.bounds);
                codes[j >> 1] = (uint8_t)(c0 | (c1 << 4));
            }
            fwrite(codes, 1, code_bytes, out);
        }
        done += (uint64_t)n;
        if (done % 1000000 < (uint64_t)CHUNK)
            fprintf(stderr, "  tq quantized %llu / %llu\n",
                    (unsigned long long)done, (unsigned long long)n_docs);
    }
    free(big); free(codes); free(rowbuf); free(rawrow); free(signs);
    fclose(out); close(fd);
    fprintf(stderr, "  tq done: %llu docs -> %s\n",
            (unsigned long long)done, out_path);
    return (done == n_docs) ? 0 : -1;
}

/* ---------- reader ---------- */
struct TqReader {
    int fd;
    TqHeader h;
    float* signs;          /* ROUNDS × pad_dim */
    size_t code_bytes;     /* pad_dim / 2 */
};

TqReader* tq_open(const char* tq4_path) {
    int fd = open(tq4_path, O_RDONLY);
    if (fd < 0) return NULL;
    TqReader* r = (TqReader*)calloc(1, sizeof(TqReader));
    r->fd = fd;
    if (pread(fd, &r->h, sizeof(TqHeader), 0) != sizeof(TqHeader) ||
        r->h.magic != TQ_MAGIC) {
        close(fd); free(r); return NULL;
    }
    r->code_bytes = (size_t)r->h.pad_dim / 2;
    r->signs = (float*)malloc((size_t)TQ_ROUNDS * r->h.pad_dim * sizeof(float));
    gen_signs(r->h.seed, (int)r->h.pad_dim, r->signs);
    return r;
}

void tq_close(TqReader* r) {
    if (!r) return;
    close(r->fd); free(r->signs); free(r);
}

/* min-heap on score for top-kprime selection */
typedef struct { float s; int32_t id; } ScoreId;
static void sift_down_min(ScoreId* h, int n, int i) {
    for (;;) {
        int l = 2*i+1, rr = 2*i+2, m = i;
        if (l < n && h[l].s < h[m].s) m = l;
        if (rr < n && h[rr].s < h[m].s) m = rr;
        if (m == i) break;
        ScoreId t = h[i]; h[i] = h[m]; h[m] = t;
        i = m;
    }
}

int tq_select(TqReader* r, struct io_uring* ring,
              const float* qvec,
              const int32_t* cand_ids, int n_cands,
              int kprime, int32_t* out_ids) {
    if (!r || n_cands <= 0) return 0;
    int pad = (int)r->h.pad_dim, dim = (int)r->h.dim;
    int k = (kprime < n_cands) ? kprime : n_cands;

    /* rotate the query once, build the per-coordinate LUT */
    float* qr = (float*)calloc((size_t)pad, sizeof(float));
    memcpy(qr, qvec, (size_t)dim * sizeof(float));
    rotate(qr, pad, r->signs);
    float* lut = (float*)malloc((size_t)pad * TQ_LEVELS * sizeof(float));
    for (int j = 0; j < pad; j++)
        for (int c = 0; c < TQ_LEVELS; c++)
            lut[j * TQ_LEVELS + c] = qr[j] * r->h.centers[c];
    free(qr);

    uint8_t* raw = (uint8_t*)malloc((size_t)n_cands * r->code_bytes);
    if (!raw) { free(lut); return -1; }

    /* Sort cand_ids by ascending doc_id so io_uring submissions hit the
       file in near-sequential order. NVMe loves this (random→sequential
       gives ~3-5×). We keep a perm[] back-pointer so the score loop maps
       the read slot back to its original cand_id.                       */
    int32_t* sorted_ids = (int32_t*)malloc((size_t)n_cands * sizeof(int32_t));
    int*     perm       = (int*)    malloc((size_t)n_cands * sizeof(int));
    if (!sorted_ids || !perm) {
        free(sorted_ids); free(perm); free(raw); free(lut); return -1;
    }
    for (int i = 0; i < n_cands; i++) perm[i] = i;
    /* indirect insertion sort would scale; use qsort with closure via
       static — but n_cands ≤ ~128k so use a simple counting/qsort path. */
    /* qsort_r-free: build (id, orig_idx) pairs, sort, then split.       */
    {
        typedef struct { int32_t id; int32_t orig; } Pair2;
        Pair2* pp = (Pair2*)malloc((size_t)n_cands * sizeof(Pair2));
        for (int i = 0; i < n_cands; i++) {
            pp[i].id = cand_ids[i]; pp[i].orig = (int32_t)i;
        }
        /* sort ascending by id */
        int cmp_pair(const void* a, const void* b) {
            int32_t x = ((const Pair2*)a)->id, y = ((const Pair2*)b)->id;
            return (x < y) ? -1 : (x > y);
        }
        qsort(pp, (size_t)n_cands, sizeof(Pair2), cmp_pair);
        for (int i = 0; i < n_cands; i++) {
            sorted_ids[i] = pp[i].id;
            perm[i]       = pp[i].orig;   /* perm[sorted_slot] = orig_slot */
        }
        free(pp);
    }

    /* batched reads of the code rows */
    int inflight = 0, submitted_total = 0;
    ScoreId* heap = (ScoreId*)malloc((size_t)k * sizeof(ScoreId));
    int heap_n = 0;

    /* submit in waves to bound queue depth */
    const int WAVE = 1024;
    int next = 0, reaped = 0;
    while (reaped < n_cands) {
        while (next < n_cands && inflight < WAVE) {
            struct io_uring_sqe* sqe = io_uring_get_sqe(ring);
            if (!sqe) break;
            off_t off = (off_t)sizeof(TqHeader) +
                        (off_t)sorted_ids[next] * (off_t)r->code_bytes;
            io_uring_prep_read(sqe, r->fd, raw + (size_t)next * r->code_bytes,
                               (unsigned)r->code_bytes, (uint64_t)off);
            io_uring_sqe_set_data(sqe, (void*)(uintptr_t)next);
            next++; inflight++;
        }
        int s = io_uring_submit(ring);
        if (s < 0) break;
        submitted_total += s;

        struct io_uring_cqe* cqe;
        if (io_uring_wait_cqe(ring, &cqe) < 0) break;
        do {
            int idx = (int)(uintptr_t)io_uring_cqe_get_data(cqe);
            int res = cqe->res;
            io_uring_cqe_seen(ring, cqe);
            inflight--; reaped++;
            if (res == (int)r->code_bytes) {
                const uint8_t* cd = raw + (size_t)idx * r->code_bytes;
                float sc = 0.0f;
                for (int j = 0; j < pad; j += 2) {
                    uint8_t b = cd[j >> 1];
                    sc += lut[j * TQ_LEVELS + (b & 0x0F)];
                    sc += lut[(j + 1) * TQ_LEVELS + (b >> 4)];
                }
                int32_t this_id = sorted_ids[idx];
                if (heap_n < k) {
                    heap[heap_n].s = sc; heap[heap_n].id = this_id;
                    heap_n++;
                    if (heap_n == k)
                        for (int j2 = k/2 - 1; j2 >= 0; j2--)
                            sift_down_min(heap, k, j2);
                } else if (sc > heap[0].s) {
                    heap[0].s = sc; heap[0].id = this_id;
                    sift_down_min(heap, k, 0);
                }
            }
        } while (io_uring_peek_cqe(ring, &cqe) == 0);
    }

    int n_out = heap_n;
    for (int i = 0; i < n_out; i++) out_ids[i] = heap[i].id;
    free(heap); free(raw); free(lut);
    free(sorted_ids); free(perm);
    return n_out;
}
