#define _POSIX_C_SOURCE 200809L
#include "tq1.h"
#include "vec_format.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>

#if defined(__aarch64__) || defined(__ARM_NEON)
#  include <arm_neon.h>
#  define TQ1_HAVE_NEON 1
#else
#  define TQ1_HAVE_NEON 0
#endif

/* Hot scoring kernel : returns Σ qr[j] for indices j where bit j of `code`
 * is set. The full score `Σ qr[j] × (bit_j ? c_pos : c_neg)` is monotone
 * in this masked sum (Δc > 0), so we use it directly for top-k ranking;
 * the constant offset `c_neg × Σqr` cancels in the heap comparisons.
 *
 * Layout : bit j sits in code[j/8] at lane (j & 7). We process 8 coords
 * per loop iteration (one input byte → 8 floats).                       */
static inline float tq1_masked_sum_scalar(const uint8_t* code,
                                          const float* qr, int pad)
{
    float acc = 0.0f;
    for (int j = 0; j < pad; j++) {
        if ((code[j >> 3] >> (j & 7)) & 1) acc += qr[j];
    }
    return acc;
}

#if TQ1_HAVE_NEON
/* NEON kernel : one byte (8 bits → 8 floats) per loop. ~4-8× faster than
 * the scalar path on Cortex-X925 / SVE2.                                */
static inline float tq1_masked_sum_neon(const uint8_t* code,
                                        const float* qr, int pad)
{
    /* Shifts {0, -1, -2, ..., -7} placed in 8 int8 lanes — vshl_u8 with a
     * negative count is a logical right shift, so each lane will hold
     * (byte >> i) for i ∈ 0..7. */
    static const int8_t SHIFTS[8] = { 0, -1, -2, -3, -4, -5, -6, -7 };
    const int8x8_t   sh   = vld1_s8(SHIFTS);
    const uint8x8_t  ONE  = vdup_n_u8(1);
    float32x4_t      acc0 = vdupq_n_f32(0.0f);
    float32x4_t      acc1 = vdupq_n_f32(0.0f);

    int n_bytes = pad >> 3;
    for (int b = 0; b < n_bytes; b++) {
        /* Splat the byte across 8 lanes, shift to bring each bit to the
         * LSB, mask the LSB.                                            */
        uint8x8_t  bits = vshl_u8(vdup_n_u8(code[b]), sh);
        bits = vand_u8(bits, ONE);
        /* Widen 8×u8 → 8×u32 (each lane is 0 or 1), then to float. */
        uint16x8_t b16 = vmovl_u8(bits);
        uint32x4_t lo  = vmovl_u16(vget_low_u16 (b16));
        uint32x4_t hi  = vmovl_u16(vget_high_u16(b16));
        float32x4_t flo = vcvtq_f32_u32(lo);
        float32x4_t fhi = vcvtq_f32_u32(hi);
        float32x4_t qlo = vld1q_f32(qr + (b << 3));
        float32x4_t qhi = vld1q_f32(qr + (b << 3) + 4);
        acc0 = vfmaq_f32(acc0, qlo, flo);
        acc1 = vfmaq_f32(acc1, qhi, fhi);
    }
    /* Tail (pad not a multiple of 8 — should never happen since pad is a
     * power of two ≥ 8, but defensive).                                 */
    float tail = 0.0f;
    for (int j = (n_bytes << 3); j < pad; j++) {
        if ((code[j >> 3] >> (j & 7)) & 1) tail += qr[j];
    }
    return vaddvq_f32(vaddq_f32(acc0, acc1)) + tail;
}
#endif

static inline float tq1_masked_sum(const uint8_t* code,
                                   const float* qr, int pad) {
#if TQ1_HAVE_NEON
    return tq1_masked_sum_neon(code, qr, pad);
#else
    return tq1_masked_sum_scalar(code, qr, pad);
#endif
}

static uint64_t sm64(uint64_t* st) {
    uint64_t z = (*st += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

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

static void gen_signs(uint64_t seed, int pad_dim, float* signs) {
    uint64_t st = seed ? seed : 1;
    for (int r = 0; r < TQ1_ROUNDS; r++)
        for (int i = 0; i < pad_dim; i++)
            signs[r * pad_dim + i] = (sm64(&st) & 1) ? 1.0f : -1.0f;
}

static void rotate(float* x, int pad_dim, const float* signs) {
    for (int r = 0; r < TQ1_ROUNDS; r++) {
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

/* 1-D Lloyd-Max at 2 levels: optimal {c_neg, c_pos} for IP estimation.
   Converges in <10 iters on rotated-Hadamard data (near-Gaussian). */
static void lloyd_2(const float* s, size_t n, float* c_neg, float* c_pos) {
    if (n == 0) { *c_neg = -1.0f; *c_pos = 1.0f; return; }
    double m = 0, m2 = 0;
    for (size_t i = 0; i < n; i++) { m += s[i]; m2 += (double)s[i] * s[i]; }
    m /= (double)n; m2 /= (double)n;
    double sd = sqrt(m2 - m * m);
    if (sd < 1e-12) sd = 1.0;
    float cn = (float)(-sd), cp = (float)sd;
    for (int it = 0; it < 40; it++) {
        float bound = 0.5f * (cn + cp);
        double sn = 0, sp = 0; size_t nn = 0, np = 0;
        for (size_t i = 0; i < n; i++) {
            if (s[i] <= bound) { sn += s[i]; nn++; }
            else                { sp += s[i]; np++; }
        }
        if (nn) cn = (float)(sn / (double)nn);
        if (np) cp = (float)(sp / (double)np);
    }
    *c_neg = cn; *c_pos = cp;
}

int tq1_build(const char* fvecs_path, const char* out_path,
              int dim, uint64_t seed, int calib_rows,
              uint64_t n_docs_cap) {
    VecFmt fmt = vec_fmt_from_path(fvecs_path);
    if (fmt != VECFMT_FVECS && fmt != VECFMT_FBIN
            && fmt != VECFMT_BVECS && fmt != VECFMT_U8BIN) {
        fprintf(stderr, "tq1_build: input must be fvecs/fbin/bvecs/u8bin\n");
        return -1;
    }
    int fd = open(fvecs_path, O_RDONLY);
    if (fd < 0) { perror("tq1_build open"); return -1; }
    struct stat st; fstat(fd, &st);
    int has_row_hdr = (fmt == VECFMT_FVECS || fmt == VECFMT_BVECS);
    int is_uint8    = (fmt == VECFMT_BVECS || fmt == VECFMT_U8BIN);
    off_t file_hdr  = (fmt == VECFMT_FBIN || fmt == VECFMT_U8BIN) ? 8 : 0;
    size_t elem     = is_uint8 ? 1u : 4u;
    size_t row_bytes = (has_row_hdr ? 4 : 0) + (size_t)dim * elem;
    uint64_t n_docs = (uint64_t)((st.st_size - file_hdr) / (off_t)row_bytes);
    if (n_docs_cap > 0 && n_docs_cap < n_docs) n_docs = n_docs_cap;

    int pad = next_pow2(dim);
    float* signs = (float*)malloc((size_t)TQ1_ROUNDS * pad * sizeof(float));
    gen_signs(seed, pad, signs);

    if (calib_rows <= 0) calib_rows = 20000;
    if ((uint64_t)calib_rows > n_docs) calib_rows = (int)n_docs;
    uint64_t stride = n_docs / (uint64_t)calib_rows;
    if (stride == 0) stride = 1;
    size_t want = 2 * 1000 * 1000;
    size_t per_row = want / (size_t)calib_rows + 1;
    if (per_row > (size_t)pad) per_row = (size_t)pad;
    float* sample = (float*)malloc((want + (size_t)pad) * sizeof(float));
    float* rowbuf = (float*)calloc((size_t)pad, sizeof(float));
    uint8_t* rawrow = (uint8_t*)malloc(row_bytes);
    size_t n_s = 0;
    size_t hdr_skip = has_row_hdr ? 4 : 0;
    fprintf(stderr, "  tq1 calibrating: %d rows (stride %llu) ...\n",
            calib_rows, (unsigned long long)stride);
    for (int i = 0; i < calib_rows; i++) {
        off_t off = file_hdr + (off_t)((uint64_t)i * stride) * (off_t)row_bytes;
        if (pread(fd, rawrow, row_bytes, off) != (ssize_t)row_bytes) break;
        memset(rowbuf, 0, (size_t)pad * sizeof(float));
        memcpy(rowbuf, rawrow + hdr_skip, (size_t)dim * sizeof(float));
        rotate(rowbuf, pad, signs);
        for (size_t j = 0; j < per_row && n_s < want; j++)
            sample[n_s++] = rowbuf[(j * 911) % (size_t)pad];
    }

    Tq1Header h; memset(&h, 0, sizeof(h));
    h.magic = TQ1_MAGIC; h.version = 1;
    h.dim = (uint32_t)dim; h.pad_dim = (uint32_t)pad;
    h.seed = seed; h.n_docs = n_docs;
    lloyd_2(sample, n_s, &h.c_neg, &h.c_pos);
    free(sample);
    fprintf(stderr, "  tq1 codebook: c_neg=%.4f c_pos=%.4f\n",
            h.c_neg, h.c_pos);

    FILE* out = fopen(out_path, "wb");
    if (!out) { perror("tq1_build fopen out"); free(signs); close(fd); return -1; }
    fwrite(&h, sizeof(h), 1, out);

    size_t code_bytes = (size_t)pad / 8;
    uint8_t* codes = (uint8_t*)malloc(code_bytes);

    const int CHUNK = 8192;
    uint8_t* big = (uint8_t*)malloc((size_t)CHUNK * row_bytes);
    uint64_t done = 0;
    lseek(fd, file_hdr, SEEK_SET);
    while (done < n_docs) {
        int n = (int)((n_docs - done) < CHUNK ? (n_docs - done) : CHUNK);
        ssize_t got = read(fd, big, (size_t)n * row_bytes);
        if (got != (ssize_t)((size_t)n * row_bytes)) {
            fprintf(stderr, "tq1_build short read at doc %llu\n",
                    (unsigned long long)done);
            break;
        }
        for (int i = 0; i < n; i++) {
            memset(rowbuf, 0, (size_t)pad * sizeof(float));
            const uint8_t* src = big + (size_t)i * row_bytes + hdr_skip;
            if (is_uint8) {
                for (int j = 0; j < dim; j++) rowbuf[j] = (float)src[j];
            } else {
                memcpy(rowbuf, src, (size_t)dim * sizeof(float));
            }
            rotate(rowbuf, pad, signs);
            memset(codes, 0, code_bytes);
            for (int j = 0; j < pad; j++)
                if (rowbuf[j] > 0.0f) codes[j >> 3] |= (uint8_t)(1u << (j & 7));
            fwrite(codes, 1, code_bytes, out);
        }
        done += (uint64_t)n;
        if (done % 1000000 < (uint64_t)CHUNK)
            fprintf(stderr, "  tq1 quantized %llu / %llu\n",
                    (unsigned long long)done, (unsigned long long)n_docs);
    }
    free(big); free(codes); free(rowbuf); free(rawrow); free(signs);
    fclose(out); close(fd);
    fprintf(stderr, "  tq1 done: %llu docs -> %s\n",
            (unsigned long long)done, out_path);
    return (done == n_docs) ? 0 : -1;
}

struct Tq1Reader {
    int fd;
    Tq1Header h;
    float* signs;
    size_t code_bytes;
};

Tq1Reader* tq1_open(const char* path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;
    Tq1Reader* r = (Tq1Reader*)calloc(1, sizeof(Tq1Reader));
    r->fd = fd;
    if (pread(fd, &r->h, sizeof(Tq1Header), 0) != sizeof(Tq1Header) ||
        r->h.magic != TQ1_MAGIC) {
        close(fd); free(r); return NULL;
    }
    r->code_bytes = (size_t)r->h.pad_dim / 8;
    r->signs = (float*)malloc((size_t)TQ1_ROUNDS * r->h.pad_dim * sizeof(float));
    gen_signs(r->h.seed, (int)r->h.pad_dim, r->signs);
    return r;
}

void tq1_close(Tq1Reader* r) {
    if (!r) return;
    close(r->fd); free(r->signs); free(r);
}

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

int tq1_select_fixed(Tq1Reader* r, struct io_uring* ring,
                     void* fixed_buf, size_t fixed_size, int fixed_buf_idx,
                     const float* qvec,
                     const int32_t* cand_ids, int n_cands,
                     int kprime, int32_t* out_ids) {
    if (!r || n_cands <= 0) return 0;
    int pad = (int)r->h.pad_dim, dim = (int)r->h.dim;
    int k = (kprime < n_cands) ? kprime : n_cands;

    float* qr = (float*)calloc((size_t)pad, sizeof(float));
    memcpy(qr, qvec, (size_t)dim * sizeof(float));
    rotate(qr, pad, r->signs);
    /* No per-coord LUT needed any more — the SIMD kernel scores directly
     * from `qr` and the packed code bits (see tq1_masked_sum). */

    /* If the caller's registered (fixed) buffer is large enough to hold all
     * codes for this query, write into it and use io_uring_prep_read_fixed
     * — skips the per-completion copy_to_user. Else fall back to malloc +
     * prep_read.                                                          */
    const size_t need = (size_t)n_cands * r->code_bytes;
    int use_fixed = (fixed_buf != NULL && fixed_size >= need);
    uint8_t* raw = use_fixed ? (uint8_t*)fixed_buf
                             : (uint8_t*)malloc(need);
    if (!raw) { free(qr); return -1; }

    int32_t* sorted_ids = (int32_t*)malloc((size_t)n_cands * sizeof(int32_t));
    if (!sorted_ids) { free(raw); free(qr); return -1; }
    {
        typedef struct { int32_t id; int32_t orig; } Pair2;
        Pair2* pp = (Pair2*)malloc((size_t)n_cands * sizeof(Pair2));
        for (int i = 0; i < n_cands; i++) {
            pp[i].id = cand_ids[i]; pp[i].orig = (int32_t)i;
        }
        int cmp_pair(const void* a, const void* b) {
            int32_t x = ((const Pair2*)a)->id, y = ((const Pair2*)b)->id;
            return (x < y) ? -1 : (x > y);
        }
        qsort(pp, (size_t)n_cands, sizeof(Pair2), cmp_pair);
        for (int i = 0; i < n_cands; i++) sorted_ids[i] = pp[i].id;
        free(pp);
    }

    int inflight = 0;
    ScoreId* heap = (ScoreId*)malloc((size_t)k * sizeof(ScoreId));
    int heap_n = 0;
    const int WAVE = 1024;
    int next = 0, reaped = 0;
    while (reaped < n_cands) {
        while (next < n_cands && inflight < WAVE) {
            struct io_uring_sqe* sqe = io_uring_get_sqe(ring);
            if (!sqe) break;
            off_t off = (off_t)sizeof(Tq1Header) +
                        (off_t)sorted_ids[next] * (off_t)r->code_bytes;
            uint8_t* dst = raw + (size_t)next * r->code_bytes;
            if (use_fixed) {
                io_uring_prep_read_fixed(sqe, r->fd, dst,
                                         (unsigned)r->code_bytes,
                                         (uint64_t)off, fixed_buf_idx);
            } else {
                io_uring_prep_read(sqe, r->fd, dst,
                                   (unsigned)r->code_bytes,
                                   (uint64_t)off);
            }
            io_uring_sqe_set_data(sqe, (void*)(uintptr_t)next);
            next++; inflight++;
        }
        int s = io_uring_submit(ring);
        if (s < 0) break;
        struct io_uring_cqe* cqe;
        if (io_uring_wait_cqe(ring, &cqe) < 0) break;
        do {
            int idx = (int)(uintptr_t)io_uring_cqe_get_data(cqe);
            int res = cqe->res;
            io_uring_cqe_seen(ring, cqe);
            inflight--; reaped++;
            if (res == (int)r->code_bytes) {
                const uint8_t* cd = raw + (size_t)idx * r->code_bytes;
                /* Score = Σ qr[j] × (bit_j ? c_pos : c_neg).
                 *       = c_neg × Σqr + Δc × masked_sum(qr, bits)
                 * For top-k ranking only the monotone part `masked_sum`
                 * matters (Δc > 0), so we score with that directly and
                 * skip the constant offset to halve the per-cand work. */
                float sc = tq1_masked_sum(cd, qr, pad);
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
    free(heap);
    if (!use_fixed) free(raw);   /* fixed buffer is owned by the caller */
    free(qr); free(sorted_ids);
    return n_out;
}

/* Backward-compat wrapper : no fixed buffer → always uses prep_read. */
int tq1_select(Tq1Reader* r, struct io_uring* ring,
               const float* qvec,
               const int32_t* cand_ids, int n_cands,
               int kprime, int32_t* out_ids) {
    return tq1_select_fixed(r, ring, NULL, 0, 0,
                             qvec, cand_ids, n_cands, kprime, out_ids);
}
