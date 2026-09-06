/* Index ANCRES : cellules de Voronoi sur K ancres echantillonnees dans la
   base (equilibre statistique gratuit), spill frontiere (un doc rejoint
   les ancres a distance <= (1+eps) de la plus proche, max M), blocs par
   cellule [u32 id + code TQ] colocalises — l'unite de lecture requete.
   Requete : descente ancres (RAM) -> nprobe blocs (io_uring) -> scoring
   TQ asymetrique (SDOT int8) -> top-R -> rerank exact (pread base f16).
   Mode cosinus : docs, ancres et requetes normalises ; pas de normes.
   Rotation : FWHT + signes seedes (recomputable), scale int par dim.

   Fichiers (out_dir) : meta.txt, anchors.bin (K x dim f32),
   offs.bin ((K+1) x u64), blocks.bin, scale.bin (dim f32).            */
#define _POSIX_C_SOURCE 200809L
#include <fcntl.h>
#include <liburing.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#define ANC_NEON 1
#else
#define ANC_NEON 0
#endif

/* ---------- PRNG splitmix64 ---------- */
static uint64_t asm64(uint64_t* st) {
    uint64_t z = (*st += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* ---------- FWHT (dim = puissance de 2) + signes seedes ---------- */
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

static void rot_seeded(const float* in, float* out, const int8_t* sgn,
                       int d) {
    for (int i = 0; i < d; i++) out[i] = in[i] * (float)sgn[i];
    fwht(out, d);
}

static void make_signs(int8_t* sgn, int d, uint64_t seed) {
    uint64_t st = seed;
    for (int i = 0; i < d; i++)
        sgn[i] = (asm64(&st) & 1) ? 1 : -1;
}

/* ---------- f16 -> f32 ---------- */
static inline float h2f(uint16_t h) {
#if ANC_NEON
    __fp16 x;
    memcpy(&x, &h, 2);
    return (float)x;
#else
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    if (e == 0) return (s ? -1.f : 1.f) * (float)m * 5.9604645e-8f;
    if (e == 31) return s ? -65504.f : 65504.f;
    union { uint32_t u; float f; } v;
    v.u = (s << 31) | ((e + 112) << 23) | (m << 13);
    return v.f;
#endif
}

static void row_f16_to_unit(const uint16_t* src, float* dst, int d) {
    float n2 = 0.0f;
    for (int i = 0; i < d; i++) { dst[i] = h2f(src[i]); n2 += dst[i] * dst[i]; }
    float inv = 1.0f / (sqrtf(n2) + 1e-9f);
    for (int i = 0; i < d; i++) dst[i] *= inv;
}

/* ---------- dot f32 NEON ---------- */
static inline float dotf(const float* a, const float* b, int d) {
#if ANC_NEON
    float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
    int i = 0;
    for (; i + 8 <= d; i += 8) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
        acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    }
    float s = vaddvq_f32(vaddq_f32(acc0, acc1));
    for (; i < d; i++) s += a[i] * b[i];
    return s;
#else
    float s = 0;
    for (int i = 0; i < d; i++) s += a[i] * b[i];
    return s;
#endif
}

/* ---------- dot int8 (SDOT si dispo, sinon scalaire) ---------- */
static inline int32_t doti8(const int8_t* a, const int8_t* b, int d) {
#if ANC_NEON && defined(__ARM_FEATURE_DOTPROD)
    int32x4_t acc = vdupq_n_s32(0);
    int i = 0;
    for (; i + 16 <= d; i += 16)
        acc = vdotq_s32(acc, vld1q_s8(a + i), vld1q_s8(b + i));
    int32_t s = vaddvq_s32(acc);
    for (; i < d; i++) s += (int32_t)a[i] * b[i];
    return s;
#else
    int32_t s = 0;
    for (int i = 0; i < d; i++) s += (int32_t)a[i] * b[i];
    return s;
#endif
}

/* Scoring TQ4 sans depaquetage : nibbles bas/hauts sign-etendus en NEON
   (shl4+shr4 / shr4 arithmetique), SDOT contre la requete REORDONNEE
   (qlo = dims paires, qhi = dims impaires). ~2 instr / 32 dims.        */
static inline int32_t score_tq4(const uint8_t* code, const int8_t* qlo,
                                const int8_t* qhi, int dim) {
#if ANC_NEON && defined(__ARM_FEATURE_DOTPROD)
    int32x4_t acc = vdupq_n_s32(0);
    int nb = dim / 2;
    for (int i = 0; i + 16 <= nb; i += 16) {
        int8x16_t b = vld1q_s8((const int8_t*)code + i);
        int8x16_t lo = vshrq_n_s8(vshlq_n_s8(b, 4), 4);
        int8x16_t hi = vshrq_n_s8(b, 4);
        acc = vdotq_s32(acc, lo, vld1q_s8(qlo + i));
        acc = vdotq_s32(acc, hi, vld1q_s8(qhi + i));
    }
    return vaddvq_s32(acc);
#else
    int32_t s = 0;
    for (int d = 0; d < dim; d += 2) {
        uint8_t b = code[d >> 1];
        int q0 = (int)(b & 15); if (q0 > 7) q0 -= 16;
        int q1 = (int)(b >> 4); if (q1 > 7) q1 -= 16;
        s += q0 * qlo[d >> 1] + q1 * qhi[d >> 1];
    }
    return s;
#endif
}

/* TQ1 : somme masquee des q8 aux bits leves (offset constant par requete,
   sans effet sur l ordre). 8 dims par octet via vtst.                   */
static inline int32_t score_tq1(const uint8_t* code, const int8_t* q8,
                                int dim) {
#if ANC_NEON && defined(__ARM_FEATURE_DOTPROD)
    static const uint8_t bitsel[16] = {1, 2, 4, 8, 16, 32, 64, 128,
                                       1, 2, 4, 8, 16, 32, 64, 128};
    uint8x16_t sel = vld1q_u8(bitsel);
    int8x16_t one = vdupq_n_s8(1);
    int32x4_t acc = vdupq_n_s32(0);
    for (int d = 0; d + 16 <= dim; d += 16) {
        uint8_t b0 = code[d >> 3], b1 = code[(d >> 3) + 1];
        uint8x16_t bb = vcombine_u8(vdup_n_u8(b0), vdup_n_u8(b1));
        uint8x16_t msk = vtstq_u8(bb, sel);
        int8x16_t v = vandq_s8(vld1q_s8(q8 + d),
                               vreinterpretq_s8_u8(msk));
        acc = vdotq_s32(acc, v, one);
    }
    return vaddvq_s32(acc);
#else
    int32_t s = 0;
    for (int d = 0; d < dim; d++)
        if ((code[d >> 3] >> (d & 7)) & 1) s += q8[d];
    return s;
#endif
}

typedef struct {
    int K, dim, M, tqbits;
    float eps;
    int64_t n;
    uint64_t seed;
} AMeta;

static int meta_load(const char* dir, AMeta* m) {
    char p[1024];
    snprintf(p, sizeof(p), "%s/meta.txt", dir);
    FILE* f = fopen(p, "r");
    if (!f) return -1;
    long long n = 0; unsigned long long sd = 0;
    int r = fscanf(f, "%d %d %d %d %f %lld %llu", &m->K, &m->dim, &m->M,
                   &m->tqbits, &m->eps, &n, &sd);
    fclose(f);
    m->n = n; m->seed = sd;
    return r == 7 ? 0 : -1;
}

/* ================= BUILD ================= */
int cmd_anchor_build(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: rpforest abuild <base.f16bin> <out_dir> <K> "
                        "[--eps 0.20] [--m 3] [--tqbits 4] [--seed 42] "
                        "[--nmax 0]\n");
        return 1;
    }
    const char* base_path = argv[2];
    const char* out = argv[3];
    int K = atoi(argv[4]);
    float eps = 0.20f;
    int M = 3, tqbits = 4;
    uint64_t seed = 42;
    int64_t nmax = 0;
    for (int i = 5; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--eps")) eps = atof(argv[i + 1]);
        else if (!strcmp(argv[i], "--m")) M = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--tqbits")) tqbits = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--seed")) seed = strtoull(argv[i + 1], 0, 10);
        else if (!strcmp(argv[i], "--nmax")) nmax = atoll(argv[i + 1]);
    }
    if (M > 4) M = 4;
    FILE* bf = fopen(base_path, "rb");
    if (!bf) { perror("base"); return 1; }
    uint32_t hdr[2];
    if (fread(hdr, 4, 2, bf) != 2) { fclose(bf); return 1; }
    int64_t n = hdr[0];
    int dim = (int)hdr[1];
    if (nmax > 0 && nmax < n) n = nmax;
    fprintf(stderr, "abuild: n=%lld dim=%d K=%d eps=%.2f M=%d tq%d\n",
            (long long)n, dim, K, eps, M, tqbits);

    /* --- ancres : K ids sans remise (seed) --- */
    uint8_t* taken = (uint8_t*)calloc((size_t)n, 1);
    int64_t* aids = (int64_t*)malloc((size_t)K * 8);
    uint64_t st = seed;
    for (int k = 0; k < K; k++) {
        int64_t id;
        do { id = (int64_t)(asm64(&st) % (uint64_t)n); } while (taken[id]);
        taken[id] = 1;
        aids[k] = id;
    }
    free(taken);
    float* A = (float*)malloc((size_t)K * dim * 4);
    uint16_t* rowbuf = (uint16_t*)malloc((size_t)dim * 2);
    for (int k = 0; k < K; k++) {
        fseeko(bf, 8 + aids[k] * (int64_t)dim * 2, SEEK_SET);
        if (fread(rowbuf, 2, dim, bf) != (size_t)dim) return 1;
        row_f16_to_unit(rowbuf, A + (size_t)k * dim, dim);
    }
    fprintf(stderr, "abuild: ancres chargees\n");

    int8_t* sgn = (int8_t*)malloc(dim);
    make_signs(sgn, dim, seed ^ 0x51CA);

    /* Ancres en int8 pour l assignation : SDOT = 4x les ops/cycle du FMA
       f32, et 41 Mo au lieu de 164 — tuilables en L2. La quantization ne
       perturbe le choix des 2 plus proches que sur des ex aequo (spill
       rang-2 : sans consequence). argmax(dot) invariant par echelle.    */
    float amax = 0.0f;
    for (size_t i = 0; i < (size_t)K * dim; i++) {
        float v = fabsf(A[i]);
        if (v > amax) amax = v;
    }
    float aq = 127.0f / (amax + 1e-9f);
    int8_t* A8 = (int8_t*)malloc((size_t)K * dim);
    for (size_t i = 0; i < (size_t)K * dim; i++)
        A8[i] = (int8_t)lrintf(A[i] * aq);

    /* --- passe 1 : assignation top-M + scale (chunks streames) --- */
    const int64_t CHUNK = 200000;
    uint16_t* raw = (uint16_t*)malloc((size_t)CHUNK * dim * 2);
    int32_t* topm = (int32_t*)malloc((size_t)n * M * 4);
    float* topd = (float*)malloc((size_t)n * M * 4);
    double* scale_acc = (double*)calloc(dim, 8);
    int64_t scale_n = 0;
    double t0 = omp_get_wtime();
    /* cache d assignation : la passe 1 (3h a 40M) est independante de
       tqbits/eps — reutilisable pour rebuilder avec d autres codes.
       Layout : [i64 n][i64 M][topm nxM i32][topd nxM f32][sigmean dim f32] */
    char apath[1024];
    snprintf(apath, sizeof(apath), "%s/assign.bin", out);
    float* sigmean = (float*)malloc((size_t)dim * 4);
    int skip_pass1 = 0;
    {
        FILE* af = fopen(apath, "rb");
        if (af) {
            int64_t ah[2];
            if (fread(ah, 8, 2, af) == 2 && ah[0] == n && ah[1] == M
                && fread(topm, 4, (size_t)n * M, af) == (size_t)n * M
                && fread(topd, 4, (size_t)n * M, af) == (size_t)n * M
                && fread(sigmean, 4, dim, af) == (size_t)dim) {
                skip_pass1 = 1;
                fprintf(stderr, "abuild: assignation reprise du cache\n");
            }
            fclose(af);
        }
    }
    fseeko(bf, 8, SEEK_SET);
    for (int64_t off = 0; skip_pass1 == 0 && off < n; off += CHUNK) {
        int64_t c = n - off < CHUNK ? n - off : CHUNK;
        if (fread(raw, (size_t)dim * 2, c, bf) != (size_t)c) return 1;
        #pragma omp parallel
        {
            float* v = (float*)malloc((size_t)dim * 4);
            float* r = (float*)malloc((size_t)dim * 4);
            int8_t* v8 = (int8_t*)malloc(dim);
            #pragma omp for schedule(dynamic, 64)
            for (int64_t i = 0; i < c; i++) {
                row_f16_to_unit(raw + (size_t)i * dim, v, dim);
                float vmax = 0.0f;
                for (int d = 0; d < dim; d++) {
                    float x = fabsf(v[d]);
                    if (x > vmax) vmax = x;
                }
                float vq = 127.0f / (vmax + 1e-9f);
                for (int d = 0; d < dim; d++)
                    v8[d] = (int8_t)lrintf(v[d] * vq);
                float bd[4] = {2e9f, 2e9f, 2e9f, 2e9f};
                int32_t bi[4] = {-1, -1, -1, -1};
                for (int k = 0; k < K; k++) {
                    float d2 = -(float)doti8(v8, A8 + (size_t)k * dim, dim);
                    if (d2 < bd[M - 1]) {
                        int j = M - 1;
                        while (j > 0 && d2 < bd[j - 1]) {
                            bd[j] = bd[j - 1]; bi[j] = bi[j - 1]; j--;
                        }
                        bd[j] = d2; bi[j] = k;
                    }
                }
                /* topd en ||q-a||^2 approx via cos int8 renormalise */
                float inv = 1.0f / (vq * aq);
                for (int j = 0; j < M; j++) {
                    topm[(off + i) * M + j] = bi[j];
                    topd[(off + i) * M + j] = 2.0f + 2.0f * bd[j] * inv;
                }
                if (((off + i) & 63) == 0) {
                    rot_seeded(v, r, sgn, dim);
                    #pragma omp critical
                    {
                        for (int d = 0; d < dim; d++)
                            scale_acc[d] += fabsf(r[d]);
                        scale_n++;
                    }
                }
            }
            free(v); free(r); free(v8);
        }
        fprintf(stderr, "abuild: assign %lld/%lld (%.0fs)\n",
                (long long)(off + c), (long long)n, omp_get_wtime() - t0);
    }
    if (!skip_pass1) {
        for (int d = 0; d < dim; d++)
            sigmean[d] = (float)(scale_acc[d] / (double)(scale_n ? scale_n : 1));
        FILE* af = fopen(apath, "wb");
        if (af) {
            int64_t ah[2] = {n, M};
            fwrite(ah, 8, 2, af);
            fwrite(topm, 4, (size_t)n * M, af);
            fwrite(topd, 4, (size_t)n * M, af);
            fwrite(sigmean, 4, dim, af);
            fclose(af);
        }
    }
    /* scale : ~3 sigma de |x| moyen (demi-normale) par dim */
    float* scale = (float*)malloc((size_t)dim * 4);
    float qlevels = (float)((1 << (tqbits - 1)) - 1) + 0.5f;
    for (int d = 0; d < dim; d++) {
        float sig = sigmean[d] * 1.2533f;
        scale[d] = qlevels / (3.0f * sig + 1e-9f);
    }
    free(scale_acc); free(sigmean);

    /* --- comptage cellules avec spill --- */
    int64_t* cnt = (int64_t*)calloc((size_t)K + 1, 8);
    int64_t entries = 0;
    for (int64_t i = 0; i < n; i++) {
        float lim = (1.0f + eps) * (1.0f + eps) * topd[i * M];
        for (int j = 0; j < M; j++) {
            if (j > 0 && topd[i * M + j] > lim) break;
            cnt[topm[i * M + j]]++;
            entries++;
        }
    }
    int code_b = dim * tqbits / 8;
    int ent_b = 4 + code_b;
    uint64_t* offs = (uint64_t*)malloc(((size_t)K + 1) * 8);
    offs[0] = 0;
    for (int k = 0; k < K; k++)
        offs[k + 1] = offs[k] + (uint64_t)cnt[k] * ent_b;
    fprintf(stderr, "abuild: %lld entrees (x%.2f), blocks %.1f Go\n",
            (long long)entries, (double)entries / n,
            (double)offs[K] / 1e9);

    /* --- passe 2 : rotation + quantization + ecriture blocs --- */
    char p[1024];
    snprintf(p, sizeof(p), "%s/blocks.bin", out);
    int bfd = open(p, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (bfd < 0 || ftruncate(bfd, (off_t)offs[K]) != 0) {
        perror("blocks"); return 1;
    }
    uint64_t* cur = (uint64_t*)malloc((size_t)K * 8);
    memcpy(cur, offs, (size_t)K * 8);
    fseeko(bf, 8, SEEK_SET);
    t0 = omp_get_wtime();
    uint8_t* entbuf = (uint8_t*)malloc((size_t)CHUNK * M * ent_b);
    int64_t* entoff = (int64_t*)malloc((size_t)CHUNK * M * 8);
    for (int64_t off = 0; off < n; off += CHUNK) {
        int64_t c = n - off < CHUNK ? n - off : CHUNK;
        if (fread(raw, (size_t)dim * 2, c, bf) != (size_t)c) return 1;
        int64_t ne = 0;
        /* offsets sequentiels (ordre doc) + index premiere entree/doc */
        int64_t* first_ent = (int64_t*)malloc((size_t)c * 8);
        for (int64_t i = 0; i < c; i++) {
            int64_t g = off + i;
            first_ent[i] = ne;
            float lim = (1.0f + eps) * (1.0f + eps) * topd[g * M];
            for (int j = 0; j < M; j++) {
                if (j > 0 && topd[g * M + j] > lim) break;
                int32_t cell = topm[g * M + j];
                entoff[ne] = (int64_t)cur[cell];
                cur[cell] += ent_b;
                ne++;
            }
        }
        #pragma omp parallel
        {
            float* v = (float*)malloc((size_t)dim * 4);
            float* r = (float*)malloc((size_t)dim * 4);
            uint8_t* code = (uint8_t*)malloc(code_b);
            #pragma omp for schedule(dynamic, 64)
            for (int64_t i = 0; i < c; i++) {
                int64_t g = off + i;
                row_f16_to_unit(raw + (size_t)i * dim, v, dim);
                rot_seeded(v, r, sgn, dim);
                if (tqbits == 4) {
                    for (int d = 0; d < dim; d += 2) {
                        int q0 = (int)lrintf(r[d] * scale[d]);
                        int q1 = (int)lrintf(r[d + 1] * scale[d + 1]);
                        if (q0 < -8) q0 = -8; if (q0 > 7) q0 = 7;
                        if (q1 < -8) q1 = -8; if (q1 > 7) q1 = 7;
                        code[d >> 1] = (uint8_t)((q0 & 15) | ((q1 & 15) << 4));
                    }
                } else { /* tq1 : bit de signe */
                    memset(code, 0, code_b);
                    for (int d = 0; d < dim; d++)
                        if (r[d] >= 0) code[d >> 3] |= (uint8_t)(1 << (d & 7));
                }
                /* copie vers chaque entree du doc */
                float lim = (1.0f + eps) * (1.0f + eps) * topd[g * M];
                int64_t idx = first_ent[i];
                for (int j = 0; j < M; j++) {
                    if (j > 0 && topd[g * M + j] > lim) break;
                    uint8_t* e = entbuf + (idx + j) * ent_b;
                    uint32_t id32 = (uint32_t)g;
                    memcpy(e, &id32, 4);
                    memcpy(e + 4, code, (size_t)code_b);
                }
            }
            free(v); free(r); free(code);
        }
        for (int64_t e = 0; e < ne; e++) {
            if (pwrite(bfd, entbuf + e * ent_b, (size_t)ent_b,
                       (off_t)entoff[e]) != ent_b) {
                perror("pwrite"); return 1;
            }
        }
        free(first_ent);
        fprintf(stderr, "abuild: blocs %lld/%lld (%.0fs)\n",
                (long long)(off + c), (long long)n, omp_get_wtime() - t0);
    }
    close(bfd);
    free(entbuf); free(entoff);

    snprintf(p, sizeof(p), "%s/anchors.bin", out);
    FILE* fo = fopen(p, "wb");
    fwrite(A, 4, (size_t)K * dim, fo); fclose(fo);
    snprintf(p, sizeof(p), "%s/offs.bin", out);
    fo = fopen(p, "wb");
    fwrite(offs, 8, (size_t)K + 1, fo); fclose(fo);
    snprintf(p, sizeof(p), "%s/scale.bin", out);
    fo = fopen(p, "wb");
    fwrite(scale, 4, dim, fo); fclose(fo);
    snprintf(p, sizeof(p), "%s/meta.txt", out);
    fo = fopen(p, "w");
    fprintf(fo, "%d %d %d %d %.4f %lld %llu\n", K, dim, M, tqbits, eps,
            (long long)n, (unsigned long long)seed);
    fclose(fo);
    fprintf(stderr, "abuild: DONE\n");
    fclose(bf);
    free(A); free(A8); free(aids); free(raw); free(topm); free(topd);
    free(cnt); free(offs); free(cur); free(scale); free(sgn); free(rowbuf);
    return 0;
}

/* ================= BENCH (query + latences) ================= */
typedef struct { float s; uint32_t id; } ScId;

static int scid_cmp(const void* a, const void* b) {
    float d = ((const ScId*)b)->s - ((const ScId*)a)->s;
    return d > 0 ? 1 : (d < 0 ? -1 : 0);
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

int cmd_anchor_bench(int argc, char** argv) {
    if (argc < 8) {
        fprintf(stderr, "usage: rpforest abench <dir> <base.f16bin> "
                        "<queries.fbin> <nq> <nprobe> <rerank> "
                        "[--out res.bin] [--drop <script>]\n");
        return 1;
    }
    const char* dir = argv[2];
    const char* base_path = argv[3];
    const char* qpath = argv[4];
    int nq = atoi(argv[5]);
    int nprobe = atoi(argv[6]);
    int rerank = atoi(argv[7]);
    const char* outp = NULL;
    const char* drop = NULL;
    for (int i = 8; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--out")) outp = argv[i + 1];
        else if (!strcmp(argv[i], "--drop")) drop = argv[i + 1];
    }
    AMeta m;
    if (meta_load(dir, &m) != 0) { fprintf(stderr, "meta?\n"); return 1; }
    int dim = m.dim, K = m.K;
    int code_b = dim * m.tqbits / 8;
    int ent_b = 4 + code_b;
    char p[1024];
    snprintf(p, sizeof(p), "%s/anchors.bin", dir);
    FILE* f = fopen(p, "rb");
    float* A = (float*)malloc((size_t)K * dim * 4);
    if (fread(A, 4, (size_t)K * dim, f) != (size_t)K * dim) return 1;
    fclose(f);
    snprintf(p, sizeof(p), "%s/offs.bin", dir);
    f = fopen(p, "rb");
    uint64_t* offs = (uint64_t*)malloc(((size_t)K + 1) * 8);
    if (fread(offs, 8, (size_t)K + 1, f) != (size_t)K + 1) return 1;
    fclose(f);
    snprintf(p, sizeof(p), "%s/scale.bin", dir);
    f = fopen(p, "rb");
    float* scale = (float*)malloc((size_t)dim * 4);
    if (fread(scale, 4, dim, f) != (size_t)dim) return 1;
    fclose(f);
    int8_t* sgn = (int8_t*)malloc(dim);
    make_signs(sgn, dim, m.seed ^ 0x51CA);

    snprintf(p, sizeof(p), "%s/blocks.bin", dir);
    int bfd = open(p, O_RDONLY);
    int basefd = open(base_path, O_RDONLY);
    if (bfd < 0 || basefd < 0) { perror("open"); return 1; }

    FILE* qf = fopen(qpath, "rb");
    uint32_t qh[2];
    if (fread(qh, 4, 2, qf) != 2 || (int)qh[1] != dim) return 1;
    if ((int)qh[0] < nq) nq = (int)qh[0];
    float* Q = (float*)malloc((size_t)nq * dim * 4);
    if (fread(Q, 4, (size_t)nq * dim, qf) != (size_t)nq * dim) return 1;
    fclose(qf);

    struct io_uring ring;
    io_uring_queue_init(256, &ring, 0);
    uint64_t max_cell = 0;
    for (int k = 0; k < K; k++)
        if (offs[k + 1] - offs[k] > max_cell) max_cell = offs[k + 1] - offs[k];
    uint8_t* blk = (uint8_t*)malloc((size_t)nprobe * max_cell);
    uint64_t* boff = (uint64_t*)malloc((size_t)nprobe * 8);
    uint64_t* blen = (uint64_t*)malloc((size_t)nprobe * 8);
    ScId* heap = (ScId*)malloc(((size_t)rerank + 1) * sizeof(ScId));
    uint32_t* out_ids = (uint32_t*)malloc((size_t)nq * 11 * 4);
    double *t_anc = malloc(nq * 8), *t_io = malloc(nq * 8),
           *t_sc = malloc(nq * 8), *t_rr = malloc(nq * 8),
           *t_tot = malloc(nq * 8);
    int64_t docs_seen_tot = 0;

    float* qn = (float*)malloc((size_t)dim * 4);
    float* qr = (float*)malloc((size_t)dim * 4);
    int8_t* q8 = (int8_t*)malloc(dim);
    int8_t* c8 = (int8_t*)malloc(dim);
    ScId* cand = (ScId*)malloc(sizeof(ScId) * (size_t)nprobe);
    uint16_t* vrow = (uint16_t*)malloc((size_t)dim * 2);
    float* vf = (float*)malloc((size_t)dim * 4);

    for (int qi = 0; qi < nq; qi++) {
        if (drop) { int rc = system(drop); (void)rc; }
        double T0 = now_ms();
        /* normalise + descente ancres */
        memcpy(qn, Q + (size_t)qi * dim, (size_t)dim * 4);
        float n2 = 0;
        for (int d = 0; d < dim; d++) n2 += qn[d] * qn[d];
        float inv = 1.0f / (sqrtf(n2) + 1e-9f);
        for (int d = 0; d < dim; d++) qn[d] *= inv;
        for (int c = 0; c < nprobe; c++) cand[c].s = -2e9f;
        #pragma omp parallel
        {
            ScId* loc = (ScId*)malloc(sizeof(ScId) * (size_t)nprobe);
            for (int c = 0; c < nprobe; c++) loc[c].s = -2e9f;
            #pragma omp for schedule(static)
            for (int k = 0; k < K; k++) {
                float s = dotf(qn, A + (size_t)k * dim, dim);
                if (s > loc[nprobe - 1].s) {
                    int j = nprobe - 1;
                    while (j > 0 && s > loc[j - 1].s) {
                        loc[j] = loc[j - 1]; j--;
                    }
                    loc[j].s = s; loc[j].id = (uint32_t)k;
                }
            }
            #pragma omp critical
            for (int c = 0; c < nprobe; c++) {
                float s = loc[c].s;
                if (s > cand[nprobe - 1].s) {
                    int j = nprobe - 1;
                    while (j > 0 && s > cand[j - 1].s) {
                        cand[j] = cand[j - 1]; j--;
                    }
                    cand[j].s = s; cand[j].id = loc[c].id;
                }
            }
            free(loc);
        }
        double T1 = now_ms();
        /* vague io_uring : nprobe blocs */
        uint64_t bo = 0;
        for (int c = 0; c < nprobe; c++) {
            int k = (int)cand[c].id;
            boff[c] = bo;
            blen[c] = offs[k + 1] - offs[k];
            struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
            io_uring_prep_read(sqe, bfd, blk + bo, (unsigned)blen[c],
                               (off_t)offs[k]);
            bo += blen[c];
        }
        io_uring_submit(&ring);
        for (int c = 0; c < nprobe; c++) {
            struct io_uring_cqe* cqe;
            io_uring_wait_cqe(&ring, &cqe);
            io_uring_cqe_seen(&ring, cqe);
        }
        double T2 = now_ms();
        /* scoring TQ : rotation requete + int8, top-rerank */
        rot_seeded(qn, qr, sgn, dim);
        float qmax = 0;
        for (int d = 0; d < dim; d++) {
            float v = fabsf(qr[d]);
            if (v > qmax) qmax = v;
        }
        for (int d = 0; d < dim; d++) {
            int q = (int)lrintf(qr[d] / (qmax + 1e-9f) * 127.0f);
            q8[d] = (int8_t)(q < -127 ? -127 : (q > 127 ? 127 : q));
        }
        /* requete reordonnee pour score_tq4 : dims paires / impaires */
        int8_t* qlo = c8;               /* reutilise le scratch */
        int8_t* qhi = c8 + dim / 2;
        for (int d = 0; d < dim; d += 2) {
            qlo[d >> 1] = q8[d];
            qhi[d >> 1] = q8[d + 1];
        }
        int hn = 0;
        int64_t docs_seen = 0;
        #pragma omp parallel reduction(+ : docs_seen)
        {
            ScId* lh = (ScId*)malloc(sizeof(ScId) * (size_t)rerank);
            int ln = 0;
            #pragma omp for schedule(dynamic, 1)
            for (int c = 0; c < nprobe; c++) {
                const uint8_t* base = blk + boff[c];
                int64_t ne = (int64_t)(blen[c] / ent_b);
                docs_seen += ne;
                for (int64_t e = 0; e < ne; e++) {
                    const uint8_t* ent = base + e * ent_b;
                    const uint8_t* code = ent + 4;
                    float s = (m.tqbits == 4)
                        ? (float)score_tq4(code, qlo, qhi, dim)
                        : (float)score_tq1(code, q8, dim);
                    if (ln < rerank) {
                        lh[ln].s = s;
                        memcpy(&lh[ln].id, ent, 4);
                        ln++;
                        if (ln == rerank)
                            qsort(lh, ln, sizeof(ScId), scid_cmp);
                    } else if (s > lh[rerank - 1].s) {
                        int j = rerank - 1;
                        while (j > 0 && s > lh[j - 1].s) {
                            lh[j] = lh[j - 1]; j--;
                        }
                        lh[j].s = s;
                        memcpy(&lh[j].id, ent, 4);
                    }
                }
            }
            if (ln < rerank) qsort(lh, ln, sizeof(ScId), scid_cmp);
            #pragma omp critical
            for (int e = 0; e < ln; e++) {
                float s = lh[e].s;
                if (hn < rerank) {
                    heap[hn++] = lh[e];
                    if (hn == rerank)
                        qsort(heap, hn, sizeof(ScId), scid_cmp);
                } else if (s > heap[rerank - 1].s) {
                    int j = rerank - 1;
                    while (j > 0 && s > heap[j - 1].s) {
                        heap[j] = heap[j - 1]; j--;
                    }
                    heap[j] = lh[e];
                } else break; /* lh trie : plus rien a inserer */
            }
            free(lh);
        }
        if (hn < rerank) qsort(heap, hn, sizeof(ScId), scid_cmp);
        docs_seen_tot += docs_seen;
        double T3 = now_ms();
        /* rerank exact : pread f16 */
        int nr = hn < rerank ? hn : rerank;
        ScId* fin = (ScId*)malloc(sizeof(ScId) * (size_t)nr);
        int nf = 0;
        for (int e = 0; e < nr; e++) {
            uint32_t id = heap[e].id;
            int dup = 0;
            for (int x = 0; x < nf; x++)
                if (fin[x].id == id) { dup = 1; break; }
            if (dup) continue;
            if (pread(basefd, vrow, (size_t)dim * 2,
                      8 + (int64_t)id * dim * 2) != (int64_t)dim * 2)
                continue;
            row_f16_to_unit(vrow, vf, dim);
            fin[nf].id = id;
            fin[nf].s = dotf(qn, vf, dim);
            nf++;
        }
        qsort(fin, nf, sizeof(ScId), scid_cmp);
        for (int e = 0; e < 11; e++)
            out_ids[qi * 11 + e] = e < nf ? fin[e].id : 0xFFFFFFFFu;
        free(fin);
        double T4 = now_ms();
        t_anc[qi] = T1 - T0; t_io[qi] = T2 - T1;
        t_sc[qi] = T3 - T2; t_rr[qi] = T4 - T3; t_tot[qi] = T4 - T0;
    }
    /* stats */
    for (int s = 0; s < 5; s++) {
        double* t = (double*[]){t_anc, t_io, t_sc, t_rr, t_tot}[s];
        const char* nm = (const char*[]){"ancres", "io", "score", "rerank",
                                         "TOTAL"}[s];
        double tmp[4096];
        int nn = nq < 4096 ? nq : 4096;
        memcpy(tmp, t, (size_t)nn * 8);
        for (int i = 1; i < nn; i++) {
            double v = tmp[i];
            int j = i - 1;
            while (j >= 0 && tmp[j] > v) { tmp[j + 1] = tmp[j]; j--; }
            tmp[j + 1] = v;
        }
        printf("%-7s p50 %7.2f ms  p99 %7.2f ms\n", nm,
               tmp[nn / 2], tmp[(int)(nn * 0.99)]);
    }
    printf("docs vus/req : %lld\n", (long long)(docs_seen_tot / nq));
    if (outp) {
        FILE* fo = fopen(outp, "wb");
        fwrite(out_ids, 4, (size_t)nq * 11, fo);
        fclose(fo);
    }
    io_uring_queue_exit(&ring);
    return 0;
}
