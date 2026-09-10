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

/* ---------- mode S3 : vagues de range-GETs paralleles (libcurl multi,
   SigV4 natif). Cles UNIQUEMENT via l environnement AWS_ACCESS_KEY_ID /
   AWS_SECRET_ACCESS_KEY (jamais en argv ni en fichier). ---------- */
#include <curl/curl.h>
typedef struct { uint8_t* dst; size_t cap; size_t got; } S3Buf;

static size_t s3_write_cb(void* p, size_t s, size_t n, void* u) {
    S3Buf* b = (S3Buf*)u;
    size_t k = s * n;
    if (b->got + k > b->cap) k = b->cap - b->got;
    memcpy(b->dst + b->got, p, k);
    b->got += k;
    return s * n;
}

typedef struct {
    CURLM* multi;
    char userpwd[512];
    int has_auth;
    int hedge_ms;       /* 0 = pas de hedging */
    int last_hedged;    /* GETs doubles lors de la derniere vague */
    long last_newconn;  /* connexions TCP ouvertes (0 = tout reutilise) */
} S3Ctx;

static double now_ms_s3(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static int s3_init(S3Ctx* c) {
    curl_global_init(CURL_GLOBAL_DEFAULT);
    c->multi = curl_multi_init();
    curl_multi_setopt(c->multi, CURLMOPT_MAX_HOST_CONNECTIONS, 512L);
    curl_multi_setopt(c->multi, CURLMOPT_MAX_TOTAL_CONNECTIONS, 512L);
    curl_multi_setopt(c->multi, CURLMOPT_MAXCONNECTS, 512L);
    const char* k = getenv("AWS_ACCESS_KEY_ID");
    const char* s = getenv("AWS_SECRET_ACCESS_KEY");
    c->has_auth = (k && s);
    if (c->has_auth) snprintf(c->userpwd, sizeof(c->userpwd), "%s:%s", k, s);
    return c->multi ? 0 : -1;
}

/* n range-GETs sur `url`, [off[i], off[i]+len[i]) -> dst[i]. Renvoie le
   nombre de reponses completes. Hedging : pas encore (v1).            */
static int s3_wave(S3Ctx* c, const char* url, const uint64_t* off,
                   const uint64_t* len, uint8_t* const* dst, int n) {
    CURL** hs = (CURL**)malloc(sizeof(CURL*) * (size_t)n);
    S3Buf* bufs = (S3Buf*)malloc(sizeof(S3Buf) * (size_t)n);
    char rng[64];
    for (int i = 0; i < n; i++) {
        hs[i] = curl_easy_init();
        bufs[i].dst = dst[i]; bufs[i].cap = (size_t)len[i]; bufs[i].got = 0;
        snprintf(rng, sizeof(rng), "%llu-%llu", (unsigned long long)off[i],
                 (unsigned long long)(off[i] + len[i] - 1));
        curl_easy_setopt(hs[i], CURLOPT_URL, url);
        curl_easy_setopt(hs[i], CURLOPT_RANGE, rng);
        curl_easy_setopt(hs[i], CURLOPT_WRITEFUNCTION, s3_write_cb);
        curl_easy_setopt(hs[i], CURLOPT_WRITEDATA, &bufs[i]);
        curl_easy_setopt(hs[i], CURLOPT_TCP_KEEPALIVE, 1L);
        if (c->has_auth) {
            curl_easy_setopt(hs[i], CURLOPT_AWS_SIGV4, "aws:amz:us-east-1:s3");
            curl_easy_setopt(hs[i], CURLOPT_USERPWD, c->userpwd);
        }
        curl_multi_add_handle(c->multi, hs[i]);
    }
    /* HEDGING sur deadline : quand la vague depasse hedge_ms sans etre
       complete, les GETs encore en vol sont DOUBLES (nouvelle connexion,
       meme plage) ; le premier arrive gagne. Tue la queue p99 des SSD
       objets / du WAN sans attendre les trainards.                     */
    int hedge_ms = c->hedge_ms > 0 ? c->hedge_ms : 100000;
    CURL** hh = (CURL**)calloc((size_t)n, sizeof(CURL*));
    S3Buf* hb = (S3Buf*)calloc((size_t)n, sizeof(S3Buf));
    uint8_t** hmem = (uint8_t**)calloc((size_t)n, sizeof(uint8_t*));
    int hedged = 0;
    double t0 = now_ms_s3();
    int running = 1;
    int hedge_done = 0;
    while (running) {
        curl_multi_perform(c->multi, &running);
        if (!running) break;
        if (!hedge_done && now_ms_s3() - t0 > hedge_ms) {
            hedge_done = 1;
            for (int i = 0; i < n; i++) {
                if (bufs[i].got >= bufs[i].cap) continue;   /* deja recu */
                hmem[i] = (uint8_t*)malloc(bufs[i].cap);
                hb[i].dst = hmem[i]; hb[i].cap = bufs[i].cap; hb[i].got = 0;
                hh[i] = curl_easy_init();
                snprintf(rng, sizeof(rng), "%llu-%llu",
                         (unsigned long long)off[i],
                         (unsigned long long)(off[i] + len[i] - 1));
                curl_easy_setopt(hh[i], CURLOPT_URL, url);
                curl_easy_setopt(hh[i], CURLOPT_RANGE, rng);
                curl_easy_setopt(hh[i], CURLOPT_WRITEFUNCTION, s3_write_cb);
                curl_easy_setopt(hh[i], CURLOPT_WRITEDATA, &hb[i]);
                /* sur connexion CHAUDE du pool : un hedge en connexion
                   neuve paie handshake + slow-start et ne gagne jamais */
                if (c->has_auth) {
                    curl_easy_setopt(hh[i], CURLOPT_AWS_SIGV4,
                                     "aws:amz:us-east-1:s3");
                    curl_easy_setopt(hh[i], CURLOPT_USERPWD, c->userpwd);
                }
                curl_multi_add_handle(c->multi, hh[i]);
                hedged++;
            }
        }
        /* fin anticipee : chaque plage recue par l un OU l autre */
        int all = 1;
        for (int i = 0; i < n; i++)
            if (bufs[i].got < bufs[i].cap && !(hh[i] && hb[i].got >= hb[i].cap)) {
                all = 0; break;
            }
        if (all) break;
        curl_multi_poll(c->multi, NULL, 0, 20, NULL);
    }
    int ok = 0;
    long newconn = 0;
    for (int i = 0; i < n; i++) {
        long nc = 0;
        curl_easy_getinfo(hs[i], CURLINFO_NUM_CONNECTS, &nc);
        newconn += nc;                 /* 0 = connexion reutilisee */
        if (bufs[i].got >= bufs[i].cap) ok++;
        else if (hh[i] && hb[i].got >= hb[i].cap) {
            memcpy(bufs[i].dst, hmem[i], bufs[i].cap); ok++;
        }
        curl_multi_remove_handle(c->multi, hs[i]);
        curl_easy_cleanup(hs[i]);
        if (hh[i]) { curl_multi_remove_handle(c->multi, hh[i]);
                     curl_easy_cleanup(hh[i]); }
        free(hmem[i]);
    }
    c->last_hedged = hedged;
    c->last_newconn = newconn;
    free(hs); free(bufs); free(hh); free(hb); free(hmem);
    return ok;
}

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

/* TQ2 : 2 bits/dim (4 dims/octet, valeur stockee (q+2)&3, q in [-2,1]).
   Decodage NEON par champs de 2 bits + 4 SDOT contre la requete
   deinterlacee en 4 flux (dims = 0,1,2,3 mod 4).                        */
static inline int32_t score_tq2(const uint8_t* code, const int8_t* q0,
                                const int8_t* q1, const int8_t* q2,
                                const int8_t* q3, int dim) {
#if ANC_NEON && defined(__ARM_FEATURE_DOTPROD)
    int32x4_t acc = vdupq_n_s32(0);
    int8x16_t three = vdupq_n_s8(3), two = vdupq_n_s8(2);
    int nb = dim / 4;
    for (int i = 0; i + 16 <= nb; i += 16) {
        uint8x16_t b = vld1q_u8(code + i);
        int8x16_t d0 = vsubq_s8(vandq_s8(vreinterpretq_s8_u8(b), three), two);
        int8x16_t d1 = vsubq_s8(vandq_s8(
            vreinterpretq_s8_u8(vshrq_n_u8(b, 2)), three), two);
        int8x16_t d2 = vsubq_s8(vandq_s8(
            vreinterpretq_s8_u8(vshrq_n_u8(b, 4)), three), two);
        int8x16_t d3 = vsubq_s8(
            vreinterpretq_s8_u8(vshrq_n_u8(b, 6)), two);
        acc = vdotq_s32(acc, d0, vld1q_s8(q0 + i));
        acc = vdotq_s32(acc, d1, vld1q_s8(q1 + i));
        acc = vdotq_s32(acc, d2, vld1q_s8(q2 + i));
        acc = vdotq_s32(acc, d3, vld1q_s8(q3 + i));
    }
    return vaddvq_s32(acc);
#else
    int32_t s = 0;
    for (int d = 0; d < dim; d += 4) {
        uint8_t b = code[d >> 2];
        s += ((int)(b & 3) - 2) * q0[d >> 2]
           + ((int)((b >> 2) & 3) - 2) * q1[d >> 2]
           + ((int)((b >> 4) & 3) - 2) * q2[d >> 2]
           + ((int)((b >> 6) & 3) - 2) * q3[d >> 2];
    }
    return s;
#endif
}

/* TQ1 : somme masquee des q8 aux bits leves (offset constant par requete,
   sans effet sur l ordre). 8 dims par octet via vtst.                   */
/* TQ1 sur le motif TQ2 (qui est rapide) : 8 flux de bits extraits par
   decalage+AND, SDOT contre la requete deinterlacee en 8 flux
   (flux b = dims congrues a b mod 8). qs1 : 8 flux contigus de `stride`
   octets chacun ; `dim` = dims a traiter (prefixe ou tout).           */
static inline int32_t score_tq1(const uint8_t* code, const int8_t* qs1,
                                int stride, int dim) {
#if ANC_NEON && defined(__ARM_FEATURE_DOTPROD)
    int32x4_t acc = vdupq_n_s32(0);
    int8x16_t one = vdupq_n_s8(1);
    int nb = dim / 8;
    for (int i = 0; i + 16 <= nb; i += 16) {
        uint8x16_t b = vld1q_u8(code + i);
        /* deroule explicitement les 8 decalages (vshrq_n exige une cst) */
        acc = vdotq_s32(acc, vreinterpretq_s8_u8(vandq_u8(b, vreinterpretq_u8_s8(one))), vld1q_s8(qs1 + 0 * stride + i));
        acc = vdotq_s32(acc, vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(b, 1), vreinterpretq_u8_s8(one))), vld1q_s8(qs1 + 1 * stride + i));
        acc = vdotq_s32(acc, vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(b, 2), vreinterpretq_u8_s8(one))), vld1q_s8(qs1 + 2 * stride + i));
        acc = vdotq_s32(acc, vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(b, 3), vreinterpretq_u8_s8(one))), vld1q_s8(qs1 + 3 * stride + i));
        acc = vdotq_s32(acc, vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(b, 4), vreinterpretq_u8_s8(one))), vld1q_s8(qs1 + 4 * stride + i));
        acc = vdotq_s32(acc, vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(b, 5), vreinterpretq_u8_s8(one))), vld1q_s8(qs1 + 5 * stride + i));
        acc = vdotq_s32(acc, vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(b, 6), vreinterpretq_u8_s8(one))), vld1q_s8(qs1 + 6 * stride + i));
        acc = vdotq_s32(acc, vreinterpretq_s8_u8(vshrq_n_u8(b, 7)), vld1q_s8(qs1 + 7 * stride + i));
    }
    return vaddvq_s32(acc);
#else
    int32_t s = 0;
    for (int d = 0; d < dim; d++)
        if ((code[d >> 3] >> (d & 7)) & 1) s += qs1[(d & 7) * stride + (d >> 3)];
    return s;
#endif
}

typedef struct {
    int K, dim, M, tqbits;
    float eps;
    int64_t n;
    uint64_t seed;
    int cdim;   /* dims couvertes par le code stocke (layout leger) */
} AMeta;

static int meta_load(const char* dir, AMeta* m) {
    char p[1024];
    snprintf(p, sizeof(p), "%s/meta.txt", dir);
    FILE* f = fopen(p, "r");
    if (!f) return -1;
    long long n = 0; unsigned long long sd = 0;
    int cd = 0;
    int r = fscanf(f, "%d %d %d %d %f %lld %llu %d", &m->K, &m->dim, &m->M,
                   &m->tqbits, &m->eps, &n, &sd, &cd);
    fclose(f);
    m->n = n; m->seed = sd;
    m->cdim = (r >= 8 && cd > 0) ? cd : m->dim;
    return r >= 7 ? 0 : -1;
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
    int cdim = 0;
    const char* coarse = NULL;
    for (int i = 5; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--eps")) eps = atof(argv[i + 1]);
        else if (!strcmp(argv[i], "--m")) M = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--tqbits")) tqbits = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--seed")) seed = strtoull(argv[i + 1], 0, 10);
        else if (!strcmp(argv[i], "--nmax")) nmax = atoll(argv[i + 1]);
        else if (!strcmp(argv[i], "--coarse")) coarse = argv[i + 1];
        else if (!strcmp(argv[i], "--cdim")) cdim = atoi(argv[i + 1]);
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
    /* --- assignation HIERARCHIQUE via un index existant (--coarse) :
       les plus proches parmi les K nouvelles ancres se cherchent dans le
       voisinage des anciennes ancres du doc. O(N x ~96) au lieu de
       O(N x K) — c est aussi le chemin de production a 1B.             */
    if (!skip_pass1 && coarse) {
        AMeta om;
        char op[1024];
        if (meta_load(coarse, &om) != 0 || om.dim != dim || om.n != n) {
            fprintf(stderr, "coarse: meta incompatible\n");
            return 1;
        }
        int Ko = om.K, Mo = om.M;
        snprintf(op, sizeof(op), "%s/anchors.bin", coarse);
        FILE* f = fopen(op, "rb");
        float* Ao = (float*)malloc((size_t)Ko * dim * 4);
        if (!f || fread(Ao, 4, (size_t)Ko * dim, f) != (size_t)Ko * dim)
            return 1;
        fclose(f);
        snprintf(op, sizeof(op), "%s/assign.bin", coarse);
        f = fopen(op, "rb");
        int64_t ah[2];
        int32_t* topm_o = (int32_t*)malloc((size_t)n * Mo * 4);
        if (!f || fread(ah, 8, 2, f) != 2 || ah[0] != n || ah[1] != Mo
            || fread(topm_o, 4, (size_t)n * Mo, f) != (size_t)n * Mo) {
            fprintf(stderr, "coarse: assign.bin absent/incompatible\n");
            return 1;
        }
        fseeko(f, (off_t)((size_t)n * Mo * 4), SEEK_CUR); /* saute topd */
        if (fread(sigmean, 4, dim, f) != (size_t)dim) return 1;
        fclose(f);
        /* table de voisinage ancienne -> nouvelles : NBR plus proches */
        const int NBR = 64;
        int32_t* nbrs = (int32_t*)malloc((size_t)Ko * NBR * 4);
        double t1 = omp_get_wtime();
        #pragma omp parallel
        {
            int8_t* v8 = (int8_t*)malloc(dim);
            #pragma omp for schedule(dynamic, 16)
            for (int ko = 0; ko < Ko; ko++) {
                const float* v = Ao + (size_t)ko * dim;
                float vmax = 0;
                for (int d = 0; d < dim; d++) {
                    float x = fabsf(v[d]);
                    if (x > vmax) vmax = x;
                }
                float vq = 127.0f / (vmax + 1e-9f);
                for (int d = 0; d < dim; d++)
                    v8[d] = (int8_t)lrintf(v[d] * vq);
                float bd[64];
                int32_t bi[64];
                for (int j = 0; j < NBR; j++) bd[j] = -2e9f;
                for (int k = 0; k < K; k++) {
                    float s = (float)doti8(v8, A8 + (size_t)k * dim, dim);
                    if (s > bd[NBR - 1]) {
                        int j = NBR - 1;
                        while (j > 0 && s > bd[j - 1]) {
                            bd[j] = bd[j - 1]; bi[j] = bi[j - 1]; j--;
                        }
                        bd[j] = s; bi[j] = k;
                    }
                }
                memcpy(nbrs + (size_t)ko * NBR, bi, NBR * 4);
            }
            free(v8);
        }
        fprintf(stderr, "abuild: voisinage %dx%d en %.0fs\n", Ko, NBR,
                omp_get_wtime() - t1);
        /* passe docs : candidats = union des voisinages des Mo anciennes */
        fseeko(bf, 8, SEEK_SET);
        t1 = omp_get_wtime();
        for (int64_t off = 0; off < n; off += CHUNK) {
            int64_t c = n - off < CHUNK ? n - off : CHUNK;
            if (fread(raw, (size_t)dim * 2, c, bf) != (size_t)c) return 1;
            #pragma omp parallel
            {
                float* v = (float*)malloc((size_t)dim * 4);
                int8_t* v8 = (int8_t*)malloc(dim);
                int32_t cnd[128];
                #pragma omp for schedule(dynamic, 64)
                for (int64_t i = 0; i < c; i++) {
                    int64_t g = off + i;
                    row_f16_to_unit(raw + (size_t)i * dim, v, dim);
                    float vmax = 0;
                    for (int d = 0; d < dim; d++) {
                        float x = fabsf(v[d]);
                        if (x > vmax) vmax = x;
                    }
                    float vq = 127.0f / (vmax + 1e-9f);
                    for (int d = 0; d < dim; d++)
                        v8[d] = (int8_t)lrintf(v[d] * vq);
                    int nc = 0;
                    for (int mo = 0; mo < Mo; mo++) {
                        int32_t ko = topm_o[g * Mo + mo];
                        if (ko < 0) continue;
                        const int32_t* nb = nbrs + (size_t)ko * NBR;
                        for (int j = 0; j < NBR; j++) {
                            int32_t k = nb[j];
                            int dup = 0;
                            for (int x = 0; x < nc; x++)
                                if (cnd[x] == k) { dup = 1; break; }
                            if (!dup) cnd[nc++] = k;
                        }
                    }
                    float bd[4] = {2e9f, 2e9f, 2e9f, 2e9f};
                    int32_t bi[4] = {-1, -1, -1, -1};
                    for (int x = 0; x < nc; x++) {
                        float d2 = -(float)doti8(
                            v8, A8 + (size_t)cnd[x] * dim, dim);
                        if (d2 < bd[M - 1]) {
                            int j = M - 1;
                            while (j > 0 && d2 < bd[j - 1]) {
                                bd[j] = bd[j - 1]; bi[j] = bi[j - 1]; j--;
                            }
                            bd[j] = d2; bi[j] = cnd[x];
                        }
                    }
                    float inv = 1.0f / (vq * aq);
                    for (int j = 0; j < M; j++) {
                        topm[g * M + j] = bi[j];
                        topd[g * M + j] = 2.0f + 2.0f * bd[j] * inv;
                    }
                }
                free(v); free(v8);
            }
            if (off % 4000000 == 0)
                fprintf(stderr, "abuild: coarse-assign %lld/%lld (%.0fs)\n",
                        (long long)off, (long long)n, omp_get_wtime() - t1);
        }
        free(Ao); free(topm_o); free(nbrs);
        skip_pass1 = 1;
        FILE* af = fopen(apath, "wb");
        if (af) {
            int64_t h2[2] = {n, M};
            fwrite(h2, 8, 2, af);
            fwrite(topm, 4, (size_t)n * M, af);
            fwrite(topd, 4, (size_t)n * M, af);
            fwrite(sigmean, 4, dim, af);
            fclose(af);
        }
        fprintf(stderr, "abuild: assignation hierarchique OK\n");
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
    if (cdim <= 0 || cdim > dim) cdim = dim;
    int code_b = cdim * tqbits / 8;
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
                    for (int d = 0; d < cdim; d += 2) {
                        int q0 = (int)lrintf(r[d] * scale[d]);
                        int q1 = (int)lrintf(r[d + 1] * scale[d + 1]);
                        if (q0 < -8) q0 = -8; if (q0 > 7) q0 = 7;
                        if (q1 < -8) q1 = -8; if (q1 > 7) q1 = 7;
                        code[d >> 1] = (uint8_t)((q0 & 15) | ((q1 & 15) << 4));
                    }
                } else if (tqbits == 2) {
                    for (int d = 0; d < cdim; d += 4) {
                        uint8_t b = 0;
                        for (int j = 0; j < 4; j++) {
                            int q = (int)lrintf(r[d + j] * scale[d + j]);
                            if (q < -2) q = -2;
                            if (q > 1) q = 1;
                            b |= (uint8_t)((q + 2) & 3) << (2 * j);
                        }
                        code[d >> 2] = b;
                    }
                } else { /* tq1 : bit de signe */
                    memset(code, 0, code_b);
                    for (int d = 0; d < cdim; d++)
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
    fprintf(fo, "%d %d %d %d %.4f %lld %llu %d\n", K, dim, M, tqbits, eps,
            (long long)n, (unsigned long long)seed, cdim);
    fclose(fo);
    fprintf(stderr, "abuild: DONE\n");
    fclose(bf);
    free(A); free(A8); free(aids); free(raw); free(topm); free(topd);
    free(cnt); free(offs); free(cur); free(scale); free(sgn); free(rowbuf);
    return 0;
}

/* ================= BENCH (query + latences) ================= */
typedef struct { float s; uint32_t id; } ScId;
typedef struct { float s; const uint8_t* ent; } ScEnt;

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
    const char* s3url = NULL;   /* ex: http://127.0.0.1:9000/wikiit */
    int hedge_ms = 0;
    for (int i = 8; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--out")) outp = argv[i + 1];
        else if (!strcmp(argv[i], "--drop")) drop = argv[i + 1];
        else if (!strcmp(argv[i], "--s3")) s3url = argv[i + 1];
        else if (!strcmp(argv[i], "--hedge")) hedge_ms = atoi(argv[i + 1]);
    }
    S3Ctx s3;
    memset(&s3, 0, sizeof(s3));
    char s3_blocks[1024], s3_base[1024];
    long hedged_tot = 0, newconn_tot = 0;
    if (s3url) {
        if (s3_init(&s3) != 0) { fprintf(stderr, "curl init?\n"); return 1; }
        s3.hedge_ms = hedge_ms;
        snprintf(s3_blocks, sizeof(s3_blocks), "%s/blocks.bin", s3url);
        snprintf(s3_base, sizeof(s3_base), "%s/base.f16bin", s3url);
        fprintf(stderr, "mode S3 : %s (auth %s)\n", s3url,
                s3.has_auth ? "SigV4" : "aucune");
    }
    AMeta m;
    if (meta_load(dir, &m) != 0) { fprintf(stderr, "meta?\n"); return 1; }
    int dim = m.dim, K = m.K;
    int cdim = m.cdim;
    int code_b = cdim * m.tqbits / 8;
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
    if (!s3url && (bfd < 0 || basefd < 0)) { perror("open"); return 1; }

    FILE* qf = fopen(qpath, "rb");
    uint32_t qh[2];
    if (fread(qh, 4, 2, qf) != 2 || (int)qh[1] != dim) return 1;
    if ((int)qh[0] < nq) nq = (int)qh[0];
    float* Q = (float*)malloc((size_t)nq * dim * 4);
    if (fread(Q, 4, (size_t)nq * dim, qf) != (size_t)nq * dim) return 1;
    fclose(qf);

    struct io_uring ring;
    io_uring_queue_init(1024, &ring, 0);
    /* buffer blocs : nprobe x p99 des cellules (pas le max — une cellule
       geante x nprobe faisait des GB), reallocation si une requete depasse */
    uint64_t* csz = (uint64_t*)malloc((size_t)K * 8);
    for (int k = 0; k < K; k++) csz[k] = offs[k + 1] - offs[k];
    /* dimensionne au p99 x2 des cellules et PRE-TOUCHE : un buffer
       realloue/refaulte a chaque requete coutait ~90 ms de noyau
       (copy_to_user + clear_page) en mono-thread — vu au perf.        */
    uint64_t* csrt = (uint64_t*)malloc((size_t)K * 8);
    memcpy(csrt, csz, (size_t)K * 8);
    for (int i = 1; i < K; i++) {           /* tri insertion partiel suffit */
        uint64_t v = csrt[i]; int j = i - 1;
        while (j >= 0 && csrt[j] > v) { csrt[j + 1] = csrt[j]; j--; }
        csrt[j + 1] = v;
        if (i > 20000) break;
    }
    uint64_t p99 = csrt[(int)(K * 0.99)];
    if (K > 20000) { /* tri partiel invalide au-dela : prend le max/2 */
        uint64_t mx = 0;
        for (int k = 0; k < K; k++) if (csz[k] > mx) mx = csz[k];
        p99 = mx / 2;
    }
    free(csrt);
    size_t blk_cap = (size_t)nprobe * (p99 * 2 + 65536);
    uint8_t* blk = (uint8_t*)malloc(blk_cap);
    if (!blk) { fprintf(stderr, "OOM blocs\n"); return 1; }
    memset(blk, 0, blk_cap);                /* pre-fault des pages */
    free(csz);
    uint64_t* boff = (uint64_t*)malloc((size_t)nprobe * 8);
    uint64_t* blen = (uint64_t*)malloc((size_t)nprobe * 8);
    ScId* heap = (ScId*)malloc(((size_t)rerank + 1) * sizeof(ScId));
    /* scratchs persistants (alloues + pre-touches UNE fois) */
    int nth = omp_get_max_threads();
    size_t ph_stride = (size_t)(16384 > rerank * 4 ? 16384 : rerank * 4);
    ScEnt* ph_all = (ScEnt*)malloc((size_t)nth * ph_stride * sizeof(ScEnt));
    ScId* lh_all = (ScId*)malloc((size_t)nth * rerank * sizeof(ScId));
    ScId* fin_all = (ScId*)malloc((size_t)rerank * sizeof(ScId));
    uint8_t* rows_all = (uint8_t*)malloc((size_t)rerank * dim * 2 + 16);
    if (!ph_all || !lh_all || !fin_all || !rows_all) return 1;
    memset(ph_all, 0, (size_t)nth * ph_stride * sizeof(ScEnt));
    memset(lh_all, 0, (size_t)nth * rerank * sizeof(ScId));
    memset(rows_all, 0, (size_t)rerank * dim * 2 + 16);
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
        uint64_t need = 0;
        for (int c = 0; c < nprobe; c++)
            need += offs[cand[c].id + 1] - offs[cand[c].id];
        if (need > blk_cap) {
            uint8_t* nb = (uint8_t*)realloc(blk, need);
            if (!nb) { fprintf(stderr, "OOM blocs req\n"); return 1; }
            blk = nb; blk_cap = need;
        }
        uint64_t bo = 0;
        if (s3url) {
            uint64_t* soff = (uint64_t*)malloc((size_t)nprobe * 8);
            uint8_t** sdst = (uint8_t**)malloc((size_t)nprobe * sizeof(void*));
            for (int c = 0; c < nprobe; c++) {
                int k = (int)cand[c].id;
                boff[c] = bo;
                blen[c] = offs[k + 1] - offs[k];
                soff[c] = offs[k];
                sdst[c] = blk + bo;
                bo += blen[c];
            }
            int ok = s3_wave(&s3, s3_blocks, soff, blen, sdst, nprobe);
            hedged_tot += s3.last_hedged;
            newconn_tot += s3.last_newconn;
            if (ok < nprobe && qi == 0)
                fprintf(stderr, "S3 : %d/%d blocs recus\n", ok, nprobe);
            free(soff); free(sdst);
        } else {
        for (int c = 0; c < nprobe; c++) {
            int k = (int)cand[c].id;
            boff[c] = bo;
            blen[c] = offs[k + 1] - offs[k];
            struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
            if (!sqe) {
                io_uring_submit(&ring);
                sqe = io_uring_get_sqe(&ring);
                if (!sqe) { fprintf(stderr, "sqe?\n"); return 1; }
            }
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
        /* requete reordonnee selon le mode : tq4 = 2 flux pair/impair,
           tq2 = 4 flux (dims mod 4), tq1 = q8 direct.                   */
        int8_t* qlo = c8;               /* reutilise le scratch */
        int8_t* qhi = c8 + dim / 2;
        int8_t* qs2[4] = {c8, c8 + dim / 4, c8 + dim / 2,
                          c8 + 3 * (dim / 4)};
        if (m.tqbits == 4) {
            for (int d = 0; d < dim; d += 2) {
                qlo[d >> 1] = q8[d];
                qhi[d >> 1] = q8[d + 1];
            }
        } else if (m.tqbits == 2) {
            for (int d = 0; d < dim; d++)
                qs2[d & 3][d >> 2] = q8[d];
        } else { /* tq1 : 8 flux de cdim/8 octets (flux b = dims = b mod 8) */
            for (int d = 0; d < cdim; d++)
                c8[(d & 7) * (cdim / 8) + (d >> 3)] = q8[d];
        }
        /* SCORING PROGRESSIF : pre-score sur les DIM_PRE premieres dims
           tournees (la FWHT egalise l energie -> le prefixe porte
           DIM_PRE/dim de la variance), preselection top-PRE_KEEP par
           thread, puis score COMPLET des seuls survivants. CPU ~/4.    */
        /* min-tas binaire sur s : remplacement du minimum en O(log n)
           (l insertion decalee coutait O(n) — mur a 16k en mono-thread) */
        #define PH_SIFT(ph, n) do {                                      \
            int _i = 0;                                                  \
            for (;;) {                                                   \
                int _l = 2 * _i + 1, _r = _l + 1, _m = _i;               \
                if (_l < (n) && (ph)[_l].s < (ph)[_m].s) _m = _l;        \
                if (_r < (n) && (ph)[_r].s < (ph)[_m].s) _m = _r;        \
                if (_m == _i) break;                                     \
                ScEnt _t = (ph)[_i]; (ph)[_i] = (ph)[_m];                \
                (ph)[_m] = _t; _i = _m;                                  \
            }                                                            \
        } while (0)
        const int DIM_PRE = (cdim >= 512) ? 256 : cdim;
        int hn = 0;
        int64_t docs_seen = 0;
        #pragma omp parallel reduction(+ : docs_seen)
        {
            int PRE_KEEP = 16384 / omp_get_num_threads();
            if (PRE_KEEP < rerank * 4) PRE_KEEP = rerank * 4;
            /* scratch persistant par thread (pas de malloc par requete) */
            ScEnt* ph = ph_all + (size_t)omp_get_thread_num() * ph_stride;
            int pn = 0;
            #pragma omp for schedule(dynamic, 1)
            for (int c = 0; c < nprobe; c++) {
                const uint8_t* base = blk + boff[c];
                int64_t ne = (int64_t)(blen[c] / ent_b);
                docs_seen += ne;
                for (int64_t e = 0; e < ne; e++) {
                    const uint8_t* ent = base + e * ent_b;
                    const uint8_t* code = ent + 4;
                    float s = (m.tqbits == 4)
                        ? (float)score_tq4(code, qlo, qhi, DIM_PRE)
                        : (m.tqbits == 2)
                        ? (float)score_tq2(code, qs2[0], qs2[1],
                                           qs2[2], qs2[3], DIM_PRE)
                        : (float)score_tq1(code, c8, cdim / 8, DIM_PRE);
                    if (pn < PRE_KEEP) {
                        /* construction : sift-up */
                        int i2 = pn++;
                        ph[i2].s = s; ph[i2].ent = ent;
                        while (i2 > 0) {
                            int p2 = (i2 - 1) / 2;
                            if (ph[p2].s <= ph[i2].s) break;
                            ScEnt t = ph[p2]; ph[p2] = ph[i2];
                            ph[i2] = t; i2 = p2;
                        }
                    } else if (s > ph[0].s) {
                        ph[0].s = s; ph[0].ent = ent;
                        PH_SIFT(ph, PRE_KEEP);
                    }
                }
            }
            /* score complet des survivants locaux -> top-rerank local */
            ScId* lh = lh_all + (size_t)omp_get_thread_num() * rerank;
            int ln = 0;
            for (int e = 0; e < pn; e++) {
                const uint8_t* ent = ph[e].ent;
                const uint8_t* code = ent + 4;
                float s = (m.tqbits == 4)
                    ? (float)score_tq4(code, qlo, qhi, cdim)
                    : (m.tqbits == 2)
                    ? (float)score_tq2(code, qs2[0], qs2[1],
                                       qs2[2], qs2[3], cdim)
                    : (float)score_tq1(code, c8, cdim / 8, cdim);
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
        }
        if (hn < rerank) qsort(heap, hn, sizeof(ScId), scid_cmp);
        docs_seen_tot += docs_seen;
        double T3 = now_ms();
        /* rerank exact : pread f16 */
        /* rerank exact en UNE vague io_uring (les preads sequentiels
           coutaient 12-13 ms pour 300 lignes ; en vague ~2-3 ms) */
        int nr = hn < rerank ? hn : rerank;
        ScId* fin = fin_all;
        int nf = 0;
        for (int e = 0; e < nr; e++) {
            uint32_t id = heap[e].id;
            int dup = 0;
            for (int x = 0; x < nf; x++)
                if (fin[x].id == id) { dup = 1; break; }
            if (!dup) fin[nf++].id = id;
        }
        uint8_t* rows = rows_all;
        int inflight = 0;
        if (s3url) {
            uint64_t* roff = (uint64_t*)malloc((size_t)nf * 8);
            uint64_t* rlen = (uint64_t*)malloc((size_t)nf * 8);
            uint8_t** rdst = (uint8_t**)malloc((size_t)nf * sizeof(void*));
            for (int e = 0; e < nf; e++) {
                roff[e] = 8 + (uint64_t)fin[e].id * dim * 2;
                rlen[e] = (uint64_t)dim * 2;
                rdst[e] = rows + (size_t)e * dim * 2;
            }
            s3_wave(&s3, s3_base, roff, rlen, rdst, nf);
            hedged_tot += s3.last_hedged;
            free(roff); free(rlen); free(rdst);
        } else
        for (int e = 0; e < nf; e++) {
            struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
            if (!sqe) {
                io_uring_submit(&ring);
                struct io_uring_cqe* cqe;
                while (inflight > 0 && io_uring_wait_cqe(&ring, &cqe) == 0) {
                    io_uring_cqe_seen(&ring, cqe); inflight--;
                }
                sqe = io_uring_get_sqe(&ring);
                if (!sqe) { fprintf(stderr, "sqe rerank?\n"); return 1; }
            }
            io_uring_prep_read(sqe, basefd, rows + (size_t)e * dim * 2,
                               (unsigned)(dim * 2),
                               (off_t)(8 + (int64_t)fin[e].id * dim * 2));
            inflight++;
        }
        io_uring_submit(&ring);
        while (inflight > 0) {
            struct io_uring_cqe* cqe;
            if (io_uring_wait_cqe(&ring, &cqe) != 0) break;
            io_uring_cqe_seen(&ring, cqe);
            inflight--;
        }
        for (int e = 0; e < nf; e++) {
            row_f16_to_unit((const uint16_t*)(rows + (size_t)e * dim * 2),
                            vf, dim);
            fin[e].s = dotf(qn, vf, dim);
        }
        qsort(fin, nf, sizeof(ScId), scid_cmp);
        for (int e = 0; e < 11; e++)
            out_ids[qi * 11 + e] = e < nf ? fin[e].id : 0xFFFFFFFFu;
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
    if (s3url)
        printf("S3 : hedge %d ms, GETs doubles/req : %.1f, connexions TCP "
               "ouvertes/req (vague blocs) : %.1f\n", hedge_ms,
               (double)hedged_tot / nq, (double)newconn_tot / nq);
    if (outp) {
        FILE* fo = fopen(outp, "wb");
        fwrite(out_ids, 4, (size_t)nq * 11, fo);
        fclose(fo);
    }
    io_uring_queue_exit(&ring);
    return 0;
}
