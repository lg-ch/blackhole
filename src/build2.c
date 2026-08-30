/* BUILD2 — construction V2 niveau-synchrone : medianes EXACTES par noeud
   (population entiere, pas d echantillon), snap int8 par niveau, routage
   avec le seuil snappe + tie-break hash(doc_id) sur les ex aequo, sortie
   NATIVE v2 : medians.bin (MED1) + fichiers slots (tree%05d.slt).
   Ni phase paires, ni conversion srt, ni etape de calibration : les
   medianes sont un sous-produit du build.

   Regle memoire projet : AUCUN mmap — base chargee en malloc explicite
   (--base-ram, seul mode pour l instant ; le mode streaming doc-major a
   quelques Mo de RSS viendra comme second regime sur la meme structure).
   Budget : base (N*dim*4) + G arbres * N * 8 (chemin u32 + proj f32)
   + perm par thread. gen v3 uniquement.                                 */
#define _POSIX_C_SOURCE 200809L
#include "gen_vec.h"
#include "traversal.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MED_MAGIC 0x3144454Du

static inline int tie_bit(uint32_t doc) {
    return (int)(((uint64_t)doc * 0x9E3779B97F4A7C15ull) >> 37 & 1ull);
}

/* Quickselect mediane (moyenne basse pour n pair : element n/2, comme
   np.median a un epsilon pres — l exactitude ABSOLUE de la definition
   importe moins que sa constance build/route). */
static float med_select(float* a, int64_t n) {
    int64_t k = n / 2;
    int64_t lo = 0, hi = n - 1;
    while (lo < hi) {
        float pivot = a[(lo + hi) / 2];
        int64_t i = lo, j = hi;
        while (i <= j) {
            while (a[i] < pivot) i++;
            while (a[j] > pivot) j--;
            if (i <= j) {
                float t = a[i]; a[i] = a[j]; a[j] = t;
                i++; j--;
            }
        }
        if (k <= j) hi = j;
        else if (k >= i) lo = i;
        else break;
    }
    return a[k];
}

typedef struct {
    const float* base;
    int64_t n;
    int dim, sub_dim, depth, slot_bytes;
} B2;

/* Un niveau, un arbre : tri par comptage sur le chemin courant, mediane
   exacte + snap par noeud, routage. paths est mis a jour en place.
   perm/proj/pbuf : scratch N par thread. Renvoie les thetas bruts du
   niveau dans raw[] (2^level).                                          */
static void level_tree(const B2* b, uint64_t ts, int level,
                       uint32_t* paths, uint32_t* perm, float* proj,
                       float* pbuf, int64_t* offs, float* raw,
                       float* theta_out) {
    const int64_t n = b->n;
    const int64_t nk = 1ll << level;
    /* comptage + offsets */
    memset(offs, 0, (size_t)(nk + 1) * sizeof(int64_t));
    for (int64_t i = 0; i < n; i++) offs[paths[i] + 1]++;
    for (int64_t k = 0; k < nk; k++) offs[k + 1] += offs[k];
    {   /* permutation stable doc -> ordre par chemin */
        int64_t* cur = (int64_t*)malloc((size_t)nk * sizeof(int64_t));
        memcpy(cur, offs, (size_t)nk * sizeof(int64_t));
        for (int64_t i = 0; i < n; i++)
            perm[cur[paths[i]]++] = (uint32_t)i;
        free(cur);
    }
    int dims[256];
    float v0[256], v1[256], w[256];
    /* passe A : projections + medianes brutes */
    for (int64_t k = 0; k < nk; k++) {
        int64_t s = offs[k], e = offs[k + 1];
        if (s == e) { raw[k] = 0.0f; continue; }
        int32_t heap = (int32_t)((nk - 1) + k);
        pick_dims(ts, heap, b->dim, b->sub_dim, dims);
        gen_vec_v3(node_seed(ts, (int)(2 * heap)),     v0, b->sub_dim);
        gen_vec_v3(node_seed(ts, (int)(2 * heap) + 1), v1, b->sub_dim);
        for (int d = 0; d < b->sub_dim; d++) w[d] = v1[d] - v0[d];
        for (int64_t i = s; i < e; i++) {
            const float* x = b->base + (int64_t)perm[i] * b->dim;
            float acc = 0.0f;
            for (int d = 0; d < b->sub_dim; d++) acc += x[dims[d]] * w[d];
            proj[i] = acc;
        }
        memcpy(pbuf, proj + s, (size_t)(e - s) * sizeof(float));
        raw[k] = med_select(pbuf, e - s);
    }
    /* grille int8 du niveau */
    float lo = raw[0], hi = raw[0];
    for (int64_t k = 1; k < nk; k++) {
        if (raw[k] < lo) lo = raw[k];
        if (raw[k] > hi) hi = raw[k];
    }
    float span = hi - lo;
    /* passe B : snap + routage */
    for (int64_t k = 0; k < nk; k++) {
        int64_t s = offs[k], e = offs[k + 1];
        float th = raw[k];
        if (span > 0.0f)
            th = roundf((th - lo) / span * 254.0f) / 254.0f * span + lo;
        theta_out[(nk - 1) + k] = th;
        for (int64_t i = s; i < e; i++) {
            uint32_t doc = perm[i];
            int bit;
            if (proj[i] > th) bit = 1;
            else if (proj[i] < th) bit = 0;
            else bit = tie_bit(doc);
            paths[doc] = (uint32_t)(2 * k + bit);
        }
    }
}

static int write_slots(const B2* b, int tree, const uint32_t* paths,
                       uint32_t* perm, int64_t* offs, const char* out_dir,
                       uint8_t* slotbuf, int64_t* trunc) {
    const int64_t n = b->n;
    const int64_t nl = 1ll << b->depth;
    const int cap = (b->slot_bytes - 4) / 4;
    memset(offs, 0, (size_t)(nl + 1) * sizeof(int64_t));
    for (int64_t i = 0; i < n; i++) offs[paths[i] + 1]++;
    for (int64_t k = 0; k < nl; k++) offs[k + 1] += offs[k];
    {
        int64_t* cur = (int64_t*)malloc((size_t)nl * sizeof(int64_t));
        memcpy(cur, offs, (size_t)nl * sizeof(int64_t));
        for (int64_t i = 0; i < n; i++)
            perm[cur[paths[i]]++] = (uint32_t)i;
        free(cur);
    }
    memset(slotbuf, 0, (size_t)nl * b->slot_bytes);
    for (int64_t k = 0; k < nl; k++) {
        int64_t s = offs[k], e = offs[k + 1];
        int64_t cnt = e - s;
        if (cnt > cap) { *trunc += cnt - cap; cnt = cap; }
        uint8_t* p = slotbuf + k * b->slot_bytes;
        uint32_t c32 = (uint32_t)cnt;
        memcpy(p, &c32, 4);
        memcpy(p + 4, perm + s, (size_t)cnt * 4);
    }
    char path[1024];
    snprintf(path, sizeof(path), "%s/tree%05d.slt", out_dir, tree);
    FILE* f = fopen(path, "wb");
    if (!f) return -1;
    size_t wr = fwrite(slotbuf, 1, (size_t)nl * b->slot_bytes, f);
    fclose(f);
    return wr == (size_t)nl * b->slot_bytes ? 0 : -1;
}

int cmd_build2(int argc, char** argv) {
    if (argc < 6) {
        fprintf(stderr,
                "usage: rpforest build2 <base.fbin> <out_dir> <n_trees> "
                "<depth> [--sub_dim 16] [--dim 96] [--slot_bytes 512] "
                "[--group 32]\n");
        return 1;
    }
    const char* base_path = argv[2];
    const char* out_dir = argv[3];
    int n_trees = atoi(argv[4]);
    int depth = atoi(argv[5]);
    int sub_dim = 16, dim = 96, slot_bytes = 512, group = 32;
    for (int i = 6; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--sub_dim")) sub_dim = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--dim")) dim = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--slot_bytes"))
            slot_bytes = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--group")) group = atoi(argv[i + 1]);
    }
    if (depth < 1 || depth > 24) { fprintf(stderr, "depth 1..24\n"); return 1; }

    FILE* f = fopen(base_path, "rb");
    if (!f) { perror("base"); return 1; }
    uint32_t hdr[2];
    if (fread(hdr, 4, 2, f) != 2) { fclose(f); return 1; }
    int64_t n = hdr[0];
    if ((int)hdr[1] != dim) {
        fprintf(stderr, "dim header %u != %d\n", hdr[1], dim);
        fclose(f); return 1;
    }
    fprintf(stderr, "build2: n=%lld dim=%d depth=%d trees=%d slot=%d "
                    "group=%d\n", (long long)n, dim, depth, n_trees,
            slot_bytes, group);
    /* base en RAM EXPLICITE (pas de mmap — regle projet) */
    float* base = (float*)malloc((size_t)n * dim * sizeof(float));
    if (!base) { fprintf(stderr, "OOM base\n"); fclose(f); return 1; }
    for (int64_t off = 0; off < n; off += 2000000) {
        int64_t c = n - off < 2000000 ? n - off : 2000000;
        if (fread(base + off * dim, (size_t)dim * 4, (size_t)c, f)
            != (size_t)c) {
            fprintf(stderr, "read base\n"); return 1;
        }
    }
    fclose(f);
    fprintf(stderr, "build2: base chargee (%.1f Go)\n",
            (double)n * dim * 4 / 1e9);

    B2 b = {base, n, dim, sub_dim, depth, slot_bytes};
    int64_t med_nodes = (1ll << depth) - 1;
    float* med = (float*)calloc((size_t)n_trees * med_nodes,
                                sizeof(float));
    int64_t trunc_tot = 0;
    double t0 = (double)clock() / CLOCKS_PER_SEC;

    for (int g0 = 0; g0 < n_trees; g0 += group) {
        int G = n_trees - g0 < group ? n_trees - g0 : group;
        uint32_t* paths = (uint32_t*)calloc((size_t)G * n, 4);
        if (!paths) { fprintf(stderr, "OOM paths\n"); return 1; }
        int64_t trunc_g = 0;
        #pragma omp parallel reduction(+ : trunc_g)
        {
            uint32_t* perm = (uint32_t*)malloc((size_t)n * 4);
            float* proj = (float*)malloc((size_t)n * 4);
            float* pbuf = (float*)malloc((size_t)n * 4);
            int64_t* offs = (int64_t*)malloc(
                ((size_t)1 << depth) * 8 + 8);
            float* raw = (float*)malloc(((size_t)1 << depth) * 4);
            uint8_t* slotbuf = (uint8_t*)malloc(
                ((size_t)1 << depth) * slot_bytes);
            #pragma omp for schedule(dynamic)
            for (int j = 0; j < G; j++) {
                int t = g0 + j;
                uint64_t ts = tree_seed(t);
                uint32_t* pt = paths + (size_t)j * n;
                for (int level = 0; level < depth; level++)
                    level_tree(&b, ts, level, pt, perm, proj, pbuf,
                               offs, raw, med + (size_t)t * med_nodes);
                int64_t tr = 0;
                write_slots(&b, t, pt, perm, offs, out_dir, slotbuf, &tr);
                trunc_g += tr;
            }
            free(perm); free(proj); free(pbuf); free(offs);
            free(raw); free(slotbuf);
        }
        trunc_tot += trunc_g;
        fprintf(stderr, "build2: arbres %d..%d ok (%.0fs cpu)\n",
                g0, g0 + G - 1,
                (double)clock() / CLOCKS_PER_SEC - t0);
        free(paths);
    }

    char mpath[1024];
    snprintf(mpath, sizeof(mpath), "%s/medians.bin", out_dir);
    FILE* mf = fopen(mpath, "wb");
    if (!mf) { perror("medians"); return 1; }
    uint32_t mh[4] = {MED_MAGIC, (uint32_t)n_trees, (uint32_t)depth, 0};
    fwrite(mh, 4, 4, mf);
    fwrite(med, sizeof(float), (size_t)n_trees * med_nodes, mf);
    fclose(mf);
    fprintf(stderr,
            "build2: DONE — medians.bin %.1f Mo, tronques %lld "
            "(%.4f%%)\n",
            (double)n_trees * med_nodes * 4 / 1e6, (long long)trunc_tot,
            100.0 * (double)trunc_tot / ((double)n * n_trees));
    free(med); free(base);
    return 0;
}
