#include "gen_vec.h"
#include <math.h>

void gen_vec_v0(uint64_t seed, float* out, int dim) {
    uint64_t st = seed ? seed : 1;
    float ns = 0.0f;
    for (int i = 0; i < dim; i++) {
        float s = 0.0f;
        for (int j = 0; j < 4; j++) {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            s += (float)(st & 0xFFFFFF) / (float)0xFFFFFF - 0.5f;
        }
        out[i] = s;
        ns += s * s;
    }
    float inv = 1.0f / sqrtf(ns + 1e-10f);
    for (int i = 0; i < dim; i++) out[i] *= inv;
}

/* v3: 1 xorshift produces 2 floats (high 24 bits + low 24 bits).
   Distribution uniform [-0.5, +0.5] per coord, then normalized.        */
void gen_vec_v3(uint64_t seed, float* out, int dim) {
    uint64_t st = seed ? seed : 1;
    float ns = 0.0f;
    int i = 0;
    for (; i + 1 < dim; i += 2) {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        float a = (float)(st & 0xFFFFFF)       / (float)0xFFFFFF - 0.5f;
        float b = (float)((st >> 24) & 0xFFFFFF)/ (float)0xFFFFFF - 0.5f;
        out[i]   = a;
        out[i+1] = b;
        ns += a*a + b*b;
    }
    /* tail dim if odd */
    if (i < dim) {
        st ^= st << 13; st ^= st >> 7; st ^= st << 17;
        float a = (float)(st & 0xFFFFFF) / (float)0xFFFFFF - 0.5f;
        out[i] = a;
        ns += a * a;
    }
    float inv = 1.0f / sqrtf(ns + 1e-10f);
    for (int k = 0; k < dim; k++) out[k] *= inv;
}

gen_vec_fn gen_vec = gen_vec_v0;
static int g_gen_version = 0;

void set_gen_version(int v) {
    if (v == 3) { gen_vec = gen_vec_v3; g_gen_version = 3; }
    else        { gen_vec = gen_vec_v0; g_gen_version = 0; }
}
int get_gen_version(void) { return g_gen_version; }

void normalize(float* v, int dim) {
    float ns = 0.0f;
    for (int i = 0; i < dim; i++) ns += v[i] * v[i];
    float inv = 1.0f / sqrtf(ns + 1e-10f);
    for (int i = 0; i < dim; i++) v[i] *= inv;
}
