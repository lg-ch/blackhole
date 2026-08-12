#ifndef VEC_FORMAT_H
#define VEC_FORMAT_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <sys/types.h>

/* Supported vector file formats:
   - VECFMT_FVECS  : per-row [int32 dim][dim × float32], no global header
   - VECFMT_U8BIN  : 8 B header [uint32 n][uint32 dim], then n × (dim × uint8)
   - VECFMT_BVECS  : per-row [int32 dim][dim × uint8], no global header (SIFT 1B legacy)
   - VECFMT_FBIN   : 8 B header [uint32 n][uint32 dim], then n × (dim × float32)
                     (big-ann-benchmarks competition format, e.g. DEEP1B)
   - VECFMT_F16BIN : 8 B header [uint32 n][uint32 dim], then n × (dim × float16
                     IEEE 754 binary16) — le format natif des embeddings CLIP
                     LAION ; ÷2 sur le disque ET sur les octets lus au rerank
   Auto-detected by file extension via vec_fmt_from_path().                  */
typedef enum { VECFMT_FVECS = 0, VECFMT_U8BIN = 1,
               VECFMT_BVECS = 2, VECFMT_FBIN  = 3,
               VECFMT_F16BIN = 4 } VecFmt;

static inline VecFmt vec_fmt_from_path(const char* p) {
    size_t n = strlen(p);
    if (n >= 7 && strcmp(p + n - 7, ".f16bin") == 0) return VECFMT_F16BIN;
    if (n >= 6 && strcmp(p + n - 6, ".u8bin") == 0) return VECFMT_U8BIN;
    if (n >= 6 && strcmp(p + n - 6, ".bvecs") == 0) return VECFMT_BVECS;
    if (n >= 5 && strcmp(p + n - 5, ".fbin")  == 0) return VECFMT_FBIN;
    return VECFMT_FVECS;
}

/* Byte offset of the START of vector data for row idx (post-header). */
static inline off_t vec_row_offset(VecFmt fmt, int idx, int dim) {
    if (fmt == VECFMT_U8BIN)  return (off_t)8 + (off_t)idx * (off_t)dim;
    if (fmt == VECFMT_BVECS)  return (off_t)idx * ((off_t)4 + (off_t)dim) + 4;
    if (fmt == VECFMT_FBIN)   return (off_t)8 + (off_t)idx * (off_t)dim * 4;
    if (fmt == VECFMT_F16BIN) return (off_t)8 + (off_t)idx * (off_t)dim * 2;
    return (off_t)idx * ((off_t)4 + (off_t)dim * 4) + 4;
}

/* Bytes to read for one row's payload (excluding any per-row header). */
static inline size_t vec_row_bytes(VecFmt fmt, int dim) {
    if (fmt == VECFMT_U8BIN)  return (size_t)dim;
    if (fmt == VECFMT_BVECS)  return (size_t)dim;
    if (fmt == VECFMT_FBIN)   return (size_t)dim * 4;
    if (fmt == VECFMT_F16BIN) return (size_t)dim * 2;
    return (size_t)dim * 4;
}

/* Convert dim uint8 values to float32 (in-place output). */
static inline void vec_u8_to_f32(const uint8_t* in, float* out, int dim) {
    for (int i = 0; i < dim; i++) out[i] = (float)in[i];
}

/* IEEE 754 binary16 → binary32, portable (pas d'intrinsics F16C requis :
   les compilateurs vectorisent bien cette forme, et le chemin est de toute
   façon dominé par l'I/O). Gère dénormaux, ±inf, NaN. */
static inline float vec_f16_to_f32_one(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t man  = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;                               /* ±0 */
        } else {                                       /* dénormal */
            uint32_t e = 127 - 15 + 1;
            while (!(man & 0x400u)) { man <<= 1; e--; }
            man &= 0x3FFu;
            bits = sign | (e << 23) | (man << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);       /* inf / NaN */
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static inline void vec_f16_to_f32(const uint16_t* in, float* out, int dim) {
    for (int i = 0; i < dim; i++) out[i] = vec_f16_to_f32_one(in[i]);
}

#endif
