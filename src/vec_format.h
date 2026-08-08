#ifndef VEC_FORMAT_H
#define VEC_FORMAT_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <sys/types.h>

/* Supported vector file formats:
   - VECFMT_FVECS : per-row [int32 dim][dim × float32], no global header
   - VECFMT_U8BIN : 8 B header [uint32 n][uint32 dim], then n × (dim × uint8)
   - VECFMT_BVECS : per-row [int32 dim][dim × uint8], no global header (SIFT 1B legacy)
   - VECFMT_FBIN  : 8 B header [uint32 n][uint32 dim], then n × (dim × float32)
                    (big-ann-benchmarks competition format, e.g. DEEP1B)
   Auto-detected by file extension via vec_fmt_from_path().                  */
typedef enum { VECFMT_FVECS = 0, VECFMT_U8BIN = 1,
               VECFMT_BVECS = 2, VECFMT_FBIN  = 3 } VecFmt;

static inline VecFmt vec_fmt_from_path(const char* p) {
    size_t n = strlen(p);
    if (n >= 6 && strcmp(p + n - 6, ".u8bin") == 0) return VECFMT_U8BIN;
    if (n >= 6 && strcmp(p + n - 6, ".bvecs") == 0) return VECFMT_BVECS;
    if (n >= 5 && strcmp(p + n - 5, ".fbin")  == 0) return VECFMT_FBIN;
    return VECFMT_FVECS;
}

/* Byte offset of the START of vector data for row idx (post-header). */
static inline off_t vec_row_offset(VecFmt fmt, int idx, int dim) {
    if (fmt == VECFMT_U8BIN) return (off_t)8 + (off_t)idx * (off_t)dim;
    if (fmt == VECFMT_BVECS) return (off_t)idx * ((off_t)4 + (off_t)dim) + 4;
    if (fmt == VECFMT_FBIN)  return (off_t)8 + (off_t)idx * (off_t)dim * 4;
    return (off_t)idx * ((off_t)4 + (off_t)dim * 4) + 4;
}

/* Bytes to read for one row's payload (excluding any per-row header). */
static inline size_t vec_row_bytes(VecFmt fmt, int dim) {
    if (fmt == VECFMT_U8BIN) return (size_t)dim;
    if (fmt == VECFMT_BVECS) return (size_t)dim;
    if (fmt == VECFMT_FBIN)  return (size_t)dim * 4;
    return (size_t)dim * 4;
}

/* Convert dim uint8 values to float32 (in-place output). */
static inline void vec_u8_to_f32(const uint8_t* in, float* out, int dim) {
    for (int i = 0; i < dim; i++) out[i] = (float)in[i];
}

#endif
