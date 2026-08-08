/* TurboQuant-style 4-bit sidecar (.tq4) for two-stage rerank.
 *
 * Codes are data-oblivious up to a shared 1-D Lloyd-Max codebook
 * calibrated once at build: rotate with seeded (HD)^3 (sign flips +
 * fast Walsh-Hadamard, O(d log d)), then quantize each coordinate
 * with the shared codebook. dim is zero-padded to the next power of
 * two before rotation.
 *
 * File layout (little-endian):
 *   TqHeader, then n_docs rows of pad_dim/2 bytes (packed nibbles,
 *   low nibble = even coord).
 */
#ifndef TQUANT_H
#define TQUANT_H

#include <stdint.h>
#include <liburing.h>

#define TQ_MAGIC   0x34515400u   /* "\0TQ4" */
#define TQ_LEVELS  16            /* 4-bit */
#define TQ_ROUNDS  3

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t dim;          /* original vector dim */
    uint32_t pad_dim;      /* power-of-two rotation dim */
    uint64_t seed;
    uint64_t n_docs;
    float    centers[TQ_LEVELS];
    float    bounds[TQ_LEVELS - 1];
} TqHeader;

/* Build the sidecar by streaming an fvecs file.
   calib_rows vectors (strided across the file) calibrate the codebook.
   Returns 0 on success.                                                */
int tq_build(const char* fvecs_path, const char* out_path,
             int dim, uint64_t seed, int calib_rows);

/* Open / close a sidecar for query-time use. Returns NULL on error.   */
typedef struct TqReader TqReader;
TqReader* tq_open(const char* tq4_path);
void      tq_close(TqReader* r);

/* Two-stage rerank.
   1. read the packed codes of cand_ids (io_uring batch on `ring`,
      caller-owned), score with a per-query LUT, keep top `kprime`;
   2. caller passes survivors to the exact rerank.
   out_ids must hold kprime entries; returns #survivors or -1.         */
int tq_select(TqReader* r, struct io_uring* ring,
              const float* qvec,
              const int32_t* cand_ids, int n_cands,
              int kprime, int32_t* out_ids);

#endif
