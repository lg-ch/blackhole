/* TurboQuant 1-bit sidecar (.tq1) for two-stage rerank.
 *
 * Same data-oblivious pipeline as TQ4: rotate with seeded (HD)^3 (sign
 * flips + Walsh-Hadamard), then quantize each coordinate to a single
 * bit. Codebook is the 2-level Lloyd-Max optimum {c_neg, c_pos},
 * calibrated once on a sample. Code = pad_dim bits, packed
 * little-endian (bit i of byte b stores coord 8*b+i).
 *
 * 4x smaller than .tq4 on disk and per-I/O. Stage 1 recall per code is
 * lower → caller picks a larger kprime (≈ 500-2000 typical).
 */
#ifndef TQ1_H
#define TQ1_H

#include <stdint.h>
#include <liburing.h>

#define TQ1_MAGIC  0x31515400u   /* "\0TQ1" */
#define TQ1_ROUNDS 3

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t dim;
    uint32_t pad_dim;
    uint64_t seed;
    uint64_t n_docs;
    float    c_neg;
    float    c_pos;
} Tq1Header;

/* n_docs_cap : 0 = all docs in fvecs_path ; otherwise cap the sidecar to
 *              the first `n_docs_cap` rows. Useful for building a sidecar
 *              over a subset of a large fbin (e.g. 10M-doc sub-index).    */
int tq1_build(const char* fvecs_path, const char* out_path,
              int dim, uint64_t seed, int calib_rows,
              uint64_t n_docs_cap);

typedef struct Tq1Reader Tq1Reader;
Tq1Reader* tq1_open(const char* path);
void       tq1_close(Tq1Reader* r);

int tq1_select(Tq1Reader* r, struct io_uring* ring,
               const float* qvec,
               const int32_t* cand_ids, int n_cands,
               int kprime, int32_t* out_ids);

/* Same as tq1_select, but uses an io_uring registered buffer when provided
 * (fixed_buf != NULL && fixed_size ≥ n_cands × code_bytes). Eliminates the
 * kernel copy_to_user during read completion — typically -10..15 % query
 * latency on the rerank path. Falls back to tq1_select() transparently
 * when the registered buffer is too small.                                */
int tq1_select_fixed(Tq1Reader* r, struct io_uring* ring,
                     void* fixed_buf, size_t fixed_size, int fixed_buf_idx,
                     const float* qvec,
                     const int32_t* cand_ids, int n_cands,
                     int kprime, int32_t* out_ids);

#endif
