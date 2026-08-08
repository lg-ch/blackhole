#ifndef RECALL_H
#define RECALL_H

#include "query_tree.h"
#include "vec_format.h"

/* Read row `idx` of a vecs file via pread into out (dim floats).
   Handles both fvecs and u8bin formats. Returns 0 on success.            */
int read_vec_at(int fd, VecFmt fmt, int idx, int dim, float* out);

/* Legacy fvecs-only reader (kept for back-compat).                       */
int read_fvec(int fd, int idx, int dim, float* out);

/* Read row `idx` of an ivecs file via pread. Caller passes len.          */
int read_ivec(int fd, int idx, int len, int* out);

/* Squared L2 distance. */
float l2sq(const float* a, const float* b, int n);

/* Partial-sort top_k candidates by ascending L2 distance to qvec.
   Reads candidate vectors from base_fd (fmt = fvecs or u8bin).           */
int rerank_l2(int base_fd, VecFmt base_fmt, int dim, const float* qvec,
              const int32_t* cand_ids, int n_cands,
              int top_k, int32_t* out_ids);

/* io_uring-batched variant. Same semantics as rerank_l2. */
struct io_uring;
int rerank_l2_uring(struct io_uring* ring,
                    int base_fd, VecFmt base_fmt, int dim, const float* qvec,
                    const int32_t* cand_ids, int n_cands,
                    int top_k, int32_t* out_ids);

/* eval_recall / eval_recall_topn auto-detect formats from the file paths
   (fvecs_path / base_path) via their extensions.                          */
int eval_recall(Forest* f,
                const char* qvecs_path,
                const char* gt_path,
                const char* base_path,
                int n_queries, int top_k, int threshold,
                int max_candidates,
                double* out_recall, double* out_avg_cands,
                double* out_ms_per_query);

int eval_recall_topn(Forest* f,
                     const char* qvecs_path,
                     const char* gt_path,
                     const char* base_path,
                     int n_queries, int top_k, int top_n_cands,
                     int query_depth,
                     const roaring_bitmap_t* allowed,
                     double* out_recall, double* out_avg_cands,
                     double* out_ms_per_query);

#endif
