#ifndef CALIBRATION_H
#define CALIBRATION_H

#include <stdint.h>

/* Online calibration : during index build (streaming), maintain the true
   top-K nearest neighbors of a small pool of query docs (self-excluded).
   Snapshots are periodically written to disk as `calibration_queries.bin`
   and `calibration_gt.bin` in the index dir. These are consumed by the
   Python calibrator to sweep query params under strict cgroup 1G + cold
   conditions and recommend production settings without user GT.

   Design points :
     - Query pool is FILLED with the first N docs streamed. Simple, no
       reservoir sampling (statistically biased to early data but fine
       for stationary corpora).
     - Each subsequent doc's L2 distance is computed to every query
       (SIMD when available), and each query's max-heap of top-K is
       updated if the new dist is smaller than the current worst.
     - Self-exclusion : a query never gets its own doc_id in the GT.
     - Cost per batch ~ K × batch_n × dim × 4 flops (few ms, <1% of
       batch build time at K=50, batch_n=8192, dim=128).                 */

int  calib_init(int n_queries, int top_k, int dim);
void calib_update(const float* vec_batch, const int32_t* doc_ids, int batch_n);
int  calib_snapshot(const char* index_dir);
void calib_free(void);
int  calib_is_enabled(void);

#endif
