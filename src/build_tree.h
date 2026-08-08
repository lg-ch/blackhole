#ifndef BUILD_TREE_H
#define BUILD_TREE_H

#include <stdint.h>
#include "vec_format.h"

/* On-disk pair: (leaf_id, doc_id) — 8 bytes. Max depth 30 (int32). */
typedef struct { int32_t leaf_id; int32_t doc_id; } Pair;

/* Stream-build n_trees tree files in index_dir/tree%05d.bin.
   Reads `vecs_path` starting at row `doc_offset`, for `n_vecs` rows.
   Stored doc_ids are doc_offset + i (i = 0..n_vecs-1) so the index has
   global doc_ids — required to combine multiple sub-indexes built on
   disjoint slices of a shared corpus into a multi-index family.         */
int build_forest(const char* vecs_path, int doc_offset, int n_vecs,
                 int dim, int sub_dim,
                 int n_trees, int depth, const char* index_dir);

/* Same as build_forest but the stored doc_ids start at `doc_id_base`
   instead of doc_offset. Use when the file slice [doc_offset, +n_vecs)
   represents a buffer of streamed docs whose global ids are
   [doc_id_base, doc_id_base + n_vecs).                                   */
int build_forest_ex(const char* vecs_path, int doc_offset, int n_vecs,
                    int doc_id_base, int dim, int sub_dim,
                    int n_trees, int depth, const char* index_dir);

/* Shift build seeds: tree file t is built with tree_seed(t + off).
   For growing an existing forest tree-by-tree (default 0).             */
void set_tree_seed_offset(int off);
/* Tree-batched build: total tree count in the final index (for medians.bin
   validation + global median-slice indexing). 0 = single-shot (== n_trees). */
void set_total_trees(int n);

/* --fast build mode : trade RAM for speed. Precomputes every inner-node
   hyperplane (v0, v1) once per tree at build start, then reads them from
   RAM instead of re-generating each node visit. Extra RAM :
     n_trees × (2^depth − 1) × 2 × sub_dim × 4 B
   ≈ 2 GB at d=16 / 256 trees / sub_dim=16 (grows ×2 per depth step).
   Default 0 (memory-bounded classic build).                             */
void set_build_fast_mode(int on);

/* --batch N : per-batch doc buffer size (default 256, or 4096 with --fast).
   Larger → fewer OMP team creations and fwrite() calls (1 per tree per
   batch instead of one per doc per tree). Cost : BATCH × (dim × 4 B +
   sizeof(Pair) × n_threads) extra RAM.                                    */
void set_build_batch(int b);

/* --calib-queries N [--calib-topk K --calib-interval SEC] :
   Enable online calibration (see calibration.h). Snapshots written to
   index_dir/calibration_{queries,gt}.bin at end of build ; also every
   `interval` seconds during build if >0 (0 = end-only).                   */
void set_build_calib(int n_queries, int top_k, double interval_s);

/* --recalib-doubling [--recalib-script PATH] :
   Every time indexed doc count crosses a doubling threshold (100k, 200k,
   400k, ...), pause the build, snapshot calibration GT, invoke the calibrator
   subprocess (default script :
     /home/chatelet/mangrove-search/scripts/mangrove_calibrate.py),
   then resume. Blocking pause : simple state machine, no concurrency issues.
   Producer WAL disk queue for live ingestion is out of scope here (build is
   file-batch), but the pattern is identical : producer writes to disk, build
   drains on resume. See [[autotune-doubling]]. */
void set_build_recalib_doubling(int enable, const char* script_path);

/* --no-varbyte : produce SRT V2 (raw uint32 doc_ids) instead of V3 (varbyte
   delta-compressed). Simpler decode, ~1.7-2× larger on disk. Useful for
   prototype/streaming where compression convert step is unwanted. Default 0
   (V3 compressed). */
void set_build_no_varbyte(int on);
int  get_build_no_varbyte(void);

/* --tail : block on fread short reads by polling (500 ms) up to 5 min per
   record. Enables WAL disk queue consumption : producer appends to the file,
   consumer reads sequentially and waits when caught up. */
void set_build_tail(int on);
int  get_build_tail(void);

/* Convert one pair file into a sorted posting-list file (tree%05d.srt).
   Two-pass counting sort: (1) count per leaf, (2) re-scan writing each
   doc_id into its leaf's slot. On success, deletes the source pair file.
   RAM ≈ n_leaves × 8 B + total_docs × 4 B per tree.                      */
int convert_tree_to_sorted(const char* pair_path, const char* sorted_path,
                           int depth);

/* Convert all n_trees pair files to sorted layout. */
int convert_all_to_sorted(const char* index_dir, int n_trees, int depth);

/* Compact (deepen) an existing segment from depth_old to depth_new without
   re-traversing from root: for each doc, read its current leaf_id from the
   old .srt, read its vector from base, run traverse_*_continue for the
   (depth_new - depth_old) extra levels, emit new pair, then convert. */
int compact_segment_deeper(const char* index_old, const char* index_new,
                           const char* base_path,
                           int n_trees, int depth_old, int depth_new,
                           int dim, int sub_dim);

/* Multi-source compaction: merge N old segments (all at same depth_old)
   into ONE new segment at depth_new. Each old segment contributes its docs
   via incremental traversal (D_new - D_old extra levels).                 */
int compact_multi_deeper(const char* index_new,
                         const char* const* index_olds, int n_olds,
                         const char* base_path,
                         int n_trees, int depth_old, int depth_new,
                         int dim, int sub_dim);

#endif

/* ---- Top-level medians ----
   Calibrate per-node median thresholds from the first `sample_n` vectors of
   `vecs_path` and write <index_dir>/medians.bin. Run BEFORE build_forest :
   the build routing and forest_open both pick the file up automatically. */
int calibrate_and_save_medians(const char* vecs_path, const char* index_dir,
                               int n_trees, int dim, int sub_dim,
                               int med_depth, int sample_n);
/* Load medians.bin ; returns malloc'd table (n_trees × (2^md - 1) floats)
   or NULL. */
float* medians_load(const char* path, int* out_n_trees, int* out_med_depth);
