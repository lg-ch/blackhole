#ifndef TRAVERSAL_H
#define TRAVERSAL_H

#include <stdint.h>

uint64_t tree_seed(int tree_idx);
uint64_t node_seed(uint64_t ts, int bfs_idx);

/* Full-dim traversal. v0 and v1 are scratch buffers of size dim.        */
int32_t traverse(const float* vec, int dim, int depth, uint64_t ts,
                 float* scratch_v0, float* scratch_v1);

/* Sub-dim traversal: at each node, pick sub_dim random coords (derived from
   the node's seed) and run gen_vec/dot on those coords only.
   scratch_v0, scratch_v1: size sub_dim. dims_buf: int[sub_dim].          */
int32_t traverse_sub(const float* vec, int full_dim, int sub_dim, int depth,
                     uint64_t ts,
                     float* scratch_v0, float* scratch_v1, int* dims_buf);

/* Pick sub_dim coordinate indices for the given (tree_seed, node_id).    */
void pick_dims(uint64_t ts, int node_id, int full_dim, int sub_dim,
               int* out_dims);

/* Per-tree input subspace (random-subspace ensemble). set_tree_sub(k) makes
   every tree restrict its input vector to k seed-derived coords before the
   node-level sub_dim sampling; 0 = disabled (default, legacy path). Set ONCE
   before the build / query thread fan-out (plain global, read-only after).  */
void set_tree_sub(int k);
int  get_tree_sub(void);
/* Number of distinct per-tree subspaces (0 = one per tree). G>0 → only G
   subspaces, hashed across the trees (~n_trees/G trees share each). */
void set_tree_sub_groups(int g);
int  get_tree_sub_groups(void);
void pick_tree_dims(uint64_t ts, int full_dim, int tree_sub, int* out_dims);

/* Per-tree path-permuted dim selection: at each level d of a tree, pick the
   d-th block of sub_dim coords from a tree-seeded permutation of full_dim.
   Successive levels of a path see disjoint dims (max coverage) ; cycle of
   `full_dim / sub_dim` levels before the perm restarts. 0 = off (default).  */
void set_node_perm(int on);
int  get_node_perm(void);

int32_t leaf_base(int depth);
int32_t leaf_count(int depth);

/* Cached sub-dim traversal: reads pre-computed hyperplanes from
   cache_v0 / cache_v1 (row-major, node_bfs × sub_dim) instead of calling
   gen_vec per node visit. pick_dims is still called (cheap). Used by
   --fast build mode.                                                    */
int32_t traverse_sub_cached(const float* vec, int full_dim, int sub_dim,
                            int depth, uint64_t ts,
                            const float* cache_v0, const float* cache_v1,
                            int* dims);

/* Continue traversal from an existing leaf node (at start_depth) for
   `n_extra` more levels. Returns the final node id at depth start_depth+n_extra.
   start_node is the GLOBAL node id (i.e. value evolved from 0 down to depth,
   = leaf_id_stored_in_srt + leaf_base(start_depth)).                        */
int32_t traverse_continue    (const float* vec, int dim,
                              int start_depth, int n_extra,
                              int32_t start_node, uint64_t ts,
                              float* v0, float* v1);
int32_t traverse_sub_continue(const float* vec, int full_dim, int sub_dim,
                              int start_depth, int n_extra,
                              int32_t start_node, uint64_t ts,
                              float* v0, float* v1, int* dims);

/* Trace variant: same as traverse_sub but also exports the per-level decision
   margins |c1 - c0| along the chosen path. `out_margins` must hold `depth`
   floats. Useful for per-tree quality scoring of a given query (without
   touching the sparse_index / leaves data — pure forest traversal). */
int32_t traverse_sub_trace(const float* vec, int full_dim, int sub_dim, int depth,
                           uint64_t ts,
                           float* v0, float* v1, int* dims,
                           float* out_margins);

/* Multi-probe (sub-dim): primary leaf + up to n_probes neighbour leaves,
   chosen by flipping the smallest-margin decisions and re-traversing.
   Only levels in [min_flip_level, depth) are flippable (deep-only probing
   avoids flooding from shallow flips; pass 0 for any-level legacy behaviour).
   out_nodes[0..count-1] = absolute node ids. Returns count (1+#probes). */
/* Same as traverse_sub_probes but also writes a per-path "min margin" score
   into out_scores (size n_probes+1). Score = smallest margin along the actual
   path leading to that leaf (combining main-path margins before flip + sibling
   continuation margins after flip). Larger score = clearer decisions = path
   landed in a denser leaf (empirically correlated with GT hit rate).         */
/* Multi-flip (séquence de perturbation) : dépasse le plafond single-flip
   de depth probes/arbre en énumérant les ensembles de flips par coût
   croissant. Mêmes conventions de sortie que la variante scored.        */
int traverse_sub_probes_multi_scored(const float* vec, int full_dim,
                                     int sub_dim, int depth, uint64_t ts,
                                     float* v0, float* v1, int* dims,
                                     int n_probes, int min_flip_level,
                                     int32_t* out_nodes, float* out_scores);

int traverse_sub_probes_scored(const float* vec, int full_dim, int sub_dim, int depth,
                               uint64_t ts, float* v0, float* v1, int* dims,
                               int n_probes, int min_flip_level,
                               int32_t* out_nodes, float* out_scores);

int traverse_sub_probes(const float* vec, int full_dim, int sub_dim, int depth,
                        uint64_t ts, float* v0, float* v1, int* dims,
                        int n_probes, int min_flip_level, int32_t* out_nodes);

#endif

/* ---- Top-level median thresholds ----
   Optional per-node thresholds θ on the split projection (c1 - c0) for heap
   nodes < 2^med_depth - 1, equalising doc mass in the tree top. Thread-local
   current-tree table : set before traversing EACH tree (OMP loops included).
   NULL disables (classic sign splits).                                      */
void traversal_set_medians(const float* table, int med_depth);
/* Mode MED2 int8 : codes u8 (2^md - 1 par arbre) + scales du meme arbre
   ([lo x md][span x md]). Exclusif du mode f32 (chaque set efface l'autre). */
void traversal_set_medians8(const uint8_t* codes, const float* scales,
                            int med_depth);

/* Calibrate per-node medians for all trees from an in-RAM sample.
   sample = n_sample × full_dim floats. out_table = n_trees × (2^med_depth - 1)
   floats (caller-allocated). Level-by-level exact medians of the sample
   (= converged value of an online estimator). OMP over trees.
   Assumes tree_sub == 0. Returns 0 on success.                              */
int traversal_calibrate_medians(const float* sample, int n_sample,
                                int full_dim, int sub_dim, int n_trees,
                                int med_depth, float* out_table);
