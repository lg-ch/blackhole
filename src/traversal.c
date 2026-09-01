#include "traversal.h"
#include "gen_vec.h"
#include <math.h>
#include <stddef.h>
#include <stdlib.h>   /* malloc/free/qsort — implicit decl is an error on gcc 15 */
#include <string.h>   /* memcpy — idem */

#ifdef __ARM_NEON
#include <arm_neon.h>
static float dot(const float* a, const float* b, int n) {
    float32x4_t s = vdupq_n_f32(0);
    int i = 0;
    for (; i + 3 < n; i += 4)
        s = vfmaq_f32(s, vld1q_f32(a + i), vld1q_f32(b + i));
    float r = vaddvq_f32(s);
    for (; i < n; i++) r += a[i] * b[i];
    return r;
}
#else
static float dot(const float* a, const float* b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}
#endif

static float dot_sub(const float* vec, const float* nvec,
                     const int* dims, int sub_dim) {
    float s = 0.0f;
    for (int i = 0; i < sub_dim; i++) s += vec[dims[i]] * nvec[i];
    return s;
}

uint64_t tree_seed(int tree_idx) {
    return (uint64_t)tree_idx * 99991ULL + 7ULL;
}

uint64_t node_seed(uint64_t ts, int bfs_idx) {
    uint64_t h = ts * 2654435761ULL ^ (uint64_t)bfs_idx * 40503ULL;
    return h ? h : 1;
}

/* Forward decls — definitions live further down with the node_perm block. */
static int g_node_perm;
static const int* compute_or_get_perm(uint64_t ts, int dim);
static inline int depth_of_node(int32_t node_id);

void pick_dims(uint64_t ts, int node_id, int full_dim, int sub_dim,
               int* out_dims) {
    /* Identity when full_dim <= sub_dim: sampling-with-replacement is
       degenerate (~38% duplicate indices at full_dim==sub_dim, splits collapse).
       Triggered by tree_sub == sub_dim — zero node-level RNG in that regime. */
    if (full_dim <= sub_dim) {
        for (int i = 0; i < sub_dim; i++) out_dims[i] = i % full_dim;
        return;
    }
    if (g_node_perm) {
        const int* p = compute_or_get_perm(ts, full_dim);
        int cycle_len = full_dim / sub_dim;
        if (cycle_len < 1) cycle_len = 1;
        int cycle_pos = depth_of_node(node_id) % cycle_len;
        int base = cycle_pos * sub_dim;
        for (int i = 0; i < sub_dim; i++) out_dims[i] = p[base + i];
        return;
    }
    uint64_t h = ts * 2654435761ULL ^ (uint64_t)(node_id * 3) * 40503ULL;
    for (int i = 0; i < sub_dim; i++) {
        h = h * 2654435761ULL + (uint64_t)i * 40503ULL;
        out_dims[i] = (int)(h % (uint64_t)full_dim);
    }
}

/* --- Per-tree input subspace (random-subspace ensemble) --------------------
   When g_tree_sub > 0, each tree restricts the input vector to a fixed random
   subset of g_tree_sub coordinates (derived from the tree seed) BEFORE the
   node-level sub_dim sampling. The whole forest still covers the full space
   because every tree draws a different subset. Gathering the subset into a
   contiguous buffer also turns the scattered dot_sub gather into a cache-hot
   contiguous one. Plain global (NOT __thread): set once before the OpenMP /
   query thread fan-out, read-only afterwards — so build workers see it too.
   tree_sub = 0 (default) disables the feature → byte-identical legacy path. */
static int g_tree_sub = 0;
void set_tree_sub(int k) { g_tree_sub = k < 0 ? 0 : k; }
int  get_tree_sub(void)  { return g_tree_sub; }

/* Number of DISTINCT per-tree subspaces. 0 (default) = one per tree (n_trees
   distinct). G > 0 = only G distinct subspaces, each shared by ~n_trees/G trees
   (a tree is hashed to a group from its seed). Lets us test whether fewer,
   better-spread subspaces beat many noisy independent draws. Plain global,
   set-once-before-fan-out like g_tree_sub. */
static int g_tree_sub_groups = 0;
void set_tree_sub_groups(int g) { g_tree_sub_groups = g < 0 ? 0 : g; }
int  get_tree_sub_groups(void)  { return g_tree_sub_groups; }

/* Map a tree seed to the seed that drives its input subspace. With grouping
   off → identity (subspace per tree). With G groups → bucket the tree into one
   of G groups (seed-derived) and return a fixed per-bucket seed, so all trees
   in a bucket share one subspace while keeping distinct hyperplanes (those
   still come from ts). Stateless → preserves the zero-data-RAM property. */
static uint64_t tree_sub_seed(uint64_t ts) {
    if (g_tree_sub_groups <= 0) return ts;
    uint64_t h = (ts ^ (ts >> 29)) * 0x9E3779B97F4A7C15ULL;
    uint64_t bucket = (h >> 33) % (uint64_t)g_tree_sub_groups;
    return bucket * 0x100000001B3ULL + 0x9E3779B1ULL;
}

/* Pick tree_sub coordinate indices for the given tree seed (sampling with
   replacement, like pick_dims — distinct salt so it doesn't track pick_dims). */
void pick_tree_dims(uint64_t ts, int full_dim, int tree_sub, int* out_dims) {
    uint64_t h = ts * 2654435761ULL ^ 0x9E3779B97F4A7C15ULL;
    for (int i = 0; i < tree_sub; i++) {
        h = h * 2654435761ULL + (uint64_t)(i * 7 + 3) * 40503ULL;
        out_dims[i] = (int)(h % (uint64_t)full_dim);
    }
}

/* --- Per-tree node-permutation dim selection -------------------------------
   When g_node_perm > 0: at level d of the tree, pick out_dims = perm[cycle*sub_dim
   .. (cycle+1)*sub_dim) where perm is a Fisher-Yates permutation of [0..full_dim-1]
   seeded by ts, and cycle = d mod (full_dim/sub_dim). Successive levels along
   any path see DISJOINT 16-dim blocks (max coverage per path); the cycle
   restarts every full_dim/sub_dim levels. Identical sibling nodes at the same
   depth see the same dims — diversity comes from gen_vec hyperplanes only.
   0 = off (default, byte-identical legacy).                                   */
/* Definition (forward-declared above so pick_dims can branch on it). */
void set_node_perm(int on) { g_node_perm = on ? 1 : 0; }
int  get_node_perm(void)   { return g_node_perm; }

/* Per-thread cache of recently-computed perms, indexed by ts mod slots. With
   the build OMP layout (parallel over trees, static slices), each thread sees
   ~n_trees/n_threads distinct ts → 16 slots is enough for full hit rate.
   On query the cache churns (~256 trees over 16 slots = ~16x recompute), but
   each recompute is O(dim) ≈ 4 us — negligible vs query latency.            */
#define NODE_PERM_CACHE_SZ 16
#define NODE_PERM_DIM_MAX  2048
typedef struct {
    uint64_t ts;
    int dim;
    int perm[NODE_PERM_DIM_MAX];
} NodePermEntry;
static __thread NodePermEntry g_perm_cache[NODE_PERM_CACHE_SZ];
static __thread int g_perm_cache_init = 0;

static const int* compute_or_get_perm(uint64_t ts, int dim) {
    if (!g_perm_cache_init) {
        for (int i = 0; i < NODE_PERM_CACHE_SZ; i++) g_perm_cache[i].ts = 0;
        g_perm_cache_init = 1;
    }
    int slot = (int)((ts * 0x9E3779B97F4A7C15ULL >> 32) % NODE_PERM_CACHE_SZ);
    NodePermEntry* e = &g_perm_cache[slot];
    if (e->ts == ts && e->dim == dim) return e->perm;
    if (dim > NODE_PERM_DIM_MAX) dim = NODE_PERM_DIM_MAX;
    for (int i = 0; i < dim; i++) e->perm[i] = i;
    uint64_t st = (ts | 1ULL) ^ 0xDEADBEEFCAFEBABEULL;
    for (int i = dim - 1; i > 0; i--) {
        st = st * 6364136223846793005ULL + 1442695040888963407ULL;
        int j = (int)((st >> 32) % (uint64_t)(i + 1));
        int t = e->perm[i]; e->perm[i] = e->perm[j]; e->perm[j] = t;
    }
    e->ts = ts; e->dim = dim;
    return e->perm;
}

static inline int depth_of_node(int32_t node_id) {
    /* depth = floor(log2(node_id + 1)) ; root (node_id=0) → 0 */
    uint32_t n = (uint32_t)(node_id + 1);
    int d = 0;
    while (n >>= 1) d++;
    return d;
}

/* ---- Top-level median thresholds (optional) ----
   When set, splits at heap nodes < g_med_limit use threshold θ[node] on the
   projection difference (c1 - c0) instead of 0, equalising the doc mass in
   the top of the tree. Heap numbering makes the level test implicit :
   node < 2^med_depth - 1  ⟺  level < med_depth.
   Thread-local : caller sets the CURRENT TREE's table before traversing it
   (mandatory in per-tree OMP loops). NULL = classic sign splits.           */
static __thread const float* g_med_table = NULL;
static __thread int32_t      g_med_limit = 0;
/* Mode MED2 int8 : codes u8 par noeud + echelles par niveau
   (scales = [lo x md][span x md] du MEME arbre). Decode a la volee :
   th = (code/254)*span[lvl] + lo[lvl] — bit-identique au f32 que build2
   ecrivait (memes operations flottantes). Cout : depth FMA par descente. */
static __thread const uint8_t* g_med8_table = NULL;
static __thread const float*   g_med8_scale = NULL;
static __thread int32_t        g_med8_limit = 0;
static __thread int            g_med8_md = 0;
void traversal_set_medians(const float* table, int med_depth) {
    g_med_table = table;
    g_med_limit = (table && med_depth > 0) ? ((1 << med_depth) - 1) : 0;
    g_med8_table = NULL; g_med8_scale = NULL;
    g_med8_limit = 0; g_med8_md = 0;
}
void traversal_set_medians8(const uint8_t* codes, const float* scales,
                            int med_depth) {
    g_med8_table = codes;
    g_med8_scale = scales;
    g_med8_limit = (codes && med_depth > 0) ? ((1 << med_depth) - 1) : 0;
    g_med8_md = med_depth;
    g_med_table = NULL; g_med_limit = 0;
}
static inline int depth_of_node_fwd(int32_t node_id) {
    uint32_t n = (uint32_t)(node_id + 1);
    int d = 0;
    while (n >>= 1) d++;
    return d;
}
static inline float med_th(int32_t node) {
    if (node < g_med8_limit) {
        int lvl = depth_of_node_fwd(node);
        return ((float)g_med8_table[node] / 254.0f)
               * g_med8_scale[g_med8_md + lvl] + g_med8_scale[lvl];
    }
    return (node < g_med_limit) ? g_med_table[node] : 0.0f;
}

int32_t traverse(const float* vec, int dim, int depth, uint64_t ts,
                 float* v0, float* v1) {
    int32_t node = 0;
    for (int level = 0; level < depth; level++) {
        gen_vec(node_seed(ts, node * 2),     v0, dim);
        gen_vec(node_seed(ts, node * 2 + 1), v1, dim);
        float mdiff = dot(vec, v1, dim) - dot(vec, v0, dim) - med_th(node);
        node = (mdiff > 0.0f) ? 2*node + 2 : 2*node + 1;
    }
    return node;
}

int32_t traverse_sub(const float* vec, int full_dim, int sub_dim, int depth,
                     uint64_t ts,
                     float* v0, float* v1, int* dims) {
    int active = (g_tree_sub > 0 && g_tree_sub < full_dim);
    int   td  [active ? g_tree_sub : 1];
    float vsub[active ? g_tree_sub : 1];
    if (active) {
        pick_tree_dims(tree_sub_seed(ts), full_dim, g_tree_sub, td);
        for (int i = 0; i < g_tree_sub; i++) vsub[i] = vec[td[i]];
        vec = vsub; full_dim = g_tree_sub;
    }
    int32_t node = 0;
    for (int level = 0; level < depth; level++) {
        pick_dims(ts, node, full_dim, sub_dim, dims);
        gen_vec(node_seed(ts, node * 2),     v0, sub_dim);
        gen_vec(node_seed(ts, node * 2 + 1), v1, sub_dim);
        float c0 = dot_sub(vec, v0, dims, sub_dim);
        float c1 = dot_sub(vec, v1, dims, sub_dim);
        node = (c1 - c0 > med_th(node)) ? 2*node + 2 : 2*node + 1;
    }
    return node;
}

/* Trace variant : returns the final leaf AND the per-level |c1 - c0| margins
   along the chosen path. `out_margins` must hold `depth` floats. Used to
   probe whether a given (query, tree) pair is well-routed: large margins
   along the path → query naturally belongs in that tree's leaf structure. */
int32_t traverse_sub_cached(const float* vec, int full_dim, int sub_dim,
                            int depth, uint64_t ts,
                            const float* cache_v0, const float* cache_v1,
                            int* dims) {
    int active = (g_tree_sub > 0 && g_tree_sub < full_dim);
    int   td  [active ? g_tree_sub : 1];
    float vsub[active ? g_tree_sub : 1];
    if (active) {
        pick_tree_dims(tree_sub_seed(ts), full_dim, g_tree_sub, td);
        for (int i = 0; i < g_tree_sub; i++) vsub[i] = vec[td[i]];
        vec = vsub; full_dim = g_tree_sub;
    }
    int32_t node = 0;
    for (int level = 0; level < depth; level++) {
        pick_dims(ts, node, full_dim, sub_dim, dims);
        const float* v0 = cache_v0 + (size_t)node * sub_dim;
        const float* v1 = cache_v1 + (size_t)node * sub_dim;
        float c0 = dot_sub(vec, v0, dims, sub_dim);
        float c1 = dot_sub(vec, v1, dims, sub_dim);
        node = (c1 - c0 > med_th(node)) ? 2*node + 2 : 2*node + 1;
    }
    return node;
}

int32_t traverse_sub_trace(const float* vec, int full_dim, int sub_dim, int depth,
                           uint64_t ts,
                           float* v0, float* v1, int* dims,
                           float* out_margins) {
    int active = (g_tree_sub > 0 && g_tree_sub < full_dim);
    int   td  [active ? g_tree_sub : 1];
    float vsub[active ? g_tree_sub : 1];
    if (active) {
        pick_tree_dims(tree_sub_seed(ts), full_dim, g_tree_sub, td);
        for (int i = 0; i < g_tree_sub; i++) vsub[i] = vec[td[i]];
        vec = vsub; full_dim = g_tree_sub;
    }
    int32_t node = 0;
    for (int level = 0; level < depth; level++) {
        pick_dims(ts, node, full_dim, sub_dim, dims);
        gen_vec(node_seed(ts, node * 2),     v0, sub_dim);
        gen_vec(node_seed(ts, node * 2 + 1), v1, sub_dim);
        float c0 = dot_sub(vec, v0, dims, sub_dim);
        float c1 = dot_sub(vec, v1, dims, sub_dim);
        float mdiff = c1 - c0 - med_th(node);
        out_margins[level] = fabsf(mdiff);
        node = (mdiff > 0.0f) ? 2*node + 2 : 2*node + 1;
    }
    return node;
}

int32_t leaf_base(int depth)  { return (1 << depth) - 1; }
int32_t leaf_count(int depth) { return 1 << depth; }

int32_t traverse_continue(const float* vec, int dim,
                          int start_depth, int n_extra,
                          int32_t start_node, uint64_t ts,
                          float* v0, float* v1) {
    int32_t node = start_node;
    for (int level = start_depth; level < start_depth + n_extra; level++) {
        gen_vec(node_seed(ts, node * 2),     v0, dim);
        gen_vec(node_seed(ts, node * 2 + 1), v1, dim);
        float mdiff = dot(vec, v1, dim) - dot(vec, v0, dim) - med_th(node);
        node = (mdiff > 0.0f) ? 2*node + 2 : 2*node + 1;
    }
    return node;
}

int32_t traverse_sub_continue(const float* vec, int full_dim, int sub_dim,
                              int start_depth, int n_extra,
                              int32_t start_node, uint64_t ts,
                              float* v0, float* v1, int* dims) {
    int active = (g_tree_sub > 0 && g_tree_sub < full_dim);
    int   td  [active ? g_tree_sub : 1];
    float vsub[active ? g_tree_sub : 1];
    if (active) {
        pick_tree_dims(tree_sub_seed(ts), full_dim, g_tree_sub, td);
        for (int i = 0; i < g_tree_sub; i++) vsub[i] = vec[td[i]];
        vec = vsub; full_dim = g_tree_sub;
    }
    int32_t node = start_node;
    for (int level = start_depth; level < start_depth + n_extra; level++) {
        pick_dims(ts, node, full_dim, sub_dim, dims);
        gen_vec(node_seed(ts, node * 2),     v0, sub_dim);
        gen_vec(node_seed(ts, node * 2 + 1), v1, sub_dim);
        float c0 = dot_sub(vec, v0, dims, sub_dim);
        float c1 = dot_sub(vec, v1, dims, sub_dim);
        node = (c1 - c0 > med_th(node)) ? 2*node + 2 : 2*node + 1;
    }
    return node;
}

/* Multi-probe (sub-dim mode): compute the primary leaf node plus up to
   n_probes neighbour leaf nodes. Neighbours are obtained by flipping the
   decision at the levels where the query is closest to the split (smallest
   |c1 - c0| margin) and re-traversing from there to full depth — because
   splits are path-dependent, a flipped bit needs a real re-descent.

   min_flip_level restricts which levels may be flipped: only levels in
   [min_flip_level, depth) are eligible. Flipping a SHALLOW level re-descends
   into a huge subtree (tens of thousands of docs at 1B), which floods the
   candidate set and drowns the vote signal — so for high probe counts we
   constrain flips to DEEP levels (min_flip_level = depth - span), where each
   flip yields a genuine neighbour leaf (~few docs). Pass 0 for the legacy
   "any level" behaviour.

   out_nodes receives absolute node ids: out_nodes[0] = primary, then the
   probes in increasing-margin order. Returns the number written (1+#probes).
   Caller scratch: v0/v1 length sub_dim, dims length sub_dim. depth <= 63.   */
/* Variant of traverse_sub_probes that also reports a per-path score :
   `out_scores[i]` = min margin along path i (smaller = more ambiguous decisions).
   Path 0 is the main path; path 1..count-1 are the probes (in selection order).
   For probes, the score combines the kept margins of the main path before the
   flip with the margins of the continuation past the flip — i.e. the true
   min margin of the actual leaf's path. Same complexity as the plain variant.
   out_scores must hold (n_probes + 1) floats.                                 */
int traverse_sub_probes_scored(const float* vec, int full_dim, int sub_dim, int depth,
                               uint64_t ts, float* v0, float* v1, int* dims,
                               int n_probes, int min_flip_level,
                               int32_t* out_nodes, float* out_scores) {
    int32_t other_child[64];
    float   margin[64];
    char    used[64];
    if (depth > 64) depth = 64;
    if (min_flip_level < 0) min_flip_level = 0;
    if (min_flip_level >= depth) min_flip_level = depth > 0 ? depth - 1 : 0;

    int active = (g_tree_sub > 0 && g_tree_sub < full_dim);
    int   td  [active ? g_tree_sub : 1];
    float vsub[active ? g_tree_sub : 1];
    if (active) {
        pick_tree_dims(tree_sub_seed(ts), full_dim, g_tree_sub, td);
        for (int i = 0; i < g_tree_sub; i++) vsub[i] = vec[td[i]];
        vec = vsub; full_dim = g_tree_sub;
    }

    int32_t node = 0;
    float min_main = 0.0f;
    for (int level = 0; level < depth; level++) {
        pick_dims(ts, node, full_dim, sub_dim, dims);
        gen_vec(node_seed(ts, node * 2),     v0, sub_dim);
        gen_vec(node_seed(ts, node * 2 + 1), v1, sub_dim);
        float c0 = dot_sub(vec, v0, dims, sub_dim);
        float c1 = dot_sub(vec, v1, dims, sub_dim);
        float mdiff = c1 - c0 - med_th(node);
        int32_t chosen, other;
        if (mdiff > 0.0f) { chosen = 2 * node + 2; other = 2 * node + 1; }
        else              { chosen = 2 * node + 1; other = 2 * node + 2; }
        other_child[level] = other;
        margin[level]      = fabsf(mdiff);
        used[level]        = 0;
        if (level == 0 || margin[level] < min_main) min_main = margin[level];
        node = chosen;
    }
    out_nodes[0]  = node;
    out_scores[0] = min_main;
    int count = 1;
    int eligible = depth - min_flip_level;
    if (n_probes > eligible) n_probes = eligible;

    /* Per-probe : flip the smallest-margin level among unused, retraverse the
       remainder of the tree from the sibling, and compute that path's min
       margin too. */
    float vp0[1024], vp1[1024]; int pdims[256];
    (void)vp0; (void)vp1; (void)pdims;  /* unused if depth small */
    for (int p = 0; p < n_probes; p++) {
        int best = -1; float bm = 0.0f;
        for (int level = min_flip_level; level < depth; level++) {
            if (used[level]) continue;
            if (best < 0 || margin[level] < bm) { best = level; bm = margin[level]; }
        }
        if (best < 0) break;
        used[best] = 1;

        /* min margin BEFORE the flip (levels 0..best-1, from the main path). */
        float min_kept = (best == 0) ? margin[best] : margin[0];
        for (int j = 1; j < best; j++)
            if (margin[j] < min_kept) min_kept = margin[j];
        if (best > 0 && margin[best] < min_kept) min_kept = margin[best];
        /* margin[best] is itself a "decision margin" the probe takes (flipped),
           so include it too. */

        /* Continuation from sibling, computing margins as we go. */
        int32_t cnode = other_child[best];
        float min_post = INFINITY;
        for (int level = best + 1; level < depth; level++) {
            pick_dims(ts, cnode, full_dim, sub_dim, dims);
            gen_vec(node_seed(ts, cnode * 2),     v0, sub_dim);
            gen_vec(node_seed(ts, cnode * 2 + 1), v1, sub_dim);
            float c0 = dot_sub(vec, v0, dims, sub_dim);
            float c1 = dot_sub(vec, v1, dims, sub_dim);
            float mm = fabsf(c1 - c0 - med_th(cnode));
            if (mm < min_post) min_post = mm;
            cnode = (c1 - c0 > med_th(cnode)) ? 2*cnode + 2 : 2*cnode + 1;
        }
        out_nodes[count]  = cnode;
        out_scores[count] = (min_post < min_kept) ? min_post : min_kept;
        count++;
    }
    return count;
}

/* ---- Générateur MULTI-FLIP (séquence de perturbation, style multi-probe
   LSH) ----------------------------------------------------------------
   Le single-flip plafonne à `depth − min_flip_level` probes distinctes par
   arbre (~16-32) : mesuré sur LAION-10M, au-delà les pools sont
   bit-identiques. Ici on énumère les ENSEMBLES de niveaux flippés S par
   coût croissant (somme des marges des niveaux flippés) via le tas
   extend/shift classique : {i} → {i, i+1} (extension) et {..i} → {..i+1}
   (décalage), sur les niveaux triés par marge croissante. Chaque probe
   re-descend réellement l'arbre (hyperplans par nœud, médianes armées),
   flips appliqués aux niveaux de S : géométrie exacte, masques distincts
   ⇒ feuilles distinctes. Score = min-marge du chemin réellement parcouru
   (compatible avec le classement global de pathrank).                    */
typedef struct { float cost; uint64_t mask; int last; } PSetEnt;

static inline void pset_push(PSetEnt* h, int* n, PSetEnt e) {
    int i = (*n)++;
    h[i] = e;
    while (i > 0) {
        int p = (i - 1) >> 1;
        if (h[p].cost <= h[i].cost) break;
        PSetEnt t = h[p]; h[p] = h[i]; h[i] = t;
        i = p;
    }
}

static inline PSetEnt pset_pop(PSetEnt* h, int* n) {
    PSetEnt top = h[0];
    h[0] = h[--(*n)];
    int i = 0;
    for (;;) {
        int l = 2 * i + 1, r = l + 1, m = i;
        if (l < *n && h[l].cost < h[m].cost) m = l;
        if (r < *n && h[r].cost < h[m].cost) m = r;
        if (m == i) break;
        PSetEnt t = h[m]; h[m] = h[i]; h[i] = t;
        i = m;
    }
    return top;
}

int traverse_sub_probes_multi_scored(const float* vec, int full_dim,
                                     int sub_dim, int depth, uint64_t ts,
                                     float* v0, float* v1, int* dims,
                                     int n_probes, int min_flip_level,
                                     int32_t* out_nodes, float* out_scores) {
    float margin[64];
    if (depth > 64) depth = 64;
    if (min_flip_level < 0) min_flip_level = 0;
    if (min_flip_level >= depth) min_flip_level = depth > 0 ? depth - 1 : 0;

    int active = (g_tree_sub > 0 && g_tree_sub < full_dim);
    int   td  [active ? g_tree_sub : 1];
    float vsub[active ? g_tree_sub : 1];
    if (active) {
        pick_tree_dims(tree_sub_seed(ts), full_dim, g_tree_sub, td);
        for (int i = 0; i < g_tree_sub; i++) vsub[i] = vec[td[i]];
        vec = vsub; full_dim = g_tree_sub;
    }

    /* Chemin principal : marges par niveau. */
    int32_t node = 0;
    float min_main = INFINITY;
    for (int level = 0; level < depth; level++) {
        pick_dims(ts, node, full_dim, sub_dim, dims);
        gen_vec(node_seed(ts, node * 2),     v0, sub_dim);
        gen_vec(node_seed(ts, node * 2 + 1), v1, sub_dim);
        float c0 = dot_sub(vec, v0, dims, sub_dim);
        float c1 = dot_sub(vec, v1, dims, sub_dim);
        float mdiff = c1 - c0 - med_th(node);
        margin[level] = fabsf(mdiff);
        if (margin[level] < min_main) min_main = margin[level];
        node = (mdiff > 0.0f) ? 2 * node + 2 : 2 * node + 1;
    }
    out_nodes[0]  = node;
    out_scores[0] = min_main;
    int count = 1;

    /* Niveaux éligibles triés par marge croissante (tri par insertion :
       ≤ 64 éléments). Cap à 63 pour tenir dans un masque uint64. */
    int   ord[64];
    float cst[64];
    int ne = 0;
    for (int level = min_flip_level; level < depth && ne < 63; level++) {
        int j = ne++;
        while (j > 0 && cst[j - 1] > margin[level]) {
            ord[j] = ord[j - 1]; cst[j] = cst[j - 1]; j--;
        }
        ord[j] = level; cst[j] = margin[level];
    }
    if (ne == 0 || n_probes <= 0) return count;

    int cap = 2 * n_probes + 8;
    PSetEnt* heap = (PSetEnt*)malloc((size_t)cap * sizeof(PSetEnt));
    if (!heap) return count;
    int hn = 0;
    pset_push(heap, &hn, (PSetEnt){cst[0], 1ull, 0});

    while (hn > 0 && count < n_probes + 1) {
        PSetEnt e = pset_pop(heap, &hn);
        /* Masque par NIVEAU depuis le masque par index trié. */
        uint64_t lvl_mask = 0;
        for (int i = 0; i <= e.last; i++)
            if (e.mask & (1ull << i)) lvl_mask |= 1ull << ord[i];
        /* Re-descente exacte avec flips. */
        int32_t pn = 0;
        float minm = INFINITY;
        for (int level = 0; level < depth; level++) {
            pick_dims(ts, pn, full_dim, sub_dim, dims);
            gen_vec(node_seed(ts, pn * 2),     v0, sub_dim);
            gen_vec(node_seed(ts, pn * 2 + 1), v1, sub_dim);
            float c0 = dot_sub(vec, v0, dims, sub_dim);
            float c1 = dot_sub(vec, v1, dims, sub_dim);
            float mdiff = c1 - c0 - med_th(pn);
            float am = fabsf(mdiff);
            if (am < minm) minm = am;
            int go_right = (mdiff > 0.0f);
            if (lvl_mask & (1ull << level)) go_right = !go_right;
            pn = go_right ? 2 * pn + 2 : 2 * pn + 1;
        }
        out_nodes[count]  = pn;
        out_scores[count] = minm;
        count++;
        /* Successeurs extend / shift. */
        if (e.last + 1 < ne && hn + 2 <= cap) {
            pset_push(heap, &hn, (PSetEnt){
                e.cost + cst[e.last + 1],
                e.mask | (1ull << (e.last + 1)), e.last + 1});
            pset_push(heap, &hn, (PSetEnt){
                e.cost - cst[e.last] + cst[e.last + 1],
                (e.mask ^ (1ull << e.last)) | (1ull << (e.last + 1)),
                e.last + 1});
        }
    }
    free(heap);
    return count;
}

int traverse_sub_probes(const float* vec, int full_dim, int sub_dim, int depth,
                        uint64_t ts, float* v0, float* v1, int* dims,
                        int n_probes, int min_flip_level, int32_t* out_nodes) {
    int32_t other_child[64];
    float   margin[64];
    char    used[64];
    if (depth > 64) depth = 64;
    if (min_flip_level < 0) min_flip_level = 0;
    if (min_flip_level >= depth) min_flip_level = depth > 0 ? depth - 1 : 0;

    int active = (g_tree_sub > 0 && g_tree_sub < full_dim);
    int   td  [active ? g_tree_sub : 1];
    float vsub[active ? g_tree_sub : 1];
    if (active) {
        pick_tree_dims(tree_sub_seed(ts), full_dim, g_tree_sub, td);
        for (int i = 0; i < g_tree_sub; i++) vsub[i] = vec[td[i]];
        vec = vsub; full_dim = g_tree_sub;   /* continue calls inherit this */
    }

    int32_t node = 0;
    for (int level = 0; level < depth; level++) {
        pick_dims(ts, node, full_dim, sub_dim, dims);
        gen_vec(node_seed(ts, node * 2),     v0, sub_dim);
        gen_vec(node_seed(ts, node * 2 + 1), v1, sub_dim);
        float c0 = dot_sub(vec, v0, dims, sub_dim);
        float c1 = dot_sub(vec, v1, dims, sub_dim);
        float mdiff = c1 - c0 - med_th(node);
        int32_t chosen, other;
        if (mdiff > 0.0f) { chosen = 2 * node + 2; other = 2 * node + 1; }
        else              { chosen = 2 * node + 1; other = 2 * node + 2; }
        other_child[level] = other;
        margin[level]      = fabsf(mdiff);
        used[level]        = 0;
        node = chosen;
    }
    out_nodes[0] = node;
    int count = 1;
    int eligible = depth - min_flip_level;          /* # flippable levels */
    if (n_probes > eligible) n_probes = eligible;
    for (int p = 0; p < n_probes; p++) {
        int best = -1; float bm = 0.0f;
        for (int level = min_flip_level; level < depth; level++) {
            if (used[level]) continue;
            if (best < 0 || margin[level] < bm) { best = level; bm = margin[level]; }
        }
        if (best < 0) break;
        used[best] = 1;
        out_nodes[count++] = traverse_sub_continue(
            vec, full_dim, sub_dim, best + 1, depth - best - 1,
            other_child[best], ts, v0, v1, dims);
    }
    return count;
}

/* ---- Median calibration (see traversal.h) ---- */
static int med_cmp_float(const void* a, const void* b) {
    float x = *(const float*)a, y = *(const float*)b;
    return (x > y) - (x < y);
}

int traversal_calibrate_medians(const float* sample, int n_sample,
                                int full_dim, int sub_dim, int n_trees,
                                int med_depth, float* out_table) {
    if (!sample || n_sample <= 0 || med_depth <= 0 || !out_table) return -1;
    int med_nodes = (1 << med_depth) - 1;
    int use_sub = (sub_dim > 0 && sub_dim <= full_dim);
    int sd = use_sub ? sub_dim : full_dim;

    int rc = 0;
    #pragma omp parallel for schedule(dynamic)
    for (int t = 0; t < n_trees; t++) {
        uint64_t ts = tree_seed(t);
        float* table = out_table + (size_t)t * med_nodes;
        int32_t* node = (int32_t*)malloc((size_t)n_sample * sizeof(int32_t));
        float*   diff = (float*)malloc((size_t)n_sample * sizeof(float));
        float*   v0   = (float*)malloc((size_t)sd * sizeof(float));
        float*   v1   = (float*)malloc((size_t)sd * sizeof(float));
        int*     dims = (int*)malloc((size_t)sd * sizeof(int));
        /* scratch for per-bucket grouping at the widest level */
        int      max_buckets = 1 << (med_depth - 1);
        int*     bcnt = (int*)malloc((size_t)max_buckets * sizeof(int));
        int*     boff = (int*)malloc((size_t)max_buckets * sizeof(int));
        float*   bval = (float*)malloc((size_t)n_sample * sizeof(float));
        if (!node || !diff || !v0 || !v1 || !dims || !bcnt || !boff || !bval) {
            rc = -1;
            goto cleanup;
        }
        for (int i = 0; i < n_sample; i++) node[i] = 0;

        for (int level = 0; level < med_depth; level++) {
            int level_base = (1 << level) - 1;
            int n_lv_nodes = 1 << level;
            /* Projection diff of each sample doc at its current node. */
            for (int i = 0; i < n_sample; i++) {
                const float* vec = sample + (size_t)i * full_dim;
                int32_t nd = node[i];
                float c0, c1;
                if (use_sub) {
                    pick_dims(ts, nd, full_dim, sub_dim, dims);
                    gen_vec(node_seed(ts, nd * 2),     v0, sub_dim);
                    gen_vec(node_seed(ts, nd * 2 + 1), v1, sub_dim);
                    c0 = dot_sub(vec, v0, dims, sub_dim);
                    c1 = dot_sub(vec, v1, dims, sub_dim);
                } else {
                    gen_vec(node_seed(ts, nd * 2),     v0, full_dim);
                    gen_vec(node_seed(ts, nd * 2 + 1), v1, full_dim);
                    c0 = dot(vec, v0, full_dim);
                    c1 = dot(vec, v1, full_dim);
                }
                diff[i] = c1 - c0;
            }
            /* Group diffs by node bucket, median per bucket. */
            memset(bcnt, 0, (size_t)n_lv_nodes * sizeof(int));
            for (int i = 0; i < n_sample; i++)
                bcnt[node[i] - level_base]++;
            boff[0] = 0;
            for (int b = 1; b < n_lv_nodes; b++) boff[b] = boff[b-1] + bcnt[b-1];
            {
                int* cur = (int*)malloc((size_t)n_lv_nodes * sizeof(int));
                if (!cur) { rc = -1; goto cleanup; }
                memcpy(cur, boff, (size_t)n_lv_nodes * sizeof(int));
                for (int i = 0; i < n_sample; i++)
                    bval[cur[node[i] - level_base]++] = diff[i];
                free(cur);
            }
            for (int b = 0; b < n_lv_nodes; b++) {
                float th = 0.0f;
                if (bcnt[b] >= 8) {   /* too few samples → keep sign split */
                    float* v = bval + boff[b];
                    qsort(v, (size_t)bcnt[b], sizeof(float), med_cmp_float);
                    th = (bcnt[b] & 1) ? v[bcnt[b]/2]
                                       : 0.5f*(v[bcnt[b]/2 - 1] + v[bcnt[b]/2]);
                }
                table[level_base + b] = th;
            }
            /* Route docs one level down with the fresh thresholds. */
            for (int i = 0; i < n_sample; i++) {
                int32_t nd = node[i];
                float th = table[nd];
                node[i] = (diff[i] > th) ? 2*nd + 2 : 2*nd + 1;
            }
        }
cleanup:
        free(node); free(diff); free(v0); free(v1); free(dims);
        free(bcnt); free(boff); free(bval);
    }
    return rc;
}
