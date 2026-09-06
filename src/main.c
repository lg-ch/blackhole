#define _POSIX_C_SOURCE 200809L
#include "gen_vec.h"
#include "traversal.h"
#include "build_tree.h"
#include "tquant.h"
#include "tq1.h"
#include "query_tree.h"
#include "recall.h"
#include "vec_format.h"
#include "croaring_io.h"
#include "srt_hash.h"

#include <roaring/roaring.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <time.h>
#include <dirent.h>
#include "sorted_store.h"
#include <fcntl.h>
#include <unistd.h>

/* ---- CLI flag parser ----
   Pulls out `--sub_dim N` and `--gen v0|v3` from argv, leaving positional
   args in place. Returns the index just past the last positional arg.    */
static int g_cli_sub_dim     = 0;
static int g_cli_tree_sub    = 0;   /* per-tree input subspace (0 = off) */
static int g_cli_tree_sub_groups = 0; /* # distinct subspaces (0 = per-tree) */
static int g_cli_node_perm   = 0;   /* path-permuted dim selection (0 = off) */
static int g_cli_gen_ver     = 0;
static int g_cli_query_depth = 0;   /* 0 = use build depth */
static int g_cli_doc_offset  = 0;   /* first doc row to read       */
static int g_cli_doc_count   = 0;   /* 0 = read all rows after offset */
static int g_cli_doc_id_base = -1;  /* base for stored doc_id; -1 = use doc_offset */
static int g_cli_dim         = 0;   /* 0 = default per cmd (128 SIFT) */
static int g_cli_tree_offset = 0;   /* build seeds = tree_seed(t + off) */
static int g_cli_total_trees = 0;   /* tree-batched build: total trees in final
                                       index (meta + medians). 0 = single-shot */
static const char* g_cli_filter_path     = NULL;  /* int32 doc_ids file    */
static const char* g_cli_filter_ch_path  = NULL;  /* CH groupBitmap state  */
static int g_cli_auto_qd = 0;       /* derive query_depth from filter size */
static int g_cli_auto_qd_v2 = 0;    /* probe-based qd from n_distinct telemetry */
static double g_cli_target_ratio = 0.4;   /* target frac of corpus visited     */
static int g_cli_probe_n = 5;       /* #queries used to estimate n_distinct  */
static int g_cli_no_filter_comp = 0;  /* disable filter-sparseness compensation */
static int g_cli_filter_comp_mode = 1;  /* 1 = floor(log2(N/F))-1, 2 = floor(log2(N/F)/2) */
static int g_cli_fast_build = 0;    /* --fast : trade RAM for speed at build time
                                       (precompute hyperplane cache, ~n_trees ×
                                        2^depth × 2 × sub_dim × 4 B extra RAM).  */
static int g_cli_build_batch_size = 0;  /* --batch N : per-batch doc count. 0 =
                                           auto (256 classic / 4096 with --fast). */
static int    g_cli_calib_queries  = 0;    /* --calib-queries N : 0 = off */
static int    g_cli_calib_topk     = 10;
static double g_cli_calib_interval = 0.0;  /* --calib-interval SEC : 0 = end-only */
static int         g_cli_recalib_doubling = 0;    /* --recalib-doubling : 0 = off */
static const char* g_cli_recalib_script   = NULL; /* --recalib-script PATH : NULL = default */
static int         g_cli_no_varbyte       = 0;    /* --no-varbyte : convert to SRT V2 */
static int         g_cli_tail              = 0;    /* --tail : WAL tail-follow mode */
static void parse_flags(int* pargc, char*** pargv) {
    int argc = *pargc;
    char** argv = *pargv;
    int w = 0;
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--sub_dim") == 0 && i + 1 < argc) {
            g_cli_sub_dim = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--tree_sub") == 0 && i + 1 < argc) {
            g_cli_tree_sub = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--tree_sub_groups") == 0 && i + 1 < argc) {
            g_cli_tree_sub_groups = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--node_perm") == 0) {
            g_cli_node_perm = 1;
        } else if (strcmp(argv[i], "--fast") == 0) {
            g_cli_fast_build = 1;
        } else if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
            g_cli_build_batch_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--calib-queries") == 0 && i + 1 < argc) {
            g_cli_calib_queries = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--calib-topk") == 0 && i + 1 < argc) {
            g_cli_calib_topk = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--calib-interval") == 0 && i + 1 < argc) {
            g_cli_calib_interval = atof(argv[++i]);
        } else if (strcmp(argv[i], "--recalib-doubling") == 0) {
            g_cli_recalib_doubling = 1;
        } else if (strcmp(argv[i], "--recalib-script") == 0 && i + 1 < argc) {
            g_cli_recalib_script = argv[++i];
        } else if (strcmp(argv[i], "--no-varbyte") == 0) {
            g_cli_no_varbyte = 1;
        } else if (strcmp(argv[i], "--tail") == 0) {
            g_cli_tail = 1;
        } else if (strcmp(argv[i], "--gen") == 0 && i + 1 < argc) {
            const char* v = argv[++i];
            g_cli_gen_ver = (strcmp(v, "v3") == 0 || strcmp(v, "3") == 0) ? 3 : 0;
        } else if (strcmp(argv[i], "--query_depth") == 0 && i + 1 < argc) {
            g_cli_query_depth = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--doc_offset") == 0 && i + 1 < argc) {
            g_cli_doc_offset = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--doc_count") == 0 && i + 1 < argc) {
            g_cli_doc_count = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--doc_id_base") == 0 && i + 1 < argc) {
            g_cli_doc_id_base = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--dim") == 0 && i + 1 < argc) {
            g_cli_dim = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--tree_offset") == 0 && i + 1 < argc) {
            g_cli_tree_offset = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--total_trees") == 0 && i + 1 < argc) {
            g_cli_total_trees = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--filter") == 0 && i + 1 < argc) {
            g_cli_filter_path = argv[++i];
        } else if (strcmp(argv[i], "--filter_ch") == 0 && i + 1 < argc) {
            g_cli_filter_ch_path = argv[++i];
        } else if (strcmp(argv[i], "--auto_qd") == 0) {
            g_cli_auto_qd = 1;
        } else if (strcmp(argv[i], "--auto_qd_v2") == 0) {
            g_cli_auto_qd_v2 = 1;
        } else if (strcmp(argv[i], "--target_ratio") == 0 && i + 1 < argc) {
            g_cli_target_ratio = atof(argv[++i]);
        } else if (strcmp(argv[i], "--probe_n") == 0 && i + 1 < argc) {
            g_cli_probe_n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--no_filter_compensate") == 0) {
            g_cli_no_filter_comp = 1;
        } else if (strcmp(argv[i], "--filter_comp_mode") == 0 && i + 1 < argc) {
            g_cli_filter_comp_mode = atoi(argv[++i]);
        } else {
            argv[w++] = argv[i];
        }
    }
    *pargc = w;
}

/* Resolve the active filter source : either `--filter <int32_ids.bin>` or
   `--filter_ch <ch_state.bin>`. Returns NULL if neither was provided.
   On return, *n_filter_out = number of ids in the bitmap (0 if no filter).
   Caller frees with roaring_bitmap_free.                                */
static roaring_bitmap_t* load_filter_any(int n_corpus, int* n_filter_out) {
    if (n_filter_out) *n_filter_out = 0;
    if (g_cli_filter_ch_path) {
        return roaring_load_ch_state_file(g_cli_filter_ch_path, n_filter_out);
    }
    if (g_cli_filter_path) {
        return roaring_load_int32_file(g_cli_filter_path, n_corpus, n_filter_out);
    }
    return NULL;
}

#include <math.h>

/* Per-query probe : average n_distinct over the first `probe` queries
   at a given qd.                                                          */
static double probe_avg_distinct(Forest* f, int qd,
                                 int qfd, VecFmt qfmt, int dim,
                                 int top_n, const roaring_bitmap_t* allowed,
                                 int probe) {
    float*    qvec = (float*)   malloc((size_t)dim * sizeof(float));
    int32_t*  ids  = (int32_t*) malloc((size_t)top_n * sizeof(int32_t));
    int32_t*  vot  = (int32_t*) malloc((size_t)top_n * sizeof(int32_t));
    long sum = 0; int n = 0;
    for (int q = 0; q < probe; q++) {
        if (read_vec_at(qfd, qfmt, q, dim, qvec) != 0) break;
        forest_collect_topn(f, qvec, top_n, qd, allowed, ids, vot);
        sum += forest_get_last_n_distinct();
        n++;
    }
    free(qvec); free(ids); free(vot);
    return n > 0 ? (double)sum / n : 0.0;
}

/* auto_qd_v2 : drive qd from observed n_distinct candidates in the merge.
   Two-probe calibration : measure `n_distinct` at qd=build_depth AND
   qd=build_depth-2, derive the actual per-level expansion factor
   (theoretical ~2× but reality drops to ~1.5-1.7× due to inter-tree
   deduplication saturation), then solve for qd_target :
       qd_target = build_depth − ceil(log_f(target / probe_native))
   Target = `target_ratio × N_corpus` (filter-agnostic, see comment below).*/
static int auto_query_depth_v2(Forest* f, int build_depth,
                               int qfd, VecFmt qfmt, int dim,
                               int top_n, const roaring_bitmap_t* allowed,
                               int n_pool, int filter_card,
                               int* out_avg_distinct) {
    int probe = g_cli_probe_n;
    if (probe < 1) probe = 1;

    double probe0 = probe_avg_distinct(f, build_depth,
                                       qfd, qfmt, dim, top_n, allowed, probe);
    int probe_qd2 = build_depth - 2;
    if (probe_qd2 < 1) probe_qd2 = 1;
    double probe2 = probe_avg_distinct(f, probe_qd2,
                                       qfd, qfmt, dim, top_n, allowed, probe);
    if (out_avg_distinct) *out_avg_distinct = (int)probe0;

    /* Target depends on the n_distinct semantics. Since the K-way merge is
       filter-aware (cursor_seek_allowed), `n_distinct` counts ONLY docs
       that pass the filter — so the target must be in the same unit :
       `ratio × filter_card` when filtered, `ratio × N_corpus` otherwise.
       Mixing the two crashes : a filter-agnostic target paired with a
       filter-aware probe sends qd into the floor.                       */
    double pool = (filter_card > 0) ? (double)filter_card : (double)n_pool;
    double target = g_cli_target_ratio * pool;
    if (probe0 <= 0.0 || target <= 0.0) return build_depth;

    double factor_per_level = 1.7;   /* fallback */
    if (probe2 > probe0 && probe_qd2 < build_depth) {
        factor_per_level = pow(probe2 / probe0,
                               1.0 / (double)(build_depth - probe_qd2));
        if (factor_per_level < 1.05) factor_per_level = 1.05;  /* avoid div0 */
    }
    int levels = (int)ceil(log(target / probe0) / log(factor_per_level));
    if (levels < 0) levels = 0;

    /* Sub-corpus compensation : the forest's hyperplanes were built on the
       full corpus, so they're sub-optimal at routing NNs that live in an
       arbitrary subset. Visiting `ratio × filter_card` (the base target) is
       the right effort budget but the recall absolute ceiling drops with
       density. Empirical fix (validated 2026-05-18) : descend
       `floor(log2(N_corpus / filter_card)) - 1` extra levels to overshoot
       the visited fraction and compensate for the structural blind spots.
       Disabled with --no_filter_compensate.                                */
    int extra = 0;
    if (filter_card > 0 && filter_card < n_pool && !g_cli_no_filter_comp) {
        double ratio_corpus = (double)n_pool / (double)filter_card;
        double lg = log2(ratio_corpus);
        if (g_cli_filter_comp_mode == 2) {
            /* Lighter compensation : half the levels. Lower latency, slightly
               lower recall ceiling. Worth it when bruteforce isn't available
               but full compensation is too slow.                            */
            extra = (int)floor(lg / 2.0);
        } else {
            /* Default : floor(log2(N/F)) - 1. Aggressive recall recovery.   */
            extra = (int)floor(lg) - 1;
        }
        if (extra < 0) extra = 0;
    }
    int qd = build_depth - levels - extra;
    if (qd < 1) qd = 1;
    if (qd > build_depth) qd = build_depth;
    fprintf(stderr,
        "auto_qd_v2: probe_native=%.0f probe_-2=%.0f factor/lvl=%.2f "
        "corpus=%d filter=%d target=%.0f base_levels=%d extra=%d -> qd=%d\n",
        probe0, probe2, factor_per_level, n_pool, filter_card,
        target, levels, extra, qd);
    return qd;
}

/* Heuristic: pick query_depth so the EXPECTED filter-matches per query is
   ~top_n (so the rerank gets enough candidates after filtering). Per
   super-leaf at qd, expected matches = (n_filter / n_corpus) * 2^(build_depth - qd).
   Over n_trees trees: matches_per_query ≈ n_trees * leaf_size_at_qd * (n_filter / n_corpus).
   Bound qd above by build_depth (no point going deeper than native at very
   high filter density).                                                  */
static int auto_query_depth(int build_depth, int n_corpus, int n_filter) {
    if (n_filter <= 0 || n_filter >= n_corpus) return build_depth;
    /* Target: 1 match per super-leaf (× ~1024 trees = ~1024 matches total). */
    double ratio = (double)n_corpus / (double)n_filter;
    int log2_ratio = 0;
    while ((1L << log2_ratio) < (long)ratio && log2_ratio < 32) log2_ratio++;
    int qd = build_depth - log2_ratio;
    if (qd < 1) qd = 1;
    if (qd > build_depth) qd = build_depth;
    return qd;
}

/* ---- index/meta.txt ---- */
typedef struct {
    int n_trees, depth, dim, sub_dim, gen_version, n_docs, tree_sub,
        tree_sub_groups, node_perm;
} Meta;
static void meta_default(Meta* m) {
    m->n_trees = 0; m->depth = 0; m->dim = 128;
    m->sub_dim = 0; m->gen_version = 0; m->n_docs = 0; m->tree_sub = 0;
    m->tree_sub_groups = 0; m->node_perm = 0;
}
static int meta_write(const char* dir, const Meta* m) {
    char p[512]; snprintf(p, sizeof(p), "%s/meta.txt", dir);
    FILE* f = fopen(p, "w"); if (!f) return -1;
    fprintf(f, "n_trees %d\ndepth %d\ndim %d\nsub_dim %d\n"
               "gen_version %d\nn_docs %d\ntree_sub %d\ntree_sub_groups %d\n"
               "node_perm %d\n",
            m->n_trees, m->depth, m->dim, m->sub_dim,
            m->gen_version, m->n_docs, m->tree_sub, m->tree_sub_groups,
            m->node_perm);
    fclose(f); return 0;
}
static int meta_read(const char* dir, Meta* m) {
    char p[512]; snprintf(p, sizeof(p), "%s/meta.txt", dir);
    FILE* f = fopen(p, "r"); if (!f) return -1;
    char key[64]; int val;
    while (fscanf(f, "%63s %d", key, &val) == 2) {
        if      (!strcmp(key, "n_trees"))     m->n_trees     = val;
        else if (!strcmp(key, "depth"))       m->depth       = val;
        else if (!strcmp(key, "dim"))         m->dim         = val;
        else if (!strcmp(key, "sub_dim"))     m->sub_dim     = val;
        else if (!strcmp(key, "gen_version")) m->gen_version = val;
        else if (!strcmp(key, "n_docs"))      m->n_docs      = val;
        else if (!strcmp(key, "tree_sub"))    m->tree_sub    = val;
        else if (!strcmp(key, "tree_sub_groups")) m->tree_sub_groups = val;
        else if (!strcmp(key, "node_perm"))   m->node_perm   = val;
    }
    fclose(f);
    /* Apply the per-tree subspace globally here (not at each query site) so an
       index built with tree_sub != 0 can never be queried with the global
       unset — that mismatch silently routes to the wrong leaves (recall ~0),
       the same silent-KO class as a depth/gen mismatch. */
    set_tree_sub(m->tree_sub);
    set_tree_sub_groups(m->tree_sub_groups);
    set_node_perm(m->node_perm);
    return 0;
}

static long peak_rss_kb(void) {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_maxrss;
}

/* Read the current anonymous (private) and file-backed RSS in KB from
   /proc/self/status. Returns 0 on success.                              */
static int read_proc_rss(long* anon, long* file) {
    *anon = 0; *file = 0;
    FILE* f = fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        long v;
        if (sscanf(line, "RssAnon: %ld kB", &v) == 1) *anon = v;
        else if (sscanf(line, "RssFile: %ld kB", &v) == 1) *file = v;
    }
    fclose(f);
    return 0;
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int count_vecs(const char* path, int dim_expected) {
    VecFmt fmt = vec_fmt_from_path(path);
    struct stat st;
    if (stat(path, &st) != 0) { perror("stat"); return -1; }

    if (fmt == VECFMT_U8BIN || fmt == VECFMT_FBIN || fmt == VECFMT_F16BIN) {
        FILE* f = fopen(path, "rb");
        if (!f) { perror("fopen header"); return -1; }
        uint32_t hdr[2];
        if (fread(hdr, 4, 2, f) != 2) { fclose(f); return -1; }
        fclose(f);
        if ((int)hdr[1] != dim_expected) {
            fprintf(stderr, "dim %u != expected %d\n", hdr[1], dim_expected);
            return -1;
        }
        long elem = (fmt == VECFMT_U8BIN) ? 1L
                  : (fmt == VECFMT_F16BIN) ? 2L : 4L;
        long expected = 8L + (long)hdr[0] * dim_expected * elem;
        if (st.st_size > expected) {
            fprintf(stderr, "size %ld > expected %ld — trailing garbage?\n",
                    (long)st.st_size, expected);
            return -1;
        }
        if (st.st_size < expected) {
            /* Partial file (download in progress / on purpose subset).
               Return the count derivable from on-disk bytes; caller can
               clamp further via --doc_count.                          */
            long n_actual = (st.st_size - 8) / (dim_expected * elem);
            fprintf(stderr,
                "  note: partial file — header says %u, on-disk %ld vecs\n",
                hdr[0], n_actual);
            return (int)n_actual;
        }
        return (int)hdr[0];
    }

    long row = (fmt == VECFMT_BVECS) ? (4 + (long)dim_expected)
                                     : (4 + (long)dim_expected * 4);
    if (st.st_size % row != 0) {
        fprintf(stderr, "%s size %ld not divisible by row %ld\n",
                (fmt == VECFMT_BVECS) ? "bvecs" : "fvecs",
                (long)st.st_size, row);
        return -1;
    }
    return (int)(st.st_size / row);
}

static int usage(void) {
    fprintf(stderr,
        "Usage:\n"
        "  rpforest build  <fvecs> <index_dir> <n_trees> <depth>\n"
        "                  [--sub_dim N] [--gen v0|v3]\n"
        "  rpforest query  <qvecs> <index_dir> <n_trees> <depth> <n_docs>\n"
        "                  [top_k=10] [threshold=2] [n_queries=10]\n"
        "  rpforest recall <qvecs> <gt_ivecs> <base_fvecs> <index_dir>\n"
        "                  <n_trees> <depth> <n_docs>\n"
        "                  [top_k=10] [threshold=2] [n_queries=1000]\n"
        "                  [max_cands=4096]\n"
        "  rpforest topn   <qvecs> <gt_ivecs> <base_fvecs> <index_dir>\n"
        "                  <n_trees> <depth> <n_docs>\n"
        "                  [top_k=10] [top_n=500] [n_queries=1000]\n"
        "  rpforest search <qvecs> <base_fvecs> <index_dir>\n"
        "                  <n_trees> <depth> <n_docs>\n"
        "                  [top_k=10] [top_n=500] [n_queries=10]\n"
        "                  [--filter <int32_ids> | --filter_ch <ch_state>]\n"
        "  rpforest verify <index_dir> [n_trees]\n"
        "  rpforest delete <index_dir> <doc_id> [doc_id ...]\n"
        "  rpforest compact <new_index> <base_path> <new_depth> <old_index> [old ...]\n"
        "  rpforest stats <index_dir>\n"
        "  rpforest health <index_dir>\n"
        "\n"
        "  --sub_dim N : subsample N dims per node (0 = full dim, default).\n"
        "  --gen vN    : v0 = 4 xor/dim quasi-gaussian; v3 = 0.5 xor/dim uniform.\n"
        "  --dim N     : query/search vector dim (default 128).\n"
        "  --filter    : roaring filter from raw int32 doc_ids file.\n"
        "  --filter_ch : roaring filter from ClickHouse groupBitmap state.\n"
        "  Build writes index_dir/meta.txt; query/topn/recall/search load it back.\n");
    return 2;
}

static int cmd_build(int argc, char** argv) {
    if (argc < 6) return usage();
    const char* fvecs = argv[2];
    const char* idir  = argv[3];
    int n_trees = atoi(argv[4]);
    int depth   = atoi(argv[5]);
    const int dim = (g_cli_dim > 0) ? g_cli_dim : 128;
    int sub_dim = g_cli_sub_dim;
    int gen_ver = g_cli_gen_ver;
    set_gen_version(gen_ver);
    set_tree_sub(g_cli_tree_sub);
    set_tree_sub_groups(g_cli_tree_sub_groups);
    set_node_perm(g_cli_node_perm);

    mkdir(idir, 0755);

    int n_total = count_vecs(fvecs, dim);
    if (n_total < 0) return 1;

    int doc_offset = g_cli_doc_offset;
    int n_vecs     = g_cli_doc_count > 0 ? g_cli_doc_count : (n_total - doc_offset);
    /* En --tail, le file peut grandir pendant le build → skip la vérif de fin.
       Le build waite les records manquants (cf. build_tree.c). */
    int upper_check = g_cli_tail ? 0 : (doc_offset + n_vecs > n_total);
    if (doc_offset < 0 || (n_total > 0 && doc_offset >= n_total) ||
        n_vecs <= 0 || upper_check) {
        fprintf(stderr, "invalid --doc_offset/--doc_count: %d/%d (total=%d, tail=%d)\n",
                doc_offset, n_vecs, n_total, g_cli_tail);
        return 1;
    }

    fprintf(stderr,
        "Build RP forest:\n"
        "  fvecs       : %s  (%d vectors × %d dim)\n"
        "  slice       : [%d, %d)  (%d vectors indexed)\n"
        "  index       : %s\n"
        "  n_trees     : %d\n"
        "  depth       : %d  (%d leaves per tree)\n"
        "  sub_dim     : %d  (0 = full dim)\n"
        "  tree_sub    : %d  (0 = full dim per tree)\n"
        "  tree_sub_grp: %d  (0 = one subspace per tree)\n"
        "  node_perm   : %d  (0 = legacy with-replacement pick_dims)\n"
        "  gen_version : v%d\n"
        "  on-disk     : ~%.2f GB total\n\n",
        fvecs, n_total, dim, doc_offset, doc_offset + n_vecs, n_vecs,
        idir, n_trees, depth, 1 << depth,
        sub_dim, g_cli_tree_sub, g_cli_tree_sub_groups, g_cli_node_perm, gen_ver,
        (double)n_vecs * n_trees * 8.0 / (1024.0 * 1024.0 * 1024.0));

    int doc_id_base = (g_cli_doc_id_base >= 0) ? g_cli_doc_id_base : doc_offset;

    if (g_cli_tree_offset > 0) {
        set_tree_seed_offset(g_cli_tree_offset);
        fprintf(stderr, "  tree_offset : %d (seeds for trees %d..%d)\n",
                g_cli_tree_offset, g_cli_tree_offset,
                g_cli_tree_offset + n_trees - 1);
    }
    /* Tree-batched build: this invocation writes trees [tree_offset,
       tree_offset+n_trees), but meta.txt + medians must reflect the FULL
       index. --total_trees carries that count.                             */
    int meta_n_trees = (g_cli_total_trees > 0) ? g_cli_total_trees : n_trees;
    if (g_cli_total_trees > 0) {
        set_total_trees(g_cli_total_trees);
        fprintf(stderr, "  total_trees : %d (this batch builds %d)\n",
                g_cli_total_trees, n_trees);
    }

    double t0 = now_s();
    set_build_fast_mode(g_cli_fast_build);
    set_build_batch(g_cli_build_batch_size);
    set_build_calib(g_cli_calib_queries, g_cli_calib_topk, g_cli_calib_interval);
    set_build_recalib_doubling(g_cli_recalib_doubling, g_cli_recalib_script);
    set_build_no_varbyte(g_cli_no_varbyte);
    set_build_tail(g_cli_tail);

    /* Write meta.txt EARLY (params already known) so calibration subprocess
       fired during Phase 1 can read the index shape (even if trees aren't
       queryable yet). n_docs = final target ; may be > current indexed count
       during Phase 1 but the reader tolerates that.                        */
    Meta m = { meta_n_trees, depth, dim, sub_dim, gen_ver,
               doc_id_base + n_vecs, g_cli_tree_sub, g_cli_tree_sub_groups,
               g_cli_node_perm };
    meta_write(idir, &m);

    fprintf(stderr, "[1/2] streaming build ...\n");
    if (build_forest_ex(fvecs, doc_offset, n_vecs, doc_id_base, dim, sub_dim,
                        n_trees, depth, idir) != 0)
        return 1;

    fprintf(stderr, "[2/2] converting to sorted layout ...\n");
    if (convert_all_to_sorted(idir, n_trees, depth) != 0) return 1;

    /* Rewrite meta at end (no-op if unchanged) — ensures durable final state. */
    meta_write(idir, &m);

    /* Post-Phase-2 final calibration : now the index is queryable, run the
       real auto-tune to write recommended_config.json. Only if recalib is
       enabled (same flag that triggered mid-build attempts). */
    if (g_cli_recalib_doubling) {
        char cmd[2048];
        const char* script = g_cli_recalib_script
            ? g_cli_recalib_script
            : "/home/chatelet/mangrove-search/scripts/mangrove_calibrate.py";
        snprintf(cmd, sizeof(cmd),
                 "python3 -u %s --index %s --n-docs %d 2>&1",
                 script, idir, doc_id_base + n_vecs);
        fprintf(stderr, "[3/3] final calibration → recommended_config.json\n");
        double rc0 = now_s();
        int rc = system(cmd);
        fprintf(stderr, "  final calibrate rc=%d in %.1fs\n", rc, now_s() - rc0);
    }

    fprintf(stderr,
        "\nDONE in %.2fs   peak RSS = %ld KB (%.2f MB)\n",
        now_s() - t0, peak_rss_kb(), peak_rss_kb() / 1024.0);
    return 0;
}

static int cmd_tquant(int argc, char** argv) {
    if (argc < 4) return usage();
    const char* fvecs = argv[2];
    const char* out   = argv[3];
    const int dim = (g_cli_dim > 0) ? g_cli_dim : 128;
    uint64_t seed = 4242;
    int calib = 20000;
    fprintf(stderr, "TurboQuant sidecar: %s -> %s (dim %d, seed %llu)\n",
            fvecs, out, dim, (unsigned long long)seed);
    double t0 = now_s();
    if (tq_build(fvecs, out, dim, seed, calib) != 0) return 1;
    fprintf(stderr, "DONE in %.2fs\n", now_s() - t0);
    return 0;
}

static int cmd_tquant1(int argc, char** argv) {
    if (argc < 4) return usage();
    const char* fvecs = argv[2];
    const char* out   = argv[3];
    const int dim = (g_cli_dim > 0) ? g_cli_dim : 128;
    uint64_t seed = 4242;
    int calib = 20000;
    fprintf(stderr, "TurboQuant1 sidecar: %s -> %s (dim %d, seed %llu)\n",
            fvecs, out, dim, (unsigned long long)seed);
    double t0 = now_s();
    uint64_t n_docs_cap = (g_cli_doc_count > 0) ? (uint64_t)g_cli_doc_count : 0;
    if (tq1_build(fvecs, out, dim, seed, calib, n_docs_cap) != 0) return 1;
    fprintf(stderr, "DONE in %.2fs\n", now_s() - t0);
    return 0;
}

static int cmd_query(int argc, char** argv) {
    if (argc < 7) return usage();
    const char* qvecs = argv[2];
    const char* idir  = argv[3];
    int n_trees = atoi(argv[4]);
    int depth   = atoi(argv[5]);
    int n_docs  = atoi(argv[6]);
    int top_k     = (argc > 7) ? atoi(argv[7]) : 10;
    int threshold = (argc > 8) ? atoi(argv[8]) : 2;
    int n_q       = (argc > 9) ? atoi(argv[9]) : 10;
    const int dim = (g_cli_dim > 0) ? g_cli_dim : 128;

    Meta m; meta_default(&m);
    if (meta_read(idir, &m) == 0) {
        /* meta overrides params that MUST match the on-disk layout. */
        set_gen_version(m.gen_version);
        /* sub_dim taken from meta below in forest_open. n_trees/depth/n_docs
           come from CLI so the user can query a subset of trees/docs.     */
    } else {
        set_gen_version(0);
    }

    Forest f;
    if (forest_open(&f, idir, n_trees, dim, m.sub_dim, depth, n_docs) != 0) return 1;

    FILE* fq = fopen(qvecs, "rb");
    if (!fq) { perror("open qvecs"); forest_close(&f); return 1; }

    float* q = (float*)malloc((size_t)dim * sizeof(float));
    uint16_t* votes = (uint16_t*)malloc((size_t)n_docs * sizeof(uint16_t));
    Result* out = (Result*)malloc((size_t)top_k * sizeof(Result));

    fprintf(stderr,
        "Query: %d queries, top_k=%d, threshold=%d  (RSS now %ld KB)\n",
        n_q, top_k, threshold, peak_rss_kb());

    double t0 = now_s();
    int dread;
    for (int i = 0; i < n_q; i++) {
        if (fread(&dread, 4, 1, fq) != 1) break;
        if (dread != dim) { fprintf(stderr, "bad dim %d\n", dread); break; }
        if (fread(q, 4, dim, fq) != (size_t)dim) break;

        int n_res = forest_search(&f, q, top_k, threshold, out, votes);
        printf("Q%-3d  hits=%d", i, n_res);
        for (int k = 0; k < (n_res < 5 ? n_res : 5); k++)
            printf("  [%d:%d]", out[k].doc_id, out[k].votes);
        printf("\n");
    }
    double dt = now_s() - t0;
    fprintf(stderr,
        "Query done: %.3fs (%.1f ms/query),  peak RSS %ld KB (%.2f MB)\n",
        dt, dt * 1000.0 / (n_q > 0 ? n_q : 1),
        peak_rss_kb(), peak_rss_kb() / 1024.0);

    free(q); free(votes); free(out);
    fclose(fq);
    forest_close(&f);
    return 0;
}

static int cmd_recall(int argc, char** argv) {
    if (argc < 9) return usage();
    const char* qvecs = argv[2];
    const char* gt    = argv[3];
    const char* base  = argv[4];
    const char* idir  = argv[5];
    int n_trees = atoi(argv[6]);
    int depth   = atoi(argv[7]);
    int n_docs  = atoi(argv[8]);
    int top_k     = (argc > 9)  ? atoi(argv[9])  : 10;
    int threshold = (argc > 10) ? atoi(argv[10]) : 2;
    int n_q       = (argc > 11) ? atoi(argv[11]) : 1000;
    int max_cands = (argc > 12) ? atoi(argv[12]) : 4096;
    const int dim = (g_cli_dim > 0) ? g_cli_dim : 128;

    Meta m; meta_default(&m);
    if (meta_read(idir, &m) == 0) {
        /* meta overrides params that MUST match the on-disk layout. */
        set_gen_version(m.gen_version);
        /* sub_dim taken from meta below in forest_open. n_trees/depth/n_docs
           come from CLI so the user can query a subset of trees/docs.     */
    } else {
        set_gen_version(0);
    }

    Forest f;
    if (forest_open(&f, idir, n_trees, dim, m.sub_dim, depth, n_docs) != 0) return 1;

    double recall = 0, avg_cands = 0, ms = 0;
    double t0 = now_s();
    int rc = eval_recall(&f, qvecs, gt, base, n_q, top_k, threshold, max_cands,
                         &recall, &avg_cands, &ms);
    if (rc != 0) { forest_close(&f); return 1; }

    fprintf(stderr,
        "\nRecall@%d = %.4f  |  avg_cands = %.0f  |  %.1f ms/query\n",
        top_k, recall, avg_cands, ms);
    fprintf(stderr,
        "total: %.2fs   peak RSS = %ld KB (%.2f MB)\n",
        now_s() - t0, peak_rss_kb(), peak_rss_kb() / 1024.0);

    forest_close(&f);
    return 0;
}

static int cmd_topn(int argc, char** argv) {
    if (argc < 9) return usage();
    const char* qvecs = argv[2];
    const char* gt    = argv[3];
    const char* base  = argv[4];
    const char* idir  = argv[5];
    int n_trees = atoi(argv[6]);
    int depth   = atoi(argv[7]);
    int n_docs  = atoi(argv[8]);
    int top_k = (argc > 9)  ? atoi(argv[9])  : 10;
    int top_n = (argc > 10) ? atoi(argv[10]) : 500;
    int n_q   = (argc > 11) ? atoi(argv[11]) : 1000;
    const int dim = (g_cli_dim > 0) ? g_cli_dim : 128;

    Meta m; meta_default(&m);
    if (meta_read(idir, &m) == 0) {
        set_gen_version(m.gen_version);
    } else {
        set_gen_version(0);
    }

    /* Heap-allocate Forest so its address (and the embedded io_uring) is
       stable across function calls and not on a moving stack frame.       */
    Forest* f = (Forest*)calloc(1, sizeof(Forest));
    if (!f) return 1;
    if (forest_open(f, idir, n_trees, dim, m.sub_dim, depth, n_docs) != 0)
        return 1;

    /* Load global filter (CRoaring) from --filter or --filter_ch. */
    int n_filter = 0;
    roaring_bitmap_t* allowed = load_filter_any(n_docs, &n_filter);
    int qd = g_cli_query_depth;
    if (g_cli_auto_qd_v2) {
        VecFmt qf = vec_fmt_from_path(qvecs);
        int qfd_probe = open(qvecs, O_RDONLY);
        if (qfd_probe >= 0) {
            int probe_avg = 0;
            qd = auto_query_depth_v2(f, depth, qfd_probe, qf, dim,
                                     top_n, allowed, n_docs, n_filter,
                                     &probe_avg);
            close(qfd_probe);
        }
    } else if (g_cli_auto_qd && n_filter > 0) {
        qd = auto_query_depth(depth, n_docs, n_filter);
        fprintf(stderr, "auto_qd: filter=%d corpus=%d build_depth=%d -> qd=%d\n",
                n_filter, n_docs, depth, qd);
    }

    double recall = 0, avg_cands = 0, ms = 0;
    double t0 = now_s();
    int rc = eval_recall_topn(f, qvecs, gt, base, n_q, top_k, top_n,
                              qd, allowed,
                              &recall, &avg_cands, &ms);
    if (allowed) roaring_bitmap_free(allowed);
    if (rc != 0) { forest_close(f); free(f); return 1; }

    long anon = 0, fileb = 0;
    read_proc_rss(&anon, &fileb);
    fprintf(stderr,
        "\nRecall@%d = %.4f  |  top_n = %d  |  avg_cands = %.0f  |  %.2f ms/query\n",
        top_k, recall, top_n, avg_cands, ms);
    fprintf(stderr,
        "total: %.2fs   RSS anon %.2f MB | mapped %.2f MB | peak ru_maxrss %.2f MB\n",
        now_s() - t0, anon / 1024.0, fileb / 1024.0, peak_rss_kb() / 1024.0);

    forest_close(f);
    free(f);
    return 0;
}

/* multitopn_auto : per-forest qd auto-calibration.
   Same args as multitopn but each forest gets its own qd derived from a
   2-probe `auto_qd_v2` on its own corpus. Calls `forest_collect_topn`
   sequentially per forest (loses the pairwise-seed traversal sharing, but
   isolates the recall-per-forest decision). Merges N × top_n cands by
   summing votes per doc_id (concat-sort-dedup) and reranks L2 against base. */
static int cmd_multitopn_auto(int argc, char** argv) {
    if (argc < 7) return usage();
    const char* qvecs_p  = argv[2];
    const char* gt_p     = argv[3];
    const char* base_p   = argv[4];
    const char* idx_list = argv[5];
    int n_trees = atoi(argv[6]);
    int depth   = atoi(argv[7]);
    int top_k   = (argc > 8)  ? atoi(argv[8])  : 10;
    int top_n   = (argc > 9)  ? atoi(argv[9])  : 500;
    int n_q     = (argc > 10) ? atoi(argv[10]) : 1000;
    const int dim = (g_cli_dim > 0) ? g_cli_dim : 128;

    char idxbuf[2048]; snprintf(idxbuf, sizeof(idxbuf), "%s", idx_list);
    char* dirs[32]; int n_forests = 0;
    char* p = idxbuf;
    while (*p && n_forests < 32) {
        dirs[n_forests++] = p;
        while (*p && *p != ',') p++;
        if (*p == ',') { *p = 0; p++; }
    }
    if (n_forests == 0) return usage();

    Meta m0; meta_default(&m0);
    if (meta_read(dirs[0], &m0) == 0) set_gen_version(m0.gen_version);

    Forest** forests = (Forest**)calloc((size_t)n_forests, sizeof(Forest*));
    int*     n_docs_arr = (int*)calloc((size_t)n_forests, sizeof(int));
    int*     qd_arr     = (int*)calloc((size_t)n_forests, sizeof(int));
    int      total_docs = 0;
    for (int i = 0; i < n_forests; i++) {
        forests[i] = (Forest*)calloc(1, sizeof(Forest));
        Meta mi; meta_default(&mi); meta_read(dirs[i], &mi);
        if (forest_open(forests[i], dirs[i], n_trees, dim, mi.sub_dim,
                        depth, mi.n_docs) != 0) {
            fprintf(stderr, "cannot open %s\n", dirs[i]); return 1;
        }
        n_docs_arr[i] = mi.n_docs;
        if (mi.n_docs > total_docs) total_docs = mi.n_docs;
        fprintf(stderr, "  opened forest %d (%s, n_docs=%d)\n", i, dirs[i], mi.n_docs);
    }

    VecFmt qfmt = vec_fmt_from_path(qvecs_p);
    VecFmt bfmt = vec_fmt_from_path(base_p);
    int qfd = open(qvecs_p, O_RDONLY);
    int gfd = open(gt_p,    O_RDONLY);
    int bfd = open(base_p,  O_RDONLY);
    if (qfd < 0 || gfd < 0 || bfd < 0) { perror("open"); return 1; }

    int n_filter = 0;
    roaring_bitmap_t* allowed = load_filter_any(total_docs, &n_filter);

    /* Per-forest qd via auto_qd_v2 probe — uses each forest's own
       n_distinct profile.                                                */
    if (g_cli_auto_qd_v2) {
        for (int i = 0; i < n_forests; i++) {
            int probe_avg = 0;
            qd_arr[i] = auto_query_depth_v2(forests[i], depth, qfd, qfmt, dim,
                                            top_n, allowed,
                                            n_docs_arr[i], n_filter, &probe_avg);
            fprintf(stderr, "  forest %d -> qd=%d\n", i, qd_arr[i]);
        }
    } else {
        for (int i = 0; i < n_forests; i++)
            qd_arr[i] = (g_cli_query_depth > 0 ? g_cli_query_depth : depth);
    }

    const int GT_K = 100;
    float*    qvec     = (float*) malloc((size_t)dim * sizeof(float));
    /* cat_ids / cand_ids hold up to n_forests * top_n cands pre-cut. */
    size_t cap = (size_t)n_forests * top_n;
    int32_t*  cat_ids  = (int32_t*)malloc(cap * sizeof(int32_t));
    int32_t*  cat_vot  = (int32_t*)malloc(cap * sizeof(int32_t));
    int32_t*  cand_ids = (int32_t*)malloc(cap * sizeof(int32_t));
    int32_t*  cand_vot = (int32_t*)malloc(cap * sizeof(int32_t));
    int32_t*  top_ids  = (int32_t*)malloc((size_t)top_k * sizeof(int32_t));
    int*      gt_buf   = (int*)    malloc((size_t)GT_K * sizeof(int));

    long long hits = 0;
    double t0 = now_s();
    for (int q = 0; q < n_q; q++) {
        if (read_vec_at(qfd, qfmt, q, dim, qvec) != 0) { n_q = q; break; }
        if (read_ivec(gfd, q, GT_K, gt_buf)       != 0) { n_q = q; break; }

        /* Collect from each forest with its own qd. */
        int n_total = 0;
        for (int i = 0; i < n_forests; i++) {
            int n_c = forest_collect_topn(forests[i], qvec, top_n, qd_arr[i],
                                          allowed,
                                          cat_ids + n_total, cat_vot + n_total);
            n_total += n_c;
        }

        /* Merge : sort by (id, vote) then collapse and pick top_n by votes. */
        for (int i = 0; i < n_total - 1; i++)
            for (int j = i + 1; j < n_total; j++)
                if (cat_ids[i] > cat_ids[j]) {
                    int32_t ti = cat_ids[i], tv = cat_vot[i];
                    cat_ids[i] = cat_ids[j]; cat_vot[i] = cat_vot[j];
                    cat_ids[j] = ti; cat_vot[j] = tv;
                }
        /* O(N²) is fine when n_forests * top_n is small. For prod, replace
           with qsort. Collapse dup ids by summing votes.                  */
        int n_uniq = 0;
        for (int i = 0; i < n_total; i++) {
            if (n_uniq > 0 && cand_ids[n_uniq - 1] == cat_ids[i]) {
                cand_vot[n_uniq - 1] += cat_vot[i];
            } else {
                cand_ids[n_uniq] = cat_ids[i];
                cand_vot[n_uniq] = cat_vot[i];
                n_uniq++;
            }
        }
        /* Pick top_n by votes (partial sort). */
        int keep = n_uniq < top_n ? n_uniq : top_n;
        for (int k = 0; k < keep; k++) {
            int best = k;
            for (int j = k + 1; j < n_uniq; j++)
                if (cand_vot[j] > cand_vot[best]) best = j;
            int32_t ti = cand_ids[k], tv = cand_vot[k];
            cand_ids[k] = cand_ids[best]; cand_vot[k] = cand_vot[best];
            cand_ids[best] = ti; cand_vot[best] = tv;
        }

        int n_top = forests[0]->ring_ok
            ? rerank_l2_uring(&forests[0]->ring, bfd, bfmt, dim, qvec,
                              cand_ids, keep, top_k, top_ids)
            : rerank_l2      (                  bfd, bfmt, dim, qvec,
                              cand_ids, keep, top_k, top_ids);

        for (int i = 0; i < n_top; i++) {
            int id = top_ids[i];
            for (int j = 0; j < top_k && j < GT_K; j++)
                if (gt_buf[j] == id) { hits++; break; }
        }
    }

    double dt = now_s() - t0;
    fprintf(stderr,
        "\nmultitopn_auto: %d forests, total_docs=%d, filter_card=%d\n"
        "Recall@%d = %.4f  |  top_n = %d  |  %.2f ms/query\n",
        n_forests, total_docs, n_filter,
        top_k, (double)hits / ((double)n_q * top_k), top_n,
        dt * 1000.0 / (n_q > 0 ? n_q : 1));

    free(qvec); free(cat_ids); free(cat_vot); free(cand_ids); free(cand_vot);
    free(top_ids); free(gt_buf); free(n_docs_arr); free(qd_arr);
    if (allowed) roaring_bitmap_free(allowed);
    close(qfd); close(gfd); close(bfd);
    for (int i = 0; i < n_forests; i++) { forest_close(forests[i]); free(forests[i]); }
    free(forests);
    return 0;
}

/* multitopn <qvecs> <gt> <base> <idx1,idx2,...> <n_trees> <depth>
              [top_k=10] [top_n=500] [n_queries=1000]
   Opens N forests sharing seeds (same n_trees/depth/sub_dim), queries them
   all at once, reranks against the shared base.fvecs.                    */
static int cmd_multitopn(int argc, char** argv) {
    if (argc < 7) return usage();
    const char* qvecs    = argv[2];
    const char* gt       = argv[3];
    const char* base     = argv[4];
    const char* idx_list = argv[5];
    int n_trees = atoi(argv[6]);
    int depth   = atoi(argv[7]);
    int top_k   = (argc > 8)  ? atoi(argv[8])  : 10;
    int top_n   = (argc > 9)  ? atoi(argv[9])  : 500;
    int n_q     = (argc > 10) ? atoi(argv[10]) : 1000;
    const int dim = (g_cli_dim > 0) ? g_cli_dim : 128;

    /* Split comma-separated index dirs. */
    char idxbuf[2048];
    snprintf(idxbuf, sizeof(idxbuf), "%s", idx_list);
    char* dirs[32];
    int n_forests = 0;
    char* p = idxbuf;
    while (*p && n_forests < 32) {
        dirs[n_forests++] = p;
        while (*p && *p != ',') p++;
        if (*p == ',') { *p = 0; p++; }
    }
    if (n_forests == 0) return usage();

    /* Read meta from forest 0 (must match across all). */
    Meta m; meta_default(&m);
    if (meta_read(dirs[0], &m) == 0) {
        set_gen_version(m.gen_version);
    } else {
        set_gen_version(0);
    }

    /* Raise FD limit BEFORE opening forests — we'll need n_forests × n_trees
       file descriptors plus rerank base fd plus queries.                  */
    {
        struct rlimit rl;
        if (getrlimit(RLIMIT_NOFILE, &rl) == 0) {
            rlim_t want = (rlim_t)(n_forests * n_trees + 64);
            if (rl.rlim_cur < want) {
                rl.rlim_cur = (want < rl.rlim_max) ? want : rl.rlim_max;
                if (setrlimit(RLIMIT_NOFILE, &rl) != 0)
                    fprintf(stderr, "warning: cannot raise FD limit to %lu\n",
                            (unsigned long)want);
            }
        }
    }

    Forest** forests = (Forest**)calloc((size_t)n_forests, sizeof(Forest*));
    int total_docs = 0;
    for (int i = 0; i < n_forests; i++) {
        forests[i] = (Forest*)calloc(1, sizeof(Forest));
        Meta mi; meta_default(&mi);
        meta_read(dirs[i], &mi);
        if (forest_open(forests[i], dirs[i], n_trees, dim, mi.sub_dim,
                        depth, mi.n_docs) != 0) {
            fprintf(stderr, "cannot open forest %s\n", dirs[i]);
            for (int j = 0; j < i; j++) { forest_close(forests[j]); free(forests[j]); }
            free(forests); return 1;
        }
        if (mi.n_docs > total_docs) total_docs = mi.n_docs;
        fprintf(stderr, "  opened forest %d: %s  n_docs(global max)=%d\n",
                i, dirs[i], mi.n_docs);
    }

    /* Open shared corpus and GT. */
    VecFmt qfmt = vec_fmt_from_path(qvecs);
    VecFmt bfmt = vec_fmt_from_path(base);
    int qfd = open(qvecs, O_RDONLY);
    int gfd = open(gt,    O_RDONLY);
    int bfd = open(base,  O_RDONLY);
    if (qfd < 0 || gfd < 0 || bfd < 0) { perror("open"); return 1; }

    /* Optional filter (CRoaring) + auto query_depth. */
    int n_filter = 0;
    roaring_bitmap_t* allowed = load_filter_any(total_docs, &n_filter);
    int qd = g_cli_query_depth;
    if (g_cli_auto_qd && n_filter > 0) {
        qd = auto_query_depth(depth, total_docs, n_filter);
        fprintf(stderr, "auto_qd: filter=%d total_docs=%d build_depth=%d -> qd=%d\n",
                n_filter, total_docs, depth, qd);
    }

    const int GT_K = 100;
    float*   qvec     = (float*)  malloc((size_t)dim * 4);
    int32_t* cand_ids = (int32_t*)malloc((size_t)top_n * 4);
    int32_t* cand_vot = (int32_t*)malloc((size_t)top_n * 4);
    int32_t* top_ids  = (int32_t*)malloc((size_t)top_k * 4);
    int*     gt_buf   = (int*)    malloc((size_t)GT_K * 4);

    long long hits = 0;
    double t0 = now_s();
    for (int q = 0; q < n_q; q++) {
        if (read_vec_at(qfd, qfmt, q, dim, qvec) != 0) { n_q = q; break; }
        if (read_ivec(gfd, q, GT_K, gt_buf)       != 0) { n_q = q; break; }

        int n_c = forest_collect_topn_multi(forests, n_forests, qvec, top_n,
                                            qd, allowed,
                                            cand_ids, cand_vot);
        int n_top = forests[0]->ring_ok
            ? rerank_l2_uring(&forests[0]->ring, bfd, bfmt, dim, qvec,
                              cand_ids, n_c, top_k, top_ids)
            : rerank_l2      (                  bfd, bfmt, dim, qvec,
                              cand_ids, n_c, top_k, top_ids);

        for (int i = 0; i < n_top; i++) {
            int id = top_ids[i];
            for (int j = 0; j < top_k && j < GT_K; j++)
                if (gt_buf[j] == id) { hits++; break; }
        }

        if ((q + 1) % 50 == 0 || q + 1 == n_q) {
            double dt = now_s() - t0;
            fprintf(stderr,
                "  q=%4d/%d  recall@%d=%.4f  %.2f ms/q\n",
                q + 1, n_q, top_k,
                (double)hits / ((q + 1) * (double)top_k),
                dt * 1000.0 / (q + 1));
        }
    }

    long anon = 0, fileb = 0; read_proc_rss(&anon, &fileb);
    double dt = now_s() - t0;
    fprintf(stderr,
        "\nMultitopn: %d forests, total_docs=%d, query_depth=%d, filter=%d\n"
        "Recall@%d = %.4f  |  top_n = %d  |  %.2f ms/query\n"
        "RSS anon %.2f MB | mapped %.2f MB | peak ru_maxrss %.2f MB\n",
        n_forests, total_docs, qd, n_filter,
        top_k, (double)hits / ((double)n_q * top_k), top_n,
        dt * 1000.0 / (n_q > 0 ? n_q : 1),
        anon / 1024.0, fileb / 1024.0, peak_rss_kb() / 1024.0);

    free(qvec); free(cand_ids); free(cand_vot); free(top_ids); free(gt_buf);
    if (allowed) roaring_bitmap_free(allowed);
    close(qfd); close(gfd); close(bfd);
    for (int i = 0; i < n_forests; i++) { forest_close(forests[i]); free(forests[i]); }
    free(forests);
    return 0;
}

/* search <qvecs> <base_fvecs> <index_dir> <n_trees> <depth> <n_docs>
          [top_k=10] [top_n=500] [n_queries=10]
          [--filter / --filter_ch <path>]
   Single-shot search : runs the forest, reranks against base, prints
   top_k doc_ids per query as CSV. No GT required. Stdout =
   `q,doc0,doc1,...` (one row per query). */
static int cmd_search(int argc, char** argv) {
    if (argc < 7) return usage();
    const char* qvecs_p = argv[2];
    const char* base_p  = argv[3];
    const char* idir    = argv[4];
    int n_trees = atoi(argv[5]);
    int depth   = atoi(argv[6]);
    int n_docs  = atoi(argv[7]);
    int top_k   = (argc > 8)  ? atoi(argv[8])  : 10;
    int top_n   = (argc > 9)  ? atoi(argv[9])  : 500;
    int n_q     = (argc > 10) ? atoi(argv[10]) : 10;
    const int dim = (g_cli_dim > 0) ? g_cli_dim : 128;

    Meta m; meta_default(&m);
    if (meta_read(idir, &m) == 0) set_gen_version(m.gen_version);
    else                          set_gen_version(0);

    Forest* f = (Forest*)calloc(1, sizeof(Forest));
    if (!f || forest_open(f, idir, n_trees, dim, m.sub_dim, depth, n_docs) != 0)
        return 1;

    int n_filter = 0;
    roaring_bitmap_t* allowed = load_filter_any(n_docs, &n_filter);
    int qd = g_cli_query_depth;
    if (g_cli_auto_qd && n_filter > 0) {
        qd = auto_query_depth(depth, n_docs, n_filter);
        fprintf(stderr, "auto_qd: filter=%d corpus=%d build_depth=%d -> qd=%d\n",
                n_filter, n_docs, depth, qd);
    }

    VecFmt qfmt = vec_fmt_from_path(qvecs_p);
    VecFmt bfmt = vec_fmt_from_path(base_p);
    int qfd = open(qvecs_p, O_RDONLY);
    int bfd = open(base_p,  O_RDONLY);
    if (qfd < 0 || bfd < 0) { perror("open"); return 1; }

    float*    qvec     = (float*)   malloc((size_t)dim * sizeof(float));
    int32_t*  cand_ids = (int32_t*) malloc((size_t)top_n * sizeof(int32_t));
    int32_t*  cand_vot = (int32_t*) malloc((size_t)top_n * sizeof(int32_t));
    int32_t*  top_ids  = (int32_t*) malloc((size_t)top_k * sizeof(int32_t));

    double t0 = now_s();
    for (int q = 0; q < n_q; q++) {
        if (read_vec_at(qfd, qfmt, q, dim, qvec) != 0) { n_q = q; break; }
        double tq = now_s();
        int n_c = forest_collect_topn(f, qvec, top_n, qd, allowed,
                                      cand_ids, cand_vot);
        int n_top = f->ring_ok
            ? rerank_l2_uring(&f->ring, bfd, bfmt, dim, qvec,
                              cand_ids, n_c, top_k, top_ids)
            : rerank_l2      (         bfd, bfmt, dim, qvec,
                              cand_ids, n_c, top_k, top_ids);
        double q_ms = (now_s() - tq) * 1000.0;
        fprintf(stderr, "Q %d %.3f\n", q, q_ms);
        printf("%d", q);
        for (int i = 0; i < n_top; i++) printf(",%d", top_ids[i]);
        printf("\n");
    }
    double dt = now_s() - t0;
    fprintf(stderr,
        "search: %d queries, top_k=%d, top_n=%d, qd=%d, filter_card=%d, %.2f ms/q\n",
        n_q, top_k, top_n, qd, n_filter,
        dt * 1000.0 / (n_q > 0 ? n_q : 1));

    free(qvec); free(cand_ids); free(cand_vot); free(top_ids);
    if (allowed) roaring_bitmap_free(allowed);
    close(qfd); close(bfd);
    forest_close(f); free(f);
    return 0;
}

/* compact <new_index> <base_path> <new_depth> <old_index> [old_index ...]
   Deepen 1+ segments into a single new segment at <new_depth>. Each old
   segment must have the same params (n_trees, depth, sub_dim, gen, dim).
   Saves compute vs a full rebuild — only (new_depth - old_depth) extra
   levels are traversed per doc. */
static int cmd_compact(int argc, char** argv) {
    if (argc < 6) return usage();
    const char* new_idx = argv[2];
    const char* base    = argv[3];
    int new_depth       = atoi(argv[4]);
    int n_olds          = argc - 5;
    const char** olds = (const char**)&argv[5];

    Meta m; meta_default(&m);
    if (meta_read(olds[0], &m) != 0) {
        fprintf(stderr, "compact: cannot read meta from %s\n", olds[0]);
        return 1;
    }
    /* Verify all sources share params (we trust the user — light check). */
    for (int i = 1; i < n_olds; i++) {
        Meta mi; meta_default(&mi);
        if (meta_read(olds[i], &mi) != 0 ||
            mi.n_trees != m.n_trees || mi.depth != m.depth ||
            mi.sub_dim != m.sub_dim || mi.dim != m.dim ||
            mi.gen_version != m.gen_version) {
            fprintf(stderr, "compact: param mismatch in %s\n", olds[i]);
            return 1;
        }
    }
    set_gen_version(m.gen_version);

    /* Sum n_docs across sources for the new meta (assumes disjoint slices). */
    int total_docs = 0;
    for (int i = 0; i < n_olds; i++) {
        Meta mi; meta_default(&mi);
        meta_read(olds[i], &mi);
        total_docs += mi.n_docs;
    }

    int rc = (n_olds == 1)
        ? compact_segment_deeper(olds[0], new_idx, base,
                                  m.n_trees, m.depth, new_depth, m.dim, m.sub_dim)
        : compact_multi_deeper(new_idx, olds, n_olds, base,
                               m.n_trees, m.depth, new_depth, m.dim, m.sub_dim);
    if (rc != 0) return 1;
    Meta nm = { m.n_trees, new_depth, m.dim, m.sub_dim, m.gen_version,
                total_docs, m.tree_sub, m.tree_sub_groups, m.node_perm };
    meta_write(new_idx, &nm);
    fprintf(stderr, "compact: done. %d source(s) -> %s (depth %d, n_docs %d)\n",
            n_olds, new_idx, new_depth, total_docs);
    return 0;
}

/* delete <index_dir> <doc_id> [...] — adds doc_ids to tombstones bitmap
   then flushes atomically. Multiple ids in one call = one save.            */
static int cmd_delete(int argc, char** argv) {
    if (argc < 4) return usage();
    const char* idir = argv[2];
    Meta m; meta_default(&m);
    if (meta_read(idir, &m) != 0) {
        fprintf(stderr, "delete: cannot read meta.txt\n");
        return 1;
    }
    set_gen_version(m.gen_version);
    Forest f; memset(&f, 0, sizeof(f));
    if (forest_open(&f, idir, m.n_trees, m.dim, m.sub_dim, m.depth, m.n_docs) != 0)
        return 1;
    int added = 0;
    for (int i = 3; i < argc; i++) {
        unsigned int id = (unsigned int)strtoul(argv[i], NULL, 10);
        if (forest_add_tombstone(&f, id) == 0) added++;
    }
    int rc = forest_save_tombstones(&f);
    fprintf(stderr, "delete: %d ids added, total tombstones=%d, rc=%d\n",
            added, forest_tombstone_count(&f), rc);
    forest_close(&f);
    return rc;
}

/* stats <index_dir> — prints index metadata, per-tree leaf stats (sampled
   from tree 0), total disk usage. Useful for ops / capacity planning.    */
static int cmd_stats(int argc, char** argv) {
    if (argc < 3) return usage();
    const char* idir = argv[2];

    Meta m; meta_default(&m);
    if (meta_read(idir, &m) != 0) {
        fprintf(stderr, "stats: cannot read %s/meta.txt\n", idir);
        return 1;
    }
    printf("Index: %s\n", idir);
    printf("  meta : n_trees=%d depth=%d dim=%d sub_dim=%d gen=v%d n_docs=%d\n",
           m.n_trees, m.depth, m.dim, m.sub_dim, m.gen_version, m.n_docs);

    /* Per-tree sample stats (tree 0). */
    char p[512];
    snprintf(p, sizeof(p), "%s/tree%05d.srt", idir, 0);
    SortedStore s;
    if (sorted_store_open_rdonly(&s, p) != 0) {
        fprintf(stderr, "stats: cannot open %s\n", p);
        return 1;
    }
    struct stat st;
    long tree_size = (stat(p, &st) == 0) ? (long)st.st_size : 0;
    const char* fmt_name = sorted_is_delta(&s) ? "SRT3 (delta+VarByte)"
                                                : "SRT2 (raw uint32)";
    long sparse_b = (long)s.n_nonempty * 8 + 8;
    long data_b   = sorted_is_delta(&s) ? (long)s.data_bytes
                                         : (long)s.total_docs * 4;
    printf("  format : %s, hash footer: %s\n",
           fmt_name, sorted_is_delta(&s) ? "yes (xxhash64)" : "no");
    printf("  tree 0 sample :\n");
    printf("    file size    : %.2f MB\n", tree_size / 1024.0 / 1024.0);
    printf("    n_nonempty   : %u leaves (%.3f%% of 2^%u)\n",
           s.n_nonempty, 100.0 * s.n_nonempty / (1ULL << s.depth), s.depth);
    printf("    avg docs/leaf: %.2f\n",
           (double)s.total_docs / (double)(s.n_nonempty ? s.n_nonempty : 1));
    printf("    bytes/doc    : %.2f\n",
           (double)data_b / (double)(s.total_docs ? s.total_docs : 1));
    printf("    sparse_index : %.2f MB\n", sparse_b / 1024.0 / 1024.0);
    printf("    data block   : %.2f MB\n", data_b / 1024.0 / 1024.0);
    sorted_store_close(&s);

    /* Walk index dir, sum sizes. */
    DIR* d = opendir(idir);
    long total_bytes = 0;
    int  n_srt_files = 0;
    if (d) {
        struct dirent* ent;
        while ((ent = readdir(d)) != NULL) {
            if (strstr(ent->d_name, ".srt") != NULL &&
                strstr(ent->d_name, ".srt.tmp") == NULL) {
                char fp[600];
                snprintf(fp, sizeof(fp), "%s/%s", idir, ent->d_name);
                struct stat ss;
                if (stat(fp, &ss) == 0) {
                    total_bytes += ss.st_size;
                    n_srt_files++;
                }
            }
        }
        closedir(d);
    }
    printf("  total disk   : %.2f GB across %d .srt files\n",
           total_bytes / 1024.0 / 1024.0 / 1024.0, n_srt_files);

    /* Tombstones. */
    char tomb_p[600];
    snprintf(tomb_p, sizeof(tomb_p), "%s/tombstones.roaring", idir);
    struct stat tss;
    if (stat(tomb_p, &tss) == 0) {
        printf("  tombstones   : %s/tombstones.roaring (%.2f KB on disk)\n",
               idir, tss.st_size / 1024.0);
    } else {
        printf("  tombstones   : (none)\n");
    }
    return 0;
}

/* health <index_dir> — runs a battery of fast checks and returns 0/1.
   Useful for liveness probes / startup gates.                            */
static int cmd_health(int argc, char** argv) {
    if (argc < 3) return usage();
    const char* idir = argv[2];

    Meta m; meta_default(&m);
    if (meta_read(idir, &m) != 0) {
        fprintf(stderr, "health: FAIL (no meta.txt)\n"); return 1;
    }
    printf("[meta]      OK n_trees=%d depth=%d dim=%d\n",
           m.n_trees, m.depth, m.dim);

    /* All n_trees .srt files exist + non-empty */
    int missing = 0;
    long total = 0;
    for (int t = 0; t < m.n_trees; t++) {
        char p[512];
        snprintf(p, sizeof(p), "%s/tree%05d.srt", idir, t);
        struct stat st;
        if (stat(p, &st) != 0 || st.st_size == 0) missing++;
        else total += st.st_size;
    }
    if (missing > 0) {
        fprintf(stderr, "[srt]       FAIL %d/%d trees missing or empty\n",
                missing, m.n_trees);
        return 1;
    }
    printf("[srt]       OK %d trees, %.2f GB total\n",
           m.n_trees, total / 1024.0 / 1024.0 / 1024.0);

    /* Magic check on tree 0 (cheap proxy for format consistency). */
    SortedStore s;
    char p0[512];
    snprintf(p0, sizeof(p0), "%s/tree%05d.srt", idir, 0);
    if (sorted_store_open_rdonly(&s, p0) != 0) {
        fprintf(stderr, "[magic]     FAIL cannot open tree 0\n"); return 1;
    }
    printf("[magic]     OK %s, depth=%u\n",
           sorted_is_delta(&s) ? "SRT3" : "SRT2", s.depth);
    sorted_store_close(&s);

    /* xxhash on tree 0 only (full verify is the `verify` cmd). */
    int rc = srt_verify_hash(p0);
    if (rc != 1) {
        fprintf(stderr, "[hash:tree0] FAIL (rc=%d)\n", rc); return 1;
    }
    printf("[hash:tree0] OK\n");
    return 0;
}

/* verify <index_dir> [n_trees] — checks xxhash64 footer on every tree.srt.
   If n_trees omitted, reads from meta.txt. Reports OK/FAIL/MISSING per tree
   and exits non-zero on any failure. Fast: O(file size) per tree, no decode.*/
static int cmd_verify(int argc, char** argv) {
    if (argc < 3) return usage();
    const char* idir = argv[2];
    int n_trees = (argc > 3) ? atoi(argv[3]) : 0;
    Meta m; meta_default(&m);
    if (meta_read(idir, &m) == 0 && n_trees == 0) n_trees = m.n_trees;
    if (n_trees <= 0) {
        fprintf(stderr, "verify: n_trees unknown (give explicit arg or write meta.txt)\n");
        return 1;
    }

    int ok = 0, fail = 0, missing = 0;
    double t0 = now_s();
    for (int t = 0; t < n_trees; t++) {
        char path[768];
        snprintf(path, sizeof(path), "%s/tree%05d.srt", idir, t);
        int rc = srt_verify_hash(path);
        if (rc == 1)      ok++;
        else if (rc == 0) { fail++; fprintf(stderr, "FAIL %s\n", path); }
        else              { missing++; fprintf(stderr, "MISSING %s\n", path); }
        if ((t + 1) % 100 == 0 || t + 1 == n_trees) {
            fprintf(stderr, "  verify: %4d / %d (ok=%d fail=%d missing=%d)\n",
                    t + 1, n_trees, ok, fail, missing);
        }
    }
    fprintf(stderr, "\nverify done in %.2fs: ok=%d fail=%d missing=%d\n",
            now_s() - t0, ok, fail, missing);
    return (fail == 0 && missing == 0) ? 0 : 1;
}

int main(int argc, char** argv) {
    parse_flags(&argc, &argv);
    if (argc < 2) return usage();
    if (strcmp(argv[1], "build")     == 0) return cmd_build(argc, argv);
    if (strcmp(argv[1], "build2")    == 0) {
        extern int cmd_build2(int, char**);
        return cmd_build2(argc, argv);
    }
    if (strcmp(argv[1], "abuild")    == 0) {
        extern int cmd_anchor_build(int, char**);
        return cmd_anchor_build(argc, argv);
    }
    if (strcmp(argv[1], "abench")    == 0) {
        extern int cmd_anchor_bench(int, char**);
        return cmd_anchor_bench(argc, argv);
    }
    if (strcmp(argv[1], "tquant")    == 0) return cmd_tquant(argc, argv);
    if (strcmp(argv[1], "tquant1")   == 0) return cmd_tquant1(argc, argv);
    if (strcmp(argv[1], "query")     == 0) return cmd_query(argc, argv);
    if (strcmp(argv[1], "recall")    == 0) return cmd_recall(argc, argv);
    if (strcmp(argv[1], "topn")      == 0) return cmd_topn(argc, argv);
    if (strcmp(argv[1], "multitopn") == 0) return cmd_multitopn(argc, argv);
    if (strcmp(argv[1], "multitopn_auto") == 0) return cmd_multitopn_auto(argc, argv);
    if (strcmp(argv[1], "search")    == 0) return cmd_search(argc, argv);
    if (strcmp(argv[1], "verify")    == 0) return cmd_verify(argc, argv);
    if (strcmp(argv[1], "delete")    == 0) return cmd_delete(argc, argv);
    if (strcmp(argv[1], "compact")   == 0) return cmd_compact(argc, argv);
    if (strcmp(argv[1], "stats")     == 0) return cmd_stats(argc, argv);
    if (strcmp(argv[1], "health")    == 0) return cmd_health(argc, argv);
    return usage();
}
