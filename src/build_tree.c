#define _POSIX_C_SOURCE 200809L
#include "build_tree.h"
#include "sorted_store.h"
#include "traversal.h"
#include "gen_vec.h"
#include "varbyte.h"
#include "srt_hash.h"
#include "calibration.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/resource.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static long peak_rss_kb(void) {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_maxrss;  /* KB on Linux */
}

/* Build trees with seeds shifted by this offset (tree file t gets
   tree_seed(t + offset)). Lets us grow an existing forest by building
   only the new trees, then renaming the files into the main index.   */
static int g_tree_seed_offset = 0;
void set_tree_seed_offset(int off) { g_tree_seed_offset = off; }

/* Tree-batched build: total number of trees in the FINAL index. When a build
   invocation only produces a slice [offset, offset+n_trees), this is the true
   count used for medians.bin validation (the table always covers all trees)
   and for indexing each tree's median slice by its GLOBAL id. 0 = single-shot
   build (total == n_trees).                                                  */
static int g_total_trees = 0;
void set_total_trees(int n) { g_total_trees = n; }

static int g_fast_mode = 0;
void set_build_fast_mode(int on) { g_fast_mode = on ? 1 : 0; }
static int g_cli_build_batch = 0;
void set_build_batch(int b) { g_cli_build_batch = b > 0 ? b : 0; }
/* Online calibration settings (0/off by default). */
static int    g_calib_n_queries    = 0;    /* --calib-queries : pool size */
static int    g_calib_top_k        = 10;
static double g_calib_interval_s   = 0.0;  /* --calib-interval : snapshot cadence sec, 0 = end only */
void set_build_calib(int n_queries, int top_k, double interval_s) {
    g_calib_n_queries  = n_queries > 0 ? n_queries : 0;
    g_calib_top_k      = top_k > 0 ? top_k : 10;
    g_calib_interval_s = interval_s;
}

/* Auto-tune schedule "doubling" : recalibrate the config at 100k, 200k, 400k,
   ... indexed docs. Each trigger pauses the build, snapshots calibration
   GT to disk, invokes an external calibrator subprocess, then resumes. */
static int   g_recalib_doubling = 0;         /* 0 = disabled (default) */
static char* g_recalib_script   = NULL;      /* NULL = default path */
void set_build_recalib_doubling(int enable, const char* script_path) {
    g_recalib_doubling = enable ? 1 : 0;
    free(g_recalib_script);
    g_recalib_script = (script_path && *script_path)
        ? strdup(script_path) : NULL;
}

/* --no-varbyte : convert to SRT V2 (raw uint32 doc_ids) instead of V3 (varbyte
   delta). V2 est ~1.7-2× plus lourd sur disque mais plus simple à décoder,
   utile pour prototype/streaming où on veut skipper la compression. */
static int g_build_no_varbyte = 0;
void set_build_no_varbyte(int on) { g_build_no_varbyte = on ? 1 : 0; }
int  get_build_no_varbyte(void)   { return g_build_no_varbyte; }

/* --tail : if the input file has fewer than n_vecs records at read time, WAIT
   (poll every 500 ms) for more data to be appended by the upstream producer.
   Producer writes via O_APPEND, consumer reads sequentially (safe on POSIX).
   Enables the "WAL disk queue" pattern : producer never blocks, consumer
   catches up on pause. */
static int g_build_tail = 0;
void set_build_tail(int on) { g_build_tail = on ? 1 : 0; }
int  get_build_tail(void)   { return g_build_tail; }

static int raise_fd_limit(int target) {
    struct rlimit rl;
    if (getrlimit(RLIMIT_NOFILE, &rl) != 0) return -1;
    if ((long)rl.rlim_cur >= (long)target) return 0;
    rl.rlim_cur = ((rlim_t)target < rl.rlim_max) ? (rlim_t)target : rl.rlim_max;
    return setrlimit(RLIMIT_NOFILE, &rl);
}

/* ---- Top-level medians : persistence + calibration driver ----
   File medians.bin : [u32 magic "MED1"][u32 n_trees][u32 med_depth][u32 rsvd]
                      [n_trees × (2^med_depth - 1) floats]                   */
#define MED_MAGIC 0x3144454du

/* MED2 : [u32 "MED2"][u32 n_trees][u32 med_depth][u32 rsvd]
          [n_trees x 2 x md floats : par arbre lo[md] puis span[md]]
          [n_trees x (2^md - 1) u8 codes]
   4x moins de RAM que MED1 (268 Mo vs 1,07 Go a 256 arbres d20) ;
   decode bit-identique au f32 de build2 (voir traversal_set_medians8). */
#define MED2_MAGIC 0x3244454du

uint8_t* medians8_load(const char* path, int* out_n_trees,
                       int* out_med_depth, float** out_scales) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    uint32_t hdr[4];
    if (fread(hdr, 4, 4, f) != 4 || hdr[0] != MED2_MAGIC) {
        fclose(f); return NULL;
    }
    int nt = (int)hdr[1], md = (int)hdr[2];
    if (nt <= 0 || md <= 0 || md > 24) { fclose(f); return NULL; }
    size_t ns = (size_t)nt * 2 * md;
    size_t nc = (size_t)nt * ((1u << md) - 1);
    float* scales = (float*)malloc(ns * sizeof(float));
    uint8_t* codes = (uint8_t*)malloc(nc);
    if (!scales || !codes
        || fread(scales, sizeof(float), ns, f) != ns
        || fread(codes, 1, nc, f) != nc) {
        free(scales); free(codes); fclose(f);
        return NULL;
    }
    fclose(f);
    *out_n_trees = nt; *out_med_depth = md; *out_scales = scales;
    return codes;
}

float* medians_load(const char* path, int* out_n_trees, int* out_med_depth) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    uint32_t hdr[4];
    if (fread(hdr, 4, 4, f) != 4 || hdr[0] != MED_MAGIC) { fclose(f); return NULL; }
    int nt = (int)hdr[1], md = (int)hdr[2];
    if (nt <= 0 || md <= 0 || md > 24) { fclose(f); return NULL; }
    size_t n = (size_t)nt * ((1u << md) - 1);
    float* tab = (float*)malloc(n * sizeof(float));
    if (!tab || fread(tab, sizeof(float), n, f) != n) {
        free(tab); fclose(f); return NULL;
    }
    fclose(f);
    *out_n_trees = nt; *out_med_depth = md;
    return tab;
}

int calibrate_and_save_medians(const char* vecs_path, const char* index_dir,
                               int n_trees, int dim, int sub_dim,
                               int med_depth, int sample_n) {
    /* NOTE : caller must have set the SAME gen_version as the build. */
    if (med_depth <= 0 || med_depth > 24 || sample_n <= 0) return -1;
    VecFmt fmt = vec_fmt_from_path(vecs_path);
    int fd = open(vecs_path, O_RDONLY);
    if (fd < 0) { perror("calibrate: open vecs"); return -1; }
    float* sample = (float*)malloc((size_t)sample_n * dim * sizeof(float));
    /* scratch de conversion : dim octets (u8) ou dim×2 (f16) */
    uint8_t* u8row = (uint8_t*)malloc((size_t)dim * 2);
    if (!sample || !u8row) { free(sample); free(u8row); close(fd); return -1; }
    /* Strided sample : row i × stride, avoids file-order bias. Stride
       derived from the file size so the sample spans the whole corpus. */
    struct stat stt; fstat(fd, &stt);
    long n_total = (long)((stt.st_size -
        ((fmt==VECFMT_U8BIN||fmt==VECFMT_FBIN||fmt==VECFMT_F16BIN)?8:0))
        / (vec_row_bytes(fmt, dim) + ((fmt==VECFMT_BVECS||fmt==VECFMT_FVECS)?4:0)));
    long stride_rows = n_total > sample_n ? n_total / sample_n : 1;
    int got = 0;
    for (int i = 0; i < sample_n; i++) {
        off_t off = vec_row_offset(fmt, (int)((long)i * stride_rows), dim);
        size_t rb = vec_row_bytes(fmt, dim);
        if (fmt == VECFMT_U8BIN || fmt == VECFMT_BVECS) {
            if (pread(fd, u8row, rb, off) != (ssize_t)rb) break;
            vec_u8_to_f32(u8row, sample + (size_t)got * dim, dim);
        } else if (fmt == VECFMT_F16BIN) {
            if (pread(fd, u8row, rb, off) != (ssize_t)rb) break;
            vec_f16_to_f32((const uint16_t*)u8row,
                           sample + (size_t)got * dim, dim);
        } else {
            if (pread(fd, sample + (size_t)got * dim, rb, off) != (ssize_t)rb) break;
        }
        got++;
    }
    close(fd); free(u8row);
    if (got < 1024) { free(sample); return -1; }   /* sample trop petit */
    fprintf(stderr, "  calibrating medians : %d samples, depth %d, %d trees...\n",
            got, med_depth, n_trees);

    size_t med_nodes = (1u << med_depth) - 1;
    float* table = (float*)malloc((size_t)n_trees * med_nodes * sizeof(float));
    if (!table) { free(sample); return -1; }
    int rc = traversal_calibrate_medians(sample, got, dim, sub_dim,
                                         n_trees, med_depth, table);
    free(sample);
    if (rc != 0) { free(table); return -1; }

    char mpath[600], tpath[620];
    snprintf(mpath, sizeof(mpath), "%s/medians.bin", index_dir);
    snprintf(tpath, sizeof(tpath), "%s.tmp", mpath);
    FILE* f = fopen(tpath, "wb");
    int ok = f != NULL;
    if (ok) {
        uint32_t hdr[4] = { MED_MAGIC, (uint32_t)n_trees, (uint32_t)med_depth, 0 };
        ok = fwrite(hdr, 4, 4, f) == 4
          && fwrite(table, sizeof(float), (size_t)n_trees * med_nodes, f)
             == (size_t)n_trees * med_nodes;
        ok = (fclose(f) == 0) && ok;
    }
    free(table);
    if (!ok) { unlink(tpath); return -1; }
    if (rename(tpath, mpath) != 0) { unlink(tpath); return -1; }
    fprintf(stderr, "  medians.bin written (%zu KB)\n",
            (16 + (size_t)n_trees * med_nodes * 4) / 1024);
    return 0;
}

int build_forest(const char* vecs_path, int doc_offset, int n_vecs,
                 int dim, int sub_dim,
                 int n_trees, int depth, const char* index_dir) {
    return build_forest_ex(vecs_path, doc_offset, n_vecs, doc_offset,
                           dim, sub_dim, n_trees, depth, index_dir);
}

int build_forest_ex(const char* vecs_path, int doc_offset, int n_vecs,
                    int doc_id_base, int dim, int sub_dim,
                    int n_trees, int depth, const char* index_dir) {
    int use_sub = (sub_dim > 0 && sub_dim <= dim);
    int scratch_dim = use_sub ? sub_dim : dim;

    /* Top-level medians : if <index_dir>/medians.bin exists (written by
       calibrate_and_save_medians before the build), routing applies its
       thresholds — queries load the same file so doc/query routing match. */
    float*   bmed_table = NULL;
    uint8_t* bmed8 = NULL;
    float*   bmed8_scales = NULL;
    int      bmed_depth = 0, bmed_nodes = 0;
    {
        char mpath[600];
        snprintf(mpath, sizeof(mpath), "%s/medians.bin", index_dir);
        int mnt = 0;
        int med_expect = g_total_trees > 0 ? g_total_trees : n_trees;
        bmed8 = medians8_load(mpath, &mnt, &bmed_depth, &bmed8_scales);
        if (!bmed8) bmed_table = medians_load(mpath, &mnt, &bmed_depth);
        if (bmed8 || bmed_table) {
            if (mnt != med_expect) {
                fprintf(stderr, "build: medians.bin n_trees mismatch (%d vs %d), ignored\n",
                        mnt, med_expect);
                free(bmed_table); bmed_table = NULL;
                free(bmed8); free(bmed8_scales);
                bmed8 = NULL; bmed8_scales = NULL; bmed_depth = 0;
            } else {
                bmed_nodes = (1 << bmed_depth) - 1;
                fprintf(stderr, "  medians ACTIVE (%s) : depth %d (%d nodes/tree)\n",
                        bmed8 ? "MED2 int8" : "MED1 f32",
                        bmed_depth, bmed_nodes);
            }
        }
    }
    if (raise_fd_limit(n_trees + 16) != 0) {
        fprintf(stderr, "warning: could not raise fd limit\n");
    }

    /* Resume: read prior progress (number of vectors processed). */
    char progress_path[600];
    snprintf(progress_path, sizeof(progress_path), "%s/phase1_t%05d.progress",
             index_dir, g_tree_seed_offset);
    int n_done = 0;
    {
        FILE* pf = fopen(progress_path, "r");
        if (pf) {
            if (fscanf(pf, "%d", &n_done) != 1) n_done = 0;
            fclose(pf);
            if (n_done < 0 || n_done > n_vecs) n_done = 0;
            if (n_done > 0)
                fprintf(stderr, "  resume: phase1 skips first %d vecs (of %d)\n",
                        n_done, n_vecs);
        }
    }
    if (n_done == n_vecs) {
        fprintf(stderr, "  phase1 already complete (progress=%d)\n", n_done);
        return 0;
    }
    /* Strong resume: if every .srt exists, phase 1 + 2 are already done. */
    {
        int all_srt = 1;
        for (int t = 0; t < n_trees; t++) {
            char p[600];
            snprintf(p, sizeof(p), "%s/tree%05d.srt", index_dir, t + g_tree_seed_offset);
            struct stat sst;
            if (stat(p, &sst) != 0 || sst.st_size == 0) { all_srt = 0; break; }
        }
        if (all_srt) {
            fprintf(stderr, "  all .srt present — skipping phase 1\n");
            return 0;
        }
    }

    VecFmt fmt = vec_fmt_from_path(vecs_path);
    FILE* fin = fopen(vecs_path, "rb");
    if (!fin) { perror("fopen vecs"); return -1; }

    /* Skip the per-file header for u8bin / fbin / f16bin (8 B = n, dim). */
    if (fmt == VECFMT_U8BIN || fmt == VECFMT_FBIN || fmt == VECFMT_F16BIN) {
        uint32_t hdr[2];
        if (fread(hdr, 4, 2, fin) != 2) {
            fprintf(stderr, "header read failed\n");
            fclose(fin); return -1;
        }
        if ((int)hdr[1] != dim) {
            fprintf(stderr, "dim mismatch in header %u != %d\n", hdr[1], dim);
            fclose(fin); return -1;
        }
    }

    /* Seek past (doc_offset + n_done) rows. */
    int seek_rows = doc_offset + n_done;
    if (seek_rows > 0) {
        off_t row;
        if      (fmt == VECFMT_U8BIN)  row = (off_t)dim;
        else if (fmt == VECFMT_BVECS)  row = (off_t)4 + (off_t)dim;
        else if (fmt == VECFMT_FBIN)   row = (off_t)dim * 4;
        else if (fmt == VECFMT_F16BIN) row = (off_t)dim * 2;
        else                           row = (off_t)4 + (off_t)dim * 4;
        if (fseeko(fin, row * (off_t)seek_rows, SEEK_CUR) != 0) {
            perror("seek doc_offset/resume"); fclose(fin); return -1;
        }
    }

    FILE** trees = (FILE**)malloc((size_t)n_trees * sizeof(FILE*));
    if (!trees) { fclose(fin); return -1; }

    const char* tree_mode = (n_done > 0) ? "ab" : "wb";
    char path[512];
    for (int t = 0; t < n_trees; t++) {
        snprintf(path, sizeof(path), "%s/tree%05d.bin", index_dir, t + g_tree_seed_offset);
        trees[t] = fopen(path, tree_mode);
        if (!trees[t]) {
            fprintf(stderr, "cannot open %s: ", path); perror("");
            for (int j = 0; j < t; j++) fclose(trees[j]);
            free(trees); fclose(fin);
            return -1;
        }
    }

    float*   vec    = (float*)  malloc((size_t)dim * sizeof(float));
    /* scratch de conversion partagé : u8 (dim o) ou f16 (dim×2 o) */
    int      need_u8 = (fmt == VECFMT_U8BIN || fmt == VECFMT_BVECS
                        || fmt == VECFMT_F16BIN);
    uint8_t* u8_buf = need_u8 ? (uint8_t*)malloc((size_t)dim * 2) : NULL;
    if (!vec || (need_u8 && !u8_buf)) {
        free(vec); free(u8_buf); fclose(fin); return -1;
    }

    double t0 = now_s();
    long max_n = n_vecs;
    int dread = 0;

#ifdef _OPENMP
    int n_threads = omp_get_max_threads();
#else
    int n_threads = 1;
#endif
    const char* fmt_name = (fmt == VECFMT_U8BIN)  ? "u8bin"
                         : (fmt == VECFMT_BVECS)  ? "bvecs"
                         : (fmt == VECFMT_FBIN)   ? "fbin"
                         : (fmt == VECFMT_F16BIN) ? "f16bin" : "fvecs";
    fprintf(stderr, "  using %d threads, sub_dim=%d, gen_v%d, fmt=%s\n",
            n_threads, use_sub ? sub_dim : dim,
            get_gen_version(),
            fmt_name);

    const int progress_chunk = 100000;  /* checkpoint cadence */

    /* Batched build : amortize OpenMP team creation + fwrite() overhead
       across BATCH docs. BATCH can be raised via `--batch N` for a one-shot
       fast import (more RAM for vec_buf + local Pair scratch, fewer fwrite
       syscalls). Defaults tuned for the memory-bounded classic path.       */
    int BATCH = g_fast_mode ? 4096 : 256;
    if (g_cli_build_batch > 0) BATCH = g_cli_build_batch;
    float*   vec_buf  = (float*)malloc((size_t)BATCH * dim * sizeof(float));
    uint8_t* u8_batch = need_u8 ? (uint8_t*)malloc((size_t)BATCH * dim * 2)
                                : NULL;
    int*     batch_doc_ids = (int*)malloc((size_t)BATCH * sizeof(int));
    char*    batch_ok = (char*)malloc((size_t)BATCH);
    if (!vec_buf || (need_u8 && !u8_batch) || !batch_doc_ids || !batch_ok) {
        free(vec); free(u8_buf); free(vec_buf); free(u8_batch);
        free(batch_doc_ids); free(batch_ok);
        fclose(fin); return -1;
    }

    /* --fast mode : precompute all inner-node hyperplanes once per tree,
       reuse across every doc traversal. RAM cost :
         n_trees × (2^depth − 1) × 2 × sub_dim × 4 B
       (~2 GB at d=16 / 256 trees / sub_dim=16).                          */
    float** hp_cache_v0 = NULL;
    float** hp_cache_v1 = NULL;
    size_t  n_inner = (use_sub && depth > 0 && g_fast_mode)
                    ? ((size_t)1 << depth) - 1 : 0;
    if (n_inner > 0) {
        double tc0 = now_s();
        size_t per_tree_bytes = n_inner * (size_t)sub_dim * sizeof(float);
        hp_cache_v0 = (float**)malloc((size_t)n_trees * sizeof(float*));
        hp_cache_v1 = (float**)malloc((size_t)n_trees * sizeof(float*));
        int alloc_ok = (hp_cache_v0 && hp_cache_v1);
        if (alloc_ok) {
            for (int t = 0; t < n_trees; t++) {
                hp_cache_v0[t] = (float*)malloc(per_tree_bytes);
                hp_cache_v1[t] = (float*)malloc(per_tree_bytes);
                if (!hp_cache_v0[t] || !hp_cache_v1[t]) { alloc_ok = 0; break; }
            }
        }
        if (!alloc_ok) {
            fprintf(stderr, "  --fast : hyperplane cache alloc failed (%.0f MB), "
                            "falling back to classic build\n",
                    (double)per_tree_bytes * 2 * n_trees / 1024.0 / 1024.0);
            if (hp_cache_v0) for (int t = 0; t < n_trees; t++) free(hp_cache_v0[t]);
            if (hp_cache_v1) for (int t = 0; t < n_trees; t++) free(hp_cache_v1[t]);
            free(hp_cache_v0); free(hp_cache_v1);
            hp_cache_v0 = hp_cache_v1 = NULL;
        } else {
            fprintf(stderr, "  --fast : hyperplane cache %.0f MB "
                            "(n_inner=%zu × %d trees × 2 × %d floats)\n",
                    (double)per_tree_bytes * 2 * n_trees / 1024.0 / 1024.0,
                    n_inner, n_trees, sub_dim);
            #pragma omp parallel for schedule(static)
            for (int t = 0; t < n_trees; t++) {
                uint64_t ts = tree_seed(t + g_tree_seed_offset);
                for (size_t node = 0; node < n_inner; node++) {
                    gen_vec(node_seed(ts, (int)(node * 2)),
                            hp_cache_v0[t] + node * (size_t)sub_dim, sub_dim);
                    gen_vec(node_seed(ts, (int)(node * 2 + 1)),
                            hp_cache_v1[t] + node * (size_t)sub_dim, sub_dim);
                }
            }
            fprintf(stderr, "  --fast : hyperplane cache built in %.1fs\n",
                    now_s() - tc0);
        }
    }

    /* Online calibration init : builds top-K GT of a small query pool
       incrementally during streaming (see calibration.h).                */
    if (g_calib_n_queries > 0) {
        calib_init(g_calib_n_queries, g_calib_top_k, dim);
    }
    double calib_last_snapshot = now_s();

    /* Doubling recalibration schedule : starts at 100k, doubles each trigger.
       Skipped if the corpus is too small (never crosses 100k). */
    int64_t next_recalib = 100000;
    int doc_id = n_done;
    if (g_recalib_doubling && n_done > 0) {
        while (next_recalib <= n_done) next_recalib *= 2;
    }
    while (doc_id < n_vecs) {
        int batch_n = (n_vecs - doc_id < BATCH) ? (n_vecs - doc_id) : BATCH;

        /* --- serial reads of batch_n vectors into vec_buf --- */
        int b;
        for (b = 0; b < batch_n; b++) {
            float* vb = vec_buf + (size_t)b * dim;
            uint8_t* ub = u8_batch ? (u8_batch + (size_t)b * dim * 2) : NULL;
            int ok = 1;
            /* TAIL mode : if fread returns short, retry with wait (WAL not
               yet at position). Otherwise break batch as before. */
            int retries = 0;
            for (;;) {
                if (fmt == VECFMT_U8BIN) {
                    if (fread(ub, 1, (size_t)dim, fin) != (size_t)dim) ok = 0;
                    else vec_u8_to_f32(ub, vb, dim);
                } else if (fmt == VECFMT_BVECS) {
                    if (fread(&dread, 4, 1, fin) != 1 || dread != dim ||
                        fread(ub, 1, (size_t)dim, fin) != (size_t)dim) ok = 0;
                    else vec_u8_to_f32(ub, vb, dim);
                } else if (fmt == VECFMT_FBIN) {
                    if (fread(vb, 4, dim, fin) != (size_t)dim) ok = 0;
                } else if (fmt == VECFMT_F16BIN) {
                    if (fread(ub, 2, dim, fin) != (size_t)dim) ok = 0;
                    else vec_f16_to_f32((const uint16_t*)ub, vb, dim);
                } else {
                    if (fread(&dread, 4, 1, fin) != 1 || dread != dim ||
                        fread(vb, 4, dim, fin) != (size_t)dim) ok = 0;
                }
                if (ok || !g_build_tail || retries >= 600) break;
                /* WAL tail : wait 500 ms and retry, up to 5 min per record. */
                clearerr(fin);
                struct timespec ts = {0, 500 * 1000 * 1000};
                nanosleep(&ts, NULL);
                retries++;
                ok = 1;   /* reset for next attempt */
            }
            batch_ok[b] = (char)ok;
            batch_doc_ids[b] = doc_id + b;
            if (!ok) { batch_n = b; break; }
            normalize(vb, dim);
        }
        if (batch_n == 0) break;

        /* --- Online calibration : update GT for the K query pool with
               the batch we just read (single thread, negligible cost). */
        if (calib_is_enabled()) {
            calib_update(vec_buf, batch_doc_ids, batch_n);
            if (g_calib_interval_s > 0.0 &&
                now_s() - calib_last_snapshot >= g_calib_interval_s) {
                calib_snapshot(index_dir);
                calib_last_snapshot = now_s();
            }
        }

        /* --- one parallel section per batch : each thread owns a chunk
               of trees → FILE* single-writer, no race --- */
        #pragma omp parallel
        {
            float v0_t[scratch_dim], v1_t[scratch_dim];
            int   dims_t[use_sub ? sub_dim : 1];
            Pair* pair_buf = (Pair*)malloc((size_t)BATCH * sizeof(Pair));
            #pragma omp for schedule(static)
            for (int t = 0; t < n_trees; t++) {
                uint64_t ts = tree_seed(t + g_tree_seed_offset);
                if (bmed8)
                    traversal_set_medians8(
                        bmed8 + (size_t)(t + g_tree_seed_offset) * bmed_nodes,
                        bmed8_scales
                            + (size_t)(t + g_tree_seed_offset) * 2 * bmed_depth,
                        bmed_depth);
                else
                    traversal_set_medians(
                        bmed_table ? bmed_table + (size_t)(t + g_tree_seed_offset) * bmed_nodes : NULL,
                        bmed_depth);
                for (int bb = 0; bb < batch_n; bb++) {
                    const float* vb = vec_buf + (size_t)bb * dim;
                    int32_t leaf;
                    if (use_sub) {
                        leaf = hp_cache_v0
                            ? traverse_sub_cached(vb, dim, sub_dim, depth, ts,
                                                  hp_cache_v0[t], hp_cache_v1[t],
                                                  dims_t)
                            : traverse_sub(vb, dim, sub_dim, depth, ts,
                                           v0_t, v1_t, dims_t);
                    } else {
                        leaf = traverse(vb, dim, depth, ts, v0_t, v1_t);
                    }
                    pair_buf[bb].leaf_id = leaf;
                    pair_buf[bb].doc_id  = doc_id_base + batch_doc_ids[bb];
                }
                /* Single fwrite per (thread, tree) per batch -- was one
                   per doc previously (~65k calls / batch at BATCH=256).   */
                fwrite(pair_buf, sizeof(Pair), (size_t)batch_n, trees[t]);
            }
            free(pair_buf);
        }

        doc_id += batch_n;
        if (doc_id % 50000 < BATCH || doc_id == n_vecs) {
            double dt = now_s() - t0;
            fprintf(stderr,
                    "  build: %7d / %d  (%.1f vec/s, peak RSS %ld KB)\n",
                    doc_id, n_vecs,
                    doc_id / (dt > 0 ? dt : 1.0),
                    peak_rss_kb());
        }
        /* Checkpoint: flush all pair files and update progress.txt every
           progress_chunk vecs (and at end). Crash-resume reads this counter. */
        if ((doc_id / progress_chunk) !=
            ((doc_id - batch_n) / progress_chunk) || doc_id == n_vecs) {
            for (int t = 0; t < n_trees; t++) {
                fflush(trees[t]);
                fsync(fileno(trees[t]));
            }
            char tmp_progress[640];
            snprintf(tmp_progress, sizeof(tmp_progress), "%s.tmp", progress_path);
            FILE* pf = fopen(tmp_progress, "w");
            if (pf) {
                fprintf(pf, "%d\n", doc_id);
                fflush(pf); fsync(fileno(pf)); fclose(pf);
                rename(tmp_progress, progress_path);
            }
        }

        /* Doubling recalibration trigger : pause build, snapshot calibration
           GT, invoke external calibrator subprocess, then resume. Design =
           see [[autotune-doubling]]. Simple system() call — blocks the build
           thread until calibration exits. During the pause, upstream producer
           can write to a WAL disk queue (drain-on-resume handled at ingest
           layer, out of scope for now : we build from a static file). */
        if (g_recalib_doubling && doc_id >= next_recalib) {
            fprintf(stderr, "  [recalib doubling] threshold %lld reached "
                            "at doc_id=%d — pausing build for calibration\n",
                    (long long)next_recalib, doc_id);
            /* Flush all trees + progress so the calibrator sees a consistent
               partial index. */
            for (int t = 0; t < n_trees; t++) {
                fflush(trees[t]); fsync(fileno(trees[t]));
            }
            if (calib_is_enabled()) calib_snapshot(index_dir);

            char cmd[2048];
            const char* script = g_recalib_script
                ? g_recalib_script
                : "/home/chatelet/mangrove-search/scripts/mangrove_calibrate.py";
            snprintf(cmd, sizeof(cmd),
                     "python3 -u %s --index %s --n-docs %d 2>&1",
                     script, index_dir, doc_id);
            double rc0 = now_s();
            int rc = system(cmd);
            fprintf(stderr, "  [recalib doubling] returned rc=%d in %.1fs "
                            "— resuming build\n", rc, now_s() - rc0);
            next_recalib *= 2;
        }
    }

    for (int t = 0; t < n_trees; t++) fclose(trees[t]);
    free(trees); fclose(fin);
    free(vec); free(u8_buf);
    free(vec_buf); free(u8_batch); free(batch_doc_ids); free(batch_ok);
    /* Final calibration snapshot + cleanup. */
    if (calib_is_enabled()) {
        calib_snapshot(index_dir);
        fprintf(stderr, "  calib snapshot : %s/calibration_{queries,gt}.bin\n",
                index_dir);
        calib_free();
    }
    if (hp_cache_v0) {
        for (int t = 0; t < n_trees; t++) free(hp_cache_v0[t]);
        free(hp_cache_v0);
    }
    if (hp_cache_v1) {
        for (int t = 0; t < n_trees; t++) free(hp_cache_v1[t]);
        free(hp_cache_v1);
    }

    /* Phase 1 complete: remove progress file. Phase 2 then takes over. */
    unlink(progress_path);

    fprintf(stderr, "  build done in %.2fs, peak RSS %ld KB\n",
            now_s() - t0, peak_rss_kb());
    (void)max_n;
    return 0;
}

static int cmp_pair_leaf_doc(const void* a, const void* b) {
    const Pair* p1 = (const Pair*)a;
    const Pair* p2 = (const Pair*)b;
    if (p1->leaf_id != p2->leaf_id)
        return (p1->leaf_id < p2->leaf_id) ? -1 : 1;
    return (p1->doc_id < p2->doc_id) ? -1 : (p1->doc_id > p2->doc_id);
}

/* Bucket-partitioned external sort: handles arbitrary depth and n_docs with
   bounded RAM (~1 bucket size, default 64 MB). Disk transient ≈ 2× the
   pair-file size (bucket dump + sparse/data temps), released after the
   final .srt is written.

   Plan:
     Pass 1: route each pair into K bucket files based on (leaf >> shift).
     Pass 2: for each bucket in order, load into RAM, qsort by leaf_id then
             doc_id, emit (leaf_id, offset) entries to tmp_sparse and doc_ids
             to tmp_data.
     Pass 3: build samples by re-reading tmp_sparse; write final .srt as
             header + samples + tmp_sparse + sentinel + tmp_data.            */
int convert_tree_to_sorted(const char* pair_path, const char* sorted_path,
                           int depth) {
    FILE* fp = fopen(pair_path, "rb");
    if (!fp) { perror("convert: fopen pair"); return -1; }
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    rewind(fp);
    long n_pairs = file_size / (long)sizeof(Pair);
    if (n_pairs < 0) { fclose(fp); return -1; }

    uint32_t n_leaves_total = (uint32_t)leaf_count(depth);
    int32_t  base = leaf_base(depth);

    /* Choose bucket count K (power of 2) so each bucket holds ~64 MB. */
    const size_t target_bucket_bytes = 64ULL * 1024 * 1024;
    int K = 1;
    while ((size_t)K * target_bucket_bytes < (size_t)file_size && K < 4096) K *= 2;
    if ((uint32_t)K > n_leaves_total) K = (int)n_leaves_total;
    int klog2 = 0;
    while ((1 << klog2) < K) klog2++;
    int shift = depth - klog2;
    if (shift < 0) shift = 0;

    /* Open K bucket files (rb+ for write then read). */
    FILE** buckets = (FILE**)calloc((size_t)K, sizeof(FILE*));
    char tpath[768];
    for (int k = 0; k < K; k++) {
        snprintf(tpath, sizeof(tpath), "%s.b%03d.tmp", sorted_path, k);
        buckets[k] = fopen(tpath, "wb+");
        if (!buckets[k]) {
            perror("convert: fopen bucket");
            for (int j = 0; j < k; j++) fclose(buckets[j]);
            free(buckets); fclose(fp); return -1;
        }
    }

    /* Pass 1: route. */
    Pair pbuf[4096];
    long total_routed = 0;
    while (1) {
        size_t got = fread(pbuf, sizeof(Pair), 4096, fp);
        if (got == 0) break;
        for (size_t i = 0; i < got; i++) {
            uint32_t leaf = (uint32_t)(pbuf[i].leaf_id - base);
            if (leaf >= n_leaves_total) continue;
            int k = (int)(leaf >> shift);
            if (k < 0) k = 0;
            if (k >= K) k = K - 1;
            fwrite(&pbuf[i], sizeof(Pair), 1, buckets[k]);
            total_routed++;
        }
    }
    fclose(fp);

    /* Open temp output files for sparse_index entries and data. */
    char tmp_sparse_path[768], tmp_data_path[768];
    snprintf(tmp_sparse_path, sizeof(tmp_sparse_path), "%s.sparse.tmp", sorted_path);
    snprintf(tmp_data_path,   sizeof(tmp_data_path),   "%s.data.tmp",   sorted_path);
    FILE* tmp_sparse = fopen(tmp_sparse_path, "wb+");
    FILE* tmp_data   = fopen(tmp_data_path,   "wb+");
    if (!tmp_sparse || !tmp_data) {
        if (tmp_sparse) fclose(tmp_sparse);
        if (tmp_data)   fclose(tmp_data);
        for (int k = 0; k < K; k++) fclose(buckets[k]);
        free(buckets); return -1;
    }

    /* Pass 2: load each bucket in order, sort, emit.
       SRT3 layout: for each leaf, write [u32 first_doc] then VarByte(delta)
       for each remaining doc. SparseEntry.offset = BYTE offset in data block.
       SRT2 layout (--no-varbyte): raw uint32 doc_ids for every doc.
                                   SparseEntry.offset = DOC index (not bytes). */
    int use_v3 = !g_build_no_varbyte;
    uint32_t n_nonempty = 0;
    uint64_t data_bytes = 0;   /* V3 only : byte offset into encoded data block */
    uint64_t total_docs = 0;
    for (int k = 0; k < K; k++) {
        long bsz = ftell(buckets[k]);
        if (bsz <= 0) {
            fclose(buckets[k]);
            snprintf(tpath, sizeof(tpath), "%s.b%03d.tmp", sorted_path, k);
            unlink(tpath);
            continue;
        }
        long nb = bsz / (long)sizeof(Pair);
        rewind(buckets[k]);
        Pair* bp = (Pair*)malloc((size_t)nb * sizeof(Pair));
        if (!bp || fread(bp, sizeof(Pair), nb, buckets[k]) != (size_t)nb) {
            fprintf(stderr, "convert: bucket %d load failed\n", k);
            if (bp) free(bp);
            fclose(buckets[k]);
            snprintf(tpath, sizeof(tpath), "%s.b%03d.tmp", sorted_path, k);
            unlink(tpath);
            continue;
        }
        fclose(buckets[k]);
        snprintf(tpath, sizeof(tpath), "%s.b%03d.tmp", sorted_path, k);
        unlink(tpath);

        qsort(bp, (size_t)nb, sizeof(Pair), cmp_pair_leaf_doc);

        long i = 0;
        while (i < nb) {
            uint32_t leaf = (uint32_t)(bp[i].leaf_id - base);
            /* V3 : offset = byte offset. V2 : offset = doc index. */
            uint32_t off_field = use_v3 ? (uint32_t)data_bytes : (uint32_t)total_docs;
            SparseEntry entry = { leaf, off_field };
            fwrite(&entry, sizeof(SparseEntry), 1, tmp_sparse);
            n_nonempty++;

            /* First doc of the leaf: raw u32 (both V2 and V3). */
            uint32_t first_doc = (uint32_t)bp[i].doc_id;
            fwrite(&first_doc, sizeof(uint32_t), 1, tmp_data);
            if (use_v3) data_bytes += 4;
            total_docs++;
            uint32_t prev = first_doc;

            long j = i + 1;
            while (j < nb && bp[j].leaf_id == bp[i].leaf_id) {
                uint32_t doc = (uint32_t)bp[j].doc_id;
                if (use_v3) {
                    uint8_t  vbuf[8];
                    size_t   vpos = 0;
                    varbyte_encode_u32(vbuf, &vpos, doc - prev);
                    fwrite(vbuf, 1, vpos, tmp_data);
                    data_bytes += vpos;
                } else {
                    /* V2: raw uint32 doc_id, no delta encoding. */
                    fwrite(&doc, sizeof(uint32_t), 1, tmp_data);
                }
                total_docs++;
                prev = doc;
                j++;
            }
            i = j;
        }
        free(bp);
    }
    free(buckets);
    (void)total_routed;

    if (total_docs > 0xFFFFFFFFu) {
        fprintf(stderr, "convert: total_docs %llu exceeds uint32\n",
                (unsigned long long)total_docs);
        fclose(tmp_sparse); fclose(tmp_data);
        unlink(tmp_sparse_path); unlink(tmp_data_path);
        return -1;
    }
    if (data_bytes > 0xFFFFFFFFu) {
        fprintf(stderr, "convert: data_bytes %llu exceeds uint32 — split tree or use larger offset\n",
                (unsigned long long)data_bytes);
        fclose(tmp_sparse); fclose(tmp_data);
        unlink(tmp_sparse_path); unlink(tmp_data_path);
        return -1;
    }

    /* Adaptive stride. */
    uint32_t stride = (n_nonempty + 65535u) / 65536u;
    if (stride < 512u) stride = 512u;
    uint32_t n_samples = (n_nonempty + stride - 1) / stride;

    /* Build samples by re-walking tmp_sparse. uint32 leaf_ids. */
    uint32_t* samples = (uint32_t*)malloc((size_t)(n_samples ? n_samples : 1)
                                          * sizeof(uint32_t));
    rewind(tmp_sparse);
    {
        SparseEntry entry;
        uint32_t j = 0;
        for (uint32_t i = 0; i < n_nonempty; i++) {
            if (fread(&entry, sizeof(SparseEntry), 1, tmp_sparse) != 1) break;
            if ((i % stride) == 0 && j < n_samples)
                samples[j++] = entry.leaf_id;
        }
    }

    /* Write final .srt via atomic tmp+rename: header + samples + sparse +
       sentinel + data. Crash mid-write leaves only the .tmp visible.       */
    char out_tmp_path[800];
    snprintf(out_tmp_path, sizeof(out_tmp_path), "%s.tmp", sorted_path);
    FILE* out = fopen(out_tmp_path, "wb");
    if (!out) {
        perror("convert: fopen out.tmp");
        free(samples);
        fclose(tmp_sparse); fclose(tmp_data);
        unlink(tmp_sparse_path); unlink(tmp_data_path);
        return -1;
    }
    uint32_t hdr[6] = {
        use_v3 ? SRT_MAGIC_V3 : SRT_MAGIC_V2,
        (uint32_t)depth, n_nonempty,
        (uint32_t)total_docs, stride,
        use_v3 ? (uint32_t)data_bytes : 0u
    };
    fwrite(hdr, sizeof(uint32_t), 6, out);
    fwrite(samples, sizeof(uint32_t), n_samples, out);
    free(samples);

    /* Stream tmp_sparse into out. SparseEntry is 8 bytes (uint32+uint32). */
    rewind(tmp_sparse);
    {
        SparseEntry buf[1024];
        long remaining = (long)n_nonempty;
        while (remaining > 0) {
            long take = (remaining > 1024) ? 1024 : remaining;
            if (fread(buf, sizeof(SparseEntry), (size_t)take, tmp_sparse) != (size_t)take)
                break;
            fwrite(buf, sizeof(SparseEntry), (size_t)take, out);
            remaining -= take;
        }
    }
    /* Sentinel : V3 → offset = total bytes ; V2 → offset = total docs. */
    uint32_t end_off = use_v3 ? (uint32_t)data_bytes : (uint32_t)total_docs;
    SparseEntry sentinel = { 0xFFFFFFFFu, end_off };
    fwrite(&sentinel, sizeof(SparseEntry), 1, out);

    /* Stream tmp_data (VarByte bytes for V3, raw uint32 for V2). */
    rewind(tmp_data);
    {
        uint8_t buf[64 * 1024];
        /* V3 : data_bytes ; V2 : total_docs × 4 bytes (raw uint32 per doc). */
        uint64_t remaining = use_v3 ? data_bytes : (uint64_t)total_docs * 4u;
        while (remaining > 0) {
            size_t take = (remaining > sizeof(buf)) ? sizeof(buf) : (size_t)remaining;
            if (fread(buf, 1, take, tmp_data) != take) break;
            fwrite(buf, 1, take, out);
            remaining -= take;
        }
    }

    fclose(tmp_sparse); fclose(tmp_data);
    unlink(tmp_sparse_path); unlink(tmp_data_path);

    /* fflush+fsync, append xxhash64 footer, then atomic rename. */
    fflush(out);
    int outfd = fileno(out);
    if (outfd >= 0) fsync(outfd);
    fclose(out);
    if (srt_finalize_with_hash(out_tmp_path) != 0) {
        fprintf(stderr, "convert: hash finalize failed for %s\n", out_tmp_path);
        unlink(out_tmp_path);
        return -1;
    }
    if (rename(out_tmp_path, sorted_path) != 0) {
        perror("convert: rename .tmp -> .srt");
        unlink(out_tmp_path);
        return -1;
    }

    unlink(pair_path);
    return 0;
}

int convert_all_to_sorted(const char* index_dir, int n_trees, int depth) {
    double t0 = now_s();
    int rc = 0;
    int skipped = 0;
    for (int t = 0; t < n_trees; t++) {
        char pair_path[512], srt_path[512];
        snprintf(pair_path, sizeof(pair_path), "%s/tree%05d.bin", index_dir, t + g_tree_seed_offset);
        snprintf(srt_path,  sizeof(srt_path),  "%s/tree%05d.srt", index_dir, t + g_tree_seed_offset);
        /* Resume: skip trees whose .srt already exists (atomic tmp+rename
           guarantees existence => fully written and hashed).             */
        struct stat sst;
        if (stat(srt_path, &sst) == 0 && sst.st_size > 0) {
            unlink(pair_path);  /* free pair file if still around */
            skipped++;
            if ((t + 1) % 64 == 0 || t + 1 == n_trees) {
                fprintf(stderr, "  convert: %4d / %d  (skipped=%d)\n",
                        t + 1, n_trees, skipped);
            }
            continue;
        }
        if (convert_tree_to_sorted(pair_path, srt_path, depth) != 0) {
            fprintf(stderr, "convert failed on %s\n", pair_path);
            rc = -1; break;
        }
        if ((t + 1) % 64 == 0 || t + 1 == n_trees) {
            fprintf(stderr,
                    "  convert: %4d / %d  (peak RSS %ld KB)\n",
                    t + 1, n_trees, peak_rss_kb());
        }
    }
    fprintf(stderr, "  convert done in %.2fs, peak RSS %ld KB\n",
            now_s() - t0, peak_rss_kb());
    return rc;
}

/* Read doc's vector from base into `out` (float32). Handles fvecs / bvecs / u8bin
   via vec_format helpers. Returns 0 on success.                            */
static int read_doc_vec(int bfd, VecFmt fmt, int doc_id, int dim,
                        float* out, uint8_t* u8scratch) {
    off_t off = vec_row_offset(fmt, doc_id, dim);
    if (fmt == VECFMT_U8BIN || fmt == VECFMT_BVECS) {
        if (pread(bfd, u8scratch, (size_t)dim, off) != (ssize_t)dim)
            return -1;
        vec_u8_to_f32(u8scratch, out, dim);
        return 0;
    }
    if (fmt == VECFMT_F16BIN) {
        /* scratch local : les appelants n'allouent que dim octets */
        uint16_t h16[1024];
        if (dim > 1024) return -1;
        if (pread(bfd, h16, (size_t)dim * 2, off) != (ssize_t)(dim * 2))
            return -1;
        vec_f16_to_f32(h16, out, dim);
        return 0;
    }
    if (pread(bfd, out, (size_t)dim * 4, off) != (ssize_t)(dim * 4))
        return -1;
    return 0;
}

/* Compact (deepen) one tree: read old .srt, for each leaf decode its docs,
   continue traversal from leaf's global node by `n_extra` levels using each
   doc's vector, emit a pair file (new_global_node, doc_id) ready for convert.*/
static int deepen_one_tree(int tree_idx,
                           const char* old_srt_path, const char* new_pair_path,
                           int bfd, VecFmt bfmt,
                           int dim, int sub_dim,
                           int depth_old, int depth_new, int append) {
    int use_sub = (sub_dim > 0 && sub_dim <= dim);
    int sd = use_sub ? sub_dim : dim;
    int n_extra = depth_new - depth_old;
    uint64_t ts = tree_seed(tree_idx);

    int fd = open(old_srt_path, O_RDONLY);
    if (fd < 0) { perror("compact: open old .srt"); return -1; }

    uint32_t hdr[6];
    if (pread(fd, hdr, sizeof(hdr), 0) != (ssize_t)sizeof(hdr)) {
        fprintf(stderr, "compact: bad header in %s\n", old_srt_path);
        close(fd); return -1;
    }
    if (hdr[0] != 0x53525433u /* SRT3 */ && hdr[0] != 0x53525432u /* SRT2 */) {
        fprintf(stderr, "compact: bad magic 0x%08x in %s\n", hdr[0], old_srt_path);
        close(fd); return -1;
    }
    int is_v3              = (hdr[0] == 0x53525433u);
    uint32_t depth_in_file = hdr[1];
    uint32_t n_nonempty    = hdr[2];
    uint32_t sample_stride = hdr[4];
    /* hdr[3] (total_docs) and hdr[5] (data_bytes) are present in the
       header but unused in this code path. */
    if ((int)depth_in_file != depth_old) {
        fprintf(stderr, "compact: depth mismatch %u != %d\n",
                depth_in_file, depth_old);
        close(fd); return -1;
    }
    uint32_t n_samples = (n_nonempty + sample_stride - 1) / sample_stride;
    uint64_t index_base = 24 + (uint64_t)n_samples * 4;
    uint64_t data_base  = index_base + (uint64_t)(n_nonempty + 1) * 8;

    FILE* out = fopen(new_pair_path, append ? "ab" : "wb");
    if (!out) { perror("compact: fopen pair out"); close(fd); return -1; }

    int32_t base_old = leaf_base(depth_old);

    /* Stream sparse_index entries: read in chunks. Pair each with next entry's
       offset to delimit the leaf's data block, then decode & deepen.       */
    const uint32_t CHUNK = 4096;
    SparseEntry* sbuf = (SparseEntry*)malloc(sizeof(SparseEntry) * (CHUNK + 1));
    uint8_t*     dbuf = (uint8_t*)malloc(64 * 1024);  /* per-leaf bytes scratch */
    size_t       dcap = 64 * 1024;
    float*       vec  = (float*)malloc((size_t)dim * sizeof(float));
    float        v0[1024], v1[1024];
    int          dims_buf[256];
    uint8_t*     u8scratch = (uint8_t*)malloc((size_t)dim);
    if (!sbuf || !dbuf || !vec || (use_sub && sub_dim > 256) || sd > 1024) {
        free(sbuf); free(dbuf); free(vec); free(u8scratch);
        fclose(out); close(fd); return -1;
    }

    uint64_t processed = 0;
    uint32_t pos = 0;  /* sparse_index entry position */
    while (pos < n_nonempty) {
        /* Read [pos, pos+CHUNK] + 1 sentinel into sbuf */
        uint32_t take = (n_nonempty - pos > CHUNK) ? CHUNK : (n_nonempty - pos);
        uint32_t read_n = take + 1;  /* +1 for next-offset of last entry */
        if (pos + read_n > n_nonempty + 1) read_n = n_nonempty + 1 - pos;
        off_t off = (off_t)(index_base + (uint64_t)pos * 8);
        if (pread(fd, sbuf, read_n * 8, off) != (ssize_t)(read_n * 8)) {
            fprintf(stderr, "compact: sparse_index pread fail at pos %u\n", pos);
            break;
        }
        /* For each entry in [0, take), process the leaf. */
        for (uint32_t k = 0; k < take; k++) {
            uint32_t leaf_id_rel = sbuf[k].leaf_id;
            uint32_t b_start     = sbuf[k].offset;
            uint32_t b_end       = sbuf[k + 1].offset;
            if (b_end <= b_start) continue;
            uint32_t b_len = b_end - b_start;
            if (b_len > dcap) {
                free(dbuf);
                dcap = b_len * 2;
                dbuf = (uint8_t*)malloc(dcap);
                if (!dbuf) { fprintf(stderr, "compact: dbuf alloc fail\n"); break; }
            }
            if (pread(fd, dbuf, b_len, (off_t)(data_base + b_start))
                != (ssize_t)b_len) {
                fprintf(stderr, "compact: data pread fail\n"); break;
            }

            int32_t global_node = (int32_t)leaf_id_rel + base_old;

            /* Decode docs: SRT3 = u32 first + varbyte deltas; SRT2 = u32 array */
            if (is_v3) {
                if (b_len < 4) continue;
                uint32_t prev;
                memcpy(&prev, dbuf, 4);
                /* Process doc */
                if (read_doc_vec(bfd, bfmt, (int)prev, dim, vec, u8scratch) == 0) {
                    normalize(vec, dim);
                    int32_t new_node = use_sub
                        ? traverse_sub_continue(vec, dim, sub_dim,
                                                depth_old, n_extra,
                                                global_node, ts, v0, v1, dims_buf)
                        : traverse_continue(vec, dim,
                                            depth_old, n_extra,
                                            global_node, ts, v0, v1);
                    Pair p = { new_node, (int32_t)prev };
                    fwrite(&p, sizeof(Pair), 1, out);
                    processed++;
                }
                size_t bpos = 4;
                while (bpos < b_len) {
                    uint32_t delta = 0; int shift = 0; uint8_t bb;
                    do {
                        bb = dbuf[bpos++];
                        delta |= (uint32_t)(bb & 0x7Fu) << shift;
                        shift += 7;
                    } while (bb & 0x80u);
                    prev += delta;
                    if (read_doc_vec(bfd, bfmt, (int)prev, dim, vec, u8scratch) == 0) {
                        normalize(vec, dim);
                        int32_t new_node = use_sub
                            ? traverse_sub_continue(vec, dim, sub_dim,
                                                    depth_old, n_extra,
                                                    global_node, ts, v0, v1, dims_buf)
                            : traverse_continue(vec, dim,
                                                depth_old, n_extra,
                                                global_node, ts, v0, v1);
                        Pair p = { new_node, (int32_t)prev };
                        fwrite(&p, sizeof(Pair), 1, out);
                        processed++;
                    }
                }
            } else {
                /* SRT2: dbuf is uint32[doc_count] */
                uint32_t doc_count = b_len / 4;
                for (uint32_t i = 0; i < doc_count; i++) {
                    uint32_t doc_id;
                    memcpy(&doc_id, dbuf + i * 4, 4);
                    if (read_doc_vec(bfd, bfmt, (int)doc_id, dim, vec, u8scratch) == 0) {
                        normalize(vec, dim);
                        int32_t new_node = use_sub
                            ? traverse_sub_continue(vec, dim, sub_dim,
                                                    depth_old, n_extra,
                                                    global_node, ts, v0, v1, dims_buf)
                            : traverse_continue(vec, dim,
                                                depth_old, n_extra,
                                                global_node, ts, v0, v1);
                        Pair p = { new_node, (int32_t)doc_id };
                        fwrite(&p, sizeof(Pair), 1, out);
                        processed++;
                    }
                }
            }
        }
        pos += take;
    }
    fflush(out); fclose(out);
    close(fd);
    free(sbuf); free(dbuf); free(vec); free(u8scratch);
    (void)processed;
    return 0;
}

int compact_multi_deeper(const char* index_new,
                         const char* const* index_olds, int n_olds,
                         const char* base_path,
                         int n_trees, int depth_old, int depth_new,
                         int dim, int sub_dim) {
    if (depth_new <= depth_old) {
        fprintf(stderr, "compact: depth_new (%d) must be > depth_old (%d)\n",
                depth_new, depth_old);
        return -1;
    }
    if (raise_fd_limit(n_trees + 16) != 0) {
        fprintf(stderr, "warning: could not raise fd limit\n");
    }
    mkdir(index_new, 0755);
    int bfd = open(base_path, O_RDONLY);
    if (bfd < 0) { perror("compact_multi: open base"); return -1; }
    VecFmt bfmt = vec_fmt_from_path(base_path);

    double t0 = now_s();
    fprintf(stderr,
        "compact_multi: %d sources -> %s  depth %d -> %d  (%d extra levels)\n",
        n_olds, index_new, depth_old, depth_new, depth_new - depth_old);

    int rc = 0;
    for (int t = 0; t < n_trees; t++) {
        char new_p[600];
        snprintf(new_p, sizeof(new_p), "%s/tree%05d.bin", index_new, t);
        for (int s = 0; s < n_olds; s++) {
            char old_p[600];
            snprintf(old_p, sizeof(old_p), "%s/tree%05d.srt", index_olds[s], t);
            if (deepen_one_tree(t, old_p, new_p, bfd, bfmt,
                                dim, sub_dim, depth_old, depth_new,
                                s > 0 /* append after first */) != 0) {
                fprintf(stderr, "compact_multi: tree %d source %d failed\n", t, s);
                rc = -1; break;
            }
        }
        if (rc != 0) break;
        if ((t + 1) % 64 == 0 || t + 1 == n_trees) {
            fprintf(stderr, "  compact_multi phase1: %4d / %d  (%.2fs)\n",
                    t + 1, n_trees, now_s() - t0);
        }
    }
    close(bfd);
    if (rc != 0) return rc;

    fprintf(stderr, "  compact_multi phase1 done in %.2fs, converting ...\n",
            now_s() - t0);
    return convert_all_to_sorted(index_new, n_trees, depth_new);
}

int compact_segment_deeper(const char* index_old, const char* index_new,
                           const char* base_path,
                           int n_trees, int depth_old, int depth_new,
                           int dim, int sub_dim) {
    if (depth_new <= depth_old) {
        fprintf(stderr, "compact: depth_new (%d) must be > depth_old (%d)\n",
                depth_new, depth_old);
        return -1;
    }
    if (raise_fd_limit(n_trees + 16) != 0) {
        fprintf(stderr, "warning: could not raise fd limit\n");
    }
    mkdir(index_new, 0755);

    int bfd = open(base_path, O_RDONLY);
    if (bfd < 0) { perror("compact: open base"); return -1; }
    VecFmt bfmt = vec_fmt_from_path(base_path);
    /* Skip u8bin header (8 B) - vec_row_offset already accounts for it. */

    double t0 = now_s();
    fprintf(stderr,
        "compact: %s -> %s  depth %d -> %d  (%d extra levels)\n",
        index_old, index_new, depth_old, depth_new, depth_new - depth_old);

    int rc = 0;
    for (int t = 0; t < n_trees; t++) {
        char old_p[600], new_p[600];
        snprintf(old_p, sizeof(old_p), "%s/tree%05d.srt", index_old, t);
        snprintf(new_p, sizeof(new_p), "%s/tree%05d.bin", index_new, t);
        if (deepen_one_tree(t, old_p, new_p, bfd, bfmt,
                            dim, sub_dim, depth_old, depth_new, 0) != 0) {
            fprintf(stderr, "compact: deepen tree %d failed\n", t);
            rc = -1; break;
        }
        if ((t + 1) % 64 == 0 || t + 1 == n_trees) {
            fprintf(stderr, "  compact phase1: %4d / %d  (%.2fs, peak RSS %ld KB)\n",
                    t + 1, n_trees, now_s() - t0, peak_rss_kb());
        }
    }
    close(bfd);
    if (rc != 0) return rc;

    fprintf(stderr, "  compact phase1 done in %.2fs, converting to .srt ...\n",
            now_s() - t0);
    return convert_all_to_sorted(index_new, n_trees, depth_new);
}
