/* HOT → SRT V2 per-tree compaction. Streaming leaf-by-leaf so RAM stays
 * bounded (only sparse index of one tree + per-leaf buffer in RAM). */
#define _POSIX_C_SOURCE 200809L
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hot_compact.h"
#include "sorted_store.h"
#include "varbyte.h"

static int u32_cmp(const void* a, const void* b) {
    uint32_t x = *(const uint32_t*)a, y = *(const uint32_t*)b;
    return (x > y) - (x < y);
}

/* Read one leaf's docs from MAIN, decoding V3 if needed.
 * Returns count in *out_n. */
static int read_main_leaf(int fd, int is_v3,
                          uint64_t data_base,
                          uint32_t off_now, uint32_t off_next,
                          uint32_t* out_docs, uint32_t out_cap, uint32_t* out_n) {
    if (off_next <= off_now) { *out_n = 0; return 0; }
    if (is_v3) {
        uint32_t nb = off_next - off_now;
        if (nb > 8 * 1024 * 1024) return -1;   /* sanity : 8 MB per leaf max */
        uint8_t* buf = (uint8_t*)malloc(nb);
        if (!buf) return -1;
        if (pread(fd, buf, nb, (off_t)(data_base + off_now)) != (ssize_t)nb) {
            free(buf); return -1;
        }
        size_t pos = 0;
        uint32_t first = 0; /* first doc raw uint32 */
        memcpy(&first, buf + pos, 4); pos += 4;
        uint32_t n = 0;
        if (out_cap > 0) out_docs[n++] = first;
        uint32_t prev = first;
        while (pos < nb && n < out_cap) {
            uint32_t d = varbyte_decode_u32(buf, &pos);
            prev += d;
            out_docs[n++] = prev;
        }
        free(buf);
        *out_n = n;
        return 0;
    } else {
        uint32_t n = off_next - off_now;
        if (n > out_cap) return -1;
        if (pread(fd, out_docs, (size_t)n * 4u, (off_t)(data_base + (uint64_t)off_now * 4u))
            != (ssize_t)(n * 4u)) return -1;
        *out_n = n;
        return 0;
    }
}

int hot_compact_tree(const char* main_path,
                     const HotSnapEntry* hot_snap, int hot_n,
                     const char* out_path,
                     int out_format,
                     int* out_new_n_nonempty,
                     uint64_t* out_new_total_docs) {
    if (out_format != 2 && out_format != 3) return -1;
    int out_v3 = (out_format == 3);
    /* Open MAIN. */
    int rfd = open(main_path, O_RDONLY);
    if (rfd < 0) { perror("hot_compact: open main"); return -1; }
    uint32_t hdr[6];
    if (pread(rfd, hdr, sizeof(hdr), 0) != (ssize_t)sizeof(hdr)) {
        close(rfd); return -1;
    }
    int is_v3 = (hdr[0] == SRT_MAGIC_V3);
    if (!is_v3 && hdr[0] != SRT_MAGIC_V2) {
        fprintf(stderr, "hot_compact: bad magic 0x%08x\n", hdr[0]);
        close(rfd); return -1;
    }
    uint32_t depth       = hdr[1];
    uint32_t n_nonempty  = hdr[2];
    uint32_t total_docs  = hdr[3];
    uint32_t stride      = hdr[4];
    uint32_t data_bytes  = is_v3 ? hdr[5] : 0u;
    uint32_t n_samples   = (n_nonempty + stride - 1) / stride;
    uint64_t index_base  = SRT_HEADER_BYTES + (uint64_t)n_samples * 4u;
    uint64_t data_base   = index_base + (uint64_t)(n_nonempty + 1) * SRT_INDEX_ENTRY;
    (void)total_docs; (void)data_bytes;

    /* Load MAIN sparse index into RAM. Size = (n_nonempty + 1) * 8 B. */
    size_t sparse_bytes = (size_t)(n_nonempty + 1) * SRT_INDEX_ENTRY;
    SparseEntry* main_sparse = (SparseEntry*)malloc(sparse_bytes);
    if (!main_sparse) { close(rfd); return -1; }
    if (pread(rfd, main_sparse, sparse_bytes, (off_t)index_base)
        != (ssize_t)sparse_bytes) {
        free(main_sparse); close(rfd); return -1;
    }

    /* Two-pointer merge. Pass 1 : compute new n_nonempty + total_docs. */
    uint32_t new_n = 0;
    uint64_t new_total = 0;
    uint32_t im = 0, ih = 0;
    while (im < n_nonempty || ih < (uint32_t)hot_n) {
        uint32_t lid_m = (im < n_nonempty)       ? main_sparse[im].leaf_id : 0xFFFFFFFFu;
        uint32_t lid_h = (ih < (uint32_t)hot_n)  ? hot_snap[ih].leaf_id  : 0xFFFFFFFFu;
        uint32_t lid = (lid_m < lid_h) ? lid_m : lid_h;
        uint32_t main_docs = 0, hot_docs = 0;
        if (lid_m == lid) {
            uint32_t off_now  = main_sparse[im].offset;
            uint32_t off_next = main_sparse[im + 1].offset;
            /* V2 offset = doc units, V3 offset = bytes (variable per doc) →
               we don't know count from bytes alone in V3 without decoding.
               For V3, count only accurate after decode ; here we bound via
               "at most (nb + 3) / 1" but that's loose. Just decode. */
            if (is_v3) {
                /* Peek : count docs by scanning varbyte. */
                if (off_next > off_now) {
                    uint32_t nb = off_next - off_now;
                    uint8_t* buf = (uint8_t*)malloc(nb);
                    if (buf) {
                        if (pread(rfd, buf, nb, (off_t)(data_base + off_now)) == (ssize_t)nb) {
                            size_t pos = 4;   /* skip first_doc raw uint32 */
                            main_docs = 1;
                            while (pos < nb) { varbyte_decode_u32(buf, &pos); main_docs++; }
                        }
                        free(buf);
                    }
                }
            } else {
                main_docs = off_next - off_now;
            }
            im++;
        }
        if (lid_h == lid) {
            hot_docs = hot_snap[ih].n_docs;
            ih++;
        }
        /* Union dedup happens per-leaf ; count is at MOST main_docs + hot_docs. */
        uint32_t leaf_max = main_docs + hot_docs;
        if (leaf_max > 0) {
            new_n++;
            new_total += leaf_max;   /* upper bound — actual after dedup ≤ this */
        }
    }

    /* Pass 2 : write output. Two-file scheme : write header + index later
       (after we know exact new_n and offsets), stream data into a temp part.
       Simpler : allocate output file, seek to data section, stream data,
       backpatch header + sparse. */
    int wfd = open(out_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (wfd < 0) { perror("hot_compact: open out"); free(main_sparse); close(rfd); return -1; }

    uint32_t new_n_samples = (new_n + stride - 1) / stride;
    uint64_t new_index_base = SRT_HEADER_BYTES + (uint64_t)new_n_samples * 4u;
    uint64_t new_data_base  = new_index_base + (uint64_t)(new_n + 1) * SRT_INDEX_ENTRY;

    /* Allocate sparse array + samples in RAM. */
    SparseEntry* new_sparse = (SparseEntry*)malloc((size_t)(new_n + 1) * SRT_INDEX_ENTRY);
    uint32_t*    new_samples = new_n_samples ? (uint32_t*)malloc((size_t)new_n_samples * 4u) : NULL;
    if (!new_sparse || (new_n_samples && !new_samples)) {
        free(new_sparse); free(new_samples); free(main_sparse); close(rfd); close(wfd); return -1;
    }

    /* Per-leaf buffer : up to leaf_max docs. Sized for the largest expected
       leaf. Grow as needed. */
    uint32_t buf_cap = 4096;
    uint32_t* leaf_buf = (uint32_t*)malloc((size_t)buf_cap * sizeof(uint32_t));
    if (!leaf_buf) {
        free(new_sparse); free(new_samples); free(main_sparse); close(rfd); close(wfd);
        return -1;
    }

    im = ih = 0;
    uint32_t out_i = 0;               /* new sparse index cursor */
    uint32_t out_doc_off = 0;         /* running doc-unit offset (V2)          */
    uint32_t out_byte_off = 0;        /* running byte offset in data block (V3) */
    /* V3 encode scratch : 4 B first_doc + up to 5 B per delta. */
    uint32_t vb_cap = 65536;
    uint8_t* vb_buf = (uint8_t*)malloc(vb_cap);
    if (!vb_buf) {
        free(new_sparse); free(new_samples); free(main_sparse); close(rfd); close(wfd); return -1;
    }
    while (im < n_nonempty || ih < (uint32_t)hot_n) {
        uint32_t lid_m = (im < n_nonempty)       ? main_sparse[im].leaf_id : 0xFFFFFFFFu;
        uint32_t lid_h = (ih < (uint32_t)hot_n)  ? hot_snap[ih].leaf_id  : 0xFFFFFFFFu;
        uint32_t lid = (lid_m < lid_h) ? lid_m : lid_h;

        uint32_t main_n = 0, hot_docs_n = 0;
        const uint32_t* hot_docs_p = NULL;

        /* Load MAIN docs for this leaf if present. */
        if (lid_m == lid) {
            uint32_t off_now  = main_sparse[im].offset;
            uint32_t off_next = main_sparse[im + 1].offset;
            /* Grow leaf_buf if needed (upper bound = leaf byte size). */
            uint32_t upper = is_v3 ? (off_next - off_now) : (off_next - off_now);
            if (upper + hot_snap[ih].n_docs > buf_cap) {
                while (buf_cap < upper + hot_snap[ih].n_docs + 8) buf_cap *= 2;
                uint32_t* nb = (uint32_t*)realloc(leaf_buf, (size_t)buf_cap * sizeof(uint32_t));
                if (!nb) goto err;
                leaf_buf = nb;
            }
            if (read_main_leaf(rfd, is_v3, data_base, off_now, off_next,
                               leaf_buf, buf_cap, &main_n) != 0) goto err;
            im++;
        }
        /* HOT docs for this leaf. */
        if (lid_h == lid) {
            hot_docs_p = hot_snap[ih].docs;
            hot_docs_n = hot_snap[ih].n_docs;
            /* Ensure leaf_buf has room for MAIN + HOT. */
            if (main_n + hot_docs_n > buf_cap) {
                while (buf_cap < main_n + hot_docs_n) buf_cap *= 2;
                uint32_t* nb = (uint32_t*)realloc(leaf_buf, (size_t)buf_cap * sizeof(uint32_t));
                if (!nb) goto err;
                leaf_buf = nb;
            }
            memcpy(leaf_buf + main_n, hot_docs_p, (size_t)hot_docs_n * sizeof(uint32_t));
            ih++;
        }
        uint32_t merged_n = main_n + hot_docs_n;
        if (merged_n == 0) continue;

        /* Sort + dedupe. */
        qsort(leaf_buf, merged_n, sizeof(uint32_t), u32_cmp);
        uint32_t w = 1;
        for (uint32_t r = 1; r < merged_n; r++) {
            if (leaf_buf[r] != leaf_buf[w-1]) leaf_buf[w++] = leaf_buf[r];
        }
        merged_n = w;

        /* Emit sparse entry + write data. */
        new_sparse[out_i].leaf_id = lid;
        if (out_v3) {
            /* Encode leaf : [u32 first_doc][varbyte deltas...]. */
            uint32_t max_bytes = 4 + merged_n * 5u;
            if (max_bytes > vb_cap) {
                while (vb_cap < max_bytes) vb_cap *= 2;
                uint8_t* nb = (uint8_t*)realloc(vb_buf, vb_cap);
                if (!nb) goto err;
                vb_buf = nb;
            }
            size_t pos = 0;
            uint32_t first_doc = leaf_buf[0];
            memcpy(vb_buf, &first_doc, 4); pos += 4;
            uint32_t prev = first_doc;
            for (uint32_t r = 1; r < merged_n; r++) {
                varbyte_encode_u32(vb_buf, &pos, leaf_buf[r] - prev);
                prev = leaf_buf[r];
            }
            new_sparse[out_i].offset = out_byte_off;
            if (pwrite(wfd, vb_buf, pos,
                       (off_t)(new_data_base + out_byte_off))
                != (ssize_t)pos) goto err;
            out_byte_off += (uint32_t)pos;
        } else {
            new_sparse[out_i].offset = out_doc_off;
            if (pwrite(wfd, leaf_buf, (size_t)merged_n * 4u,
                       (off_t)(new_data_base + (uint64_t)out_doc_off * 4u))
                != (ssize_t)(merged_n * 4u)) goto err;
        }
        out_doc_off += merged_n;
        out_i++;
    }
    /* Sentinel. */
    new_sparse[out_i].leaf_id = 0xFFFFFFFFu;
    new_sparse[out_i].offset  = out_v3 ? out_byte_off : out_doc_off;

    /* Samples : leaf_id at every `stride`-th sparse entry. */
    for (uint32_t i = 0; i < new_n_samples; i++) {
        uint32_t idx = i * stride;
        if (idx >= out_i) idx = out_i - 1;
        new_samples[i] = new_sparse[idx].leaf_id;
    }

    /* Header. V3 stores data_bytes in hdr[5]; V2 zeros it. */
    uint32_t whdr[6] = {
        out_v3 ? SRT_MAGIC_V3 : SRT_MAGIC_V2,
        depth, out_i, out_doc_off, stride,
        out_v3 ? out_byte_off : 0u
    };
    if (pwrite(wfd, whdr, sizeof(whdr), 0) != (ssize_t)sizeof(whdr)) goto err;
    if (new_n_samples > 0) {
        if (pwrite(wfd, new_samples, (size_t)new_n_samples * 4u,
                   (off_t)SRT_HEADER_BYTES) != (ssize_t)(new_n_samples * 4u)) goto err;
    }
    if (pwrite(wfd, new_sparse, (size_t)(out_i + 1) * SRT_INDEX_ENTRY,
               (off_t)new_index_base) != (ssize_t)((out_i + 1) * SRT_INDEX_ENTRY)) goto err;
    fsync(wfd);

    if (out_new_n_nonempty) *out_new_n_nonempty = (int)out_i;
    if (out_new_total_docs) *out_new_total_docs = out_doc_off;
    free(vb_buf); free(leaf_buf); free(new_sparse); free(new_samples); free(main_sparse);
    close(rfd); close(wfd);
    return 0;

err:
    free(vb_buf); free(leaf_buf); free(new_sparse); free(new_samples); free(main_sparse);
    close(rfd); close(wfd);
    unlink(out_path);
    return -1;
}
