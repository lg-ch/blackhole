#define _POSIX_C_SOURCE 200809L
#include "slot_store.h"
#include "sorted_store.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

int slt_open_rdonly(SltStore* s, const char* path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    uint32_t hdr[8];
    if (pread(fd, hdr, sizeof(hdr), 0) != (ssize_t)sizeof(hdr)) {
        close(fd); return -1;
    }
    if (hdr[0] != SLT_MAGIC) { close(fd); return -1; }
    s->fd            = fd;
    s->magic         = hdr[0];
    s->depth         = hdr[1];
    s->n_nonempty    = hdr[2];
    s->total_docs    = hdr[3];
    s->n_classes     = hdr[4];
    s->sample_stride = hdr[5];
    if (s->n_classes != SLT_N_CLASSES) { close(fd); return -1; }

    off_t off = SLT_HEADER_BYTES;
    if (pread(fd, s->class_sizes, SLT_N_CLASSES * 4, off) != SLT_N_CLASSES * 4) {
        close(fd); return -1;
    }
    off += SLT_N_CLASSES * 4;
    if (pread(fd, s->class_n_slots, SLT_N_CLASSES * 4, off) != SLT_N_CLASSES * 4) {
        close(fd); return -1;
    }
    off += SLT_N_CLASSES * 4;

    s->n_samples = (s->n_nonempty + s->sample_stride - 1) / s->sample_stride;
    if (s->n_samples > 0) {
        s->sample_leaves = (uint32_t*)malloc((size_t)s->n_samples * 4u);
        if (!s->sample_leaves) { close(fd); return -1; }
        if (pread(fd, s->sample_leaves, (size_t)s->n_samples * 4u, off)
            != (ssize_t)(s->n_samples * 4u)) {
            free(s->sample_leaves); close(fd); return -1;
        }
    } else {
        s->sample_leaves = NULL;
    }
    off += (uint64_t)s->n_samples * 4u;

    s->index_base = (uint64_t)off;
    /* Sparse entries = n_nonempty × 8 B, no sentinel needed (all sizes known via class). */
    s->data_base = s->index_base + (uint64_t)s->n_nonempty * sizeof(SltSparseEntry);

    /* Precompute class_data_offset (cumulative bytes of allocated slots) */
    uint64_t cum = 0;
    for (int c = 0; c < SLT_N_CLASSES; c++) {
        s->class_data_offset[c] = cum;
        cum += (uint64_t)s->class_n_slots[c] * s->class_sizes[c];
    }
    return 0;
}

void slt_close(SltStore* s) {
    if (!s) return;
    if (s->sample_leaves) { free(s->sample_leaves); s->sample_leaves = NULL; }
    if (s->fd >= 0) { close(s->fd); s->fd = -1; }
}

uint32_t slt_sample_bucket(const SltStore* s, uint32_t leaf_id) {
    if (s->n_samples == 0) return 0;
    uint32_t lo = 0, hi = s->n_samples;
    while (lo < hi) {
        uint32_t mid = (lo + hi) / 2;
        if (s->sample_leaves[mid] <= leaf_id) lo = mid + 1;
        else                                  hi = mid;
    }
    return lo > 0 ? lo - 1 : 0;
}

/* Convert .srt V2 to .slt.
 * Two passes over the .srt : (1) classify each leaf, count slots per class.
 * (2) write .slt : header, class metadata, samples, sparse entries, data blocks. */
int slt_convert_from_srt_v2(const char* srt_path, const char* slt_path,
                            int* out_n_overflow) {
    /* Read .srt V2 header + sparse. */
    int rfd = open(srt_path, O_RDONLY);
    if (rfd < 0) { perror("slt_convert: open srt"); return -1; }
    uint32_t rhdr[6];
    if (pread(rfd, rhdr, sizeof(rhdr), 0) != sizeof(rhdr)) {
        close(rfd); return -1;
    }
    if (rhdr[0] != SRT_MAGIC_V2) {
        fprintf(stderr, "slt_convert: expected SRT V2 magic, got 0x%08x\n", rhdr[0]);
        close(rfd); return -1;
    }
    uint32_t depth = rhdr[1], n_nonempty = rhdr[2],
             total_docs = rhdr[3], stride = rhdr[4];
    uint32_t n_samples = (n_nonempty + stride - 1) / stride;
    /* Read sparse entries (uint32 leaf_id, uint32 offset) + sentinel */
    uint64_t sparse_off = SRT_HEADER_BYTES + (uint64_t)n_samples * 4u;
    size_t sparse_bytes = (size_t)(n_nonempty + 1) * 8u;
    uint32_t* srt_sparse = (uint32_t*)malloc(sparse_bytes);
    if (!srt_sparse) { close(rfd); return -1; }
    if (pread(rfd, srt_sparse, sparse_bytes, (off_t)sparse_off)
        != (ssize_t)sparse_bytes) {
        free(srt_sparse); close(rfd); return -1;
    }
    uint64_t data_base_r = sparse_off + sparse_bytes;

    /* Classify each leaf, count slots per class. */
    uint32_t class_count[SLT_N_CLASSES] = {0};
    uint8_t* leaf_class = (uint8_t*)malloc((size_t)n_nonempty);
    uint32_t* leaf_slot = (uint32_t*)malloc((size_t)n_nonempty * 4u);
    if (!leaf_class || !leaf_slot) {
        free(srt_sparse); free(leaf_class); free(leaf_slot);
        close(rfd); return -1;
    }
    int n_overflow = 0;
    for (uint32_t i = 0; i < n_nonempty; i++) {
        uint32_t off_now  = srt_sparse[i * 2 + 1];
        uint32_t off_next = srt_sparse[(i + 1) * 2 + 1];
        uint32_t size_bytes = (off_next - off_now) * 4u;  /* V2 offsets in doc-units */
        int cls = -1;
        for (int c = 0; c < SLT_N_CLASSES; c++) {
            if (size_bytes <= SLT_CLASS_SIZES[c]) { cls = c; break; }
        }
        if (cls < 0) {
            /* Overflow — assign to last class, will truncate data. */
            cls = SLT_N_CLASSES - 1;
            n_overflow++;
        }
        leaf_class[i] = (uint8_t)cls;
        leaf_slot[i]  = class_count[cls];
        class_count[cls]++;
    }
    if (out_n_overflow) *out_n_overflow = n_overflow;

    /* Compute file layout offsets. */
    uint64_t class_data_offset[SLT_N_CLASSES];
    uint64_t cum = 0;
    for (int c = 0; c < SLT_N_CLASSES; c++) {
        class_data_offset[c] = cum;
        cum += (uint64_t)class_count[c] * SLT_CLASS_SIZES[c];
    }
    uint64_t data_total = cum;

    /* Open output. Write header + class metadata + samples + sparse first,
       then seek to data section and stream data per class. */
    int wfd = open(slt_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (wfd < 0) { perror("slt_convert: open slt"); goto err; }

    uint32_t whdr[8] = {
        SLT_MAGIC, depth, n_nonempty, total_docs,
        SLT_N_CLASSES, stride, 0, 0
    };
    write(wfd, whdr, sizeof(whdr));
    /* class_sizes */
    for (int c = 0; c < SLT_N_CLASSES; c++) {
        uint32_t sz = SLT_CLASS_SIZES[c];
        write(wfd, &sz, 4);
    }
    /* class_n_slots */
    write(wfd, class_count, SLT_N_CLASSES * 4);

    /* Samples : reuse from .srt (same format, sampled by stride). */
    if (n_samples > 0) {
        uint32_t* samples = (uint32_t*)malloc((size_t)n_samples * 4u);
        if (!samples) { close(wfd); goto err; }
        for (uint32_t i = 0; i < n_samples; i++) {
            uint32_t src_idx = i * stride;
            if (src_idx >= n_nonempty) src_idx = n_nonempty - 1;
            samples[i] = srt_sparse[src_idx * 2];
        }
        write(wfd, samples, (size_t)n_samples * 4u);
        free(samples);
    }

    /* Sparse entries : (leaf_id, packed = class<<28 | slot, n_docs) */
    for (uint32_t i = 0; i < n_nonempty; i++) {
        SltSparseEntry e;
        uint32_t off_now  = srt_sparse[i * 2 + 1];
        uint32_t off_next = srt_sparse[(i + 1) * 2 + 1];
        uint32_t n_docs   = off_next - off_now;
        if (n_docs * 4u > SLT_CLASS_SIZES[leaf_class[i]])
            n_docs = SLT_CLASS_SIZES[leaf_class[i]] / 4u;
        e.leaf_id = srt_sparse[i * 2];
        e.packed  = SLT_PACK(leaf_class[i], leaf_slot[i]);
        e.n_docs  = n_docs;
        write(wfd, &e, sizeof(e));
    }

    /* Data blocks : re-read each leaf's docs from .srt V2, pad with zeros to slot size.
       Strategy : for each class, iterate leaves, seek to their doc data in .srt,
       read, pad. Simple but multiple seek passes over .srt. */
    /* Compute total data size and pre-allocate output file (sparse). */
    if (ftruncate(wfd, SLT_HEADER_BYTES + SLT_N_CLASSES * 8
                       + (uint64_t)n_samples * 4u
                       + (uint64_t)n_nonempty * sizeof(SltSparseEntry)
                       + data_total) != 0) {
        perror("slt_convert: ftruncate");
        close(wfd); goto err;
    }
    uint64_t data_base_w = SLT_HEADER_BYTES + SLT_N_CLASSES * 8
                         + (uint64_t)n_samples * 4u
                         + (uint64_t)n_nonempty * sizeof(SltSparseEntry);

    /* Write leaf data slot-by-slot */
    uint8_t* zero_pad = (uint8_t*)calloc(SLT_CLASS_SIZES[SLT_N_CLASSES-1], 1);
    if (!zero_pad) { close(wfd); goto err; }
    for (uint32_t i = 0; i < n_nonempty; i++) {
        int cls = leaf_class[i];
        uint32_t slot = leaf_slot[i];
        uint32_t off_now = srt_sparse[i * 2 + 1];
        uint32_t off_next = srt_sparse[(i + 1) * 2 + 1];
        uint32_t n_docs = off_next - off_now;
        uint32_t data_bytes = n_docs * 4u;
        if (data_bytes > SLT_CLASS_SIZES[cls]) data_bytes = SLT_CLASS_SIZES[cls];

        /* Read docs from .srt V2 */
        uint32_t docs[SLT_CLASS_SIZES[SLT_N_CLASSES-1] / 4];
        pread(rfd, docs, data_bytes, (off_t)(data_base_r + (uint64_t)off_now * 4u));

        /* Write into slot in .slt */
        uint64_t slot_off = data_base_w + class_data_offset[cls]
                          + (uint64_t)slot * SLT_CLASS_SIZES[cls];
        pwrite(wfd, docs, data_bytes, (off_t)slot_off);
        /* Pad with zeros if leaf < slot size */
        if (data_bytes < SLT_CLASS_SIZES[cls]) {
            pwrite(wfd, zero_pad, SLT_CLASS_SIZES[cls] - data_bytes,
                   (off_t)(slot_off + data_bytes));
        }
    }
    free(zero_pad);
    close(wfd);
    close(rfd);
    free(srt_sparse); free(leaf_class); free(leaf_slot);
    return 0;

err:
    free(srt_sparse); free(leaf_class); free(leaf_slot);
    close(rfd);
    return -1;
}
