#define _POSIX_C_SOURCE 200809L
#include "sorted_store.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>

int sorted_store_open_rdonly(SortedStore* s, const char* path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    uint32_t hdr[6];
    if (pread(fd, hdr, sizeof(hdr), 0) != (ssize_t)sizeof(hdr)) {
        close(fd); return -1;
    }
    if (hdr[0] != SRT_MAGIC_V2 && hdr[0] != SRT_MAGIC_V3) {
        fprintf(stderr, "sorted_store: bad magic 0x%08x in %s\n", hdr[0], path);
        close(fd); return -1;
    }
    s->fd            = fd;
    s->magic         = hdr[0];
    s->depth         = hdr[1];
    s->n_nonempty    = hdr[2];
    s->total_docs    = hdr[3];
    s->sample_stride = hdr[4];
    s->data_bytes    = (hdr[0] == SRT_MAGIC_V3) ? hdr[5] : 0u;
    s->n_samples     = (s->n_nonempty + s->sample_stride - 1) / s->sample_stride;
    s->index_base    = (uint64_t)SRT_HEADER_BYTES
                       + (uint64_t)s->n_samples * 4u;
    s->data_base     = s->index_base
                       + (uint64_t)s->n_nonempty * (uint64_t)SRT_INDEX_ENTRY
                       + (uint64_t)SRT_INDEX_ENTRY;     /* + sentinel entry */

    /* Load the small RAM sample table (uint32 leaf_ids). */
    s->sample_leaves = NULL;
    if (s->n_samples > 0) {
        s->sample_leaves = (uint32_t*)malloc((size_t)s->n_samples * 4u);
        if (!s->sample_leaves) { close(fd); return -1; }
        if (pread(fd, s->sample_leaves, (size_t)s->n_samples * 4u,
                  (off_t)SRT_HEADER_BYTES) != (ssize_t)(s->n_samples * 4u)) {
            free(s->sample_leaves); s->sample_leaves = NULL;
            close(fd); return -1;
        }
    }

    /* Dense-samples sidecar : if <path>.smp exists and matches this .srt,
       swap in its finer samples. Any inconsistency → silently keep the
       in-file samples (correctness never depends on the sidecar).         */
    {
        char smp_path[600];
        snprintf(smp_path, sizeof(smp_path), "%s.smp", path);
        int sfd = open(smp_path, O_RDONLY);
        if (sfd >= 0) {
            uint32_t shdr[4];
            if (pread(sfd, shdr, sizeof(shdr), 0) == (ssize_t)sizeof(shdr)
                && shdr[0] == SMP_MAGIC
                && shdr[1] > 0
                && shdr[3] == s->n_nonempty) {
                uint32_t stride    = shdr[1];
                uint32_t n_samples = shdr[2];
                uint32_t expect = (s->n_nonempty + stride - 1) / stride;
                if (n_samples == expect && n_samples > 0) {
                    uint32_t* dense = (uint32_t*)malloc((size_t)n_samples * 4u);
                    if (dense && pread(sfd, dense, (size_t)n_samples * 4u,
                                       SMP_HEADER_BYTES)
                                 == (ssize_t)(n_samples * 4u)) {
                        free(s->sample_leaves);
                        s->sample_leaves = dense;
                        s->sample_stride = stride;
                        s->n_samples     = n_samples;
                    } else {
                        free(dense);
                    }
                }
            }
            close(sfd);
        }
    }
    return 0;
}

int srt_build_smp(const char* srt_path, uint32_t stride) {
    if (stride == 0) stride = SMP_DEFAULT_STRIDE;
    int fd = open(srt_path, O_RDONLY);
    if (fd < 0) return -1;
    uint32_t hdr[6];
    if (pread(fd, hdr, sizeof(hdr), 0) != (ssize_t)sizeof(hdr)
        || (hdr[0] != SRT_MAGIC_V2 && hdr[0] != SRT_MAGIC_V3)) {
        close(fd); return -1;
    }
    uint32_t n_nonempty = hdr[2];
    uint32_t native_stride = hdr[4];
    uint32_t native_samples = (n_nonempty + native_stride - 1) / native_stride;
    uint64_t index_base = (uint64_t)SRT_HEADER_BYTES
                        + (uint64_t)native_samples * 4u;
    uint32_t n_samples = (n_nonempty + stride - 1) / stride;
    uint32_t* out = (uint32_t*)malloc((size_t)(n_samples ? n_samples : 1) * 4u);
    if (!out) { close(fd); return -1; }

    /* Stream the sparse index in chunks, keep every stride-th leaf_id. */
    const uint32_t CHUNK = 1u << 20;   /* entries per read (8 MB) */
    SparseEntry* buf = (SparseEntry*)malloc((size_t)CHUNK * SRT_INDEX_ENTRY);
    if (!buf) { free(out); close(fd); return -1; }
    uint32_t done = 0, wr = 0;
    while (done < n_nonempty) {
        uint32_t take = n_nonempty - done;
        if (take > CHUNK) take = CHUNK;
        if (pread(fd, buf, (size_t)take * SRT_INDEX_ENTRY,
                  (off_t)(index_base + (uint64_t)done * SRT_INDEX_ENTRY))
            != (ssize_t)(take * SRT_INDEX_ENTRY)) {
            free(buf); free(out); close(fd); return -1;
        }
        /* First multiple of stride ≥ done. */
        uint32_t i = (done % stride == 0) ? 0 : stride - (done % stride);
        for (; i < take; i += stride) out[wr++] = buf[i].leaf_id;
        done += take;
    }
    free(buf); close(fd);
    if (wr != n_samples) { free(out); return -1; }

    char smp_path[600], tmp_path[620];
    snprintf(smp_path, sizeof(smp_path), "%s.smp", srt_path);
    snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", smp_path);
    int wfd = open(tmp_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (wfd < 0) { free(out); return -1; }
    uint32_t whdr[4] = { SMP_MAGIC, stride, n_samples, n_nonempty };
    int ok = (pwrite(wfd, whdr, sizeof(whdr), 0) == (ssize_t)sizeof(whdr))
          && (pwrite(wfd, out, (size_t)n_samples * 4u, SMP_HEADER_BYTES)
              == (ssize_t)(n_samples * 4u))
          && (fsync(wfd) == 0);
    close(wfd); free(out);
    if (!ok) { unlink(tmp_path); return -1; }
    if (rename(tmp_path, smp_path) != 0) { unlink(tmp_path); return -1; }
    return 0;
}

void sorted_store_close(SortedStore* s) {
    if (s->sample_leaves) { free(s->sample_leaves); s->sample_leaves = NULL; }
    if (s->fd >= 0) { close(s->fd); s->fd = -1; }
}

uint32_t sorted_sample_bucket(const SortedStore* s, uint32_t leaf_id) {
    uint32_t lo = 0, hi = s->n_samples;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (s->sample_leaves[mid] <= leaf_id) lo = mid + 1;
        else                                  hi = mid;
    }
    return (lo == 0) ? 0 : (lo - 1);
}
