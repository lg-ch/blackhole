#define _POSIX_C_SOURCE 200809L
#include "croaring_io.h"

#include <roaring/roaring.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

roaring_bitmap_t* roaring_load_int32_file(const char* path, int n_corpus,
                                          int* n_filter_out) {
    if (n_filter_out) *n_filter_out = 0;
    if (!path) return NULL;
    FILE* f = fopen(path, "rb");
    if (!f) { perror("open filter"); return NULL; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    rewind(f);
    int n = (int)(sz / 4);
    int32_t* ids = (int32_t*)malloc((size_t)n * sizeof(int32_t));
    if (!ids || (n > 0 && fread(ids, sizeof(int32_t), n, f) != (size_t)n)) {
        free(ids); fclose(f); return NULL;
    }
    fclose(f);

    roaring_bitmap_t* rb = roaring_bitmap_create();
    int set = 0;
    for (int i = 0; i < n; i++) {
        int32_t d = ids[i];
        if (d < 0) continue;
        if (n_corpus > 0 && d >= n_corpus) continue;
        roaring_bitmap_add(rb, (uint32_t)d);
        set++;
    }
    free(ids);
    roaring_bitmap_run_optimize(rb);
    roaring_bitmap_shrink_to_fit(rb);
    if (n_filter_out) *n_filter_out = set;
    size_t pb = roaring_bitmap_portable_size_in_bytes(rb);
    fprintf(stderr, "filter: %d ids → roaring portable=%zu bytes (%.2f MB)\n",
            set, pb, pb / 1024.0 / 1024.0);
    return rb;
}

/* Read ClickHouse varint (7-bit groups, MSB = continuation). Returns the
   number of bytes consumed (>=1), or 0 on overflow / truncated input.    */
static size_t ch_read_varint(const uint8_t* p, size_t n, uint64_t* out) {
    uint64_t v = 0;
    int shift = 0;
    for (size_t i = 0; i < n && i < 10; i++) {
        uint8_t b = p[i];
        v |= ((uint64_t)(b & 0x7F)) << shift;
        if (!(b & 0x80)) { *out = v; return i + 1; }
        shift += 7;
    }
    return 0;
}

roaring_bitmap_t* roaring_from_ch_state(const uint8_t* data, size_t len,
                                        int* n_filter_out) {
    if (n_filter_out) *n_filter_out = 0;
    if (!data || len < 2) return NULL;

    /* Type tag : 0x01 = "Roaring portable" branch in CH's bitmap impl.    */
    if (data[0] != 0x01) {
        fprintf(stderr, "ch_state: unexpected tag 0x%02x (want 0x01)\n", data[0]);
        return NULL;
    }
    uint64_t cr_size = 0;
    size_t vlen = ch_read_varint(data + 1, len - 1, &cr_size);
    if (vlen == 0) {
        fprintf(stderr, "ch_state: bad varint size\n");
        return NULL;
    }
    size_t hdr = 1 + vlen;
    if (hdr + cr_size > len) {
        fprintf(stderr, "ch_state: short body (need %zu, have %zu)\n",
                hdr + (size_t)cr_size, len);
        return NULL;
    }
    roaring_bitmap_t* rb = roaring_bitmap_portable_deserialize_safe(
        (const char*)(data + hdr), (size_t)cr_size);
    if (!rb) {
        fprintf(stderr, "ch_state: portable_deserialize failed\n");
        return NULL;
    }
    uint64_t card = roaring_bitmap_get_cardinality(rb);
    if (n_filter_out) *n_filter_out = (card > (uint64_t)INT32_MAX)
                                          ? INT32_MAX : (int)card;
    fprintf(stderr, "filter: ch_state %zu bytes (cr=%zu) → roaring card=%llu\n",
            len, (size_t)cr_size, (unsigned long long)card);
    return rb;
}

roaring_bitmap_t* roaring_load_ch_state_file(const char* path,
                                             int* n_filter_out) {
    if (n_filter_out) *n_filter_out = 0;
    if (!path) return NULL;
    FILE* f = fopen(path, "rb");
    if (!f) { perror("open ch_state"); return NULL; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    rewind(f);
    if (sz <= 0) { fclose(f); return NULL; }
    uint8_t* buf = (uint8_t*)malloc((size_t)sz);
    if (!buf || fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf); fclose(f); return NULL;
    }
    fclose(f);
    roaring_bitmap_t* rb = roaring_from_ch_state(buf, (size_t)sz,
                                                 n_filter_out);
    free(buf);
    return rb;
}
