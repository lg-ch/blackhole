#ifndef SORTED_STORE_H
#define SORTED_STORE_H

#include <stdint.h>
#include <stddef.h>

/* Sparse-offset posting-list format `.srt`.

   Two formats supported:
   - SRT2 (legacy): doc_ids stored as raw uint32 in data block.
                    SparseEntry.offset = doc index into data.
   - SRT3 (delta):  per-leaf header [u32 first_doc] then VarByte(delta) per
                    subsequent doc. SparseEntry.offset = BYTE offset into data.
                    See varbyte.h for encoding.

   File layout (per tree):
     [24 B header]
        uint32 magic        = 0x53525432 ("SRT2") | 0x53525433 ("SRT3")
        uint32 depth        (tree depth used at build)
        uint32 n_nonempty   (# of leaves with ≥ 1 doc)
        uint32 total_docs
        uint32 sample_stride
        uint32 data_bytes   (SRT3 only; reserved=0 in SRT2)
     [n_samples × 4 B sample_leaf_ids]
        sample_leaf_ids[i] = sparse_index[i × sample_stride].leaf_id
        (n_samples = (n_nonempty + sample_stride - 1) / sample_stride)
     [n_nonempty × 8 B sparse_index]
        sorted ascending by leaf_id.
        index[i] = (uint32 leaf_id, uint32 offset)
          SRT2: offset = doc-index into data
          SRT3: offset = byte-index into data
     [4 B sentinel + 4 B trailing]
        SRT2: SparseEntry{0xFFFFFFFFu, total_docs}
        SRT3: SparseEntry{0xFFFFFFFFu, data_bytes}
     [data]
        SRT2: total_docs × uint32 doc_ids
        SRT3: data_bytes bytes (VarByte-encoded; see varbyte.h)        */

#define SRT_MAGIC_V2       0x53525432u   /* "SRT2" — legacy uint32        */
#define SRT_MAGIC_V3       0x53525433u   /* "SRT3" — VarByte delta        */
#define SRT_MAGIC          SRT_MAGIC_V2  /* default writer alias (legacy) */
#define SRT_HEADER_BYTES   24
#define SRT_SAMPLE_STRIDE  4096u  /* tested 256 (smaller window reads) but the
                                     fixed NVMe per-request latency dominates
                                     and resident sample_leaves overflows L2
                                     cache → query slower. 4096 stays optimal. */
#define SRT_INDEX_ENTRY    8u            /* sizeof(SparseEntry)           */

typedef struct {
    uint32_t leaf_id;
    uint32_t offset;   /* SRT2: doc index, SRT3: byte offset in data block */
} SparseEntry;

typedef struct {
    int       fd;
    uint32_t  magic;          /* SRT_MAGIC_V2 or SRT_MAGIC_V3              */
    uint32_t  depth;
    uint32_t  n_nonempty;
    uint32_t  total_docs;
    uint32_t  sample_stride;
    uint32_t  data_bytes;     /* SRT3: byte length of data block; SRT2: 0  */
    uint32_t  n_samples;
    uint32_t* sample_leaves;  /* RAM-resident: n_samples × uint32 */
    uint64_t  index_base;     /* byte offset of sparse_index[0]   */
    uint64_t  data_base;      /* byte offset of data[0]           */
} SortedStore;

static inline int sorted_is_delta(const SortedStore* s) {
    return s->magic == SRT_MAGIC_V3;
}

int  sorted_store_open_rdonly(SortedStore* s, const char* path);
void sorted_store_close      (SortedStore* s);

uint32_t sorted_sample_bucket(const SortedStore* s, uint32_t leaf_id);

/* ---- Dense-samples sidecar `.smp` ----
   Optional file `<tree>.srt.smp` holding samples of the sparse index at a
   FINER stride than the in-file SRT_SAMPLE_STRIDE (4096). Loaded
   transparently by sorted_store_open_rdonly when present and consistent
   (n_nonempty must match); it then overrides sample_stride / n_samples /
   sample_leaves, which shrinks every Phase-1 window read by the stride
   ratio (4096→128 ⇒ 32 KB → 1 KB per window) with zero query-code change.
   Layout : [u32 magic][u32 stride][u32 n_samples][u32 n_nonempty]
            [n_samples × u32 sample_leaf_ids]                              */
#define SMP_MAGIC          0x31504d53u   /* "SMP1" */
#define SMP_HEADER_BYTES   16
#define SMP_DEFAULT_STRIDE 128u

/* Build `<srt_path>.smp` by streaming the .srt sparse index. Returns 0 on
   success. Sequential read of n_nonempty × 8 B ; a few seconds per tree.  */
int srt_build_smp(const char* srt_path, uint32_t stride);

#endif
