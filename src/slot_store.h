#ifndef SLOT_STORE_H
#define SLOT_STORE_H

/* .slt format : slot-allocator alternative to .srt.
 * Leaves stored in fixed-size slots grouped by size class. Enables :
 *   - Structural bounded max leaf size (last class = mlb)
 *   - Predictable read pattern (all slots in a class same size)
 *   - Streaming HOT overlay (allocate slot in class dynamically)
 */

#include <stdint.h>

#define SLT_MAGIC        0x32544c53u   /* "SLT2" little-endian — v2 adds n_docs field */
#define SLT_HEADER_BYTES 32
#define SLT_N_CLASSES    16

/* Power-of-2 slot classes covering leaves from 16 B to 384 KB. */
static const uint32_t SLT_CLASS_SIZES[SLT_N_CLASSES] = {
    16, 32, 64, 128, 256, 512,
    1024, 2048, 4096, 8192, 16384, 32768,
    65536, 131072, 262144, 393216
};

/* Sparse entry : (leaf_id, packed, n_docs) where packed = class_id<<28 | slot_id.
 * class_id fits in 4 bits (16 classes), slot_id in 28 bits (256 M per class).
 * n_docs = actual doc count in the slot (rest is zero-padded to class size).
 * Total 12 B — +50 % vs SRT sparse but necessary to trim padding at read time. */
typedef struct __attribute__((packed)) {
    uint32_t leaf_id;
    uint32_t packed;   /* upper 4 bits = class, lower 28 = slot */
    uint32_t n_docs;   /* actual valid doc count in the slot */
} SltSparseEntry;

#define SLT_CLASS(packed)   (((packed) >> 28) & 0xFu)
#define SLT_SLOT(packed)    ((packed) & 0x0FFFFFFFu)
#define SLT_PACK(cls, slot) ((((uint32_t)(cls) & 0xFu) << 28) | ((uint32_t)(slot) & 0x0FFFFFFFu))

typedef struct {
    int       fd;
    uint32_t  magic;
    uint32_t  depth;
    uint32_t  n_nonempty;
    uint32_t  total_docs;
    uint32_t  n_classes;    /* always SLT_N_CLASSES for now */
    uint32_t  sample_stride;
    uint32_t  class_sizes[SLT_N_CLASSES];
    uint32_t  class_n_slots[SLT_N_CLASSES];
    uint64_t  class_data_offset[SLT_N_CLASSES];  /* byte offset within data block */
    uint32_t  n_samples;
    uint32_t* sample_leaves;  /* RAM-resident (as .srt) */
    uint64_t  index_base;     /* byte offset of sparse index start */
    uint64_t  data_base;      /* byte offset of data block start */
} SltStore;

int  slt_open_rdonly(SltStore* s, const char* path);
void slt_close      (SltStore* s);
uint32_t slt_sample_bucket(const SltStore* s, uint32_t leaf_id);

/* Convert .srt V2 (uncompressed uint32 doc_ids) to .slt.
 * Skips leaves > max class (overflow) — logs count. */
int slt_convert_from_srt_v2(const char* srt_path, const char* slt_path,
                            int* out_n_overflow);

#endif
