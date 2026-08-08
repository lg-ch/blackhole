#ifndef CROARING_IO_H
#define CROARING_IO_H

#include <stdint.h>
#include <stddef.h>

/* Opaque forward decl so callers can keep this header light. */
struct roaring_bitmap_s;
typedef struct roaring_bitmap_s roaring_bitmap_t;

/* Load int32 doc_ids from a binary file (one int32 LE per id) and pack into
   a roaring bitmap. `n_corpus` (>0) bounds valid ids to [0, n_corpus).
   *n_filter_out gets the number of ids actually set.
   Returns a new bitmap (caller frees with roaring_bitmap_free) or NULL.   */
roaring_bitmap_t* roaring_load_int32_file(const char* path, int n_corpus,
                                          int* n_filter_out);

/* Decode a ClickHouse `AggregateFunction(groupBitmap, UInt32)` state blob
   (whole file = state bytes :
       [1 byte tag = 0x01][varint size][CRoaring portable bytes...]).
   Use this when bytes were obtained via `SELECT cast(bm, 'String')`
   and piped to disk by the Python orchestrator.
   Returns a new bitmap or NULL. *n_filter_out = cardinality.              */
roaring_bitmap_t* roaring_load_ch_state_file(const char* path,
                                             int* n_filter_out);

/* Same decoder but from an in-memory buffer (no file I/O). */
roaring_bitmap_t* roaring_from_ch_state(const uint8_t* data, size_t len,
                                        int* n_filter_out);

#endif
