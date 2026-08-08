#ifndef TOMBSTONES_H
#define TOMBSTONES_H

#include <stdint.h>
#include <stddef.h>

/* Soft-delete via a persistent CRoaring bitmap.

   File: <index_dir>/tombstones.roaring  — portable roaring serialization.
   Empty / missing file = no tombstones, query path skips the AND-NOT.

   Load: tombstones_load() reads the file if present, returns NULL otherwise.
   Add:  add doc_id(s) to the in-memory bitmap.
   Save: tombstones_save() does atomic tmp+rename, fsyncs.

   At query time, the effective filter is (user_filter AND NOT tombstones),
   composed once per query in forest_collect_topn before the K-way merge.

   For >= 1B docs, worst case bitmap = ~125 MB. Typical (tombstones << N)
   is << 1 MB. Memory is real RSS (in-memory bitmap), not page-cache.       */

struct roaring_bitmap_s;
typedef struct roaring_bitmap_s roaring_bitmap_t;

/* Load tombstones from <index_dir>/tombstones.roaring; returns NULL if the
   file does not exist (caller treats as empty). On corruption: returns NULL
   and prints a warning. Caller owns the bitmap (roaring_bitmap_free).      */
roaring_bitmap_t* tombstones_load(const char* index_dir);

/* Atomic save: serialize, write .tmp, fsync, rename. Returns 0 on success. */
int tombstones_save(const char* index_dir, const roaring_bitmap_t* tomb);

#endif
