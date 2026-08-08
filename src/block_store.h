#ifndef BLOCK_STORE_H
#define BLOCK_STORE_H

#include <stdint.h>
#include <stddef.h>

/* Chained-blocks on-disk layout per tree:
     block 0 : sentinel (unused).
     block i for i in [1, n_leaves] : root block for leaf (i - 1).
     blocks beyond n_leaves : overflow blocks allocated on demand.
   next = 0 means end of chain. block_index 0 is therefore "no block".  */

#define BS_BYTES 32
#define DOCS_PER_BLOCK 6

typedef struct {
    uint32_t count;            /* 0..DOCS_PER_BLOCK */
    uint32_t doc_id[DOCS_PER_BLOCK];
    uint32_t next;             /* block index, 0 = end of chain */
} Block;

typedef struct {
    int           fd;
    uint32_t      n_leaves;
    uint32_t      next_alloc;   /* next block index to allocate (write mode) */
    const Block*  map;          /* mmap'd file base, NULL in write mode */
    size_t        map_size;
} BlockStore;

/* Create new file of size (n_leaves + 1) * BS_BYTES, zeroed (sparse). */
int  block_store_create(BlockStore* s, const char* path, uint32_t n_leaves);

/* Open existing file for read. n_leaves is recovered from the meta. */
int  block_store_open_rdonly(BlockStore* s, const char* path, uint32_t n_leaves);

void block_store_close(BlockStore* s);

/* Convert leaf_id to root-block index. */
static inline uint32_t root_block_of(uint32_t leaf_id) { return leaf_id + 1; }

/* Direct pointer to block at given index (read mode). */
static inline const Block* block_ptr(const BlockStore* s, uint32_t block_idx) {
    return &s->map[block_idx];
}

/* Read a single block at given index (write mode or fallback). */
int  block_read (const BlockStore* s, uint32_t block_idx, Block* out);

/* Write a block at given index. Extends file as needed. */
int  block_write(BlockStore* s, uint32_t block_idx, const Block* b);

/* Allocate a new block (returns its index). Does not write contents. */
uint32_t block_alloc(BlockStore* s);

#endif
