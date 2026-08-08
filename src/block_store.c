#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE
#include "block_store.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>

int block_store_create(BlockStore* s, const char* path, uint32_t n_leaves) {
    s->fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (s->fd < 0) return -1;
    off_t initial = (off_t)(n_leaves + 1) * BS_BYTES;
    if (ftruncate(s->fd, initial) != 0) { close(s->fd); return -1; }
    s->n_leaves = n_leaves;
    s->next_alloc = n_leaves + 1;
    s->map = NULL;
    s->map_size = 0;
    return 0;
}

int block_store_open_rdonly(BlockStore* s, const char* path, uint32_t n_leaves) {
    s->fd = open(path, O_RDONLY);
    if (s->fd < 0) return -1;
    struct stat st;
    if (fstat(s->fd, &st) != 0) { close(s->fd); return -1; }
    s->map_size = (size_t)st.st_size;
    void* m = mmap(NULL, s->map_size, PROT_READ, MAP_PRIVATE, s->fd, 0);
    if (m == MAP_FAILED) {
        s->map = NULL;
    } else {
        s->map = (const Block*)m;
        madvise(m, s->map_size, MADV_RANDOM);
    }
    s->n_leaves = n_leaves;
    s->next_alloc = 0;
    return 0;
}

void block_store_close(BlockStore* s) {
    if (s->map) { munmap((void*)s->map, s->map_size); s->map = NULL; }
    if (s->fd >= 0) close(s->fd);
    s->fd = -1;
}

int block_read(const BlockStore* s, uint32_t block_idx, Block* out) {
    off_t off = (off_t)block_idx * BS_BYTES;
    ssize_t n = pread(s->fd, out, BS_BYTES, off);
    return (n == BS_BYTES) ? 0 : -1;
}

int block_write(BlockStore* s, uint32_t block_idx, const Block* b) {
    off_t off = (off_t)block_idx * BS_BYTES;
    ssize_t n = pwrite(s->fd, b, BS_BYTES, off);
    return (n == BS_BYTES) ? 0 : -1;
}

uint32_t block_alloc(BlockStore* s) {
    return s->next_alloc++;
}
