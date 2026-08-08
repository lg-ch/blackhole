#define _POSIX_C_SOURCE 200809L
#include "srt_hash.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#define XXH_INLINE_ALL 0
#include <xxhash.h>

#define HASH_CHUNK (256 * 1024)

int srt_finalize_with_hash(const char* path) {
    int fd = open(path, O_RDWR);
    if (fd < 0) { perror("srt_hash: open"); return -1; }

    XXH64_state_t* state = XXH64_createState();
    if (!state) { close(fd); return -1; }
    XXH64_reset(state, 0);

    uint8_t* buf = (uint8_t*)malloc(HASH_CHUNK);
    if (!buf) { XXH64_freeState(state); close(fd); return -1; }

    if (lseek(fd, 0, SEEK_SET) < 0) {
        perror("srt_hash: lseek 0");
        free(buf); XXH64_freeState(state); close(fd); return -1;
    }
    ssize_t n;
    while ((n = read(fd, buf, HASH_CHUNK)) > 0) {
        XXH64_update(state, buf, (size_t)n);
    }
    free(buf);
    if (n < 0) {
        perror("srt_hash: read");
        XXH64_freeState(state); close(fd); return -1;
    }

    uint64_t hash = XXH64_digest(state);
    XXH64_freeState(state);

    if (lseek(fd, 0, SEEK_END) < 0) {
        perror("srt_hash: lseek end");
        close(fd); return -1;
    }
    if (write(fd, &hash, sizeof(hash)) != (ssize_t)sizeof(hash)) {
        perror("srt_hash: write footer");
        close(fd); return -1;
    }
    fsync(fd);
    close(fd);
    return 0;
}

int srt_verify_hash(const char* path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("srt_verify: open"); return -1; }
    struct stat st;
    if (fstat(fd, &st) != 0) {
        perror("srt_verify: fstat"); close(fd); return -1;
    }
    if (st.st_size < (off_t)SRT_HASH_BYTES) {
        fprintf(stderr, "srt_verify: file too small\n");
        close(fd); return -1;
    }
    /* Check magic: SRT2 indexes have no hash footer, treat as "OK skip". */
    uint32_t magic;
    if (pread(fd, &magic, sizeof(magic), 0) != (ssize_t)sizeof(magic)) {
        close(fd); return -1;
    }
    if (magic == 0x53525432u /* SRT2 */) {
        close(fd); return 1;  /* legacy, no hash to verify */
    }
    if (magic != 0x53525433u /* SRT3 */) {
        fprintf(stderr, "srt_verify: bad magic 0x%08x in %s\n", magic, path);
        close(fd); return -1;
    }
    off_t data_len = st.st_size - SRT_HASH_BYTES;

    XXH64_state_t* state = XXH64_createState();
    if (!state) { close(fd); return -1; }
    XXH64_reset(state, 0);

    uint8_t* buf = (uint8_t*)malloc(HASH_CHUNK);
    if (!buf) { XXH64_freeState(state); close(fd); return -1; }

    off_t left = data_len;
    while (left > 0) {
        size_t take = (left > HASH_CHUNK) ? HASH_CHUNK : (size_t)left;
        ssize_t n = read(fd, buf, take);
        if (n <= 0) {
            perror("srt_verify: read");
            free(buf); XXH64_freeState(state); close(fd); return -1;
        }
        XXH64_update(state, buf, (size_t)n);
        left -= n;
    }
    free(buf);
    uint64_t computed = XXH64_digest(state);
    XXH64_freeState(state);

    uint64_t footer;
    if (pread(fd, &footer, sizeof(footer), data_len) != (ssize_t)sizeof(footer)) {
        perror("srt_verify: read footer");
        close(fd); return -1;
    }
    close(fd);

    return (computed == footer) ? 1 : 0;
}
