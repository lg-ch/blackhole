#define _POSIX_C_SOURCE 200809L
#include "tombstones.h"

#include <roaring/roaring.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

static void tomb_path(char* out, size_t n, const char* dir) {
    snprintf(out, n, "%s/tombstones.roaring", dir);
}

roaring_bitmap_t* tombstones_load(const char* index_dir) {
    char p[768];
    tomb_path(p, sizeof(p), index_dir);
    struct stat st;
    if (stat(p, &st) != 0) return NULL;       /* none = ok                */
    if (st.st_size == 0) return NULL;
    int fd = open(p, O_RDONLY);
    if (fd < 0) { perror("tombstones: open"); return NULL; }
    char* buf = (char*)malloc((size_t)st.st_size);
    if (!buf) { close(fd); return NULL; }
    ssize_t n = read(fd, buf, (size_t)st.st_size);
    close(fd);
    if (n != (ssize_t)st.st_size) { free(buf); return NULL; }

    roaring_bitmap_t* rb = roaring_bitmap_portable_deserialize_safe(
        buf, (size_t)st.st_size);
    free(buf);
    if (!rb) {
        fprintf(stderr, "tombstones: deserialize failed for %s — treating as empty\n", p);
        return NULL;
    }
    return rb;
}

int tombstones_save(const char* index_dir, const roaring_bitmap_t* tomb) {
    if (!tomb) return 0;
    char p[768], ptmp[800];
    tomb_path(p, sizeof(p), index_dir);
    snprintf(ptmp, sizeof(ptmp), "%s.tmp", p);

    size_t sz = roaring_bitmap_portable_size_in_bytes(tomb);
    char* buf = (char*)malloc(sz);
    if (!buf) return -1;
    size_t wrote = roaring_bitmap_portable_serialize(tomb, buf);
    if (wrote != sz) { free(buf); return -1; }

    int fd = open(ptmp, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) { perror("tombstones: open tmp"); free(buf); return -1; }
    ssize_t n = write(fd, buf, sz);
    free(buf);
    if (n != (ssize_t)sz) { close(fd); unlink(ptmp); return -1; }
    fsync(fd);
    close(fd);
    if (rename(ptmp, p) != 0) {
        perror("tombstones: rename"); unlink(ptmp); return -1;
    }
    return 0;
}
