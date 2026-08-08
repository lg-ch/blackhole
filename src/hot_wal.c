/* HOT WAL — buffered append + background fsync. */
#define _POSIX_C_SOURCE 200809L
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <stdatomic.h>
#include <time.h>
#include "hot_wal.h"

#define WAL_REC_BYTES 12u
#define WAL_BUF_BYTES 65536u

struct HotWal {
    int             fd;
    char            path[600];
    pthread_mutex_t mu;
    uint8_t*        buf;
    uint32_t        buf_pos;
    uint64_t        n_records;
    uint64_t        file_size;
    int             fsync_interval_ms;
    pthread_t       flusher_th;
    atomic_int      stop;
    int             have_flusher;
};

static void* flusher_loop(void* arg) {
    HotWal* w = (HotWal*)arg;
    while (!atomic_load(&w->stop)) {
        struct timespec ts = {
            .tv_sec = w->fsync_interval_ms / 1000,
            .tv_nsec = (w->fsync_interval_ms % 1000) * 1000000L
        };
        nanosleep(&ts, NULL);
        hot_wal_flush(w);
    }
    return NULL;
}

HotWal* hot_wal_open(const char* path, int fsync_interval_ms) {
    if (!path) return NULL;
    HotWal* w = (HotWal*)calloc(1, sizeof(*w));
    if (!w) return NULL;
    strncpy(w->path, path, sizeof(w->path) - 1);
    w->fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (w->fd < 0) { free(w); return NULL; }
    pthread_mutex_init(&w->mu, NULL);
    w->buf = (uint8_t*)malloc(WAL_BUF_BYTES);
    if (!w->buf) { close(w->fd); free(w); return NULL; }
    w->buf_pos = 0;
    w->n_records = 0;
    /* file_size = current end (in case we reopen a non-empty WAL). */
    off_t sz = lseek(w->fd, 0, SEEK_END);
    w->file_size = (sz > 0) ? (uint64_t)sz : 0;
    w->fsync_interval_ms = fsync_interval_ms;

    atomic_store(&w->stop, 0);
    if (fsync_interval_ms > 0) {
        if (pthread_create(&w->flusher_th, NULL, flusher_loop, w) == 0) {
            w->have_flusher = 1;
        }
    }
    return w;
}

void hot_wal_close(HotWal* w) {
    if (!w) return;
    atomic_store(&w->stop, 1);
    if (w->have_flusher) pthread_join(w->flusher_th, NULL);
    hot_wal_flush(w);
    if (w->fd >= 0) close(w->fd);
    free(w->buf);
    pthread_mutex_destroy(&w->mu);
    free(w);
}

int hot_wal_append(HotWal* w, uint32_t tree_id, uint32_t leaf_id, uint32_t doc_id) {
    if (!w) return -1;
    pthread_mutex_lock(&w->mu);
    if (w->buf_pos + WAL_REC_BYTES > WAL_BUF_BYTES) {
        /* Flush inline. */
        ssize_t r = write(w->fd, w->buf, w->buf_pos);
        if (r != (ssize_t)w->buf_pos) { pthread_mutex_unlock(&w->mu); return -1; }
        w->file_size += w->buf_pos;
        w->buf_pos = 0;
    }
    memcpy(w->buf + w->buf_pos + 0, &tree_id, 4);
    memcpy(w->buf + w->buf_pos + 4, &leaf_id, 4);
    memcpy(w->buf + w->buf_pos + 8, &doc_id,  4);
    w->buf_pos += WAL_REC_BYTES;
    w->n_records++;
    pthread_mutex_unlock(&w->mu);
    return 0;
}

int hot_wal_flush(HotWal* w) {
    if (!w) return -1;
    pthread_mutex_lock(&w->mu);
    if (w->buf_pos > 0) {
        ssize_t r = write(w->fd, w->buf, w->buf_pos);
        if (r != (ssize_t)w->buf_pos) { pthread_mutex_unlock(&w->mu); return -1; }
        w->file_size += w->buf_pos;
        w->buf_pos = 0;
    }
    int rc = fsync(w->fd);
    pthread_mutex_unlock(&w->mu);
    return rc;
}

int hot_wal_truncate(HotWal* w) {
    if (!w) return -1;
    pthread_mutex_lock(&w->mu);
    w->buf_pos = 0;
    if (ftruncate(w->fd, 0) != 0) { pthread_mutex_unlock(&w->mu); return -1; }
    lseek(w->fd, 0, SEEK_SET);
    w->file_size = 0;
    w->n_records = 0;
    fsync(w->fd);
    pthread_mutex_unlock(&w->mu);
    return 0;
}

uint64_t hot_wal_size_bytes(const HotWal* w) { return w ? w->file_size + w->buf_pos : 0; }
uint64_t hot_wal_n_records (const HotWal* w) { return w ? w->n_records : 0; }

int hot_wal_replay(const char* path, hot_wal_cb cb, void* ctx) {
    if (!path || !cb) return -1;
    int fd = open(path, O_RDONLY);
    if (fd < 0) return 0;  /* no WAL file → nothing to replay */
    int n = 0;
    uint8_t buf[WAL_BUF_BYTES];
    off_t off = 0;
    for (;;) {
        ssize_t r = pread(fd, buf, sizeof(buf), off);
        if (r <= 0) break;
        size_t nrec = (size_t)r / WAL_REC_BYTES;
        for (size_t i = 0; i < nrec; i++) {
            uint32_t tid, lid, did;
            memcpy(&tid, buf + i * WAL_REC_BYTES + 0, 4);
            memcpy(&lid, buf + i * WAL_REC_BYTES + 4, 4);
            memcpy(&did, buf + i * WAL_REC_BYTES + 8, 4);
            if (cb(ctx, tid, lid, did) != 0) { close(fd); return -1; }
            n++;
        }
        off += (off_t)(nrec * WAL_REC_BYTES);
        if ((size_t)r < sizeof(buf)) break;
    }
    close(fd);
    return n;
}
