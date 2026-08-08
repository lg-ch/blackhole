/* Background compaction thread — round-robin per-tree compaction throttled. */
#define _POSIX_C_SOURCE 200809L
#include <pthread.h>
#include <stdatomic.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "hot_compact_bg.h"
#include "hot_store.h"
#include "hot_compact.h"
#include "query_tree.h"

struct HotCompactBg {
    Forest*      forest;
    HotOverlay*  hot;
    char         index_dir[512];
    int          threshold_docs;
    int          sleep_ms;
    int          out_format;
    pthread_t    thr;
    atomic_int   stop;
    atomic_uint_least64_t n_compactions;
    atomic_uint_least64_t n_docs_merged;

    /* Query ↔ compaction synchronization : during the tiny fd-swap window
     * (forest_reopen_tree), queries must not read from that tree. Use a
     * rwlock : queries take read lock at forest access ; compaction takes
     * write lock only for the swap moment. */
    pthread_rwlock_t forest_rw;
    int              rwlock_ok;
};

/* Compaction thread reads snap from HOT (own lock), does merge + rewrite
 * (lock-free on disk, only touches its own tmp file), then briefly takes
 * write lock to swap fd. */
static void* compact_loop(void* arg) {
    HotCompactBg* bg = (HotCompactBg*)arg;
    int nt = bg->forest->n_trees;
    int tid = 0;
    while (!atomic_load(&bg->stop)) {
        /* Peek : is this tree's HOT above threshold? */
        HotTree* ht = &bg->hot->trees[tid];
        int should = 0;
        pthread_mutex_lock(&ht->mu);
        int entries = ht->n_entries;
        int docs = 0;
        for (int i = 0; i < entries; i++) docs += ht->entries[i].n_docs;
        pthread_mutex_unlock(&ht->mu);
        if (bg->threshold_docs <= 0 || docs >= bg->threshold_docs) should = 1;

        if (should && entries > 0) {
            HotSnapEntry* snap = NULL; int snap_n = 0;
            if (hot_snapshot_and_clear(bg->hot, tid, &snap, &snap_n) == 0 && snap_n > 0) {
                char main_path[520], tmp_path[540];
                snprintf(main_path, sizeof(main_path), "%s/tree%05d.srt",
                         bg->index_dir, tid);
                snprintf(tmp_path,  sizeof(tmp_path),  "%s/tree%05d.srt.tmp",
                         bg->index_dir, tid);
                int new_n = 0; uint64_t new_total = 0;
                int rc = hot_compact_tree(main_path, snap, snap_n, tmp_path,
                                          bg->out_format, &new_n, &new_total);
                if (rc == 0) {
                    /* Atomic rename → fd swap under write lock. */
                    if (rename(tmp_path, main_path) == 0) {
                        if (bg->rwlock_ok) pthread_rwlock_wrlock(&bg->forest_rw);
                        forest_reopen_tree(bg->forest, tid);
                        if (bg->rwlock_ok) pthread_rwlock_unlock(&bg->forest_rw);
                        atomic_fetch_add(&bg->n_compactions, 1);
                        atomic_fetch_add(&bg->n_docs_merged, (uint64_t)docs);
                    } else {
                        perror("bg_compact: rename");
                    }
                }
                hot_snapshot_free(snap, snap_n);
            }
        }

        tid = (tid + 1) % nt;
        if (tid == 0) {
            /* Full pass done — sleep to yield SSD bandwidth. */
            struct timespec ts = {
                .tv_sec  = bg->sleep_ms / 1000,
                .tv_nsec = (bg->sleep_ms % 1000) * 1000000L
            };
            nanosleep(&ts, NULL);
        }
    }
    return NULL;
}

HotCompactBg* hot_compact_bg_start(Forest* forest, HotOverlay* hot,
                                   const char* index_dir,
                                   int threshold_docs, int sleep_ms,
                                   int out_format) {
    if (!forest || !hot || !index_dir) return NULL;
    if (out_format != 2 && out_format != 3) out_format = 3;
    HotCompactBg* bg = (HotCompactBg*)calloc(1, sizeof(*bg));
    if (!bg) return NULL;
    bg->forest = forest;
    bg->hot = hot;
    strncpy(bg->index_dir, index_dir, sizeof(bg->index_dir) - 1);
    bg->threshold_docs = threshold_docs;
    bg->sleep_ms = sleep_ms > 0 ? sleep_ms : 100;
    bg->out_format = out_format;
    atomic_store(&bg->stop, 0);
    atomic_store(&bg->n_compactions, 0);
    atomic_store(&bg->n_docs_merged, 0);
    if (pthread_rwlock_init(&bg->forest_rw, NULL) == 0) bg->rwlock_ok = 1;
    if (pthread_create(&bg->thr, NULL, compact_loop, bg) != 0) {
        if (bg->rwlock_ok) pthread_rwlock_destroy(&bg->forest_rw);
        free(bg); return NULL;
    }
    return bg;
}

void hot_compact_bg_stop(HotCompactBg* bg) {
    if (!bg) return;
    atomic_store(&bg->stop, 1);
    pthread_join(bg->thr, NULL);
    if (bg->rwlock_ok) pthread_rwlock_destroy(&bg->forest_rw);
    free(bg);
}

uint64_t hot_compact_bg_n_compactions(const HotCompactBg* bg) {
    if (!bg) return 0;
    return atomic_load(&((HotCompactBg*)bg)->n_compactions);
}

uint64_t hot_compact_bg_n_docs_merged(const HotCompactBg* bg) {
    if (!bg) return 0;
    return atomic_load(&((HotCompactBg*)bg)->n_docs_merged);
}
