/* Store V2 a slots fixes — voir slots_v2.h. Pipeline de collecte :
   1 vague io_uring (offsets calcules, pas de phase fenetres) -> parse
   [count][ids] -> paires (tree|doc) -> radix par doc -> votes dedup par
   arbre -> top_n par histogramme de votes. Self-contained. */
#define _POSIX_C_SOURCE 200809L
#include "slots_v2.h"

#include <fcntl.h>
#include <liburing.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    int   n_trees;
    int   n_leaves;
    int   slot_bytes;
    int*  fds;
    struct io_uring ring;
    int   ring_ok;
} SlotsV2;

void* slots_v2_open(const char* dir, int n_trees, int n_leaves,
                    int slot_bytes) {
    SlotsV2* s = (SlotsV2*)calloc(1, sizeof(SlotsV2));
    if (!s) return NULL;
    s->n_trees = n_trees;
    s->n_leaves = n_leaves;
    s->slot_bytes = slot_bytes;
    s->fds = (int*)malloc((size_t)n_trees * sizeof(int));
    if (!s->fds) { free(s); return NULL; }
    char path[1024];
    for (int t = 0; t < n_trees; t++) {
        snprintf(path, sizeof(path), "%s/tree%05d.slt", dir, t);
        s->fds[t] = open(path, O_RDONLY);
        if (s->fds[t] < 0) {
            for (int j = 0; j < t; j++) close(s->fds[j]);
            free(s->fds); free(s);
            return NULL;
        }
    }
    s->ring_ok = (io_uring_queue_init(4096, &s->ring, 0) == 0);
    return s;
}

void slots_v2_close(void* h) {
    SlotsV2* s = (SlotsV2*)h;
    if (!s) return;
    if (s->ring_ok) io_uring_queue_exit(&s->ring);
    for (int t = 0; t < s->n_trees; t++) close(s->fds[t]);
    free(s->fds);
    free(s);
}

int slots_v2_collect(void* h, const int32_t* leaves,
                     const int32_t* leaf_tree, int nL, int k_shift,
                     int n_trees, int top_n,
                     int32_t* out_ids, int32_t* out_votes) {
    SlotsV2* s = (SlotsV2*)h;
    if (!s || top_n <= 0) return -1;
    const size_t range_bytes = (size_t)s->slot_bytes << k_shift;
    const int slot_cap = (s->slot_bytes - 4) / 4;

    /* Probes actives. */
    int n_act = 0;
    for (int li = 0; li < nL; li++) if (leaves[li] >= 0) n_act++;
    if (n_act == 0) return 0;

    uint8_t* buf = (uint8_t*)malloc((size_t)n_act * range_bytes);
    int* act = (int*)malloc((size_t)n_act * sizeof(int));
    if (!buf || !act) { free(buf); free(act); return -1; }

    /* --- LA vague unique : offsets calcules, lectures contiguës. --- */
    int k = 0, inflight = 0, submitted_total = 0;
    const int WINDOW = 2048;
    for (int li = 0; li < nL; li++) {
        if (leaves[li] < 0) continue;
        act[submitted_total] = li;
        struct io_uring_sqe* sqe = io_uring_get_sqe(&s->ring);
        if (!sqe) {
            io_uring_submit(&s->ring);
            sqe = io_uring_get_sqe(&s->ring);
            if (!sqe) { free(buf); free(act); return -1; }
        }
        uint64_t off = ((uint64_t)leaves[li] << k_shift)
                       * (uint64_t)s->slot_bytes;
        io_uring_prep_read(sqe, s->fds[leaf_tree[li]],
                           buf + (size_t)submitted_total * range_bytes,
                           (unsigned)range_bytes, off);
        io_uring_sqe_set_data(sqe, (void*)(uintptr_t)submitted_total);
        submitted_total++;
        if (++inflight >= WINDOW) {
            io_uring_submit(&s->ring);
            struct io_uring_cqe* cqe;
            while (inflight > WINDOW / 2 &&
                   io_uring_wait_cqe(&s->ring, &cqe) == 0) {
                io_uring_cqe_seen(&s->ring, cqe);
                inflight--;
            }
        }
        (void)k;
    }
    io_uring_submit(&s->ring);
    while (inflight > 0) {
        struct io_uring_cqe* cqe;
        if (io_uring_wait_cqe(&s->ring, &cqe) != 0) break;
        io_uring_cqe_seen(&s->ring, cqe);
        inflight--;
    }

    /* --- Parse + pack (tree|doc). --- */
    size_t cap_pairs = (size_t)n_act * ((size_t)slot_cap << k_shift);
    uint64_t* pairs = (uint64_t*)malloc(cap_pairs * sizeof(uint64_t));
    uint64_t* scratch = (uint64_t*)malloc(cap_pairs * sizeof(uint64_t));
    if (!pairs || !scratch) {
        free(pairs); free(scratch); free(buf); free(act);
        return -1;
    }
    size_t N = 0;
    for (int a = 0; a < submitted_total; a++) {
        uint64_t tag = (uint64_t)leaf_tree[act[a]] << 32;
        const uint8_t* rb = buf + (size_t)a * range_bytes;
        for (int sl = 0; sl < (1 << k_shift); sl++) {
            const uint8_t* p = rb + (size_t)sl * s->slot_bytes;
            uint32_t cnt;
            memcpy(&cnt, p, 4);
            if (cnt > (uint32_t)slot_cap) cnt = (uint32_t)slot_cap;
            const uint32_t* ids = (const uint32_t*)(p + 4);
            for (uint32_t i = 0; i < cnt; i++)
                pairs[N++] = tag | (uint64_t)ids[i];
        }
    }
    free(buf); free(act);
    if (N == 0) { free(pairs); free(scratch); return 0; }

    /* --- Radix 11/11/10 par doc (bits bas). --- */
    uint64_t* src = pairs;
    uint64_t* dst = scratch;
    const int BITS[3] = {11, 11, 10};
    const int SHIFTS[3] = {0, 11, 22};
    const uint32_t MASKS[3] = {0x7FFu, 0x7FFu, 0x3FFu};
    for (int pass = 0; pass < 3; pass++) {
        int nb = 1 << BITS[pass];
        int sh = SHIFTS[pass];
        uint32_t mask = MASKS[pass];
        uint32_t* hist = (uint32_t*)calloc((size_t)nb, 4);
        for (size_t i = 0; i < N; i++)
            hist[(src[i] >> sh) & mask]++;
        uint32_t acc = 0;
        for (int b = 0; b < nb; b++) {
            uint32_t c = hist[b]; hist[b] = acc; acc += c;
        }
        for (size_t i = 0; i < N; i++)
            dst[hist[(src[i] >> sh) & mask]++] = src[i];
        free(hist);
        uint64_t* t = src; src = dst; dst = t;
    }

    /* --- Votes (dedup par arbre) + top_n par histogramme. --- */
    typedef struct { int32_t id; int32_t vote; } IdVote2;
    IdVote2* dv = (IdVote2*)malloc((N + 1) * sizeof(IdVote2));
    int32_t* tree_seen = (int32_t*)malloc((size_t)n_trees * 4);
    if (!dv || !tree_seen) {
        free(dv); free(tree_seen); free(pairs); free(scratch);
        return -1;
    }
    for (int t = 0; t < n_trees; t++) tree_seen[t] = -1;
    uint32_t vote_hist[257] = {0};
    const int MAX_VOTE = (n_trees < 256) ? n_trees : 256;
    size_t nd = 0;
    int32_t prev = -1;
    int vote = 0;
    for (size_t i = 0; i < N; i++) {
        int32_t doc = (int32_t)(src[i] & 0xffffffffu);
        int32_t tr = (int32_t)(src[i] >> 32);
        if (doc != prev) {
            if (prev >= 0) {
                int v = vote > MAX_VOTE ? MAX_VOTE : vote;
                dv[nd++] = (IdVote2){prev, v};
                vote_hist[v]++;
            }
            prev = doc; vote = 0;
        }
        if (tree_seen[tr] != doc) { tree_seen[tr] = doc; vote++; }
    }
    if (prev >= 0) {
        int v = vote > MAX_VOTE ? MAX_VOTE : vote;
        dv[nd++] = (IdVote2){prev, v};
        vote_hist[v]++;
    }
    /* seuil de vote pour tenir dans top_n */
    int thr = MAX_VOTE;
    long take = 0;
    while (thr > 1 && take + (long)vote_hist[thr] <= top_n)
        take += vote_hist[thr--];
    /* Deux passes : d'abord TOUS les votes > seuil (ils tiennent par
       construction), puis completer avec les votes == seuil. Une seule
       passe premier-arrive laissait les petits ids a vote-seuil evincer
       les gros votes a id eleve (recall 0,97 -> 0,56, vecu).            */
    int topn_size = 0;
    for (size_t i = 0; i < nd; i++) {
        if (dv[i].vote > thr) {
            out_ids[topn_size] = dv[i].id;
            out_votes[topn_size] = dv[i].vote;
            topn_size++;
        }
    }
    for (size_t i = 0; i < nd && topn_size < top_n; i++) {
        if (dv[i].vote == thr) {
            out_ids[topn_size] = dv[i].id;
            out_votes[topn_size] = dv[i].vote;
            topn_size++;
        }
    }
    free(dv); free(tree_seen); free(pairs); free(scratch);
    return topn_size;
}
