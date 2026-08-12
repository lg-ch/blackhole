/* meta_store — métadonnées natives : frozen views mmap + deltas WAL.
 * Voir meta_store.h pour le contrat. */
#define _POSIX_C_SOURCE 200809L
#include "meta_store.h"

#include <roaring/roaring.h>

#include <dirent.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define META_KEY_MAX   256
#define META_WAL_NAME  "meta.wal"

typedef struct {
    char   key[META_KEY_MAX];
    /* couche gelée (NULL si pas encore de fichier) */
    const roaring_bitmap_t* frozen;   /* frozen view sur map_base */
    void*  map_base;
    size_t map_len;
    /* couche delta (NULL tant que vide) */
    roaring_bitmap_t* delta;
} MetaKey;

struct MetaStore {
    char            dir[512];
    MetaKey*        keys;
    int             n_keys, cap_keys;
    FILE*           wal;
    pthread_mutex_t mu;
};

/* ---- helpers ---- */

static int key_ok(const char* k) {
    size_t n = strlen(k);
    if (n == 0 || n >= META_KEY_MAX) return 0;
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)k[i];
        if (c == '/' || c < 0x20 || c == 0x7f) return 0;
    }
    return 1;
}

static void frozen_path(const MetaStore* m, const char* key,
                        char* out, size_t cap) {
    snprintf(out, cap, "%s/%s.rbm", m->dir, key);
}

static MetaKey* key_find(MetaStore* m, const char* key) {
    for (int i = 0; i < m->n_keys; i++) {
        if (strcmp(m->keys[i].key, key) == 0) return &m->keys[i];
    }
    return NULL;
}

static MetaKey* key_get(MetaStore* m, const char* key) {
    MetaKey* k = key_find(m, key);
    if (k) return k;
    if (m->n_keys == m->cap_keys) {
        int nc = m->cap_keys ? m->cap_keys * 2 : 64;
        MetaKey* nk = (MetaKey*)realloc(m->keys, (size_t)nc * sizeof(MetaKey));
        if (!nk) return NULL;
        m->keys = nk;
        m->cap_keys = nc;
    }
    k = &m->keys[m->n_keys++];
    memset(k, 0, sizeof(*k));
    snprintf(k->key, META_KEY_MAX, "%s", key);
    return k;
}

/* Monte (mmap) le fichier gelé d'une clé s'il existe. 0 ok / -1 erreur. */
static int key_map_frozen(MetaStore* m, MetaKey* k) {
    if (k->frozen) return 0;
    char path[600];
    frozen_path(m, k->key, path, sizeof(path));
    int fd = open(path, O_RDONLY);
    if (fd < 0) return 0;                 /* pas de gelé : delta seul */
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size == 0) { close(fd); return 0; }
    void* base = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (base == MAP_FAILED) return -1;
    const roaring_bitmap_t* v =
        roaring_bitmap_frozen_view((const char*)base, (size_t)st.st_size);
    if (!v) { munmap(base, (size_t)st.st_size); return -1; }
    k->map_base = base;
    k->map_len  = (size_t)st.st_size;
    k->frozen   = v;
    return 0;
}

static void key_unmap_frozen(MetaKey* k) {
    /* Une frozen view vit DANS le buffer mmapé : pas de
       roaring_bitmap_free ici, munmap suffit (free = UB). */
    k->frozen = NULL;
    if (k->map_base) {
        munmap(k->map_base, k->map_len);
        k->map_base = NULL;
        k->map_len = 0;
    }
}

/* WAL : [u16 klen][klen octets][u32 doc_id], little-endian, append-only. */
static int wal_append(MetaStore* m, const char* key, uint32_t doc) {
    uint16_t kl = (uint16_t)strlen(key);
    if (fwrite(&kl, 2, 1, m->wal) != 1) return -1;
    if (fwrite(key, 1, kl, m->wal) != kl) return -1;
    if (fwrite(&doc, 4, 1, m->wal) != 1) return -1;
    return 0;
}

static int delta_add(MetaStore* m, const char* key, uint32_t doc) {
    MetaKey* k = key_get(m, key);
    if (!k) return -1;
    if (!k->delta) {
        k->delta = roaring_bitmap_create();
        if (!k->delta) return -1;
    }
    roaring_bitmap_add(k->delta, doc);
    return 0;
}

static void wal_replay(MetaStore* m) {
    char path[600];
    snprintf(path, sizeof(path), "%s/%s", m->dir, META_WAL_NAME);
    FILE* f = fopen(path, "rb");
    if (!f) return;
    char key[META_KEY_MAX];
    for (;;) {
        uint16_t kl;
        uint32_t doc;
        if (fread(&kl, 2, 1, f) != 1) break;
        if (kl == 0 || kl >= META_KEY_MAX) break;      /* corrompu : stop */
        if (fread(key, 1, kl, f) != kl) break;
        key[kl] = 0;
        if (fread(&doc, 4, 1, f) != 1) break;          /* enreg. tronqué */
        delta_add(m, key, doc);
    }
    fclose(f);
}

/* ---- API ---- */

MetaStore* meta_open(const char* dir) {
    if (!dir) return NULL;
    MetaStore* m = (MetaStore*)calloc(1, sizeof(MetaStore));
    if (!m) return NULL;
    snprintf(m->dir, sizeof(m->dir), "%s", dir);
    mkdir(dir, 0755);
    pthread_mutex_init(&m->mu, NULL);

    /* Recense les clés gelées (montées paresseusement au premier filter). */
    DIR* d = opendir(dir);
    if (d) {
        struct dirent* e;
        while ((e = readdir(d)) != NULL) {
            size_t n = strlen(e->d_name);
            if (n > 4 && strcmp(e->d_name + n - 4, ".rbm") == 0
                && n - 4 < META_KEY_MAX) {
                char key[META_KEY_MAX];
                memcpy(key, e->d_name, n - 4);
                key[n - 4] = 0;
                key_get(m, key);
            }
        }
        closedir(d);
    }

    wal_replay(m);

    char wpath[600];
    snprintf(wpath, sizeof(wpath), "%s/%s", m->dir, META_WAL_NAME);
    m->wal = fopen(wpath, "ab");
    if (!m->wal) { meta_close(m); return NULL; }
    return m;
}

void meta_close(MetaStore* m) {
    if (!m) return;
    for (int i = 0; i < m->n_keys; i++) {
        key_unmap_frozen(&m->keys[i]);
        if (m->keys[i].delta) roaring_bitmap_free(m->keys[i].delta);
    }
    free(m->keys);
    if (m->wal) fclose(m->wal);
    pthread_mutex_destroy(&m->mu);
    free(m);
}

int meta_add(MetaStore* m, const char* key, uint32_t doc_id) {
    return meta_add_batch(m, key, &doc_id, 1);
}

int meta_add_batch(MetaStore* m, const char* key,
                   const uint32_t* doc_ids, int n) {
    if (!m || !key_ok(key) || !doc_ids || n <= 0) return -1;
    pthread_mutex_lock(&m->mu);
    int rc = 0;
    for (int i = 0; i < n; i++) {
        if (delta_add(m, key, doc_ids[i]) != 0 ||
            wal_append(m, key, doc_ids[i]) != 0) { rc = -1; break; }
    }
    if (rc == 0 && fflush(m->wal) != 0) rc = -1;
    pthread_mutex_unlock(&m->mu);
    return rc;
}

int meta_compact(MetaStore* m) {
    if (!m) return -1;
    pthread_mutex_lock(&m->mu);
    int frozen_count = 0;
    for (int i = 0; i < m->n_keys; i++) {
        MetaKey* k = &m->keys[i];
        if (!k->delta || roaring_bitmap_is_empty(k->delta)) continue;
        if (key_map_frozen(m, k) != 0) { frozen_count = -1; break; }

        /* union gelé + delta → nouveau gelé */
        roaring_bitmap_t* merged = roaring_bitmap_copy(k->delta);
        if (!merged) { frozen_count = -1; break; }
        if (k->frozen) roaring_bitmap_or_inplace(merged, k->frozen);
        roaring_bitmap_run_optimize(merged);
        roaring_bitmap_shrink_to_fit(merged);

        size_t sz = roaring_bitmap_frozen_size_in_bytes(merged);
        char* buf = (char*)aligned_alloc(32, (sz + 31) & ~(size_t)31);
        if (!buf) { roaring_bitmap_free(merged); frozen_count = -1; break; }
        roaring_bitmap_frozen_serialize(merged, buf);
        roaring_bitmap_free(merged);

        char path[600], tmp[608];
        frozen_path(m, k->key, path, sizeof(path));
        snprintf(tmp, sizeof(tmp), "%s.tmp", path);
        FILE* f = fopen(tmp, "wb");
        int ok = f && fwrite(buf, 1, sz, f) == sz;
        if (f) { fflush(f); fsync(fileno(f)); fclose(f); }
        free(buf);
        if (!ok || rename(tmp, path) != 0) {
            unlink(tmp);
            frozen_count = -1;
            break;
        }

        key_unmap_frozen(k);              /* remonte la nouvelle version */
        if (key_map_frozen(m, k) != 0) { frozen_count = -1; break; }
        roaring_bitmap_free(k->delta);
        k->delta = NULL;
        frozen_count++;
    }
    if (frozen_count >= 0) {
        /* tous les deltas sont gelés : le WAL peut repartir de zéro */
        char wpath[600];
        snprintf(wpath, sizeof(wpath), "%s/%s", m->dir, META_WAL_NAME);
        fclose(m->wal);
        m->wal = fopen(wpath, "wb");      /* truncate */
        if (m->wal) { fclose(m->wal); m->wal = fopen(wpath, "ab"); }
        if (!m->wal) frozen_count = -1;
    }
    pthread_mutex_unlock(&m->mu);
    return frozen_count;
}

void* meta_filter(MetaStore* m, const char** keys,
                  const int* group_lens, int n_groups) {
    if (!m || !keys || !group_lens || n_groups <= 0) return NULL;
    pthread_mutex_lock(&m->mu);
    roaring_bitmap_t* result = NULL;
    int ki = 0, fail = 0;
    for (int g = 0; g < n_groups && !fail; g++) {
        roaring_bitmap_t* grp = roaring_bitmap_create();
        if (!grp) { fail = 1; break; }
        for (int j = 0; j < group_lens[g]; j++, ki++) {
            MetaKey* k = key_find(m, keys[ki]);
            if (!k) continue;                       /* clé inconnue = vide */
            if (key_map_frozen(m, k) != 0) { fail = 1; break; }
            if (k->frozen) roaring_bitmap_or_inplace(grp, k->frozen);
            if (k->delta)  roaring_bitmap_or_inplace(grp, k->delta);
        }
        if (fail) { roaring_bitmap_free(grp); break; }
        if (!result) {
            result = grp;
        } else {
            roaring_bitmap_and_inplace(result, grp);
            roaring_bitmap_free(grp);
        }
    }
    pthread_mutex_unlock(&m->mu);
    if (fail) {
        if (result) roaring_bitmap_free(result);
        return NULL;
    }
    return result;
}

void meta_filter_free(void* bmp) {
    if (bmp) roaring_bitmap_free((roaring_bitmap_t*)bmp);
}

int64_t meta_bitmap_card(void* bmp) {
    if (!bmp) return -1;
    return (int64_t)roaring_bitmap_get_cardinality((roaring_bitmap_t*)bmp);
}

int meta_n_keys(MetaStore* m) { return m ? m->n_keys : -1; }

int64_t meta_delta_docs(MetaStore* m) {
    if (!m) return -1;
    int64_t n = 0;
    pthread_mutex_lock(&m->mu);
    for (int i = 0; i < m->n_keys; i++) {
        if (m->keys[i].delta)
            n += (int64_t)roaring_bitmap_get_cardinality(m->keys[i].delta);
    }
    pthread_mutex_unlock(&m->mu);
    return n;
}
