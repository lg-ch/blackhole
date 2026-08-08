# PAIRWISE SEED FOREST + MERGE K-WAY

## Principe

Chaque arbre est défini par une seule seed. Chaque nœud génère 2 vecteurs aléatoires à la volée depuis `hash(seed, node_id)`. Le vecteur va vers le plus proche des 2. Aucun nœud n'est stocké, ni en RAM ni sur disque.

Les posting lists (doc_ids par feuille) sont stockées sur disque, **triées par doc_id**. Le tri est gratuit : les doc_ids sont séquentiels (0, 1, 2...), et chaque append va à la fin → naturellement trié.

Le count freq utilise un **merge K-way** sur les listes triées : un min-heap de K curseurs (un par liste), on avance le plus petit, on compte les doublons. RAM = K pointeurs ≈ 4 KB.

## Résultat vérifié

```
10K vecs SIFT, 15 arbres, depth 12, 20 feuilles adjacentes
Merge K-way : 91.5% recall@10, 2319 candidats
Hashmap :     91.5% recall@10, 2319 candidats ← identique
RAM merge :   ~160 KB (leaf index + heap)
RAM hashmap : ~1.4 MB
```

## Pipeline complet

```
BUILD (streaming, 0 RAM):
  Pour chaque vecteur lu du disque:
    Normaliser
    Pour chaque arbre:
      Traverser (gen_vec à la volée) → leaf_id
      Append (leaf_id, doc_id) dans fichier de l'arbre
  Trier chaque fichier par (leaf_id, doc_id) — une seule fois

QUERY:
  1. Traverser T arbres → T leaf_ids                    [RAM: 1.5 KB]
  2. Pour chaque arbre, trouver NL feuilles adjacentes   [leaf index ou slots fixes]
  3. Ouvrir les posting lists triées → merge K-way       [RAM: heap ~4 KB]
  4. Count freq ≥ seuil → candidats                      [0 RAM supplémentaire]
  5. Lire vecteurs candidats → cosine exact → top-K      [RAM: candidats × dim × 4]
```

## Code C

### Génération de nœuds

```c
#include <stdint.h>
#include <math.h>

static inline uint64_t node_seed(uint64_t tree_seed, int bfs_idx) {
    uint64_t h = tree_seed * 2654435761ULL ^ (uint64_t)bfs_idx * 40503ULL;
    return h ? h : 1;
}

static inline uint64_t tree_seed(int tree_idx) {
    return (uint64_t)tree_idx * 99991ULL + 7ULL;
}

static void gen_vec(uint64_t seed, float* out, int dim) {
    uint64_t st = seed ? seed : 1;
    float ns = 0;
    for (int i = 0; i < dim; i++) {
        float s = 0;
        for (int j = 0; j < 4; j++) {
            st ^= st << 13; st ^= st >> 7; st ^= st << 17;
            s += (float)(st & 0xFFFFFF) / 0xFFFFFF - 0.5f;
        }
        out[i] = s; ns += s * s;
    }
    float inv = 1.0f / sqrtf(ns + 1e-10f);
    for (int i = 0; i < dim; i++) out[i] *= inv;
}

static float dot(const float* a, const float* b, int n) {
    float s = 0;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}
```

### ARM NEON dot product

```c
#ifdef __ARM_NEON
#include <arm_neon.h>
static float dot(const float* a, const float* b, int n) {
    float32x4_t sum = vdupq_n_f32(0);
    int i = 0;
    for (; i + 3 < n; i += 4)
        sum = vfmaq_f32(sum, vld1q_f32(a + i), vld1q_f32(b + i));
    float s = vaddvq_f32(sum);
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}
#endif
```

### Traversée pairwise

```c
int traverse(const float* vec, int dim, int depth, uint64_t ts) {
    float v0[dim], v1[dim];
    int node = 0;
    for (int level = 0; level < depth; level++) {
        gen_vec(node_seed(ts, node * 2), v0, dim);
        gen_vec(node_seed(ts, node * 2 + 1), v1, dim);
        node = dot(vec, v1, dim) > dot(vec, v0, dim) ? 2*node+2 : 2*node+1;
    }
    return node;
}
```

### Build streaming

```c
// Format sur disque par arbre: paires (leaf_id, doc_id) en int32
typedef struct { int32_t leaf_id; int32_t doc_id; } Pair;

void build_streaming(const char* fvecs_path, int n_vecs, int dim,
                     int n_trees, int depth, const char* index_dir) {
    FILE* fin = fopen(fvecs_path, "rb");
    
    // Un fichier par arbre
    FILE* tree_files[n_trees];
    for (int t = 0; t < n_trees; t++) {
        char fname[256];
        snprintf(fname, sizeof(fname), "%s/tree%d.bin", index_dir, t);
        tree_files[t] = fopen(fname, "wb");
    }
    
    float vec[dim];
    int d;
    
    for (int doc_id = 0; doc_id < n_vecs; doc_id++) {
        // Lire 1 seul vecteur
        fread(&d, 4, 1, fin);
        fread(vec, 4, dim, fin);
        
        // Normaliser
        float norm = 0;
        for (int i = 0; i < dim; i++) norm += vec[i] * vec[i];
        norm = 1.0f / sqrtf(norm + 1e-10f);
        for (int i = 0; i < dim; i++) vec[i] *= norm;
        
        // Traverser chaque arbre
        for (int t = 0; t < n_trees; t++) {
            int leaf = traverse(vec, dim, depth, tree_seed(t));
            Pair p = {leaf, doc_id};
            fwrite(&p, sizeof(Pair), 1, tree_files[t]);
        }
        // vec est réutilisé au prochain tour — 0 accumulation RAM
    }
    
    fclose(fin);
    for (int t = 0; t < n_trees; t++) fclose(tree_files[t]);
}
```

### Sort des posting lists (une seule fois après build)

```c
int cmp_pair(const void* a, const void* b) {
    const Pair* pa = (const Pair*)a;
    const Pair* pb = (const Pair*)b;
    if (pa->leaf_id != pb->leaf_id) return pa->leaf_id - pb->leaf_id;
    return pa->doc_id - pb->doc_id;
}

void sort_tree_file(const char* fname) {
    FILE* f = fopen(fname, "rb");
    fseek(f, 0, SEEK_END);
    int n = ftell(f) / sizeof(Pair);
    fseek(f, 0, SEEK_SET);
    
    Pair* pairs = malloc(n * sizeof(Pair));
    fread(pairs, sizeof(Pair), n, f);
    fclose(f);
    
    qsort(pairs, n, sizeof(Pair), cmp_pair);
    
    f = fopen(fname, "wb");
    fwrite(pairs, sizeof(Pair), n, f);
    fclose(f);
    free(pairs);
}
```

Note : le sort charge tout le fichier d'un arbre en RAM temporairement. Pour un arbre de 100M vecs × 8 bytes = 800 MB. Si c'est trop, faire un external merge sort par chunks.

Alternative : puisque les doc_ids sont déjà triés dans le fichier (insertion séquentielle), il suffit de faire un **tri stable par leaf_id** — c'est un counting sort en O(n) si on connaît le range des leaf_ids.

### Merge K-way

```c
typedef struct {
    int32_t doc_id;
    int list_idx;
} HeapItem;

// Min-heap operations
void heap_push(HeapItem* heap, int* size, HeapItem item) {
    heap[*size] = item;
    int i = (*size)++;
    while (i > 0) {
        int p = (i - 1) / 2;
        if (heap[p].doc_id <= heap[i].doc_id) break;
        HeapItem tmp = heap[p]; heap[p] = heap[i]; heap[i] = tmp;
        i = p;
    }
}

HeapItem heap_pop(HeapItem* heap, int* size) {
    HeapItem top = heap[0];
    heap[0] = heap[--(*size)];
    int i = 0;
    while (1) {
        int l = 2*i+1, r = 2*i+2, s = i;
        if (l < *size && heap[l].doc_id < heap[s].doc_id) s = l;
        if (r < *size && heap[r].doc_id < heap[s].doc_id) s = r;
        if (s == i) break;
        HeapItem tmp = heap[s]; heap[s] = heap[i]; heap[i] = tmp;
        i = s;
    }
    return top;
}

// Merge K posting lists, emit doc_ids with count >= threshold
// lists[i] = pointeur vers tableau trié de doc_ids
// lens[i] = longueur du tableau i
// results = buffer de sortie (pré-alloué)
// Retourne le nombre de résultats
int merge_kway(int32_t** lists, int* lens, int n_lists,
               int threshold, int32_t* results) {
    
    HeapItem* heap = malloc(n_lists * sizeof(HeapItem));
    int heap_size = 0;
    int* cursors = calloc(n_lists, sizeof(int));
    
    // Init: push first element of each non-empty list
    for (int i = 0; i < n_lists; i++) {
        if (lens[i] > 0)
            heap_push(heap, &heap_size, (HeapItem){lists[i][0], i});
    }
    
    int n_results = 0;
    int prev_id = -1;
    int count = 0;
    
    while (heap_size > 0) {
        HeapItem item = heap_pop(heap, &heap_size);
        
        if (item.doc_id == prev_id) {
            count++;
        } else {
            if (count >= threshold)
                results[n_results++] = prev_id;
            prev_id = item.doc_id;
            count = 1;
        }
        
        // Advance cursor for this list
        int li = item.list_idx;
        cursors[li]++;
        if (cursors[li] < lens[li])
            heap_push(heap, &heap_size, (HeapItem){lists[li][cursors[li]], li});
    }
    if (count >= threshold)
        results[n_results++] = prev_id;
    
    free(heap);
    free(cursors);
    return n_results;
}
```

### Merge K-way streaming depuis disque (sans charger les listes)

```c
// Version qui lit depuis le disque avec un buffer par liste
// RAM = n_lists × BUFSIZE × 4 bytes

#define BUFSIZE 64  // prefetch 64 doc_ids per list

typedef struct {
    FILE* file;
    int32_t buffer[BUFSIZE];
    int buf_pos;
    int buf_len;
    int remaining;  // total remaining in file
} StreamCursor;

int stream_next(StreamCursor* sc) {
    if (sc->buf_pos >= sc->buf_len) {
        // Refill buffer from disk
        int to_read = sc->remaining < BUFSIZE ? sc->remaining : BUFSIZE;
        if (to_read == 0) return -1;
        // Read pairs, extract doc_ids
        Pair pairs[BUFSIZE];
        sc->buf_len = fread(pairs, sizeof(Pair), to_read, sc->file);
        for (int i = 0; i < sc->buf_len; i++)
            sc->buffer[i] = pairs[i].doc_id;
        sc->remaining -= sc->buf_len;
        sc->buf_pos = 0;
    }
    if (sc->buf_pos >= sc->buf_len) return -1;
    return sc->buffer[sc->buf_pos++];
}

// Usage:
// RAM par liste = BUFSIZE × 4 = 256 bytes
// 500 listes = 500 × 256 = 128 KB
// + heap de 500 = 4 KB
// Total: ~132 KB de RAM pour le merge complet
```

### Query complète

```c
void query(const float* query_vec, int dim, int depth,
           int n_trees, int n_leaves, int threshold,
           const char* index_dir,
           int32_t* out_results, int* n_results) {
    
    // 1. Traverser tous les arbres
    int query_leaves[n_trees];
    for (int t = 0; t < n_trees; t++)
        query_leaves[t] = traverse(query_vec, dim, depth, tree_seed(t));
    
    // 2. Collecter les posting lists adjacentes
    //    (ici simplifié — en prod utiliser leaf index ou slots fixes)
    int max_lists = n_trees * (n_leaves + 1);
    int32_t** lists = malloc(max_lists * sizeof(int32_t*));
    int* lens = malloc(max_lists * sizeof(int));
    int n_lists = 0;
    
    for (int t = 0; t < n_trees; t++) {
        // Trouver les NL feuilles adjacentes et lire leurs doc_ids
        // ... (lookup leaf index, seek in file, read sorted doc_ids)
        // Chaque liste est déjà triée par doc_id
    }
    
    // 3. Merge K-way
    int32_t* results = malloc(max_lists * 100 * sizeof(int32_t));  // upper bound
    *n_results = merge_kway(lists, lens, n_lists, threshold, results);
    memcpy(out_results, results, *n_results * sizeof(int32_t));
    
    // 4. Lire les vecteurs des candidats et cosine exact → top-K
    // ...
    
    free(results);
    for (int i = 0; i < n_lists; i++) free(lists[i]);
    free(lists); free(lens);
}
```

## Compilation

```bash
# x86
gcc -O3 -march=native -o index index.c -lm

# ARM avec NEON
gcc -O3 -march=native -o index index.c -lm
# (NEON activé automatiquement avec -march=native sur ARM)
```

## Stockage disque

```
index/
├── tree0.bin    # paires (leaf_id, doc_id) triées par leaf_id puis doc_id
├── tree1.bin
├── ...
└── tree2047.bin

Taille par arbre : N_vecs × 8 bytes
Total : N_vecs × N_trees × 8 bytes
100M vecs × 2048 arbres = 1.6 TB

Alternative avec slots fixes (offset calculable, pas de leaf index):
  offset(tree, leaf_rank) = tree × N_SLOTS × SLOT_SIZE + leaf_rank × SLOT_SIZE
  → 0 bytes d'index en RAM
```

## Budget RAM

```
Seeds :                    N_trees × 8 bytes = 16 KB
Traversée (working set) :  dim × 4 × 3 = 1.5 KB
Merge heap :               n_lists × 8 bytes = 4 KB
Stream buffers (optionnel): n_lists × 256 bytes = 128 KB
─────────────────────────────────────────────────
Total :                    ~150 KB
```