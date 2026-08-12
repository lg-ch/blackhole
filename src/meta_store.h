#ifndef META_STORE_H
#define META_STORE_H

/* meta_store — métadonnées natives de mangrove.
 *
 * Un bitmap roaring par clé "champ=valeur" :
 *   - couche GELÉE : <dir>/<clé>.rbm au format frozen CRoaring, montée en
 *     frozen view mmap → zéro copie, la RAM ne paie que les pages touchées.
 *   - couche DELTA : bitmap RAM alimenté par les inserts live, journalisé
 *     dans <dir>/meta.wal (rejoué à l'ouverture), compacté dans le gelé à
 *     la demande. Le même motif que HOT → MAIN côté vecteurs.
 *
 * Prédicat = AND de groupes OR sur des clés : évalué in-process, retourne
 * un roaring_bitmap_t* directement consommable par la query (voir
 * mg_query_pathrank_bm) — zéro sérialisation, zéro réseau.
 *
 * Clés : ASCII sans '/', longueur < 256. La composition champ=valeur est
 * la convention du SDK ; le store ne voit que des clés opaques.
 * Thread-safety : mutex global sur les écritures ; les lectures (filter)
 * prennent le même mutex le temps de composer le bitmap résultat.
 */

#include <stdint.h>

typedef struct MetaStore MetaStore;

MetaStore* meta_open(const char* dir);
void       meta_close(MetaStore* m);

/* Ajoute doc_id sous la clé (delta + WAL). Retourne 0, -1 sinon. */
int meta_add(MetaStore* m, const char* key, uint32_t doc_id);
int meta_add_batch(MetaStore* m, const char* key,
                   const uint32_t* doc_ids, int n);

/* Fusionne tous les deltas dans les fichiers gelés (tmp + rename par clé),
   tronque le WAL, remonte les frozen views. Retourne le nb de clés gelées. */
int meta_compact(MetaStore* m);

/* Bitmap composé du prédicat : AND des groupes, OR des clés d'un groupe.
   keys = tableau plat de n_total clés ; group_lens[i] = taille du groupe i.
   Clé inconnue = bitmap vide (le groupe peut quand même matcher via ses
   autres clés). Retourne un bitmap NEUF (à libérer via meta_filter_free),
   NULL si allocation impossible. */
void* meta_filter(MetaStore* m, const char** keys,
                  const int* group_lens, int n_groups);
void  meta_filter_free(void* bmp);
int64_t meta_bitmap_card(void* bmp);

/* Stats : nombre de clés connues, docs en delta (non gelés). */
int      meta_n_keys(MetaStore* m);
int64_t  meta_delta_docs(MetaStore* m);

#endif
