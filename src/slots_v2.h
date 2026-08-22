/* Store V2 : feuilles a SLOTS FIXES, adressage arithmetique pur.
   Format par arbre (tree%05d.slt) : n_leaves slots de slot_bytes octets,
   slot = [u32 count][u32 doc_ids x count][padding zero].
   offset(leaf) = (uint64_t)leaf * slot_bytes — AUCUN index, AUCUNE phase 1.
   A qd < depth, une probe couvre 2^k slots CONTIGUS : une seule lecture
   sequentielle par (arbre, probe). Une seule vague io_uring par requete. */
#ifndef SLOTS_V2_H
#define SLOTS_V2_H

#include <stdint.h>

void* slots_v2_open(const char* dir, int n_trees, int n_leaves,
                    int slot_bytes);
void  slots_v2_close(void* h);

/* Lit les ranges de slots des probes (une vague io_uring), parse les ids,
   vote par (doc, arbre distinct), renvoie le top_n par votes.
   leaves[li] : feuille a qd (forme node - base), -1 = probe inactive.
   leaf_tree[li] : arbre de la probe. k_shift = depth - qd.
   Retourne topn_size ou -1. */
int slots_v2_collect(void* h, const int32_t* leaves,
                     const int32_t* leaf_tree, int nL, int k_shift,
                     int n_trees, int top_n,
                     int32_t* out_ids, int32_t* out_votes);

#endif
