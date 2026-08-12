"""E2E : métadonnées NATIVES (meta_store) × injection live.

Valide la thèse "filtres sans dépendance externe" :
  - bitmaps gelés mmap (persistance via compact + reopen)
  - deltas live journalisés WAL par-dessus les gelés
  - prédicat AND-de-OR évalué in-process, bitmap passé DIRECTEMENT à la
    query (mg_query_pathrank_bm) — zéro sérialisation, zéro réseau
  - équivalence exacte avec le chemin ch_state (ClickHouse) historique

Checks :
  [1] filtre simple lang=fr : cible trouvée, non-fr jamais présents
  [2] équivalence native vs ch_state sur le même prédicat (mêmes résultats)
  [3] persistance : compact → close → reopen → mêmes cardinalités, frozen
  [4] docs LIVE (HOT) métadonnés en delta post-compact : trouvés via filtre
      natif, et exclus quand le filtre ne les couvre pas
  [5] prédicat composé (lang=fr OU en) ET src=a — cardinalité et résultats
"""
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mangrove_ffi as mf                                   # noqa: E402
from mangrove_ffi import Forest, MetaStore, set_gen_version  # noqa: E402
import test_live_medians as T                               # noqa: E402
from test_live_filters import roaring_ch_state              # noqa: E402

N_LIVE = 100
LIVE_BASE = 30_000


def main():
    tmp = tempfile.mkdtemp(prefix='mangrove_meta_')
    try:
        vecs = T.make_vecs(T.N_DOCS, T.DIM, seed=42)
        base = os.path.join(tmp, 'base.fvecs')
        T.write_fvecs(base, vecs)
        idir = os.path.join(tmp, 'idx_med')
        T.build_index(base, idir, with_medians=True)
        set_gen_version(T.GEN_V)
        f = Forest(idir, n_trees=T.N_TREES, dim=T.DIM, sub_dim=T.SUB_DIM,
                   depth=T.DEPTH, n_docs=T.N_DOCS, gen_version=T.GEN_V)
        mf.clear_live_medians()
        assert mf.load_live_medians(idir) == T.MED_DEPTH

        # métadonnées des docs MAIN : lang par parité, src par modulo 3
        meta_dir = os.path.join(tmp, 'meta')
        ms = MetaStore(meta_dir)
        all_ids = np.arange(T.N_DOCS, dtype=np.uint32)
        ms.add('lang', 'fr', all_ids[all_ids % 2 == 0])
        ms.add('lang', 'en', all_ids[all_ids % 2 == 1])
        for i, s in enumerate(('a', 'b', 'c')):
            ms.add('src', s, all_ids[all_ids % 3 == i])

        def q_native(vec, bmp):
            ids, votes, n = f.query_pathrank_meta(vec, 3, 64, bmp,
                                                  top_n=500,
                                                  query_depth=T.DEPTH)
            return {int(ids[j]): int(votes[j]) for j in range(n)}

        def q_ch(vec, id_list):
            ids, votes, n = f.query_pathrank(vec, 3, 64, 500,
                                             query_depth=T.DEPTH,
                                             allowed_state=roaring_ch_state(
                                                 id_list))
            return {int(ids[j]): int(votes[j]) for j in range(n)}

        # [1] filtre simple : lang=fr
        bmp = ms.filter({'lang': 'fr'})
        card_fr = MetaStore.filter_card(bmp)
        target = 4242                                   # pair → fr
        r1 = q_native(vecs[target], bmp)
        only_fr = all(d % 2 == 0 for d in r1)
        MetaStore.filter_free(bmp)
        print(f'[1] lang=fr : card={card_fr} (attendu {T.N_DOCS//2}), '
              f'cible trouvée={target in r1}, que des fr={only_fr}')
        ok1 = card_fr == T.N_DOCS // 2 and target in r1 and only_fr

        # [2] équivalence native vs ch_state : src=b
        bmp = ms.filter({'src': 'b'})
        src_b = [int(x) for x in all_ids[all_ids % 3 == 1]]
        same = 0
        for qi in (10, 500, 7777):
            rn = q_native(vecs[qi], bmp)
            rc = q_ch(vecs[qi], src_b)
            same += (rn == rc)
        MetaStore.filter_free(bmp)
        print(f'[2] équivalence native vs ch_state : {same}/3 requêtes '
              f'identiques (ids ET votes)')

        # [3] persistance : compact, reopen
        n_frozen = ms.compact()
        assert ms.delta_docs == 0
        ms.close()
        ms = MetaStore(meta_dir)
        bmp = ms.filter({'lang': 'fr'})
        card_fr2 = MetaStore.filter_card(bmp)
        MetaStore.filter_free(bmp)
        print(f'[3] persistance : {n_frozen} clés gelées, reopen → '
              f'card lang=fr {card_fr2} (delta_docs={ms.delta_docs})')
        ok3 = card_fr2 == card_fr and ms.delta_docs == 0

        # [4] docs LIVE : hot insert + meta en DELTA par-dessus les gelés
        hotdir = os.path.join(tmp, 'hot')
        os.makedirs(hotdir, exist_ok=True)
        hot = mf._lib.mg_hot_init(T.N_TREES, T.DEPTH, hotdir.encode())
        assert hot
        mf._lib.mg_forest_set_hot_overlay(hot)
        live = T.make_vecs(N_LIVE, T.DIM, seed=999)
        for i in range(N_LIVE):
            for t in range(T.N_TREES):
                assert mf._lib.mg_hot_append(
                    hot, t, T.insert_leaf(live[i], t), LIVE_BASE + i) == 0
        live_ids = np.arange(LIVE_BASE, LIVE_BASE + N_LIVE, dtype=np.uint32)
        ms.add('lang', 'fr', live_ids)          # delta sur frozen lang=fr
        ms.add('src', 'live', live_ids)

        bmp_fr = ms.filter({'lang': 'fr'})
        found = sum(1 for i in range(N_LIVE)
                    if (LIVE_BASE + i) in q_native(live[i], bmp_fr))
        card_mixed = MetaStore.filter_card(bmp_fr)
        MetaStore.filter_free(bmp_fr)
        bmp_en = ms.filter({'lang': 'en'})
        leaks = sum(1 for i in range(N_LIVE)
                    if (LIVE_BASE + i) in q_native(live[i], bmp_en))
        MetaStore.filter_free(bmp_en)
        print(f'[4] live × meta : {found}/{N_LIVE} trouvés via lang=fr '
              f'(frozen+delta, card {card_mixed}), {leaks} fuites via lang=en')
        ok4 = (found == N_LIVE and leaks == 0
               and card_mixed == card_fr + N_LIVE)

        # [5] composé : (lang=fr OU en) ET src=a — sur les MAIN
        bmp = ms.filter({'lang': ['fr', 'en'], 'src': 'a'})
        card5 = MetaStore.filter_card(bmp)
        t5 = 9999                                # 9999 % 3 == 0 → src=a
        r5 = q_native(vecs[t5], bmp)
        only_a = all(d % 3 == 0 or d >= LIVE_BASE for d in r5)
        MetaStore.filter_free(bmp)
        att5 = len([x for x in range(T.N_DOCS) if x % 3 == 0])
        print(f'[5] (fr|en)&src=a : card={card5} (attendu {att5}), '
              f'cible={t5 in r5}, que src=a={only_a}')
        ok5 = card5 == att5 and t5 in r5 and only_a

        mf._lib.mg_forest_set_hot_overlay(None)
        mf._lib.mg_hot_free(hot)
        ms.close()
        f.close()

        ok = ok1 and same == 3 and ok3 and ok4 and ok5
        print('PASS' if ok else 'FAIL')
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
