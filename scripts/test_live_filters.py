"""E2E : filtres métadonnées × injection live (HOT overlay) × tombstones.

Le FILTER PATH du collect (query_tree.c) ne lisait jamais le HOT overlay.
Conséquence : avec un bitmap de filtre actif — ou la MOINDRE tombstone,
composée dans le même bitmap — les docs insérés en live devenaient
invisibles pour la query jusqu'à la compaction. Pour du RAG en prod
(filtres partout, deletes fréquents), c'est fatal.

  [0] filtre sur docs MAIN, sans HOT : sémantique de filtre inchangée
      (et valide notre encodage roaring portable fait main)
  [1] sans filtre         : docs live trouvés (sanité, cf test_live_medians)
  [2] filtre INCLUANT les live : ils DOIVENT être trouvés
      (avant le fix : 0/100 — invisibles)
  [3] filtre EXCLUANT les live : ils ne doivent PAS fuiter
  [4] une tombstone sur UN doc MAIN, aucun filtre appelant :
      les live restent visibles (avant : tous cachés) et le doc
      supprimé disparaît

Les doc_ids du test restent < 65536 → un seul container roaring
(encodage portable single-array fait main, pas de dépendance pyroaring).
"""
import os
import shutil
import struct
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mangrove_ffi as mf                                   # noqa: E402
from mangrove_ffi import Forest, set_gen_version            # noqa: E402
# réutilise le harnais (constantes, build, bindings hot déclarés à l'import)
import test_live_medians as T                               # noqa: E402
from ctypes import c_uint32, POINTER, c_int                 # noqa: E402

N_LIVE = 100
LIVE_BASE = 30_000          # ids live : 30000..30099 (< 65536)


def roaring_ch_state(ids):
    """Bitmap roaring portable (1 container) enveloppé au format état
    ClickHouse groupBitmap : [0x01][varint taille][portable].
    Spec roaring : container ARRAY (u16 triés) jusqu'à 4096 éléments,
    container BITSET (8 Ko de bits) au-delà — un array sur-rempli est
    parsé comme du bitset → bitmap poubelle silencieuse."""
    ids = sorted(set(int(x) for x in ids))
    assert ids and ids[-1] < 65536, 'test limité à 1 container'
    body = struct.pack('<IIHHI', 12346, 1, 0, len(ids) - 1, 16)
    if len(ids) > 4096:
        bits = np.zeros(1024, dtype=np.uint64)
        arr = np.asarray(ids, dtype=np.uint64)
        np.bitwise_or.at(bits, (arr >> 6).astype(np.int64),
                         np.uint64(1) << (arr & np.uint64(63)))
        body += bits.tobytes()
    else:
        body += struct.pack('<%dH' % len(ids), *ids)
    n = len(body)
    varint = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        varint.append(b | (0x80 if n else 0))
        if not n:
            break
    return b'\x01' + bytes(varint) + body


def main():
    tmp = tempfile.mkdtemp(prefix='mangrove_live_filters_')
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

        rng = np.random.default_rng(77)
        some_main = [int(x) for x in rng.choice(T.N_DOCS, 500, replace=False)]

        def q(vec, allowed=None):
            ids, votes, n = f.query_pathrank(vec, 3, 64, 500,
                                             query_depth=T.DEPTH,
                                             allowed_state=allowed)
            return {int(ids[j]): int(votes[j]) for j in range(n)}

        # [0] filtre MAIN sans HOT : la query d'un doc filtré-in le trouve,
        #     un doc hors filtre n'apparaît jamais.
        target = some_main[0]
        r = q(vecs[target], roaring_ch_state(some_main))
        only_allowed = all(d in set(some_main) for d in r)
        print(f'[0] filtre main sans HOT : cible trouvée={target in r}, '
              f'hors-filtre exclus={only_allowed}')
        ok0 = target in r and only_allowed

        # HOT + inserts live routés médianes
        hotdir = os.path.join(tmp, 'hot')
        os.makedirs(hotdir, exist_ok=True)
        hot = mf._lib.mg_hot_init(T.N_TREES, T.DEPTH, hotdir.encode())
        assert hot
        mf._lib.mg_forest_set_hot_overlay(hot)
        live = T.make_vecs(N_LIVE, T.DIM, seed=999)
        for i in range(N_LIVE):
            for t in range(T.N_TREES):
                rc = mf._lib.mg_hot_append(
                    hot, t, T.insert_leaf(live[i], t), LIVE_BASE + i)
                assert rc == 0
        live_ids = list(range(LIVE_BASE, LIVE_BASE + N_LIVE))

        # [1] sans filtre
        f1 = sum(1 for i in range(N_LIVE)
                 if (LIVE_BASE + i) in q(live[i]))
        print(f'[1] sans filtre          : {f1}/{N_LIVE} live trouvés')

        # [2] filtre incluant les live (+ du main)
        st_incl = roaring_ch_state(live_ids + some_main)
        f2 = sum(1 for i in range(N_LIVE)
                 if (LIVE_BASE + i) in q(live[i], st_incl))
        print(f'[2] filtre INCLUANT live : {f2}/{N_LIVE} live trouvés '
              f'(le bug les cachait tous)')

        # [3] filtre excluant les live
        st_excl = roaring_ch_state(some_main)
        leaks = sum(1 for i in range(N_LIVE)
                    if (LIVE_BASE + i) in q(live[i], st_excl))
        print(f'[3] filtre EXCLUANT live : {leaks} fuites (attendu 0)')

        # [4] une tombstone sur un doc main, aucun filtre appelant
        f.tombstone_add(target)
        f4 = sum(1 for i in range(N_LIVE)
                 if (LIVE_BASE + i) in q(live[i]))
        gone = target not in q(vecs[target])
        f.tombstone_remove(target)
        print(f'[4] avec 1 tombstone     : {f4}/{N_LIVE} live visibles '
              f'(le bug les cachait tous), doc supprimé absent={gone}')

        mf._lib.mg_forest_set_hot_overlay(None)
        mf._lib.mg_hot_free(hot)
        f.close()

        ok = (ok0 and f1 == N_LIVE and f2 == N_LIVE and leaks == 0
              and f4 == N_LIVE and gone)
        print('PASS' if ok else 'FAIL')
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
