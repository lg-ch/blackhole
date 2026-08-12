"""E2E : couche document complète (chunks + métadonnées typées + live).

Scénario : un index médianes de 20k vecteurs de fond, puis 60 documents
insérés EN LIVE via DocumentStore (chunking, embed factice déterministe,
métadonnées typées : str, int, date, float, tags), et des recherches
groupées par document avec fenêtres et filtres.

  [1] retrouvabilité : la requête = texte d'un chunk → son document en
      top-1, la fenêtre contient le texte du hit et ses voisins
  [2] filtre catégoriel lang=fr : aucun doc en dans les résultats
  [3] plage d'années ('range', 2019, 2021) sur un champ int
  [4] regex sur le champ source (expansion dictionnaire)
  [5] plage float avec FloatSpec (score ∈ [0.40, 0.60])
  [6] delete_document → le doc disparaît des recherches
"""
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mangrove_ffi as mf                                   # noqa: E402
from mangrove_ffi import Forest, set_gen_version            # noqa: E402
from mangrove.documents import DocumentStore                # noqa: E402
from mangrove.metatypes import FloatSpec                    # noqa: E402
import test_live_medians as T                               # noqa: E402

N_DOCS_LIVE = 60
LANGS = ('fr', 'en')
SRCS = ('wiki', 'blog', 'arxiv')


def fake_embed_factory(dim):
    """Embedding factice déterministe : le même texte donne toujours le
    même vecteur (hash → graine → gaussienne normalisée)."""
    def embed(texts):
        out = np.empty((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = abs(hash(('emb', t))) % (2 ** 31)
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(dim).astype(np.float32)
            out[i] = v / np.linalg.norm(v)
        return out
    return embed


def make_doc_text(d):
    parts = []
    for s in range(6):
        parts.append(f'Document {d} section {s} : ' +
                     ' '.join(f'mot{d}_{s}_{w}' for w in range(60)))
    return '\n'.join(parts)


def main():
    tmp = tempfile.mkdtemp(prefix='mangrove_docs_')
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
        hotdir = os.path.join(tmp, 'hot')
        os.makedirs(hotdir, exist_ok=True)
        hot = mf._lib.mg_hot_init(T.N_TREES, T.DEPTH, hotdir.encode())
        assert hot
        mf._lib.mg_forest_set_hot_overlay(hot)

        ds = DocumentStore(os.path.join(tmp, 'docstore'), f, hot,
                           fake_embed_factory(T.DIM),
                           dim=T.DIM, depth=T.DEPTH, sub_dim=T.SUB_DIM,
                           n_trees=T.N_TREES,
                           chunk_size=200, chunk_overlap=40,
                           float_specs={'score': FloatSpec(decimals=2)})
        # ids de chunks après le corpus de fond
        ds.state['next_chunk_id'] = T.N_DOCS

        rng = np.random.default_rng(7)
        infos = {}
        for d in range(N_DOCS_LIVE):
            key = f'doc-{d:03d}'
            meta = {'lang': LANGS[d % 2], 'src': SRCS[d % 3],
                    'year': 2015 + d % 10,
                    'score': round(float(rng.uniform(0, 1)), 2),
                    'tags': [f'tag{d % 5}', 'commun']}
            ds.insert_document(key, text=make_doc_text(d), metadata=meta)
            infos[key] = meta
        print(f'[0] {N_DOCS_LIVE} documents insérés en live '
              f'({ds.state["next_chunk_id"] - T.N_DOCS} chunks, '
              f'{ds.meta.n_keys} clés méta)')

        # [1] retrouvabilité + fenêtre
        idx = ds._idx_view()
        probe_doc, probe_seq = 'doc-017', 3
        rec = ds.get_document(probe_doc)
        e = idx[rec['first_chunk'] + probe_seq]
        probe_text = ds._read_chunk_text(int(e['off']), int(e['len']))
        r = ds.search(text=probe_text, top_docs=3, window=1)
        top1 = r and r[0]['key'] == probe_doc
        seqs = [c['seq'] for c in r[0]['chunks']] if r else []
        win_ok = top1 and r[0]['best_seq'] == probe_seq \
            and seqs == [probe_seq - 1, probe_seq, probe_seq + 1] \
            and any(probe_text == c['text'] for c in r[0]['chunks'])
        print(f'[1] retrouvabilité : top1={top1}, fenêtre={seqs}, '
              f'texte exact={win_ok}')

        # [2] filtre catégoriel
        r = ds.search(text=probe_text, where={'lang': 'fr'}, top_docs=5)
        only_fr = all(infos[x['key']]['lang'] == 'fr' for x in r)
        print(f'[2] lang=fr : {len(r)} docs, que du fr={only_fr}')

        # [3] plage d'années
        r = ds.search(text=probe_text, where={'year': ('range', 2019, 2021)},
                      top_docs=8)
        yr_ok = r and all(2019 <= infos[x['key']]['year'] <= 2021 for x in r)
        print(f'[3] year∈[2019,2021] : {len(r)} docs, bornes ok={yr_ok}')

        # [4] regex sur src : ^(wi|ar) → wiki|arxiv
        r = ds.search(text=probe_text, where={'src': ('re', '^(wi|ar)')},
                      top_docs=8)
        re_ok = r and all(infos[x['key']]['src'] in ('wiki', 'arxiv')
                          for x in r)
        print(f'[4] src~^(wi|ar) : {len(r)} docs, conformes={re_ok}')

        # [5] plage float (score quantifié à 2 décimales)
        r = ds.search(text=probe_text,
                      where={'score': ('range', 0.40, 0.60)}, top_docs=8)
        fl_ok = r and all(0.40 <= infos[x['key']]['score'] <= 0.60
                          for x in r)
        print(f'[5] score∈[0.40,0.60] : {len(r)} docs, bornes ok={fl_ok}')

        # [6] delete
        ds.delete_document(probe_doc)
        r = ds.search(text=probe_text, top_docs=3)
        gone = all(x['key'] != probe_doc for x in r) \
            and ds.get_document(probe_doc) is None
        print(f'[6] delete doc-017 : absent des résultats={gone}')

        # [7] persistance : close + reopen (registre disque + hash disque)
        ds.close()
        ds = DocumentStore(os.path.join(tmp, 'docstore'), f, hot,
                           fake_embed_factory(T.DIM),
                           dim=T.DIM, depth=T.DEPTH, sub_dim=T.SUB_DIM,
                           n_trees=T.N_TREES,
                           float_specs={'score': FloatSpec(decimals=2)})
        rec2 = ds.get_document('doc-030')
        e2 = idx[rec2['first_chunk'] + 2]
        t2 = ds._read_chunk_text(int(e2['off']), int(e2['len']))
        r = ds.search(text=t2, top_docs=3)
        persist_ok = (rec2 is not None and rec2['meta'] == infos['doc-030']
                      and r and r[0]['key'] == 'doc-030'
                      and ds.get_document(probe_doc) is None)
        print(f'[7] reopen : doc-030 retrouvé={persist_ok}')

        mf._lib.mg_forest_set_hot_overlay(None)
        mf._lib.mg_hot_free(hot)
        ds.close()
        f.close()

        ok = (win_ok and only_fr and bool(yr_ok) and bool(re_ok)
              and bool(fl_ok) and gone and persist_ok)
        print('PASS' if ok else 'FAIL')
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
