"""documents — la couche document de mangrove (RAG-ready).

Transforme le moteur vectoriel (Forest + HOT + MetaStore) en base
documentaire : insertion de documents chunkés, métadonnées typées,
recherche groupée par document parent avec fenêtres de contexte.

Layout <root>/ (tout est fichiers + pread/mmap — zéro RAM résidente) :
  state.json    compteurs (next_chunk_id, next_doc_id) + config
  docs.log      textes des chunks, append-only, utf-8
  chunks.idx    24 o/chunk : [u64 off][u32 len][u32 parent][u32 seq][u32 pad]
  doc_map.jsonl 1 ligne/document : clé, plage de chunks, métadonnées
  vecs.fbin     vecteurs des chunks (rerank L2), header fbin entretenu
  meta/         MetaStore natif (bitmaps par champ=valeur)

Le côté vectoriel (Forest ouvert, HOT overlay armé) est fourni par
l'appelant — la couche route les chunks via traverse_batch (médianes
live comprises) et les insère via hot_append_block.

Cohérence crash (MVP) : textes/index/vecs/méta sont écrits AVANT le HOT
(qui a son propre WAL applicatif) ; un crash entre les deux se rejoue en
réinsérant le document (les doublons HOT se dédupliquent à la compaction).
"""
from __future__ import annotations

import json
import os
import struct

import numpy as np

import mangrove_ffi as mf
from .metatypes import compile_where, encode_meta

IDX_DTYPE = np.dtype([('off', '<u8'), ('len', '<u4'), ('parent', '<u4'),
                      ('seq', '<u4'), ('pad', '<u4')])


def chunk_text(text: str, size: int = 800, overlap: int = 160) -> list[str]:
    """Fenêtres de caractères avec chevauchement, coupées de préférence
    sur une frontière d'espace proche de la fin de fenêtre."""
    if size <= 0 or overlap >= size:
        raise ValueError('size > 0 et overlap < size requis')
    out = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + size, n)
        if j < n:
            k = text.rfind(' ', i + size - min(80, size // 4), j)
            if k > i:
                j = k
        out.append(text[i:j])
        if j >= n:
            break
        i = max(j - overlap, i + 1)
    return out


class DocumentStore:
    def __init__(self, root: str, forest, hot_handle, embed_fn,
                 dim: int, depth: int, sub_dim: int, n_trees: int,
                 chunk_size: int = 800, chunk_overlap: int = 160,
                 float_specs: dict | None = None):
        self.root = root
        self.forest = forest
        self.hot = hot_handle
        self.embed = embed_fn
        self.dim, self.depth = dim, depth
        self.sub_dim, self.n_trees = sub_dim, n_trees
        self.float_specs = float_specs or {}
        os.makedirs(root, exist_ok=True)
        self.meta = mf.MetaStore(os.path.join(root, 'meta'))
        self._paths = {n: os.path.join(root, n) for n in
                       ('state.json', 'docs.log', 'chunks.idx',
                        'doc_map.jsonl', 'vecs.fbin')}
        if os.path.exists(self._paths['state.json']):
            self.state = json.load(open(self._paths['state.json']))
        else:
            self.state = {'next_chunk_id': 0, 'next_doc_id': 0,
                          'chunk_size': chunk_size,
                          'chunk_overlap': chunk_overlap}
        self.state.setdefault('chunk_size', chunk_size)
        self.state.setdefault('chunk_overlap', chunk_overlap)
        # doc_map en RAM : petit (1 entrée/document, pas par chunk)
        self.docs: dict[str, dict] = {}
        if os.path.exists(self._paths['doc_map.jsonl']):
            with open(self._paths['doc_map.jsonl']) as fh:
                for line in fh:
                    rec = json.loads(line)
                    if rec.get('deleted'):
                        self.docs.pop(rec['key'], None)
                    else:
                        self.docs[rec['key']] = rec
        if not os.path.exists(self._paths['vecs.fbin']):
            with open(self._paths['vecs.fbin'], 'wb') as fh:
                fh.write(struct.pack('<ii', 0, dim))
        if not os.path.exists(self._paths['chunks.idx']):
            open(self._paths['chunks.idx'], 'wb').close()

    # ----- internes -----

    def _save_state(self):
        tmp = self._paths['state.json'] + '.tmp'
        with open(tmp, 'w') as fh:
            json.dump(self.state, fh)
        os.replace(tmp, self._paths['state.json'])

    def _append_vecs(self, vecs: np.ndarray):
        """Le fichier vecs.fbin est indexé par chunk_id GLOBAL : le rerank
        fait pread(8 + chunk_id*dim*4). On étend donc à l'offset du
        premier chunk du bloc (les ids sont contigus)."""
        first = self.state['next_chunk_id']
        with open(self._paths['vecs.fbin'], 'r+b') as fh:
            fh.seek(8 + first * self.dim * 4)
            fh.write(vecs.astype(np.float32).tobytes())
            n_total = first + len(vecs)
            fh.seek(0)
            fh.write(struct.pack('<ii', n_total, self.dim))

    def _idx_view(self) -> np.ndarray:
        return np.memmap(self._paths['chunks.idx'], dtype=IDX_DTYPE,
                         mode='r')

    def _read_chunk_text(self, off: int, ln: int) -> str:
        with open(self._paths['docs.log'], 'rb') as fh:
            fh.seek(off)
            return fh.read(ln).decode('utf-8', errors='replace')

    # ----- API -----

    def insert_document(self, key: str, text: str | None = None,
                        chunks: list[str] | None = None,
                        metadata: dict | None = None) -> dict:
        if key in self.docs:
            raise ValueError(f'document {key!r} existe déjà '
                             '(delete_document d abord)')
        if chunks is None:
            if text is None:
                raise ValueError('text ou chunks requis')
            chunks = chunk_text(text, self.state['chunk_size'],
                                self.state['chunk_overlap'])
        if not chunks:
            raise ValueError('document vide')

        vecs = np.asarray(self.embed(chunks), dtype=np.float32)
        if vecs.shape != (len(chunks), self.dim):
            raise ValueError(f'embed_fn: forme {vecs.shape} != '
                             f'({len(chunks)}, {self.dim})')
        vecs = vecs / np.maximum(
            np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)

        c0 = self.state['next_chunk_id']
        doc_id = self.state['next_doc_id']
        n = len(chunks)

        # 1. textes + index (durables avant le HOT). chunks.idx est indexé
        #    par chunk_id GLOBAL (position = cid × 24 o, sparse au début si
        #    les ids démarrent après un corpus de fond) — même convention
        #    que vecs.fbin, pour des pread directs par id.
        recs = np.empty(n, dtype=IDX_DTYPE)
        with open(self._paths['docs.log'], 'ab') as fh:
            for i, c in enumerate(chunks):
                b = c.encode('utf-8')
                recs[i] = (fh.tell(), len(b), doc_id, i, 0)
                fh.write(b)
        with open(self._paths['chunks.idx'], 'r+b') as fh:
            fh.seek(c0 * IDX_DTYPE.itemsize)
            fh.write(recs.tobytes())

        # 2. vecteurs (rerank) — indexés par chunk_id global
        self._append_vecs(vecs)

        # 3. métadonnées typées (les chunks héritent du document)
        chunk_ids = np.arange(c0, c0 + n, dtype=np.uint32)
        if metadata:
            self.meta.add_keys(encode_meta(metadata, self.float_specs),
                               chunk_ids)

        # 4. routage médianes + HOT (en dernier : rejouable)
        leaves = mf.traverse_batch(vecs, self.sub_dim, self.depth,
                                   self.n_trees)
        mf.hot_append_block(self.hot, leaves, chunk_ids)

        # 5. doc_map + state
        rec = {'key': key, 'doc_id': doc_id, 'first_chunk': int(c0),
               'n_chunks': n, 'meta': metadata or {}}
        with open(self._paths['doc_map.jsonl'], 'a') as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
        self.docs[key] = rec
        self.state['next_chunk_id'] = int(c0 + n)
        self.state['next_doc_id'] = doc_id + 1
        self._save_state()
        return rec

    def delete_document(self, key: str) -> int:
        rec = self.docs.get(key)
        if rec is None:
            raise KeyError(key)
        for cid in range(rec['first_chunk'],
                         rec['first_chunk'] + rec['n_chunks']):
            self.forest.tombstone_add(cid)
        with open(self._paths['doc_map.jsonl'], 'a') as fh:
            fh.write(json.dumps({'deleted': True, 'key': key}) + '\n')
        del self.docs[key]
        return rec['n_chunks']

    def search(self, text: str | None = None, qvec=None,
               where: dict | None = None, top_docs: int = 5,
               window: int = 1, n_probes: int = 3, top_paths: int = 1024,
               top_n: int = 4000, query_depth: int = 0) -> list[dict]:
        """Top documents pour la requête, filtrés par `where` (typé),
        groupés par parent (rang du meilleur chunk), avec `window` chunks
        de contexte de part et d'autre du hit."""
        if qvec is None:
            if text is None:
                raise ValueError('text ou qvec requis')
            qvec = np.asarray(self.embed([text]), dtype=np.float32)[0]
        qvec = qvec / max(float(np.linalg.norm(qvec)), 1e-12)

        bmp = None
        if where:
            flat, lens = compile_where(where, self.meta.keys(),
                                       self.float_specs)
            bmp = self.meta.filter_keys(flat, lens)
        try:
            ids, votes, nn = self.forest.query_pathrank_meta(
                qvec, n_probes, top_paths, bmp, top_n=top_n,
                query_depth=query_depth)
        finally:
            if bmp:
                mf.MetaStore.filter_free(bmp)
        if nn == 0:
            return []
        top = self.forest.rerank_l2(self._paths['vecs.fbin'], qvec,
                                    ids[:nn],
                                    top_k=min(nn, max(50, top_docs * 10)))

        idx = self._idx_view()
        by_doc: dict[int, int] = {}          # doc_id → meilleur chunk (rang)
        order: list[int] = []
        for cid in (int(x) for x in top):
            if cid >= len(idx):
                continue
            parent = int(idx[cid]['parent'])
            if parent not in by_doc:
                by_doc[parent] = cid
                order.append(parent)
                if len(order) >= top_docs:
                    break

        key_by_id = {r['doc_id']: k for k, r in self.docs.items()}
        results = []
        for rank, doc_id in enumerate(order):
            key = key_by_id.get(doc_id)
            if key is None:
                continue                      # document supprimé
            rec = self.docs[key]
            hit = by_doc[doc_id]
            seq = int(idx[hit]['seq'])
            lo = max(0, seq - window)
            hi = min(rec['n_chunks'] - 1, seq + window)
            ctx = []
            for s in range(lo, hi + 1):
                e = idx[rec['first_chunk'] + s]
                ctx.append({'seq': s,
                            'text': self._read_chunk_text(int(e['off']),
                                                          int(e['len']))})
            results.append({'key': key, 'rank': rank, 'best_seq': seq,
                            'meta': rec['meta'], 'chunks': ctx})
        return results

    def close(self):
        self.meta.close()
