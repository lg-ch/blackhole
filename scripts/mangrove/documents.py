"""documents — la couche document de mangrove (RAG-ready).

Transforme le moteur vectoriel (Forest + HOT + MetaStore) en base
documentaire : insertion de documents chunkés, métadonnées typées,
recherche groupée par document parent avec fenêtres de contexte.

Layout <root>/ — TOUT est fichiers + pread/mmap, y compris le registre
des documents : la RAM ne croît ni avec les chunks NI avec les documents
(scénario 10M docs sous cgroup 1 Go : ok) :
  state.json     compteurs (next_chunk_id, next_doc_id) + config
  docs.log       textes des chunks, append-only, utf-8
  chunks.idx     24 o/chunk (par chunk_id GLOBAL) : off/len/parent/seq
  doc_recs.idx   32 o/document (par doc_id) : first_chunk, n_chunks,
                 flags (bit0=supprimé), offsets clé+méta dans docs_meta.log
  docs_meta.log  blobs append-only : clé utf-8 puis json des métadonnées
  keys.hash      table de hachage disque clé→doc_id (sondage linéaire,
                 redimensionnée par réécriture ; ~12 o/slot sur disque)
  vecs.fbin      vecteurs des chunks (rerank L2), indexés par chunk_id
  meta/          MetaStore natif (bitmaps par champ=valeur)

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
DOC_DTYPE = np.dtype([('first_chunk', '<u8'), ('n_chunks', '<u4'),
                      ('flags', '<u4'), ('blob_off', '<u8'),
                      ('key_len', '<u2'), ('pad', '<u2'),
                      ('meta_len', '<u4')])
DOC_DELETED = 1


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


def _fnv1a64(s: str) -> int:
    h = 0xcbf29ce484222325
    for b in s.encode('utf-8'):
        h = ((h ^ b) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h or 1


class _DiskHash:
    """clé → doc_id sur disque. Slots de 12 o [u64 hash][u32 doc_id+1],
    0 = vide ; slot 0 réservé au compteur. Sondage linéaire, resize ×2 par
    réécriture au-delà de 70 % de charge. RAM : néant (pread par sonde).
    Les collisions de hash sont levées par verify(doc_id) → clé réelle."""

    SLOT = struct.Struct('<QI')

    def __init__(self, path: str, verify):
        self.path = path
        self._verify = verify
        if not os.path.exists(path):
            self._init_file(4096)
        self.cap = os.path.getsize(path) // 12 - 1

    def _init_file(self, cap: int):
        with open(self.path, 'wb') as fh:
            fh.write(self.SLOT.pack(0xC0FFEE, 0))          # header : count
            fh.write(b'\x00' * (12 * cap))

    def _count(self, fh) -> int:
        fh.seek(0)
        return self.SLOT.unpack(fh.read(12))[1]

    def get(self, key: str) -> int | None:
        h = _fnv1a64(key)
        with open(self.path, 'rb') as fh:
            i = h % self.cap
            for _ in range(self.cap):
                fh.seek(12 * (1 + i))
                sh, sd = self.SLOT.unpack(fh.read(12))
                if sh == 0:
                    return None
                if sh == h and sd != 0 and self._verify(sd - 1) == key:
                    return sd - 1
                i = (i + 1) % self.cap
        return None

    def put(self, key: str, doc_id: int):
        with open(self.path, 'r+b') as fh:
            count = self._count(fh)
            if (count + 1) * 10 > self.cap * 7:
                self._resize(fh)
                return self.put(key, doc_id)
            h = _fnv1a64(key)
            i = h % self.cap
            for _ in range(self.cap):
                fh.seek(12 * (1 + i))
                sh, _sd = self.SLOT.unpack(fh.read(12))
                if sh == 0:
                    fh.seek(12 * (1 + i))
                    fh.write(self.SLOT.pack(h, doc_id + 1))
                    fh.seek(0)
                    fh.write(self.SLOT.pack(0xC0FFEE, count + 1))
                    return
                i = (i + 1) % self.cap
        raise RuntimeError('keys.hash pleine (resize défaillant)')

    def _resize(self, fh):
        fh.seek(12)
        slots = [self.SLOT.unpack(fh.read(12))
                 for _ in range(self.cap)]
        fh.close()
        new_cap = self.cap * 2
        tmp = self.path + '.tmp'
        with open(tmp, 'wb') as out:
            out.write(self.SLOT.pack(0xC0FFEE, 0))
            out.write(b'\x00' * (12 * new_cap))
        old_cap, self.cap = self.cap, new_cap
        os.replace(tmp, self.path)
        count = 0
        with open(self.path, 'r+b') as out:
            for sh, sd in slots:
                if sh == 0 or sd == 0:
                    continue
                i = sh % new_cap
                while True:
                    out.seek(12 * (1 + i))
                    eh, _ = self.SLOT.unpack(out.read(12))
                    if eh == 0:
                        out.seek(12 * (1 + i))
                        out.write(self.SLOT.pack(sh, sd))
                        count += 1
                        break
                    i = (i + 1) % new_cap
            out.seek(0)
            out.write(self.SLOT.pack(0xC0FFEE, count))
        del old_cap


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
                        'doc_recs.idx', 'docs_meta.log', 'keys.hash',
                        'vecs.fbin')}
        if os.path.exists(self._paths['state.json']):
            self.state = json.load(open(self._paths['state.json']))
        else:
            self.state = {'next_chunk_id': 0, 'next_doc_id': 0,
                          'chunk_size': chunk_size,
                          'chunk_overlap': chunk_overlap}
        self.state.setdefault('chunk_size', chunk_size)
        self.state.setdefault('chunk_overlap', chunk_overlap)
        for name, header in (('vecs.fbin', struct.pack('<ii', 0, dim)),
                             ('chunks.idx', b''), ('doc_recs.idx', b''),
                             ('docs_meta.log', b''), ('docs.log', b'')):
            if not os.path.exists(self._paths[name]):
                with open(self._paths[name], 'wb') as fh:
                    fh.write(header)
        self._keys = _DiskHash(self._paths['keys.hash'],
                               self._doc_key_of)

    # ----- registre documents (disque, zéro RAM par doc) -----

    def _doc_raw(self, doc_id: int):
        sz = DOC_DTYPE.itemsize
        with open(self._paths['doc_recs.idx'], 'rb') as fh:
            fh.seek(doc_id * sz)
            b = fh.read(sz)
        if len(b) < sz:
            return None
        return np.frombuffer(b, dtype=DOC_DTYPE)[0]

    def _doc_key_of(self, doc_id: int) -> str | None:
        r = self._doc_raw(doc_id)
        if r is None:
            return None
        with open(self._paths['docs_meta.log'], 'rb') as fh:
            fh.seek(int(r['blob_off']))
            return fh.read(int(r['key_len'])).decode('utf-8')

    def get_document(self, key: str) -> dict | None:
        """Record complet {key, doc_id, first_chunk, n_chunks, meta} ou
        None (inconnu ou supprimé). Coût : 2-3 pread, zéro RAM résidente."""
        doc_id = self._keys.get(key)
        if doc_id is None:
            return None
        r = self._doc_raw(doc_id)
        if r is None or (int(r['flags']) & DOC_DELETED):
            return None
        with open(self._paths['docs_meta.log'], 'rb') as fh:
            fh.seek(int(r['blob_off']) + int(r['key_len']))
            mb = fh.read(int(r['meta_len']))
        return {'key': key, 'doc_id': doc_id,
                'first_chunk': int(r['first_chunk']),
                'n_chunks': int(r['n_chunks']),
                'meta': json.loads(mb) if mb else {}}

    def _doc_by_id(self, doc_id: int) -> dict | None:
        r = self._doc_raw(doc_id)
        if r is None or (int(r['flags']) & DOC_DELETED):
            return None
        with open(self._paths['docs_meta.log'], 'rb') as fh:
            fh.seek(int(r['blob_off']))
            blob = fh.read(int(r['key_len']) + int(r['meta_len']))
        key = blob[:int(r['key_len'])].decode('utf-8')
        mb = blob[int(r['key_len']):]
        return {'key': key, 'doc_id': doc_id,
                'first_chunk': int(r['first_chunk']),
                'n_chunks': int(r['n_chunks']),
                'meta': json.loads(mb) if mb else {}}

    def _put_doc(self, key: str, doc_id: int, first_chunk: int,
                 n_chunks: int, metadata: dict):
        kb = key.encode('utf-8')
        mb = json.dumps(metadata or {}, ensure_ascii=False).encode('utf-8')
        with open(self._paths['docs_meta.log'], 'ab') as fh:
            blob_off = fh.tell()
            fh.write(kb + mb)
        rec = np.zeros(1, dtype=DOC_DTYPE)
        rec[0] = (first_chunk, n_chunks, 0, blob_off, len(kb), 0, len(mb))
        with open(self._paths['doc_recs.idx'], 'r+b') as fh:
            fh.seek(doc_id * DOC_DTYPE.itemsize)
            fh.write(rec.tobytes())
        self._keys.put(key, doc_id)

    # ----- internes chunks -----

    def _save_state(self):
        tmp = self._paths['state.json'] + '.tmp'
        with open(tmp, 'w') as fh:
            json.dump(self.state, fh)
        os.replace(tmp, self._paths['state.json'])

    def _append_vecs(self, vecs: np.ndarray):
        first = self.state['next_chunk_id']
        with open(self._paths['vecs.fbin'], 'r+b') as fh:
            fh.seek(8 + first * self.dim * 4)
            fh.write(vecs.astype(np.float32).tobytes())
            fh.seek(0)
            fh.write(struct.pack('<ii', first + len(vecs), self.dim))

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
        if self.get_document(key) is not None:
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

        # 1. textes + index chunks (chunks.idx par chunk_id GLOBAL)
        recs = np.empty(n, dtype=IDX_DTYPE)
        with open(self._paths['docs.log'], 'ab') as fh:
            for i, c in enumerate(chunks):
                b = c.encode('utf-8')
                recs[i] = (fh.tell(), len(b), doc_id, i, 0)
                fh.write(b)
        with open(self._paths['chunks.idx'], 'r+b') as fh:
            fh.seek(c0 * IDX_DTYPE.itemsize)
            fh.write(recs.tobytes())

        # 2. vecteurs (rerank), indexés par chunk_id global
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

        # 5. registre document (disque) + state
        self._put_doc(key, doc_id, c0, n, metadata or {})
        self.state['next_chunk_id'] = int(c0 + n)
        self.state['next_doc_id'] = doc_id + 1
        self._save_state()
        return {'key': key, 'doc_id': doc_id, 'first_chunk': int(c0),
                'n_chunks': n, 'meta': metadata or {}}

    def delete_document(self, key: str) -> int:
        rec = self.get_document(key)
        if rec is None:
            raise KeyError(key)
        for cid in range(rec['first_chunk'],
                         rec['first_chunk'] + rec['n_chunks']):
            self.forest.tombstone_add(cid)
        with open(self._paths['doc_recs.idx'], 'r+b') as fh:
            off = rec['doc_id'] * DOC_DTYPE.itemsize
            fh.seek(off + 12)                     # champ flags (u32)
            fh.write(struct.pack('<I', DOC_DELETED))
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
        by_doc: dict[int, int] = {}
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

        results = []
        for rank, doc_id in enumerate(order):
            rec = self._doc_by_id(doc_id)
            if rec is None:
                continue                          # document supprimé
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
            results.append({'key': rec['key'], 'rank': rank,
                            'best_seq': seq, 'meta': rec['meta'],
                            'chunks': ctx})
        return results

    def close(self):
        self.meta.close()
