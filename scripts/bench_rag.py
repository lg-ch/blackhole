"""bench_rag — LE bench produit : la pile VDB complète sous cgroup, à froid.

Corpus : l'index 50M "vécu" de la campagne (grandi ×50 par injection
live) habillé en base documentaire — 6,25M documents de 8 chunks,
métadonnées à l'échelle (lang 4 valeurs, year 10 valeurs + buckets).

Phase --setup (hors cgroup, une fois) :
  chunks.idx 50M×24o (parent = cid//8), registre 6,25M docs (32o/doc,
  blobs), vecs.fbin = copie du corpus (rerank + appends live),
  bitmaps méta gelés (~70M entrées de clés au total).

Phase --run (à lancer sous systemd-run -p MemoryMax=512M) :
  50 requêtes DEEP, drop_caches avant CHACUNE, 4 modes :
    A  documentaire pur (top 5 docs, fenêtre ±1)
    B  filtre lang=fr (12,5M chunks)
    C  filtre lang=fr ET year∈[2019,2021]
    D  mode A pendant une injection live concurrente (thread
       insert_document en continu)
  Métriques : doc-recall@5 (le parent du top-1 GT chunk est dans les 5
  docs), latence e2e p50/p99 (pathrank+rerank+group-by+textes), RSS max.
"""
import argparse
import json
import os
import shutil
import struct
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mangrove_ffi as mf                                   # noqa: E402
from mangrove_ffi import Forest, MetaStore, set_gen_version  # noqa: E402
from mangrove.documents import DocumentStore, IDX_DTYPE, DOC_DTYPE  # noqa: E402

ROOT = os.environ.get('RAG_ROOT', '/root/mangrove-campaign')
IDX = os.path.join(ROOT, 'run', 'idx_live')
DATA = os.path.join(ROOT, 'data')
DS = os.path.join(ROOT, 'ragstore')
N_CHUNKS = 50_000_000
PER_DOC = 8
N_DOCS = N_CHUNKS // PER_DOC
DIM, SD, NT = 96, 16, 256


def read_meta_txt():
    meta = {}
    for line in open(os.path.join(IDX, 'meta.txt')):
        p = line.split()
        if len(p) == 2:
            meta[p[0]] = int(p[1]) if p[1].lstrip('-').isdigit() else p[1]
    return meta


def setup():
    os.makedirs(DS, exist_ok=True)
    t0 = time.time()

    # texte partagé factice (le contenu n'est pas l'objet du bench)
    shared = ('Chunk de démonstration mangrove — le texte réel vivrait '
              'ici ; le bench mesure le moteur, pas la prose. ' * 3)
    sb = shared.encode()
    with open(os.path.join(DS, 'docs.log'), 'wb') as fh:
        fh.write(sb)

    # chunks.idx : 50M records, parent = cid // PER_DOC
    print('[setup] chunks.idx (50M × 24 o)...', flush=True)
    B = 1_000_000
    with open(os.path.join(DS, 'chunks.idx'), 'wb') as fh:
        for s in range(0, N_CHUNKS, B):
            n = min(B, N_CHUNKS - s)
            r = np.zeros(n, dtype=IDX_DTYPE)
            cid = np.arange(s, s + n, dtype=np.uint64)
            r['off'] = 0
            r['len'] = len(sb)
            r['parent'] = (cid // PER_DOC).astype(np.uint32)
            r['seq'] = (cid % PER_DOC).astype(np.uint32)
            fh.write(r.tobytes())

    # registre documents : blobs 'd<did>' + '{}' puis records 32 o
    print('[setup] registre 6,25M docs...', flush=True)
    with open(os.path.join(DS, 'docs_meta.log'), 'wb') as bf, \
         open(os.path.join(DS, 'doc_recs.idx'), 'wb') as rf:
        off = 0
        for s in range(0, N_DOCS, B):
            n = min(B, N_DOCS - s)
            keys = [f'd{d}'.encode() for d in range(s, s + n)]
            blob = b''.join(k + b'{}' for k in keys)
            bf.write(blob)
            r = np.zeros(n, dtype=DOC_DTYPE)
            did = np.arange(s, s + n, dtype=np.uint64)
            r['first_chunk'] = did * PER_DOC
            r['n_chunks'] = PER_DOC
            kl = np.array([len(k) for k in keys], dtype=np.uint64)
            r['blob_off'] = off + np.concatenate(
                ([0], np.cumsum(kl[:-1] + 2)))
            r['key_len'] = kl.astype(np.uint16)
            r['meta_len'] = 2
            rf.write(r.tobytes())
            off += len(blob)

    # vecs.fbin : copie du corpus (rerank par chunk_id + appends live)
    dst = os.path.join(DS, 'vecs.fbin')
    if not os.path.exists(dst):
        print('[setup] copie vecs.fbin (19 Go)...', flush=True)
        shutil.copyfile(os.path.join(DATA, 'deep50m.fbin'), dst)

    # état
    json.dump({'next_chunk_id': N_CHUNKS, 'next_doc_id': N_DOCS,
               'chunk_size': 800, 'chunk_overlap': 160},
              open(os.path.join(DS, 'state.json'), 'w'))

    # métadonnées à l'échelle : lang (cid%4), year ((cid//8)%10 + 2015)
    print('[setup] bitmaps méta (~70M entrées)...', flush=True)
    ms = MetaStore(os.path.join(DS, 'meta'))
    cid = np.arange(N_CHUNKS, dtype=np.uint32)
    for i, lang in enumerate(('fr', 'en', 'de', 'es')):
        ms.add('lang', lang, cid[cid % 4 == i])
        print(f'  lang={lang} ok', flush=True)
    yv = (cid // PER_DOC) % 10 + 2015
    for y in range(2015, 2025):
        ids = cid[yv == y]
        ms.add('year', str(y), ids)                    # clé exacte
        ms.add('year.b3', str(y // 1000), ids)         # bucket 10^3
        print(f'  year={y} ok', flush=True)
    n = ms.compact()
    print(f'[setup] compact : {n} clés gelées en '
          f'{time.time()-t0:.0f}s total', flush=True)
    ms.close()
    compute_filtered_gt()
    print('[setup] FINI', flush=True)


def compute_filtered_gt():
    """GT brute-force FILTRÉES pour les modes B et C : le top-10 global ne
    contient pas forcément le meilleur voisin fr — comparer un résultat
    filtré à un GT non filtré sous-estime mécaniquement le recall."""
    Q = np.load(os.path.join(DATA, 'queries.npy'))[:50]
    masks = {
        'gt_50m_fr.npy': lambda cid: cid % 4 == 0,
        'gt_50m_fr_1921.npy': lambda cid: (cid % 4 == 0) & (
            ((cid // PER_DOC) % 10 + 2015 >= 2019) &
            ((cid // PER_DOC) % 10 + 2015 <= 2021)),
    }
    todo = {n: m for n, m in masks.items()
            if not os.path.exists(os.path.join(DS, n))}
    if not todo:
        return
    print('[setup] GT filtrées (scan 19 Go)...', flush=True)
    t0 = time.time()
    CH = 2_000_000
    best = {n: (np.full((50, 10), np.inf),
                np.zeros((50, 10), dtype=np.int64)) for n in todo}
    with open(os.path.join(DATA, 'deep50m.fbin'), 'rb') as fh:
        fh.read(8)
        for s in range(0, N_CHUNKS, CH):
            n = min(CH, N_CHUNKS - s)
            block = np.frombuffer(fh.read(n * DIM * 4),
                                  dtype=np.float32).reshape(n, DIM)
            cid = np.arange(s, s + n, dtype=np.int64)
            for name, mfn in todo.items():
                sel = mfn(cid)
                sub = block[sel]
                ids = cid[sel]
                bn = (sub ** 2).sum(1)
                bd, bi = best[name]
                for qi in range(50):
                    d2 = bn - 2.0 * (sub @ Q[qi])
                    p = np.argpartition(d2, 10)[:10]
                    cd = np.concatenate([bd[qi], d2[p]])
                    ci = np.concatenate([bi[qi], ids[p]])
                    o = np.argsort(cd)[:10]
                    bd[qi], bi[qi] = cd[o], ci[o]
    for name, (bd, bi) in best.items():
        np.save(os.path.join(DS, name), bi)
    print(f'[setup] GT filtrées écrites en {time.time()-t0:.0f}s',
          flush=True)


def drop_caches():
    os.system('sync')
    try:
        with open('/proc/sys/vm/drop_caches', 'w') as fh:
            fh.write('3')
    except PermissionError:
        os.system("echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null")


def rss_mb():
    for line in open('/proc/self/status'):
        if line.startswith('VmRSS:'):
            return int(line.split()[1]) / 1024.0
    return 0.0


def run():
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    meta = read_meta_txt()
    depth = int(meta['depth'])
    qd = json.load(open(os.path.join(
        IDX, 'recommended_qd.json')))['recommended_qd'] \
        if os.path.exists(os.path.join(IDX, 'recommended_qd.json')) \
        else depth - 2
    set_gen_version(int(meta.get('gen_version', 3)))
    mf.set_max_leaf_bytes(200_000)
    f = Forest(IDX, n_trees=NT, dim=DIM, sub_dim=SD, depth=depth,
               n_docs=N_CHUNKS, gen_version=int(meta.get('gen_version', 3)))
    mf.clear_live_medians()
    mf.load_live_medians(IDX)
    hotdir = os.path.join(DS, 'hot')
    os.makedirs(hotdir, exist_ok=True)
    hot = mf._lib.mg_hot_init(NT, depth, hotdir.encode())
    if not hot:
        raise RuntimeError('mg_hot_init a échoué — vérifier LimitNOFILE '
                           '(1024 fichiers HOT + 256 forêt > défaut 1024)')
    mf._lib.mg_forest_set_hot_overlay(hot)

    rng = np.random.default_rng(42)

    def embed(texts):
        return rng.standard_normal((len(texts), DIM)).astype(np.float32)

    ds = DocumentStore(DS, f, hot, embed, dim=DIM, depth=depth,
                       sub_dim=SD, n_trees=NT)
    Q = np.load(os.path.join(DATA, 'queries.npy'))[:50]
    GT = np.load(os.path.join(DATA, 'gt_50m.npy'))[:50, :10]

    GT_FR = np.load(os.path.join(DS, 'gt_50m_fr.npy'))
    GT_C = np.load(os.path.join(DS, 'gt_50m_fr_1921.npy'))

    def one_pass(tag, where, gt):
        lats, hits = [], 0
        peak = rss_mb()
        for qi in range(len(Q)):
            drop_caches()
            t0 = time.time()
            r = ds.search(qvec=Q[qi], where=where, top_docs=5, window=1,
                          n_probes=3, top_paths=1024, top_n=4000,
                          query_depth=qd)
            lats.append((time.time() - t0) * 1000)
            peak = max(peak, rss_mb())
            # doc-recall@5 : le parent du top-1 de la GT (filtrée pour le
            # mode) doit être dans les 5 documents retournés
            want = int(gt[qi][0]) // PER_DOC
            if any(x['key'] == f'd{want}' for x in r):
                hits += 1
        la = np.array(lats)
        print(f'[{tag}] doc-recall@5 {hits}/{len(Q)}  '
              f'e2e p50 {np.median(la):.0f} ms  '
              f'p99 {np.percentile(la, 99):.0f} ms  RSS {peak:.0f} MB',
              flush=True)

    one_pass('A pur', None, GT)
    one_pass('B lang=fr', {'lang': 'fr'}, GT_FR)
    one_pass('C fr×2019-21', {'lang': 'fr',
                              'year': ('range', 2019, 2021)}, GT_C)

    # D : requêtes pendant injection live concurrente
    stop = threading.Event()
    n_ins = [0]

    run_tag = os.getpid()          # le registre est persistant : clés
                                   # uniques par run pour la rejouabilité
    def injector():
        while not stop.is_set():
            key = f'live-{run_tag}-{n_ins[0]}'
            ds.insert_document(key, chunks=[f'chunk {key} {i}'
                                            for i in range(PER_DOC)],
                               metadata={'lang': 'fr', 'year': 2024})
            n_ins[0] += 1

    th = threading.Thread(target=injector, daemon=True)
    th.start()
    one_pass('D pur+inject', None, GT)
    stop.set()
    th.join(timeout=30)
    print(f'[D] {n_ins[0]} documents insérés pendant la passe '
          f'({n_ins[0] * PER_DOC} chunks)', flush=True)

    mf._lib.mg_forest_set_hot_overlay(None)
    mf._lib.mg_hot_free(hot)
    ds.close()
    f.close()
    print('BENCH FINI', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--setup', action='store_true')
    args = ap.parse_args()
    if args.setup:
        setup()
    else:
        run()
