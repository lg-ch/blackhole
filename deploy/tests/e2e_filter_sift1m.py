"""Deployment e2e — Phase: metadata filtering via ClickHouseSink.

Validates the full filter path on the deployed stack:
  SDK where= -> ClickHouseSink builds a doc_id bitmap from the deployed
  ClickHouse -> shipped to the mangrove pod as allowed_bitmap -> applied
  inside the K-way merge (pre) or as a post-filter over-fetch (post).

Seeds docs_metadata with a synthetic categorical label
  category = 'c{doc_id % 10}'  (~10% density per class)
then asserts every returned id satisfies the predicate, in BOTH filter modes,
and checks filtered ranking quality against a brute-force baseline on a sample.

Run after e2e_index_sift1m.py, with both port-forwards up (8000 + 8123).
Usage:  python3 deploy/tests/e2e_filter_sift1m.py [n_queries]
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import mangrove as mg
from _sift import read_fvecs

URL      = os.environ.get('MG_URL', 'http://localhost:8000')
CH_URL   = os.environ.get('CH_URL', 'http://localhost:8123')
BASE     = os.environ.get('SIFT_BASE', 'sift/sift_base.fvecs')
QUERY    = os.environ.get('SIFT_QUERY', 'sift/sift_query.fvecs')
NAME     = os.environ.get('MG_INDEX', 'sift1m')
NQ       = int(sys.argv[1]) if len(sys.argv) > 1 else 200
TARGET   = os.environ.get('MG_FILTER_CAT', 'c3')   # keep ids where doc_id % 10 == 3
TOP_K    = 10
NCLASS   = 10


def seed_metadata(sink, n_docs):
    import clickhouse_connect
    ch = clickhouse_connect.get_client(
        host='localhost', port=int(CH_URL.split(':')[-1]), database='default')
    have = int(ch.query('SELECT count() FROM docs_metadata').result_rows[0][0])
    if have >= n_docs:
        print(f'[filter] metadata already seeded ({have} rows)', flush=True)
        return
    if have:
        ch.command('TRUNCATE TABLE docs_metadata')
    print(f'[filter] seeding {n_docs} metadata rows (category=c0..c{NCLASS-1}) ...', flush=True)
    import datetime as _dt
    TS = _dt.datetime(2026, 1, 1)
    CH = 200_000
    t0 = time.time()
    for off in range(0, n_docs, CH):
        ids = range(off, min(off + CH, n_docs))
        rows = [[int(i), TS, f'c{int(i) % NCLASS}', '', '']
                for i in ids]
        ch.insert('docs_metadata', rows,
                  column_names=['doc_id', 'ts', 'category', 'region', 'lang'])
    print(f'[filter] seeded in {time.time()-t0:.1f}s', flush=True)
    ch.close()


def main():
    sink = mg.ClickHouseSink(
        url=CH_URL, table='docs_metadata',
        schema={'category': 'LowCardinality(String)',
                'region': 'LowCardinality(String)',
                'lang': 'LowCardinality(String)'})
    client = mg.Client(URL, timeout=30.0, metadata_sink=sink)

    n_docs = int(client.stats(name=NAME).get('total_docs'))
    seed_metadata(sink, n_docs)

    where = f"category='{TARGET}'"
    keep = lambda d: d % NCLASS == int(TARGET[1:])
    density = sink.count_matching(where) / n_docs
    print(f'[filter] where={where!r}  density={density:.3f}', flush=True)

    queries = read_fvecs(QUERY)
    nq = min(NQ, len(queries))

    results = {}
    for mode in ('pre', 'post'):
        viol = 0
        short = 0
        lat = []
        t0 = time.time()
        for i in range(nq):
            r = client.search(queries[i], name=NAME, top_k=TOP_K,
                              where=where, filter_mode=mode)
            ids = r['ids']
            if len(ids) < TOP_K:
                short += 1
            for d in ids:
                if not keep(int(d)):
                    viol += 1
        wall = time.time() - t0
        results[mode] = (viol, short)
        print(f'[filter] mode={mode:4}  predicate_violations={viol}  '
              f'under_filled_results={short}/{nq}  '
              f'wall={wall:.1f}s ({nq/wall:.1f} q/s)', flush=True)

    # Ranking-quality spot check vs brute force on the filtered subset.
    print('[filter] loading base vectors for brute-force baseline ...', flush=True)
    base = read_fvecs(BASE, limit=n_docs)
    mask = (np.arange(n_docs) % NCLASS) == int(TARGET[1:])
    sub_idx = np.nonzero(mask)[0]
    sub = base[sub_idx]
    SAMPLE = min(50, nq)
    fr_hits = 0
    for i in range(SAMPLE):
        q = queries[i]
        d2 = np.sum((sub - q) ** 2, axis=1)
        bf = set(int(sub_idx[j]) for j in np.argsort(d2)[:TOP_K])
        ids = set(int(x) for x in client.search(
            q, name=NAME, top_k=TOP_K, where=where, filter_mode='pre')['ids'])
        fr_hits += len(bf & ids)
    filt_recall = fr_hits / (SAMPLE * TOP_K)

    print('\n=== SIFT 1M FILTERING RESULT ===', flush=True)
    print(f'  where          : {where}  (density {density:.3f})', flush=True)
    for mode in ('pre', 'post'):
        viol, short = results[mode]
        print(f'  {mode:4} mode      : violations={viol}  under_filled={short}', flush=True)
    print(f'  filtered recall: {filt_recall:.4f}  (vs brute force, {SAMPLE} q)', flush=True)
    assert results['pre'][0] == 0 and results['post'][0] == 0, 'predicate violated'
    assert filt_recall > 0.90, f'filtered recall too low: {filt_recall:.4f}'
    print('  STATUS         : OK', flush=True)


if __name__ == '__main__':
    main()
