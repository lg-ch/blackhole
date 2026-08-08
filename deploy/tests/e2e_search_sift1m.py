"""Deployment e2e — Phase: SDK search + recall against the deployed index.

Runs SIFT queries through the SDK against the kube pod, computes recall@10 vs
groundtruth (doc_id == base row index, guaranteed by the contiguous ingest),
and reports latency percentiles.

Run after e2e_index_sift1m.py and with the port-forward up.
Usage:  python3 deploy/tests/e2e_search_sift1m.py [n_queries] [top_k]
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import mangrove as mg
from _sift import read_fvecs, read_ivecs

URL     = os.environ.get('MG_URL', 'http://localhost:8000')
QUERY   = os.environ.get('SIFT_QUERY', 'sift/sift_query.fvecs')
GT      = os.environ.get('SIFT_GT', 'sift/sift_groundtruth.ivecs')
NAME    = os.environ.get('MG_INDEX', 'sift1m')
NQ      = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
TOP_K   = int(sys.argv[2]) if len(sys.argv) > 2 else 10


def main():
    queries = read_fvecs(QUERY)
    gt      = read_ivecs(GT)
    nq = min(NQ, len(queries))
    print(f'[search] {nq} queries, top_k={TOP_K}, index={NAME!r}', flush=True)

    client = mg.Client(URL, timeout=30.0)
    st = client.stats(name=NAME)
    print(f'[search] index total_docs={st.get("total_docs")}', flush=True)

    hits = 0
    total = 0
    lat = []
    t0 = time.time()
    for i in range(nq):
        r = client.search(queries[i], name=NAME, top_k=TOP_K)
        ids = r['ids']
        lat.append(r.get('latency_ms', 0.0))
        truth = set(int(x) for x in gt[i][:TOP_K])
        hits += len(truth & set(int(x) for x in ids))
        total += TOP_K
    wall = time.time() - t0

    lat = np.array(lat)
    recall = hits / total
    print('\n=== SIFT 1M SEARCH RESULT ===', flush=True)
    print(f'  queries       : {nq}', flush=True)
    print(f'  recall@{TOP_K:<2}     : {recall:.4f}  ({hits}/{total})', flush=True)
    print(f'  server lat ms : p50={np.percentile(lat,50):.2f}  '
          f'p95={np.percentile(lat,95):.2f}  p99={np.percentile(lat,99):.2f}', flush=True)
    print(f'  e2e QPS       : {nq/wall:.1f}  (incl. client+network over port-forward)', flush=True)
    assert recall > 0.90, f'recall too low: {recall:.4f}'
    print('  STATUS        : OK', flush=True)


if __name__ == '__main__':
    main()
