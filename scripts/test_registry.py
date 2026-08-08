"""Smoke test for MangroveCluster registry + prefix search."""
from __future__ import annotations
import os, shutil, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from registry import MangroveCluster


def main():
    ROOT = '/tmp/cluster_test'
    if os.path.exists(ROOT):
        shutil.rmtree(ROOT)

    cl = MangroveCluster(ROOT)

    print('=== 1. Create 4 indexes ===')
    for name in ['docs-202401', 'docs-202402', 'docs-202403', 'media-202401']:
        cl.create_index(name, dim=4, sub_dim=0, n_trees=20, depth=6)
        print(f'  created {name}')

    print(f'\n=== 2. list_indexes() ===')
    print(f'  all                : {cl.list_indexes()}')
    print(f'  pattern "docs-*"   : {cl.list_indexes("docs-*")}')
    print(f'  pattern "*-202401" : {cl.list_indexes("*-202401")}')
    print(f'  pattern "docs-202?02" : {cl.list_indexes("docs-2024??")}')

    print(f'\n=== 3. Insert different docs in each index ===')
    for i, name in enumerate(['docs-202401', 'docs-202402',
                              'docs-202403', 'media-202401']):
        li = cl.get(name)
        # 5 docs each, with index-specific signature
        for j in range(5):
            v = np.array([float(i * 10 + j), 1.0, 0.0, 0.0],
                         dtype=np.float32)
            li.insert(v, doc_id=j)
        li.freeze()
        print(f'  {name}: 5 docs frozen')

    print(f'\n=== 4. Prefix search "docs-*" with qvec [12, 1, 0, 0] ===')
    # qvec matches "doc 2 in index docs-202402" (idx=1, j=2 → vec=[12,1,0,0])
    r = cl.search('docs-*', np.array([12.0, 1.0, 0.0, 0.0], dtype=np.float32),
                  top_n=20, top_k=5)
    print(f'  matched : {r["matched_indexes"]}')
    print(f'  results :')
    for hit in r['results']:
        print(f'    {hit}')

    print(f'\n=== 5. Search "*" (all 4 indexes) ===')
    r2 = cl.search('*', np.array([12.0, 1.0, 0.0, 0.0], dtype=np.float32),
                   top_n=20, top_k=3)
    print(f'  matched : {r2["matched_indexes"]}')
    for hit in r2['results']:
        print(f'    {hit}')

    print(f'\n=== 6. stats() ===')
    for name, st in cl.stats().items():
        print(f'  {name:20} {st}')

    print(f'\n=== 7. drop docs-202401 ===')
    cl.drop_index('docs-202401')
    print(f'  after drop : {cl.list_indexes()}')
    assert 'docs-202401' not in cl.list_indexes()
    assert not os.path.exists(os.path.join(ROOT, 'docs-202401'))

    cl.close()
    print('\nOK')


if __name__ == '__main__':
    main()
