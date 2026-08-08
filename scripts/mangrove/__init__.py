"""mangrove — Python SDK for the mangrove-search service.

INSTALL :
    pip install mangrove-search          # PyPI distribution name

IMPORT :
    import mangrove as mg                # importable name = 'mangrove'

(The PyPI distribution name and the import name differ — same pattern as
'scikit-learn' / 'sklearn' or 'PyYAML' / 'yaml'.)

USAGE :
    import mangrove as mg
    client = mg.Client('http://localhost:8000', api_key='optional-secret')

    # Manage indexes
    client.indexes.create('arxiv-2026', dim=768, sub_dim=16,
                          n_trees=1000, depth=20)
    client.indexes.list()                  # all
    client.indexes.list('arxiv-*')         # glob
    client.indexes.drop('arxiv-2026')

    # Per-index operations
    idx = client.indexes['arxiv-2026']
    doc_id = idx.insert(vec)
    idx.insert_batch(vecs)                          # bulk
    idx.search(qvec, top_k=10)                      # standard
    idx.search(qvec, top_k=10, allowed_ids=[...])   # filter pre-filter
    idx.search(qvec, top_k=10, client_side=True)    # privacy mode
    idx.delete(doc_id=42)
    seg_name = idx.freeze(timeout=600)              # blocking, can be slow

    # Cluster-wide search across an index-name glob
    client.search(pattern='arxiv-*', qvec=qvec, top_k=10)

    # Health & stats
    client.health()
    client.stats()                                  # cluster summary
    client.stats('arxiv-2026')                      # per-index detail

ERROR HANDLING :
    try:
        idx.insert(vec)
    except mg.MangroveError as e:
        if e.code == 503:        # backpressure
            ...
        elif e.code == 429:      # rate limited
            ...
"""
from .client       import Client, IndexHandle, MangroveError
from .sinks        import ClickHouseSink, MetadataSink
from .batcher      import BatchedInserter, AsyncBatchedInserter
try:
    from .async_client import AsyncClient
except ImportError:        # httpx not installed → AsyncClient unavailable
    AsyncClient = None     # type: ignore

__all__ = ['Client', 'AsyncClient', 'IndexHandle', 'MangroveError',
           'ClickHouseSink', 'MetadataSink',
           'BatchedInserter', 'AsyncBatchedInserter']
__version__ = '0.2.0'
