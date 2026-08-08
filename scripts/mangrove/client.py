"""HTTP client implementation for the mangrove-search SDK.

Flat API : everything hangs off the Client. No nested `client.indexes[...]`
shenanigans. Use `name=` for single-index ops and `pattern=` for
multi-index ops.

Uses urllib3 with connection pooling + automatic retry/backoff for
transient failures (503, 502, 504, 429 with Retry-After).
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import urllib3
from urllib3.util.retry import Retry


__all__ = ['Client', 'IndexHandle', 'MangroveError']


class MangroveError(RuntimeError):
    """HTTP error returned by the mangrove service (after retries).

    Attributes :
      code : HTTP status code. Common values :
        400 — malformed request body
        401 — missing X-API-Key
        403 — key lacks scope/perm
        404 — index not found
        409 — index name already exists
        429 — rate-limited (Retry-After header honored by SDK)
        500 — server-side exception
        503 — backpressure (retried automatically by default)
        504 — server slow → SDK read-timeout
        599 — exhausted retries / network unreachable (synthetic)
      body : parsed JSON payload, typically {'error': <kind>, ...}
    """
    def __init__(self, code: int, body: Any) -> None:
        self.code = code
        self.body = body
        super().__init__(f'HTTP {code}: {body!r}')


def _to_list(v: Any) -> list:
    """Normalize numpy / tuple / list → list of float."""
    if isinstance(v, np.ndarray):
        return v.astype(np.float32, copy=False).tolist()
    return list(v)


def _wrap_ch_state(b: bytes) -> bytes:
    """Wrap raw CRoaring portable bytes into the ClickHouse
       AggregateFunction(groupBitmap, UInt32) state envelope.
       Bytes already starting with 0x01 pass through. """
    if b and b[0] == 0x01:
        return b
    n = len(b)
    varint = bytearray()
    while n >= 0x80:
        varint.append((n & 0x7F) | 0x80)
        n >>= 7
    varint.append(n)
    return b'\x01' + bytes(varint) + b


def _default_retry(total: int = 3, backoff: float = 0.3) -> Retry:
    return Retry(
        total=total, connect=total, read=total, status=total,
        backoff_factor=backoff,
        status_forcelist=[502, 503, 504, 429],
        allowed_methods=frozenset(['GET', 'POST', 'PUT', 'DELETE', 'HEAD']),
        respect_retry_after_header=True,
        raise_on_status=False,
    )


class Client:
    """Flat mangrove client — one method per operation, attached to the client.

    Args :
      url           — base URL, e.g. 'http://localhost:8000'
      api_key       — optional, sent as X-API-Key header
      timeout       — default per-request read timeout (seconds, default 10.0)
      retries       — total retry attempts on transient failures (default 3)
      pool_size     — concurrent connections in the pool (default 10)
      metadata_sink — optional sink for metadata (typically a
                      mg.ClickHouseSink) ; enables insert(name, vec,
                      metadata=...) and search(name|pattern, where=...).

    PUBLIC API (everything attaches to the client) :

      # Index management
      client.create(name, dim, **kwargs)              — create one index
      client.drop(name|pattern=...)                   — delete one or many
      client.list(pattern=None)                       — list (optional glob)
      client.exists(name)
      client.stats(name|pattern=None)                 — exhaustive stats
      client.health()                                 — cluster health

      # Document operations
      client.insert(name, vec, doc_id=None, metadata=None)
      client.insert_batch(name, vecs, metadatas=None)
      client.delete(name|pattern=..., doc_id|doc_ids|where=...)
      client.freeze(name, timeout=600)                — advanced (auto-called)

      # Search
      client.search(qvec, name|pattern=..., **kwargs)
      # kwargs : top_k, top_n, metric, where, filter_mode, allowed_ids,
      #          allowed_bitmap, client_side, cursor_after

    Most users only need : create() once, insert() in a loop, search() to query.
    See SDK.md and QUICKSTART.md.
    """

    def __init__(self,
                 url: str,
                 api_key: str | None = None,
                 timeout: float = 10.0,
                 retries: int = 3,
                 pool_size: int = 10,
                 metadata_sink: Any | None = None,
                 ingest_rate_limit: float | None = 1000.0,
                 ingest_rate_burst: int = 1024) -> None:
        """`ingest_rate_limit` : max vec/s for insert()/insert_batch().
           Defaults to 1000 vec/s (streaming-safe steady state that HOT +
           throttled background compaction can absorb comfortably).
           Pass None to disable capping. `ingest_rate_burst` = max tokens
           accumulated at rest (allows short bursts up to this size)."""
        self.url = url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.metadata_sink = metadata_sink
        self._pool = urllib3.PoolManager(
            num_pools=pool_size,
            maxsize=pool_size,
            timeout=urllib3.Timeout(connect=timeout, read=timeout),
            retries=_default_retry(retries),
        )
        self._rate_limit = ingest_rate_limit
        self._rate_burst = float(ingest_rate_burst)
        self._rate_tokens = self._rate_burst
        self._rate_last = None  # lazy init on first call
        import threading as _th
        self._rate_lock = _th.Lock()

    def _rate_acquire(self, n: int) -> None:
        """Token-bucket : block until n tokens available, refill at rate_limit
           tokens/s up to burst cap. Thread-safe. No-op if disabled."""
        if self._rate_limit is None or n <= 0:
            return
        import time as _t
        while True:
            with self._rate_lock:
                now = _t.monotonic()
                if self._rate_last is None:
                    self._rate_last = now
                elapsed = now - self._rate_last
                self._rate_last = now
                self._rate_tokens = min(
                    self._rate_burst,
                    self._rate_tokens + elapsed * self._rate_limit,
                )
                if self._rate_tokens >= n:
                    self._rate_tokens -= n
                    return
                needed = n - self._rate_tokens
                sleep_s = needed / self._rate_limit
            _t.sleep(sleep_s)

    # ---- internal HTTP plumbing ----

    def _request(self, method: str, path: str,
                 body: dict | None = None,
                 query: dict | None = None,
                 timeout: float | None = None,
                 idempotent: bool = True) -> dict:
        if query:
            qs = '&'.join(f'{k}={v}' for k, v in query.items() if v is not None)
            path = f'{path}?{qs}'
        headers = {'Content-Type': 'application/json',
                   'Accept':       'application/json'}
        if self.api_key:
            headers['X-API-Key'] = self.api_key
        data = json.dumps(body).encode() if body is not None else None
        req_kwargs: dict[str, Any] = {'body': data, 'headers': headers}
        if timeout is not None:
            req_kwargs['timeout'] = urllib3.Timeout(connect=self.timeout,
                                                    read=timeout)
        if not idempotent:
            req_kwargs['retries'] = urllib3.Retry(total=0)
        try:
            r = self._pool.request(method, self.url + path, **req_kwargs)
        except urllib3.exceptions.MaxRetryError as e:
            raise MangroveError(599, {'error': 'max_retries_exhausted',
                                      'detail': str(e)}) from None
        except urllib3.exceptions.ReadTimeoutError as e:
            raise MangroveError(504, {'error': 'read_timeout',
                                      'detail': str(e)}) from None
        raw = r.data
        try:
            payload: Any = json.loads(raw.decode()) if raw else {}
        except Exception:
            payload = {'raw': raw.decode(errors='replace')[:500]}
        if r.status >= 400:
            raise MangroveError(r.status, payload)
        return payload

    # ---- cluster ops ----

    def health(self) -> dict:
        return self._request('GET', '/health')

    def list(self, pattern: str | None = None) -> list[str]:
        """List indexes, optionally filtered by a glob pattern (e.g. 'arxiv-*')."""
        q = {'pattern': pattern} if pattern else None
        return self._request('GET', '/indexes', query=q)['names']

    def exists(self, name: str) -> bool:
        return name in self.list()

    def stats(self, name: str | None = None,
              pattern: str | None = None) -> dict:
        """Return EXHAUSTIVE stats.

           - stats(name='foo')      : detailed stats for one index
           - stats(pattern='arxiv-*'): per-index stats + cluster aggregates
           - stats()                : cluster-wide summary across all indexes
        """
        if name and pattern:
            raise ValueError("pass either `name` or `pattern`, not both")
        if name:
            return self._request('GET', f'/indexes/{name}/stats')
        if pattern:
            return self._request('GET', '/stats', query={'pattern': pattern})
        return self._request('GET', '/stats')

    # ---- index management ----

    def create(self, name: str, *, dim: int,
               sub_dim: int | None = None,
               n_trees: int | None = None,
               depth: int | None = None,
               max_active: int | None = None,
               gen_version: int | None = None) -> None:
        """Create a new index. Only `name` and `dim` are required ; the
           rest are tuned by the server to sensible defaults (sub_dim=16,
           n_trees=1000, depth=14, max_active=100k). See SDK.md §Defaults."""
        body: dict[str, Any] = {'name': name, 'dim': int(dim)}
        if sub_dim     is not None: body['sub_dim']     = int(sub_dim)
        if n_trees     is not None: body['n_trees']     = int(n_trees)
        if depth       is not None: body['depth']       = int(depth)
        if max_active  is not None: body['max_active']  = int(max_active)
        if gen_version is not None: body['gen_version'] = int(gen_version)
        self._request('POST', '/indexes', body=body)

    def drop(self, name: str | None = None, *,
             pattern: str | None = None) -> list[str]:
        """Drop one index by `name`, or all indexes matching a glob `pattern`.

           Returns the list of names actually dropped (empty if none matched).

           >>> client.drop('arxiv-2025')
           ['arxiv-2025']
           >>> client.drop(pattern='tmp-*')
           ['tmp-foo', 'tmp-bar']
        """
        if (name is None) == (pattern is None):
            raise ValueError("pass exactly one of `name` or `pattern`")
        if name is not None:
            self._request('DELETE', f'/indexes/{name}')
            return [name]
        targets = self.list(pattern=pattern)
        for n in targets:
            self._request('DELETE', f'/indexes/{n}')
        return targets

    def freeze(self, name: str, timeout: float = 600.0) -> str | None:
        """Force-build a segment from the in-memory active buffer.
           Synchronous, can take seconds-to-minutes. Returns segment name
           or None if active was empty. Not retried (non-idempotent)."""
        resp = self._request('POST', f'/indexes/{name}/freeze', body={},
                             timeout=timeout, idempotent=False)
        return resp.get('segment')

    # ---- document ops ----

    def insert(self, name: str, vec: Any,
               doc_id: int | None = None,
               metadata: dict[str, Any] | None = None) -> int:
        """Insert one vector. If `metadata` is given AND a metadata_sink
           is configured, push (doc_id, ts, ...metadata...) to the sink too.
           Blocks per client `ingest_rate_limit` token bucket."""
        self._rate_acquire(1)
        body: dict[str, Any] = {'vec': _to_list(vec)}
        if doc_id is not None:
            body['doc_id'] = int(doc_id)
        result = self._request('POST', f'/indexes/{name}/insert', body=body)
        doc_id = int(result['doc_id'])
        if metadata is not None and self.metadata_sink is not None:
            self.metadata_sink.insert(doc_id, metadata)
        return doc_id

    def insert_batch(self, name: str, vecs: Any,
                     metadatas: list[dict[str, Any]] | None = None,
                     timeout: float | None = None,
                     auto_freeze_on_full: bool = True) -> list[int]:
        """Insert N vectors in one HTTP call.

           Timeout scales with batch size (~10 ms per vec + 30s floor).
           Override via `timeout=` if you have a slow link.

           On 503 backpressure (active buffer full), auto-calls freeze()
           and retries once. Set `auto_freeze_on_full=False` to bubble
           the error to the caller. The auto behavior is what most
           streaming-ingest workloads want."""
        vecs_list = [_to_list(v) for v in vecs]
        if metadatas is not None and len(metadatas) != len(vecs_list):
            raise ValueError(
                f'len(metadatas)={len(metadatas)} != len(vecs)={len(vecs_list)}')
        self._rate_acquire(len(vecs_list))
        if timeout is None:
            timeout = max(30.0, 0.01 * len(vecs_list))

        def _try() -> list[int]:
            return self._request('POST', f'/indexes/{name}/insert_batch',
                                 body={'vecs': vecs_list},
                                 timeout=timeout)['doc_ids']
        try:
            ids = _try()
        except MangroveError as e:
            if (auto_freeze_on_full and e.code == 503
                and isinstance(e.body, dict)
                and e.body.get('error') == 'backpressure'):
                self.freeze(name)
                ids = _try()
            else:
                raise
        if metadatas is not None and self.metadata_sink is not None:
            self.metadata_sink.insert_batch(list(zip(ids, metadatas)))
        return ids

    def delete(self, name: str | None = None,
               doc_id: int | None = None, *,
               pattern: str | None = None,
               doc_ids: list[int] | None = None,
               where: str | None = None) -> dict[str, int] | None:
        """Tombstone documents in one or many indexes.

           Modes (one of) :
             delete('docs', 42)                       — single id (legacy form,
                                                        returns None)
             delete('docs', doc_ids=[1,2,3])          — bulk
             delete('docs', where="lang='fr'")        — metadata-driven
             delete(pattern='arxiv-*', doc_id=42)     — same id across many
             delete(pattern='arxiv-*', doc_ids=[...]) — bulk × many
             delete(pattern='arxiv-*', where='...')   — metadata × many

           Returns :
             None         — legacy single-id form
             {name: cnt}  — every other form (cnt = deletes attempted per index)

           For `where=`, the sink is queried once for matching doc_ids, then
           each id is tombstoned in every targeted index. The sink is the
           source of truth for what's a member ; mangrove tombstones are
           idempotent (deleting a non-existent doc_id is a no-op).
        """
        if (name is None) == (pattern is None):
            raise ValueError("pass exactly one of `name` or `pattern`")
        modes = sum(x is not None for x in (doc_id, doc_ids, where))
        if modes != 1:
            raise ValueError("pass exactly one of `doc_id=`, `doc_ids=`, `where=`")

        # Resolve target ids
        if where is not None:
            if self.metadata_sink is None:
                raise RuntimeError("where='...' requires Client(metadata_sink=...)")
            ids = list(self.metadata_sink.matching_ids(where))
        elif doc_ids is not None:
            ids = [int(i) for i in doc_ids]
        else:
            ids = [int(doc_id)]

        # Resolve target indexes
        targets = [name] if name is not None else self.list(pattern=pattern)

        # Legacy compat : single id + name + no kwargs → return None
        legacy_form = (name is not None and doc_id is not None
                       and doc_ids is None and where is None)

        counts: dict[str, int] = {}
        for n in targets:
            c = 0
            for i in ids:
                try:
                    self._request('POST', f'/indexes/{n}/delete',
                                  body={'doc_id': i})
                    c += 1
                except MangroveError as e:
                    if e.code != 404:
                        raise
            counts[n] = c

        if where is not None and self.metadata_sink is not None:
            # also clean the sink so future filters don't return dead ids
            try:
                self.metadata_sink.delete_ids(ids)
            except (AttributeError, NotImplementedError):
                pass

        return None if legacy_form else counts

    # ---- search (everything attaches here) ----

    def search(self, qvec: Any, *,
               name: str | None = None,
               pattern: str | None = None,
               top_k: int = 5,
               top_n: int | None = None,
               query_depth: int | None = None,
               n_probes: int | None = None,
               max_leaf_bytes: int | None = None,
               metric: str = 'l2',
               where: str | None = None,
               filter_mode: str = 'pre',
               allowed_ids: Any | None = None,
               allowed_bitmap: bytes | None = None,
               client_side: bool = False,
               cursor_after: int | None = None) -> dict:
        """Search one index or a glob pattern of indexes.

           Args (required) :
             qvec    — the query vector (list/numpy/iterable of floats)
             name OR pattern — exactly one.

           Args (filtering, optional) :
             where        — SQL WHERE clause for the configured metadata_sink.
             allowed_bitmap — raw CRoaring bytes (typically from ClickHouse).
             allowed_ids  — list/np.ndarray of int doc_ids.
             filter_mode  — 'pre' (default) | 'post'. Pre-filter pushes the
                            bitmap into the K-way merge ; fast for selective
                            clauses (<~3% match rate). Post-filter runs the
                            ANN search unfiltered, over-fetches, then drops
                            non-matching ids ; better for non-selective
                            clauses. See SDK.md §Pre- vs post-filter.

           Args (tuning, optional) :
             top_k          — number of results (default 5)
             top_n          — K-way merge cap. Default None → server picks.
             n_probes       — multi-probe routing (0 = single-probe legacy).
                              Recommended 5 for high-recall workloads.
             max_leaf_bytes — skip oversized leaves at query time. Bounds
                              p99 latency. Recommended 20000 for SLA.
             query_depth    — runtime tree-walk depth override.
             metric         — 'l2' (default) | 'cosine' | 'ip'
             cursor_after   — cursor pagination
             client_side    — privacy mode, single-index only.
        """
        if (name is None) == (pattern is None):
            raise ValueError("pass exactly one of `name` or `pattern`")
        if filter_mode not in ('pre', 'post'):
            raise ValueError(f"filter_mode must be 'pre' or 'post', got {filter_mode!r}")

        # In 'pre' mode : resolve where → bitmap and push into the merge.
        # In 'post' mode : skip the bitmap; we'll over-fetch and filter
        # the returned doc_ids against the WHERE clause after the search.
        if where is not None and allowed_bitmap is None and filter_mode == 'pre':
            if self.metadata_sink is None:
                raise RuntimeError("where='...' requires Client(metadata_sink=...)")
            allowed_bitmap = self.metadata_sink.filter_bitmap(where)

        if filter_mode == 'post' and where is not None and self.metadata_sink is None:
            raise RuntimeError("where='...' requires Client(metadata_sink=...)")

        if client_side and pattern is not None:
            raise ValueError("client_side=True only supported with name=, not pattern=")

        if filter_mode == 'post' and (where is not None
                                      or allowed_bitmap is not None
                                      or allowed_ids is not None):
            return self._search_post_filter(
                qvec, name=name, pattern=pattern,
                top_k=top_k, top_n=top_n, metric=metric,
                where=where, allowed_bitmap=allowed_bitmap,
                allowed_ids=allowed_ids,
                client_side=client_side, cursor_after=cursor_after,
                query_depth=query_depth, n_probes=n_probes,
                max_leaf_bytes=max_leaf_bytes)

        if name:
            return self._search_one(name, qvec, top_k, top_n, metric,
                                    allowed_bitmap, allowed_ids,
                                    client_side, cursor_after,
                                    query_depth, n_probes, max_leaf_bytes)
        return self._search_pattern(pattern, qvec, top_k, top_n,
                                    allowed_bitmap, query_depth,
                                    n_probes, max_leaf_bytes)

    def _estimate_density(self, name: str | None,
                          where: str | None,
                          allowed_bitmap: bytes | None,
                          allowed_ids: Any | None) -> float:
        """Estimate filter selectivity in [0, 1]. Used to size the post-filter
           over-fetch. Returns 1.0 (no over-fetch needed) when we can't
           estimate (pattern mode, sink not available, etc.)."""
        if name is None:
            return 1.0          # pattern mode : skip estimation
        try:
            n_docs = int(self.stats(name=name).get('total_docs') or 0)
        except Exception:
            n_docs = 0
        if n_docs <= 0:
            return 1.0
        if where is not None and self.metadata_sink is not None:
            try:
                n_match = self.metadata_sink.count_matching(where)
                return min(1.0, n_match / n_docs)
            except (AttributeError, Exception):
                return 1.0
        if allowed_bitmap is not None:
            try:
                from pyroaring import BitMap       # type: ignore
                bm = BitMap.deserialize(bytes(allowed_bitmap))
                return min(1.0, len(bm) / n_docs)
            except Exception:
                return 1.0
        if allowed_ids is not None:
            return min(1.0, len(list(allowed_ids)) / n_docs)
        return 1.0

    @staticmethod
    def _post_oversample(top_k: int, density: float) -> int:
        """How many candidates to over-fetch so that ~top_k survive the
           post-filter at the estimated density. 2× safety, capped at 500×."""
        if density >= 0.5:
            return max(2 * top_k, 20)
        import math
        target = math.ceil(top_k / max(density, 1e-6))
        return min(500 * top_k, max(10 * top_k, 2 * target))

    def _search_post_filter(self, qvec, *, name, pattern, top_k, top_n,
                            metric, where, allowed_bitmap, allowed_ids,
                            client_side, cursor_after, query_depth=None,
                            n_probes=None, max_leaf_bytes=None):
        """Post-filter strategy : ANN search without filter, over-fetch
           candidates adaptively from estimated density, then drop
           non-matching ids."""
        density = self._estimate_density(name, where, allowed_bitmap, allowed_ids)
        wide_top_k = self._post_oversample(top_k, density)
        wide_top_n = top_n if top_n is not None else max(2000, wide_top_k * 4)
        if name:
            raw = self._search_one(name, qvec, wide_top_k, wide_top_n, metric,
                                   None, None, client_side, cursor_after,
                                   query_depth, n_probes, max_leaf_bytes)
            ids = raw.get('ids', [])
        else:
            raw = self._search_pattern(pattern, qvec, wide_top_k, wide_top_n,
                                       None, query_depth, n_probes,
                                       max_leaf_bytes)
            ids = [r['doc_id'] for r in raw.get('results', [])]
        if not ids:
            return raw
        if where is not None:
            keep = set(self.metadata_sink.filter_ids(ids, where))
        elif allowed_bitmap is not None:
            from pyroaring import BitMap          # type: ignore
            bm = BitMap.deserialize(bytes(allowed_bitmap))
            keep = {i for i in ids if i in bm}
        else:
            keep = set(int(i) for i in allowed_ids)
        if name:
            kept = [i for i in ids if i in keep][:top_k]
            raw['ids'] = kept
        else:
            kept = [r for r in raw['results'] if r['doc_id'] in keep][:top_k]
            raw['results'] = kept
        return raw

    def _search_one(self, name, qvec, top_k, top_n, metric,
                    allowed_bitmap, allowed_ids, client_side, cursor_after,
                    query_depth=None, n_probes=None, max_leaf_bytes=None):
        if client_side:
            from .traversal import compute_leaves
            st = self.stats(name=name)
            segments = st.get('segments', [])
            if not segments:
                raise MangroveError(412, {'error': 'index empty (no segments)'})
            depth   = max(s['depth'] for s in segments)
            dim     = st.get('dim')
            sub_dim = st.get('sub_dim', 16)
            n_trees = st.get('n_trees', 1000)
            gen     = st.get('gen_version', 3)
            q       = np.asarray(qvec, dtype=np.float32)
            leaves  = compute_leaves(q, n_trees=n_trees, depth=depth,
                                     sub_dim=sub_dim, dim=dim,
                                     gen_version=gen)
            return self._request('POST', f'/indexes/{name}/search_by_leaves',
                                 body={'leaves': leaves,
                                       'top_n':  int(top_n or 4000),
                                       'top_k':  int(top_k)})

        body: dict[str, Any] = {
            'qvec':   _to_list(qvec),
            'top_k':  int(top_k),
            'metric': metric,
        }
        if top_n is not None:
            body['top_n'] = int(top_n)
        if query_depth is not None:
            body['query_depth'] = int(query_depth)
        if n_probes is not None:
            body['n_probes'] = int(n_probes)
        if max_leaf_bytes is not None:
            body['max_leaf_bytes'] = int(max_leaf_bytes)
        if cursor_after is not None:
            body['cursor_after'] = int(cursor_after)
        if allowed_bitmap is not None:
            import base64 as _b64
            wrapped = _wrap_ch_state(bytes(allowed_bitmap))
            body['allowed_bitmap_b64'] = _b64.b64encode(wrapped).decode()
        elif allowed_ids is not None:
            body['allowed_ids'] = (list(allowed_ids)
                                   if not isinstance(allowed_ids, list)
                                   else allowed_ids)
        return self._request('POST', f'/indexes/{name}/search', body=body)

    def _search_pattern(self, pattern, qvec, top_k, top_n, allowed_bitmap,
                        query_depth=None, n_probes=None, max_leaf_bytes=None):
        body: dict[str, Any] = {
            'pattern': pattern,
            'qvec':    _to_list(qvec),
            'top_k':   int(top_k),
        }
        if top_n is not None:
            body['top_n'] = int(top_n)
        if query_depth is not None:
            body['query_depth'] = int(query_depth)
        if n_probes is not None:
            body['n_probes'] = int(n_probes)
        if max_leaf_bytes is not None:
            body['max_leaf_bytes'] = int(max_leaf_bytes)
        if allowed_bitmap is not None:
            import base64 as _b64
            wrapped = _wrap_ch_state(bytes(allowed_bitmap))
            body['allowed_bitmap_b64'] = _b64.b64encode(wrapped).decode()
        return self._request('POST', '/search', body=body)

    # ---- escape hatch for advanced users ----

    def index(self, name: str) -> 'IndexHandle':
        """Return an IndexHandle for advanced/legacy use. Most users
           should call client.<op>(name=...) directly."""
        return IndexHandle(self, name)

    def close(self) -> None:
        self._pool.clear()


# ============================================================================
# Legacy / advanced — IndexHandle for callers who want a chained interface.
# ============================================================================

class IndexHandle:
    """Per-index handle. PREFER `client.<op>(name=...)` over this for new code.

    Kept for backward compatibility and for callers who pass a single index
    handle around between functions."""

    def __init__(self, client: Client, name: str) -> None:
        self._c = client
        self.name = name

    def insert(self, vec, doc_id=None, metadata=None):
        return self._c.insert(self.name, vec, doc_id=doc_id, metadata=metadata)

    def insert_batch(self, vecs, metadatas=None):
        return self._c.insert_batch(self.name, vecs, metadatas=metadatas)

    def search(self, qvec, **kw):
        return self._c.search(qvec, name=self.name, **kw)

    def delete(self, doc_id=None, *, doc_ids=None, where=None):
        return self._c.delete(self.name, doc_id,
                              doc_ids=doc_ids, where=where)

    def freeze(self, timeout=600.0):
        return self._c.freeze(self.name, timeout=timeout)

    def stats(self):
        return self._c.stats(name=self.name)

    def health(self):
        return self._c._request('GET', f'/indexes/{self.name}/health')
