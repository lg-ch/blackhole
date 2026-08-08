"""Async mangrove client based on httpx (soft dep).

Mirror of mangrove.Client but with async/await semantics. Useful when
you ingest at high throughput and want to issue many insert_batch
requests in parallel from one asyncio task.

Soft-imports httpx — if missing, raises a clear message.

Usage :
    import asyncio, mangrove as mg

    async def main():
        async with mg.AsyncClient('http://localhost:8000') as c:
            await c.create('docs', dim=128)
            await c.insert('docs', [1.0, 2.0, ...])
            # Parallel insert_batch :
            await asyncio.gather(*[
                c.insert_batch('docs', batch)
                for batch in iter_batches()
            ])
            r = await c.search([1.0, 2.0, ...], name='docs', top_k=10)

    asyncio.run(main())
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from .client import _to_list, _wrap_ch_state, MangroveError

__all__ = ['AsyncClient']


class AsyncClient:
    """Async mirror of mg.Client. Same public surface, awaited methods."""

    def __init__(self,
                 url: str,
                 api_key: str | None = None,
                 timeout: float = 10.0,
                 metadata_sink: Any | None = None) -> None:
        try:
            import httpx
        except ImportError as e:
            raise RuntimeError(
                'mangrove.AsyncClient requires the `httpx` package. '
                'Install : pip install httpx') from e
        self.url = url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.metadata_sink = metadata_sink
        headers = {'Content-Type': 'application/json',
                   'Accept':       'application/json'}
        if api_key:
            headers['X-API-Key'] = api_key
        self._client = httpx.AsyncClient(
            base_url=self.url,
            headers=headers,
            timeout=timeout,
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
        )

    async def __aenter__(self) -> 'AsyncClient':
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    async def _request(self, method: str, path: str,
                       body: dict | None = None,
                       query: dict | None = None,
                       timeout: float | None = None) -> dict:
        if query:
            qs = '&'.join(f'{k}={v}' for k, v in query.items() if v is not None)
            path = f'{path}?{qs}'
        data = json.dumps(body).encode() if body is not None else None
        r = await self._client.request(method, path, content=data,
                                       timeout=timeout or self.timeout)
        try:
            payload: Any = r.json() if r.content else {}
        except Exception:
            payload = {'raw': r.text[:500]}
        if r.status_code >= 400:
            raise MangroveError(r.status_code, payload)
        return payload

    # ---- API mirror ----

    async def health(self) -> dict:
        return await self._request('GET', '/health')

    async def list(self, pattern: str | None = None) -> list[str]:
        q = {'pattern': pattern} if pattern else None
        return (await self._request('GET', '/indexes', query=q))['names']

    async def stats(self, name: str | None = None,
                    pattern: str | None = None) -> dict:
        if name and pattern:
            raise ValueError("pass either `name` or `pattern`")
        if name:
            return await self._request('GET', f'/indexes/{name}/stats')
        q = {'pattern': pattern} if pattern else None
        return await self._request('GET', '/stats', query=q)

    async def create(self, name: str, *, dim: int, **kw) -> None:
        body: dict[str, Any] = {'name': name, 'dim': int(dim)}
        for k in ('sub_dim', 'n_trees', 'depth', 'max_active', 'gen_version'):
            if k in kw and kw[k] is not None:
                body[k] = int(kw[k])
        await self._request('POST', '/indexes', body=body)

    async def drop(self, name: str) -> None:
        await self._request('DELETE', f'/indexes/{name}')

    async def freeze(self, name: str, timeout: float = 600.0) -> str | None:
        resp = await self._request('POST', f'/indexes/{name}/freeze',
                                   body={}, timeout=timeout)
        return resp.get('segment')

    async def insert(self, name: str, vec: Any,
                     doc_id: int | None = None,
                     metadata: dict[str, Any] | None = None) -> int:
        body: dict[str, Any] = {'vec': _to_list(vec)}
        if doc_id is not None:
            body['doc_id'] = int(doc_id)
        result = await self._request('POST', f'/indexes/{name}/insert', body=body)
        doc_id = int(result['doc_id'])
        if metadata is not None and self.metadata_sink is not None:
            self.metadata_sink.insert(doc_id, metadata)
        return doc_id

    async def insert_batch(self, name: str, vecs: Any,
                           metadatas: list[dict[str, Any]] | None = None,
                           timeout: float | None = None,
                           auto_freeze_on_full: bool = True) -> list[int]:
        vecs_list = [_to_list(v) for v in vecs]
        if metadatas is not None and len(metadatas) != len(vecs_list):
            raise ValueError(
                f'len(metadatas)={len(metadatas)} != len(vecs)={len(vecs_list)}')
        if timeout is None:
            timeout = max(30.0, 0.01 * len(vecs_list))
        async def _try() -> list[int]:
            r = await self._request('POST', f'/indexes/{name}/insert_batch',
                                    body={'vecs': vecs_list}, timeout=timeout)
            return r['doc_ids']
        try:
            ids = await _try()
        except MangroveError as e:
            if (auto_freeze_on_full and e.code == 503
                and isinstance(e.body, dict)
                and e.body.get('error') == 'backpressure'):
                await self.freeze(name)
                ids = await _try()
            else:
                raise
        if metadatas is not None and self.metadata_sink is not None:
            self.metadata_sink.insert_batch(list(zip(ids, metadatas)))
        return ids

    async def delete(self, name: str, doc_id: int) -> None:
        await self._request('POST', f'/indexes/{name}/delete',
                            body={'doc_id': int(doc_id)})

    async def search(self, qvec: Any, *,
                     name: str | None = None,
                     pattern: str | None = None,
                     top_k: int = 5,
                     top_n: int | None = None,
                     metric: str = 'l2',
                     where: str | None = None,
                     allowed_bitmap: bytes | None = None,
                     allowed_ids: Any | None = None,
                     cursor_after: int | None = None) -> dict:
        if (name is None) == (pattern is None):
            raise ValueError("pass exactly one of `name` or `pattern`")
        if where is not None and allowed_bitmap is None:
            if self.metadata_sink is None:
                raise RuntimeError("where='...' requires metadata_sink=")
            allowed_bitmap = self.metadata_sink.filter_bitmap(where)
        body: dict[str, Any] = {'qvec': _to_list(qvec),
                                'top_k': int(top_k), 'metric': metric}
        if top_n is not None: body['top_n'] = int(top_n)
        if cursor_after is not None: body['cursor_after'] = int(cursor_after)
        if allowed_bitmap is not None:
            import base64 as _b64
            body['allowed_bitmap_b64'] = _b64.b64encode(
                _wrap_ch_state(bytes(allowed_bitmap))).decode()
        elif allowed_ids is not None:
            body['allowed_ids'] = list(allowed_ids) if not isinstance(
                allowed_ids, list) else allowed_ids
        if name:
            return await self._request('POST', f'/indexes/{name}/search', body=body)
        body['pattern'] = pattern
        body.pop('cursor_after', None)
        return await self._request('POST', '/search', body=body)
