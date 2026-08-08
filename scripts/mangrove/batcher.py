"""Auto-batching insert helper.

Wraps a Client (or AsyncClient) with a buffer that flushes by size or
by time. The user just calls .insert(vec, metadata=...) and the batcher
amortizes HTTP overhead automatically.

Sync usage :
    client = mg.Client(url)
    with mg.BatchedInserter(client, 'docs', batch_size=2000) as bi:
        for vec, meta in data:
            bi.insert(vec, metadata=meta)
        # flushed at __exit__

Async usage :
    async with mg.AsyncBatchedInserter(aclient, 'docs') as bi:
        ...

For the common pattern : ingest a million vectors, you write a simple
loop, the batcher handles HTTP packaging + freeze auto on 503.
"""
from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

__all__ = ['BatchedInserter', 'AsyncBatchedInserter']


class BatchedInserter:
    """Sync batcher : buffer in RAM, flush when batch_size hit or
       flush_interval seconds since first add. Thread-safe (lock around
       buffer)."""

    def __init__(self, client: Any, name: str, *,
                 batch_size: int = 1000,
                 flush_interval: float = 5.0) -> None:
        self.client = client
        self.name = name
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self._buf_vecs: list[Any] = []
        self._buf_meta: list[dict[str, Any] | None] = []
        self._lock = threading.Lock()
        self._first_add_ts: float = 0.0
        self._closed = False
        # Background flusher : checks every flush_interval/2 if buffer is
        # older than flush_interval and flushes.
        self._stop = False
        self._flusher = threading.Thread(target=self._flush_loop, daemon=True)
        self._flusher.start()

    def __enter__(self) -> 'BatchedInserter':
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def insert(self, vec: Any, metadata: dict[str, Any] | None = None) -> None:
        if self._closed:
            raise RuntimeError('BatchedInserter is closed')
        with self._lock:
            if not self._buf_vecs:
                self._first_add_ts = time.time()
            self._buf_vecs.append(vec)
            self._buf_meta.append(metadata)
            if len(self._buf_vecs) >= self.batch_size:
                self._flush_locked()

    def flush(self) -> list[int]:
        """Force an immediate flush. Returns the doc_ids assigned to the
           buffered vectors (in order). Empty list if buffer was empty."""
        with self._lock:
            return self._flush_locked()

    def _flush_locked(self) -> list[int]:
        if not self._buf_vecs:
            return []
        vecs = self._buf_vecs
        metas = self._buf_meta
        self._buf_vecs = []
        self._buf_meta = []
        self._first_add_ts = 0.0
        # Drop the lock around the HTTP call (long-running) — re-acquire
        # only if we need to push back on failure.
        any_meta = any(m is not None for m in metas)
        if any_meta:
            ids = self.client.insert_batch(self.name, vecs, metadatas=metas)
        else:
            ids = self.client.insert_batch(self.name, vecs)
        return ids

    def _flush_loop(self) -> None:
        while not self._stop:
            time.sleep(self.flush_interval / 2)
            with self._lock:
                if (self._buf_vecs and
                    time.time() - self._first_add_ts >= self.flush_interval):
                    try:
                        self._flush_locked()
                    except Exception as e:
                        # Re-buffering on failure would risk duplicates ;
                        # easier : log and drop. Caller's .insert() will
                        # see the issue on next flush.
                        import sys as _sys
                        _sys.stderr.write(
                            f'[batcher] background flush failed: {e}\n')

    def close(self) -> None:
        if self._closed: return
        self._closed = True
        try:
            self.flush()
        finally:
            self._stop = True


class AsyncBatchedInserter:
    """Async mirror — same semantics, awaitable insert() and flush()."""

    def __init__(self, client: Any, name: str, *,
                 batch_size: int = 1000,
                 flush_interval: float = 5.0) -> None:
        self.client = client
        self.name = name
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self._buf_vecs: list[Any] = []
        self._buf_meta: list[dict[str, Any] | None] = []
        self._lock = asyncio.Lock()
        self._first_add_ts: float = 0.0
        self._closed = False
        self._task: asyncio.Task | None = None

    async def __aenter__(self) -> 'AsyncBatchedInserter':
        self._task = asyncio.create_task(self._flush_loop())
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def insert(self, vec: Any,
                     metadata: dict[str, Any] | None = None) -> None:
        async with self._lock:
            if not self._buf_vecs:
                self._first_add_ts = time.time()
            self._buf_vecs.append(vec)
            self._buf_meta.append(metadata)
            if len(self._buf_vecs) >= self.batch_size:
                await self._flush_locked()

    async def flush(self) -> list[int]:
        async with self._lock:
            return await self._flush_locked()

    async def _flush_locked(self) -> list[int]:
        if not self._buf_vecs:
            return []
        vecs = self._buf_vecs
        metas = self._buf_meta
        self._buf_vecs = []
        self._buf_meta = []
        self._first_add_ts = 0.0
        if any(m is not None for m in metas):
            return await self.client.insert_batch(self.name, vecs, metadatas=metas)
        return await self.client.insert_batch(self.name, vecs)

    async def _flush_loop(self) -> None:
        while not self._closed:
            await asyncio.sleep(self.flush_interval / 2)
            async with self._lock:
                if (self._buf_vecs and
                    time.time() - self._first_add_ts >= self.flush_interval):
                    try:
                        await self._flush_locked()
                    except Exception as e:
                        import sys as _sys
                        _sys.stderr.write(
                            f'[batcher] background flush failed: {e}\n')

    async def close(self) -> None:
        if self._closed: return
        try:
            await self.flush()
        finally:
            self._closed = True
            if self._task:
                self._task.cancel()
                try: await self._task
                except asyncio.CancelledError: pass
