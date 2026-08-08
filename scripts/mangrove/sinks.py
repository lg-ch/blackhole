"""Metadata sinks — bridge between mangrove (vectors) and a sidecar
metadata store (typically ClickHouse) used for filtering.

The pattern :
   - Vectors live in mangrove, indexed by `doc_id` (uint32, server-assigned)
   - Metadata lives in ClickHouse, keyed by the SAME `doc_id`
   - At query time, ClickHouse filters by metadata, returns a doc_id bitmap,
     mangrove receives that bitmap as `allowed_bitmap=` and applies it
     inside the K-way merge

This module provides a thin abstraction so the SDK's insert/search can
talk to both backends transparently. Hard dep on `clickhouse-connect` for
the ClickHouse sink (soft-imported : only loaded when you actually use it).
"""
from __future__ import annotations

import datetime as _dt
import time
from typing import Any, Iterable, Sequence


class MetadataSink:
    """Abstract interface. A sink is anything that can :
         - record (doc_id, ts, metadata) on insert
         - return a bitmap of doc_ids matching a `where` clause on search
    """
    def insert(self, doc_id: int, metadata: dict[str, Any]) -> None: ...
    def insert_batch(self, docs: list[tuple[int, dict[str, Any]]]) -> None: ...
    def filter_bitmap(self, where: str) -> bytes: ...
    def filter_ids(self, ids: Sequence[int], where: str) -> list[int]: ...
    def matching_ids(self, where: str) -> Iterable[int]: ...
    def count_matching(self, where: str) -> int: ...
    def delete_ids(self, ids: Sequence[int]) -> None: ...


class ClickHouseSink(MetadataSink):
    """A metadata sink backed by a ClickHouse table.

    Expects (or creates on first insert) a table with at minimum these
    columns :
        doc_id UInt32          — primary key, joins to mangrove
        ts     DateTime64(3)   — server-assigned at insert time if absent
    Plus any user metadata columns (typed in `schema` or inferred from
    the first insert).

    Usage :
        from mangrove.sinks import ClickHouseSink
        sink = ClickHouseSink(
            url   = 'http://clickhouse:8123',
            table = 'docs_metadata',
            schema = {
                'category': 'LowCardinality(String)',
                'region':   'LowCardinality(String)',
                'lang':     'LowCardinality(String)',
            },
        )

    Pass it at Client construction or attach to an IndexHandle :
        client = mg.Client('http://mg:8000', metadata_sink=sink)
        # or
        idx = client.indexes['docs']
        idx._sink = sink   # programmatic per-index override
    """

    DEFAULT_BASE_COLUMNS = {
        'doc_id': 'UInt32',
        'ts':     'DateTime64(3)',
    }

    def __init__(self,
                 url: str,
                 table: str,
                 schema: dict[str, str] | None = None,
                 database: str = 'default',
                 username: str = 'default',
                 password: str = '',
                 create_table: bool = True,
                 connect_timeout: float = 5.0) -> None:
        try:
            import clickhouse_connect    # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                'mangrove.sinks.ClickHouseSink requires the '
                '`clickhouse-connect` package. Install via : '
                'pip install clickhouse-connect') from e
        import clickhouse_connect
        self._ch = clickhouse_connect.get_client(
            host=url.replace('http://', '').replace('https://', '').split(':')[0],
            port=int(url.split(':')[-1].rstrip('/')) if ':' in url[7:] else 8123,
            username=username,
            password=password,
            database=database,
            connect_timeout=connect_timeout,
        )
        self.table  = table
        self.schema = dict(schema or {})    # user metadata column types
        if create_table:
            self._ensure_table()

    def _ensure_table(self) -> None:
        """CREATE TABLE IF NOT EXISTS with the base columns + user schema.
           Engine choice : AggregatingMergeTree groups by metadata combo
           but here we just want raw rows for groupBitmapState() at query
           time, so a plain MergeTree ORDER BY (ts, doc_id) is correct.   */"""
        cols = ', '.join(
            f'  `{k}` {v}'
            for k, v in (self.DEFAULT_BASE_COLUMNS | self.schema).items())
        sql = (f'CREATE TABLE IF NOT EXISTS `{self.table}` (\n'
               f'{cols}\n'
               f') ENGINE = MergeTree() ORDER BY (ts, doc_id)')
        self._ch.command(sql)

    def insert(self, doc_id: int, metadata: dict[str, Any]) -> None:
        self.insert_batch([(doc_id, metadata)])

    def insert_batch(self, docs: list[tuple[int, dict[str, Any]]]) -> None:
        """One INSERT per call regardless of len(docs). ClickHouse strongly
           prefers batches over single-row inserts — call this with
           several thousand rows at once if you can."""
        if not docs: return
        # Union of all metadata keys ; missing values default to None
        all_keys: list[str] = list(self.schema.keys())
        for _, m in docs:
            for k in m.keys():
                if k not in all_keys and k != 'ts':
                    all_keys.append(k)
        now_ms = lambda: _dt.datetime.fromtimestamp(time.time())
        rows = []
        for doc_id, m in docs:
            row = [int(doc_id), m.get('ts', now_ms())]
            for k in all_keys:
                row.append(m.get(k))
            rows.append(row)
        cols = ['doc_id', 'ts'] + all_keys
        self._ch.insert(self.table, rows, column_names=cols)

    def filter_bitmap(self, where: str) -> bytes:
        """SELECT groupBitmapState(doc_id) FROM <table> WHERE <where>,
           returns CH wire-format bitmap bytes ready to feed mangrove.

           ClickHouse uses TWO bitmap formats internally :
             tag 0x00 (small) : [0x00][varuint N][N × uint32 LE]
             tag 0x01 (large) : [0x01][varuint size][portable bytes]

           mangrove's C-side parser only accepts 0x01. When CH emits 0x00
           (typical for small filters, threshold ≈ 32 ids), we convert
           the inline uint32 list to a portable Roaring bitmap and wrap
           it in the 0x01 envelope here.                                   */"""
        sql = (f"SELECT groupBitmapState(doc_id) "
               f"FROM `{self.table}` WHERE {where} FORMAT RowBinary")
        raw = self._ch.raw_query(sql)
        if not raw:
            return b'\x01\x00'
        # Detect format and convert if needed.
        if raw[0] == 0x01:
            return bytes(raw)               # large, ready as-is
        if raw[0] == 0x00:
            # Small : parse uint32s, build a Roaring bitmap, wrap as 0x01.
            return self._small_to_ch_state(raw)
        # Unknown tag — pass through, mangrove will reject loudly.
        return bytes(raw)

    @staticmethod
    def _small_to_ch_state(raw: bytes) -> bytes:
        """Convert CH's small-format bitmap bytes
              [0x00][varuint N][N × uint32 LE]
           into the large-format envelope
              [0x01][varuint size][portable Roaring bytes]
           Requires pyroaring."""
        try:
            from pyroaring import BitMap
        except ImportError as e:
            raise RuntimeError(
                'ClickHouseSink small-format bitmap conversion requires '
                'pyroaring. Install : pip install pyroaring') from e
        # Parse [0x00][varuint N][N × uint32]
        pos = 1
        n = 0; shift = 0
        while True:
            b = raw[pos]; pos += 1
            n |= (b & 0x7F) << shift
            if not (b & 0x80): break
            shift += 7
        import struct
        ids = struct.unpack_from(f'<{n}I', raw, pos)
        bm = BitMap(ids)
        portable = bm.serialize()
        # Wrap : 0x01 tag + varuint(len) + portable
        sz = len(portable)
        v = bytearray()
        while sz >= 0x80:
            v.append((sz & 0x7F) | 0x80); sz >>= 7
        v.append(sz)
        return b'\x01' + bytes(v) + portable

    def filter_ids(self, ids: Sequence[int], where: str) -> list[int]:
        """Post-filter helper : among the given `ids`, return those that
           also match `where`. Used by Client.search(filter_mode='post')."""
        if not ids:
            return []
        ids_list = ','.join(str(int(i)) for i in ids)
        sql = (f"SELECT doc_id FROM `{self.table}` "
               f"WHERE doc_id IN ({ids_list}) AND ({where})")
        rows = self._ch.query(sql).result_rows
        return [int(r[0]) for r in rows]

    def matching_ids(self, where: str) -> list[int]:
        """Return ALL doc_ids matching `where`. Used by
           Client.delete(where=...) to drive metadata-based deletion."""
        sql = f"SELECT doc_id FROM `{self.table}` WHERE {where}"
        rows = self._ch.query(sql).result_rows
        return [int(r[0]) for r in rows]

    def count_matching(self, where: str) -> int:
        """Count rows matching `where`. Used by post-filter to estimate
           density and size the over-fetch accordingly."""
        sql = f"SELECT count() FROM `{self.table}` WHERE {where}"
        return int(self._ch.query(sql).result_rows[0][0])

    def delete_ids(self, ids: Sequence[int]) -> None:
        """Remove the given doc_ids from the sink so future WHERE-driven
           operations don't see them. Uses ALTER TABLE DELETE (mutation) ;
           if your CH version doesn't support it, override or no-op."""
        if not ids: return
        ids_list = ','.join(str(int(i)) for i in ids)
        sql = (f"ALTER TABLE `{self.table}` "
               f"DELETE WHERE doc_id IN ({ids_list})")
        self._ch.command(sql)

    def close(self) -> None:
        self._ch.close()
