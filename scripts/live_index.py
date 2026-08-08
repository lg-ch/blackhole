"""LiveIndex — streaming-ingest + WAL durability on top of mangrove-search.

Layout on disk:
  <root>/
    manifest.json       # immutable history of frozen segments + index params
    wal.bin             # append-only log of pending mutations
    active.fvecs        # accumulated vectors not yet frozen (vec format)
    seg0/ seg1/ ...     # frozen segments (standard rpforest index dirs)
    tombstones.roaring  # global soft-deletes (handled by Forest at query time)

WAL record format (binary):
  [op:u8][doc_id:u32][len:u32][payload:len bytes]
    op = 0x01 INSERT — payload = dim × float32 (vector)
    op = 0x02 DELETE — payload empty (just the doc_id)

On open, the WAL is replayed: INSERT rebuilds the active buffer, DELETE
goes into the in-memory tombstones list. Once the active is frozen
(its segment.srt files are committed), the WAL is truncated.

Query path: opens every frozen segment via FFI Forest, calls
mg_forest_query_multi, then does L2 rerank against base file. Active
buffer is not queried in this MVP (next iteration: queryable active).

Usage as a library:
    from live_index import LiveIndex
    li = LiveIndex.create('/path/to/idx',
                          dim=128, sub_dim=16, n_trees=200, depth=15,
                          gen_version=3, base_path='/path/to/base.fvecs')
    for vec in stream:
        li.insert(vec)         # writes WAL + buffers in active
        if li.active_size() >= 10_000:
            li.freeze()        # builds segment, atomic publish, truncates WAL
    ids = li.query(qvec, top_n=4000)
    li.close()
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import time
from typing import Iterable

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mangrove_ffi import (Forest, query_multi, set_gen_version,
                          set_shared_scratch_pool)  # noqa: E402


WAL_INSERT = 0x01
WAL_DELETE = 0x02
WAL_RECORD_HDR = struct.Struct('<BII')   # op, doc_id, payload_len

# Manifest schema versioning. Bump SUPPORTED_MAX when adding any breaking
# manifest layout change. Older versions are still readable (forward-
# compatible reader) as long as their fields stay legal. Newer manifests
# are REJECTED with a clear error so users notice the upgrade need.
MANIFEST_VERSION_MIN = 1
MANIFEST_VERSION_MAX = 1


class IncompatibleManifest(Exception):
    """Raised when a manifest version is outside the supported range."""
    pass

# LSM compaction policy : compact K segments of the same tier into one
# segment at tier+1 with depth += DEPTH_PER_TIER. Matches the size factor :
# 4 segments × N docs = 1 segment × 4N docs needs +2 levels of depth
# (since log2(4N) = log2(N) + 2).
COMPACT_TIER_K       = 4
DEPTH_PER_TIER       = 2

# Default backpressure : reject insert() when active buffer reaches this
# many docs. With dim=128 × 4B + 8B overhead per doc this caps RAM at
# ~70 MB; for larger dims the caller should pass a lower max_active in
# create(). 0 disables.
DEFAULT_MAX_ACTIVE = 100_000


class Backpressure(Exception):
    """Raised by insert() when the active buffer is at capacity."""
    pass


def _atomic_write_json(path: str, data: dict) -> None:
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp, path)


class LiveIndex:
    """Streaming-ingest mangrove index with WAL-backed durability."""

    def __init__(self, root: str, mode: str = 'primary') -> None:
        """mode = 'primary' : full read+write
           mode = 'replica' : read-only, periodically rescans manifest"""
        self.root = root
        self.manifest_path = os.path.join(root, 'manifest.json')
        self.wal_path      = os.path.join(root, 'wal.bin')
        self.manifest: dict = {}
        self._base_path_abs: str = ''  # resolved at load, NOT persisted
        # active doc buffer in memory (rebuilt from WAL on open)
        self._active_ids:  list[int]         = []
        self._active_vecs: list[np.ndarray]  = []
        self._wal_fd:      int | None        = None
        # name → Forest map so queries find a stable set during compactions.
        self._forests:     dict[str, Forest] = {}
        if mode not in ('primary', 'replica'):
            raise ValueError(f'unknown mode {mode!r}')
        self.mode = mode
        # Replica state : last seen mtime of manifest.json
        self._last_manifest_mtime: float = 0.0
        self._replica_stop = False
        self._replica_thread = None
        # Async compaction thread (primary only). Workers pick from
        # _compact_queue : each item triggers a _maybe_compact() pass.
        # Coalesced — if a pass is already pending, we don't enqueue again.
        import threading, queue
        self._compact_lock = threading.Lock()
        self._compact_pending = False    # debounce flag
        self._compact_thread = None
        self._compact_stop = False

    # ----- creation / open -----

    @classmethod
    def create(cls, root: str, dim: int, sub_dim: int, n_trees: int,
               depth: int, gen_version: int,
               base_path: str | None = None,
               rpforest_bin: str | None = None,
               max_active: int = DEFAULT_MAX_ACTIVE) -> 'LiveIndex':
        """If base_path is None, LiveIndex maintains its own vecs.fvecs at
           root and auto-appends on every insert(). This is the streaming
           prod pattern. If base_path is provided (e.g. an existing SIFT
           base file), it is used as-is — caller is responsible for it.   */"""
        os.makedirs(root, exist_ok=True)
        auto_store = base_path is None
        if auto_store:
            base_path = os.path.join(root, 'vecs.fvecs')
            # touch empty vecs file (no header in fvecs format)
            open(base_path, 'ab').close()
            # Stored relative to root so the manifest is location-portable
            # (backup → restore on another host without path patching).
            base_path_stored = 'vecs.fvecs'
        else:
            base_path_stored = os.path.abspath(base_path)
        manifest = {
            'version':      1,
            'dim':          dim,
            'sub_dim':      sub_dim,
            'n_trees':      n_trees,
            'depth':        depth,
            'gen_version':  gen_version,
            'base_path':    base_path_stored,
            'auto_store':   auto_store,
            'rpforest_bin': rpforest_bin or os.path.join(
                os.path.dirname(HERE), 'rpforest'),
            'max_active':   max_active,
            'segments':     [],
            'next_doc_id':  0,
        }
        _atomic_write_json(os.path.join(root, 'manifest.json'), manifest)
        # touch empty WAL
        open(os.path.join(root, 'wal.bin'), 'ab').close()
        li = cls(root)
        li._load()
        return li

    @classmethod
    def open(cls, root: str, mode: str = 'primary') -> 'LiveIndex':
        li = cls(root, mode=mode)
        li._load()
        if mode == 'replica':
            li._start_replica_watch()
        return li

    def _start_replica_watch(self, interval: float = 5.0) -> None:
        """Background thread that re-reads manifest.json on mtime change.
           New segments are opened, compacted-away ones are closed.       */"""
        import threading
        def loop():
            while not self._replica_stop:
                time.sleep(interval)
                if self._replica_stop:
                    break
                try:
                    mt = os.path.getmtime(self.manifest_path)
                    if mt > self._last_manifest_mtime:
                        self._refresh_replica()
                        self._last_manifest_mtime = mt
                except FileNotFoundError:
                    pass
                except Exception as e:
                    sys.stderr.write(f'[replica] watch error: {e}\n')
        self._last_manifest_mtime = os.path.getmtime(self.manifest_path)
        t = threading.Thread(target=loop, daemon=True, name='replica-watch')
        t.start()
        self._replica_thread = t

    def _refresh_replica(self) -> None:
        """Pick up new segments from disk ; close ones no longer in manifest."""
        with open(self.manifest_path) as f:
            new_manifest = json.load(f)
        current = {s['name'] for s in self.manifest.get('segments', [])}
        latest  = {s['name'] for s in new_manifest.get('segments', [])}
        # Open new ones
        for seg in new_manifest['segments']:
            if seg['name'] not in self._forests:
                self.manifest = new_manifest
                self._open_segment(seg)
        # Close removed ones (compacted away)
        for name in list(self._forests.keys()):
            if name not in latest:
                try:
                    self._forests[name].close()
                except Exception:
                    pass
                del self._forests[name]
        self.manifest = new_manifest
        sys.stderr.write(
            f'[replica] refreshed: now {len(self._forests)} segment(s)\n')

    def _load(self) -> None:
        with open(self.manifest_path) as f:
            self.manifest = json.load(f)
        # Schema version check — must be in [MIN, MAX]. A future-version
        # manifest produced by a newer mangrove gets a clear error rather
        # than silent misinterpretation. Old versions get migrated forward
        # in-memory (legacy fields populated with sensible defaults).
        v = self.manifest.get('version', 1)
        if v < MANIFEST_VERSION_MIN:
            raise IncompatibleManifest(
                f'manifest version {v} too old (min {MANIFEST_VERSION_MIN}); '
                f'run scripts/migrate_manifest.py first')
        if v > MANIFEST_VERSION_MAX:
            raise IncompatibleManifest(
                f'manifest version {v} too new for this binary '
                f'(supported up to {MANIFEST_VERSION_MAX}); upgrade mangrove')
        # Forward-compat defaults : when a v1 manifest doesn't have fields
        # introduced in v2+, fall back to safe values.
        self.manifest.setdefault('auto_store', False)
        self.manifest.setdefault('max_active', DEFAULT_MAX_ACTIVE)
        # Resolve base_path: relative paths are stored against root, absolute
        # paths are taken as-is. Cached in _base_path_abs; manifest['base_path']
        # is kept as authored so it survives backup/restore unchanged.
        bp = self.manifest['base_path']
        self._base_path_abs = bp if os.path.isabs(bp) else os.path.join(self.root, bp)
        # Replay WAL to rebuild the active buffer.
        self._active_ids.clear()
        self._active_vecs.clear()
        if os.path.exists(self.wal_path):
            self._replay_wal()
        # Recover next_doc_id from WAL replay : crash between insert()'s
        # WAL fsync and the next manifest write may leave next_doc_id
        # stale. The active buffer + frozen segments together define the
        # high-water mark of seen ids.
        seen_max = max(self._active_ids, default=-1)
        seg_max  = max(
            (s['doc_offset'] + s['n_docs'] - 1
             for s in self.manifest['segments']), default=-1)
        recovered = max(seen_max, seg_max) + 1
        if recovered > self.manifest.get('next_doc_id', 0):
            self.manifest['next_doc_id'] = recovered
        # Open frozen segments via FFI for query_multi.
        # Each segment carries its own depth (LSM tiers may differ per segment).
        set_gen_version(self.manifest['gen_version'])
        # Multi-forest clusters benefit massively from the shared scratch
        # pool : per-query bytes_buf and docs_buf (~300-500 MB at SIFT 1B
        # scale) become thread-local globals shared across all Forests.
        # No latency cost — queries are already serial inside this LiveIndex.
        set_shared_scratch_pool(True)
        for seg in self.manifest['segments']:
            seg.setdefault('tier', 0)  # backwards-compat for pre-LSM manifests
            self._open_segment(seg)

    def _open_segment(self, seg: dict) -> None:
        seg_dir = os.path.join(self.root, seg['name'])
        # n_docs sizes the doc-id space for the K-way merge; must exceed
        # every doc_id present. With explicit doc_ids next_doc_id may be 0,
        # so fall back to max(doc_offset + n_docs) across segments.
        n_docs_total = max(
            self.manifest['next_doc_id'],
            max((s['doc_offset'] + s['n_docs']
                 for s in self.manifest['segments']), default=1),
            1,
        )
        self._forests[seg['name']] = Forest(
            seg_dir,
            n_trees=self.manifest['n_trees'],
            dim=self.manifest['dim'],
            sub_dim=self.manifest['sub_dim'],
            depth=seg['depth'],
            n_docs=n_docs_total,
            gen_version=self.manifest['gen_version'],
        )

    def _replay_wal(self) -> None:
        # If auto_store, also re-apply inserts to vecs.fvecs (insert order
        # writes WAL first then vecs.fvecs; the vec store may have a hole
        # for the last few records).
        auto_store = self.manifest.get('auto_store', False)
        row_bytes = 4 + self.manifest['dim'] * 4 if auto_store else 0
        store_fd = (open(self._base_path_abs, 'r+b')
                    if auto_store else None)
        with open(self.wal_path, 'rb') as f:
            while True:
                hdr = f.read(WAL_RECORD_HDR.size)
                if not hdr or len(hdr) < WAL_RECORD_HDR.size:
                    break
                op, doc_id, plen = WAL_RECORD_HDR.unpack(hdr)
                payload = f.read(plen)
                if op == WAL_INSERT:
                    vec = np.frombuffer(payload, dtype=np.float32).copy()
                    if len(vec) != self.manifest['dim']:
                        sys.stderr.write(
                            f'[wal] truncated INSERT for doc {doc_id}\n')
                        break
                    self._active_ids.append(doc_id)
                    self._active_vecs.append(vec)
                    if store_fd is not None:
                        store_fd.seek(doc_id * row_bytes)
                        store_fd.write(struct.pack('<i', self.manifest['dim']))
                        store_fd.write(vec.tobytes())
                elif op == WAL_DELETE:
                    # Tombstones survive via tombstones.roaring (loaded by
                    # each segment's Forest). WAL DELETE record is informative
                    # only at this point.
                    pass
                else:
                    sys.stderr.write(f'[wal] unknown op {op:#x} — stop\n')
                    break
        if store_fd is not None:
            store_fd.flush()
            os.fsync(store_fd.fileno())
            store_fd.close()

    # ----- write path -----

    def _open_wal(self) -> int:
        if self._wal_fd is None:
            self._wal_fd = os.open(
                self.wal_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        return self._wal_fd

    def insert(self, vec: np.ndarray, doc_id: int | None = None) -> int:
        if self.mode == 'replica':
            raise RuntimeError(
                'replica mode is read-only — route writes to the primary')
        cap = self.manifest.get('max_active', DEFAULT_MAX_ACTIVE)
        if cap > 0 and len(self._active_ids) >= cap:
            raise Backpressure(
                f'active buffer at capacity ({len(self._active_ids)}/{cap}); '
                f'call freeze() or wait')
        if vec.dtype != np.float32:
            vec = vec.astype(np.float32, copy=False)
        if doc_id is None:
            doc_id = self.manifest['next_doc_id']
            self.manifest['next_doc_id'] += 1
        # WAL first (durable), then buffer.
        fd = self._open_wal()
        hdr = WAL_RECORD_HDR.pack(WAL_INSERT, doc_id, vec.nbytes)
        os.write(fd, hdr + vec.tobytes())
        os.fsync(fd)
        # If auto-store, also append to the managed vecs.fvecs at offset doc_id.
        # We assume insertions arrive with monotonic doc_ids in [0, next_doc_id);
        # the fvecs row layout is [int32 dim][dim*float32] per record so we
        # seek to record × row_bytes and write.
        if self.manifest.get('auto_store', False):
            row_bytes = 4 + self.manifest['dim'] * 4
            with open(self._base_path_abs, 'r+b') as bf:
                bf.seek(doc_id * row_bytes)
                bf.write(struct.pack('<i', self.manifest['dim']))
                bf.write(vec.astype(np.float32, copy=False).tobytes())
                bf.flush()
                os.fsync(bf.fileno())
        self._active_ids.append(doc_id)
        self._active_vecs.append(vec.copy())
        return doc_id

    def delete(self, doc_id: int) -> None:
        if self.mode == 'replica':
            raise RuntimeError('replica mode is read-only')
        fd = self._open_wal()
        hdr = WAL_RECORD_HDR.pack(WAL_DELETE, doc_id, 0)
        os.write(fd, hdr)
        os.fsync(fd)
        # Apply on every loaded segment whose doc range covers doc_id.
        # Forest tombstones are per-segment; we add to all (the bitmap is
        # cheap to maintain and only fired for ids actually in the segment).
        for f in self._forests.values():
            try:
                f.tombstone_add(doc_id)
            except Exception:
                pass
        for f in self._forests.values():
            try:
                f.tombstones_flush()
            except Exception:
                pass

    def active_size(self) -> int:
        return len(self._active_ids)

    # ----- freeze -----

    def freeze(self, depth_override: int | None = None) -> str | None:
        """Build a frozen segment from the current active buffer.
           Returns the segment name, or None if active is empty.            """
        if self.mode == 'replica':
            raise RuntimeError('replica mode is read-only')
        if not self._active_ids:
            return None

        n_segs = len(self.manifest['segments'])
        seg_name = f'seg{n_segs}'
        seg_dir = os.path.join(self.root, seg_name)
        seg_dir_tmp = seg_dir + '.tmp'
        if os.path.exists(seg_dir_tmp):
            subprocess.check_call(['rm', '-rf', seg_dir_tmp])
        os.makedirs(seg_dir_tmp, exist_ok=True)

        dim = self.manifest['dim']
        doc_offset = min(self._active_ids)
        n_vecs = len(self._active_vecs)
        auto_store = self.manifest.get('auto_store', False)

        if auto_store:
            # Vectors are already in the managed vecs.fvecs at rows
            # [doc_offset, doc_offset + n_vecs). Build reads them directly.
            vecs_path = self._base_path_abs
            file_offset = doc_offset
            doc_id_base = doc_offset  # same as file row → doc_id
        else:
            # Write a temporary in.fvecs from the active buffer.
            vecs_path = os.path.join(seg_dir_tmp, 'in.fvecs')
            with open(vecs_path, 'wb') as f:
                for v in self._active_vecs:
                    f.write(struct.pack('<i', dim))
                    f.write(v.astype(np.float32, copy=False).tobytes())
            file_offset = 0
            doc_id_base = doc_offset

        depth = depth_override or max(self.manifest['depth'],
                                      int(np.log2(max(2, n_vecs))) - 2)
        cmd = [
            self.manifest['rpforest_bin'],
            '--dim', str(dim),
            '--sub_dim', str(self.manifest['sub_dim']),
            '--gen', f'v{self.manifest["gen_version"]}',
            '--doc_offset', str(file_offset),
            '--doc_count', str(n_vecs),
            '--doc_id_base', str(doc_id_base),
            'build', vecs_path, seg_dir_tmp,
            str(self.manifest['n_trees']), str(depth),
        ]
        sys.stderr.write(f'[freeze] building {seg_name}: '
                         f'doc_offset={doc_offset} n={n_vecs} depth={depth} '
                         f'(auto_store={auto_store})\n')
        subprocess.check_call(cmd, stdout=sys.stderr)
        if not auto_store:
            os.remove(vecs_path)

        # Atomic publish: rename .tmp → final, append to manifest.
        os.rename(seg_dir_tmp, seg_dir)
        self.manifest['segments'].append({
            'name':       seg_name,
            'n_docs':     n_vecs,
            'doc_offset': doc_offset,
            'depth':      depth,
            'tier':       0,
        })
        _atomic_write_json(self.manifest_path, self.manifest)

        # Open the new segment, drop the active buffer + truncate WAL.
        self._open_segment(self.manifest['segments'][-1])
        # End of fast freeze path. Compaction trigger is dispatched to a
        # background thread so freeze() returns quickly to the caller.
        # The thread picks up the work, runs _maybe_compact() with its own
        # lock against the LiveIndex. Multiple back-to-back freezes coalesce
        # into one pending pass.
        self._schedule_compact_async()
        self._active_ids.clear()
        self._active_vecs.clear()
        if self._wal_fd is not None:
            os.close(self._wal_fd)
            self._wal_fd = None
        # truncate WAL atomically by re-creating the file
        open(self.wal_path, 'wb').close()

        return seg_name

    # ----- LSM compaction (async) -----

    def _schedule_compact_async(self) -> None:
        """Schedule a _maybe_compact() pass on the background thread.
           If a pass is already pending or running, this is a no-op
           (the existing pass will see the new segments and cascade
           through tiers naturally).                                   */"""
        import threading
        with self._compact_lock:
            if self._compact_pending or self._compact_stop:
                return
            self._compact_pending = True
        if self._compact_thread is None or not self._compact_thread.is_alive():
            self._compact_thread = threading.Thread(
                target=self._compact_worker, name='lsm-compact', daemon=True)
            self._compact_thread.start()

    def _compact_worker(self) -> None:
        """Drain pending compaction work. Sleeps when idle, exits on stop. */"""
        while not self._compact_stop:
            with self._compact_lock:
                pending = self._compact_pending
                self._compact_pending = False
            if not pending:
                import time as _t
                _t.sleep(0.2)
                continue
            try:
                # Run the cascade. Each iteration may free up a tier
                # then re-trigger _maybe_compact for cascading effects.
                self._maybe_compact()
            except Exception as e:
                sys.stderr.write(f'[compact-worker] error: {e}\n')

    def compact_wait(self, timeout: float = 600.0) -> bool:
        """Block until no pending compaction work remains. Returns False
           on timeout. Useful in tests and for explicit barriers."""
        import time as _t
        start = _t.time()
        while _t.time() - start < timeout:
            with self._compact_lock:
                if not self._compact_pending:
                    # Worker might still be inside _maybe_compact ; check
                    # if its thread is idle via a tiny sleep + recheck.
                    pass
                else:
                    _t.sleep(0.1); continue
            # Idle moment : peek if any tier is over K. If yes, wait more.
            from collections import defaultdict
            by_tier = defaultdict(int)
            for s in self.manifest.get('segments', []):
                by_tier[s.get('tier', 0)] += 1
            if all(c < COMPACT_TIER_K for c in by_tier.values()):
                return True
            _t.sleep(0.2)
        return False

    def _maybe_compact(self) -> None:
        """Walk manifest tiers, compact any tier holding ≥ COMPACT_TIER_K
           segments into one segment at tier+1 with depth += DEPTH_PER_TIER.
           Cascades: a new tier+1 segment might itself trigger tier+1's compact. """
        while True:
            from collections import defaultdict
            by_tier: dict[int, list[dict]] = defaultdict(list)
            for s in self.manifest['segments']:
                by_tier[s.get('tier', 0)].append(s)
            ready = sorted(t for t, segs in by_tier.items()
                              if len(segs) >= COMPACT_TIER_K)
            if not ready:
                return
            tier = ready[0]
            self._compact_tier(tier, by_tier[tier][:COMPACT_TIER_K])
            # loop: the newly-created tier+1 segment might trigger another

    def _compact_tier(self, tier: int, sources: list[dict]) -> None:
        """Atomic-swap compaction: the source segments stay queryable
           throughout the compact subprocess. Only AFTER the new segment
           is built do we open it, swap the manifest+Forest dict atomically,
           then close+rm the sources. Queries during compaction see the
           OLD set (or the NEW set after the swap) — never an empty/partial
           state.                                                          """
        src_dirs = [os.path.join(self.root, s['name']) for s in sources]
        src_names = [s['name'] for s in sources]
        new_tier  = tier + 1
        new_depth = sources[0]['depth'] + DEPTH_PER_TIER
        n_new = sum(s['n_docs'] for s in sources)
        new_name = f'seg_t{new_tier}_{len(self.manifest["segments"])}'
        new_dir  = os.path.join(self.root, new_name)
        sys.stderr.write(
            f'[compact] tier {tier}→{new_tier}: {len(sources)} segs × '
            f'depth {sources[0]["depth"]} → 1 seg × depth {new_depth} '
            f'({n_new} docs) — source forests stay live during build\n')

        # Run compact subprocess. Source dirs are read but not modified; their
        # Forest objects continue serving queries. New segment lands in new_dir.
        cmd = [
            self.manifest['rpforest_bin'], '--dim', str(self.manifest['dim']),
            'compact', new_dir, self._base_path_abs,
            str(new_depth),
        ] + src_dirs
        subprocess.check_call(cmd, stdout=sys.stderr)

        # Open the new Forest while sources are still live.
        new_seg = {
            'name':       new_name,
            'n_docs':     n_new,
            'doc_offset': min(s['doc_offset'] for s in sources),
            'depth':      new_depth,
            'tier':       new_tier,
        }
        new_segments = [s for s in self.manifest['segments']
                        if s not in sources] + [new_seg]
        # Open new Forest before swap (sources still in self._forests).
        self._open_segment(new_seg)

        # ATOMIC SWAP : manifest first (durable), then pop sources from
        # self._forests so queries hitting in-between still see all live
        # forests. Order matters : write manifest -> drop sources -> close.
        self.manifest['segments'] = new_segments
        _atomic_write_json(self.manifest_path, self.manifest)
        # Now sources are no longer in the manifest; pull them out of the
        # live dict and close. Queries between the pop and close are fine —
        # they will not iterate over sources but may have references to the
        # Forest objects mid-call; the close() races with that, so we close
        # *after* yielding briefly. For Python-level concurrency this is
        # protected by the GIL : the query loop and this close don't run
        # interleaved within a single iteration.
        dropped = [self._forests.pop(name) for name in src_names
                   if name in self._forests]
        for f in dropped:
            f.close()
        for d in src_dirs:
            subprocess.check_call(['rm', '-rf', d])

    # ----- query -----

    def query(self, qvec: np.ndarray, top_n: int = 4000,
              top_k: int = 10, include_active: bool = True,
              metric: str = 'l2',
              allowed_ids: np.ndarray | None = None,
              allowed_bitmap: bytes | None = None,
              query_depth: int = 0,
              n_probes: int = 0,
              max_leaf_bytes: int = 0) -> np.ndarray:
        """Query each segment at its own build depth, merge votes, L2 rerank.

           Filter options (mutually exclusive — bitmap wins if both set) :
             allowed_ids    : list/np.ndarray of int doc_ids. Internally
                              built into a CRoaring bitmap on each call.
                              OK for small filters built in Python.
             allowed_bitmap : raw bytes of a CRoaring portable-serialize
                              bitmap. THE preferred path for ClickHouse
                              integration : the bytes from
                              `SELECT groupBitmapState(doc_id) FROM ...`
                              are exactly this format, can be passed
                              through with zero parsing/rebuild cost.

           If include_active, also L2-rank the in-memory active buffer
           (docs inserted but not yet frozen) — gives read-your-writes
           semantics.                                                     */"""
        forests_snapshot = list(self._forests.values())
        # Snapshot active buffer separately — a concurrent freeze may clear it,
        # but the doc_ids would then be in the new frozen segment which we
        # MIGHT or MIGHT NOT have in forests_snapshot. The dup is harmless
        # (same doc_id seen twice → vote_acc adds, doesn't double-count
        # since the rerank dedups on doc_id).
        active_ids = list(self._active_ids)
        active_vecs = list(self._active_vecs)

        if not forests_snapshot and not active_ids:
            return np.empty(0, dtype=np.int32)

        # Pre-filter : bitmap path (preferred, zero rebuild cost) or
        # int-array path (built into bitmap inside the C K-way merge).
        allowed_arr = None
        if allowed_ids is not None and allowed_bitmap is None:
            allowed_arr = np.asarray(allowed_ids, dtype=np.int32)
            if not allowed_arr.flags['C_CONTIGUOUS']:
                allowed_arr = np.ascontiguousarray(allowed_arr)

        # Tail cap (thread-local C global) : skip oversized leaves to bound
        # p99. 0 = no cap (legacy). Set per-query to honor caller intent.
        from mangrove_ffi import set_max_leaf_bytes
        set_max_leaf_bytes(int(max_leaf_bytes) if max_leaf_bytes else 0)

        vote_acc: dict[int, int] = {}
        for f in forests_snapshot:
            if allowed_bitmap is not None:
                if n_probes > 0:
                    ids_seg, votes_seg, n_seg = f.query_probes(
                        qvec, n_probes, top_n=top_n,
                        probe_depth=query_depth,
                        allowed_state=allowed_bitmap)
                else:
                    ids_seg, votes_seg, n_seg = f.query(
                        qvec, top_n=top_n, query_depth=query_depth,
                        allowed_state=allowed_bitmap)
            elif allowed_arr is not None:
                # int-array filter — no multi-probe support yet (filter path
                # uses per-leaf roaring iterator, incompatible with the
                # fused multi-probe merge). Single-probe at query_depth.
                ids_seg, votes_seg, n_seg = f.query_with_ids(
                    qvec, allowed_arr, top_n=top_n,
                    query_depth=query_depth)
            else:
                if n_probes > 0:
                    ids_seg, votes_seg, n_seg = f.query_probes(
                        qvec, n_probes, top_n=top_n,
                        probe_depth=query_depth)
                else:
                    ids_seg, votes_seg, n_seg = f.query(
                        qvec, top_n=top_n, query_depth=query_depth)
            for i in range(n_seg):
                vote_acc[int(ids_seg[i])] = vote_acc.get(int(ids_seg[i]), 0) + int(votes_seg[i])

        # Active buffer brute-force : compute L2 to each active doc and emit
        # synthetic high votes so they survive into rerank. Vote weight is
        # chosen so they consistently make the rerank candidate set; the
        # subsequent L2 rerank decides the final ranking against all cands.
        if include_active and active_ids:
            if qvec.dtype != np.float32:
                qvec_f = qvec.astype(np.float32, copy=False)
            else:
                qvec_f = qvec
            # Filter active by allowed_ids if requested
            if allowed_arr is not None:
                allowed_set = set(int(x) for x in allowed_arr)
                keep_idx = [i for i, d in enumerate(active_ids) if d in allowed_set]
                act_ids = [active_ids[i] for i in keep_idx]
                act_vecs = [active_vecs[i] for i in keep_idx]
            else:
                act_ids = active_ids
                act_vecs = active_vecs
            if not act_ids:
                pass   # filter excluded everything
            else:
                arr = np.stack(act_vecs)
                d2 = ((arr - qvec_f) ** 2).sum(axis=1)
                n_pick = min(len(act_ids), max(1, top_n // 10))
                order = np.argpartition(d2, n_pick - 1)[:n_pick]
                for j in order:
                    # synthetic vote = high so they survive the rerank cut
                    vote_acc[act_ids[j]] = vote_acc.get(act_ids[j], 0) + 10000

        if not vote_acc:
            return np.empty(0, dtype=np.int32)
        items = sorted(vote_acc.items(), key=lambda kv: -kv[1])[:top_n]
        cand_ids = np.array([k for k, _ in items], dtype=np.int32)
        # Rerank with the chosen metric. L2 uses the C-side rerank_l2 for
        # speed; cosine and inner-product (ip) go through a numpy fallback
        # that reads candidate vecs from the base file.
        if metric == 'l2' and forests_snapshot:
            return forests_snapshot[0].rerank_l2(
                self._base_path_abs, qvec, cand_ids, top_k=top_k)
        if qvec.dtype != np.float32:
            qvec = qvec.astype(np.float32, copy=False)
        dim = self.manifest['dim']
        row_bytes = 4 + dim * 4
        scores: list[tuple[float, int]] = []
        with open(self._base_path_abs, 'rb') as bf:
            for d in cand_ids:
                bf.seek(int(d) * row_bytes + 4)
                v = np.frombuffer(bf.read(dim * 4), dtype=np.float32)
                if len(v) != dim:
                    continue
                if metric == 'cosine':
                    nv = float(np.linalg.norm(v))
                    nq = float(np.linalg.norm(qvec))
                    score = -float(np.dot(v, qvec)) / max(1e-12, nv * nq)
                elif metric == 'ip':
                    score = -float(np.dot(v, qvec))   # smaller = better after negation
                else:
                    score = float(((v - qvec) ** 2).sum())
                scores.append((score, int(d)))
        scores.sort()
        return np.array([d for _, d in scores[:top_k]], dtype=np.int32)

    # ----- lifecycle -----

    def close(self) -> None:
        # Stop replica watch thread if any
        self._replica_stop = True
        if self._replica_thread is not None:
            self._replica_thread.join(timeout=10)
            self._replica_thread = None
        # Stop async compaction thread (let in-flight pass finish naturally)
        self._compact_stop = True
        if self._compact_thread is not None:
            self._compact_thread.join(timeout=30)
            self._compact_thread = None
        if self._wal_fd is not None:
            os.close(self._wal_fd)
            self._wal_fd = None
        for f in self._forests.values():
            try:
                f.close()
            except Exception:
                pass
        self._forests.clear()


# ---------------- CLI ----------------

def _cli_init(args):
    LiveIndex.create(args.root,
                     dim=args.dim, sub_dim=args.sub_dim,
                     n_trees=args.n_trees, depth=args.depth,
                     gen_version=args.gen, base_path=args.base)
    print(f'created live index at {args.root}')


def _cli_stats(args):
    li = LiveIndex.open(args.root)
    m = li.manifest
    print(f'  segments    : {len(m["segments"])}')
    for s in m['segments']:
        print(f'    {s["name"]:6} n={s["n_docs"]:9}  offset={s["doc_offset"]:9}  depth={s["depth"]}')
    print(f'  active      : {li.active_size()} docs')
    print(f'  next_doc_id : {m["next_doc_id"]}')
    li.close()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd')

    p = sub.add_parser('init')
    p.add_argument('root')
    p.add_argument('--dim', type=int, required=True)
    p.add_argument('--sub_dim', type=int, default=16)
    p.add_argument('--n_trees', type=int, default=1000)
    p.add_argument('--depth', type=int, default=18)
    p.add_argument('--gen', type=int, default=3)
    p.add_argument('--base', required=True)
    p.set_defaults(fn=_cli_init)

    p = sub.add_parser('stats')
    p.add_argument('root')
    p.set_defaults(fn=_cli_stats)

    args = ap.parse_args()
    if not args.cmd:
        ap.print_help()
        return
    args.fn(args)


if __name__ == '__main__':
    main()
