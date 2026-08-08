"""MangroveCluster — a registry of LiveIndex instances under a single root.

Layout :
   <cluster_root>/
     registry.json       — list of registered indexes + their params
     <index_name>/       — each index = a full LiveIndex dir
       manifest.json
       wal.bin
       ...

Operations :
   - create_index(name, ...)        → adds to registry + creates LiveIndex
   - drop_index(name)               → removes from registry + rms dir
   - list_indexes(pattern=None)     → optional glob filter (e.g. "arxiv-*")
   - get(name)                      → LiveIndex
   - search(pattern, qvec, ...)     → multi-index search across pattern

Multi-index search merges votes per doc_id across matching indexes
then L2 reranks. Caller MUST ensure indexes share the same dim
(we enforce this at create() time when joining a cluster).
"""
from __future__ import annotations

import fnmatch
import json
import os
import shutil
import sys
import threading
from typing import Iterable

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from live_index import LiveIndex


def _atomic_write_json(path: str, data: dict) -> None:
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp, path)


class MangroveCluster:
    """A multi-index registry over a single root directory."""

    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(root)
        self.registry_path = os.path.join(self.root, 'registry.json')
        self._lock = threading.Lock()
        self._indexes: dict[str, LiveIndex] = {}
        self.registry: dict = {}
        if not os.path.exists(self.root):
            os.makedirs(self.root, exist_ok=True)
        if not os.path.exists(self.registry_path):
            self.registry = {'version': 1, 'indexes': {}}
            _atomic_write_json(self.registry_path, self.registry)
        else:
            with open(self.registry_path) as f:
                self.registry = json.load(f)
        # Reconcile : on-disk index dirs may exist without an entry in
        # the registry (server restart against a populated root, ops
        # tooling that creates dirs out-of-band, etc.). Adopt them.
        self._discover_and_adopt_orphans()

    def _discover_and_adopt_orphans(self) -> None:
        """Scan `self.root` for subdirs containing manifest.json that are
           NOT yet in the registry, and register them. Writes a new
           registry.json if anything was adopted."""
        try:
            entries = os.listdir(self.root)
        except FileNotFoundError:
            return
        adopted: list[str] = []
        for name in entries:
            if name in self.registry['indexes']:
                continue
            if name in ('lost+found', 'registry.json',
                        'registry.json.tmp', '.', '..'):
                continue
            d = os.path.join(self.root, name)
            mpath = os.path.join(d, 'manifest.json')
            if not (os.path.isdir(d) and os.path.isfile(mpath)):
                continue
            try:
                with open(mpath) as f:
                    m = json.load(f)
                self.registry['indexes'][name] = {
                    'dim':         int(m['dim']),
                    'sub_dim':     int(m.get('sub_dim', 16)),
                    'n_trees':     int(m.get('n_trees', 1000)),
                    'depth':       int(m.get('depth', 14)),
                    'gen_version': int(m.get('gen_version', 3)),
                }
                adopted.append(name)
            except (KeyError, ValueError, OSError, json.JSONDecodeError) as e:
                print(f'registry: skip orphan {name!r}: {e}', file=sys.stderr)
        if adopted:
            _atomic_write_json(self.registry_path, self.registry)
            print(f'registry: adopted {len(adopted)} orphan index(es): '
                  f'{", ".join(adopted)}', file=sys.stderr)

    # ---- lifecycle ----

    def create_index(self, name: str, *, dim: int, sub_dim: int,
                     n_trees: int, depth: int, gen_version: int = 3,
                     base_path: str | None = None,
                     max_active: int = 100_000) -> LiveIndex:
        """Register a new index ; LiveIndex is created lazily on first access."""
        with self._lock:
            if name in self.registry['indexes']:
                raise ValueError(f'index {name!r} already exists')
            idx_dir = os.path.join(self.root, name)
            li = LiveIndex.create(idx_dir, dim=dim, sub_dim=sub_dim,
                                  n_trees=n_trees, depth=depth,
                                  gen_version=gen_version,
                                  base_path=base_path, max_active=max_active)
            self.registry['indexes'][name] = {
                'dim': dim, 'sub_dim': sub_dim, 'n_trees': n_trees,
                'depth': depth, 'gen_version': gen_version,
            }
            _atomic_write_json(self.registry_path, self.registry)
            self._indexes[name] = li
            return li

    def drop_index(self, name: str) -> None:
        with self._lock:
            if name not in self.registry['indexes']:
                raise KeyError(name)
            if name in self._indexes:
                try:
                    self._indexes[name].close()
                except Exception:
                    pass
                del self._indexes[name]
            shutil.rmtree(os.path.join(self.root, name), ignore_errors=True)
            del self.registry['indexes'][name]
            _atomic_write_json(self.registry_path, self.registry)

    def get(self, name: str) -> LiveIndex:
        with self._lock:
            if name in self._indexes:
                return self._indexes[name]
            if name not in self.registry['indexes']:
                raise KeyError(name)
            idx_dir = os.path.join(self.root, name)
            li = LiveIndex.open(idx_dir)
            self._indexes[name] = li
            return li

    def close(self) -> None:
        for li in list(self._indexes.values()):
            try:
                li.close()
            except Exception:
                pass
        self._indexes.clear()

    # ---- listing ----

    def list_indexes(self, pattern: str | None = None) -> list[str]:
        """Return registered index names, optionally filtered by glob.
           Examples : 'arxiv-*', '*-2026*', 'docs-202?-1*'.            */"""
        names = list(self.registry['indexes'].keys())
        if pattern:
            names = [n for n in names if fnmatch.fnmatch(n, pattern)]
        return sorted(names)

    def stats(self) -> dict:
        """Aggregate registry stats : per-index segment count + total docs."""
        out = {}
        for name in self.registry['indexes']:
            try:
                li = self.get(name)
                out[name] = {
                    'n_segments':  len(li.manifest['segments']),
                    'active_size': li.active_size(),
                    'next_doc_id': li.manifest['next_doc_id'],
                    'dim':         li.manifest['dim'],
                }
            except Exception as e:
                out[name] = {'error': str(e)}
        return out

    # ---- multi-index search ----

    def search(self, pattern: str, qvec: np.ndarray, top_n: int = 4000,
               top_k: int = 10, include_active: bool = True) -> dict:
        """Search across all indexes whose name matches `pattern`. Merges
           per-segment votes across all matching indexes via the same
           K-way pattern as LiveIndex.query, then a single L2 rerank
           against each matching index's base_path (the rerank winner
           wins overall). Returns {ids:[...], per_index_counts:{...}}.  */"""
        names = self.list_indexes(pattern)
        if not names:
            return {'ids': [], 'matched_indexes': [], 'note': 'no match'}
        # Sanity: enforce common dim
        dims = {self.registry['indexes'][n]['dim'] for n in names}
        if len(dims) > 1:
            raise ValueError(f'mixed dims in pattern {pattern!r}: {dims}')

        all_cands: list[tuple[int, int, str]] = []  # (doc_id, vote, index_name)
        per_idx = {}
        for n in names:
            li = self.get(n)
            forests = list(li._forests.values())
            for f in forests:
                ids, votes, k = f.query(qvec, top_n=top_n)
                for j in range(k):
                    all_cands.append((int(ids[j]), int(votes[j]), n))
            # active buffer
            if include_active and li._active_ids:
                arr = np.stack(li._active_vecs)
                qv = qvec.astype(np.float32, copy=False)
                d2 = ((arr - qv) ** 2).sum(axis=1)
                n_pick = min(len(li._active_ids), max(1, top_n // 10))
                for j in np.argpartition(d2, n_pick - 1)[:n_pick]:
                    all_cands.append((li._active_ids[j], 10000, n))
            per_idx[n] = sum(1 for c in all_cands if c[2] == n)
        if not all_cands:
            return {'ids': [], 'matched_indexes': names, 'per_index_counts': per_idx}

        # Aggregate votes by (doc_id, index_name) — same doc_id across
        # different indexes is treated as DIFFERENT docs (they have
        # different semantics). Caller can globally namespace if needed.
        scored: dict[tuple[int, str], int] = {}
        for doc_id, vote, n in all_cands:
            key = (doc_id, n)
            scored[key] = scored.get(key, 0) + vote
        top_cands = sorted(scored.items(), key=lambda kv: -kv[1])[:top_n]

        # L2 rerank each candidate against its OWN index's base file.
        if qvec.dtype != np.float32:
            qvec = qvec.astype(np.float32, copy=False)
        dim = self.registry['indexes'][names[0]]['dim']
        row_bytes = 4 + dim * 4
        scored_l2: list[tuple[float, int, str]] = []
        # Group by index for one fd per index
        cands_by_idx: dict[str, list[int]] = {}
        for (doc_id, idx_name), _ in top_cands:
            cands_by_idx.setdefault(idx_name, []).append(doc_id)
        for idx_name, doc_ids in cands_by_idx.items():
            li = self.get(idx_name)
            with open(li._base_path_abs, 'rb') as bf:
                for d in doc_ids:
                    bf.seek(d * row_bytes + 4)
                    vec = np.frombuffer(bf.read(dim * 4), dtype=np.float32)
                    if len(vec) != dim:
                        continue  # missing row (shouldn't happen)
                    d2 = float(((vec - qvec) ** 2).sum())
                    scored_l2.append((d2, d, idx_name))
        scored_l2.sort()
        out = [{'index': idx_name, 'doc_id': d, 'l2': d2}
               for d2, d, idx_name in scored_l2[:top_k]]
        return {'results': out, 'matched_indexes': names,
                'per_index_counts': per_idx}
