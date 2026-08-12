"""Python ctypes bindings for libmangrove.so.

Usage:
    from mangrove_ffi import Forest
    f = Forest('/path/to/index', n_trees=200, dim=128, sub_dim=16,
               depth=25, n_docs=100_000_000, gen_version=3)
    ids, votes, n = f.query(qvec, top_n=500, query_depth=0)
    f.close()

Eliminates the ~10ms subprocess overhead of running ./rpforest per query.
Forest handles are not thread-safe (one ring per forest, serial use).
"""
from __future__ import annotations

import ctypes
import os
import numpy as np
from ctypes import c_int, c_int32, c_uint32, c_void_p, c_char_p, c_float, c_uint8, POINTER


def _load_lib() -> ctypes.CDLL:
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, '..', 'libmangrove.so'),
        os.path.join(os.path.dirname(here), 'libmangrove.so'),
        '/usr/local/lib/libmangrove.so',
    ]
    for p in candidates:
        if os.path.exists(p):
            return ctypes.CDLL(p)
    raise OSError(
        f'libmangrove.so not found in: {candidates}. Run `make` first.'
    )


_lib = _load_lib()

# void* mg_forest_open(const char*, int, int, int, int, int, int);
_lib.mg_forest_open.argtypes = [c_char_p, c_int, c_int, c_int, c_int, c_int]
_lib.mg_forest_open.restype  = c_void_p

_lib.mg_forest_close.argtypes = [c_void_p]
_lib.mg_forest_close.restype  = None

# int mg_forest_query(void*, const float*, int, int,
#                     const uint8_t*, int, int*, int*);
_lib.mg_forest_query.argtypes = [
    c_void_p, POINTER(c_float), c_int, c_int,
    POINTER(c_uint8), c_int,
    POINTER(c_int32), POINTER(c_int32),
]
_lib.mg_forest_query.restype = c_int

# Convenience: pass a raw int32 doc_id array as filter (no CH state needed).
_lib.mg_forest_query_ids.argtypes = [
    c_void_p, POINTER(c_float), c_int, c_int,
    POINTER(c_int32), c_int,
    POINTER(c_int32), POINTER(c_int32),
]
_lib.mg_forest_query_ids.restype = c_int

_lib.mg_forest_query_multi.argtypes = [
    POINTER(c_void_p), c_int, POINTER(c_float), c_int, c_int,
    POINTER(c_uint8), c_int,
    POINTER(c_int32), POINTER(c_int32),
]
_lib.mg_forest_query_multi.restype = c_int

_lib.mg_last_n_distinct.argtypes = []
_lib.mg_last_n_distinct.restype  = c_int

# Query deadline (thread-local, ns absolute CLOCK_MONOTONIC).
_lib.mg_now_ns.argtypes                = []
_lib.mg_now_ns.restype                 = ctypes.c_int64
_lib.mg_set_query_deadline_ns.argtypes = [ctypes.c_int64]
_lib.mg_set_query_deadline_ns.restype  = None
_lib.mg_last_query_partial.argtypes    = []
_lib.mg_last_query_partial.restype     = c_int


def set_query_deadline_ms(ms: int) -> None:
    """Arm a per-query wall-clock deadline (ms from now, thread-local).
       Pass 0 to disable. When the deadline fires mid-query, the C code
       returns the partial top-N built so far and last_query_partial()
       reports True. Effective abort granularity : Phase 1 / Phase 2
       boundaries + merge inner loop. In-flight io_uring reads are NOT
       cancelled mid-batch (limitation, see project_query_deadline_gap)."""
    if ms <= 0:
        _lib.mg_set_query_deadline_ns(0)
        return
    _lib.mg_set_query_deadline_ns(_lib.mg_now_ns() + int(ms) * 1_000_000)


def last_query_partial() -> bool:
    """True iff the last collect returned partial results due to deadline."""
    return _lib.mg_last_query_partial() != 0

for getter in ('mg_n_trees', 'mg_dim', 'mg_sub_dim', 'mg_depth', 'mg_n_docs',
               'mg_srt_version'):
    fn = getattr(_lib, getter)
    fn.argtypes = [c_void_p]
    fn.restype  = c_int

_lib.mg_set_gen_version.argtypes = [c_int]
_lib.mg_set_gen_version.restype  = None

_lib.mg_set_tree_sub.argtypes = [c_int]
_lib.mg_set_tree_sub.restype  = None
_lib.mg_get_tree_sub.argtypes = []
_lib.mg_get_tree_sub.restype  = c_int

_lib.mg_set_tree_sub_groups.argtypes = [c_int]
_lib.mg_set_tree_sub_groups.restype  = None
_lib.mg_get_tree_sub_groups.argtypes = []
_lib.mg_get_tree_sub_groups.restype  = c_int

_lib.mg_set_node_perm.argtypes = [c_int]
_lib.mg_set_node_perm.restype  = None
_lib.mg_get_node_perm.argtypes = []
_lib.mg_get_node_perm.restype  = c_int

_lib.mg_set_max_distinct.argtypes = [c_int]
_lib.mg_set_max_distinct.restype  = None
_lib.mg_get_max_distinct.argtypes = []
_lib.mg_get_max_distinct.restype  = c_int

_lib.mg_set_max_leaf_bytes.argtypes = [c_uint32]
_lib.mg_set_max_leaf_bytes.restype  = None

def set_max_leaf_bytes(n):
    """Skip leaves whose posting list exceeds `n` bytes (SRT3) at query
       time. Bounds p99 by dropping degenerate dense leaves. 0 = no cap.
       Recall trade : a query that routes into a capped leaf loses that
       tree's vote for the doc, but other trees compensate."""
    _lib.mg_set_max_leaf_bytes(int(n))

_lib.mg_set_max_stable_rejects.argtypes = [c_int]
_lib.mg_set_max_stable_rejects.restype  = None
_lib.mg_get_max_stable_rejects.argtypes = []
_lib.mg_get_max_stable_rejects.restype  = c_int

# Privacy : arm thread-local override so the next query() skips traversal
# and uses these caller-supplied per-tree leaf_ids instead.
_lib.mg_set_external_leaves.argtypes   = [POINTER(c_int32), c_int]
_lib.mg_set_external_leaves.restype    = None

_lib.mg_probe_leaves.argtypes = [c_void_p, POINTER(c_float), c_int, c_int, c_int,
                                 POINTER(c_int32)]
_lib.mg_probe_leaves.restype  = c_int32

# Fused multi-probe query: probe-leaf compute + single-pass collect in one C call.
# Signature : (forest, qvec, n_probes, probe_span, probe_depth, top_n,
#              allowed_state, allowed_state_len, out_ids, out_votes)
_lib.mg_query_probes.argtypes = [c_void_p, POINTER(c_float),
                                 c_int, c_int, c_int, c_int,
                                 POINTER(c_uint8), c_int,
                                 POINTER(c_int32), POINTER(c_int32)]
_lib.mg_query_probes.restype  = c_int32

# Path-rank query: cross-tree top-K paths by margin, then vote dedup + cap.
# Signature : (forest, qvec, n_probes, top_paths, top_n, query_depth,
#              allowed_state, allowed_state_len, out_ids, out_votes)
_lib.mg_query_pathrank.argtypes = [c_void_p, POINTER(c_float),
                                   c_int, c_int, c_int, c_int,
                                   POINTER(c_uint8), c_int,
                                   POINTER(c_int32), POINTER(c_int32)]
_lib.mg_query_pathrank.restype  = c_int32

# LSM : query pathrank multi-segment (shared traversal, single FFI round-trip)
_lib.mg_query_pathrank_multi.argtypes = [POINTER(c_void_p), c_int,
                                         POINTER(c_float),
                                         c_int, c_int, c_int, c_int,
                                         POINTER(c_uint8), c_int,
                                         POINTER(c_int32), POINTER(c_int32)]
_lib.mg_query_pathrank_multi.restype  = c_int32


def query_pathrank_multi(forests, qvec, n_probes, top_paths, top_n=4000,
                         query_depth=0, allowed_state=None):
    """LSM query : plusieurs segments partageant les seeds trees.
       Traversal partagée + collect par segment + merge final = 1 seul FFI call."""
    import numpy as _np
    q = _np.ascontiguousarray(qvec, dtype=_np.float32)
    ids   = _np.empty(top_n, dtype=_np.int32)
    votes = _np.empty(top_n, dtype=_np.int32)
    n_handles = len(forests)
    handles = (c_void_p * n_handles)(*[c_void_p(f._h) for f in forests])
    as_len = 0 if allowed_state is None else len(allowed_state)
    as_buf = (c_uint8 * as_len).from_buffer_copy(allowed_state) if as_len > 0 else None
    n = _lib.mg_query_pathrank_multi(
        handles, n_handles, q.ctypes.data_as(POINTER(c_float)),
        int(n_probes), int(top_paths), int(top_n), int(query_depth),
        as_buf, as_len,
        ids.ctypes.data_as(POINTER(c_int32)),
        votes.ctypes.data_as(POINTER(c_int32)))
    if n < 0:
        raise RuntimeError('mg_query_pathrank_multi failed')
    return ids[:n], votes[:n], n
_lib.mg_clear_external_leaves.argtypes = []
_lib.mg_clear_external_leaves.restype  = None

# Shared scratch pool : drops RAM ~10× on multi-segment clusters.
_lib.mg_set_shared_scratch_pool.argtypes = [c_int]
_lib.mg_set_shared_scratch_pool.restype  = None
_lib.mg_shared_scratch_bytes.argtypes    = []
_lib.mg_shared_scratch_bytes.restype     = ctypes.c_size_t


def set_shared_scratch_pool(enable: bool) -> None:
    """Enable the thread-local shared scratch pool. All Forests queried
       in this thread will share their per-query bytes_buf and docs_buf
       (the largest buffers, ~hundreds of MB per forest at SIFT 1B scale).
       Queries are serial inside LiveIndex anyway, so sharing is safe."""
    _lib.mg_set_shared_scratch_pool(1 if enable else 0)


def shared_scratch_bytes() -> int:
    return int(_lib.mg_shared_scratch_bytes())


def set_external_leaves(leaves):
    """Arm thread-local override : next forest.query() uses these leaves
       instead of traversing. Pass None to clear and revert to normal."""
    import numpy as np
    if leaves is None:
        _lib.mg_clear_external_leaves()
    else:
        arr = np.asarray(leaves, dtype=np.int32)
        if not arr.flags['C_CONTIGUOUS']:
            arr = np.ascontiguousarray(arr)
        _lib.mg_set_external_leaves(arr.ctypes.data_as(POINTER(c_int32)),
                                    int(arr.size))
        # Keep a reference so GC doesn't free the buffer between set + query
        global _external_leaves_keepalive
        _external_leaves_keepalive = arr


_external_leaves_keepalive = None

# Test-only exports
_lib.mg_varbyte_encode.argtypes = [ctypes.c_uint, POINTER(c_uint8)]
_lib.mg_varbyte_encode.restype  = c_int
_lib.mg_varbyte_decode.argtypes = [POINTER(c_uint8), POINTER(ctypes.c_size_t)]
_lib.mg_varbyte_decode.restype  = ctypes.c_uint
_lib.mg_traverse_sub.argtypes   = [POINTER(c_float), c_int, c_int, c_int, c_int]
_lib.mg_traverse_sub.restype    = c_int
_lib.mg_traverse_sub_continue.argtypes = [POINTER(c_float), c_int, c_int,
                                          c_int, c_int, c_int, c_int]
_lib.mg_traverse_sub_continue.restype  = c_int

# int mg_trace_margins(const float* qvec, int dim, int sub_dim,
#                      int depth, int tree_idx, float* out_margins)
_lib.mg_trace_margins.argtypes = [POINTER(c_float), c_int, c_int,
                                  c_int, c_int, POINTER(c_float)]
_lib.mg_trace_margins.restype  = c_int


def trace_margins(qvec, dim: int, sub_dim: int, depth: int, tree_idx: int):
    """Return (leaf_id, margins[depth]) for query qvec on tree tree_idx.

    Margins are |c1 - c0| at each path level (CPU-only, no disk I/O)."""
    if qvec.dtype != np.float32:
        qvec = qvec.astype(np.float32, copy=False)
    if not qvec.flags['C_CONTIGUOUS']:
        qvec = np.ascontiguousarray(qvec)
    out = np.empty(depth, dtype=np.float32)
    leaf = _lib.mg_trace_margins(
        qvec.ctypes.data_as(POINTER(c_float)),
        dim, sub_dim, depth, tree_idx,
        out.ctypes.data_as(POINTER(c_float)),
    )
    return leaf, out


# int mg_query_probes_scored(qvec, dim, sub_dim, depth, tree_idx, n_probes,
#                             out_leaves[NP+1], out_scores[NP+1])
_lib.mg_query_probes_scored.argtypes = [
    POINTER(c_float), c_int, c_int, c_int, c_int, c_int,
    POINTER(c_int32), POINTER(c_float),
]
_lib.mg_query_probes_scored.restype = c_int


def query_probes_scored(qvec, dim: int, sub_dim: int, depth: int,
                         tree_idx: int, n_probes: int):
    """Return (leaves, scores) arrays of length 1+n_probes for the given
    (query, tree). Score = min margin along path (larger = clearer routing)."""
    if qvec.dtype != np.float32:
        qvec = qvec.astype(np.float32, copy=False)
    if not qvec.flags['C_CONTIGUOUS']:
        qvec = np.ascontiguousarray(qvec)
    L = n_probes + 1
    leaves = np.empty(L, dtype=np.int32)
    scores = np.empty(L, dtype=np.float32)
    n = _lib.mg_query_probes_scored(
        qvec.ctypes.data_as(POINTER(c_float)),
        dim, sub_dim, depth, tree_idx, n_probes,
        leaves.ctypes.data_as(POINTER(c_int32)),
        scores.ctypes.data_as(POINTER(c_float)),
    )
    return leaves[:n], scores[:n]


# int mg_leaf_docs(void*, int tree_idx, int leaf_id, int* out, int max_n)
_lib.mg_leaf_docs.argtypes = [c_void_p, c_int, c_int, POINTER(c_int32), c_int]
_lib.mg_leaf_docs.restype  = c_int


def leaf_docs(forest_handle, tree_idx: int, leaf_id: int, max_n: int = 65536):
    """Return the doc_ids stored in (tree_idx, leaf_id). 1 disk read per
    call — used for per-tree-quality observations. Auto-retries with a
    larger buffer if the leaf is bigger than max_n (shallow-depth safe). """
    while True:
        out = np.empty(max_n, dtype=np.int32)
        n = _lib.mg_leaf_docs(
            forest_handle, tree_idx, leaf_id,
            out.ctypes.data_as(POINTER(c_int32)), max_n,
        )
        if n < max_n:
            return out[:n] if n > 0 else np.array([], dtype=np.int32)
        max_n *= 4


def varbyte_encode(v: int) -> bytes:
    buf = (c_uint8 * 5)()
    n = _lib.mg_varbyte_encode(v, buf)
    return bytes(buf[:n])


def varbyte_decode(data: bytes) -> tuple[int, int]:
    arr = (c_uint8 * len(data)).from_buffer_copy(data)
    pos = ctypes.c_size_t(0)
    val = _lib.mg_varbyte_decode(arr, ctypes.byref(pos))
    return val, pos.value


# ---- Live-ingest medians ----
# int  mg_live_medians_load(const char* path)   → med_depth or -1
# void mg_live_medians_clear(void)
# int  mg_live_medians_depth(void)              → 0 when not loaded
_lib.mg_live_medians_load.argtypes  = [c_char_p]
_lib.mg_live_medians_load.restype   = c_int
_lib.mg_live_medians_clear.argtypes = []
_lib.mg_live_medians_clear.restype  = None
_lib.mg_live_medians_depth.argtypes = []
_lib.mg_live_medians_depth.restype  = c_int


def load_live_medians(index_dir: str) -> int:
    """Arm the stateless traversal helpers (traverse_sub & co — the live
    insert routing path) with the index's frozen median thresholds.

    On a median-built index this MUST be called before routing inserts,
    otherwise docs are routed sign-split and land in leaves the query
    never visits (stranded). Returns med_depth, or 0 if the index has no
    medians.bin (classic index — sign splits, nothing to arm)."""
    path = os.path.join(index_dir, 'medians.bin')
    if not os.path.exists(path):
        return 0
    md = _lib.mg_live_medians_load(path.encode())
    if md < 0:
        raise RuntimeError(f'medians.bin unreadable: {path}')
    return md


def clear_live_medians() -> None:
    _lib.mg_live_medians_clear()


def live_medians_depth() -> int:
    return _lib.mg_live_medians_depth()


# ---- Métadonnées natives (meta_store : frozen views mmap + deltas WAL) ----
_lib.mg_meta_open.argtypes  = [c_char_p]
_lib.mg_meta_open.restype   = c_void_p
_lib.mg_meta_close.argtypes = [c_void_p]
_lib.mg_meta_close.restype  = None
_lib.mg_meta_add_batch.argtypes = [c_void_p, c_char_p, POINTER(c_uint32), c_int]
_lib.mg_meta_add_batch.restype  = c_int
_lib.mg_meta_compact.argtypes = [c_void_p]
_lib.mg_meta_compact.restype  = c_int
_lib.mg_meta_filter.argtypes = [c_void_p, POINTER(c_char_p), POINTER(c_int), c_int]
_lib.mg_meta_filter.restype  = c_void_p
_lib.mg_meta_filter_free.argtypes = [c_void_p]
_lib.mg_meta_filter_free.restype  = None
_lib.mg_meta_filter_card.argtypes = [c_void_p]
_lib.mg_meta_filter_card.restype  = ctypes.c_int64
_lib.mg_meta_n_keys.argtypes = [c_void_p]
_lib.mg_meta_n_keys.restype  = c_int
_lib.mg_meta_delta_docs.argtypes = [c_void_p]
_lib.mg_meta_delta_docs.restype  = ctypes.c_int64
_lib.mg_meta_list_keys.argtypes = [c_void_p, ctypes.c_char_p, c_int]
_lib.mg_meta_list_keys.restype  = c_int
# int mg_query_pathrank_bm(h, qvec, np, tp, top_n, qd, bitmap, out_ids, out_votes)
_lib.mg_query_pathrank_bm.argtypes = [c_void_p, POINTER(c_float), c_int, c_int,
                                      c_int, c_int, c_void_p,
                                      POINTER(c_int32), POINTER(c_int32)]
_lib.mg_query_pathrank_bm.restype = c_int


class MetaStore:
    """Métadonnées natives : un bitmap roaring par (champ=valeur), gelés
    mmap zéro-copie + deltas WAL. Le filtre s'évalue in-process et se passe
    DIRECTEMENT à la query (query_pathrank_meta) : zéro sérialisation,
    zéro réseau — contrairement au chemin ClickHouse ch_state."""

    def __init__(self, directory: str):
        self._h = _lib.mg_meta_open(directory.encode())
        if not self._h:
            raise RuntimeError(f'mg_meta_open failed: {directory}')

    def add(self, field: str, value: str, doc_ids) -> None:
        arr = np.ascontiguousarray(np.asarray(doc_ids, dtype=np.uint32))
        key = f'{field}={value}'.encode()
        rc = _lib.mg_meta_add_batch(
            self._h, key, arr.ctypes.data_as(POINTER(c_uint32)), len(arr))
        if rc != 0:
            raise RuntimeError(f'meta add({key}) failed')

    def add_keys(self, keys: list[str], doc_ids) -> None:
        """Ajoute les mêmes doc_ids sous plusieurs clés DÉJÀ ENCODÉES
        (sortie de metatypes.encode_meta — inclut les clés de présence)."""
        arr = np.ascontiguousarray(np.asarray(doc_ids, dtype=np.uint32))
        for k in keys:
            rc = _lib.mg_meta_add_batch(
                self._h, k.encode(),
                arr.ctypes.data_as(POINTER(c_uint32)), len(arr))
            if rc != 0:
                raise RuntimeError(f'meta add_keys({k}) failed')

    def compact(self) -> int:
        rc = _lib.mg_meta_compact(self._h)
        if rc < 0:
            raise RuntimeError('meta compact failed')
        return rc

    def filter(self, where: dict):
        """where = {champ: valeur | [valeurs]} — groupes ANDés entre champs,
        valeurs ORées dans un champ. Retourne un handle bitmap (à libérer
        via filter_free après la/les queries)."""
        keys, lens = [], []
        for field, values in where.items():
            vals = values if isinstance(values, (list, tuple)) else [values]
            for v in vals:
                keys.append(f'{field}={v}'.encode())
            lens.append(len(vals))
        karr = (c_char_p * len(keys))(*keys)
        larr = (c_int * len(lens))(*lens)
        bmp = _lib.mg_meta_filter(self._h, karr, larr, len(lens))
        if not bmp:
            raise RuntimeError('meta filter failed')
        return bmp

    def filter_keys(self, flat_keys: list[str], group_lens: list[int]):
        """Variante bas-niveau : clés déjà compilées (metatypes.compile_where)."""
        karr = (c_char_p * len(flat_keys))(*[k.encode() for k in flat_keys])
        larr = (c_int * len(group_lens))(*group_lens)
        bmp = _lib.mg_meta_filter(self._h, karr, larr, len(group_lens))
        if not bmp:
            raise RuntimeError('meta filter_keys failed')
        return bmp

    @staticmethod
    def filter_free(bmp) -> None:
        _lib.mg_meta_filter_free(bmp)

    @staticmethod
    def filter_card(bmp) -> int:
        return _lib.mg_meta_filter_card(bmp)

    @property
    def n_keys(self) -> int:
        return _lib.mg_meta_n_keys(self._h)

    def keys(self) -> list[str]:
        """Dictionnaire des clés connues (gelées + deltas). Sert à
        l'expansion regex/plages côté SDK."""
        cap = 1 << 20
        buf = ctypes.create_string_buffer(cap)
        n = _lib.mg_meta_list_keys(self._h, buf, cap)
        if n < 0:
            raise RuntimeError('meta list_keys failed')
        raw = buf.value.decode()
        return raw.split('\n') if raw else []

    @property
    def delta_docs(self) -> int:
        return _lib.mg_meta_delta_docs(self._h)

    def close(self) -> None:
        if self._h:
            _lib.mg_meta_close(self._h)
            self._h = None


# int mg_traverse_batch(vecs, n_vecs, dim, sub_dim, depth, n_trees, out)
_lib.mg_traverse_batch.argtypes = [POINTER(c_float), c_int, c_int, c_int,
                                   c_int, c_int, POINTER(c_int32)]
_lib.mg_traverse_batch.restype = c_int
# int mg_hot_append_block(hot, leaves, doc_ids, n_vecs, n_trees)
_lib.mg_hot_append_block.argtypes = [c_void_p, POINTER(c_int32),
                                     POINTER(c_uint32), c_int, c_int]
_lib.mg_hot_append_block.restype = c_int


def traverse_batch(vecs: np.ndarray, sub_dim: int, depth: int,
                   n_trees: int) -> np.ndarray:
    """Leaf ids d'un BLOC de vecteurs (n, dim) sur les n_trees arbres en un
    seul appel FFI, OMP sur les vecteurs (gros grain — c'est le chemin
    d'ingest rapide ; traverse_all_trees parallélise sur les arbres, trop
    fin à faible depth). Retour : int32 (n, n_trees)."""
    if vecs.dtype != np.float32:
        vecs = vecs.astype(np.float32)
    if not vecs.flags['C_CONTIGUOUS']:
        vecs = np.ascontiguousarray(vecs)
    n, dim = vecs.shape
    out = np.empty((n, n_trees), dtype=np.int32)
    rc = _lib.mg_traverse_batch(
        vecs.ctypes.data_as(POINTER(c_float)), n, dim, sub_dim,
        depth, n_trees, out.ctypes.data_as(POINTER(c_int32)))
    if rc != 0:
        raise ValueError('mg_traverse_batch failed')
    return out


def hot_append_block(hot_handle, leaves: np.ndarray,
                     doc_ids: np.ndarray) -> None:
    """Append d'un bloc routé (sortie de traverse_batch) dans le HOT,
    parallèle par arbre côté C."""
    n, n_trees = leaves.shape
    if leaves.dtype != np.int32:
        leaves = leaves.astype(np.int32)
    if doc_ids.dtype != np.uint32:
        doc_ids = doc_ids.astype(np.uint32)
    rc = _lib.mg_hot_append_block(
        hot_handle, leaves.ctypes.data_as(POINTER(c_int32)),
        doc_ids.ctypes.data_as(POINTER(c_uint32)), n, n_trees)
    if rc != 0:
        raise RuntimeError('mg_hot_append_block failed')


# int mg_traverse_all_trees(qvec, dim, sub_dim, depth, n_trees, out_leaves)
_lib.mg_traverse_all_trees.argtypes = [POINTER(c_float), c_int, c_int,
                                       c_int, c_int, POINTER(c_int32)]
_lib.mg_traverse_all_trees.restype = c_int


def traverse_all_trees(qvec: np.ndarray, sub_dim: int, depth: int,
                       n_trees: int) -> np.ndarray:
    """Leaf ids d'UN vecteur sur les n_trees arbres en un seul appel FFI
    (médianes live appliquées si chargées). C'est le chemin de routage
    d'insert batché — ~100× moins d'overhead FFI que n_trees appels."""
    if qvec.dtype != np.float32:
        qvec = qvec.astype(np.float32)
    if not qvec.flags['C_CONTIGUOUS']:
        qvec = np.ascontiguousarray(qvec)
    out = np.empty(n_trees, dtype=np.int32)
    rc = _lib.mg_traverse_all_trees(
        qvec.ctypes.data_as(POINTER(c_float)), len(qvec), sub_dim,
        depth, n_trees, out.ctypes.data_as(POINTER(c_int32)))
    if rc != 0:
        raise ValueError('mg_traverse_all_trees failed')
    return out


def traverse_sub(qvec: np.ndarray, sub_dim: int, depth: int, tree_idx: int) -> int:
    if qvec.dtype != np.float32: qvec = qvec.astype(np.float32)
    if not qvec.flags['C_CONTIGUOUS']: qvec = np.ascontiguousarray(qvec)
    return _lib.mg_traverse_sub(qvec.ctypes.data_as(POINTER(c_float)),
                                 len(qvec), sub_dim, depth, tree_idx)


def traverse_sub_continue(qvec: np.ndarray, sub_dim: int, start_depth: int,
                          n_extra: int, start_node: int, tree_idx: int) -> int:
    if qvec.dtype != np.float32: qvec = qvec.astype(np.float32)
    if not qvec.flags['C_CONTIGUOUS']: qvec = np.ascontiguousarray(qvec)
    return _lib.mg_traverse_sub_continue(qvec.ctypes.data_as(POINTER(c_float)),
                                          len(qvec), sub_dim,
                                          start_depth, n_extra,
                                          start_node, tree_idx)

_lib.mg_verify_srt.argtypes = [c_char_p]
_lib.mg_verify_srt.restype  = c_int

# int mg_rerank_l2(void*, const char*, const float*, const int*, int, int, int*)
_lib.mg_rerank_l2.argtypes = [
    c_void_p, c_char_p, POINTER(c_float),
    POINTER(c_int32), c_int, c_int, POINTER(c_int32),
]
_lib.mg_rerank_l2.restype = c_int

# int mg_rerank_tq(void*, const char*, const char*, const float*,
#                  const int*, int, int, int, int*)
_lib.mg_rerank_tq.argtypes = [
    c_void_p, c_char_p, c_char_p, POINTER(c_float),
    POINTER(c_int32), c_int, c_int, c_int, POINTER(c_int32),
]
_lib.mg_rerank_tq.restype = c_int

# int mg_rerank_tq1(void*, const char*, const char*, const float*,
#                   const int*, int, int, int, int*)
_lib.mg_rerank_tq1.argtypes = [
    c_void_p, c_char_p, c_char_p, POINTER(c_float),
    POINTER(c_int32), c_int, c_int, c_int, POINTER(c_int32),
]
_lib.mg_rerank_tq1.restype = c_int

# int mg_auto_qd_v2(void*, const float*, int, double, int, int)
_lib.mg_auto_qd_v2.argtypes = [
    c_void_p, POINTER(c_float), c_int, ctypes.c_double, c_int, c_int,
]
_lib.mg_auto_qd_v2.restype = c_int

# Tombstones (mutations API).
for fn_name in ('mg_tombstone_add', 'mg_tombstone_remove'):
    fn = getattr(_lib, fn_name)
    fn.argtypes = [c_void_p, ctypes.c_uint]
    fn.restype  = c_int

for fn_name in ('mg_tombstones_flush', 'mg_tombstones_count'):
    fn = getattr(_lib, fn_name)
    fn.argtypes = [c_void_p]
    fn.restype  = c_int


def set_gen_version(v: int) -> None:
    """Call before opening a Forest if the index was built with gen v3 (etc)."""
    _lib.mg_set_gen_version(v)


def set_tree_sub(k: int) -> None:
    """Per-tree input subspace. MUST match meta.txt's `tree_sub` for the index
    being queried (0 = disabled / legacy) or recall collapses."""
    _lib.mg_set_tree_sub(int(k))


def set_tree_sub_groups(g: int) -> None:
    """# distinct per-tree subspaces (0 = one per tree). MUST match meta.txt's
    `tree_sub_groups` for the index being queried or recall collapses."""
    _lib.mg_set_tree_sub_groups(int(g))


def set_node_perm(on: int) -> None:
    """Path-permuted dim selection. MUST match meta.txt's `node_perm` for the
    index being queried or recall collapses. Forest.__init__ reads meta.txt and
    sets this automatically; manual setter is for in-process builds only."""
    _lib.mg_set_node_perm(int(on))


def set_max_distinct(n: int) -> None:
    """Cap the K-way merge to n distinct candidates per query.
    0 = no cap (default). Hard-bounds p99 on heavy queries at the cost
    of recall on the tail of candidates beyond n.                      """
    _lib.mg_set_max_distinct(n)


def set_max_stable_rejects(n: int) -> None:
    """Exit the K-way merge after n consecutive non-inserting pushes
    (= heap is stable). Preserves recall much better than max_distinct
    because we only exit when the heap stops accepting new candidates.
    0 = no cap (default).                                              """
    _lib.mg_set_max_stable_rejects(n)


def verify_srt(path: str) -> int:
    """Verify a single .srt's xxhash. Returns 1 OK / 0 bad / -1 IO error."""
    return _lib.mg_verify_srt(path.encode())


class Forest:
    """Wraps a single mangrove forest with the persistent io_uring ring."""

    def __init__(self, index_dir: str, n_trees: int, dim: int,
                 sub_dim: int, depth: int, n_docs: int,
                 gen_version: int = 3) -> None:
        set_gen_version(gen_version)
        self._h = _lib.mg_forest_open(
            index_dir.encode(), n_trees, dim, sub_dim, depth, n_docs)
        if not self._h:
            raise RuntimeError(f'forest_open failed for {index_dir}')
        self.n_trees = n_trees
        self.dim     = dim
        self.sub_dim = sub_dim
        self.depth   = depth
        self.n_docs  = n_docs
        self.srt_version = _lib.mg_srt_version(self._h)

    def probe_leaves(self, qvec, n_probes, probe_depth=0, probe_span=0):
        """Multi-probe leaf ids (computed in C, trusted routing).
           probe_depth: 0 = native build depth; lower → traverse/flip to that
           depth (leaves are bigger; pair with query(query_depth=probe_depth)).
           probe_span: # of DEEPEST levels eligible to flip (0 = any level,
           legacy). Deep-only probing keeps each probe a genuine neighbour leaf
           instead of re-descending a huge shallow subtree — essential for high
           n_probes (else candidates explode and recall collapses).
           Returns int32 array shape (n_probes+1, n_trees); row p = the p-th
           probe leaf per tree, in (node - leaf_base) form for
           set_external_leaves(). Run one query() pass per row and merge votes."""
        import numpy as _np
        q = _np.ascontiguousarray(qvec, dtype=_np.float32)
        out = _np.empty((n_probes + 1) * self.n_trees, dtype=_np.int32)
        nt = _lib.mg_probe_leaves(self._h, q.ctypes.data_as(POINTER(c_float)),
                                  int(n_probes), int(probe_depth), int(probe_span),
                                  out.ctypes.data_as(POINTER(c_int32)))
        if nt != self.n_trees:
            raise RuntimeError('mg_probe_leaves failed')
        return out.reshape(n_probes + 1, self.n_trees)

    def query_probes(self, qvec, n_probes, top_n=500, probe_span=0,
                     probe_depth=0, allowed_state=None):
        """Multi-probe query in ONE C call. Two regimes by `probe_depth` :

           - probe_depth = 0 (default) or = build_depth → FUSED native path :
             single-pass K-way merge across all probes, vote = # distinct trees
             containing the doc (≤ n_trees). Best for low-dim corpora (SIFT-like).

           - 0 < probe_depth < build_depth → MULTI-PASS at qd path :
             n_probes+1 internal calls to forest_collect_topn at probe_depth,
             vote-accumulated in C. Subtree expansion is handled by collect_topn.
             Vote semantics : sum across probe-sets of per-set tree-counts (a
             tree can vote up to n_probes+1× for the same doc → max vote
             = (n_probes+1) × n_trees). Best for high-dim corpora where qd↓
             is the dominant recall lever (dim ≥ 384).

           Returns (ids, votes, n). Releases the GIL → parallelizable across
           segments with real threads."""
        import numpy as _np
        q = _np.ascontiguousarray(qvec, dtype=_np.float32)
        ids   = _np.empty(top_n, dtype=_np.int32)
        votes = _np.empty(top_n, dtype=_np.int32)
        as_len = 0 if allowed_state is None else len(allowed_state)
        as_buf = (c_uint8 * as_len).from_buffer_copy(allowed_state) if as_len > 0 else None
        n = _lib.mg_query_probes(
            self._h, q.ctypes.data_as(POINTER(c_float)),
            int(n_probes), int(probe_span), int(probe_depth), int(top_n),
            as_buf, as_len,
            ids.ctypes.data_as(POINTER(c_int32)),
            votes.ctypes.data_as(POINTER(c_int32)))
        if n < 0:
            raise RuntimeError('mg_query_probes failed')
        return ids[:n], votes[:n], n

    def query_pathrank(self, qvec, n_probes: int, top_paths: int, top_n: int = 4000,
                       query_depth: int = 0, allowed_state=None):
        """Cross-tree path-rank query : traverse all trees × (n_probes+1),
           keep the globally best `top_paths` by min-margin score, then
           vote-dedup + top_n cap in one C call.

           query_depth : 0 (default) or ≥ build_depth → native path.
                        0 < qd < build_depth → traverse only to level qd,
                        each path expands to 2^(build_depth − qd) leaves
                        via subtree-expansion in collect_topn_probes."""
        import numpy as _np
        q = _np.ascontiguousarray(qvec, dtype=_np.float32)
        ids   = _np.empty(top_n, dtype=_np.int32)
        votes = _np.empty(top_n, dtype=_np.int32)
        as_len = 0 if allowed_state is None else len(allowed_state)
        as_buf = (c_uint8 * as_len).from_buffer_copy(allowed_state) if as_len > 0 else None
        n = _lib.mg_query_pathrank(
            self._h, q.ctypes.data_as(POINTER(c_float)),
            int(n_probes), int(top_paths), int(top_n), int(query_depth),
            as_buf, as_len,
            ids.ctypes.data_as(POINTER(c_int32)),
            votes.ctypes.data_as(POINTER(c_int32)))
        if n < 0:
            raise RuntimeError('mg_query_pathrank failed')
        return ids[:n], votes[:n], n

    def query_pathrank_with_fallback(self, qvec, n_probes: int, top_paths: int,
                                     top_n: int = 4000, query_depth: int = 0,
                                     allowed_state=None,
                                     total_deadline_ms: int = 1800,
                                     mlb_ladder: tuple = (400_000, 300_000, 200_000, 150_000),
                                     min_remaining_ms: int = 200):
        """Graceful-degradation query: on partial deadline hit, retry with a
           smaller max_leaf_bytes cap (mlb ladder) within the same total wall
           budget. Preserves recall for pathological fat-tail queries by paying
           multiple attempts with progressively tighter working sets.

           Amortized cost : 1 call on healthy queries (~99 %), 2-3 on fat-tail.

           Returns (ids, votes, n, meta) where meta = {
               'attempts': [(mlb, elapsed_ms, partial), ...],
               'mlb_used': int,       # mlb of the returned result
               'partial': bool,       # True iff final call was partial
           }.
        """
        import time as _time
        t0 = _time.time()
        attempts = []
        best_ids = best_votes = None
        best_n = 0
        best_partial = True
        best_mlb = mlb_ladder[0]
        # Budget par attempt = total // len(ladder). Ainsi la 1ère tentative
        # n'engloutit pas tout le budget wall ; les retries plus courts
        # (mlb réduit → moins d'IO) tiennent aisément dans leur slot.
        per_attempt_ms = max(min_remaining_ms, total_deadline_ms // len(mlb_ladder))
        for mlb in mlb_ladder:
            elapsed_ms = int((_time.time() - t0) * 1000)
            remaining = total_deadline_ms - elapsed_ms
            if remaining < min_remaining_ms and attempts:
                break
            attempt_budget = min(per_attempt_ms, remaining)
            set_max_leaf_bytes(mlb)
            set_query_deadline_ms(attempt_budget)
            ta = _time.time()
            ids, votes, n = self.query_pathrank(qvec, n_probes, top_paths,
                                                top_n, query_depth, allowed_state)
            dt = int((_time.time() - ta) * 1000)
            was_partial = last_query_partial()
            attempts.append((mlb, dt, was_partial))
            # Keep best result: prefer non-partial, then higher n
            if not was_partial:
                best_ids, best_votes, best_n = ids, votes, n
                best_partial = False; best_mlb = mlb
                break
            if n > best_n:
                best_ids, best_votes, best_n = ids, votes, n
                best_mlb = mlb
        set_query_deadline_ms(0)  # disarm
        if best_ids is None:
            best_ids = np.empty(0, dtype=np.int32)
            best_votes = np.empty(0, dtype=np.int32)
        return best_ids, best_votes, best_n, {
            'attempts': attempts, 'mlb_used': best_mlb, 'partial': best_partial,
        }

    def query_pathrank_meta(self, qvec, n_probes: int, top_paths: int,
                            meta_bitmap, top_n: int = 4000,
                            query_depth: int = 0):
        """query_pathrank filtré par un bitmap métadonnées NATIF (handle
        retourné par MetaStore.filter) — même process, zéro sérialisation.
        meta_bitmap=None → pas de filtre."""
        if qvec.dtype != np.float32:
            qvec = qvec.astype(np.float32, copy=False)
        if not qvec.flags['C_CONTIGUOUS']:
            qvec = np.ascontiguousarray(qvec)
        ids   = np.empty(top_n, dtype=np.int32)
        votes = np.empty(top_n, dtype=np.int32)
        n = _lib.mg_query_pathrank_bm(
            self._h, qvec.ctypes.data_as(POINTER(c_float)),
            n_probes, top_paths, top_n, query_depth, meta_bitmap,
            ids.ctypes.data_as(POINTER(c_int32)),
            votes.ctypes.data_as(POINTER(c_int32)))
        if n < 0:
            raise RuntimeError('mg_query_pathrank_bm failed')
        return ids, votes, n

    def query(self, qvec: np.ndarray, top_n: int = 500, query_depth: int = 0,
              allowed_state: bytes | None = None
              ) -> tuple[np.ndarray, np.ndarray, int]:
        """Top-N by votes via K-way merge.

        qvec: dim float32 numpy array. Will be cast/copied if needed.
        top_n: max candidates returned (heap capacity).
        query_depth: 0 (or build_depth) = native depth; lower → broader.
        allowed_state: ClickHouse groupBitmap wire bytes, or None.

        Returns (ids, votes, n) where ids[:n], votes[:n] are valid.
        """
        if qvec.dtype != np.float32:
            qvec = qvec.astype(np.float32, copy=False)
        if not qvec.flags['C_CONTIGUOUS']:
            qvec = np.ascontiguousarray(qvec)
        ids   = np.empty(top_n, dtype=np.int32)
        votes = np.empty(top_n, dtype=np.int32)

        as_ptr = allowed_state
        as_len = 0 if as_ptr is None else len(as_ptr)
        as_buf = (c_uint8 * as_len).from_buffer_copy(as_ptr) if as_len > 0 else None

        n = _lib.mg_forest_query(
            self._h,
            qvec.ctypes.data_as(POINTER(c_float)),
            top_n, query_depth,
            as_buf, as_len,
            ids.ctypes.data_as(POINTER(c_int32)),
            votes.ctypes.data_as(POINTER(c_int32)),
        )
        if n < 0:
            raise RuntimeError('mg_forest_query failed')
        return ids, votes, n

    def query_with_ids(self, qvec: np.ndarray, allowed_ids: np.ndarray,
                       top_n: int = 500, query_depth: int = 0
                       ) -> tuple[np.ndarray, np.ndarray, int]:
        """Like query() but accepts a raw int32 doc_id filter array.

        Builds the roaring bitmap inside the C extension (avoids needing
        pyroaring or a ClickHouse state blob in Python). Convenient for
        synthetic filter tests.
        """
        if qvec.dtype != np.float32:
            qvec = qvec.astype(np.float32, copy=False)
        if not qvec.flags['C_CONTIGUOUS']:
            qvec = np.ascontiguousarray(qvec)
        if allowed_ids.dtype != np.int32:
            allowed_ids = allowed_ids.astype(np.int32, copy=False)
        if not allowed_ids.flags['C_CONTIGUOUS']:
            allowed_ids = np.ascontiguousarray(allowed_ids)
        ids   = np.empty(top_n, dtype=np.int32)
        votes = np.empty(top_n, dtype=np.int32)
        n = _lib.mg_forest_query_ids(
            self._h,
            qvec.ctypes.data_as(POINTER(c_float)),
            top_n, query_depth,
            allowed_ids.ctypes.data_as(POINTER(c_int32)),
            len(allowed_ids),
            ids.ctypes.data_as(POINTER(c_int32)),
            votes.ctypes.data_as(POINTER(c_int32)),
        )
        if n < 0:
            raise RuntimeError('mg_forest_query_ids failed')
        return ids, votes, n

    def n_distinct(self) -> int:
        """Number of distinct docs the K-way merge saw on the last query."""
        return _lib.mg_last_n_distinct()

    # --- Mutations: soft delete via tombstones ---
    def tombstone_add(self, doc_id: int) -> None:
        rc = _lib.mg_tombstone_add(self._h, doc_id)
        if rc != 0:
            raise RuntimeError(f'tombstone_add({doc_id}) failed')

    def tombstone_remove(self, doc_id: int) -> None:
        _lib.mg_tombstone_remove(self._h, doc_id)

    def tombstones_flush(self) -> None:
        """Atomic write of the tombstone bitmap to <index_dir>/tombstones.roaring."""
        rc = _lib.mg_tombstones_flush(self._h)
        if rc != 0:
            raise RuntimeError('tombstones_flush failed')

    def tombstones_count(self) -> int:
        return _lib.mg_tombstones_count(self._h)

    def auto_qd_v2(self, probe_qvec: np.ndarray, top_n: int = 4000,
                   target_ratio: float = 0.001,
                   n_pool: int | None = None,
                   filter_card: int = 0) -> int:
        """2-probe auto-calibration of query_depth.

        Runs two queries (at build_depth and build_depth-2) with the given
        probe vector, measures the n_distinct expansion factor, then picks
        the qd that should land target_ratio × n_pool candidates per query.
        For filtered queries, pass filter_card > 0 and the sub-corpus
        compensation kicks in.
        """
        if probe_qvec.dtype != np.float32:
            probe_qvec = probe_qvec.astype(np.float32, copy=False)
        if not probe_qvec.flags['C_CONTIGUOUS']:
            probe_qvec = np.ascontiguousarray(probe_qvec)
        n_pool = n_pool if n_pool is not None else self.n_docs
        qd = _lib.mg_auto_qd_v2(
            self._h, probe_qvec.ctypes.data_as(POINTER(c_float)),
            top_n, target_ratio, n_pool, filter_card,
        )
        if qd < 0:
            raise RuntimeError('mg_auto_qd_v2 failed')
        return qd

    def rerank_l2(self, base_path: str, qvec: np.ndarray,
                  cand_ids: np.ndarray, top_k: int = 10) -> np.ndarray:
        """L2 rerank the forest's candidates against raw base vectors.

        Reads base_path on each call (caches in OS page cache). Returns
        top_k doc_ids sorted by ascending L2.                              """
        if qvec.dtype != np.float32:
            qvec = qvec.astype(np.float32, copy=False)
        if not qvec.flags['C_CONTIGUOUS']:
            qvec = np.ascontiguousarray(qvec)
        if cand_ids.dtype != np.int32:
            cand_ids = cand_ids.astype(np.int32, copy=False)
        if not cand_ids.flags['C_CONTIGUOUS']:
            cand_ids = np.ascontiguousarray(cand_ids)
        out = np.empty(top_k, dtype=np.int32)
        n = _lib.mg_rerank_l2(
            self._h, base_path.encode(),
            qvec.ctypes.data_as(POINTER(c_float)),
            cand_ids.ctypes.data_as(POINTER(c_int32)),
            len(cand_ids), top_k,
            out.ctypes.data_as(POINTER(c_int32)),
        )
        if n < 0:
            raise RuntimeError('mg_rerank_l2 failed')
        return out[:n]

    def rerank_tq(self, tq4_path: str, base_path: str, qvec: np.ndarray,
                  cand_ids: np.ndarray, kprime: int = 100,
                  top_k: int = 10) -> np.ndarray:
        """Two-stage rerank: approx IP on .tq4 codes -> kprime survivors
        -> exact L2 on raw base. Returns top_k doc_ids (ascending L2)."""
        if qvec.dtype != np.float32:
            qvec = qvec.astype(np.float32, copy=False)
        if not qvec.flags['C_CONTIGUOUS']:
            qvec = np.ascontiguousarray(qvec)
        if cand_ids.dtype != np.int32:
            cand_ids = cand_ids.astype(np.int32, copy=False)
        if not cand_ids.flags['C_CONTIGUOUS']:
            cand_ids = np.ascontiguousarray(cand_ids)
        out = np.empty(top_k, dtype=np.int32)
        n = _lib.mg_rerank_tq(
            self._h, tq4_path.encode(), base_path.encode(),
            qvec.ctypes.data_as(POINTER(c_float)),
            cand_ids.ctypes.data_as(POINTER(c_int32)),
            len(cand_ids), kprime, top_k,
            out.ctypes.data_as(POINTER(c_int32)),
        )
        if n < 0:
            raise RuntimeError('mg_rerank_tq failed')
        return out[:n]

    def rerank_tq1(self, tq1_path: str, base_path: str, qvec: np.ndarray,
                   cand_ids: np.ndarray, kprime: int = 1000,
                   top_k: int = 10) -> np.ndarray:
        """Same cascade as rerank_tq but with the 1-bit .tq1 sidecar.
        4x smaller I/O per code; kprime default 1000 (vs 100 for tq4)."""
        if qvec.dtype != np.float32:
            qvec = qvec.astype(np.float32, copy=False)
        if not qvec.flags['C_CONTIGUOUS']:
            qvec = np.ascontiguousarray(qvec)
        if cand_ids.dtype != np.int32:
            cand_ids = cand_ids.astype(np.int32, copy=False)
        if not cand_ids.flags['C_CONTIGUOUS']:
            cand_ids = np.ascontiguousarray(cand_ids)
        out = np.empty(top_k, dtype=np.int32)
        n = _lib.mg_rerank_tq1(
            self._h, tq1_path.encode(), base_path.encode(),
            qvec.ctypes.data_as(POINTER(c_float)),
            cand_ids.ctypes.data_as(POINTER(c_int32)),
            len(cand_ids), kprime, top_k,
            out.ctypes.data_as(POINTER(c_int32)),
        )
        if n < 0:
            raise RuntimeError('mg_rerank_tq1 failed')
        return out[:n]

    def close(self) -> None:
        if self._h:
            _lib.mg_forest_close(self._h)
            self._h = None

    def __enter__(self) -> 'Forest':
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


def query_multi(forests: list[Forest], qvec: np.ndarray,
                top_n: int = 500, query_depth: int = 0,
                allowed_state: bytes | None = None
                ) -> tuple[np.ndarray, np.ndarray, int]:
    """Multi-forest K-way merge (all forests share tree seeds)."""
    if qvec.dtype != np.float32:
        qvec = qvec.astype(np.float32, copy=False)
    if not qvec.flags['C_CONTIGUOUS']:
        qvec = np.ascontiguousarray(qvec)
    handles = (c_void_p * len(forests))(*[f._h for f in forests])
    ids   = np.empty(top_n, dtype=np.int32)
    votes = np.empty(top_n, dtype=np.int32)
    as_len = 0 if allowed_state is None else len(allowed_state)
    as_buf = (c_uint8 * as_len).from_buffer_copy(allowed_state) if as_len > 0 else None
    n = _lib.mg_forest_query_multi(
        handles, len(forests),
        qvec.ctypes.data_as(POINTER(c_float)),
        top_n, query_depth,
        as_buf, as_len,
        ids.ctypes.data_as(POINTER(c_int32)),
        votes.ctypes.data_as(POINTER(c_int32)),
    )
    if n < 0:
        raise RuntimeError('mg_forest_query_multi failed')
    return ids, votes, n
