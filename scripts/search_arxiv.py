#!/usr/bin/env python3
"""
J3 orchestrator : assemble une recherche ANN+metadata pour la base arxiv.

Flow :
  1. Récupère la query vector (--id depuis le corpus, ou --vec file).
  2. Si --where, demande à ClickHouse en natif le `groupBitmapState`
     correspondant (via `cast(..., 'String')`) + sa cardinality.
  3. Décide pre-filter vs post-filter sur le seuil de densité.
     - density ≤ 3 % : écrit le state CH dans un .bin et exec
       `rpforest search --filter_ch ...` (forest skip via bitmap).
     - density >  3 % : forest sans filtre, puis SQL post-filter
       `WHERE internal_id IN (...) AND <conditions>`.
  4. Imprime les top_k arxiv_id + l'origine du choix de stratégie.

Pas de HTTP : tout transite par clickhouse-driver (TCP/9000). Le bitmap
arrive en bytes décodables tel quel par `croaring_io.c`.
"""
import argparse, os, struct, subprocess, sys, tempfile, time

import numpy as np
try:
    from clickhouse_driver import Client
except ImportError:
    Client = None

ROOT      = '/home/chatelet/mangrove-search'
BASE      = f'{ROOT}/datasets/arxiv/arxiv_base.fvecs'
INDEX     = f'{ROOT}/index_arxiv'
RPFOREST  = f'{ROOT}/rpforest'
N_DOCS    = 2_058_751
DIM       = 768
N_TREES   = 1000
DEPTH     = 20
DENSITY_THRESHOLD  = 0.03           # ≤ 3 % → pre-filter discriminant
BRUTEFORCE_MAX_MB  = 50             # filter_card × dim × 4 < this → bruteforce
                                    # (recall 1.0 by construction, beats the
                                    # forest's intrinsic ceiling on sparse
                                    # sub-corpora — see journal_2026-05-18).

_CH_CLIENT = None
def ch_client():
    """Lazy CH client; returns None if unavailable.
       Lets queries proceed without filter when CH is down (graceful degrade). */"""
    global _CH_CLIENT
    if _CH_CLIENT is False:
        return None
    if _CH_CLIENT is None:
        if Client is None:
            sys.stderr.write('[degrade] clickhouse-driver not installed; running without filter\n')
            _CH_CLIENT = False
            return None
        try:
            _CH_CLIENT = Client('127.0.0.1', connect_timeout=2)
            _CH_CLIENT.execute('SELECT 1')
        except Exception as e:
            sys.stderr.write(f'[degrade] ClickHouse unreachable ({e}); running without filter\n')
            _CH_CLIENT = False
            return None
    return _CH_CLIENT

# Compatibility shim for code that still uses `ch.execute(...)`.
class _ChProxy:
    def execute(self, sql, *a, **kw):
        c = ch_client()
        if c is None:
            return []
        return c.execute(sql, *a, **kw)

ch = _ChProxy()

_BASE_MMAP = None
def base_mmap():
    global _BASE_MMAP
    if _BASE_MMAP is None:
        _BASE_MMAP = np.memmap(BASE, dtype='float32', mode='r'
                              ).reshape(N_DOCS, DIM + 1)
    return _BASE_MMAP


def bruteforce_topk(qvec, allowed_ids, top_k):
    """Direct L2 over `allowed_ids` rows of the fvecs. Sorted ASC list."""
    arr = base_mmap()
    sub = arr[allowed_ids, 1:]               # (n, DIM)
    sq  = np.einsum('ij,ij->i', sub, sub)
    d   = sq - 2.0 * (sub @ qvec) + float(np.dot(qvec, qvec))
    k   = min(top_k, len(allowed_ids))
    idx = np.argpartition(d, k - 1)[:k]
    order = idx[np.argsort(d[idx])]
    return [int(allowed_ids[i]) for i in order]


def fetch_allowed_ids(where: str) -> np.ndarray:
    rows = ch.execute(f'SELECT internal_id FROM mangrove.docs WHERE {where}')
    a = np.fromiter((r[0] for r in rows), dtype=np.int64)
    a.sort()
    return a


def read_corpus_vec(internal_id: int) -> np.ndarray:
    """Read one fvecs row from the base file."""
    record_bytes = 4 + DIM * 4
    with open(BASE, 'rb') as f:
        f.seek(internal_id * record_bytes)
        dim_hdr = struct.unpack('<i', f.read(4))[0]
        if dim_hdr != DIM:
            raise RuntimeError(f'bad dim in fvecs: {dim_hdr}')
        return np.frombuffer(f.read(DIM * 4), dtype=np.float32).copy()


def fetch_filter(where: str) -> tuple[bytes, int]:
    """Returns (ch_state_bytes, cardinality) for the WHERE clause.
       Empty WHERE → (b'', N_DOCS). CH down → also (b'', N_DOCS) with warning. */"""
    if not where:
        return b'', N_DOCS
    if ch_client() is None:
        sys.stderr.write('[degrade] CH unavailable; ignoring filter\n')
        return b'', N_DOCS
    sql = (f"SELECT cast(groupBitmapState(internal_id), 'String') AS state, "
           f"count() AS card FROM mangrove.docs WHERE {where}")
    try:
        rows = ch.execute(sql)
        if not rows:
            return b'', N_DOCS
        state, card = rows[0]
        return state, card
    except Exception as e:
        sys.stderr.write(f'[degrade] CH query failed ({e}); ignoring filter\n')
        return b'', N_DOCS


def write_qvec_fvecs(vec: np.ndarray, path: str) -> None:
    with open(path, 'wb') as f:
        f.write(struct.pack('<i', DIM))
        f.write(vec.astype(np.float32).tobytes())


def run_forest(qvecs_path: str, filter_ch_path: str | None,
               top_k: int, top_n: int) -> list[int]:
    cmd = [
        RPFOREST, '--dim', str(DIM),
        'search', qvecs_path, BASE, INDEX,
        str(N_TREES), str(DEPTH), str(N_DOCS),
        str(top_k), str(top_n), '1',
    ]
    if filter_ch_path:
        cmd += ['--filter_ch', filter_ch_path]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           check=True, timeout=30)
    except subprocess.TimeoutExpired:
        sys.stderr.write('[degrade] rpforest timed out (>30s)\n')
        return []
    except subprocess.CalledProcessError as e:
        sys.stderr.write(f'[degrade] rpforest exit {e.returncode}\n{e.stderr}\n')
        return []
    except FileNotFoundError:
        sys.stderr.write(f'[degrade] rpforest binary not found at {RPFOREST}\n')
        return []
    dt = time.time() - t0
    sys.stderr.write(r.stderr)
    sys.stderr.write(f'[forest] subprocess wall {dt*1000:.1f} ms\n')
    line = r.stdout.strip().splitlines()
    if not line:
        sys.stderr.write('[degrade] rpforest returned no output\n')
        return []
    parts = line[0].split(',')
    return [int(x) for x in parts[1:]]


def post_filter(ids: list[int], where: str) -> list[int]:
    """Keep ids that also match `where`, preserving forest ordering."""
    if not where:
        return ids
    placeholders = ','.join(str(i) for i in ids)
    sql = (f"SELECT internal_id FROM mangrove.docs "
           f"WHERE internal_id IN ({placeholders}) AND ({where})")
    kept = {row[0] for row in ch.execute(sql)}
    return [i for i in ids if i in kept]


def lookup_metadata(ids: list[int]) -> list[tuple]:
    if not ids:
        return []
    placeholders = ','.join(str(i) for i in ids)
    rows = ch.execute(
        f"SELECT internal_id, arxiv_id, year, primary_cat, top_cat "
        f"FROM mangrove.docs WHERE internal_id IN ({placeholders})"
    )
    by_id = {r[0]: r for r in rows}
    return [by_id[i] for i in ids if i in by_id]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--id',    type=int,
                    help='internal_id whose corpus vec is used as query')
    ap.add_argument('--where', default='',
                    help='SQL WHERE clause (e.g. "year=2023 AND top_cat=\'cs\'")')
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--top_n', type=int, default=500)
    args = ap.parse_args()

    if args.id is None:
        ap.error('--id INTERNAL_ID required (no embedder wired in yet)')

    qvec = read_corpus_vec(args.id)
    sys.stderr.write(f'query: internal_id={args.id}  dim={DIM}\n')

    state, card = fetch_filter(args.where)
    density = card / N_DOCS if N_DOCS else 0.0
    sys.stderr.write(
        f'filter: where={args.where!r}  card={card}  density={density:.4%}\n')

    # Routing :
    #   - tiny filter (subset fits in RAM cheaply) → bruteforce direct
    #   - selective filter (density ≤ 3 %) → pre-filter forest
    #   - broad filter (density > 3 %) → post-filter forest
    sub_mb = (card * DIM * 4) / 1024**2
    if state and sub_mb <= BRUTEFORCE_MAX_MB:
        strategy = 'bruteforce'
    elif state and density <= DENSITY_THRESHOLD:
        strategy = 'pre'
    elif state:
        strategy = 'post'
    else:
        strategy = 'none'
    sys.stderr.write(
        f'strategy: {strategy} (subset={sub_mb:.0f} MB, '
        f'BF≤{BRUTEFORCE_MAX_MB}, density≤{DENSITY_THRESHOLD:.0%})\n')

    if strategy == 'bruteforce':
        t0 = time.time()
        allowed = fetch_allowed_ids(args.where)
        ids = bruteforce_topk(qvec, allowed, args.top_k)
        sys.stderr.write(f'[bruteforce] {len(allowed)} docs, '
                         f'{(time.time()-t0)*1000:.1f} ms wall\n')
        rows = lookup_metadata(ids)
        print(f'\n#  internal_id  arxiv_id    year  primary_cat  top_cat')
        print('-' * 60)
        for rank, (iid, aid, yr, pc, tc) in enumerate(rows):
            print(f'{rank:<2} {iid:<12} {aid:<12} {yr:<5} {pc:<12} {tc}')
        return

    with tempfile.TemporaryDirectory() as td:
        qvecs_path = os.path.join(td, 'q.fvecs')
        write_qvec_fvecs(qvec, qvecs_path)

        filter_path = None
        if strategy == 'pre':
            filter_path = os.path.join(td, 'filter_ch.bin')
            with open(filter_path, 'wb') as f:
                f.write(state)

        # If post-filter, overfetch — we'll drop misses below.
        eff_top_k = args.top_k * (10 if strategy == 'post' else 1)
        eff_top_k = min(eff_top_k, args.top_n)
        ids = run_forest(qvecs_path, filter_path, eff_top_k, args.top_n)

    if strategy == 'post':
        ids = post_filter(ids, args.where)
    ids = ids[: args.top_k]

    rows = lookup_metadata(ids)
    print(f'\n#  internal_id  arxiv_id    year  primary_cat  top_cat')
    print('-' * 60)
    for rank, (iid, aid, yr, pc, tc) in enumerate(rows):
        print(f'{rank:<2} {iid:<12} {aid:<12} {yr:<5} {pc:<12} {tc}')


if __name__ == '__main__':
    main()
