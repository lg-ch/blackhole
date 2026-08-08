#!/usr/bin/env python3
"""
Bench latence du pipeline ClickHouse+forest sur arxiv 2 M / dim 768.

Scénarios :
  - no_filter
  - pre  year=2007                  ~2 %
  - pre  year IN (2023,2024)         ~14 %
  - pre  year=2023 AND primary_cat='cs.LG'   << 1 %
  - post top_cat='cs'                ~24 %

Pour chaque scénario : 100 queries (vecs corpus aléatoires), un seul
subprocess `rpforest search`, mesure p50 / p95 / p99 / max.
"""
import argparse, os, struct, subprocess, sys, tempfile, time
import numpy as np
from clickhouse_driver import Client

ROOT     = '/home/chatelet/mangrove-search'
BASE     = f'{ROOT}/datasets/arxiv/arxiv_base.fvecs'
INDEX    = f'{ROOT}/index_arxiv'
RPFOREST = f'{ROOT}/rpforest'
N_DOCS   = 2_058_751
DIM      = 768
N_TREES  = 1000
DEPTH    = 20

ch = Client('127.0.0.1')

SCENARIOS = [
    ('no_filter',      None,                                                  'none'),
    ('pre_2007',       "year=2007",                                           'pre'),
    ('pre_2023_24',    "year IN (2023,2024)",                                 'pre'),
    ('pre_2023_csLG',  "year=2023 AND primary_cat='cs.LG'",                   'pre'),
    ('post_cs',        "top_cat='cs'",                                        'post'),
]


def read_corpus_vec(internal_id: int) -> np.ndarray:
    rec = 4 + DIM * 4
    with open(BASE, 'rb') as f:
        f.seek(internal_id * rec)
        d = struct.unpack('<i', f.read(4))[0]
        assert d == DIM
        return np.frombuffer(f.read(DIM * 4), dtype=np.float32).copy()


def write_fvecs(vecs: np.ndarray, path: str) -> None:
    n, d = vecs.shape
    with open(path, 'wb') as f:
        for i in range(n):
            f.write(struct.pack('<i', d))
            f.write(vecs[i].astype(np.float32).tobytes())


def fetch_filter(where: str) -> tuple[bytes, int, float]:
    """(state_bytes, cardinality, ch_lookup_ms)"""
    if not where:
        return b'', N_DOCS, 0.0
    sql = (f"SELECT cast(groupBitmapState(internal_id), 'String'), count() "
           f"FROM mangrove.docs WHERE {where}")
    t0 = time.time()
    state, card = ch.execute(sql)[0]
    return state, card, (time.time() - t0) * 1000


def run_forest(qvecs_path: str, filter_ch: str | None,
               n_q: int, top_k: int, top_n: int) -> tuple[list[list[int]], list[float]]:
    cmd = [
        RPFOREST, '--dim', str(DIM),
        'search', qvecs_path, BASE, INDEX,
        str(N_TREES), str(DEPTH), str(N_DOCS),
        str(top_k), str(top_n), str(n_q),
    ]
    if filter_ch:
        cmd += ['--filter_ch', filter_ch]
    r = subprocess.run(cmd, capture_output=True, text=True, check=True)
    rows = [
        [int(x) for x in line.split(',')[1:]]
        for line in r.stdout.strip().splitlines() if line
    ]
    # parse stderr lines "Q <idx> <ms>"
    per_q = []
    for ln in r.stderr.splitlines():
        if ln.startswith('Q '):
            try:
                per_q.append(float(ln.split()[2]))
            except (IndexError, ValueError):
                pass
    return rows, per_q


def percentiles(xs: list[float]) -> dict:
    if not xs: return {'p50': 0, 'p95': 0, 'p99': 0, 'max': 0}
    a = np.asarray(xs)
    return {
        'p50': float(np.percentile(a, 50)),
        'p95': float(np.percentile(a, 95)),
        'p99': float(np.percentile(a, 99)),
        'max': float(a.max()),
    }


def bench_scenario(name: str, where: str | None, strategy: str,
                   qvecs_path: str, n_q: int, top_k: int, top_n: int) -> dict:
    state = b''
    card  = N_DOCS
    ch_ms = 0.0
    if where:
        state, card, ch_ms = fetch_filter(where)

    pre = (strategy == 'pre')
    post = (strategy == 'post')

    # write filter
    filter_path = None
    if pre and state:
        filter_path = qvecs_path + '.filter'
        with open(filter_path, 'wb') as f:
            f.write(state)

    # overfetch if post
    eff_top_k = top_k * (10 if post else 1)
    eff_top_k = min(eff_top_k, top_n)

    rows, per_q = run_forest(qvecs_path, filter_path, n_q, eff_top_k, top_n)
    p = percentiles(per_q)

    # post-filter : 1 CH SQL per query (prod path)
    post_per_q = []
    if post and where:
        for row in rows:
            t0 = time.time()
            if row:
                placeholders = ','.join(str(i) for i in row)
                ch.execute(
                    f"SELECT internal_id FROM mangrove.docs "
                    f"WHERE internal_id IN ({placeholders}) AND ({where})"
                )
            post_per_q.append((time.time() - t0) * 1000)
    pp = percentiles(post_per_q) if post_per_q else {'p50': 0, 'p95': 0, 'p99': 0, 'max': 0}

    return {
        'name': name, 'where': where or '-',
        'card': card, 'density': card / N_DOCS,
        'strategy': strategy,
        'state_bytes': len(state) if state else 0,
        'ch_lookup_ms': ch_ms,            # one-shot cost, amortized over batch
        'forest_p': p,
        'post_p': pp,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_q',   type=int, default=100)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--top_n', type=int, default=500)
    ap.add_argument('--seed',  type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    qids = rng.integers(0, N_DOCS, args.n_q).tolist()
    qvecs = np.stack([read_corpus_vec(i) for i in qids])

    with tempfile.TemporaryDirectory() as td:
        qvecs_path = os.path.join(td, 'q.fvecs')
        write_fvecs(qvecs, qvecs_path)

        print(f'\nbench: n_q={args.n_q} top_k={args.top_k} top_n={args.top_n}\n')
        hdr = (f"{'scenario':<17} {'card':>9} {'den%':>6} {'strat':<5} {'state':>7} "
               f"{'ch_ms':>6}  {'forest p50/p95/p99/max':>26}  "
               f"{'post p50/p95/p99/max':>22}")
        print(hdr)
        print('-' * len(hdr))
        for name, where, strat in SCENARIOS:
            r = bench_scenario(name, where, strat, qvecs_path,
                               args.n_q, args.top_k, args.top_n)
            fp, pp = r['forest_p'], r['post_p']
            print(f"{r['name']:<17} {r['card']:>9} {r['density']*100:>5.2f}% "
                  f"{r['strategy']:<5} {r['state_bytes']:>6}B {r['ch_lookup_ms']:>5.1f} "
                  f" {fp['p50']:>5.1f}/{fp['p95']:>5.1f}/{fp['p99']:>5.1f}/{fp['max']:>5.1f} "
                  f" {pp['p50']:>4.1f}/{pp['p95']:>4.1f}/{pp['p99']:>4.1f}/{pp['max']:>4.1f}")


if __name__ == '__main__':
    main()
