#!/usr/bin/env python3
"""
Bench complet : recall + latence + RAM, par scénario filter.

Inputs (préalable) :
  scripts/build_gt_arxiv.py --n_q 100  ->  bench_q.fvecs + bench_gt.ivecs

Output : table recall@10 + p50/p95/p99/max latence + RSS forest + RSS CH.

NB : GT bruteforce est sur le corpus complet (recall basé sur la vraie
top-100 L2). Pour les scénarios filtre, le recall@10 mesure les hits du
top-10 forest+rerank parmi le top-100 GT. Si le filtre exclut certains GT
votes, le recall théorique max < 1 (c'est attendu, c'est la vraie sémantique
filter-aware).
"""
import argparse, os, re, struct, subprocess, sys, tempfile, time
import numpy as np
from clickhouse_driver import Client

ROOT     = '/home/chatelet/mangrove-search'
BASE     = f'{ROOT}/datasets/arxiv/arxiv_base.fvecs'
INDEX    = f'{ROOT}/index_arxiv'
RPFOREST = f'{ROOT}/rpforest'
QVECS    = f'{ROOT}/datasets/arxiv/bench_q.fvecs'
GT       = f'{ROOT}/datasets/arxiv/bench_gt.ivecs'
N_DOCS   = 2_058_751
DIM      = 768
N_TREES  = 1000
DEPTH    = 20

ch = Client('127.0.0.1')

ARX = f'{ROOT}/datasets/arxiv'
SCENARIOS = [
    # (name, where, strategy, gt_path)
    ('no_filter',     None,                                       'none', f'{ARX}/bench_gt.ivecs'),
    ('pre_2007',      "year=2007",                                'pre',  f'{ARX}/bench_gt_2007.ivecs'),
    ('pre_2023_24',   "year IN (2023,2024)",                      'pre',  f'{ARX}/bench_gt_2023_24.ivecs'),
    ('pre_2023_csLG', "year=2023 AND primary_cat='cs.LG'",        'pre',  f'{ARX}/bench_gt_2023_csLG.ivecs'),
    ('post_cs',       "top_cat='cs'",                             'post', f'{ARX}/bench_gt_cs.ivecs'),
]


def ch_rss_mb() -> tuple[float, float]:
    """Return (RssAnon_MB, RssFile_MB) of the clickhouse-server process."""
    try:
        out = subprocess.check_output(['ps', '-C', 'clickhouse', '-o', 'pid='], text=True)
        pid = int(out.strip().split('\n')[0])
        with open(f'/proc/{pid}/status') as f:
            anon = filed = 0
            for line in f:
                if   line.startswith('RssAnon:'): anon  = int(line.split()[1]) / 1024
                elif line.startswith('RssFile:'): filed = int(line.split()[1]) / 1024
            return anon, filed
    except Exception as e:
        sys.stderr.write(f'ch_rss probe failed: {e}\n')
        return 0.0, 0.0


def fetch_filter(where: str) -> tuple[bytes, int]:
    if not where:
        return b'', N_DOCS
    sql = (f"SELECT cast(groupBitmapState(internal_id), 'String'), count() "
           f"FROM mangrove.docs WHERE {where}")
    state, card = ch.execute(sql)[0]
    return state, card


def run_topn(filter_ch: str | None, gt_path: str,
             n_q: int, top_k: int, top_n: int, qd: int = 0) -> dict:
    cmd = [
        RPFOREST, '--dim', str(DIM),
    ]
    if qd > 0:
        cmd += ['--query_depth', str(qd)]
    cmd += [
        'topn', QVECS, gt_path, BASE, INDEX,
        str(N_TREES), str(DEPTH), str(N_DOCS),
        str(top_k), str(top_n), str(n_q),
    ]
    if filter_ch:
        cmd += ['--filter_ch', filter_ch]
    r = subprocess.run(cmd, capture_output=True, text=True, check=True)
    err = r.stderr
    # parse :  "Recall@10 = 0.XXXX  |  top_n = X  |  avg_cands = X  |  X.XX ms/query"
    # plus :   "total: X.XXs   RSS anon X.XX MB | mapped X.XX MB | peak ru_maxrss X.XX MB"
    out = {'raw': err}
    m = re.search(r'Recall@\d+ = ([\d.]+)', err)
    out['recall']     = float(m.group(1)) if m else float('nan')
    m = re.search(r'avg_cands = (\d+)', err)
    out['avg_cands']  = int(m.group(1)) if m else -1
    m = re.search(r'([\d.]+) ms/query', err)
    out['ms_per_q']   = float(m.group(1)) if m else float('nan')
    m = re.search(r'RSS anon ([\d.]+) MB \| mapped ([\d.]+) MB \| peak ru_maxrss ([\d.]+) MB', err)
    if m:
        out['rss_anon_mb'] = float(m.group(1))
        out['rss_map_mb']  = float(m.group(2))
        out['rss_peak_mb'] = float(m.group(3))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_q',   type=int, default=100)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--top_n', type=int, default=2000)
    ap.add_argument('--qd',    type=int, default=16,
                    help='query_depth (lower = wider leaves, higher recall)')
    args = ap.parse_args()

    assert os.path.exists(QVECS), 'run build_gt_arxiv.py first'
    assert os.path.exists(GT),    'run build_gt_arxiv.py first'

    print(f'\nbench: n_q={args.n_q} top_k={args.top_k} top_n={args.top_n} qd={args.qd}')
    ch_anon0, ch_map0 = ch_rss_mb()
    print(f'CH idle  RssAnon={ch_anon0:.1f} MB  RssFile={ch_map0:.1f} MB\n')

    hdr = (f"{'scenario':<17} {'card':>9} {'den%':>6} {'strat':<5} "
           f"{'state':>8} {'recall@10':>9} {'ms/q':>6} "
           f"{'f_anon':>6} {'f_map':>7} {'f_peak':>7} {'ch_anon':>8}")
    print(hdr); print('-' * len(hdr))

    with tempfile.TemporaryDirectory() as td:
        for name, where, strat, gt_path in SCENARIOS:
            state, card = fetch_filter(where or '')
            density = card / N_DOCS

            filter_path = None
            if strat == 'pre' and state:
                filter_path = f'{td}/{name}.filter'
                with open(filter_path, 'wb') as f:
                    f.write(state)

            # For 'post' we don't pre-filter the forest, but the GT used here
            # is filter-aware (top-100 over the cs subset). Recall therefore
            # measures how often the forest top-K naturally contains the
            # filter-restricted NN. The app-side post-filter step would shrink
            # the result set further but doesn't affect the upstream recall.
            anon_before, _ = ch_rss_mb()
            r = run_topn(filter_path, gt_path, args.n_q, args.top_k, args.top_n, args.qd)
            anon_after, _ = ch_rss_mb()

            print(f"{name:<17} {card:>9} {density*100:>5.2f}% {strat:<5} "
                  f"{len(state) if state else 0:>7}B "
                  f"{r.get('recall', float('nan')):>9.4f} "
                  f"{r.get('ms_per_q', float('nan')):>5.1f} "
                  f"{r.get('rss_anon_mb', 0):>5.1f} "
                  f"{r.get('rss_map_mb', 0):>6.1f} "
                  f"{r.get('rss_peak_mb', 0):>6.1f} "
                  f"{anon_after:>7.1f}")

    ch_anonF, ch_mapF = ch_rss_mb()
    print(f'\nCH final RssAnon={ch_anonF:.1f} MB  RssFile={ch_mapF:.1f} MB')


if __name__ == '__main__':
    main()
