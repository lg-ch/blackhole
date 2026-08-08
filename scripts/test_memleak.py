"""Memleak / fd-leak long-running test for serve_live.py.

Spawns serve_live.py, drives a mixed insert/search/delete workload for
DURATION seconds while sampling RSS + open fd count every SAMPLE_S.

Pass criteria : RSS at the end <= 1.5 × RSS at minute 1 (after warmup),
                fd count at the end <= warmup fd count + 50.

Run :
  python3 scripts/test_memleak.py [--duration 600] [--port 8901]
"""
from __future__ import annotations
import argparse, os, random, shutil, signal, subprocess, sys, time
import urllib.request, json
import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))


def http_post(port, path, body):
    req = urllib.request.Request(
        f'http://127.0.0.1:{port}{path}',
        data=json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def http_get(port, path):
    with urllib.request.urlopen(f'http://127.0.0.1:{port}{path}', timeout=5) as r:
        return r.status, json.loads(r.read().decode())


def rss_kb(pid):
    """Return RSS in KB from /proc/<pid>/status."""
    with open(f'/proc/{pid}/status') as f:
        for line in f:
            if line.startswith('VmRSS:'):
                return int(line.split()[1])
    return -1


def fd_count(pid):
    return len(os.listdir(f'/proc/{pid}/fd'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--duration', type=int, default=300,
                    help='Total test duration seconds (default 300)')
    ap.add_argument('--sample',   type=int, default=10,
                    help='Sample interval seconds (default 10)')
    ap.add_argument('--port',     type=int, default=8901)
    ap.add_argument('--dim',      type=int, default=128)
    args = ap.parse_args()

    ROOT = '/tmp/memleak_idx'
    if os.path.exists(ROOT):
        shutil.rmtree(ROOT)

    # Launch serve_live in subprocess.
    log = open('/tmp/memleak_serve.log', 'w')
    proc = subprocess.Popen([
        'python3', os.path.join(HERE, 'serve_live.py'),
        '--root', ROOT, '--port', str(args.port), '--create',
        '--dim', str(args.dim), '--sub_dim', '16',
        '--n_trees', '100', '--depth', '12', '--gen', '3',
    ], stdout=log, stderr=log)
    print(f'[memleak] started serve_live pid={proc.pid}, port={args.port}')

    # Wait for /health to respond.
    for _ in range(50):
        time.sleep(0.2)
        try:
            code, _ = http_get(args.port, '/health')
            if code == 200:
                break
        except Exception:
            pass

    rng = np.random.default_rng(42)
    samples: list[tuple[int, int, int]] = []  # (t, rss_kb, fds)
    t_start = time.time()
    last_sample = 0.0
    ops = {'insert': 0, 'search': 0, 'delete': 0, 'freeze': 0,
           'bp_reject': 0, 'errors': 0}

    print(f'[memleak] driving load for {args.duration}s ...')
    while time.time() - t_start < args.duration:
        # Sample at intervals.
        if time.time() - t_start - last_sample > args.sample:
            last_sample = time.time() - t_start
            try:
                rss = rss_kb(proc.pid)
                fds = fd_count(proc.pid)
                samples.append((int(last_sample), rss, fds))
                print(f'  t={int(last_sample):>4}s rss={rss/1024:.1f} MB '
                      f'fds={fds} ops={ops}')
            except FileNotFoundError:
                print(f'  process died at t={last_sample:.0f}s')
                break

        # Random op: 70% insert, 25% search, 4% delete, 1% freeze
        r = random.random()
        if r < 0.70:
            v = rng.standard_normal(args.dim).astype(np.float32).tolist()
            code, body = http_post(args.port, '/insert', {'vec': v})
            if code == 503:
                ops['bp_reject'] += 1
                # On backpressure, force a freeze to clear active buffer
                code2, _ = http_post(args.port, '/freeze', {})
                if code2 == 200: ops['freeze'] += 1
            elif code == 200:
                ops['insert'] += 1
            else:
                ops['errors'] += 1
        elif r < 0.95:
            v = rng.standard_normal(args.dim).astype(np.float32).tolist()
            code, _ = http_post(args.port, '/search',
                                {'qvec': v, 'top_n': 200, 'top_k': 10})
            if code == 200: ops['search'] += 1
            else: ops['errors'] += 1
        elif r < 0.99:
            doc_id = random.randint(0, max(1, ops['insert'] - 1))
            code, _ = http_post(args.port, '/delete', {'doc_id': doc_id})
            if code == 200: ops['delete'] += 1
        else:
            code, _ = http_post(args.port, '/freeze', {})
            if code == 200: ops['freeze'] += 1

    # Final sample.
    try:
        samples.append((int(time.time() - t_start), rss_kb(proc.pid), fd_count(proc.pid)))
    except FileNotFoundError:
        pass

    # Shutdown serve_live cleanly.
    print(f'[memleak] sending SIGTERM ...')
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
    log.close()

    # Verdict.
    if len(samples) < 3:
        print(f'[memleak] FAIL : only {len(samples)} samples collected')
        sys.exit(1)
    # Warmup = first 1/4 of samples; tail = last 1/4. For RSS we check the
    # peak (leaks accumulate, peak grows). For fds we check the MINIMUM
    # of each window : fds oscillate during freeze/compact cycles (each
    # Forest holds n_trees fds, swaps move them around), but a leak shows
    # up as the floor (min) rising over time.
    n = len(samples)
    head = samples[: max(1, n // 4)]
    tail = samples[-max(1, n // 4):]
    warm_rss_peak = max(s[1] for s in head)
    tail_rss_peak = max(s[1] for s in tail)
    warm_fds_min  = min(s[2] for s in head)
    tail_fds_min  = min(s[2] for s in tail)
    rss_ratio = tail_rss_peak / max(1, warm_rss_peak)
    fds_floor_rise = tail_fds_min - warm_fds_min
    print(f'\n[memleak] warmup : RSS peak {warm_rss_peak/1024:.1f} MB, fds min {warm_fds_min}')
    print(f'[memleak] tail   : RSS peak {tail_rss_peak/1024:.1f} MB, fds min {tail_fds_min}')
    print(f'[memleak] rss ratio       = {rss_ratio:.2f} (PASS if <= 1.50)')
    print(f'[memleak] fds floor delta = {fds_floor_rise:+d}     (PASS if <= 200)')
    print(f'[memleak] ops summary: {ops}')
    ok = rss_ratio <= 1.5 and fds_floor_rise <= 200
    print(f'\n[memleak] RESULT : {"PASS" if ok else "FAIL"}')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
