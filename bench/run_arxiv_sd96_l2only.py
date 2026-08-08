"""arxiv 2M sd96 — exact L2 directly, no TQ1.
NP=10 QD=16 top_n=1000, single CPU pinned.
"""
import os, sys, time
os.environ['OMP_NUM_THREADS'] = '1'
os.sched_setaffinity(0, {0})
import numpy as np, resource
sys.path.insert(0, '/home/chatelet/mangrove-search/scripts')
import mangrove_ffi as mf
from mangrove_ffi import Forest

DATA = '/home/chatelet/mangrove-search/datasets/arxiv'
IDX  = '/mnt/mangrove/indexes/arxiv_sd96'
BASE = f'{DATA}/arxiv_base.fvecs'
QP   = f'{DATA}/bench_q.fvecs'
GP   = f'{DATA}/bench_gt.ivecs'
DIM = 768
NP, QD, TN = 10, 16, 1000

def drop_caches():
    os.sync(); open('/proc/sys/vm/drop_caches', 'w').write('3')

qraw = np.fromfile(QP, dtype=np.int32)
nq = qraw.size // (1 + DIM)
Q = qraw.reshape(nq, 1 + DIM)[:, 1:].view(np.float32).copy()
graw = np.fromfile(GP, dtype=np.int32)
gk = graw.size // nq - 1
GT = graw.reshape(nq, 1 + gk)[:, 1:11].copy()

mf.set_shared_scratch_pool(True)
f = Forest(IDX, n_trees=256, dim=DIM, sub_dim=96, depth=20,
           n_docs=2058751, gen_version=3)

print(f'arxiv 2M sd96 — exact L2 direct, no TQ1')
print(f'NP={NP}  QD={QD}  top_n={TN}, single CPU pinned core 0\n')

drop_caches()
# warm
for q in Q[:5]:
    ids, _, n = f.query_probes(q, NP, top_n=TN, probe_depth=QD)
    cand = np.asarray(ids[:n], dtype=np.int32); cand = cand[cand >= 0]
    f.rerank_l2(BASE, q, cand, top_k=10)

recs, lats, ncands = [], [], []
for qi in range(5, 105):
    q = Q[qi % len(Q)]; ref = set(int(x) for x in GT[qi % len(GT)])
    t0 = time.time()
    ids, _, n = f.query_probes(q, NP, top_n=TN, probe_depth=QD)
    cand = np.asarray(ids[:n], dtype=np.int32); cand = cand[cand >= 0]
    top = f.rerank_l2(BASE, q, cand, top_k=10)
    lats.append((time.time() - t0) * 1000.0)
    recs.append(sum(1 for x in top if int(x) in ref) / 10.0)
    ncands.append(len(cand))

drop_caches()
t0 = time.time()
ids, _, n = f.query_probes(Q[0], NP, top_n=TN, probe_depth=QD)
cand = np.asarray(ids[:n], dtype=np.int32); cand = cand[cand >= 0]
f.rerank_l2(BASE, Q[0], cand, top_k=10)
cold = (time.time() - t0) * 1000.0

rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
print(f'recall@10   = {np.mean(recs):.4f}')
print(f'avg cands   = {int(np.mean(ncands))}')
print(f'p50 warm    = {np.percentile(lats, 50):.1f} ms')
print(f'p95 warm    = {np.percentile(lats, 95):.1f} ms')
print(f'cold q0     = {cold:.1f} ms')
print(f'peak RSS    = {rss:.1f} MB')
