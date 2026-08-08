"""Core unit tests for mangrove-search C primitives via FFI.

Run:
    python3 -m pytest tests/test_core.py -v
or:
    python3 tests/test_core.py
"""
from __future__ import annotations

import os
import struct
import sys
import tempfile

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import mangrove_ffi as ffi
from mangrove_ffi import (Forest, verify_srt, set_gen_version,
                          varbyte_encode, varbyte_decode,
                          traverse_sub, traverse_sub_continue)


# -------------------- VarByte --------------------

@pytest.mark.parametrize("v", [
    0, 1, 127, 128, 255, 256, 16383, 16384,
    65535, 1_000_000, (1 << 20) - 1, (1 << 31) - 1, (1 << 32) - 1,
])
def test_varbyte_roundtrip(v):
    """encode(v) → decode → v exactly, for all uint32 magnitudes."""
    enc = varbyte_encode(v)
    decoded, n_consumed = varbyte_decode(enc)
    assert decoded == v
    assert n_consumed == len(enc)


def test_varbyte_size():
    """Small values fit in 1 byte, large ones grow predictably."""
    assert len(varbyte_encode(0)) == 1
    assert len(varbyte_encode(127)) == 1
    assert len(varbyte_encode(128)) == 2
    assert len(varbyte_encode(16383)) == 2
    assert len(varbyte_encode(16384)) == 3
    assert len(varbyte_encode((1 << 31) - 1)) == 5


# -------------------- Traversal continue --------------------

@pytest.mark.parametrize("seed_tree", [0, 17, 999])
def test_traverse_continue_matches_full(seed_tree):
    """traverse(qvec, depth) ≡ traverse(qvec, D1) + traverse_continue(qvec, D1, depth-D1).
       Continuing from an intermediate leaf MUST land on the same final leaf."""
    np.random.seed(42 + seed_tree)
    dim, sub_dim, depth = 128, 16, 20
    qvec = np.random.randn(dim).astype(np.float32)
    # normalize to match build-time
    qvec /= max(1e-9, np.linalg.norm(qvec))

    full_leaf = traverse_sub(qvec, sub_dim, depth, seed_tree)

    for split in [5, 10, 15, 18]:
        mid_node = traverse_sub(qvec, sub_dim, split, seed_tree)
        n_extra = depth - split
        final_node = traverse_sub_continue(qvec, sub_dim, split, n_extra,
                                            mid_node, seed_tree)
        assert final_node == full_leaf, (
            f'split={split}: full={full_leaf} vs split+continue={final_node}')


# -------------------- xxhash footer --------------------

def test_verify_srt_on_tiny_build(tmp_path):
    """Build a tiny SIFT 1M index slice, verify_srt returns 1 (OK)
       on all trees; corrupting a byte returns 0 (mismatch)."""
    import subprocess
    rpf = os.path.join(ROOT, 'rpforest')
    sift = os.path.join(ROOT, 'sift/sift_base.fvecs')
    if not os.path.exists(sift):
        pytest.skip(f'no SIFT base at {sift}')

    idx = str(tmp_path / 'verify_test')
    subprocess.check_call([rpf, '--dim', '128', '--sub_dim', '0',
                           '--gen', 'v0', '--doc_count', '10000',
                           'build', sift, idx, '5', '10'],
                          stdout=subprocess.DEVNULL)

    for t in range(5):
        rc = verify_srt(f'{idx}/tree{t:05d}.srt')
        assert rc == 1, f'tree {t} verify failed (rc={rc})'

    # Corrupt 4 bytes mid-file → verify fails
    with open(f'{idx}/tree00000.srt', 'r+b') as f:
        f.seek(500)
        f.write(b'\x00\xff\x00\xff')
    assert verify_srt(f'{idx}/tree00000.srt') == 0


# -------------------- Tombstones --------------------

def test_tombstones_roundtrip(tmp_path):
    """Forest.tombstone_add → flush → reopen → count preserved."""
    import subprocess
    rpf = os.path.join(ROOT, 'rpforest')
    sift = os.path.join(ROOT, 'sift/sift_base.fvecs')
    if not os.path.exists(sift):
        pytest.skip(f'no SIFT base at {sift}')

    idx = str(tmp_path / 'tomb_test')
    subprocess.check_call([rpf, '--dim', '128', '--sub_dim', '0',
                           '--gen', 'v0', '--doc_count', '10000',
                           'build', sift, idx, '5', '10'],
                          stdout=subprocess.DEVNULL)

    set_gen_version(0)
    f = Forest(idx, n_trees=5, dim=128, sub_dim=0, depth=10,
               n_docs=10000, gen_version=0)
    assert f.tombstones_count() == 0
    for doc_id in [42, 100, 200, 300]:
        f.tombstone_add(doc_id)
    assert f.tombstones_count() == 4
    f.tombstones_flush()
    f.close()

    f2 = Forest(idx, n_trees=5, dim=128, sub_dim=0, depth=10,
                n_docs=10000, gen_version=0)
    assert f2.tombstones_count() == 4, 'tombstones not persisted'
    f2.close()


def test_tombstone_blocks_doc_from_results(tmp_path):
    """After tombstoning a doc_id, it must not appear in query results."""
    import subprocess
    rpf = os.path.join(ROOT, 'rpforest')
    sift = os.path.join(ROOT, 'sift/sift_base.fvecs')
    if not os.path.exists(sift):
        pytest.skip(f'no SIFT base at {sift}')

    idx = str(tmp_path / 'tomb_block_test')
    subprocess.check_call([rpf, '--dim', '128', '--sub_dim', '0',
                           '--gen', 'v0', '--doc_count', '10000',
                           'build', sift, idx, '20', '10'],
                          stdout=subprocess.DEVNULL)

    set_gen_version(0)
    f = Forest(idx, n_trees=20, dim=128, sub_dim=0, depth=10,
               n_docs=10000, gen_version=0)
    # Use an arbitrary query
    with open(sift, 'rb') as src:
        src.read(4)
        qvec = np.frombuffer(src.read(128 * 4), dtype=np.float32).copy()
    ids, _, n = f.query(qvec, top_n=100)
    assert n > 0
    blocked = int(ids[0])
    f.tombstone_add(blocked)
    ids2, _, n2 = f.query(qvec, top_n=100)
    assert blocked not in ids2[:n2].tolist(), (
        f'blocked id {blocked} still appears in top-{n2}')
    f.close()


# -------------------- Smoke main --------------------

if __name__ == '__main__':
    # Allow running without pytest (CI fallback)
    failures = []
    tests = [
        (test_varbyte_roundtrip, [0, 1, 127, 128, 16384, (1 << 31) - 1]),
        (test_varbyte_size, [None]),
        (test_traverse_continue_matches_full, [0, 17, 999]),
    ]
    for fn, args in tests:
        for a in args:
            try:
                fn(a) if a is not None else fn()
                print(f'  ✓ {fn.__name__}({a})')
            except AssertionError as e:
                failures.append((fn.__name__, a, e))
                print(f'  ✗ {fn.__name__}({a}): {e}')
    if failures:
        print(f'\n{len(failures)} failures:')
        for n, a, e in failures:
            print(f'  - {n}({a}): {e}')
        sys.exit(1)
    print('\nAll smoke tests passed.')
