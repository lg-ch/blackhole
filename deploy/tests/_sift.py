"""Shared helpers for the deployment e2e tests: fvecs/ivecs readers."""
import numpy as np


def read_fvecs(path, limit=None):
    """Read a .fvecs file → (N, dim) float32 array.
    Format: per vector [int32 dim][dim × float32]."""
    a = np.fromfile(path, dtype=np.int32)
    dim = int(a[0])
    a = a.reshape(-1, dim + 1)
    if limit is not None:
        a = a[:limit]
    return a[:, 1:].view(np.float32)


def read_ivecs(path, limit=None):
    """Read a .ivecs file → (N, k) int32 array (groundtruth)."""
    a = np.fromfile(path, dtype=np.int32)
    k = int(a[0])
    a = a.reshape(-1, k + 1)
    if limit is not None:
        a = a[:limit]
    return a[:, 1:]
