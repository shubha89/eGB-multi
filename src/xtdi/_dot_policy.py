"""Identify NumPy's geometry-dot rounding once, without a Numba dependency.

Physical wavefront seconds must retain the independent NumPy reference's
expression order. The checked NumPy kernels use (x*kx + z*kz) + y*ky,
with either a rounded first sum or a fused z*kz + rounded(x*kx).
This setup probe selects between those exact expressions; it does not adjust
physical parameters, interpolate queries, or relax an accuracy tolerance.
"""
from functools import lru_cache
import ctypes
import math
import numpy as np


@lru_cache(maxsize=1)
def host_fma():
    """libm/stdlib IEEE binary64 fma, used only for the setup probe."""
    if hasattr(math, 'fma'):
        return math.fma
    operation = ctypes.CDLL(None).fma
    operation.argtypes = [ctypes.c_double] * 3
    operation.restype = ctypes.c_double
    return operation


@lru_cache(maxsize=1)
def numpy_dot_uses_fma():
    """Fail closed if the host uses neither independently checked expression."""
    rng = np.random.default_rng(20261104)
    vectors = np.ascontiguousarray(rng.normal(size=(4, 256, 3)) * 500.)
    directions = rng.normal(size=(12, 3))
    directions /= np.linalg.norm(directions, axis=-1)[:, None]
    references = [np.einsum('lti,i->lt', vectors, k) for k in directions]
    separate = [((vectors[..., 0]*k[0] + vectors[..., 2]*k[2])
                 + vectors[..., 1]*k[1]) for k in directions]
    if all(a.tobytes() == b.tobytes() for a, b in zip(separate, references)):
        return False
    fma = host_fma()
    for k, expected in zip(directions, references):
        actual = np.array([fma(float(v[2]), float(k[2]), float(v[0]*k[0]))
                           + v[1]*k[1] for v in vectors.reshape(-1, 3)])
        if actual.tobytes() != expected.tobytes():
            raise RuntimeError('Unsupported host NumPy geometry-dot rounding; exact source queries are not qualified')
    return True
