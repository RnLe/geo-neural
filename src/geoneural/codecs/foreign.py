"""Conventional codecs as products: each encodes a float field under an absolute bound and decodes from its bytes.

Versions are recorded with every result, because codec behaviour changes between releases. SZ3, SPERR and raw
LERC come from imagecodecs (no GeoTIFF wrapper), zfp from zfpy in fixed-accuracy mode, q32 from this package.
"""
from __future__ import annotations

import numpy as np


def _ic():
    import imagecodecs
    return imagecodecs


def versions() -> dict:
    out = {}
    try:
        ic = _ic()
        out["imagecodecs"] = ic.__version__
        for name in ("sz3", "sperr", "lerc", "zfp"):
            fn = getattr(ic, f"{name}_version", None)
            if fn:
                out[name] = fn()
    except ImportError:
        pass
    try:
        import zfpy
        out["zfpy"] = getattr(zfpy, "__version__", "unknown")
    except ImportError:
        pass
    return out


def encode(name: str, field: np.ndarray, bound_m: float) -> bytes:
    a = np.ascontiguousarray(field, np.float32)
    if name == "sz3":
        ic = _ic()
        return ic.sz3_encode(a, mode=ic.SZ3.MODE.ABS, abs=bound_m)
    if name == "sperr":
        ic = _ic()
        return ic.sperr_encode(a.astype(np.float64), level=bound_m, mode=ic.SPERR.MODE.PWE)
    if name == "lerc":
        return _ic().lerc_encode(a, level=bound_m)
    if name == "zfp":
        import zfpy
        return zfpy.compress_numpy(a, tolerance=bound_m)
    if name == "q32dz":
        from geoneural.codecs.codecs import registry
        return registry()["q32-delta-zstd"].encode(a, bound_m)
    raise ValueError(f"unknown codec {name}")


def decode(name: str, blob: bytes, shape) -> np.ndarray:
    if name == "sz3":
        return np.asarray(_ic().sz3_decode(blob, shape=tuple(shape), dtype=np.float32), np.float64)
    if name == "sperr":
        return np.asarray(_ic().sperr_decode(blob, shape=tuple(shape), dtype=np.float64), np.float64)
    if name == "lerc":
        return np.asarray(_ic().lerc_decode(blob), np.float64).reshape(shape)
    if name == "zfp":
        import zfpy
        return np.asarray(zfpy.decompress_numpy(blob), np.float64).reshape(shape)
    if name == "q32dz":
        from geoneural.codecs.codecs import registry
        return np.asarray(registry()["q32-delta-zstd"].decode(blob), np.float64).reshape(shape)
    raise ValueError(f"unknown codec {name}")


NAMES = ("sz3", "sperr", "zfp", "lerc", "q32dz")
