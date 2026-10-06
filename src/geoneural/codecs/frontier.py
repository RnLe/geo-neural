"""Conventional arms that isolate parts of the learned coder (protocol arms C1 to C3) and encoder-side task corrections.

C1: a coarse lattice coded by a conventional codec, cubic interpolation to the full lattice, and the residual coded by a
conventional codec at the target bound. It isolates multiscale prediction without a neural network.
C2: a uniform conventional product plus a sparse correction stream on cells the encoder chooses from the reference
(stream bands, or cells whose flow direction has a small margin), judged at equal total bytes.
C3: the C1 base plus a per-field ridge regression of the residual on coarse derivatives, with or without geology
classes; coefficients and the class raster are charged.

Every product is a `.gnc` file (coder "foreign"); `decode` rebuilds the field from the bytes alone.
"""
from __future__ import annotations

import struct

import numpy as np
import zstandard

from geoneural.codecs import foreign, package
from geoneural.metrics import corrections, drainage

CODEC_IDS = {name: i for i, name in enumerate(foreign.NAMES)}
CODEC_NAMES = {i: name for name, i in CODEC_IDS.items()}
_C1 = struct.Struct("<BBBBd")  # factor, base codec, residual codec, flags (bit 0: regression), base bound


def upsample(coarse: np.ndarray, factor: int, side: int) -> np.ndarray:
    """Node-aligned cubic interpolation of a stride-`factor` lattice to `side` nodes (float64, deterministic)."""
    from scipy.ndimage import map_coordinates
    t = np.arange(side) / factor
    rr, cc = np.meshgrid(t, t, indexing="ij")
    return map_coordinates(np.asarray(coarse, np.float64), [rr, cc], order=3, mode="nearest")


def _features(up: np.ndarray, classes: np.ndarray | None, spacing_m: float, n_classes: int) -> np.ndarray:
    """Design matrix for the residual regression: intercept, slope, curvature and their product, plus class
    indicators. All terms come from what the decoder has (the decoded base and the transmitted raster)."""
    gr, gc = np.gradient(up, spacing_m)
    slope = np.hypot(gr, gc)
    lap = (np.gradient(gr, spacing_m, axis=0) + np.gradient(gc, spacing_m, axis=1)) * spacing_m
    cols = [np.ones(up.size), slope.ravel(), lap.ravel(), (slope * lap).ravel()]
    if classes is not None:
        flat = classes.ravel()
        cols += [(flat == k).astype(np.float64) for k in range(1, n_classes)]
    return np.stack(cols, 1)


def encode_base(z: np.ndarray, bound_m: float, factor: int, base_bound_m: float, base: str = "sz3",
                residual: str = "sz3", raster: np.ndarray | None = None, regression: bool = False,
                spatial: dict | None = None, spacing_m: float = 10.0):
    """C1 (regression False) or C3 (regression True, with class indicators from `raster`, which is charged; the
    classes per node come from the raster exactly as the decoder derives them). Returns (blob, reconstruction)."""
    side = z.shape[0]
    sub = np.ascontiguousarray(z[::factor, ::factor], np.float32)
    bblob = foreign.encode(base, sub, base_bound_m)
    up = upsample(foreign.decode(base, bblob, sub.shape), factor, side)
    comps = {"params": _C1.pack(factor, CODEC_IDS[base], CODEC_IDS[residual], int(regression), base_bound_m),
             "base": bblob}
    pred = up
    if regression:
        n_classes = int(raster.max()) + 1 if raster is not None else 0
        classes = package.context_nodes(raster, side) if raster is not None else None
        X = _features(up, classes, spacing_m, n_classes)
        y = (np.asarray(z, np.float64) - up).ravel()
        lam = 1e-3 * X.shape[0]
        beta = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y).astype(np.float32)
        comps["coefficients"] = struct.pack("<H", n_classes) + beta.tobytes()
        if raster is not None:
            comps["context"] = zstandard.ZstdCompressor(level=19).compress(np.ascontiguousarray(raster, np.uint8).tobytes())
        pred = up + (X @ beta.astype(np.float64)).reshape(z.shape)
    r = (np.asarray(z, np.float64) - pred).astype(np.float32)
    comps["residual"] = foreign.encode(residual, r, bound_m)
    blob = _product(z.shape, bound_m, "c1" if not regression else "c3", comps, spatial)
    return blob, decode(blob)


def _product(shape, bound_m, name, comps, spatial) -> bytes:
    E = package.ml.bound_units(bound_m)
    return package.Product(rows=shape[0], cols=shape[1], E=E, coder="foreign", foreign=name, components=comps,
                           **(spatial or {})).to_bytes()


def encode_corrected(z: np.ndarray, bound_m: float, codec: str, mask: np.ndarray, quantum_m: float,
                     spatial: dict | None = None):
    """C2: the uniform `codec` product at `bound_m` and a sparse correction of the masked cells to `quantum_m`."""
    blob = foreign.encode(codec, np.ascontiguousarray(z, np.float32), bound_m)
    dec = foreign.decode(codec, blob, z.shape).astype(np.float64)
    idx = np.flatnonzero(mask)
    codes = np.rint((np.asarray(z, np.float64).ravel()[idx] - dec.ravel()[idx]) / quantum_m).astype(np.int64)
    corr = corrections.encode_corrections(idx, codes, quantum_m)["blob"]
    comps = {"params": struct.pack("<B", CODEC_IDS[codec]), "foreign": blob, "corrections": corr}
    out = _product(z.shape, bound_m, "c2", comps, spatial)
    return out, decode(out)


def decode(blob: bytes) -> np.ndarray:
    """The field from a C1, C2 or C3 product's bytes alone (metres, float64)."""
    p = package.read(blob)
    side = p.rows
    c = p.components
    if p.foreign == "c2":
        codec = CODEC_NAMES[c["params"][0]]
        dec = foreign.decode(codec, c["foreign"], (p.rows, p.cols)).astype(np.float64)
        return corrections.apply_decoded(dec, c["corrections"])
    factor, base_id, res_id, flags, _ = _C1.unpack(c["params"])
    sub_side = (side - 1) // factor + 1
    up = upsample(foreign.decode(CODEC_NAMES[base_id], c["base"], (sub_side, sub_side)), factor, side)
    pred = up
    if flags & 1:
        n_classes = struct.unpack_from("<H", c["coefficients"])[0]
        beta = np.frombuffer(c["coefficients"], np.float32, offset=2).astype(np.float64)
        classes = None
        if "context" in c:
            raster = np.frombuffer(zstandard.ZstdDecompressor().decompress(c["context"]), np.uint8)
            n = int(round(np.sqrt(raster.size)))
            classes = package.context_nodes(raster.reshape(n, n), side)
        X = _features(up, classes, p.spacing_m, n_classes)
        pred = up + (X @ beta).reshape(up.shape)
    r = foreign.decode(CODEC_NAMES[res_id], c["residual"], (p.rows, p.cols)).astype(np.float64)
    return pred + r


def margin_mask(z: np.ndarray, bound_m: float, spacing_m: float, min_area_m2: float) -> np.ndarray:
    """Cells on flow paths (contributing area at least `min_area_m2`) whose steepest and second steepest drops
    differ by less than twice the bound, so an error within the bound could turn their flow direction."""
    routed = drainage.route(z, spacing_m)
    filled = routed["filled"]
    pad = np.pad(filled, 1, mode="edge")
    n = filled.shape[0]
    drops = []
    for dr, dc in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
        d = np.hypot(dr, dc)
        drops.append((filled - pad[1 + dr:1 + dr + n, 1 + dc:1 + dc + n]) / d)
    drops = np.sort(np.stack(drops), 0)
    margin = (drops[-1] - drops[-2]) * spacing_m
    return (margin < 2 * bound_m) & (routed["cells"] >= min_area_m2 / spacing_m ** 2)

