"""Observation operators and interpolators, all as separable sparse matrices.

An observation operator H maps a fine lattice to a coarse one whose node j sits on fine node j*f. Every operator
here is separable (one weight matrix per axis), applied as two sparse products, and identified by a fingerprint
of the weights it actually applies and of the code that builds them, never by its name.

The declared operator is the trapezoidal node average of `superres.observe`. The others exist because a model
trained on one coarsening and scored on the same one may have learned the kernel rather than the terrain:

* `gaussian-s`: a Gaussian average of width s coarse cells, centred on the node.
* `decimate`: point samples, no averaging at all.
* `provider-style`: a Gaussian average with a registration offset, the shape that best explains the provider's
  own 10 m product from its 1 m product (fitted by `fit_provider_style`; about 0.55 coarse cells wide and
  displaced 0.3 of a cell (3 m) north and 0.1 of a cell east). It reproduces the systematic part of that
  difference only.
* `box-offset`: f samples averaged with the window displaced by half a fine sample, the start-at-node block
  average that `superres._axis_weights` warns about. Never used in training.

Interpolators map coarse to fine: `bilinear`, `keys` (cubic convolution, a = -1/2, the usual bicubic) and
`bspline` (interpolating cubic B-spline). All mirror the coarse grid at its edges and are local, so a window of
an interpolated field equals the same window of the whole-field result.
"""
from __future__ import annotations

import functools
import hashlib
import inspect

import numpy as np
from scipy import ndimage, sparse

from geoneural.superres import superres

#: Gaussian width (coarse cells) and offset (coarse cells, rows then columns) of the provider-style operator,
#: from `fit_provider_style` on the three development regions with 1 m data. Rows grow southwards, so a
#: negative row offset is a displacement to the north.
PROVIDER_STYLE = {"sigmaCells": 0.55, "shiftCells": (-0.3, 0.1)}
TRAIN_GAUSSIANS = (0.3, 0.45, 0.6, 0.8)
#: Operators scored for every method. `box-offset` and `gaussian-1.0` never enter training.
EVALUATION = ("trapezoid", "gaussian-0.5", "decimate", "provider-style", "box-offset", "gaussian-1.0")
HELD_OUT = ("box-offset", "gaussian-1.0")


def _renormalised(weights: np.ndarray) -> np.ndarray:
    total = weights.sum(axis=1, keepdims=True)
    return weights / np.where(total > 0, total, 1.0)


def trapezoid_axis(side: int, factor: int, shift: float = 0.0) -> np.ndarray:
    return superres._axis_weights(side, factor)


def gaussian_axis(side: int, factor: int, sigma: float = 0.5, shift: float = 0.0) -> np.ndarray:
    """Gaussian weights of width `sigma` coarse cells centred `shift` cells from the node, cut at 4 sigma."""
    nodes = (side - 1) // factor + 1
    centre = (np.arange(nodes)[:, None] + shift) * factor
    offset = np.arange(side)[None, :] - centre
    width = sigma * factor
    weights = np.where(np.abs(offset) <= 4.0 * width, np.exp(-0.5 * (offset / width) ** 2), 0.0)
    return _renormalised(weights)


def decimate_axis(side: int, factor: int, shift: float = 0.0) -> np.ndarray:
    nodes = (side - 1) // factor + 1
    weights = np.zeros((nodes, side))
    weights[np.arange(nodes), np.arange(nodes) * factor] = 1.0
    return weights


def box_offset_axis(side: int, factor: int, shift: float = 0.0) -> np.ndarray:
    """Equal weights on fine nodes j*f - f/2 .. j*f + f/2 - 1: a block average displaced half a fine sample."""
    nodes = (side - 1) // factor + 1
    weights = np.zeros((nodes, side))
    half = factor // 2
    for j in range(nodes):
        lo, hi = max(j * factor - half, 0), min(j * factor + factor - half - 1, side - 1)
        weights[j, lo:hi + 1] = 1.0
    return _renormalised(weights)


class Operator:
    """One separable observation operator: weights per axis, observation, back-projection and identity."""

    def __init__(self, name: str, factor: int, axis, params: dict | None = None):
        self.name, self.factor, self.axis, self.params = name, int(factor), axis, dict(params or {})
        self._cache: dict = {}

    def _weights(self, side: int, which: int) -> np.ndarray:
        shift = tuple(self.params.get("shiftCells", (0.0, 0.0)))[which]
        extra = {k: v for k, v in self.params.items() if k not in ("shiftCells",)}
        if "sigmaCells" in extra:
            extra = {"sigma": extra["sigmaCells"]}
        return self.axis(side, self.factor, shift=shift, **extra)

    def matrices(self, shape):
        key = tuple(shape)
        if key not in self._cache:
            self._cache[key] = (sparse.csr_matrix(self._weights(shape[0], 0)),
                                sparse.csr_matrix(self._weights(shape[1], 1)))
        return self._cache[key]

    def observe(self, fine: np.ndarray) -> np.ndarray:
        rows, columns = self.matrices(np.shape(fine))
        return np.asarray((columns @ (rows @ np.asarray(fine, dtype=np.float64)).T).T)

    def project(self, estimate: np.ndarray, coarse: np.ndarray, tolerance: float = 1e-4,
                max_iterations: int = 500):
        """Back-projection to `tolerance` under this operator; returns (field, report)."""
        return superres.back_project(estimate, coarse, self.factor, tolerance, max_iterations,
                                     observe_fn=self.observe)

    def fingerprint(self) -> str:
        digest = hashlib.sha256(f"{self.factor}".encode())
        for side in (4 * self.factor + 1, 5 * self.factor + 3):
            for which in (0, 1):
                digest.update(np.ascontiguousarray(self._weights(side, which), dtype="<f8").tobytes())
        probe = np.random.default_rng(1729).standard_normal((3 * self.factor + 2, 4 * self.factor + 1))
        digest.update(np.ascontiguousarray(np.round(Operator(self.name, self.factor, self.axis, self.params)
                                                    .observe(probe), 10), dtype="<f8").tobytes())
        for function in (self.axis, Operator.observe, Operator._weights):
            digest.update(inspect.getsource(function).encode())
        return digest.hexdigest()[:16]

    def schema(self) -> dict:
        return {"name": self.name, "factor": self.factor, "params": {k: list(v) if isinstance(v, tuple) else v
                                                                     for k, v in self.params.items()},
                "separable": True, "fingerprint": self.fingerprint()}


def make(name: str, factor: int) -> Operator:
    """Operator by name: trapezoid, decimate, box-offset, provider-style or gaussian-<width in cells>."""
    if name == "trapezoid":
        return Operator(name, factor, trapezoid_axis)
    if name == "decimate":
        return Operator(name, factor, decimate_axis)
    if name == "box-offset":
        return Operator(name, factor, box_offset_axis)
    if name == "provider-style":
        return Operator(name, factor, gaussian_axis, {"sigmaCells": PROVIDER_STYLE["sigmaCells"],
                                                      "shiftCells": tuple(PROVIDER_STYLE["shiftCells"])})
    if name.startswith("gaussian-"):
        return Operator(name, factor, gaussian_axis, {"sigmaCells": float(name.split("-", 1)[1])})
    raise ValueError(f"unknown operator {name}")


def training_mix(factor: int) -> list[tuple[Operator, float]]:
    """The operators a mixed-operator model is trained on, with sampling weights (a quarter per family)."""
    gaussians = [make(f"gaussian-{s}", factor) for s in TRAIN_GAUSSIANS]
    return ([(make("trapezoid", factor), 0.25), (make("decimate", factor), 0.25),
             (make("provider-style", factor), 0.25)] + [(g, 0.25 / len(gaussians)) for g in gaussians])


# --- interpolation -------------------------------------------------------------------------------------------

def _mirror(index: np.ndarray, side: int) -> np.ndarray:
    if side == 1:
        return np.zeros_like(index)
    period = 2 * (side - 1)
    index = np.abs(index) % period
    return np.where(index >= side, period - index, index)


def _keys(u: np.ndarray) -> np.ndarray:
    """Cubic convolution weights (a = -1/2) for taps at -1, 0, 1, 2 around fractional position u."""
    a = -0.5
    d = np.stack([1 + u, u, 1 - u, 2 - u], axis=-1)
    near = (a + 2) * d ** 3 - (a + 3) * d ** 2 + 1
    far = a * d ** 3 - 5 * a * d ** 2 + 8 * a * d - 4 * a
    return np.where(d <= 1, near, far)


def _bspline(u: np.ndarray) -> np.ndarray:
    """Cubic B-spline basis for taps at -1, 0, 1, 2."""
    return np.stack([(1 - u) ** 3 / 6, (3 * u ** 3 - 6 * u ** 2 + 4) / 6,
                     (-3 * u ** 3 + 3 * u ** 2 + 3 * u + 1) / 6, u ** 3 / 6], axis=-1)


@functools.lru_cache(maxsize=32)
def axis_interpolation(kind: str, coarse_side: int, factor: int):
    """Sparse (fine x coarse) matrix of one axis of an interpolator, mirrored at the edges."""
    if kind == "bilinear":
        return superres._bilinear_axis(coarse_side, factor)
    fine_side = (coarse_side - 1) * factor + 1
    position = np.arange(fine_side) / factor
    base = np.minimum(np.floor(position).astype(np.int64), coarse_side - 1)
    u = position - base
    taps = {"keys": _keys, "bspline": _bspline}[kind](u)
    columns = _mirror(base[:, None] + np.arange(-1, 3)[None, :], coarse_side)
    rows = np.repeat(np.arange(fine_side), 4)
    return sparse.csr_matrix((taps.ravel(), (rows, columns.ravel())), shape=(fine_side, coarse_side))


def coefficients(coarse: np.ndarray, kind: str) -> np.ndarray:
    """What the interpolation matrices act on: the grid itself, or its B-spline coefficients."""
    coarse = np.asarray(coarse, dtype=np.float64)
    if kind == "bspline":
        return ndimage.spline_filter(coarse, order=3, mode="mirror")
    return coarse


def upsample(coarse: np.ndarray, factor: int, kind: str = "keys") -> np.ndarray:
    c = coefficients(coarse, kind)
    rows = axis_interpolation(kind, c.shape[0], factor)
    columns = axis_interpolation(kind, c.shape[1], factor)
    return np.asarray((columns @ (rows @ c).T).T)


def upsample_window(coeff: np.ndarray, factor: int, kind: str, rows: slice, columns: slice) -> np.ndarray:
    """Fine nodes rows x columns of `upsample`, from the whole-field coefficients; identical to cropping."""
    r = axis_interpolation(kind, coeff.shape[0], factor)[rows]
    c = axis_interpolation(kind, coeff.shape[1], factor)[columns]
    return np.asarray((c @ (r @ coeff).T).T)


# --- the provider's own coarsening ----------------------------------------------------------------------------

def fit_provider_style(pairs, sigmas=(0.4, 0.45, 0.5, 0.55, 0.6, 0.7), shifts=np.arange(-0.4, 0.21, 0.05),
                       window=(2000, 6001)) -> dict:
    """Gaussian width and offset that best map 1 m data onto the provider's 10 m grid.

    `pairs` yields (region, fine 1 m memmap, provider 10 m grid). A square of each region (`window`, fine
    nodes) is coarsened with every candidate and compared with the provider grid; the candidate with the
    lowest mean absolute difference pooled over regions wins. What it leaves over is reported: it is the part
    of the provider's difference no linear shift-invariant operator explains (survey epoch, filtering).
    """
    lo, hi = window
    rows = []
    for region, fine, provider in pairs:
        sub = np.asarray(fine[lo:hi, lo:hi], dtype=np.float64)
        target = np.asarray(provider[lo // 10:(hi - 1) // 10 + 1, lo // 10:(hi - 1) // 10 + 1], dtype=np.float64)
        inner = (slice(4, -4), slice(4, -4))
        trapezoid = superres.observe(sub, 10)
        row = {"region": region, "trapezoidMaeM": float(np.abs(trapezoid - target)[inner].mean()), "grid": {}}
        for sigma in sigmas:
            for sy in shifts:
                ry = sparse.csr_matrix(gaussian_axis(sub.shape[0], 10, sigma, sy))
                half = ry @ sub
                for sx in shifts:
                    rx = sparse.csr_matrix(gaussian_axis(sub.shape[1], 10, sigma, sx))
                    out = np.asarray((rx @ half.T).T)
                    row["grid"][(sigma, round(float(sy), 2), round(float(sx), 2))] = \
                        float(np.abs(out - target)[inner].mean())
        rows.append(row)
    keys = rows[0]["grid"].keys()
    pooled = {k: float(np.mean([r["grid"][k] for r in rows])) for k in keys}
    best = min(pooled, key=pooled.get)
    return {"sigmaCells": best[0], "shiftCells": [best[1], best[2]], "pooledMaeM": pooled[best],
            "byRegion": {r["region"]: {"trapezoidMaeM": r["trapezoidMaeM"], "fittedMaeM": r["grid"][best]}
                         for r in rows},
            "window": list(window)}
