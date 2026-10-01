"""Non-neural comparators. Each one sees exactly the observations the network sees.

* Per-phase linear kernels: every fine node at phase (p, q) of the coarse lattice is a fixed linear combination
  of the 8 x 8 coarse nodes around it, constants passing through, least-squares fitted on training regions. It is
  the best any linear, shift-invariant upsampler can do on those regions, so a network gain over it is nonlinear.
* Regression-kriging, area to point: a trend over the interpolated base (its Laplacian, its slope and, where
  available, lithology one-hot intercepts) fitted on training regions; the trend residual is treated as a
  stationary field with an exponential covariance fitted on training regions, observed through the declared
  averaging operator, and kriged from the 64 nearest coarse residuals. On a regular lattice with a stationary
  covariance the kriging weights depend only on the phase, so the predictor is again a per-phase kernel, and
  its kriging variance gives a Gaussian interval.
* Biharmonic-regularised fit: argmin ||H z - y||^2 + alpha ||L z||^2 with the five-point Laplacian L, solved by
  conjugate gradients from the back-projected spline; alpha is chosen on the validation region.
* Harmonic and biharmonic infill of a square hole (exact sparse solves, Dirichlet data from one or two rings).
* Local universal kriging with a linear drift, for holes and scattered samples, with an optional declared
  observation-noise covariance. Two covariance families are fitted on training regions: exponential and
  Matern 3/2. At 10 m terrain is smooth beyond the cell scale; an exponential covariance is linear at the origin
  (Brownian), which turns kriging into a membrane fill, so the once-differentiable Matern is the main variant.
"""
from __future__ import annotations

import functools

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse import linalg as splinalg

K, LO = 8, 3  # coarse neighbourhood: offsets -3 .. 4 around the coarse node at or above-left of the target


# --- per-phase kernels ---------------------------------------------------------------------------------------

def _phase_nodes(coarse_side: int, factor: int, phase: int) -> np.ndarray:
    return np.arange(coarse_side if phase == 0 else coarse_side - 1)


def _neighbourhoods(padded: np.ndarray, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    offsets = np.arange(-LO, K - LO)
    r = rows[:, None, None] + K + offsets[None, :, None]
    c = columns[:, None, None] + K + offsets[None, None, :]
    return padded[r, c].reshape(len(rows), K * K)


def fit_phase_kernels(pairs, factor: int, max_per_phase: int = 200_000, ridge: float = 1e-6,
                      seed: int = 0) -> np.ndarray:
    """Least-squares kernels (f, f, K, K) with unit sum, from (coarse, fine) training pairs."""
    rng = np.random.default_rng(seed)
    gram = {}
    for coarse, fine in pairs:
        padded = np.pad(coarse, K, mode="reflect")
        n = coarse.shape[0]
        share = max(1, max_per_phase // len(pairs))
        for pr in range(factor):
            for pc in range(factor):
                jr, jc = _phase_nodes(n, factor, pr), _phase_nodes(coarse.shape[1], factor, pc)
                pick = rng.integers(0, len(jr) * len(jc), min(share, len(jr) * len(jc)))
                rows, columns = jr[pick // len(jc)], jc[pick % len(jc)]
                x = _neighbourhoods(padded, rows, columns)
                t = np.asarray(fine[factor * rows + pr, factor * columns + pc], dtype=np.float64)
                mean = x.mean(1, keepdims=True)
                a, b = gram.get((pr, pc), (0.0, 0.0))
                gram[(pr, pc)] = (a + (x - mean).T @ (x - mean), b + (x - mean).T @ (t - mean[:, 0]))
    kernels = np.zeros((factor, factor, K, K))
    for (pr, pc), (a, b) in gram.items():
        w = np.linalg.solve(a + ridge * np.trace(a) / a.shape[0] * np.eye(a.shape[0]), b)
        kernels[pr, pc] = (w + (1.0 - w.sum()) / w.size).reshape(K, K)
    return kernels


def apply_phase_kernels(coarse: np.ndarray, kernels: np.ndarray, factor: int) -> np.ndarray:
    """Fine field from per-phase kernels: node (f j + p, f k + q) = sum kernels[p, q] * coarse window."""
    coarse = np.asarray(coarse, dtype=np.float64)
    padded = np.pad(coarse, K, mode="reflect")
    n_r, n_c = coarse.shape
    out = np.zeros(((n_r - 1) * factor + 1, (n_c - 1) * factor + 1))
    shift = K + K // 2 - LO
    for pr in range(factor):
        for pc in range(factor):
            jr, jc = _phase_nodes(n_r, factor, pr), _phase_nodes(n_c, factor, pc)
            full = ndimage.correlate(padded, kernels[pr, pc], mode="constant")
            out[np.ix_(factor * jr + pr, factor * jc + pc)] = full[np.ix_(jr + shift, jc + shift)]
    return out


# --- regression-kriging, area to point -----------------------------------------------------------------------

def covariates(base: np.ndarray, slots: np.ndarray | None, n_slots: int) -> np.ndarray:
    """Trend covariates per fine node: Laplacian and slope of the base, then lithology intercepts or a constant."""
    gy, gx = np.gradient(base)
    p = np.pad(base, 1, mode="edge")
    lap = p[1:-1, 2:] + p[1:-1, :-2] + p[2:, 1:-1] + p[:-2, 1:-1] - 4.0 * base
    columns = [lap.ravel(), np.hypot(gx, gy).ravel()]
    if slots is None:
        columns.append(np.ones(base.size))
    else:
        columns += [(slots.ravel() == k).astype(np.float64) for k in range(n_slots)]
    return np.stack(columns, axis=1)


def fit_trend(samples, n_slots: int, geology: bool, per_region: int = 300_000, seed: int = 0) -> np.ndarray:
    """OLS trend of (reference - base) on `covariates`, from (base, reference, slots) training triples."""
    rng = np.random.default_rng(seed)
    xs, ts = [], []
    for base, reference, slots in samples:
        pick = rng.choice(base.size, min(per_region, base.size), replace=False)
        x = covariates(base, slots if geology else None, n_slots)[pick]
        xs.append(x)
        ts.append((reference - base).ravel()[pick])
    x, t = np.concatenate(xs), np.concatenate(ts)
    return np.linalg.lstsq(x.T @ x + 1e-8 * np.eye(x.shape[1]), x.T @ t, rcond=None)[0]


def empirical_covariance(fields, max_lag: int = 24) -> np.ndarray:
    """Radially averaged autocovariance (lags 0..max_lag, fine nodes) of zero-mean residual fields, region-balanced."""
    out = []
    for field in fields:
        f = np.asarray(field, dtype=np.float64)
        f = f - f.mean()
        shape = [2 * s for s in f.shape]
        spectrum = np.fft.rfft2(f, shape)
        auto = np.fft.irfft2(spectrum * np.conj(spectrum), shape)
        counts = np.fft.irfft2(np.abs(np.fft.rfft2(np.ones_like(f), shape)) ** 2, shape)
        auto = np.fft.fftshift(auto / np.maximum(counts, 1.0))
        cy, cx = shape[0] // 2, shape[1] // 2
        yy, xx = np.mgrid[-max_lag:max_lag + 1, -max_lag:max_lag + 1]
        r = np.rint(np.hypot(yy, xx)).astype(int)
        window = auto[cy - max_lag:cy + max_lag + 1, cx - max_lag:cx + max_lag + 1]
        out.append(np.array([window[r == h].mean() for h in range(max_lag + 1)]))
    return np.mean(out, axis=0)


def fit_exponential(covariance: np.ndarray, fit_lags: int, ranges=None) -> dict:
    """sill * exp(-h / range), no nugget, least squares over lags 0..fit_lags.

    The trend residual is correlated over about one coarse cell and has a hole effect beyond it, which no
    exponential follows; fitting where the correlation lives (up to one coarse cell) is the declared choice.
    """
    lags = np.arange(fit_lags + 1, dtype=np.float64)
    target = covariance[:fit_lags + 1]
    ranges = np.geomspace(0.2, 50.0, 200) if ranges is None else ranges
    best = None
    for rho in ranges:
        basis = np.exp(-lags / rho)
        sill = max(float(basis @ target / (basis @ basis)), 1e-12)
        err = float(((sill * basis - target) ** 2).sum())
        if best is None or err < best[0]:
            best = (err, rho, sill)
    _, rho, sill = best
    return {"sill": sill, "rangeCells": float(rho), "nugget": 0.0, "fitLags": int(fit_lags)}


def _point_covariance(model: dict, radius: int) -> np.ndarray:
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    c = model["sill"] * np.exp(-np.hypot(yy, xx) / model["rangeCells"])
    c[radius, radius] += model["nugget"]
    return c


def atpk_kernels(model: dict, row_weights: np.ndarray, column_weights: np.ndarray, factor: int):
    """Simple-kriging weights (f, f, K, K) from 64 coarse areal residuals to a point, and the kriging variance.

    `row_weights`/`column_weights` are the declared operator's interior axis weights (offsets -h .. h).
    """
    h = (len(row_weights) - 1) // 2
    w2 = np.outer(row_weights, column_weights)
    radius = (K + 1) * factor + 2 * h + 2
    point = _point_covariance(model, radius)
    point_area = ndimage.correlate(point, w2, mode="constant")
    area_area = ndimage.correlate(point_area, w2[::-1, ::-1], mode="constant")
    offsets = np.arange(-LO, K - LO)
    grid = [(a, b) for a in offsets for b in offsets]
    caa = np.array([[area_area[radius + (a - a2) * factor, radius + (b - b2) * factor] for a2, b2 in grid]
                    for a, b in grid])
    kernels = np.zeros((factor, factor, K, K))
    variance = np.zeros((factor, factor))
    for pr in range(factor):
        for pc in range(factor):
            cpa = np.array([point_area[radius + pr - a * factor, radius + pc - b * factor] for a, b in grid])
            lam = np.linalg.solve(caa, cpa)
            kernels[pr, pc] = lam.reshape(K, K)
            variance[pr, pc] = point[radius, radius] - lam @ cpa
    return kernels, np.maximum(variance, 0.0)


# --- regularised fit -----------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=8)
def _laplacian(rows: int, columns: int):
    """Five-point Laplacian with mirrored (zero-flux) edges, as a sparse matrix on the flattened grid."""
    def axis(n):
        main = -2.0 * np.ones(n)
        off = np.ones(n - 1)
        m = sparse.diags([off, main, off], [-1, 0, 1], format="lil")
        m[0, 1], m[n - 1, n - 2] = 2.0, 2.0
        return m.tocsr()
    return (sparse.kron(axis(rows), sparse.identity(columns)) +
            sparse.kron(sparse.identity(rows), axis(columns))).tocsr()


def biharmonic_fit(y: np.ndarray, operator, shape, alpha: float, start: np.ndarray, rtol: float = 1e-7,
                   maxiter: int = 3000):
    """argmin ||H z - y||^2 + alpha ||L z||^2 by conjugate gradients; returns (field, report)."""
    rows, columns = operator.matrices(shape)
    lap = _laplacian(*shape)
    biharmonic = (lap.T @ lap).tocsr()

    def normal(v):
        z = v.reshape(shape)
        hz = np.asarray((columns @ (rows @ z).T).T)
        back = np.asarray((columns.T @ (rows.T @ hz).T).T)
        return back.ravel() + alpha * (biharmonic @ v)

    rhs = np.asarray((columns.T @ (rows.T @ np.asarray(y, dtype=np.float64)).T).T).ravel()
    system = splinalg.LinearOperator((rhs.size, rhs.size), matvec=normal, dtype=np.float64)
    count = [0]
    solution, info = splinalg.cg(system, rhs, x0=np.asarray(start, dtype=np.float64).ravel(), rtol=rtol,
                                 maxiter=maxiter, callback=lambda _: count.__setitem__(0, count[0] + 1))
    return solution.reshape(shape), {"alpha": alpha, "iterations": count[0], "converged": info == 0,
                                     "rtol": rtol}


# --- holes ---------------------------------------------------------------------------------------------------

RING = 2


@functools.lru_cache(maxsize=8)
def infill_matrix(size: int, kind: str):
    """Dense map from known ring values to a size x size hole in a (size + 4)^2 window: (G, known flat index).

    Harmonic: five-point Laplace = 0 in the hole, Dirichlet data from the first ring. Biharmonic: squared
    Laplacian = 0 in the hole, data from two rings. Exact sparse LU solves.
    """
    s = size + 2 * RING
    index = np.arange(s * s).reshape(s, s)
    rows, cols, vals = [], [], []
    for r in range(1, s - 1):
        for c in range(1, s - 1):
            i = index[r, c]
            rows += [i] * 5
            cols += [i, index[r - 1, c], index[r + 1, c], index[r, c - 1], index[r, c + 1]]
            vals += [-4.0, 1.0, 1.0, 1.0, 1.0]
    lap = sparse.csr_matrix((vals, (rows, cols)), shape=(s * s, s * s))
    hole = np.zeros((s, s), bool)
    hole[RING:-RING, RING:-RING] = True
    ring1 = np.zeros((s, s), bool)
    ring1[RING - 1:s - RING + 1, RING - 1:s - RING + 1] = True
    ring1 &= ~hole
    matrix, known = {"harmonic": (lap, index[ring1]), "biharmonic": ((lap @ lap).tocsr(), index[~hole])}[kind]
    h = index[hole]
    a = matrix[h][:, h].tocsc()
    b = matrix[h][:, known].toarray()
    return -splinalg.splu(a).solve(b), known


def fill_hole(window: np.ndarray, row: int, column: int, size: int, kind: str) -> np.ndarray:
    """Fill window[row:row+size, column:column+size] from its rings; the hole's own values are never read."""
    g, known = infill_matrix(size, kind)
    s = size + 2 * RING
    sub = window[row - RING:row - RING + s, column - RING:column - RING + s].ravel()[known]
    mean = sub.mean()
    out = np.array(window, dtype=np.float64, copy=True)
    out[row:row + size, column:column + size] = (g @ (sub - mean) + mean).reshape(size, size)
    return out


# --- local kriging -------------------------------------------------------------------------------------------

def semivariogram(fields, window: int = 160, max_lag: int = 64, per_field: int = 40, seed: int = 0) -> np.ndarray:
    """Region-balanced semivariogram (lags 0..max_lag, cells, along both axes) of locally detrended windows."""
    rng = np.random.default_rng(seed)
    out = []
    yy, xx = np.mgrid[0:window, 0:window]
    design = np.stack([np.ones(window * window), yy.ravel(), xx.ravel()], 1)
    for field in fields:
        gam = np.zeros(max_lag + 1)
        for _ in range(per_field):
            r, c = rng.integers(0, field.shape[0] - window, 2)
            w = field[r:r + window, c:c + window].ravel()
            w = (w - design @ np.linalg.lstsq(design, w, rcond=None)[0]).reshape(window, window)
            for h in range(1, max_lag + 1):
                gam[h] += 0.25 * (np.mean((w[h:] - w[:-h]) ** 2) + np.mean((w[:, h:] - w[:, :-h]) ** 2))
        out.append(gam / per_field)
    return np.mean(out, axis=0)


def correlation(h: np.ndarray, model: dict) -> np.ndarray:
    """Correlation at distance h (cells) of the fitted family: exponential, or Matern with smoothness 3/2."""
    if model.get("family", "exponential") == "matern32":
        a = np.sqrt(3.0) * h / model["rangeCells"]
        return (1.0 + a) * np.exp(-a)
    return np.exp(-h / model["rangeCells"])


def fit_variogram(gamma: np.ndarray, family: str = "exponential") -> dict:
    """nugget + sill * (1 - correlation(h)) to lags >= 1, least squares in relative error over a range grid.

    Relative error weights every lag alike, so the short lags that decide a local interpolation are not
    swamped by the large semivariances at long lags.
    """
    lags = np.arange(len(gamma), dtype=np.float64)[1:]
    w = 1.0 / np.maximum(gamma[1:], 1e-12)
    best = None
    for rho in np.geomspace(1.0, 2000.0, 150):
        x = np.stack([np.ones_like(lags), 1.0 - correlation(lags, {"family": family, "rangeCells": rho})], 1)
        coef = np.linalg.lstsq(x * w[:, None], gamma[1:] * w, rcond=None)[0]
        nugget, sill = max(coef[0], 0.0), max(coef[1], 1e-9)
        err = float((((nugget + sill * x[:, 1] - gamma[1:]) * w) ** 2).sum())
        if best is None or err < best[0]:
            best = (err, rho, sill, nugget)
    return {"family": family, "sill": best[2], "rangeCells": best[1], "nugget": best[3]}


def universal_kriging(points: np.ndarray, values: np.ndarray, targets: np.ndarray, model: dict,
                      noise=None, local_sill: bool = True, device=None):
    """Kriging with a linear drift from `points` (n,2 in cells) to `targets` (m,2); returns (mean, variance).

    `noise(points)` returns the declared observation-noise covariance added to the data covariance. With
    `local_sill` the variogram is rescaled so that it matches these data at their shortest separations (the
    closest tenth of point pairs, noise semivariance removed), so a flat neighbourhood is not given the
    interval of the steep regions the model was fitted on; the shape (family, range, nugget ratio) stays the
    one fitted on training regions. With a torch `device` the same algebra runs there in float64.
    """
    n = len(points)
    drift = np.column_stack([np.ones(n), points])
    scale = 1.0
    if local_sill and n > 6:
        i, j = np.triu_indices(n, 1)
        d = np.hypot(*(points[i] - points[j]).T)
        near = d <= max(np.quantile(d, 0.1), 1.5)
        empirical = 0.5 * np.mean((values[i[near]] - values[j[near]]) ** 2)
        if noise is not None:
            c_noise = noise(points)
            diagonal = np.diag(c_noise)
            empirical -= np.mean(0.5 * (diagonal[i[near]] + diagonal[j[near]]) - c_noise[i[near], j[near]])
        modelled = np.mean(model["nugget"] + model["sill"] * (1.0 - correlation(d[near], model)))
        scale = float(np.clip(empirical / max(modelled, 1e-12), 1e-3, 1e3))
    if device is None:
        def cov(a, b):
            d = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])
            return model["sill"] * correlation(d, model)
        c = cov(points, points) * scale + model["nugget"] * scale * np.eye(n)
        if noise is not None:
            c = c + noise(points)
        system = np.zeros((n + 3, n + 3))
        system[:n, :n], system[:n, n:], system[n:, :n] = c, drift, drift.T
        rhs = np.zeros((n + 3, len(targets)))
        rhs[:n] = cov(points, targets) * scale
        rhs[n:] = np.column_stack([np.ones(len(targets)), targets]).T
        weights = np.linalg.solve(system, rhs)
        mean = weights[:n].T @ values
        variance = (model["sill"] + model["nugget"]) * scale - np.einsum("ij,ij->j", weights, rhs)
        return mean, np.maximum(variance, 0.0)
    import torch
    t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64), device=device)
    p, q = t(points), t(targets)

    def cov_t(a, b):
        d = torch.cdist(a, b)
        if model.get("family", "exponential") == "matern32":
            u = np.sqrt(3.0) * d / model["rangeCells"]
            return model["sill"] * (1.0 + u) * torch.exp(-u)
        return model["sill"] * torch.exp(-d / model["rangeCells"])
    c = cov_t(p, p) * scale + model["nugget"] * scale * torch.eye(n, dtype=torch.float64, device=device)
    if noise is not None:
        c = c + t(noise(points))
    system = torch.zeros((n + 3, n + 3), dtype=torch.float64, device=device)
    dr = t(drift)
    system[:n, :n], system[:n, n:], system[n:, :n] = c, dr, dr.T
    rhs = torch.zeros((n + 3, len(targets)), dtype=torch.float64, device=device)
    rhs[:n] = cov_t(p, q) * scale
    rhs[n] = 1.0
    rhs[n + 1:] = q.T
    weights = torch.linalg.solve(system, rhs)
    mean = weights[:n].T @ t(values)
    variance = (model["sill"] + model["nugget"]) * scale - (weights * rhs).sum(0)
    return mean.cpu().numpy(), np.maximum(variance.cpu().numpy(), 0.0)
