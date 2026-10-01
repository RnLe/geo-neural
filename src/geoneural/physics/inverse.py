"""Inferring landscape history, and what a surface cannot tell you.

The attractive claim here is that a present-day surface identifies the history
that made it. Much of that claim is false for structural reasons, and this
module shows which part before fitting anything, because a posterior over
unidentifiable parameters looks exactly like a posterior over identifiable
ones.

The identifiability statement is arithmetic. The stream-power system obeys a
similarity law: scaling (U, K, D) -> (c U, c K, c D) and t -> t / c leaves the
entire trajectory unchanged. From a terminal surface with a known initial
condition the data can constrain at most the products U t, K t and D t
(equivalently U / K, D / K and K t), never the rates and the duration
separately. In the four coordinates (log U, log K, log D, log t) this is a
one-dimensional common-scaling ridge along (1, 1, 1, -1); further
degeneracies (an unknown initial surface, base-level history or near
equilibrium) can only add to it.

Observation designs, all at absolute epochs:

* terminal: the surface at t;
* fractional (control): the surfaces at alpha t and t. Both epochs move with
  t under the rescaling, so the ridge survives exactly; this design is kept
  to show that, not as a way to break it;
* fixed lag: the surfaces at t and t + Delta with Delta fixed in years. The
  ridge maps t + Delta to t / c + Delta, not to (t + Delta) / c, so a
  measurable change between the epochs can identify c; near equilibrium the
  change is too small and the ridge returns.

Noise is drawn from the declared covariance (exponential kernel, sigma and
correlation length in cells, independent between epochs) as L z with L its
Cholesky factor: no per-draw mean removal and no normalisation by the sample
standard deviation, which would make the noise something other than the
Gaussian the likelihood assumes. The likelihood uses the same covariance by
Cholesky, with the log determinant, and optionally an unknown datum offset per
epoch (Gaussian prior, marginalised analytically by adding tau^2 1 1^T to that
epoch's covariance).
"""
from __future__ import annotations

import functools
import math
import multiprocessing
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from geoneural.metrics import hydrology_fast
from geoneural.physics import landscape, units

SCHEMA = "geoneural-inverse-history-v2"

SIDE, SPACING_M = 32, 50.0
BASE = {"uplift": 1e-4, "kIncision": 1e-5, "diffusivity": 1e-2}
DT_YEARS = 400.0
INITIAL_RELIEF_M = 15.0
INITIAL_SEED = 1729
SIGMA_M, LENGTH_CELLS, DATUM_SD_M = 0.5, 3.0, 5.0
FRACTION = 0.4
LAGS_YEARS = (20_000.0, 50_000.0)
TRUTHS = {"transient": 5e5, "late": 2e6, "near-equilibrium": 5e6}
C_COARSE = (0.5, 0.75, 1.0, 1.5, 2.0)
C_FINE = tuple(sorted(set([round(2.0 ** (-1 + k / 12), 6) for k in range(25)] + list(C_COARSE))))
# The ratio problem: theta = (log10 U/K, log10 D/K) with K and t known, so K t is known.
RATIO_NAMES = ("logUpliftOverIncision", "logDiffusivityOverIncision")
RATIO_TRUTH = (math.log10(BASE["uplift"] / BASE["kIncision"]),
               math.log10(BASE["diffusivity"] / BASE["kIncision"]))
RATIO_BOX = ((0.5, 1.5), (2.5, 3.5))
RATIO_YEARS = 5e5
PARAMETERS = ("logUplift", "logIncision", "logDiffusivity", "logYears", "datum")


def ridge_direction() -> dict:
    """The exact null direction of a terminal-surface likelihood.

    Under (U, K, D, t) -> (c U, c K, c D, t / c) the surface is unchanged, so in
    log coordinates the direction (+1, +1, +1, -1) normalised is a flat
    direction by construction: a one-dimensional ridge in the four rate and
    time coordinates. Nothing was fitted to obtain this.
    """
    vector = np.array([1.0, 1.0, 1.0, -1.0, 0.0])
    return {"direction": (vector / np.linalg.norm(vector)).tolist(),
            "parameters": list(PARAMETERS), "dimension": 1,
            "identified": ["logUplift + logYears", "logIncision + logYears",
                           "logDiffusivity + logYears"],
            "notIdentified": ["the common scale c along the direction, from a terminal or fractional design"],
            "atSteadyState": ["logFluvialNumber", "logHillslopeNumber"],
            "note": "A one-dimensional common-scaling ridge in four rate and time coordinates, "
                    "before any further degeneracy (unknown initial surface, base level, "
                    "equilibrium). An 'inferred age' read off it reports the prior."}


@functools.lru_cache(maxsize=8)
def initial_surface(side: int = SIDE, seed: int = INITIAL_SEED, relief_m: float = INITIAL_RELIEF_M) -> np.ndarray:
    """Red noise (amplitude proportional to 1/k), the known initial condition of every run."""
    rng = np.random.default_rng(seed)
    spectrum = np.fft.rfft2(rng.normal(0.0, 1.0, (side, side)))
    ky = np.fft.fftfreq(side)[:, None]
    kx = np.fft.rfftfreq(side)[None, :]
    k = np.sqrt(ky * ky + kx * kx)
    k[0, 0] = 1.0
    field = np.fft.irfft2(spectrum / k, s=(side, side))
    out = relief_m * field / max(float(field.std()), 1e-9)
    out.flags.writeable = False
    return out


class Observation:
    """Interior nodes of each epoch, with correlated Gaussian error of a declared covariance."""

    def __init__(self, side: int = SIDE, crop: int = 1, sigma_m: float = SIGMA_M,
                 length_cells: float = LENGTH_CELLS, datum_sd_m: float = DATUM_SD_M):
        self.side, self.crop = int(side), int(crop)
        self.sigma_m, self.length_cells, self.datum_sd_m = float(sigma_m), float(length_cells), float(datum_sd_m)
        inner = self.side - 2 * self.crop
        rows, cols = np.meshgrid(np.arange(inner), np.arange(inner), indexing="ij")
        points = np.stack([rows.ravel(), cols.ravel()], axis=1).astype(np.float64)
        distance = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(-1))
        self.covariance = self.sigma_m ** 2 * np.exp(-distance / self.length_cells)
        self.covariance += 1e-10 * np.eye(len(points))
        self.size = len(points)
        self.lower = np.linalg.cholesky(self.covariance)
        self._factors = {}

    def apply(self, surface: np.ndarray) -> np.ndarray:
        c = self.crop
        return np.asarray(surface, dtype=np.float64)[c:self.side - c, c:self.side - c].reshape(-1).copy()

    def draw(self, rng) -> np.ndarray:
        """One noise vector, exactly N(0, C): L z, nothing removed, nothing rescaled."""
        return self.lower @ rng.normal(0.0, 1.0, self.size)

    def factor(self, datum: bool):
        if datum not in self._factors:
            cov = self.covariance + (self.datum_sd_m ** 2 if datum else 0.0)
            chol = cho_factor(cov, lower=True)
            self._factors[datum] = (chol, 2.0 * float(np.log(np.diag(chol[0])).sum()))
        return self._factors[datum]

    def loglik(self, residuals, datum: bool = False) -> float:
        """Sum over epochs of the Gaussian log density of each residual vector, with constants."""
        chol, logdet = self.factor(datum)
        total = 0.0
        for r in residuals:
            total += -0.5 * float(r @ cho_solve(chol, r)) - 0.5 * logdet - 0.5 * r.size * math.log(2 * math.pi)
        return total

    def loglik_iid(self, residuals) -> float:
        """The misspecified scalar likelihood an earlier version used: independent errors."""
        return float(sum(-0.5 * float(r @ r) / self.sigma_m ** 2 for r in residuals))

    def chi2(self, residuals, datum: bool = False) -> float:
        chol, _ = self.factor(datum)
        return float(sum(float(r @ cho_solve(chol, r)) for r in residuals))

    def record(self) -> dict:
        return {"side": self.side, "crop": self.crop, "observedPerEpoch": self.size,
                "sigmaM": self.sigma_m, "lengthCells": self.length_cells,
                "kernel": "sigma^2 exp(-d / length), d in cells, independent between epochs",
                "datumSdM": self.datum_sd_m,
                "noise": "L z with L the Cholesky factor of the same covariance; no mean removal, "
                         "no per-draw normalisation"}


def dense_loglik(residual: np.ndarray, covariance: np.ndarray) -> float:
    """The textbook formula with an explicit inverse and determinant, for checking `loglik`."""
    sign, logdet = np.linalg.slogdet(covariance)
    return float(-0.5 * residual @ np.linalg.inv(covariance) @ residual - 0.5 * logdet
                 - 0.5 * residual.size * math.log(2 * math.pi))


def old_noise(shape, rng, sigma_m: float = SIGMA_M, length_cells: float = LENGTH_CELLS) -> np.ndarray:
    """The noise recipe of version 1, kept only to measure what it did to coverage.

    A Gaussian-filtered white field with its mean removed and divided by its
    own sample standard deviation: not a draw from the covariance the
    likelihood assumes.
    """
    white = rng.normal(0.0, 1.0, shape)
    spectrum = np.fft.rfft2(white)
    ky = np.fft.fftfreq(shape[0])[:, None]
    kx = np.fft.rfftfreq(shape[1])[None, :]
    kernel = np.exp(-0.5 * (np.sqrt(ky * ky + kx * kx) * length_cells * 2 * np.pi) ** 2)
    spectrum[0, 0] = 0.0
    field = np.fft.irfft2(spectrum * kernel, s=shape)
    field = field - field.mean()
    return sigma_m * field / max(float(field.std()), 1e-12)


def design_epochs(design: str, years: float, lag: float | None = None) -> list[float]:
    if design == "terminal":
        return [years]
    if design == "fractional":
        return [FRACTION * years, years]
    if design == "fixed-lag":
        return [years, years + float(lag)]
    raise ValueError(f"unknown design {design!r}")


def _fastscape_snapshots(start, rates, epochs, dt):
    import fastscapelib as fs
    rows, cols = start.shape
    grid = fs.RasterGrid([rows, cols], [SPACING_M, SPACING_M], fs.NodeStatus.FIXED_VALUE)
    graph = fs.FlowGraph(grid, [fs.PFloodSinkResolver(), fs.SingleFlowRouter()])
    spl = fs.SPLEroder(graph, k_coef=rates["kIncision"], area_exp=0.5, slope_exp=1.0, tolerance=1e-8)
    diffusion = fs.DiffusionADIEroder(grid, rates["diffusivity"])
    core = np.zeros(start.shape, dtype=bool)
    core[1:-1, 1:-1] = True
    z, now, out = np.array(start, dtype=np.float64), 0.0, {}
    for target in sorted(epochs):
        for h in landscape.step_plan(target - now, dt):
            z = z + np.where(core, rates["uplift"] * h, 0.0)
            graph.update_routes(z)
            z = z - spl.erode(z, graph.accumulate(1.0), h)
            z = z - diffusion.erode(z, h)
        now = target
        out[target] = z.copy()
    return out


def simulate(job: dict) -> dict:
    """One forward run. `job`: rates, epochs (absolute years), dt, and optional variations.

    Variations, used for misspecified truths: `solver` ("teacher" or
    "fastscape"), `upliftGradient` (fractional change of U across the domain,
    west to east), `boundaryOffsetsM` (constant offsets of the N, S, W and E
    edges), `initialSeed`. Returns the surfaces at the requested epochs.
    """
    seed = int(job.get("initialSeed", INITIAL_SEED))
    start = np.array(initial_surface(SIDE, seed), dtype=np.float64)
    offsets = job.get("boundaryOffsetsM")
    if offsets:
        start[0, :] += offsets[0]
        start[-1, :] += offsets[1]
        start[1:-1, 0] += offsets[2]
        start[1:-1, -1] += offsets[3]
    rates = job["rates"]
    epochs = sorted(float(e) for e in job["epochs"])
    if job.get("solver", "teacher") == "fastscape":
        return _fastscape_snapshots(start, rates, epochs, float(job["dt"]))
    parameters = landscape.Parameters(uplift_m_per_year=rates["uplift"], k_incision=rates["kIncision"],
                                      diffusivity_m2_per_year=rates["diffusivity"], spacing_m=SPACING_M)
    uplift = None
    if job.get("upliftGradient"):
        ramp = np.linspace(-0.5, 0.5, SIDE)[None, :] * np.ones((SIDE, 1))
        uplift = rates["uplift"] * (1.0 + float(job["upliftGradient"]) * ramp)
    final, record = landscape.evolve(start, parameters, epochs[-1], dt_years=float(job["dt"]), uplift=uplift,
                                     base_level="fixed-edges", router=hydrology_fast, snapshots=epochs[:-1])
    out = {e: record["snapshotSurfaces"][e] for e in epochs[:-1]}
    out[epochs[-1]] = final
    return out


def _observe(job: dict) -> dict:
    """Worker entry: simulate and return observed vectors keyed by epoch (picklable, small)."""
    observation = _observation()
    surfaces = simulate(job)
    return {"key": job.get("key"), "obs": {e: observation.apply(s) for e, s in surfaces.items()}}


@functools.lru_cache(maxsize=1)
def _observation() -> Observation:
    return Observation()


_POOL = {}


def _pool_map(jobs, workers: int):
    """Map forward runs over one persistent worker pool per size (forked, so caches are shared)."""
    if workers <= 1:
        return [_observe(job) for job in jobs]
    if workers not in _POOL:
        _POOL[workers] = multiprocessing.get_context("fork").Pool(workers)
    return _POOL[workers].map(_observe, jobs, chunksize=max(1, len(jobs) // (4 * workers)))


def close_pools() -> None:
    for pool in _POOL.values():
        pool.close()
        pool.join()
    _POOL.clear()


def along_ridge(c: float, years: float, dt: float = DT_YEARS, rescale_dt: bool = True) -> tuple[dict, float, float]:
    rates = {k: c * v for k, v in BASE.items()}
    return rates, years / c, (dt / c if rescale_dt else dt)


# 1. The decisive profile along the ridge.

def ridge_profile(workers: int = 20, seed: int = 20261004, draws: int = 300) -> dict:
    """Likelihood along c for terminal, fractional and fixed-lag designs, three truths.

    Every c runs the teacher at (c U, c K, c D) to t / c with dt / c, so every
    run takes the same steps and a fixed numerical step cannot masquerade as
    information. Two controls: the same profile with dt held at 400 years for
    every c (the masquerade, made visible) and with dt halved (refinement).
    Expected profiles use the noiseless truth as data; one noisy profile is
    reported per design, and coverage of the 2-log-unit interval for c is
    counted over `draws` noise realisations with the correct, the independent
    and the version 1 noise models.
    """
    observation = _observation()
    rng = np.random.default_rng(seed)
    variants = {"rescaledDt": dict(rescale_dt=True, dt=DT_YEARS),
                "fixedDtCap": dict(rescale_dt=False, dt=DT_YEARS),
                "refinedDt": dict(rescale_dt=True, dt=DT_YEARS / 2.0)}
    jobs = []
    for truth, years in TRUTHS.items():
        for variant, opts in variants.items():
            grid = C_FINE if variant == "rescaledDt" else C_COARSE
            for c in grid:
                rates, t, dt = along_ridge(c, years, opts["dt"], opts["rescale_dt"])
                epochs = sorted({FRACTION * t, t} | {t + lag for lag in LAGS_YEARS})
                jobs.append({"key": (truth, variant, c), "rates": rates, "epochs": epochs, "dt": dt,
                             "c": c, "t": t})
    started = time.perf_counter()
    results = {r["key"]: r["obs"] for r in _pool_map(jobs, workers)}
    seconds = time.perf_counter() - started
    out = {"truths": {}, "seconds": seconds}
    for truth, years in TRUTHS.items():
        block = {"years": years, "variants": {}}
        for variant, opts in variants.items():
            grid = C_FINE if variant == "rescaledDt" else C_COARSE
            models = {}
            for c in grid:
                obs = results[(truth, variant, c)]
                t = years / c
                times = sorted(obs)
                models[c] = {"terminal": [obs[t]], "fractional": [obs[times[0]], obs[t]]}
                for lag in LAGS_YEARS:
                    later = min(times, key=lambda e: abs(e - (t + lag)))
                    models[c][f"fixed-lag-{int(lag / 1000)}k"] = [obs[t], obs[later]]
            truth_model = models[1.0]
            designs = {}
            for design in truth_model:
                expected = [observation.loglik([m - d for m, d in zip(models[c][design], truth_model[design])])
                            for c in grid]
                expected = [v - max(expected) for v in expected]
                surface_shift = max(float(max(np.abs(m - d).max() for m, d in zip(models[c][design],
                                                                                   truth_model[design])))
                                    for c in grid)
                row = {"deltaLogLikExpected": expected, "rangeExpected": float(-min(expected)),
                       "maxObservedDifferenceAlongRidgeM": surface_shift,
                       "coarse": {str(c): expected[grid.index(c)] for c in C_COARSE}}
                if variant == "rescaledDt":
                    row["interval2Expected"] = interval(grid, expected)
                    row["sdLnCExpected"] = curvature(grid, expected)
                    noisy = [d + observation.draw(rng) for d in truth_model[design]]
                    ll = [observation.loglik([m - d for m, d in zip(models[c][design], noisy)]) for c in grid]
                    row["deltaLogLikNoisy"] = [v - max(ll) for v in ll]
                    row["interval2Noisy"] = interval(grid, row["deltaLogLikNoisy"])
                    if design.startswith("fixed-lag") and truth != "near-equilibrium":
                        row["coverage"] = _c_coverage(models, design, grid, observation, rng, draws)
                designs[design] = row
            if variant == "rescaledDt":
                change = {f"fixed-lag-{int(lag / 1000)}k": float(np.sqrt(np.mean(
                    (truth_model[f"fixed-lag-{int(lag / 1000)}k"][1] - truth_model["terminal"][0]) ** 2)))
                    for lag in LAGS_YEARS}
                block["rmsChangeOverLagM"] = change
                block["reliefM"] = float(np.ptp(truth_model["terminal"][0]))
            block["variants"][variant] = {"cGrid": list(grid), "designs": designs}
        out["truths"][truth] = block
    out["flatWithinRounding"] = {truth: {variant: {d: out["truths"][truth]["variants"][variant]["designs"][d]
                                                   ["rangeExpected"] for d in ("terminal", "fractional")}
                                         for variant in variants} for truth in TRUTHS}
    return out


def _c_coverage(models, design, grid, observation, rng, draws: int) -> dict:
    """Does the 2-log-unit interval for c cover c = 1, over many noise draws, per noise model?"""
    hits = {"correct": 0, "iid": 0, "iidVersion1Noise": 0}
    truth = models[1.0][design]
    for _ in range(draws):
        noisy = [d + observation.draw(rng) for d in truth]
        old = [d + observation.apply(np.pad(old_noise((SIDE - 2, SIDE - 2), rng), 1)) for d in truth]
        for name, data, iid in (("correct", noisy, False), ("iid", noisy, True), ("iidVersion1Noise", old, True)):
            residual = lambda c: [m - d for m, d in zip(models[c][design], data)]  # noqa: E731
            ll = [observation.loglik_iid(residual(c)) if iid else observation.loglik(residual(c)) for c in grid]
            delta = [v - max(ll) for v in ll]
            low, high = interval(grid, delta)["interval"]
            hits[name] += int(low <= 1.0 <= high)
    return {"draws": draws, "nominal": 0.954,
            **{name: hits[name] / draws for name in hits},
            "note": "2-log-unit likelihood interval over the c grid (Wilks: about 95.4 % for one "
                    "parameter). iid ignores the correlation; iidVersion1Noise also uses the "
                    "version 1 noise recipe."}


def interval(grid, delta, level: float = 2.0) -> dict:
    x, y = np.log(np.asarray(grid)), np.asarray(delta)
    ok = y >= -level
    i0, i1 = np.flatnonzero(ok)[0], np.flatnonzero(ok)[-1]
    lo = grid[0] if i0 == 0 else float(np.exp(np.interp(-level, [y[i0 - 1], y[i0]], [x[i0 - 1], x[i0]])))
    hi = grid[-1] if i1 == len(grid) - 1 else float(np.exp(np.interp(-level, [y[i1 + 1], y[i1]], [x[i1 + 1], x[i1]])))
    return {"interval": [float(lo), float(hi)], "hitsGridEdge": bool(i0 == 0 or i1 == len(grid) - 1)}


def curvature(grid, delta) -> float | None:
    """sd of ln c from a quadratic fit to points within 3 log units of the maximum."""
    x, y = np.log(np.asarray(grid)), np.asarray(delta)
    keep = y >= -3.0
    if keep.sum() < 3:
        keep = np.argsort(-y)[:3]
    q = np.polyfit(x[keep], y[keep], 2)[0]
    return float(1.0 / math.sqrt(-2.0 * q)) if q < 0 else None


def nuisance_profile(workers: int = 20, truth: str = "transient", lag: float = 50_000.0,
                     cs=(0.5, 0.75, 0.9, 1.0, 1.1, 1.5, 2.0), span: float = 0.03, points: int = 7) -> dict:
    """The fixed-lag profile for c with log10 U/K and log10 D/K profiled out, not held at the truth.

    A slice through the truth can look informative because the other
    parameters are fixed at values the data were made with. Here, at each c,
    the expected log-likelihood (noiseless data) is maximised over offsets of
    the two ratios on a grid of +-`span` and then on a grid three times finer
    around the best point. The profile is compared with the slice.
    """
    observation = _observation()
    years = TRUTHS[truth]
    rates0 = dict(BASE)
    data = _observe({"rates": rates0, "epochs": [years, years + lag], "dt": DT_YEARS})["obs"]
    data = [data[years], data[years + lag]]

    def evaluate(c, offsets):
        jobs = []
        for i, (du, dd) in enumerate(offsets):
            rates = {"uplift": c * BASE["uplift"] * 10.0 ** du, "kIncision": c * BASE["kIncision"],
                     "diffusivity": c * BASE["diffusivity"] * 10.0 ** dd}
            t = years / c
            jobs.append({"key": i, "rates": rates, "epochs": [t, t + lag], "dt": DT_YEARS / c, "t": t})
        outs = sorted(_pool_map(jobs, workers), key=lambda r: r["key"])
        values = []
        for job, out in zip(jobs, outs):
            obs = out["obs"]
            first, second = sorted(obs)
            values.append(observation.loglik([obs[first] - data[0], obs[second] - data[1]]))
        return np.asarray(values)

    rows = []
    for c in cs:
        axis = np.linspace(-span, span, points)
        grid = [(a, b) for a in axis for b in axis]
        values = evaluate(c, grid)
        best = grid[int(np.argmax(values))]
        fine_axis = np.linspace(-1.0, 1.0, points) * (axis[1] - axis[0])
        fine = [(best[0] + a, best[1] + b) for a in fine_axis for b in fine_axis]
        fine_values = evaluate(c, fine)
        slice_value = float(values[grid.index((axis[points // 2], axis[points // 2]))])
        top = int(np.argmax(fine_values))
        rows.append({"c": c, "profileLogLik": float(max(values.max(), fine_values.max())), "sliceLogLik": slice_value,
                     "bestOffsets": list(fine[top]) if fine_values.max() >= values.max() else list(best),
                     "bestOnGridEdge": bool(max(abs(best[0]), abs(best[1])) >= span - 1e-12)})
    profile_max = max(r["profileLogLik"] for r in rows)
    slice_max = max(r["sliceLogLik"] for r in rows)
    for r in rows:
        r["deltaProfile"] = r["profileLogLik"] - profile_max
        r["deltaSlice"] = r["sliceLogLik"] - slice_max
    grid_c = [r["c"] for r in rows]
    return {"truth": truth, "years": years, "lagYears": lag, "offsetSpan": span, "rows": rows,
            "interval2Profile": interval(grid_c, [r["deltaProfile"] for r in rows]),
            "interval2Slice": interval(grid_c, [r["deltaSlice"] for r in rows]),
            "sdLnCProfile": curvature(grid_c, [r["deltaProfile"] for r in rows]),
            "sdLnCSlice": curvature(grid_c, [r["deltaSlice"] for r in rows]),
            "note": "Expected (noiseless) data. Offsets are in log10 of U/K and D/K at each c; a best "
                    "offset on the grid edge means the profile may be underestimated there."}


# 2 to 5. The identifiable ratios.

def ratio_rates(theta) -> dict:
    """theta = (log10 U/K, log10 D/K) at the base K."""
    k = BASE["kIncision"]
    return {"uplift": k * 10.0 ** theta[0], "kIncision": k, "diffusivity": k * 10.0 ** theta[1]}


def ratio_job(theta, key=None, dt: float = DT_YEARS, years: float = RATIO_YEARS, **variation) -> dict:
    return {"key": key, "rates": ratio_rates(theta), "epochs": [years], "dt": dt, **variation}


def _quadratic(mesh, values, dims, fit_span: float = 50.0):
    """Vertex and covariance of a quadratic fitted to every grid point within `fit_span` of the maximum.

    A global fit rather than one through the neighbours of the maximum: the
    teacher's D8 routing makes the likelihood piecewise smooth with narrow
    jumps where a receiver flips, and a fit through three points can mistake
    one jump for the curvature of the whole posterior.
    """
    near = values >= values.max() - fit_span
    if near.sum() < (dims + 1) * (dims + 2) // 2 + 1:
        return None, None
    centre = mesh[int(np.argmax(values))]
    x = mesh[near] - centre
    terms = [np.ones(len(x))] + [x[:, i] for i in range(dims)] + \
            [x[:, i] * x[:, j] for i in range(dims) for j in range(i, dims)]
    coef = np.linalg.lstsq(np.column_stack(terms), values[near], rcond=None)[0]
    gradient = coef[1:1 + dims]
    hessian = np.zeros((dims, dims))
    index = 1 + dims
    for i in range(dims):
        for j in range(i, dims):
            hessian[i, j] = hessian[j, i] = coef[index] * (2.0 if i == j else 1.0)
            index += 1
    if not np.all(np.linalg.eigvalsh(-hessian) > 0):
        return None, None
    covariance = np.linalg.inv(-hessian)
    return centre + covariance @ gradient, covariance


def grid_posterior(evaluate, box, dims: int, points=(9, 15), span: float = 8.0,
                   width_sd: float = 5.0, max_levels: int = 7, min_half_width: float = 1e-3) -> dict:
    """A likelihood grid that homes in on the mass: the reference posterior under a flat prior.

    `evaluate(list of theta) -> list of log-likelihoods`. Level 0 covers the
    prior box. While fewer than three grid points per axis lie within `span`
    log units of the maximum, the box shrinks to the bounding box of those
    points padded by one cell. Once resolved, a quadratic fitted to all
    points within 50 log units gives the next box, `width_sd` standard
    deviations either side of its vertex, never narrower than
    `min_half_width` (a floor, in log10 units, below which a box can only be
    chasing a single receiver flip). If the edges are not `span` log units
    below the maximum the box doubles. The final grid carries the normalised
    posterior; `jumpFraction` reports how rough the likelihood was on it.
    """
    prior_lows = np.array([b[0] for b in box[:dims]], dtype=float)
    prior_highs = np.array([b[1] for b in box[:dims]], dtype=float)
    levels, evaluated, fitted = [], 0, False
    lows, highs, n = prior_lows.copy(), prior_highs.copy(), points[0]
    sd, laplace_centre = None, None
    for level in range(max_levels):
        axes = [np.linspace(lows[i], highs[i], n) for i in range(dims)]
        mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, dims)
        values = np.asarray(evaluate([tuple(p) for p in mesh]), dtype=float)
        evaluated += len(mesh)
        grid = values.reshape((n,) * dims)
        edge_max = max(float(np.take(grid, i, axis=d).max()) for d in range(dims) for i in (0, -1))
        step = (highs - lows) / (n - 1)
        top = mesh[values >= values.max() - span]
        per_axis = [len(np.unique(np.round(top[:, d] / step[d]))) for d in range(dims)]
        resolved = min(per_axis) >= 3
        whole = bool(np.all(lows <= prior_lows + 1e-12) and np.all(highs >= prior_highs - 1e-12))
        contained = edge_max <= values.max() - span or whole
        levels.append({"lows": lows.tolist(), "highs": highs.tolist(), "points": n,
                       "edgeDropLogLik": float(values.max() - edge_max), "pointsWithinSpanPerAxis": per_axis})
        if fitted and resolved and contained:
            break
        best = mesh[int(np.argmax(values))]
        if not resolved:
            lows = np.maximum(top.min(axis=0) - step, prior_lows)
            highs = np.minimum(top.max(axis=0) + step, prior_highs)
            centre, half = 0.5 * (lows + highs), np.maximum(0.5 * (highs - lows), min_half_width)
        elif not contained and fitted:
            centre, half = best, (highs - lows)
        else:
            vertex, covariance = _quadratic(mesh, values, dims)
            if covariance is None:
                centre, half = best, np.maximum((highs - lows) / 4.0, min_half_width)
            else:
                sd = np.sqrt(np.diag(covariance))
                laplace_centre = vertex
                centre, half = vertex, np.maximum(width_sd * sd, min_half_width)
                fitted = True
        lows = np.maximum(centre - half, prior_lows)
        highs = np.minimum(centre + half, prior_highs)
        n = points[1]
    jumps = []
    for d in range(dims):
        jumps.append(np.abs(np.diff(grid, axis=d)).ravel())
    jumps = np.concatenate(jumps)
    weights = np.exp(values - values.max())
    weights /= weights.sum()
    shape = (n,) * dims
    marginals = []
    for i in range(dims):
        axis_weights = weights.reshape(shape).sum(axis=tuple(j for j in range(dims) if j != i))
        marginals.append({"axis": axes[i].tolist(), "weights": axis_weights.tolist(),
                          "mean": float((axes[i] * axis_weights).sum()),
                          "quantiles": _quantiles(axes[i], axis_weights)})
    order = np.argsort(-weights)
    cumulative = np.cumsum(weights[order])
    hpd95 = float(values[order][min(np.searchsorted(cumulative, 0.95), len(order) - 1)])
    return {"mesh": mesh, "logLik": values, "weights": weights, "marginals": marginals,
            "hpd95LogLik": hpd95, "maxLogLik": float(values.max()),
            "map": mesh[int(np.argmax(values))].tolist(), "levels": levels, "evaluations": evaluated,
            "laplaceSd": None if sd is None else sd.tolist(),
            "laplaceCentre": None if laplace_centre is None else laplace_centre.tolist(),
            "gridStep": [float(a[1] - a[0]) for a in axes], "edgeDropLogLik": levels[-1]["edgeDropLogLik"],
            "massContained": bool(levels[-1]["edgeDropLogLik"] >= span), "resolved": bool(resolved),
            "jumpFraction": float(np.mean(jumps > 20.0)),
            "note": "jumpFraction: share of neighbouring grid values differing by more than 20 log units"}


def _quantiles(axis, weights) -> dict:
    """Quantiles of a gridded density, treating each grid value as the centre of its cell."""
    step = axis[1] - axis[0]
    edges = np.concatenate([[axis[0] - step / 2], axis + step / 2])
    cdf = np.concatenate([[0.0], np.cumsum(weights)])
    return {name: float(np.interp(q, cdf, edges))
            for name, q in (("q025", 0.025), ("q16", 0.16), ("q50", 0.5), ("q84", 0.84), ("q975", 0.975))}


def _ratio_evaluator(data, observation, workers, datum=False, **variation):
    def evaluate(thetas):
        jobs = [ratio_job(theta, key=i, **variation) for i, theta in enumerate(thetas)]
        outs = sorted(_pool_map(jobs, workers), key=lambda r: r["key"])
        return [observation.loglik([o["obs"][RATIO_YEARS] - data], datum) for o in outs]
    return evaluate


def ratio_truth_data(theta, rng, observation, **variation):
    """Noiseless and noisy terminal observations at a truth (possibly generated by another model)."""
    clean = _observe(ratio_job(theta, **variation))["obs"][RATIO_YEARS]
    return clean, clean + observation.draw(rng)


def metropolis(job: dict) -> dict:
    """One random-walk Metropolis chain on the ratio posterior (flat prior on the box)."""
    observation = _observation()
    rng = np.random.default_rng(job["seed"])
    dims, data = job["dims"], job["data"]
    lows = np.array([b[0] for b in RATIO_BOX[:dims]])
    highs = np.array([b[1] for b in RATIO_BOX[:dims]])

    def logpost(theta):
        if np.any(theta < lows) or np.any(theta > highs):
            return -np.inf
        full = list(theta) + list(RATIO_TRUTH[dims:])
        obs = _observe(ratio_job(full))["obs"][RATIO_YEARS]
        return observation.loglik([obs - data])

    state = np.array(job["start"], dtype=float)
    value = logpost(state)
    chol = np.linalg.cholesky(np.atleast_2d(job["proposal"]))
    draws, accepted = [], 0
    for index in range(job["burn"] + job["draws"]):
        candidate = state + chol @ rng.normal(size=dims)
        candidate_value = logpost(candidate)
        if math.log(max(rng.random(), 1e-300)) < candidate_value - value:
            state, value = candidate, candidate_value
            accepted += index >= job["burn"]
        if index >= job["burn"]:
            draws.append(state.copy())
    return {"draws": np.array(draws), "acceptance": accepted / job["draws"]}


def ratio_reference(workers: int = 20, seed: int = 7, chains: int = 16, draws: int = 400, burn: int = 150) -> dict:
    """One- and two-ratio posteriors on a likelihood grid, checked by MCMC with the new diagnostics."""
    from geoneural.physics import diagnostics
    observation = _observation()
    rng = np.random.default_rng(seed)
    _, data = ratio_truth_data(RATIO_TRUTH, rng, observation)
    out = {"truth": dict(zip(RATIO_NAMES, RATIO_TRUTH)), "years": RATIO_YEARS, "box": RATIO_BOX}
    for dims in (1, 2):
        evaluate = _ratio_evaluator(data, observation, workers)
        fixed = list(RATIO_TRUTH[dims:])
        grid = grid_posterior(lambda thetas: evaluate([list(t) + fixed for t in thetas]), RATIO_BOX, dims,
                              points=(17, 41) if dims == 1 else (9, 21))
        weights, mesh = grid["weights"], grid["mesh"]
        mean = (weights[:, None] * mesh).sum(axis=0)
        cov = ((mesh - mean).T * weights) @ (mesh - mean)
        cov = np.atleast_2d(cov) + np.diag(np.asarray(grid["gridStep"]) ** 2 / 12.0)
        # The proposal follows the smooth (quadratic) scale when the gridded posterior is
        # narrower, so a chain is not tuned to the width of a single receiver flip.
        smooth = np.diag(np.asarray(grid["laplaceSd"]) ** 2) if grid["laplaceSd"] else cov
        proposal = (2.38 ** 2 / dims) * (cov if np.trace(cov) >= np.trace(smooth) else smooth)
        spread = np.sqrt(np.maximum(np.diag(cov), np.diag(smooth)))
        starts = [mean + 3.0 * spread * rng.uniform(-1, 1, dims) for _ in range(chains)]
        jobs = [{"seed": int(rng.integers(2 ** 31)), "dims": dims, "data": data, "start": s.tolist(),
                 "proposal": proposal, "burn": burn, "draws": draws} for s in starts]
        started = time.perf_counter()
        with multiprocessing.get_context("fork").Pool(min(workers, chains)) as pool:
            chain_out = pool.map(metropolis, jobs)
        stacked = np.stack([c["draws"] for c in chain_out])
        summary = diagnostics.summary(stacked, RATIO_NAMES[:dims])
        compare = {}
        for i, name in enumerate(RATIO_NAMES[:dims]):
            g = grid["marginals"][i]["quantiles"]
            m = summary["parameters"][name]["quantiles"]
            sd = math.sqrt(cov[i, i])
            compare[name] = {"grid": g, "mcmc": {k: m[k] for k in ("q025", "q16", "q50", "q84", "q975")},
                             "gridSd": sd, "maxQuantileDifferenceInSd": max(abs(g[k] - m[k]) for k in g) / sd}
        out[f"{dims}d"] = {"grid": {k: v for k, v in grid.items() if k not in ("mesh", "logLik", "weights")},
                           "gridMean": mean.tolist(), "gridCovariance": np.atleast_2d(cov).tolist(),
                           "mcmc": {**summary, "acceptance": [c["acceptance"] for c in chain_out],
                                    "seconds": time.perf_counter() - started, "burn": burn,
                                    "proposalSd": np.sqrt(np.diag(proposal)).tolist()},
                           "jumpFraction": grid["jumpFraction"], "laplaceSd": grid["laplaceSd"],
                           "gridVersusMcmc": compare}
    return out


def coverage(workers: int = 20, seed: int = 11, truths_1d: int = 100, truths_2d: int = 50,
             variation: dict | None = None, label: str = "correct") -> dict:
    """Repeated synthetic truths: how often do the grid posteriors' intervals cover the truth?

    Truths are drawn uniformly inside the inner 80 % of the prior box. With
    `variation`, the data come from a different model (a misspecified truth)
    while inference keeps the nominal teacher; a posterior-predictive check
    (chi-square of the whitened residual at the MAP against its n degrees of
    freedom) says whether the mismatch is detectable.
    """
    observation = _observation()
    rng = np.random.default_rng(seed)
    out = {"label": label, "variation": variation or {}, "rows": {}}
    for dims, count in ((1, truths_1d), (2, truths_2d)):
        if count == 0:
            continue
        rows = []
        for _ in range(count):
            theta = [float(rng.uniform(lo + 0.1 * (hi - lo), hi - 0.1 * (hi - lo))) for lo, hi in RATIO_BOX[:dims]]
            full = theta + list(RATIO_TRUTH[dims:])
            _, data = ratio_truth_data(full, rng, observation, **(variation or {}))
            evaluate = _ratio_evaluator(data, observation, workers)
            fixed = list(RATIO_TRUTH[dims:])
            grid = grid_posterior(lambda thetas: evaluate([list(t) + fixed for t in thetas]), RATIO_BOX, dims,
                                  points=(17, 25) if dims == 1 else (9, 13))
            map_obs = _observe(ratio_job(list(grid["map"]) + fixed))["obs"][RATIO_YEARS]
            chi2 = observation.chi2([map_obs - data])
            row = {"truth": theta, "map": grid["map"], "chi2": chi2, "n": observation.size, "_data": data,
                   "gridStep": grid["gridStep"], "laplaceSd": grid["laplaceSd"],
                   "massContained": grid["massContained"], "resolved": grid["resolved"],
                   "evaluations": grid["evaluations"],
                   "chi2Z": (chi2 - observation.size) / math.sqrt(2 * observation.size), "marginals": []}
            row["jumpFraction"] = grid["jumpFraction"]
            for i in range(dims):
                q = grid["marginals"][i]["quantiles"]
                entry = {"mean": grid["marginals"][i]["mean"], "quantiles": q,
                         "in68": bool(q["q16"] <= theta[i] <= q["q84"]),
                         "in95": bool(q["q025"] <= theta[i] <= q["q975"])}
                if grid["laplaceSd"] is not None:
                    z = abs(theta[i] - grid["laplaceCentre"][i]) / grid["laplaceSd"][i]
                    entry.update({"laplaceIn68": bool(z <= 1.0), "laplaceIn95": bool(z <= 1.96)})
                row["marginals"].append(entry)
            if dims == 2:
                truth_ll = evaluate([full])[0]
                row["inJoint95"] = bool(truth_ll >= grid["hpd95LogLik"])
            rows.append(row)
        summary = {"truths": count}
        for i, name in enumerate(RATIO_NAMES[:dims]):
            errors = [r["marginals"][i]["mean"] - r["truth"][i] for r in rows]
            widths = [r["marginals"][i]["quantiles"]["q975"] - r["marginals"][i]["quantiles"]["q025"] for r in rows]
            laplace = [r["marginals"][i] for r in rows if "laplaceIn95" in r["marginals"][i]]
            summary[name] = {"coverage68": float(np.mean([r["marginals"][i]["in68"] for r in rows])),
                             "coverage95": float(np.mean([r["marginals"][i]["in95"] for r in rows])),
                             "laplaceCoverage68": float(np.mean([m["laplaceIn68"] for m in laplace])) if laplace else None,
                             "laplaceCoverage95": float(np.mean([m["laplaceIn95"] for m in laplace])) if laplace else None,
                             "rmseLog10": float(np.sqrt(np.mean(np.square(errors)))),
                             "biasLog10": float(np.mean(errors)),
                             "medianWidth95Log10": float(np.median(widths))}
        if dims == 2:
            summary["jointCoverage95"] = float(np.mean([r["inJoint95"] for r in rows]))
        summary["ppcDetectionRate"] = float(np.mean([r["chi2Z"] > 2.326 for r in rows]))
        summary["medianJumpFraction"] = float(np.median([r["jumpFraction"] for r in rows]))
        summary["massContainedFraction"] = float(np.mean([r["massContained"] for r in rows]))
        summary["resolvedFraction"] = float(np.mean([r["resolved"] for r in rows]))
        summary["medianChi2Z"] = float(np.median([r["chi2Z"] for r in rows]))
        out["rows"][f"{dims}d"] = {"summary": summary, "truths": rows}
    return out


def strip_private(value):
    """Drop the `_`-prefixed working arrays before a result is written."""
    if isinstance(value, dict):
        return {k: strip_private(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, list):
        return [strip_private(v) for v in value]
    return value


def features(observed: np.ndarray) -> np.ndarray:
    """Slope, curvature and drainage statistics of an observed surface and of its change.

    The change is taken against the known initial surface, which the
    likelihood also knows. Drainage comes from the same D8 routing as the
    teacher, run on the noisy observation.
    """
    inner = SIDE - 2
    z = observed.reshape(inner, inner)
    start = initial_surface()[1:-1, 1:-1]
    change = z - start
    gy, gx = np.gradient(z, SPACING_M)
    slope = np.hypot(gx, gy)
    curvature = landscape.laplacian(z, SPACING_M)[1:-1, 1:-1]
    routed = landscape.routing(z, SPACING_M, hydrology_fast)
    area, d8 = routed["area"], routed["slope"]
    cells = area / SPACING_M ** 2
    channel = cells >= 8
    use = (cells >= 2) & (d8 > 1e-6)
    if use.sum() >= 8:
        fit = np.polyfit(np.log10(area[use]), np.log10(d8[use]), 1)
    else:
        fit = (0.0, 0.0)
    span = max(float(z.max() - z.min()), 1e-9)
    return np.array([z.std(), np.quantile(z, 0.99) - np.quantile(z, 0.01), slope.mean(), np.median(slope),
                     np.quantile(slope, 0.9), curvature.std(), np.abs(curvature).mean(),
                     float(np.mean(curvature < 0)), (z.mean() - z.min()) / span, fit[0], fit[1],
                     float(d8[channel].mean()) if channel.any() else 0.0, float(channel.mean()),
                     change.mean(), change.std(), np.quantile(change, 0.9) - np.quantile(change, 0.1)])


FEATURE_NAMES = ("heightStd", "relief98", "slopeMean", "slopeMedian", "slopeP90", "curvatureStd",
                 "curvatureMeanAbs", "convexFraction", "hypsometricIntegral", "slopeAreaExponent",
                 "slopeAreaLogIntercept", "channelSlopeMean", "channelFraction", "changeMean", "changeStd",
                 "changeP90MinusP10")


def _design(x: np.ndarray, centre=None, scale=None):
    centre = x.mean(axis=0) if centre is None else centre
    scale = x.std(axis=0) + 1e-12 if scale is None else scale
    u = (x - centre) / scale
    quad = [u[:, i] * u[:, j] for i in range(u.shape[1]) for j in range(i, u.shape[1])]
    return np.column_stack([np.ones(len(u)), u] + quad), centre, scale


def estimators(test_rows: list, workers: int = 20, seed: int = 17, count: int = 2000,
               out_of_range: int = 20, device: str = "cuda", steps: int = 4000) -> dict:
    """A CNN and a feature regressor for the two ratios, against the reference posterior.

    Both are trained on `count` simulations drawn uniformly from the prior box
    (one fresh noise draw per simulation for the regressor, a fresh draw every
    batch for the CNN), with the last 20 % held out for calibration. The CNN
    predicts a mean and a log variance per ratio (Gaussian negative log
    likelihood), so its intervals are its own; the regressor's intervals are
    split-conformal from the calibration residuals. Test truths are the
    two-ratio coverage truths with the same noisy data the reference
    posterior saw, plus `out_of_range` truths with log10 U/K above the box,
    where an honest estimator should widen or fail visibly.
    """
    import torch
    observation = _observation()
    rng = np.random.default_rng(seed)
    thetas = np.column_stack([rng.uniform(lo, hi, count) for lo, hi in RATIO_BOX])
    jobs = [ratio_job(t, key=i) for i, t in enumerate(thetas)]
    beyond = np.column_stack([rng.uniform(1.55, 1.8, out_of_range), rng.uniform(2.6, 3.4, out_of_range)])
    jobs += [ratio_job(t, key=count + i) for i, t in enumerate(beyond)]
    started = time.perf_counter()
    clean = {r["key"]: r["obs"][RATIO_YEARS] for r in _pool_map(jobs, workers)}
    simulate_seconds = time.perf_counter() - started
    train_clean = np.stack([clean[i] for i in range(count)])
    beyond_data = np.stack([clean[count + i] + observation.draw(rng) for i in range(out_of_range)])
    test_data = np.stack([r["_data"] for r in test_rows])
    test_truth = np.array([r["truth"] for r in test_rows])
    cut = int(0.8 * count)

    # Feature regressor: quadratic least squares on standardised features, ridge-regularised.
    noisy = train_clean + np.stack([observation.draw(rng) for _ in range(count)])
    feats = np.stack([features(v) for v in noisy])
    design, centre, scale = _design(feats[:cut])
    penalty = 1e-3 * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + penalty * len(design), design.T @ thetas[:cut])
    predict_features = lambda x: _design(np.stack([features(v) for v in x]), centre, scale)[0] @ coef  # noqa: E731
    calibration = np.abs(_design(feats[cut:], centre, scale)[0] @ coef - thetas[cut:])
    q68, q95 = np.quantile(calibration, 0.68, axis=0), np.quantile(calibration, 0.95, axis=0)

    # CNN with a Gaussian head.
    torch.manual_seed(seed)
    inner = SIDE - 2
    start = torch.tensor(initial_surface()[1:-1, 1:-1], dtype=torch.float32, device=device)
    net = torch.nn.Sequential(
        torch.nn.Conv2d(2, 32, 3, padding=1), torch.nn.GELU(),
        torch.nn.Conv2d(32, 32, 3, padding=1, stride=2), torch.nn.GELU(),
        torch.nn.Conv2d(32, 64, 3, padding=1, stride=2), torch.nn.GELU(),
        torch.nn.Conv2d(64, 64, 3, padding=1, stride=2), torch.nn.GELU(),
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(64, 4)).to(device)
    lower = torch.tensor(observation.lower, dtype=torch.float32, device=device)
    box_lo = torch.tensor([b[0] for b in RATIO_BOX], device=device)
    box_span = torch.tensor([b[1] - b[0] for b in RATIO_BOX], device=device)
    x_train = torch.tensor(train_clean[:cut], dtype=torch.float32, device=device)
    y_train = torch.tensor(thetas[:cut], dtype=torch.float32, device=device)

    def inputs(vectors):
        z = vectors.view(-1, 1, inner, inner)
        return torch.cat([z / 50.0, (z - start) / 50.0], dim=1)

    def head(vectors):
        out = net(inputs(vectors))
        return box_lo + box_span * out[:, :2], out[:, 2:].clamp(-12.0, 4.0)

    optimiser = torch.optim.Adam(net.parameters(), lr=2e-3)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, steps)
    generator = torch.Generator(device=device).manual_seed(seed)
    began = time.perf_counter()
    for _ in range(steps):
        pick = torch.randint(0, cut, (128,), device=device, generator=generator)
        noise = (lower @ torch.randn(observation.size, 128, device=device, generator=generator)).T
        mean, log_var = head(x_train[pick] + noise)
        loss = (0.5 * ((mean - y_train[pick]) ** 2 / log_var.exp() + log_var)).mean()
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        schedule.step()
    train_seconds = time.perf_counter() - began
    net.eval()

    def predict_cnn(vectors):
        with torch.no_grad():
            mean, log_var = head(torch.tensor(vectors, dtype=torch.float32, device=device))
        return mean.cpu().numpy().astype(np.float64), np.exp(0.5 * log_var.cpu().numpy().astype(np.float64))

    def score(pred, truth, low95, high95, low68, high68):
        error = pred - truth
        return {name: {"rmseLog10": float(np.sqrt(np.mean(error[:, i] ** 2))),
                       "biasLog10": float(np.mean(error[:, i])),
                       "coverage68": float(np.mean((low68[:, i] <= truth[:, i]) & (truth[:, i] <= high68[:, i]))),
                       "coverage95": float(np.mean((low95[:, i] <= truth[:, i]) & (truth[:, i] <= high95[:, i]))),
                       "medianWidth95Log10": float(np.median(high95[:, i] - low95[:, i]))}
                for i, name in enumerate(RATIO_NAMES)}

    results = {}
    for label, data, truth in (("inRange", test_data, test_truth), ("outOfRange", beyond_data, beyond)):
        mean, sd = predict_cnn(data)
        feature_pred = predict_features(data)
        results[label] = {
            "truths": int(len(truth)),
            "cnn": score(mean, truth, mean - 1.96 * sd, mean + 1.96 * sd, mean - sd, mean + sd),
            "featureRegressor": score(feature_pred, truth, feature_pred - q95, feature_pred + q95,
                                      feature_pred - q68, feature_pred + q68)}
    reference = {}
    for i, name in enumerate(RATIO_NAMES):
        means = np.array([r["marginals"][i]["mean"] for r in test_rows])
        q = [r["marginals"][i]["quantiles"] for r in test_rows]
        reference[name] = {"rmseLog10": float(np.sqrt(np.mean((means - test_truth[:, i]) ** 2))),
                           "coverage95": float(np.mean([a["q025"] <= t <= a["q975"] for a, t in zip(q, test_truth[:, i])])),
                           "medianWidth95Log10": float(np.median([a["q975"] - a["q025"] for a in q]))}
    results["inRange"]["referencePosterior"] = reference
    return {"trainSimulations": cut, "calibrationSimulations": count - cut, "features": list(FEATURE_NAMES),
            "cnn": {"steps": steps, "batch": 128, "parameters": int(sum(p.numel() for p in net.parameters())),
                    "inputs": "observed surface and its change from the known initial surface, metres / 50",
                    "seconds": train_seconds},
            "featureRegressor": {"form": "quadratic least squares, ridge 1e-3, split-conformal intervals",
                                 "conformalHalfWidth95": q95.tolist(), "conformalHalfWidth68": q68.tolist()},
            "simulateSeconds": simulate_seconds, "results": results,
            "outOfRangeBox": {"logUpliftOverIncision": [1.55, 1.8], "logDiffusivityOverIncision": [2.6, 3.4]}}


MISSPECIFIED = {
    "finer-dt": {"dt": 50.0},
    "fastscape-solver": {"solver": "fastscape"},
    "uplift-gradient-10pct": {"upliftGradient": 0.1},
    "boundary-offsets": {"boundaryOffsetsM": [0.8, -0.6, 0.5, -0.9]},
    "initial-surface": {"initialSeed": 4242},
}


def misspecification(workers: int = 20, truths: int = 40, seed: int = 13) -> dict:
    """Coverage of the one-ratio posterior when the data come from another model."""
    return {name: coverage(workers, seed + i, truths_1d=truths, truths_2d=0, variation=variation, label=name)
            for i, (name, variation) in enumerate(MISSPECIFIED.items())}


def discretisation_check(workers: int = 4) -> dict:
    """The teacher's own time-discretisation error in the inverse setting, against the noise."""
    jobs = [ratio_job(RATIO_TRUTH, key=dt, dt=dt) for dt in (400.0, 200.0, 100.0, 25.0)]
    outs = {r["key"]: r["obs"][RATIO_YEARS] for r in _pool_map(jobs, workers)}
    ref = outs[25.0]
    return {"referenceDtYears": 25.0, "noiseSigmaM": SIGMA_M,
            "rows": [{"dtYears": dt, "meanAbsM": float(np.abs(outs[dt] - ref).mean()),
                      "maxAbsM": float(np.abs(outs[dt] - ref).max()),
                      "meanAbsOverSigma": float(np.abs(outs[dt] - ref).mean()) / SIGMA_M}
                     for dt in (400.0, 200.0, 100.0)]}


def groups_of_truth() -> dict:
    """The dimensionless groups of the base truth, from `units`, for the record."""
    length = SPACING_M * (SIDE - 1)
    return {name: units.uplift_groups(BASE["uplift"], BASE["kIncision"], BASE["diffusivity"], length,
                                      INITIAL_RELIEF_M) | {"dimensionlessTime": years * BASE["uplift"] / INITIAL_RELIEF_M}
            for name, years in TRUTHS.items()}


def setup_record() -> dict:
    return {"side": SIDE, "spacingM": SPACING_M, "base": BASE, "dtYears": DT_YEARS,
            "initial": {"seed": INITIAL_SEED, "reliefM": INITIAL_RELIEF_M, "spectrum": "1/k amplitude"},
            "boundary": "all edges fixed at their initial values", "observation": _observation().record(),
            "fractionalEpoch": FRACTION, "lagsYears": list(LAGS_YEARS), "truthYears": TRUTHS,
            "ratioProblem": {"parameters": list(RATIO_NAMES), "truth": RATIO_TRUTH, "box": RATIO_BOX,
                             "years": RATIO_YEARS, "known": "K and t (so K t), the initial surface, the boundary"},
            "groups": groups_of_truth(), "ridge": ridge_direction()}


def campaign(torch=None, side: int = SIDE, chains: int = 8, draws: int = 600, burn: int = 200,
             seed: int = 1729, workers: int = 20) -> dict:
    """The ridge profile and the ratio reference, as one combined result (the legacy CLI entry).

    The experiments are defined on the module's 32 x 32 grid; a different
    `side` is recorded as ignored rather than silently honoured in part.
    """
    try:
        return {"schema": SCHEMA, "setup": setup_record(),
                "sideArgument": {"requested": side, "used": SIDE, "note": "the grid is fixed by the module"},
                "ridgeProfile": ridge_profile(workers, seed),
                "ratioReference": ratio_reference(workers, seed, chains, draws, burn)}
    finally:
        close_pools()
