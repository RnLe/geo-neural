"""Inferring landscape history, and what a surface cannot tell you.

The attractive claim here is that a present-day surface identifies the history
that made it. Most of that claim is false for structural reasons, and this
module shows which part is false before fitting anything, because a posterior
over unidentifiable parameters looks exactly like a posterior over
identifiable ones.

The identifiability statement is arithmetic, not an empirical finding. The
stream-power system obeys a similarity law: scaling
(U, K, D) -> (lambda U, lambda K, lambda D) and t -> t / lambda leaves the
entire trajectory unchanged. So from a terminal surface with a known initial
condition, the data can constrain at most the products `U t`, `K t` and `D t`,
never the rates and the duration separately. A four-parameter posterior over
(log U, log K, log D, log t) therefore has an exact three-dimensional ridge in
it, and any "inferred age" read off such a posterior is reporting the prior.

At steady state it is worse: the surface depends only on the dimensionless
groups Nf and Nh, so even the products are gone and two numbers survive out of
four. `ridge_direction` returns the exact null direction, and
`profile_along_ridge` walks it to show that the likelihood is flat there.

What breaks the degeneracy is a second time slice, which `campaign` compares
against a single slice: the rate of change between two epochs carries the
timescale that a single snapshot cannot.
"""
from __future__ import annotations

import math
import multiprocessing
import time

import numpy as np

from geoneural.physics import landscape

SCHEMA = "geoneural-inverse-history-v1"

PARAMETERS = ("logUplift", "logIncision", "logDiffusivity", "logYears", "datum")

# Model discrepancy as a fraction of the surface's relief. The teacher omits
# climate, lithology, glaciation and base-level history, so treating it as exact
# is wrong and makes the posterior too narrow to sample (see `log_likelihood`).
DISCREPANCY_FRACTION = 0.05


class Observation:
    """What is actually measured, rather than the field itself.

    Three things stand between a simulated surface and a measurement, and each
    of them changes the posterior: the grid is subsampled, the domain is cropped,
    and the vertical datum is unknown. The datum matters most: an unknown
    constant offset is perfectly correlated with everything that raises the
    surface uniformly, which is most of what uplift does.
    """

    def __init__(self, stride: int = 4, crop: int = 8, noise_m: float = 0.5,
                 correlation_cells: float = 3.0, seed: int = 1729):
        self.stride = int(stride)
        self.crop = int(crop)
        self.noise_m = float(noise_m)
        self.correlation_cells = float(correlation_cells)
        self.seed = int(seed)

    def apply(self, surface: np.ndarray, datum: float = 0.0) -> np.ndarray:
        inner = surface[self.crop:-self.crop or None, self.crop:-self.crop or None]
        return inner[::self.stride, ::self.stride] + datum

    def noise(self, shape, rng) -> np.ndarray:
        """Spatially correlated error, because survey error is not white.

        White noise would make every subsampled node an independent measurement
        and shrink the posterior by roughly the square root of their count. The
        correlation length is declared and the covariance is the one the
        likelihood uses, so the two cannot drift apart.
        """
        white = rng.normal(0.0, 1.0, shape)
        spectrum = np.fft.rfft2(white)
        ky = np.fft.fftfreq(shape[0])[:, None]
        kx = np.fft.rfftfreq(shape[1])[None, :]
        wavenumber = np.sqrt(ky * ky + kx * kx)
        kernel = np.exp(-0.5 * (wavenumber * self.correlation_cells * 2 * np.pi) ** 2)
        # Remove the constant mode. The kernel passes it at full weight (a
        # smoothing filter never attenuates wavenumber zero), and the
        # normalisation below divides by the standard deviation, which measures
        # spread about the mean and not the mean itself. On a small grid the
        # remaining deviations can be a hundred-thousandth of the mean, so
        # dividing by their scale can inflate the offset to tens of kilometres on
        # a 66 m surface. Measurement noise is zero-mean by definition; a
        # constant offset is the `datum` parameter, which is sampled separately
        # and bounded to +-5 m, so an offset here would make the data unfittable.
        spectrum[0, 0] = 0.0
        field = np.fft.irfft2(spectrum * kernel, s=shape)
        field = field - float(field.mean())
        spread = float(np.std(field))
        if spread < 1e-6:
            raise ValueError(
                f"correlated noise on a {shape} grid with a {self.correlation_cells}-cell "
                "correlation length has no variance left: every mode above DC is "
                "filtered out. Generate the noise at the full grid resolution and "
                "subsample it, rather than generating it on the subsampled grid")
        return self.noise_m * field / spread

    def record(self) -> dict:
        return {"stride": self.stride, "crop": self.crop, "noiseM": self.noise_m,
                "correlationCells": self.correlation_cells,
                "note": "Correlated noise, not white: white error would make every "
                        "subsampled node independent and shrink the posterior by "
                        "roughly the square root of their count."}


def ridge_direction() -> dict:
    """The exact null direction of a terminal-surface likelihood.

    Under (U, K, D, t) -> (lambda U, lambda K, lambda D, t/lambda) the surface is
    unchanged, so in log coordinates the direction (+1, +1, +1, -1) normalised is
    a flat direction of the likelihood by construction. Nothing was fitted to
    obtain this; it is the similarity law written in the coordinates the sampler
    uses.
    """
    vector = np.array([1.0, 1.0, 1.0, -1.0, 0.0])
    return {"direction": (vector / np.linalg.norm(vector)).tolist(),
            "parameters": list(PARAMETERS),
            "identified": ["logUplift + logYears", "logIncision + logYears",
                           "logDiffusivity + logYears"],
            "notIdentified": ["any of the four separately"],
            "atSteadyState": ["logFluvialNumber", "logHillslopeNumber"],
            "note": "Arithmetic, not a fitted result. A posterior over the four "
                    "parameters has this exact ridge in it, and an 'inferred age' "
                    "read off one is reporting the prior along this direction."}


def simulate(theta: dict, side: int = 48, spacing_m: float = 50.0,
             initial_relief_m: float = 15.0, seed: int = 1729,
             router=None) -> np.ndarray:
    """Run the teacher at one parameter vector. The forward model of the inference."""
    rng = np.random.default_rng(seed)
    white = rng.normal(0.0, 1.0, (side, side))
    spectrum = np.fft.rfft2(white)
    ky = np.fft.fftfreq(side)[:, None]
    kx = np.fft.rfftfreq(side)[None, :]
    wavenumber = np.sqrt(ky * ky + kx * kx)
    wavenumber[0, 0] = 1.0
    field = np.fft.irfft2(spectrum / wavenumber, s=(side, side))
    height = initial_relief_m * field / max(float(np.std(field)), 1e-9)
    parameters = landscape.Parameters(
        uplift_m_per_year=10.0 ** theta["logUplift"],
        k_incision=10.0 ** theta["logIncision"],
        area_exponent=0.5, slope_exponent=1.0,
        diffusivity_m2_per_year=10.0 ** theta["logDiffusivity"],
        spacing_m=spacing_m)
    years = 10.0 ** theta["logYears"]
    dt = min(landscape.stable_timestep(parameters), 400.0)
    final, _ = landscape.evolve(height, parameters, years, dt_years=dt,
                                base_level="fixed-edges", router=router)
    return final


def profile_along_ridge(theta: dict, observation: Observation, steps: int = 7,
                        span: float = 0.6, side: int = 48, spacing_m: float = 50.0,
                        router=None) -> dict:
    """Walk the similarity direction and show the misfit does not move.

    This is the demonstration the identifiability statement needs. Walking a
    direction the theory says is flat and finding the misfit flat is evidence;
    asserting the ridge exists is not.
    """
    direction = np.array(ridge_direction()["direction"])
    base = np.array([theta[name] for name in PARAMETERS])
    reference = observation.apply(simulate(theta, side, spacing_m, router=router))
    rows = []
    for offset in np.linspace(-span, span, steps):
        moved = base + offset * direction
        candidate = dict(zip(PARAMETERS, moved))
        surface = observation.apply(simulate(candidate, side, spacing_m, router=router))
        rows.append({"offset": float(offset),
                     "logUplift": float(candidate["logUplift"]),
                     "logYears": float(candidate["logYears"]),
                     "rmseM": float(np.sqrt(np.mean((surface - reference) ** 2)))})
    spread = float(max(r["rmseM"] for r in rows))
    return {"rows": rows, "maxRmseAlongRidgeM": spread,
            "note": "The misfit along the similarity direction. A likelihood that "
                    "moves here would falsify the identifiability statement; one "
                    "that does not is the ridge, measured rather than asserted."}


def log_likelihood(surface, data, observation: Observation, datum: float,
                   discrepancy_m: float = 0.0) -> float:
    """Gaussian in the observation noise plus a model-discrepancy term.

    Without the second term the likelihood asserts that the teacher is the truth.
    It is not: it omits climate, lithology, glaciation and base-level history,
    and the teacher audit shows it needs a mesh at or below 100 m to obey its
    steady-state law. A likelihood that ignores this is not only optimistic but
    unsamplable: without the term, a 12 % change in uplift rate costs about 504
    log units, which makes the posterior some four hundred times narrower than
    the prior box, so a proposal wide enough to cross the box is almost never
    accepted.

    The inflation is declared as a fraction of the surface's own relief rather
    than tuned, so it scales with the problem and is recorded in the result.
    """
    predicted = observation.apply(surface, datum)
    residual = predicted - data
    variance = max(observation.noise_m ** 2 + discrepancy_m ** 2, 1e-12)
    return float(-0.5 * np.sum(residual * residual) / variance)


PRIOR = {"logUplift": (-4.5, -3.0), "logIncision": (-6.0, -4.0),
         "logDiffusivity": (-3.0, -1.0), "logYears": (5.0, 6.5),
         "datum": (-5.0, 5.0)}


def _log_prior(vector) -> float:
    for name, value in zip(PARAMETERS, vector):
        low, high = PRIOR[name]
        if not (low <= value <= high):
            return -np.inf
    return 0.0



def _chain_posterior(vector, data, observation, side, spacing_m, slices,
                     discrepancy_m=0.0):
    from geoneural.metrics import hydrology_fast
    prior = _log_prior(vector)
    if not np.isfinite(prior):
        return -np.inf
    theta = dict(zip(PARAMETERS, vector))
    if slices is None:
        surface = simulate(theta, side, spacing_m, router=hydrology_fast)
        return prior + log_likelihood(surface, data, observation, theta["datum"],
                                      discrepancy_m)
    total = prior
    for fraction, observed in slices:
        partial = dict(theta)
        partial["logYears"] = theta["logYears"] + math.log10(fraction)
        surface = simulate(partial, side, spacing_m, router=hydrology_fast)
        total += log_likelihood(surface, observed, observation, theta["datum"],
                                discrepancy_m)
    return total


def _run_chain(job: dict) -> dict:
    """One Metropolis chain. Module level so a process pool can pickle it."""
    rng = np.random.default_rng(job["seed"])
    lows, highs = job["lows"], job["highs"]
    step = job["scale"].copy()
    args = (job["data"], job["observation"], job["side"], job["spacingM"],
            job["slices"], job.get("discrepancyM", 0.0))
    state = lows + rng.random(len(PARAMETERS)) * (highs - lows)
    value = _chain_posterior(state, *args)
    guard = 0
    while not np.isfinite(value) and guard < 200:
        state = lows + rng.random(len(PARAMETERS)) * (highs - lows)
        value = _chain_posterior(state, *args)
        guard += 1
    history, accepted, proposed = [], 0, 0
    for index in range(job["draws"]):
        candidate = state + rng.normal(0.0, 1.0, len(PARAMETERS)) * step
        candidate_value = _chain_posterior(candidate, *args)
        proposed += 1
        if math.log(max(rng.random(), 1e-300)) < candidate_value - value:
            state, value = candidate, candidate_value
            accepted += 1
        history.append(state.copy())
        # Windowed adaptation with acceptance targeting, frozen after burn-in. A
        # single adaptation at the end of burn-in cannot find both the scale and
        # the shape of the posterior; it leaves acceptance near 0.07 and R-hat
        # above 12, and the chain has explored nothing by the time it is frozen.
        window = job.get("window", 50)
        if index < job["burn"] and index >= window and (index + 1) % window == 0:
            recent = np.array(history[index + 1 - window:index + 1])
            moved = np.mean(np.any(np.diff(recent, axis=0) != 0.0, axis=1))
            spread = np.maximum(recent.std(axis=0), 1e-4 * (highs - lows))
            step = spread * 2.4 / math.sqrt(len(PARAMETERS))
            # Target the multivariate optimum of about 0.234.
            step = step * (1.6 if moved > 0.35 else (0.6 if moved < 0.15 else 1.0))
            step = np.maximum(step, 1e-4 * (highs - lows))
    return {"draws": np.array(history[job["burn"]:]),
            "accepted": accepted, "proposed": proposed}


def adaptive_metropolis(data, observation: Observation, chains: int = 4,
                        draws: int = 600, burn: int = 200, side: int = 48,
                        spacing_m: float = 50.0, seed: int = 1729,
                        router=None, slices=None, workers: int = 4,
                        window: int = 50, discrepancy_m: float = 0.0) -> dict:
    """Adaptive Metropolis with R-hat and ESS, over the four rates and the datum.

    The proposal covariance adapts to the chain history during burn-in and is
    then frozen, because a proposal that keeps adapting is not a Markov chain and
    its stationary distribution is not the posterior.

    `slices` optionally supplies a second epoch; when present the likelihood sums
    over both, which is the arrangement that breaks the similarity ridge.
    """
    rng = np.random.default_rng(seed)
    lows = np.array([PRIOR[name][0] for name in PARAMETERS])
    highs = np.array([PRIOR[name][1] for name in PARAMETERS])
    scale = 0.05 * (highs - lows)
    started = time.perf_counter()

    def posterior(vector):
        prior = _log_prior(vector)
        if not np.isfinite(prior):
            return -np.inf
        theta = dict(zip(PARAMETERS, vector))
        if slices is None:
            surface = simulate(theta, side, spacing_m, router=router)
            return prior + log_likelihood(surface, data, observation, theta["datum"])
        total = prior
        for fraction, observed in slices:
            partial = dict(theta)
            partial["logYears"] = theta["logYears"] + math.log10(fraction)
            surface = simulate(partial, side, spacing_m, router=router)
            total += log_likelihood(surface, observed, observation, theta["datum"])
        return total

    jobs = [{"chain": index, "seed": int(rng.integers(0, 2 ** 31)),
             "draws": draws, "burn": burn, "window": window,
             "lows": lows, "highs": highs,
             "scale": scale, "data": data, "observation": observation,
             "side": side, "spacingM": spacing_m, "slices": slices,
             "discrepancyM": discrepancy_m}
            for index in range(chains)]
    # Chains are independent until R-hat is computed, so they run in parallel:
    # four chains in series take about half an hour of forward models.
    if workers and workers > 1 and chains > 1:
        with multiprocessing.Pool(min(workers, chains)) as pool:
            outcomes = pool.map(_run_chain, jobs)
    else:
        outcomes = [_run_chain(job) for job in jobs]
    all_draws = [outcome["draws"] for outcome in outcomes]
    accepted_total = sum(outcome["accepted"] for outcome in outcomes)
    proposed_total = sum(outcome["proposed"] for outcome in outcomes)

    stacked = np.stack(all_draws)
    chain_means = stacked.mean(axis=1)
    chain_vars = stacked.var(axis=1, ddof=1)
    length = stacked.shape[1]
    between = length * chain_means.var(axis=0, ddof=1) if chains > 1 else np.zeros(len(PARAMETERS))
    within = chain_vars.mean(axis=0)
    target = ((length - 1) / length) * within + between / length
    r_hat = np.sqrt(np.maximum(target, 1e-30) / np.maximum(within, 1e-30))

    def ess(column):
        flat = stacked[:, :, column].reshape(-1)
        flat = flat - flat.mean()
        if flat.std() < 1e-12:
            return 0.0
        correlation = np.correlate(flat, flat, mode="full")[len(flat) - 1:]
        correlation = correlation / correlation[0]
        total, index = 0.0, 1
        while index < len(correlation) and correlation[index] > 0.05:
            total += correlation[index]
            index += 1
        return float(len(flat) / (1.0 + 2.0 * total))

    def r_hat_of(values):
        """R-hat for an arbitrary per-draw quantity, shape (chains, draws)."""
        if values.shape[0] < 2:
            return float("nan")
        means = values.mean(axis=1)
        variances = values.var(axis=1, ddof=1)
        length_ = values.shape[1]
        between_ = length_ * means.var(ddof=1)
        within_ = variances.mean()
        target_ = ((length_ - 1) / length_) * within_ + between_ / length_
        return float(math.sqrt(max(target_, 1e-30) / max(within_, 1e-30)))

    # The identifiability claim, measured: the products the similarity law says
    # are identified should mix even when the four parameters separately do not.
    # Reporting only the per-parameter R-hat would say the sampler failed;
    # reporting both says which part of it failed and why.
    combinations = {
        "logUpliftTimesYears": stacked[:, :, 0] + stacked[:, :, 3],
        "logIncisionTimesYears": stacked[:, :, 1] + stacked[:, :, 3],
        "logDiffusivityTimesYears": stacked[:, :, 2] + stacked[:, :, 3],
        "ridgeCoordinate": (stacked[:, :, 0] + stacked[:, :, 1] +
                            stacked[:, :, 2] - stacked[:, :, 3]) / 2.0,
    }
    derived_r_hat = {name: r_hat_of(values) for name, values in combinations.items()}

    flat_draws = stacked.reshape(-1, len(PARAMETERS))
    return {
        "derivedRHat": derived_r_hat,
        "derivedNote": "The similarity law says the products are identified and the "
                       "ridge coordinate is not. These are the numbers that test it: "
                       "a sampler mixing on the products while failing on the ridge "
                       "coordinate has found the degeneracy rather than failed to "
                       "sample.",
        "parameters": list(PARAMETERS),
        "chains": chains, "draws": draws, "burn": burn,
        "acceptanceRate": accepted_total / max(proposed_total, 1),
        "rHat": {name: float(r_hat[i]) for i, name in enumerate(PARAMETERS)},
        "effectiveSampleSize": {name: ess(i) for i, name in enumerate(PARAMETERS)},
        "posteriorMean": {name: float(flat_draws[:, i].mean())
                          for i, name in enumerate(PARAMETERS)},
        "posteriorSd": {name: float(flat_draws[:, i].std())
                        for i, name in enumerate(PARAMETERS)},
        "derived": {
            # The products the data can actually constrain.
            "logUpliftTimesYears": {
                "mean": float((flat_draws[:, 0] + flat_draws[:, 3]).mean()),
                "sd": float((flat_draws[:, 0] + flat_draws[:, 3]).std())},
            "logIncisionTimesYears": {
                "mean": float((flat_draws[:, 1] + flat_draws[:, 3]).mean()),
                "sd": float((flat_draws[:, 1] + flat_draws[:, 3]).std())}},
        "converged": bool(np.all(r_hat < 1.05)),
        "seconds": time.perf_counter() - started,
        "observation": observation.record(),
        "discrepancyM": discrepancy_m,
        "note": "The proposal adapts during burn-in and is then frozen: a proposal "
                "that keeps adapting is not a Markov chain and its stationary "
                "distribution is not the posterior.",
    }


def campaign(torch=None, side: int = 24, spacing_m: float = 50.0,
             chains: int = 4, draws: int = 400, burn: int = 150,
             seed: int = 1729, workers: int = 4) -> dict:
    """The identifiability demonstration, then one-slice and two-slice posteriors.

    The order matters. The ridge is established first, from the similarity law
    and then by walking it, so the posteriors that follow are read as
    constrained where the theory says they can be, rather than as an inference
    that happened to be wide. An analysis that fits first and explains the width
    afterwards cannot tell a flat likelihood from a weak one.
    """
    from geoneural.metrics import hydrology_fast
    truth = {"logUplift": -4.0, "logIncision": -5.0, "logDiffusivity": -2.0,
             "logYears": 5.7, "datum": 0.0}
    observation = Observation(stride=4, crop=4, noise_m=0.5, seed=seed)
    rng = np.random.default_rng(seed)

    final = simulate(truth, side, spacing_m, router=hydrology_fast)
    clean = observation.apply(final)
    # Noise at the surface's own resolution, then observed through the same
    # operator: the correlation length is in surface cells, which is what it
    # claims to be, and the subsampled grid is never asked to carry a structure
    # finer than it can represent.
    data = clean + observation.apply(observation.noise(final.shape, rng))

    # A second, earlier epoch of the same history. This is what carries the
    # timescale a single snapshot cannot.
    early_theta = dict(truth)
    early_theta["logYears"] = truth["logYears"] + math.log10(0.4)
    early = simulate(early_theta, side, spacing_m, router=hydrology_fast)
    early_clean = observation.apply(early)
    early_data = early_clean + observation.apply(observation.noise(early.shape, rng))

    ridge = ridge_direction()
    profile = profile_along_ridge(truth, observation, steps=7, span=0.5,
                                  side=side, spacing_m=spacing_m,
                                  router=hydrology_fast)
    discrepancy = DISCREPANCY_FRACTION * float(np.ptp(final))
    one = adaptive_metropolis(data, observation, chains=chains, draws=draws,
                              burn=burn, side=side, spacing_m=spacing_m,
                              seed=seed, workers=workers,
                              discrepancy_m=discrepancy)
    two = adaptive_metropolis(data, observation, chains=chains, draws=draws,
                              burn=burn, side=side, spacing_m=spacing_m,
                              seed=seed + 1, workers=workers,
                              discrepancy_m=discrepancy,
                              slices=[(0.4, early_data), (1.0, data)])

    def recovery(posterior):
        """Does the interval cover the truth, and is it narrower than the prior?"""
        out = {}
        for name in PARAMETERS:
            low, high = PRIOR[name]
            mean = posterior["posteriorMean"][name]
            sd = posterior["posteriorSd"][name]
            prior_sd = (high - low) / math.sqrt(12.0)
            out[name] = {
                "truth": truth[name], "posteriorMean": mean, "posteriorSd": sd,
                "coversTruth": abs(mean - truth[name]) <= 2.0 * sd,
                "shrinkageVsPrior": float(sd / prior_sd) if prior_sd > 0 else None,
                "informative": bool(prior_sd > 0 and sd / prior_sd < 0.5)}
        return out

    return {
        "schema": SCHEMA, "truth": truth, "side": side, "spacingM": spacing_m,
        "observation": observation.record(),
        "identifiability": ridge,
        "discrepancyM": DISCREPANCY_FRACTION * float(np.ptp(final)),
        "discrepancyFraction": DISCREPANCY_FRACTION,
        "ridgeProfile": profile,
        "oneSlice": {"posterior": one, "recovery": recovery(one)},
        "twoSlice": {"posterior": two, "recovery": recovery(two),
                     "secondEpochFraction": 0.4},
        "qualification":
            "The forward model is the teacher itself, not the emulator: the "
            "emulator fails its drainage and conservation checks, and an inference "
            "built on a forward model known to route water wrongly would compound "
            "the error this study characterises. The cost is a small grid and a "
            "modest chain length, which the diagnostics report rather than hide. "
            "The posterior is under a model that omits climate, lithology, "
            "glaciation and base-level history, and says nothing about any real "
            "landscape.",
    }
