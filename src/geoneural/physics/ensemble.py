"""Training ensemble for the landscape emulator.

An emulator learns the teacher, so the ensemble decides what it can know. Three
rules apply.

It runs at or below the teacher's adequate spacing. The teacher audit found
100 m to be the coarsest mesh on which the teacher still recovers its own
steady-state concavity: at 200 m the exponent is 0.988 against a predicted 0.5
and the drainage network is unresolved. A 128-node domain at 50 m is 6.35 km and
sits inside that limit. An ensemble generated above it would train an emulator
on a teacher that is not solving its own equation, and nothing downstream could
detect it.

It is sampled and split in non-dimensional space. The stream-power system has a
similarity law: (U, K, D, t) -> (lambda U, lambda K, lambda D, t/lambda) leaves
the surface unchanged, so dimensional parameters are not the coordinates the
physics lives in. Sampling and splitting on the dimensionless groups of `units`
(log Nf and log Pe with explicit L, H and T = H / U) stops a test simulation
from being a rescaled copy of a training one, a leak that no random split over
U, K, D would catch. The relief scale H is the declared amplitude scale of the
initial families, `RELIEF_SCALE_M`; an earlier version omitted it, which left
its "fluvial number" with units of 1/m.

The hardest corner is entirely test. The legacy split holds the top decile of
(log Nf, log Nh) out whole; the pilot split (`split_by_block`) cuts the window
into blocks and holds out corner blocks (extrapolation), interior blocks
(interpolation) and one initial family, so the three kinds of generalisation
are reported separately. Trajectories are never divided between splits.

Every simulation records its realised frame times, a volume ledger per
interval (uplift, incision, diffusion, boundary outflow, observed change),
its incision-limiter count and the hash of the solver source that made it.
"""
from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import pathlib
import time

import numpy as np

from geoneural.metrics import hydrology_fast

from geoneural.physics import landscape, units

SCHEMA = "geoneural-ensemble-v2"

# Coarsest spacing at which the teacher audit found the teacher recovers its own
# steady-state concavity. Measured, not chosen.
ADEQUATE_SPACING_M = 100.0

INITIAL_FAMILIES = ("sinusoid", "red-noise", "plateau-notch", "abrupt-uplift-shift")
BOUNDARIES = ("fixed-edges",)

# The relief scale H of the groups: the amplitude scale of the initial families
# (12 to 30 m). Declared, because the groups are only defined once it is.
RELIEF_SCALE_M = 20.0

# The dimensionless window the ensemble is restricted to, as log10 of
# Nf = K L^(2m-n) H^n / U and Pe = Nf / Nh = K L^(2m-n+2) H^(n-1) / D. Outside it
# the system is either uneroded (tiny Nf) or flattened within a step (huge Nf),
# and neither teaches anything about landscape evolution.
LOG_FLUVIAL_RANGE = (0.0, 3.0)

# With K fixed at REFERENCE_K on a 6.35 km domain this window is
# D in [4e-4, 0.4] m^2/yr, which brackets the measured range of hillslope
# diffusivities and keeps the explicit diffusion limit dx^2/4D affordable. By
# the similarity law the restriction limits which landscapes are reachable, not
# which scales.
LOG_PECLET_RANGE = (3.0, 6.0)

# Teacher audit v2: at dt 400 yr (incision Courant number K sqrt(A) dt / dx of
# about 0.6) the surface after 200 kyr differs from a dt 12.5 yr reference by
# 9.4 % of its mean change, with 16 % of D8 receivers different; at Courant
# 0.07 (dt 50 yr) by 0.4 %. Every simulation therefore also respects this
# Courant cap at the largest drainage area expected, a third of the domain.
COURANT_MAX = 0.1

# In the teacher audit (version 1), 204 cells hit the incision limiter at
# dt = 800 yr and none at 400 yr. The bare stability limit is set by diffusion alone and reaches
# 310,000 yr at the low-Nh end of this window, 775x the timestep the teacher was
# validated at, so every simulation is capped here.
AUDITED_MAX_DT_YEARS = 400.0

# The dimensional scale the ensemble is expressed at. Any value gives the same
# landscapes; this one puts U in [2e-7, 2e-4] m/yr and D in [4e-4, 0.4] m^2/yr,
# both inside the measured range for real settings, so the dimensional record is
# readable as well as correct.
REFERENCE_K = 1e-5

# Surfaces kept per simulation, including the initial condition. Eight intervals
# gives the emulator one-step targets and a rollout long enough for the 2, 4 and
# 8-step rollout gates.
FRAMES_PER_SIMULATION = 9


def dimensionless(uplift: float, k_incision: float, diffusivity: float,
                  spacing_m: float, side: int, area_exponent: float = 0.5,
                  slope_exponent: float = 1.0, relief_m: float = RELIEF_SCALE_M) -> dict:
    """The groups of `units` at L = the domain length, H = `relief_m`, T = H / U.

    Two simulations with the same groups, initial family and dimensionless
    elapsed time are the same landscape at different speeds.
    """
    length = spacing_m * (side - 1)
    groups = units.uplift_groups(uplift, k_incision, diffusivity, length, relief_m,
                                 area_exponent, slope_exponent)
    return {key: groups[key] for key in ("logFluvialNumber", "logHillslopeNumber", "logPecletNumber",
                                         "lengthM", "reliefM", "timeYears")}


def initial_surface(family: str, side: int, spacing_m: float, rng) -> np.ndarray:
    """One of four initial conditions, so the emulator cannot learn one shape."""
    axis = np.arange(side, dtype=np.float64) * spacing_m
    grid_y, grid_x = np.meshgrid(axis, axis, indexing="ij")
    length = spacing_m * (side - 1)
    if family == "sinusoid":
        return (12.0 * np.sin(2.0 * np.pi * grid_x / length) *
                np.cos(2.0 * np.pi * grid_y / length))
    if family == "red-noise":
        white = rng.normal(0.0, 1.0, (side, side))
        spectrum = np.fft.rfft2(white)
        ky = np.fft.fftfreq(side)[:, None]
        kx = np.fft.rfftfreq(side)[None, :]
        wavenumber = np.sqrt(ky * ky + kx * kx)
        wavenumber[0, 0] = 1.0
        field = np.fft.irfft2(spectrum / wavenumber, s=(side, side))
        return 15.0 * field / max(float(np.std(field)), 1e-9)
    if family == "plateau-notch":
        surface = np.where(grid_y < length * 0.5, 30.0, 0.0)
        notch = np.abs(grid_x - length * 0.5) < length * 0.08
        return np.where(notch & (grid_y < length * 0.5), 5.0, surface)
    if family == "abrupt-uplift-shift":
        return 20.0 * (grid_y / length) + rng.normal(0.0, 0.5, (side, side))
    raise ValueError(f"Unknown initial family: {family}")


def _simulate(job: dict) -> dict:
    """One simulation. Runs in a worker process; returns the record and the frames.

    The frames are reached one interval at a time, each with full steps and one
    remainder step, so the recorded times are the realised ones and every
    interval carries its own volume ledger (uplift, incision, diffusion,
    boundary outflow, observed change). The interval is the emulator's
    prediction interval, and its dimensionless size `dt* = interval / T` with
    `T = H / U` is the third conditioner.
    """
    rng = np.random.default_rng(job["seed"])
    side = int(job["side"])
    spacing = float(job["spacingM"])
    parameters = landscape.Parameters(
        uplift_m_per_year=job["uplift"], k_incision=job["kIncision"],
        area_exponent=job["areaExponent"], slope_exponent=job["slopeExponent"],
        diffusivity_m2_per_year=job["diffusivity"], spacing_m=spacing)
    surface = initial_surface(job["initialFamily"], side, spacing, rng)
    chunks = int(job["frames"]) - 1
    span = float(job["years"]) / chunks
    started = time.perf_counter()
    frames, times, balances, now = [surface.astype(np.float32)], [0.0], [], 0.0
    steps = partial = clipped = 0
    worst = 0.0
    for _ in range(chunks):
        surface, record = landscape.evolve(
            surface, parameters, span, dt_years=job.get("dtYears"),
            record_every=0, base_level=job["baseLevel"], router=hydrology_fast)
        ledger = record["ledger"]
        now += record["realisedYears"]
        frames.append(surface.astype(np.float32))
        times.append(now)
        steps += int(record["steps"])
        partial += int(record["partialSteps"])
        clipped += int(record["cellsIncisionLimitedTotal"])
        worst = max(worst, abs(float(ledger["closureResidualRelative"])))
        balances.append({key: float(ledger[key]) for key in (
            "upliftVolumeM3", "incisionVolumeM3", "diffusionVolumeM3", "boundaryOutflowVolumeM3",
            "observedVolumeChangeM3", "closureResidualM3")})
    return {
        "id": job["id"], "job": job,
        "seconds": time.perf_counter() - started,
        "finalReliefM": float(np.ptp(surface)),
        "frameTimesYears": times, "realisedYears": now,
        "steps": steps, "partialSteps": partial, "cellsIncisionLimitedTotal": clipped,
        "worstClosureResidualRelative": worst, "balances": balances,
        "dimensionlessStep": span / float(job["timeYears"]),
        "frames": np.stack(frames),
    }


def plan(count: int, side: int, spacing_m: float, years: float, seed: int,
         record_every: int, rng=None) -> list[dict]:
    """Draw the dimensionless groups directly, then solve back for (U, K, D).

    Sampling (U, K, D) log-uniformly and rejecting whatever misses the
    dimensionless window covers only a thin sliver of it, because the groups
    are products of the rates: such an ensemble spans about half a decade of
    each group, a single regime on which an emulator would learn one landscape
    and still score well.

    Sampling (log Nf, log Pe) and solving back through `units` covers the window
    by construction, with no rejection. The one thing the groups do not fix is
    the dimensional speed, which is held constant through `REFERENCE_K`.
    """
    if spacing_m > ADEQUATE_SPACING_M:
        raise ValueError(
            f"{spacing_m} m is coarser than the {ADEQUATE_SPACING_M} m the teacher "
            "audit found adequate; above it the teacher does not recover its own "
            "steady-state exponent and an emulator trained on it learns a wrong law")
    rng = np.random.default_rng(seed) if rng is None else rng
    length = spacing_m * (side - 1)
    area_exponent, slope_exponent = 0.5, 1.0
    jobs = []
    for index in range(count):
        log_fluvial = rng.uniform(*LOG_FLUVIAL_RANGE)
        log_peclet = rng.uniform(*LOG_PECLET_RANGE)
        # The speed is fixed, not sampled. The similarity law
        # (U, K, D, t) -> (lambda U, lambda K, lambda D, t/lambda) leaves the
        # surface unchanged, so sampling it as well would generate rescaled twins
        # of landscapes already in the set, which the dimensionless split exists
        # to keep out. Fixing K also keeps run costs comparable.
        rates = units.rates_for_fixed_incision(log_fluvial, log_peclet, REFERENCE_K, length,
                                               RELIEF_SCALE_M, area_exponent, slope_exponent)
        groups = dimensionless(rates["uplift"], rates["kIncision"], rates["diffusivity"],
                               spacing_m, side, area_exponent, slope_exponent)
        if abs(groups["logFluvialNumber"] - log_fluvial) > 1e-9 or \
                abs(groups["logPecletNumber"] - log_peclet) > 1e-9:
            raise ValueError(
                "the inversion from (Nf, Pe) back to (U, D) does not reproduce the "
                "groups it was given; the two formulas have drifted apart")
        parameters = landscape.Parameters(
            uplift_m_per_year=rates["uplift"], k_incision=rates["kIncision"],
            area_exponent=area_exponent, slope_exponent=slope_exponent,
            diffusivity_m2_per_year=rates["diffusivity"], spacing_m=spacing_m)
        jobs.append({
            "id": f"sim-{index:04d}", "seed": int(rng.integers(0, 2 ** 31)),
            "side": side, "spacingM": spacing_m, "years": years,
            "uplift": rates["uplift"], "kIncision": rates["kIncision"],
            "diffusivity": rates["diffusivity"],
            "areaExponent": area_exponent, "slopeExponent": slope_exponent,
            "initialFamily": INITIAL_FAMILIES[index % len(INITIAL_FAMILIES)],
            "baseLevel": "fixed-edges", "recordEvery": record_every,
            "frames": FRAMES_PER_SIMULATION,
            "dtYears": min(landscape.stable_timestep(parameters), AUDITED_MAX_DT_YEARS,
                           COURANT_MAX * spacing_m / (rates["kIncision"] * length / math.sqrt(3.0))),
            "dimensionlessYears": years / groups["timeYears"],
            **groups})
    return jobs


def split_by_corner(jobs: list[dict], decile: float = 0.9) -> dict:
    """Hold out the hardest corner of (log Nf, log Nh) wholesale.

    A random split over simulations would put a rescaled twin of every test
    simulation in the training set, because the similarity law makes such twins
    exist. Splitting on the dimensionless groups is the only split here that
    holds anything out.
    """
    fluvial = np.array([j["logFluvialNumber"] for j in jobs])
    hillslope = np.array([j["logHillslopeNumber"] for j in jobs])
    # A joint top-decile corner (both groups above their 90th percentile) holds
    # only 1 % of independently drawn samples, which on a 400-simulation plan can
    # be a single simulation. The corner is taken along the diagonal of the two
    # normalised groups instead: still a contiguous region of the plane in the
    # direction of strongest erosion, and actually a decile.
    def unit(values):
        span = float(values.max() - values.min())
        return (values - values.min()) / (span if span > 0 else 1.0)

    hardness = unit(fluvial) + unit(hillslope)
    cut = float(np.quantile(hardness, decile))
    corner = hardness >= cut
    cut_f = float(np.quantile(fluvial, decile))
    cut_h = float(np.quantile(hillslope, decile))
    rest = np.flatnonzero(~corner)
    held = np.flatnonzero(corner)
    order = rest[np.argsort(fluvial[rest])]
    validation = order[::5]
    training = np.setdiff1d(rest, validation)
    if held.size == 0:
        raise ValueError(
            "the top-decile corner of (log Nf, log Nh) is empty, so the test set "
            "holds nothing out. That happens when the ensemble spans too little "
            "of the dimensionless window to have a corner at all")
    return {
        "trainIds": [jobs[i]["id"] for i in sorted(training)],
        "validationIds": [jobs[i]["id"] for i in sorted(validation)],
        "testIds": [jobs[i]["id"] for i in sorted(held)],
        "cornerCut": {"logFluvialNumber": cut_f, "logHillslopeNumber": cut_h,
                      "decile": decile, "hardnessCut": cut,
                      "hardness": "normalised log Nf + normalised log Nh"},
        "note": "The test set is the top-decile corner of (log Nf, log Nh) taken "
                "whole, not a random sample. Under the similarity law a random "
                "split leaves a rescaled twin of every test simulation in "
                "training, so it holds nothing out.",
    }


def split_by_block(jobs: list[dict], blocks: int = 4, held_family: str = "abrupt-uplift-shift") -> dict:
    """Whole trajectories into splits by parameter block, plus a held-out initial family.

    The (log Nf, log Pe) window is cut into `blocks` x `blocks` cells. The two
    corner cells along the strong-erosion diagonal are extrapolation test,
    two interior cells interpolation test, two further cells validation, and
    the rest training. Within the training cells, every trajectory of
    `held_family` is moved to its own test split, so no training trajectory
    starts from that family. A trajectory is never divided between splits.
    """
    f_lo, f_hi = LOG_FLUVIAL_RANGE
    p_lo, p_hi = LOG_PECLET_RANGE
    def cell(job):
        i = min(int((job["logFluvialNumber"] - f_lo) / (f_hi - f_lo) * blocks), blocks - 1)
        j = min(int((job["logPecletNumber"] - p_lo) / (p_hi - p_lo) * blocks), blocks - 1)
        return i, j
    extrapolation = {(blocks - 1, blocks - 1), (0, 0)}
    interpolation = {(1, 2), (2, 1)}
    validation = {(1, 1), (2, 2)}
    out = {"trainIds": [], "validationIds": [], "testInterpolationIds": [], "testExtrapolationIds": [],
           "testInitialFamilyIds": []}
    for job in jobs:
        where = cell(job)
        if where in extrapolation:
            out["testExtrapolationIds"].append(job["id"])
        elif where in interpolation:
            out["testInterpolationIds"].append(job["id"])
        elif where in validation:
            out["validationIds"].append(job["id"])
        elif job["initialFamily"] == held_family:
            out["testInitialFamilyIds"].append(job["id"])
        else:
            out["trainIds"].append(job["id"])
    out["testIds"] = out["testInterpolationIds"] + out["testExtrapolationIds"]
    out["rule"] = {"blocks": blocks, "extrapolationCells": sorted(extrapolation),
                   "interpolationCells": sorted(interpolation), "validationCells": sorted(validation),
                   "heldInitialFamily": held_family, "coordinates": "log10 Nf and log10 Pe from units"}
    return out


def solver_hash() -> str:
    """sha256 over the teacher and router source, so a manifest names the code that made it."""
    from geoneural.metrics import hydrology
    digest = hashlib.sha256()
    for module in (landscape, units, hydrology, hydrology_fast):
        digest.update(pathlib.Path(module.__file__).read_bytes())
    return digest.hexdigest()


def generate(out_dir, count: int = 400, side: int = 128, spacing_m: float = 50.0,
             years: float = 2_000_000.0, seed: int = 20260914,
             record_every: int = 0, workers: int = 0,
             frames: int = 9, split_rule: str = "corner") -> dict:
    """Run the ensemble and write it as one npz plus a manifest."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    every = record_every or max(1, int(years / (frames - 1) / 1000.0))
    jobs = plan(count, side, spacing_m, years, seed, every)
    workers = workers or max(1, min(20, (multiprocessing.cpu_count() or 2) - 2))
    started = time.perf_counter()
    with multiprocessing.Pool(workers) as pool:
        results = pool.map(_simulate, jobs, chunksize=1)
    seconds = time.perf_counter() - started
    stack = {r["id"]: r["frames"] for r in results}
    np.savez_compressed(out_dir / "ensemble.npz", **stack)
    split = split_by_block(jobs) if split_rule == "block" else split_by_corner(jobs)
    manifest = {
        "schema": SCHEMA, "count": len(jobs), "side": side, "spacingM": spacing_m,
        "domainM": spacing_m * (side - 1), "years": years, "seed": seed,
        "recordEvery": every, "workers": workers, "seconds": seconds,
        "adequateSpacingM": ADEQUATE_SPACING_M,
        "spacingIsAdequate": spacing_m <= ADEQUATE_SPACING_M,
        "initialFamilies": list(INITIAL_FAMILIES),
        "logFluvialRange": list(LOG_FLUVIAL_RANGE), "logPecletRange": list(LOG_PECLET_RANGE),
        "reliefScaleM": RELIEF_SCALE_M, "solver": landscape.SCHEMA, "solverHash": solver_hash(),
        "courantMax": COURANT_MAX, "splitRule": split_rule,
        "split": split,
        "framesPerSimulation": FRAMES_PER_SIMULATION,
        "simulations": [{k: v for k, v in r.items() if k != "frames"}
                        for r in results],
        "arrays": "ensemble.npz, one entry per simulation id, shape (frames, side, side) float32",
        "classification": "RESEARCH_ONLY: synthetic teacher output, not Earth data.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, default=float))
    manifest["contentId"] = hashlib.sha256(
        json.dumps(manifest["simulations"], sort_keys=True, default=float)
        .encode("utf-8")).hexdigest()[:16]
    return manifest
