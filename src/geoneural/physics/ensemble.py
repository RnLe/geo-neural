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
physics lives in. Sampling and splitting on the dimensionless groups (Nf, Nh)
stops a test simulation from being a rescaled copy of a training one, a leak
that no random split over U, K, D would catch.

The hardest corner is entirely test. The top decile of (log Nf, log Nh) is held
out whole rather than sampled, so the reported generalisation is to a regime the
emulator has not seen rather than to the interior of one it has.
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

from geoneural.physics import landscape

SCHEMA = "geoneural-ensemble-v1"

# Coarsest spacing at which the teacher audit found the teacher recovers its own
# steady-state concavity. Measured, not chosen.
ADEQUATE_SPACING_M = 100.0

INITIAL_FAMILIES = ("sinusoid", "red-noise", "plateau-notch", "abrupt-uplift-shift")
BOUNDARIES = ("fixed-edges",)

# The dimensionless window the ensemble is restricted to. Outside it the system
# is either uneroded (tiny Nf) or flattened within a step (huge Nf), and neither
# teaches anything about landscape evolution.
FLUVIAL_RANGE = (0.05, 50.0)

# `Nh = D / (K L^2)`, so on a 6.35 km domain with K = 1e-5 an upper end of
# Nh = 1 would mean D = 4e5 m^2/yr, about four million times any measured
# hillslope diffusivity, and an explicit diffusion limit dt < dx^2/4D needing
# ~1e9 steps per simulation. This window is D in [4e-4, 0.4] m^2/yr, which
# brackets the real range; the teacher audit's parameters (D = 5e-3, K = 4e-5)
# sit at Nh = 3.1e-6 inside it. By the similarity law the restriction limits
# which landscapes are reachable, not which scales.
HILLSLOPE_RANGE = (1e-6, 1e-3)

# In the teacher audit, 204 cells hit the incision limiter at dt = 800 yr and
# none at 400 yr. The bare stability limit is set by diffusion alone and reaches
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
                  slope_exponent: float = 1.0) -> dict:
    """The groups the similarity law says the surface actually depends on.

    `Nf` compares fluvial incision to uplift at the domain scale; `Nh` compares
    hillslope diffusion to fluvial incision. Two simulations with the same pair
    are the same landscape at different speeds.
    """
    length = spacing_m * (side - 1)
    area = length * length
    fluvial = k_incision * (area ** area_exponent) * (length ** -slope_exponent) / uplift
    hillslope = diffusivity / (k_incision * (area ** area_exponent) *
                               (length ** (2.0 - slope_exponent)))
    return {"logFluvialNumber": math.log10(fluvial),
            "logHillslopeNumber": math.log10(hillslope),
            "fluvialNumber": fluvial, "hillslopeNumber": hillslope}


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

    `evolve` records scalar history, not surfaces, so the frames an emulator
    needs are captured by running it in equal chunks and keeping the surface at
    each boundary. The chunk is the emulator's prediction interval, and its
    dimensionless size `dt* = dt K` is the third conditioner: two simulations
    asked to advance the same dimensionless amount are asked the same question,
    whatever their dimensional timescales.
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
    frames = [surface.astype(np.float32)]
    closure, steps, clipped = 0.0, 0, 0
    for _ in range(chunks):
        surface, record = landscape.evolve(
            surface, parameters, span, dt_years=job.get("dtYears"),
            record_every=0, base_level=job["baseLevel"], router=hydrology_fast)
        frames.append(surface.astype(np.float32))
        closure = max(closure, abs(float(record.get("closureResidualRelative") or 0.0)))
        steps += int(record.get("steps") or 0)
        clipped += int(record.get("cellsIncisionLimitedTotal") or 0)
    return {
        "id": job["id"], "job": job,
        "seconds": time.perf_counter() - started,
        "finalReliefM": float(np.ptp(surface)),
        "worstClosureResidualRelative": closure,
        "steps": steps, "cellsIncisionLimitedTotal": clipped,
        "dimensionlessStep": span * job["kIncision"],
        "frames": np.stack(frames),
    }


def plan(count: int, side: int, spacing_m: float, years: float, seed: int,
         record_every: int, rng=None) -> list[dict]:
    """Draw the dimensionless groups directly, then solve back for (U, K, D).

    Sampling (U, K, D) log-uniformly and rejecting whatever misses the
    dimensionless window does not work here. With m = 1/2, n = 1 and a square
    domain, `Nf = K A^m L^-n / U` collapses to exactly `K/U`, and `Nh` to
    `D / (K L^2)`, so a box in (U, K, D) meets the window only in a thin sliver
    of its corner. Such an ensemble spans about 0.5 decades of each group:
    a single regime, on which an emulator would learn one landscape and still
    score well.

    Sampling the groups and solving back covers the window by construction, with
    no rejection. The one thing the groups do not fix is the dimensional scale,
    which is held constant through `REFERENCE_K`.
    """
    if spacing_m > ADEQUATE_SPACING_M:
        raise ValueError(
            f"{spacing_m} m is coarser than the {ADEQUATE_SPACING_M} m the teacher "
            "audit found adequate; above it the teacher does not recover its own "
            "steady-state exponent and an emulator trained on it learns a wrong law")
    rng = np.random.default_rng(seed) if rng is None else rng
    length = spacing_m * (side - 1)
    area_exponent, slope_exponent = 0.5, 1.0
    scale = (length ** (2.0 * area_exponent)) ** 1.0  # A^m with A = L^2, i.e. L
    jobs = []
    for index in range(count):
        log_fluvial = rng.uniform(math.log10(FLUVIAL_RANGE[0]), math.log10(FLUVIAL_RANGE[1]))
        log_hillslope = rng.uniform(math.log10(HILLSLOPE_RANGE[0]), math.log10(HILLSLOPE_RANGE[1]))
        # The scale is fixed, not sampled. The similarity law
        # (U, K, D, t) -> (lambda U, lambda K, lambda D, t/lambda) leaves the
        # surface unchanged, so sampling the scale as well would generate
        # rescaled twins of landscapes already in the set, which the
        # dimensionless split exists to keep out. Fixing K also keeps run costs
        # comparable: with K free, the implied D can reach 427 m^2/yr with a
        # 0.3-year stability limit, 6.8 M steps for one simulation against
        # 5,000 for another.
        k_incision = REFERENCE_K
        uplift = k_incision / (10.0 ** log_fluvial)
        diffusivity = (10.0 ** log_hillslope) * k_incision * \
            (length ** (2.0 * area_exponent)) * (length ** (2.0 - slope_exponent))
        groups = dimensionless(uplift, k_incision, diffusivity, spacing_m, side,
                               area_exponent, slope_exponent)
        if abs(groups["logFluvialNumber"] - log_fluvial) > 1e-6 or \
                abs(groups["logHillslopeNumber"] - log_hillslope) > 1e-6:
            raise ValueError(
                "the inversion from (Nf, Nh) back to (K, D) does not reproduce the "
                "groups it was given; the two formulas have drifted apart")
        jobs.append({
            "id": f"sim-{index:04d}", "seed": int(rng.integers(0, 2 ** 31)),
            "side": side, "spacingM": spacing_m, "years": years,
            "uplift": uplift, "kIncision": k_incision, "diffusivity": diffusivity,
            "areaExponent": area_exponent, "slopeExponent": slope_exponent,
            "initialFamily": INITIAL_FAMILIES[index % len(INITIAL_FAMILIES)],
            "baseLevel": "fixed-edges", "recordEvery": record_every,
            "frames": FRAMES_PER_SIMULATION,
            "dtYears": min(landscape.stable_timestep(landscape.Parameters(
                uplift_m_per_year=uplift, k_incision=k_incision,
                area_exponent=area_exponent, slope_exponent=slope_exponent,
                diffusivity_m2_per_year=diffusivity, spacing_m=spacing_m)),
                AUDITED_MAX_DT_YEARS),
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


def generate(out_dir, count: int = 400, side: int = 128, spacing_m: float = 50.0,
             years: float = 2_000_000.0, seed: int = 20260914,
             record_every: int = 0, workers: int = 0,
             frames: int = 9) -> dict:
    """Run the ensemble and write it as one npz plus a manifest."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    every = record_every or max(1, int(years / (frames - 1) / 1000.0))
    jobs = plan(count, side, spacing_m, years, seed, every)
    workers = workers or max(1, min(30, (multiprocessing.cpu_count() or 2) - 2))
    started = time.perf_counter()
    with multiprocessing.Pool(workers) as pool:
        results = pool.map(_simulate, jobs, chunksize=1)
    seconds = time.perf_counter() - started
    stack = {r["id"]: r["frames"] for r in results}
    np.savez_compressed(out_dir / "ensemble.npz", **stack)
    split = split_by_corner(jobs)
    manifest = {
        "schema": SCHEMA, "count": len(jobs), "side": side, "spacingM": spacing_m,
        "domainM": spacing_m * (side - 1), "years": years, "seed": seed,
        "recordEvery": every, "workers": workers, "seconds": seconds,
        "adequateSpacingM": ADEQUATE_SPACING_M,
        "spacingIsAdequate": spacing_m <= ADEQUATE_SPACING_M,
        "initialFamilies": list(INITIAL_FAMILIES),
        "fluvialRange": list(FLUVIAL_RANGE), "hillslopeRange": list(HILLSLOPE_RANGE),
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
