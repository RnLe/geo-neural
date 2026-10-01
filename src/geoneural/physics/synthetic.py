"""Matched synthetic terrain: process-made against procedural, for the codec campaign.

The question this data answers is whether a codec gains anything from terrain
having been made by erosion, as opposed to merely having terrain-like
statistics. Two sets of fields are generated on the same lattice and matched
pair by pair in the two statistics a generic codec is most sensitive to, the
height standard deviation and the median slope:

* `process`: the landscape teacher (stream-power incision with m = 1/2, n = 1,
  linear hillslope diffusion and uniform uplift, all four edges held at a fixed
  base level) run from a random low-relief surface. Rates are drawn
  log-uniformly over plausible ranges and the duration is a multiple of the
  fluvial response time 1 / K, so the set mixes transient surfaces (a plateau
  still being dissected) with near-steady ones. Each run is simulated natively
  on the 513 x 513 lattice at 10 m; nothing is upsampled.
* `procedural`: spectral synthesis with a power-law spectrum plus ridged noise,
  synthesised on a larger grid and cropped so that it is not periodic. No
  routing, filling or erosion is involved. Each field is scaled to the height
  standard deviation of its process partner, and its spectral exponent is
  bisected until its median slope matches the partner's.

Relief is set after the simulation. With n = 1 the equation is linear in z for
fixed routing, so multiplying a solution and its uplift rate by the same factor
b gives another exact solution (the depression-filling epsilon of 1e-6 m is the
only term that does not scale). Each process field is therefore the teacher's
output multiplied by b, chosen so that its relief equals a target drawn
log-uniformly between 20 and 400 m, which is the span of 5 km tiles in NRW
from the Lower Rhine plain to the Sauerland. The manifest records b and the
effective uplift b U the field is an exact solution for.

What this does not provide: the process fields are a restricted model (no
sediment deposition, lithology, climate, glacial or periglacial history), and
the procedural fields are matched in two statistics only. They differ in
everything else by design, most visibly in drainage: a procedural field has
closed depressions everywhere and no organised network.
"""
from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import pathlib
import time

import numpy as np

from geoneural.common import HOME, utc
from geoneural.metrics import hydrology, hydrology_fast
from geoneural.physics import landscape, units

SCHEMA = "geoneural-synthetic-terrain-v1"
SIDE = 513
SPACING_M = 10.0
COUNT = 48
SEED = 20261004

# Log-uniform ranges for U, K and D, with (K, D) restricted to hillslope lengths
# sqrt(D / K) of 10 to 40 m by rejection. Longer hillslopes would make the
# teacher's diffusion limit, not the incision Courant number, set the step and
# multiply the cost of the near-steady runs several times. U sets only how much
# of the initial noise survives once relief is rescaled.
UPLIFT_RANGE = (5e-5, 5e-4)          # m/yr
INCISION_RANGE = (5e-6, 2e-5)        # 1/yr for m = 1/2, n = 1
DIFFUSIVITY_RANGE = (2e-3, 2e-2)     # m^2/yr
HILLSLOPE_LENGTH_RANGE = (10.0, 40.0)  # m, sqrt(D / K)
# Duration as dimensionless time K t. Dissecting a low-relief plateau is slow:
# in a calibration run (K = 1e-5, D = 1e-2, U = 2e-4) the interior still rose
# at 87 % of the uplift rate at K t = 3, 59 % at K t = 10 and 44 % at
# K t = 14 and 29 % at K t = 18, falling more slowly as it goes. Three strata give
# transient, intermediate and near-steady fields in equal numbers.
DURATION_STRATA = ((1.0, 4.0), (4.0, 15.0), (35.0, 55.0))
# Courant number of the explicit incision at the largest drainage area
# expected (a third of the domain): dt = COURANT dx / (K sqrt(A)). The same
# calibration run, at 1.17, hit the incision limiter in about 13 cells per step
# early on and almost none once the largest basin had shrunk below 6.6 km^2.
COURANT = 0.9
RELIEF_RANGE_M = (20.0, 400.0)
BASE_ELEVATION_RANGE_M = (30.0, 300.0)
# Initial noise std as a fraction of U / K, the relief scale of the steady state
# (S = (U / K) A^-1/2): low relief in the units the equation cares about.
INITIAL_STD_FRACTION = (0.01, 0.1)
STEADY_RATE = 0.05                   # mean |dz/dt| / U below this is called near-steady
PROCEDURAL_PAD = 1024                # synthesis grid; the field is a central crop
BETA_RANGE = (1.2, 5.0)


def _spectral(rng, side: int, beta: float, white=None) -> np.ndarray:
    """A field with power spectrum proportional to k^-beta, zero mean, unit std."""
    white = rng.normal(0.0, 1.0, (side, side)) if white is None else white
    spectrum = np.fft.rfft2(white)
    ky = np.fft.fftfreq(side)[:, None]
    kx = np.fft.rfftfreq(side)[None, :]
    k = np.sqrt(ky * ky + kx * kx)
    k[0, 0] = np.inf
    field = np.fft.irfft2(spectrum * k ** (-beta / 2.0), s=(side, side))
    return (field - field.mean()) / max(float(field.std()), 1e-12)


def initial_surface(rng, relief_scale_m: float, side: int = SIDE) -> tuple[np.ndarray, dict]:
    """Random low-relief start: red noise of 1 to 10 % of U / K, edges at base level 0."""
    beta = float(rng.uniform(1.5, 3.0))
    std = relief_scale_m * float(math.exp(rng.uniform(*np.log(INITIAL_STD_FRACTION))))
    field = std * _spectral(rng, side, beta)
    field[0, :] = field[-1, :] = field[:, 0] = field[:, -1] = 0.0
    return field, {"spectralExponent": beta, "stdM": std, "edges": "0 m"}


def summary(field: np.ndarray, spacing_m: float = SPACING_M) -> dict:
    """Statistics recorded per field. Slope is the centred-difference gradient magnitude."""
    field = np.asarray(field, dtype=np.float64)
    gy, gx = np.gradient(field, spacing_m)
    slope = np.hypot(gx, gy)[1:-1, 1:-1]
    lap = landscape.laplacian(field, spacing_m)[1:-1, 1:-1]
    filled = hydrology_fast.fill_depressions(field)
    depth = filled - field
    side = field.shape[0]
    spectrum = np.abs(np.fft.rfft2(field - field.mean())) ** 2
    ky = np.fft.fftfreq(side)[:, None]
    kx = np.fft.rfftfreq(side)[None, :]
    k = np.sqrt(ky * ky + kx * kx)
    band = (k > 4.0 / side) & (k < 0.25)
    fit = np.polyfit(np.log10(k[band]), np.log10(spectrum[band] + 1e-30), 1)
    return {"meanM": float(field.mean()), "stdM": float(field.std()),
            "minM": float(field.min()), "maxM": float(field.max()),
            "reliefM": float(field.max() - field.min()),
            "slopeMedian": float(np.median(slope)), "slopeP10": float(np.quantile(slope, 0.1)),
            "slopeP90": float(np.quantile(slope, 0.9)),
            "laplacianStdPerM": float(lap.std()),
            "depressionCellFraction": float(np.mean(depth > 1e-3)),
            "depressionMaxDepthM": float(depth.max()),
            "spectralExponent": float(-fit[0])}


def plan_process(count: int = COUNT, seed: int = SEED) -> list[dict]:
    """Rates, durations, relief targets and seeds for every process field."""
    rng = np.random.default_rng(seed)
    children = np.random.SeedSequence(seed).spawn(count)
    jobs = []
    for index in range(count):
        while True:
            k = float(math.exp(rng.uniform(*np.log(INCISION_RANGE))))
            d = float(math.exp(rng.uniform(*np.log(DIFFUSIVITY_RANGE))))
            if HILLSLOPE_LENGTH_RANGE[0] <= math.sqrt(d / k) <= HILLSLOPE_LENGTH_RANGE[1]:
                break
        u = float(math.exp(rng.uniform(*np.log(UPLIFT_RANGE))))
        low, high = DURATION_STRATA[index % len(DURATION_STRATA)]
        kt = float(math.exp(rng.uniform(math.log(low), math.log(high))))
        relief = float(math.exp(rng.uniform(*np.log(RELIEF_RANGE_M))))
        base = float(rng.uniform(*BASE_ELEVATION_RANGE_M))
        parameters = landscape.Parameters(uplift_m_per_year=u, k_incision=k,
                                          diffusivity_m2_per_year=d, spacing_m=SPACING_M)
        largest_area = (SIDE - 1) ** 2 * SPACING_M ** 2 / 3.0
        dt = min(landscape.stable_timestep(parameters),
                 COURANT * SPACING_M / (k * math.sqrt(largest_area)))
        years = kt / k
        jobs.append({"index": index, "seed": int(children[index].generate_state(1)[0]),
                     "uplift": u, "kIncision": k, "diffusivity": d,
                     "hillslopeLengthM": math.sqrt(d / k), "dimensionlessTime": kt,
                     "years": years, "dtYears": dt, "steps": len(landscape.step_plan(years, dt)),
                     "targetReliefM": relief, "baseElevationM": base,
                     "stratum": [low, high]})
    return jobs


def run_process(job: dict) -> dict:
    """One teacher run on the native lattice, rescaled in relief. Worker entry point."""
    rng = np.random.default_rng(job["seed"])
    start, initial = initial_surface(rng, job["uplift"] / job["kIncision"])
    parameters = landscape.Parameters(
        uplift_m_per_year=job["uplift"], k_incision=job["kIncision"],
        diffusivity_m2_per_year=job["diffusivity"], spacing_m=SPACING_M)
    began = time.perf_counter()
    final, record = landscape.evolve(start, parameters, job["years"], dt_years=job["dtYears"],
                                     base_level="fixed-edges", router=hydrology_fast)
    seconds = time.perf_counter() - began
    # Instantaneous rate at the end: one more step, not applied.
    probe, _ = landscape.step(final, parameters, job["dtYears"], router=hydrology_fast)
    interior = (slice(1, -1), slice(1, -1))
    rate = float(np.abs(probe[interior] - final[interior]).mean()) / (job["uplift"] * job["dtYears"])
    relief = float(final.max() - final.min())
    factor = job["targetReliefM"] / max(relief, 1e-9)
    field = (job["baseElevationM"] + factor * (final - final.min())).astype(np.float32)
    ledger = record["ledger"]
    return {**job, "initial": initial, "seconds": seconds,
            "realisedYears": record["realisedYears"], "stepsTaken": record["steps"],
            "cellsIncisionLimitedTotal": record["cellsIncisionLimitedTotal"],
            "closureResidualRelative": ledger["closureResidualRelative"],
            "simulatedReliefM": relief, "reliefFactor": factor,
            "effectiveUplift": factor * job["uplift"],
            "normalisedRate": rate, "nearSteady": bool(rate < STEADY_RATE),
            "field": field}


def _ridged(rng, side: int, octaves: int = 6, base_wavelength: float = 256.0) -> np.ndarray:
    """Ridged multifractal noise: sum over octaves of (1 - |n|)^2, n band-limited noise."""
    total = np.zeros((side, side))
    weight, wavelength = 1.0, base_wavelength
    ky = np.fft.fftfreq(side)[:, None]
    kx = np.fft.rfftfreq(side)[None, :]
    k = np.sqrt(ky * ky + kx * kx)
    for _ in range(octaves):
        centre = 1.0 / wavelength
        band = np.exp(-0.5 * (np.log(np.maximum(k, 1e-12) / centre) / 0.35) ** 2)
        noise = np.fft.irfft2(np.fft.rfft2(rng.normal(0.0, 1.0, (side, side))) * band, s=(side, side))
        noise /= max(float(noise.std()), 1e-12)
        total += weight * (1.0 - np.minimum(np.abs(noise) / 2.0, 1.0)) ** 2
        weight *= 0.5
        wavelength /= 2.0
    return (total - total.mean()) / max(float(total.std()), 1e-12)


def _median_slope(field: np.ndarray, spacing_m: float = SPACING_M) -> float:
    gy, gx = np.gradient(field, spacing_m)
    return float(np.median(np.hypot(gx, gy)[1:-1, 1:-1]))


def run_procedural(job: dict) -> dict:
    """One procedural field matched to its partner's std and median slope. Worker entry point."""
    rng = np.random.default_rng(job["seed"])
    pad = PROCEDURAL_PAD
    white = rng.normal(0.0, 1.0, (pad, pad))
    ridged = _ridged(rng, pad)
    weight = float(rng.uniform(0.2, 0.6))
    offset = (pad - SIDE) // 2
    crop = (slice(offset, offset + SIDE), slice(offset, offset + SIDE))
    target_std, target_slope = job["targetStdM"], job["targetSlopeMedian"]

    def build(beta, weight):
        field = _spectral(None, pad, beta, white) + weight * ridged
        field = field[crop]
        return target_std * (field - field.mean()) / max(float(field.std()), 1e-12)

    # Bisect beta at the drawn ridged weight. If even the smoothest spectrum is
    # too rough (the ridged creases carry slope at every scale), halve the
    # weight and try again, down to a pure power-law field.
    drawn = weight
    for weight in (drawn, drawn / 2.0, drawn / 4.0, 0.0):
        low, high = BETA_RANGE
        slope_low, slope_high = _median_slope(build(low, weight)), _median_slope(build(high, weight))
        matched = slope_high <= target_slope <= slope_low
        if matched:
            break
    warp = 0.0
    if not matched and target_slope < slope_high:
        # Even the smoothest spectrum is too steep: the partner is a flat upland cut by
        # steep margins, whose median slope is far below what its std implies for a
        # Gaussian field. A monotone warp z -> tanh(k z) / k flattens high and low ground
        # into terraces joined by steep steps, at fixed std and with bounded relief; k is
        # bisected instead of beta.
        beta, weight = BETA_RANGE[1], drawn
        base = build(beta, weight) / target_std

        def warped(k):
            field = np.tanh(k * base) if k > 0 else base
            return target_std * (field - field.mean()) / max(float(field.std()), 1e-12)

        low_k, high_k = 0.0, 40.0
        if _median_slope(warped(high_k)) <= target_slope:
            for _ in range(40):
                warp = 0.5 * (low_k + high_k)
                if _median_slope(warped(warp)) > target_slope:
                    low_k = warp
                else:
                    high_k = warp
            warp = 0.5 * (low_k + high_k)
            matched = True
        else:
            warp = high_k
        build_final = lambda: warped(warp)  # noqa: E731
    elif not matched:
        beta = low if target_slope > slope_low else high
        build_final = lambda: build(beta, weight)  # noqa: E731
    else:
        for _ in range(40):
            beta = 0.5 * (low + high)
            if _median_slope(build(beta, weight)) > target_slope:
                low = beta
            else:
                high = beta
        beta = 0.5 * (low + high)
        build_final = lambda: build(beta, weight)  # noqa: E731
    field = build_final()
    field = (job["baseElevationM"] + field - field.min()).astype(np.float32)
    return {**job, "spectralExponent": beta, "ridgedWeight": weight, "ridgedWeightDrawn": drawn,
            "warp": warp, "slopeMatched": bool(matched), "field": field}


def code_hash() -> str:
    """sha256 over the source of every module that shapes the fields."""
    digest = hashlib.sha256()
    for module in (landscape, units, hydrology, hydrology_fast):
        digest.update(pathlib.Path(module.__file__).read_bytes())
    digest.update(pathlib.Path(__file__).read_bytes())
    return digest.hexdigest()


DEFINITIONS = {
    "lattice": "513 x 513 nodes at 10 m spacing (5120 m square), row-major, float32 metres, row 0 north",
    "process": ("landscape teacher v2 (explicit Euler; D8 stream power with m = 1/2, n = 1, drainage area "
                "from priority-flood filled D8; five-point linear diffusion; uniform uplift; all four edges "
                "fixed at 0 m) from red noise with std 1 to 10 % of U / K, simulated natively on the lattice; "
                "U log-uniform in 5e-5 to 5e-4 m/yr, K in 5e-6 to 2e-5 1/yr and D in 2e-3 to 2e-2 m^2/yr, "
                "(K, D) restricted to sqrt(D/K) in 10-40 m; dt = min(teacher diffusion limit, "
                "0.9 dx / (K sqrt(A))) with A a third of the domain; "
                "duration K t drawn in three strata (1-4, 4-15, 35-55); output = base + b (z - min z) with "
                "b = target relief / simulated relief, target relief log-uniform in 20-400 m, base uniform "
                "in 30-300 m. With n = 1 the rescaled field is an exact solution for uplift b U."),
    "procedural": ("power-law spectral field (exponent beta) plus w times ridged multifractal noise (6 octaves "
                   "of (1 - |n|/2)^2 over band-limited noise, wavelengths 2560 m down to 80 m, w uniform in "
                   "0.2-0.6, halved up to twice and then set to 0 if no beta matches), synthesised on 1024 x 1024 "
                   "and centre-cropped; scaled to the partner's height std; beta bisected in [1.2, 5] until "
                   "the median slope equals the partner's; where even beta = 5 is too steep (flat uplands "
                   "with steep margins), beta = 5 and the field is warped monotonically, z -> tanh(k z) on "
                   "the standardised field, with k bisected in [0, 40] to match, then rescaled to the std; "
                   "shifted so "
                   "its minimum equals the partner's base elevation. No routing, filling or erosion."),
    "slope": "centred-difference gradient magnitude (numpy.gradient at 10 m), interior nodes",
    "nearSteady": f"mean |dz/dt| over interior nodes divided by U, one teacher step after the end, below {STEADY_RATE}",
}


def generate(out_dir=None, count: int = COUNT, seed: int = SEED, workers: int = 20) -> dict:
    """Write both sets, their manifests and a README. Process first, then the matched procedural set."""
    out = pathlib.Path(out_dir) if out_dir else HOME / "synthetic"
    for name in ("process", "procedural"):
        (out / name).mkdir(parents=True, exist_ok=True)
    jobs = plan_process(count, seed)
    # Longest runs first, so the pool does not end on one straggler.
    order = sorted(jobs, key=lambda j: -j["steps"])
    started = time.perf_counter()
    with multiprocessing.Pool(min(workers, count)) as pool:
        process = sorted(pool.map(run_process, order, chunksize=1), key=lambda r: r["index"])
    process_seconds = time.perf_counter() - started
    partners = []
    children = np.random.SeedSequence(seed + 1).spawn(count)
    for row in process:
        stats = summary(row["field"])
        row["summary"] = stats
        partners.append({"index": row["index"], "seed": int(children[row["index"]].generate_state(1)[0]),
                         "targetStdM": stats["stdM"], "targetSlopeMedian": stats["slopeMedian"],
                         "baseElevationM": row["baseElevationM"]})
    started = time.perf_counter()
    with multiprocessing.Pool(min(workers, count)) as pool:
        procedural = sorted(pool.map(run_procedural, partners, chunksize=1), key=lambda r: r["index"])
    procedural_seconds = time.perf_counter() - started
    for row in procedural:
        row["summary"] = summary(row["field"])
    digest = code_hash()
    manifests = {}
    for name, rows, seconds in (("process", process, process_seconds),
                                ("procedural", procedural, procedural_seconds)):
        entries = []
        for row in rows:
            path = out / name / f"{row['index']:03d}.npy"
            np.save(path, row["field"])
            entries.append({"file": f"{name}/{path.name}",
                            **{k: v for k, v in row.items() if k != "field"}})
        manifest = {"schema": SCHEMA, "set": name, "count": len(rows), "side": SIDE,
                    "spacingM": SPACING_M, "dtype": "float32", "seed": seed,
                    "generatorCodeHash": digest, "createdUtc": utc(),
                    "wallSeconds": seconds, "workers": workers,
                    "definition": DEFINITIONS[name], "lattice": DEFINITIONS["lattice"],
                    "slopeDefinition": DEFINITIONS["slope"], "fields": entries,
                    "classification": "RESEARCH_ONLY: synthetic, not Earth data."}
        if name == "process":
            manifest["nearSteadyDefinition"] = DEFINITIONS["nearSteady"]
            manifest["solver"] = landscape.SCHEMA
        (out / name / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
        manifests[name] = manifest
    (out / "README.txt").write_text(readme(manifests))
    return manifests


def regenerate_procedural(out_dir=None, workers: int = 20) -> dict:
    """Rebuild only the procedural set against an existing process set, and rewrite the README."""
    out = pathlib.Path(out_dir) if out_dir else HOME / "synthetic"
    process = json.loads((out / "process" / "manifest.json").read_text())
    seed = int(process["seed"])
    children = np.random.SeedSequence(seed + 1).spawn(process["count"])
    partners = [{"index": f["index"], "seed": int(children[f["index"]].generate_state(1)[0]),
                 "targetStdM": f["summary"]["stdM"], "targetSlopeMedian": f["summary"]["slopeMedian"],
                 "baseElevationM": f["baseElevationM"]} for f in process["fields"]]
    started = time.perf_counter()
    with multiprocessing.Pool(min(workers, len(partners))) as pool:
        rows = sorted(pool.map(run_procedural, partners, chunksize=1), key=lambda r: r["index"])
    entries = []
    for row in rows:
        row["summary"] = summary(row["field"])
        path = out / "procedural" / f"{row['index']:03d}.npy"
        np.save(path, row["field"])
        entries.append({"file": f"procedural/{path.name}", **{k: v for k, v in row.items() if k != "field"}})
    manifest = {"schema": SCHEMA, "set": "procedural", "count": len(rows), "side": SIDE, "spacingM": SPACING_M,
                "dtype": "float32", "seed": seed, "generatorCodeHash": code_hash(), "createdUtc": utc(),
                "wallSeconds": time.perf_counter() - started, "workers": workers,
                "definition": DEFINITIONS["procedural"], "lattice": DEFINITIONS["lattice"],
                "slopeDefinition": DEFINITIONS["slope"], "fields": entries,
                "classification": "RESEARCH_ONLY: synthetic, not Earth data."}
    (out / "procedural" / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    manifests = {"process": process, "procedural": manifest}
    (out / "README.txt").write_text(readme(manifests))
    return manifests


def readme(manifests: dict) -> str:
    """The plain-text description written next to the data."""
    process, procedural = manifests["process"]["fields"], manifests["procedural"]["fields"]

    def span(rows, key):
        values = [r["summary"][key] for r in rows]
        return f"{min(values):.3g} to {max(values):.3g} (median {float(np.median(values)):.3g})"

    steady = sum(1 for r in process if r["nearSteady"])
    matched = sum(1 for r in procedural if r["slopeMatched"])
    warped = sum(1 for r in procedural if r.get("warp", 0.0) > 0.0)
    rates = {}
    for r in process:
        rates.setdefault(tuple(r["stratum"]), []).append(r["normalisedRate"])
    clipped = [r["cellsIncisionLimitedTotal"] / max(r["stepsTaken"], 1) for r in process]
    lines = [
        "Synthetic terrain for H4 (process-made versus procedural). Generated by",
        "geoneural.physics.synthetic (command: geoneural synthetic-terrain).",
        "",
        f"Lattice: {DEFINITIONS['lattice']}.",
        f"Files: process/NNN.npy and procedural/NNN.npy, {len(process)} each; field i of one set is",
        "matched to field i of the other. manifest.json in each folder lists parameters, seeds,",
        "realised times and summary statistics; generatorCodeHash covers the generating source.",
        "",
        "process: " + DEFINITIONS["process"],
        "",
        "procedural: " + DEFINITIONS["procedural"],
        "",
        "Matching: height std equal by construction; median slope matched by bisection",
        f"({matched} of {len(procedural)} matched, {warped} of them through the monotone warp).",
        f"Slope: {DEFINITIONS['slope']}.",
        "",
        f"Process set: {steady} of {len(process)} near-steady ({DEFINITIONS['nearSteady']}).",
        "  normalised rate by duration stratum (K t): " + "; ".join(
            f"{lo:g}-{hi:g}: {min(v):.3f} to {max(v):.3f}" for (lo, hi), v in sorted(rates.items())),
        "  (the longest stratum is late-transient, the interior still rising at a few to 12 % of U)",
        f"  incision limiter: median {float(np.median(clipped)):.2g} cells per step "
        f"(max {max(clipped):.3g} of 263169); per-run totals are in the manifest, not attributed to locations",
        f"  wall time {manifests['process']['wallSeconds'] / 60:.0f} min on {manifests['process']['workers']} workers "
        "(shared host)",
        f"  height std {span(process, 'stdM')} m; relief {span(process, 'reliefM')} m",
        f"  median slope {span(process, 'slopeMedian')}; closed-depression cell fraction "
        f"{span(process, 'depressionCellFraction')}",
        f"Procedural set: height std {span(procedural, 'stdM')} m; relief {span(procedural, 'reliefM')} m",
        f"  median slope {span(procedural, 'slopeMedian')}; closed-depression cell fraction "
        f"{span(procedural, 'depressionCellFraction')}",
        "",
        "Known artefacts: the process fields carry D8 routing artefacts (straight channel segments",
        "along the eight grid directions and parallel valleys on planar slopes; the teacher audit",
        "measures the directional bias), and transient fields keep flat uplifted plateaus. The",
        "procedural fields are rougher at short wavelengths than their partners at equal median slope,",
        "and the warped ones are terraced.",
        "",
        "Not provided: deposition, lithology, climate or glacial history in the process set; the",
        "procedural set matches two statistics only and differs in everything else, most visibly",
        "in drainage (closed depressions everywhere, no organised network).",
        "",
    ]
    return "\n".join(lines)
