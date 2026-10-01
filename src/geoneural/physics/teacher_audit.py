"""Is the landscape teacher a solver, or just a plausible picture?

Matching the steady-state law against the closed form is necessary but not
sufficient: a scheme can reproduce an asymptotic relation and still mis-integrate
the transient, lose mass, or depend on its timestep. A neural network that
accurately reproduces a flawed discretisation is still a flawed physical model,
so resolution and timestep convergence, boundary behaviour and conservation are
established before anything is learned from the teacher.

Version 2 audits version 2 of the teacher (exact elapsed time, receiver-drop
limiter). Each audit has a gate that can fail:

* Timestep refinement. Halve dt against a reference run at the finest dt and
  fit the observed order. Explicit Euler on this operator splitting is first
  order, so the gate is order >= 0.8; the incision limiter and D8 receiver
  switching are both discontinuous and can drag it lower, which is why the number
  of limited cells and the fraction of cells whose D8 receiver differs from the
  reference are reported beside it.
* Non-divisible durations. A duration that is not a multiple of dt is
  integrated with a remainder step; its error must sit on the refinement curve
  and its realised time must equal the request.
* Grid refinement. Hold the physical domain and parameters fixed and refine
  dx. The steady-state exponent must survive, because it is the one property the
  closed form predicts independently of the mesh.
* Conservation. The ledger `evolve` keeps must close.
* Steady state, by a persistence criterion: the interior rate stays below the
  threshold for a declared window, not merely crosses it once.
* Analytic incision. With uplift and stream power only, the steady state on
  the final drainage network is known in closed form,
  z_i = z_r + d_ir (U / K)^(1/n) A_i^(-m/n); the teacher must reproduce it cell
  by cell.
* Manufactured diffusion. A solution chosen in advance, with the source term it
  implies supplied as a spatial uplift field, must be recovered at second order
  in space.
* Rotation. A 90 degree rotation must commute with the teacher; a radially
  symmetric start shows how far D8 routing is from rotational invariance.
* An established solver. Two to three stream-power cases against fastscapelib
  with conventions matched explicitly.
* Budget. The discretisation error at the production timestep, relative to the
  prediction error budget of the tasks that learn from the teacher (proposed
  criterion: below 10 % of it).
"""
from __future__ import annotations

import math
import time

import numpy as np

from geoneural.metrics import hydrology, hydrology_fast
from geoneural.physics import landscape

SCHEMA = "geoneural-teacher-audit-v2"
ROUTER = hydrology_fast  # bit-identical to the reference router; speed only


def initial_surface(side: int, relief_m: float = 30.0, tilt_m: float = 10.0,
                    seed: int = 1729) -> np.ndarray:
    """A two-dimensional start surface. One-dimensional variation routes flow in
    one direction only, so no cell accumulates a channel and nothing exceeds a
    channel threshold."""
    axis = np.linspace(0.0, 1.0, side)
    field = relief_m * np.sin(2.0 * np.pi * axis)[None, :] * np.cos(2.0 * np.pi * axis)[:, None]
    return field + np.linspace(0.0, tilt_m, side)[:, None]


def _receivers(surface: np.ndarray, spacing_m: float) -> np.ndarray:
    return hydrology.d8_receivers(ROUTER.fill_depressions(surface), spacing_m)


def _evolve(start, parameters, years, dt, **kwargs):
    return landscape.evolve(start, parameters, years, dt_years=dt, base_level="fixed-edges",
                            router=ROUTER, **kwargs)


def timestep_refinement(parameters: landscape.Parameters, side: int = 64,
                        years: float = 200_000.0,
                        steps_list=(800.0, 400.0, 200.0, 100.0, 50.0),
                        reference_dt: float = 12.5) -> dict:
    """Halve dt and watch the answer stop moving. Or fail to."""
    start = initial_surface(side)
    reference, _ = _evolve(start, parameters, years, reference_dt)
    reference_receivers = _receivers(reference, parameters.spacing_m)
    change = float(np.abs(reference - start).mean())
    rows = []
    for dt in steps_list:
        surface, report = _evolve(start, parameters, years, dt)
        error = np.abs(surface - reference)
        rows.append({"dtYears": float(dt), "realisedYears": report["realisedYears"],
                     "l1M": float(error.mean()), "lInfM": float(error.max()),
                     "l1RelativeToChange": float(error.mean()) / max(change, 1e-12),
                     "receiverChangeFraction": float(np.mean(
                         _receivers(surface, parameters.spacing_m) != reference_receivers)),
                     "cellsIncisionLimitedTotal": report["cellsIncisionLimitedTotal"],
                     "closureResidualRelative": report["ledger"]["closureResidualRelative"]})
    orders = []
    for coarse, fine in zip(rows[:-1], rows[1:]):
        if fine["l1M"] > 0.0 and coarse["l1M"] > 0.0:
            ratio = coarse["dtYears"] / fine["dtYears"]
            orders.append(math.log(coarse["l1M"] / fine["l1M"]) / math.log(ratio))
    finest_order = orders[-1] if orders else 0.0
    monotone = all(a["l1M"] >= b["l1M"] for a, b in zip(rows[:-1], rows[1:]))
    return {"referenceDtYears": reference_dt, "years": years, "meanAbsChangeM": change,
            "rows": rows, "observedOrders": orders,
            "finestObservedOrder": float(finest_order), "monotone": monotone,
            "agrees": bool(finest_order >= 0.8 and monotone),
            "note": "Explicit Euler on this splitting is first order. receiverChangeFraction is the "
                    "share of cells whose D8 receiver on the final surface differs from the "
                    "reference run's; receiver switching is discontinuous and is what can pull "
                    "the observed order below one."}


def non_divisible(parameters: landscape.Parameters, side: int = 64, years: float = 99_900.0,
                  dt: float = 400.0, reference_dt: float = 12.5) -> dict:
    """A duration that dt does not divide: exact time, and an error on the refinement curve."""
    flat = landscape.Parameters(uplift_m_per_year=1.0, k_incision=0.0, diffusivity_m2_per_year=0.0)
    uplift_only, record = landscape.evolve(np.zeros((8, 8)), flat, 1000.0, dt_years=400.0)
    start = initial_surface(side)
    reference, _ = _evolve(start, parameters, years, reference_dt)
    odd, odd_record = _evolve(start, parameters, years, dt)
    near, _ = _evolve(start, parameters, math.floor(years / dt) * dt, dt)
    near_reference, _ = _evolve(start, parameters, math.floor(years / dt) * dt, reference_dt)
    odd_error = float(np.abs(odd - reference).mean())
    near_error = float(np.abs(near - near_reference).mean())
    exact = abs(float(uplift_only.mean()) - 1000.0) < 1e-9 and abs(record["realisedYears"] - 1000.0) < 1e-9
    timed = abs(odd_record["realisedYears"] - years) < 1e-6
    return {"upliftOnly": {"requestedYears": 1000.0, "dtYears": 400.0,
                           "meanM": float(uplift_only.mean()), "realisedYears": record["realisedYears"],
                           "steps": record["steps"], "partialSteps": record["partialSteps"]},
            "years": years, "dtYears": dt, "realisedYears": odd_record["realisedYears"],
            "partialSteps": odd_record["partialSteps"],
            "l1M": odd_error, "divisibleNeighbourL1M": near_error,
            "errorRatio": odd_error / max(near_error, 1e-30),
            "agrees": bool(exact and timed and odd_error <= 1.5 * near_error),
            "note": "The odd duration ends with a remainder step; its error against the dt 12.5 "
                    "reference must be comparable to that of the nearest divisible duration."}


def grid_refinement(parameters: landscape.Parameters, domain_m: float = 6400.0,
                    sides=(33, 65, 129), years: float = 200_000.0,
                    dt_years: float = 50.0) -> dict:
    """Refine dx at fixed physical domain. The exponent must survive the mesh."""
    rows = []
    for side in sides:
        spacing = domain_m / (side - 1)
        scaled = landscape.Parameters(
            uplift_m_per_year=parameters.uplift_m_per_year,
            k_incision=parameters.k_incision,
            diffusivity_m2_per_year=parameters.diffusivity_m2_per_year,
            spacing_m=spacing,
            area_exponent=parameters.area_exponent,
            slope_exponent=parameters.slope_exponent)
        limit = landscape.stable_timestep(scaled)
        dt = min(dt_years, limit * 0.9)
        surface, report = _evolve(initial_surface(side), scaled, years, dt)
        steady = landscape.steady_state_report(surface, scaled)
        rows.append({"side": side, "spacingM": spacing, "dtYears": dt,
                     "reliefM": report["reliefM"],
                     "concavity": steady.get("concavity"),
                     "medianRatio": steady.get("medianRatio"),
                     "agrees": steady.get("agrees"),
                     "closureResidualRelative": report["ledger"]["closureResidualRelative"]})
    predicted = parameters.area_exponent / parameters.slope_exponent
    for row in rows:
        row["predictedConcavity"] = predicted
        row["concavityError"] = (None if row["concavity"] is None
                                 else abs(row["concavity"] - predicted))
    errors = [(row["spacingM"], row["concavityError"]) for row in rows
              if row["concavityError"] is not None]
    converging = all(a[1] >= b[1] for a, b in zip(errors[:-1], errors[1:]))
    adequate = [row["spacingM"] for row in rows
                if row["concavityError"] is not None and row["concavityError"] <= 0.15
                and row["agrees"]]
    return {"domainM": domain_m, "rows": rows,
            "predictedConcavity": predicted,
            "converging": bool(converging),
            "adequateSpacingM": max(adequate) if adequate else None,
            "agrees": bool(converging and adequate),
            "note": "The steady-state exponent is the one property the closed form predicts "
                    "independently of the mesh, so it is what a grid refinement can falsify. "
                    "`adequateSpacingM` is the coarsest spacing that still recovers it."}


def conservation(parameters: landscape.Parameters, side: int = 64,
                 years: float = 100_000.0, dt_years: float = 200.0,
                 tolerance: float = 1e-9) -> dict:
    """The books must close, and diffusion must take exactly nothing out."""
    rows = []
    for base_level in (None, "fixed-edges"):
        _, report = landscape.evolve(initial_surface(side), parameters, years,
                                     dt_years=dt_years, base_level=base_level, router=ROUTER)
        ledger = report["ledger"]
        rows.append({"baseLevel": base_level or "none (closed system)", **ledger,
                     "closes": bool(ledger["closureResidualRelative"] < tolerance),
                     "diffusionIsZero": bool(abs(ledger["diffusionVolumeM3"])
                                             < tolerance * max(abs(ledger["upliftVolumeM3"]), 1.0))})
    return {"tolerance": tolerance, "rows": rows,
            "agrees": all(row["closes"] and row["diffusionIsZero"] for row in rows),
            "note": "Zero-flux edges make the discrete Laplacian sum to zero exactly. Boundary "
                    "outflow is non-zero only under a base level, which makes the system open."}


def steady_state_time(parameters: landscape.Parameters, side: int = 64,
                      years: float = 2_000_000.0, dt_years: float = 200.0,
                      threshold: float = 1e-3, persistence_years: float = 200_000.0) -> dict:
    """When does it stop moving, by a criterion that requires it to stay stopped?

    The rate is the interior mean |dh| per step over the uplift per step. A
    single crossing can be a pause between two receiver reorganisations, so the
    surface counts as settled only from the first time after which the rate
    stays below the threshold for `persistence_years`.
    """
    surface = initial_surface(side)
    uplift_per_step = parameters.uplift_m_per_year * dt_years
    plan = landscape.step_plan(years, dt_years)
    interior = (slice(1, -1), slice(1, -1))
    boundary = np.zeros(surface.shape, dtype=bool)
    boundary[0, :] = boundary[-1, :] = boundary[:, 0] = boundary[:, -1] = True
    pinned = surface[boundary].copy()
    rates, times, trace, now = [], [], [], 0.0
    for index, h in enumerate(plan):
        previous = surface
        surface, _ = landscape.step(surface, parameters, h, router=ROUTER)
        surface[boundary] = pinned
        now += h
        rate = float(np.abs(surface[interior] - previous[interior]).mean()) / (
            parameters.uplift_m_per_year * h)
        rates.append(rate)
        times.append(now)
        if index % max(len(plan) // 40, 1) == 0:
            trace.append({"step": index, "years": now, "normalisedRate": rate,
                          "reliefM": float(surface.max() - surface.min())})
    rates_a, times_a = np.asarray(rates), np.asarray(times)
    first = next((t for r, t in zip(rates, times) if r < threshold), None)
    above = np.flatnonzero(rates_a >= threshold)
    settled = None
    if above.size == 0:
        settled = float(times_a[0])
    elif above[-1] + 1 < len(rates):
        candidate = float(times_a[above[-1] + 1])
        if times_a[-1] - candidate >= persistence_years:
            settled = candidate
    return {"thresholdNormalisedRate": threshold, "persistenceYears": persistence_years,
            "firstCrossingYears": first, "settledAtYears": settled,
            "finalRate": float(rates_a[-1]), "maxRateInLastWindow": float(
                rates_a[times_a >= times_a[-1] - persistence_years].max()),
            "reliefTimescaleYears": float((surface.max() - surface.min())
                                          / parameters.uplift_m_per_year),
            "finalReliefM": float(surface.max() - surface.min()),
            "trace": trace, "upliftPerStepM": uplift_per_step,
            "note": "Settled means the rate fell below the threshold and stayed below it for "
                    "persistenceYears until the end of the run; a first crossing alone is "
                    "reported but is not the criterion."}


def analytic_incision(side: int = 48, spacing_m: float = 100.0, uplift: float = 5e-4,
                      k_incision: float = 4e-5, years: float = 3_000_000.0,
                      dt_years: float = 400.0) -> dict:
    """Uplift and stream power only: the steady state is known on the final network.

    At steady state every interior cell satisfies U = K A^m S^n with S the drop
    to its receiver over the receiver distance, so integrating
    z_i = z_r + d_ir (U / K)^(1/n) A_i^(-m/n) from the fixed edges along the
    teacher's own final receivers gives the exact discrete steady state for
    that network. The explicit update leaves such a surface unchanged for any
    dt, so a residual measures convergence and routing consistency, not the
    timestep.
    """
    parameters = landscape.Parameters(uplift_m_per_year=uplift, k_incision=k_incision,
                                      diffusivity_m2_per_year=0.0, spacing_m=spacing_m)
    start = initial_surface(side)
    final, record = _evolve(start, parameters, years, dt_years)
    routed = landscape.routing(final, spacing_m, ROUTER)
    receiver = routed["receiver"].reshape(-1)
    area = routed["area"].reshape(-1)
    m, n = parameters.area_exponent, parameters.slope_exponent
    cols = side
    exact = final.reshape(-1).copy()
    interior = np.zeros((side, side), dtype=bool)
    interior[1:-1, 1:-1] = True
    flat_interior = interior.reshape(-1)
    order = np.argsort(routed["filled"].reshape(-1), kind="stable")
    for index in order:
        target = receiver[index]
        if not flat_interior[index] or target == hydrology.NO_RECEIVER:
            continue
        distance = math.hypot(target // cols - index // cols, target % cols - index % cols) * spacing_m
        exact[index] = exact[target] + distance * (uplift / k_incision) ** (1.0 / n) * area[index] ** (-m / n)
    exact = exact.reshape(side, side)
    difference = np.abs(final - exact)[interior]
    relief = float(final.max() - final.min())
    return {"side": side, "spacingM": spacing_m, "years": years, "dtYears": dt_years,
            "parameters": parameters.as_dict(), "reliefM": relief,
            "maxAbsDifferenceM": float(difference.max()), "meanAbsDifferenceM": float(difference.mean()),
            "maxRelativeToRelief": float(difference.max()) / max(relief, 1e-12),
            "cellsIncisionLimitedTotal": record["cellsIncisionLimitedTotal"],
            "agrees": bool(float(difference.max()) <= 1e-3 * max(relief, 1.0)),
            "note": "Exact discrete steady state on the teacher's final D8 network, integrated "
                    "upstream from the fixed edges. Tolerance 0.1 % of relief."}


def manufactured_diffusion(sides=(17, 33, 65), length_m: float = 400.0, diffusivity: float = 1e-2,
                           years: float = 400_000.0, amplitude_m: float = 10.0) -> dict:
    """z = z_s + B exp(-D k_t^2 t) phi, with the source of z_s supplied as uplift.

    phi = sin(pi x / L) sin(pi y / L) decays freely; z_s = A sin(2 pi x / L)
    sin(pi y / L) is held steady by the source D k_s^2 z_s. Both vanish on the
    fixed edges. The time step is tied to dx^2, so the error falls at second
    order in dx if the operator, the edges and the source are all right.
    """
    rows = []
    for side in sides:
        spacing = length_m / (side - 1)
        axis = np.arange(side) * spacing
        y, x = np.meshgrid(axis, axis, indexing="ij")
        k_free2 = 2.0 * (math.pi / length_m) ** 2
        k_steady2 = (2.0 * math.pi / length_m) ** 2 + (math.pi / length_m) ** 2
        free = np.sin(math.pi * x / length_m) * np.sin(math.pi * y / length_m)
        steady = amplitude_m * np.sin(2.0 * math.pi * x / length_m) * np.sin(math.pi * y / length_m)
        source = diffusivity * k_steady2 * steady
        start = steady + amplitude_m * free
        parameters = landscape.Parameters(uplift_m_per_year=0.0, k_incision=0.0,
                                          diffusivity_m2_per_year=diffusivity, spacing_m=spacing)
        dt = landscape.stable_timestep(parameters)
        final, record = landscape.evolve(start, parameters, years, dt_years=dt, uplift=source,
                                         base_level="fixed-edges", router=ROUTER)
        exact = steady + amplitude_m * math.exp(-diffusivity * k_free2 * years) * free
        error = np.abs(final - exact)
        rows.append({"side": side, "spacingM": spacing, "dtYears": dt,
                     "realisedYears": record["realisedYears"],
                     "maxErrorM": float(error.max()), "meanErrorM": float(error.mean())})
    orders = [math.log(a["maxErrorM"] / b["maxErrorM"]) / math.log(a["spacingM"] / b["spacingM"])
              for a, b in zip(rows[:-1], rows[1:])]
    return {"lengthM": length_m, "diffusivity": diffusivity, "years": years, "rows": rows,
            "observedOrders": orders, "agrees": bool(orders and min(orders) >= 1.7),
            "note": "Manufactured solution with a steady source passed as the spatial uplift "
                    "field and fixed (Dirichlet) edges. dt is the teacher's own limit, so it "
                    "scales with dx^2 and the expected order is two."}


def rotation(parameters: landscape.Parameters, side: int = 64, years: float = 100_000.0,
             dt_years: float = 200.0, seed: int = 7) -> dict:
    """Does a quarter turn commute with the teacher, and how anisotropic is D8?

    A generic surface has no ties, so D8, the five-point Laplacian and the fixed
    edges are all equivariant under a 90 degree rotation and the two runs
    should agree to rounding, except where priority-flood tie-breaking on
    filled flats depends on scan order. A radially symmetric cone shows the
    other side: D8 routes along eight directions only, so a surface that should
    stay radially symmetric develops an eight-fold pattern. That is a property
    of the teacher, recorded rather than corrected.
    """
    rng = np.random.default_rng(seed)
    start = initial_surface(side) + rng.normal(0.0, 0.05, (side, side))
    one, _ = _evolve(start, parameters, years, dt_years)
    # The quarter turn uses fixed edges as everywhere else; the cone below
    # sits inside the domain so the square boundary does not shape it.
    two, _ = _evolve(np.rot90(start), parameters, years, dt_years)
    quarter = float(np.abs(np.rot90(one) - two).max())
    axis = (np.arange(side) - (side - 1) / 2.0) * parameters.spacing_m
    y, x = np.meshgrid(axis, axis, indexing="ij")
    radius = np.hypot(x, y)
    cone_radius = 0.45 * (side - 1) * parameters.spacing_m
    cone = 60.0 * np.clip(1.0 - radius / cone_radius, 0.0, None)
    incision_only = landscape.Parameters(uplift_m_per_year=0.0, k_incision=parameters.k_incision,
                                         diffusivity_m2_per_year=0.0, spacing_m=parameters.spacing_m)
    evolved, _ = _evolve(cone, incision_only, 20_000.0, dt_years)
    depth = (cone - evolved)
    ring = (radius > 0.3 * cone_radius) & (radius < 0.7 * cone_radius)
    angle = np.degrees(np.arctan2(y, x))[ring] % 45.0
    spoke = np.minimum(angle, 45.0 - angle) < 3.0
    values = depth[ring]
    return {"quarterTurnMaxDifferenceM": quarter,
            "quarterTurnCommutes": bool(quarter <= 1e-9 * max(float(np.ptp(one)), 1.0)),
            "coneIncisionMeanM": float(values.mean()),
            "coneIncisionAzimuthalCv": float(values.std() / max(values.mean(), 1e-12)),
            "coneIncisionOnSpokesOverOffSpokes": float(values[spoke].mean() / max(values[~spoke].mean(), 1e-12)),
            "agrees": bool(quarter <= 1e-9 * max(float(np.ptp(one)), 1.0)),
            "note": "Gate: the quarter turn. The cone numbers are a recorded property of D8: "
                    "incision only, 20 kyr, on a circular cone whose exact solution is radially "
                    "symmetric. On the ring between 30 and 70 % of the cone radius, the azimuthal "
                    "coefficient of variation of the incised depth and the ratio of incision within "
                    "3 degrees of the eight D8 directions to incision elsewhere measure how far the "
                    "routing is from rotational invariance."}


def _fastscape_run(start, uplift, k_incision, diffusivity, spacing_m, years, dt, m=0.5, n=1.0):
    """Uplift, routing, implicit stream power and ADI diffusion, in fastscapelib's usual order."""
    import fastscapelib as fs
    rows, cols = start.shape
    grid = fs.RasterGrid([rows, cols], [spacing_m, spacing_m], fs.NodeStatus.FIXED_VALUE)
    graph = fs.FlowGraph(grid, [fs.PFloodSinkResolver(), fs.SingleFlowRouter()])
    spl = fs.SPLEroder(graph, k_coef=k_incision, area_exp=m, slope_exp=n, tolerance=1e-8)
    diffusion = fs.DiffusionADIEroder(grid, diffusivity) if diffusivity > 0 else None
    core = np.zeros(start.shape, dtype=bool)
    core[1:-1, 1:-1] = True
    z = np.array(start, dtype=np.float64)
    for h in landscape.step_plan(years, dt):
        z = z + np.where(core, uplift * h, 0.0)
        graph.update_routes(z)
        area = graph.accumulate(1.0)
        z = z - spl.erode(z, area, h)
        if diffusion is not None:
            z = z - diffusion.erode(z, h)
    receivers = graph.impl().receivers[:, 0].reshape(start.shape)
    return z, receivers


def fastscape_crosscheck(side: int = 48, spacing_m: float = 100.0) -> dict:
    """Three stream-power cases against fastscapelib 0.3, conventions matched.

    Matched: D8 single-flow routing after priority-flood filling (fastscapelib
    adds its own epsilon), drainage area = contributing cells times dx^2
    including the cell itself (`accumulate(1.0)` on a raster with dx^2 node
    areas), slope = drop to the receiver over the receiver distance, all four
    edges fixed value, m = 1/2, n = 1, uplift on core nodes only. Not matched,
    and the reason the transient cases are refined in dt: the teacher is
    explicit and unsplit, fastscapelib is implicit (Braun and Willett 2013) and
    applies uplift, incision and ADI diffusion in sequence. Both are first
    order in time, so their difference must shrink as dt does.
    """
    try:
        import fastscapelib  # noqa: F401
        version = __import__("importlib.metadata").metadata.version("fastscapelib")
    except Exception as exc:  # pragma: no cover - recorded, not worked around
        return {"available": False, "reason": repr(exc), "agrees": False}
    rng = np.random.default_rng(11)
    axis = (np.arange(side) - (side - 1) / 2.0) * spacing_m
    y, x = np.meshgrid(axis, axis, indexing="ij")
    half = (side - 1) / 2.0 * spacing_m
    cone = 80.0 * (1.0 - np.maximum(np.abs(x), np.abs(y)) / half) + rng.normal(0.0, 0.05, (side, side))
    cone[0, :] = cone[-1, :] = cone[:, 0] = cone[:, -1] = 0.0
    cases = []
    # 1. Stream power alone, transient, refined in dt.
    for name, uplift, k, d, years in (("spl-transient", 0.0, 2e-5, 0.0, 50_000.0),
                                      ("spl-diffusion-uplift-transient", 2e-4, 2e-5, 1e-2, 50_000.0)):
        rows = []
        for dt in (400.0, 100.0, 25.0):
            parameters = landscape.Parameters(uplift_m_per_year=uplift, k_incision=k,
                                              diffusivity_m2_per_year=d, spacing_m=spacing_m)
            ours, _ = _evolve(cone, parameters, years, dt)
            theirs, receivers = _fastscape_run(cone, uplift, k, d, spacing_m, years, dt)
            mine = _receivers(ours, spacing_m).reshape(-1)
            same = np.where(mine == hydrology.NO_RECEIVER, np.arange(mine.size), mine)
            difference = np.abs(ours - theirs)[1:-1, 1:-1]
            rows.append({"dtYears": dt, "meanAbsDifferenceM": float(difference.mean()),
                         "maxAbsDifferenceM": float(difference.max()),
                         "meanAbsChangeM": float(np.abs(ours - cone)[1:-1, 1:-1].mean()),
                         "receiverAgreement": float(np.mean(same == receivers.reshape(-1)))})
        shrinking = all(a["meanAbsDifferenceM"] > b["meanAbsDifferenceM"] for a, b in zip(rows[:-1], rows[1:]))
        cases.append({"case": name, "uplift": uplift, "kIncision": k, "diffusivity": d,
                      "years": years, "rows": rows, "differenceShrinksWithDt": shrinking,
                      "agrees": bool(shrinking and rows[-1]["meanAbsDifferenceM"]
                                     <= 0.02 * max(rows[-1]["meanAbsChangeM"], 1e-9))})
    # 2. Uplift and stream power to steady state: both satisfy the same closed form.
    uplift, k = 5e-4, 4e-5
    parameters = landscape.Parameters(uplift_m_per_year=uplift, k_incision=k,
                                      diffusivity_m2_per_year=0.0, spacing_m=spacing_m)
    ours, _ = _evolve(cone, parameters, 2_000_000.0, 400.0)
    theirs, receivers = _fastscape_run(cone, uplift, k, 0.0, spacing_m, 2_000_000.0, 2000.0)
    mine = _receivers(ours, spacing_m).reshape(-1)
    same = np.where(mine == hydrology.NO_RECEIVER, np.arange(mine.size), mine)
    agreement = float(np.mean(same == receivers.reshape(-1)))
    difference = np.abs(ours - theirs)[1:-1, 1:-1]
    relief = float(np.ptp(ours))
    cases.append({"case": "spl-uplift-steady", "uplift": uplift, "kIncision": k, "diffusivity": 0.0,
                  "years": 2_000_000.0, "dtYears": {"teacher": 400.0, "fastscapelib": 2000.0},
                  "receiverAgreement": agreement, "meanAbsDifferenceM": float(difference.mean()),
                  "maxAbsDifferenceM": float(difference.max()), "reliefM": relief,
                  "meanElevationM": {"teacher": float(ours.mean()), "fastscapelib": float(theirs.mean())},
                  "agrees": bool(abs(float(ours.mean() - theirs.mean())) <= 0.02 * relief)})
    return {"available": True, "version": version, "side": side, "spacingM": spacing_m,
            "cases": cases, "agrees": all(c["agrees"] for c in cases),
            "conventions": {"routing": "D8 after priority-flood filling", "area": "cells times dx^2, self included",
                            "slope": "drop to receiver over receiver distance", "boundary": "all edges fixed value",
                            "m": 0.5, "n": 1.0, "uplift": "core nodes"},
            "note": "Steady states are compared by mean elevation because a steady state depends "
                    "on the network, and the two codes break D8 ties and fill flats differently; "
                    "receiverAgreement says how often they chose the same network."}


def budget(timestep: dict, production_dt: float = 400.0, noise_m: float = 0.5) -> dict:
    """Teacher discretisation error against the prediction error budget (proposed: below 10 %).

    Dynamics: a surrogate is judged on predicting the change over the interval,
    so the budget is declared as 10 % of the mean absolute change, and the
    teacher's own error at the production timestep must be below a tenth of
    that. Inverse: the likelihood's observation noise is the error scale the
    data can resolve, so the teacher error must be below a tenth of it.
    """
    row = next(r for r in timestep["rows"] if r["dtYears"] == production_dt)
    change = timestep["meanAbsChangeM"]
    dynamics_budget = 0.1 * change
    passing = {"dynamics": [r["dtYears"] for r in timestep["rows"] if r["l1M"] <= 0.1 * dynamics_budget],
               "inverse": [r["dtYears"] for r in timestep["rows"] if r["l1M"] <= 0.1 * noise_m]}
    return {"productionDtYears": production_dt, "teacherL1M": row["l1M"],
            "largestPassingDtYears": {k: (max(v) if v else None) for k, v in passing.items()},
            "receiverChangeFractionAtProductionDt": row["receiverChangeFraction"],
            "dynamics": {"meanAbsChangeM": change, "budgetM": dynamics_budget,
                         "ratio": row["l1M"] / max(dynamics_budget, 1e-30),
                         "passes": bool(row["l1M"] <= 0.1 * dynamics_budget)},
            "inverse": {"noiseM": noise_m, "ratio": row["l1M"] / noise_m,
                        "passes": bool(row["l1M"] <= 0.1 * noise_m)},
            "note": "Criterion proposed in the research plan (teacher error below 10 % of the "
                    "prediction error budget). The dynamics budget of 10 % of the change over "
                    "the interval and the inverse budget of the observation noise are declared "
                    "here, not derived."}


def run(parameters: landscape.Parameters | None = None, include_fastscape: bool = True) -> dict:
    """The whole audit. Every gate reported, failures included."""
    parameters = parameters or landscape.Parameters(
        uplift_m_per_year=5e-4, k_incision=4e-5,
        diffusivity_m2_per_year=5e-3, spacing_m=100.0)
    started = time.perf_counter()
    timestep = timestep_refinement(parameters)
    sections = {
        "timestepRefinement": timestep,
        "nonDivisible": non_divisible(parameters),
        "gridRefinement": grid_refinement(parameters),
        "conservation": conservation(parameters),
        "steadyState": steady_state_time(parameters),
        "analyticIncision": analytic_incision(),
        "manufacturedDiffusion": manufactured_diffusion(),
        "rotation": rotation(parameters),
        "fastscape": fastscape_crosscheck() if include_fastscape else {"available": False, "agrees": False},
        "budget": budget(timestep),
    }
    verdict = {
        "timestepConverges": timestep["agrees"],
        "nonDivisibleExact": sections["nonDivisible"]["agrees"],
        "exponentSurvivesTheMesh": sections["gridRefinement"]["agrees"],
        "coarsestAdequateSpacingM": sections["gridRefinement"]["adequateSpacingM"],
        "booksClose": sections["conservation"]["agrees"],
        "reachesPersistentSteadyState": sections["steadyState"]["settledAtYears"] is not None,
        "analyticIncision": sections["analyticIncision"]["agrees"],
        "manufacturedDiffusion": sections["manufacturedDiffusion"]["agrees"],
        "quarterTurnCommutes": sections["rotation"]["agrees"],
        "agreesWithFastscape": sections["fastscape"]["agrees"],
        "dynamicsBudget": sections["budget"]["dynamics"]["passes"],
        "inverseBudget": sections["budget"]["inverse"]["passes"],
    }
    verdict["fitToTeach"] = bool(verdict["timestepConverges"] and verdict["exponentSurvivesTheMesh"]
                                 and verdict["booksClose"] and verdict["analyticIncision"]
                                 and verdict["manufacturedDiffusion"])
    return {"schema": SCHEMA, "solver": landscape.SCHEMA, "parameters": parameters.as_dict(),
            **sections, "verdict": verdict, "seconds": time.perf_counter() - started,
            "qualification": "These gates establish that the solver integrates the equation it "
                             "claims to. They establish nothing about whether that equation "
                             "describes any real landscape: the omissions listed by `evolve` "
                             "(sediment transport, lithology, flexure, climate, sea level, human "
                             "alteration) are unaffected by any amount of numerical convergence."}
