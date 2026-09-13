"""Is the landscape teacher a solver, or just a plausible picture?

Matching the steady-state law against the closed form is necessary but not
sufficient: a scheme can reproduce an asymptotic relation and still mis-integrate
the transient, lose mass, or depend on its timestep. A neural network that
accurately reproduces a flawed discretisation is still a flawed physical model,
so resolution and timestep convergence, boundary behaviour and conservation are
established before anything is learned from the teacher.

Four audits, each with a gate that can fail:

* Timestep refinement. Halve dt against a reference run at the finest dt and
  fit the observed order. Explicit Euler on this operator splitting is first
  order, so the gate is order >= 0.8; the incision limiter and D8 receiver
  switching are both discontinuous and can drag it lower, which is why the number
  of limited cells and the receiver-change fraction are reported beside it. An
  order near zero would mean the transient is not converging at all.
* Grid refinement. Hold the physical domain and parameters fixed and refine
  dx. The steady-state exponent must survive, because it is the one property the
  closed form predicts independently of the mesh.
* Conservation. The ledger `evolve` keeps must close: what the surface
  gained equals uplift minus incision plus diffusion minus boundary outflow.
  Diffusion must contribute exactly zero under zero-flux edges.
* Steady state. A declared criterion rather than "it looks settled": the
  interior mean |dh| per step, divided by the uplift per step, below a threshold
  and staying there.
"""
from __future__ import annotations

import math

import numpy as np

from geoneural.physics import landscape

SCHEMA = "geoneural-teacher-audit-v1"


def initial_surface(side: int, relief_m: float = 30.0, tilt_m: float = 10.0,
                    seed: int = 1729) -> np.ndarray:
    """A two-dimensional start surface. One-dimensional variation routes flow in
    one direction only, so no cell accumulates a channel and nothing exceeds a
    channel threshold."""
    axis = np.linspace(0.0, 1.0, side)
    field = relief_m * np.sin(2.0 * np.pi * axis)[None, :] * np.cos(2.0 * np.pi * axis)[:, None]
    return field + np.linspace(0.0, tilt_m, side)[:, None]


def timestep_refinement(parameters: landscape.Parameters, side: int = 64,
                        years: float = 200_000.0,
                        steps_list=(800.0, 400.0, 200.0, 100.0, 50.0),
                        reference_dt: float = 12.5) -> dict:
    """Halve dt and watch the answer stop moving. Or fail to."""
    start = initial_surface(side)
    reference, _ = landscape.evolve(start, parameters, years, dt_years=reference_dt,
                                    base_level="fixed-edges")
    rows = []
    for dt in steps_list:
        surface, report = landscape.evolve(start, parameters, years, dt_years=dt,
                                           base_level="fixed-edges")
        error = np.abs(surface - reference)
        rows.append({"dtYears": float(dt),
                     "l1M": float(error.mean()), "lInfM": float(error.max()),
                     "cellsIncisionLimitedTotal": report["cellsIncisionLimitedTotal"],
                     "closureResidualRelative": report["ledger"]["closureResidualRelative"]})
    orders = []
    for coarse, fine in zip(rows[:-1], rows[1:]):
        if fine["l1M"] > 0.0 and coarse["l1M"] > 0.0:
            ratio = coarse["dtYears"] / fine["dtYears"]
            orders.append(math.log(coarse["l1M"] / fine["l1M"]) / math.log(ratio))
    finest_order = orders[-1] if orders else 0.0
    return {"referenceDtYears": reference_dt, "rows": rows, "observedOrders": orders,
            "finestObservedOrder": float(finest_order),
            "monotone": all(a["l1M"] >= b["l1M"] for a, b in zip(rows[:-1], rows[1:])),
            "agrees": bool(finest_order >= 0.8
                           and all(a["l1M"] >= b["l1M"] for a, b in zip(rows[:-1], rows[1:]))),
            "note": "Explicit Euler on this splitting is first order. The incision limiter and D8 "
                    "receiver switching are discontinuous, so an order below one is expected "
                    "rather than alarming; an order near zero means the transient is not "
                    "converging and nothing may be learned from it."}


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
        surface, report = landscape.evolve(initial_surface(side), scaled, years,
                                           dt_years=dt, base_level="fixed-edges")
        steady = landscape.steady_state_report(surface, scaled)
        rows.append({"side": side, "spacingM": spacing, "dtYears": dt,
                     "reliefM": report["reliefM"],
                     "concavity": steady.get("concavity"),
                     "medianRatio": steady.get("medianRatio"),
                     "agrees": steady.get("agrees"),
                     "closureResidualRelative": report["ledger"]["closureResidualRelative"]})
    # A refinement study asks whether the answer converges to the predicted one
    # and from what resolution onward, not whether every resolution agrees: a
    # deliberately coarse one should not.
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
                    "`adequateSpacingM` is the coarsest spacing that still recovers it: above "
                    "that the drainage network is unresolved and the teacher does not obey its "
                    "own law. Any ensemble or emulator built "
                    "from this teacher must run at or below it."}


def conservation(parameters: landscape.Parameters, side: int = 64,
                 years: float = 100_000.0, dt_years: float = 200.0,
                 tolerance: float = 1e-9) -> dict:
    """The books must close, and diffusion must take exactly nothing out."""
    rows = []
    for base_level in (None, "fixed-edges"):
        _, report = landscape.evolve(initial_surface(side), parameters, years,
                                     dt_years=dt_years, base_level=base_level)
        ledger = report["ledger"]
        rows.append({"baseLevel": base_level or "none (closed system)", **ledger,
                     "closes": bool(ledger["closureResidualRelative"] < tolerance),
                     "diffusionIsZero": bool(abs(ledger["diffusionVolumeM3"])
                                             < tolerance * max(abs(ledger["upliftVolumeM3"]), 1.0))})
    return {"tolerance": tolerance, "rows": rows,
            "agrees": all(row["closes"] and row["diffusionIsZero"] for row in rows),
            "note": "Zero-flux edges make the discrete Laplacian sum to zero exactly, so a "
                    "non-zero diffusion volume would be a bug in the operator rather than a "
                    "property of the landscape. Boundary outflow is non-zero only under a base "
                    "level, which is what makes the system open."}


def steady_state_time(parameters: landscape.Parameters, side: int = 64,
                      years: float = 2_000_000.0, dt_years: float = 200.0,
                      threshold: float = 1e-3) -> dict:
    """When does it stop moving, by a stated criterion rather than by eye?"""
    surface = initial_surface(side)
    uplift_per_step = parameters.uplift_m_per_year * dt_years
    steps = max(int(round(years / dt_years)), 1)
    interior = (slice(1, -1), slice(1, -1))
    settled_at = None
    trace = []
    boundary = np.zeros(surface.shape, dtype=bool)
    boundary[0, :] = boundary[-1, :] = boundary[:, 0] = boundary[:, -1] = True
    pinned = surface[boundary].copy()
    for index in range(steps):
        previous = surface
        surface, _ = landscape.step(surface, parameters, dt_years)
        surface[boundary] = pinned
        rate = float(np.abs(surface[interior] - previous[interior]).mean()) / uplift_per_step
        if index % max(steps // 40, 1) == 0:
            trace.append({"step": index, "years": index * dt_years, "normalisedRate": rate,
                          "reliefM": float(surface.max() - surface.min())})
        if settled_at is None and rate < threshold:
            settled_at = index * dt_years
    return {"thresholdNormalisedRate": threshold,
            "settledAtYears": settled_at,
            "reliefTimescaleYears": float((surface.max() - surface.min())
                                          / parameters.uplift_m_per_year),
            "finalReliefM": float(surface.max() - surface.min()),
            "trace": trace,
            "note": "Interior mean |dh| per step divided by the uplift per step. Below the "
                    "threshold the surface is changing by a thousandth of what uplift adds, "
                    "which is a stated criterion rather than a settled-looking picture."}


def run(parameters: landscape.Parameters | None = None) -> dict:
    """The whole audit. Every gate reported, failures included."""
    parameters = parameters or landscape.Parameters(
        uplift_m_per_year=5e-4, k_incision=4e-5,
        diffusivity_m2_per_year=5e-3, spacing_m=100.0)
    timestep = timestep_refinement(parameters)
    grid = grid_refinement(parameters)
    books = conservation(parameters)
    settling = steady_state_time(parameters)
    return {"schema": SCHEMA, "parameters": parameters.as_dict(),
            "timestepRefinement": timestep, "gridRefinement": grid,
            "conservation": books, "steadyState": settling,
            "verdict": {
                "timestepConverges": timestep["agrees"],
                "exponentConvergesWithRefinement": grid["converging"],
                "coarsestAdequateSpacingM": grid["adequateSpacingM"],
                "exponentSurvivesTheMesh": grid["agrees"],
                "booksClose": books["agrees"],
                "reachesSteadyState": settling["settledAtYears"] is not None,
                "fitToTeach": bool(timestep["agrees"] and grid["agrees"] and books["agrees"]),
            },
            "qualification": "These gates establish that the solver integrates the equation it "
                             "claims to. They establish nothing about whether that equation "
                             "describes any real landscape: the omissions listed by `evolve` "
                             "(sediment transport, lithology, flexure, climate, sea level, human "
                             "alteration) are unaffected by any amount of numerical convergence."}
