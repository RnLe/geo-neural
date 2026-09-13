"""A landscape-evolution teacher, and the checks that validate it.

The model is a deliberately limited evolution equation:

    dh/dt = U - K * A^m * |grad h|^n + div(D grad h)

uplift, stream-power incision and hillslope diffusion. It implements exactly
that and nothing more, because a teacher is useful for what it is known to do,
not for how much it contains.

`A` is drainage area, not discharge. Substituting one for the other assumes a
uniform effective precipitation and a steady state that this model does not
provide, so the name stays literal. The area comes from the D8 accumulation in
`hydrology.py`, the same routine (with the same depression filling) that the
drainage-preservation metric uses to judge codecs, so a teacher and its
evaluation cannot drift apart.

The model omits sediment transport and deposition, lithologic variation,
flexure, tectonic rotation, climate, sea level and human alteration. Agreement
with the teacher alone is not sufficient validation, so the missing terms are
listed in every record this module writes.

Nothing here is fitted to NRW. It is a synthetic generator whose behaviour is
verified against cases with known answers, which is what gives a later
emulator's agreement with it any meaning.

Two properties of this discretisation are artefacts rather than physics, and are
documented because an emulator would learn them as though they were physics:

A tilted plane is not a steady state near the boundary. Edge-replicated padding
makes the edges zero-flux, so the ghost cell repeats the edge rather than
continuing the plane, and a plane therefore has non-zero curvature in the last
row and column. This is what zero-flux means, and it is the same property that
makes the diffusion term conserve total height exactly; one cannot be had
without the other.

A perfectly flat surface incises very slightly. `hydrology.fill_depressions`
raises each step across a filled flat by 1e-6 m so that flow stays defined, and
that artificial gradient drives an artificial incision of order
`K * sqrt(A) * (epsilon / spacing) * t`, about 3e-07 m over ten thousand years
at the parameters used here. It is negligible in magnitude but systematic in
direction, growing with drainage area, so it is documented rather than rounded
away.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, asdict

import numpy as np

from geoneural.metrics import hydrology

SCHEMA = "geoneural-landscape-teacher-v1"


@dataclass(frozen=True)
class Parameters:
    """Physical parameters, in metres, years and their combinations.

    `k_incision` carries units that depend on `area_exponent` and
    `slope_exponent`, which is a property of the stream-power form rather than an
    oversight; the exponents are therefore stored beside it in every record so a
    value is never reported without the units it belongs to.
    """
    uplift_m_per_year: float = 1e-4
    k_incision: float = 1e-5
    area_exponent: float = 0.5          # m
    slope_exponent: float = 1.0         # n
    diffusivity_m2_per_year: float = 1e-2
    spacing_m: float = 100.0

    def as_dict(self) -> dict:
        return asdict(self)


def laplacian(height: np.ndarray, spacing_m: float) -> np.ndarray:
    """Five-point Laplacian with zero-flux edges.

    Edge-replicated padding makes the boundary reflective, so the diffusion term
    moves no material across it. That is a modelling choice and the reason the
    conservation check below can be exact: with uplift and incision off, the
    total must not change at all.
    """
    padded = np.pad(height, 1, mode="edge")
    return (padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2]
            + padded[1:-1, 2:] - 4.0 * height) / (spacing_m ** 2)


def steepest_slope(height: np.ndarray, spacing_m: float) -> np.ndarray:
    """Downhill slope to the D8 receiver, in rise over run.

    Incision is driven by the gradient along the flow path, so this is the slope
    to the cell water actually leaves by, not a centred gradient magnitude,
    which would mix in directions no water takes.
    """
    filled = hydrology.fill_depressions(height)
    receiver = hydrology.d8_receivers(filled, spacing_m)
    flat = filled.reshape(-1)
    rows, cols = height.shape
    slope = np.zeros(height.size, dtype=np.float64)
    has_receiver = receiver.reshape(-1) != hydrology.NO_RECEIVER
    target = receiver.reshape(-1)[has_receiver]
    source = np.flatnonzero(has_receiver)
    dr = np.abs(target // cols - source // cols)
    dc = np.abs(target % cols - source % cols)
    distance = np.hypot(dr, dc) * spacing_m
    slope[source] = np.maximum(flat[source] - flat[target], 0.0) / np.maximum(distance, 1e-12)
    return slope.reshape(height.shape)


def drainage_area(height: np.ndarray, spacing_m: float, router=None) -> np.ndarray:
    """Contributing area in square metres, from the drainage-metric accumulation.

    `router` swaps in a compiled implementation for the two loops that dominate
    an ensemble. It defaults to `hydrology`, so a caller that does not ask gets
    the reference implementation exactly. The only admissible substitute is one
    that passes `hydrology_fast.bit_identical_to_reference`; agreement to a
    tolerance is not enough, because a router that differs anywhere silently
    splits every drainage number into two populations.
    """
    router = hydrology if router is None else router
    filled = router.fill_depressions(height)
    receiver = hydrology.d8_receivers(filled, spacing_m)
    cells = router.flow_accumulation(filled, receiver)
    return cells.astype(np.float64) * (spacing_m ** 2)


def stable_timestep(parameters: Parameters, safety: float = 0.2) -> float:
    """Largest explicit step the diffusion term tolerates, times a safety factor.

    Forward Euler on a five-point Laplacian is stable for dt <= dx^2 / (4 D) in
    two dimensions. The incision term has no comparable closed condition, so this
    bounds only what can be bounded, and `step` additionally limits incision so
    it cannot cut a cell below its receiver. That is how the incision term
    actually fails: by overshooting into a reversed gradient.
    """
    if parameters.diffusivity_m2_per_year <= 0.0:
        return float("inf")
    return safety * parameters.spacing_m ** 2 / (4.0 * parameters.diffusivity_m2_per_year)


def step(height: np.ndarray, parameters: Parameters, dt_years: float,
         uplift: np.ndarray | None = None, router=None) -> tuple[np.ndarray, dict]:
    """One explicit Euler step, returning the new surface and what each term did.

    The per-term contributions are returned rather than summed away because a
    teacher whose terms cannot be inspected separately cannot be debugged, and
    because validation needs source and sink accounts rather than a net change.
    """
    spacing = parameters.spacing_m
    rise = (parameters.uplift_m_per_year if uplift is None else uplift) * dt_years
    area = drainage_area(height, spacing, router)
    slope = steepest_slope(height, spacing)
    incision = (parameters.k_incision
                * area ** parameters.area_exponent
                * slope ** parameters.slope_exponent) * dt_years
    # Never incise a cell below the neighbour it drains to within one step: that
    # inverts the gradient that produced the incision and is the explicit
    # scheme's characteristic failure here, not a physical outcome.
    reach = slope * spacing * math.sqrt(2.0)
    limited = np.minimum(incision, np.maximum(reach, 0.0))
    clipped = int(np.count_nonzero(limited < incision))
    spread = parameters.diffusivity_m2_per_year * laplacian(height, spacing) * dt_years
    updated = height + rise - limited + spread
    return updated, {
        "dtYears": dt_years,
        "upliftMeanM": float(np.mean(rise)) if uplift is not None else float(rise),
        "incisionMeanM": float(incision.mean()), "incisionMaxM": float(incision.max()),
        # The ledger records what was applied, not what was requested: the
        # limiter is the difference between the equation and the scheme, and
        # charging the unlimited value would account for a step that never
        # happened.
        "incisionAppliedMeanM": float(limited.mean()),
        "diffusionMeanM": float(spread.mean()),
        "cellsIncisionLimited": clipped,
        "maxDrainageAreaM2": float(area.max()),
    }


def evolve(height: np.ndarray, parameters: Parameters, years: float,
           dt_years: float | None = None, uplift: np.ndarray | None = None,
           record_every: int = 0, base_level: str | None = None,
           router=None) -> tuple[np.ndarray, dict]:
    """Integrate forward, refusing a step the scheme cannot take.

    A run that silently exceeded its stability limit would produce a smooth,
    plausible, wrong field, so the limit is checked rather than assumed, and an
    unstable request raises instead of being quietly reduced.

    `base_level` decides whether the domain has anywhere to erode to, and it
    changes the answer completely rather than adjusting it:

    * `None`: the closed system. Uplift raises every cell including the edges
      the water leaves through, so nothing is ever lowered relative to anything
      else. On a 64-node grid, relief is 22.0 m at 200 kyr, 1.6 m at 400 kyr and
      0.0 m at 1.2 Myr, after which the surface is a flat plane translating
      upward at exactly `U`. Slope goes to zero, incision goes with it, and the
      concavity drifts to 1.0 against a predicted `m/n` of 0.5. This is the
      correct behaviour of the equation as posed, and it is useless as a
      teacher: an emulator would learn a rising plane.
    * `"fixed-edges"`: the boundary is held at its initial elevation. Uplift in
      the interior then creates relief against a base level, which is what makes
      a stream-power steady state exist at all. Total height is no longer
      conserved: mass leaves through the boundary, as it must in an open system.

    The default is `None` so that existing callers keep their behaviour. Nothing
    should be distilled from a `None` run.
    """
    limit = stable_timestep(parameters)
    dt = min(limit, years / 100.0) if dt_years is None else float(dt_years)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("timestep must be positive and finite")
    if dt > limit:
        raise ValueError(
            f"timestep {dt} yr exceeds the diffusion stability limit {limit} yr; "
            "reduce dt or the diffusivity rather than accepting a smooth wrong answer")
    if base_level not in (None, "fixed-edges"):
        raise ValueError(f"Unsupported base level: {base_level!r}")
    steps = max(int(round(years / dt)), 1)
    surface = np.array(height, dtype=np.float64, copy=True)
    initial = surface.copy()
    boundary = np.zeros(surface.shape, dtype=bool)
    boundary[0, :] = boundary[-1, :] = boundary[:, 0] = boundary[:, -1] = True
    pinned = surface[boundary].copy()
    # A volume ledger of sources and sinks, because a solver that silently loses
    # mass produces a plausible wrong landscape. Each term is integrated as a
    # volume over the domain, and the closure residual is what the surface
    # actually changed minus what the terms say it should have.
    # Diffusion contributes exactly zero with zero-flux edges, so a non-zero
    # diffusion volume is a bug in the Laplacian rather than physics.
    cell_area = parameters.spacing_m ** 2
    ledger = {"upliftVolumeM3": 0.0, "incisionVolumeM3": 0.0,
              "diffusionVolumeM3": 0.0, "boundaryOutflowVolumeM3": 0.0}
    history, clipped_total = [], 0
    for index in range(steps):
        before = surface
        surface, record = step(surface, parameters, dt, uplift, router)
        ledger["upliftVolumeM3"] += record["upliftMeanM"] * surface.size * cell_area
        ledger["incisionVolumeM3"] += record["incisionAppliedMeanM"] * surface.size * cell_area
        ledger["diffusionVolumeM3"] += record["diffusionMeanM"] * surface.size * cell_area
        if base_level == "fixed-edges":
            # Re-pinning removes (or adds) whatever the interior physics did to
            # the boundary. That is the open system's outflow and it is counted,
            # not absorbed.
            ledger["boundaryOutflowVolumeM3"] += float(
                (surface[boundary] - pinned).sum()) * cell_area
            surface[boundary] = pinned
        clipped_total += record["cellsIncisionLimited"]
        if record_every and (index % record_every == 0 or index == steps - 1):
            history.append({"step": index, "meanM": float(surface.mean()),
                            "maxM": float(surface.max()), **record})
        if not np.isfinite(surface).all():
            raise FloatingPointError(f"surface became non-finite at step {index}")
    return surface, {
        "schema": SCHEMA, "parameters": parameters.as_dict(),
        "years": years, "dtYears": dt, "steps": steps,
        "baseLevel": base_level or "none (closed system)",
        "reliefM": float(surface.max() - surface.min()),
        "baseLevelNote": (
            "With no base level the edges rise with the interior, relief decays to zero and the "
            "surface becomes a plane translating upward at U. Measured: 22.0 m of relief at "
            "200 kyr, 0.0 m at 1.2 Myr. A stream-power steady state requires fixed-edges."),
        "stabilityLimitYears": limit,
        "cellsIncisionLimitedTotal": clipped_total,
        "ledger": {**ledger, **_closure(surface, initial, ledger, cell_area)},
        "history": history,
        "discretisationArtefacts": [
            "zero-flux edges give a tilted plane non-zero curvature in the outermost row and column, "
            "so a plane is not a steady state there; this is the same choice that makes diffusion "
            "conserve total height exactly",
            "the 1e-6 m flat-resolution epsilon in depression filling drives an artificial incision "
            "of order K*sqrt(A)*(epsilon/spacing)*t on perfectly flat ground, growing with drainage "
            "area",
        ],
        "omits": ["sediment transport and deposition", "lithologic variation", "flexure",
                  "tectonic rotation", "climate and precipitation variation", "sea level",
                  "human alteration"],
        "qualification": "A synthetic generator, not a model of any real landscape. Drainage area is "
                         "area, not discharge. Explicit Euler on a five-point Laplacian with reflective "
                         "edges; the incision term is limited per step so it cannot invert the gradient "
                         "that drives it, and the number of cells that hit that limit is reported.",
    }


def nondimensional(parameters: Parameters, length_m: float, relief_m: float) -> dict:
    """The dimensionless groups that actually govern the equation.

    Non-dimensionalisation says which parameter combinations a landscape can
    identify, and therefore which inverse-history problems are well posed. With
    `h* = h/H`, `x* = x/L` and `t* = t U / H`, the stream-power equation

        dh/dt = U - K A^m |grad h|^n + D div(grad h)

    becomes

        dh*/dt* = 1 - Nf A*^m S*^n + Nh div*(grad* h*)

    with only two free numbers:

        Nf = K L^(2m - n) H^n / U      fluvial efficiency against uplift
        Nh = D H / (U L^2)             hillslope efficiency against uplift

    Their ratio is the landscape Peclet number: how far a signal travels by
    channel incision before hillslope diffusion erases it, and hence whether the
    terrain is ridge-and-valley or smooth.

    The consequence is a similarity law: two landscapes with equal (Nf, Nh, m, n)
    are the same landscape up to rescaling, whatever their dimensional U, K and
    D. So `U` and `K` are not separately identifiable from a single present-day
    surface; only `Nf` is.
    """
    if length_m <= 0.0 or relief_m <= 0.0:
        raise ValueError("length and relief scales must be positive")
    m, n = parameters.area_exponent, parameters.slope_exponent
    uplift = parameters.uplift_m_per_year
    if uplift <= 0.0:
        raise ValueError("non-dimensionalisation by uplift needs a positive uplift rate")
    fluvial = parameters.k_incision * length_m ** (2 * m - n) * relief_m ** n / uplift
    hillslope = parameters.diffusivity_m2_per_year * relief_m / (uplift * length_m ** 2)
    return {"lengthM": float(length_m), "reliefM": float(relief_m),
            "fluvialNumber": float(fluvial), "hillslopeNumber": float(hillslope),
            "pecletNumber": float(fluvial / hillslope) if hillslope > 0 else float("inf"),
            "areaExponent": float(m), "slopeExponent": float(n),
            "timescaleYears": float(relief_m / uplift),
            "similarity": "Two parameter sets with equal (fluvialNumber, hillslopeNumber, m, n) "
                          "produce the same landscape up to rescaling by L and H. U and K are not "
                          "separately identifiable from one present-day surface; only their "
                          "combination in fluvialNumber is.",
            "convention": "h* = h/H, x* = x/L, t* = t U / H."}


def slope_area(height: np.ndarray, spacing_m: float, min_area_m2: float | None = None,
               min_cells: int = 32, bins: int = 12, min_per_bin: int = 8) -> dict:
    """Fit `log S = log ks - theta * log A` over the channel network.

    This is the standard check on a stream-power landscape, and the one thing
    that distinguishes a solver that integrates the equation from one that merely
    produces plausible-looking terrain. At steady state with negligible
    diffusion, `U = K A^m S^n` gives

        S = (U / K)^(1/n) * A^(-m/n)

    so a log-log regression of slope on drainage area must return a straight line
    of slope `-m/n` (the concavity index) with intercept `(U/K)^(1/n)`. Both
    are predictions, not fits: the exponents come from the parameters and the
    regression is free to disagree.

    The fit is restricted to cells above `min_area_m2` because the law only holds
    where fluvial incision dominates; on hillslopes, diffusion sets the slope and
    the relationship bends over. Defaulting that threshold to a fixed number of
    cells rather than a fraction of the domain keeps the fitted window comparable
    between grid sizes.
    """
    height = np.asarray(height, dtype=np.float64)
    area = drainage_area(height, spacing_m)
    slope = steepest_slope(height, spacing_m)
    if min_area_m2 is None:
        min_area_m2 = 50.0 * spacing_m * spacing_m
    usable = (area >= min_area_m2) & (slope > 0.0) & np.isfinite(slope) & np.isfinite(area)
    count = int(usable.sum())
    if count < min_cells:
        return {"cells": count, "sufficient": False,
                "note": f"only {count} cells exceed the channel threshold; no fit attempted"}
    log_area = np.log10(area[usable])
    log_slope = np.log10(slope[usable])
    # Bin by log area and fit the bin medians (the standard construction). A
    # single cell's slope is a finite difference over one spacing, so the
    # per-cell scatter is dominated by that noise rather than by the law;
    # regressing raw cells returns a concavity near zero even where the binned
    # medians follow A^(-m/n) to within 3 per cent. Medians also keep the fit
    # from being dragged by the few cells nearest the outlet, where the fixed
    # boundary and not the stream-power balance sets the slope.
    edges = np.linspace(log_area.min(), log_area.max(), bins + 1)
    which = np.clip(np.digitize(log_area, edges[1:-1]), 0, bins - 1)
    centres, medians = [], []
    for index in range(bins):
        inside = which == index
        if int(inside.sum()) < min_per_bin:
            continue
        centres.append(float(np.median(log_area[inside])))
        medians.append(float(np.median(log_slope[inside])))
    if len(centres) < 3:
        return {"cells": count, "sufficient": False, "bins": len(centres),
                "note": "fewer than three populated area bins; no fit attempted"}
    centres_a = np.asarray(centres)
    medians_a = np.asarray(medians)
    gradient, intercept = np.polyfit(centres_a, medians_a, 1)
    predicted = gradient * centres_a + intercept
    residual = medians_a - predicted
    total = medians_a - medians_a.mean()
    r_squared = 1.0 - float(residual @ residual) / max(float(total @ total), 1e-30)
    return {"cells": count, "sufficient": True, "bins": len(centres),
            "binnedLogArea": centres, "binnedLogSlope": medians,
            "concavity": float(-gradient),
            "steepnessLog10": float(intercept),
            "steepness": float(10.0 ** intercept),
            "rSquared": r_squared,
            "minAreaM2": float(min_area_m2),
            "law": "S = ks * A^(-concavity); at steady state concavity = m/n and ks = (U/K)^(1/n)."}


def steady_state_report(height: np.ndarray, parameters: Parameters,
                        tolerance: float = 0.15,
                        outlet_area_fraction: float = 0.05) -> dict:
    """Test the surface against the closed-form steady state, bin by bin.

    The obvious test (regress concavity and compare it with `m/n`) is weaker,
    and on this solver misleading. It checks only the exponent, ignores the
    coefficient entirely, and is dragged by the few cells next to the pinned
    boundary, where the base level sets the slope rather than the stream-power
    balance. The regression can return 0.325 against a predicted 0.500 on a
    field whose binned slopes match the closed form to within 7 per cent
    everywhere the law applies.

    So the comparison is made where it is strongest. At steady state,

        S(A) = (U / K)^(1/n) * A^(-m/n)

    is a complete prediction with no free parameter, and each area bin is checked
    against it directly. Bins whose drainage area exceeds `outlet_area_fraction`
    of the domain are excluded, and the exclusion is reported: those cells drain
    into a boundary held at fixed elevation, which is a condition imposed on the
    run and not a balance the equation reached.
    """
    measured = slope_area(height, parameters.spacing_m)
    expected = parameters.area_exponent / parameters.slope_exponent
    if not measured.get("sufficient"):
        return {**measured, "expectedConcavity": float(expected), "agrees": False}
    m, n = parameters.area_exponent, parameters.slope_exponent
    coefficient = (parameters.uplift_m_per_year / parameters.k_incision) ** (1.0 / n)
    areas = np.power(10.0, np.asarray(measured["binnedLogArea"]))
    slopes = np.power(10.0, np.asarray(measured["binnedLogSlope"]))
    predicted = coefficient * areas ** (-m / n)
    domain_area = float(height.size) * parameters.spacing_m ** 2
    keep = areas <= outlet_area_fraction * domain_area
    ratios = slopes[keep] / predicted[keep]
    if ratios.size < 3:
        return {**measured, "expectedConcavity": float(expected), "agrees": False,
                "note": "too few bins remain after excluding the outlet bins"}
    worst = float(np.max(np.abs(np.log(ratios))))
    return {**measured, "expectedConcavity": float(expected),
            "predictedSteepness": float(coefficient),
            "binRatios": [float(v) for v in ratios],
            "medianRatio": float(np.median(ratios)),
            "worstLogRatio": worst,
            "outletAreaFraction": float(outlet_area_fraction),
            "binsExcludedAsOutlet": int(areas.size - int(keep.sum())),
            "binsCompared": int(ratios.size),
            "tolerance": float(tolerance),
            "agrees": bool(worst <= math.log(1.0 + tolerance)),
            "law": "S = (U/K)^(1/n) * A^(-m/n), a prediction with no free parameter.",
            "qualification": (
                "Every bin is compared against the closed-form steady state, so both the exponent "
                "and the coefficient are tested. The concavity from the regression is reported "
                "beside it but is not the criterion: it tests only the exponent and is biased by "
                "the outlet-adjacent bins, which are excluded here by a geometric criterion and "
                "counted in binsExcludedAsOutlet. Holds only at steady state and where fluvial "
                "incision dominates.")}


def _closure(surface: np.ndarray, initial: np.ndarray, ledger: dict,
             cell_area: float) -> dict:
    """Does the surface change equal what the terms claim they did?

    `observed` is the volume the field actually gained. `expected` is uplift
    minus incision plus diffusion minus whatever the base level took out through
    the boundary. If they disagree by more than rounding, a term is unaccounted
    for and the run should not be used for distillation.
    """
    observed = float((surface - initial).sum()) * cell_area
    expected = (ledger["upliftVolumeM3"] - ledger["incisionVolumeM3"]
                + ledger["diffusionVolumeM3"] - ledger["boundaryOutflowVolumeM3"])
    scale = max(abs(ledger["upliftVolumeM3"]), abs(observed), 1e-12)
    return {"observedVolumeChangeM3": observed,
            "expectedVolumeChangeM3": expected,
            "closureResidualM3": observed - expected,
            "closureResidualRelative": abs(observed - expected) / scale,
            "closureNote": "observed minus expected, relative to the uplift volume. Diffusion "
                           "must contribute exactly zero with zero-flux edges; a non-zero "
                           "diffusion volume is a Laplacian bug, not physics."}
