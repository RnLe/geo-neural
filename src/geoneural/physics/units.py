"""The one place the landscape equation is made dimensionless.

The teacher integrates

    dz/dt = U - K A^m S^n + div(D grad z)

with z in metres, horizontal distance in metres and time in years, so U is in
m/yr, D in m^2/yr and K in m^(1 - 2m)/yr. Choose a length scale L, a relief
scale H and a time scale T, write x = L x*, z = H z*, t = T t*, and the
equation becomes

    dz*/dt* = Pi_U - Pi_K A*^m S*^n + Pi_D div*(grad* z*)

with three coefficient groups

    Pi_U = U T / H
    Pi_K = K T L^(2m - n) H^(n - 1)
    Pi_D = D T / L^2

All three scales are explicit inputs. Earlier versions of the ensemble wrote a
fluvial number without the relief scale, which is not dimensionless unless
n = 1 and H happens to be one metre; every module now takes its groups from
here instead, and every group is returned as a base-10 logarithm, because
that is the coordinate the parameters are sampled, split, conditioned on and
inferred in.

With positive uplift the usual choice is T = H / U (`uplift_scales`). Then
Pi_U = 1 and the two remaining groups are the familiar

    Nf = K L^(2m - n) H^n / U       fluvial efficiency against uplift
    Nh = D H / (U L^2)              hillslope efficiency against uplift

and their ratio, the landscape Peclet number Nf / Nh = K L^(2m - n + 2)
H^(n - 1) / D, says whether channels or hillslope creep shape the terrain at
the scale L. Equal groups are not enough to make two landscapes the same: the
dimensionless initial surface, boundary geometry, forcing history and elapsed
time t* = t / T must match as well.

Two consequences used elsewhere. Scaling (U, K, D) by c and time by 1/c leaves
every group and t* unchanged when T is tied to the rates, which is the
similarity ridge of the inverse problem. And with n = 1 the equation is linear
in z for fixed routing, so multiplying the surface and U by the same factor is
again an exact solution; the synthetic terrain generator relies on that when it
rescales relief.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

GROUPS = ("logPiU", "logPiK", "logPiD")


@dataclass(frozen=True)
class Scales:
    """Length L (m), relief H (m) and time T (yr) that make the equation dimensionless."""
    length_m: float
    relief_m: float
    time_years: float

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be positive and finite, got {value}")

    def as_dict(self) -> dict:
        return {"lengthM": self.length_m, "reliefM": self.relief_m, "timeYears": self.time_years}


def uplift_scales(length_m: float, relief_m: float, uplift_m_per_year: float) -> Scales:
    """The uplift time scale T = H / U, for which Pi_U = 1."""
    if not (math.isfinite(uplift_m_per_year) and uplift_m_per_year > 0.0):
        raise ValueError("the uplift time scale needs a positive uplift rate; pass an explicit T otherwise")
    return Scales(length_m, relief_m, relief_m / uplift_m_per_year)


def groups(uplift: float, k_incision: float, diffusivity: float, scales: Scales,
           area_exponent: float = 0.5, slope_exponent: float = 1.0) -> dict:
    """log10 of (Pi_U, Pi_K, Pi_D) for the given rates and scales.

    A rate of exactly zero has no logarithm; it is returned as None rather than
    as minus infinity, so that a record can never carry a non-finite number.
    """
    m, n = float(area_exponent), float(slope_exponent)
    L, H, T = scales.length_m, scales.relief_m, scales.time_years

    def log(value, factor):
        if value < 0.0:
            raise ValueError("rates must be non-negative")
        return None if value == 0.0 else math.log10(value) + factor

    return {
        "logPiU": log(uplift, math.log10(T) - math.log10(H)),
        "logPiK": log(k_incision, math.log10(T) + (2 * m - n) * math.log10(L) + (n - 1) * math.log10(H)),
        "logPiD": log(diffusivity, math.log10(T) - 2 * math.log10(L)),
        "areaExponent": m, "slopeExponent": n, **scales.as_dict(),
    }


def uplift_groups(uplift: float, k_incision: float, diffusivity: float, length_m: float,
                  relief_m: float, area_exponent: float = 0.5, slope_exponent: float = 1.0) -> dict:
    """The fluvial and hillslope numbers under T = H / U, plus the Peclet number.

    `logFluvialNumber` is log10 Pi_K and `logHillslopeNumber` is log10 Pi_D at
    that time scale; `logPecletNumber` is their difference and does not depend
    on U at all.
    """
    scales = uplift_scales(length_m, relief_m, uplift)
    g = groups(uplift, k_incision, diffusivity, scales, area_exponent, slope_exponent)
    fluvial, hillslope = g["logPiK"], g["logPiD"]
    return {"logFluvialNumber": fluvial, "logHillslopeNumber": hillslope,
            "logPecletNumber": (None if fluvial is None or hillslope is None else fluvial - hillslope),
            "areaExponent": g["areaExponent"], "slopeExponent": g["slopeExponent"],
            **scales.as_dict(), "convention": "T = H / U, so log Pi_U = 0"}


def rates_from_uplift_groups(log_fluvial: float, log_hillslope: float, uplift: float,
                             length_m: float, relief_m: float, area_exponent: float = 0.5,
                             slope_exponent: float = 1.0) -> dict:
    """Invert `uplift_groups`: (log Nf, log Nh) and U back to (K, D)."""
    m, n = float(area_exponent), float(slope_exponent)
    L, H = float(length_m), float(relief_m)
    T = H / uplift
    k_incision = 10.0 ** (log_fluvial - math.log10(T) - (2 * m - n) * math.log10(L)
                          - (n - 1) * math.log10(H))
    diffusivity = 10.0 ** (log_hillslope - math.log10(T) + 2 * math.log10(L))
    return {"uplift": float(uplift), "kIncision": float(k_incision), "diffusivity": float(diffusivity)}


def rates_for_fixed_incision(log_fluvial: float, log_peclet: float, k_incision: float,
                             length_m: float, relief_m: float, area_exponent: float = 0.5,
                             slope_exponent: float = 1.0) -> dict:
    """(log Nf, log Pe) and a fixed K back to (U, K, D), under T = H / U.

    Nf = K L^(2m - n) H^n / U gives U; Pe = K L^(2m - n + 2) H^(n - 1) / D gives
    D. Neither inversion involves the other group, which is why an ensemble that
    holds K fixed samples (Nf, Pe) rather than (Nf, Nh).
    """
    m, n = float(area_exponent), float(slope_exponent)
    L, H = float(length_m), float(relief_m)
    uplift = k_incision * L ** (2 * m - n) * H ** n / 10.0 ** log_fluvial
    diffusivity = k_incision * L ** (2 * m - n + 2) * H ** (n - 1) / 10.0 ** log_peclet
    return {"uplift": float(uplift), "kIncision": float(k_incision), "diffusivity": float(diffusivity)}


def dimensionless_time(years: float, scales: Scales) -> float:
    """t* = t / T."""
    return float(years) / scales.time_years


def rescale(uplift: float, k_incision: float, diffusivity: float, length_factor: float,
            relief_factor: float, time_factor: float, area_exponent: float = 0.5,
            slope_exponent: float = 1.0) -> dict:
    """Rates that make (a L, b H, c T) describe the same dimensionless problem as (L, H, T).

    U' = U b / c, K' = K / (c a^(2m - n) b^(n - 1)), D' = D a^2 / c. A run with
    spacing a dx, heights b z, the primed rates and duration c t is the original
    run drawn at another scale, which is the strongest available check that the
    groups above are the ones the solver obeys.
    """
    a, b, c = float(length_factor), float(relief_factor), float(time_factor)
    m, n = float(area_exponent), float(slope_exponent)
    return {"uplift": uplift * b / c,
            "kIncision": k_incision / (c * a ** (2 * m - n) * b ** (n - 1)),
            "diffusivity": diffusivity * a * a / c}
