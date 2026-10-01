"""Scores shared by every reconstruction task: heights, observation consistency, drainage, intervals, statistics.

Heights are in metres against the reference raster, which is itself a measurement and not error-free ground.
Drainage uses `geoneural.metrics.drainage` with a stream threshold of 0.05 km^2 at any resolution; every
drainage number is reported beside the noise floor of the reference against itself plus 1 cm of white noise,
because exact-cell overlap of a perfect method cannot exceed it.

Statistics: per-region values, a region-balanced mean (each region weighs the same however many cells or blocks
it holds), and paired differences with percentile bootstrap intervals. With six regions an interval over regions
is a description of spread, not a precise test; the unit resampled is always stated.
"""
from __future__ import annotations

import numpy as np

from geoneural.metrics import drainage

LEVELS = (0.5, 0.8, 0.9, 0.95)


def heights(estimate: np.ndarray, reference: np.ndarray, mask=None) -> dict:
    error = np.asarray(estimate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    if mask is not None:
        error = error[mask]
    a = np.abs(error)
    return {"maeM": float(a.mean()), "rmseM": float(np.sqrt((a ** 2).mean())), "biasM": float(error.mean()),
            "p95M": float(np.quantile(a, 0.95)), "p99M": float(np.quantile(a, 0.99)), "maxM": float(a.max())}


def observation(estimate: np.ndarray, coarse: np.ndarray, operator, margin: int = 2) -> dict:
    """||H z - y|| under `operator`, interior coarse nodes (edge nodes average a truncated window)."""
    d = operator.observe(estimate) - np.asarray(coarse, dtype=np.float64)
    d = d[margin:-margin, margin:-margin] if margin else d
    return {"obsRmsM": float(np.sqrt((d ** 2).mean())), "obsMaxM": float(np.abs(d).max())}


def tiles(estimate: np.ndarray, reference: np.ndarray, size: int = 256) -> list[float]:
    """MAE per non-overlapping tile, row-major: the within-region paired unit."""
    e = np.abs(np.asarray(estimate) - np.asarray(reference))
    n_r, n_c = e.shape[0] // size, e.shape[1] // size
    return [float(e[i * size:(i + 1) * size, j * size:(j + 1) * size].mean()) for i in range(n_r) for j in range(n_c)]


COMPACT = ("jaccard", "recall", "precision", "tolerantF1", "receiverAgreement", "outletAgreement",
           "maxFillDepthDifferenceM", "fillVolumeDifferenceM3", "referenceCells", "estimateCells")


def streams(reference_routed: dict, estimate: np.ndarray, spacing_m: float) -> dict:
    """Drainage of one surface against a routed reference, at 0.05 km^2 and the secondary areas."""
    out = drainage.stream_metrics(reference_routed["surface"], estimate, spacing_m,
                                  area_m2=drainage.STREAM_AREA_M2, tolerance_cells=1,
                                  reference_routed=reference_routed)
    compact = {k: out.get(k) for k in COMPACT}
    compact["secondary"] = {a: {k: v.get(k) for k in ("jaccard", "tolerantF1")} for a, v in out["secondary"].items()}
    return compact


def floor(reference: np.ndarray, spacing_m: float) -> dict:
    """Noise floor of exact and tolerant drainage agreement: the reference against itself plus 1 cm and 10 cm."""
    return {f"{sigma}m": drainage.noise_floor(reference, spacing_m, sigma_m=sigma) for sigma in (0.01, 0.1)}


def masked_streams(ref_routed: dict, est_routed: dict, mask: np.ndarray, tolerance: int = 1) -> dict:
    """Drainage agreement restricted to `mask` (both surfaces routed over the whole field first).

    For holes: everything outside the holes is the reference itself, so a whole-field score is mostly
    agreement of the reference with itself; this reads only where the methods differ.
    """
    from scipy.ndimage import binary_dilation
    a, b = drainage.streams(ref_routed), drainage.streams(est_routed)
    structure = np.ones((3, 3), bool)
    da, db = binary_dilation(a, structure, tolerance), binary_dilation(b, structure, tolerance)
    am, bm = a & mask, b & mask
    na, nb = int(am.sum()), int(bm.sum())
    out = {"referenceCells": na, "estimateCells": nb}
    if na and nb:
        recall_t, precision_t = float((am & db).sum()) / na, float((bm & da).sum()) / nb
        out.update({"jaccard": float((am & bm).sum()) / float((am | bm).sum()),
                    "tolerantF1": 2 * recall_t * precision_t / (recall_t + precision_t)
                    if recall_t + precision_t else 0.0})
    else:
        out.update({"jaccard": None, "tolerantF1": None, "empty": "reference" if na == 0 else "estimate"})
    shape = mask.shape
    same = ref_routed["receiver"].reshape(shape) == est_routed["receiver"].reshape(shape)
    out["receiverAgreement"] = float(same[mask].mean())
    out["outletAgreement"] = float((drainage.outlets(ref_routed["receiver"]).reshape(shape) ==
                                    drainage.outlets(est_routed["receiver"]).reshape(shape))[mask].mean())
    depth_ref = (ref_routed["filled"] - ref_routed["surface"])[mask]
    depth_est = (est_routed["filled"] - est_routed["surface"])[mask]
    out["maxFillDepthDifferenceM"] = float(depth_est.max() - depth_ref.max())
    out["fillVolumeDifferenceM3"] = float((depth_est.sum() - depth_ref.sum()) * ref_routed["spacingM"] ** 2)
    return out


def masked_floor(reference: np.ndarray, spacing_m: float, mask: np.ndarray, sigma_m: float = 0.01,
                 seeds=(0, 1, 2)) -> dict:
    ref = drainage.route(reference, spacing_m)
    rows = [masked_streams(ref, drainage.route(reference + np.random.default_rng(s).normal(0.0, sigma_m,
                                                                                          reference.shape),
                                               spacing_m), mask) for s in seeds]
    keys = ("jaccard", "tolerantF1", "receiverAgreement", "outletAgreement")
    return {"sigmaM": sigma_m, **{k: _mean([r[k] for r in rows]) for k in keys}}


def _mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


# --- intervals -----------------------------------------------------------------------------------------------

def intervals(truth: np.ndarray, quantiles: dict, classes: np.ndarray | None = None, n_classes: int = 4) -> dict:
    """Coverage and width of central intervals, overall and per terrain class.

    `quantiles` maps a probability to a field (metres), and must hold (1-p)/2 and (1+p)/2 for every level p in
    LEVELS. Coverage below nominal means the intervals claim more certainty than the errors allow.
    """
    out = {"overall": _coverage(truth, quantiles, None)}
    if classes is not None:
        out["bySlopeClass"] = {str(k): _coverage(truth, quantiles, classes == k) for k in range(n_classes)
                               if (classes == k).any()}
    return out


def _coverage(truth, quantiles, mask):
    rows = {}
    for p in LEVELS:
        lo, hi = quantiles[round((1 - p) / 2, 4)], quantiles[round((1 + p) / 2, 4)]
        inside = (truth >= lo) & (truth <= hi)
        width = hi - lo
        if mask is not None:
            inside, width = inside[mask], width[mask]
        rows[str(p)] = {"coverage": float(inside.mean()), "meanWidthM": float(width.mean())}
    rows["cells"] = int(truth.size if mask is None else mask.sum())
    return rows


def quantile_levels() -> list[float]:
    return sorted({round((1 - p) / 2, 4) for p in LEVELS} | {round((1 + p) / 2, 4) for p in LEVELS})


def pit_histogram(pit: np.ndarray, bins: int = 10) -> list[float]:
    """Share of cells per decile of the predictive CDF at the truth; flat at 0.1 when calibrated."""
    h, _ = np.histogram(np.clip(pit, 0, 1), bins=bins, range=(0, 1))
    return (h / max(h.sum(), 1)).tolist()


# --- statistics ----------------------------------------------------------------------------------------------

def bootstrap(values, n: int = 10_000, seed: int = 0, level: float = 0.95) -> dict:
    """Mean and percentile bootstrap interval over the given units."""
    v = np.asarray([x for x in values if x is not None], dtype=np.float64)
    if v.size == 0:
        return {"mean": None, "low": None, "high": None, "units": 0}
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, (n, v.size))].mean(1)
    a = (1 - level) / 2
    return {"mean": float(v.mean()), "low": float(np.quantile(means, a)), "high": float(np.quantile(means, 1 - a)),
            "units": int(v.size), "positive": int((v > 0).sum()), "negative": int((v < 0).sum())}


def region_balanced(per_region: dict) -> float | None:
    values = [v for v in per_region.values() if v is not None]
    return float(np.mean(values)) if values else None


# --- summaries -----------------------------------------------------------------------------------------------

SUMMARY_KEYS = ("maeM", "rmseM", "p99M", "maxM", "biasM", "obsRmsM", "obsMaxM")
DRAINAGE_KEYS = ("jaccard", "tolerantF1", "receiverAgreement", "outletAgreement", "maxFillDepthDifferenceM",
                 "fillVolumeDifferenceM3")


def merge_seeds(methods: dict) -> dict:
    """Collapse `<label>-seed<k>[+bp]` rows into `<label>[+bp]` (mean over seeds, with the seed spread)."""
    import re
    out = {k: v for k, v in methods.items() if "-seed" not in k}
    groups: dict = {}
    for key, row in methods.items():
        m = re.match(r"(.+)-seed\d+(\+bp)?$", key)
        if m:
            groups.setdefault(m.group(1) + (m.group(2) or ""), []).append(row)
    for key, rows in groups.items():
        merged = {k: float(np.mean([r[k] for r in rows])) for k in SUMMARY_KEYS if k in rows[0]}
        merged["seedSdMaeM"] = float(np.std([r["maeM"] for r in rows]))
        merged["seeds"] = len(rows)
        for unit in ("tileMaeM", "blockMaeM"):
            if unit in rows[0]:
                merged[unit] = np.mean([r[unit] for r in rows], axis=0).tolist()
        if "drainage" in rows[0]:
            merged["drainage"] = {k: _mean([r["drainage"].get(k) for r in rows]) for k in DRAINAGE_KEYS}
        if "sampleMisfitRmsM" in rows[0]:
            merged["sampleMisfitRmsM"] = float(np.mean([r["sampleMisfitRmsM"] for r in rows]))
        if "backProjection" in rows[0]:
            merged["backProjection"] = {
                "toleranceM": rows[0]["backProjection"].get("toleranceM"),
                "achievedMaxM": max(r["backProjection"]["achievedMaxM"] for r in rows),
                "iterations": [r["backProjection"]["iterations"] for r in rows],
                "converged": all(r["backProjection"]["converged"] for r in rows)}
        out[key] = merged
    return out


def compact(row: dict) -> dict:
    out = {k: row[k] for k in SUMMARY_KEYS + ("seedSdMaeM", "seeds") if k in row}
    if "drainage" in row:
        out["drainage"] = {k: row["drainage"].get(k) for k in DRAINAGE_KEYS}
    if "backProjection" in row:
        bp = row["backProjection"]
        out["backProjection"] = {k: bp.get(k) for k in ("toleranceM", "achievedMaxM", "iterations", "converged")}
    if "solver" in row:
        out["solver"] = row["solver"]
    return out


def paired(per_region: dict, a: str, b: str, key: str = "maeM", unit: str = "tileMaeM") -> dict:
    """a minus b per region, bootstrap over regions, and per region a bootstrap over its tiles or blocks.

    Negative favours a. The within-region intervals treat tiles (or blocks) as independent, which spatial
    correlation makes optimistic; the interval over regions is the one that speaks to new geography.
    """
    diffs = {r: m[a][key] - m[b][key] for r, m in per_region.items() if a in m and b in m}
    out = {"comparison": f"{a} minus {b}", "metric": key, "byRegion": diffs,
           "relativeByRegion": {r: diffs[r] / m[b][key] for r, m in per_region.items() if r in diffs and m[b][key]},
           "overRegions": bootstrap(list(diffs.values())), "unit": "region"}
    out["withinRegion"] = {r: bootstrap(np.subtract(m[a][unit], m[b][unit]).tolist())
                           for r, m in per_region.items() if a in m and b in m and unit in m[a] and unit in m[b]}
    out["withinRegionUnit"] = unit
    return out


def merge_seed_metrics(rows: dict) -> dict:
    """Collapse `<label>-seed<k>[+bp]` entries of flat metric dicts into their mean over seeds."""
    import re
    out = {k: v for k, v in rows.items() if "-seed" not in k}
    groups: dict = {}
    for key, row in rows.items():
        m = re.match(r"(.+)-seed\d+(\+bp)?$", key)
        if m:
            groups.setdefault(m.group(1) + (m.group(2) or ""), []).append(row)
    for key, group in groups.items():
        out[key] = {k: _mean([g.get(k) for g in group]) for k in group[0]}
    return out
