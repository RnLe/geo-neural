"""Drainage utility of a decoded or predicted surface, compared with the reference.

The same routing is applied to both surfaces: priority-flood depression filling, D8 steepest descent per metre of
ground, accumulation. Every domain edge is an outlet (`EDGE_POLICY`), which makes basin counts depend on the crop;
they are a defined comparison, not properties of the regional drainage system.

Streams are cells whose contributing area reaches a fixed physical area (0.05 km2 by default, 500 cells at 10 m), so
a change of resolution does not silently change the task. Exact-cell overlap is fragile: a centimetre of noise on
the reference alone moves channels across flat ground. `noise_floor` measures that, and `stream_metrics` also
reports a tolerant overlap in which a stream cell counts as found when the other surface has one within
`tolerance_cells`.

Two depression measures are kept apart: the change of the deepest fill (metres) and the change of the filled
volume (cubic metres). The v1 reports called the first one a volume.
"""
from __future__ import annotations

import numpy as np

from geoneural.metrics import hydrology, hydrology_fast

EDGE_POLICY = "every edge cell is an outlet"
STREAM_AREA_M2 = 50_000.0
SECONDARY_AREAS_M2 = (25_000.0, 100_000.0, 200_000.0)


def route(surface: np.ndarray, spacing_m: float) -> dict:
    """Filled surface, flat receiver index (NO_RECEIVER at outlets) and contributing cells."""
    z = np.asarray(surface, dtype=np.float64)
    filled = hydrology_fast.fill_depressions(z)
    receiver = hydrology.d8_receivers(filled, spacing_m)
    cells = hydrology_fast.flow_accumulation(filled, receiver)
    return {"surface": z, "filled": filled, "receiver": receiver.reshape(-1), "cells": np.asarray(cells).reshape(z.shape),
            "spacingM": float(spacing_m)}


def streams(routed: dict, area_m2: float = STREAM_AREA_M2) -> np.ndarray:
    return routed["cells"] >= area_m2 / routed["spacingM"] ** 2


def outlets(receiver: np.ndarray) -> np.ndarray:
    """Terminal cell of every cell's flow path, by pointer jumping (log2 of the longest path passes)."""
    n = receiver.size
    root = np.where(receiver == hydrology.NO_RECEIVER, np.arange(n), receiver).astype(np.int64)
    while True:
        nxt = root[root]
        if np.array_equal(nxt, root):
            return root
        root = nxt


def _dilate(mask: np.ndarray, cells: int) -> np.ndarray:
    if cells <= 0:
        return mask
    from scipy.ndimage import binary_dilation
    return binary_dilation(mask, structure=np.ones((3, 3), bool), iterations=cells)


def stream_metrics(reference: np.ndarray, estimate: np.ndarray, spacing_m: float, *,
                   area_m2: float = STREAM_AREA_M2, tolerance_cells: int = 1, secondary: bool = True,
                   reference_routed: dict | None = None) -> dict:
    """Drainage agreement of `estimate` with `reference`. Empty stream sets are reported, not scored as 0 or 1."""
    ref = reference_routed or route(reference, spacing_m)
    est = route(estimate, spacing_m)
    out = {"edgePolicy": EDGE_POLICY, "streamAreaM2": area_m2, "toleranceCells": tolerance_cells}
    out.update(_overlap(streams(ref, area_m2), streams(est, area_m2), tolerance_cells))
    if secondary:
        out["secondary"] = {str(int(a)): _overlap(streams(ref, a), streams(est, a), tolerance_cells)
                            for a in SECONDARY_AREAS_M2}
    interior = np.zeros(ref["surface"].shape, bool)
    interior[1:-1, 1:-1] = True
    same = ref["receiver"].reshape(interior.shape) == est["receiver"].reshape(interior.shape)
    out["receiverAgreement"] = float(same[interior].mean())
    out["outletAgreement"] = float((outlets(ref["receiver"]) == outlets(est["receiver"])).mean())
    depth_ref = ref["filled"] - ref["surface"]
    depth_est = est["filled"] - est["surface"]
    out["maxFillDepthDifferenceM"] = float(depth_est.max() - depth_ref.max())
    out["fillVolumeDifferenceM3"] = float((depth_est.sum() - depth_ref.sum()) * spacing_m ** 2)
    out["basinsReference"] = int((ref["receiver"] == hydrology.NO_RECEIVER).sum())
    out["basinsEstimate"] = int((est["receiver"] == hydrology.NO_RECEIVER).sum())
    return out


def _overlap(a: np.ndarray, b: np.ndarray, tol: int) -> dict:
    na, nb = int(a.sum()), int(b.sum())
    if na == 0 or nb == 0:
        return {"referenceCells": na, "estimateCells": nb, "jaccard": None, "recall": None, "precision": None,
                "tolerantF1": None, "empty": "reference" if na == 0 else "estimate"}
    inter = float((a & b).sum())
    union = float((a | b).sum())
    recall_t = float((a & _dilate(b, tol)).sum()) / na
    precision_t = float((b & _dilate(a, tol)).sum()) / nb
    f1 = 2 * recall_t * precision_t / (recall_t + precision_t) if recall_t + precision_t else 0.0
    return {"referenceCells": na, "estimateCells": nb, "jaccard": inter / union, "recall": inter / na,
            "precision": inter / nb, "tolerantRecall": recall_t, "tolerantPrecision": precision_t, "tolerantF1": f1}


def noise_floor(reference: np.ndarray, spacing_m: float, sigma_m: float = 0.01, seeds=(0, 1, 2),
                area_m2: float = STREAM_AREA_M2, tolerance_cells: int = 1) -> dict:
    """Agreement of the reference with itself plus white noise of `sigma_m`: the best any estimate can expect."""
    ref = route(reference, spacing_m)
    rows = []
    for seed in seeds:
        noisy = np.asarray(reference, np.float64) + np.random.default_rng(seed).normal(0.0, sigma_m, reference.shape)
        rows.append(stream_metrics(reference, noisy, spacing_m, area_m2=area_m2, tolerance_cells=tolerance_cells,
                                   secondary=False, reference_routed=ref))
    keys = ("jaccard", "tolerantF1", "receiverAgreement", "outletAgreement")
    return {"sigmaM": sigma_m, "seeds": list(seeds), **{k: float(np.mean([r[k] for r in rows])) for k in keys}}
