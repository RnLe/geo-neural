"""Paired comparisons with regions as the statistical unit.

Pixels and patches are not independent, so uncertainty is taken over regions: a paired log-ratio per region, its
region-balanced mean, a percentile bootstrap over regions and the leave-one-region-out range. With six regions the
interval is coarse; it is reported as such, not as a precise confidence statement.
"""
from __future__ import annotations

import numpy as np


def paired_log_ratio(a: dict[str, float], b: dict[str, float], draws: int = 20000, seed: int = 0) -> dict:
    """Compare a against b on the regions both have. Ratios are a / b; below 1 means a is smaller."""
    regions = sorted(set(a) & set(b))
    if not regions:
        return {"regions": [], "n": 0}
    lr = np.array([np.log(a[r] / b[r]) for r in regions])
    rng = np.random.default_rng(seed)
    boot = lr[rng.integers(0, lr.size, (draws, lr.size))].mean(1)
    loo = [float(np.exp(np.delete(lr, i).mean())) for i in range(lr.size)] if lr.size > 1 else []
    return {"regions": regions, "n": int(lr.size), "perRegion": {r: float(np.exp(v)) for r, v in zip(regions, lr)},
            "geometricMeanRatio": float(np.exp(lr.mean())),
            "bootstrap95": [float(np.exp(np.quantile(boot, 0.025))), float(np.exp(np.quantile(boot, 0.975)))],
            "leaveOneRegionOut": [min(loo), max(loo)] if loo else None,
            "allRegionsSmaller": bool((lr < 0).all())}
