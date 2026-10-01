"""Two different fine terrains with exactly the same coarse observation.

If z2 = z1 + n with H n = 0, every method that sees only y = H z1 gives the same answer for both, so whatever it
predicts is wrong by at least |n| / 2 for one of them. A model that claims certainty here is claiming more than
the observation contains. Two constructions, at 40 m -> 10 m under the declared trapezoid:

* detail swap: n is the null-space projection of (fine detail of another region) minus (fine detail of z1),
  where detail is the residual over the B-spline of the region's own coarse grid; z2 has z1's coarse signal
  and someone else's 10 m texture, both plausible terrain.
* channel shift: a planar slope with a 2 m deep, 20 m wide channel, moved two cells sideways inside the same
  coarse cells and projected so the coarse grids agree.

Scored: the gap |z1 - z2|, the error of the shared prediction against each, and whether the 90 % intervals of
the single model, the seed ensemble and regression-kriging cover z1, z2 and both, in the cells where the two
differ by more than 0.5 m.
"""
from __future__ import annotations

import numpy as np
import torch

from geoneural.recon import baselines, coarse, fields, models, operators, train

CROP = 513
GAP_M = 0.5


def null_projection(v: np.ndarray, operator) -> np.ndarray:
    """v minus its least-norm component that H can see: H applied to the result is zero."""
    wr, wc = (m.toarray() for m in operator.matrices(v.shape))
    gr, gc = np.linalg.inv(wr @ wr.T), np.linalg.inv(wc @ wc.T)
    return v - wr.T @ (gr @ (wr @ v @ wc.T) @ gc) @ wc


def _detail(z, operator):
    return z - operators.upsample(operator.observe(z), coarse.FACTOR, coarse.BASE)


def cases(region: str, other: str, origin=(256, 256)):
    trap = operators.make("trapezoid", coarse.FACTOR)
    sl = (slice(origin[0], origin[0] + CROP), slice(origin[1], origin[1] + CROP))
    z1 = fields.reference(region)[sl]
    q = fields.reference(other)[sl]
    n = null_projection(_detail(q, trap) - _detail(z1, trap), trap)
    yy, xx = np.mgrid[0:CROP, 0:CROP].astype(np.float64)
    ramp = 100.0 + 0.03 * 10.0 * yy + 0.01 * 10.0 * xx

    def channel(x0):
        return -2.0 * np.clip(1.0 - np.abs(xx - x0) / 1.0, 0.0, 1.0)

    c1 = ramp + channel(CROP // 2)
    m = null_projection(channel(CROP // 2 + 2) - channel(CROP // 2), trap)
    return {"detailSwap": (z1, z1 + n), "channelShift": (c1, c1 + m)}, trap


def score(members, z1, z2, operator, fitted_rk, device) -> dict:
    y = operator.observe(z1)
    check = float(np.abs(operator.observe(z2) - y).max())
    base = operators.upsample(y, coarse.FACTOR, coarse.BASE)
    locs, spreads = [], []
    for model in members:
        loc, sp = train.predict(model, coarse.SPEC, base, None, device=device)
        locs.append(base + loc)
        spreads.append(sp)
    gap = np.abs(z1 - z2)
    differ = gap > GAP_M
    inner = np.zeros_like(differ)
    inner[8:-8, 8:-8] = True
    differ &= inner
    out = {"coarseDisagreementM": check, "gapMeanM": float(gap[inner].mean()), "gapMaxM": float(gap.max()),
           "cellsDiffering": int(differ.sum())}
    mu, sp = torch.tensor(np.stack(locs), device=device), torch.tensor(np.stack(spreads), device=device)
    for name, sel in (("single", slice(0, 1)), ("ensemble", slice(None))):
        q = models.mixture_quantiles(mu[sel], sp[sel], (0.05, 0.95))
        lo, hi = q[0].cpu().numpy(), q[1].cpu().numpy()
        mean = mu[sel].mean(0).cpu().numpy()
        out[name] = _cover(mean, lo, hi, z1, z2, differ)
    from scipy.stats import norm
    rk = coarse.rk_predict(fitted_rk, y, base, None, operator)
    sd = np.sqrt(np.tile(fitted_rk["variance"], (CROP // coarse.FACTOR + 1,) * 2)[:CROP, :CROP])
    out["rkKriging"] = _cover(rk, rk + norm.ppf(0.05) * sd, rk + norm.ppf(0.95) * sd, z1, z2, differ)
    return out


def _cover(mean, lo, hi, z1, z2, mask):
    if not mask.any():
        return {"cells": 0}
    in1 = (z1 >= lo) & (z1 <= hi)
    in2 = (z2 >= lo) & (z2 <= hi)
    return {"cells": int(mask.sum()), "maeToZ1M": float(np.abs(mean - z1)[mask].mean()),
            "maeToZ2M": float(np.abs(mean - z2)[mask].mean()),
            "coverZ1": float(in1[mask].mean()), "coverZ2": float(in2[mask].mean()),
            "coverBoth": float((in1 & in2)[mask].mean()),
            "meanHalfWidthM": float(((hi - lo) / 2)[mask].mean()),
            "meanHalfGapM": float((np.abs(z1 - z2) / 2)[mask].mean()),
            "nominal": 0.9}


def run(test_regions, device, models_dir, parts_dir) -> dict:
    """Uses the coarse-task models of the fold that held each region out, and that fold's RK fit."""
    from geoneural.common import read_json
    out = {}
    for region in test_regions:
        part = read_json(parts_dir / f"{region}.json", max_bytes=256 * 1024 * 1024)
        fold = part["fold"]
        other = fold["validation"][0]
        members = []
        for k in range(len(part["seeds"])):
            model = models.UNet(coarse.SPEC.base_channels(), 32).to(device)
            model.load_state_dict(torch.load(models_dir / region / f"neural-seed{k}.pt", map_location=device))
            members.append(model.eval())
        rk = part["baselines"]["rkNoGeology"]
        trap = operators.make("trapezoid", coarse.FACTOR)
        weights = trap._weights(4 * coarse.FACTOR + 1, 0)[2]
        support = np.flatnonzero(weights)
        kernels, variance = baselines.atpk_kernels(rk["covariance"], weights[support], weights[support], coarse.FACTOR)
        fitted = {"beta": np.asarray(rk["beta"]), "kernels": kernels, "variance": variance, "geology": False}
        built, op = cases(region, other)
        out[region] = {"alternativeDetailFrom": other,
                       **{name: score(members, z1, z2, op, fitted, device) for name, (z1, z2) in built.items()}}
    return out
