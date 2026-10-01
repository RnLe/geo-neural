"""Sampler diagnostics: rank-normalised split R-hat and bulk and tail effective sample size.

These follow Vehtari, Gelman, Simpson, Carpenter and Buerkner (2021), "Rank-
normalization, folding, and localization: an improved R-hat for assessing
convergence of MCMC", Bayesian Analysis 16(2). In plain terms:

* Every chain is split in half, so a chain that drifts (its first half
  differs from its second) is caught as two chains that disagree.
* Draws are replaced by their pooled ranks, mapped through the normal quantile
  function. That makes R-hat and ESS meaningful for heavy tails and for
  quantities with infinite variance, where the classical versions are not.
* R-hat is the larger of the value on the rank-normalised draws (location)
  and on the rank-normalised distances from the median (scale), so chains
  that agree in location but not in spread are also caught.
* ESS uses the multi-chain autocorrelation and Geyer's initial monotone
  sequence, never the autocorrelation of chains concatenated end to end
  (which treats the jump from one chain to the next as part of one series).
  Bulk ESS is computed on the rank-normalised draws; tail ESS is the smaller
  of the ESS of the indicators for the 5 % and 95 % quantiles.

Proposed gate (research plan): R-hat <= 1.01 and an adequate ESS for every
reported quantity. Passing it does not prove the physical model is right; it
says the chains agree with one another.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata


def _as_chains(draws) -> np.ndarray:
    values = np.asarray(draws, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 4:
        raise ValueError("draws must be (chains, iterations) with at least four iterations")
    return values


def split(draws) -> np.ndarray:
    """Halve every chain; an odd middle draw is dropped."""
    values = _as_chains(draws)
    half = values.shape[1] // 2
    return np.concatenate([values[:, :half], values[:, -half:]], axis=0)


def rank_normalise(draws) -> np.ndarray:
    """Pooled fractional ranks (ties averaged) through the normal quantile, (r - 3/8) / (S + 1/4)."""
    values = np.asarray(draws, dtype=np.float64)
    ranks = rankdata(values, method="average").reshape(values.shape)
    return ndtri((ranks - 0.375) / (values.size + 0.25))


def _rhat(chains: np.ndarray) -> float:
    m, n = chains.shape
    if m < 2:
        return float("nan")
    means = chains.mean(axis=1)
    within = chains.var(axis=1, ddof=1).mean()
    between = n * means.var(ddof=1)
    if within <= 0.0:
        return 1.0 if between <= 0.0 else float("inf")
    pooled = (n - 1) / n * within + between / n
    return float(np.sqrt(pooled / within))


def rhat(draws) -> float:
    """Rank-normalised split R-hat: the larger of the bulk and the folded value."""
    chains = split(draws)
    bulk = _rhat(rank_normalise(chains))
    folded = _rhat(rank_normalise(np.abs(chains - np.median(chains))))
    return float(max(bulk, folded))


def _autocovariance(x: np.ndarray) -> np.ndarray:
    n = x.size
    size = 1 << int(np.ceil(np.log2(2 * n)))
    centred = x - x.mean()
    spectrum = np.fft.rfft(centred, size)
    acov = np.fft.irfft(spectrum * np.conjugate(spectrum), size)[:n].real
    return acov / n


def ess(draws) -> float:
    """Multi-chain effective sample size with Geyer's initial monotone sequence (Stan's estimator)."""
    chains = _as_chains(draws)
    m, n = chains.shape
    acov = np.stack([_autocovariance(c) for c in chains])
    chain_var = acov[:, 0] * n / (n - 1.0)
    within = chain_var.mean()
    pooled = within * (n - 1.0) / n
    if m > 1:
        pooled += chains.mean(axis=1).var(ddof=1)
    if pooled <= 0.0 or not np.isfinite(pooled):
        return float("nan")
    rho = 1.0 - (within - acov.mean(axis=0)) / pooled
    rho[0] = 1.0
    # Geyer: sum adjacent pairs while positive, then force them non-increasing.
    pairs = []
    for t in range(0, n - 1, 2):
        value = rho[t] + rho[t + 1]
        if value <= 0.0:
            break
        pairs.append(value)
    pairs = np.minimum.accumulate(np.asarray(pairs)) if pairs else np.asarray([1.0])
    tau = -1.0 + 2.0 * float(pairs.sum())
    tau = max(tau, 1.0 / np.log10(m * n))
    return float(m * n / tau)


def bulk_ess(draws) -> float:
    return ess(rank_normalise(split(draws)))


def tail_ess(draws) -> float:
    """The smaller ESS of the indicators for the 5 % and 95 % quantiles."""
    chains = split(draws)
    out = []
    for q in (0.05, 0.95):
        cut = np.quantile(chains, q)
        out.append(ess((chains <= cut).astype(np.float64)))
    return float(min(out))


def summary(draws, names=None, threshold: float = 1.01, min_ess: float = 400.0) -> dict:
    """R-hat, bulk and tail ESS and central quantiles for each parameter.

    `draws` is (chains, iterations) for one parameter, or (chains, iterations,
    parameters). The gate is R-hat <= `threshold` and both ESS >= `min_ess`
    for every parameter.
    """
    values = np.asarray(draws, dtype=np.float64)
    if values.ndim == 2:
        values = values[:, :, None]
    names = list(names) if names is not None else [f"p{i}" for i in range(values.shape[2])]
    rows = {}
    for index, name in enumerate(names):
        x = values[:, :, index]
        q = np.quantile(x, (0.025, 0.05, 0.16, 0.5, 0.84, 0.95, 0.975))
        rows[name] = {"rhat": rhat(x), "bulkEss": bulk_ess(x), "tailEss": tail_ess(x),
                      "mean": float(x.mean()), "sd": float(x.std(ddof=1)),
                      "quantiles": dict(zip(("q025", "q05", "q16", "q50", "q84", "q95", "q975"),
                                            (float(v) for v in q)))}
    passes = all(r["rhat"] <= threshold and r["bulkEss"] >= min_ess and r["tailEss"] >= min_ess
                 for r in rows.values())
    return {"parameters": rows, "chains": int(values.shape[0]), "iterations": int(values.shape[1]),
            "gate": {"rhatMax": threshold, "essMin": min_ess, "passes": bool(passes)},
            "method": "rank-normalised split R-hat (max of bulk and folded), bulk and tail ESS "
                      "(Vehtari et al. 2021), Geyer initial monotone sequence"}
