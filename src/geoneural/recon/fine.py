"""Coarse-to-fine reconstruction, 10 m -> 1 m, on the three development regions with 1 m data.

Same contract as the 40 m task at factor 10: y = H z from the 1 m reference, every method receives y and the
operator name. Leave one region out: two regions train, one is scored. With two training regions there is no
spare region to validate on, so the southern band of each training region (fine rows from 8192, 1.9 km) is held
out of training windows with a 256 m buffer and validates the step count and learning rate by the same declared
rule as the 40 m task.

The real operator-mismatch test exists here: the provider's own 10 m product of the same ground. It is fed in
place of y; back-projection then enforces the declared trapezoid, which is the wrong operator for it, and the raw
output is reported beside it. Streams start at 0.05 km^2, which is 50,000 cells at 1 m, read on sixteen
2041-node windows per region with their noise floor. 1 m detail predicted here is a statistical estimate, not
new measured information.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from geoneural.common import HOME
from geoneural.recon import baselines, evaluate, fields, models, operators, train

FACTOR = 10
SPEC = models.Features(scale_sigma=10.0, highpass_sigma=5.0, unit=10.0, floor_m=0.02, factor=FACTOR, margin=40)
CORE, BORDER, BATCH = 160, 24, 16
TILE, HALO = 320, 64
VALIDATION_ROW, BUFFER = 8192, 256
VALIDATION_WINDOWS = ((8448, 400), (8448, 3000), (8448, 5600), (8448, 8200))
VALIDATION_SIZE = 1280
LR_GRID = (5e-4, 1e-3)
CHECKPOINTS = (500, 1000, 1500, 2000, 3000)
MAX_STEPS = 3000
DRAINAGE_SIZE = 2041
DRAINAGE_ORIGINS = tuple((400 + 2400 * i, 400 + 2400 * j) for i in range(4) for j in range(4))
FIT_WINDOWS = 2
#: Windows (2048 nodes) of each training region on which the regression-kriging trend and covariance are fitted.
RK_WINDOWS = ((1024, 1024), (1024, 5120), (5120, 1024), (5120, 5120))
ALPHA = 0.03
BASE = "bspline"
INNER = 50


def provider_grid(region: str) -> np.ndarray:
    return np.load(HOME / "atlases" / region / "reference.npy").astype(np.float64)


class SplineWindows:
    """B-spline base of aligned windows from whole-field coefficients, on the device, equal to a crop."""

    def __init__(self, size: int, device):
        index = np.arange(size)
        j, u = index // FACTOR, (index % FACTOR) / FACTOR
        self.blocks = (size - 1) // FACTOR + 4
        m = np.zeros((size, self.blocks))
        taps = operators._bspline(u)
        for t in range(4):
            m[index, j + t] += taps[:, t]
        self.matrix = torch.tensor(m, dtype=torch.float32, device=device)
        self.device = device

    def __call__(self, padded, coarse_origins):
        """padded: (C,H+4,W+4) coefficients mirrored by two nodes; origins (B,3) = (field, row, col) coarse."""
        k = self.blocks
        rows = torch.as_tensor(coarse_origins[:, 1, None] + 1 + np.arange(k)[None], device=self.device)
        cols = torch.as_tensor(coarse_origins[:, 2, None] + 1 + np.arange(k)[None], device=self.device)
        f = torch.as_tensor(coarse_origins[:, 0], device=self.device)
        block = padded[f[:, None, None], rows[:, :, None], cols[:, None, :]]
        return self.matrix @ block @ self.matrix.T


class Task:
    def __init__(self, regions, mix, device):
        self.device, self.mix = device, mix
        self.weights = np.array([w for _, w in mix]) / sum(w for _, w in mix)
        self.fine = [np.asarray(fields.fine(r), dtype=np.float32) for r in regions]
        coeffs, self.index = [], {}
        for ri, region in enumerate(regions):
            for oi, (op, _) in enumerate(mix):
                y = op.observe(self.fine[ri])
                self.index[(ri, oi)] = len(coeffs)
                coeffs.append(np.pad(operators.coefficients(y, BASE), 2, mode="reflect"))
        self.coeffs = torch.tensor(np.stack(coeffs), dtype=torch.float32, device=device)
        self.size = CORE + 2 * SPEC.margin
        self.windows = SplineWindows(self.size + 1, device)
        self.sampler = models.WindowSampler(self.size + 1, self.size, device)
        self.phases = models.phase_maps(0, 0, self.size, self.size, FACTOR, device)[None].expand(BATCH, -1, -1, -1)
        self.weight = torch.zeros(BATCH, 1, CORE, CORE, device=device)
        self.weight[..., BORDER:-BORDER, BORDER:-BORDER] = 1.0
        self.channels = SPEC.base_channels()
        trap = operators.make("trapezoid", FACTOR)
        self.validation = []
        for ri in range(len(regions)):
            coeff = operators.coefficients(trap.observe(self.fine[ri]), BASE)
            for r, c in VALIDATION_WINDOWS:
                sl = (slice(r, r + VALIDATION_SIZE), slice(c, c + VALIDATION_SIZE))
                base = operators.upsample_window(coeff, FACTOR, BASE, *sl)
                self.validation.append((self.fine[ri][sl].astype(np.float64), base))

    def batch(self, rng):
        span = self.size + 1
        region = rng.integers(len(self.fine), size=BATCH)
        op = rng.choice(len(self.mix), size=BATCH, p=self.weights)
        top = (VALIDATION_ROW - BUFFER - span) // FACTOR
        left = (self.fine[0].shape[1] - span) // FACTOR
        origins = np.column_stack([rng.integers(0, top + 1, BATCH), rng.integers(0, left + 1, BATCH)])
        z = np.stack([self.fine[ri][r * FACTOR:r * FACTOR + span, c * FACTOR:c * FACTOR + span]
                      for ri, (r, c) in zip(region, origins)])
        field = np.array([self.index[(ri, oi)] for ri, oi in zip(region, op)])
        base = self.windows(self.coeffs, np.column_stack([field, origins]))
        stack = torch.stack([base, torch.as_tensor(z, device=self.device)], 1)
        stack = self.sampler.gather(stack, np.arange(BATCH), np.tile([0, 1], (BATCH, 1)),
                                    np.zeros((BATCH, 2), int), rng)
        base, z = stack[:, :1], stack[:, 1:2]
        x, sigma = models.inputs(SPEC, base, self.phases, None)
        m = SPEC.margin
        return x, sigma, (z - base)[..., m:-m, m:-m] / sigma, self.weight

    def validate(self, model):
        maes = []
        for z, base in self.validation:
            loc, _ = train.predict(model, SPEC, base, None, tile=TILE, halo=HALO, device=self.device)
            maes.append(np.abs(base + loc - z)[INNER:-INNER, INNER:-INNER].mean())
        return float(np.mean(maes))


def fit_baselines(regions, log=print) -> dict:
    trap = operators.make("trapezoid", FACTOR)
    pairs = []
    for region in regions:
        fine = np.asarray(fields.fine(region), dtype=np.float32)
        pairs.append((trap.observe(fine), fine))
    out = {"linear": baselines.fit_phase_kernels(pairs, FACTOR, max_per_phase=100_000)}
    rng = np.random.default_rng(0)
    xs, ts, windows = [], [], []
    for y, fine in pairs:
        for r, c in RK_WINDOWS:
            sl = (slice(r, r + 2048), slice(c, c + 2048))
            base = operators.upsample_window(operators.coefficients(y, BASE), FACTOR, BASE, *sl)
            x = baselines.covariates(base, None, 1)
            pick = rng.choice(base.size, 100_000, replace=False)
            xs.append(x[pick])
            ts.append((fine[sl] - base).ravel()[pick])
            windows.append((base, fine[sl].astype(np.float64)))
    x, t = np.concatenate(xs), np.concatenate(ts)
    beta = np.linalg.lstsq(x.T @ x + 1e-8 * np.eye(x.shape[1]), x.T @ t, rcond=None)[0]
    residuals = [f - b - (baselines.covariates(b, None, 1) @ beta).reshape(b.shape) for b, f in windows]
    model = baselines.fit_exponential(baselines.empirical_covariance(residuals, 3 * FACTOR), FACTOR)
    weights = trap._weights(4 * FACTOR + 1, 0)[2]
    support = np.flatnonzero(weights)
    kernels, variance = baselines.atpk_kernels(model, weights[support], weights[support], FACTOR)
    out["rk"] = {"beta": beta, "covariance": model, "kernels": kernels, "variance": variance}
    log(f"fine baselines: rk covariance {model}")
    return out


def rk_predict(fitted, y, base, operator):
    rk = fitted["rk"]
    gy, gx = np.gradient(base)
    p = np.pad(base, 1, mode="edge")
    lap = p[1:-1, 2:] + p[1:-1, :-2] + p[2:, 1:-1] + p[:-2, 1:-1] - 4.0 * base
    trend = base + rk["beta"][0] * lap + rk["beta"][1] * np.hypot(gx, gy) + rk["beta"][2]
    del gy, gx, p, lap
    return trend + baselines.apply_phase_kernels(y - operator.observe(trend), rk["kernels"], FACTOR)


def score(estimate, z, y, op) -> dict:
    inner = (slice(INNER, -INNER), slice(INNER, -INNER))
    row = evaluate.heights(estimate[inner], z[inner])
    row.update(evaluate.observation(estimate, y, op))
    row["tileMaeM"] = evaluate.tiles(estimate, z, 2048)
    return row


def evaluate_region(region, members, fitted, device, log=print) -> dict:
    from geoneural.metrics import drainage
    z = np.asarray(fields.fine(region), dtype=np.float64)
    trap = operators.make("trapezoid", FACTOR)
    windows = [(slice(r, r + DRAINAGE_SIZE), slice(c, c + DRAINAGE_SIZE)) for r, c in DRAINAGE_ORIGINS]
    routed = [drainage.route(z[w], 1.0) for w in windows]
    out = {"region": region, "noiseFloor": [evaluate.floor(z[w], 1.0) for w in windows], "operators": {}}
    for name in ("trapezoid", "provider", "decimate"):
        start = time.perf_counter()
        op = trap if name == "provider" else operators.make(name, FACTOR)
        y = provider_grid(region) if name == "provider" else op.observe(z)
        rows, drain = {}, {}

        def add(key, surface, project=True):
            rows[key] = score(surface, z, y, op)
            if name == "trapezoid":
                drain[key] = [evaluate.streams(r, surface[w], 1.0) for r, w in zip(routed, windows)]
            if project:
                fixed, report = op.project(surface, y)
                rows[key + "+bp"] = score(fixed, z, y, op)
                rows[key + "+bp"]["backProjection"] = report
                if name == "trapezoid":
                    drain[key + "+bp"] = [evaluate.streams(r, fixed[w], 1.0) for r, w in zip(routed, windows)]
                return fixed
            return surface

        add("bicubic", operators.upsample(y, FACTOR, "keys"))
        base = operators.upsample(y, FACTOR, BASE)
        add("bspline", base)
        add("linear", baselines.apply_phase_kernels(y, fitted["linear"], FACTOR))
        add("rkNoGeology", rk_predict(fitted, y, base, trap))
        locs = []
        for k, model in enumerate(members):
            loc, _ = train.predict(model, SPEC, base, None, tile=TILE, halo=HALO, device=device)
            locs.append(loc.astype(np.float32))
            add(f"neural-seed{k}", base + loc)
            del loc
        if len(locs) > 1:
            add("neural-ensemble", base + np.mean(locs, axis=0))
        if name == "trapezoid":
            fits = {}
            for i, w in enumerate(windows[:FIT_WINDOWS]):
                cw = (slice(w[0].start // FACTOR, (w[0].stop - 1) // FACTOR + 1),
                      slice(w[1].start // FACTOR, (w[1].stop - 1) // FACTOR + 1))
                start_w, _ = op.project(base[w], y[cw])
                fit, report = baselines.biharmonic_fit(y[cw], op, base[w].shape, ALPHA, start_w, rtol=1e-5,
                                                       maxiter=800)
                inner = (slice(INNER, -INNER), slice(INNER, -INNER))
                fits[str(i)] = {"heights": evaluate.heights(fit[inner], z[w][inner]), "solver": report,
                                "drainage": evaluate.streams(routed[i], fit, 1.0),
                                "sameWindowOthers": {k: evaluate.heights(v[inner], z[w][inner])
                                                     for k, v in (("bspline+bp", start_w),)}}
            out["biharmonicFitWindows"] = fits
        out["operators"][name] = {"methods": rows, "drainageByWindow": drain, "seconds": time.perf_counter() - start}
        log(f"{region} fine {name}: " + ", ".join(f"{k} {v['maeM']:.4f}" for k, v in rows.items() if "seed" not in k))
        del base, locs
    return out


def run_fold(fold, seeds, device, models_dir, frozen=None, log=print, max_steps=MAX_STEPS, checkpoints=CHECKPOINTS,
             lr_grid=LR_GRID):
    start = time.perf_counter()
    task = Task(fold["train"], operators.training_mix(FACTOR), device)
    runs = []
    if frozen:
        recipe = dict(frozen["neural"])
    else:
        for lr in lr_grid:
            run = train.fit(task, lr=lr, steps=max_steps, checkpoints=checkpoints, seed=seeds[0], device=device)
            log(f"{fold['test']} fine lr {lr} curve {train.curve(run)}")
            runs.append(run)
        recipe = train.select(runs)
    members = train.members(task, runs, recipe, seeds, device, not frozen, log)
    folder = models_dir / "-".join(fold["test"])
    folder.mkdir(parents=True, exist_ok=True)
    for k, m in enumerate(members):
        torch.save(m["model"].state_dict(), folder / f"neural-seed{k}.pt")
    del task
    fitted = fit_baselines(fold["train"], log)
    regions = {r: evaluate_region(r, [m["model"] for m in members], fitted, device, log) for r in fold["test"]}
    return {"fold": fold, "seeds": list(seeds), "learningRateGrid": list(lr_grid), "recipes": {"neural": recipe},
            "training": {"neural": train.history(runs, members)},
            "failures": [{"variant": "neural", "seed": r["seed"], "lr": r["lr"], "reason": r["collapseReason"]}
                         for r in runs + members if r["collapsed"] or r["stalled"]],
            "baselines": {"rk": {"beta": fitted["rk"]["beta"].tolist(), "covariance": fitted["rk"]["covariance"]}},
            "regions": regions, "seconds": time.perf_counter() - start}


def folds(regions, test_regions=None):
    """Leave one region out with no validation region: validation is the southern band of the training regions."""
    regions = list(regions)
    if test_regions:
        return [{"test": list(test_regions), "validation": [], "train": regions}]
    return [{"test": [r], "validation": [f"{t}:rows>={VALIDATION_ROW}" for t in regions if t != r],
             "train": [t for t in regions if t != r]} for r in regions]


NOTES = ("Task 2, 10 m -> 1 m on the three regions with 1 m data. Leave one region out: two regions train, the "
         "southern band of each training region validates, the third region is scored over its whole 10.24 km "
         "interior (50 m trimmed). 'provider' feeds the provider's own 10 m product of the same ground; "
         "back-projection then enforces the declared trapezoid (the wrong operator for that input). Drainage at "
         "0.05 km^2 (50,000 cells) on sixteen 2041-node windows per region, mean over windows, noise floor "
         "beside it. Paired differences over 25 tiles of 2 km within a region, and over the three regions.")
COMPARISONS = (("neural+bp", "bspline+bp"), ("neural+bp", "bicubic+bp"), ("neural+bp", "linear+bp"),
               ("neural+bp", "rkNoGeology+bp"), ("neural", "bspline"), ("linear+bp", "bspline+bp"))


def selection_at_cap(parts) -> dict:
    """Folds whose selected step is the largest checkpoint: the rule may have been limited by the cap."""
    return {"-".join(p["fold"]["test"]): {v: r["steps"] for v, r in p["recipes"].items()
                                          if isinstance(r, dict) and r.get("steps") == max(CHECKPOINTS)}
            for p in parts}


def summarise(parts):
    per_op: dict = {}
    drainage, floors, fits, recipes, training, failures, fitted = {}, {}, {}, {}, {}, [], {}
    for part in parts:
        key = "-".join(part["fold"]["test"])
        recipes[key] = {"fold": part["fold"], "recipes": part["recipes"]}
        training[key] = part["training"]
        fitted[key] = part["baselines"]
        failures += [dict(f, fold=key) for f in part["failures"]]
        for region, result in part["regions"].items():
            floors[region] = {s: {k: evaluate._mean([w[s].get(k) for w in result["noiseFloor"]])
                                  for k in ("jaccard", "tolerantF1", "receiverAgreement", "outletAgreement")}
                              for s in ("0.01m", "0.1m")}
            fits[region] = result.get("biharmonicFitWindows")
            for op, block in result["operators"].items():
                per_op.setdefault(op, {})[region] = evaluate.merge_seeds(block["methods"])
                if block["drainageByWindow"]:
                    drainage[region] = evaluate.merge_seed_metrics(
                        {m: {k: evaluate._mean([w.get(k) for w in rows]) for k in evaluate.DRAINAGE_KEYS}
                         for m, rows in block["drainageByWindow"].items()})
    tables = {op: {r: {m: evaluate.compact(row) for m, row in methods.items()} for r, methods in regions.items()}
              for op, regions in per_op.items()}
    balanced = {op: {m: {k: evaluate.region_balanced({r: tables[op][r][m].get(k) for r in tables[op]
                                                       if m in tables[op][r]})
                         for k in ("maeM", "rmseM", "p99M", "maxM", "obsRmsM")}
                     for m in next(iter(tables[op].values()))} for op in tables}
    pairs = {op: [evaluate.paired(per_op[op], a, b) for a, b in COMPARISONS
                  if all(a in m and b in m for m in per_op[op].values())] for op in per_op}
    drain_bal = {m: {k: evaluate.region_balanced({r: d.get(m, {}).get(k) for r, d in drainage.items()})
                     for k in evaluate.DRAINAGE_KEYS} for m in next(iter(drainage.values()))} if drainage else {}
    floor_bal = {s: {k: evaluate.region_balanced({r: f[s][k] for r, f in floors.items()})
                     for k in ("jaccard", "tolerantF1", "receiverAgreement", "outletAgreement")}
                 for s in ("0.01m", "0.1m")}
    return {"perRegion": tables, "regionBalanced": balanced, "paired": pairs,
            "drainage": {"byRegion": drainage, "regionBalanced": drain_bal, "noiseFloor": floors,
                         "noiseFloorRegionBalanced": floor_bal},
            "biharmonicFitWindows": fits, "recipes": recipes, "selectionAtCap": selection_at_cap(parts),
            "training": training, "fittedBaselines": fitted}, failures


def describe(parts):
    recipe = {"task": "coarse-to-fine 10 m -> 1 m", "factor": FACTOR, "base": BASE, "features": SPEC.record(),
              "trainingOperators": [{"operator": op.schema(), "weight": w} for op, w in operators.training_mix(FACTOR)],
              "evaluationInputs": ["trapezoid (declared)", "provider 10 m product (real mismatch)", "decimate"],
              "model": {"type": "U-Net width 32, as the 40 m task", "window": CORE, "lossBorder": BORDER,
                        "batch": BATCH, "inferenceTile": TILE, "inferenceHalo": HALO},
              "validation": {"rows": f">= {VALIDATION_ROW} of each training region", "buffer": BUFFER,
                             "windows": [list(w) for w in VALIDATION_WINDOWS], "size": VALIDATION_SIZE},
              "selection": {"learningRates": {"-".join(p["fold"]["test"]): p.get("learningRateGrid", list(LR_GRID))
                                              for p in parts},
                            "checkpoints": sorted({e["step"] for p in parts for runs in p["training"].values()
                                                   for r in runs for e in r["curve"]})},
              "baselines": {"bicubic": "Keys", "bspline": "cubic B-spline", "linear": "per-phase 8x8 kernels "
                            "(100 phases), training regions", "rkNoGeology": "trend plus area-to-point kriging",
                            "biharmonicFit": f"alpha {ALPHA} (the 40 m validation choice), first {FIT_WINDOWS} "
                                             "drainage windows only"},
              "drainage": {"windowSize": DRAINAGE_SIZE, "origins": [list(o) for o in DRAINAGE_ORIGINS],
                           "streamAreaM2": 50000}}
    return recipe, {"folds": [p["fold"] for p in parts], "rule": "leave one region out, southern-band validation"}


def gates(results):
    from geoneural.evaluation import protocol
    bal = results["regionBalanced"]["trapezoid"]
    best = min(("bspline+bp", "bicubic+bp", "linear+bp", "rkNoGeology+bp"), key=lambda m: bal[m]["maeM"])
    pair = next(p for p in results["paired"]["trapezoid"] if p["comparison"] == f"neural+bp minus {best}")
    gain = 1 - bal["neural+bp"]["maeM"] / bal[best]["maeM"]
    out = {"beatsStrongestNonNeural": protocol.gate(
        "pass" if gain >= 0.02 and pair["overRegions"]["high"] < 0 else "fail",
        f"neural+bp {bal['neural+bp']['maeM']:.4f} m against {best} {bal[best]['maeM']:.4f} m (gain {gain:.1%})",
        minimumGain=0.02, intervalExcludesZero=True)}
    d = results["drainage"]["regionBalanced"]
    if d:
        diff = d["neural+bp"]["tolerantF1"] - d["bspline+bp"]["tolerantF1"]
        out["drainageNotWorse"] = protocol.gate("pass" if diff >= -0.01 else "fail",
                                                f"tolerant F1 neural+bp minus bspline+bp {diff:+.3f}", margin=0.01)
    prov = results["regionBalanced"]["provider"]
    out["providerInputNotWorse"] = protocol.gate(
        "pass" if prov["neural"]["maeM"] <= prov["bspline"]["maeM"] else "fail",
        f"raw neural {prov['neural']['maeM']:.4f} m against raw B-spline {prov['bspline']['maeM']:.4f} m on the "
        "provider 10 m input")
    return out


def frozen_recipe(parts):
    chosen = [p["recipes"]["neural"] for p in parts]
    rates = [c["lr"] for c in chosen]
    lr = min(set(rates), key=lambda r: (-rates.count(r), r))
    steps = int(np.median([c["steps"] for c in chosen]))
    steps = min(CHECKPOINTS, key=lambda c: (abs(c - steps), c))
    return {"task": "fine", "recipes": {"neural": {"lr": lr, "steps": steps}},
            "fromFolds": [p["fold"] for p in parts]}
