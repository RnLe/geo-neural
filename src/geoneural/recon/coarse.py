"""Coarse-to-fine reconstruction, 40 m -> 10 m, on the 10 m development regions.

The reference z is observed through an operator H (factor 4: 1025^2 nodes -> 257^2). Every method receives y = H z
and the name of the operator; nothing else about the test region. The neural model is trained on a mix of
operators (trapezoid, Gaussian of four widths, point decimation, provider-style), so it cannot rely on one kernel,
and it is always scored under every evaluation operator, two of which it never saw (the operator-mismatch test).
Back-projection to 1e-4 m uses the operator that produced y; that is an oracle when the operator is not the
declared one, and the raw output is reported beside it.

The neural base is the interpolating cubic B-spline of y (the better of the two cubic interpolators on the
development regions; Keys bicubic is reported as the conventional baseline). The model predicts the residual
over it, in units of a local scale computed from the base alone.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from geoneural.recon import baselines, evaluate, fields, models, operators, train

FACTOR = 4
SPEC = models.Features(scale_sigma=4.0, highpass_sigma=2.0, unit=4.0, floor_m=0.05, factor=FACTOR, margin=16)
CORE, BORDER, BATCH = 128, 16, 16
LR_GRID = (5e-4, 1e-3)
CHECKPOINTS = (500, 1000, 1500, 2000, 3000, 4000, 5000)
MAX_STEPS = 5000
ALPHAS = (0.003, 0.03, 0.3)
GEOLOGY_SHIFT = (512, 512)
BASE = "bspline"


class Data:
    """Training regions on the device as one (regions, channels, H, W) tensor: a base per mix operator, the
    reference, then geology slots."""

    def __init__(self, regions, mix, slots_map, device):
        self.device, self.mix = device, mix
        self.weights = np.array([w for _, w in mix]) / sum(w for _, w in mix)
        stacks = []
        for region in regions:
            z = fields.reference(region)
            layers = [operators.upsample(op.observe(z), FACTOR, BASE) for op, _ in mix]
            layers += [z, slots_map[fields.geology(region, DICTIONARY)].astype(np.float64)]
            stacks.append(np.stack(layers).astype(np.float32))
        self.tensor = torch.tensor(np.stack(stacks), device=device)
        self.reference, self.geology = len(mix), len(mix) + 1


DICTIONARY = fields.class_dictionary()
SLOTS = len(DICTIONARY) + 1


class Task:
    """One training configuration: which data, with or without geology, and what validates it."""

    def __init__(self, data: Data, geology: bool, validation: str | None, slots_map, device):
        self.data, self.geology, self.device = data, geology, device
        self.channels = SPEC.base_channels() + (SLOTS if geology else 0)
        self.validation = validation
        if validation:
            z = fields.reference(validation)
            y = operators.make("trapezoid", FACTOR).observe(z)
            self._val = (z, operators.upsample(y, FACTOR, BASE),
                         extras(validation, slots_map) if geology else None)
        self.size = CORE + 2 * SPEC.margin
        self.sampler = models.WindowSampler(self.size + 1, self.size, device)
        self.phases = models.phase_maps(0, 0, self.size, self.size, FACTOR, device)[None].expand(BATCH, -1, -1, -1)
        self.weight = torch.zeros(BATCH, 1, CORE, CORE, device=device)
        self.weight[..., BORDER:-BORDER, BORDER:-BORDER] = 1.0

    def batch(self, rng):
        n_regions, _, side, _ = self.data.tensor.shape
        region = rng.integers(n_regions, size=BATCH)
        op = rng.choice(len(self.data.mix), size=BATCH, p=self.data.weights)
        channels = np.stack([op, np.full(BATCH, self.data.reference), np.full(BATCH, self.data.geology)], 1)
        # Windows start on a coarse node and span a multiple of f plus one node, so any rotation or flip
        # keeps coarse nodes on coarse nodes; the sampler then drops the last node, keeping the phase origin.
        limit = (side - self.size - 1) // FACTOR
        origins = rng.integers(0, limit + 1, (BATCH, 2)) * FACTOR
        stack = self.sampler.gather(self.data.tensor, region, channels, origins, rng)
        base, z, geo = stack[:, :1], stack[:, 1:2], stack[:, 2:3]
        ex = train.one_hot(geo, SLOTS) if self.geology else None
        x, sigma = models.inputs(SPEC, base, self.phases, ex)
        m = SPEC.margin
        target = (z - base)[..., m:-m, m:-m] / sigma
        return x, sigma, target, self.weight

    def validate(self, model):
        z, base, ex = self._val
        loc, _ = train.predict(model, SPEC, base, ex, device=self.device)
        return float(np.abs(base + loc - z)[8:-8, 8:-8].mean())


def extras(region: str, slots_map, shift=None) -> np.ndarray:
    classes = slots_map[fields.geology(region, DICTIONARY)]
    if shift is not None:
        classes = np.roll(classes, shift, axis=(0, 1))
    return np.eye(SLOTS, dtype=np.float32)[classes].transpose(2, 0, 1).copy()


def train_variants(fold: dict, seeds, device: str, frozen: dict | None = None, log=print,
                   max_steps: int = MAX_STEPS, checkpoints=CHECKPOINTS, lr_grid=LR_GRID,
                   geology_seeds: int | None = None) -> dict:
    """Train the variants without and with geology for one fold by the declared selection rule."""
    slots_map = fields.fold_classes(fold["train"], DICTIONARY)
    data = Data(fold["train"], operators.training_mix(FACTOR), slots_map, device)
    validation = fold["validation"][0] if fold["validation"] else None
    out = {"slotsMap": slots_map.tolist(), "variants": {}, "failures": []}
    for variant, geology in (("neural", False), ("neural-geology", True)):
        task = Task(data, geology, validation, slots_map, device)
        runs = []
        if frozen:
            recipe = dict(frozen[variant])
        else:
            grid = tuple(lr_grid) if variant == "neural" else (out["variants"]["neural"]["recipe"]["lr"],)
            for lr in grid:
                run = train.fit(task, lr=lr, steps=max_steps, checkpoints=checkpoints, seed=seeds[0], device=device)
                log(f"{fold['test']} {variant} lr {lr} curve {train.curve(run)}")
                runs.append(run)
            recipe = train.select(runs)
        chosen = seeds if variant == "neural" or not geology_seeds else seeds[:geology_seeds]
        members = train.members(task, runs, recipe, chosen, device, bool(validation), log)
        for run in runs + members:
            if run["collapsed"] or run["stalled"]:
                out["failures"].append({"variant": variant, "seed": run["seed"], "lr": run["lr"],
                                        "reason": run["collapseReason"]})
        out["variants"][variant] = {"recipe": recipe, "members": [m["model"] for m in members],
                                    "training": train.history(runs, members)}
    return out


def fit_baselines(fold: dict, slots_map, log=print) -> dict:
    """Linear kernels, regression-kriging and the regularisation weight, from training (and validation) regions."""
    trap = operators.make("trapezoid", FACTOR)
    refs = {r: fields.reference(r) for r in fold["train"]}
    pairs = [(trap.observe(z), z) for z in refs.values()]
    out = {"linear": baselines.fit_phase_kernels(pairs, FACTOR)}
    mix_pairs = [(op.observe(z), z) for z in refs.values() for op, _ in operators.training_mix(FACTOR)]
    out["linearMix"] = baselines.fit_phase_kernels(mix_pairs, FACTOR)
    samples = [(operators.upsample(y, FACTOR, BASE), z, slots_map[fields.geology(r, DICTIONARY)])
               for (y, z), r in zip(pairs, refs)]
    for name, geology in (("rk", True), ("rkNoGeology", False)):
        beta = baselines.fit_trend(samples, SLOTS, geology)
        residuals = [z - b - (baselines.covariates(b, s if geology else None, SLOTS) @ beta).reshape(b.shape)
                     for b, z, s in samples]
        model = baselines.fit_exponential(baselines.empirical_covariance(residuals), FACTOR)
        weights = trap._weights(4 * FACTOR + 1, 0)[2]
        support = np.flatnonzero(weights)
        kernels, variance = baselines.atpk_kernels(model, weights[support], weights[support], FACTOR)
        out[name] = {"beta": beta, "covariance": model, "kernels": kernels, "variance": variance, "geology": geology}
    out["alpha"] = {"grid": list(ALPHAS), "validationMaeM": {}}
    if fold["validation"]:
        z = fields.reference(fold["validation"][0])
        y = trap.observe(z)
        start, _ = trap.project(operators.upsample(y, FACTOR, BASE), y)
        for alpha in ALPHAS:
            fit, _ = baselines.biharmonic_fit(y, trap, z.shape, alpha, start)
            out["alpha"]["validationMaeM"][str(alpha)] = float(np.abs(fit - z)[8:-8, 8:-8].mean())
        out["alpha"]["selected"] = float(min(ALPHAS, key=lambda a: out["alpha"]["validationMaeM"][str(a)]))
    else:
        out["alpha"]["selected"] = None
    log(f"baselines fitted for {fold['test']}: alpha {out['alpha']}")
    return out


def rk_predict(fitted: dict, y, base, slots, operator):
    beta = fitted["beta"]
    trend = base + (baselines.covariates(base, slots if fitted["geology"] else None, SLOTS) @ beta).reshape(base.shape)
    d = y - operator.observe(trend)
    return trend + baselines.apply_phase_kernels(d, fitted["kernels"], FACTOR)


def score_surface(estimate, z, y, op, routed=None, region_tiles=True) -> dict:
    inner = (slice(8, -8), slice(8, -8))
    row = evaluate.heights(estimate[inner], z[inner])
    row.update(evaluate.observation(estimate, y, op))
    if region_tiles:
        row["tileMaeM"] = evaluate.tiles(estimate, z)
    if routed is not None:
        row["drainage"] = evaluate.streams(routed, estimate, fields.SPACING_M)
    return row


def evaluate_region(region: str, trained: dict, fitted: dict, slots_map, device: str, alpha_fallback: float,
                    log=print) -> dict:
    """Every method under every evaluation operator on one held-out region."""
    from geoneural.metrics import drainage
    z = fields.reference(region)
    slots = slots_map[fields.geology(region, DICTIONARY)]
    geo = extras(region, slots_map)
    geo_shifted = extras(region, slots_map, GEOLOGY_SHIFT)
    routed = drainage.route(z, fields.SPACING_M)
    slope = fields.slope_class(z)
    out = {"region": region, "noiseFloor": evaluate.floor(z, fields.SPACING_M), "operators": {},
           "geologyUnknownFraction": float((slots == 0).mean())}
    trap = operators.make("trapezoid", FACTOR)
    for name in operators.EVALUATION:
        start = time.perf_counter()
        op = operators.make(name, FACTOR)
        primary = name in ("trapezoid", "provider-style")
        y = op.observe(z)
        rows, projections = {}, {}

        def add(key, surface, project=True):
            rows[key] = score_surface(surface, z, y, op, routed if primary else None)
            if project:
                fixed, report = op.project(surface, y)
                projections[key] = report
                rows[key + "+bp"] = score_surface(fixed, z, y, op, routed if primary else None)
                rows[key + "+bp"]["backProjection"] = report
                return fixed
            return surface

        keys = operators.upsample(y, FACTOR, "keys")
        add("bicubic", keys)
        base = operators.upsample(y, FACTOR, BASE)
        add("bspline", base)
        add("linear", baselines.apply_phase_kernels(y, fitted["linear"], FACTOR))
        add("linearMix", baselines.apply_phase_kernels(y, fitted["linearMix"], FACTOR))
        for rk in ("rk", "rkNoGeology"):
            add(rk, rk_predict(fitted[rk], y, base, slots, trap))
        start_bh, _ = op.project(base, y)
        alpha = fitted["alpha"]["selected"] or alpha_fallback
        fit, report = baselines.biharmonic_fit(y, op, z.shape, alpha, start_bh)
        rows["biharmonicFit"] = score_surface(fit, z, y, op, routed if primary else None)
        rows["biharmonicFit"]["solver"] = report
        surfaces = {}
        for variant, members in trained.items():
            controls = [(variant, geo if variant == "neural-geology" else None)]
            if variant == "neural-geology":
                controls.append(("neural-geology-shifted", geo_shifted))
            for label, ex in controls:
                locs, spreads, per_seed = [], [], []
                for k, model in enumerate(members):
                    loc, spread = train.predict(model, SPEC, base, ex, device=device)
                    locs.append(loc)
                    spreads.append(spread)
                    est = base + loc
                    fixed = add(f"{label}-seed{k}", est)
                    per_seed.append(fixed)
                add(f"{label}-ensemble", base + np.mean(locs, axis=0))
                surfaces[label] = (locs, spreads, base)
        out["operators"][name] = {"methods": rows, "seconds": time.perf_counter() - start,
                                  "operator": op.schema()}
        if name == "trapezoid":
            out["uncertainty"] = uncertainty(z, slope, surfaces, fitted, y, base, slots, trap, device)
        log(f"{region} {name}: " + ", ".join(f"{k} {v['maeM']:.3f}" for k, v in rows.items()
                                            if k.endswith("+bp") and "seed" not in k))
    return out


def uncertainty(z, slope, surfaces, fitted, y, base, slots, op, device) -> dict:
    """Interval coverage and width of the single-model, ensemble and kriging intervals, by slope class.

    Intervals sit around the raw predictions (before back-projection), which is what the spread was trained
    on; back-projection moves the trapezoid estimate by a few millimetres.
    """
    levels = evaluate.quantile_levels()
    out = {}
    for label, (locs, spreads, b) in surfaces.items():
        if label.endswith("shifted"):
            continue
        mu = torch.tensor(np.stack(locs), device=device) + torch.tensor(b, device=device)
        sp = torch.tensor(np.stack(spreads), device=device)
        truth = torch.tensor(z, device=device)
        for name, members in (("single", slice(0, 1)), ("ensemble", slice(None))):
            q = models.mixture_quantiles(mu[members], sp[members], levels)
            quantiles = {lv: q[i].cpu().numpy() for i, lv in enumerate(levels)}
            pit = models.mixture_cdf(mu[members], sp[members], truth).cpu().numpy()
            out[f"{label}-{name}"] = {"intervals": evaluate.intervals(z, quantiles, slope),
                                      "pit": evaluate.pit_histogram(pit)}
    from scipy.stats import norm
    for rk in ("rk", "rkNoGeology"):
        mean = rk_predict(fitted[rk], y, base, slots, op)
        sd = np.sqrt(np.tile(fitted[rk]["variance"], (z.shape[0] // FACTOR + 1, z.shape[1] // FACTOR + 1))
                     [:z.shape[0], :z.shape[1]])
        quantiles = {lv: mean + norm.ppf(lv) * sd for lv in levels}
        out[f"{rk}-kriging"] = {"intervals": evaluate.intervals(z, quantiles, slope),
                                "pit": evaluate.pit_histogram(norm.cdf((z - mean) / np.maximum(sd, 1e-9)))}
    return out


def reload(part: dict, folder, device) -> dict:
    """The trained variants of an earlier part, from its recipes and the saved weights."""
    variants = {}
    for variant, geology in (("neural", False), ("neural-geology", True)):
        members = []
        for path in sorted(folder.glob(f"{variant}-seed*.pt")):
            model = models.UNet(SPEC.base_channels() + (SLOTS if geology else 0), 32).to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            members.append(model.eval())
        variants[variant] = {"recipe": part["recipes"][variant], "members": members,
                             "training": part["training"][variant]}
    return {"slotsMap": part["slotsMap"], "variants": variants, "failures": part["failures"]}


def run_fold(fold: dict, seeds, device: str, models_dir, frozen: dict | None = None, log=print,
             max_steps: int = MAX_STEPS, checkpoints=CHECKPOINTS, lr_grid=LR_GRID,
             geology_seeds: int | None = None, reuse: dict | None = None) -> dict:
    """Train, fit baselines and score the test regions of one fold. Returns a JSON-safe part record."""
    start = time.perf_counter()
    folder = models_dir / "-".join(fold["test"])
    if reuse:
        trained = reload(reuse, folder, device)
        lr_grid, geology_seeds = reuse.get("learningRateGrid", LR_GRID), reuse.get("geologySeeds")
        seeds = reuse["seeds"]
        log(f"{fold['test']}: reusing the trained models, scoring again")
    else:
        trained = train_variants(fold, seeds, device, frozen, log, max_steps, checkpoints, lr_grid, geology_seeds)
    slots_map = np.asarray(trained["slotsMap"])
    fitted = fit_baselines(fold, slots_map, log)
    folder.mkdir(parents=True, exist_ok=True)
    for variant, info in trained["variants"].items():
        for k, model in enumerate(info["members"]):
            torch.save(model.state_dict(), folder / f"{variant}-seed{k}.pt")
    members = {v: info["members"] for v, info in trained["variants"].items()}
    regions = {r: evaluate_region(r, members, fitted, slots_map, device, ALPHAS[1], log) for r in fold["test"]}
    return {"fold": fold, "seeds": list(seeds), "slotsMap": trained["slotsMap"], "learningRateGrid": list(lr_grid),
            "geologySeeds": geology_seeds or len(seeds), "inferenceHalo": 64, "rescored": bool(reuse),
            "recipes": {v: info["recipe"] for v, info in trained["variants"].items()},
            "training": {v: info["training"] for v, info in trained["variants"].items()},
            "failures": trained["failures"],
            "baselines": {"alpha": fitted["alpha"],
                          **{rk: {"beta": fitted[rk]["beta"].tolist(), "covariance": fitted[rk]["covariance"],
                                  "krigingVarianceByPhase": fitted[rk]["variance"].tolist()}
                             for rk in ("rk", "rkNoGeology")}},
            "regions": regions, "seconds": time.perf_counter() - start}


DEVELOPMENT_FAILURES = [
    {"stage": "development", "run": "first launch with at most 3000 steps",
     "reason": "validation MAE still fell between the last two checkpoints in both first folds (0.1049 -> 0.1018 m "
               "on muensterland-plain), so the cap rather than the rule would have chosen the step; stopped after "
               "the first learning-rate run and restarted with checkpoints to 5000. No result of it is reported."},
    {"stage": "budget", "run": "folds testing lower-rhine, teutoburg-forest, bergisches-land, rothaar-sauerland",
     "reason": "the shared GPU slowed training about threefold after the first two folds; their first attempt was "
               "stopped part-way through its first run and rerun with the learning-rate grid reduced to 5e-4 (the "
               "rate both completed folds selected, lower at every checkpoint than 1e-3) and the geology variant "
               "on the first seed only. Per-fold grids and seed counts are in the parts (learningRateGrid, "
               "geologySeeds)."}]
COMPARISONS = (("neural+bp", "bspline+bp"), ("neural+bp", "bicubic+bp"), ("neural+bp", "linear+bp"),
               ("neural+bp", "rk+bp"), ("neural+bp", "biharmonicFit"), ("neural-geology+bp", "neural+bp"),
               ("neural-geology+bp", "rk+bp"), ("neural-geology+bp", "neural-geology-shifted+bp"),
               ("neural-ensemble+bp", "neural+bp"), ("rk+bp", "rkNoGeology+bp"), ("linear+bp", "bspline+bp"))
MISMATCH = ("bicubic", "bspline", "bspline+bp", "linear", "linearMix", "rk", "neural", "neural+bp",
            "neural-geology", "neural-geology+bp")


def selection_at_cap(parts) -> dict:
    """Folds whose selected step is the largest checkpoint: the rule may have been limited by the cap."""
    return {"-".join(p["fold"]["test"]): {v: r["steps"] for v, r in p["recipes"].items()
                                          if isinstance(r, dict) and r.get("steps") == max(CHECKPOINTS)}
            for p in parts}


def summarise(parts: list[dict]) -> dict:
    """Per-region tables, region-balanced means, paired differences, mismatch, drainage and calibration."""
    per_op: dict = {}
    floors, uncertainty, recipes, training, failures, fitted = {}, {}, {}, {}, [], {}
    for part in parts:
        for region, result in part["regions"].items():
            floors[region] = result["noiseFloor"]
            uncertainty[region] = result.get("uncertainty")
            for op, block in result["operators"].items():
                per_op.setdefault(op, {})[region] = evaluate.merge_seeds(block["methods"])
        key = "-".join(part["fold"]["test"])
        recipes[key] = {"fold": part["fold"], "recipes": part["recipes"]}
        training[key] = part["training"]
        fitted[key] = part["baselines"]
        failures += [dict(f, fold=key) for f in part["failures"]]
    tables = {op: {r: {m: evaluate.compact(row) for m, row in methods.items()} for r, methods in regions.items()}
              for op, regions in per_op.items()}
    balanced = {op: {m: {k: evaluate.region_balanced({r: tables[op][r][m].get(k) for r in tables[op]
                                                       if m in tables[op][r]})
                         for k in ("maeM", "rmseM", "p99M", "maxM", "obsRmsM")}
                     for m in next(iter(tables[op].values()))} for op in tables}
    pairs = {op: [evaluate.paired(per_op[op], a, b) for a, b in COMPARISONS
                  if all(a in m and b in m for m in per_op[op].values())] for op in per_op}
    drainage = {}
    for op in ("trapezoid", "provider-style"):
        if op not in tables:
            continue
        drainage[op] = {m: {k: evaluate.region_balanced({r: (tables[op][r][m].get("drainage") or {}).get(k)
                                                          for r in tables[op] if m in tables[op][r]})
                            for k in evaluate.DRAINAGE_KEYS}
                        for m in next(iter(tables[op].values())) if m.endswith("+bp") or m == "biharmonicFit"}
    drainage["noiseFloorRegionBalanced"] = {
        s: {k: evaluate.region_balanced({r: f[s].get(k) for r, f in floors.items()})
            for k in ("jaccard", "tolerantF1", "receiverAgreement", "outletAgreement")} for s in ("0.01m", "0.1m")}
    mismatch = {op: {m: balanced[op][m]["maeM"] for m in MISMATCH if m in balanced[op]} for op in balanced}
    calibration = {}
    for region, unc in uncertainty.items():
        for label, row in (unc or {}).items():
            calibration.setdefault(label, {})[region] = row
    calibration_balanced = {
        label: {lv: {"coverage": evaluate.region_balanced({r: v["intervals"]["overall"][lv]["coverage"]
                                                          for r, v in rows.items()}),
                     "meanWidthM": evaluate.region_balanced({r: v["intervals"]["overall"][lv]["meanWidthM"]
                                                            for r, v in rows.items()})}
                for lv in map(str, evaluate.LEVELS)} for label, rows in calibration.items()}
    return {"perRegion": tables, "regionBalanced": balanced, "paired": pairs, "drainage": drainage,
            "noiseFloor": floors, "operatorMismatchRegionBalancedMaeM": mismatch,
            "uncertainty": {"byRegion": calibration, "regionBalanced": calibration_balanced},
            "recipes": recipes, "selectionAtCap": selection_at_cap(parts), "training": training,
            "fittedBaselines": fitted}, failures + DEVELOPMENT_FAILURES


NOTES = ("Task 1, 40 m -> 10 m. Leave one region out over the six development regions: the next region in the "
         "list validates (learning rate and step count by the declared rule), the remaining four train. The "
         "1 m data play no part. Neural rows are means over seeds unless named -ensemble; +bp rows are "
         "back-projected to 1e-4 m under the operator that produced the coarse grid. Paired differences are "
         "negative when the first method has the lower error; intervals are percentile bootstraps over the "
         "unit named (six regions, or 16 tiles of 2.56 km within a region). Drainage is at 0.05 km^2 with the "
         "reference noise floor beside it.")


def describe(parts):
    recipe = {"task": "coarse-to-fine 40 m -> 10 m", "factor": FACTOR, "base": BASE, "features": SPEC.record(),
              "trainingOperators": [{"operator": op.schema(), "weight": w} for op, w in operators.training_mix(FACTOR)],
              "evaluationOperators": list(operators.EVALUATION), "heldOutOperators": list(operators.HELD_OUT),
              "model": {"type": "U-Net, 3 levels, width 32, GELU, no normalisation layers, Laplace spread head on "
                                "detached features", "window": CORE, "lossBorder": BORDER, "batch": BATCH,
                        "inferenceTile": 256, "inferenceHalo": sorted({p.get("inferenceHalo", 48) for p in parts}),
                        "ema": 0.998, "optimiser": "AdamW wd 1e-4, "
                        "100-step warmup, constant rate, gradient clip 1"},
              "selection": {"learningRates": {"-".join(p["fold"]["test"]): p.get("learningRateGrid", list(LR_GRID))
                                              for p in parts},
                            "checkpoints": sorted({e["step"] for p in parts for runs in p["training"].values()
                                                   for r in runs for e in r["curve"]}),
                            "rule": "lowest validation-region MAE (declared operator, before back-projection) "
                                    "over learning rate and checkpoint, first seed; the geology variant reuses "
                                    "the rate and selects its own step"},
              "baselines": {"bicubic": "Keys cubic convolution a=-1/2 (as GDAL cubic)",
                            "bspline": "interpolating cubic B-spline",
                            "linear": "per-phase 8x8 least-squares kernel, declared operator, training regions",
                            "linearMix": "the same fitted on all training-mix operators",
                            "rk": "trend (base Laplacian, slope, lithology intercepts) plus area-to-point "
                                  "simple kriging of the trend residual from 64 coarse nodes, exponential "
                                  "covariance fitted on training regions",
                            "rkNoGeology": "the same with a constant instead of lithology",
                            "biharmonicFit": "argmin ||Hz-y||^2 + alpha ||Lz||^2, alpha on the validation region",
                            "backProjection": "z <- z + bilinear(y - Hz) to max |Hz - y| <= 1e-4 m"},
              "geology": {"dictionary": DICTIONARY, "minimumTrainingShare": 0.005,
                          "shiftedControl": list(GEOLOGY_SHIFT)},
              "scoring": "heights on nodes 8..1016, observation residual on interior coarse nodes"}
    split = {"folds": [p["fold"] for p in parts],
             "rule": "test region k, validation region k+1 in the region list, train the rest; regions do not "
                     "overlap, so no buffer is needed between groups"}
    return recipe, split


def gates(results):
    from geoneural.evaluation import protocol
    bal = results["regionBalanced"]["trapezoid"]
    strongest = min(("bspline+bp", "bicubic+bp", "linear+bp", "rk+bp", "biharmonicFit"), key=lambda m: bal[m]["maeM"])
    pair = next(p for p in results["paired"]["trapezoid"] if p["comparison"] == f"neural+bp minus {strongest}")
    gain = 1 - bal["neural+bp"]["maeM"] / bal[strongest]["maeM"]
    out = {"beatsStrongestNonNeural": protocol.gate(
        "pass" if gain >= 0.05 and pair["overRegions"]["high"] < 0 else "fail",
        f"neural+bp {bal['neural+bp']['maeM']:.4f} m against {strongest} {bal[strongest]['maeM']:.4f} m "
        f"(region-balanced MAE, gain {gain:.1%})", minimumGain=0.05, intervalExcludesZero=True)}
    worst = max(((r, m) for r, ms in results["perRegion"]["trapezoid"].items() for m, row in ms.items()
                 if m.endswith("+bp")), key=lambda rm: results["perRegion"]["trapezoid"][rm[0]][rm[1]].get(
        "backProjection", {}).get("achievedMaxM", 0.0))
    achieved = results["perRegion"]["trapezoid"][worst[0]][worst[1]]["backProjection"]["achievedMaxM"]
    out["observationConsistency"] = protocol.gate("pass" if achieved <= 1e-4 else "fail",
                                                  f"worst achieved max |Hz - y| {achieved:.2e} m", toleranceM=1e-4)
    dr = results["drainage"]["trapezoid"]
    d = dr["neural+bp"]["tolerantF1"] - dr["bspline+bp"]["tolerantF1"]
    out["drainageNotWorse"] = protocol.gate("pass" if d >= -0.01 else "fail",
                                            f"tolerant F1 neural+bp minus bspline+bp {d:+.3f}", margin=0.01)
    mm = results["operatorMismatchRegionBalancedMaeM"]
    worse = [op for op in operators.HELD_OUT if mm[op]["neural"] > mm[op]["bspline"]]
    out["heldOutOperatorNotWorse"] = protocol.gate(
        "fail" if worse else "pass", "neural raw against B-spline raw under operators never trained on"
        + (f"; worse under {worse}" if worse else ""), operators=list(operators.HELD_OUT))
    g = next(p for p in results["paired"]["trapezoid"] if p["comparison"] == "neural-geology+bp minus neural+bp")
    out["geologyHelps"] = protocol.gate("pass" if g["overRegions"]["high"] < 0 else "fail",
                                        f"mean difference {g['overRegions']['mean']:+.4f} m over regions",
                                        intervalExcludesZero=True)
    cov = results["uncertainty"]["regionBalanced"].get("neural-ensemble", {}).get("0.9", {}).get("coverage")
    out["ensembleCalibration90"] = protocol.gate("not-run" if cov is None else
                                                 ("pass" if 0.85 <= cov <= 0.95 else "fail"),
                                                 f"region-balanced 90% coverage {cov}", band=[0.85, 0.95])
    return out


def frozen_recipe(parts) -> dict:
    """Recipe for a confirmation run: per variant the most often selected rate and the median selected step."""
    out = {}
    for variant in ("neural", "neural-geology"):
        chosen = [p["recipes"][variant] for p in parts]
        rates = [c["lr"] for c in chosen]
        lr = min(set(rates), key=lambda r: (-rates.count(r), r))
        steps = int(np.median([c["steps"] for c in chosen]))
        steps = min(CHECKPOINTS, key=lambda c: (abs(c - steps), c))
        out[variant] = {"lr": lr, "steps": steps}
    return {"task": "coarse", "recipes": out, "fromFolds": [p["fold"] for p in parts],
            "rule": "most frequently selected learning rate (ties to the lower), median selected step rounded "
                    "to the checkpoint grid; frozen before any confirmation region is read"}


def tiling_check(parts, models_dir, device: str = "cpu") -> dict:
    """Largest difference between the tiled prediction used everywhere (256-node tiles, 64-node halo) and one
    window over the whole region, per fold, for the first-seed model without geology. Run in float32 on the
    host, so it measures the tiling and not the reduced precision of device inference."""
    from geoneural.recon import models as nets
    out = {}
    for part in parts:
        # Models live in the fold's folder: one test region in development, the whole cohort in a confirmation run.
        model = nets.UNet(SPEC.base_channels(), 32)
        model.load_state_dict(torch.load(models_dir / "-".join(part["fold"]["test"]) / "neural-seed0.pt",
                                         map_location="cpu"))
        for region in part["fold"]["test"]:
            z = fields.reference(region)
            base = operators.upsample(operators.make("trapezoid", FACTOR).observe(z), FACTOR, BASE)
            tiled, _ = train.predict(model, SPEC, base, None, device="cpu")
            whole, _ = train.predict(model, SPEC, base, None, tile=1280, halo=64, device="cpu", batch=1)
            out[region] = {"maxDifferenceM": float(np.abs(tiled - whole).max()),
                           "maxResidualM": float(np.abs(whole).max())}
    return out
