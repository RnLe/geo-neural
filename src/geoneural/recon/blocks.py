"""Missing blocks: square holes of 8, 32 and 128 cells in the 10 m reference, filled from their surroundings.

Each block is an independent problem: everything outside it is the reference, everything inside is unknown, and
no method reads the hole's own values. Blocks are drawn per region before any method runs, stratified by local
relief (thirds of the relief distribution of candidate positions in that region), with a fixed seed. A block's
anthropogenic share (GK100 'anthropogenic unconsolidated material' inside it) is kept as a land-use proxy.

Methods: harmonic and biharmonic infill (exact solves from one or two rings), local universal kriging with a
linear drift and a Matern 3/2 (main) or exponential covariance fitted on the training regions (neighbourhood:
rings around the hole, dense near it and sparse far out, at most 600 cells), and the neural residual model over
the biharmonic fill, with and without geology, which reads a 256-cell window centred on the block.

Drainage is read on a non-overlapping subset of blocks per size, all filled at once, routed over the whole region
and compared only within 16 cells of a block, beside the noise floor of the same cells.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from geoneural.recon import baselines, evaluate, fields, models, train

SIZES = (8, 32, 128)
WINDOW = 256
SPEC = models.Features(scale_sigma=4.0, highpass_sigma=2.0, unit=1.0, floor_m=0.05, factor=None, margin=16)
BLOCKS_PER_SIZE = 200
VALIDATION_BLOCKS = 20
BLOCK_SEED = 20261004
CONTEXT = 32
DRAINAGE_MARGIN = 16
BAND = {8: 8, 32: 16, 128: 32}
MAX_POINTS = 600
LR_GRID = (2e-4, 5e-4)
CHECKPOINTS = (500, 1000, 1500, 2000, 3000)
MAX_STEPS = 3000
BATCH = 8
#: U-Net depth: the centre of a 128-cell hole is 64 cells from known terrain, beyond three levels' reach.
LEVELS = 5
DICTIONARY = fields.class_dictionary()
SLOTS = len(DICTIONARY) + 1
ANTHROPOGENIC = DICTIONARY.get("anthropogenic unconsolidated material")


def choose_blocks(region: str, z: np.ndarray, count: int = BLOCKS_PER_SIZE, seed: int = BLOCK_SEED) -> dict:
    """Block origins per size, stratified by relief, deterministic in (region, size, seed).

    Candidates lie on an 8-cell grid wherever the 256-cell window fits. Relief is the range of the reference
    over the block plus 8 cells. Each third of the relief distribution contributes a third of the blocks; if
    spacing leaves a third short, the others make up the count, and the realised strata are recorded.
    """
    from scipy import ndimage
    out = {}
    material = fields.geology(region, DICTIONARY)
    for size in SIZES:
        rng = np.random.default_rng([seed, size, sum(map(ord, region))])
        lo, hi = WINDOW // 2 - size // 2, z.shape[0] - WINDOW // 2 - size // 2 - 1
        grid = np.arange(lo, hi + 1, 8)
        span = size + 16
        top = ndimage.maximum_filter(z, span, origin=-(span // 2), mode="nearest")
        low = ndimage.minimum_filter(z, span, origin=-(span // 2), mode="nearest")
        candidates = [(int(r), int(c)) for r in grid for c in grid]
        relief = np.array([top[r - 8, c - 8] - low[r - 8, c - 8] for r, c in candidates])
        stratum = np.digitize(relief, np.quantile(relief, [1 / 3, 2 / 3]))
        separation = {8: 16, 32: 24, 128: 16}[size]
        chosen = []

        def take(pool, want):
            taken = 0
            for i in pool:
                if taken == want:
                    break
                r, c = candidates[i]
                if all(max(abs(r - a), abs(c - b)) >= separation for a, b, *_ in chosen):
                    share = float((material[r:r + size, c:c + size] == ANTHROPOGENIC).mean()) if ANTHROPOGENIC else 0.0
                    chosen.append((r, c, int(stratum[i]), float(relief[i]), share))
                    taken += 1
            return taken

        for s in range(3):
            take(rng.permutation(np.flatnonzero(stratum == s)), count // 3 + (1 if s < count % 3 else 0))
        take(rng.permutation(len(candidates)), count - len(chosen))
        out[size] = chosen
    return out


def drainage_subset(blocks, size: int) -> list:
    """Greedy non-overlapping subset whose dilated masks do not touch, for an all-at-once drainage reading."""
    gap = size + 2 * DRAINAGE_MARGIN + 2
    keep = []
    for b in blocks:
        if all(max(abs(b[0] - k[0]), abs(b[1] - k[1])) >= gap for k in keep):
            keep.append(b)
    return keep


#: The residual of an interpolated hole grows with distance from the nearest known cell, so the target is
#: divided by sigma * (1 + d / DISTANCE_SCALE) as well as the local scale; d is known from the mask alone.
DISTANCE_SCALE = 4.0


def distance_map(rows: int, columns: int, holes, sizes, device) -> torch.Tensor:
    """(N,H,W) Chebyshev distance from each hole cell to the nearest known cell (zero outside holes)."""
    out = torch.zeros(len(holes), rows, columns, device=device)
    for k, ((r, c), s) in enumerate(zip(holes, sizes)):
        i = torch.arange(s, device=device, dtype=torch.float32)
        edge = torch.minimum(i, s - 1 - i) + 1.0
        out[k, r:r + s, c:c + s] = torch.minimum(edge[:, None], edge[None, :])
    return out


_INFILL: dict = {}


def _device_infill(size: int, device):
    key = (size, str(device))
    if key not in _INFILL:
        g, known = baselines.infill_matrix(size, "biharmonic")
        _INFILL[key] = (torch.tensor(g, dtype=torch.float32, device=device), torch.tensor(known, device=device))
    return _INFILL[key]


class Data:
    def __init__(self, regions, slots_map, device):
        stacks = [np.stack([fields.reference(r), slots_map[fields.geology(r, DICTIONARY)].astype(np.float64)])
                  for r in regions]
        self.tensor = torch.tensor(np.stack(stacks).astype(np.float32), device=device)
        self.g = {s: _device_infill(s, device) for s in SIZES}


def fill_batch(windows: torch.Tensor, origins, size: int, g) -> torch.Tensor:
    """Biharmonic fill of one hole per window (N,H,W) on the device; the hole values are never read."""
    matrix, known = g
    s = size + 2 * baselines.RING
    subs = torch.stack([windows[k, r - 2:r - 2 + s, c - 2:c - 2 + s] for k, (r, c) in enumerate(origins)])
    kv = subs.reshape(len(origins), -1)[:, known]
    mean = kv.mean(1, keepdim=True)
    values = (kv - mean) @ matrix.T + mean
    out = windows.clone()
    for k, (r, c) in enumerate(origins):
        out[k, r:r + size, c:c + size] = values[k].view(size, size)
    return out


class Task:
    levels = LEVELS

    def __init__(self, data: Data, geology: bool, validation: str | None, slots_map, device):
        self.data, self.geology, self.device = data, geology, device
        self.channels = SPEC.base_channels() + 2 + (SLOTS if geology else 0)
        self.validation = validation
        self.size = WINDOW + 2 * SPEC.margin
        self.sampler = models.WindowSampler(self.size, self.size, device)
        if validation:
            z = fields.reference(validation)
            chosen = choose_blocks(validation, z, VALIDATION_BLOCKS, seed=BLOCK_SEED + 1)
            self._val = (z, {s: [b[:2] for b in v] for s, v in chosen.items()},
                         slots_map[fields.geology(validation, DICTIONARY)] if geology else None)

    def batch(self, rng):
        n_regions, _, side, _ = self.data.tensor.shape
        region = rng.integers(n_regions, size=BATCH)
        origins = rng.integers(0, side - self.size + 1, (BATCH, 2))
        stack = self.sampler.gather(self.data.tensor, region, np.tile([0, 1], (BATCH, 1)), origins, rng)
        z, geo = stack[:, 0], stack[:, 1:2]
        sizes = rng.choice(SIZES, size=BATCH)
        m = SPEC.margin
        holes = [tuple((m + rng.integers(CONTEXT, WINDOW - CONTEXT - s + 1, 2)).tolist()) for s in sizes]
        base = z.clone()
        mask = torch.zeros_like(z)
        weight = torch.zeros_like(z)
        for s in SIZES:
            idx = [k for k in range(BATCH) if sizes[k] == s]
            if idx:
                base[idx] = fill_batch(z[idx], [holes[k] for k in idx], s, self.data.g[s])
        for k, (r, c) in enumerate(holes):
            mask[k, r:r + sizes[k], c:c + sizes[k]] = 1.0
            weight[k, r:r + sizes[k], c:c + sizes[k]] = 1.0 / float(sizes[k] ** 2)
        d = distance_map(self.size, self.size, holes, sizes, self.device)
        ex = [mask[:, None], d[:, None] / 64.0] + ([train.one_hot(geo, SLOTS)] if self.geology else [])
        x, sigma = models.inputs(SPEC, base[:, None], None, torch.cat(ex, 1))
        core = (slice(None), slice(None), slice(m, -m), slice(m, -m))
        scale = sigma * (1.0 + d[:, None][core] / DISTANCE_SCALE)
        target = (z[:, None] - base[:, None])[core] / scale
        return x, scale, target, weight[:, None][core]

    def validate(self, model):
        z, chosen, slots = self._val
        maes = []
        for size, blocks in chosen.items():
            fills = neural_fill(model, z, blocks, size, slots, self.device)
            maes.append(np.mean([np.abs(f - z[r:r + size, c:c + size]).mean() for f, (r, c) in zip(fills[0], blocks)]))
        return float(np.mean(maes))


@torch.no_grad()
def neural_fill(model, z: np.ndarray, blocks, size: int, slots, device, chunk: int = 16):
    """Hole values (location, spread) for each block, from a WINDOW window centred on it."""
    model.eval()
    m = SPEC.margin
    side = WINDOW + 2 * m
    locs, spreads = [], []
    padded = np.pad(z, m, mode="reflect")
    padded_slots = None if slots is None else np.pad(slots, m, mode="edge")
    for start in range(0, len(blocks), chunk):
        part = blocks[start:start + chunk]
        windows, offsets, geo = [], [], []
        for r, c in part:
            r0, c0 = r + size // 2 - WINDOW // 2, c + size // 2 - WINDOW // 2
            w = padded[r0:r0 + side, c0:c0 + side].copy()
            hr, hc = r - r0 + m, c - c0 + m
            w[hr:hr + size, hc:hc + size] = np.nan
            windows.append(w)
            offsets.append((hr, hc))
            if padded_slots is not None:
                geo.append(padded_slots[r0:r0 + side, c0:c0 + side])
        win = torch.tensor(np.stack(windows), dtype=torch.float32, device=device)
        base = fill_batch(win, offsets, size, _device_infill(size, device))
        mask = torch.zeros_like(base)
        for k, (r, c) in enumerate(offsets):
            mask[k, r:r + size, c:c + size] = 1.0
        d = distance_map(side, side, offsets, [size] * len(offsets), device)
        ex = [mask[:, None], d[:, None] / 64.0]
        if padded_slots is not None:
            ex.append(train.one_hot(torch.tensor(np.stack(geo), device=device)[:, None].float(), SLOTS))
        x, sigma = models.inputs(SPEC, base[:, None], None, torch.cat(ex, 1))
        scale = sigma * (1.0 + d[:, None, m:-m, m:-m] / DISTANCE_SCALE)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
            loc, log_spread = model(x)
        loc = (loc.float() * scale)[:, 0]
        sp = (torch.exp(log_spread.float().clamp(-7, 6)) * scale)[:, 0]
        core_base = base[:, m:-m, m:-m]
        for k, (r, c) in enumerate(offsets):
            rr, cc = r - m, c - m
            locs.append((core_base[k, rr:rr + size, cc:cc + size] + loc[k, rr:rr + size, cc:cc + size])
                        .double().cpu().numpy())
            spreads.append(sp[k, rr:rr + size, cc:cc + size].double().cpu().numpy())
    return locs, spreads


RINGS = (1, 2, 3, 5, 8, 12, 16, 24, 32)


def neighbourhood(r: int, c: int, size: int) -> np.ndarray:
    """Observed cells on rings 1, 2, 3, 5, 8, ... (Chebyshev distance from the hole, up to the band), each ring
    thinned to an equal share of MAX_POINTS, so near cells are dense and far ones sparse."""
    rings = [d for d in RINGS if d <= BAND[size]]
    quota = MAX_POINTS / len(rings)
    points = []
    for d in rings:
        r0, r1, c0, c1 = r - d, r + size - 1 + d, c - d, c + size - 1 + d
        ring = ([(r0, x) for x in range(c0, c1 + 1)] + [(y, c1) for y in range(r0 + 1, r1 + 1)] +
                [(r1, x) for x in range(c1 - 1, c0 - 1, -1)] + [(y, c0) for y in range(r1 - 1, r0, -1)])
        stride = max(1, int(np.ceil(len(ring) / quota)))
        points += ring[::stride]
    return np.asarray(points, dtype=np.float64)


def kriging_fill(z: np.ndarray, r: int, c: int, size: int, model: dict, device=None):
    """Universal kriging of a hole from its ring neighbourhood; the hole's own values are never read."""
    points = neighbourhood(r, c, size)
    values = z[points[:, 0].astype(int), points[:, 1].astype(int)]
    ty, tx = np.mgrid[r:r + size, c:c + size]
    targets = np.column_stack([ty.ravel(), tx.ravel()]).astype(np.float64)
    centre = points.mean(0)
    mean, var = baselines.universal_kriging(points - centre, values, targets - centre, model,
                                            device=device if size >= 32 else None)
    return mean.reshape(size, size), np.sqrt(var).reshape(size, size)


def train_variants(fold, seeds, device, frozen=None, log=print,
                   max_steps=MAX_STEPS, checkpoints=CHECKPOINTS, lr_grid=LR_GRID,
                   geology_seeds: int | None = None):
    slots_map = fields.fold_classes(fold["train"], DICTIONARY)
    data = Data(fold["train"], slots_map, device)
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
                log(f"{fold['test']} blocks {variant} lr {lr} curve {train.curve(run)}")
                runs.append(run)
            recipe = train.select(runs)
        chosen = seeds if variant == "neural" or not geology_seeds else seeds[:geology_seeds]
        members = train.members(task, runs, recipe, chosen, device, bool(validation), log)
        out["failures"] += [{"variant": variant, "seed": r["seed"], "lr": r["lr"], "reason": r["collapseReason"]}
                            for r in runs + members if r["collapsed"] or r["stalled"]]
        out["variants"][variant] = {"recipe": recipe, "members": [m["model"] for m in members],
                                    "training": train.history(runs, members)}
    return out


def evaluate_region(region, trained, slots_map, kriging_model, device, log=print) -> dict:
    from geoneural.metrics import drainage
    z = fields.reference(region)
    slots = slots_map[fields.geology(region, DICTIONARY)]
    slope = fields.slope_class(z)
    chosen = choose_blocks(region, z)
    routed = drainage.route(z, fields.SPACING_M)
    out = {"region": region, "sizes": {}}
    for size in SIZES:
        start = time.perf_counter()
        blocks = [b[:2] for b in chosen[size]]
        fills = {"harmonic": [], "biharmonic": [], "kriging": [], "krigingExponential": []}
        kriging_sd = {"kriging": [], "krigingExponential": []}
        for r, c in blocks:
            for kind in ("harmonic", "biharmonic"):
                window = z[r - 2:r + size + 2, c - 2:c + size + 2]
                fills[kind].append(baselines.fill_hole(window, 2, 2, size, kind)[2:-2, 2:-2])
            for name, family in (("kriging", "matern32"), ("krigingExponential", "exponential")):
                mean, sd = kriging_fill(z, r, c, size, kriging_model[family], device)
                fills[name].append(mean)
                kriging_sd[name].append(sd)
        spreads = {}
        for variant, members in trained.items():
            variants = [(variant, slots if variant == "neural-geology" else None)]
            if variant == "neural-geology":
                variants.append(("neural-geology-shifted", np.roll(slots, (512, 512), axis=(0, 1))))
            for label, sl in variants:
                locs, sps = [], []
                for k, model in enumerate(members):
                    loc, sp = neural_fill(model, z, blocks, size, sl, device)
                    fills[f"{label}-seed{k}"] = loc
                    locs.append(loc)
                    sps.append(sp)
                fills[f"{label}-ensemble"] = [np.mean([l[i] for l in locs], 0) for i in range(len(blocks))]
                spreads[label] = (locs, sps)
        truth = [z[r:r + size, c:c + size] for r, c in blocks]
        per_block = {name: [float(np.abs(f - t).mean()) for f, t in zip(vals, truth)] for name, vals in fills.items()}
        pooled = {name: evaluate.heights(np.concatenate([f.ravel() for f in vals]),
                                         np.concatenate([t.ravel() for t in truth])) for name, vals in fills.items()}
        rows = {name: {**pooled[name], "blockMaeM": per_block[name]} for name in fills}
        info = [{"origin": [r, c], "stratum": s, "reliefM": rel, "anthropogenicShare": a}
                for r, c, s, rel, a in chosen[size]]
        out["sizes"][str(size)] = {"blocks": info, "methods": rows,
                                   "drainage": block_drainage(z, routed, chosen[size], size, fills, blocks),
                                   "uncertainty": block_uncertainty(truth, spreads, fills, kriging_sd, blocks, size,
                                                                    slope, device),
                                   "seconds": time.perf_counter() - start}
        log(f"{region} blocks {size}: " + ", ".join(f"{k} {v['maeM']:.3f}" for k, v in rows.items() if "seed" not in k))
    return out


def block_drainage(z, routed, chosen, size, fills, blocks) -> dict:
    from geoneural.metrics import drainage
    subset = drainage_subset(chosen, size)
    index = {tuple(b): i for i, b in enumerate(blocks)}
    mask = np.zeros(z.shape, bool)
    for r, c, *_ in subset:
        m = DRAINAGE_MARGIN
        mask[max(r - m, 0):r + size + m, max(c - m, 0):c + size + m] = True
    out = {"blocks": len(subset), "maskCells": int(mask.sum()),
           "noiseFloor": evaluate.masked_floor(z, fields.SPACING_M, mask)}
    for name, vals in fills.items():
        if "seed" in name and not name.endswith("seed0"):
            continue
        est = z.copy()
        for r, c, *_ in subset:
            est[r:r + size, c:c + size] = vals[index[(r, c)]]
        out[name] = evaluate.masked_streams(routed, drainage.route(est, fields.SPACING_M), mask)
    return out


def block_uncertainty(truth, spreads, fills, kriging_sd, blocks, size, slope, device) -> dict:
    """Coverage of neural (single, ensemble) and kriging intervals over hole cells, by slope class."""
    from scipy.stats import norm
    levels = evaluate.quantile_levels()
    t = np.concatenate([x.ravel() for x in truth])
    cls = np.concatenate([slope[r:r + size, c:c + size].ravel() for r, c in blocks])
    out = {}
    for label, (locs, sps) in spreads.items():
        if label.endswith("shifted"):
            continue
        mu = torch.tensor(np.stack([np.concatenate([x.ravel() for x in member]) for member in locs]), device=device)
        sp = torch.tensor(np.stack([np.concatenate([x.ravel() for x in member]) for member in sps]), device=device)
        for name, members in (("single", slice(0, 1)), ("ensemble", slice(None))):
            q = models.mixture_quantiles(mu[members], sp[members], levels)
            quantiles = {lv: q[i].cpu().numpy() for i, lv in enumerate(levels)}
            pit = models.mixture_cdf(mu[members], sp[members], torch.tensor(t, device=device)).cpu().numpy()
            out[f"{label}-{name}"] = {"intervals": evaluate.intervals(t, quantiles, cls),
                                      "pit": evaluate.pit_histogram(pit)}
    for name in ("kriging", "krigingExponential"):
        mean = np.concatenate([x.ravel() for x in fills[name]])
        sd = np.maximum(np.concatenate([x.ravel() for x in kriging_sd[name]]), 1e-9)
        out[name] = {"intervals": evaluate.intervals(t, {lv: mean + norm.ppf(lv) * sd for lv in levels}, cls),
                     "pit": evaluate.pit_histogram(norm.cdf((t - mean) / sd))}
    return out


def fit_kriging(regions) -> dict:
    gamma = baselines.semivariogram([fields.reference(r) for r in regions])
    return {"matern32": baselines.fit_variogram(gamma, "matern32"),
            "exponential": baselines.fit_variogram(gamma, "exponential"), "semivariogram": gamma.tolist()}


def run_fold(fold, seeds, device, models_dir, frozen=None, log=print, max_steps=MAX_STEPS, checkpoints=CHECKPOINTS,
             lr_grid=LR_GRID, geology_seeds: int | None = None):
    start = time.perf_counter()
    trained = train_variants(fold, seeds, device, frozen, log, max_steps, checkpoints, lr_grid, geology_seeds)
    slots_map = np.asarray(trained["slotsMap"])
    kriging_model = fit_kriging(fold["train"])
    log(f"kriging models {kriging_model['matern32']} {kriging_model['exponential']}")
    folder = models_dir / "-".join(fold["test"])
    folder.mkdir(parents=True, exist_ok=True)
    for variant, info in trained["variants"].items():
        for k, model in enumerate(info["members"]):
            torch.save(model.state_dict(), folder / f"{variant}-seed{k}.pt")
    members = {v: info["members"] for v, info in trained["variants"].items()}
    regions = {r: evaluate_region(r, members, slots_map, kriging_model, device, log) for r in fold["test"]}
    return {"fold": fold, "seeds": list(seeds), "slotsMap": trained["slotsMap"], "learningRateGrid": list(lr_grid),
            "geologySeeds": geology_seeds or len(seeds),
            "recipes": {v: info["recipe"] for v, info in trained["variants"].items()},
            "training": {v: info["training"] for v, info in trained["variants"].items()},
            "failures": trained["failures"], "kriging": kriging_model, "regions": regions,
            "seconds": time.perf_counter() - start}


DEVELOPMENT_FAILURES = [
    {"stage": "development", "run": "first fold (test essen-ruhr), learning rate 2e-4, 3000 steps",
     "reason": "targets divided by the local scale alone were about 10 (90th percentile 30) inside 128-cell holes; "
               "validation MAE moved from 0.3698 m (biharmonic) to 0.3683 m in 3000 steps, no useful learning. "
               "Stopped and replaced by the distance-scaled target; no result of that run is reported."},
    {"stage": "development", "run": "first fold, distance-scaled target, three-level U-Net, rates 2e-4 and 5e-4",
     "reason": "validation MAE 0.3711 -> 0.3715 m (2e-4) and 0.3711 -> 0.3675 m (5e-4) against 0.3698 m for the "
               "biharmonic fill in 3000 steps; both runs were flagged as not learning at step 500 and the then "
               "rule refused to select, ending the process. The three-level receptive field (about 45 cells) "
               "does not reach known terrain from the centre of a 128-cell hole; replaced by five levels, and "
               "stalled runs now stay selectable. No result of these runs is reported."},
    {"stage": "budget", "run": "all folds",
     "reason": "on the shared GPU a five-level step took about 0.4 s, so the grid was cut to one learning rate "
               "(4e-4, the rate that trained the prototype's hole model), at most 1500 steps (checkpoints 500, "
               "1000, 1500, selected on the validation region) and one seed per variant. Uncertainty for this "
               "task therefore rests on the Laplace spread head; the -ensemble rows equal the single model."}]
COMPARISONS = (("neural", "biharmonic"), ("neural", "kriging"), ("neural", "harmonic"), ("kriging", "biharmonic"),
               ("kriging", "krigingExponential"),
               ("neural-geology", "neural"), ("neural-geology", "neural-geology-shifted"),
               ("neural-ensemble", "neural"), ("neural-geology", "biharmonic"))
NOTES = ("Task 3, missing blocks. Leave one region out over the six development regions (validation region "
         "next in the list). 200 blocks per size per region, stratified by relief; every block is an independent "
         "fill from its surroundings. Errors are over hole cells in metres; blockMaeM is per block. Neural rows "
         "are means over seeds unless named -ensemble. Paired differences are negative when the first method "
         "is better; within a region the bootstrap unit is the block (optimistic, blocks of 128 overlap), over "
         "regions the unit is the region. Drainage is read within 16 cells of a non-overlapping block subset "
         "with the masked noise floor beside it.")


def selection_at_cap(parts) -> dict:
    """Folds whose selected step is the largest checkpoint: the rule may have been limited by the cap."""
    return {"-".join(p["fold"]["test"]): {v: r["steps"] for v, r in p["recipes"].items()
                                          if isinstance(r, dict) and r.get("steps") == max(CHECKPOINTS)}
            for p in parts}


def summarise(parts):
    per_size: dict = {}
    strata: dict = {}
    drainage: dict = {}
    uncertainty: dict = {}
    recipes, training, failures, kriging = {}, {}, [], {}
    for part in parts:
        key = "-".join(part["fold"]["test"])
        recipes[key] = {"fold": part["fold"], "recipes": part["recipes"]}
        training[key] = part["training"]
        kriging[key] = {k: v for k, v in part["kriging"].items() if k != "semivariogram"}
        failures += [dict(f, fold=key) for f in part["failures"]]
        for region, result in part["regions"].items():
            for size, block in result["sizes"].items():
                methods = evaluate.merge_seeds(block["methods"])
                per_size.setdefault(size, {})[region] = methods
                stratum = np.array([b["stratum"] for b in block["blocks"]])
                anthropogenic = np.array([b["anthropogenicShare"] for b in block["blocks"]]) > 0.25
                strata.setdefault(size, {})[region] = {
                    m: {**{f"relief{s}": float(np.mean(np.asarray(row["blockMaeM"])[stratum == s]))
                           for s in range(3) if (stratum == s).any()},
                        "anthropogenic": float(np.mean(np.asarray(row["blockMaeM"])[anthropogenic]))
                        if anthropogenic.any() else None,
                        "anthropogenicBlocks": int(anthropogenic.sum())}
                    for m, row in methods.items() if "seed" not in m}
                drainage.setdefault(size, {})[region] = block["drainage"]
                uncertainty.setdefault(size, {})[region] = block["uncertainty"]
    tables = {size: {r: {m: {k: row[k] for k in ("maeM", "rmseM", "p99M", "maxM", "biasM", "seedSdMaeM")
                             if k in row} | {"blockBalancedMaeM": float(np.mean(row["blockMaeM"]))}
                         for m, row in methods.items() if "seed" not in m or m.endswith("seed0")}
                     for r, methods in regions.items()} for size, regions in per_size.items()}
    balanced = {size: {m: evaluate.region_balanced({r: tables[size][r][m]["maeM"] for r in tables[size]})
                       for m in next(iter(tables[size].values()))} for size in tables}
    pairs = {size: [evaluate.paired(per_size[size], a, b, unit="blockMaeM") for a, b in COMPARISONS
                    if all(a in m and b in m for m in per_size[size].values())] for size in per_size}
    drain_bal = {size: {m: {k: evaluate.region_balanced({r: (d.get(m) or {}).get(k) for r, d in regions.items()})
                            for k in ("jaccard", "tolerantF1", "receiverAgreement", "outletAgreement",
                                      "fillVolumeDifferenceM3")}
                        for m in next(iter(regions.values())) if m not in ("blocks", "maskCells", "noiseFloor")}
                 | {"noiseFloor": {k: evaluate.region_balanced({r: d["noiseFloor"].get(k) for r, d in regions.items()})
                                   for k in ("jaccard", "tolerantF1", "receiverAgreement", "outletAgreement")}}
                 for size, regions in drainage.items()}
    calib = {size: {label: {lv: {"coverage": evaluate.region_balanced(
        {r: u[label]["intervals"]["overall"][lv]["coverage"] for r, u in regions.items()}),
        "meanWidthM": evaluate.region_balanced(
        {r: u[label]["intervals"]["overall"][lv]["meanWidthM"] for r, u in regions.items()})}
        for lv in map(str, evaluate.LEVELS)} for label in next(iter(regions.values()))}
        for size, regions in uncertainty.items()}
    return {"perRegion": tables, "regionBalancedMaeM": balanced, "paired": pairs, "byStratum": strata,
            "drainage": {"byRegion": drainage, "regionBalanced": drain_bal},
            "uncertainty": {"byRegion": uncertainty, "regionBalanced": calib},
            "recipes": recipes, "selectionAtCap": selection_at_cap(parts), "training": training,
            "kriging": kriging}, failures + DEVELOPMENT_FAILURES


def describe(parts):
    recipe = {"task": "missing blocks", "sizes": list(SIZES), "blocksPerSize": BLOCKS_PER_SIZE,
              "blockSeed": BLOCK_SEED, "stratification": "thirds of local relief (range over the block plus 8 "
              "cells) among candidate positions of the region", "landUseProxy": "GK100 anthropogenic material share",
              "window": WINDOW, "features": SPEC.record(), "context": CONTEXT,
              "model": {"levels": LEVELS, "type": "U-Net width 32 over the biharmonic fill, inputs plus hole mask "
                                                  "and distance to the nearest known cell (plus geology one-hot)",
                        "batch": BATCH,
                        "targetScale": f"local sigma * (1 + d / {DISTANCE_SCALE}), d = Chebyshev distance to the "
                                       "nearest known cell", "trainingHoles": "one per window, size uniform over "
                                "8/32/128, at least 32 cells of context", "loss": "per-block mean L1 plus Laplace NLL"},
              "selection": {"learningRates": {"-".join(p["fold"]["test"]): p.get("learningRateGrid", list(LR_GRID))
                                              for p in parts},
                            "checkpoints": sorted({e["step"] for p in parts for runs in p["training"].values()
                                                   for r in runs for e in r["curve"]}),
                            "rule": "lowest validation MAE over 20 blocks per size in the validation region"},
              "baselines": {"harmonic": "Laplace equation, first ring", "biharmonic": "squared Laplacian, two rings",
                            "kriging": f"universal kriging, linear drift, Matern 3/2 variogram fitted on training "
                                       f"regions, local sill rescaled to the neighbourhood, band {BAND} cells, at "
                                       f"most {MAX_POINTS} points",
                            "krigingExponential": "the same with an exponential variogram"},
              "geology": {"dictionary": DICTIONARY, "shiftedControl": [512, 512]}}
    return recipe, {"folds": [p["fold"] for p in parts], "rule": "test region k, validation k+1, train the rest"}


def gates(results):
    from geoneural.evaluation import protocol
    out = {}
    for size in map(str, SIZES):
        bal = results["regionBalancedMaeM"][size]
        best = min(("biharmonic", "kriging", "krigingExponential", "harmonic"), key=bal.get)
        pair = next(p for p in results["paired"][size] if p["comparison"] == f"neural minus {best}")
        gain = 1 - bal["neural"] / bal[best]
        out[f"neuralBeatsBest{size}"] = protocol.gate(
            "pass" if gain >= 0.02 and pair["overRegions"]["high"] < 0 else "fail",
            f"neural {bal['neural']:.3f} m against {best} {bal[best]:.3f} m (region-balanced MAE over hole cells, "
            f"gain {gain:.1%})", minimumGain=0.02, intervalExcludesZero=True)
    return out


def frozen_recipe(parts):
    out = {}
    for variant in ("neural", "neural-geology"):
        chosen = [p["recipes"][variant] for p in parts]
        rates = [c["lr"] for c in chosen]
        lr = min(set(rates), key=lambda r: (-rates.count(r), r))
        steps = int(np.median([c["steps"] for c in chosen]))
        out[variant] = {"lr": lr, "steps": min(CHECKPOINTS, key=lambda c: (abs(c - steps), c))}
    return {"task": "blocks", "recipes": out, "fromFolds": [p["fold"] for p in parts],
            "rule": "most frequently selected learning rate, median selected step on the checkpoint grid"}
