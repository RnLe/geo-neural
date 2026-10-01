"""Training, selection and whole-field inference for the reconstruction model.

Selection rule, declared before any test region is scored: for the first seed, every learning rate in the grid
is trained to the largest step count, the exponential moving average of the weights is scored on the validation
region at every checkpoint, and the (learning rate, step) pair with the lowest validation MAE wins; ties go to
fewer steps. Later seeds and the variant with geology reuse that learning rate and stop at their own selected
step (the variant with geology selects its step on its own first-seed curve). A confirmation run uses a frozen
(learning rate, steps) pair and no validation.

A run that fails is kept, not hidden: `collapsed` is set when the loss stops being finite (the run ends there and
cannot be selected), `stalled` when the training L1 relative to predicting zero residual is not below
`STALL_RATIO` by the first checkpoint (the run stays selectable; the validation curve decides).
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn.functional as F

from geoneural.recon import models

STALL_RATIO = 0.98


def fit(task, *, lr: float, steps: int, checkpoints, seed: int, device: str = "cuda", width: int = 32,
        ema_decay: float = 0.998, warmup: int = 100, log_every: int = 250, validate: bool = True) -> dict:
    """Train one model on `task` (see the task modules for `batch`, `channels` and `validate`)."""
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    rng = np.random.default_rng(seed)
    model = models.UNet(task.channels, width, getattr(task, "levels", 3)).to(device)
    average = copy.deepcopy(model).eval()
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    checkpoints = sorted({int(c) for c in checkpoints if c <= steps} | {int(steps)})
    curve, log, best = [], [], None
    collapsed, stalled, reason = False, False, ""
    running, running_ref, count = 0.0, 0.0, 0
    torch.cuda.reset_peak_memory_stats() if device.startswith("cuda") else None
    start = time.perf_counter()
    for step in range(1, steps + 1):
        for group in optimiser.param_groups:
            group["lr"] = lr * min(1.0, step / warmup)
        x, sigma, target, weight = task.batch(rng)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            location, spread = model(x)
        l1, nll = models.laplace_loss(location.float(), spread.float(), target, weight)
        loss = l1 + nll
        if not torch.isfinite(loss):
            collapsed, reason = True, f"non-finite loss at step {step}"
            break
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        decay = min(ema_decay, (1.0 + step) / (10.0 + step))
        with torch.no_grad():
            for a, p in zip(average.parameters(), model.parameters()):
                a.mul_(decay).add_(p.detach(), alpha=1.0 - decay)
        running += l1.item()
        running_ref += float((target.abs() * weight).sum() / weight.sum().clamp_min(1.0))
        count += 1
        if step % log_every == 0:
            log.append({"step": step, "l1": running / count, "relativeL1": running / max(running_ref, 1e-12),
                        "nll": nll.item(), "seconds": time.perf_counter() - start})
            running, running_ref, count = 0.0, 0.0, 0
        if step in checkpoints:
            if step == checkpoints[0] and log and log[-1]["relativeL1"] > STALL_RATIO:
                stalled, reason = True, (f"relative L1 {log[-1]['relativeL1']:.3f} at step {step}: barely "
                                         "learning beyond the base")
            entry = {"step": step}
            if validate and getattr(task, "validation", None):
                entry["validationMaeM"] = float(task.validate(average))
                if best is None or entry["validationMaeM"] < best[0]:
                    best = (entry["validationMaeM"], step, copy.deepcopy(average.state_dict()))
            curve.append(entry)
    seconds = time.perf_counter() - start
    if best is not None:
        average.load_state_dict(best[2])
    peak = torch.cuda.max_memory_allocated() / 1e9 if device.startswith("cuda") else None
    return {"model": average, "curve": curve, "log": log, "seconds": seconds, "peakGpuGB": peak,
            "selectedStep": best[1] if best else steps, "selectedValidationMaeM": best[0] if best else None,
            "collapsed": collapsed, "stalled": stalled, "collapseReason": reason, "lr": lr, "seed": seed,
            "steps": steps, "levels": getattr(task, "levels", 3),
            "parameters": models.parameters(model)}


def select(runs: list[dict]) -> dict:
    """The declared rule: lowest validation MAE over (lr, checkpoint), ties to fewer steps then lower lr."""
    candidates = [(e["validationMaeM"], e["step"], r["lr"]) for r in runs if not r["collapsed"]
                  for e in r["curve"] if "validationMaeM" in e]
    if not candidates:
        raise RuntimeError("every candidate run collapsed; nothing to select")
    mae, step, lr = min(candidates)
    return {"lr": lr, "steps": step, "validationMaeM": mae}


def members(task, runs: list[dict], recipe: dict, seeds, device: str, validate: bool, log=None) -> list[dict]:
    """One model per seed at the selected recipe, reusing a selection run where it already is that model."""
    out = []
    for seed in seeds:
        reuse = [r for r in runs
                 if r["seed"] == seed and r["lr"] == recipe["lr"] and r["selectedStep"] == recipe["steps"]]
        if reuse:
            out.append(reuse[0])
            continue
        out.append(fit(task, lr=recipe["lr"], steps=recipe["steps"], checkpoints=(recipe["steps"],), seed=seed,
                       device=device, validate=validate))
        if log:
            log(f"seed {seed} steps {recipe['steps']} {out[-1]['seconds']:.0f}s")
    return out


def curve(run: dict) -> str:
    return f"{[(e['step'], round(e.get('validationMaeM', 0), 4)) for e in run['curve']]} {run['seconds']:.0f}s"


TRAINING_KEYS = ("lr", "seed", "steps", "selectedStep", "selectedValidationMaeM", "curve", "log", "seconds",
                 "peakGpuGB", "collapsed", "stalled", "collapseReason", "parameters", "levels")


def history(runs: list[dict], chosen: list[dict]) -> list[dict]:
    return [{k: r[k] for k in TRAINING_KEYS} for r in runs + [m for m in chosen if m not in runs]]


@torch.no_grad()
def predict(model, spec: models.Features, base: np.ndarray, extras: np.ndarray | None = None,
            tile: int = 256, halo: int = 64, device: str = "cuda", batch: int = 4):
    """Whole-field residual location and spread in metres, by tiles with a halo.

    The base and extras are mirror-padded once and every tile is a window of that one padded field, with the
    margin the stencils need, so the network reads exactly what a training window cut from the same place
    would hold. Returns (location, spread) as float64 arrays shaped like `base`.
    """
    model.eval()
    rows, columns = base.shape
    pad = spec.margin + halo
    n_r, n_c = -(-rows // tile), -(-columns // tile)
    extra_r, extra_c = n_r * tile - rows, n_c * tile - columns
    widths = ((pad, pad + extra_r), (pad, pad + extra_c))
    padded = torch.from_numpy(np.pad(np.asarray(base, dtype=np.float32), widths, mode="reflect"))
    padded_extras = None if extras is None else torch.from_numpy(
        np.pad(np.asarray(extras, dtype=np.float32), ((0, 0),) + widths, mode="reflect"))
    side = tile + 2 * pad
    location = np.zeros((rows, columns))
    spread = np.zeros((rows, columns))
    jobs = [(i, j) for i in range(n_r) for j in range(n_c)]
    for start in range(0, len(jobs), batch):
        part = jobs[start:start + batch]
        windows = torch.stack([padded[i * tile:i * tile + side, j * tile:j * tile + side]
                               for i, j in part])[:, None].to(device)
        phases = None
        if spec.factor:
            phases = torch.stack([models.phase_maps(i * tile - pad, j * tile - pad, side, side, spec.factor, device)
                                  for i, j in part])
        ex = None if padded_extras is None else torch.stack(
            [padded_extras[:, i * tile:i * tile + side, j * tile:j * tile + side] for i, j in part]).to(device)
        x, sigma = models.inputs(spec, windows, phases, ex)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            loc, log_spread = model(x)
        loc = (loc.float() * sigma)[:, 0, halo:-halo, halo:-halo].double().cpu().numpy()
        sp = torch.exp(log_spread.float().clamp(-7.0, 6.0)) * sigma
        sp = sp[:, 0, halo:-halo, halo:-halo].double().cpu().numpy()
        for k, (i, j) in enumerate(part):
            r1, c1 = min((i + 1) * tile, rows), min((j + 1) * tile, columns)
            location[i * tile:r1, j * tile:c1] = loc[k, :r1 - i * tile, :c1 - j * tile]
            spread[i * tile:r1, j * tile:c1] = sp[k, :r1 - i * tile, :c1 - j * tile]
    return location, spread


def one_hot(classes: torch.Tensor, slots: int) -> torch.Tensor:
    """(N,1,H,W) integer classes -> (N,slots,H,W) float one-hot; slot 0 is unknown."""
    return F.one_hot(classes[:, 0].long(), slots).permute(0, 3, 1, 2).float()
