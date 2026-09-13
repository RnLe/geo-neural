"""One training loop, shared by the CLI and the hyperparameter search.

Comparing architectures under one fixed recipe (say Adam at 1e-4 with no
schedule) measures the pairing of family and recipe, not the architecture: a
family that would win at 3e-3 with cosine decay loses to one that happens to
suit 1e-4. The recipe is therefore parameterised here and tuned per family. One
loop serves both the command line and the search so that their results stay
comparable.

Three things in here are guards rather than features:

The slope term only uses node pairs where both nodes are inside the training
mask. A finite difference reaching one node outside it would read a held-out
height and train on the test set, which inflates holdout scores.

`evaluate` chunks and runs under `inference_mode`, and converts to float64 before
taking errors, because a float32 mean over 190k residuals loses digits where the
small numbers matter.

Deployed bytes are the actual serialized safetensors length, not a parameter
count times four. The difference is small (640 bytes on a width-128, depth-3
SIREN), but it is the number a deployment pays.
"""
from __future__ import annotations
import math
import time
from dataclasses import dataclass, asdict, replace

import numpy as np

SCHEMA = "geoneural-training-v1"
EVAL_CHUNK = 262_144


@dataclass(frozen=True)
class Recipe:
    """Everything about how a model is trained, separate from what it is."""
    steps: int = 6000
    batch: int = 8192
    lr: float = 1e-3
    schedule: str = "cosine"          # 'none' | 'cosine'
    warmup: int = 0
    weight_decay: float = 0.0
    loss: str = "mse"                 # 'mse' | 'huber' | 'l1'
    huber_delta: float = 0.05         # normalised units
    slope_weight: float = 0.0
    multiscale_weight: float = 0.0
    seed: int = 1729
    device: str = "cuda"
    eval_every: int = 0               # 0 disables intermediate evaluation
    # Where the lattice lives during training, and how a batch is drawn from it.
    # These are recipe fields rather than call arguments because they change the
    # random stream and therefore the result, so a run's record must state them.
    # "host" is the original path and stays the default so every stored recipe
    # replays exactly. See `device_data` for why the device path exists and why
    # with-replacement sampling is inside the seed spread.
    data_path: str = "host"           # 'host' | 'device'
    sampling: str = "without"         # 'without' | 'with' | 'reshuffle'

    def as_dict(self) -> dict:
        return asdict(self)


def _schedule(recipe: Recipe, step: int) -> float:
    if step < recipe.warmup:
        return recipe.lr * (step + 1) / max(recipe.warmup, 1)
    if recipe.schedule == "cosine":
        span = max(recipe.steps - recipe.warmup, 1)
        progress = min((step - recipe.warmup) / span, 1.0)
        return recipe.lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    return recipe.lr


def _objective(predicted, truth, recipe: Recipe):
    import torch
    if recipe.loss == "mse":
        return (predicted - truth).square().mean()
    if recipe.loss == "l1":
        return (predicted - truth).abs().mean()
    if recipe.loss == "huber":
        return torch.nn.functional.huber_loss(predicted, truth, delta=recipe.huber_delta)
    raise ValueError(f"Unsupported loss: {recipe.loss}")


def training_pairs(train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Neighbour pairs with both ends inside the training region.

    A slope penalty needs two heights. Taking the second from wherever the
    lattice happens to go would read held-out nodes, so pairs that leave the
    training mask are dropped rather than clipped.
    """
    side = train_mask.shape[0]
    flat = np.arange(train_mask.size, dtype=np.int64).reshape(train_mask.shape)
    east = train_mask[:, :-1] & train_mask[:, 1:]
    south = train_mask[:-1, :] & train_mask[1:, :]
    a = np.concatenate([flat[:, :-1][east], flat[:-1, :][south]])
    b = np.concatenate([flat[:, 1:][east], flat[1:, :][south]])
    return a, b


def evaluate(model, features_fn, indexes: np.ndarray, reference_flat: np.ndarray,
             side: int, intervals: int, shared: bool, mean: float, scale: float,
             device: str, torch, limit: int | None = None,
             rng: np.random.Generator | None = None) -> dict:
    """Physical-unit errors on an index set, with the maximum kept.

    MAE is reported alongside p99 and max and never alone: a lower mean that
    breaks one ridge is a rejection.
    """
    if indexes.size == 0:
        return {"samples": 0}
    if limit is not None and indexes.size > limit:
        picked = (rng or np.random.default_rng(0)).choice(indexes, limit, replace=False)
    else:
        picked = indexes
    was_training = model.training
    model.eval()
    errors = np.empty(picked.size, dtype=np.float64)
    with torch.inference_mode():
        for begin in range(0, picked.size, EVAL_CHUNK):
            block = picked[begin:begin + EVAL_CHUNK]
            coords, tiles = features_fn(block, side, intervals, shared)
            out = model(torch.from_numpy(coords).to(device), torch.from_numpy(tiles).to(device))
            predicted = out.squeeze(-1).detach().to(torch.float32).cpu().numpy().astype(np.float64)
            errors[begin:begin + block.size] = np.abs(
                predicted * scale + mean - reference_flat[block].astype(np.float64))
    if was_training:
        model.train()
    # A model can finish training with a finite loss and still produce non-finite
    # predictions: the loss check inside `fit` only looks at reporting steps and
    # only at the training batches. Returning NaN here would put NaN into a study
    # objective and onto a Pareto front, where it is an absent result rather than
    # a bad one. Raising makes it a pruned trial that records its reason.
    if not np.all(np.isfinite(errors)):
        raise FloatingPointError(
            f"{int((~np.isfinite(errors)).sum())} of {errors.size} predictions were not finite")
    return {
        "samples": int(picked.size),
        "mae_m": float(errors.mean()),
        "rmse_m": float(np.sqrt(np.mean(errors ** 2))),
        "p95_m": float(np.quantile(errors, 0.95)),
        "p99_m": float(np.quantile(errors, 0.99)),
        "max_m": float(errors.max()),
    }


# What a deployment may ship stored tensors as. float16 is the default because the
# precision ladder measured it: it halves the rate for about 0.002 m of mean
# error, while bfloat16 costs the same bytes for 67x that error and every float8
# form destroys the model. Widening to float64 is the control that shows the
# ladder measures storage at all. The ladder itself lives in
# `search.precision_sweep`; this is the subset a search is allowed to price at.
STORAGE_PRECISIONS = ("float64", "float32", "float16")


def store_state(model, torch, precision: str = "float32") -> dict:
    """The tensors a deployment ships, at the width it ships them."""
    if precision not in STORAGE_PRECISIONS:
        raise ValueError(f"Unsupported storage precision: {precision}")
    dtype = getattr(torch, precision)
    return {k: (v.detach().cpu().contiguous().to(dtype) if v.is_floating_point()
                else v.detach().cpu().contiguous())
            for k, v in model.state_dict().items()}


def deployed_bytes(model, torch, precision: str = "float32") -> int:
    """Bytes a deployment ships: the serialized tensors, framing included."""
    from safetensors.torch import save
    return len(save(store_state(model, torch, precision)))


def tensor_bytes(model, torch, precision: str = "float32") -> int:
    """The stored tensors alone, without the container that names and shapes them.

    Reported beside `deployed_bytes` because the difference is the one accounting
    asymmetry that runs against the neural side: a neural row is charged its
    safetensors header, while the conventional index (which names and locates 256
    pages) is charged to neither side. The header is 448 to 968 bytes depending
    on tensor count, under one per cent of a deployable model and about forty per
    cent of a 1,122-byte one, so it matters only at the very cheap end of a
    front. A compact fixed-schema container would shrink it, so it is a
    deployment choice rather than an intrinsic cost.
    """
    dtype = getattr(torch, precision)
    return sum(v.numel() * dtype.itemsize if v.is_floating_point() else v.numel() * v.dtype.itemsize
               for v in model.state_dict().values())


def round_to_storage(model, torch, precision: str) -> dict:
    """Round every stored tensor through `precision` in place; return the original.

    Deployment ships weights at the stored width and widens them on load, so a
    model priced at that width has to be measured at it too. Charging float16
    bytes while evaluating float32 weights would understate the error at the
    quoted size.

    Activations are untouched. Nothing here is a runtime or resident-memory claim.
    """
    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    dtype = getattr(torch, precision)
    with torch.no_grad():
        for tensor in model.state_dict().values():
            if tensor.is_floating_point():
                tensor.copy_(tensor.to(dtype).to(tensor.dtype))
    return original


def fit(model, features_fn, reference_flat: np.ndarray, side: int, intervals: int,
        shared: bool, train_indexes: np.ndarray, mean: float, scale: float,
        recipe: Recipe, torch, train_mask: np.ndarray | None = None,
        level_samplers=None, on_report=None, checkpoint_path=None,
        checkpoint_every: int = 0, resume: dict | None = None, tables=None,
        engine_mode: str = "eager") -> dict:
    """Train in place and return a history. Nothing is written to disk here.

    `on_report(step, record)` may raise to abandon a run; the search uses that
    for pruning, and an abandoned run leaves the model at its current weights
    rather than losing them.
    """
    if recipe.data_path == "device":
        return _fit_device(model, features_fn, reference_flat, side, intervals, shared,
                           train_indexes, mean, scale, recipe, torch, train_mask,
                           level_samplers, on_report, checkpoint_path, checkpoint_every,
                           resume, tables, engine_mode)
    if recipe.data_path != "host":
        raise ValueError(f"Unknown data path: {recipe.data_path}")
    device = recipe.device
    torch.manual_seed(recipe.seed)
    # One stream per purpose. If the slope term drew from the batch stream,
    # switching it on would also change every batch, and the ablation would
    # measure two things at once.
    rng = np.random.default_rng(recipe.seed)
    slope_rng = np.random.default_rng(recipe.seed ^ 0x51073)
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=recipe.lr,
                                 weight_decay=recipe.weight_decay)
    first_step = 0
    resumed_history: list[dict] = []
    if resume is not None:
        # A resumed run must be indistinguishable from the uninterrupted one, so
        # the batch stream, the slope stream, the torch stream and Adam's moments
        # all come back before the first step is taken.
        for field in ("batch", "loss", "slope_weight", "seed", "data_path", "sampling"):
            was = resume.get("recipe", {}).get(field)
            if was is not None and was != getattr(recipe, field):
                raise ValueError(
                    f"checkpoint was written with {field}={was!r}, resuming with "
                    f"{getattr(recipe, field)!r}: that is a different experiment")
        first_step = restore_state(resume, model, optimizer, rng, slope_rng, torch) + 1
        resumed_history = list(resume.get("history", []))

    pair_a = pair_b = None
    if recipe.slope_weight > 0.0:
        if train_mask is None:
            raise ValueError("a slope term needs the training mask to keep its pairs inside it")
        pair_a, pair_b = training_pairs(train_mask)
        if pair_a.size == 0:
            raise ValueError("no neighbour pair lies wholly inside the training region")

    history: list[dict] = list(resumed_history)
    start = time.perf_counter()
    for step in range(first_step, recipe.steps):
        for group in optimizer.param_groups:
            group["lr"] = _schedule(recipe, step)
        indexes = rng.choice(train_indexes, recipe.batch, replace=False)
        coords, tiles = features_fn(indexes, side, intervals, shared)
        coords_t = torch.from_numpy(coords).to(device)
        tiles_t = torch.from_numpy(tiles).to(device)
        truth = torch.from_numpy(((reference_flat[indexes] - mean) / scale).astype(np.float32)).to(device)
        predicted = model(coords_t, tiles_t).squeeze(-1)
        loss = _objective(predicted, truth, recipe)

        if recipe.multiscale_weight > 0.0 and level_samplers:
            # Each band-limited head is trained against the atlas level whose
            # bandwidth it is allowed to carry, so a coarse query is answerable
            # from the early layers alone rather than by truncating a fine field.
            outputs = model.all_levels(coords_t)
            extra = 0.0
            for head, sampler in zip(outputs, level_samplers):
                target = torch.from_numpy(((sampler(indexes) - mean) / scale).astype(np.float32)).to(device)
                extra = extra + _objective(head.squeeze(-1), target, recipe)
            loss = loss + recipe.multiscale_weight * extra / max(len(level_samplers), 1)

        if recipe.slope_weight > 0.0:
            picked = slope_rng.choice(pair_a.size, min(recipe.batch, pair_a.size), replace=False)
            left, right = pair_a[picked], pair_b[picked]
            ca, ta = features_fn(left, side, intervals, shared)
            cb, tb = features_fn(right, side, intervals, shared)
            pa = model(torch.from_numpy(ca).to(device), torch.from_numpy(ta).to(device)).squeeze(-1)
            pb = model(torch.from_numpy(cb).to(device), torch.from_numpy(tb).to(device)).squeeze(-1)
            observed = torch.from_numpy(
                ((reference_flat[right] - reference_flat[left]) / scale).astype(np.float32)).to(device)
            loss = loss + recipe.slope_weight * (pb - pa - observed).square().mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        report_now = recipe.eval_every > 0 and (step % recipe.eval_every == 0 or step == recipe.steps - 1)
        if report_now or step % max(recipe.steps // 20, 1) == 0 or step == recipe.steps - 1:
            value = float(loss.detach().cpu())
            if not math.isfinite(value):
                raise FloatingPointError(f"training loss diverged at step {step}")
            record = {"step": step, "loss": value,
                      "batch_rmse_m": float(math.sqrt(max(value, 0.0))) * scale if recipe.loss == "mse" else None,
                      "lr": _schedule(recipe, step),
                      "elapsed_s": time.perf_counter() - start}
            history.append(record)
            if on_report is not None and report_now:
                on_report(step, record)
        if checkpoint_every > 0 and checkpoint_path is not None \
                and (step + 1) % checkpoint_every == 0 and step + 1 < recipe.steps:
            save_checkpoint(checkpoint_path,
                            capture_state(model, optimizer, rng, slope_rng, step, history,
                                          recipe, torch), torch)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    return {"history": history, "seconds": time.perf_counter() - start,
            "recipe": recipe.as_dict(), "resumedFromStep": first_step or None}


def capture_state(model, optimizer, rng, slope_rng, step: int, history: list,
                  recipe: Recipe, torch) -> dict:
    """Everything needed to resume training bit-identically from `step`.

    Weights alone are not training state. Adam carries first and second moment
    estimates per parameter, the schedule is a function of the step counter, and
    three independent random streams choose the batches, the slope pairs and the
    initialisation. Restoring only `state_dict()` and calling `fit` again restarts
    the optimiser cold, rewinds the learning rate to warmup, and redraws the same
    batches the first run already used. That is not a continuation but a
    differently initialised second run that happens to start from good weights.

    The two numpy streams are stored as their bit-generator state rather than
    their seed, because a seed only reproduces a stream from its beginning.

    Checkpoints are research bytes. `accounting.package_bytes` classifies them
    with optimizer state and the reference raster, never into deployed size.
    """
    state = {"schema": "geoneural-training-checkpoint-v1",
             "step": int(step),
             "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
             "optimizer": optimizer.state_dict(),
             "batch_rng": rng.bit_generator.state,
             "slope_rng": slope_rng.bit_generator.state,
             "torch_rng": torch.get_rng_state(),
             "history": list(history),
             "recipe": recipe.as_dict()}
    if torch.cuda.is_available():
        state["cuda_rng"] = torch.cuda.get_rng_state_all()
    return state


def restore_state(state: dict, model, optimizer, rng, slope_rng, torch) -> int:
    """Put a captured state back and return the step to resume at.

    The recipe is not restored from the checkpoint: a caller may legitimately
    resume with more steps than the original run planned. It is returned in the
    state for provenance, and `fit` checks the parts that would silently change
    the meaning of a resumed run.
    """
    if state.get("schema") != "geoneural-training-checkpoint-v1":
        raise ValueError(f"Unrecognised checkpoint schema: {state.get('schema')!r}")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    rng.bit_generator.state = state["batch_rng"]
    slope_rng.bit_generator.state = state["slope_rng"]
    torch.set_rng_state(state["torch_rng"].to("cpu", torch.uint8)
                        if hasattr(state["torch_rng"], "to") else state["torch_rng"])
    if "cuda_rng" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return int(state["step"])


def save_checkpoint(path, state: dict, torch) -> int:
    """Write a checkpoint and return its size. Atomic: a killed run never leaves
    a half-written checkpoint that would fail to load on the retry it exists for."""
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(state, temporary)
    temporary.replace(path)
    return path.stat().st_size


def load_checkpoint(path, torch) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def with_overrides(recipe: Recipe, **changes) -> Recipe:
    return replace(recipe, **changes)


def _fit_device(model, features_fn, reference_flat, side, intervals, shared,
                train_indexes, mean, scale, recipe: Recipe, torch,
                train_mask=None, level_samplers=None, on_report=None,
                checkpoint_path=None, checkpoint_every: int = 0,
                resume: dict | None = None, tables=None,
                engine_mode: str = "eager") -> dict:
    """`fit` with the lattice resident on the device and no per-step host traffic.

    Same optimiser, same schedule, same loss, same reporting cadence, same
    divergence check. What differs is that the batch indices are drawn by a
    device generator and the features are gathered rather than rebuilt, so the
    three synchronous copies per step are gone and the CPU leaves the critical
    path. See `device_data` for the measurement and for why this changes the
    study identity rather than being a pure optimisation.

    Adam is `fused=True` here. It is a different kernel from the foreach default
    and therefore not bit-equal to the host path. The path as a whole is not
    bit-equal either, which is why it is checked by the equivalence protocol
    rather than by a bitwise test.
    """
    from geoneural.neural import device_data

    if level_samplers and recipe.multiscale_weight > 0.0:
        # The band-limited multiscale term samples atlas levels on the host. No
        # search uses it, so it has not been ported; refusing avoids an untested
        # device path.
        raise ValueError("the multiscale term has no device path; run it with data_path='host'")

    device = recipe.device
    torch.manual_seed(recipe.seed)
    if tables is None:
        raise ValueError("the device path needs DeviceTables; pass tables=")
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=recipe.lr,
                                 weight_decay=recipe.weight_decay, fused=True)

    pool = torch.from_numpy(np.asarray(train_indexes, dtype=np.int64)).to(device)
    batcher = device_data.Batcher(pool, recipe.batch, recipe.seed, torch, device, recipe.sampling)
    slope_batcher = None
    pair_left = pair_right = None
    if recipe.slope_weight > 0.0:
        if train_mask is None:
            raise ValueError("a slope term needs the training mask to keep its pairs inside it")
        pair_left, pair_right = tables.pairs(train_mask)
        if pair_left.numel() == 0:
            raise ValueError("no neighbour pair lies wholly inside the training region")
        slope_pool = torch.arange(pair_left.numel(), device=device)
        slope_batcher = device_data.Batcher(
            slope_pool, min(recipe.batch, pair_left.numel()),
            recipe.seed ^ 0x51073, torch, device, recipe.sampling)

    first_step = 0
    history: list[dict] = []
    if resume is not None:
        for field in ("batch", "loss", "slope_weight", "seed", "data_path", "sampling"):
            was = resume.get("recipe", {}).get(field)
            if was is not None and was != getattr(recipe, field):
                raise ValueError(
                    f"checkpoint was written with {field}={was!r}, resuming with "
                    f"{getattr(recipe, field)!r}: that is a different experiment")
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        torch.set_rng_state(resume["torch_rng"])
        batcher.load_state(resume["batcher"])
        if slope_batcher is not None and resume.get("slope_batcher") is not None:
            slope_batcher.load_state(resume["slope_batcher"])
        first_step = int(resume["step"]) + 1
        history = list(resume.get("history", []))

    # A captured graph cannot call back into Python for a per-step learning rate,
    # so the schedule is precomputed on the device and the optimiser reads a
    # tensor. `capturable=True` is what makes Adam's own state safe to replay.
    engine = None
    schedule = None
    lr_tensor = None
    static: dict = {}
    if engine_mode == "cudagraph" and recipe.slope_weight == 0.0 \
            and str(device).startswith("cuda"):
        from geoneural.neural import engine as engine_module
        schedule = engine_module.schedule_tensor(recipe, torch, device)
        lr_tensor = torch.tensor(float(recipe.lr), device=device)
        for group in optimizer.param_groups:
            group["lr"] = lr_tensor
            group["capturable"] = True
        static["indexes"] = torch.zeros(recipe.batch, dtype=torch.int64, device=device)
        static["loss"] = torch.zeros((), device=device)

        def captured_step():
            picked = batcher.fill(static["indexes"])
            coords, tiles, truth = tables.gather(picked)
            value = _objective(model(coords, tiles).squeeze(-1), truth, recipe)
            optimizer.zero_grad(set_to_none=False)
            value.backward()
            optimizer.step()
            static["loss"].copy_(value.detach())

        engine = engine_module.StepEngine("cudagraph", torch, device)
        static["generators"] = (batcher.generator,)
        if not engine.try_capture(captured_step, static):
            for group in optimizer.param_groups:
                group["lr"] = float(recipe.lr)
                group["capturable"] = False
        else:
            first_step += engine.WARMUP_STEPS

    start = time.perf_counter()
    for step in range(first_step, recipe.steps):
        if engine is not None and engine.mode == "cudagraph":
            lr_tensor.copy_(schedule[step])
            engine.replay()
            report_now = recipe.eval_every > 0 and (step % recipe.eval_every == 0
                                                    or step == recipe.steps - 1)
            if report_now or step % max(recipe.steps // 20, 1) == 0 or step == recipe.steps - 1:
                value = float(static["loss"])
                if not math.isfinite(value):
                    raise FloatingPointError(f"training loss diverged at step {step}")
                record = {"step": step, "loss": value,
                          "batch_rmse_m": float(math.sqrt(max(value, 0.0))) * scale
                          if recipe.loss == "mse" else None,
                          "lr": _schedule(recipe, step),
                          "elapsed_s": time.perf_counter() - start}
                history.append(record)
                if on_report is not None and report_now:
                    on_report(step, record)
            continue
        for group in optimizer.param_groups:
            group["lr"] = _schedule(recipe, step)
        coords, tiles, truth = tables.gather(batcher.next())
        predicted = model(coords, tiles).squeeze(-1)
        loss = _objective(predicted, truth, recipe)

        if recipe.slope_weight > 0.0:
            picked = slope_batcher.next()
            left = pair_left.index_select(0, picked)
            right = pair_right.index_select(0, picked)
            ca, ta, _ = tables.gather(left)
            cb, tb, _ = tables.gather(right)
            pa = model(ca, ta).squeeze(-1)
            pb = model(cb, tb).squeeze(-1)
            observed = (tables.reference_m.index_select(0, right)
                        - tables.reference_m.index_select(0, left)).to(torch.float32) / scale
            loss = loss + recipe.slope_weight * (pb - pa - observed).square().mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        report_now = recipe.eval_every > 0 and (step % recipe.eval_every == 0
                                                or step == recipe.steps - 1)
        if report_now or step % max(recipe.steps // 20, 1) == 0 or step == recipe.steps - 1:
            value = float(loss.detach())
            if not math.isfinite(value):
                raise FloatingPointError(f"training loss diverged at step {step}")
            record = {"step": step, "loss": value,
                      "batch_rmse_m": float(math.sqrt(max(value, 0.0))) * scale
                      if recipe.loss == "mse" else None,
                      "lr": _schedule(recipe, step),
                      "elapsed_s": time.perf_counter() - start}
            history.append(record)
            if on_report is not None and report_now:
                on_report(step, record)
        if checkpoint_every > 0 and checkpoint_path is not None \
                and (step + 1) % checkpoint_every == 0 and step + 1 < recipe.steps:
            state = {"schema": "geoneural-training-checkpoint-v2", "step": int(step),
                     "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                     "optimizer": optimizer.state_dict(),
                     "batcher": batcher.state(),
                     "slope_batcher": None if slope_batcher is None else slope_batcher.state(),
                     "torch_rng": torch.get_rng_state(),
                     "history": list(history), "recipe": recipe.as_dict()}
            save_checkpoint(checkpoint_path, state, torch)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    return {"history": history, "seconds": time.perf_counter() - start,
            "recipe": recipe.as_dict(), "resumedFromStep": first_step or None,
            "engine": None if engine is None else engine.record()}
