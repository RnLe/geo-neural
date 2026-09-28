"""Choose architectures and frequency bands on held-out criteria.

A single width, depth, learning rate, seed and the SIREN paper's default
frequency scale establish a baseline, but say little about the families: a
family that needs a different frequency scale or learning rate looks bad for
reasons unrelated to the family. Model size and frequency bands are therefore
selected with held-out criteria, and no single favourable seed is chosen.

Three properties make this a measurement rather than a leaderboard:

The objective is rate and error, not error. Minimising held-out error alone has
one answer (spend more parameters) and gives a number that cannot be compared
with a conventional codec. Each study is therefore multi-objective over
(deployed bytes, selection MAE), and its result is a Pareto front in the same
plane as the conventional codec rate-distortion curve. Maximum and p99 error are
recorded for every trial but do not drive the search, so the front can be re-cut
on them afterwards without refitting.

The training budget is fixed inside a study. If steps were searchable, every
trial could buy accuracy with time and the front would measure compute, not
rate. Steps, batch size and the evaluation protocol are study constants and are
recorded as such.

Selection and reporting use different pages. Trials see only the selection half
of the held-out checkerboard, as often as they like. The test half and the
extrapolation block are untouched until `confirm`, which retrains the chosen
configurations across several seeds. A configuration that wins by seed luck
shows up there as a spread.

This does not by itself make the comparison with conventional compression fair.
A neural field returns point heights; a conventional page returns point heights
plus a hash, a height range and an error bound. Equal bytes at equal MAE is not
an equal product, and the maximum-error column is where that shows.
"""
from __future__ import annotations
import math
import time
from pathlib import Path

import numpy as np

from geoneural.codecs import codecs

from geoneural.neural import splits

from geoneural.neural import training
from geoneural.common import read_json, sha_file
from geoneural.neural.training import Recipe

SCHEMA = "geoneural-architecture-search-v1"
FAMILIES = ("mlp", "siren", "fourier", "shared", "codegrid", "bandlimited", "grid", "residual",
            "hybrid", "liif")

# What `residual` may put on top of its coarse base. None of these carries local
# codes, which is why the preferred design has its own arm (`hybrid`) rather than
# a fourth entry here: widening this categorical would change what every
# previously run residual trial was sampling from.
RESIDUAL_INNER = ("siren", "grid", "mlp")

# A search arm is not always a model kind. "hybrid" is the preferred design (a
# coarse base carrying a modulated code-grid decoder), which builds as a
# `residual` model. Keeping them distinct lets the two arms be tuned and reported
# separately without pretending they are different architectures.
ARM_KIND = {"hybrid": "residual"}
# Rate targets for the conventional comparator, matching the codec comparison's
# declared accuracy targets.
CONVENTIONAL_TARGETS = (0.01, 0.05, 0.1, 0.5, 1.0)

# The coarse axis needs its own, coarser, targets. CONVENTIONAL_TARGETS are the
# declared accuracy targets, the right sweep for full-resolution quantization
# because they are the guarantees on offer. On the downsample axis the target is
# not a guarantee but a rate knob on top of a grid whose error is already set by
# its level, and stopping that knob at 1.0 m means a fine level can never be
# bought cheaply.
#
# Stopping at 1.0 m removes exactly the conventional points at the rates a small
# network occupies, and can manufacture a neural win: `fourier` at 21,608 bytes
# and 1.368 m appears to beat the conventional front on mean and maximum, while
# level 2 at a 2.0 m target (20,020 bytes, 1.059 m, 14.69 m max) is cheaper and
# better on both.
COARSE_TARGETS = (0.05, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)

# The full-resolution axis needs the coarse end too, for the same reason.
# CONVENTIONAL_TARGETS stop at 1.0 m because 0.01-1.0 m is the accuracy regime of
# interest, but a network at half a bit per sample is not competing with a 1 cm
# guarantee. It competes with whatever the conventional codec does at its rate,
# and quantizing the full lattice to a coarse bound is an ordinary way to get
# there.
#
# Example: hybrid at 120,951 bytes, 0.370 m mean and 7.67 m measured maximum looks
# 50% cheaper than any conventional point with a better maximum if the cheapest
# swept point is q32-delta-zstd at a 1.0 m target (241,060 bytes). At an 8 m
# target the same codec costs 73,961 bytes and guarantees 8 m: 39% cheaper than
# the neural point, with a bound rather than an observation.
ENVELOPE_TARGETS = tuple(sorted(set(CONVENTIONAL_TARGETS + (2.0, 4.0, 8.0, 16.0))))


def paged_bytes(codec, grid: np.ndarray, target: float, intervals: int) -> int:
    """Independently compressed pages, the way a randomly accessible archive ships.

    Both conventional curves charge this. Charging one the monolithic rate and
    the other the paged rate would bias the comparison by about 26 %.
    """
    return paged_decode(codec, grid, target, intervals)[1]


def paged_decode(codec, grid: np.ndarray, target: float, intervals: int) -> tuple[np.ndarray, int]:
    """The decoded pages that are charged, stitched, and their bytes (see codecs.envelope.paged)."""
    from geoneural.codecs.envelope import paged
    return paged(codec, grid, target, intervals)


class Problem:
    """The frozen data, split and normalisation every trial shares.

    Built once per process. A trial that rebuilt the split or renormalised would
    be a different experiment under the same name.
    """

    def __init__(self, atlas_path: Path, device: str = "cuda",
                 extrapolation_fraction: float = 0.25, normalisation: str = "all"):
        import torch
        self.torch = torch
        self.atlas_path = Path(atlas_path)
        self.manifest = read_json(self.atlas_path)
        reference_path = self.atlas_path.parent / "reference.npy"
        if sha_file(reference_path) != self.manifest["reference_sha256"]:
            raise ValueError("Reference changed since the atlas was built")
        self.reference = np.load(reference_path, allow_pickle=False).astype(np.float64)
        self.side = int(self.reference.shape[0])
        self.intervals = int(self.manifest["page_intervals"])
        self.spacing_m = float(self.manifest["spacing_m"])
        self.device = device
        self.flat = self.reference.reshape(-1)
        self.split = splits.build(self.side, self.intervals, extrapolation_fraction,
                                  selection_split=True)
        # Where the normalisation statistics come from is part of the experiment's
        # information contract. Whole-reference mean and standard deviation let
        # withheld target information into the preprocessing of a holdout
        # experiment.
        #
        # "all" is correct for a codec: full-target statistics are allowed for
        # encoding provided they are stored, and these two scalars are stored.
        # "train" is what a predictive evaluation needs, because a holdout page
        # must not reach the model even through a mean. The choice is made per
        # experiment rather than inherited.
        if normalisation not in ("all", "train"):
            raise ValueError(f"Unknown normalisation scope: {normalisation}")
        self.normalisation = normalisation
        source = self.flat if normalisation == "all" \
            else self.flat[np.flatnonzero(self.split["trainMask"].reshape(-1))]
        self.mean = float(np.mean(source))
        self.scale = max(float(np.std(source)), 1.0)
        self.tiles = ((self.side - 1) // self.intervals) ** 2
        self.indexes = {name: np.flatnonzero(self.split[key].reshape(-1)) for name, key in (
            ("train", "trainMask"), ("selection", "selectionMask"),
            ("test", "testMask"), ("extrapolation", "extrapolationMask"))}
        self.indexes["all"] = np.arange(self.flat.size, dtype=np.int64)
        self._levels: dict[int, np.ndarray] = {}
        self._tables: dict[bool, object] = {}

    def tables(self, shared: bool):
        """Device-resident lattice tables, built once per (problem, shared) pair.

        Cached on the Problem because a study runs hundreds of trials against one
        problem and the upload is the same every time.
        """
        from geoneural.neural import device_data
        key = bool(shared)
        if key not in self._tables:
            self._tables[key] = device_data.DeviceTables(self, key, self.torch)
        return self._tables[key]

    def level_grid(self, level: int) -> np.ndarray:
        """The atlas's own decoded page data at a pyramid level, cached."""
        if level not in self._levels:
            from geoneural.codecs import bounds
            self._levels[level] = bounds.level_grid(self.atlas_path.parent, self.manifest, level)
        return self._levels[level]

    def base(self, level: int, target_m: float) -> tuple[np.ndarray, int, dict]:
        """A conventional coarse grid, its decoded values and its real cost.

        The bytes come from encoding that grid with the winning conventional
        codec, at a declared max-error target. The conventional half of a hybrid
        must be charged for, or the hybrid is subsidised.
        """
        grid = self.level_grid(level)
        codec = codecs.registry()["q32-delta-zstd"]
        measured = codecs.measure(codec, grid.astype(np.float32), target_m)
        # Charged as independently compressed pages, the convention
        # conventional_curve and base_only_curve both use. Billing a hybrid at the
        # monolithic rate while its control pays the paged rate would give it an
        # unearned 20-30 % discount. The values are decoded from those same pages.
        decoded, paged = paged_decode(codec, grid.astype(np.float32), target_m, self.intervals)
        return decoded, paged, {
            "level": level, "targetM": target_m, "side": int(grid.shape[0]),
            "bytes": paged, "monolithicBytes": int(measured["bytes"]),
            "maxErrorM": float(measured["max_error_m"]), "codec": "q32-delta-zstd"}


def build_model(config: dict, problem: Problem):
    """Instantiate a family, resolving the pieces a Problem has to supply."""
    from geoneural.neural.models import make_model
    config = dict(config)
    extra = {}
    if config["kind"] == "shared":
        config["tiles"] = problem.tiles
    if config["kind"] == "residual":
        decoded, base_bytes, record = problem.base(int(config.pop("base_level")),
                                                   float(config.pop("base_target_m")))
        normalised = (decoded - problem.mean) / problem.scale
        config["base"] = problem.torch.from_numpy(normalised.astype(np.float32))
        inner = dict(config["inner"])
        if inner["kind"] == "shared":
            inner["tiles"] = problem.tiles
        config["inner"] = inner
        extra = {"baseBytes": base_bytes, "base": record}
    return make_model(config), extra


def describe(config: dict) -> dict:
    """A JSON-safe copy of a model config, with any bulk array stripped.

    The filter is on the value rather than the key, so a new family that carries
    an array (such as `context` with its geology raster) is stripped without
    having to be listed here.

    Small lists survive, because a config legitimately carries things like layer
    sizes; anything with a `shape` does not.
    """
    out = {}
    for key, value in config.items():
        if hasattr(value, "shape") or hasattr(value, "detach"):
            out[f"{key}Shape"] = list(getattr(value, "shape", ()))
            continue
        out[key] = value
    if isinstance(out.get("inner"), dict):
        out["inner"] = describe(out["inner"])
    return out


# The precision a search prices and measures at. Deployment ships stored tensors at
# this width and widens them on load; pricing one width while evaluating another
# would be inconsistent. See `training.round_to_storage` for why this is float16 by default.
DEFAULT_STORE_PRECISION = "float16"

# Initialisation draws from its own stream so that changing the training seed
# still changes the initial weights, while construction stays independent of
# anything the process did earlier.
MODEL_SEED_SALT = 0x1A17



def measure(config: dict, recipe: Recipe, problem: Problem,
            evaluate_on=("selection",), limit: int | None = 200_000,
            on_report=None, train_on: str = "train",
            store_precision: str = DEFAULT_STORE_PRECISION,
            engine_mode: str = "eager") -> dict:
    """Train one configuration and report it. The unit of every experiment here.

    Training runs in float32 throughout; `store_precision` is the width the result
    is stored at, applied once after fitting. The returned model is the deployed
    model (already rounded), so every metric taken from it, here or by a caller,
    is a metric of what would actually ship.
    """
    from geoneural.neural.learning import features as _lattice_features
    # A multi-region problem addresses several atlases through one index space and
    # supplies its own unpacking. Everything else about the run (training loop,
    # evaluation, byte accounting) is the same code, because a multi-region arm
    # measured through a different path would not be comparable with the
    # single-region numbers it is read against.
    features = getattr(problem, "features_fn", None) or _lattice_features
    torch = problem.torch
    # Seed before constructing the model. `fit` seeds torch at the top of the
    # training loop, which is after initialisation; without this, a model's
    # initial weights would come from whatever state the previous trial left in
    # the global generator, and the first model in a process from an unseeded
    # one. A trial's result must not depend on which trials ran before it.
    torch.manual_seed(recipe.seed ^ MODEL_SEED_SALT)
    model, extra = build_model(config, problem)
    shared = config["kind"] == "shared" or (
        config.get("inner", {}).get("kind") == "shared" if isinstance(config.get("inner"), dict) else False)
    weight_bytes = training.deployed_bytes(model, torch, store_precision)
    float32_bytes = training.deployed_bytes(model, torch, "float32")
    tensor_only = training.tensor_bytes(model, torch, store_precision)
    base_bytes = int(extra.get("baseBytes", 0))
    train_mask = problem.split["trainMask"] if train_on == "train" \
        else np.ones((problem.side, problem.side), dtype=bool)
    tables = problem.tables(shared) if recipe.data_path == "device" else None
    result = training.fit(model, features, problem.flat, problem.side, problem.intervals,
                          shared, problem.indexes[train_on], problem.mean, problem.scale,
                          recipe, torch, train_mask=train_mask, on_report=on_report,
                          tables=tables, engine_mode=engine_mode)
    # Round before evaluating, not after reporting, so metrics are measured at
    # the precision they are priced at.
    # Code coverage is measured on both sides of storage rounding, because they
    # answer different questions and can disagree completely. Before rounding:
    # did training reach the codes? After: does the deployed model still use
    # them? A modulation weight below the fp16 subnormal floor flushes to zero on
    # rounding, so a network can learn a real but tiny conditioning path and
    # deploy one that is dead. Its code payload and weights are then inert bytes
    # and the deployed model is exactly its conventional base (for example 0.645
    # coverage after training, 0.0 deployed, and an error equal to the base's).
    coverage_trained = code_coverage(model, problem, shared, train_on=train_on)
    training.round_to_storage(model, torch, store_precision)
    rng = np.random.default_rng(recipe.seed)
    if tables is not None:
        from geoneural.neural import device_data
        # Subsampling stays on the host so the picked indexes are the same ones
        # the host path would have picked; only the decode and the reduction move.
        metrics = {}
        for name in evaluate_on:
            picked = problem.indexes[name]
            if limit is not None and picked.size > limit:
                picked = rng.choice(picked, limit, replace=False)
            metrics[name] = device_data.evaluate(
                model, tables, torch.from_numpy(np.asarray(picked, dtype=np.int64)).to(
                    problem.device), torch)
    else:
        metrics = {name: training.evaluate(model, features, problem.indexes[name], problem.flat,
                                           problem.side, problem.intervals, shared, problem.mean,
                                           problem.scale, problem.device, torch, limit, rng)
                   for name in evaluate_on}
    return {
        "config": describe(config), "recipe": recipe.as_dict(),
        "weightsBytes": weight_bytes, "baseBytes": base_bytes,
        "deployedBytes": weight_bytes + base_bytes,
        "storePrecision": store_precision,
        "weightsBytesFloat32": float32_bytes,
        "tensorBytes": tensor_only,
        "containerBytes": weight_bytes - tensor_only,
        "deployedBytesFloat32": float32_bytes + base_bytes,
        "bitsPerSample": 8.0 * (weight_bytes + base_bytes) / problem.flat.size,
        "codeCoverageTrained": coverage_trained,
        "engine": result.get("engine"),
        "trainingSeconds": result["seconds"], "history": result["history"],
        "trainedOn": train_on,
        "metrics": metrics, "model": model, "shared": shared,
        **{k: v for k, v in extra.items() if k != "baseBytes"},
    }


# ---------------------------------------------------------------------------
# Search spaces. Ranges are deliberately wide enough to contain configurations
# that will fail, because a space drawn around the answer proves nothing.


def sample_model(trial, family: str) -> dict:
    if family == "mlp":
        return {"kind": "mlp", "width": trial.suggest_int("width", 16, 384, log=True),
                "depth": trial.suggest_int("depth", 2, 7),
                "activation": trial.suggest_categorical("activation", ["gelu", "relu", "silu"])}
    if family == "siren":
        omega = trial.suggest_float("omega", 2.0, 200.0, log=True)
        return {"kind": "siren", "width": trial.suggest_int("width", 16, 384, log=True),
                "depth": trial.suggest_int("depth", 2, 7), "omega": omega,
                "hidden_omega": trial.suggest_float("hidden_omega", 2.0, 100.0, log=True)}
    if family == "fourier":
        mode = trial.suggest_categorical("mode", ["dyadic", "gaussian"])
        config = {"kind": "fourier", "width": trial.suggest_int("width", 16, 384, log=True),
                  "depth": trial.suggest_int("depth", 2, 6), "mode": mode,
                  "bands": trial.suggest_int("bands", 4, 96, log=True)}
        if mode == "gaussian":
            config["scale"] = trial.suggest_float("scale", 0.5, 64.0, log=True)
        return config
    if family == "shared":
        # omega ranges match the siren arm's exactly. Since a zero code reduces
        # this family to that one, an unmatched frequency range would reintroduce
        # by the back door the confound the shared backbone just removed.
        return {"kind": "shared", "width": trial.suggest_int("width", 16, 256, log=True),
                "depth": trial.suggest_int("depth", 2, 6),
                "latent": trial.suggest_int("latent", 2, 64, log=True),
                "omega": trial.suggest_float("omega", 2.0, 200.0, log=True),
                "hidden_omega": trial.suggest_float("hidden_omega", 2.0, 100.0, log=True)}
    if family == "liif":
        # The one family here that is not a function of an absolute coordinate.
        # Patch counts share codegrid's power-of-two convention so the two are
        # directly comparable at equal raster resolution: same payload shape,
        # different decoding frame.
        return {"kind": "liif",
                "patches": trial.suggest_categorical("patches", [4, 8, 16, 32, 64]),
                "latent": trial.suggest_int("latent", 4, 64, log=True),
                "width": trial.suggest_int("width", 16, 256, log=True),
                "depth": trial.suggest_int("depth", 1, 5),
                "activation": trial.suggest_categorical("activation", ["relu", "gelu"]),
                "cell_decoding": trial.suggest_categorical("cell_decoding", [True, False])}
    if family == "codegrid":
        # Patch counts are powers of two so a patch is a whole number of atlas
        # pages: 16 patches is one page each (640 m), 8 is four pages (1.28 km),
        # 4 is sixteen (2.56 km), i.e. 64/128/256 sample intervals.
        return {"kind": "codegrid",
                "patches": trial.suggest_categorical("patches", [4, 8, 16, 32]),
                "latent": trial.suggest_int("latent", 4, 64, log=True),
                "width": trial.suggest_int("width", 16, 256, log=True),
                "depth": trial.suggest_int("depth", 2, 6),
                "omega": trial.suggest_float("omega", 2.0, 200.0, log=True),
                "hidden_omega": trial.suggest_float("hidden_omega", 2.0, 100.0, log=True)}
    if family == "bandlimited":
        return {"kind": "bandlimited", "width": trial.suggest_int("width", 16, 384, log=True),
                "depth": trial.suggest_int("depth", 2, 7),
                "bandwidth": trial.suggest_float("bandwidth", 0.5, 64.0, log=True)}
    if family == "grid":
        return {"kind": "grid", "levels": trial.suggest_int("levels", 2, 16),
                "base_resolution": trial.suggest_int("base_resolution", 2, 32, log=True),
                "growth": trial.suggest_float("growth", 1.1, 2.5),
                "features": trial.suggest_int("features", 1, 4),
                "table_size": 1 << trial.suggest_int("log2_table", 8, 17),
                "hashed": trial.suggest_categorical("hashed", [True, False]),
                "width": trial.suggest_int("width", 8, 128, log=True),
                "depth": trial.suggest_int("depth", 1, 4)}
    if family == "residual":
        inner_family = trial.suggest_categorical("inner", list(RESIDUAL_INNER))
        inner = sample_model(trial, inner_family)
        return {"kind": "residual", "inner": inner,
                "base_level": trial.suggest_int("base_level", 1, 4),
                "base_target_m": trial.suggest_categorical("base_target_m", [0.05, 0.5, 1.0, 4.0])}
    if family == "hybrid":
        # The preferred architecture, which the `residual` arm cannot reach: a
        # fixed coarse base plus a shared width-128, three-layer modulated
        # coordinate decoder with 32-dimensional codes on 128-interval patches.
        # `residual` samples its inner model from siren, grid and mlp, none of
        # which carries local codes.
        #
        # It is a separate arm rather than a fourth `inner` option so that
        # searches already run against the three-way space stay comparable:
        # widening a categorical changes what every trial of that family was
        # sampling from.
        inner = sample_model(trial, "codegrid")
        return {"kind": "residual", "inner": inner,
                "base_level": trial.suggest_int("base_level", 1, 4),
                "base_target_m": trial.suggest_categorical("base_target_m", [0.05, 0.5, 1.0, 4.0])}
    raise ValueError(f"Unknown family: {family}")


def sample_recipe(trial, steps: int, batch: int, device: str, seed: int,
                  data_path: str = "host", sampling: str = "without") -> Recipe:
    """The recipe is tuned per family, so a family is never judged on another's
    learning rate. Steps and batch are study constants and are not searched."""
    loss = trial.suggest_categorical("loss", ["mse", "huber", "l1"])
    return Recipe(
        steps=steps, batch=batch, device=device, seed=seed,
        data_path=data_path, sampling=sampling,
        lr=trial.suggest_float("lr", 1e-4, 2e-2, log=True),
        schedule=trial.suggest_categorical("schedule", ["none", "cosine"]),
        warmup=trial.suggest_int("warmup", 0, max(steps // 10, 1)),
        weight_decay=trial.suggest_float("weight_decay", 1e-9, 1e-3, log=True),
        loss=loss,
        huber_delta=trial.suggest_float("huber_delta", 0.002, 0.2, log=True) if loss == "huber" else 0.05,
        slope_weight=trial.suggest_float("slope_weight", 1e-4, 1.0, log=True)
        if trial.suggest_categorical("use_slope", [True, False]) else 0.0,
    )


# What a study is allowed to optimise. These are different scientific questions
# and keep different names and score tables: "holdout" asks whether the
# representation generalises to pages it never saw, and "codec" asks how few bytes
# reproduce what it did see. Only the second is comparable with the
# rate-distortion baseline.
MODES = {
    "holdout": {"trainOn": "train", "scoreOn": "selection"},
    "codec": {"trainOn": "all", "scoreOn": "all"},
}


def study(family: str, problem: Problem, trials: int = 30, steps: int = 5000,
          batch: int = 8192, seed: int = 1729, byte_ceiling: int = 1_200_000,
          sampler_seed: int = 20260912, progress: bool = True,
          mode: str = "holdout",
          store_precision: str = DEFAULT_STORE_PRECISION,
          data_path: str = "host", sampling: str = "without",
          engine_mode: str = "eager") -> dict:
    """One multi-objective study for one family: minimise bytes and the mode's error.

    `mode` selects the question (see `MODES`) and is part of the study identity, so
    a codec-objective front is never read as a generalisation result or the reverse.

    `progress` prints one JSON line per trial, so a study of a few hundred
    trainings shows how far along it is.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown study mode: {mode}")
    train_on, score_on = MODES[mode]["trainOn"], MODES[mode]["scoreOn"]
    import json
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    started = time.perf_counter()
    records: list[dict] = []

    def announce(**fields) -> None:
        if progress:
            elapsed = time.perf_counter() - started
            done = len(records)
            rate = elapsed / max(done, 1)
            print(json.dumps({"family": family, "elapsed_s": round(elapsed, 1),
                              "completed": done, "of": trials,
                              "eta_s": round(rate * max(trials - done, 0), 1) if done else None,
                              **fields}), flush=True)

    def objective(trial):
        config = sample_model(trial, family)
        recipe = sample_recipe(trial, steps, batch, problem.device, seed,
                               data_path, sampling)
        try:
            model, extra = build_model(config, problem)
        except (ValueError, KeyError) as error:
            raise optuna.TrialPruned(f"invalid configuration: {error}")
        bytes_total = training.deployed_bytes(model, problem.torch, store_precision) \
            + int(extra.get("baseBytes", 0))
        del model
        # Reject over-budget configurations before spending a second on them.
        if bytes_total > byte_ceiling:
            trial.set_user_attr("prunedBytes", bytes_total)
            announce(trial=trial.number, pruned="over byte ceiling", bytes=bytes_total)
            raise optuna.TrialPruned(f"{bytes_total} bytes over the {byte_ceiling} ceiling")
        try:
            # The scored split is evaluated in full, never subsampled. The
            # selection half is 190,512 nodes and fits under the default limit
            # anyway, but codec mode scores all 1,050,625, and a subsample would
            # make the recorded maximum a sample maximum. A sampled maximum must
            # not be reported as a global error bound.
            result = measure(config, recipe, problem, evaluate_on=(score_on,),
                             train_on=train_on, store_precision=store_precision,
                             limit=None, engine_mode=engine_mode)
        except (FloatingPointError, RuntimeError) as error:
            # Divergence is a real outcome of a hyperparameter choice, kept as a
            # pruned trial with its reason rather than silently dropped.
            trial.set_user_attr("failure", f"{type(error).__name__}: {error}")
            announce(trial=trial.number, pruned=f"{type(error).__name__}", bytes=bytes_total)
            raise optuna.TrialPruned(str(error))
        coverage = code_coverage(result["model"], problem, result["shared"], train_on=train_on)
        if coverage is not None:
            trial.set_user_attr("codeCoverageFraction", coverage["coverageFraction"])
            record_coverage = coverage
        else:
            record_coverage = None
        selection = result["metrics"][score_on]
        for key in ("mae_m", "rmse_m", "p95_m", "p99_m", "max_m"):
            trial.set_user_attr(key, selection[key])
        trial.set_user_attr("deployedBytes", result["deployedBytes"])
        trial.set_user_attr("trainingSeconds", result["trainingSeconds"])
        record = {k: v for k, v in result.items() if k not in ("model", "history")}
        record["codeCoverage"] = record_coverage
        record["trial"] = trial.number
        record["finalLoss"] = result["history"][-1]["loss"]
        records.append(record)
        del result
        if problem.device.startswith("cuda"):
            problem.torch.cuda.empty_cache()
        announce(trial=trial.number, bytes=record["deployedBytes"], scored_on=score_on,
                 score_mae_m=round(selection["mae_m"], 4),
                 score_max_m=round(selection["max_m"], 3))
        return float(record["deployedBytes"]), float(selection["mae_m"])

    # NSGA-II with its default population of 50 never leaves generation 0 inside a
    # 24-trial budget, and generation 0 is uniform random, so it would be random
    # search under a GA's name. The population is sized to the budget so the GA
    # actually breeds, and the sampler that ran is recorded with its parameters.
    # Below four generations a GA has nothing to select on, so MOTPE is used
    # instead: it is designed for small multi-objective budgets and starts
    # modelling after ten trials.
    generations = 4
    population = max(8, trials // generations)
    if trials >= population * generations:
        sampler = optuna.samplers.NSGAIISampler(seed=sampler_seed, population_size=population)
        sampler_record = {"sampler": "NSGAII", "populationSize": population,
                          "generations": trials // population}
    else:
        sampler = optuna.samplers.TPESampler(seed=sampler_seed, multivariate=True, group=True,
                                             constant_liar=True, n_startup_trials=10)
        sampler_record = {"sampler": "MOTPE", "nStartupTrials": 10,
                          "why": f"{trials} trials is under {population * generations}, the "
                                 "budget an NSGA-II population of "
                                 f"{population} would need to breed four generations"}
    optuna_study = optuna.create_study(
        directions=["minimize", "minimize"],
        sampler=sampler,
        study_name=f"geoneural-{family}")
    optuna_study.optimize(objective, n_trials=trials, catch=())

    front = [{"trial": t.number, "deployedBytes": t.values[0], "scoreMaeM": t.values[1],
              "scoreMaxM": t.user_attrs.get("max_m"), "scoreP99M": t.user_attrs.get("p99_m"),
              "codeCoverageFraction": t.user_attrs.get("codeCoverageFraction"),
              "params": t.params}
             for t in optuna_study.best_trials]
    front.sort(key=lambda row: row["deployedBytes"])
    pruned = [{"trial": t.number, **t.user_attrs} for t in optuna_study.trials
              if t.state == optuna.trial.TrialState.PRUNED]
    return {
        "family": family, "trialsRequested": trials,
        "trialsCompleted": sum(1 for t in optuna_study.trials
                               if t.state == optuna.trial.TrialState.COMPLETE),
        "trialsPruned": len(pruned), "prunedTrials": pruned,
        "mode": mode, "trainedOn": train_on, "scoredOn": score_on,
        "studyConstants": {"steps": steps, "batch": batch, "seed": seed,
                           "byteCeiling": byte_ceiling, "samplerSeed": sampler_seed,
                           "storePrecision": store_precision,
                           "dataPath": data_path, "sampling": sampling,
                           "engineMode": engine_mode,
                           **sampler_record,
                           "objectives": ["deployedBytes", "scoreMaeM"]},
        "paretoFront": front, "trials": records,
        "parameterImportances": _importances(optuna_study),
        "codeCoverageCaveat": "A family storing local codes cannot be read on the holdout splits unless "
                              "codeCoverageFraction is 1.0; below that, part of the reported error is "
                              "the code initialiser on regions training never reached. Codec-fit runs "
                              "are unaffected because every region is trained there.",
    }


def _importances(optuna_study) -> dict:
    """PED-ANOVA associations within this sampled, pruned study, and no more.

    Optuna 5.0 makes PED-ANOVA the default evaluator. An importance here says
    which knob moved the objective across the trials that actually ran; it is not
    evidence that the knob matters in general, and a high score earns a
    controlled ablation rather than a conclusion.
    """
    import optuna
    out: dict = {"method": "optuna.importance.get_param_importances (PED-ANOVA default)",
                 "direction": "PED-ANOVA ranks parameters by their influence on reaching LOW target "
                              "values. Both objectives here are minimised, so that is the wanted "
                              "direction and the evaluator's warning about it is expected.",
                 "qualification": "Associations within a sampled and pruned search. Not an "
                                  "architectural conclusion and not causal. A high score earns a "
                                  "controlled ablation, not a statement about architectures."}
    for index, name in enumerate(("deployedBytes", "scoreMaeM")):
        try:
            out[name] = {k: float(v) for k, v in optuna.importance.get_param_importances(
                optuna_study, target=lambda t, i=index: t.values[i]).items()}
        except (ValueError, RuntimeError) as error:
            out[name] = {"unavailable": f"{type(error).__name__}: {error}"}
    return out


def screen(problem: Problem, families=FAMILIES, steps: int = 5000, batch: int = 8192,
           seed: int = 1729, recipe: Recipe | None = None,
           store_precision: str = DEFAULT_STORE_PRECISION,
           evaluate_on=("train", "selection")) -> dict:
    """Every family at one matched recipe, before any tuning.

    This is the control for the search itself. If a family only wins after being
    tuned, then tuning found it; the untuned comparison matches the single-config
    baseline and is the only way to say how much of an improvement came from the
    search rather than from the architecture.

    Selection pages only. Screening decides which families get searched at all,
    so it is part of selection and must not see the test half or the
    extrapolation block; `confirm` is the only function that reads them. Pass
    `evaluate_on` explicitly to read more, and say why in the artifact.
    """
    base = recipe or Recipe(steps=steps, batch=batch, seed=seed, device=problem.device,
                            lr=1e-3, schedule="cosine", loss="mse")
    defaults = {
        "mlp": {"kind": "mlp", "width": 128, "depth": 3},
        "siren": {"kind": "siren", "width": 128, "depth": 3},
        "fourier": {"kind": "fourier", "width": 128, "depth": 3},
        "shared": {"kind": "shared", "width": 128, "depth": 3},
        "codegrid": {"kind": "codegrid", "patches": 16, "latent": 16, "width": 128, "depth": 3},
        "liif": {"kind": "liif", "patches": 16, "latent": 16, "width": 128, "depth": 3},
        "bandlimited": {"kind": "bandlimited", "width": 128, "depth": 3, "bandwidth": 8.0},
        "grid": {"kind": "grid", "levels": 8, "base_resolution": 8, "growth": 1.7,
                 "features": 2, "table_size": 1 << 14, "width": 32, "depth": 2},
        "residual": {"kind": "residual", "base_level": 3, "base_target_m": 1.0,
                     "inner": {"kind": "siren", "width": 128, "depth": 3}},
        # The preferred hybrid design: a fixed coarse base, a shared width-128
        # three-layer modulated decoder, 32-dimensional codes on 128-interval
        # patches. At 64 intervals per page and a 1025-node lattice, 8 patches is
        # 128 sample intervals a side.
        "hybrid": {"kind": "residual", "base_level": 3, "base_target_m": 1.0,
                   "inner": {"kind": "codegrid", "patches": 8, "latent": 32,
                             "width": 128, "depth": 3}},
    }
    rows = []
    for family in families:
        result = measure(defaults[family], base, problem, evaluate_on=tuple(evaluate_on),
                          store_precision=store_precision)
        rows.append({k: v for k, v in result.items() if k not in ("model", "history")})
        if problem.device.startswith("cuda"):
            problem.torch.cuda.empty_cache()
    return {"schema": SCHEMA, "mode": "matched-recipe screening",
            "recipe": base.as_dict(), "rows": rows,
            "storePrecision": store_precision, "evaluatedOn": list(evaluate_on),
            "qualification": "One configuration and one seed per family at a shared recipe. It says "
                             "which families are worth searching, not which family is better; the "
                             "shared recipe suits some of them better than others, which is why the "
                             "per-family search exists."}


def rate_denominators(problem: Problem) -> dict:
    """The two sample counts a bits-per-sample figure can be divided by.

    The codec tournament normalises by every sample it stores across the whole
    pyramid; a coordinate network stores no pyramid, so its natural denominator
    is the unique valid samples of the finest domain. The two differ by a factor
    of about 1.37 on this atlas, and a neural bits-per-sample quoted against a
    tournament bits-per-sample without that adjustment understates the
    conventional side by that factor.

    Every byte figure elsewhere in this module is absolute, so the comparisons
    here do not depend on this. It is recorded so that a bits-per-sample number
    states which denominator produced it.
    """
    side, intervals = problem.side, problem.intervals
    finest = side * side
    across_levels, level, current = 0, 0, side
    pages_per_side = (side - 1) // intervals
    while True:
        pages = max(pages_per_side >> level, 1) ** 2
        across_levels += pages * (intervals + 1) ** 2
        if pages == 1:
            break
        level += 1
    return {
        "uniqueFinestSamples": finest,
        "samplesAcrossLevels": across_levels,
        "multiscaleOverPrimary": across_levels / finest,
        "primary": "uniqueFinestSamples",
        "note": "Divide deployed bytes by uniqueFinestSamples for the primary rate. The codec "
                "tournament divides by samplesAcrossLevels, which is larger because page edges are "
                "shared between neighbours and coarser levels are stored too, so its bits-per-sample "
                "figures are smaller for the same archive. Multiply a tournament figure by "
                "multiscaleOverPrimary before setting it beside a neural one.",
        "qualification": "A count of stored samples, not of independent information: adjacent pages "
                         "share their edge samples and every coarse sample is derived from finer ones.",
    }


def conventional_curve(problem: Problem, targets=CONVENTIONAL_TARGETS,
                       codec_name: str = "q32-delta-zstd") -> dict:
    """The comparator, stated four ways so it is clear which one is meant.

    The whole five-level pyramid plus the compact EATIDX1 index costs 390,343
    bytes at the 1.0 m target. That is a correct package total but the wrong
    comparator: the network ships no pyramid, no index, no per-page hash and no
    declared height range, so charging the conventional side for all of them
    inflates the gap in the network's favour. The like-for-like figure is 241,060
    bytes. The four definitions below are recomputed from the frozen reference
    every time this report is written.

    `perPageFinestBytes` is the primary comparator and the one to quote. A neural
    field answers a query at any coordinate without reading anything else, so the
    conventional product that matches it must also be randomly accessible, which
    means independently compressed pages rather than one monolithic stream.
    Independence costs about 29 % here, and charging conventional the monolithic
    rate would credit it with a product it cannot deliver.

    `monolithicFinestBytes` is the smallest valid number for the same heights
    and is kept because it is the right comparator for a whole-region decode.
    `pyramidBytes` additionally encodes every coarser level, which the atlas ships
    and a coordinate network does not reproduce: querying a network at coarse
    spacing returns point samples, not the [1,4,6,4,1] low-passed parent that the
    HLOD selector actually needs. `indexBytes` is charged to neither side.
    """
    codec = codecs.registry()[codec_name]
    finest = problem.reference.astype(np.float32)
    intervals, side = problem.intervals, problem.side
    per_side = (side - 1) // intervals

    def paged(grid, target):
        return paged_bytes(codec, grid, target, intervals)

    def per_split(decoded: np.ndarray) -> dict:
        """Error on each split, so this front can be read on the same pages a
        neural score was taken on. The selection half is measurably easier than
        the whole field (conventional error is 0.07 to 0.50 m lower there), so
        comparing a whole-field conventional number against a selection-half
        neural one credits the network with a difference in the terrain rather
        than in the representation."""
        error = np.abs(decoded.astype(np.float64).reshape(-1) - problem.flat)
        out = {}
        for name, index in problem.indexes.items():
            picked = error[index]
            out[name] = {"mae_m": float(picked.mean()),
                         "rmse_m": float(np.sqrt((picked ** 2).mean())),
                         "p99_m": float(np.quantile(picked, 0.99)),
                         "max_m": float(picked.max()),
                         "samples": int(picked.size)}
        return out

    rows = []
    for target in targets:
        measured = codecs.measure(codec, finest, target)
        decoded, per_page = paged_decode(codec, finest, target, intervals)
        splits_error = per_split(decoded)
        pyramid = per_page
        for level in range(1, int(problem.manifest["max_level"]) + 1):
            pyramid += paged(problem.level_grid(level).astype(np.float32), target)
        rows.append({
            "targetMaxErrorM": target, "codec": codec_name,
            "monolithicFinestBytes": int(measured["bytes"]),
            "perPageFinestBytes": per_page,
            "perPageFinestBitsPerSample": 8.0 * per_page / problem.flat.size,
            "randomAccessOverheadFraction": per_page / max(int(measured["bytes"]), 1) - 1.0,
            "pyramidBytes": pyramid,
            "maeM": float(measured["mae_m"]), "maxErrorM": float(measured["max_error_m"]),
            "rmseM": float(measured["rmse_m"]), "metrics": splits_error,
        })
    return {
        "rows": rows, "sampleCount": int(problem.flat.size),
        "pagesAtFinest": per_side * per_side,
        "indexBytes": {"json": 220796, "eatidx1": 16844,
                       "note": "Neither side is charged the index. It carries per-page hashes, height "
                               "ranges and error figures that a neural field does not provide at all."},
        "primaryComparator": "perPageFinestBytes",
        "primaryRationale": "A neural field answers any coordinate without reading neighbouring data, so "
                            "the matching conventional product must be randomly accessible: independently "
                            "compressed pages, not one monolithic stream.",
        "supersedes": "390,343 bytes at the 1.0 m target is the full pyramid plus the EATIDX1 "
                      "index. That is correct as a package total but wrong as the "
                      "comparator for a point-evaluating field that ships none of it. The like-for-like "
                      "row here is perPageFinestBytes.",
        "boundedness": "Every row meets its target by construction on the WHOLE field; the maxErrorM "
                       "column is that guarantee. The per-split max in metrics is an observation on a "
                       "subset and is never larger. No neural row in this report has any bound.",
    }


def finalists(report: dict, per_family: int = 2, limit: int = 8) -> dict:
    """Pick a diverse, reproducible set of front points to confirm across seeds.

    Confirmation uses about six diverse finalists at a larger fixed budget under
    three fixed seeds. Choosing them by hand is where a favourable point gets
    picked by eye, so the choice is a stated rule: from each family's Pareto
    front, the cheapest point and the most accurate one, the two ends a
    deployment would choose between. Ties and extra slots are taken in byte order
    so the same report always yields the same finalists.

    `per_family=1` keeps only the cheapest point. The picks are built cheapest
    first and then truncated, so a single slot always takes the cheap end and
    drops the accurate one, which is where any win lives (for example siren at
    2,742 bytes and 3.208 m instead of its 68,458-byte, 0.526 m point). Pass at
    least 2, and a `limit` of at least twice the family count, whenever the
    confirmation is meant to validate a front rather than sample it.

    Returns the `{name: {config, recipe}}` mapping `confirm` and `codec_fit` read.
    """
    chosen: dict = {}
    for family, study_report in sorted(report.get("byFamily", {}).items()):
        front = sorted(study_report.get("paretoFront", []), key=lambda row: row["deployedBytes"])
        if not front:
            continue
        by_trial = {record["trial"]: record for record in study_report.get("trials", [])}
        # Cheapest first, then most accurate; both are on the front by definition,
        # and on a one-point front they are the same trial.
        picks = [front[0]["trial"]]
        best = min(front, key=lambda row: row["scoreMaeM"])["trial"]
        if best not in picks:
            picks.append(best)
        for row in front:
            if len(picks) >= per_family:
                break
            if row["trial"] not in picks:
                picks.append(row["trial"])
        for trial in picks[:per_family]:
            record = by_trial.get(trial)
            if record is None:
                continue
            chosen[f"{family}-t{trial}"] = {"config": record["config"], "recipe": record["recipe"],
                                            "fromStudy": {"family": family, "trial": trial,
                                                          "mode": study_report.get("mode"),
                                                          "deployedBytes": record["deployedBytes"]}}
    if len(chosen) <= limit:
        return chosen
    # Trim round-robin across families, not by global byte order. Taking the
    # cheapest `limit` overall would hand every slot to whichever families sit at
    # the small end and drop others entirely, while the finalists are meant to be
    # diverse. Every family gets its first pick before any family gets a second,
    # and within a rank the cheaper point wins, so the small end is still
    # preferred where there is a choice.
    by_family: dict[str, list] = {}
    for name, entry in chosen.items():
        by_family.setdefault(entry["fromStudy"]["family"], []).append((name, entry))
    for entries in by_family.values():
        entries.sort(key=lambda item: item[1]["fromStudy"]["deployedBytes"])
    trimmed: list = []
    rank = 0
    while len(trimmed) < limit and any(len(e) > rank for e in by_family.values()):
        rung = [entries[rank] for entries in by_family.values() if len(entries) > rank]
        rung.sort(key=lambda item: item[1]["fromStudy"]["deployedBytes"])
        for item in rung:
            if len(trimmed) >= limit:
                break
            trimmed.append(item)
        rank += 1
    return dict(trimmed)


def confirm(named: dict, problem: Problem, seeds=(1729, 20260912, 31337),
            steps: int = 5000, batch: int = 8192, report_test: bool = True,
            store_precision: str = DEFAULT_STORE_PRECISION) -> dict:
    """Retrain chosen configurations across seeds and read the test pages once.

    Everything before this point may look at the selection half as often as it
    likes. This function is the only one that touches the test half and the
    extrapolation block, and it does so after the choices are fixed. The seed
    spread is the result, not an error bar on a number that was already decided,
    so a favourable single seed cannot be mistaken for a result.
    """
    evaluate_on = ("train", "selection", "test", "extrapolation") if report_test \
        else ("train", "selection")
    floor_mae = constant_baseline(problem, evaluate_on)
    rows = []
    for name, entry in named.items():
        config, recipe = entry["config"], entry["recipe"]
        runs = []
        for seed in seeds:
            result = measure(config, training.with_overrides(
                Recipe(**{**recipe, "steps": steps, "batch": batch, "device": problem.device}),
                seed=seed), problem, evaluate_on=evaluate_on,
                store_precision=store_precision)
            runs.append({k: v for k, v in result.items() if k not in ("model", "history")})
            if problem.device.startswith("cuda"):
                problem.torch.cuda.empty_cache()
        summary = {}
        for split_name in evaluate_on:
            for metric in ("mae_m", "max_m", "p99_m"):
                values = [run["metrics"][split_name][metric] for run in runs]
                summary[f"{split_name}.{metric}"] = {
                    "median": float(np.median(values)), "min": float(min(values)),
                    "max": float(max(values)),
                    "spreadFraction": float((max(values) - min(values)) / max(np.median(values), 1e-12))}
        row = {"name": name, "config": describe(config), "recipe": recipe,
               "seeds": list(seeds), "deployedBytes": runs[0]["deployedBytes"],
               "storePrecision": store_precision,
               "deployedBytesFloat32": runs[0]["deployedBytesFloat32"],
               "bitsPerSample": runs[0]["bitsPerSample"],
               "perSeed": runs, "summary": summary}
        # Did it train at all? A row at the constant-predictor floor is a failed
        # optimisation, and ranking it against architectures that did converge
        # reads as a finding about its family when it is a finding about its
        # hyperparameters.
        row["convergence"] = {
            split: convergence(summary[f"{split}.mae_m"]["median"], floor_mae["bySplit"][split]["maeM"])
            for split in evaluate_on if f"{split}.mae_m" in summary}
        # `trained` is about the optimisation, so it reads only the splits the model
        # was fitted against. Extrapolation is a different property: a model can
        # fit its domain perfectly and still diverge outside it (every non-residual
        # family does here), and folding it into one boolean would report a
        # converged network as a failed one.
        fitted = [name for name in ("train", "selection", "test") if name in row["convergence"]]
        row["trained"] = all(row["convergence"][name]["trained"] for name in fitted)
        if "extrapolation" in row["convergence"]:
            row["extrapolates"] = row["convergence"]["extrapolation"]["trained"]
        # A residual row cannot be read without its base, so the base's own
        # contribution is attached here rather than left to a separate table.
        # Otherwise most of the base's accuracy reads as a residual win.
        if config.get("kind") == "residual":
            control = base_row(problem, config["base_level"], config["base_target_m"])
            row["baseContribution"] = base_contribution(row, control,
                                                        tuple(s for s in ("selection", "test")
                                                              if s in evaluate_on))
        rows.append(row)
    return {"schema": SCHEMA, "mode": "multi-seed confirmation", "seeds": list(seeds),
            "steps": steps, "batch": batch, "storePrecision": store_precision, "rows": rows,
            "constantBaseline": floor_mae,
            "nonConvergent": [r["name"] for r in rows if not r["trained"]],
            "doesNotExtrapolate": [r["name"] for r in rows if r.get("extrapolates") is False],
            "extrapolationNote": "A model that fits its pages and fails here has not failed to train; "
                                 "it has no anchor outside the coordinates it saw. Only the families "
                                 "carrying a conventional base stay bounded.",
            "qualification": "Test-half and extrapolation errors here were read once, after the "
                             "configurations were fixed on the selection half. The selection column "
                             "is retained so selection-to-test drift is visible rather than implied."}


def predict_field(model, problem: Problem, shared: bool, chunk: int = 262_144) -> np.ndarray:
    """Decode the whole lattice, for full-reference and hydrological comparison."""
    from geoneural.neural.learning import features
    torch = problem.torch
    out = np.empty(problem.flat.size, dtype=np.float64)
    model.eval()
    with torch.inference_mode():
        for begin in range(0, problem.flat.size, chunk):
            indexes = np.arange(begin, min(begin + chunk, problem.flat.size), dtype=np.int64)
            coords, tiles = features(indexes, problem.side, problem.intervals, shared)
            decoded = model(torch.from_numpy(coords).to(problem.device),
                            torch.from_numpy(tiles).to(problem.device)).squeeze(-1)
            out[begin:begin + indexes.size] = decoded.to(torch.float32).cpu().numpy().astype(np.float64)
    return (out * problem.scale + problem.mean).reshape(problem.side, problem.side)


def half_precision(model, problem: Problem) -> tuple[int, dict]:
    """Cast every stored tensor to float16 and report the real cost and damage.

    Halving the stored weights halves the rate, the cheapest lever in this
    study, but only if the error it adds is small, which is measured here rather
    than assumed. The cast is applied to the stored tensors and then restored,
    so the caller's model is unchanged.
    """
    torch = problem.torch
    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        for tensor in model.state_dict().values():
            if tensor.is_floating_point():
                tensor.copy_(tensor.to(torch.float16).to(tensor.dtype))
    halved = {k: (v.to(torch.float16) if v.is_floating_point() else v)
              for k, v in model.state_dict().items()}
    from safetensors.torch import save
    stored = len(save({k: v.detach().cpu().contiguous() for k, v in halved.items()}))
    return stored, original


# Storage dtypes worth asking about, widest first. A "-scaled" suffix divides each
# tensor by its own maximum magnitude before casting and multiplies back on load,
# which costs one float32 per tensor and is the only way the eight-bit formats are
# a real option: a SIREN's hidden weights sit near 0.007 and float8_e4m3fn's
# smallest normal is 0.0156, so an unscaled cast pushes most of the model into
# subnormals. Both forms are measured, because the unscaled row is what shows why
# the scale is needed.
#
# int8 is absent on purpose. It is a fixed-point format needing a zero point as
# well as a scale and different rounding, so it belongs in the quantised ladder
# as its own measurement rather than as another row here.
PRECISIONS = ("float64", "float32", "bfloat16", "float16",
              "float8_e4m3fn", "float8_e4m3fn-scaled", "float8_e5m2", "float8_e5m2-scaled")


def precision_sweep(model, problem: Problem, shared: bool,
                    precisions=PRECISIONS) -> dict:
    """What each storage precision costs and what it damages.

    The cast is applied to the stored tensors and the model then computes in its
    original dtype, because that is what deployment does: weights ship at the
    stored width and are widened on load. Activations are untouched, so nothing
    here is a runtime or memory claim; checkpoint storage is not a smaller active
    model.

    Call this on a model stored at float32. `measure` rounds to its deployed
    precision before returning, so a model taken from a float16 run has already
    lost those bits and every row of the ladder would inherit the loss.

    float64 is included as a control. Widening weights that were trained in
    float32 is exactly lossless, so it doubles the bytes for an error change of
    zero, which shows the ladder is measuring storage and not something else. A
    model trained in float64 is a different question this does not answer.
    """
    import torch
    from safetensors.torch import save
    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    rows = []
    for name in precisions:
        scaled = name.endswith("-scaled")
        dtype = getattr(torch, name[:-len("-scaled")] if scaled else name)
        scale_count = 0
        with torch.no_grad():
            for key, tensor in model.state_dict().items():
                if not tensor.is_floating_point():
                    continue
                source = original[key]
                if scaled:
                    peak = source.abs().max()
                    limit = torch.finfo(dtype).max
                    factor = (limit / peak) if peak > 0 else torch.ones_like(peak)
                    tensor.copy_((source * factor).to(dtype).to(tensor.dtype) / factor)
                    scale_count += 1
                else:
                    tensor.copy_(source.to(dtype).to(tensor.dtype))
        stored = {k: (v.to(dtype) if v.is_floating_point() else v)
                  for k, v in model.state_dict().items()}
        size = len(save({k: v.detach().cpu().contiguous() for k, v in stored.items()}))
        # One float32 per scaled tensor, charged rather than waved away.
        scale_bytes = 4 * scale_count
        field = predict_field(model, problem, shared)
        errors = np.abs(field - problem.reference)
        rows.append({"precision": name, "weightsBytes": size + scale_bytes,
                     "tensorBytes": size, "scaleBytes": scale_bytes,
                     "perTensorScale": scaled,
                     "bitsPerSample": 8.0 * (size + scale_bytes) / problem.flat.size,
                     "mae_m": float(errors.mean()),
                     "p99_m": float(np.quantile(errors, 0.99)),
                     "max_m": float(errors.max())})
    model.load_state_dict(original)
    baseline = next((r for r in rows if r["precision"] == "float32"), rows[0])
    for row in rows:
        row["maeChangeFromFloat32M"] = row["mae_m"] - baseline["mae_m"]
        row["bytesRelativeToFloat32"] = row["weightsBytes"] / max(baseline["weightsBytes"], 1)
    return {"rows": rows, "baseline": "float32",
            "note": "Stored-tensor precision only. Weights are widened back for compute and activations "
                    "are never quantised, so no runtime or resident-memory claim follows.",
            "int8": "Absent on purpose: it needs per-tensor scales and zero points which are themselves "
                    "payload, so it belongs in the quantised ladder as its own measurement rather than "
                    "as another row."}


def ladder(named: dict, problem: Problem, seed: int = 1729, steps: int = 5000,
           batch: int = 8192, precisions=PRECISIONS, regions=(1, 16, 256, 4096)) -> dict:
    """The storage-precision ladder, code coverage and amortisation, per candidate.

    The ladder is re-run against whatever the current search chose, because the
    float16 finding decides the deployed precision and must not be inherited from
    an old checkpoint.

    Trains at float32 on every page, because the ladder needs an unrounded model:
    `measure` rounds to its deployed precision before returning, and every row of
    a ladder taken from a float16 run would inherit that loss.
    """
    torch = problem.torch
    rows = []
    for name, entry in named.items():
        config, recipe = entry["config"], entry["recipe"]
        result = measure(config, Recipe(**{**recipe, "steps": steps, "batch": batch,
                                           "device": problem.device, "seed": seed}),
                         problem, evaluate_on=("all",), limit=None, train_on="all",
                         store_precision="float32")
        model, shared = result["model"], result["shared"]
        sweep = precision_sweep(model, problem, shared, precisions)
        coverage = code_coverage(model, problem, shared)
        found = code_parameter(model)
        stored_codes = int(found[1].numel()) if found is not None else 0
        # Codes are per region; everything else is paid once however many regions
        # there are. A hybrid's conventional base is per region too.
        code_bytes = 2 * stored_codes  # float16, the width these now ship at
        shared_bytes = max(result["weightsBytes"] - code_bytes, 0)
        spread = amortisation(shared_bytes, code_bytes, int(result["baseBytes"]), regions)
        rows.append({"name": name, "config": describe(config), "recipe": recipe,
                     "deployedBytesFloat32": result["deployedBytesFloat32"],
                     "storedCodeParameters": stored_codes,
                     "codeCoverage": coverage, "precisionLadder": sweep,
                     "amortisation": spread, "trainingSeconds": result["trainingSeconds"]})
        del result, model
        if problem.device.startswith("cuda"):
            torch.cuda.empty_cache()
    return {"schema": SCHEMA, "mode": "precision ladder, code coverage and amortisation",
            "seed": seed, "steps": steps, "batch": batch, "rows": rows,
            "qualification": "Trained on every page, so the errors are codec fit and not "
                             "generalisation. Storage precision only: weights are widened back for "
                             "compute and activations are never quantised, so no runtime or "
                             "resident-memory claim follows. Amortisation is arithmetic over a "
                             "measured single-region cost, not a measurement on an acquired "
                             "multi-region corpus."}


def codec_fit(named: dict, problem: Problem, seeds=(1729,), steps: int = 5000,
              batch: int = 8192, stream_cells: int = 500,
              store_precision: str = "float32", fields_dir: Path | None = None) -> dict:
    """Train on every page and judge the result as a codec, drainage included.

    A codec's job is to reproduce what it encoded, so this trains without a
    holdout on purpose and the resulting error is codec fit, not generalization.
    It is the only comparison that belongs beside the conventional rate-distortion
    curve, and it is reported with the drainage metric, because a lower mean error
    that breaks a critical outlet is a rejection, not a win.
    """
    from geoneural.metrics import hydrology
    torch = problem.torch
    rows = []
    for name, entry in named.items():
        config, recipe = entry["config"], entry["recipe"]
        runs = []
        for seed in seeds:
            # Stored at float32 on purpose: this function's own float16 round-trip
            # below is the comparison, and it would be a no-op against a model that
            # had already been rounded. Both rates are reported per run.
            result = measure(config, Recipe(**{**recipe, "steps": steps, "batch": batch,
                                               "device": problem.device, "seed": seed}),
                             problem, evaluate_on=("all",), limit=None, train_on="all",
                             store_precision=store_precision)
            model = result["model"]
            field = predict_field(model, problem, result["shared"])
            if fields_dir is not None:
                # The decoded heights behind the reported numbers, for figures and the viewer.
                Path(fields_dir).mkdir(parents=True, exist_ok=True)
                stem = f"fit--{name}" if len(seeds) == 1 else f"fit--{name}-s{seed}"
                np.save(Path(fields_dir) / f"{stem}.npy", field.astype(np.float32), allow_pickle=False)
            errors = np.abs(field - problem.reference)
            drainage = hydrology.compare(problem.reference, field, problem.spacing_m, stream_cells)
            fp16_bytes, original = half_precision(model, problem)
            fp16_field = predict_field(model, problem, result["shared"])
            fp16_errors = np.abs(fp16_field - problem.reference)
            model.load_state_dict(original)
            runs.append({
                "seed": seed,
                "deployedBytes": result["deployedBytes"], "baseBytes": result["baseBytes"],
                "bitsPerSample": result["bitsPerSample"],
                "fullReference": {
                    "mae_m": float(errors.mean()), "rmse_m": float(np.sqrt((errors ** 2).mean())),
                    "p95_m": float(np.quantile(errors, 0.95)), "p99_m": float(np.quantile(errors, 0.99)),
                    "max_m": float(errors.max())},
                "halfPrecision": {
                    "deployedBytes": fp16_bytes + result["baseBytes"],
                    "bitsPerSample": 8.0 * (fp16_bytes + result["baseBytes"]) / problem.flat.size,
                    "mae_m": float(fp16_errors.mean()), "max_m": float(fp16_errors.max()),
                    "maeChangeM": float(fp16_errors.mean() - errors.mean())},
                "drainage": {k: drainage[k] for k in (
                    "streamJaccard", "streamRecall", "receiverAgreementFraction",
                    "basinAgreementFraction", "referenceBasins", "reconstructedBasins",
                    "reconstructedFilledCells", "referenceFilledCells", "bySlopeClass")},
                "trainingSeconds": result["trainingSeconds"],
            })
            del result, model, field
            if problem.device.startswith("cuda"):
                torch.cuda.empty_cache()
        rows.append({"name": name, "config": describe(config), "recipe": recipe, "runs": runs})
    return {"schema": SCHEMA, "mode": "codec fit with drainage",
            "streamThresholdCells": stream_cells, "rows": rows,
            "qualification": "Trained on every page, so these errors are codec fit on the encoded "
                             "region and are NOT evidence of generalization. Drainage is a routing "
                             "comparison, not a hydrological simulation. No neural row carries an "
                             "error bound; the conventional rows meet theirs by construction.",
            "interpretation": "This is the comparison that belongs beside the conventional "
                              "rate-distortion curve. Read the maximum-error and drainage columns "
                              "before the mean: a lower MAE that loses the stream network is a "
                              "rejection under the drainage-preservation check."}


def dominance(deployed_bytes: int, mae_m: float, points, max_m: float | None = None) -> dict:
    """Is this point on the joint Pareto front, or does something already beat it?

    A ratio against "the best conventional point at or below my rate" is the wrong
    test, because the conventional front is a staircase: the error is set by the
    pyramid level and the bytes by the quantization target, so within a level three
    times the bytes buys three per cent of the error. A network landing in a gap
    between levels always shows a flattering ratio, which says more about where
    the levels fall than about the network.

    Dominance is the correct test. A point survives only if no conventional point
    is both no dearer and no worse. For the ones that survive, the two neighbours
    state the claim: how much cheaper than the nearest thing that beats it, and
    how much better than the nearest thing that undercuts it.

    `points` are `(bytes, mae_m, max_m, label)` tuples. When `max_m` is given, a
    dominator must also be no worse on the maximum, the column the
    drainage-preservation check reads.
    """
    on_mean = [p for p in points if p[0] <= deployed_bytes and p[1] <= mae_m]
    beaten = [p for p in on_mean if max_m is None or p[2] <= max_m]
    cheaper = [p for p in points if p[0] < deployed_bytes]
    better = [p for p in points if p[1] < mae_m]
    nearest_better = min(better, key=lambda p: p[0]) if better else None
    best_cheaper = min(cheaper, key=lambda p: p[1]) if cheaper else None

    def _at(point):
        return {"label": point[3], "deployedBytes": point[0], "maeM": point[1], "maxM": point[2]}

    verdict = {"deployedBytes": int(deployed_bytes), "maeM": float(mae_m),
               "dominated": bool(beaten),
               # Kept separate because they are different claims. A point that is
               # dearer and no more accurate, but has a lower maximum, is still on
               # the joint front (the maximum is read too), yet saying it is
               # "cheaper than the nearest better point" would be false.
               "dominatedOnMeanAndRate": bool(on_mean),
               "survivesOnMaximumOnly": bool(on_mean) and not beaten,
               "dominatedBy": _at(beaten[0]) if beaten else None,
               "dominatedOnMeanBy": _at(min(on_mean, key=lambda p: p[1])) if on_mean else None}
    if not beaten and not on_mean:
        if nearest_better:
            verdict["cheaperThanNearestBetter"] = {
                **_at(nearest_better),
                "fraction": 1.0 - deployed_bytes / max(nearest_better[0], 1)}
        if best_cheaper:
            verdict["betterThanBestCheaper"] = {
                **_at(best_cheaper),
                "fraction": 1.0 - mae_m / max(best_cheaper[1], 1e-12)}
    verdict["note"] = ("A ratio against the best point at or below this rate is not a claim of "
                       "dominance: the conventional front is a staircase in pyramid level, so a "
                       "point sitting in a gap shows a good ratio without beating anything.")
    return verdict


def constant_baseline(problem: Problem, splits=("train", "selection", "test", "extrapolation", "all")) -> dict:
    """What predicting one number everywhere already achieves, per split.

    The floor every model must clear to have learned anything at all. A network
    that converges to the field mean scores the mean absolute deviation, which on
    this region is 23.5 m, so a 23.28 m row is not a weak architecture but an
    optimisation that did not start. Ranking such a row beside converged ones
    reads as a finding about its family when it is not: `shared` collapses at its
    screening defaults yet reaches 0.915 m in the codec search once its
    hyperparameters are sampled.

    The mean is taken over the whole field, because that is the single scalar a
    deployed model would ship, and reported per split.
    """
    whole = float(problem.flat.mean())
    rows = {}
    for name in splits:
        values = problem.flat[problem.indexes[name]] if name in problem.indexes else problem.flat
        rows[name] = {"maeM": float(np.abs(values - whole).mean()),
                      "rmseM": float(np.sqrt(((values - whole) ** 2).mean())),
                      "samples": int(values.size)}
    return {"constantM": whole, "bySplit": rows,
            "method": "predict the whole-field mean at every coordinate",
            "purpose": "the floor a model must clear to have learned anything; a row at or above "
                       "this did not train and is not an architecture result"}


# A model has to remove this fraction of the constant predictor's error before its
# number is read as an architecture comparison at all. Two per cent is deliberately
# permissive: it is a liveness check, not a quality bar.
CONVERGENCE_SKILL_FLOOR = 0.02


def convergence(mae_m: float, baseline_mae_m: float,
                floor: float = CONVERGENCE_SKILL_FLOOR) -> dict:
    """Did this row train? Skill against the constant predictor, and a verdict.

    `skill` is the fraction of the constant predictor's error removed: 1.0 is
    perfect, 0.0 is the constant itself, negative is worse than predicting one
    number. Rows below the floor carry `trained: false` and must be reported as
    non-convergent rather than as a slow architecture.
    """
    skill = 1.0 - mae_m / max(baseline_mae_m, 1e-12)
    return {"baselineMaeM": baseline_mae_m, "maeM": mae_m, "skill": skill,
            "trained": bool(skill > floor), "skillFloor": floor,
            "note": "skill is the fraction of the constant-predictor error removed. A row with "
                    "trained false did not converge and is not a statement about its architecture."}


def base_row(problem: Problem, level: int, target: float) -> dict:
    """The base-only control for one `(level, target)`: the single-point form.

    `base_only_curve` sweeps the grid for the report's front. This evaluates the
    one base a given residual config actually stands on, so the control can be
    attached to that row instead of living in a separate table the reader has to
    join by hand. A residual row whose base beats it is a negative result, however
    good its byte count looks.
    """
    import torch
    from geoneural.neural.models import ResidualDecoder
    from geoneural.neural.learning import features

    class _Silent(torch.nn.Module):
        def forward(self, coords, tiles=None):
            return torch.zeros(coords.shape[:-1] + (1,), dtype=coords.dtype)

    decoded, base_bytes, record = problem.base(level, target)
    normalised = (decoded - problem.mean) / problem.scale
    model = ResidualDecoder(torch.from_numpy(normalised.astype(np.float32)), _Silent())
    metrics = {name: training.evaluate(model, features, problem.indexes[name], problem.flat,
                                       problem.side, problem.intervals, False, problem.mean,
                                       problem.scale, "cpu", torch, None)
               for name in ("train", "selection", "test", "extrapolation", "all")}
    return {"baseLevel": int(level), "baseTargetM": float(target), "baseSide": record["side"],
            "deployedBytes": int(base_bytes), "metrics": metrics}


def spent_better_on_the_base(row: dict, control: dict, better_bases, split: str = "selection") -> dict:
    """Could the same bytes have bought a better plain grid?

    The question `base_contribution` asks (is base+network better than this
    base?) is the weak form. A residual model at B total bytes is competing with
    every conventional grid costing at most B, including grids finer than its own
    base, and if one of those beats it then the network has not merely failed to
    earn its bytes, the bytes were misallocated.

    `better_bases` are `(bytes, mae_m, label)` for base-only points on the same
    split. Returns the best one that beats the combined model, or None.
    """
    total = int(row["deployedBytes"])
    combined = float((row.get("summary") or {})[f"{split}.mae_m"]["median"])
    affordable = [b for b in better_bases if b[0] <= total and b[1] < combined]
    best = min(affordable, key=lambda b: b[1]) if affordable else None
    return {"split": split, "totalBytes": total, "combinedMaeM": combined,
            "bestAffordableBase": ({"deployedBytes": best[0], "maeM": best[1], "label": best[2]}
                                   if best else None),
            "bytesWereWellSpent": best is None,
            "note": "A residual model competes with every plain grid it could have bought for the "
                    "same total bytes, not only with the one it happens to sit on. A model with "
                    "bytesWereWellSpent false would have been better off as a bigger grid."}


def base_contribution(row: dict, control: dict, splits=("selection", "test")) -> dict:
    """What the network added on top of its base, in metres and in bytes.

    Signed so that a negative delta is an improvement. A residual model whose
    delta is positive spent its network bytes making the answer worse, and only
    the comparison against its own base reveals that; comparing it with other
    networks does not.
    """
    combined = row.get("summary") or {}
    deltas = {}
    for split in splits:
        key = f"{split}.mae_m"
        if key not in combined:
            continue
        neural = float(combined[key]["median"])
        base = float(control["metrics"][split]["mae_m"])
        deltas[split] = {"baseMaeM": base, "combinedMaeM": neural,
                         "deltaM": neural - base, "improves": neural < base}
    network_bytes = int(row["deployedBytes"]) - int(control["deployedBytes"])
    improved = [d["improves"] for d in deltas.values()]
    # The strict inequality above is far too weak on its own: a model can spend
    # 29,160 bytes to improve its base by 0.0003 m and pass. A deployment compares
    # against the best base it could buy for the same total bytes, which is what
    # `spent_better_on_the_base` checks.
    return {"control": {k: v for k, v in control.items() if k != "metrics"},
            "controlMetrics": {s: control["metrics"][s] for s in splits if s in control["metrics"]},
            "networkBytes": network_bytes, "bySplit": deltas,
            "networkEarnsItsBytes": bool(improved) and all(improved),
            "note": "deltaM is combined minus base on the same split: negative means the network "
                    "helped. networkBytes is what the network cost over shipping the base alone. "
                    "A row with networkEarnsItsBytes false is a conventional coarse grid that has "
                    "been made worse and more expensive by a network."}


def annotate_convergence(report: dict, problem: Problem, split: str = "all") -> dict:
    """Flag every search trial that never cleared the constant predictor.

    Roughly one trial in six collapses to the field mean on this region, across
    siren, fourier and shared alike. A collapsed trial is not a problem for the
    search (the sampler is meant to explore and be rejected), but it is for
    anything downstream that counts trials or reads a family's worst points, and
    for the screening defaults, one of which (`shared`) is itself a collapsed
    configuration.
    """
    floor = constant_baseline(problem, (split,))
    floor_mae = floor["bySplit"][split]["maeM"]
    summary = {}
    for family, study_report in report.get("byFamily", {}).items():
        collapsed = 0
        for trial in study_report.get("trials", []):
            metrics = trial.get("metrics") or {}
            if split not in metrics:
                continue
            verdict = convergence(float(metrics[split]["mae_m"]), floor_mae)
            trial["convergence"] = verdict
            collapsed += not verdict["trained"]
        scored = sum(1 for t in study_report.get("trials", []) if (t.get("metrics") or {}).get(split))
        study_report["nonConvergentTrials"] = collapsed
        study_report["scoredTrials"] = scored
        summary[family] = {"scored": scored, "nonConvergent": collapsed}
    report["constantBaseline"] = floor
    report["convergenceByFamily"] = summary
    report["convergenceNote"] = (
        "A trial whose error matches the constant predictor never left the field mean. These are "
        "counted, not discarded: a family's completed-trial count overstates its effective budget "
        "by the collapsed fraction.")
    return report


def annotate_residual_fronts(report: dict, problem: Problem, split: str = "all") -> dict:
    """Attach each residual/hybrid front point's own base, in place.

    A search front reports one row per Pareto point, and for the residual arms
    every one of those rows may stand on a different base, because the search
    samples `base_level` and `base_target_m` per trial. Joining them by hand
    against `conventional.coarse` means re-deriving which coarse row each point
    used, so it is done here, once, keyed on the trial's own parameters.

    Bases are evaluated once per distinct `(level, target)` and reused, which
    keeps this to a handful of evaluations rather than one per Pareto point.
    """
    cache: dict[tuple[int, float], dict] = {}
    for family, study_report in report.get("byFamily", {}).items():
        for point in study_report.get("paretoFront", []):
            params = point.get("params", {})
            if "base_level" not in params or "base_target_m" not in params:
                continue
            key = (int(params["base_level"]), float(params["base_target_m"]))
            if key not in cache:
                cache[key] = base_row(problem, *key)
            control = cache[key]
            base_mae = float(control["metrics"][split]["mae_m"])
            neural_mae = float(point["scoreMaeM"])
            point["baseContribution"] = {
                "baseLevel": key[0], "baseTargetM": key[1],
                "baseBytes": control["deployedBytes"],
                "networkBytes": int(point["deployedBytes"]) - control["deployedBytes"],
                "split": split, "baseMaeM": base_mae, "combinedMaeM": neural_mae,
                "deltaM": neural_mae - base_mae,
                "baseMaxM": float(control["metrics"][split]["max_m"]),
                "networkEarnsItsBytes": neural_mae < base_mae}
    report["residualBaseNote"] = (
        "Every residual and hybrid Pareto point carries the coarse base it stands on. deltaM is "
        "combined minus base on the scored split: negative means the network helped. A point with "
        "networkEarnsItsBytes false is a conventional grid made worse by a network, whatever its "
        "ratio against the envelope says.")
    return report


def base_only_curve(problem: Problem, levels=(1, 2, 3, 4),
                    targets=(0.05, 0.5, 1.0, 4.0)) -> dict:
    """What the conventional coarse grid achieves with no network at all.

    This is the control the residual family cannot be read without. That family
    predicts `base + network`, and if the base alone already reaches the same
    error then the network adds nothing and the result is a statement about
    bilinear upsampling of a cheap grid. Reported on the same splits so the
    comparison is direct.

    Bilinear upsampling is the same reconstruction the atlas declares for its own
    parent levels, so this is the conventional multiscale codec answering the
    query the network is asked.
    """
    import torch
    from geoneural.neural.models import ResidualDecoder

    class _Silent(torch.nn.Module):
        def forward(self, coords, tiles=None):
            return torch.zeros(coords.shape[:-1] + (1,), dtype=coords.dtype)

    from geoneural.neural.learning import features
    rows = []
    for level in levels:
        for target in targets:
            # `Problem.base` returns the paged byte count as its second value. The
            # single-stream figure is on the record, so it is read from there;
            # dividing the paged figure by itself would make
            # randomAccessOverheadFraction zero.
            decoded, base_bytes, record = problem.base(level, target)
            monolithic_bytes = int(record["monolithicBytes"])
            normalised = (decoded - problem.mean) / problem.scale
            model = ResidualDecoder(torch.from_numpy(normalised.astype(np.float32)), _Silent())
            metrics = {name: training.evaluate(model, features, problem.indexes[name], problem.flat,
                                               problem.side, problem.intervals, False, problem.mean,
                                               problem.scale, "cpu", torch, None)
                       for name in ("train", "selection", "test", "extrapolation", "all")}
            rows.append({"baseLevel": level, "baseTargetM": target, "baseSide": record["side"],
                         "deployedBytes": base_bytes, "monolithicBytes": monolithic_bytes,
                         "randomAccessOverheadFraction": base_bytes / max(monolithic_bytes, 1) - 1.0,
                         "bitsPerSample": 8.0 * base_bytes / problem.flat.size,
                         "metrics": metrics})
    return {"rows": rows,
            "rateConvention": "deployedBytes is independently compressed pages, the same convention "
                              "conventional_curve uses, so the two fronts are charged alike. "
                              "monolithicBytes is the single-stream figure for reference.",
            "method": "the atlas's own page data at one pyramid level, re-encoded at a declared "
                      "max-error target and reconstructed bilinearly, the reconstruction the "
                      "manifest already declares for parent levels",
            "purpose": "the control for the residual family: a hybrid whose base alone matches its "
                       "combined error has learned nothing worth its weights",
            "note": "Errors on held-out splits are not held out in any meaningful sense here; the "
                    "base is conventional data covering the whole region. They are listed so the "
                    "residual family's holdout columns can be compared against the right control."}


def conventional_envelope(problem: Problem, targets=CONVENTIONAL_TARGETS,
                          levels=(1, 2, 3, 4), split: str = "all") -> dict:
    """The full conventional front a neural codec has to beat, both axes of it.

    Sweeping only the error target at full resolution does not cover the whole
    conventional envelope. Dropping resolution and reconstructing bilinearly is
    an ordinary conventional way to spend fewer bytes, and at the rates a small
    network occupies it is the stronger competitor. Leaving it out makes a neural
    result look better than it is, so both axes are swept here and merged into
    one front.

    The two axes fail differently, so the front is reported per metric.
    Full-resolution quantization keeps a hard maximum-error guarantee and pays for
    it in bytes; downsampling gets a far better mean for the same bytes and gives
    the guarantee up entirely. Which one wins depends on which column is read.
    """
    # Both axes are swept to the coarse end: see ENVELOPE_TARGETS and COARSE_TARGETS.
    # An envelope that stops at the declared guarantees is not the envelope a
    # sub-bit-per-sample model actually competes against.
    full = conventional_curve(problem, ENVELOPE_TARGETS)
    # Deliberately not `targets`: see COARSE_TARGETS. Sweeping the downsample axis
    # only over the declared accuracy targets prices every coarse level as though
    # it had to carry a fine guarantee, and removes the conventional competition
    # at exactly the rates a small network occupies.
    coarse = base_only_curve(problem, levels, COARSE_TARGETS)
    # Both arms read the same split, and the caller says which. A holdout search
    # scores the selection half; a codec search scores everything. The selection
    # half is easier than the whole field by 0.07 to 0.50 m, so setting a
    # selection-half neural number against a whole-field conventional one would
    # credit the network with a property of the terrain.
    if split not in problem.indexes:
        raise ValueError(f"Unknown split for the conventional envelope: {split}")
    points = [{"kind": "full-resolution-quantized", "label": f"q32-delta-zstd @ {row['targetMaxErrorM']} m",
               "deployedBytes": row["perPageFinestBytes"],
               "maeM": row["metrics"][split]["mae_m"],
               "maxM": row["metrics"][split]["max_m"], "boundGuaranteed": True}
              for row in full["rows"]]
    points += [{"kind": "downsampled-bilinear",
                "label": f"level {row['baseLevel']} @ {row['baseTargetM']} m",
                "deployedBytes": row["deployedBytes"], "maeM": row["metrics"][split]["mae_m"],
                "maxM": row["metrics"][split]["max_m"], "boundGuaranteed": False}
               for row in coarse["rows"]]
    points.sort(key=lambda point: point["deployedBytes"])

    def front(metric: str) -> list[dict]:
        best, out = float("inf"), []
        for point in points:
            if point[metric] < best:
                best = point[metric]
                out.append(point)
        return out

    return {"full": full, "coarse": coarse, "points": points, "split": split,
            "paretoByMae": front("maeM"), "paretoByMax": front("maxM"),
            "splitNote": "Every point is the error on the named split. Compare a neural score only "
                         "against a front read on the same split: the selection half is easier than "
                         "the whole field, so mixing them moves the answer by up to 0.5 m in the "
                         "network's favour.",
            "note": "paretoByMax contains only full-resolution rows wherever a guarantee matters: a "
                    "downsampled grid has no bound on the detail it discarded, and neither does a "
                    "neural field.",
            "qualification": "Rate and error on the frozen prepared reference. Not accuracy against "
                             "the ground, not IO cost, not whole-renderer behaviour."}


def code_coverage(model, problem: Problem, shared: bool, chunk: int = 131_072,
                  train_on: str = "train") -> dict | None:
    """How much of a model's local-code payload the training region can reach.

    A decoder with per-region codes stores a parameter per region; regions
    withheld from training never receive a gradient, so their codes stay at
    initialisation and a holdout error for that family measures the initialiser,
    not the architecture.

    This runs one backward pass over the whole training index set and counts the
    code entries that actually received gradient. A family reporting a holdout
    number with coverage below one is reporting something else, and an
    interpolated code grid is expected to score higher than per-tile vectors
    because its nodes are shared across neighbouring patches.

    `train_on` must name the split the model was actually fitted on. A codec-fit
    model trains on every page, so measuring its coverage against the training
    split alone would understate it.

    The feature unpacker comes from the problem for the same reason `measure` takes
    it from there: a multi-region problem addresses several atlases through one
    index space, and the single-region unpacker would compute wrong tile indices
    for it rather than fail.
    """
    from geoneural.neural.learning import features as _lattice_features
    features = getattr(problem, "features_fn", None) or _lattice_features
    torch = problem.torch
    # One place decides what a code payload is, and it matches wrapped decoders,
    # so a code grid carried by a residual base (the preferred design) is found.
    found = code_parameter(model)
    if found is None:
        return None
    parameter = found[1]
    model.zero_grad(set_to_none=True)
    indexes = problem.indexes[train_on]
    for begin in range(0, indexes.size, chunk):
        block = indexes[begin:begin + chunk]
        coords, tiles = features(block, problem.side, problem.intervals, shared)
        out = model(torch.from_numpy(coords).to(problem.device),
                    torch.from_numpy(tiles).to(problem.device))
        out.sum().backward()
    # Flatten every leading dimension so one "entry" is one code vector, whether
    # the payload is an embedding table (tiles, latent) or a 2-D node grid
    # (nodes, nodes, latent). Counting rows of a grid would report 17 entries for
    # 289 nodes and turn a partial coverage into a perfect one.
    gradient = parameter.grad
    if gradient is None:
        return {"codeEntries": int(parameter.reshape(-1, parameter.shape[-1]).shape[0]),
                "entriesReachedByTraining": 0, "coverageFraction": 0.0,
                "note": "No gradient reached this code payload at all."}
    flat = gradient.reshape(-1, gradient.shape[-1])
    reached = int((flat.abs().sum(-1) > 0).sum())
    total = int(flat.shape[0])
    model.zero_grad(set_to_none=True)
    # Two different failures share one number. A code entry can be unreached
    # because the training split never visits its region, or because the model's
    # own modulation path contributes nothing, so the gradient is zero even where
    # training does visit. A coverage of 0.0 on a split that visits every region
    # of the code grid is a dead conditioning path (the network is its base), not
    # a withheld-region problem. Reporting visited entries separately tells the
    # two apart.
    visited = _visited_code_entries(model, problem, shared, train_on, chunk)
    return {
        "codeEntries": total, "entriesReachedByTraining": reached,
        "coverageFraction": reached / max(total, 1),
        "entriesVisitedByTraining": visited,
        "visitedFraction": None if visited is None else visited / max(total, 1),
        # No entry reached by a training set that visits hundreds of thousands of
        # nodes cannot be a withheld-region effect: a node grid's entries are
        # shared between neighbouring cells, and an embedding table addressed by
        # a visited tile must receive gradient. Zero means the path contributes
        # nothing to the output at all.
        "deadModulationPath": bool(reached == 0 and problem.indexes[train_on].size > 0),
        "measuredOn": train_on,
        "note": "Code entries the training region can influence, measured over the split the model "
                "was actually fitted on. Below 1.0, the holdout error of this family is partly the "
                "initialiser and is not an architecture comparison. visitedFraction counts entries "
                "the training indexes address at all: visited high with reached zero means the "
                "modulation path is dead, not that regions were withheld.",
    }


def _visited_code_entries(model, problem: Problem, shared: bool, train_on: str,
                          chunk: int) -> int | None:
    """How many code entries the training indexes address, ignoring the network.

    Counted from the tile indices the feature unpacker produces, so it is a fact
    about the split and the lookup and cannot be changed by what the decoder does
    with a code. Only defined for families whose codes are addressed by tile; a
    node-grid payload (`liif`, `codegrid`) is addressed by interpolation over
    several entries at once, so this returns None rather than a wrong integer.
    """
    import numpy as _np
    found = code_parameter(model)
    if found is None:
        return None
    name, parameter = found
    inner = getattr(model, "inner", model)
    if not hasattr(inner, "codes") or not hasattr(getattr(inner, "codes"), "num_embeddings"):
        return None  # a node grid, not an embedding table
    from geoneural.neural.learning import features as _lattice_features
    features = getattr(problem, "features_fn", None) or _lattice_features
    indexes = problem.indexes[train_on]
    seen: set[int] = set()
    for begin in range(0, indexes.size, chunk):
        block = indexes[begin:begin + chunk]
        _, tiles = features(block, problem.side, problem.intervals, shared)
        seen.update(_np.unique(tiles).tolist())
    return len(seen)


def code_parameter(model):
    """The local-code payload of a decoder, or None if it has no codes.

    Matched on the suffix, because a code-carrying decoder is often wrapped: the
    preferred architecture is a coarse base carrying a modulated code grid, where
    the parameter is `inner.codes` rather than `codes`. Matching only the
    top-level name would return None for that model and disable `code_coverage`
    on it.
    """
    for name, parameter in model.named_parameters():
        if name == "codes" or name.endswith(".codes") \
                or name == "codes.weight" or name.endswith(".codes.weight"):
            return name, parameter
    return None


def fit_codes(model, problem: Problem, region: str, recipe: Recipe,
              shared: bool, steps: int | None = None) -> dict:
    """Freeze the decoder and optimise only the codes on one region.

    This is what an encoder does, and without it a local-code family cannot be
    read on a held-out split at all: `code_coverage` shows only 0.375 of a
    per-tile payload is reached during training, so the rest of the field is
    being reconstructed from the initialiser.

    It is deliberately not a generalization experiment. The encoder has the
    region's own heights and is compressing them; the codes it produces are
    payload and are counted. This measures whether the shared decoder is a good
    codec given properly fitted codes. Whether codes can be predicted for a
    region whose fine detail was never seen is a different experiment needing a
    code predictor, and this function cannot answer it.

    The decoder is frozen rather than fine-tuned, because a decoder that adapted
    to each region would no longer be shared and its bytes would no longer
    amortise.
    """
    from geoneural.neural.learning import features
    torch = problem.torch
    found = code_parameter(model)
    if found is None:
        raise ValueError("this model has no code payload to fit")
    name, codes = found
    frozen = []
    for parameter_name, parameter in model.named_parameters():
        if parameter_name != name:
            frozen.append((parameter, parameter.requires_grad))
            parameter.requires_grad_(False)
    before = codes.detach().clone()

    indexes = problem.indexes[region]
    budget = recipe.steps if steps is None else steps
    optimiser = torch.optim.Adam([codes], lr=recipe.lr)
    rng = np.random.default_rng(recipe.seed ^ 0xC0DE)
    model.train()
    start = time.perf_counter()
    for iteration in range(budget):
        for group in optimiser.param_groups:
            group["lr"] = _schedule_for(recipe, iteration, budget)
        picked = rng.choice(indexes, min(recipe.batch, indexes.size), replace=False)
        coords, tiles = features(picked, problem.side, problem.intervals, shared)
        truth = torch.from_numpy(
            ((problem.flat[picked] - problem.mean) / problem.scale).astype(np.float32)).to(problem.device)
        predicted = model(torch.from_numpy(coords).to(problem.device),
                          torch.from_numpy(tiles).to(problem.device)).squeeze(-1)
        loss = (predicted - truth).square().mean()
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
    for parameter, was in frozen:
        parameter.requires_grad_(was)
    moved = int((codes.detach() - before).abs().reshape(-1, codes.shape[-1]).sum(-1).gt(0).sum())
    total = int(codes.detach().reshape(-1, codes.shape[-1]).shape[0])
    return {
        "region": region, "steps": budget, "seconds": time.perf_counter() - start,
        "codeParameter": name, "codeEntries": total, "entriesMoved": moved,
        "coverageAfterFitting": moved / max(total, 1),
        "finalLoss": float(loss.detach().cpu()),
        "note": "Decoder frozen; only the code payload was optimised, on the region's own heights. "
                "This is encode-time code fitting, which is what a codec does. It is NOT evidence of "
                "generalization to a region whose detail was never seen; that needs a code predictor.",
    }


def _schedule_for(recipe: Recipe, iteration: int, budget: int) -> float:
    """Cosine decay over an arbitrary budget, reusing the recipe's rate."""
    if recipe.schedule != "cosine":
        return recipe.lr
    return recipe.lr * 0.5 * (1.0 + math.cos(math.pi * min(iteration / max(budget, 1), 1.0)))


def amortisation(shared_bytes: int, per_region_bytes: int, base_bytes_per_region: int = 0,
                 regions=(1, 16, 256, 4096)) -> dict:
    """Total deployed bytes as a corpus grows, which is where sharing is decided.

    This is the only accounting in which a shared decoder can be judged: a model
    that loses on one 10.24 km tile may win over a corpus, and a family with
    large per-region codes can lose the other way. A single region's total is not
    the rate.

    `shared_bytes` is paid once however many regions there are: decoder weights,
    and a code predictor if one is used. `per_region_bytes` is paid per region:
    stored codes, and zero for a predictor. `base_bytes_per_region` is the
    conventional payload each region ships regardless; a hybrid must be charged
    for it.
    """
    rows = []
    for count in regions:
        total = shared_bytes + count * (per_region_bytes + base_bytes_per_region)
        rows.append({"regions": count, "totalBytes": total,
                     "bytesPerRegion": total / count,
                     "sharedFraction": shared_bytes / max(total, 1)})
    return {"sharedBytes": shared_bytes, "perRegionBytes": per_region_bytes,
            "baseBytesPerRegion": base_bytes_per_region, "byRegionCount": rows,
            "note": "A per-region cost dominates any shared cost eventually, so the interesting "
                    "question is where the crossover sits relative to the corpus actually intended, "
                    "not whether sharing helps in the limit."}


def amortisation_crossover(first: dict, second: dict, limit: int = 1 << 20) -> dict:
    """The region count at which one accounting overtakes the other.

    Closed form where it exists: with totals `a0 + n*a1` and `b0 + n*b1`, they
    cross at `n = (b0 - a0) / (a1 - b1)` when the per-region costs differ. Equal
    per-region costs never cross, and that is reported as such rather than as a
    very large number.
    """
    slope = first["perRegionBytes"] + first["baseBytesPerRegion"] \
        - (second["perRegionBytes"] + second["baseBytesPerRegion"])
    gap = second["sharedBytes"] - first["sharedBytes"]
    if slope == 0:
        cheaper = "first" if first["sharedBytes"] < second["sharedBytes"] else (
            "second" if second["sharedBytes"] < first["sharedBytes"] else "identical")
        return {"crosses": False, "reason": "equal per-region cost; the ranking never changes",
                "cheaperAtEveryCount": cheaper}
    at = gap / slope
    # Below one region there is no corpus size at which the ranking differs, so
    # this is not a crossover however positive the arithmetic is (a crossover at
    # 0.65 regions has no meaning).
    if at < 1.0 or at > limit:
        cheaper = "first" if (first["sharedBytes"] + first["perRegionBytes"]
                              + first["baseBytesPerRegion"]
                              <= second["sharedBytes"] + second["perRegionBytes"]
                              + second["baseBytesPerRegion"]) else "second"
        return {"crosses": False,
                "reason": f"the totals meet at {at:.2f} regions, outside the range 1..{limit} where a "
                          f"corpus can actually sit",
                "crossoverRegions": at, "cheaperAtEveryCount": cheaper}
    return {"crosses": True, "crossoverRegions": at,
            "interpretation": "below this count the arrangement with the smaller shared cost wins; "
                              "above it the one with the smaller per-region cost does"}


def context_ablation(problem: Problem, arms: dict, config: dict, recipe: Recipe,
                     seeds=(1729, 20260912, 31337),
                     store_precision: str = DEFAULT_STORE_PRECISION,
                     base: tuple[int, float] | None = None) -> dict:
    """Geology conditioning ablation: the same architecture on each context arm,
    across seeds.

    The architecture, the recipe and the seeds are identical in every arm; only
    the raster differs. That is why `none` ships a single-class raster rather
    than no raster: an arm that also changed the code path would confound the
    information with the plumbing.

    Deployed bytes are weights plus the context raster. A conditioning gain that
    costs more context bytes than it saves in weights is a net loss, and it
    cannot be seen at all if the raster is free.

    `base` makes every arm predict a correction to a conventional coarse grid
    rather than the whole field: `(level, target_m)` selects the grid, each arm
    becomes `ResidualDecoder(base, ContextDecoder(...))`, and the base is charged
    to every arm alike. The residual form is the only one here where a network
    has stayed bounded outside its training footprint, and conditioning a
    correction is a different question from conditioning a field: most of the
    signal a whole-field decoder must spend capacity on is the regional trend,
    which the base already carries, so whatever geology can explain is a larger
    share of what is left.

    `baseOnly` is computed and attached rather than left for the reader to join:
    an arm whose base alone matches it has measured bilinear upsampling, not
    conditioning.
    """
    rows = []
    level = target = None
    base_control = None
    if base is not None:
        level, target = int(base[0]), float(base[1])
        base_control = base_row(problem, level, target)
    for name in sorted(arms):
        arm = arms[name]
        classes = np.asarray(arm["classes"])
        runs = []
        for seed in seeds:
            inner_config = {**config, "classes": classes,
                            "class_count": int(classes.max()) + 1}
            arm_config = (inner_config if base is None else
                          {"kind": "residual", "base_level": level,
                           "base_target_m": target, "inner": inner_config})
            result = measure(arm_config, training.with_overrides(recipe, seed=seed), problem,
                             evaluate_on=("train", "selection", "test", "extrapolation"),
                             store_precision=store_precision)
            run = {k: v for k, v in result.items() if k not in ("model", "history")}
            if isinstance(run.get("config"), dict):
                run["config"] = describe(run["config"])
            runs.append(run)
            if problem.device.startswith("cuda"):
                problem.torch.cuda.empty_cache()
        summary = {}
        for split in ("train", "selection", "test", "extrapolation"):
            for metric in ("mae_m", "max_m"):
                values = [run["metrics"][split][metric] for run in runs]
                summary[f"{split}.{metric}"] = {
                    "median": float(np.median(values)), "min": float(min(values)),
                    "max": float(max(values)),
                    "spreadFraction": float((max(values) - min(values))
                                            / max(np.median(values), 1e-12))}
        # weightsBytes and baseBytes are kept apart because they answer different
        # questions. The conditioning claim is about the weights; the base is a
        # constant charged identically to every arm, and folding it into one
        # figure would let a large base hide a small difference between arms.
        weight_bytes = int(runs[0]["weightsBytes"])
        base_bytes = int(runs[0].get("baseBytes", 0))
        row = {"arm": name, "contextBytes": int(arm["bytes"]),
               "weightBytes": weight_bytes, "baseBytes": base_bytes,
               "deployedBytes": weight_bytes + base_bytes + int(arm["bytes"]),
               "distinctClasses": int(arm.get("distinctClasses", 0)),
               "perSeed": runs, "summary": summary}
        if base_control is not None:
            for split in ("selection", "test", "extrapolation"):
                base_mae = float(base_control["metrics"][split]["mae_m"])
                row[f"baseOnly.{split}.mae_m"] = base_mae
                row[f"beatsItsBase.{split}"] = bool(
                    summary[f"{split}.mae_m"]["median"] < base_mae)
        rows.append(row)
    by_arm = {row["arm"]: row for row in rows}
    verdict = None
    if {"real", "misaligned", "none"} <= set(by_arm):
        real = by_arm["real"]["summary"]["test.mae_m"]
        control = by_arm["misaligned"]["summary"]["test.mae_m"]
        spread = max(real["max"] - real["min"], control["max"] - control["min"])
        gain = control["median"] - real["median"]
        verdict = {
            "gainOverMisalignedM": float(gain),
            "seedSpreadM": float(spread),
            "exceedsSeedSpread": bool(gain > spread),
            "extraBytesOverNone": int(by_arm["real"]["deployedBytes"]
                                      - by_arm["none"]["deployedBytes"]),
            "note": "The claim is that real geology helps because it is geology. The comparison "
                    "that supports it is against misaligned: identical shapes, sizes, class "
                    "frequencies and compressed size, only the registration broken. A gain smaller "
                    "than the seed spread is not a gain."}
    if base_control is not None and verdict is not None:
        # A residual arm competes with every conventional grid it could have
        # bought instead, not only with the one it stands on. Without this the
        # arm could beat its own base, lose to a cheaper finer grid, and still
        # read as a win.
        affordable = [(r["deployedBytes"], r["metrics"]["test"]["mae_m"],
                       f"L{r['baseLevel']}@{r['baseTargetM']}")
                      for r in base_only_curve(problem)["rows"]]
        verdict["spentBetterOnTheBase"] = spent_better_on_the_base(
            by_arm["real"], base_control, affordable, split="test")
        verdict["baseOnlyTestMaeM"] = float(base_control["metrics"]["test"]["mae_m"])
    return {"schema": SCHEMA, "mode": "context ablation", "seeds": list(seeds),
            "rows": rows, "verdict": verdict,
            "base": ({"level": level, "targetM": target, "control": base_control}
                     if base_control is not None else None),
            "conditioningForm": ("whole field" if base is None else
                                 "correction to a conventional coarse base"),
            "qualification": "Same architecture, recipe and seeds in every arm; only the context "
                             "raster differs. Deployed bytes include the raster. "
                             "`shuffled` is not byte-matched (it costs about thirteen times real "
                             "because permutation destroys compressibility), so it is a diagnostic "
                             "for per-node class reading, not a rate comparison."}


EQUIVALENCE_SCHEMA = "geoneural-data-path-equivalence-v1"


def equivalence(problem: Problem, steps: int = 5000, batch: int = 8192,
                seeds=(1729, 20260912, 31337),
                store_precision: str = DEFAULT_STORE_PRECISION) -> dict:
    """Is the device path the same experiment as the host path, or a different one?

    Moving the lattice onto the GPU changes the batch stream, and with-replacement
    sampling changes the estimator. Both are inside the seed spread by argument
    (see `device_data`); this measures it instead, because "inside the noise" is
    a claim about numbers.

    Three families, chosen for their kernel classes rather than their scores: a
    dense backbone (`siren`), an interpolated node grid with `index_put_`
    (`codegrid`), and an embedding table (`shared`). Three seeds each, three arms.

    The test is one-sided: the median across seeds must move by less than the
    larger of the two arms' own seed spreads. A device path that shifted a family
    further than reseeding it would be a different experiment and would have to
    be reported as one.
    """
    import numpy as _np
    arms = (("host", "without", "eager"),
            ("device", "without", "eager"),
            ("device", "with", "cudagraph"))
    families = {
        "siren": {"kind": "siren", "width": 128, "depth": 3,
                  "omega": 30.0, "hidden_omega": 30.0},
        "codegrid": {"kind": "codegrid", "patches": 16, "latent": 16,
                     "width": 128, "depth": 3, "omega": 30.0},
        "shared": {"kind": "shared", "width": 128, "depth": 3, "latent": 16, "omega": 30.0},
    }
    rows = []
    for family, config in families.items():
        by_arm = {}
        for path, sampling, engine_mode in arms:
            values, seconds = [], []
            for seed in seeds:
                recipe = Recipe(steps=steps, batch=batch, device=problem.device, seed=seed,
                                data_path=path, sampling=sampling)
                if config["kind"] == "shared":
                    config = {**config, "tiles": problem.tiles}
                result = measure(config, recipe, problem, evaluate_on=("selection",),
                                 store_precision=store_precision, limit=None,
                                 engine_mode=engine_mode)
                values.append(result["metrics"]["selection"]["mae_m"])
                seconds.append(result["trainingSeconds"])
                del result
                if problem.device.startswith("cuda"):
                    problem.torch.cuda.empty_cache()
            by_arm[f"{path}/{sampling}/{engine_mode}"] = {
                "perSeed": values, "medianMaeM": float(_np.median(values)),
                "spreadM": float(max(values) - min(values)),
                "medianTrainingSeconds": float(_np.median(seconds))}
        reference = by_arm["host/without/eager"]
        for name, arm in by_arm.items():
            if name == "host/without/eager":
                continue
            shift = abs(arm["medianMaeM"] - reference["medianMaeM"])
            tolerance = max(arm["spreadM"], reference["spreadM"])
            arm["shiftFromHostM"] = float(shift)
            arm["toleranceM"] = float(tolerance)
            arm["withinSeedSpread"] = bool(shift < tolerance)
            arm["speedup"] = float(reference["medianTrainingSeconds"]
                                   / max(arm["medianTrainingSeconds"], 1e-9))
        rows.append({"family": family, "byArm": by_arm})
    verdict = {
        "allWithinSeedSpread": all(arm["withinSeedSpread"] for row in rows
                                   for name, arm in row["byArm"].items()
                                   if name != "host/without/eager"),
        "note": "A device-path study is a different study identity from a host-path one whatever "
                "this says: the stream changed. What this establishes is whether it is also a "
                "different RESULT. If every arm sits inside the seed spread, the two paths measure "
                "the same thing and host-path fronts stay readable beside device-path ones; if any "
                "does not, the paths must be reported as separate experiments and no host-path "
                "number may be quoted against a device-path one."}
    return {"schema": EQUIVALENCE_SCHEMA, "seeds": list(seeds), "steps": steps, "batch": batch,
            "storePrecision": store_precision, "rows": rows, "verdict": verdict,
            "qualification": "Training seconds here are contended only by this process, but they "
                             "are not a qualified timing measurement and are reported as a ratio "
                             "between arms run back to back on the same host, nothing more."}
