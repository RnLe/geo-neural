"""Several regions under one decoder, and what a new region costs.

A planetary storage estimate that multiplies one region's measured bytes by the
number of regions on Earth assumes the regions are independent. A shared decoder
tests that assumption. If a backbone trained on several regions lets a new one
arrive carrying only its own codes, the per-region cost falls towards the code
size and the estimate changes by the backbone/code ratio. If it does not, the
estimate stands.

The experiment is built so it cannot prefer either outcome.

Two design choices matter and are easy to get wrong:

Per-region normalisation, charged. The six NRW regions sit between 15 m and
818 m. Normalised globally, the dominant signal across the dataset is which region
a sample came from: a constant per region, learnable by any architecture, and
unrelated to terrain. A shared decoder would then show a large apparent gain for
storing six numbers. So each region is normalised by its own mean and scale, and
those 2N scalars are counted as deployed side information in
`normalisation_bytes`. They are a small but real cost.

Region-local coordinates. Every region maps to the same [-1, 1]^2. A decoder
with no per-region parameters therefore sees N contradictory targets at identical
inputs and can do no better than their pointwise mean. This is the control: it
measures how much of a region is unpredictable from position alone, and it is the
floor the shared arm has to clear.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from geoneural.common import read_json, sha_file
from geoneural.neural import splits


def features(flat: np.ndarray, side: int, intervals: int, shared: bool):
    """`learning.features`, with the region unpacked out of the global index.

    A global index is `region * side**2 + local`, so the region falls out by
    division and nothing extra has to be threaded through `fit` or `evaluate`.
    Tile indices are offset by region, which is what gives a shared decoder one
    code per (region, tile) while the backbone stays common.

    Coordinates are region-local by design; see the module docstring.
    """
    area = side * side
    region = flat // area
    local = flat % area
    row, col = local // side, local % side
    per_side = (side - 1) // intervals
    tx = np.minimum(col // intervals, per_side - 1)
    ty = np.minimum(row // intervals, per_side - 1)
    tiles = (region * per_side * per_side + ty * per_side + tx).astype(np.int64)
    if shared:
        coords = np.column_stack([(col - tx * intervals) / intervals,
                                  (row - ty * intervals) / intervals]) * 2 - 1
    else:
        coords = np.column_stack([col / (side - 1), row / (side - 1)]) * 2 - 1
    return coords.astype(np.float32), tiles


class MultiRegionProblem:
    """Several atlases addressed as one index space, sharing a decoder.

    Mirrors `search.Problem` closely enough for `search.measure` to run against
    it unchanged: a multi-region arm that went through a different training or
    evaluation path would not be comparable with the single-region numbers it is
    read against.
    """

    #: Every region must agree on these; a decoder cannot be shared across
    #: lattices that do not line up, and silently resampling one to match would
    #: make the comparison a statement about the resampler.
    MUST_MATCH = ("page_intervals", "spacing_m")

    def __init__(self, atlas_paths, device: str = "cuda",
                 extrapolation_fraction: float = 0.25,
                 normalisation: str = "train", holdout_region: str | None = None):
        import torch
        self.torch = torch
        self.device = device
        self.atlas_paths = [Path(p) for p in atlas_paths]
        if len(self.atlas_paths) < 2:
            raise ValueError("A multi-region problem needs at least two regions")
        if normalisation not in ("all", "train"):
            raise ValueError(f"Unknown normalisation scope: {normalisation}")
        self.normalisation = normalisation
        self.features_fn = features

        self.names, references, manifests = [], [], []
        for path in self.atlas_paths:
            manifest = read_json(path)
            reference_path = path.parent / "reference.npy"
            if sha_file(reference_path) != manifest["reference_sha256"]:
                raise ValueError(f"Reference changed since the atlas was built: {path}")
            references.append(np.load(reference_path, allow_pickle=False).astype(np.float64))
            manifests.append(manifest)
            self.names.append(path.parent.name)
        self.manifests = manifests
        self.manifest = manifests[0]

        sides = {r.shape[0] for r in references}
        if len(sides) != 1:
            raise ValueError(f"Regions disagree on lattice side: {sorted(sides)}")
        self.side = int(sides.pop())
        for key in self.MUST_MATCH:
            values = {m[key] for m in manifests}
            if len(values) != 1:
                raise ValueError(f"Regions disagree on {key}: {sorted(values)}")
        self.intervals = int(self.manifest["page_intervals"])
        self.spacing_m = float(self.manifest["spacing_m"])
        self.regions = len(references)
        self.area = self.side * self.side
        self.tiles = self.regions * ((self.side - 1) // self.intervals) ** 2

        # One split per region, built with the same rule and the same seedless
        # geometry, then shifted into the global index space. Holding the split
        # geometry identical across regions means a per-region difference in
        # error is a difference in terrain, not in which pages were withheld.
        self.split_by_region = [
            splits.build(self.side, self.intervals, extrapolation_fraction,
                         selection_split=True)
            for _ in references]
        self.split = dict(self.split_by_region[0])
        self.split["regions"] = self.names
        self.split["note"] = (
            "One geographic split per region, identical in geometry, concatenated into a "
            "global index space. Region membership is never a split: every region "
            "contributes training, selection, test and extrapolation pages, so a shared "
            "decoder is asked to generalise within each region it has seen. Holding out a "
            "whole region is a different question and is `holdout_region`.")

        self.holdout_region = holdout_region
        self.holdout_index = (self.names.index(holdout_region)
                              if holdout_region is not None else None)

        # Per-region normalisation, charged. See the module docstring.
        self.means, self.scales = [], []
        flats = []
        for reference, split in zip(references, self.split_by_region):
            flat = reference.reshape(-1)
            source = flat if normalisation == "all" \
                else flat[np.flatnonzero(split["trainMask"].reshape(-1))]
            mean = float(np.mean(source))
            scale = max(float(np.std(source)), 1.0)
            self.means.append(mean)
            self.scales.append(scale)
            flats.append((flat - mean) / scale)
        # `fit` and `evaluate` apply `(value - mean) / scale` with the single pair
        # they are given, so the regions are pre-normalised here and the pair
        # handed on is the identity. Anything else would silently renormalise
        # already-normalised values.
        self.flat = np.concatenate(flats)
        self.reference_metres = np.concatenate([r.reshape(-1) for r in references])
        self.mean = 0.0
        # Errors are reported in metres. Every region was divided by its own scale,
        # so there is no single multiplier that returns metres for all of them; the
        # scales differ by 30x across this panel. `scale` is set to 1.0 so that
        # `fit` and `evaluate` do not apply a wrong one, and `metres_by_region`
        # converts properly, per region.
        self.scale = 1.0

        self.indexes = {}
        for name, key in (("train", "trainMask"), ("selection", "selectionMask"),
                          ("test", "testMask"), ("extrapolation", "extrapolationMask")):
            parts = []
            for region, split in enumerate(self.split_by_region):
                if self.holdout_index is not None and region == self.holdout_index \
                        and name == "train":
                    continue  # the withheld region contributes no training samples
                parts.append(np.flatnonzero(split[key].reshape(-1)) + region * self.area)
            self.indexes[name] = np.concatenate(parts)
        self.indexes["all"] = np.arange(self.flat.size, dtype=np.int64)
        for region, name in enumerate(self.names):
            self.indexes[f"region:{name}"] = (
                np.flatnonzero(self.split_by_region[region]["selectionMask"].reshape(-1))
                + region * self.area)

    def normalisation_bytes(self, store_precision: str = "float16") -> int:
        """The 2N stored scalars, at the width everything else is priced at."""
        width = {"float64": 8, "float32": 4, "float16": 2}[store_precision]
        return 2 * self.regions * width

    def region_of(self, indexes: np.ndarray) -> np.ndarray:
        return indexes // self.area

    def metres_by_region(self, predicted_normalised: np.ndarray,
                         indexes: np.ndarray) -> np.ndarray:
        """Absolute error in metres, undoing each region's own normalisation."""
        region = self.region_of(indexes)
        scale = np.asarray(self.scales)[region]
        mean = np.asarray(self.means)[region]
        predicted_m = predicted_normalised * scale + mean
        return np.abs(predicted_m - self.reference_metres[indexes])

    def summary(self) -> dict:
        return {
            "regions": self.names,
            "side": self.side, "intervals": self.intervals, "spacingM": self.spacing_m,
            "tiles": self.tiles, "samplesPerRegion": self.area,
            "holdoutRegion": self.holdout_region,
            "perRegion": [
                {"name": name, "meanM": mean, "scaleM": scale,
                 "reliefM": float(np.ptp(self.reference_metres[i * self.area:(i + 1) * self.area]))}
                for i, (name, mean, scale) in enumerate(zip(self.names, self.means, self.scales))],
            "normalisation": {
                "scope": self.normalisation, "perRegion": True,
                "storedScalars": 2 * self.regions,
                "why": "Globally normalised, the dominant variance across a 15 m-to-818 m panel is "
                       "which region a sample came from: a constant per region that any "
                       "architecture learns and that says nothing about terrain. Per-region "
                       "scalars remove that confound and are charged as deployed side "
                       "information rather than treated as free.",
            },
        }


#: `training.evaluate` chunk size, matched so the two paths behave alike.
EVAL_CHUNK = 262_144


def evaluate_by_region(model, problem: MultiRegionProblem, indexes: np.ndarray,
                       torch) -> dict:
    """Errors in true metres, per region and pooled.

    `training.evaluate` cannot be used here, and would not fail if it were. It
    converts predictions with the single `(mean, scale)` pair it is handed, and a
    multi-region problem has no such pair: the scales across this panel differ by
    about thirty times. Called with `problem.mean`/`problem.scale` it would return
    errors in normalised units under field names ending `_m`.

    Pooled figures are sample-weighted, and every region contributes the same
    number of samples, so the pooled mean is the unweighted mean across regions.
    A roughness panel spanning 2.7 m to 88.6 m of standard deviation will be
    dominated by its roughest member on any absolute metric. Read the per-region
    rows; the pooled row is for decomposition, not to be quoted alone.
    """
    was_training = model.training
    model.eval()
    errors = np.empty(indexes.size, dtype=np.float64)
    with torch.inference_mode():
        for begin in range(0, indexes.size, EVAL_CHUNK):
            block = indexes[begin:begin + EVAL_CHUNK]
            coords, tiles = problem.features_fn(block, problem.side, problem.intervals,
                                                _is_shared(model))
            out = model(torch.from_numpy(coords).to(problem.device),
                        torch.from_numpy(tiles).to(problem.device))
            predicted = out.squeeze(-1).detach().to(torch.float32).cpu().numpy().astype(np.float64)
            errors[begin:begin + block.size] = problem.metres_by_region(predicted, block)
    if was_training:
        model.train()
    if not np.all(np.isfinite(errors)):
        raise FloatingPointError(
            f"{int((~np.isfinite(errors)).sum())} of {errors.size} predictions were not finite")

    def stats(values: np.ndarray) -> dict:
        return {"samples": int(values.size), "mae_m": float(values.mean()),
                "rmse_m": float(np.sqrt(np.mean(values ** 2))),
                "p95_m": float(np.percentile(values, 95)),
                "p99_m": float(np.percentile(values, 99)),
                "max_m": float(values.max())}

    region = problem.region_of(indexes)
    by_region = {problem.names[r]: stats(errors[region == r])
                 for r in np.unique(region)}
    return {"pooled": stats(errors), "byRegion": by_region,
            "note": "Per-region metres, each undoing that region's own normalisation. The pooled "
                    "row is sample-weighted and is dominated by the roughest region; read the "
                    "per-region rows."}


def _is_shared(model) -> bool:
    """Whether this model wants region-local tile coordinates and a tile index."""
    from geoneural.neural.models import SharedDecoder
    inner = getattr(model, "inner", model)
    return isinstance(inner, SharedDecoder) or isinstance(model, SharedDecoder)


def _freeze_backbone(model, torch) -> dict:
    """Everything but the code table stops learning.

    The transfer arm asks what a new region costs once a backbone exists. That
    only means something if the backbone does not move: a run that let it drift
    would be measuring joint training on N regions and calling it transfer.
    Sampling is restricted to the new region as well, so the other regions' code
    rows receive no gradient either. Embedding gradients are indexed, so this
    needs no masking, but the sampling restriction must hold; the returned counts
    are how that is checked.
    """
    trainable = frozen = 0
    for name, parameter in model.named_parameters():
        if name.startswith("codes"):
            parameter.requires_grad_(True)
            trainable += parameter.numel()
        else:
            parameter.requires_grad_(False)
            frozen += parameter.numel()
    if trainable == 0:
        raise ValueError("Nothing left trainable: this model has no code table to fit")
    return {"trainableParameters": trainable, "frozenParameters": frozen}


#: The conventional axis the multi-region comparison is read against. It reaches
#: far coarser than the accuracy targets because a shared decoder over six regions
#: will not land near a metre, and an axis that stops short of where the neural
#: arm sits would manufacture a win for it.
REGION_TARGETS = (0.05, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)


def conventional_curve(problem: MultiRegionProblem, targets=REGION_TARGETS) -> dict:
    """The conventional control swept across targets, not pinned at one.

    A single target cannot answer whether this error could have been bought more
    cheaply conventionally unless the neural arm happens to land on it, and it
    will not. The curve lets the verdict look up the cheapest conventional point
    that guarantees an error the neural arm merely achieved, which is how neural
    and conventional points are compared throughout the package.
    """
    return {"targets": [conventional_per_region(problem, t) for t in targets],
            "note": "Each row is every region encoded alone at one max-error target. The codec's "
                    "error is a guarantee; the neural arms' is a measurement, and the comparison "
                    "is deliberately in the codec's favour on that point."}


def conventional_per_region(problem: MultiRegionProblem, target_m: float) -> dict:
    """The conventional codec on each region at a declared max-error target.

    Computed the same way as everywhere else in the package (independently
    compressed pages), so the rate convention matches the other rate-distortion
    comparisons. Its total is the number the shared arm has to come in under, and
    it amortises nothing across regions, which is the property under test.
    """
    from geoneural.codecs import codecs
    from geoneural.neural.search import paged_bytes
    codec = codecs.registry()["q32-delta-zstd"]
    rows = []
    for region, name in enumerate(problem.names):
        grid = problem.reference_metres[region * problem.area:(region + 1) * problem.area]
        grid = grid.reshape(problem.side, problem.side).astype(np.float32)
        measured = codecs.measure(codec, grid, target_m)
        rows.append({"region": name,
                     "bytes": int(paged_bytes(codec, grid, target_m, problem.intervals)),
                     "monolithicBytes": int(measured["bytes"]),
                     "maxErrorM": float(measured["max_error_m"])})
    return {"targetM": float(target_m), "codec": "q32-delta-zstd", "perRegion": rows,
            "totalBytes": int(sum(r["bytes"] for r in rows)),
            "worstMaxErrorM": float(max(r["maxErrorM"] for r in rows)),
            "amortisation": "none by construction: each region is encoded alone. This is the "
                            "quantity the shared arm must beat, and the reason it might: a codec "
                            "cannot reuse anything it learned about one region on the next.",
            "rateConvention": "independently compressed pages, matching the other rate-distortion "
                              "comparisons"}


SCHEMA = "geoneural-multiregion-v1"


#: Why the codec objective trains on every page of every training region.
#:
#: A decoder with one local code per page cannot be read on a page holdout. On the
#: standard page split a `shared` model covers only 37.5 % of its codes; the rest
#: never receive a gradient, so most of its reported error is the code initialiser
#: and it collapses to the constant predictor. That is a property of the split,
#: not of the architecture, and a multi-region run on the same split would measure
#: two collapsed models agreeing with each other.
#:
#: The question here is whether a backbone amortises across REGIONS, so the region
#: is the held-out unit and every page within a training region is trained. That
#: makes the within-region numbers codec fit, and they are labelled as such
#: everywhere they appear. They are not generalisation and must not be quoted as
#: such.
OBJECTIVES = {
    "codec": {"trainOn": "all", "scoreOn": "all",
              "question": "Does a backbone shared across regions cost less than one model per region, "
                          "at the same within-region fit? Every page of every training region is "
                          "trained, so local codes are fully covered and the comparison is between "
                          "architectures rather than between initialisers.",
              "isNotGeneralisation": "Within-region error here is codec fit. The only held-out unit is "
                                     "the transfer region."},
    "holdout": {"trainOn": "train", "scoreOn": "selection",
                "question": "Does a backbone shared across regions generalise to unseen pages within "
                            "each region?",
                "isNotGeneralisation": None,
                "warning": "Any family with one local code per page is structurally unreadable here; "
                           "check codeCoverageFraction before believing a number from this objective."},
}


def amortisation(atlas_paths, config: dict, recipe, device: str = "cuda",
                 store_precision: str = "float16", conventional_target_m: float = 1.0,
                 transfer_region: str | None = None, seeds=(1729,),
                 objective: str = "codec") -> dict:
    """Does a decoder shared across regions cost less than N separate ones?

    Four arms:

    * `independent`: N separate models, the same architecture each. This is what
      a per-region planetary estimate assumes.
    * `shared`: one backbone, one code per (region, tile). If a backbone
      amortises, this is cheaper at the same error.
    * `transfer`: the backbone from `shared` trained WITHOUT one region, then
      frozen while only that region's codes are fitted. Its marginal bytes are
      what a new region costs once a backbone exists, which is the number a
      planetary estimate should multiply.
    * `conventional`: the codec on each region alone. It amortises nothing, and
      it is the control every other arm has to beat.

    A shared decoder can look good here for a reason that is not amortisation:
    regions differ in roughness by about thirty times across this panel, so a
    pooled error is mostly the roughest region. Every arm is therefore reported
    per region as well as pooled, and the comparison that counts is per region at
    matched error.
    """
    from geoneural.neural.search import measure, describe, code_coverage
    from geoneural.neural import training

    if objective not in OBJECTIVES:
        raise ValueError(f"Unknown multi-region objective: {objective}")
    spec = OBJECTIVES[objective]
    train_on, score_on = spec["trainOn"], spec["scoreOn"]
    # "all" normalisation is permitted for a codec objective and required to be
    # stored; it is two scalars per region and they are charged. A holdout
    # objective may not use it, because a withheld page must not reach the model
    # even through a mean.
    scope = "all" if objective == "codec" else "train"
    torch_problem = MultiRegionProblem(atlas_paths, device, normalisation=scope)
    torch = torch_problem.torch
    report = {"schema": SCHEMA, "mode": "multi-region amortisation",
              "objective": objective, "objectiveSpec": spec,
              "trainedOn": train_on, "scoredOn": score_on,
              "problem": torch_problem.summary(), "storePrecision": store_precision,
              "seeds": list(seeds), "arms": {}}

    # ---- conventional, first, so a partial report still carries its control ----
    report["arms"]["conventional"] = conventional_per_region(
        torch_problem, conventional_target_m)
    report["arms"]["conventionalCurve"] = conventional_curve(torch_problem)

    # ---- independent: one model per region -------------------------------------
    from geoneural.neural.search import Problem
    rows, total = [], 0
    for path, name in zip(torch_problem.atlas_paths, torch_problem.names):
        single = Problem(path, device, normalisation="train")
        single_config = dict(config)
        if single_config["kind"] == "shared":
            single_config["tiles"] = single.tiles
        result = measure(single_config, training.with_overrides(recipe, seed=seeds[0]), single,
                         evaluate_on=(score_on,), train_on=train_on,
                         store_precision=store_precision, limit=None)
        coverage = code_coverage(result["model"], single, result["shared"], train_on=train_on)
        rows.append({"region": name, "deployedBytes": int(result["deployedBytes"]),
                     "codeCoverageFraction": None if coverage is None
                     else coverage["coverageFraction"],
                     score_on: {k: result["metrics"][score_on][k]
                                for k in ("mae_m", "p99_m", "max_m")}})
        total += int(result["deployedBytes"])
        del result
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    scalars = torch_problem.normalisation_bytes(store_precision)
    report["arms"]["independent"] = {
        "perRegion": rows, "weightBytes": total,
        "normalisationBytes": scalars, "deployedBytes": total + scalars,
        "note": "N models with nothing in common. The per-region cost is the total divided by N "
                "and does not fall as regions are added, which is the assumption a per-region "
                "planetary estimate makes."}

    # ---- shared: one backbone, codes per (region, tile) -------------------------
    shared_config = {**config, "kind": "shared", "tiles": torch_problem.tiles}
    shared_result = measure(shared_config, training.with_overrides(recipe, seed=seeds[0]),
                            torch_problem, evaluate_on=(), train_on=train_on,
                            store_precision=store_precision, limit=None)
    shared_model = shared_result["model"]
    shared_eval = {score_on: evaluate_by_region(shared_model, torch_problem,
                                                torch_problem.indexes[score_on], torch)}
    shared_coverage = code_coverage(shared_model, torch_problem, True, train_on=train_on)
    code_bytes = _code_bytes(shared_model, store_precision)
    report["arms"]["shared"] = {
        "deployedBytes": int(shared_result["deployedBytes"]) + scalars,
        "weightBytes": int(shared_result["deployedBytes"]),
        "codeBytes": code_bytes,
        "backboneBytes": int(shared_result["deployedBytes"]) - code_bytes,
        "normalisationBytes": scalars,
        "config": describe(shared_config), "evaluation": shared_eval,
        "codeCoverageFraction": None if shared_coverage is None
        else shared_coverage["coverageFraction"],
        "note": "Codes and backbone are separated because only the backbone can amortise. The "
                "code table grows with every region; the backbone does not."}

    # ---- transfer: freeze the backbone, pay only for a new region's codes -------
    if transfer_region is not None:
        held = MultiRegionProblem(atlas_paths, device, normalisation=scope,
                                  holdout_region=transfer_region)
        pre_config = {**config, "kind": "shared", "tiles": held.tiles}
        pre = measure(pre_config, training.with_overrides(recipe, seed=seeds[0]), held,
                      evaluate_on=(), train_on=train_on, store_precision=store_precision,
                      limit=None)
        backbone = pre["model"]
        counts = _freeze_backbone(backbone, torch)
        before = {name: parameter.detach().clone()
                  for name, parameter in backbone.named_parameters()
                  if not name.startswith("codes")}
        # Fit ONLY the withheld region: its samples, its code rows, nothing else.
        # Every page of it under the codec objective, so the arriving region's codes
        # are fully covered and the marginal cost is a real cost rather than a
        # partly-untrained table.
        region_index = held.names.index(transfer_region)
        mask = ("all" if train_on == "all" else "trainMask")
        only = (np.arange(held.area, dtype=np.int64) if mask == "all"
                else np.flatnonzero(held.split_by_region[region_index]["trainMask"].reshape(-1)))
        only = only + region_index * held.area
        fit_history = training.fit(
            backbone, held.features_fn, held.flat, held.side, held.intervals, True,
            only, held.mean, held.scale, training.with_overrides(recipe, seed=seeds[0]),
            torch=torch, train_mask=None)
        drift = max(float((before[name] - parameter.detach()).abs().max())
                    for name, parameter in backbone.named_parameters()
                    if not name.startswith("codes"))
        training.round_to_storage(backbone, torch, store_precision)
        scored = (np.arange(held.area, dtype=np.int64) if score_on == "all"
                  else np.flatnonzero(
                      held.split_by_region[region_index]["selectionMask"].reshape(-1)))
        transfer_eval = evaluate_by_region(backbone, held, scored + region_index * held.area, torch)
        per_region_codes = _code_bytes(backbone, store_precision) // held.regions
        report["arms"]["transfer"] = {
            "region": transfer_region, **counts,
            "backboneDriftMax": drift,
            "backboneReallyFrozen": bool(drift == 0.0),
            "marginalBytesForANewRegion": per_region_codes + 2 * {"float64": 8, "float32": 4,
                                                                  "float16": 2}[store_precision],
            "evaluation": transfer_eval,
            "finalLoss": fit_history["history"][-1]["loss"],
            "conventionalBytesForThatRegion": next(
                r["bytes"] for r in report["arms"]["conventional"]["perRegion"]
                if r["region"] == transfer_region),
            "note": "Marginal bytes are this region's code rows plus its two normalisation "
                    "scalars: what arrives with a new region once the backbone exists. Compare "
                    "it against conventionalBytesForThatRegion, not against the shared total; "
                    "the backbone is already paid for, and a marginal cost is not comparable "
                    "with a total.",
        }
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    report["verdict"] = _verdict(report)
    report["qualification"] = (
        "One architecture and one recipe across every arm, one seed unless more are given. "
        "This measures whether amortisation exists for THIS configuration on THIS panel of six "
        "NRW regions at 10 m; it is not a tuned comparison and a negative result does not close "
        "the question for architectures not tried. Regions are all NRW DGM1 (same provider, "
        "datum, epoch and licence), so this isolates terrain from acquisition, and correspondingly "
        "says nothing about transfer across providers.")
    return report


def _code_bytes(model, store_precision: str) -> int:
    width = {"float64": 8, "float32": 4, "float16": 2}[store_precision]
    codes = getattr(model, "codes", None)
    if codes is None:
        return 0
    weight = getattr(codes, "weight", codes)
    return int(weight.numel()) * width


def _verdict(report: dict) -> dict:
    arms = report["arms"]
    conventional = arms["conventional"]["totalBytes"]
    target = arms["conventional"]["targetM"]
    independent = arms["independent"]["deployedBytes"]
    shared = arms["shared"]["deployedBytes"]

    # A byte comparison between arms at different errors is not a comparison: a
    # shared arm 45x cheaper than the conventional control and sixty times worse
    # reads as a win in the byte columns. So the error test is computed first and
    # every byte claim below is gated on it.
    scored_on = report.get("scoredOn", "selection")
    worst = max(row["max_m"]
                for row in arms["shared"]["evaluation"][scored_on]["byRegion"].values())
    matched = bool(worst <= target)
    out = {
        "errorsAreMatched": matched,
        "sharedWorstMaxErrorM": float(worst),
        "conventionalTargetM": float(target),
        "sharedBeatsIndependentOnBytes": bool(shared < independent),
        "bytesSavedBySharing": int(independent - shared),
        "sharingFraction": float(1.0 - shared / independent) if independent else None,
        "conventionalTotalBytes": int(conventional),
        "bothLoseToConventional": bool(min(shared, independent) > conventional),
        "comparableWithConventional": matched,
        "note": "Bytes alone decide nothing: the arms must be read at matched error, and the "
                "per-region rows are where that is visible. A shared arm that is cheaper and "
                "worse has not amortised, it has undertrained.",
    }
    if not matched:
        out["readThisFirst"] = (
            f"The shared arm's worst per-region maximum is {worst:.2f} m against a conventional "
            f"control encoded to guarantee {target:.2f} m. The arms are NOT at matched error, so "
            "every byte comparison in this record is between things that do different jobs and "
            "none of them supports a compression claim. The shared-vs-independent comparison "
            "below is still valid, since those two share an architecture, a recipe and a seed, "
            "but the conventional column is not a like-for-like competitor at this error.")
    curve = arms.get("conventionalCurve")
    if curve:
        # The cheapest conventional point that GUARANTEES an error the shared arm
        # only achieved. If one exists and is cheaper, the bytes were misallocated
        # (the same test `spent_better_on_the_base` applies to residuals).
        affordable = [row for row in curve["targets"] if row["worstMaxErrorM"] <= worst]
        cheapest = min(affordable, key=lambda r: r["totalBytes"]) if affordable else None
        out["cheapestConventionalAtThisError"] = None if cheapest is None else {
            "targetM": cheapest["targetM"], "totalBytes": cheapest["totalBytes"],
            "sharedTotalBytes": int(shared),
            "conventionalIsCheaper": bool(cheapest["totalBytes"] < shared),
            "timesCheaper": float(shared / max(cheapest["totalBytes"], 1)),
        }
        out["conventionalCurveTotals"] = {
            f"{row['targetM']}": row["totalBytes"] for row in curve["targets"]}

    transfer = arms.get("transfer")
    if transfer:
        transfer_worst = max(row["max_m"]
                             for row in transfer["evaluation"]["byRegion"].values())
        transfer_matched = bool(transfer_worst <= target)
        out["marginalVsConventionalForThatRegion"] = {
            "marginalBytes": transfer["marginalBytesForANewRegion"],
            "conventionalBytes": transfer["conventionalBytesForThatRegion"],
            "atMatchedError": transfer_matched,
            "transferWorstMaxErrorM": float(transfer_worst),
            "marginalIsCheaper": bool(transfer_matched
                                      and transfer["marginalBytesForANewRegion"]
                                      < transfer["conventionalBytesForThatRegion"]),
            "ratio": (transfer["conventionalBytesForThatRegion"]
                      / max(transfer["marginalBytesForANewRegion"], 1)),
            "note": "ratio is reported whatever the error, because it says how much room the "
                    "transfer arm has; marginalIsCheaper is false unless the error is matched, "
                    "because a cheaper arm that misses the target has not bought the same thing.",
        }
    return out
