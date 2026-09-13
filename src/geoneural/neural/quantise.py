"""Charge the neural side the way the conventional side is charged.

`q32-delta-zstd` quantises to a declared tolerance, takes a spatial delta and
entropy-codes the result. Pricing a neural model as raw float16 safetensors (no
quantisation below 16 bits, no prediction, no entropy coder) is asymmetric in the
codec's favour. The weights are not incompressible (they are a smooth-ish
distribution with a lot of near-zero mass), so that pricing charges the network
for bytes a real deployment would not ship.

Candidates are ranked by actual encoded bytes, including any learned probability
models, tables and normalisation. A Shannon entropy estimate without an
implemented coder is a lower bound, not a file size, so everything here runs a
real coder and reports real lengths. The entropy is computed too, as a
clearly-labelled bound, never as a result.

Three things are charged that are easy to forget:

* The scales. One float32 scale (and, asymmetrically, one zero point) per
  quantised tensor is stored side information. At int8 with a hundred tensors
  that is a few hundred bytes against tens of kilobytes: small, but not zero,
  and close byte comparisons can turn on it.
* The shapes and the dtype table. A decoder that cannot tell how to unpack
  the stream is not a decoder. The container is measured, not estimated.
* The error after quantisation, not before. A model priced at int8 and
  measured at float16 would be priced as one model and scored as another.

Sub-byte widths are packed rather than stored in int8 containers, because an int4
tensor that occupies eight bits per value is an int8 tensor with a smaller range.
"""
from __future__ import annotations

import numpy as np

SCHEMA = "geoneural-quantised-package-v1"

#: Integer widths worth measuring. int8 is the first step below float16;
#: 6 and 4 bits are where a small decoder either survives or does not.
WIDTHS = (8, 6, 4)


def _pack(values: np.ndarray, bits: int) -> bytes:
    """Bit-pack unsigned integers at an arbitrary width.

    A 6-bit tensor stored one-per-byte is an 8-bit tensor with a smaller range,
    and pricing it at six bits would undercount it. Widths that do not
    divide eight (6, 5, 3) are exactly the interesting ones for a small decoder,
    so the packer works at the bit level rather than assuming a whole number of
    values per byte.
    """
    flat = values.reshape(-1).astype(np.uint8)
    if bits == 8:
        return flat.tobytes()
    if not 1 <= bits <= 8:
        raise ValueError(f"width {bits} is outside 1..8 bits")
    # Most-significant-first within each value, so the stream is the obvious one
    # a decoder would write by hand.
    unpacked = np.unpackbits(flat[:, None], axis=1)[:, 8 - bits:]
    return np.packbits(unpacked.reshape(-1)).tobytes()


def quantise_tensor(tensor: np.ndarray, bits: int, symmetric: bool = True) -> dict:
    """One tensor to integers, with the side information it needs to come back."""
    values = np.asarray(tensor, dtype=np.float64)
    levels = (1 << bits) - 1
    if symmetric:
        scale = float(np.abs(values).max()) / (levels // 2) if values.size else 1.0
        scale = scale or 1.0
        codes = np.clip(np.round(values / scale) + (levels // 2), 0, levels).astype(np.uint8)
        zero_point = float(levels // 2)
    else:
        low, high = float(values.min()), float(values.max())
        scale = (high - low) / levels if high > low else 1.0
        codes = np.clip(np.round((values - low) / scale), 0, levels).astype(np.uint8)
        zero_point = low
    restored = ((codes.astype(np.float64) - (levels // 2)) * scale if symmetric
                else codes.astype(np.float64) * scale + zero_point)
    return {"codes": codes, "scale": scale, "zeroPoint": zero_point,
            "restored": restored.reshape(values.shape),
            "maxAbsError": float(np.abs(restored.reshape(values.shape) - values).max())
            if values.size else 0.0}


def entropy_bits(codes: np.ndarray) -> float:
    """Order-0 Shannon entropy of the symbol stream. A BOUND, not a size."""
    if codes.size == 0:
        return 0.0
    counts = np.bincount(codes.reshape(-1))
    probabilities = counts[counts > 0] / codes.size
    return float(-(probabilities * np.log2(probabilities)).sum())


def encode_state(state: dict, bits: int, level: int = 19) -> dict:
    """Quantise every stored tensor, entropy-code it, and count what ships.

    `state` is a `{name: ndarray}` mapping (`training.store_state`'s output moved
    to numpy). The stream is the packed codes of every tensor concatenated and
    compressed once, because a real container would not restart the coder per
    tensor and per-tensor compression would overstate the cost by a header each.
    """
    import zstandard

    packed, table, total_values = [], [], 0
    restored: dict[str, np.ndarray] = {}
    worst = 0.0
    for name in sorted(state):
        values = np.asarray(state[name])
        result = quantise_tensor(values, bits)
        packed.append(_pack(result["codes"], bits))
        restored[name] = result["restored"]
        worst = max(worst, result["maxAbsError"])
        total_values += values.size
        table.append({"name": name, "shape": list(values.shape),
                      "scale": result["scale"], "zeroPoint": result["zeroPoint"],
                      "entropyBitsPerSymbol": entropy_bits(result["codes"])})
    stream = b"".join(packed)
    coded = zstandard.ZstdCompressor(level=level).compress(stream)

    # Side information, counted explicitly. Two float32 per tensor for
    # scale and zero point, plus a compact shape record: 1 byte of rank and 4
    # bytes per dimension. Names are not counted; a deployment ships an ordered
    # container, not a dictionary of strings.
    side = sum(8 + 1 + 4 * len(entry["shape"]) for entry in table)
    weighted_entropy = (sum(entry["entropyBitsPerSymbol"] * int(np.prod(entry["shape"]))
                            for entry in table) / max(total_values, 1))
    return {"bits": bits, "tensors": table,
            "packedBytes": len(stream), "codedBytes": len(coded),
            "sideInformationBytes": side,
            "deployedBytes": len(coded) + side,
            "values": total_values,
            "bitsPerWeight": 8.0 * (len(coded) + side) / max(total_values, 1),
            "entropyBitsPerWeight": weighted_entropy,
            "entropyBoundBytes": int(np.ceil(weighted_entropy * total_values / 8.0)) + side,
            "worstWeightError": worst,
            "restored": restored,
            "coder": f"zstd level {level} over bit-packed symbols, one stream",
            "note": "codedBytes is a measured length from a real coder. "
                    "entropyBoundBytes is an order-0 Shannon bound on the same symbols and is "
                    "NOT a file size: it is reported so the gap between what the coder achieved "
                    "and what the symbol statistics permit is visible."}


def package(model, torch, widths=WIDTHS, float16_reference: bool = True) -> dict:
    """Every integer width for one model, against its float16 price.

    Returns the restored tensors per width so a caller can load them back and
    measure the error the deployed model actually makes. Nothing here decides
    which width wins; that needs the terrain error, which lives with the caller.
    """
    from geoneural.neural import training
    state = {name: value.to(torch.float32).numpy()
             for name, value in training.store_state(model, torch, "float32").items()
             if value.is_floating_point()}
    integer_state = {name: value for name, value in
                     training.store_state(model, torch, "float32").items()
                     if not value.is_floating_point()}
    reference = training.deployed_bytes(model, torch, "float16") if float16_reference else None
    rows = []
    for bits in widths:
        encoded = encode_state(state, bits)
        rows.append({k: v for k, v in encoded.items() if k != "restored"})
        rows[-1]["restored"] = encoded["restored"]
    return {"schema": SCHEMA,
            "float16DeployedBytes": reference,
            "nonFloatTensors": sorted(integer_state),
            "byWidth": rows,
            "qualification": "Weights only. Any conventional base, context raster or code payload "
                             "shipped beside the model is charged separately and is not made "
                             "cheaper by quantising the network. Error after quantisation is the "
                             "caller's measurement: a width priced here and evaluated at float16 "
                             "would be a subsidy."}


def load_restored(model, restored: dict, torch) -> None:
    """Put a width's restored tensors back into a model, in place."""
    state = model.state_dict()
    for name, values in restored.items():
        state[name].copy_(torch.from_numpy(np.asarray(values, dtype=np.float32)).to(
            state[name].dtype).reshape(state[name].shape))


class FakeQuantise(np.lib.mixins.NDArrayOperatorsMixin):
    """Marker type only; the real parametrisation is `_FakeQuantise` below."""


def attach(model, bits: int, torch):
    """Fake-quantise every weight in the forward pass, with a straight-through gradient.

    Post-training quantisation moves a trained model to the nearest grid point and
    hopes the loss surface is flat there. On a trained SIREN it is not: int8
    rounding raised the codec error from 0.576 m to 1.227 m, costing more accuracy
    than it saved in bytes.

    Training through the quantiser is the standard answer. The forward pass sees
    `dequantise(quantise(w))`; the backward pass sees the identity, because the
    rounding derivative is zero almost everywhere and would stop learning.
    The weights that survive are ones whose rounded values are good, rather than
    ones that happened to be good before rounding.

    The scale is recomputed from the current weights each forward, so it tracks
    the distribution as it moves rather than freezing an early estimate.
    """
    from torch.nn.utils import parametrize

    class _FakeQuantise(torch.nn.Module):
        def __init__(self, bits: int):
            super().__init__()
            self.levels = (1 << bits) - 1
            self.half = self.levels // 2

        def forward(self, weight):
            scale = weight.detach().abs().max() / self.half
            scale = torch.clamp(scale, min=1e-12)
            rounded = torch.clamp(torch.round(weight / scale), -self.half, self.half) * scale
            # Straight through: value of the rounded tensor, gradient of the raw one.
            return weight + (rounded - weight).detach()

    attached = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) and hasattr(module, "weight"):
            parametrize.register_parametrization(module, "weight", _FakeQuantise(bits))
            attached += 1
    return attached


def detach_all(model, torch) -> None:
    """Bake the quantised weights in and remove the parametrisations."""
    from torch.nn.utils import parametrize
    for module in model.modules():
        if parametrize.is_parametrized(module, "weight"):
            parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)


LADDER_SCHEMA = "geoneural-quantised-ladder-v1"


def _survival(outcome) -> dict:
    """`survives` returns (verdict, why-not); flatten it into the report row."""
    survives, why = outcome
    return {"survives": survives} if why is None else {"survives": survives, "whyNot": why}


def constant_floor(problem) -> float:
    """MAE of predicting the field mean everywhere, on the split the ladder scores.

    A compressed model that cannot beat this has not learned the terrain, and no
    byte count it achieves is worth reporting as a win.
    """
    values = np.asarray(problem.flat, dtype=np.float64)[problem.indexes["all"]]
    return float(np.mean(np.abs(values - float(np.mean(values)))))


def ladder(chosen: dict, problem, steps: int = 5000, batch: int = 8192,
           finetune_steps: int = 1500, widths=WIDTHS, seed: int = 1729,
           engine_mode: str = "cudagraph", data_path: str = "device",
           sampling: str = "with") -> dict:
    """The quantised ladder: what the network costs when it is charged like a codec.

    For each chosen configuration: train it, price and measure it at float16 (the
    default storage convention), then at every integer width both by
    post-training rounding and by fine-tuning through the quantiser. Each row is
    placed against the conventional envelope on the same split, so a width that
    moves a point off the front says so.

    `q32-delta-zstd` quantises to a tolerance, predicts spatially and
    entropy-codes the residual, while raw float16 safetensors do none of those
    three. Pricing the network at float16 alone is not a like-for-like rate.
    """
    import copy
    import torch

    from geoneural.neural import search as arch

    from geoneural.neural import training
    from geoneural.neural.learning import features

    envelope = arch.conventional_envelope(problem, split="all")
    points = sorted((int(row.get("deployedBytes", row.get("bytes", 0))),
                     float(row.get("maeM", row.get("mae_m"))),
                     float(row.get("maxM", row.get("max_m"))),
                     str(row.get("label")))
                    for row in envelope["points"])

    cheapest_conventional = min(row[0] for row in points) if points else 0
    floor_mae = float(constant_floor(problem))

    def beaten_by(byte_count: int, mae: float):
        affordable = [row for row in points if row[0] <= byte_count and row[1] < mae]
        best = min(affordable, key=lambda row: row[1]) if affordable else None
        return None if best is None else {
            "label": best[3], "bytes": best[0], "maeM": best[1], "maxM": best[2]}

    def beaten_on_max(byte_count: int, max_m: float):
        affordable = [row for row in points if row[0] <= byte_count and row[2] < max_m]
        best = min(affordable, key=lambda row: row[2]) if affordable else None
        return None if best is None else {
            "label": best[3], "bytes": best[0], "maeM": best[1], "maxM": best[2]}

    def survives(byte_count: int, mae: float) -> tuple[bool, str | None]:
        """Unbeaten is not the same as winning.

        A row cheaper than every conventional point has an empty comparison set,
        so `beaten_by` returns None for it, which reads as a win and is not one.
        Such a row can also be worse than predicting the field mean (for example
        268 bytes at 36.86 m MAE against a 23.32 m constant predictor). A row only
        survives if the envelope covers its budget and it clears the
        constant-predictor floor.
        """
        if mae >= floor_mae:
            return False, "worse than the constant predictor"
        if byte_count < cheapest_conventional:
            return False, "below the cheapest conventional point; nothing to compare against"
        if beaten_by(byte_count, mae) is not None:
            return False, None
        return True, None

    # The fine-tune fits the whole field, the same objective `measure` was given
    # above, so the slope term's pairs belong over the whole field too. Built once
    # and reused: `DeviceTables.pairs` caches per mask, and a fresh array per
    # finalist would upload the same pairs again each time.
    full_mask = np.ones((problem.side, problem.side), dtype=bool)

    rows = []
    for name in sorted(chosen):
        entry = chosen[name]
        config = dict(entry["config"])
        recipe = training.with_overrides(
            Recipe_from(entry.get("recipe"), steps, batch, problem.device, seed),
            data_path=data_path, sampling=sampling)
        result = arch.measure(config, recipe, problem, evaluate_on=("all",), train_on="all",
                              store_precision="float32", limit=None, engine_mode=engine_mode)
        trained = result["model"]
        # A residual family ships a conventional base alongside its weights, and
        # that base can be most of the payload. Pricing the weights alone can turn
        # a dominated point into an apparent win.
        base_bytes = int(result.get("baseBytes", 0))
        float16_bytes = training.deployed_bytes(trained, torch, "float16") + base_bytes
        reference = copy.deepcopy(trained)
        training.round_to_storage(reference, torch, "float16")
        float16_metrics = training.evaluate(
            reference, features, problem.indexes["all"], problem.flat, problem.side,
            problem.intervals, False, problem.mean, problem.scale, problem.device, torch, None)
        del reference

        widths_rows = [{
            "width": "float16", "bits": 16, "method": "storage rounding",
            "deployedBytes": int(float16_bytes),
            "bitsPerWeight": 8.0 * float16_bytes / max(sum(
                int(np.prod(v.shape)) for v in trained.state_dict().values()), 1),
            "maeM": float16_metrics["mae_m"], "maxM": float16_metrics["max_m"],
            "baseBytes": base_bytes,
            "beatenBy": beaten_by(int(float16_bytes), float16_metrics["mae_m"]),
            "beatenOnMaxBy": beaten_on_max(int(float16_bytes), float16_metrics["max_m"]),
            **_survival(survives(int(float16_bytes), float16_metrics["mae_m"]))}]

        for bits in widths:
            for method in ("post-training", "quantisation-aware"):
                model = copy.deepcopy(trained)
                if method == "quantisation-aware":
                    attach(model, bits, torch)
                    fine = training.with_overrides(recipe, steps=finetune_steps,
                                                   lr=recipe.lr * 0.1, seed=seed ^ 0x9E37,
                                                   schedule="cosine", warmup=0)
                    training.fit(model, features, problem.flat, problem.side, problem.intervals,
                                 False, problem.indexes["all"], problem.mean, problem.scale,
                                 fine, torch, train_mask=full_mask,
                                 tables=problem.tables(False) if data_path == "device" else None)
                    detach_all(model, torch)
                encoded = package(model, torch, widths=(bits,), float16_reference=False)
                row = encoded["byWidth"][0]
                load_restored(model, row["restored"], torch)
                metrics = training.evaluate(
                    model, features, problem.indexes["all"], problem.flat, problem.side,
                    problem.intervals, False, problem.mean, problem.scale,
                    problem.device, torch, None)
                deployed = int(row["deployedBytes"]) + base_bytes
                widths_rows.append({
                    "width": f"int{bits}", "bits": bits, "method": method,
                    "deployedBytes": deployed,
                    "weightBytes": int(row["deployedBytes"]),
                    "baseBytes": base_bytes,
                    "codedBytes": row["codedBytes"],
                    "sideInformationBytes": row["sideInformationBytes"],
                    "bitsPerWeight": row["bitsPerWeight"],
                    "entropyBitsPerWeight": row["entropyBitsPerWeight"],
                    "maeM": metrics["mae_m"], "maxM": metrics["max_m"],
                    "p99M": metrics["p99_m"],
                    "beatenBy": beaten_by(deployed, metrics["mae_m"]),
                    "beatenOnMaxBy": beaten_on_max(deployed, metrics["max_m"]),
                    **_survival(survives(deployed, metrics["mae_m"]))})
                del model
                if problem.device.startswith("cuda"):
                    torch.cuda.empty_cache()

        survivors = [row for row in widths_rows if row["survives"]]
        # A survivor that loses on maximum error has won on the mean only. A model
        # can beat a codec point on MAE while its worst node is far worse, so
        # survivors are also listed by whether they hold on both axes.
        clean = [row for row in survivors if row["beatenOnMaxBy"] is None]
        rows.append({"name": name, "config": arch.describe(config),
                     "byWidth": widths_rows,
                     "survivingWidths": [row["width"] + "/" + row["method"] for row in survivors],
                     "survivingOnBothAxes": [row["width"] + "/" + row["method"] for row in clean],
                     "bestSurvivor": min(survivors, key=lambda row: row["deployedBytes"])
                     if survivors else None})
        del result, trained
        if problem.device.startswith("cuda"):
            torch.cuda.empty_cache()

    return {"schema": LADDER_SCHEMA, "rows": rows,
            "conventional": envelope,
            "finetuneSteps": finetune_steps, "steps": steps, "batch": batch, "seed": seed,
            "qualification": "Codec objective: trained on every page and scored on every page, so "
                             "these are rate-distortion figures and not generalisation. Error is "
                             "measured AFTER quantisation on the model that would ship. Weights "
                             "only: a conventional base or code payload beside the model is "
                             "charged separately and is not made cheaper by quantising the "
                             "network. `beatenBy` names the cheapest conventional point that is "
                             "both affordable at the row's byte count and more accurate; a row "
                             "with `beatenBy` null is one no conventional point reaches."}


def Recipe_from(stored, steps, batch, device, seed):
    """A stored recipe mapping, or the default recipe, at this budget."""
    from geoneural.neural.training import Recipe
    if not stored:
        return Recipe(steps=steps, batch=batch, device=device, seed=seed)
    fields = {k: v for k, v in stored.items()
              if k in Recipe.__dataclass_fields__}
    fields.update(steps=steps, batch=batch, device=device, seed=seed)
    return Recipe(**fields)
