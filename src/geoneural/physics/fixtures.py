"""Export the trained closures and the parity fixtures for the native kernel.

`native/landscape-core` reimplements the operators of `hybrid.py` in Rust and
runs the trained closures from exported weights. This module writes what it
needs, into one directory:

* `closure.json` and `<arm>.bin` for the three arms of `hybrid.train_closure`.
  Weights are little-endian float32, layer by layer; everything needed to run
  them is described in the JSON. The browser bundle ships these same files.
* one `<case>.json` per parity case. Arrays are stored as
  `{"dtype", "shape", "base64"}` with little-endian bytes, so values survive
  exactly.

Regenerate with

    uv run python -m geoneural.physics.fixtures [out_dir] [--device cuda]

Training on a GPU is not bitwise repeatable, so a rerun changes the weights
and the learned-arm fixtures together. They always agree with each other: the
reference rollouts run on the weights read back from the written files.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import pathlib

import numpy as np

from geoneural.physics import hybrid, landscape, teacher_audit

SCHEMA = "geoneural-closure-weights-v1"
FIXTURE_SCHEMA = "geoneural-landscape-fixture-v1"
RATIO_CLIP = 0.99          # the clip inside hybrid.nonlinear_teacher
SIDE, SPACING_M, BATCH = 48, 50.0, 8
DIFFUSIVITY, CRITICAL_SLOPE = 0.05, 0.6
DT_YEARS = 200.0           # the rollout step of hybrid.evaluate_closure
ROLLOUT_SEED = 31337       # the evaluation seed of hybrid.evaluate_closure
DEFAULT_OUT = pathlib.Path(__file__).resolve().parents[3] / "native" / "fixtures"


def _array(values, dtype: str = "f64") -> dict:
    values = np.ascontiguousarray(values, dtype={"f64": "<f8", "f32": "<f4"}[dtype])
    return {"dtype": dtype, "shape": list(values.shape),
            "base64": base64.b64encode(values.tobytes()).decode("ascii")}


def _write(path: pathlib.Path, record: dict) -> None:
    path.write_text(json.dumps({"schema": FIXTURE_SCHEMA, **record}, indent=1) + "\n")


def _integral(height: np.ndarray, spacing_m: float) -> float:
    return float(np.sum(height)) * spacing_m ** 2


def _rough_surface(seed: int) -> np.ndarray:
    """One training-distribution surface from `hybrid._batch`, float32 widened to float64."""
    fields, _ = hybrid._batch(np.random.default_rng(seed), 1, SIDE, SPACING_M,
                              DIFFUSIVITY, CRITICAL_SLOPE)
    return fields[0].astype(np.float64)


def _max_slope(fields: np.ndarray, spacing_m: float) -> float:
    return max(float(np.abs(np.diff(fields, axis=-1)).max()),
               float(np.abs(np.diff(fields, axis=-2)).max())) / spacing_m


# Reference operators the Python package does not have. Each mirrors the order
# of operations of the Rust kernel, so agreement is expected to the bit.

def periodic_divergence(height: np.ndarray, spacing_m: float, diffusivity: float) -> np.ndarray:
    """Linear diffusion on a periodic grid, in the pass order of `hybrid.divergence`."""
    east = (np.roll(height, -1, axis=1) - height) / spacing_m
    south = (np.roll(height, -1, axis=0) - height) / spacing_m
    out = np.zeros_like(height)
    out += east / spacing_m
    out -= np.roll(east, 1, axis=1) / spacing_m
    out += south / spacing_m
    out -= np.roll(south, 1, axis=0) / spacing_m
    return diffusivity * out


def _harmonic(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    total = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total > 0.0, 2.0 * a * b / total, 0.0)


def variable_divergence(height: np.ndarray, spacing_m: float, field: np.ndarray) -> np.ndarray:
    """Linear diffusion with per-cell diffusivity; faces use the harmonic mean."""
    east, south = hybrid.face_gradients(height, spacing_m)
    east = east * _harmonic(field[:, :-1], field[:, 1:])
    south = south * _harmonic(field[:-1, :], field[1:, :])
    return hybrid.divergence(east, south, spacing_m, height.shape)


def _clipped_faces(height: np.ndarray, spacing_m: float, critical_slope: float) -> int:
    east, south = hybrid.face_gradients(height, spacing_m)
    return int((np.abs(east) / critical_slope > RATIO_CLIP).sum()
               + (np.abs(south) / critical_slope > RATIO_CLIP).sum())


# Analytic and teacher cases.

def sine_case(side: int = 64, spacing_m: float = 50.0, diffusivity: float = 0.05,
              dt_years: float = 1000.0, steps: int = 200, mode=(1, 2),
              amplitude_m: float = 10.0) -> dict:
    """A periodic sine mode decaying under linear diffusion.

    The continuous answer is `exp(-D k^2 t)`; the discrete one decays at the
    five-point eigenvalue, which differs by `O((k dx)^2)`.
    """
    x = (np.arange(side) + 0.5) * spacing_m
    length = side * spacing_m
    height = (amplitude_m * np.sin(2 * np.pi * mode[1] * x / length)[:, None]
              * np.sin(2 * np.pi * mode[0] * x / length)[None, :])
    start = height.copy()
    for _ in range(steps):
        height = height + dt_years * periodic_divergence(height, spacing_m, diffusivity)
    return {"case": "linear-sine", "model": "linear", "boundary": "periodic",
            "side": side, "spacingM": spacing_m, "diffusivity": diffusivity,
            "dtYears": dt_years, "steps": steps, "modeX": mode[0], "modeY": mode[1],
            "amplitudeM": amplitude_m, "initial": _array(start), "final": _array(height),
            "note": "h = A sin(2 pi my y / L) sin(2 pi mx x / L) at cell centres, "
                    "L = side * spacing; final from numpy with the kernel's pass order."}


def closed_case(seed: int = 2024, diffusivity: float = 0.05, dt_years: float = 1000.0,
                steps: int = 100) -> dict:
    """Linear diffusion on a closed domain through `hybrid.flux_divergence`."""
    height = _rough_surface(seed)
    start, integrals = height.copy(), []
    for _ in range(steps):
        height = height + dt_years * hybrid.flux_divergence(height, SPACING_M, diffusivity)
        integrals.append(_integral(height, SPACING_M))
    return {"case": "linear-closed", "model": "linear", "boundary": "closed",
            "side": SIDE, "spacingM": SPACING_M, "diffusivity": diffusivity,
            "dtYears": dt_years, "steps": steps, "initial": _array(start),
            "final": _array(height), "integrals": _array(integrals)}


def variable_case(seed: int = 7, dt_years: float = 1000.0, steps: int = 100) -> dict:
    """Linear diffusion with a strongly varying per-cell diffusivity, closed."""
    rng = np.random.default_rng(seed)
    height = _rough_surface(seed)
    smooth = np.cumsum(np.cumsum(rng.normal(0.0, 1.0, (SIDE, SIDE)), axis=0), axis=1)
    smooth = (smooth - smooth.mean()) / smooth.std()
    field = 0.05 * np.exp(0.8 * smooth)
    field[10:20, 25:40] = 0.002          # a resistant block: a sharp contrast
    field = np.minimum(field, 0.5)
    start, integrals = height.copy(), []
    for _ in range(steps):
        height = height + dt_years * variable_divergence(height, SPACING_M, field)
        integrals.append(_integral(height, SPACING_M))
    return {"case": "linear-variable", "model": "linear", "boundary": "closed",
            "side": SIDE, "spacingM": SPACING_M, "dtYears": dt_years, "steps": steps,
            "initial": _array(start), "field": _array(field), "final": _array(height),
            "integrals": _array(integrals)}


def fixed_case(side: int = 32, diffusivity: float = 0.05, dt_years: float = 1000.0,
               steps: int = 50) -> dict:
    """`landscape.evolve` with fixed edges and only diffusion switched on.

    The audit surface alone is symmetric and its net outflow cancels, so a
    central hill is added: it spreads towards the pinned edges and leaves.
    """
    parameters = landscape.Parameters(uplift_m_per_year=0.0, k_incision=0.0,
                                      diffusivity_m2_per_year=diffusivity,
                                      spacing_m=SPACING_M)
    axis = np.linspace(0.0, 1.0, side) - 0.5
    hill = 40.0 * np.exp(-(axis[:, None] ** 2 + axis[None, :] ** 2) / (2 * 0.2 ** 2))
    start = teacher_audit.initial_surface(side) + hill
    final, report = landscape.evolve(start, parameters, steps * dt_years,
                                     dt_years=dt_years, base_level="fixed-edges")
    ledger = report["ledger"]
    return {"case": "linear-fixed", "model": "linear", "boundary": "fixed",
            "side": side, "spacingM": SPACING_M, "diffusivity": diffusivity,
            "dtYears": dt_years, "steps": steps, "initial": _array(start),
            "final": _array(final),
            "boundaryOutflowVolumeM3": ledger["boundaryOutflowVolumeM3"],
            "diffusionVolumeM3": ledger["diffusionVolumeM3"],
            "note": "From landscape.evolve(base_level='fixed-edges') with uplift and "
                    "incision at zero, on the audit surface plus a central hill. "
                    "Outflow is positive when material leaves."}


def teacher_case(dt_years: float = DT_YEARS, steps: int = 32) -> dict:
    """A short rollout of `hybrid.nonlinear_teacher`, closed."""
    height = _rough_surface(ROLLOUT_SEED)
    start, integrals, clipped = height.copy(), [], []
    for _ in range(steps):
        clipped.append(_clipped_faces(height, SPACING_M, CRITICAL_SLOPE))
        height = height + dt_years * hybrid.nonlinear_teacher(
            height, SPACING_M, DIFFUSIVITY, CRITICAL_SLOPE)
        integrals.append(_integral(height, SPACING_M))
    return {"case": "teacher", "model": "nonlinear", "boundary": "closed",
            "side": SIDE, "spacingM": SPACING_M, "diffusivity": DIFFUSIVITY,
            "criticalSlope": CRITICAL_SLOPE, "dtYears": dt_years, "steps": steps,
            "initial": _array(start), "final": _array(height),
            "integrals": _array(integrals), "clippedFacesPerStep": clipped}


# Weights.

def _check_layout(model, torch) -> list:
    """The format assumes exactly this layout; refuse anything else."""
    modules = list(model.net)
    convs = [m for m in modules if isinstance(m, torch.nn.Conv2d)]
    for index, module in enumerate(modules):
        if isinstance(module, torch.nn.Conv2d):
            ok = (module.kernel_size == (3, 3) and module.stride == (1, 1)
                  and module.padding == (1, 1) and module.dilation == (1, 1)
                  and module.groups == 1 and module.padding_mode == "replicate"
                  and module.bias is not None)
        else:
            ok = isinstance(module, torch.nn.GELU) and module.approximate == "none"
        if not ok:
            raise ValueError(f"unexpected closure layer {index}: {module}")
    if [type(m).__name__ for m in modules] != ["Conv2d", "GELU"] * (len(convs) - 1) + ["Conv2d"]:
        raise ValueError("closure layers are not conv/GELU pairs ending in a conv")
    return convs


def _write_weights(directory: pathlib.Path, arm: str, model, torch) -> dict:
    convs = _check_layout(model, torch)
    chunks, layers, offset = [], [], 0
    final = "identity" if arm == "flux" else "softplus"
    for index, conv in enumerate(convs):
        weight = conv.weight.detach().cpu().numpy().astype("<f4").ravel()
        bias = conv.bias.detach().cpu().numpy().astype("<f4").ravel()
        layers.append({"in": conv.in_channels, "out": conv.out_channels,
                       "weightOffset": offset, "biasOffset": offset + weight.size,
                       "activation": final if index == len(convs) - 1 else "gelu"})
        chunks += [weight, bias]
        offset += weight.size + bias.size
    blob = np.concatenate(chunks).astype("<f4").tobytes()
    (directory / f"{arm}.bin").write_bytes(blob)
    return {"file": f"{arm}.bin", "floats": offset, "bytes": len(blob),
            "sha256": hashlib.sha256(blob).hexdigest(), "layers": layers}


def load_closure(directory, arm: str, torch, device: str = "cpu"):
    """Rebuild one arm as the torch module `hybrid` trains, from the written files."""
    directory = pathlib.Path(directory)
    meta = json.loads((directory / "closure.json").read_text())
    spec = meta["arms"][arm]
    blob = np.frombuffer((directory / spec["file"]).read_bytes(), dtype="<f4")
    if blob.size != spec["floats"]:
        raise ValueError(f"{spec['file']} holds {blob.size} floats, expected {spec['floats']}")
    parts = hybrid._modules(torch)
    model = parts["FluxClosure"]() if spec["apply"] == "flux" else parts["DiffusivityField"]()
    convs = _check_layout(model, torch)
    if len(convs) != len(spec["layers"]):
        raise ValueError("layer count differs from the default closure")
    with torch.no_grad():
        for conv, layer in zip(convs, spec["layers"]):
            start = layer["weightOffset"]
            weight = blob[start:start + conv.weight.numel()].reshape(conv.weight.shape)
            bias = blob[layer["biasOffset"]:layer["biasOffset"] + conv.bias.numel()]
            conv.weight.copy_(torch.from_numpy(weight.copy()))
            conv.bias.copy_(torch.from_numpy(bias.copy()))
    return model.to(device).eval(), meta


def training_range(seed: int, steps: int) -> dict:
    """Slope and height range of the training data, by replaying its generator.

    `train_closure` draws every batch from `default_rng(seed)`, so the replay
    sees exactly the surfaces the arms were trained on.
    """
    rng = np.random.default_rng(seed)
    slope, low, high = 0.0, np.inf, -np.inf
    for _ in range(steps):
        fields, _ = hybrid._batch(rng, BATCH, SIDE, SPACING_M, DIFFUSIVITY, CRITICAL_SLOPE)
        fields = fields.astype(np.float64)
        slope = max(slope, _max_slope(fields, SPACING_M))
        low, high = min(low, float(fields.min())), max(high, float(fields.max()))
    return {"maxSlope": slope, "minHeightM": low, "maxHeightM": high}


def _closure_meta(arms: dict, validated: dict, training: dict) -> dict:
    return {
        "schema": SCHEMA,
        "format": "Each arm is a stack of 3x3 Conv2d layers, stride 1, padding 1 with "
                  "replicate padding. <arm>.bin is little-endian float32: for each layer "
                  "in order, weight [out][in][3][3] row-major, then bias [out]. Offsets "
                  "count float32 elements from the start of the file.",
        "activations": {"gelu": "0.5 x (1 + erf(x / sqrt 2)), exact erf",
                        "softplus": "log(1 + exp(x)), and x itself above 20",
                        "identity": "x"},
        "normalisation": "none: inputs enter the network in metres and rise over run, "
                         "as hybrid.py feeds them",
        "grid": "side x side cells, row-major, row 0 north; spacingM is the cell edge",
        "spacingM": SPACING_M,
        "dtYears": DT_YEARS,
        "dtNote": "the rollout step of hybrid.evaluate_closure; a kernel splits any step "
                  "into substeps below the stability bound",
        "teacher": {"form": "G = D g / (1 - r^2), r = clip(|g| / S_c, 0, 0.99), g = dh/dx "
                            "on a face; tendency = divergence of G",
                    "diffusivity": DIFFUSIVITY, "criticalSlope": CRITICAL_SLOPE,
                    "ratioClip": RATIO_CLIP},
        "validated": {**validated,
                      "note": "range of the training surfaces; states outside it are "
                              "extrapolation and the kernel refuses to step them"},
        "stability": "explicit Euler; kfield and penalty: dt <= spacing^2 / (4 max K); "
                     "flux: the teacher bound spacing^2 (1 - 0.99^2) / (4 D)",
        "training": training,
        "arms": arms,
    }


def _arm_description(arm: str) -> dict:
    if arm == "flux":
        return {"apply": "flux", "conservative": True,
                "inputs": ["face gradient (h[c+1] - h[c]) / spacingM",
                           "face mean height 0.5 (h[c+1] + h[c])"],
                "inputGrid": "east faces, side rows by side-1 columns; south faces run "
                             "the same network on the transposed field",
                "tendency": "output G per face; the west (north) cell gains G / spacingM "
                            "and the east (south) cell loses it"}
    return {"apply": "kfield", "conservative": False,
            "inputs": ["cell height h"], "inputGrid": "cells, side x side",
            "tendency": "output K per cell (m^2/yr, after softplus); tendency = K (h_n + "
                        "h_s + h_w + h_e - 4 h) / spacingM^2 with replicate edges",
            "trainedWith": "conservation loss" if arm == "penalty" else "no conservation term"}


# Learned-arm rollouts.

def _rollout(apply: str, model, start: np.ndarray, steps: int, dt_years: float, torch,
             dtype, device: str):
    surface = torch.from_numpy(start[None]).to(device=device, dtype=dtype)
    integrals, max_k = [], 0.0
    with torch.no_grad():
        for _ in range(steps):
            if apply == "kfield":
                max_k = max(max_k, float(model(surface).max()))
            surface = surface + dt_years * hybrid.apply_closure(
                apply, model, surface, SPACING_M, torch)
            integrals.append(float(surface.double().sum()) * SPACING_M ** 2)
    return surface[0].double().cpu().numpy(), integrals, max_k


def arm_case(arm: str, directory: pathlib.Path, torch, device: str, validated: dict,
             steps: int = 16) -> dict:
    """One arm from the written weights: a tendency and a rollout, float64 and float32.

    The float64 run (CPU, weights widened exactly) is the reference for the
    kernel's float64 inference. The float32 run on `device`, with TF32 off, is
    the rollout `hybrid.evaluate_closure` performs; its distance from the
    float64 run is the float32 noise floor.
    """
    model, meta = load_closure(directory, arm, torch, "cpu")
    apply = meta["arms"][arm]["apply"]
    seed = ROLLOUT_SEED
    start = _rough_surface(seed)
    while not (_max_slope(start, SPACING_M) <= validated["maxSlope"]
               and validated["minHeightM"] <= start.min()
               and start.max() <= validated["maxHeightM"]):
        seed += 1
        start = _rough_surface(seed)
    model64 = copy.deepcopy(model).double()
    with torch.no_grad():
        tendency = hybrid.apply_closure(apply, model64, torch.from_numpy(start[None]),
                                        SPACING_M, torch)[0].numpy()
    # The kernel splits a step beyond 0.9 of the stability bound, and a split
    # rollout is not the torch rollout. A K-field arm can exceed the teacher's
    # largest diffusivity, so its step is halved until the rollout fits.
    dt_years = DT_YEARS
    while True:
        final64, integrals64, max_k = _rollout(apply, model64, start, steps, dt_years,
                                               torch, torch.float64, "cpu")
        if apply != "kfield" or dt_years <= 0.9 * SPACING_M ** 2 / (4.0 * max_k):
            break
        dt_years /= 2.0
    flags = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    try:
        final32, integrals32, _ = _rollout(apply, model.float().to(device), start, steps,
                                           dt_years, torch, torch.float32, device)
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = flags
    scale = float(np.abs(final64).max())
    record = {"case": f"arm-{arm}", "model": arm, "apply": apply, "boundary": "closed",
              "side": SIDE, "spacingM": SPACING_M, "dtYears": dt_years, "steps": steps,
              "initialSeed": seed, "initial": _array(start), "tendency64": _array(tendency),
              "final64": _array(final64), "integrals64": _array(integrals64),
              "final32": _array(final32.astype(np.float32), "f32"),
              "integrals32": _array(integrals32),
              "float32": {"device": device, "tf32": False,
                          "maxAbsDiffFromFloat64M": float(np.abs(final32 - final64).max()),
                          "relativeToMaxHeight": float(np.abs(final32 - final64).max()) / scale}}
    if apply == "kfield":
        record["maxDiffusivity"] = max_k
    return record


def export(out_dir=DEFAULT_OUT, device: str = "cuda", seed: int = 1729, steps: int = 1500,
           train: bool = True) -> dict:
    """Train the three arms as `hybrid.campaign` does, write weights and fixtures.

    With `train=False` the weights already in `out_dir` are reused and only the
    fixtures are rewritten.
    """
    import torch

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    validated = training_range(seed, steps)
    if train:
        arms, training = {}, {"steps": steps, "batch": BATCH, "side": SIDE, "lr": 3e-4,
                              "seed": seed, "device": device, "arms": {}}
        for arm in hybrid.ARMS:
            fitted = hybrid.train_closure(arm, torch, steps=steps, side=SIDE,
                                          spacing_m=SPACING_M, device=device, seed=seed)
            arms[arm] = {**_arm_description(arm),
                         **_write_weights(out, arm, fitted["model"], torch)}
            training["arms"][arm] = {"parameters": fitted["parameters"],
                                     "finalLoss": fitted["history"][-1]["loss"],
                                     "seconds": round(fitted["seconds"], 1)}
            del fitted
        meta = _closure_meta(arms, validated, training)
        (out / "closure.json").write_text(json.dumps(meta, indent=1) + "\n")
    cases = [sine_case(), closed_case(), variable_case(), fixed_case(), teacher_case()]
    cases += [arm_case(arm, out, torch, device, validated) for arm in hybrid.ARMS]
    for case in cases:
        _write(out / f"{case['case']}.json", case)
    sizes = {p.name: p.stat().st_size for p in sorted(out.iterdir()) if p.is_file()}
    return {"outDir": str(out), "bytes": sum(sizes.values()), "files": sizes,
            "float32": {c["model"]: c["float32"] for c in cases if "float32" in c}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir", nargs="?", default=str(DEFAULT_OUT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--reuse-weights", action="store_true",
                        help="keep the weights in out_dir and rewrite only the fixtures")
    args = parser.parse_args()
    print(json.dumps(export(args.out_dir, args.device, args.seed, args.steps,
                            train=not args.reuse_weights), indent=1))
