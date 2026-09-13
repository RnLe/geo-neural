"""Train the landscape emulators and evaluate them against their gates.

The emulator advances a surface the way the teacher would, faster. Height error
alone is easy to make small: a landscape is mostly a smooth ramp, and predicting
the input reproduces most of it. So every gate is a comparison against baselines
that do exactly that, and the emulator has to beat them:

* persistence: return the input unchanged. The floor for any one-step claim.
* linear uplift: add `U dt` everywhere. What the surface does if erosion is
  ignored entirely, which on short steps is most of what happens.

Three gates beyond height, because a surface can be close in metres and wrong as
a landscape:

* drainage, through `hydrology.compare` at the ensemble's own spacing. This
  catches a plausible smooth field with the channels in the wrong places.
* conservation, against the teacher's own ledger. An emulator that gains or
  loses mass per step is integrating something other than the equation.
* long rollout, free-running to the end of the recorded window. One-step
  accuracy is not stability: a model can be excellent at one step and diverge
  over eight.

Everything is scored on the test split, the top-decile corner of
(log Nf, log Nh) held out whole: a regime the emulator has not seen rather than
the interior of one it has.
"""
from __future__ import annotations

import json
import pathlib
import time

import numpy as np

from geoneural.physics import emulator

from geoneural.metrics import hydrology

SCHEMA = "geoneural-emulator-train-v1"


class EnsembleData:
    """Frames, parameters and split, normalised per simulation.

    Normalisation is per simulation from its first frame only. Using the whole
    trajectory would leak the answer (how much relief the run ends with) into
    the input scale.
    """

    def __init__(self, folder, torch, device: str = "cuda"):
        folder = pathlib.Path(folder)
        self.manifest = json.loads((folder / "manifest.json").read_text())
        archive = np.load(folder / "ensemble.npz")
        self.ids = [record["id"] for record in self.manifest["simulations"]]
        self.by_id = {record["id"]: record for record in self.manifest["simulations"]}
        self.frames = {key: np.asarray(archive[key], dtype=np.float32) for key in self.ids}
        self.torch = torch
        self.device = device
        self.split = self.manifest["split"]
        self.side = int(self.manifest["side"])
        self.spacing_m = float(self.manifest["spacingM"])
        self.scale = {}
        self.centre = {}
        for key, stack in self.frames.items():
            first = stack[0]
            self.centre[key] = float(first.mean())
            self.scale[key] = max(float(first.std()), 1e-3)
        mask = np.zeros((self.side, self.side), dtype=np.float32)
        mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 1.0
        self.boundary = torch.from_numpy(mask).to(device)

    def conditioners(self, key):
        record = self.by_id[key]["job"]
        return (float(record["fluvialNumber"]), float(record["hillslopeNumber"]),
                float(self.by_id[key]["dimensionlessStep"]))

    def pair(self, key, index, stride: int = 1):
        """Normalised (input, target) `stride` recorded intervals apart.

        Stride is how `dimensionlessStep` becomes a real conditioner. The
        ensemble records every simulation at one interval, so dt* is 2.5 for all
        of them and, without strides, the conditioner would be a constant
        (a dead channel). Asking for one, two or four intervals gives dt* in
        {2.5, 5, 10} from the same frames, which is the variable-step behaviour
        the conditioner exists to provide.
        """
        stack = self.frames[key]
        centre, scale = self.centre[key], self.scale[key]
        target = min(index + stride, stack.shape[0] - 1)
        return ((stack[index] - centre) / scale, (stack[target] - centre) / scale)

    def batch(self, keys, indexes, stride: int = 1):
        torch = self.torch
        inputs = np.stack([self.pair(k, i, stride)[0] for k, i in zip(keys, indexes)])
        targets = np.stack([self.pair(k, i, stride)[1] for k, i in zip(keys, indexes)])
        height = torch.from_numpy(inputs).to(self.device)
        target = torch.from_numpy(targets).to(self.device)
        mask = self.boundary.unsqueeze(0).expand(height.shape[0], -1, -1)
        channels = emulator.input_channels(height, mask, torch)
        scalars = torch.tensor(
            [[np.log(max(self.conditioners(k)[0], 1e-12)),
              np.log(max(self.conditioners(k)[1], 1e-12)),
              self.conditioners(k)[2] * stride] for k in keys],
            dtype=torch.float32, device=self.device)
        return channels, scalars, height, target


def train(config: dict, data: EnsembleData, torch, steps: int = 2000,
          batch: int = 8, lr: float = 1e-3, seed: int = 1729,
          rollout_from: float = 0.5, rollout_length: int = 2,
          strides=(1, 2, 4)) -> dict:
    """One-step training, then a short rollout curriculum on the back half.

    A model trained only on single steps is optimised for a distribution it
    never sees at inference, where its own output is its next input; the
    resulting drift is invisible at one step and dominant at eight.
    """
    torch.manual_seed(seed)
    model = emulator.make_emulator(config, torch).to(data.device).train()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    train_ids = data.split["trainIds"]
    intervals = data.manifest["framesPerSimulation"] - 1
    history = []
    started = time.perf_counter()
    for step in range(steps):
        rolling = step >= int(steps * rollout_from)
        length = rollout_length if rolling else 1
        stride = 1 if rolling else int(rng.choice(strides))
        keys = [train_ids[i] for i in rng.integers(0, len(train_ids), batch)]
        starts = rng.integers(0, max(1, intervals - length * stride + 1), batch)
        channels, scalars, height, _ = data.batch(keys, starts, stride)
        loss = 0.0
        current = height
        for offset in range(length):
            if offset > 0:
                mask = data.boundary.unsqueeze(0).expand(current.shape[0], -1, -1)
                channels = emulator.input_channels(current, mask, torch)
            predicted = model(channels, scalars).squeeze(1)
            truth = torch.from_numpy(np.stack([
                data.pair(k, s + offset * stride, stride)[1]
                for k, s in zip(keys, starts)])).to(data.device)
            loss = loss + torch.nn.functional.l1_loss(predicted, truth)
            current = predicted
        loss = loss / length
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            history.append({"step": step, "loss": float(loss.detach()),
                            "rolloutLength": length})
    return {"model": model, "history": history, "strides": list(strides),
            "seconds": time.perf_counter() - started,
            "parameters": int(sum(p.numel() for p in model.parameters()))}


def _denormalise(values, centre, scale):
    return values * scale + centre


def evaluate(model, config, data, torch, ids=None, rollout=(1, 2, 4, 8),
             stream_cells: int = 500, drainage_simulations: int = 8) -> dict:
    """Every gate, on the held-out corner, against baselines that do the easy part.

    `persistence` and `linearUplift` exist because most of a landscape is a ramp
    that barely moves in one step. An emulator that beats neither has learned
    nothing, and one that beats persistence only at one step has learned nothing
    that survives being run.
    """
    ids = data.split["testIds"] if ids is None else ids
    intervals = data.manifest["framesPerSimulation"] - 1
    # Drainage over a sample of the test set rather than one simulation: routing
    # agreement varies enough between landscapes that a single case is anecdote.
    drainage_ids = set(ids[:drainage_simulations])
    model.eval()
    out = {"rollout": [], "ids": len(ids)}
    with torch.no_grad():
        for length in rollout:
            if length > intervals:
                continue
            errors = {"emulator": [], "persistence": [], "linearUplift": []}
            drainage = {"emulator": [], "persistence": []}
            drift = []
            for key in ids:
                stack = data.frames[key]
                centre, scale = data.centre[key], data.scale[key]
                job = data.by_id[key]["job"]
                start = 0
                current = torch.from_numpy(
                    ((stack[start] - centre) / scale)[None]).to(data.device)
                scalars = torch.tensor(
                    [[np.log(max(job["fluvialNumber"], 1e-12)),
                      np.log(max(job["hillslopeNumber"], 1e-12)),
                      float(data.by_id[key]["dimensionlessStep"])]],
                    dtype=torch.float32, device=data.device)
                mask = data.boundary.unsqueeze(0)
                for _ in range(length):
                    channels = emulator.input_channels(current, mask, torch)
                    current = model(channels, scalars).squeeze(1)
                predicted = _denormalise(current[0].cpu().numpy(), centre, scale)
                truth = stack[start + length]
                persistence = stack[start]
                span = float(job["years"]) / intervals * length
                uplift = persistence + float(job["uplift"]) * span
                errors["emulator"].append(float(np.abs(predicted - truth).mean()))
                errors["persistence"].append(float(np.abs(persistence - truth).mean()))
                errors["linearUplift"].append(float(np.abs(uplift - truth).mean()))
                # Conservation, reported three ways. The ratio alone is a poor
                # statistic because its denominator (the mass uplift adds over
                # the span) varies by three orders of magnitude across the
                # ensemble (U spans 2e-7 to 2e-4 m/yr), so its mean is set by
                # whichever runs uplift least. The absolute error in metres says
                # how wrong the mass is; the median ratio says how wrong it is
                # for a typical run.
                added = float(job["uplift"]) * span
                mass_error = float(abs(predicted.mean() - truth.mean()))
                drift.append({"ratio": mass_error / max(abs(added), 1e-12),
                              "massErrorM": mass_error, "upliftAddedM": abs(added)})
                if key in drainage_ids:
                    spacing = data.spacing_m
                    drainage["emulator"].append(
                        hydrology.compare(truth.astype(np.float64),
                                          predicted.astype(np.float64),
                                          spacing, stream_cells))
                    drainage["persistence"].append(
                        hydrology.compare(truth.astype(np.float64),
                                          persistence.astype(np.float64),
                                          spacing, stream_cells))
            ratios = np.array([d["ratio"] for d in drift])
            mass = np.array([d["massErrorM"] for d in drift])
            added_m = np.array([d["upliftAddedM"] for d in drift])
            row = {"steps": length,
                   "emulatorMaeM": float(np.mean(errors["emulator"])),
                   "persistenceMaeM": float(np.mean(errors["persistence"])),
                   "linearUpliftMaeM": float(np.mean(errors["linearUplift"])),
                   "conservationDriftFractionOfUplift": float(np.mean(ratios)),
                   "conservationDriftMedian": float(np.median(ratios)),
                   "massErrorMeanM": float(np.mean(mass)),
                   "massErrorMedianM": float(np.median(mass)),
                   "upliftAddedMedianM": float(np.median(added_m)),
                   "upliftAddedRangeM": [float(added_m.min()), float(added_m.max())]}
            row["beatsPersistence"] = row["emulatorMaeM"] < row["persistenceMaeM"]
            row["beatsLinearUplift"] = row["emulatorMaeM"] < row["linearUpliftMaeM"]
            if drainage["emulator"]:
                def summarise(records):
                    return {"receiverAgreementFraction": float(np.mean(
                                [r["receiverAgreementFraction"] for r in records])),
                            "streamJaccard": float(np.mean(
                                [r["streamJaccard"] for r in records])),
                            "simulations": len(records)}
                row["drainage"] = {
                    "emulator": summarise(drainage["emulator"]),
                    "persistence": summarise(drainage["persistence"]),
                    "beatsPersistenceOnStreams":
                        float(np.mean([r["streamJaccard"] for r in drainage["emulator"]])) >
                        float(np.mean([r["streamJaccard"] for r in drainage["persistence"]])),
                    "beatsPersistenceOnReceivers":
                        float(np.mean([r["receiverAgreementFraction"] for r in drainage["emulator"]])) >
                        float(np.mean([r["receiverAgreementFraction"] for r in drainage["persistence"]]))}
            out["rollout"].append(row)
    every = out["rollout"]
    out["gates"] = {
        "beatsPersistenceAtEveryLength": all(r["beatsPersistence"] for r in every),
        "beatsLinearUpliftAtEveryLength": all(r["beatsLinearUplift"] for r in every),
        "conservationWithinOnePercent": all(
            r["conservationDriftMedian"] <= 0.01 for r in every),
        # Drainage is part of the verdict, not only reported. A surface closer
        # in metres that routes water worse than not updating at all has not
        # learned landscape evolution, and height error cannot see the
        # difference.
        "beatsPersistenceOnDrainage": all(
            r["drainage"]["beatsPersistenceOnStreams"] for r in every
            if r.get("drainage")),
        "longestRollout": max((r["steps"] for r in every), default=0),
    }
    out["gates"]["passes"] = all(v for k, v in out["gates"].items()
                                 if isinstance(v, bool))
    out["note"] = ("Baselines do the easy part on purpose: most of a landscape is a "
                   "ramp that barely moves in one interval, so height error alone "
                   "flatters any model. Drainage is read at the ensemble's own "
                   "spacing with a 500-cell stream threshold.")
    return out


def campaign(folder, torch, kinds=("unet", "fno"), steps: int = 2000,
             batch: int = 8, lr: float = 1e-3, seed: int = 1729,
             device: str = "cuda") -> dict:
    """Train and gate each emulator family, returning one combined result."""
    data = EnsembleData(folder, torch, device)
    if not data.manifest.get("spacingIsAdequate", False):
        raise ValueError(
            f"the ensemble was generated at {data.manifest['spacingM']} m, coarser "
            f"than the {data.manifest['adequateSpacingM']} m the teacher audit found "
            "adequate; an emulator trained on it learns a teacher that does not obey "
            "its own steady-state law")
    rows = []
    for kind in kinds:
        config = ({"kind": "unet", "stages": 3, "base": 32, "blocks": 1, "in_channels": 4}
                  if kind == "unet" else
                  {"kind": "fno", "blocks": 4, "width": 32, "modes": 16, "in_channels": 4})
        fitted = train(config, data, torch, steps=steps, batch=batch, lr=lr, seed=seed)
        gates = evaluate(fitted["model"], config, data, torch)
        rows.append({"kind": kind, "config": config,
                     "parameters": fitted["parameters"],
                     "seconds": fitted["seconds"], "history": fitted["history"],
                     "evaluation": gates})
        del fitted
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    return {"schema": SCHEMA, "rows": rows,
            "ensemble": {k: data.manifest[k] for k in
                         ("count", "side", "spacingM", "domainM", "years",
                          "adequateSpacingM", "spacingIsAdequate",
                          "framesPerSimulation")},
            "split": {k: len(v) for k, v in data.split.items() if k.endswith("Ids")},
            "steps": steps, "batch": batch, "lr": lr, "seed": seed,
            "qualification":
                "Scored on the top-decile corner of (log Nf, log Nh), held out whole. "
                "Under the similarity law a random split over simulations leaves a "
                "rescaled twin of every test case in training, so it holds nothing "
                "out; this split is the only one here that does."}
