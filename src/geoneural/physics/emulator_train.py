"""Train the landscape emulators and evaluate them against their gates.

The emulator advances a surface the way the teacher would, faster. Height error
alone is easy to make small: a landscape is mostly a smooth ramp, and predicting
the input reproduces most of it. So every gate is a comparison against baselines
that do exactly that, and the emulator has to beat them:

* persistence: return the input unchanged. The floor for any one-step claim.
* linear uplift: add `U dt` to every free cell. What the surface does if
  erosion is ignored entirely, which on short steps is most of what happens.

The contract is identity plus increment with pinned fixed edges
(`emulator.make_increment`): an untrained model is persistence and the
outermost ring is the base level for any weights. The legacy absolute
contract is trained beside it as a labelled comparison.

Beyond height, because a surface can be close in metres and wrong as a
landscape:

* drainage, through `hydrology.compare` with a fixed physical stream
  threshold of 0.05 km^2;
* the volume balance including the boundary: the emulator's volume change over
  a rollout against the teacher's ledger for the same interval (uplift minus
  incision minus what left through the fixed edges), relative to the uplift
  volume. A full-surface emulator has no terms for these, so only their sum
  is compared;
* errors by distance to the edge: the outermost ring (exactly zero when
  pinned), the next ring and the interior, reported separately;
* rollouts of 1, 2, 4 and 8 intervals at the realised physical times of the
  teacher frames.

Everything is scored on held-out trajectories: whole parameter blocks for
interpolation and extrapolation, and a held-out initial family, reported
separately.
"""
from __future__ import annotations

import json
import pathlib
import time

import numpy as np

from geoneural.physics import emulator

from geoneural.metrics import hydrology

SCHEMA = "geoneural-emulator-train-v2"
STREAM_AREA_M2 = 5e4


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
        self.mask = mask
        self.boundary = torch.from_numpy(mask).to(device)

    def conditioners(self, key):
        record = self.by_id[key]["job"]
        return (float(record["logFluvialNumber"]), float(record["logHillslopeNumber"]),
                float(self.by_id[key]["dimensionlessStep"]))

    def pair(self, key, index, stride: int = 1):
        """Normalised (input, target) `stride` recorded intervals apart."""
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
            [[self.conditioners(k)[0], self.conditioners(k)[1],
              self.conditioners(k)[2] * stride] for k in keys],
            dtype=torch.float32, device=self.device)
        return channels, scalars, height, target


def build(config: dict, torch, contract: str = "increment"):
    return emulator.make_increment(config, torch, pin_edges=(contract == "increment"), contract=contract)


def train(config: dict, data: EnsembleData, torch, steps: int = 2000,
          batch: int = 8, lr: float = 1e-3, seed: int = 1729,
          rollout_from: float = 0.5, rollout_length: int = 2,
          strides=(1, 2, 4), contract: str = "increment") -> dict:
    """One-step training, then a short rollout curriculum on the back half.

    A model trained only on single steps is optimised for a distribution it
    never sees at inference, where its own output is its next input; the
    resulting drift is invisible at one step and dominant at eight.
    """
    torch.manual_seed(seed)
    model = build(config, torch, contract).to(data.device).train()
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
            history.append({"step": step, "loss": float(loss.detach()), "rolloutLength": length})
    return {"model": model, "history": history, "strides": list(strides), "contract": contract,
            "seconds": time.perf_counter() - started,
            "parameters": int(sum(p.numel() for p in model.parameters()))}


def _rings(side: int):
    rows, cols = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    distance = np.minimum.reduce([rows, cols, side - 1 - rows, side - 1 - cols])
    return {"edge": distance == 0, "nextRing": distance == 1, "interior": distance >= 2}


def evaluate(model, data, torch, ids, rollout=(1, 2, 4, 8), drainage_simulations: int = 8) -> dict:
    """Every gate on one set of held-out trajectories, against persistence and linear uplift."""
    intervals = data.manifest["framesPerSimulation"] - 1
    rings = _rings(data.side)
    cell_area = data.spacing_m ** 2
    stream_cells = max(1, int(round(STREAM_AREA_M2 / cell_area)))
    drainage_ids = set(ids[:drainage_simulations])
    model.eval()
    out = {"rollout": [], "ids": len(ids), "streamCells": stream_cells}
    with torch.no_grad():
        for length in rollout:
            if length > intervals:
                continue
            errors = {name: {ring: [] for ring in ("all", *rings)} for name in ("emulator", "persistence", "linearUplift")}
            drainage = {"emulator": [], "persistence": []}
            balance, physical = [], []
            for key in ids:
                stack = data.frames[key]
                centre, scale = data.centre[key], data.scale[key]
                record = data.by_id[key]
                job = record["job"]
                current = torch.from_numpy(((stack[0] - centre) / scale)[None]).to(data.device)
                scalars = torch.tensor([data.conditioners(key)], dtype=torch.float32, device=data.device)
                mask = data.boundary.unsqueeze(0)
                for _ in range(length):
                    current = model(emulator.input_channels(current, mask, torch), scalars).squeeze(1)
                predicted = current[0].cpu().numpy() * scale + centre
                truth = stack[length].astype(np.float64)
                start = stack[0].astype(np.float64)
                span = float(record["frameTimesYears"][length])
                physical.append(span)
                uplift = start + np.where(data.mask > 0.5, 0.0, float(job["uplift"]) * span)
                for name, field in (("emulator", predicted), ("persistence", start), ("linearUplift", uplift)):
                    diff = np.abs(field - truth)
                    errors[name]["all"].append(float(diff.mean()))
                    for ring, where in rings.items():
                        errors[name][ring].append(float(diff[where].mean()))
                ledger = record["balances"][:length]
                teacher_change = sum(b["observedVolumeChangeM3"] for b in ledger)
                uplift_volume = sum(b["upliftVolumeM3"] for b in ledger)
                outflow = sum(b["boundaryOutflowVolumeM3"] for b in ledger)
                emulator_change = float((predicted - start).sum()) * cell_area
                balance.append({"relative": abs(emulator_change - teacher_change) / max(abs(uplift_volume), 1e-9),
                                "outflowShareOfUplift": outflow / max(abs(uplift_volume), 1e-9)})
                if key in drainage_ids:
                    drainage["emulator"].append(hydrology.compare(truth, predicted.astype(np.float64),
                                                                  data.spacing_m, stream_cells))
                    drainage["persistence"].append(hydrology.compare(truth, start, data.spacing_m, stream_cells))
            row = {"steps": length, "physicalYearsMedian": float(np.median(physical))}
            for name in errors:
                row[name] = {ring: float(np.mean(v)) for ring, v in errors[name].items()}
            row["beatsPersistence"] = row["emulator"]["all"] < row["persistence"]["all"]
            row["beatsLinearUplift"] = row["emulator"]["all"] < row["linearUplift"]["all"]
            row["balanceErrorOverUpliftMedian"] = float(np.median([b["relative"] for b in balance]))
            row["balanceErrorOverUpliftP90"] = float(np.quantile([b["relative"] for b in balance], 0.9))
            row["teacherOutflowShareOfUpliftMedian"] = float(np.median([b["outflowShareOfUplift"] for b in balance]))
            if drainage["emulator"]:
                def summarise(records):
                    return {"receiverAgreementFraction": float(np.mean([r["receiverAgreementFraction"] for r in records])),
                            "streamJaccard": float(np.mean([r["streamJaccard"] for r in records])),
                            "simulations": len(records)}
                row["drainage"] = {"emulator": summarise(drainage["emulator"]),
                                   "persistence": summarise(drainage["persistence"])}
            out["rollout"].append(row)
    every = out["rollout"]
    out["gates"] = {
        "beatsPersistenceAtEveryLength": all(r["beatsPersistence"] for r in every),
        "beatsLinearUpliftAtEveryLength": all(r["beatsLinearUplift"] for r in every),
        "balanceWithinOnePercent": all(r["balanceErrorOverUpliftMedian"] <= 0.01 for r in every),
        "edgeExact": all(r["emulator"]["edge"] == 0.0 for r in every),
        "beatsPersistenceOnDrainage": all(
            r["drainage"]["emulator"]["receiverAgreementFraction"] >
            r["drainage"]["persistence"]["receiverAgreementFraction"] for r in every if r.get("drainage")),
    }
    return out


def campaign(folder, torch, kinds=("unet", "fno"), steps: int = 2000,
             batch: int = 8, lr: float = 1e-3, seed: int = 1729,
             device: str = "cuda", legacy: bool = True) -> dict:
    """Train and gate each emulator family, plus the legacy absolute U-Net, on every held-out split."""
    data = EnsembleData(folder, torch, device)
    if not data.manifest.get("spacingIsAdequate", False):
        raise ValueError(
            f"the ensemble was generated at {data.manifest['spacingM']} m, coarser "
            f"than the {data.manifest['adequateSpacingM']} m the teacher audit found adequate")
    configs = {"unet": {"kind": "unet", "stages": 3, "base": 32, "blocks": 1, "in_channels": 4},
               "fno": {"kind": "fno", "blocks": 4, "width": 32, "modes": 16, "in_channels": 4}}
    arms = [(kind, configs[kind], "increment") for kind in kinds]
    if legacy:
        arms.append(("unet-absolute", configs["unet"], "absolute"))
    splits = {name: data.split[name] for name in ("testInterpolationIds", "testExtrapolationIds",
                                                   "testInitialFamilyIds", "testIds") if data.split.get(name)}
    if "testInterpolationIds" in splits:
        splits.pop("testIds")
    rows = []
    for name, config, contract in arms:
        fitted = train(config, data, torch, steps=steps, batch=batch, lr=lr, seed=seed, contract=contract)
        scores = {split: evaluate(fitted["model"], data, torch, ids) for split, ids in splits.items()}
        rows.append({"arm": name, "kind": config["kind"], "contract": contract, "pinnedEdges": contract == "increment",
                     "config": config, "parameters": fitted["parameters"], "seconds": fitted["seconds"],
                     "history": fitted["history"], "evaluation": scores})
        del fitted
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    return {"schema": SCHEMA, "rows": rows,
            "ensemble": {k: data.manifest.get(k) for k in
                         ("count", "side", "spacingM", "domainM", "years", "adequateSpacingM",
                          "spacingIsAdequate", "framesPerSimulation", "solver", "solverHash", "splitRule")},
            "split": {k: len(v) for k, v in data.split.items() if k.endswith("Ids")},
            "steps": steps, "batch": batch, "lr": lr, "seed": seed,
            "qualification": "Scored on whole held-out trajectories: parameter blocks for interpolation and "
                             "extrapolation, and a held-out initial family, reported separately. The balance "
                             "compares the emulator's volume change with the teacher's ledger sum including "
                             "boundary outflow; it cannot attribute the change to individual terms."}
