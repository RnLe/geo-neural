"""A geological event field, and the ablation that has to fail.

A folded and faulted rock volume is not a function of position in any smooth
sense: across a fault the stratigraphy is displaced, so a network asked to
interpolate through one must either smear the discontinuity or memorise it. The
alternative is to represent the fault as an event: a coordinate transform
applied before the smooth field is evaluated, so the discontinuity is in the
geometry rather than in the function.

The claim to be tested is therefore not "the network fits the volume". It is
that the event parameterisation fits it at the fault, where the ablation
without the event cannot. An ablation that merely scores worse overall would be
consistent with the event arm having more capacity; the ablation has to fail
specifically at the fault and match away from it, and that is what
`by_distance_to_fault` reports.

Synthetic only: the GK100 carries 34 fault traces with no dip, no throw and no
depth extent, so a real-data version of this experiment has no target to fit.
`geoneural.data.geology.fault_rasters` supplies trace geometry and nothing
else.
"""
from __future__ import annotations

import time

import numpy as np

SCHEMA = "geoneural-structure-v1"

UNITS = 5


def synthetic_volume(side: int = 48, depth: int = 24, throw_m: float = 180.0,
                     fold_amplitude_m: float = 120.0, fold_wavelength: float = 1.0,
                     dip: float = 0.35, seed: int = 1729) -> dict:
    """A folded five-unit layer cake cut by one normal fault.

    The fault is a plane; the hanging wall is displaced down-dip by `throw_m`.
    Stratigraphic height is a smooth function of position after that
    displacement is undone, which is exactly the structure the event
    parameterisation encodes and the plain field does not.
    """
    axis_x = np.linspace(0.0, 1.0, side)
    axis_y = np.linspace(0.0, 1.0, side)
    axis_z = np.linspace(0.0, 1.0, depth)
    grid_x, grid_y, grid_z = np.meshgrid(axis_x, axis_y, axis_z, indexing="ij")

    # The fault plane: x = 0.5 + dip * z, so it leans with depth.
    plane = 0.5 + dip * (grid_z - 0.5)
    hanging_wall = grid_x > plane

    # Undo the throw, then evaluate a smooth folded stratigraphy.
    elevation_m = 1000.0 * grid_z + np.where(hanging_wall, throw_m, 0.0)
    fold = fold_amplitude_m * np.sin(2.0 * np.pi * fold_wavelength * grid_y)
    stratigraphic = elevation_m + fold
    unit = np.clip((stratigraphic / (1000.0 / UNITS)).astype(int), 0, UNITS - 1)
    rng = np.random.default_rng(seed)
    return {
        "coordinates": np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3),
        "stratigraphic": stratigraphic.reshape(-1),
        "unit": unit.reshape(-1),
        "hangingWall": hanging_wall.reshape(-1),
        "distanceToFault": np.abs(grid_x - plane).reshape(-1),
        "shape": (side, side, depth),
        "throwM": throw_m, "dip": dip, "foldAmplitudeM": fold_amplitude_m,
        "seed": int(rng.integers(0, 2 ** 31)),
        "note": "Synthetic. The GK100 carries fault traces with no dip, no throw and "
                "no depth extent, so there is no real-data target for this experiment.",
    }


def borehole_split(volume: dict, boreholes: int = 40, held_out: int = 10,
                   block: tuple = (0.55, 0.85), seed: int = 1729) -> dict:
    """Hold out whole boreholes and one spatial block, never random samples.

    Random held-out samples in a dense volume are surrounded by training points
    a few millimetres away, so a model that memorises interpolates them
    perfectly. Whole boreholes and a contiguous block are the only splits here
    that ask the model to say something it was not told.
    """
    side, _, depth = volume["shape"]
    rng = np.random.default_rng(seed)
    columns = rng.choice(side * side, boreholes, replace=False)
    held = set(columns[:held_out].tolist())
    train_columns = [c for c in columns[held_out:]]
    coordinates = volume["coordinates"]
    column_index = np.arange(side * side).repeat(depth)
    in_block = ((coordinates[:, 0] >= block[0]) & (coordinates[:, 0] <= block[1]) &
                (coordinates[:, 1] >= block[0]) & (coordinates[:, 1] <= block[1]))
    is_train = np.isin(column_index, train_columns) & ~in_block
    is_borehole_test = np.isin(column_index, list(held))
    return {"trainIndex": np.flatnonzero(is_train),
            "boreholeTestIndex": np.flatnonzero(is_borehole_test & ~in_block),
            "blockTestIndex": np.flatnonzero(in_block),
            "boreholes": boreholes, "heldOutBoreholes": held_out, "block": list(block),
            "note": "Whole boreholes and a contiguous block. A random split in a dense "
                    "volume surrounds every test point with training points and "
                    "rewards memorisation."}


ARMS = ("event", "oracle", "plain")

# `oracle` is the diagnostic that makes the other two interpretable. It is the
# event parameterisation with the fault plane and throw fixed at their true
# values, fitting only the smooth field. If `oracle` is much better than
# `event`, the event arm's problem is finding the geometry, an optimisation
# failure. If `oracle` is no better than `plain`, the parameterisation itself
# buys nothing on this problem and no amount of search would have helped.
# Without it, a tie between `event` and `plain` has two explanations and the
# experiment cannot choose.


def _modules(torch):
    nn = torch.nn

    class Siren3(nn.Module):
        """A SIREN over three coordinates. The smooth field, in every arm."""

        def __init__(self, width: int = 128, depth: int = 4, omega: float = 12.0):
            super().__init__()
            self.first = nn.Linear(3, width)
            self.hidden = nn.ModuleList(
                [nn.Linear(width, width) for _ in range(depth - 1)])
            self.out = nn.Linear(width, 1)
            self.omega = float(omega)
            with torch.no_grad():
                self.first.weight.uniform_(-1.0 / 3, 1.0 / 3)
                bound = (6.0 / width) ** 0.5 / self.omega
                for layer in self.hidden:
                    layer.weight.uniform_(-bound, bound)

        def forward(self, coordinates):
            value = torch.sin(self.omega * self.first(coordinates))
            for layer in self.hidden:
                value = torch.sin(self.omega * layer(value))
            return self.out(value).squeeze(-1)

    class EventField(nn.Module):
        """A fault as a learned coordinate transform, applied before the field.

        The displacement is a single learned vector gated by a learned sigmoid
        across a learned plane: the discontinuity lives in the geometry, and the
        field itself stays smooth and can be a plain SIREN. Nothing here is told
        where the fault is: the plane's offset, dip and the throw are all
        parameters.
        """

        def __init__(self, width: int = 128, depth: int = 4, omega: float = 12.0,
                     sharpness: float = 8.0):
            super().__init__()
            self.field = Siren3(width, depth, omega)
            self.offset = nn.Parameter(torch.tensor(0.45))
            self.dip = nn.Parameter(torch.tensor(0.0))
            # A throw of exactly zero is a stationary point: with no displacement
            # the gate multiplies nothing and no gradient reaches the plane's
            # offset or dip, so the arm can never find a fault it starts by
            # denying. Seeded small and non-zero instead.
            self.throw = nn.Parameter(torch.tensor([0.0, 0.0, 0.02]))
            # A sharp gate is worse, not better: sigmoid(60 x) is flat wherever
            # it is not vertical, so the geometry parameters see gradient from a
            # sliver of samples and stay where they were initialised. The gate is
            # annealed sharp during training instead of starting there.
            self.sharpness = float(sharpness)

        def side(self, coordinates):
            """Signed distance to the fault plane, positive in the hanging wall."""
            plane = self.offset + self.dip * (coordinates[:, 2] - 0.5)
            return coordinates[:, 0] - plane

        def forward(self, coordinates):
            gate = torch.sigmoid(self.sharpness * self.side(coordinates))
            moved = coordinates + gate.unsqueeze(-1) * self.throw
            return self.field(moved)

    return {"Siren3": Siren3, "EventField": EventField}


def train_arm(arm: str, volume: dict, split: dict, torch, steps: int = 3000,
              batch: int = 4096, lr: float = 3e-4, seed: int = 1729,
              device: str = "cuda", width: int = 128, depth: int = 4,
              eikonal_weight: float = 0.0) -> dict:
    """Fit one arm to stratigraphic height on the training boreholes."""
    if arm not in ARMS:
        raise ValueError(f"Unknown structure arm: {arm}")
    parts = _modules(torch)
    torch.manual_seed(seed)
    if arm == "plain":
        model = parts["Siren3"](width, depth)
    else:
        model = parts["EventField"](width, depth)
        if arm == "oracle":
            with torch.no_grad():
                model.offset.fill_(0.5)
                model.dip.fill_(float(volume["dip"]))
                # The true displacement in coordinate units: `throw_m` of
                # stratigraphic height over the 1000 m the depth axis spans.
                model.throw.copy_(torch.tensor(
                    [0.0, 0.0, float(volume["throwM"]) / 1000.0]))
            model.offset.requires_grad_(False)
            model.dip.requires_grad_(False)
            model.throw.requires_grad_(False)
    model = model.to(device).train()
    optimiser = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr)
    index = split["trainIndex"]
    coordinates = torch.from_numpy(
        volume["coordinates"][index].astype(np.float32)).to(device)
    centre = float(volume["stratigraphic"][index].mean())
    scale = max(float(volume["stratigraphic"][index].std()), 1e-6)
    target = torch.from_numpy(
        ((volume["stratigraphic"][index] - centre) / scale).astype(np.float32)).to(device)
    rng = np.random.default_rng(seed)
    history = []
    started = time.perf_counter()
    for step in range(steps):
        if arm == "event":
            # Anneal the gate from soft to sharp. Soft early lets the plane move;
            # sharp late makes the displacement a discontinuity rather than a ramp.
            model.sharpness = 8.0 + 52.0 * (step / max(steps - 1, 1))
        elif arm == "oracle":
            # No annealing: the oracle is not searching for the fault, so a soft
            # early gate only smears a discontinuity it already knows exactly.
            # With the `event` schedule the oracle scores worse than the free arm
            # (11.69 m against 6.70 m) despite being given the true geometry, an
            # effect of the schedule and not of the parameterisation.
            model.sharpness = 60.0
        picks = torch.from_numpy(
            rng.integers(0, len(index), min(batch, len(index)))).to(device)
        points = coordinates.index_select(0, picks)
        predicted = model(points)
        loss = torch.nn.functional.mse_loss(predicted, target.index_select(0, picks))
        if eikonal_weight > 0.0:
            probe = points.clone().requires_grad_(True)
            gradient = torch.autograd.grad(model(probe).sum(), probe, create_graph=True)[0]
            loss = loss + eikonal_weight * ((gradient.norm(dim=-1) - 1.0) ** 2).mean()
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        if step % max(1, steps // 8) == 0 or step == steps - 1:
            history.append({"step": step, "loss": float(loss.detach())})
    return {"model": model, "arm": arm, "centre": centre, "scale": scale,
            "history": history, "seconds": time.perf_counter() - started,
            "parameters": int(sum(p.numel() for p in model.parameters()))}


def by_distance_to_fault(fitted: dict, volume: dict, index, torch,
                         device: str = "cuda", edges=(0.02, 0.05, 0.15, 1.0)) -> list:
    """Error binned by distance to the fault, the central measurement here.

    An ablation that is worse everywhere has less capacity. An ablation that
    matches away from the fault and fails at it has failed for the stated
    reason, and only the binned figure can tell those apart.
    """
    model = fitted["model"]
    model.eval()
    coordinates = torch.from_numpy(
        volume["coordinates"][index].astype(np.float32)).to(device)
    truth = volume["stratigraphic"][index]
    with torch.no_grad():
        predicted = (model(coordinates).cpu().numpy() * fitted["scale"] + fitted["centre"])
    error = np.abs(predicted - truth)
    distance = volume["distanceToFault"][index]
    rows, low = [], 0.0
    for high in edges:
        mask = (distance >= low) & (distance < high)
        rows.append({"distanceFrom": low, "distanceTo": high,
                     "samples": int(mask.sum()),
                     "maeM": float(error[mask].mean()) if mask.any() else None,
                     "maxM": float(error[mask].max()) if mask.any() else None})
        low = high
    return rows


def campaign(torch, side: int = 40, depth: int = 20, steps: int = 3000,
             device: str = "cuda", seed: int = 1729) -> dict:
    """Every arm, both held-out sets, error binned by distance to the fault."""
    volume = synthetic_volume(side=side, depth=depth, seed=seed)
    # Dense enough that the field is determined away from the fault. With 40
    # boreholes the training set is about 352 points for a three-dimensional
    # volume, and the arms fit it equally badly, which tests nothing.
    split = borehole_split(volume, boreholes=side * 5, held_out=side, seed=seed)
    rows = []
    for arm in ARMS:
        fitted = train_arm(arm, volume, split, torch, steps=steps,
                           device=device, seed=seed)
        entry = {"arm": arm, "parameters": fitted["parameters"],
                 "seconds": fitted["seconds"], "history": fitted["history"]}
        for name, index in (("borehole", split["boreholeTestIndex"]),
                            ("block", split["blockTestIndex"])):
            binned = by_distance_to_fault(fitted, volume, index, torch, device)
            overall = [b for b in binned if b["maeM"] is not None]
            entry[name] = {
                "byDistanceToFault": binned,
                "maeM": float(np.average([b["maeM"] for b in overall],
                                         weights=[b["samples"] for b in overall]))}
        if arm == "event":
            entry["recovered"] = {
                "offset": float(fitted["model"].offset.detach()),
                "dip": float(fitted["model"].dip.detach()),
                "throw": [float(v) for v in fitted["model"].throw.detach()],
                "trueOffset": 0.5, "trueDip": volume["dip"],
                "note": "The plane and the throw are parameters, not inputs. "
                        "Recovering them is the event arm saying where it thinks "
                        "the fault is, which can be checked against where it is."}
        rows.append(entry)
        del fitted
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    return {"schema": SCHEMA, "rows": rows, "volume": {
                k: volume[k] for k in ("shape", "throwM", "dip", "foldAmplitudeM")},
            "split": {k: split[k] for k in ("boreholes", "heldOutBoreholes", "block")},
            "steps": steps, "seed": seed,
            "qualification":
                "Synthetic. The GK100 carries 34 fault traces with no dip, no throw "
                "and no depth extent, so there is no real-data target for this "
                "experiment and none is claimed. The test is not that the event arm "
                "fits the volume but that the ablation fails specifically at the "
                "fault and matches away from it; an ablation worse everywhere would "
                "only show it has less capacity."}
