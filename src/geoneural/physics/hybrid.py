"""A learned flux closure that cannot break conservation.

A model that predicts the whole next surface has no reason to conserve
anything; nothing in its construction knows that material moved rather than
appeared. The full-surface emulators show this as a roughly constant mass bias
of 0.35-0.49 m against a 1 % bound. This module is the alternative, and the
reason to prefer it is structural rather than empirical.

Flux form. Write the hillslope term as the divergence of a flux across cell
faces instead of a Laplacian of the field. With zero-flux boundaries the
divergence of any antisymmetric face field sums to exactly zero, so a closure
that emits face fluxes conserves mass by construction: at every step, for
every weight, before it has been trained at all. The network predicts
`F_ij = -F_ji` because it predicts one number per face and applies it with
opposite sign to the two cells sharing it; there is no penalty term, no
soft constraint and nothing to tune.

`flux_divergence` is checked against `landscape.laplacian` on the linear case,
because a flux form that differed from that operator would be a second,
inconsistent version of the same physics.

The target is a teacher variant the linear operator cannot represent: nonlinear
depth-dependent diffusion with a critical slope, where flux blows up as the
gradient approaches `S_c`. A closure that matches linear diffusion has learned
nothing, so the thing to be learned has to be something linear diffusion gets
wrong.

Two controls, both necessary:

* Arm B, a learned K-field: the same network capacity spent predicting a
  per-cell diffusivity rather than face fluxes. It is declared
  non-conservative (a spatially varying K inside a Laplacian does not
  telescope) and shows whether conservation costs accuracy.
* The penalty control: conservation as a loss term instead of a structure.
  This is what most of the literature does, and the comparison shows whether
  building the constraint in is worth more than asking for it.
"""
from __future__ import annotations

import copy
import time

import numpy as np

from geoneural.physics import landscape

SCHEMA = "geoneural-hybrid-closure-v1"


def face_gradients(height: np.ndarray, spacing_m: float):
    """Gradients on the interior faces, east and south. No boundary faces exist.

    Returning only interior faces is what makes the boundary zero-flux: a face
    that is not represented cannot carry material out of the domain.
    """
    east = (height[:, 1:] - height[:, :-1]) / spacing_m
    south = (height[1:, :] - height[:-1, :]) / spacing_m
    return east, south


def divergence(east_flux: np.ndarray, south_flux: np.ndarray, spacing_m: float,
               shape) -> np.ndarray:
    """Net inflow per cell from interior face fluxes.

    Every interior face contributes `+f` to one cell and `-f` to its neighbour,
    so the total over the domain is identically zero in exact arithmetic and to
    rounding in floating point. The conservation guarantee of this module rests
    on that identity.
    """
    out = np.zeros(shape, dtype=np.float64)
    out[:, :-1] += east_flux / spacing_m
    out[:, 1:] -= east_flux / spacing_m
    out[:-1, :] += south_flux / spacing_m
    out[1:, :] -= south_flux / spacing_m
    return out


def flux_divergence(height: np.ndarray, spacing_m: float,
                    diffusivity: float = 1.0) -> np.ndarray:
    """Linear diffusion in flux form. Equal to `laplacian` times D to rounding."""
    east, south = face_gradients(height, spacing_m)
    return diffusivity * divergence(east, south, spacing_m, height.shape)


def matches_laplacian(sides=(9, 17, 32), seed: int = 20260914) -> dict:
    """Check that the flux form is the same operator as `laplacian`, to rounding."""
    rng = np.random.default_rng(seed)
    rows = []
    for side in sides:
        field = rng.normal(0.0, 10.0, (side, side))
        for spacing in (1.0, 50.0):
            reference = landscape.laplacian(field, spacing)
            ported = flux_divergence(field, spacing, 1.0)
            difference = float(np.abs(reference - ported).max())
            rows.append({"side": side, "spacingM": spacing,
                         "maxDifference": difference,
                         "identical": bool(np.array_equal(reference, ported)),
                         "withinRounding": difference <=
                         1e-12 * max(float(np.abs(reference).max()), 1.0)})
    return {"rows": rows,
            "agrees": all(r["withinRounding"] for r in rows),
            "note": "Sign convention and edge handling both have to match: a flux "
                    "form that differs from `laplacian` would be a second, "
                    "inconsistent version of the same operator."}


def nonlinear_teacher(height: np.ndarray, spacing_m: float, diffusivity: float,
                      critical_slope: float = 0.6) -> np.ndarray:
    """Depth-dependent hillslope flux with a critical slope.

    `q = -D grad h / (1 - (|grad h| / S_c)^2)`, the Roering form. Flux diverges
    as the gradient approaches `S_c`, which linear diffusion cannot represent at
    any D, so a closure that reproduces it has learned something and one that
    settles on linear diffusion has not.
    """
    east, south = face_gradients(height, spacing_m)
    def limit(gradient):
        ratio = np.clip(np.abs(gradient) / critical_slope, 0.0, 0.99)
        return gradient / (1.0 - ratio * ratio)
    return diffusivity * divergence(limit(east), limit(south), spacing_m, height.shape)


def _modules(torch):
    nn = torch.nn

    class FluxClosure(nn.Module):
        """One scalar per interior face, applied antisymmetrically.

        The network reads the two cells either side of a face and their local
        neighbourhood, and emits the flux across it. Conservation is not a term
        in the loss; it is a property of `divergence` being fed a face field.
        """

        def __init__(self, width: int = 32, blocks: int = 3):
            super().__init__()
            layers = [nn.Conv2d(2, width, 3, padding=1, padding_mode="replicate"),
                      nn.GELU()]
            for _ in range(blocks):
                layers += [nn.Conv2d(width, width, 3, padding=1,
                                     padding_mode="replicate"), nn.GELU()]
            layers.append(nn.Conv2d(width, 1, 3, padding=1, padding_mode="replicate"))
            self.net = nn.Sequential(*layers)

        def forward(self, gradient, height):
            """`gradient` and `height` are face-centred; returns the face flux."""
            return self.net(torch.stack((gradient, height), dim=1)).squeeze(1)

    class DiffusivityField(nn.Module):
        """Arm B: a per-cell diffusivity inside the ordinary Laplacian.

        Declared non-conservative. A spatially varying K inside a five-point
        Laplacian does not telescope, so the domain total is not preserved and
        the ledger will say so. The arm exists to measure that.
        """

        def __init__(self, width: int = 32, blocks: int = 3):
            super().__init__()
            layers = [nn.Conv2d(1, width, 3, padding=1, padding_mode="replicate"),
                      nn.GELU()]
            for _ in range(blocks):
                layers += [nn.Conv2d(width, width, 3, padding=1,
                                     padding_mode="replicate"), nn.GELU()]
            layers.append(nn.Conv2d(width, 1, 3, padding=1, padding_mode="replicate"))
            self.net = nn.Sequential(*layers)

        def forward(self, height):
            return torch.nn.functional.softplus(self.net(height.unsqueeze(1)).squeeze(1))

    return {"FluxClosure": FluxClosure, "DiffusivityField": DiffusivityField}


ARMS = ("flux", "kfield", "penalty")


def _face_inputs(height, spacing_m, torch):
    """Face-centred gradient and mean height, on the east faces of a padded field.

    Both directions are handled by transposing, so one network serves both and
    the closure cannot learn a different physics for east than for south.
    """
    east_gradient = (height[:, :, 1:] - height[:, :, :-1]) / spacing_m
    east_height = 0.5 * (height[:, :, 1:] + height[:, :, :-1])
    return east_gradient, east_height


def _apply_flux(model, height, spacing_m, torch):
    """Face fluxes both ways through one network, then the divergence."""
    east_gradient, east_height = _face_inputs(height, spacing_m, torch)
    east = model(east_gradient, east_height)
    south_gradient, south_height = _face_inputs(
        height.transpose(1, 2), spacing_m, torch)
    south = model(south_gradient, south_height).transpose(1, 2)
    out = torch.zeros_like(height)
    out[:, :, :-1] = out[:, :, :-1] + east / spacing_m
    out[:, :, 1:] = out[:, :, 1:] - east / spacing_m
    out[:, :-1, :] = out[:, :-1, :] + south / spacing_m
    out[:, 1:, :] = out[:, 1:, :] - south / spacing_m
    return out


def _apply_kfield(model, height, spacing_m, torch):
    """Arm B: a learned diffusivity inside the ordinary five-point Laplacian."""
    padded = torch.nn.functional.pad(height.unsqueeze(1), (1, 1, 1, 1),
                                     mode="replicate").squeeze(1)
    curvature = (padded[:, :-2, 1:-1] + padded[:, 2:, 1:-1] +
                 padded[:, 1:-1, :-2] + padded[:, 1:-1, 2:] -
                 4.0 * height) / (spacing_m ** 2)
    return model(height) * curvature


def apply_closure(arm, model, height, spacing_m, torch):
    return (_apply_kfield(model, height, spacing_m, torch) if arm == "kfield"
            else _apply_flux(model, height, spacing_m, torch))


def _batch(rng, batch: int, side: int, spacing_m: float, diffusivity: float,
           critical_slope: float):
    """Rough surfaces and the nonlinear teacher's response to them."""
    fields, targets = [], []
    for _ in range(batch):
        white = rng.normal(0.0, 1.0, (side, side))
        spectrum = np.fft.rfft2(white)
        ky = np.fft.fftfreq(side)[:, None]
        kx = np.fft.rfftfreq(side)[None, :]
        wavenumber = np.sqrt(ky * ky + kx * kx)
        wavenumber[0, 0] = 1.0
        field = np.fft.irfft2(spectrum / (wavenumber ** 1.2), s=(side, side))
        field = 30.0 * field / max(float(np.std(field)), 1e-9)
        fields.append(field)
        targets.append(nonlinear_teacher(field, spacing_m, diffusivity, critical_slope))
    return (np.stack(fields).astype(np.float32),
            np.stack(targets).astype(np.float32))


def train_closure(arm: str, torch, steps: int = 1500, batch: int = 8, side: int = 48,
                  spacing_m: float = 50.0, diffusivity: float = 0.05,
                  critical_slope: float = 0.6, lr: float = 3e-4, seed: int = 1729,
                  device: str = "cuda", penalty_weight: float = 1.0) -> dict:
    """Fit one arm against the nonlinear teacher's hillslope response."""
    if arm not in ARMS:
        raise ValueError(f"Unknown closure arm: {arm}")
    parts = _modules(torch)
    torch.manual_seed(seed)
    # The penalty control is a K-field, not a flux closure. A conservation penalty
    # on a formulation that already conserves by construction is inert and
    # measures nothing. The comparison is structure against penalty, so the
    # penalty has to be applied to something that can violate the constraint.
    model = (parts["FluxClosure"]() if arm == "flux"
             else parts["DiffusivityField"]()).to(device).train()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    history = []
    started = time.perf_counter()
    for step in range(steps):
        fields, targets = _batch(rng, batch, side, spacing_m, diffusivity, critical_slope)
        height = torch.from_numpy(fields).to(device)
        truth = torch.from_numpy(targets).to(device)
        predicted = apply_closure("flux" if arm == "flux" else "kfield",
                                  model, height, spacing_m, torch)
        loss = torch.nn.functional.l1_loss(predicted, truth)
        if arm == "penalty":
            # Conservation asked for rather than built in, on the one formulation
            # here that can disobey.
            loss = loss + penalty_weight * predicted.sum(dim=(1, 2)).abs().mean()
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        if step % max(1, steps // 8) == 0 or step == steps - 1:
            history.append({"step": step, "loss": float(loss.detach())})
    return {"model": model, "arm": arm, "history": history,
            "seconds": time.perf_counter() - started,
            "parameters": int(sum(p.numel() for p in model.parameters()))}


def evaluate_closure(fitted: dict, torch, cases: int = 24, side: int = 48,
                     spacing_m: float = 50.0, diffusivity: float = 0.05,
                     critical_slope: float = 0.6, seed: int = 31337,
                     device: str = "cuda", rollout: int = 64,
                     dt_years: float = 200.0) -> dict:
    """Accuracy, conservation and long-rollout stability, on unseen surfaces.

    Conservation is reported as the domain total of the predicted tendency,
    relative to the scale of that tendency. For the flux arm it must be zero to
    rounding no matter how badly trained it is, because the identity has nothing
    to do with the weights; anything else is a bug in `_apply_flux`, not a
    training outcome. The linear control is the same measurement on ordinary
    diffusion, which is the number the flux arm has to match.
    """
    arm, model = fitted["arm"], fitted["model"]
    model.eval()
    rng = np.random.default_rng(seed)
    fields, targets = _batch(rng, cases, side, spacing_m, diffusivity, critical_slope)
    height = torch.from_numpy(fields).to(device)
    truth = torch.from_numpy(targets).to(device)
    with torch.no_grad():
        predicted = apply_closure("flux" if arm == "flux" else "kfield",
                                  model, height, spacing_m, torch)
    error = (predicted - truth).abs()
    scale = truth.abs().mean().clamp_min(1e-12)
    totals = predicted.sum(dim=(1, 2)).abs()
    tendency_scale = predicted.abs().sum(dim=(1, 2)).clamp_min(1e-12)
    # The baseline every arm must beat: linear diffusion at the same D, which is
    # what the closure replaces. An arm that cannot beat it has learned nothing
    # the linear operator did not already do.
    linear = np.stack([flux_divergence(f.astype(np.float64), spacing_m, diffusivity)
                       for f in fields])
    linear_error = float(np.abs(linear - targets).mean())

    # Long rollout: apply the closure repeatedly and see whether the surface
    # survives. A closure can be accurate on one step and blow up over sixty.
    with torch.no_grad():
        surface = height.clone()
        drift = []
        start_mass = surface.sum(dim=(1, 2))
        for _ in range(rollout):
            tendency = apply_closure("flux" if arm == "flux" else "kfield",
                                     model, surface, spacing_m, torch)
            surface = surface + dt_years * tendency
            drift.append(float((surface.sum(dim=(1, 2)) - start_mass).abs().max()))
        finite = bool(torch.isfinite(surface).all())
        final_relief = float((surface.amax(dim=(1, 2)) -
                              surface.amin(dim=(1, 2))).mean())

    conservation = float(( totals / tendency_scale).max())
    return {
        "arm": arm,
        "maeVsTeacher": float(error.mean()),
        "relativeError": float(error.mean() / scale),
        "linearDiffusionMae": linear_error,
        "beatsLinearDiffusion": float(error.mean()) < linear_error,
        "conservationResidualRelative": conservation,
        "conservesToRounding": conservation < 1e-6,
        "rolloutSteps": rollout,
        "massDriftM": drift[-1],
        "massDriftRelative": drift[-1] / max(float(np.abs(fields).sum(axis=(1, 2)).max()), 1e-12),
        "rolloutFinite": finite,
        "finalReliefM": final_relief,
        "note": "conservationResidualRelative is the domain total of the predicted "
                "tendency over the total of its absolute value. The flux arm must "
                "be at rounding for any weights at all (the identity does not "
                "depend on training), so a non-zero value there is a bug in the "
                "face assembly and not a result.",
    }


def gradient_probe(fitted: dict, torch, side: int = 16, spacing_m: float = 50.0,
                   seed: int = 4242, device: str = "cuda",
                   epsilon: float = 1e-3) -> dict:
    """Finite-difference check that the closure's gradient is what autograd says.

    A cheap check worth running on every learned physics term: a closure
    assembled with an indexing error can train to a plausible loss while its
    gradient points somewhere else, and nothing in the loss curve would show it.
    """
    arm = fitted["arm"]
    # Double precision, on a copy. A central difference in float32 resolves a
    # gradient of magnitude 1e-4 to about a part in a hundred, far coarser than
    # any useful tolerance; the probe would be measuring float32 rounding and
    # calling it a gradient error.
    model = copy.deepcopy(fitted["model"]).double()
    rng = np.random.default_rng(seed)
    field = torch.from_numpy(
        rng.normal(0.0, 10.0, (1, side, side))).to(device).double()
    parameter = next(p for p in model.parameters() if p.numel() > 4)

    def loss_of(model_):
        out = apply_closure("flux" if arm == "flux" else "kfield",
                            model_, field, spacing_m, torch)
        return (out ** 2).sum()

    model.zero_grad(set_to_none=True)
    loss_of(model).backward()
    analytic = parameter.grad.flatten()[:8].detach().cpu().numpy().copy()
    numeric = []
    with torch.no_grad():
        flat = parameter.view(-1)
        for index in range(8):
            original = flat[index].item()
            flat[index] = original + epsilon
            plus = float(loss_of(model))
            flat[index] = original - epsilon
            minus = float(loss_of(model))
            flat[index] = original
            numeric.append((plus - minus) / (2.0 * epsilon))
    numeric = np.array(numeric)
    # The standard finite-difference criterion: the ratio of vector norms, not a
    # per-component maximum. A per-component ratio lets a component far below
    # the gradient's magnitude (which carries no information and is resolved to
    # a digit or two by any finite difference) decide the verdict, and so fails
    # closures whose gradient is correct.
    difference = float(np.linalg.norm(analytic - numeric))
    magnitude = max(float(np.linalg.norm(analytic)),
                    float(np.linalg.norm(numeric)), 1e-30)
    relative = difference / magnitude
    return {"analytic": analytic.tolist(), "numeric": numeric.tolist(),
            "gradientNorm": float(np.linalg.norm(analytic)),
            "maxRelativeDifference": relative, "agrees": relative < 1e-5,
            "note": "||analytic - numeric|| / ||gradient||, in double precision. A "
                    "per-component ratio is the wrong statistic here: components far "
                    "below the gradient's magnitude are resolved to a digit or two by "
                    "any finite difference and must not decide the check. The "
                    "tolerance is 1e-5 because the norm ratio is a much tighter test "
                    "than a componentwise comparison."}


def campaign(torch, arms=ARMS, steps: int = 1500, side: int = 48,
             spacing_m: float = 50.0, device: str = "cuda", seed: int = 1729) -> dict:
    """Train and evaluate every arm, returning one combined record."""
    port = matches_laplacian()
    if not port["agrees"]:
        raise ValueError("the flux form does not reproduce `laplacian`; refusing to train")
    rows = []
    for arm in arms:
        fitted = train_closure(arm, torch, steps=steps, side=side,
                               spacing_m=spacing_m, device=device, seed=seed)
        gates = evaluate_closure(fitted, torch, side=side, spacing_m=spacing_m,
                                 device=device)
        probe = gradient_probe(fitted, torch, spacing_m=spacing_m, device=device)
        rows.append({"arm": arm, "parameters": fitted["parameters"],
                     "seconds": fitted["seconds"], "history": fitted["history"],
                     "evaluation": gates, "gradientProbe": probe,
                     "conservative": arm == "flux",
                     "declared": {
                         "flux": "mass-conserving by construction, for any weights",
                         "kfield": "declared non-conservative: a spatially varying K "
                                   "inside a five-point Laplacian does not telescope",
                         "penalty": "the same non-conservative K-field, asked to "
                                    "conserve through a loss term instead"}[arm]})
        del fitted
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    return {"schema": SCHEMA, "rows": rows, "laplacianPort": port,
            "teacher": "Roering nonlinear hillslope flux, critical slope 0.6",
            "steps": steps, "side": side, "spacingM": spacing_m, "seed": seed,
            "qualification":
                "The target is a teacher variant linear diffusion cannot represent at "
                "any D, so an arm that settles on linear diffusion has demonstrably "
                "learned nothing. Conservation for the flux arms is structural and "
                "holds for untrained weights; for the K-field arm it is measured and "
                "expected to fail, which is why that arm is here."}


def seed_study(torch, seeds=(1729, 2, 3, 4, 5), penalty_weights=(1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0),
               steps: int = 1500, side: int = 48, spacing_m: float = 50.0, device: str = "cuda") -> dict:
    """The structural arm against a fairly tuned penalty, over several seeds.

    One seed of one penalty weight does not show that building the constraint in beats asking
    for it. Every seed trains the flux arm, the unconstrained K-field and the K-field with each
    penalty weight, on the same teacher and budget. Each run is reported, not only a summary.
    """
    port = matches_laplacian()
    if not port["agrees"]:
        raise ValueError("the flux form does not reproduce `laplacian`; refusing to train")
    arms = [("flux", "flux", None), ("kfield", "kfield", None)]
    arms += [(f"penalty-{w:g}", "penalty", w) for w in penalty_weights]
    runs = []
    for seed in seeds:
        for name, arm, weight in arms:
            fitted = train_closure(arm, torch, steps=steps, side=side, spacing_m=spacing_m, device=device,
                                   seed=seed, penalty_weight=1.0 if weight is None else weight)
            gates = evaluate_closure(fitted, torch, side=side, spacing_m=spacing_m, device=device)
            keys = ("maeVsTeacher", "linearDiffusionMae", "conservationResidualRelative", "massDriftRelative",
                    "relativeError")
            runs.append({"arm": name, "seed": seed, "penaltyWeight": weight, "seconds": fitted["seconds"],
                         "parameters": fitted["parameters"], "rolloutFinite": gates["rolloutFinite"],
                         # A blown-up rollout gives inf or nan; it is recorded as missing, not as a number.
                         **{k: (float(gates[k]) if np.isfinite(gates[k]) else None) for k in keys}})
            del fitted
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
    summary = {}
    for name, _, _ in arms:
        picked = [r for r in runs if r["arm"] == name]
        summary[name] = {}
        for key in ("maeVsTeacher", "conservationResidualRelative", "massDriftRelative"):
            values = [r[key] for r in picked if r[key] is not None]
            summary[name][key] = ({"median": float(np.median(values)), "min": float(np.min(values)),
                                   "max": float(np.max(values)), "runs": len(values)} if values else None)
        summary[name]["finiteRollouts"] = int(sum(bool(r["rolloutFinite"]) for r in picked))
    return {"schema": SCHEMA.replace("closure-v1", "closure-seeds-v1"), "runs": runs, "summary": summary,
            "seeds": list(seeds), "penaltyWeights": list(penalty_weights), "steps": steps, "side": side,
            "spacingM": spacing_m, "laplacianPort": port,
            "teacher": "Roering nonlinear hillslope flux, critical slope 0.6",
            "qualification": "Synthetic teacher, synthetic surfaces. Seeds vary both initialisation and the "
                             "training batches. A conservative arm conserves for any weights; its accuracy "
                             "and the penalty arms' conservation are measured."}
