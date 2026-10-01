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

The free flux arm conserves, but nothing stops it moving a flat surface: on a
constant field every face sees the same input and emits the same value, which
cancels inside and not at closed edges. It also reads absolute height, so
adding a constant changes its answer, and it has no stability bound of its
own. The structured arm (`Conductance`) predicts one bounded symmetric
conductance per face from translation-invariant face features,
`a_ij = D_lin + (a_max - D_lin) * sigmoid(NN(|g|, material))`, and uses
`F_ij = a_ij (z_i - z_j)`, `F_ji = -F_ij`. A flat surface is then exactly still,
a constant offset changes nothing, `a >= D_lin > 0` dissipates the quadratic
energy under closed edges, and `dt <= dx^2 / (4 a_max)` makes every explicit
step a convex combination; rollouts use that bound instead of a fixed step.
The linear floor matters: without it the arm learned a near-zero conductance on
gentle slopes, which a single-step tendency loss dominated by steep faces
cannot see but a rollout does. `closure_study` compares it with analytic
linear diffusion, the teacher itself, a conventional conservative face
diffusivity (`FaceDiffusivity`), the free flux arm, the K-field and the penalty
arm, over five seeds, and selects on rollout error at matched physical times.

Two controls of the original study, both necessary:

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


def nonlinear_teacher(height: np.ndarray, spacing_m: float, diffusivity,
                      critical_slope: float = 0.6) -> np.ndarray:
    """Depth-dependent hillslope flux with a critical slope.

    `q = -D grad h / (1 - (|grad h| / S_c)^2)`, the Roering form. Flux diverges
    as the gradient approaches `S_c`, which linear diffusion cannot represent at
    any D, so a closure that reproduces it has learned something and one that
    settles on linear diffusion has not.

    The critical slope is applied per face, that is per grid direction: a
    ridge at 45 degrees with true slope s has face gradients s / sqrt(2) and
    is limited less than the same ridge along an axis. That is a property of
    this teacher, not of the physics it imitates, and `teacher_anisotropy`
    measures it.

    `diffusivity` is a number or a per-cell field (a lithology map); a face
    takes the harmonic mean of its two cells, which is the series conductance
    of two materials and keeps the update conservative.
    """
    east, south = face_gradients(height, spacing_m)

    def limit(gradient):
        ratio = np.clip(np.abs(gradient) / critical_slope, 0.0, 0.99)
        return gradient / (1.0 - ratio * ratio)

    if np.ndim(diffusivity) == 0:
        return diffusivity * divergence(limit(east), limit(south), spacing_m, height.shape)
    field = np.asarray(diffusivity, dtype=np.float64)
    east_d = _harmonic(field[:, :-1], field[:, 1:])
    south_d = _harmonic(field[:-1, :], field[1:, :])
    return divergence(east_d * limit(east), south_d * limit(south), spacing_m, height.shape)


def _harmonic(a, b):
    total = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total > 0.0, 2.0 * a * b / total, 0.0)


def teacher_anisotropy(spacing_m: float = 50.0, diffusivity: float = 0.05,
                       critical_slope: float = 0.6, side: int = 48) -> dict:
    """The teacher's response to one ridge profile laid along an axis and along a diagonal.

    Same true slope, same wavelength. An isotropic law gives the same peak
    tendency in both orientations; this one does not, because the critical
    slope is applied to each face gradient separately.
    """
    y, x = np.mgrid[0:side, 0:side] * spacing_m
    rows = []
    for amplitude in (10.0, 30.0, 45.0):
        out = {}
        for degrees in (0, 45):
            theta = np.radians(degrees)
            field = amplitude * np.cos(2 * np.pi * (x * np.cos(theta) + y * np.sin(theta)) / 600.0)
            response = nonlinear_teacher(field, spacing_m, diffusivity, critical_slope)[4:-4, 4:-4]
            out[degrees] = float(np.abs(response).mean())
        rows.append({"amplitudeM": amplitude, "maxTrueSlope": amplitude * 2 * np.pi / 600.0,
                     "axisMeanAbsTendency": out[0], "diagonalMeanAbsTendency": out[45],
                     "diagonalOverAxis": out[45] / max(out[0], 1e-30)})
    return {"rows": rows, "note": "Interior mean |tendency| of a cosine ridge (600 m wavelength) along "
            "an axis and along a diagonal. A ratio away from one is the per-direction critical slope."}


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

    def stack(inputs: int, width: int, blocks: int):
        layers = [nn.Conv2d(inputs, width, 3, padding=1, padding_mode="replicate"), nn.GELU()]
        for _ in range(blocks):
            layers += [nn.Conv2d(width, width, 3, padding=1, padding_mode="replicate"), nn.GELU()]
        layers.append(nn.Conv2d(width, 1, 3, padding=1, padding_mode="replicate"))
        return nn.Sequential(*layers)

    class Conductance(nn.Module):
        """A bounded symmetric face conductance with a linear floor.

        `a = floor + (a_max - floor) * sigmoid(net(features))`, one value per
        face, used as `G = a g` on that face for both cells. The features are
        the face's |g| (and, with material, the mean and absolute difference of
        the two cells' material values), all on the face grid, so the
        conductance is the same whichever cell is called i, does not change
        when a constant is added to the surface, and is unchanged when the
        surface is negated. A flat surface has g = 0 on every face and so zero
        flux for any weights; `a >= floor > 0` makes the semi-discrete update
        dissipative under closed edges, and `a <= a_max` gives the explicit
        bound dt <= dx^2 / (4 a_max).
        """

        def __init__(self, floor: float, a_max: float, material: bool = False,
                     width: int = 32, blocks: int = 3):
            super().__init__()
            if not 0.0 <= floor < a_max:
                raise ValueError("the conductance needs 0 <= floor < a_max")
            self.floor, self.a_max, self.material = float(floor), float(a_max), bool(material)
            self.net = stack(3 if material else 1, width, blocks)

        def forward(self, features):
            fraction = torch.sigmoid(self.net(features).squeeze(1))
            return self.floor + (self.a_max - self.floor) * fraction

    class FaceDiffusivity(nn.Module):
        """A conventional learned diffusivity, conservative because it lives on faces.

        One K per cell from translation-invariant cell features (the centred
        gradient magnitude, and material if given), softplus so it is
        non-negative, averaged harmonically onto faces. Unbounded above, so
        its explicit step is recomputed from the largest K at every step.
        """

        def __init__(self, material: bool = False, width: int = 32, blocks: int = 3):
            super().__init__()
            self.material = bool(material)
            self.net = stack(2 if material else 1, width, blocks)

        def forward(self, features):
            return torch.nn.functional.softplus(self.net(features).squeeze(1))

    return {"FluxClosure": FluxClosure, "DiffusivityField": DiffusivityField,
            "Conductance": Conductance, "FaceDiffusivity": FaceDiffusivity}


ARMS = ("flux", "kfield", "penalty")
STRUCTURED_ARMS = ("facek", "conductance", "conductance-nofloor")


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


def _assemble(east, south, spacing_m, height, torch):
    out = torch.zeros_like(height)
    out[:, :, :-1] = out[:, :, :-1] + east / spacing_m
    out[:, :, 1:] = out[:, :, 1:] - east / spacing_m
    out[:, :-1, :] = out[:, :-1, :] + south / spacing_m
    out[:, 1:, :] = out[:, 1:, :] - south / spacing_m
    return out


def conductances(model, height, spacing_m, torch, material=None):
    """Face conductances (east, south) of a `Conductance` model; south through the transpose."""
    def faces(field, mat):
        gradient = (field[:, :, 1:] - field[:, :, :-1]) / spacing_m
        channels = [gradient.abs()]
        if model.material:
            channels += [0.5 * (mat[:, :, 1:] + mat[:, :, :-1]), (mat[:, :, 1:] - mat[:, :, :-1]).abs()]
        return gradient, model(torch.stack(channels, dim=1))
    east_g, east_a = faces(height, material)
    south_g, south_a = faces(height.transpose(1, 2),
                             None if material is None else material.transpose(1, 2))
    return (east_g, east_a), (south_g.transpose(1, 2), south_a.transpose(1, 2))


def _apply_conductance(model, height, spacing_m, torch, material=None):
    (east_g, east_a), (south_g, south_a) = conductances(model, height, spacing_m, torch, material)
    return _assemble(east_a * east_g, south_a * south_g, spacing_m, height, torch)


def _cell_diffusivity(model, height, spacing_m, torch, material=None):
    padded = torch.nn.functional.pad(height.unsqueeze(1), (1, 1, 1, 1), mode="replicate").squeeze(1)
    gx = (padded[:, 1:-1, 2:] - padded[:, 1:-1, :-2]) / (2.0 * spacing_m)
    gy = (padded[:, 2:, 1:-1] - padded[:, :-2, 1:-1]) / (2.0 * spacing_m)
    channels = [torch.sqrt(gx * gx + gy * gy + 1e-30)]
    if model.material:
        channels.append(material)
    return model(torch.stack(channels, dim=1))


def _apply_facek(model, height, spacing_m, torch, material=None):
    k = _cell_diffusivity(model, height, spacing_m, torch, material)
    east_k = 2.0 * k[:, :, 1:] * k[:, :, :-1] / (k[:, :, 1:] + k[:, :, :-1]).clamp_min(1e-30)
    south_k = 2.0 * k[:, 1:, :] * k[:, :-1, :] / (k[:, 1:, :] + k[:, :-1, :]).clamp_min(1e-30)
    east_g = (height[:, :, 1:] - height[:, :, :-1]) / spacing_m
    south_g = (height[:, 1:, :] - height[:, :-1, :]) / spacing_m
    return _assemble(east_k * east_g, south_k * south_g, spacing_m, height, torch)


def apply_closure(arm, model, height, spacing_m, torch, material=None):
    if arm in ("kfield", "penalty"):
        return _apply_kfield(model, height, spacing_m, torch)
    if arm == "facek":
        return _apply_facek(model, height, spacing_m, torch, material)
    if arm.startswith("conductance"):
        return _apply_conductance(model, height, spacing_m, torch, material)
    return _apply_flux(model, height, spacing_m, torch)


def stable_dt(arm, model, height, spacing_m, torch, diffusivity: float = 0.05,
              material=None) -> float:
    """The explicit step each arm can take from this state, before any safety factor.

    Conductance: dx^2 / (4 a_max), a bound for every state. K-field and face-K:
    dx^2 / (4 K_max) at this state. The free flux arm has no bound of its own;
    the teacher's (the diffusivity at the slope clip) is used, which is what
    the arm was always rolled out with and is not a guarantee.
    """
    if arm.startswith("conductance"):
        return spacing_m ** 2 / (4.0 * model.a_max)
    if arm in ("kfield", "penalty"):
        return spacing_m ** 2 / (4.0 * float(model(height).max()))
    if arm == "facek":
        return spacing_m ** 2 / (4.0 * float(_cell_diffusivity(model, height, spacing_m, torch,
                                                                material).max()))
    return spacing_m ** 2 * (1.0 - 0.99 ** 2) / (4.0 * diffusivity)


def _batch(rng, batch: int, side: int, spacing_m: float, diffusivity: float,
           critical_slope: float, amplitude_m: float = 30.0, exponent: float = 1.2,
           ratio: float | None = None):
    """Rough surfaces and the nonlinear teacher's response to them.

    With `ratio`, each surface also gets a two-class lithology map (a random
    half-plane) with diffusivity `diffusivity` on class 0 and `ratio *
    diffusivity` on class 1, and the third return value is the class map.
    """
    fields, targets, materials = [], [], []
    for _ in range(batch):
        white = rng.normal(0.0, 1.0, (side, side))
        spectrum = np.fft.rfft2(white)
        ky = np.fft.fftfreq(side)[:, None]
        kx = np.fft.rfftfreq(side)[None, :]
        wavenumber = np.sqrt(ky * ky + kx * kx)
        wavenumber[0, 0] = 1.0
        field = np.fft.irfft2(spectrum / (wavenumber ** exponent), s=(side, side))
        field = amplitude_m * field / max(float(np.std(field)), 1e-9)
        fields.append(field)
        if ratio is None:
            targets.append(nonlinear_teacher(field, spacing_m, diffusivity, critical_slope))
            continue
        angle = rng.uniform(0.0, 2.0 * np.pi)
        y, x = np.mgrid[0:side, 0:side] - (side - 1) / 2.0
        offset = rng.uniform(-0.3, 0.3) * side
        classes = (x * np.cos(angle) + y * np.sin(angle) > offset).astype(np.float64)
        materials.append(classes)
        targets.append(nonlinear_teacher(field, spacing_m, diffusivity * np.where(classes > 0, ratio, 1.0),
                                         critical_slope))
    out = (np.stack(fields).astype(np.float32), np.stack(targets).astype(np.float32))
    return out if ratio is None else out + (np.stack(materials).astype(np.float32),)


def make_arm(arm: str, torch, diffusivity: float = 0.05, a_max: float = 3.0,
             material: bool = False, floor: float | None = None):
    """Build an untrained model for one arm."""
    parts = _modules(torch)
    if arm == "flux":
        return parts["FluxClosure"]()
    if arm in ("kfield", "penalty"):
        return parts["DiffusivityField"]()
    if arm == "facek":
        return parts["FaceDiffusivity"](material=material)
    if arm == "conductance":
        return parts["Conductance"](diffusivity if floor is None else floor, a_max, material=material)
    if arm == "conductance-nofloor":
        return parts["Conductance"](0.0, a_max, material=material)
    raise ValueError(f"Unknown closure arm: {arm}")


def train_closure(arm: str, torch, steps: int = 1500, batch: int = 8, side: int = 48,
                  spacing_m: float = 50.0, diffusivity: float = 0.05,
                  critical_slope: float = 0.6, lr: float = 3e-4, seed: int = 1729,
                  device: str = "cuda", penalty_weight: float = 1.0,
                  ratio: float | None = None, a_max: float = 3.0,
                  floor: float | None = None) -> dict:
    """Fit one arm against the nonlinear teacher's hillslope response.

    The penalty control is a K-field, not a flux closure. A conservation
    penalty on a formulation that already conserves by construction is inert
    and measures nothing; the comparison is structure against penalty, so the
    penalty is applied to something that can violate the constraint. With
    `ratio`, the surfaces carry a two-class lithology and the structured arms
    read the class map.
    """
    if arm not in ARMS + STRUCTURED_ARMS:
        raise ValueError(f"Unknown closure arm: {arm}")
    torch.manual_seed(seed)
    model = make_arm(arm, torch, diffusivity, a_max, material=ratio is not None,
                     floor=floor).to(device).train()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    history = []
    started = time.perf_counter()
    for step in range(steps):
        drawn = _batch(rng, batch, side, spacing_m, diffusivity, critical_slope, ratio=ratio)
        height = torch.from_numpy(drawn[0]).to(device)
        truth = torch.from_numpy(drawn[1]).to(device)
        material = torch.from_numpy(drawn[2]).to(device) if ratio is not None else None
        predicted = apply_closure(arm, model, height, spacing_m, torch, material)
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


def advance(arm, model, surface, years, spacing_m, torch, diffusivity: float = 0.05,
            material=None, safety: float = 1.0, teacher=None):
    """Advance by `years` in equal substeps no longer than `safety` times the arm's bound.

    `arm` may also be "linear" (uniform `diffusivity`) or "teacher" (the
    nonlinear teacher in float64 numpy, `teacher = (diffusivity field or value,
    critical slope)`), so that every model reaches the same physical times.
    """
    if arm in ("linear", "teacher"):
        values = surface.detach().cpu().double().numpy()
        out = []
        for index, field in enumerate(values):
            if arm == "linear":
                bound = spacing_m ** 2 / (4.0 * diffusivity)
                rate = lambda h: flux_divergence(h, spacing_m, diffusivity)  # noqa: E731
            else:
                d, critical = teacher
                d_cell = d if np.ndim(d) == 0 else d[index]
                bound = spacing_m ** 2 * (1.0 - 0.99 ** 2) / (4.0 * float(np.max(d_cell)))
                rate = lambda h: nonlinear_teacher(h, spacing_m, d_cell, critical)  # noqa: E731
            pieces = max(1, int(np.ceil(years / (safety * bound) - 1e-12)))
            for _ in range(pieces):
                field = field + (years / pieces) * rate(field)
            out.append(field)
        return torch.from_numpy(np.stack(out)).to(device=surface.device, dtype=surface.dtype)
    bound = stable_dt(arm, model, surface, spacing_m, torch, diffusivity, material)
    pieces = max(1, int(np.ceil(years / (safety * bound) - 1e-12)))
    for _ in range(pieces):
        surface = surface + (years / pieces) * apply_closure(arm, model, surface, spacing_m, torch, material)
    return surface


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
        predicted = apply_closure(arm, model, height, spacing_m, torch)
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
    # Each interval of dt_years is split into substeps below the arm's own
    # explicit bound (`stable_dt`), rather than taken whole whatever the arm.
    with torch.no_grad():
        surface = height.clone()
        drift = []
        start_mass = surface.sum(dim=(1, 2))
        for _ in range(rollout):
            surface = advance(arm, model, surface, dt_years, spacing_m, torch, diffusivity)
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
        out = apply_closure(arm, model_, field, spacing_m, torch)
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


# The structured-closure study: selection on rollouts at matched physical times.

STUDY_ARMS = ("flux", "kfield", "penalty", "facek", "conductance", "conductance-nofloor")
CANDIDATES = ("flux", "penalty", "facek", "conductance", "conductance-nofloor")
HORIZONS = (8, 32, 64)
INTERVAL_YEARS = 200.0
A_MAX = 3.0            # above the teacher's largest face conductance, D / (1 - 0.99^2) = 2.51
MATERIAL_RATIO = 0.2   # class-1 diffusivity over class-0 in the material-contrast case
MAX_SUBSTEPS = 200


def corrugations(side: int = 48, spacing_m: float = 50.0) -> np.ndarray:
    """Cosine ridges at 0, 45 and 90 degrees, two amplitudes, four phases each."""
    y, x = np.mgrid[0:side, 0:side] * spacing_m
    fields = []
    for amplitude in (30.0, 45.0):
        for degrees in (0, 45, 90):
            theta = np.radians(degrees)
            for phase in np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False):
                fields.append(amplitude * np.cos(2 * np.pi * (x * np.cos(theta) + y * np.sin(theta))
                                                 / 600.0 + phase))
    return np.stack(fields).astype(np.float32)


def evaluation_sets(seed: int = 31337, count: int = 16, side: int = 48, spacing_m: float = 50.0,
                    diffusivity: float = 0.05, critical_slope: float = 0.6) -> dict:
    """In-range surfaces, held-out relief and spectra, rotated surfaces and ridges."""
    rng = np.random.default_rng(seed)
    draw = lambda **kw: _batch(rng, count, side, spacing_m, diffusivity, critical_slope, **kw)[0]  # noqa: E731
    sets = {"in-range": draw(), "relief-x1.6": draw(amplitude_m=48.0),
            "relief-x0.4": draw(amplitude_m=12.0), "spectrum-2.0": draw(exponent=2.0)}
    sets["rot90"] = np.ascontiguousarray(np.rot90(sets["in-range"], 1, axes=(1, 2)))
    sets["ridges"] = corrugations(side, spacing_m)
    return sets


def _rollout_errors(arm, model, start, reference, torch, device, spacing_m, diffusivity,
                    material=None, teacher=None) -> dict:
    """RMSE against the reference at each horizon, plus dissipation and substep counts."""
    surface = torch.from_numpy(start).to(device=device, dtype=torch.float64)
    mat = None if material is None else torch.from_numpy(material).to(device=device, dtype=torch.float64)
    rows, energy, substeps, done = {}, [], 0, 0
    centred = surface - surface.mean(dim=(1, 2), keepdim=True)
    energy.append(float((centred ** 2).sum()))
    with torch.no_grad():
        for interval in range(1, max(HORIZONS) + 1):
            if arm not in ("linear", "teacher"):
                bound = stable_dt(arm, model, surface, spacing_m, torch, diffusivity, mat)
                pieces = int(np.ceil(INTERVAL_YEARS / bound - 1e-12))
                if not np.isfinite(bound) or pieces > MAX_SUBSTEPS:
                    break
                substeps += pieces
            surface = advance(arm, model, surface, INTERVAL_YEARS, spacing_m, torch, diffusivity,
                              mat, teacher=teacher)
            if not bool(torch.isfinite(surface).all()):
                break
            centred = surface - surface.mean(dim=(1, 2), keepdim=True)
            energy.append(float((centred ** 2).sum()))
            done = interval
            if interval in HORIZONS:
                diff = surface.cpu().numpy() - reference[interval]
                rows[interval] = float(np.sqrt(np.mean(diff ** 2)))
    increases = np.diff(np.asarray(energy))
    return {"rmseM": {str(h): rows.get(h) for h in HORIZONS},
            "intervalsCompleted": done, "substeps": substeps,
            "maxEnergyIncreaseRelative": float(max(increases.max(), 0.0) / max(energy[0], 1e-30))
            if increases.size else None}


def reference_rollouts(fields, spacing_m, diffusivity, critical_slope, torch, material_d=None,
                       safety: float = 0.25) -> dict:
    """The teacher at a quarter of its explicit bound: the truth every arm is scored against."""
    surface = torch.from_numpy(np.asarray(fields, dtype=np.float64))
    teacher = (diffusivity if material_d is None else material_d, critical_slope)
    out = {}
    for interval in range(1, max(HORIZONS) + 1):
        surface = advance("teacher", None, surface, INTERVAL_YEARS, spacing_m, torch, diffusivity,
                          safety=safety, teacher=teacher)
        if interval in HORIZONS:
            out[interval] = surface.numpy().copy()
    return out


def structural_checks(arm, model, torch, side: int = 48, spacing_m: float = 50.0,
                      diffusivity: float = 0.05, critical_slope: float = 0.6, seed: int = 31337,
                      material=None) -> dict:
    """Flat surfaces, offsets, conservation, symmetry and the explicit bound, in float64 on CPU."""
    m64 = copy.deepcopy(model).double().cpu().eval()
    rng = np.random.default_rng(seed)
    fields = torch.from_numpy(_batch(rng, 8, side, spacing_m, diffusivity, critical_slope)[0]
                              .astype(np.float64))
    mat = None if material is None else torch.zeros_like(fields)

    def f(h):
        with torch.no_grad():
            return apply_closure(arm, m64, h, spacing_m, torch, mat)

    p = f(fields)
    scale = float(p.abs().mean())
    flat = {}
    for c in (0.0, 100.0):
        level = torch.full((1, side, side), c, dtype=torch.float64)
        with torch.no_grad():
            tendency = apply_closure(arm, m64, level, spacing_m, torch,
                                     None if mat is None else mat[:1])
        surface = level.clone()
        with torch.no_grad():
            for _ in range(max(HORIZONS)):
                surface = advance(arm, m64, surface, INTERVAL_YEARS, spacing_m, torch, diffusivity,
                                  None if mat is None else mat[:1])
        flat[f"c{int(c)}"] = {"maxAbsTendencyMPerYear": float(tendency.detach().abs().max()),
                              "rollout64MaxDeviationM": float((surface - c).abs().max())}
    rotated = torch.rot90(fields, 1, dims=(1, 2))
    rot_error = float((f(rotated) - torch.rot90(p, 1, dims=(1, 2))).abs().mean()) / scale
    neg_error = float((f(-fields) + p).abs().mean()) / scale
    offset = float((f(fields + 100.0) - p).abs().mean()) / scale
    conservation = float((p.sum(dim=(1, 2)).abs() / p.abs().sum(dim=(1, 2))).max())
    out = {"flat": flat, "offset100Relative": offset, "conservationResidualRelative": conservation,
           "rot90EquivarianceRelative": rot_error, "negationEquivarianceRelative": neg_error}
    out["flatStill"] = all(v["maxAbsTendencyMPerYear"] == 0.0 and v["rollout64MaxDeviationM"] == 0.0
                           for v in flat.values())
    out["offsetInvariant"] = offset < 1e-9
    out["conservative"] = conservation < 1e-12
    if arm.startswith("conductance"):
        with torch.no_grad():
            (_, east_a), (_, south_a) = conductances(m64, fields, spacing_m, torch, mat)
        a_max = float(torch.maximum(east_a.max(), south_a.max()))
        a_min = float(torch.minimum(east_a.min(), south_a.min()))
        out.update({"conductanceRange": [a_min, a_max], "aMax": m64.a_max, "floor": m64.floor,
                    "cflDtYears": spacing_m ** 2 / (4.0 * m64.a_max),
                    "cflNumberAtInterval": INTERVAL_YEARS * 4.0 * m64.a_max / spacing_m ** 2,
                    "bounded": bool(m64.floor <= a_min and a_max <= m64.a_max)})
    return out


def conductance_curve(model, torch, spacing_m: float = 50.0, side: int = 24, material: float | None = None,
                      gradients=(0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.59, 0.7)) -> list:
    """a(|g|) read on tilted planes (interior face), against the teacher's D / (1 - (g / S_c)^2)."""
    m64 = copy.deepcopy(model).double().cpu().eval()
    x = (np.arange(side) - (side - 1) / 2.0) * spacing_m
    rows = []
    for g in gradients:
        plane = torch.from_numpy(np.tile(g * x, (side, 1))[None])
        mat = None if material is None else torch.full_like(plane, material)
        with torch.no_grad():
            (_, east_a), _ = conductances(m64, plane, spacing_m, torch, mat)
        rows.append({"gradient": g, "conductance": float(east_a[0, side // 2, side // 2 - 1])})
    return rows


def closure_study(torch, seeds=(1729, 2, 3, 4, 5), steps: int = 1500, side: int = 48,
                  spacing_m: float = 50.0, diffusivity: float = 0.05, critical_slope: float = 0.6,
                  device: str = "cuda", material: bool = True) -> dict:
    """Every arm and baseline, five seeds, scored on rollouts at matched physical times.

    Selection rule, fixed before the runs: among the learned candidates that
    pass the structural gates (flat surface exactly still, offset invariant,
    conservative, bounded where claimed), the arm with the lowest median over
    seeds of the mean in-range rollout RMSE at 8, 32 and 64 intervals of 200
    years. Single-step error is reported but does not select.
    """
    started = time.perf_counter()
    sets = evaluation_sets(side=side, spacing_m=spacing_m, diffusivity=diffusivity,
                           critical_slope=critical_slope)
    references = {name: reference_rollouts(fields, spacing_m, diffusivity, critical_slope, torch)
                  for name, fields in sets.items()}
    baselines = {}
    for name in ("linear", "teacher"):
        baselines[name] = {key: _rollout_errors(name, None, fields, references[key], torch, "cpu", spacing_m,
                                                diffusivity, teacher=(diffusivity, critical_slope))
                           for key, fields in sets.items()}
    runs = []
    for seed in seeds:
        for arm in STUDY_ARMS:
            fitted = train_closure(arm, torch, steps=steps, side=side, spacing_m=spacing_m,
                                   diffusivity=diffusivity, critical_slope=critical_slope,
                                   device=device, seed=seed, a_max=A_MAX)
            model = fitted["model"].eval()
            m64 = copy.deepcopy(model).double()
            single = evaluate_single(arm, m64, torch, device, side, spacing_m, diffusivity, critical_slope)
            rollouts = {key: _rollout_errors(arm, m64, fields, references[key], torch, device, spacing_m,
                                             diffusivity) for key, fields in sets.items()}
            row = {"arm": arm, "seed": seed, "seconds": fitted["seconds"], "parameters": fitted["parameters"],
                   "finalLoss": fitted["history"][-1]["loss"], "singleStep": single, "rollouts": rollouts,
                   "structure": structural_checks(arm, model, torch, side, spacing_m, diffusivity,
                                                  critical_slope)}
            if arm.startswith("conductance"):
                row["curve"] = conductance_curve(model, torch, spacing_m)
            runs.append(row)
            del fitted, model, m64
    summary = summarise(runs, baselines)
    selection = select(summary)
    result = {"schema": SCHEMA.replace("closure-v1", "closure-study-v1"), "runs": runs,
              "baselines": baselines, "summary": summary, "selection": selection,
              "teacherAnisotropy": teacher_anisotropy(spacing_m, diffusivity, critical_slope, side),
              "settings": {"seeds": list(seeds), "steps": steps, "side": side, "spacingM": spacing_m,
                           "diffusivity": diffusivity, "criticalSlope": critical_slope, "aMax": A_MAX,
                           "intervalYears": INTERVAL_YEARS, "horizons": list(HORIZONS),
                           "referenceSafety": 0.25, "evaluationSets": {k: int(v.shape[0]) for k, v in sets.items()}},
              "seconds": time.perf_counter() - started}
    if material:
        result["material"] = material_study(torch, seeds, steps, side, spacing_m, diffusivity,
                                            critical_slope, device)
    result["seconds"] = time.perf_counter() - started
    return result


def evaluate_single(arm, model, torch, device, side, spacing_m, diffusivity, critical_slope,
                    seed: int = 31337, cases: int = 24) -> dict:
    """One-step tendency error on unseen in-range surfaces (reported, not selecting)."""
    rng = np.random.default_rng(seed)
    fields, targets = _batch(rng, cases, side, spacing_m, diffusivity, critical_slope)
    with torch.no_grad():
        predicted = apply_closure(arm, model, torch.from_numpy(fields.astype(np.float64)).to(device),
                                  spacing_m, torch).cpu().numpy()
    linear = np.stack([flux_divergence(f.astype(np.float64), spacing_m, diffusivity) for f in fields])
    return {"maeVsTeacher": float(np.abs(predicted - targets).mean()),
            "relativeError": float(np.abs(predicted - targets).mean() / np.abs(targets).mean()),
            "linearDiffusionMae": float(np.abs(linear - targets).mean())}


def _mean_rmse(rollout: dict):
    values = [v for v in rollout["rmseM"].values()]
    return None if any(v is None for v in values) else float(np.mean(values))


def summarise(runs: list, baselines: dict) -> dict:
    """Median, min and max over seeds of every rollout score, per arm and evaluation set."""
    out = {"baselines": {name: {key: {"rmseM": r["rmseM"], "meanRmseM": _mean_rmse(r)}
                                for key, r in sets.items()} for name, sets in baselines.items()}}
    for arm in sorted({r["arm"] for r in runs}):
        picked = [r for r in runs if r["arm"] == arm]
        entry = {"seeds": len(picked)}
        for key in picked[0]["rollouts"]:
            scores = [_mean_rmse(r["rollouts"][key]) for r in picked]
            finite = [v for v in scores if v is not None]
            per_horizon = {}
            for h in HORIZONS:
                values = [r["rollouts"][key]["rmseM"][str(h)] for r in picked]
                kept = [v for v in values if v is not None]
                per_horizon[str(h)] = ({"median": float(np.median(kept)), "min": float(min(kept)),
                                        "max": float(max(kept))} if kept else None)
            entry[key] = {"meanRmseM": ({"median": float(np.median(finite)), "min": float(min(finite)),
                                         "max": float(max(finite))} if finite else None),
                          "failedRollouts": len(scores) - len(finite), "byHorizon": per_horizon,
                          "maxEnergyIncreaseRelative": max((r["rollouts"][key]["maxEnergyIncreaseRelative"] or 0.0)
                                                           for r in picked)}
        entry["singleStepMae"] = float(np.median([r["singleStep"]["maeVsTeacher"] for r in picked]))
        entry["structuralPass"] = all(r["structure"]["flatStill"] and r["structure"]["offsetInvariant"]
                                      and r["structure"]["conservative"]
                                      and r["structure"].get("bounded", True) for r in picked)
        entry["structure"] = {k: max(float(r["structure"][k]) for r in picked)
                              for k in ("offset100Relative", "conservationResidualRelative",
                                        "rot90EquivarianceRelative", "negationEquivarianceRelative")}
        out[arm] = entry
    return out


def select(summary: dict) -> dict:
    """Apply the declared rule; report the ranking and the arms the gates removed."""
    ranking, removed = [], []
    for arm in CANDIDATES:
        entry = summary.get(arm)
        if entry is None:
            continue
        score = entry["in-range"]["meanRmseM"]
        if not entry["structuralPass"]:
            removed.append({"arm": arm, "reason": "fails a structural gate"})
            continue
        if score is None or entry["in-range"]["failedRollouts"]:
            removed.append({"arm": arm, "reason": "a rollout did not complete"})
            continue
        ranking.append({"arm": arm, "medianMeanRmseM": score["median"], "worstSeed": score["max"],
                        "singleStepMae": entry["singleStepMae"]})
    ranking.sort(key=lambda r: r["medianMeanRmseM"])
    return {"rule": "lowest median over seeds of the mean in-range rollout RMSE at 8, 32 and 64 "
                    "intervals of 200 years, among candidates passing flat, offset, conservation "
                    "and bound gates", "ranking": ranking, "removed": removed,
            "accepted": ranking[0]["arm"] if ranking else None}


def material_study(torch, seeds, steps, side, spacing_m, diffusivity, critical_slope, device,
                   ratio: float = MATERIAL_RATIO, count: int = 16) -> dict:
    """Piecewise lithology with a known diffusivity ratio: can the structured arms recover it?

    Class 0 has D, class 1 has ratio * D, with a random half-plane boundary.
    The conductance arm's floor is the smaller class diffusivity, assumed
    known as a lower bound; the learned ratio is read from a(|g|) on uniform
    planes of each class and compared with the truth (the teacher's ratio is
    exactly `ratio` at every slope).
    """
    rng = np.random.default_rng(4242)
    fields, _, classes = _batch(rng, count, side, spacing_m, diffusivity, critical_slope, ratio=ratio)
    d_cells = diffusivity * np.where(classes > 0, ratio, 1.0)
    references = reference_rollouts(fields, spacing_m, diffusivity, critical_slope, torch, material_d=d_cells)
    baselines = {
        "teacher": _rollout_errors("teacher", None, fields, references, torch, "cpu", spacing_m, diffusivity,
                                   teacher=(d_cells, critical_slope)),
        "linear-uniform": _rollout_errors("linear", None, fields, references, torch, "cpu", spacing_m,
                                          diffusivity)}
    runs = []
    for seed in seeds:
        for arm in ("facek", "conductance"):
            fitted = train_closure(arm, torch, steps=steps, side=side, spacing_m=spacing_m,
                                   diffusivity=diffusivity, critical_slope=critical_slope, device=device,
                                   seed=seed, ratio=ratio, a_max=A_MAX, floor=ratio * diffusivity)
            model = fitted["model"].eval()
            m64 = copy.deepcopy(model).double()
            rollout = _rollout_errors(arm, m64, fields, references, torch, device, spacing_m, diffusivity,
                                      material=classes.astype(np.float64))
            row = {"arm": arm, "seed": seed, "rollout": rollout, "finalLoss": fitted["history"][-1]["loss"]}
            if arm == "conductance":
                zero = conductance_curve(model, torch, spacing_m, material=0.0, gradients=(0.02, 0.1, 0.3))
                one = conductance_curve(model, torch, spacing_m, material=1.0, gradients=(0.02, 0.1, 0.3))
                row["learnedRatio"] = [{"gradient": a["gradient"],
                                        "ratio": b["conductance"] / max(a["conductance"], 1e-30)}
                                       for a, b in zip(zero, one)]
            else:
                ratios = []
                for g in (0.02, 0.1, 0.3):
                    plane = np.tile(g * (np.arange(side) - (side - 1) / 2.0) * spacing_m, (side, 1))[None]
                    h = torch.from_numpy(plane)
                    with torch.no_grad():
                        k0 = _cell_diffusivity(m64.cpu(), h, spacing_m, torch, torch.zeros_like(h))
                        k1 = _cell_diffusivity(m64.cpu(), h, spacing_m, torch, torch.ones_like(h))
                    centre = (0, side // 2, side // 2)
                    ratios.append({"gradient": g, "ratio": float(k1[centre] / k0[centre])})
                row["learnedRatio"] = ratios
            runs.append(row)
            del fitted, model, m64
    summary = {}
    for arm in ("facek", "conductance"):
        picked = [r for r in runs if r["arm"] == arm]
        scores = [_mean_rmse(r["rollout"]) for r in picked]
        finite = [v for v in scores if v is not None]
        gentle = [r["learnedRatio"][0]["ratio"] for r in picked]
        summary[arm] = {"meanRmseM": ({"median": float(np.median(finite)), "min": float(min(finite)),
                                       "max": float(max(finite))} if finite else None),
                        "failedRollouts": len(scores) - len(finite),
                        "learnedRatioAtGradient0.02": {"median": float(np.median(gentle)),
                                                       "min": float(min(gentle)), "max": float(max(gentle))},
                        "trueRatio": ratio}
    return {"ratio": ratio, "classes": "random half-plane per surface", "surfaces": count,
            "baselines": {k: {"rmseM": v["rmseM"], "meanRmseM": _mean_rmse(v)} for k, v in baselines.items()},
            "runs": runs, "summary": summary}


def teacher_conductance(height, spacing_m: float, d_cells, critical_slope: float, torch):
    """The teacher's own face conductance D_face / (1 - r^2), east and south, as torch tensors."""
    def faces(h, d):
        g = (h[:, :, 1:] - h[:, :, :-1]) / spacing_m
        r = (g.abs() / critical_slope).clamp(0.0, 0.99)
        face_d = 2.0 * d[:, :, 1:] * d[:, :, :-1] / (d[:, :, 1:] + d[:, :, :-1])
        return face_d / (1.0 - r * r)
    east = faces(height, d_cells)
    south = faces(height.transpose(1, 2), d_cells.transpose(1, 2)).transpose(1, 2)
    return east, south


def material_face_loss_ablation(torch, seeds=(1729, 2, 3, 4, 5), steps: int = 1500, side: int = 48,
                                spacing_m: float = 50.0, diffusivity: float = 0.05, critical_slope: float = 0.6,
                                device: str = "cuda", ratio: float = MATERIAL_RATIO, count: int = 16) -> dict:
    """Does the material-aware conductance recover the diffusivity ratio if trained on conductances?

    In `material_study` the arm is trained, like every arm, on cell tendencies, and the L1 loss
    is dominated by steep faces; gentle-slope conductances are left at the floor. This ablation
    trains the same arm on the teacher's face conductances directly (mean |log a - log a_true|),
    which is oracle supervision that a closure fitted to observed surfaces would not have. It
    separates "the architecture cannot represent the material law" from "the tendency loss does
    not constrain it".
    """
    rng_eval = np.random.default_rng(4242)
    fields, _, classes = _batch(rng_eval, count, side, spacing_m, diffusivity, critical_slope, ratio=ratio)
    d_cells = diffusivity * np.where(classes > 0, ratio, 1.0)
    references = reference_rollouts(fields, spacing_m, diffusivity, critical_slope, torch, material_d=d_cells)
    runs = []
    for seed in seeds:
        torch.manual_seed(seed)
        model = make_arm("conductance", torch, diffusivity, A_MAX, material=True,
                         floor=ratio * diffusivity).to(device).train()
        optimiser = torch.optim.Adam(model.parameters(), lr=3e-4)
        rng = np.random.default_rng(seed)
        started = time.perf_counter()
        for _ in range(steps):
            h, _, m = _batch(rng, 8, side, spacing_m, diffusivity, critical_slope, ratio=ratio)
            h = torch.from_numpy(h).to(device)
            m = torch.from_numpy(m).to(device)
            d = diffusivity * torch.where(m > 0, ratio, 1.0)
            (_, east_a), (_, south_a) = conductances(model, h, spacing_m, torch, m)
            true_east, true_south = teacher_conductance(h, spacing_m, d, critical_slope, torch)
            loss = 0.5 * ((east_a.log() - true_east.log()).abs().mean() + (south_a.log() - true_south.log()).abs().mean())
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
        model.eval()
        m64 = copy.deepcopy(model).double()
        rollout = _rollout_errors("conductance", m64, fields, references, torch, device, spacing_m, diffusivity,
                                  material=classes.astype(np.float64))
        zero = conductance_curve(model, torch, spacing_m, material=0.0, gradients=(0.02, 0.1, 0.3))
        one = conductance_curve(model, torch, spacing_m, material=1.0, gradients=(0.02, 0.1, 0.3))
        runs.append({"seed": seed, "seconds": time.perf_counter() - started, "finalLoss": float(loss.detach()),
                     "rollout": rollout,
                     "learnedRatio": [{"gradient": a["gradient"], "ratio": b["conductance"] / a["conductance"],
                                       "class0": a["conductance"], "class1": b["conductance"]}
                                      for a, b in zip(zero, one)]})
    scores = [_mean_rmse(r["rollout"]) for r in runs]
    finite = [v for v in scores if v is not None]
    gentle = [r["learnedRatio"][0]["ratio"] for r in runs]
    return {"ratio": ratio, "loss": "mean |log a - log a_true| over faces (oracle face conductances)",
            "runs": runs,
            "summary": {"meanRmseM": {"median": float(np.median(finite)), "min": float(min(finite)),
                                      "max": float(max(finite))} if finite else None,
                        "learnedRatioAtGradient0.02": {"median": float(np.median(gentle)), "min": float(min(gentle)),
                                                       "max": float(max(gentle))},
                        "trueRatio": ratio}}


def conductance_export(model, torch, validated: dict | None = None, training: dict | None = None) -> tuple[dict, bytes]:
    """The conductance arm in the closure.json structure the exporters read, and its weight blob.

    Returns `(arm_meta, blob)`: `arm_meta` is the entry for `closure.json["arms"]["conductance"]`
    (apply, layers with offsets and activations, floats, bytes, sha256, the floor and a_max), and
    `blob` the little-endian float32 weights to write as `conductance.bin`. The last layer's
    activation is "sigmoid"; the kernel maps it to `floor + (aMax - floor) * sigmoid(x)`.
    """
    import hashlib
    modules = list(model.net)
    convs = [m for m in modules if isinstance(m, torch.nn.Conv2d)]
    if [type(m).__name__ for m in modules] != ["Conv2d", "GELU"] * (len(convs) - 1) + ["Conv2d"]:
        raise ValueError("conductance layers are not conv/GELU pairs ending in a conv")
    if model.material:
        raise ValueError("the material-aware conductance is a Python experiment and is not exported")
    chunks, layers, offset = [], [], 0
    for index, conv in enumerate(convs):
        if conv.kernel_size != (3, 3) or conv.padding_mode != "replicate" or conv.bias is None:
            raise ValueError(f"unexpected conductance layer {index}: {conv}")
        weight = conv.weight.detach().cpu().numpy().astype("<f4").ravel()
        bias = conv.bias.detach().cpu().numpy().astype("<f4").ravel()
        layers.append({"in": conv.in_channels, "out": conv.out_channels,
                       "weightOffset": offset, "biasOffset": offset + weight.size,
                       "activation": "sigmoid" if index == len(convs) - 1 else "gelu"})
        chunks += [weight, bias]
        offset += weight.size + bias.size
    blob = np.concatenate(chunks).astype("<f4").tobytes()
    meta = {"apply": "conductance", "conservative": True,
            "floor": float(model.floor), "aMax": float(model.a_max),
            "inputs": ["face gradient magnitude |h[c+1] - h[c]| / spacingM"],
            "inputGrid": "east faces, side rows by side-1 columns; south faces run the same network "
                         "on the transposed field",
            "tendency": "a = floor + (aMax - floor) * sigmoid(output) per face; G = a (h[c+1] - h[c]) / "
                        "spacingM; the west (north) cell gains G / spacingM and the east (south) cell "
                        "loses it",
            "stability": "dt <= spacingM^2 / (4 aMax) for every state",
            "file": "conductance.bin", "floats": offset, "bytes": len(blob),
            "sha256": hashlib.sha256(blob).hexdigest(), "layers": layers}
    if validated is not None:
        meta["validated"] = validated
    if training is not None:
        meta["training"] = training
    return meta, blob
