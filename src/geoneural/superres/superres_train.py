"""Train the super-resolution arms and score them against the classical bar.

Rules this driver enforces, each of which keeps the result from looking better
than it is:

* Only the coarse grid enters. In the training region the 1 m reference is the
  supervision label (the residual the arm is fitted to); it is never an input,
  never a normaliser, and in a held-out region it is consulted only as truth.
* Every arm predicts a residual over bicubic, so the reported number is the
  classical bar plus whatever the network added. An arm that learns nothing
  reproduces the bar rather than something worse, so "no effect" is
  distinguishable from a broken run.
* Normalisation is local and computed once on the whole coarse field
  (`coarse_inputs`): a Gaussian-weighted relief and gradient scale around each
  node, from coarse heights alone. A training patch, an evaluation tile and a
  drainage window are crops of the same arrays, so the network sees the same
  numbers for the same node whatever window it is run on, and the residual is
  returned in metres by multiplying with the same scale. A per-patch mean and
  standard deviation would make the input depend on the window size, which is
  what the earlier 17-node training patches and 205-node drainage windows did.
* Held-out geography is a whole region, not a crop. Training on essen-ruhr and
  testing on rothaar-sauerland is the only arrangement here that says anything
  about terrain the network has not seen.
* Patches are drawn from disjoint coarse tiles so a training patch and an
  evaluation patch never share a fine sample.
"""
from __future__ import annotations

import time

import numpy as np

from geoneural.superres import superres

from geoneural.superres import superres_models

SCHEMA = "geoneural-superres-neural-v2"

#: Gaussian width (coarse cells) of the local normaliser, its floor in metres, and the halo that
#: makes a tiled reconstruction equal to a whole-window one (encoder receptive field plus margin).
SCALE_SIGMA = 2.0
SCALE_FLOOR_M = 0.05
INPUT_CHANNELS = 4
HALO = 12


def _bicubic_base(coarse: np.ndarray, factor: int) -> np.ndarray:
    """The classical bar every arm starts from. float32: a 10241^2 float64 base
    is 839 MB and the residuals it feeds are metres, not microns."""
    return np.ascontiguousarray(superres.classical(coarse, factor, "bicubic"),
                                dtype=np.float32)


def coarse_inputs(coarse: np.ndarray):
    """Whole-field network inputs and the local height scale, from the coarse grid alone.

    Channels: relief about a Gaussian local mean, the two central-difference
    gradients (metres per coarse cell), each divided by the local scale, and
    log of the scale. The scale is the Gaussian-weighted RMS gradient plus a
    floor, so flat ground is not divided by zero. Everything is computed on the
    whole field with mirrored edges, then cropped, so the inputs of a node do
    not depend on which window it is read through.
    """
    from scipy import ndimage
    c = np.asarray(coarse, dtype=np.float64)
    local = ndimage.gaussian_filter(c, SCALE_SIGMA, mode="mirror", truncate=4.0)
    gy, gx = np.gradient(c)
    scale = np.sqrt(ndimage.gaussian_filter(gx * gx + gy * gy, SCALE_SIGMA, mode="mirror",
                                            truncate=4.0)) + SCALE_FLOOR_M
    inputs = np.stack([(c - local) / scale, gx / scale, gy / scale, np.log(scale)])
    return inputs.astype(np.float32), scale


def fine_scale(scale_window: np.ndarray, factor: int) -> np.ndarray:
    """The coarse scale on the fine lattice. Bilinear, so a crop of it equals it computed on the crop."""
    return superres.upsample_bilinear(scale_window, factor)


class PatchSampler:
    """Coarse tiles split by tile index into training and evaluation sets.

    Tiles step by `patch - 1`, so neighbouring tiles share one edge row or column of coarse
    nodes, and a training and an evaluation tile can share the fine samples along that line.
    """

    def __init__(self, coarse_side: int, patch: int, factor: int, seed: int,
                 holdout_fraction: float = 0.2):
        self.patch = int(patch)
        self.factor = int(factor)
        step = self.patch - 1
        starts = list(range(0, coarse_side - self.patch, step))
        if not starts:
            raise ValueError(
                f"a {coarse_side} coarse grid cannot hold a {patch} patch")
        tiles = [(r, c) for r in starts for c in starts]
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(tiles))
        cut = max(1, int(len(tiles) * holdout_fraction))
        self.evaluation = [tiles[i] for i in order[:cut]]
        self.training = [tiles[i] for i in order[cut:]]
        if not self.training:
            raise ValueError("no training tiles left after the holdout split")

    def record(self) -> dict:
        return {"patch": self.patch, "trainingTiles": len(self.training),
                "evaluationTiles": len(self.evaluation),
                "note": "neighbouring tiles share one edge row or column of coarse "
                        "nodes; interiors are disjoint"}


def _patch(features, coarse, fine, base, row, column, patch, factor):
    """One patch: whole-field input crop, residual target in scale units, and the raw pieces.

    `features` is `coarse_inputs(coarse)`. The target is (fine - bicubic) divided
    by the fine-lattice scale; the scale comes from the coarse grid, so the label
    is rescaled but nothing about it reaches the input.
    """
    inputs, scale = features
    c = coarse[row:row + patch, column:column + patch]
    span = (patch - 1) * factor + 1
    fr, fc = row * factor, column * factor
    f = np.asarray(fine[fr:fr + span, fc:fc + span], dtype=np.float64)
    b = np.asarray(base[fr:fr + span, fc:fc + span], dtype=np.float64)
    s = fine_scale(scale[row:row + patch, column:column + patch], factor)
    return (np.ascontiguousarray(inputs[:, row:row + patch, column:column + patch]),
            ((f - b) / s).astype(np.float32),
            np.ascontiguousarray(c, dtype=np.float64), b, s)


def train_arm(config: dict, coarse: np.ndarray, fine: np.ndarray, torch,
              steps: int = 3000, patch: int = 17, factor: int = 10,
              queries: int = 4096, batch: int = 4, lr: float = 3e-4,
              seed: int = 1729, device: str = "cuda", holdout_fraction: float = 0.2):
    """Fit one arm on one region's coarse grid. Returns (model, history, sampler)."""
    torch.manual_seed(seed)
    config = dict(config, channels=INPUT_CHANNELS)
    model = superres_models.make_model(config, torch).to(device).train()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    base = _bicubic_base(coarse, factor)
    features = coarse_inputs(coarse)
    sampler = PatchSampler(coarse.shape[0], patch, factor, seed, holdout_fraction)
    rng = np.random.default_rng(seed ^ 0x5EED)
    span = (patch - 1) * factor + 1
    arm = config["arm"]
    history = []
    start = time.perf_counter()
    for step in range(steps):
        picks = [sampler.training[i] for i in
                 rng.integers(0, len(sampler.training), batch)]
        parts = [_patch(features, coarse, fine, base, r, c, patch, factor) for r, c in picks]
        coarse_in = torch.from_numpy(np.stack([p[0] for p in parts])).to(device)
        target = torch.from_numpy(np.stack([p[1] for p in parts])).to(device)
        if arm == "edsr":
            predicted = model(coarse_in).squeeze(1)
            loss = torch.nn.functional.l1_loss(predicted, target)
        else:
            # Continuous queries: a fine node at (y, x) sits in coarse cell
            # (y // factor, x // factor) at offset (y % factor) / factor - 0.5.
            ys = torch.from_numpy(rng.integers(0, span, (batch, queries))).to(device)
            xs = torch.from_numpy(rng.integers(0, span, (batch, queries))).to(device)
            cells = torch.clamp(ys // factor, max=patch - 1) * patch + \
                torch.clamp(xs // factor, max=patch - 1)
            offsets = torch.stack((
                (ys % factor).float() / factor - 0.5,
                (xs % factor).float() / factor - 0.5), dim=-1)
            predicted = model(coarse_in, cells, offsets)
            truth = target.flatten(1).gather(1, ys * span + xs)
            loss = torch.nn.functional.l1_loss(predicted, truth)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            history.append({"step": step, "loss": float(loss.detach())})
    seconds = time.perf_counter() - start
    return model, {"history": history, "seconds": seconds,
                   "sampler": sampler.record(), "config": config}, sampler


def _accumulate():
    return {"absolute": 0.0, "squares": 0.0, "max": 0.0, "pool": []}


def _add(store, error):
    store["absolute"] += float(error.sum())
    store["squares"] += float((error ** 2).sum())
    store["max"] = max(store["max"], float(error.max()))
    store["pool"].append(error.ravel()[::97].copy())


def _finish(store, samples):
    pool = np.concatenate(store["pool"]) if store["pool"] else np.zeros(1)
    return {"maeM": store["absolute"] / samples,
            "rmseM": (store["squares"] / samples) ** 0.5,
            "maxM": store["max"], "p99M": float(np.percentile(pool, 99)),
            "samples": samples}


def evaluate_region(model, config, coarse, fine, torch, factor: int = 10,
                    patch: int = 17, device: str = "cuda",
                    tiles=None, trim: int | None = None, tolerance: float = 1e-4):
    """Reconstruction on held-out tiles, against 1 m truth.

    Four numbers on exactly the same ground, because comparing a network against
    the weakest classical method is the easiest way to overstate a
    super-resolution result:

    * `bicubic`: the free baseline the arms predict a residual over.
    * `bicubicBackProjected`: the same, with the operator constraint enforced
      to `tolerance`. This is the real classical bar.
    * `neural`: the arm's own output.
    * `neuralBackProjected`: the arm's output with the same constraint enforced.

    The network is run on each tile plus a halo of `HALO` coarse cells cut from
    the whole-field inputs, so a tile scores exactly what a whole-region
    reconstruction would put there.

    `trim` matters. Back-projection upsamples its residual, and on a 17-cell
    patch the outermost ring of that upsample is unconstrained on one side. On
    rothaar-sauerland a back-projected patch scored 0.2855 m over its outer ten
    rows against 0.0519 m inside. A margin is trimmed from every method equally;
    without it the comparison measures the patch grid rather than the methods.
    """
    trim = 2 * factor if trim is None else int(trim)
    model.eval()
    config = dict(config, channels=INPUT_CHANNELS)
    base = _bicubic_base(coarse, factor)
    features = coarse_inputs(coarse)
    stores = {name: _accumulate() for name in
              ("bicubic", "bicubicBackProjected", "neural", "neuralBackProjected")}
    consistency = {"neural": 0.0, "neuralBackProjected": 0.0}
    projection = {"toleranceM": tolerance, "achievedMaxM": 0.0, "iterations": 0, "converged": True}
    samples = 0
    inner = slice(trim, -trim) if trim > 0 else slice(None)
    for row, column in tiles:
        _, residual_scaled, raw_coarse, raw_base, scale = _patch(
            features, coarse, fine, base, row, column, patch, factor)
        truth = raw_base + residual_scaled * scale
        estimate = raw_base + reconstruct_core(model, config, features, row, column, patch,
                                               torch, factor, device)
        classical_bp, report_b = superres.back_project(raw_base, raw_coarse, factor, tolerance)
        neural_bp, report_n = superres.back_project(estimate, raw_coarse, factor, tolerance)
        for report in (report_b, report_n):
            projection["achievedMaxM"] = max(projection["achievedMaxM"], report["achievedMaxM"])
            projection["iterations"] = max(projection["iterations"], report["iterations"])
            projection["converged"] &= report["converged"]
        core = truth[inner, inner]
        _add(stores["bicubic"], np.abs(raw_base[inner, inner] - core))
        _add(stores["bicubicBackProjected"], np.abs(classical_bp[inner, inner] - core))
        _add(stores["neural"], np.abs(estimate[inner, inner] - core))
        _add(stores["neuralBackProjected"], np.abs(neural_bp[inner, inner] - core))
        consistency["neural"] = max(
            consistency["neural"],
            superres.operator_consistency(estimate, raw_coarse, factor)["maxM"])
        consistency["neuralBackProjected"] = max(
            consistency["neuralBackProjected"],
            superres.operator_consistency(neural_bp, raw_coarse, factor)["maxM"])
        samples += core.size
    out = {name: _finish(store, samples) for name, store in stores.items()}
    out["operatorConsistencyMaxM"] = consistency
    out["backProjection"] = projection
    out["tiles"] = len(tiles)
    out["trimFineSamples"] = trim
    out["note"] = ("All four figures are the same tiles, the same truth and the same "
                   "operator. The comparison that decides anything is "
                   "neuralBackProjected against bicubicBackProjected: back-projection "
                   "is free for both sides and enforcing it on only one would favour "
                   "that side.")
    return out


def operator_identity(factor: int = 10) -> str:
    """Fingerprint of the weights and code of the declared operator; see `superres.operator_fingerprint`."""
    return superres.operator_fingerprint(factor)


def _load(region: str, root: str = "fine"):
    """Coarse grid and 1 m reference for one region, memory-mapped."""
    import pathlib as _p
    folder = _p.Path(root) / region
    coarse = np.load(folder / "coarse_from_operator.npy")
    fine = np.load(folder / "reference_1m.npy", mmap_mode="r")
    expected = (coarse.shape[0] - 1) * 10 + 1
    if fine.shape[0] != expected:
        raise ValueError(
            f"{region}: {fine.shape[0]} fine nodes against {expected} implied by "
            f"a {coarse.shape[0]} coarse grid; the two are not the same lattice")
    return np.ascontiguousarray(coarse, dtype=np.float64), fine


def campaign(arms, train_region: str, test_regions, torch, steps: int = 3000,
             patch: int = 17, factor: int = 10, queries: int = 4096,
             batch: int = 8, lr: float = 3e-4, seed: int = 1729,
             device: str = "cuda", width: int = 64, blocks: int = 4,
             hidden: int = 256, depth: int = 4,
             evaluation_tiles: int = 512, root: str = "fine",
             drainage_window: int = 205, drainage_origin: int = 100,
             stream_cells: int = 50_000, tolerance: float = 1e-4) -> dict:
    """Fit each arm on one region, then read it on held-out ground and held-out geography.

    The within-region figure and the cross-region figure answer different
    questions and the difference between them is the result. A super-resolution
    model that only works where it was fitted has learned this region's terrain,
    not what terrain looks like.
    """
    coarse, fine = _load(train_region, root)
    fine = np.ascontiguousarray(fine, dtype=np.float32)
    rows = []
    for arm in arms:
        config = {"arm": arm, "width": width, "blocks": blocks,
                  "hidden": hidden, "depth": depth, "factor": factor}
        model, history, sampler = train_arm(
            config, coarse, fine, torch, steps=steps, patch=patch, factor=factor,
            queries=queries, batch=batch, lr=lr, seed=seed, device=device)
        within = evaluate_region(model, config, coarse, fine, torch, factor=factor,
                                 patch=patch, device=device,
                                 tiles=sampler.evaluation[:evaluation_tiles], tolerance=tolerance)
        row = {"arm": arm, "config": history["config"],
               "deployedBytes": superres_models.deployed_bytes(model, torch),
               "trainedOn": train_region, "training": history,
               "withinRegion": within, "heldOutGeography": {},
               "drainage": {}}
        if drainage_window:
            row["drainage"][train_region] = drainage_check(
                model, config, coarse, fine, torch, drainage_origin, drainage_origin,
                window=drainage_window, factor=factor, device=device,
                stream_area_m2=float(stream_cells), tolerance=tolerance)
        for region in test_regions:
            other_coarse, other_fine = _load(region, root)
            other_fine = np.ascontiguousarray(other_fine, dtype=np.float32)
            other_sampler = PatchSampler(other_coarse.shape[0], patch, factor, seed)
            every = other_sampler.training + other_sampler.evaluation
            row["heldOutGeography"][region] = evaluate_region(
                model, config, other_coarse, other_fine, torch, factor=factor,
                patch=patch, device=device, tiles=every[:evaluation_tiles], tolerance=tolerance)
            if drainage_window:
                row["drainage"][region] = drainage_check(
                    model, config, other_coarse, other_fine, torch,
                    drainage_origin, drainage_origin, window=drainage_window,
                    factor=factor, device=device, stream_area_m2=float(stream_cells),
                    tolerance=tolerance)
            del other_coarse, other_fine
        rows.append(row)
        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    envelopes = {}
    for region in [train_region] + list(test_regions):
        region_coarse, region_fine = (coarse, fine) if region == train_region \
            else _load(region, root)
        envelopes[region] = metre_envelope(
            region_coarse, region_fine, factor=factor, window=drainage_window,
            row=drainage_origin, column=drainage_origin)
        del region_coarse, region_fine
    return {"schema": SCHEMA, "rows": rows, "trainRegion": train_region,
            "metreEnvelope": envelopes,
            "testRegions": list(test_regions), "factor": factor, "patch": patch,
            "steps": steps, "seed": seed,
            "observationOperator": superres.operator_schema(factor),
            "observationOperatorId": operator_identity(factor),
            "normalisation": {"inputs": "relief about a Gaussian local mean and the two gradients, "
                                        "divided by the local scale, and log(scale)",
                              "scale": "sqrt(Gaussian(|grad coarse|^2)) + floor, metres",
                              "sigmaCoarseCells": SCALE_SIGMA, "floorM": SCALE_FLOOR_M,
                              "haloCoarseCells": HALO},
            "qualification":
                "Only the coarse grid enters the model. In the training region the 1 m "
                "reference is the supervision label; elsewhere it is only the truth scored "
                "against. Every arm predicts a residual over bicubic in units of a local "
                "scale computed once on the whole coarse field, so training patches, "
                "evaluation tiles and drainage windows see the same inputs for the same "
                "nodes. Held-out geography is a whole region from the same provider and "
                "datum, which tests transfer across terrain and not across source."}


def reconstruct_window(model, config, inputs_window, scale_window, torch, factor: int = 10,
                       device: str = "cuda", chunk: int = 1_000_000) -> np.ndarray:
    """Residual over bicubic, in metres, for a window cut from the whole-field inputs.

    `inputs_window` and `scale_window` are crops of `coarse_inputs`, so the
    numbers the network sees for a node are the ones it saw in training,
    whatever the window size. The arms are fully convolutional and the heads
    pointwise, so they accept any rectangle; the query grid is what has to be
    chunked, because a 2041^2 window is 4.2 M queries.
    """
    rows, columns = int(scale_window.shape[0]), int(scale_window.shape[1])
    span_r, span_c = (rows - 1) * factor + 1, (columns - 1) * factor + 1
    coarse_in = torch.from_numpy(np.ascontiguousarray(inputs_window, dtype=np.float32)[None]).to(device)
    model.eval()
    with torch.no_grad():
        if config["arm"] == "edsr":
            out = model(coarse_in).squeeze(0).squeeze(0).cpu().numpy().astype(np.float64)
        else:
            features = model.encoder(coarse_in)
            flat = features.flatten(2).transpose(1, 2)
            out = np.empty(span_r * span_c, dtype=np.float32)
            total = span_r * span_c
            for start in range(0, total, chunk):
                stop = min(start + chunk, total)
                index = np.arange(start, stop)
                ys, xs = index // span_c, index % span_c
                yt = torch.from_numpy(ys).to(device)
                xt = torch.from_numpy(xs).to(device)
                cells = torch.clamp(yt // factor, max=rows - 1) * columns + \
                    torch.clamp(xt // factor, max=columns - 1)
                offsets = torch.stack((
                    (yt % factor).float() / factor - 0.5,
                    (xt % factor).float() / factor - 0.5), dim=-1)[None]
                latent = flat.gather(
                    1, cells[None].unsqueeze(-1).expand(-1, -1, flat.shape[-1]))
                out[start:stop] = model.head(latent, offsets).squeeze(-1).squeeze(0).cpu().numpy()
            out = out.reshape(span_r, span_c).astype(np.float64)
    return out * fine_scale(scale_window, factor)


def reconstruct_core(model, config, features, row, column, patch, torch, factor: int = 10,
                     device: str = "cuda", halo: int = HALO) -> np.ndarray:
    """Residual on the fine nodes of one coarse tile, computed with a halo of whole-field inputs."""
    inputs, scale = features
    r0, r1 = max(row - halo, 0), min(row + patch + halo, scale.shape[0])
    c0, c1 = max(column - halo, 0), min(column + patch + halo, scale.shape[1])
    window = reconstruct_window(model, config, inputs[:, r0:r1, c0:c1], scale[r0:r1, c0:c1],
                                torch, factor, device)
    span = (patch - 1) * factor + 1
    fr, fc = (row - r0) * factor, (column - c0) * factor
    return window[fr:fr + span, fc:fc + span]


def reconstruct_tiled(model, config, features, torch, factor: int = 10, device: str = "cuda",
                      tile: int = 32, halo: int = HALO) -> np.ndarray:
    """Whole-field residual assembled from tiles with halos.

    Equal to `reconstruct_window` on the whole field to float precision when
    the halo covers the receptive field (`HALO` does for the default arms);
    that equality is the tiled-versus-full-domain gate.
    """
    _, scale = features
    rows, columns = scale.shape
    out = np.zeros(((rows - 1) * factor + 1, (columns - 1) * factor + 1))
    for r in range(0, rows - 1, tile):
        for c in range(0, columns - 1, tile):
            pr, pc = min(tile, rows - 1 - r) + 1, min(tile, columns - 1 - c) + 1
            inputs, sc = features
            r0, r1 = max(r - halo, 0), min(r + pr + halo, rows)
            c0, c1 = max(c - halo, 0), min(c + pc + halo, columns)
            window = reconstruct_window(model, config, inputs[:, r0:r1, c0:c1], sc[r0:r1, c0:c1],
                                        torch, factor, device)
            fr, fc = (r - r0) * factor, (c - c0) * factor
            out[r * factor:(r + pr - 1) * factor + 1, c * factor:(c + pc - 1) * factor + 1] = \
                window[fr:fr + (pr - 1) * factor + 1, fc:fc + (pc - 1) * factor + 1]
    return out


DRAINAGE_SCHEMA = "geoneural-superres-drainage-v2"


def drainage_check(model, config, coarse, fine, torch, row: int, column: int,
                   window: int = 205, factor: int = 10, device: str = "cuda",
                   stream_area_m2: float = 50_000.0, trim: int = 20,
                   tolerance: float = 1e-4) -> dict:
    """Route water over the reconstruction and over the truth, and compare.

    The third axis of the bar. A surface that scores well on height and routes
    water wrongly has not reconstructed terrain, and height error cannot see it:
    a ridge displaced by one metre costs almost nothing in MAE and moves a
    catchment boundary.

    Streams start at a physical area (0.05 km^2, 50,000 cells of 1 m, the same
    area 500 cells give at 10 m). `geoneural.metrics.drainage` gives exact and
    1-cell tolerant overlap, receiver and outlet agreement and fill depth and
    volume; its noise floor (the truth against itself plus 1 cm of noise) is
    reported next to them, because exact-cell overlap moves with centimetres.
    The window's inputs are cut from the whole-region inputs, as in training.
    """
    from geoneural.metrics import drainage
    span = (window - 1) * factor + 1
    config = dict(config, channels=INPUT_CHANNELS)
    coarse_window = np.ascontiguousarray(
        coarse[row:row + window, column:column + window], dtype=np.float64)
    truth = np.ascontiguousarray(
        fine[row * factor:row * factor + span,
             column * factor:column * factor + span], dtype=np.float64)
    base = np.ascontiguousarray(superres.classical(coarse_window, factor, "bicubic"),
                                dtype=np.float64)
    features = coarse_inputs(coarse)
    estimate = base + reconstruct_core(model, config, features, row, column, window, torch,
                                       factor, device)
    classical_bp, report_b = superres.back_project(base, coarse_window, factor, tolerance)
    neural_bp, report_n = superres.back_project(estimate, coarse_window, factor, tolerance)
    inner = slice(trim, -trim) if trim > 0 else slice(None)
    reference = truth[inner, inner]
    routed = drainage.route(reference, 1.0)
    out = {"schema": DRAINAGE_SCHEMA, "window": window, "spanFine": span,
           "streamAreaM2": stream_area_m2, "trimFineSamples": trim,
           "origin": {"row": row, "column": column},
           "backProjection": {"bicubic": report_b, "neural": report_n},
           "noiseFloor": drainage.noise_floor(reference, 1.0, area_m2=stream_area_m2)}
    for name, surface in (("bicubic", base), ("bicubicBackProjected", classical_bp),
                          ("neural", estimate), ("neuralBackProjected", neural_bp)):
        out[name] = drainage.stream_metrics(reference, surface[inner, inner], 1.0,
                                            area_m2=stream_area_m2, reference_routed=routed)
        out[name]["maeM"] = float(np.abs(surface[inner, inner] - reference).mean())
    out["note"] = ("Both fields routed identically at 1 m. Exact-cell overlap is read against "
                   "noiseFloor; tolerant F1 and receiver agreement are the discriminating "
                   "quantities.")
    return out


ENVELOPE_SCHEMA = "geoneural-superres-envelope-v1"


def metre_envelope(coarse, fine, factor: int = 10, window: int = 205,
                   row: int = 0, column: int = 0,
                   targets=(0.05, 0.5, 1.0, 2.0, 4.0),
                   coarse_spacings=(1, 2, 5)) -> dict:
    """What the conventional side costs to deliver 1 m over the same window.

    Two families, because a super-resolution claim has to beat both:

    * Store the 1 m grid: `q32-delta-zstd` at a declared max-error target. This
      is expensive, and it is the only option here that carries a bound rather
      than an observed error.
    * Store a coarser grid and interpolate: downsample the 1 m truth to 2 m or
      5 m, encode that, and bicubic back up. This is the real competitor: it is
      what a deployment would do, and the family the network has to beat to
      justify its weights.

    The 10 m grid the network reads is not charged here. The atlas ships it
    regardless, so the network's bytes are marginal, as are the extra bytes of
    any denser grid above what 10 m already costs.
    """
    from geoneural.codecs import codecs
    span = (window - 1) * factor + 1
    truth = np.ascontiguousarray(
        fine[row * factor:row * factor + span,
             column * factor:column * factor + span], dtype=np.float64)
    coarse_window = np.ascontiguousarray(
        coarse[row:row + window, column:column + window], dtype=np.float64)
    codec = codecs.registry()["q32-delta-zstd"]
    rows = []
    for spacing in coarse_spacings:
        grid = np.ascontiguousarray(truth[::spacing, ::spacing], dtype=np.float32)
        for target in targets:
            measured = codecs.measure(codec, grid, target)
            decoded = codec.decode(codec.encode(grid, target)).astype(np.float64)
            if spacing == 1:
                estimate = decoded
            else:
                estimate = superres.classical(decoded, spacing, "bicubic")
                estimate = estimate[:span, :span]
                if estimate.shape != truth.shape:
                    raise ValueError(
                        f"{spacing} m upsample gave {estimate.shape} against "
                        f"{truth.shape}; the two are not the same lattice")
            error = np.abs(estimate - truth)
            rows.append({
                "sourceSpacingM": spacing, "targetMaxErrorM": target,
                "bytes": int(measured["bytes"]),
                "samples": int(grid.size),
                "method": ("stored at 1 m" if spacing == 1
                           else f"stored at {spacing} m, bicubic to 1 m"),
                "boundGuaranteed": spacing == 1,
                "maeM": float(error.mean()), "maxM": float(error.max()),
                "p99M": float(np.percentile(error, 99))})
    return {"schema": ENVELOPE_SCHEMA, "window": window, "spanFine": span,
            "rows": rows,
            "note": "The 10 m grid the network reads is not charged: the atlas ships "
                    "it regardless, so every figure here is marginal cost above it. "
                    "boundGuaranteed marks the rows whose maximum is an enforced "
                    "encoder tolerance rather than an observed worst node. Only the "
                    "1 m rows carry one, because an interpolated surface has no bound."}
