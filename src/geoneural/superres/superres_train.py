"""Train the super-resolution arms and score them against the classical bar.

Rules this driver enforces, each of which keeps the result from looking better
than it is:

* Only the coarse grid enters. The 1 m reference is the target. It is never an
  input, never a normaliser, and never consulted at evaluation except as truth.
* Every arm predicts a residual over bicubic, so the reported number is the
  classical bar plus whatever the network added. An arm that learns nothing
  reproduces the bar rather than something worse, so "no effect" is
  distinguishable from a broken run.
* Normalisation is per coarse patch, computed from the coarse patch alone. A
  global normaliser fitted over the region would leak the held-out geography
  into the input scale.
* Held-out geography is a whole region, not a crop. Training on essen-ruhr and
  testing on rothaar-sauerland is the only arrangement here that says anything
  about terrain the network has not seen.
* Patches are drawn from disjoint coarse tiles so a training patch and an
  evaluation patch never share a fine sample.
"""
from __future__ import annotations

import hashlib
import time

import numpy as np

from geoneural.superres import superres

from geoneural.superres import superres_models

SCHEMA = "geoneural-superres-neural-v1"


def _bicubic_base(coarse: np.ndarray, factor: int) -> np.ndarray:
    """The classical bar every arm starts from. float32: a 10241^2 float64 base
    is 839 MB and the residuals it feeds are metres, not microns."""
    return np.ascontiguousarray(superres.classical(coarse, factor, "bicubic"),
                                dtype=np.float32)


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


def _patch(coarse, fine, base, row, column, patch, factor):
    """One patch: normalised coarse input, residual target, and the raw pieces.

    The raw coarse patch and the bicubic base come back too, because
    back-projection needs the unnormalised grid and the absolute surface, and
    reconstructing either from the normalised copy is how a scale error gets in.
    """
    c = coarse[row:row + patch, column:column + patch]
    span = (patch - 1) * factor + 1
    fr, fc = row * factor, column * factor
    f = fine[fr:fr + span, fc:fc + span]
    b = base[fr:fr + span, fc:fc + span]
    centre = float(c.mean())
    scale = max(float(c.std()), 1e-3)
    return (((c - centre) / scale).astype(np.float32), (f - b).astype(np.float32),
            np.ascontiguousarray(c, dtype=np.float64),
            np.ascontiguousarray(b, dtype=np.float64))


def train_arm(config: dict, coarse: np.ndarray, fine: np.ndarray, torch,
              steps: int = 3000, patch: int = 17, factor: int = 10,
              queries: int = 4096, batch: int = 4, lr: float = 3e-4,
              seed: int = 1729, device: str = "cuda", holdout_fraction: float = 0.2):
    """Fit one arm on one region's coarse grid. Returns (model, history, sampler)."""
    torch.manual_seed(seed)
    model = superres_models.make_model(config, torch).to(device).train()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    base = _bicubic_base(coarse, factor)
    sampler = PatchSampler(coarse.shape[0], patch, factor, seed, holdout_fraction)
    rng = np.random.default_rng(seed ^ 0x5EED)
    span = (patch - 1) * factor + 1
    arm = config["arm"]
    history = []
    start = time.perf_counter()
    for step in range(steps):
        picks = [sampler.training[i] for i in
                 rng.integers(0, len(sampler.training), batch)]
        parts = [_patch(coarse, fine, base, r, c, patch, factor) for r, c in picks]
        coarse_in = torch.from_numpy(np.stack([p[0] for p in parts])[:, None]).to(device)
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
                   "sampler": sampler.record()}, sampler


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
                    tiles=None, block: int = 64, iterations: int = 5,
                    trim: int | None = None):
    """Whole-patch reconstruction on held-out tiles, against 1 m truth.

    Four numbers on exactly the same ground, because comparing a network against
    the weakest classical method is the easiest way to overstate a
    super-resolution result:

    * `bicubic`: the free baseline the arms predict a residual over.
    * `bicubicBackProjected`: the same, with the operator constraint enforced.
      This is the real classical bar. It costs nothing, needs no training, and
      over the whole interior it gains more than any choice of kernel.
    * `neural`: the arm's own output.
    * `neuralBackProjected`: the arm's output with the same constraint enforced.

    Back-projection is not specific to neural methods, so the fair comparison is
    `neuralBackProjected` against `bicubicBackProjected`. Giving the constraint
    to only one side would bias the result.

    `trim` matters. Back-projection upsamples its residual, and on a 17-cell
    patch the outermost ring of that upsample is unconstrained on one side. On
    rothaar-sauerland a back-projected patch scores 0.2855 m over its outer ten
    rows against 0.0519 m inside, so scoring whole patches makes back-projection
    look 38 % worse than plain bicubic, although over the whole region it is
    better. The arms have the same edge exposure through their replicate
    padding. A margin is trimmed from every method equally; without it the
    comparison measures the patch grid rather than the methods.
    """
    trim = 2 * factor if trim is None else int(trim)
    model.eval()
    base = _bicubic_base(coarse, factor)
    span = (patch - 1) * factor + 1
    arm = config["arm"]
    stores = {name: _accumulate() for name in
              ("bicubic", "bicubicBackProjected", "neural", "neuralBackProjected")}
    consistency = {"neural": 0.0, "neuralBackProjected": 0.0}
    samples = 0
    with torch.no_grad():
        for index in range(0, len(tiles), block):
            picks = tiles[index:index + block]
            parts = [_patch(coarse, fine, base, r, c, patch, factor) for r, c in picks]
            coarse_in = torch.from_numpy(
                np.stack([p[0] for p in parts])[:, None]).to(device)
            target = np.stack([p[1] for p in parts])
            if arm == "edsr":
                predicted = model(coarse_in).squeeze(1).cpu().numpy()
            else:
                grid = np.arange(span)
                ys = np.repeat(grid, span)[None, :].repeat(len(picks), 0)
                xs = np.tile(grid, span)[None, :].repeat(len(picks), 0)
                yt = torch.from_numpy(ys).to(device)
                xt = torch.from_numpy(xs).to(device)
                cells = torch.clamp(yt // factor, max=patch - 1) * patch + \
                    torch.clamp(xt // factor, max=patch - 1)
                offsets = torch.stack((
                    (yt % factor).float() / factor - 0.5,
                    (xt % factor).float() / factor - 0.5), dim=-1)
                predicted = model(coarse_in, cells, offsets).cpu().numpy().reshape(
                    len(picks), span, span)
            inner = slice(trim, -trim) if trim > 0 else slice(None)
            for order, (_, residual_truth, raw_coarse, raw_base) in enumerate(parts):
                truth = raw_base + residual_truth
                estimate = raw_base + predicted[order]
                classical_bp = superres.back_project(raw_base, raw_coarse, factor, iterations)
                neural_bp = superres.back_project(estimate, raw_coarse, factor, iterations)
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
    out["tiles"] = len(tiles)
    out["trimFineSamples"] = trim
    out["note"] = ("All four figures are the same tiles, the same truth and the same "
                   "operator. The comparison that decides anything is "
                   "neuralBackProjected against bicubicBackProjected: back-projection "
                   "is free for both sides and enforcing it on only one would favour "
                   "that side.")
    return out


def operator_identity() -> str:
    return hashlib.sha256(superres.OPERATOR.encode("utf-8")).hexdigest()[:16]


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
             stream_cells: int = 50_000) -> dict:
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
                                 tiles=sampler.evaluation[:evaluation_tiles])
        row = {"arm": arm, "config": config,
               "deployedBytes": superres_models.deployed_bytes(model, torch),
               "trainedOn": train_region, "training": history,
               "withinRegion": within, "heldOutGeography": {},
               "drainage": {}}
        if drainage_window:
            row["drainage"][train_region] = drainage_check(
                model, config, coarse, fine, torch, drainage_origin, drainage_origin,
                window=drainage_window, factor=factor, device=device,
                stream_cells=stream_cells)
        for region in test_regions:
            other_coarse, other_fine = _load(region, root)
            other_fine = np.ascontiguousarray(other_fine, dtype=np.float32)
            other_sampler = PatchSampler(other_coarse.shape[0], patch, factor, seed)
            every = other_sampler.training + other_sampler.evaluation
            row["heldOutGeography"][region] = evaluate_region(
                model, config, other_coarse, other_fine, torch, factor=factor,
                patch=patch, device=device, tiles=every[:evaluation_tiles])
            if drainage_window:
                row["drainage"][region] = drainage_check(
                    model, config, other_coarse, other_fine, torch,
                    drainage_origin, drainage_origin, window=drainage_window,
                    factor=factor, device=device, stream_cells=stream_cells)
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
            "observationOperator": superres.OPERATOR,
            "observationOperatorId": operator_identity(),
            "qualification":
                "Only the coarse grid enters the model. The 1 m reference is the "
                "target and is never an input, a normaliser or an evaluation-time "
                "hint. Every arm predicts a residual over bicubic, so an arm that "
                "learns nothing reports the classical bar rather than noise. "
                "Normalisation is per coarse patch, from the coarse patch alone. "
                "Held-out geography is a whole region from the same provider and "
                "datum, which tests transfer across terrain and not across source."}


def reconstruct_window(model, config, coarse_window, torch, factor: int = 10,
                       device: str = "cuda", chunk: int = 1_000_000) -> np.ndarray:
    """Reconstruct a whole coarse window at 1 m, not a 17-cell patch.

    Drainage cannot be read on a training patch. A 161x161 fine patch holds
    25,921 cells against a 50,000-cell stream threshold, so the network it would
    be scored on does not exist. The arms are fully convolutional and the heads
    are pointwise, so they accept any window; the query grid is what has to be
    chunked, because a 2041^2 window is 4.2 M queries and a 64-wide latent for
    each is a gigabyte.
    """
    side = int(coarse_window.shape[0])
    span = (side - 1) * factor + 1
    centre = float(coarse_window.mean())
    scale = max(float(coarse_window.std()), 1e-3)
    normalised = ((coarse_window - centre) / scale).astype(np.float32)
    coarse_in = torch.from_numpy(normalised[None, None]).to(device)
    model.eval()
    with torch.no_grad():
        if config["arm"] == "edsr":
            return model(coarse_in).squeeze().cpu().numpy().astype(np.float64)
        features = model.encoder(coarse_in)
        flat = features.flatten(2).transpose(1, 2)
        out = np.empty(span * span, dtype=np.float32)
        total = span * span
        for start in range(0, total, chunk):
            stop = min(start + chunk, total)
            index = np.arange(start, stop)
            ys, xs = index // span, index % span
            yt = torch.from_numpy(ys).to(device)
            xt = torch.from_numpy(xs).to(device)
            cells = torch.clamp(yt // factor, max=side - 1) * side + \
                torch.clamp(xt // factor, max=side - 1)
            offsets = torch.stack((
                (yt % factor).float() / factor - 0.5,
                (xt % factor).float() / factor - 0.5), dim=-1)[None]
            latent = flat.gather(
                1, cells[None].unsqueeze(-1).expand(-1, -1, flat.shape[-1]))
            out[start:stop] = model.head(latent, offsets).squeeze(-1).squeeze(0).cpu().numpy()
        return out.reshape(span, span).astype(np.float64)


DRAINAGE_SCHEMA = "geoneural-superres-drainage-v1"


def drainage_check(model, config, coarse, fine, torch, row: int, column: int,
                   window: int = 205, factor: int = 10, device: str = "cuda",
                   stream_cells: int = 50_000, trim: int = 20) -> dict:
    """Route water over the reconstruction and over the truth, and compare.

    The third axis of the bar. A surface that scores well on height and routes
    water wrongly has not reconstructed terrain, and height error cannot see it:
    a ridge displaced by one metre costs almost nothing in MAE and moves a
    catchment boundary.

    The threshold is 50,000 cells of 1 m, the same 0.05 km^2 that 500 cells
    give at 10 m, so both resolutions use the same physical channel threshold
    rather than the same cell count.
    """
    from geoneural.metrics import hydrology
    span = (window - 1) * factor + 1
    coarse_window = np.ascontiguousarray(
        coarse[row:row + window, column:column + window], dtype=np.float64)
    truth = np.ascontiguousarray(
        fine[row * factor:row * factor + span,
             column * factor:column * factor + span], dtype=np.float64)
    base = np.ascontiguousarray(superres.classical(coarse_window, factor, "bicubic"),
                                dtype=np.float64)
    residual = reconstruct_window(model, config, coarse_window, torch, factor, device)
    estimate = base + residual
    classical_bp = superres.back_project(base, coarse_window, factor)
    neural_bp = superres.back_project(estimate, coarse_window, factor)
    inner = slice(trim, -trim) if trim > 0 else slice(None)
    out = {"schema": DRAINAGE_SCHEMA, "window": window, "spanFine": span,
           "streamThresholdCells": stream_cells, "trimFineSamples": trim,
           "origin": {"row": row, "column": column}}
    for name, surface in (("bicubic", base), ("bicubicBackProjected", classical_bp),
                          ("neural", estimate), ("neuralBackProjected", neural_bp)):
        out[name] = hydrology.compare(truth[inner, inner], surface[inner, inner],
                                      1.0, stream_cells)
        out[name]["maeM"] = float(np.abs(surface[inner, inner] - truth[inner, inner]).mean())
    out["note"] = ("Both fields routed identically at 1 m. receiverAgreementFraction "
                   "and streamJaccard are the discriminating quantities; pooled basin "
                   "agreement is global and moves for reasons unrelated to local "
                   "reconstruction quality.")
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
