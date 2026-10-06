"""Codec campaigns on the development regions.

H1: is a shared learned predictor inside the multilevel coder smaller than strong conventional codecs at the same
maximum error? Leave one region out: the model for a region is trained on the other five and never sees it.
Products are standalone rasters of the whole 1025 x 1025 field, each decoded from its own bytes and checked
against the bound on every node. The learned product is reported twice: standalone (model embedded, the complete
cost of one file) and corpus (model shared, counted once per corpus and reported separately).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from geoneural.codecs import foreign, package
from geoneural.codecs.predictor import Predictor, fit
from geoneural.common import HOME, read_json
from geoneural.metrics import drainage

REGIONS = ("essen-ruhr", "muensterland-plain", "lower-rhine", "teutoburg-forest", "bergisches-land",
           "rothaar-sauerland")
BOUNDS = (0.0005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
SPACING_M = 10.0


def load(region: str, root: Path | None = None) -> tuple[np.ndarray, dict]:
    atlas = (root or HOME / "atlases") / region
    return np.load(atlas / "reference.npy"), read_json(atlas / "atlas.json")


def height_metrics(ref: np.ndarray, est: np.ndarray, bound_m: float) -> dict:
    """Bound violations allow one float32 ulp of the height, the rounding of storing the result as float32;
    strict violations and the largest excess are recorded as well."""
    d = np.asarray(est, np.float64) - np.asarray(ref, np.float64)
    a = np.abs(d)
    ulp = np.spacing(np.abs(np.asarray(ref, np.float32))).astype(np.float64)
    return {"maxM": float(a.max()), "rmseM": float(np.sqrt((d * d).mean())), "maeM": float(a.mean()),
            "p99M": float(np.quantile(a, 0.99)), "biasM": float(d.mean()),
            "boundViolations": int((a > bound_m + ulp).sum()), "strictViolations": int((a > bound_m).sum()),
            "maxExcessM": float(max(0.0, (a - bound_m).max()))}


def score(ref, est, bound_m, routed) -> dict:
    return {**height_metrics(ref, est, bound_m),
            "drainage": drainage.stream_metrics(ref, est, SPACING_M, reference_routed=routed)}


def conventional_rows(region, z, spatial, routed, bounds, codecs=foreign.NAMES):
    rows = []
    for name in codecs:
        for b in bounds:
            t = time.perf_counter()
            payload = foreign.encode(name, z, b)
            te = time.perf_counter() - t
            blob = package.wrap_foreign(name, payload, z.shape, b, spatial)
            t = time.perf_counter()
            est = foreign.decode(name, package.read(blob).components["foreign"], z.shape)
            td = time.perf_counter() - t
            rows.append({"region": region, "product": "standalone", "coder": name, "family": "conventional",
                         "boundM": b, "bytes": len(blob), "payloadBytes": len(payload), "encodeS": te, "decodeS": td,
                         **score(z, est, b, routed)})
    return rows


def multilevel_rows(region, z, spatial, routed, bounds, coder, model=None, seed=None, rules=None, name=None):
    """Rows for one multilevel coder; `rules` maps a bound to its bound-allocation rule (none: uniform), and `name`
    replaces the coder name in the rows."""
    rows = []
    for b in bounds:
        rule = (rules or {}).get(b)
        t = time.perf_counter()
        blob, recon, info = package.encode(z, b, coder, model, embed_model=False, spatial=spatial, rule=rule)
        te = time.perf_counter() - t
        t = time.perf_counter()
        est = package.decode(blob, model)
        td = time.perf_counter() - t
        if not np.array_equal(est, recon):
            raise RuntimeError(f"{coder} decode differs from the encoder's reconstruction")
        common = {"region": region, "coder": name or coder, "boundM": b, "encodeS": te, "decodeS": td, "E": info["E"],
                  "rule": rule,
                  "idealBytes": info["idealBits"] / 8, **score(z, est, b, routed)}
        if coder == "learned":
            # The rule component is inside the corpus file already; the model and its directory entry are added.
            mb = model.nbytes() + 9
            rows.append({**common, "product": "corpus", "family": "learned", "seed": seed, "bytes": len(blob),
                         "sharedModelBytes": model.nbytes(), "modelSha256": model.sha256(),
                         "breakdown": info["breakdown"]})
            rows.append({**common, "product": "standalone", "family": "learned", "seed": seed, "bytes": len(blob) + mb,
                         "modelSha256": model.sha256()})
        else:
            rows.append({**common, "product": "standalone", "family": "multilevel-fixed", "bytes": len(blob),
                         "breakdown": info["breakdown"]})
    return rows


def model_for(held: str, seed: int, out: Path, regions=REGIONS, bounds=BOUNDS, widths=(32, 32), steps=5000,
              rounds=2) -> Predictor:
    path = out / "models" / f"loro-{held}-s{seed}.gnm"
    if path.exists():
        return Predictor.from_bytes(path.read_bytes())
    train = [load(r)[0] for r in regions if r != held]
    model = fit(train, bounds, rounds=rounds, widths=widths, steps=steps, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(model.to_bytes())
    return model


def h1(out: Path, regions=REGIONS, bounds=BOUNDS, seeds=(0, 1, 2), log=print) -> list[dict]:
    rows = []
    for held in regions:
        z, atlas = load(held)
        spatial = package.spatial_from_atlas(atlas)
        routed = drainage.route(z, SPACING_M)
        floor = drainage.noise_floor(z, SPACING_M)
        rows += [dict(r, noiseFloor=floor) for r in conventional_rows(held, z, spatial, routed, bounds)]
        for coder in ("cubic-order0", "cubic-ctx"):
            rows += multilevel_rows(held, z, spatial, routed, bounds, coder)
        for seed in seeds:
            model = model_for(held, seed, out, regions, bounds)
            rows += multilevel_rows(held, z, spatial, routed, bounds, "learned", model, seed)
            log(f"{held} seed {seed} done")
    return rows


# ---- H2: geology inside the learned coder ------------------------------------------------------------------

GEO_ARMS = ("none", "constant", "real", "shifted", "shuffled", "wrong-region")
CHARGED = {"real", "shifted", "shuffled", "wrong-region"}


def geology_rasters(regions=REGIONS, factor: int = package.CONTEXT_FACTOR) -> tuple[dict, dict]:
    """GK100 material classes per region on the context lattice (`factor` field nodes per cell, 40 m by default),
    one dictionary for all regions."""
    import glob
    from rasterio.features import rasterize
    from rasterio.transform import from_origin

    from geoneural.data import geology
    units = {r: geology.parse_units(glob.glob(str(HOME / "geology" / r / "*.gml"))) for r in regions}
    labels = sorted({(u["material"] or "").strip() for r in regions for u in units[r]} - {""})
    legend = {name: i + 1 for i, name in enumerate(labels)}  # 0: no mapped unit
    out = {}
    for r in regions:
        west, south, east, north = load(r)[1]["bounds"]
        sp = SPACING_M * factor
        side = (1025 - 1) // factor + 1
        shapes = [(u["geometry"], legend[u["material"].strip()]) for u in units[r] if (u["material"] or "").strip()]
        out[r] = rasterize(shapes, out_shape=(side, side), transform=from_origin(west - sp / 2, north + sp / 2, sp, sp),
                           fill=0, dtype="uint8")
    return out, legend


def control(arm: str, region: str, rasters: dict, regions=REGIONS, seed: int = 0) -> np.ndarray | None:
    """The context raster an arm uses for a region. Controls keep class frequencies or map statistics while
    breaking the link to this terrain: shifted rolls the map by 2.56 km, shuffled permutes 1 km blocks, and
    wrong-region uses the map of the next region in the list."""
    real = rasters[region]
    if arm == "none":
        return None
    if arm == "constant":
        return np.zeros_like(real)
    if arm == "real":
        return real
    if arm == "shifted":
        return np.roll(real, (64, 64), axis=(0, 1))
    if arm == "shuffled":
        b = 25
        n = real.shape[0] // b
        rng = np.random.default_rng(seed + 7)
        perm = rng.permutation(n * n)
        out = real.copy()
        for i, p in enumerate(perm):
            si, sj = divmod(i, n)
            pi, pj = divmod(int(p), n)
            out[si * b:(si + 1) * b, sj * b:(sj + 1) * b] = real[pi * b:(pi + 1) * b, pj * b:(pj + 1) * b]
        return out
    if arm == "wrong-region":
        return rasters[regions[(list(regions).index(region) + 1) % len(regions)]]
    raise ValueError(arm)


def extend(model: Predictor, classes: int, geo_dim: int = 4, seed: int = 0) -> Predictor:
    """The same predictor with a class embedding appended to its input: zero weights on the new inputs, so it
    starts as the model it came from."""
    (W1, b1), *rest = model.layers
    W = np.concatenate([W1.astype(np.float32), np.zeros((W1.shape[0], geo_dim), np.float32)], 1)
    embed = np.random.default_rng(seed).normal(0, 0.01, (classes, geo_dim))
    return Predictor([(W, b1)] + rest, embed=embed)


def h2(out: Path, regions=REGIONS, bounds=BOUNDS, seeds=(0, 1), arms=GEO_ARMS, steps: int = 3000,
       held_out=None, log=print) -> list[dict]:
    """Every arm starts from the H1 model of the same fold and seed and gets the same extra training (one
    closed-loop round), so the only difference between arms is the context they see. `held_out` runs only those
    folds (training still uses every other region), so folds can run in separate processes."""
    rasters, legend = geology_rasters(regions)
    n_classes = len(legend) + 1
    rows = []
    for held in held_out or regions:
        z, atlas = load(held)
        spatial = package.spatial_from_atlas(atlas)
        routed = drainage.route(z, SPACING_M)
        train = [r for r in regions if r != held]
        fields = [load(r)[0] for r in train]
        for seed in seeds:
            base = model_for(held, seed, out, regions, bounds)
            for arm in arms:
                path = out / "models" / f"h2-{held}-s{seed}-{arm}.gnm"
                if path.exists():
                    model = Predictor.from_bytes(path.read_bytes())
                else:
                    if arm == "none":
                        model = fit(fields, bounds, rounds=1, steps=steps, seed=seed + 50, init=base)
                    else:
                        ctx = [package.context_nodes(control(arm, r, rasters, regions, seed), 1025) for r in train]
                        model = fit(fields, bounds, rounds=1, steps=steps, seed=seed + 50, contexts=ctx,
                                    classes=n_classes, init=extend(base, n_classes, seed=seed))
                    path.write_bytes(model.to_bytes())
                raster = control(arm, held, rasters, regions, seed)
                for b in bounds:
                    blob, recon, info = package.encode(z, b, "learned", model, embed_model=False, spatial=spatial,
                                                       context=raster, context_charged=arm in CHARGED)
                    if not np.array_equal(package.decode(blob, model, context=raster), recon):
                        raise RuntimeError("geology product does not decode to its reconstruction")
                    rows.append({"region": held, "arm": arm, "seed": seed, "boundM": b, "bytes": len(blob),
                                 "contextBytes": info["breakdown"].get("context", 0),
                                 "sharedModelBytes": model.nbytes(), **score(z, recon, b, routed)})
                log(f"h2 {held} seed {seed} {arm} done")
    return rows


# ---- H3: bound allocation ------------------------------------------------------------------------------------

H3_RULES = {
    "streams-0.25": {"kind": "streams", "factor": 0.25, "areaM2": 50000.0, "dilate": 1, "fromStride": 8},
    "streams-0.5": {"kind": "streams", "factor": 0.5, "areaM2": 50000.0, "dilate": 2, "fromStride": 8},
    "slope-0.25": {"kind": "slope", "slopeRef": 0.02, "gamma": 1.0, "factor": 0.25, "fromStride": 8},
    "slope-0.5": {"kind": "slope", "slopeRef": 0.02, "gamma": 0.5, "factor": 0.5, "fromStride": 8},
}
H3_BOUNDS = (0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0)


def h3(out: Path, regions=REGIONS, bounds=H3_BOUNDS, rules=H3_RULES, log=print) -> list[dict]:
    """Uniform against allocated bounds, cubic-ctx and learned (H1 model, seed 0), compared at equal bytes."""
    rows = []
    for held in regions:
        z, atlas = load(held)
        spatial = package.spatial_from_atlas(atlas)
        routed = drainage.route(z, SPACING_M)
        model = model_for(held, 0, out, regions)
        for coder, m in (("cubic-ctx", None), ("learned", model)):
            for name, rule in {"uniform": None, **rules}.items():
                for b in bounds:
                    blob, recon, info = package.encode(z, b, coder, m, embed_model=False, spatial=spatial, rule=rule)
                    rows.append({"region": held, "coder": coder, "arm": name, "boundM": b, "bytes": len(blob),
                                 "tightenedFraction": info["tightenedFraction"], **score(z, recon, b, routed)})
            log(f"h3 {held} {coder} done")
    return rows


def at_equal_bytes(rows: list[dict], metric) -> list[dict]:
    """Each allocated point against the uniform curve of the same region and coder, interpolated in log bytes."""
    out = []
    keys = {(r["region"], r["coder"]) for r in rows}
    for region, coder in sorted(keys):
        uni = sorted((r for r in rows if r["region"] == region and r["coder"] == coder and r["arm"] == "uniform"),
                     key=lambda r: r["bytes"])
        xs = np.log([r["bytes"] for r in uni])
        for r in rows:
            if r["region"] != region or r["coder"] != coder or r["arm"] == "uniform":
                continue
            x = np.log(r["bytes"])
            if not xs[0] <= x <= xs[-1]:
                continue
            ref = {k: float(np.interp(x, xs, [metric(u, k) for u in uni])) for k in ("f1", "jaccard", "rmse")}
            out.append({"region": region, "coder": coder, "arm": r["arm"], "boundM": r["boundM"], "bytes": r["bytes"],
                        "f1": metric(r, "f1"), "f1Uniform": ref["f1"], "deltaF1": metric(r, "f1") - ref["f1"],
                        "deltaJaccard": metric(r, "jaccard") - ref["jaccard"],
                        "rmse": metric(r, "rmse"), "rmseUniform": ref["rmse"]})
    return out


def h3_metric(r: dict, key: str) -> float:
    if key == "f1":
        return r["drainage"]["tolerantF1"] or 0.0
    if key == "jaccard":
        return r["drainage"]["jaccard"] or 0.0
    return r["rmseM"]


# ---- Level-wise bounds: a lower typical error at the same largest error ----------------------------------------

LEVEL_RULES = {f"levels-{f}-{s}": {"kind": "levels", "factor": f, "toStride": s}
               for f in (0.5, 0.25, 0.0) for s in (4, 8, 16)}
LEVEL_BOUNDS = (0.05, 0.1, 0.25, 0.5, 1.0)


def levels(out: Path, regions=REGIONS, bounds=LEVEL_BOUNDS, seeds=(0, 1, 2), rules=LEVEL_RULES, held_out=None,
           log=print) -> list[dict]:
    """Uniform and level-wise bounds for cubic-ctx and the H1 leave-one-region-out models. Every product is
    decoded again from its bytes and must equal the encoder's reconstruction."""
    rows = []
    for held in held_out or regions:
        z, atlas = load(held)
        spatial = package.spatial_from_atlas(atlas)
        routed = drainage.route(z, SPACING_M)
        runs = [("cubic-ctx", None, None)] + [("learned", s, model_for(held, s, out, regions)) for s in seeds]
        for coder, seed, model in runs:
            for name, rule in {"uniform": None, **rules}.items():
                for b in bounds:
                    blob, recon, info = package.encode(z, b, coder, model, embed_model=coder == "learned",
                                                       spatial=spatial, rule=rule)
                    if not np.array_equal(package.decode(blob, model), recon):
                        raise RuntimeError(f"{held} {coder} {name} {b} does not decode to its reconstruction")
                    rows.append({"region": held, "coder": coder, "seed": seed, "arm": name, "boundM": b,
                                 "bytes": len(blob), "tightenedFraction": info.get("tightenedFraction"),
                                 **score(z, recon, b, routed)})
            log(f"levels {held} {coder} seed {seed} done")
    return rows


def confirm_levels(out: Path, cohort, rules: dict, bounds=LEVEL_BOUNDS, seeds=(0, 1, 2), log=print) -> list[dict]:
    """The frozen level-wise recipe, run once on a confirmation cohort: the conventional arms as in H1, the frozen
    H1 models with the frozen rule per bound (named learned, so the H1 verdict applies unchanged), and cubic-ctx
    uniform and with the same rules."""
    from geoneural.codecs import sz3tuned
    models = {s: Predictor.from_bytes((out / "models" / f"final-s{s}.gnm").read_bytes()) for s in seeds}
    rows = []
    for region in cohort:
        z, atlas = load(region)
        spatial = package.spatial_from_atlas(atlas)
        routed = drainage.route(z, SPACING_M)
        floor = drainage.noise_floor(z, SPACING_M)
        rows += [dict(r, noiseFloor=floor) for r in conventional_rows(region, z, spatial, routed, bounds)]
        if sz3tuned.binary():
            rows += sz3_best_rows((region,), bounds, log=lambda *_: None)
        rows += multilevel_rows(region, z, spatial, routed, bounds, "cubic-ctx")
        rows += multilevel_rows(region, z, spatial, routed, bounds, "cubic-ctx", rules=rules, name="cubic-ctx-levels")
        for seed, model in models.items():
            rows += multilevel_rows(region, z, spatial, routed, bounds, "learned", model, seed, rules=rules)
        log(f"confirm levels {region} done")
    return rows


# ---- Geology sensitivity: map resolution, translation, boundary blur, unknown classes ------------------------

GEO_PERTURB = ("res-10m", "res-160m", "shift-120m", "shift-320m", "shift-1km", "blur-3", "blur-5", "unknown-25")


def perturbed_raster(arm: str, region: str, rasters: dict, fine: dict, coarse: dict, seed: int = 0) -> np.ndarray:
    """The class raster of one sensitivity arm. Shifts are in whole 40 m cells; blur takes the most frequent class
    in a w x w window; unknown sets a quarter of the 1 km blocks to class 0 (no mapped unit)."""
    real = rasters[region]
    if arm == "res-10m":
        return fine[region]
    if arm == "res-160m":
        return coarse[region]
    if arm.startswith("shift-"):
        cells = {"shift-120m": 3, "shift-320m": 8, "shift-1km": 25}[arm]
        return np.roll(real, (cells, cells), axis=(0, 1))
    if arm.startswith("blur-"):
        from scipy.ndimage import uniform_filter
        w = int(arm.split("-")[1])
        counts = np.stack([uniform_filter((real == k).astype(np.float32), w, mode="nearest")
                           for k in range(int(real.max()) + 1)])
        return counts.argmax(0).astype(np.uint8)
    if arm == "unknown-25":
        out = real.copy()
        b = 25
        n = real.shape[0] // b
        rng = np.random.default_rng(seed + 11)
        for i in rng.choice(n * n, (n * n) // 4, replace=False):
            si, sj = divmod(int(i), n)
            out[si * b:(si + 1) * b, sj * b:(sj + 1) * b] = 0
        return out
    raise ValueError(arm)


def geology_sweep(out: Path, regions=REGIONS, bounds=BOUNDS, seed: int = 0, arms=GEO_PERTURB, steps: int = 3000,
                  held_out=None, log=print) -> list[dict]:
    """The H2 recipe (same start, same extra round, same seed) with perturbed maps, for comparison with the H2
    rows of the same seed (no map and the real map). Exploratory: one seed."""
    rasters, legend = geology_rasters(regions)
    fine, _ = geology_rasters(regions, 1)
    coarse, _ = geology_rasters(regions, 16)
    n_classes = len(legend) + 1
    rows = []
    for held in held_out or regions:
        z, atlas = load(held)
        spatial = package.spatial_from_atlas(atlas)
        routed = drainage.route(z, SPACING_M)
        train = [r for r in regions if r != held]
        fields = [load(r)[0] for r in train]
        base = model_for(held, seed, out, regions, bounds)
        for arm in arms:
            path = out / "models" / f"geo-{held}-s{seed}-{arm}.gnm"
            if path.exists():
                model = Predictor.from_bytes(path.read_bytes())
            else:
                ctx = [package.context_nodes(perturbed_raster(arm, r, rasters, fine, coarse, seed), 1025) for r in train]
                model = fit(fields, bounds, rounds=1, steps=steps, seed=seed + 50, contexts=ctx, classes=n_classes,
                            init=extend(base, n_classes, seed=seed))
                path.write_bytes(model.to_bytes())
            raster = perturbed_raster(arm, held, rasters, fine, coarse, seed)
            for b in bounds:
                blob, recon, info = package.encode(z, b, "learned", model, embed_model=False, spatial=spatial,
                                                   context=raster, context_charged=True)
                if not np.array_equal(package.decode(blob, model, context=raster), recon):
                    raise RuntimeError("perturbed geology product does not decode to its reconstruction")
                rows.append({"region": held, "arm": arm, "seed": seed, "boundM": b, "bytes": len(blob),
                             "contextBytes": info["breakdown"].get("context", 0),
                             "sharedModelBytes": model.nbytes(), **score(z, recon, b, routed)})
            log(f"geology sweep {held} {arm} done")
    return rows


# ---- Conventional frontier: C1, C2 and C3 (protocol arms isolating the learned coder's parts) ----------------

FRONTIER_BOUNDS = (0.05, 0.1, 0.25, 0.5, 1.0)
C1_GRID = [(f, m, rc) for f in (2, 4, 8, 16) for m in (0.5, 1.0, 2.0) for rc in ("sz3", "sperr")]
UNIFORM_GRID = (0.03, 0.04, 0.05, 0.07, 0.1, 0.14, 0.2, 0.25, 0.35, 0.5, 0.7, 1.0, 1.4)


def frontier(out: Path, regions=REGIONS, bounds=FRONTIER_BOUNDS, held_out=None, log=print) -> list[dict]:
    """C1 over its configuration grid, C3 paired with C1 at one configuration (factor 4, base bound equal to the
    target), C2 on SZ3 and SPERR with three correction masks, and the uniform SZ3 and SPERR curves the C2 points
    are judged against. Every product is decoded again from its bytes."""
    from geoneural.codecs import frontier as fr
    rasters, legend = geology_rasters(regions)
    rows = []

    def keep(region, arm, b, blob, recon, z, routed, **extra):
        if not np.array_equal(fr.decode(blob), recon):
            raise RuntimeError(f"{region} {arm} {b} does not decode to its reconstruction")
        rows.append({"region": region, "arm": arm, "boundM": b, "bytes": len(blob), **extra,
                     **score(z, recon, b, routed)})

    for region in held_out or regions:
        z, atlas = load(region)
        spatial = package.spatial_from_atlas(atlas)
        routed = drainage.route(z, SPACING_M)
        raster = rasters[region]
        for b in bounds:
            for f, m, rc in C1_GRID:
                blob, recon = fr.encode_base(z, b, f, m * b, residual=rc, spatial=spatial)
                keep(region, f"c1-f{f}-m{m}-{rc}", b, blob, recon, z, routed, family="c1")
            for arm, kw in (("c3-morph", {}),
                            ("c3-geology", {"raster": raster}),
                            ("c3-geology-shifted", {"raster": control("shifted", region, rasters, regions)})):
                ras = kw.get("raster")
                blob, recon = fr.encode_base(z, b, 4, b, regression=True, raster=ras, spatial=spatial)
                keep(region, arm, b, blob, recon, z, routed, family="c3")
            ref_streams = drainage.streams(routed)
            masks = {"band0": ref_streams, "band1": drainage._dilate(ref_streams, 1),
                     "margin": fr.margin_mask(z, b, SPACING_M, drainage.STREAM_AREA_M2 / 4)}
            for codec in ("sz3", "sperr"):
                for mname, mask in masks.items():
                    blob, recon = fr.encode_corrected(z, b, codec, mask, b / 4, spatial)
                    keep(region, f"c2-{codec}-{mname}", b, blob, recon, z, routed, family="c2", codec=codec,
                         correctedCells=int(mask.sum()))
        for codec in ("sz3", "sperr"):
            for b in UNIFORM_GRID:
                blob = foreign.encode(codec, np.ascontiguousarray(z, np.float32), b)
                recon = foreign.decode(codec, blob, z.shape).astype(np.float64)
                wrapped = package.wrap_foreign(codec, blob, z.shape, b, spatial)
                rows.append({"region": region, "arm": f"uniform-{codec}", "boundM": b, "bytes": len(wrapped),
                             "family": "uniform", "codec": codec, **score(z, recon, b, routed)})
        log(f"frontier {region} done")
    return rows


# ---- Drainage sensitivity: stream threshold, boundary policy and routing domain ----------------------------

SENS_AREAS = (25_000.0, 50_000.0, 100_000.0, 200_000.0)
POLICIES = ("edges", "lowest-edge")
WIDE_MARGIN = 512  # nodes of the 5.12 km routing margin around a development region


def _best_conventional_row(rows, region, b):
    ok = [r for r in rows if r["region"] == region and r["boundM"] == b and r.get("family") == "conventional"
          and r.get("boundViolations") == 0]
    return min(ok, key=lambda r: r["bytes"]) if ok else None


def _conventional_recon(coder, z, b):
    from geoneural.codecs import sz3tuned
    z32 = np.ascontiguousarray(z, np.float32)
    if coder == "sz3-best":
        return np.asarray(sz3tuned.best(z32, b)["best"]["decoded"], np.float64)
    return foreign.decode(coder, foreign.encode(coder, z32, b), z.shape).astype(np.float64)


def _f1s(ref_routed, est_routed, areas):
    return {str(int(a)): drainage._overlap(drainage.streams(ref_routed, a), drainage.streams(est_routed, a), 1)
            ["tolerantF1"] for a in areas}


def _wide_path(region: str) -> Path:
    return HOME / "atlases" / f"{region}-wide" / "reference.npy"


def wide_domain(region: str) -> tuple[np.ndarray, np.ndarray]:
    """The region with a WIDE_MARGIN-node margin and its nodata mask. Published atlases have none; a domain whose
    publication was refused for small holes in the margin is merged again from its acquired tiles, and the holes
    become outlets in routing (they are never filled)."""
    from geoneural.data import build
    if _wide_path(region).exists():
        W = np.load(_wide_path(region)).astype(np.float64)
        return W, np.zeros(W.shape, bool)
    W = build.merge(HOME / "raw" / f"{region}-wide" / "input.json")[0].astype(np.float64)
    holes = ~np.isfinite(W)
    W[holes] = 0.0
    return W, holes


def drainage_sensitivity(out: Path, log=print) -> list[dict]:
    """The H1 and H1b drainage comparisons again on their confirmation cohorts at four stream thresholds and under
    two boundary policies, and the development comparison on each region embedded in a routing domain 5.12 km
    wider than the scored core. Products: the learned coder (frozen models; H1b with its frozen rules) and the best
    conventional product of that region and bound as recorded by the confirmation run."""
    from geoneural.common import COHORT_B_CONFIG, COHORT_CONFIG
    recipe = read_json(Path(__file__).resolve().parents[3] / "results" / "v2" / "frozen" / "levels-recipe.json")
    rules = {float(b): r for b, r in recipe["rules"].items()}
    finals = {s: Predictor.from_bytes((out / "models" / f"final-s{s}.gnm").read_bytes()) for s in (0, 1, 2)}
    rows = []
    sets = [("A", list(read_json(COHORT_CONFIG)), read_json(out / "h1-confirmation.json")["results"]["rows"], None),
            ("B", list(read_json(COHORT_B_CONFIG)), read_json(out / "h1b-levels-confirmation.json")["results"]["rows"],
             rules)]
    for cohort, regions, report, rule_map in sets:
        for region in regions:
            z, _ = load(region)
            for b in LEVEL_BOUNDS:
                best = _best_conventional_row(report, region, b)
                products = {f"conventional:{best['coder']}": _conventional_recon(best["coder"], z, b)}
                for s, m in finals.items():
                    rule = (rule_map or {}).get(b)
                    products[f"learned:s{s}"] = package.encode(z, b, "learned", m, rule=rule)[1]
                for policy in POLICIES:
                    ref = drainage.route(z, SPACING_M, policy)
                    for name, rec in products.items():
                        rows.append({"study": "cohort", "cohort": cohort, "region": region, "boundM": b,
                                     "product": name, "policy": policy,
                                     "f1": _f1s(ref, drainage.route(rec, SPACING_M, policy), SENS_AREAS)})
            log(f"sensitivity cohort {cohort} {region} done")
    dev = read_json(out / "h1.json")["results"]["rows"]
    if (out / "sz3-best.json").exists():
        dev += [r for r in read_json(out / "sz3-best.json")["results"]["rows"] if "bytes" in r]
    for region in REGIONS:
        W, holes = wide_domain(region)
        z, _ = load(region)
        k = WIDE_MARGIN
        core = (slice(k, k + z.shape[0]), slice(k, k + z.shape[1]))
        if holes[core].any() or np.abs(W[core][1:-1, 1:-1] - z[1:-1, 1:-1]).max() > 1e-3:
            raise RuntimeError(f"{region}: the wide domain does not contain the region at the expected offset")
        # The region's outermost ring carries its own crop's resampling edge, so the core holds the region exactly.
        rows.append({"study": "wide-domain", "region": region, "holeNodes": int(holes.sum()),
                     "ringMaxDifferenceM": float(np.abs(W[core] - z).max())})
        W[core] = z
        ref_core = drainage.route(z, SPACING_M)
        ref_wide = drainage.route(W, SPACING_M, holes=holes)
        ref_wide_core = {"cells": ref_wide["cells"][core], "spacingM": SPACING_M}
        rows.append({"study": "wide-reference", "region": region,
                     "f1": {str(int(a)): drainage._overlap(drainage.streams(ref_core, a),
                                                           drainage.streams(ref_wide_core, a), 1)["tolerantF1"]
                            for a in SENS_AREAS}})
        model = model_for(region, 0, out)
        for b in LEVEL_BOUNDS:
            best = _best_conventional_row(dev, region, b)
            products = {f"conventional:{best['coder']}": _conventional_recon(best["coder"], z, b),
                        "learned:s0": package.encode(z, b, "learned", model)[1],
                        "learned-levels:s0": package.encode(z, b, "learned", model, rule=rules.get(b))[1]}
            for name, rec in products.items():
                Wd = W.copy()
                Wd[core] = rec
                est_wide = drainage.route(Wd, SPACING_M, holes=holes)
                rows.append({"study": "wide", "region": region, "boundM": b, "product": name,
                             "core": _f1s(ref_core, drainage.route(rec, SPACING_M), SENS_AREAS),
                             "wide": {str(int(a)): drainage._overlap(
                                 drainage.streams(ref_wide_core, a),
                                 drainage.streams({"cells": est_wide["cells"][core], "spacingM": SPACING_M}, a),
                                 1)["tolerantF1"] for a in SENS_AREAS}})
        log(f"sensitivity wide {region} done")
    return rows


# ---- Paged product: random access -------------------------------------------------------------------------

PAGE = 256  # pages of (PAGE + 1)^2 nodes sharing their edges


def pages(side: int = 1025):
    n = (side - 1) // PAGE
    return [(i * PAGE, j * PAGE) for i in range(n) for j in range(n)]


def paged_rows(region, z, routed, bounds, codecs=("sz3", "sperr"), model=None) -> list[dict]:
    """Every page coded alone; a page directory of 8 bytes per page is charged. Shared edges are coded twice
    (charged) and the stitched field takes each node from the first page that holds it."""
    rows = []
    tiles = pages(z.shape[0])
    directory = 8 * len(tiles)
    arms = [(c, "conventional") for c in codecs] + [("cubic-ctx", "multilevel-fixed")]
    if model is not None:
        arms.append(("learned", "learned"))
    for name, family in arms:
        for b in bounds:
            est = np.full(z.shape, np.nan)
            total = directory
            for r0, c0 in reversed(tiles):
                tile = np.ascontiguousarray(z[r0:r0 + PAGE + 1, c0:c0 + PAGE + 1])
                if family == "conventional":
                    blob = package.wrap_foreign(name, foreign.encode(name, tile, b), tile.shape, b)
                    dec = foreign.decode(name, package.read(blob).components["foreign"], tile.shape)
                else:
                    blob, dec, _ = package.encode(tile, b, name, model, embed_model=False)
                total += len(blob)
                est[r0:r0 + PAGE + 1, c0:c0 + PAGE + 1] = dec
            rows.append({"region": region, "product": "paged", "coder": name, "family": family, "boundM": b,
                         "bytes": total, "pageNodes": (PAGE + 1) ** 2, "pages": len(tiles),
                         **score(z, est, b, routed)})
    return rows


def paged_campaign(out: Path, regions=REGIONS, bounds=(0.05, 0.1, 0.25, 0.5, 1.0), log=print) -> list[dict]:
    rows = []
    for held in regions:
        z, _ = load(held)
        routed = drainage.route(z, SPACING_M)
        rows += paged_rows(held, z, routed, bounds, model=model_for(held, 0, out, regions))
        log(f"paged {held} done")
    return rows


def sz3_best_rows(regions=REGIONS, bounds=BOUNDS, log=print) -> list[dict]:
    """The best of several SZ3 configurations per field and bound (codecs/sz3tuned), as a conventional arm."""
    from geoneural.codecs import sz3tuned
    rows = []
    for held in regions:
        z, atlas = load(held)
        spatial = package.spatial_from_atlas(atlas)
        routed = drainage.route(z, SPACING_M)
        for b in bounds:
            found = sz3tuned.best(z, b)
            if found["best"] is None:
                rows.append({"region": held, "coder": "sz3-best", "boundM": b, "failure": "no configuration held the bound",
                             "tried": found["tried"]})
                continue
            blob = package.wrap_foreign("sz3cli", found["best"]["blob"], z.shape, b, spatial)
            rows.append({"region": held, "product": "standalone", "coder": "sz3-best", "family": "conventional",
                         "boundM": b, "bytes": len(blob), "config": found["best"]["config"], "tried": found["tried"],
                         **score(z, found["best"]["decoded"], b, routed)})
        log(f"sz3-best {held} done")
    return rows


# ---- H4: process prior ---------------------------------------------------------------------------------------

H4_ARMS = ("real", "real-extra", "process-real", "procedural-real", "process-only", "procedural-only")


def synthetic_fields(kind: str, root: Path | None = None) -> list[np.ndarray]:
    folder = (root or HOME / "synthetic") / kind
    paths = sorted(folder.glob("*.npy"))
    if not paths:
        raise FileNotFoundError(f"no synthetic {kind} fields under {folder}")
    return [np.load(p) for p in paths]


def pretrained(kind: str, seed: int, out: Path, bounds=BOUNDS, steps: int = 5000) -> Predictor:
    path = out / "models" / f"h4-pretrain-{kind}-s{seed}.gnm"
    if path.exists():
        return Predictor.from_bytes(path.read_bytes())
    model = fit(synthetic_fields(kind), bounds, rounds=2, steps=steps, seed=seed + 300)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(model.to_bytes())
    return model


def h4(out: Path, regions=REGIONS, bounds=BOUNDS, seeds=(0,), steps: int = 5000, held_out=None,
       log=print) -> list[dict]:
    """Arms per fold, all ending with the same number of optimiser steps except `real` (the H1 model):
    real-extra continues the H1 model on real data for two more rounds (more real exposure, same steps as the
    pretrained arms); process-real and procedural-real start from a model pretrained on simulated or procedural
    terrain and then get the two real rounds of H1; the -only arms show what transfers without real data."""
    rows = []
    for seed in seeds:
        pre = {kind: pretrained(kind, seed, out, bounds, steps) for kind in ("process", "procedural")}
        log(f"h4 pretrained models for seed {seed} ready")
        for held in held_out if held_out is not None else regions:
            z, atlas = load(held)
            spatial = package.spatial_from_atlas(atlas)
            routed = drainage.route(z, SPACING_M)
            train = [load(r)[0] for r in regions if r != held]
            base = model_for(held, seed, out, regions, bounds)
            for arm in H4_ARMS:
                path = out / "models" / f"h4-{held}-s{seed}-{arm}.gnm"
                if arm == "real":
                    model = base
                elif arm.endswith("-only"):
                    model = pre[arm.split("-")[0]]
                elif path.exists():
                    model = Predictor.from_bytes(path.read_bytes())
                else:
                    init = base if arm == "real-extra" else pre[arm.split("-")[0]]
                    model = fit(train, bounds, rounds=2, steps=steps, seed=seed + 600, init=init)
                    path.write_bytes(model.to_bytes())
                for b in bounds:
                    blob, recon, _ = package.encode(z, b, "learned", model, embed_model=False, spatial=spatial)
                    rows.append({"region": held, "arm": arm, "seed": seed, "boundM": b, "bytes": len(blob),
                                 **score(z, recon, b, routed)})
                log(f"h4 {held} seed {seed} {arm} done")
    return rows


# ---- Confirmation -----------------------------------------------------------------------------------------

def final_model(seed: int, out: Path, regions=REGIONS, bounds=BOUNDS, widths=(32, 32), steps=5000,
                rounds=2) -> Predictor:
    """The frozen learned predictor: the H1 recipe trained on all six development regions."""
    path = out / "models" / f"final-s{seed}.gnm"
    if path.exists():
        return Predictor.from_bytes(path.read_bytes())
    model = fit([load(r)[0] for r in regions], bounds, rounds=rounds, widths=widths, steps=steps, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(model.to_bytes())
    return model


def confirm_h1(out: Path, cohort, bounds=BOUNDS, seeds=(0, 1, 2), log=print) -> list[dict]:
    """The frozen H1 arms, run once on the confirmation regions. Model files must already exist (frozen)."""
    from geoneural.codecs import sz3tuned
    models = {}
    for seed in seeds:
        path = out / "models" / f"final-s{seed}.gnm"
        if not path.exists():
            raise FileNotFoundError(f"frozen model {path.name} is missing; freeze the recipe first")
        models[seed] = Predictor.from_bytes(path.read_bytes())
    rows = []
    for region in cohort:
        z, atlas = load(region)
        spatial = package.spatial_from_atlas(atlas)
        routed = drainage.route(z, SPACING_M)
        floor = drainage.noise_floor(z, SPACING_M)
        rows += [dict(r, noiseFloor=floor) for r in conventional_rows(region, z, spatial, routed, bounds)]
        if sz3tuned.binary():
            rows += sz3_best_rows((region,), bounds, log=lambda *_: None)
        for coder in ("cubic-order0", "cubic-ctx"):
            rows += multilevel_rows(region, z, spatial, routed, bounds, coder)
        for seed, model in models.items():
            rows += multilevel_rows(region, z, spatial, routed, bounds, "learned", model, seed)
        log(f"confirm h1 {region} done")
    return rows
