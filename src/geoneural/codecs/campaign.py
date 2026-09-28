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


def multilevel_rows(region, z, spatial, routed, bounds, coder, model=None, seed=None):
    rows = []
    for b in bounds:
        t = time.perf_counter()
        blob, recon, info = package.encode(z, b, coder, model, embed_model=False, spatial=spatial)
        te = time.perf_counter() - t
        t = time.perf_counter()
        est = package.decode(blob, model)
        td = time.perf_counter() - t
        if not np.array_equal(est, recon):
            raise RuntimeError(f"{coder} decode differs from the encoder's reconstruction")
        common = {"region": region, "coder": coder, "boundM": b, "encodeS": te, "decodeS": td, "E": info["E"],
                  "idealBytes": info["idealBits"] / 8, **score(z, est, b, routed)}
        if coder == "learned":
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


def geology_rasters(regions=REGIONS) -> tuple[dict, dict]:
    """GK100 material classes per region on the 40 m context lattice, one dictionary for all regions."""
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
        sp = SPACING_M * package.CONTEXT_FACTOR
        side = (1025 - 1) // package.CONTEXT_FACTOR + 1
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
       log=print) -> list[dict]:
    """Every arm starts from the H1 model of the same fold and seed and gets the same extra training (one
    closed-loop round), so the only difference between arms is the context they see."""
    rasters, legend = geology_rasters(regions)
    n_classes = len(legend) + 1
    rows = []
    for held in regions:
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


def h4(out: Path, regions=REGIONS, bounds=BOUNDS, seeds=(0,), steps: int = 5000, log=print) -> list[dict]:
    """Arms per fold, all ending with the same number of optimiser steps except `real` (the H1 model):
    real-extra continues the H1 model on real data for two more rounds (more real exposure, same steps as the
    pretrained arms); process-real and procedural-real start from a model pretrained on simulated or procedural
    terrain and then get the two real rounds of H1; the -only arms show what transfers without real data."""
    rows = []
    for seed in seeds:
        pre = {kind: pretrained(kind, seed, out, bounds, steps) for kind in ("process", "procedural")}
        for held in regions:
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
