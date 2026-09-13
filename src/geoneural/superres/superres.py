"""Reconstructing terrain the coarse grid does not contain.

A conventional codec has no answer to this task. A codec is given every sample
it must reproduce; super-resolution is given a 10 m grid and asked for 1 m
detail that was never in it. The conventional baseline is therefore not
`q32-delta-zstd` but interpolation (bicubic, Lanczos, splines), and beating
interpolation is a different question from beating an entropy coder.

Constraints on the experiment:

The observation operator is declared and hashed. O = area-average of the 1 m
source onto the canonical 10 m lattice, with the same node-centred transform,
edge trim and datum the atlas uses. Training on a coarse grid built by one
operator and testing against another measures the operator, not the model.

The operator-mismatch test is required. The provider's own 10 m response is a
different coarsening of the same terrain (on essen-ruhr: 0.196 m mean absolute
difference, 1.41 m p99, 7.2 m max). Feeding that instead of O(1 m) tests
whether a model learned terrain or one resampler's kernel.

Nothing at 1 m enters training, selection, normalisation or tuning. The fine
data is validation and test only. Normalisers come from the coarse patch.

Evaluating a continuous function at 1 m intervals is not 1 m accuracy. That
claim requires independent higher-resolution measurement, which is what the
1 m acquisitions are for, with their own acquisition and datum caveats.

Three regions have 1 m coverage: essen-ruhr, rothaar-sauerland and
muensterland-plain, chosen as the middle, the roughest and the flattest of the
regions by relief, so a held-out-geography claim is possible rather than only a
within-region one.
"""
from __future__ import annotations

import pathlib

import numpy as np

SCHEMA = "geoneural-superres-v1"

#: The declared observation operator. Changing this string changes the experiment.
OPERATOR = "area-average of the 1 m source onto the node-centred 10 m lattice " \
           "(rasterio Resampling.average, edge_trim 2, EPSG:25832 / EPSG:7837)"


def fine_reference(input_json, out_path, spacing_m: float = 1.0,
                   edge_trim: int = 2) -> dict:
    """Mosaic a 1 m acquisition onto its own lattice, refusing holes.

    As in `build.prepare` at 10 m, missing data never becomes zero height and a
    partially covered tile is refused. The result is memory-mapped by everything
    downstream because a 10.24 km region at 1 m is 10241^2 float32, about 420 MB.
    """
    import rasterio
    from rasterio.transform import Affine
    from rasterio.warp import reproject, Resampling

    from geoneural.common import read_json, sha_file
    from geoneural.data.build import erode_valid

    input_json = pathlib.Path(input_json)
    source = read_json(input_json)
    config = source["config"]
    west, south, east, north = (float(v) for v in config["bbox"])
    side = int(round((east - west) / spacing_m)) + 1
    transform = Affine(spacing_m, 0, west - spacing_m / 2, 0, -spacing_m, north + spacing_m / 2)
    merged = np.full((side, side), np.nan, dtype=np.float32)
    for item in sorted(source["files"], key=lambda x: x["path"]):
        path = (input_json.parent / item["path"]).resolve()
        if sha_file(path) != item["sha256"]:
            raise ValueError(f"Source changed after acquisition: {path}")
        temporary = np.full_like(merged, np.nan)
        with rasterio.open(path) as dataset:
            reproject(source=rasterio.band(dataset, 1), destination=temporary,
                      src_transform=dataset.transform, src_crs=dataset.crs,
                      src_nodata=dataset.nodata, dst_transform=transform,
                      dst_crs=config["crs"], dst_nodata=np.nan,
                      resampling=Resampling.nearest, num_threads=2, warp_mem_limit=256)
        valid = erode_valid(np.isfinite(temporary), edge_trim)
        take = valid & ~np.isfinite(merged)
        merged[take] = temporary[take]
        del temporary
    missing = int(np.count_nonzero(~np.isfinite(merged)))
    if missing:
        raise ValueError(
            f"{missing} of {merged.size} fine samples are nodata ({missing / merged.size:.4%}); "
            "zero-filling is forbidden. Move the tile or lower --edge-trim and accept the "
            "reprojection edge bias, which is a deliberate trade and not an automatic one")
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, merged)
    return {"schema": SCHEMA, "path": str(out_path), "side": side,
            "spacingM": spacing_m, "bbox": [west, south, east, north],
            "crs": config["crs"], "verticalCrs": config["vertical_crs"],
            "edgeTrim": edge_trim,
            "reliefM": float(np.ptp(merged)),
            "meanM": float(merged.mean()), "stdM": float(merged.std()),
            "sha256": sha_file(out_path),
            "resampling": "nearest (a 1 m source onto a 1 m lattice is a registration, not a "
                          "resampling; nearest keeps the measured values rather than blending "
                          "them, which would make the truth a filtered product)",
            "qualification": "NRW DGM1 at its native spacing, mosaicked and halo-trimmed. This is "
                             "the independent higher-resolution measurement a super-resolution "
                             "claim is validated against; it carries its own acquisition and "
                             "datum uncertainty and is not error-free ground truth."}


def _axis_weights(side: int, factor: int) -> np.ndarray:
    """Trapezoidal cell weights for one axis, one row per coarse node.

    A cell centred on a node spans half a coarse cell either side. With an even
    factor there is no symmetric window of whole samples, so the two endpoints
    carry half weight (the trapezoid rule) and the result is exact for a linear
    field, which is what makes the operator unbiased.

    The obvious alternatives both fail that test. A window of `factor` samples
    starting at the node is offset by half a cell; a window ending at it is
    offset the other way. rasterio's `Resampling.average` takes the first, so
    this operator differs from it by half a source sample, and
    `source_control`'s provider comparison inherits that bias. Half a sample is
    0.5 m at 1 m spacing, around 0.05 m of systematic height on typical NRW
    gradients, and a systematic operator error shows up in the results as
    uniform model error.

    Edge cells keep only the part of the window that exists and renormalise, so
    the operator is defined on the whole lattice rather than on an interior crop.
    """
    nodes = (side - 1) // factor + 1
    half = factor / 2.0
    weights = np.zeros((nodes, side), dtype=np.float64)
    for index in range(nodes):
        centre = index * factor
        low = centre - half
        high = centre + half
        first = int(np.ceil(low))
        last = int(np.floor(high))
        positions = np.arange(max(first, 0), min(last, side - 1) + 1)
        if positions.size == 0:
            weights[index, min(max(centre, 0), side - 1)] = 1.0
            continue
        row = np.ones(positions.size, dtype=np.float64)
        if positions[0] == first and first == low:
            row[0] = 0.5
        if positions[-1] == last and last == high:
            row[-1] = 0.5
        weights[index, positions] = row / row.sum()
    return weights


def observe(fine: np.ndarray, factor: int = 10) -> np.ndarray:
    """The declared operator: area-average the fine field onto the coarse lattice.

    Each coarse node averages the fine samples in the cell centred on it. See
    `_axis_weights` for why the endpoints are half-weighted and for what the two
    obvious alternatives get wrong.
    """
    fine = np.asarray(fine, dtype=np.float64)
    rows = _axis_weights(fine.shape[0], factor)
    columns = _axis_weights(fine.shape[1], factor)
    return rows @ fine @ columns.T


def classical(coarse: np.ndarray, factor: int, method: str) -> np.ndarray:
    """Interpolation controls. These are what a super-resolution claim must beat.

    `bicubic` and `lanczos` are the ones a practitioner would actually reach for;
    `bilinear` is the atlas's own declared reconstruction for parent levels, so it
    is the floor rather than a strawman.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import Affine
    from rasterio.warp import reproject

    resampling = {"bilinear": Resampling.bilinear, "bicubic": Resampling.cubic,
                  "lanczos": Resampling.lanczos,
                  "cubic_spline": Resampling.cubic_spline}[method]
    source = np.asarray(coarse, dtype=np.float32)
    # Non-square blocks are the normal case: a tiled evaluation passes a band of
    # coarse rows across the full width, so each axis is sized from its own length.
    rows = (source.shape[0] - 1) * factor + 1
    columns = (source.shape[1] - 1) * factor + 1
    out = np.empty((rows, columns), dtype=np.float32)
    # Both lattices are node-centred and must share their node positions, so each
    # transform's origin sits half a cell outside its first centre. Aligning the
    # origins instead would shift the coarse grid half a coarse cell against the
    # fine one: five metres of systematic offset at a factor of ten, which would
    # read as model error.
    src_transform = Affine(float(factor), 0, -float(factor) / 2.0,
                           0, -float(factor), float(factor) / 2.0)
    dst_transform = Affine(1.0, 0, -0.5, 0, -1.0, 0.5)
    reproject(source=source, destination=out, src_transform=src_transform,
              src_crs="EPSG:25832", dst_transform=dst_transform, dst_crs="EPSG:25832",
              resampling=resampling, num_threads=2)
    return out.astype(np.float64)


def back_project(estimate: np.ndarray, coarse: np.ndarray, factor: int,
                 iterations: int = 5) -> np.ndarray:
    """Force the estimate to average back to the coarse grid it was given.

    Iterative back-projection. It is not specific to neural methods, so it is
    applied to every method here, including the classical ones. A
    super-resolution result that does not reproduce its own input under the
    declared operator violates the one constraint it was given for free, and
    enforcing that constraint only for the network would bias the comparison in
    its favour.
    """
    current = np.array(estimate, dtype=np.float64, copy=True)
    for _ in range(iterations):
        residual = np.asarray(coarse, dtype=np.float64) - observe(current, factor)
        current = current + classical(residual, factor, "bilinear")
    return current


def operator_consistency(estimate: np.ndarray, coarse: np.ndarray, factor: int) -> dict:
    """How far the estimate is from reproducing its own input."""
    difference = observe(estimate, factor) - np.asarray(coarse, dtype=np.float64)
    absolute = np.abs(difference)
    return {"maeM": float(absolute.mean()), "maxM": float(absolute.max()),
            "note": "||O(estimate) - coarse||. A method that cannot reproduce the grid it was "
                    "handed is not reconstructing detail, it is changing the answer it was given."}


BASELINE_SCHEMA = "geoneural-superres-baseline-v1"


def classical_baselines(fine_path, coarse_grid, factor: int = 10,
                        methods=("bilinear", "bicubic", "lanczos", "cubic_spline"),
                        back_projection: bool = True, tile: int = 2048) -> dict:
    """What interpolation achieves at 1 m, which is the bar a network must clear.

    Evaluated in tiles because a 10241^2 float64 field and four upsampled copies
    of it do not fit in memory at once. Errors are accumulated exactly
    (sum, sum of squares, max) rather than averaged per tile, so the figures are
    the whole-field ones and not a mean of means.

    The interior is scored. Edge cells of the operator average a half cell, so
    their coarse values mean something slightly different from the interior's,
    and including them would charge every method for the operator's edge
    convention.
    """
    fine = np.load(fine_path, mmap_mode="r")
    coarse = np.asarray(coarse_grid, dtype=np.float64)
    side = fine.shape[0]
    margin = factor
    rows = {}
    for method in methods:
        for projected in ((False, True) if back_projection else (False,)):
            total = 0.0
            square = 0.0
            worst = 0.0
            count = 0
            for begin in range(0, side - 1, tile):
                stop = min(begin + tile, side)
                # Overlap by one coarse cell so interpolation has support at the
                # tile edge; the overlap is scored once.
                lo_c = max(begin // factor - 2, 0)
                hi_c = min(-(-stop // factor) + 3, coarse.shape[0])
                block = classical(coarse[lo_c:hi_c, :], factor, method)
                if projected:
                    block = back_project(block, coarse[lo_c:hi_c, :], factor, iterations=3)
                offset = lo_c * factor
                take_lo = max(begin, margin) - offset
                take_hi = min(stop, side - margin) - offset
                if take_hi <= take_lo:
                    continue
                estimate = block[take_lo:take_hi, margin:side - margin]
                truth = np.asarray(
                    fine[offset + take_lo:offset + take_hi, margin:side - margin],
                    dtype=np.float64)
                if estimate.shape != truth.shape:
                    raise ValueError(
                        f"tile shapes disagree: estimate {estimate.shape} truth {truth.shape}")
                error = np.abs(estimate - truth)
                total += float(error.sum())
                square += float((error ** 2).sum())
                worst = max(worst, float(error.max()))
                count += error.size
                del block, estimate, truth, error
            name = method + ("+backprojection" if projected else "")
            rows[name] = {"maeM": total / max(count, 1),
                          "rmseM": float(np.sqrt(square / max(count, 1))),
                          "maxM": worst, "samples": count}
    return {"schema": BASELINE_SCHEMA, "factor": factor, "operator": OPERATOR,
            "byMethod": rows,
            "qualification": "Interior only: the operator's edge cells average a half cell and "
                             "mean something different from the interior's, so scoring them would "
                             "charge every method for an edge convention. Errors are accumulated "
                             "exactly over the whole interior rather than averaged per tile."}
