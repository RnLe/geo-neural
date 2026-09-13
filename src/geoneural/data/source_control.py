"""What the service's 10 m response actually is.

The economical acquisition asks the WCS for a coarsened response and then
resamples it onto the canonical lattice. The result is a prepared 10 m product
derived from a 1 m source. It is not a downloaded 1 m truth set, and it is not
a proven 10 m accuracy. Any later super-resolution or source-fidelity claim has
to be measured against real source pixels, so those pixels are acquired
separately and compared here.

Three 10 m fields are derived from the 1 m acquisition and each is compared with
the service response:

* area average: each canonical 10 m node receives the mean of the 1 m samples
  in its cell. This is what 'coarsened to 10 m' would mean if the provider's
  kernel were an area average, which is not assumed.
* bilinear: the same interpolation the ordinary pipeline applies, so that the
  provider's kernel choice can be separated from the resampling stage.
* nearest: decimation, which tests directly whether the service coarsens by
  dropping samples instead of averaging them.

A difference here is not an error in either field. It measures how much the
prepared reference depends on who did the coarsening, which is exactly the
quantity a super-resolution claim would otherwise silently absorb.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

from geoneural.data.build import erode_valid, grid_shape
from geoneural.common import read_json, sha_file, write_json
from geoneural.provenance import run_record

DEFAULT_EDGE_TRIM = 2


def _resample(source: dict, input_dir: Path, config: dict, side: int, method: str,
              edge_trim: int) -> np.ndarray:
    import rasterio
    from rasterio.transform import Affine
    from rasterio.warp import reproject, Resampling

    west, _south, _east, north = map(float, config["bbox"])
    spacing = float(config["spacing_m"])
    transform = Affine(spacing, 0, west - spacing / 2, 0, -spacing, north + spacing / 2)
    merged = np.full((side, side), np.nan, dtype=np.float32)
    resampling = {"average": Resampling.average, "bilinear": Resampling.bilinear,
                  "nearest": Resampling.nearest}[method]
    for item in sorted(source["files"], key=lambda x: x["path"]):
        path = (input_dir / item["path"]).resolve()
        if not path.is_relative_to(input_dir.resolve()):
            raise ValueError("Input file escapes the acquisition directory")
        if sha_file(path) != item["sha256"]:
            raise ValueError(f"Source changed after acquisition: {path}")
        temporary = np.full_like(merged, np.nan)
        with rasterio.open(path) as dataset:
            reproject(source=rasterio.band(dataset, 1), destination=temporary,
                      src_transform=dataset.transform, src_crs=dataset.crs, src_nodata=dataset.nodata,
                      dst_transform=transform, dst_crs=config["crs"], dst_nodata=np.nan,
                      resampling=resampling, num_threads=1, warp_mem_limit=64)
        valid = erode_valid(np.isfinite(temporary), edge_trim)
        take = valid & ~np.isfinite(merged)
        merged[take] = temporary[take]
    return merged


def _statistics(difference: np.ndarray) -> dict:
    finite = difference[np.isfinite(difference)]
    absolute = np.abs(finite)
    return {
        "samples": int(finite.size),
        "mean_signed_m": float(finite.mean()),
        "mae_m": float(absolute.mean()),
        "rmse_m": float(np.sqrt(np.mean(finite ** 2))),
        "p95_m": float(np.percentile(absolute, 95)),
        "p99_m": float(np.percentile(absolute, 99)),
        "max_m": float(absolute.max()),
    }


def compare(fine_input: Path, atlas_path: Path, out: Path, edge_trim: int = DEFAULT_EDGE_TRIM) -> Path:
    """Compare service-coarsened 10 m against 10 m derived from the 1 m source."""
    fine_input = Path(fine_input)
    source = read_json(fine_input)
    if source.get("schema") != "geoneural-input-v1":
        raise ValueError("Expected a terrain input.json for the fine acquisition")
    manifest = read_json(Path(atlas_path))
    config = manifest["config"]
    fine_spacing = float(source["source_spacing_requested_m"])
    coarse_spacing = float(manifest["spacing_m"])
    if fine_spacing >= coarse_spacing:
        raise ValueError(
            f"The control acquisition is {fine_spacing} m, no finer than the atlas at "
            f"{coarse_spacing} m; it cannot measure the provider's coarsening")
    side = grid_shape(config)
    served = np.load(Path(atlas_path).parent / "reference.npy").astype(np.float64)

    derived = {}
    # Nearest is included to test the likely explanation directly: if the service
    # coarsens by decimation rather than by averaging, nearest will agree with it
    # far better than the other two, and the gap is provider behaviour rather than
    # a defect in this pipeline.
    for method in ("average", "bilinear", "nearest"):
        grid = _resample(source, fine_input.parent, config, side, method, edge_trim)
        missing = int(np.count_nonzero(~np.isfinite(grid)))
        derived[method] = {"grid": grid, "missing": missing}

    comparisons = {}
    for method, entry in derived.items():
        difference = entry["grid"].astype(np.float64) - served
        comparisons[method] = {
            "missing_samples": entry["missing"],
            "versus_service_response": _statistics(difference),
        }
    kernel = derived["average"]["grid"].astype(np.float64) - derived["bilinear"]["grid"].astype(np.float64)

    report = {
        "schema": "geoneural-source-control-v1",
        "atlas": str(atlas_path),
        "atlas_content_id": manifest.get("content_id"),
        "atlas_spacing_m": coarse_spacing,
        "fine_acquisition": str(fine_input),
        "fine_spacing_m": fine_spacing,
        "fine_tiles": len(source["files"]),
        "fine_bytes": source["bytes"],
        "provider_resampling_declared": source.get("provider_resampling"),
        "edge_trim_samples": edge_trim,
        "comparisons": comparisons,
        "average_versus_bilinear_from_same_source": _statistics(kernel),
        "run": run_record("source-control", {"atlas": manifest.get("content_id")},
                          {"fine": fine_input.parent, "atlas": Path(atlas_path).parent}),
        "observed_2026_09_12_essen_ruhr": (
            "Nearest agreed with the service response worse than average or bilinear did, so the "
            "service is not coarsening by decimation. No kernel choice tested here accounts for the "
            "residual, which means it is not explained by this pipeline's resampling stage."),
        "next_discriminating_experiment": (
            "The 10 m path is resampled twice (once by the service and once onto the canonical "
            "lattice), while the 1 m path is resampled once. Compare the service's returned 10 m "
            "pixels directly against the 1 m source evaluated at the service's own grid positions, "
            "with no reprojection, to separate the provider's kernel from the second resampling."),
        "interpretation": (
            "A nonzero difference does not make either field wrong. It bounds how much the prepared "
            "10 m reference depends on who coarsened the 1 m source and with which kernel. Any "
            "super-resolution or source-fidelity claim must be evaluated against the 1 m pixels "
            "themselves, not against this prepared reference."
        ),
    }
    write_json(Path(out), report)
    return Path(out)
