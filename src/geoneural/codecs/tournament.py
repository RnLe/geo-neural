"""The conventional codec tournament on a frozen atlas.

Every codec encodes the identical multiscale page decomposition that the shipped
atlas uses, at each declared maximum-error target, so the resulting table is a
rate-distortion comparison rather than a collection of incomparable operating
points. Achieved error is measured against the prepared reference and checked
against the target the codec was configured for.

What this does not establish: whole-renderer performance, random-access IO cost
under contention, or source accuracy against the ground. Encode and decode
times here are single-process diagnostics taken in one pass, not an isolated
timing cohort.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

from geoneural.codecs.accounting import package_bytes
from geoneural.data.build import smooth_decimate
from geoneural.codecs.codecs import measure, registry
from geoneural.common import read_json, write_json
from geoneural.provenance import run_record

DEFAULT_TARGETS = (0.01, 0.05, 0.1, 0.5, 1.0)


def pyramid(reference: np.ndarray, intervals: int, levels: int) -> list[tuple[int, int, int, np.ndarray]]:
    """The same level/page decomposition `build.pack` publishes."""
    pages = []
    grid = reference
    for level in range(levels + 1):
        count = (grid.shape[0] - 1) // intervals
        for y in range(count):
            for x in range(count):
                patch = grid[y * intervals:y * intervals + intervals + 1,
                             x * intervals:x * intervals + intervals + 1]
                pages.append((level, x, y, np.ascontiguousarray(patch)))
        grid = smooth_decimate(grid)
    return pages


def run(atlas_path: Path, out: Path, targets: tuple[float, ...] = DEFAULT_TARGETS,
        page_limit: int | None = None, only: list[str] | None = None) -> Path:
    atlas_path = Path(atlas_path)
    manifest = read_json(atlas_path)
    if manifest.get("schema") != "geoneural-atlas-v1":
        raise ValueError("Not a geoneural-atlas-v1 manifest")
    directory = atlas_path.parent
    reference = np.load(directory / "reference.npy")
    pages = pyramid(reference, int(manifest["page_intervals"]), int(manifest["max_level"]))
    if page_limit:
        # A bounded pilot. Recorded explicitly so a partial run is never read as full coverage.
        pages = pages[:page_limit]
    samples = sum(patch.size for _, _, _, patch in pages)

    accounted = package_bytes(directory)
    index_bytes = accounted["deployment"]["metadata_and_index"]

    codecs = registry()
    results = []
    for name, codec in codecs.items():
        if only and name not in only:
            continue
        if not codec.available:
            results.append({"codec": name, "available": False, "reason": codec.unavailable_reason,
                            "note": codec.note, "observations": []})
            continue
        observations = []
        # A lossless codec has no error target to sweep; running it five times
        # would fabricate five identical rows.
        sweep = targets if codec.error_bounded else targets[:1]
        for target in sweep:
            total_bytes = 0
            encode_ms = decode_ms = 0.0
            worst = 0.0
            squared = 0.0
            absolute = 0.0
            ulp = 0.0
            failures = []
            for level, x, y, patch in pages:
                try:
                    m = measure(codec, patch, target)
                except Exception as exc:  # noqa: BLE001 - a failing page is a result
                    failures.append({"page": f"{level}/{x}/{y}", "error": f"{type(exc).__name__}: {exc}"})
                    continue
                total_bytes += m["bytes"]
                encode_ms += m["encode_ms"]
                decode_ms += m["decode_ms"]
                worst = max(worst, m["max_error_m"])
                # Decoded pages are float32. At these elevations one ulp is a few
                # microns, so a bound is 'met' only up to the precision the output
                # format can represent; charging that to the codec would report a
                # false violation.
                ulp = max(ulp, float(np.spacing(np.float32(np.abs(patch).max()))))
                squared += (m["rmse_m"] ** 2) * patch.size
                absolute += m["mae_m"] * patch.size
            if failures and len(failures) == len(pages):
                observations.append({"target_m": target, "failed": True, "failures": failures[:4]})
                continue
            observations.append({
                "target_m": target if codec.error_bounded else None,
                "payload_bytes": total_bytes,
                "index_bytes": index_bytes,
                "total_deployment_bytes": total_bytes + index_bytes,
                "bits_per_sample": 8.0 * (total_bytes + index_bytes) / samples,
                "payload_bits_per_sample": 8.0 * total_bytes / samples,
                "max_error_m": worst,
                "rmse_m": float(np.sqrt(squared / samples)),
                "mae_m": float(absolute / samples),
                "meets_target": (worst <= target + ulp) if codec.error_bounded else None,
                "float32_representation_ulp_m": ulp,
                "excess_over_target_m": (worst - target) if codec.error_bounded else None,
                "encode_ms_total": encode_ms,
                "decode_ms_total": decode_ms,
                "page_failures": failures[:4],
            })
        results.append({"codec": name, "available": True, "error_bounded": codec.error_bounded,
                        "random_access": codec.random_access, "families": list(codec.families),
                        "note": codec.note, "observations": observations})

    report = {
        "schema": "geoneural-codec-tournament-v1",
        "atlas_content_id": manifest.get("content_id"),
        "atlas": str(atlas_path),
        "pages_encoded": len(pages),
        "pages_in_atlas": len(manifest["pages"]),
        "partial_pilot": bool(page_limit),
        "samples_across_levels": samples,
        "targets_m": list(targets),
        "index_bytes_shared_by_every_codec": index_bytes,
        "results": results,
        "run": run_record("codec-tournament", {"atlas": manifest.get("content_id")}, {"atlas": directory}),
        "qualification": (
            "Rate and error on the prepared reference only. Not whole-renderer performance, not "
            "random-access IO cost, not accuracy against the ground. Timings are one diagnostic pass "
            "in a shared process, not an isolated cohort, and the OS cache was not flushed. "
            "Bits per sample counts every level of the pyramid plus the shared index."
        ),
    }
    write_json(Path(out), report)
    return Path(out)
