"""A dense conventional envelope: every pyramid level at many error targets.

The neural comparison sweeps the conventional side at a handful of targets. Between those
samples a neural row can look undominated only because no conventional point was measured
near it. This sweep fills the gaps on both conventional axes (grid spacing and quantisation
step) with the same product definition the neural rows are charged against: the grid in
independently compressed 65 x 65 pages, q32-delta-zstd, reconstructed bilinearly onto the
10 m lattice. It also adds the trivial control, a stored field mean.

Levels come from the atlas pyramid (binomial low-pass, then decimation by two), so a coarse
point is the same product the atlas already declares for its parent levels. Levels 5 and 6
(320 m and 640 m grids, one page each) continue the same filter below the pyramid, so the
sweep reaches the few-hundred-byte budgets of the smallest networks.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from geoneural.codecs import bounds, codecs
from geoneural.common import read_json
from geoneural.data.build import expand_bilinear, smooth_decimate

SCHEMA = "geoneural-dense-envelope-v1"
TARGETS = (0.01, 0.02, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0,
           12.0, 16.0)
LEVELS = (0, 1, 2, 3, 4, 5, 6)


def paged_bytes(codec, grid: np.ndarray, target: float, intervals: int) -> int:
    """Bytes of the grid as independently compressed pages of `intervals` cells."""
    span = (grid.shape[0] - 1) // intervals
    if span < 1:
        return len(codec.encode(grid, target))
    return sum(len(codec.encode(grid[y * intervals:y * intervals + intervals + 1,
                                     x * intervals:x * intervals + intervals + 1], target))
               for y in range(span) for x in range(span))


def errors(decoded: np.ndarray, reference: np.ndarray) -> dict:
    e = np.abs(decoded.astype(np.float64) - reference)
    return {"maeM": float(e.mean()), "rmseM": float(np.sqrt((e ** 2).mean())),
            "p99M": float(np.quantile(e, 0.99)), "maxM": float(e.max())}


def decoded_field(atlas_path: Path, level: int, target: float, codec_name: str = "q32-delta-zstd",
                  manifest: dict | None = None) -> tuple[np.ndarray, int]:
    """The reconstruction on the 10 m lattice and its paged byte count."""
    atlas_path = Path(atlas_path)
    manifest = manifest or read_json(atlas_path)
    codec = codecs.registry()[codec_name]
    top = int(manifest["max_level"])
    if level == 0:
        grid = np.load(atlas_path.parent / "reference.npy").astype(np.float64)
    else:
        grid = bounds.level_grid(atlas_path.parent, manifest, min(level, top))
        # Below the pyramid's coarsest level, continue with the same filter and decimation.
        for _ in range(level - top):
            grid = smooth_decimate(grid)
    grid = grid.astype(np.float32)
    size = paged_bytes(codec, grid, target, int(manifest["page_intervals"]))
    decoded = codec.decode(codec.encode(grid, target)).astype(np.float64)
    return expand_bilinear(decoded, 2 ** level), size


def sweep(atlas_path: Path, levels=LEVELS, targets=TARGETS, streams: bool = True,
          stream_cells: int = 500) -> dict:
    from geoneural.metrics import hydrology
    atlas_path = Path(atlas_path)
    manifest = read_json(atlas_path)
    reference = np.load(atlas_path.parent / "reference.npy").astype(np.float64)
    spacing = float(manifest["spacing_m"])
    reference_analysis = hydrology.analyse(reference, spacing, stream_cells) if streams else None
    points = []
    for level in levels:
        for target in targets:
            field, size = decoded_field(atlas_path, level, target, manifest=manifest)
            point = {"level": level, "spacingM": spacing * 2 ** level, "targetM": target, "bytes": size,
                     "boundGuaranteed": level == 0, **errors(field, reference)}
            if streams:
                point["streamJaccard"] = hydrology.compare(
                    reference, field, spacing, stream_cells, reference_analysis=reference_analysis
                )["streamJaccard"]
            points.append(point)
            print(f"level {level} target {target:g} m: {size} B, MAE {point['maeM']:.3f} m", flush=True)
    mean = float(reference.mean())
    constant = {"bytes": 4, **errors(np.full_like(reference, mean), reference),
                "note": "A single float32 field mean. Anything above this line has learned something."}
    return {"schema": SCHEMA, "atlas": atlas_path.parent.name, "atlasContentId": manifest["content_id"],
            "codec": "q32-delta-zstd", "pageIntervals": int(manifest["page_intervals"]),
            "streamThresholdCells": stream_cells, "points": points, "constantMean": constant,
            "method": "pyramid level grid, q32-delta-zstd at the target, independent 65 x 65 pages, "
                      "bilinear reconstruction onto the 10 m lattice; errors on all reference nodes",
            "qualification": "Only level 0 carries a maximum-error guarantee against the reference. A coarse "
                             "level's target bounds error against its own grid, not against the 10 m terrain."}
