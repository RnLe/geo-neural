"""A conservative certificate for the mixed-level seam, not another estimate.

The parent/child mismatch at a level transition needs an explicit interface with
conservative bounds. The atlas ships `sample_error_m`, labelled
"sampled-finest-grid-bilinear-relative; not continuous-ground or triangle
proof". Two points matter when relying on that value.

It is not sampled. `build.pack` evaluates the difference at every finest-grid
node the page covers, which is exhaustive. It is the exact supremum over the
continuous domain under the declared bilinear reconstruction: on any one finest
cell both surfaces are bilinear, so their difference is bilinear; a bilinear
function's interior critical point is a saddle (its Hessian is [[0,d],[d,0]]),
and along each edge it is linear, so its extrema lie at the cell's corners,
which are finest-grid nodes. Evaluating at nodes therefore misses nothing.

But the renderer does not draw the bilinear surface. It draws two triangles per
cell, which is a different operator, and that is what the disclaimer protects
against. The gap between them has a closed form. On a cell with
corner heights h00, h01, h10, h11 the bilinear centre value is their mean, while
the diagonal triangulation gives the mean of the two diagonal endpoints; the
difference is the twist term (h00 + h11 - h01 - h10) / 4. That is also the
maximum over the whole cell: on the triangle (0,0), (1,0), (1,1) the discrepancy
is exactly d * y * (x - 1), whose largest magnitude on 0 <= y <= x <= 1 is |d|/4,
attained at the cell centre.

So a conservative triangle-to-triangle bound follows from the triangle
inequality:

    |P_tri - C_tri| <= |P_tri - P_bil| + |P_bil - C_bil| + |C_bil - C_tri|
                    <= max|twist(parent)|/4 + node_sup + max|twist(child)|/4

Every term is computed exactly from stored data. Nothing here is sampled, and
`verify_bound` checks the result against densely evaluated random points so the
derivation is tested rather than trusted.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

from geoneural.codecs import eat1 as codec
from geoneural.data.build import expand_bilinear
from geoneural.common import read_json, utc, write_json
from geoneural.provenance import run_record

SCHEMA = "geoneural-seam-bound-v1"


def twist(grid: np.ndarray) -> np.ndarray:
    """Per-cell |h00 + h11 - h01 - h10|, the bilinear cross term."""
    return np.abs(grid[:-1, :-1] + grid[1:, 1:] - grid[:-1, 1:] - grid[1:, :-1])


def triangulation_slack(grid: np.ndarray) -> float:
    """Largest gap between this grid's bilinear and triangulated surfaces."""
    if grid.shape[0] < 2 or grid.shape[1] < 2:
        return 0.0
    return float(twist(grid).max()) / 4.0


def level_grid(atlas_dir: Path, manifest: dict, level: int) -> np.ndarray:
    intervals = manifest["page_intervals"]
    entries = [e for e in manifest["pages"].values() if e["level"] == level]
    span = max(e["x"] for e in entries) + 1
    grid = np.zeros((span * intervals + 1, span * intervals + 1))
    for entry in entries:
        values, _ = codec.decode((atlas_dir / entry["path"]).read_bytes(), entry["raw_bytes"])
        grid[entry["y"] * intervals:entry["y"] * intervals + intervals + 1,
             entry["x"] * intervals:entry["x"] * intervals + intervals + 1] = values
    return grid


def page_bound(coarse: np.ndarray, fine: np.ndarray) -> dict:
    """Certificate for one parent page against the child surface beneath it.

    `fine` must be the child-level nodes covering exactly the parent's extent,
    so it is (2n-1) x (2n-1) for an n x n parent.
    """
    expected = 2 * coarse.shape[0] - 1
    if fine.shape != (expected, expected):
        raise ValueError(f"child region must be {expected}x{expected} for this parent, got {fine.shape}")
    node_sup = float(np.abs(expand_bilinear(coarse, 2) - fine).max())
    coarse_slack = triangulation_slack(coarse)
    fine_slack = triangulation_slack(fine)
    return {
        "bilinearSupM": node_sup,
        "parentTriangulationSlackM": coarse_slack,
        "childTriangulationSlackM": fine_slack,
        "triangleBoundM": node_sup + coarse_slack + fine_slack,
        "basis": "exact supremum under the declared bilinear reconstruction, plus closed-form "
                 "triangulation slack for both surfaces; conservative for the rendered triangles",
    }


def _sample_triangulated(grid: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Evaluate the rendered surface: two triangles per cell, split along the
    diagonal from the cell's (row, col) origin to its opposite corner."""
    row = np.clip(np.floor(ys).astype(int), 0, grid.shape[0] - 2)
    col = np.clip(np.floor(xs).astype(int), 0, grid.shape[1] - 2)
    fy, fx = ys - row, xs - col
    h00 = grid[row, col]
    h01 = grid[row, col + 1]
    h10 = grid[row + 1, col]
    h11 = grid[row + 1, col + 1]
    # Lower triangle where fx >= fy, upper otherwise; both interpolate linearly.
    lower = h00 + (h01 - h00) * fx + (h11 - h01) * fy
    upper = h00 + (h11 - h10) * fx + (h10 - h00) * fy
    return np.where(fx >= fy, lower, upper)


def verify_bound(coarse: np.ndarray, fine: np.ndarray, certificate: dict,
                 samples: int = 200_000, seed: int = 20260912) -> dict:
    """Test the derivation instead of trusting it.

    Random continuous points are evaluated on both rendered surfaces and the
    observed discrepancy is compared against the certificate. A bound that the
    data can exceed is not a bound.

    `observedMaxM` is a maximum over a finite random sample: a stochastic lower
    bound on the true discrepancy, not a measurement of it. It moves with the
    sample count and the seed, so both are recorded alongside it and two runs at
    different counts are not comparable. Only `holds` carries a conclusion.
    """
    rng = np.random.default_rng(seed)
    n = coarse.shape[0]
    # Parent coordinates in [0, n-1]; the child grid is twice as dense.
    cy = rng.uniform(0.0, n - 1, samples)
    cx = rng.uniform(0.0, n - 1, samples)
    parent = _sample_triangulated(coarse, cy, cx)
    child = _sample_triangulated(fine, cy * 2.0, cx * 2.0)
    observed = float(np.abs(parent - child).max())
    bound = certificate["triangleBoundM"]
    return {
        "samples": samples, "seed": seed,
        "observedIsStochasticLowerBound": True,
        "observedMaxM": observed, "boundM": bound,
        "holds": observed <= bound + 1e-9,
        "headroomM": bound - observed,
        "tightnessPercent": round(100.0 * observed / bound, 1) if bound > 0 else None,
    }


def certify(atlas_path: Path, out: Path | None = None, verify_samples: int = 200_000) -> dict:
    atlas_path = Path(atlas_path)
    atlas_dir = atlas_path.parent
    manifest = read_json(atlas_path)
    intervals = manifest["page_intervals"]
    grids = {level: level_grid(atlas_dir, manifest, level)
             for level in range(manifest["max_level"] + 1)}

    pages, per_level = {}, {}
    for level in range(1, manifest["max_level"] + 1):
        coarse_grid, fine_grid = grids[level], grids[level - 1]
        worst = {"triangleBoundM": -1.0}
        for entry in manifest["pages"].values():
            if entry["level"] != level:
                continue
            x, y = entry["x"], entry["y"]
            coarse = coarse_grid[y * intervals:y * intervals + intervals + 1,
                                 x * intervals:x * intervals + intervals + 1]
            fine = fine_grid[2 * y * intervals:2 * y * intervals + 2 * intervals + 1,
                             2 * x * intervals:2 * x * intervals + 2 * intervals + 1]
            certificate = page_bound(coarse, fine)
            key = f"{level}/{x}/{y}"
            pages[key] = certificate
            declared = entry["sample_error_m"]
            certificate["declaredSampleErrorM"] = declared
            certificate["boundExceedsDeclared"] = certificate["triangleBoundM"] > declared
            if certificate["triangleBoundM"] > worst["triangleBoundM"]:
                worst = {**certificate, "page": key}
        per_level[str(level)] = worst

    # Verify the worst page at every transition, where a wrong derivation would
    # show first, rather than an average page where it might hide.
    verifications = {}
    for level, worst in per_level.items():
        x, y = (int(part) for part in worst["page"].split("/")[1:])
        coarse_grid, fine_grid = grids[int(level)], grids[int(level) - 1]
        coarse = coarse_grid[y * intervals:y * intervals + intervals + 1,
                             x * intervals:x * intervals + intervals + 1]
        fine = fine_grid[2 * y * intervals:2 * y * intervals + 2 * intervals + 1,
                         2 * x * intervals:2 * x * intervals + 2 * intervals + 1]
        verifications[level] = {"page": worst["page"],
                                **verify_bound(coarse, fine, worst, verify_samples)}

    declared_exceeded = sum(1 for c in pages.values() if c["boundExceedsDeclared"])
    report = {
        "schema": SCHEMA, "createdUtc": utc(),
        "atlas": str(atlas_path), "atlasContentId": manifest["content_id"],
        "quantumM": manifest["quantum_m"],
        "derivation": {
            "bilinearSupIsExact": "On any finest cell both surfaces are bilinear, so their difference "
                                  "is bilinear. A bilinear function's interior critical point is a "
                                  "saddle and it is linear along each edge, so its extrema are at the "
                                  "cell corners, which are nodes. Node evaluation is therefore the "
                                  "exact supremum, not a sample.",
            "triangulationSlack": "A cell's triangulated surface departs from its bilinear surface by "
                                  "at most |h00 + h11 - h01 - h10| / 4, attained at the cell centre.",
            "combination": "|P_tri - C_tri| <= parent slack + bilinear supremum + child slack, by the "
                           "triangle inequality. Conservative, never tight by construction.",
            "relativeTo": "Both stored surfaces after quantization, and nothing else. Adding "
                          "quantum/2 reaches the unquantized reference, because quantization error "
                          "is itself bounded by quantum/2. Nothing here reaches the ground: the "
                          "measured source-fidelity figures are an MAE and percentiles, which are "
                          "averages and order statistics rather than conservative bounds, so they "
                          "cannot be added to a supremum to produce one. A bound against the real "
                          "surface would need a conservative source-error certificate, and none exists.",
            "perTransitionScope": "Each bound compares one level against the level below it, and is "
                                  "not the error of the finest level against the source. Over a "
                                  "domain both transitions cover, bounds compose by the triangle "
                                  "inequality, so level 2 against level 0 is bounded by the sum of the "
                                  "2->1 and 1->0 bounds. A composed bound is looser at every step it "
                                  "passes through, and no composition gives a bound against the source.",
        },
        "perLevelWorst": per_level,
        "verification": verifications,
        "pagesCertified": len(pages),
        "pagesWhereBoundExceedsDeclaredSampleError": declared_exceeded,
        "qualification": "A geometric certificate between two stored surfaces. It bounds the rendered "
                         "mixed-level discrepancy; it says nothing about agreement with the real "
                         "ground, and it is not a performance or coverage claim.",
        "pages": pages,
    }
    report["run"] = run_record("seam-bound-certificate", {"atlas": str(atlas_path)},
                               {"atlas": atlas_dir})
    if out is not None:
        write_json(Path(out), report)
    return report
