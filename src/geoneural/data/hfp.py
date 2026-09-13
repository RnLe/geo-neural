"""The atlas against NRW's official height benchmarks (Höhenfestpunkte).

A landmark check needs known projected points with independently surveyed
heights. NRW publishes its height benchmarks as open data from the official AFIS
register: ETRS89/UTM32 positions (zone-prefixed eastings) and DHHN2016 heights,
surveyed independently of the DGM1 laser survey.

Most benchmarks are bolts on walls, bridges and churches, so each sits an unknown
height above the ground the terrain model describes. That fixes what can be
tested, and neither test asks the terrain to hit a bolt:

* Registration. A constant bolt-above-ground offset does not change how the
  residuals spread; a horizontal misregistration correlates them with the local
  terrain slope. The shift is estimated by that regression, and its 95 %
  interval must exclude half a lattice cell. A north-south flip and a
  transpose are measured as well: if they did not worsen the spread, the test
  would have no power.
* Datum. An ellipsoidal/quasigeoid confusion would move every residual by
  the height anomaly, tens of metres in Germany. The median residual is
  reported as observed; it is not fitted to an expectation.

The provider warns that benchmark heights come from different survey epochs and
methods, and the Ruhr is a mining region with real ground movement between
epochs. Individual residuals therefore mix bolt placement, survey epoch and
terrain-model error; only the registration result is free of all three.
"""
from __future__ import annotations

import csv
import hashlib
import statistics
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

from geoneural.common import read_json

UTM32_PREFIX = 32_000_000
SOURCE = "https://www.opengeodata.nrw.de/produkte/geobasis/rb/fd/"


def _number(text: str) -> float | None:
    text = text.strip()
    return float(text.replace(",", ".")) if text else None


def parse(path: Path) -> list[dict]:
    """Rows of the HFP point list: id; name; east; north; height; gravity.

    Eastings carry the UTM zone as a prefix (32xxxxxx). Rows without a height
    or with a foreign zone are dropped and counted, never guessed.
    """
    points, dropped = [], 0
    with open(path, encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle, delimiter=";", quotechar='"'):
            if len(row) < 5:
                dropped += 1
                continue
            east, north, height = (_number(v) for v in row[2:5])
            if east is None or north is None or height is None or not 32e6 <= east < 33e6:
                dropped += 1
                continue
            points.append({"id": row[0], "name": row[1], "east": east - UTM32_PREFIX,
                           "north": north, "height": height})
    return points


def latest_epochs(path: Path) -> dict[str, date]:
    """The most recent survey date of each benchmark (Punktliste-Zeitfolge)."""
    latest: dict[str, date] = {}
    with open(path, encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle, delimiter=";", quotechar='"'):
            if len(row) < 3:
                continue
            try:
                day, month, year = (int(v) for v in row[1].split("."))
                when = date(year, month, day)
            except ValueError:
                continue
            if row[0] not in latest or when > latest[row[0]]:
                latest[row[0]] = when
    return latest


def sample(grid: np.ndarray, manifest: dict, east: np.ndarray, north: np.ndarray) -> np.ndarray:
    """Bilinear heights at projected positions; NaN outside the lattice.

    Nodes are sample centres, row 0 is north (manifest `bounds` are centres)."""
    west, _, _, top = manifest["bounds"]
    spacing = manifest["spacing_m"]
    side = grid.shape[0]
    col = (np.asarray(east, dtype=np.float64) - west) / spacing
    row = (top - np.asarray(north, dtype=np.float64)) / spacing
    inside = (col >= 0) & (row >= 0) & (col <= side - 1) & (row <= side - 1)
    c0 = np.clip(np.floor(col).astype(int), 0, side - 2)
    r0 = np.clip(np.floor(row).astype(int), 0, side - 2)
    fc, fr = col - c0, row - r0
    value = (grid[r0, c0] * (1 - fc) * (1 - fr) + grid[r0, c0 + 1] * fc * (1 - fr)
             + grid[r0 + 1, c0] * (1 - fc) * fr + grid[r0 + 1, c0 + 1] * fc * fr)
    return np.where(inside, value, np.nan)


def spread(residuals: np.ndarray) -> float:
    """Median absolute deviation: robust to the bolts on bridges and walls."""
    finite = residuals[np.isfinite(residuals)]
    return float(np.median(np.abs(finite - np.median(finite))))


def registration(grid: np.ndarray, manifest: dict, east, north, height,
                 boots: int = 2000, seed: int = 20260923) -> dict:
    """Where the benchmarks sit relative to the atlas, with a confidence interval.

    Near zero shift the residual spread is flat (bolt placement and survey
    epoch are noise at the metre scale), so the location of its minimum is not
    a sound estimate: a grid scan can put it at 6 m on a 2 % dip. A benchmark
    reported at x but truly at x + d gives residual r = c + grad(z)·d, so d is
    the regression of residuals on the local terrain gradient, over the robust
    core (within 3 robust sigmas of the median: bridges and walls excluded),
    with a bootstrap interval over benchmarks.
    """
    residual = height - sample(grid, manifest, east, north)
    rows_grad, east_grad = np.gradient(grid, manifest["spacing_m"])
    gx = sample(east_grad, manifest, east, north)
    gy = -sample(rows_grad, manifest, east, north)  # rows run south
    median, sigma = np.nanmedian(residual), 1.4826 * spread(residual)
    core = np.flatnonzero(np.isfinite(residual) & np.isfinite(gx) & np.isfinite(gy)
                          & (np.abs(residual - median) <= 3 * sigma))

    def fit(index):
        design = np.column_stack([np.ones(len(index)), gx[index], gy[index]])
        return np.linalg.lstsq(design, residual[index], rcond=None)[0]

    offset, dx, dy = fit(core)
    rng = np.random.default_rng(seed)
    samples = np.array([fit(rng.choice(core, size=len(core), replace=True))[1:] for _ in range(boots)])
    low, high = np.percentile(samples, [2.5, 97.5], axis=0)
    return {"shiftM": [float(dx), float(dy)], "offsetM": float(offset),
            "shift95M": {"east": [float(low[0]), float(high[0])], "north": [float(low[1]), float(high[1])]},
            "benchmarksInCore": int(len(core)), "bootstrap": {"samples": boots, "seed": seed},
            "medianSlope": float(np.median(np.hypot(gx[core], gy[core])))}


def run(atlas_path: Path, hfp_dir: Path) -> dict:
    atlas_path, hfp_dir = Path(atlas_path), Path(hfp_dir)
    manifest = read_json(atlas_path)
    grid = np.load(atlas_path.parent / "reference.npy").astype(np.float64)
    west, south, east_edge, north_edge = manifest["bounds"]
    everything = parse(hfp_dir / "hfp_pl.csv")
    points = [p for p in everything
              if west <= p["east"] <= east_edge and south <= p["north"] <= north_edge]
    if len(points) < 20:
        raise ValueError(f"only {len(points)} benchmarks inside the atlas; too few to test registration")
    epochs = latest_epochs(hfp_dir / "hfp_plzf.csv") if (hfp_dir / "hfp_plzf.csv").exists() else {}
    e = np.array([p["east"] for p in points])
    n = np.array([p["north"] for p in points])
    h = np.array([p["height"] for p in points])
    residual = h - sample(grid, manifest, e, n)
    q = np.nanpercentile(residual, [5, 25, 50, 75, 95])
    found = registration(grid, manifest, e, n, h)
    at_zero = spread(residual)
    flipped = spread(h - sample(grid[::-1, :], manifest, e, n))
    transposed = spread(h - sample(grid.T, manifest, e, n))
    # Half a cell is the smallest convention error the lattice could make
    # (sample centre against corner); the interval must exclude it.
    half_cell = manifest["spacing_m"] / 2
    registered = all(abs(v) < half_cell for pair in found["shift95M"].values() for v in pair)
    powered = bool(min(flipped, transposed) > at_zero)
    order = np.argsort(-np.abs(residual - q[2]))
    dated = [epochs[p["id"]] for p in points if p["id"] in epochs]
    files = {name: {"sha256": hashlib.sha256((hfp_dir / name).read_bytes()).hexdigest(),
                    "bytes": (hfp_dir / name).stat().st_size,
                    "retrievedUtc": datetime.fromtimestamp((hfp_dir / name).stat().st_mtime,
                                                           timezone.utc).isoformat(timespec="seconds")}
             for name in ("hfp_pl.csv", "hfp_plzf.csv", "Datenformatbeschreibung.pdf", "Nutzerinformation.pdf")
             if (hfp_dir / name).exists()}
    return {
        "schema": "geoneural-hfp-check-v1",
        "atlasContentId": manifest["content_id"],
        "source": {"url": SOURCE, "files": files,
                   "coordinates": "zone-prefixed UTM32 eastings; the format description labels them "
                                  "'DHDN2016', which names no datum; ETRS89/UTM32N is what this test verifies",
                   "heights": "DHHN2016 (EPSG:7837), the atlas's own vertical reference"},
        "benchmarksInNrw": len(everything), "benchmarksInAtlas": len(points),
        "surveyEpochs": {"dated": len(dated),
                         "median": statistics.median(dated).isoformat() if dated else None,
                         "oldest": min(dated).isoformat() if dated else None,
                         "newest": max(dated).isoformat() if dated else None},
        "residualPublishedMinusAtlasM": {"p5": q[0], "p25": q[1], "median": q[2], "p75": q[3], "p95": q[4],
                                         "mad": spread(residual),
                                         "withinOneMetreOfMedian": float(np.mean(np.abs(residual - q[2]) < 1.0))},
        "registration": {**found, "halfCellM": half_cell, "registered": registered, "spreadAtZeroM": at_zero,
                         "spreadIfFlippedNorthSouthM": flipped, "spreadIfTransposedM": transposed,
                         "testHasPower": powered},
        "largestResiduals": [{"id": points[i]["id"], "name": points[i]["name"],
                              "residualM": float(residual[i])} for i in order[:10]],
        # Descriptive, not a tolerance: a datum mix-up would offset the median by tens of metres.
        "verdict": {"registrationVerified": registered and powered,
                    "medianOffsetBelowOneSpread": bool(abs(q[2]) < spread(residual))},
        "scope": "Registration and datum only. Benchmarks are mostly bolts on structures, surveyed at "
                 "different epochs in a mining region, so residuals are not a ground-accuracy measure.",
    }
