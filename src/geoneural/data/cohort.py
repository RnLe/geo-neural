"""The confirmation cohort: new NRW regions chosen by a rule fixed before any of them is downloaded.

The rule (`RULE`) is part of the committed protocol. Candidates are 10.24 km squares on an 11 km lattice over the
NRW extent, so neighbouring candidates never share samples. Every candidate closer than `BUFFER_M` to a development
region (or to the two older presets that overlap them) is excluded. The rest are ranked by the SHA-256 of
`SALT` and the tile origin, which nobody can steer. In rank order, each candidate gets one coarse probe (400 m
samples from the provider's WCS): candidates with any missing value are skipped (outside NRW or across a border),
and the coarse height standard deviation assigns a relief stratum with cut points taken from the development
regions. The first candidates to fill each stratum's quota are the cohort.

The probes are the only look at these regions before the frozen recipes run. Every probed candidate, kept or
skipped, is written to the exposure log with what was seen (coarse statistics only, no codec or model output).
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from geoneural.common import COHORT_CONFIG, CONFIG, read_json, utc, write_json

RULE = {
    "version": 1,
    "salt": "geoneural confirmation cohort 2026-10-04",
    "extentEpsg25832": [280000.0, 5560000.0, 540000.0, 5830000.0],
    "latticeM": 11000.0,
    "tileM": 10240.0,
    "originOffsetM": 0.5,
    "bufferM": 15000.0,
    "excludeNearPresets": ["essen-ruhr", "ruhr-wide", "drachenfels", "muensterland-plain", "lower-rhine",
                           "teutoburg-forest", "bergisches-land", "rothaar-sauerland"],
    "probeSpacingM": 400.0,
    "strataStdM": {"flat": [0.0, 8.0], "moderate": [8.0, 30.0], "rough": [30.0, 1e9]},
    "quota": {"flat": 2, "moderate": 3, "rough": 2},
    "requireNoMissing": True,
    "note": ("Cut points from the development regions' 10 m height standard deviation: 2.7 m (Muensterland), "
             "12.2 (Lower Rhine), 21.4 (Teutoburg), 28.2 (Essen), 36.9 (Bergisches Land), 88.6 (Rothaar)."),
}


# Cohort B, for a recipe designed after cohort A was used: the same ranking, a buffer around the cohort A tiles too,
# and every candidate already probed for cohort A skipped, so no tile has been looked at before its frozen run.
RULE_B = {
    **RULE,
    "version": 2,
    "excludeNearPresets": RULE["excludeNearPresets"] + ["nrw-368-5747", "nrw-390-5780", "nrw-324-5648", "nrw-313-5659", "nrw-401-5791", "nrw-445-5703", "nrw-412-5769"],
    "excludeProbed": ["nrw-335-5802", "nrw-368-5747", "nrw-456-5604", "nrw-280-5736", "nrw-390-5780", "nrw-489-5626", "nrw-324-5648", "nrw-313-5659", "nrw-280-5604", "nrw-401-5791", "nrw-511-5791", "nrw-522-5769", "nrw-478-5593", "nrw-280-5637", "nrw-445-5703", "nrw-522-5571", "nrw-324-5791", "nrw-500-5780", "nrw-500-5615", "nrw-401-5593", "nrw-489-5571", "nrw-522-5560", "nrw-302-5648", "nrw-478-5714", "nrw-324-5747", "nrw-368-5560", "nrw-291-5692", "nrw-324-5758", "nrw-478-5571", "nrw-412-5769"],
    "note": RULE["note"] + " Cohort B: cohort A tiles buffered like the development regions; cohort A probes skipped.",
}


def _gap(a, b) -> float:
    """Distance between two axis-aligned boxes (west, south, east, north); 0 when they overlap."""
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return math.hypot(dx, dy)


def candidates(rule: dict = RULE) -> list[dict]:
    presets = read_json(CONFIG)
    if COHORT_CONFIG.exists():
        presets.update(read_json(COHORT_CONFIG))
    near = [presets[name]["bbox"] for name in rule["excludeNearPresets"]]
    skip = set(rule.get("excludeProbed", ()))
    x0, y0, x1, y1 = rule["extentEpsg25832"]
    step, tile, off = rule["latticeM"], rule["tileM"], rule["originOffsetM"]
    out = []
    for i in range(int((x1 - x0 - tile) // step) + 1):
        for j in range(int((y1 - y0 - tile) // step) + 1):
            west, south = x0 + i * step + off, y0 + j * step + off
            box = [west, south, west + tile, south + tile]
            if min(_gap(box, b) for b in near) < rule["bufferM"]:
                continue
            name = f"nrw-{int(west) // 1000:03d}-{int(south) // 1000:04d}"
            if name in skip:
                continue
            rank = hashlib.sha256(f"{rule['salt']}:{int(west)}:{int(south)}".encode()).hexdigest()
            out.append({"name": name, "bbox": box, "rank": rank})
    return sorted(out, key=lambda c: c["rank"])


def probe(candidate: dict, out: Path, rule: dict = RULE) -> dict:
    """One coarse WCS request for the whole tile; returns missing fraction and coarse relief statistics."""
    import rasterio

    from geoneural.data import acquire, wcs as wcs_schema
    west, south, east, north = candidate["bbox"]
    out.mkdir(parents=True, exist_ok=True)
    factor = 1.0 / rule["probeSpacingM"]
    params = acquire.coverage_params("nw_dgm", "x", "y", (west, east, south, north),
                                     wcs_schema.scaling_forms("x", "y", factor)[0])
    path = out / f"{candidate['name']}-probe.tif"
    acquire.download(acquire.url(acquire.WCS, params), path, 8 * 1024 * 1024, "tiff")
    with rasterio.open(path) as ds:
        a = ds.read(1).astype(np.float64)
        nodata = ds.nodata
    missing = ~np.isfinite(a) | (a < -1000)
    if nodata is not None:
        missing |= a == nodata
    valid = a[~missing]
    return {"missingFraction": float(missing.mean()), "samples": int(a.size),
            "stdM": float(valid.std()) if valid.size else None, "reliefM": float(np.ptp(valid)) if valid.size else None}


def stratum(std_m: float, rule: dict = RULE) -> str:
    for name, (lo, hi) in rule["strataStdM"].items():
        if lo <= std_m < hi:
            return name
    raise ValueError(std_m)


def select(out: Path, rule: dict = RULE, max_probes: int = 120) -> dict:
    """Walk the ranked candidates, probing each, until every stratum quota is filled. Writes the exposure log."""
    quota = dict(rule["quota"])
    chosen, log = [], []
    for cand in candidates(rule):
        if not any(quota.values()) or len(log) >= max_probes:
            break
        try:
            p = probe(cand, out / "probes", rule)
        except Exception as exc:  # noqa: BLE001 - a failed probe is recorded and the candidate skipped
            log.append({**cand, "outcome": "probe failed", "error": str(exc)[:300]})
            continue
        entry = {**cand, **p}
        if rule["requireNoMissing"] and p["missingFraction"] > 0:
            entry["outcome"] = "skipped: missing values"
        else:
            s = stratum(p["stdM"], rule)
            entry["stratum"] = s
            if quota[s] > 0:
                quota[s] -= 1
                entry["outcome"] = "selected"
                chosen.append(entry)
            else:
                entry["outcome"] = "skipped: stratum full"
        log.append(entry)
        print(f"{cand['name']}: {entry['outcome']}", flush=True)
    result = {"schema": "geoneural-cohort-v1", "rule": rule, "createdUtc": utc(), "selected": chosen,
              "exposureLog": log, "quotaLeft": quota}
    write_json(out / "cohort.json", result)
    return result


def presets(result: dict) -> dict:
    """Region presets for the selected tiles, in the format of configs/regions.json."""
    return {c["name"]: {"title": f"Confirmation tile {c['name']} ({c['stratum']}), NRW DGM1", "bbox": c["bbox"],
                        "crs": "EPSG:25832", "vertical_crs": "EPSG:7837", "spacing_m": 10.0, "page_intervals": 64,
                        "download_tile_m": 2560, "source_spacing_m": 10, "quantum_m": 0.01,
                        "notes": "Confirmation cohort; chosen by data/cohort.py before download.",
                        "cohort": {"stratum": c["stratum"], "probeStdM": c["stdM"], "rank": c["rank"]}}
            for c in result["selected"]}
