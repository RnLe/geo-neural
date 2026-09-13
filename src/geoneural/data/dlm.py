"""The atlas inspected against NRW's landscape model (ATKIS Basis-DLM).

The Basis-DLM is surveyed and maintained apart from the laser terrain model: water
axes with their flow direction, water areas, bridges and published water-surface
heights. Fetched for the atlas extent only, through the official WFS
(DL-DE-Zero-2.0), it tests what the benchmark check cannot:

* Water levels. Published water-surface heights are landmark elevations of a
  known feature. Tolerance: DGM1's stated 0.2 m accuracy (provider tile metadata)
  plus half the 0.1 m rounding of the published value.
* Orientation. Water runs downhill. Along every axis whose `fliessrichtung` is
  true (flow follows the digitised direction) and long enough to measure, heights
  are fitted against distance; a significant slope must descend far more often than
  chance allows. A north-south mirrored atlas is the control that shows the test
  can fail. Axes flagged false are reported separately: reversing them, as the
  flag's name suggests, scrambles the test, and using the terrain to decide what
  the flag means would make the test circular.
* Valley floor. An axis lies below its banks.
* River crossings. A bare-earth model carries no bridge decks, so where a bridge
  crosses an axis the axis profile must show no bump.
* Region edges. The valley test restricted to the atlas margin, where the
  reprojection halos of neighbouring tiles can disagree by metres.
"""
from __future__ import annotations

import hashlib
import math
import statistics
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from geoneural.common import read_json, write_json
from geoneural.data.hfp import sample

SERVICE = "https://www.wfs.nrw.de/geobasis/wfs_nw_atkis-basis-dlm_aaa-modell-basiert"
TYPES = ("AX_Wasserspiegelhoehe", "AX_Gewaesserachse", "AX_BauwerkImVerkehrsbereich")
BRIDGE = "1800"  # bauwerksfunktion: Brücke
STATED_ACCURACY_M = 0.2  # DGM1 tile metadata, "Genauigkeit"
PUBLISHED_ROUNDING_M = 0.05  # water levels are published to 0.1 m
STEP_M = 10.0


def fetch(bounds: list[float], out_dir: Path, page: int = 1000, limit: int = 50) -> dict:
    """Bounded GetFeature per type, paged; raw pages and a receipt are kept."""
    import requests
    out_dir.mkdir(parents=True, exist_ok=True)
    west, south, east, north = bounds
    receipt = {"service": SERVICE, "bounds": bounds, "licence": "DL-DE-Zero-2.0", "files": []}
    for kind in TYPES:
        for index in range(limit):
            params = {"SERVICE": "WFS", "VERSION": "2.0.0", "REQUEST": "GetFeature",
                      "TYPENAMES": f"adv:{kind}", "COUNT": page, "STARTINDEX": index * page,
                      "BBOX": f"{west},{south},{east},{north},urn:ogc:def:crs:EPSG::25832"}
            response = requests.get(SERVICE, params=params, timeout=120)
            response.raise_for_status()
            path = out_dir / f"{kind}-{index:03d}.xml"
            path.write_bytes(response.content)
            returned = int(ET.fromstring(response.content).get("numberReturned", "0"))
            receipt["files"].append({"path": path.name, "url": response.url, "returned": returned,
                                     "sha256": hashlib.sha256(response.content).hexdigest(),
                                     "retrievedUtc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
            if returned < page:
                break
        else:
            receipt.setdefault("incomplete", []).append(kind)
    write_json(out_dir / "receipt.json", receipt)
    return receipt


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse(directory: Path) -> list[dict]:
    """Features with their attributes and the first geometry they carry."""
    features = []
    for path in sorted(Path(directory).glob("AX_*.xml")):
        for member in ET.parse(path).getroot():
            if _local(member.tag) != "member":
                continue
            for feature in member:
                attrs, geometry = {}, None
                for node in feature.iter():
                    name = _local(node.tag)
                    if name in ("pos", "posList") and geometry is None:
                        values = [float(v) for v in node.text.split()]
                        coords = list(zip(values[0::2], values[1::2]))
                        geometry = coords
                    elif len(node) == 0 and node.text and node.text.strip() and name not in ("pos", "posList"):
                        attrs.setdefault(name, node.text.strip())
                features.append({"type": _local(feature.tag), "attrs": attrs, "coords": geometry or []})
    return features


def _densify(coords: list[tuple[float, float]], step: float = STEP_M) -> np.ndarray:
    points = [coords[0]]
    for (x0, y0), (x1, y1) in zip(coords, coords[1:]):
        length = math.hypot(x1 - x0, y1 - y0)
        for k in range(1, max(1, int(length // step)) + 1):
            t = min(1.0, k * step / length) if length else 1.0
            points.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
    return np.array(points)


def _slope(distance: np.ndarray, height: np.ndarray) -> tuple[float, float]:
    """Least-squares slope and its standard error."""
    d = distance - distance.mean()
    slope = float((d * (height - height.mean())).sum() / (d * d).sum())
    residual = height - height.mean() - slope * d
    se = math.sqrt(float((residual ** 2).sum()) / max(len(d) - 2, 1) / float((d * d).sum()))
    return slope, se


def _binomial_tail(k: int, n: int) -> float:
    """P(X >= k) for X ~ Binomial(n, 1/2)."""
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n if n else 1.0


def inspect(grid: np.ndarray, manifest: dict, features: list[dict]) -> dict:
    west, south, east, north = manifest["bounds"]
    height = lambda e, n: sample(grid, manifest, np.asarray(e), np.asarray(n))  # noqa: E731
    levels = []
    for f in features:
        if f["type"] == "AX_Wasserspiegelhoehe" and f["coords"]:
            (e, n), published = f["coords"][0], float(f["attrs"].get("hoeheDesWasserspiegels", "nan"))
            atlas = float(height([e], [n])[0])
            if math.isfinite(atlas) and math.isfinite(published):
                levels.append({"east": e, "north": n, "publishedM": published, "atlasM": atlas,
                               "differenceM": atlas - published})
    tolerance = STATED_ACCURACY_M + PUBLISHED_ROUNDING_M

    axes = []
    for f in features:
        if f["type"] != "AX_Gewaesserachse" or len(f["coords"]) < 2:
            continue
        line = _densify(f["coords"])
        inside = (line[:, 0] >= west) & (line[:, 0] <= east) & (line[:, 1] >= south) & (line[:, 1] <= north)
        line = line[inside]
        if len(line) >= 2:
            axes.append({"line": line, "width": float(f["attrs"].get("breiteDesGewaessers", "0") or 0),
                         "flow": f["attrs"].get("fliessrichtung")})

    def orientation(g: np.ndarray, flag: str) -> dict:
        """Axes carrying `fliessrichtung == flag`, taken in their digitised direction."""
        down = up = flat = 0
        for axis in axes:
            line = axis["line"]
            if axis["flow"] != flag or len(line) < 50:  # 500 m: shorter axes cannot resolve a slope
                continue
            distance = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(line, axis=0).T))])
            h = sample(g, manifest, line[:, 0], line[:, 1])
            ok = np.isfinite(h)
            slope, se = _slope(distance[ok], h[ok])
            if abs(slope) <= 2 * se:
                flat += 1
            elif slope < 0:
                down += 1
            else:
                up += 1
        return {"descending": down, "ascending": up, "notSignificant": flat,
                "pChance": _binomial_tail(down, down + up)}

    def valley(g: np.ndarray, margin: float | None = None) -> dict:
        below = total = 0
        for axis in axes:
            line = axis["line"]
            if len(line) < 3:
                continue
            tangent = np.gradient(line, axis=0)
            norm = np.hypot(tangent[:, 0], tangent[:, 1])[:, None]
            normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1) / np.where(norm == 0, 1, norm)
            offset = axis["width"] / 2 + 20.0
            keep = np.ones(len(line), dtype=bool)
            if margin is not None:
                edge = np.minimum.reduce([line[:, 0] - west, east - line[:, 0], line[:, 1] - south, north - line[:, 1]])
                keep = edge <= margin
            centre = sample(g, manifest, line[:, 0], line[:, 1])
            left = sample(g, manifest, line[:, 0] + offset * normal[:, 0], line[:, 1] + offset * normal[:, 1])
            right = sample(g, manifest, line[:, 0] - offset * normal[:, 0], line[:, 1] - offset * normal[:, 1])
            ok = keep & np.isfinite(centre) & np.isfinite(left) & np.isfinite(right)
            below += int(np.sum(centre[ok] <= np.minimum(left[ok], right[ok])))
            total += int(ok.sum())
        return {"samples": total, "belowBothBanks": below / total if total else None}

    bumps = []
    for f in features:
        if f["type"] != "AX_BauwerkImVerkehrsbereich" or f["attrs"].get("bauwerksfunktion") != BRIDGE or not f["coords"]:
            continue
        cx, cy = np.mean(np.array(f["coords"]), axis=0)
        for axis in axes:
            line = axis["line"]
            gap = np.hypot(line[:, 0] - cx, line[:, 1] - cy)
            at = int(np.argmin(gap))
            if gap[at] > 15.0:
                continue
            profile = sample(grid, manifest, line[:, 0], line[:, 1])
            window = profile[max(0, at - 6):at + 7]
            if np.isfinite(window).sum() >= 5 and np.isfinite(profile[at]):
                bumps.append(float(profile[at] - np.nanmedian(window)))
            break

    mirrored = grid[::-1, :]
    report = {
        "waterLevels": {"points": levels, "toleranceM": tolerance,
                        "allWithinTolerance": bool(levels) and all(abs(l["differenceM"]) <= tolerance for l in levels)},
        # Only 'true' has an unambiguous meaning (flow follows digitisation), so only it
        # tests the atlas. 'false' is reported, not used: in this data those axes also
        # descend as digitised, so the flag cannot mean "reversed" here.
        "orientation": {"flowAlongDigitisation": orientation(grid, "true"),
                        "flowAlongDigitisationMirrored": orientation(mirrored, "true"),
                        "flagFalseAsDigitised": orientation(grid, "false")},
        "valleyFloor": {"atlas": valley(grid), "mirroredNorthSouth": valley(mirrored),
                        "atlasMargin300m": valley(grid, 300.0)},
        "crossings": {"bridgesOnAxes": len(bumps),
                      "absBumpM": ({"median": statistics.median(abs(b) for b in bumps),
                                    "max": max(abs(b) for b in bumps)} if bumps else None)},
        "axes": len(axes),
        "scope": "Terrain features against an independently maintained landscape model. Not a height-accuracy "
                 "measure beyond the water levels; the valley and crossing tests are about placement.",
    }
    return report
