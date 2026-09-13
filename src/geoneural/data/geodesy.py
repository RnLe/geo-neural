"""Geographic, axis and datum verification of a prepared atlas.

Nothing downstream is meaningful if the lattice is transposed, flipped, shifted by
half a sample, or vertically referenced to something other than what it claims.
These checks are therefore run before any quality or rate claim.

Two classes of check live here and they are not interchangeable:

* Structural checks need no external truth. They verify the sample-centre
  convention, the north-up row order, the projected span, the GeoTIFF transform
  and the agreement between the research reference and the decoded runtime pages.
* Landmark checks need independently documented elevations, supplied by the
  operator in a references file. This module refuses to invent them. An
  unconfirmed reference produces a provisional result that is explicitly labelled
  as such, never a pass.
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np

from geoneural.common import read_json, safe_child

#: Node coordinates are sample centres: node (row, col) sits at
#: easting = west + col * spacing, northing = north - row * spacing.
ROW_ZERO_IS_NORTH = True
TRANSFORM_TOLERANCE_M = 1e-6
SPAN_TOLERANCE_M = 1e-5


def crs_facts(horizontal: str, vertical: str) -> dict:
    """Confirm the declared references are the kind of CRS they are used as."""
    import pyproj

    h = pyproj.CRS.from_user_input(horizontal)
    v = pyproj.CRS.from_user_input(vertical)
    if not h.is_projected:
        raise ValueError(f"{horizontal} is not a projected CRS; the atlas lattice is metric")
    if v.type_name != "Vertical CRS":
        raise ValueError(f"{vertical} is not a vertical CRS; heights would have no declared datum")
    return {
        "proj_version": pyproj.proj_version_str,
        "pyproj_version": pyproj.__version__,
        "horizontal": {"code": horizontal, "name": h.name, "projected": True, "axis_order_forced_xy": True},
        "vertical": {"code": vertical, "name": v.name},
        "note": (
            "The vertical reference is asserted by the source contract and is not converted here. "
            "No geoid transformation is performed."
        ),
    }


def roundtrip_corners(bounds: list[float], horizontal: str) -> dict:
    """Transform the corners to geographic coordinates and back.

    `always_xy=True` is explicit: a silent authority-axis swap is the classic way
    an easting/northing pair becomes a northing/easting pair.
    """
    import pyproj

    west, south, east, north = map(float, bounds)
    forward = pyproj.Transformer.from_crs(horizontal, "EPSG:4326", always_xy=True)
    back = pyproj.Transformer.from_crs("EPSG:4326", horizontal, always_xy=True)
    worst = 0.0
    corners = []
    for x, y in ((west, south), (west, north), (east, south), (east, north)):
        lon, lat = forward.transform(x, y)
        rx, ry = back.transform(lon, lat)
        error = math.hypot(rx - x, ry - y)
        worst = max(worst, error)
        corners.append({"easting": x, "northing": y, "lon": lon, "lat": lat, "roundtrip_error_m": error})
    return {"corners": corners, "max_roundtrip_error_m": worst, "always_xy": True}


def lattice_check(manifest: dict) -> dict:
    """Verify span, sample-centre endpoints and the declared orientation flags."""
    west, south, east, north = map(float, manifest["bounds"])
    side = int(manifest["sample_side"])
    spacing = float(manifest["spacing_m"])
    span_x, span_y = east - west, north - south
    expected = (side - 1) * spacing
    findings = []
    if abs(span_x - expected) > SPAN_TOLERANCE_M or abs(span_y - expected) > SPAN_TOLERANCE_M:
        findings.append(
            f"Span {span_x}x{span_y} m disagrees with (sample_side-1)*spacing = {expected} m. "
            "Bounds must be sample-centre endpoints, not pixel edges."
        )
    if not manifest.get("sample_centres"):
        findings.append("Manifest does not declare sample-centre bounds")
    if not manifest.get("north_up"):
        findings.append("Manifest does not declare north-up row order")
    return {
        "sample_side": side,
        "spacing_m": spacing,
        "span_x_m": span_x,
        "span_y_m": span_y,
        "expected_span_m": expected,
        "row_zero_is_north": ROW_ZERO_IS_NORTH,
        "findings": findings,
    }


def raster_transform_check(atlas_dir: Path, manifest: dict) -> dict:
    """Verify the research GeoTIFF places pixel centres on node coordinates."""
    reference = atlas_dir / "reference.tif"
    if not reference.exists():
        return {"present": False, "findings": ["reference.tif absent; transform not verified"]}
    import rasterio
    from rasterio.transform import xy

    west, _south, _east, north = map(float, manifest["bounds"])
    spacing = float(manifest["spacing_m"])
    findings = []
    with rasterio.open(reference) as dataset:
        centre_x, centre_y = xy(dataset.transform, 0, 0)
        if abs(centre_x - west) > TRANSFORM_TOLERANCE_M or abs(centre_y - north) > TRANSFORM_TOLERANCE_M:
            findings.append(
                f"First pixel centre ({centre_x}, {centre_y}) is not the north-west node ({west}, {north})"
            )
        if abs(dataset.transform.a - spacing) > TRANSFORM_TOLERANCE_M:
            findings.append(f"Pixel width {dataset.transform.a} m disagrees with spacing {spacing} m")
        if abs(dataset.transform.e + spacing) > TRANSFORM_TOLERANCE_M:
            findings.append(
                f"Pixel height {dataset.transform.e} m is not the negative spacing; rows may not run north to south"
            )
        shape = (dataset.height, dataset.width)
        crs = dataset.crs.to_string() if dataset.crs else None
    if crs != manifest["crs"]:
        findings.append(f"reference.tif CRS {crs} disagrees with manifest {manifest['crs']}")
    return {"present": True, "shape": list(shape), "crs": crs, "findings": findings}


def node_of(manifest: dict, easting: float, northing: float) -> tuple[float, float]:
    """Fractional (row, col) of a projected coordinate on the canonical lattice."""
    west, _south, _east, north = map(float, manifest["bounds"])
    spacing = float(manifest["spacing_m"])
    return (north - northing) / spacing, (easting - west) / spacing


def sample_reference(reference: np.ndarray, row: float, col: float) -> float:
    """Bilinear sample of the prepared reference at a fractional node position."""
    side = reference.shape[0]
    if not (0 <= row <= side - 1 and 0 <= col <= side - 1):
        raise ValueError("Coordinate lies outside the prepared region")
    r0, c0 = int(math.floor(row)), int(math.floor(col))
    r1, c1 = min(r0 + 1, side - 1), min(c0 + 1, side - 1)
    fr, fc = row - r0, col - c0
    top = reference[r0, c0] * (1 - fc) + reference[r0, c1] * fc
    bottom = reference[r1, c0] * (1 - fc) + reference[r1, c1] * fc
    return float(top * (1 - fr) + bottom * fr)


def runtime_agreement(atlas_dir: Path, manifest: dict, reference: np.ndarray) -> dict:
    """Compare decoded finest pages against the research reference.

    This is the orientation check that matters: a transposed or vertically flipped
    page would still decode cleanly and still look like terrain. Only the corner
    pages are read, which is enough to catch a flip or transpose while keeping the
    check cheap.
    """
    from geoneural.codecs.eat1 import decode

    intervals = int(manifest["page_intervals"])
    side = int(manifest["sample_side"])
    leaves = (side - 1) // intervals
    checks = []
    worst = 0.0
    for x, y in ((0, 0), (leaves - 1, 0), (0, leaves - 1), (leaves - 1, leaves - 1)):
        key = f"0/{x}/{y}"
        page = manifest["pages"].get(key)
        if page is None:
            checks.append({"key": key, "error": "absent from manifest"})
            continue
        values, header = decode(safe_child(atlas_dir, page["path"]).read_bytes())
        r0, c0 = y * intervals, x * intervals
        expected = reference[r0 : r0 + header["side"], c0 : c0 + header["side"]]
        difference = float(np.max(np.abs(values - expected)))
        worst = max(worst, difference)
        checks.append({"key": key, "max_difference_m": difference, "quantum_m": header["quantum_m"]})
    quantum = float(manifest["quantum_m"])
    findings = []
    if worst > quantum / 2 + 1e-9:
        findings.append(
            f"Decoded pages differ from the reference by up to {worst} m, beyond the {quantum / 2} m "
            "quantization half-step. Orientation, addressing or quantization is wrong."
        )
    return {"pages": checks, "max_difference_m": worst, "quantization_half_step_m": quantum / 2, "findings": findings}


def landmark_check(manifest: dict, reference: np.ndarray, references_path: Path, allow_unconfirmed: bool) -> dict:
    """Compare atlas elevations with independently documented landmark heights.

    The references file is operator-supplied. Each entry must name its source and
    the vertical reference its elevation is quoted against; an entry quoted
    against a different datum is reported, not silently converted.
    """
    import pyproj

    document = read_json(references_path)
    if document.get("schema") != "geoneural-landmarks-v1":
        raise ValueError("Landmark file must declare schema geoneural-landmarks-v1")
    to_projected = pyproj.Transformer.from_crs("EPSG:4326", manifest["crs"], always_xy=True)
    results = []
    findings = []
    provisional = 0
    for entry in document.get("landmarks", [])[:64]:
        confirmed = bool(entry.get("confirmed"))
        if not confirmed:
            provisional += 1
            if not allow_unconfirmed:
                results.append({"name": entry.get("name"), "status": "skipped-unconfirmed"})
                continue
        easting, northing = to_projected.transform(float(entry["lon"]), float(entry["lat"]))
        try:
            row, col = node_of(manifest, easting, northing)
            observed = sample_reference(reference, row, col)
        except ValueError as error:
            results.append({"name": entry.get("name"), "status": "outside-region", "detail": str(error)})
            continue
        published = float(entry["elevation_m"])
        tolerance = float(entry.get("tolerance_m", 0.0))
        difference = observed - published
        same_datum = entry.get("vertical_crs") == manifest["vertical_crs"]
        status = "agrees" if abs(difference) <= tolerance and same_datum else "disagrees"
        if not same_datum:
            status = "different-vertical-reference"
        if status != "agrees":
            findings.append(
                f"{entry.get('name')}: atlas {observed:.2f} m vs published {published:.2f} m "
                f"({status}; tolerance {tolerance} m)"
            )
        results.append({
            "name": entry.get("name"),
            "lat": entry.get("lat"),
            "lon": entry.get("lon"),
            "easting": easting,
            "northing": northing,
            "published_elevation_m": published,
            "published_vertical_crs": entry.get("vertical_crs"),
            "atlas_elevation_m": observed,
            "difference_m": difference,
            "tolerance_m": tolerance,
            "source": entry.get("source"),
            "confirmed": confirmed,
            "status": status,
        })
    return {
        "references": str(references_path),
        "results": results,
        "unconfirmed_entries": provisional,
        "findings": findings,
        "qualification": (
            "Landmark agreement uses operator-supplied published elevations. Unconfirmed entries "
            "yield a provisional observation, never an acceptance."
        ),
    }


def verify(atlas_path: Path, references_path: Path | None = None, allow_unconfirmed: bool = False) -> dict:
    """Run every available check against one prepared atlas."""
    atlas_path = Path(atlas_path)
    atlas_dir = atlas_path.parent
    manifest = read_json(atlas_path)
    if manifest.get("schema") != "geoneural-atlas-v1":
        raise ValueError("Not a geoneural-atlas-v1 manifest")
    reference_file = atlas_dir / "reference.npy"
    if not reference_file.exists():
        raise ValueError("reference.npy is required for geographic verification")
    reference = np.load(reference_file).astype(np.float64)

    report: dict = {
        "schema": "geoneural-lattice-verification-v1",
        "atlas_content_id": manifest.get("content_id"),
        "crs": crs_facts(manifest["crs"], manifest["vertical_crs"]),
        "roundtrip": roundtrip_corners(manifest["bounds"], manifest["crs"]),
        "lattice": lattice_check(manifest),
        "raster_transform": raster_transform_check(atlas_dir, manifest),
        "runtime_agreement": runtime_agreement(atlas_dir, manifest, reference),
    }
    if references_path is not None:
        report["landmarks"] = landmark_check(manifest, reference, Path(references_path), allow_unconfirmed)
    findings: list[str] = []
    for section in report.values():
        if isinstance(section, dict):
            findings.extend(section.get("findings", []))
    report["findings"] = findings
    report["passed_structural_checks"] = not findings
    report["qualification"] = (
        "Structural agreement of lattice, transform, orientation and codec only. This does not "
        "establish source positional or vertical accuracy, which require independent references."
    )
    return report
