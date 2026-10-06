"""Write the static data bundle the browser viewer and lab read.

Everything the page shows comes from here, so every number on it traces back to the
candidate table and the prepared reference. Fields are sent at 20 m (every second node of
the 10 m lattice) to keep the transfer small; metrics are always computed at 10 m.

Encodings (all little-endian, all gzip-compressed):

* ``u16-delta``: heights as uint16 codes, ``h = offset + quantum * code``, stored as the
  first difference of the row-major code sequence (mod 2^16). Exact to the 1 cm quantum.
* ``bits``: a boolean mask, row-major, packed eight cells per byte (most significant first).
* ``u8``: one byte per node.
"""
from __future__ import annotations

import gzip
import hashlib
import shutil
from pathlib import Path

import numpy as np

from geoneural.common import HOME, read_json, utc, write_json

SCHEMA = "geoneural-web-bundle-v1"
QUANTUM_M = 0.01
STRIDE = 2
LAB_LEVEL = 3
LAB_SEED = 4242
LAB_SURFACES = 4
STREAM_CELLS = 500
#: Candidates shown in the viewer, chosen by rule rather than by look: bounded quantisers at
#: 0.1 m and 1 m, the corrected product at 1 m, the largest saved neural field (the 135 kB SIREN),
#: its best conventional rival at no more bytes (a 20 m grid at 0.5 m), and the smallest neural
#: and hybrid codec fits of the search.
VIEWER = ("q32dz-full@0.1", "q32dz-full@1", "q32dz-level1@0.5", "corrected-r1@1", "checkpoint/siren-codec-fit",
          "fit/siren-t17", "fit/hybrid-t5")
#: Muted categorical colours for the GK100 material classes; unknown stays transparent.
PALETTE = ("#00000000", "#b9a88c", "#8fa3a8", "#7f8f6a", "#c9c3a3", "#d8b46a", "#e6d29a", "#c98f5f",
           "#a7b58a", "#8a7f9e", "#9fb7c9", "#c7a4a4", "#a4c7b4")


def _gz(path: Path, raw: bytes) -> dict:
    data = gzip.compress(raw, compresslevel=9, mtime=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def heights(path: Path, field: np.ndarray, offset: float) -> dict:
    codes = np.rint((field - offset) / QUANTUM_M)
    if codes.min() < 0 or codes.max() > 65535:
        raise ValueError("height field does not fit the uint16 range of the bundle")
    flat = codes.astype(np.uint16).ravel()
    delta = np.diff(flat, prepend=np.uint16(0)).astype(np.uint16)
    return {"file": path.name, "encoding": "u16-delta", **_gz(path, delta.astype("<u2").tobytes())}


def bits(path: Path, mask: np.ndarray) -> dict:
    return {"file": path.name, "encoding": "bits", **_gz(path, np.packbits(mask.astype(bool).ravel()).tobytes())}


def pool_any(mask: np.ndarray, stride: int) -> np.ndarray:
    """A coarse cell is set if any fine node in its neighbourhood is, so thin streams survive."""
    side = (mask.shape[0] - 1) // stride + 1
    reach = stride // 2
    padded = np.pad(mask.astype(bool), reach, constant_values=False)
    out = np.zeros((side, side), dtype=bool)
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            window = padded[reach + dy:reach + dy + mask.shape[0], reach + dx:reach + dx + mask.shape[1]]
            out |= window[::stride, ::stride]
    return out


def conventional_field(atlas: Path, candidate_id: str, reference: np.ndarray) -> np.ndarray | None:
    """Rebuild a conventional or corrected candidate's 10 m field from its definition."""
    from geoneural.codecs import envelope
    from geoneural.metrics import corrections, hydrology
    if candidate_id.startswith("q32dz-"):
        name, target = candidate_id.split("@")
        level = 0 if name == "q32dz-full" else int(name.removeprefix("q32dz-level"))
        return envelope.decoded_field(atlas, level, float(target))[0]
    if candidate_id.startswith("corrected-r"):
        radius, target = candidate_id.removeprefix("corrected-r").split("@")
        step = 2.0 * float(target)
        coarse = np.round(reference / step) * step
        stream = hydrology.analyse(reference, 10.0, STREAM_CELLS)["stream"]
        corrected, _ = corrections.apply_corrections(coarse, reference, corrections.dilate(stream, int(radius)),
                                                     QUANTUM_M)
        return corrected
    return None


def export(out: Path, candidates: dict, atlas: Path | None = None, fields_dir: Path | None = None,
           lab_dir: Path | None = None, geology_dir: Path | None = None) -> Path:
    """Write the bundle to `out` (replaced if it exists) and return its manifest path."""
    import glob

    from geoneural.data import geology
    from geoneural.metrics import hydrology
    atlas = Path(atlas or HOME / "atlases" / "essen-ruhr" / "atlas.json")
    fields_dir = Path(fields_dir or HOME / "fields")
    geology_dir = Path(geology_dir or HOME / "geology" / "essen-ruhr")
    manifest = read_json(atlas)
    reference = np.load(atlas.parent / "reference.npy").astype(np.float64)
    spacing = float(manifest["spacing_m"])
    west, south, east, north = manifest["bounds"]
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    rows = {row["id"]: row for row in candidates["candidates"]}
    fields = {}
    for cid in VIEWER:
        field = conventional_field(atlas, cid, reference)
        if field is None:
            path = fields_dir / (cid.replace("/", "--") + ".npy")
            if not path.exists():
                print(f"skipping {cid}: no saved field at {path}")
                continue
            field = np.load(path).astype(np.float64)
        if cid not in rows:
            print(f"skipping {cid}: not in the candidate table")
            continue
        fields[cid] = field

    every = [reference] + list(fields.values())
    offset = float(np.floor(min(float(f.min()) for f in every)) - 1.0)
    display = reference[::STRIDE, ::STRIDE]
    side = int(display.shape[0])
    reference_analysis = hydrology.analyse(reference, spacing, STREAM_CELLS)
    bundle = {
        "schema": SCHEMA, "createdUtc": utc(),
        "region": {"id": "essen-ruhr", "title": "Southern Essen and the Ruhr valley",
                   "crs": manifest["crs"], "verticalCrs": manifest["vertical_crs"],
                   "bounds": manifest["bounds"], "sampleCentres": True, "northUp": True,
                   "referenceSide": int(reference.shape[0]), "referenceSpacingM": spacing,
                   "referenceSha256": manifest["reference_sha256"]},
        "grid": {"side": side, "spacingM": spacing * STRIDE, "stride": STRIDE, "rowOrder": "north-to-south",
                 "heightOffsetM": offset, "heightQuantumM": QUANTUM_M,
                 "note": "Every second node of the 10 m reference. Metrics are computed at 10 m."},
        "reference": {"height": heights(out / "reference.height.bin", display, offset),
                      "streams": bits(out / "reference.streams.bin",
                                      pool_any(reference_analysis["stream"], STRIDE)),
                      "minM": float(reference.min()), "maxM": float(reference.max())},
        "streamThresholdCells": STREAM_CELLS,
        "candidates": [],
    }
    for cid, field in fields.items():
        row = rows[cid]
        analysis = hydrology.analyse(field, spacing, STREAM_CELLS)
        check = hydrology.compare(reference, field, spacing, STREAM_CELLS, reference_analysis=reference_analysis)
        stem = cid.replace("/", "--").replace("@", "-at-")
        bundle["candidates"].append({
            "id": cid, "label": row["label"], "family": row["family"], "evidence": row["evidence"],
            # Version-1 candidates: payload bytes under the old accounting, not comparable with v2 products.
            "historical": True, "accounting": "v1 payload only",
            "bytes": row["bytes"], "maeM": row["maeM"], "maxM": row["maxM"], "streamJaccard": row["streamJaccard"],
            "boundM": row["boundM"],
            "check": {"maeM": float(np.abs(field - reference).mean()),
                      "maxM": float(np.abs(field - reference).max()),
                      "streamJaccard": check["streamJaccard"]},
            "height": heights(out / f"{stem}.height.bin", field[::STRIDE, ::STRIDE], offset),
            "streams": bits(out / f"{stem}.streams.bin", pool_any(analysis["stream"], STRIDE))})

    units = geology.parse_units(glob.glob(str(geology_dir / "*.gml")))
    raster = geology.rasterise(units, manifest["bounds"], side, spacing * STRIDE)
    faults = geology.parse_faults(glob.glob(str(geology_dir / "*.gml")))
    legend = sorted(raster["legend"].items(), key=lambda kv: kv[1])
    bundle["geology"] = {
        "classes": {"file": "geology.classes.bin", "encoding": "u8",
                    **_gz(out / "geology.classes.bin", raster["classes"].astype(np.uint8).tobytes())},
        "legend": [{"code": 0, "label": "not mapped", "color": PALETTE[0]}] +
                  [{"code": code, "label": label, "color": PALETTE[1 + (code - 1) % (len(PALETTE) - 1)]} for label, code in legend],
        "faults": [[[round(x - west, 1), round(y - south, 1)] for x, y in line]
                   for fault in faults for line in _lines(fault["geometry"])],
        "faultsFrame": "metres east and north of the south-west reference node",
        "source": "GK100, Geologischer Dienst NRW, INSPIRE WFS; DL-DE-BY-2.0",
        "scale": "1:100,000", "unknownFraction": raster["unknownFraction"],
        "note": "Mapped surface units and fault traces. No depth, dip or throw."}

    (out / "lab").mkdir(exist_ok=True)
    if lab_dir and (Path(lab_dir) / "closure.json").exists():
        closure = read_json(Path(lab_dir) / "closure.json")
        shutil.copy2(Path(lab_dir) / "closure.json", out / "lab" / "closure.json")
        for arm in closure["arms"]:
            shutil.copy2(Path(lab_dir) / f"{arm}.bin", out / "lab" / f"{arm}.bin")
        bundle["closure"] = {"meta": "lab/closure.json", "weights": {arm: f"lab/{arm}.bin" for arm in closure["arms"]},
                             "encoding": "raw little-endian float32, see closure.json"}
        # The learned arms only know rough synthetic surfaces like the ones they were trained on, so
        # the conservation preset starts from fresh draws of that same distribution.
        from geoneural.physics import hybrid
        rng = np.random.default_rng(LAB_SEED)
        side_lab = int(closure["training"]["side"])
        surfaces, _ = hybrid._batch(rng, LAB_SURFACES, side_lab, float(closure["spacingM"]),
                                    float(closure["teacher"]["diffusivity"]),
                                    float(closure["teacher"]["criticalSlope"]))
        bundle["closure"]["surfaces"] = {
            "file": "lab/surfaces.f32.bin", "encoding": "f32", "count": LAB_SURFACES, "side": side_lab,
            "spacingM": float(closure["spacingM"]), "seed": LAB_SEED,
            **_gz(out / "lab" / "surfaces.f32.bin", surfaces.astype("<f4").tobytes()),
            "note": "Synthetic rough surfaces from the training distribution, not seen in training."}
    lab_grid = reference[::2 ** LAB_LEVEL, ::2 ** LAB_LEVEL]
    lab_classes = geology.rasterise(units, manifest["bounds"], lab_grid.shape[0], spacing * 2 ** LAB_LEVEL)["classes"]
    bundle["labTerrain"] = {
        "side": int(lab_grid.shape[0]), "spacingM": spacing * 2 ** LAB_LEVEL,
        "height": {"file": "lab/terrain.f32.bin", "encoding": "f32",
                   **_gz(out / "lab" / "terrain.f32.bin", lab_grid.astype("<f4").tobytes())},
        "classes": {"file": "lab/geology.u8.bin", "encoding": "u8",
                    **_gz(out / "lab" / "geology.u8.bin", lab_classes.astype(np.uint8).tobytes())},
        "note": "Every eighth reference node (80 m). A starting surface for scenarios, not a forecast."}

    bundle["chart"] = [
        {k: row[k] for k in ("id", "label", "family", "experiment", "bytes", "maeM", "maxM", "streamJaccard",
                             "boundGuaranteed", "evidence", "dominatedBy")}
        for row in candidates["candidates"] if row["package"] == "finest-per-page" and row["bytes"]]
    bundle["attribution"] = [
        "Terrain: DGM1, Geobasis NRW (Bezirksregierung Koeln), DL-DE-Zero-2.0",
        "Geology: IS GK100, Geologischer Dienst NRW, DL-DE-BY-2.0"]
    write_json(out / "manifest.json", bundle)
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"bundle: {out} ({total / 1e6:.2f} MB, {len(bundle['candidates'])} candidates)")
    return out / "manifest.json"


def _lines(geometry: dict) -> list:
    """Coordinate lines of a GeoJSON LineString or MultiLineString."""
    if geometry["type"] == "LineString":
        return [geometry["coordinates"]]
    if geometry["type"] == "MultiLineString":
        return list(geometry["coordinates"])
    raise ValueError(f"unexpected fault geometry {geometry['type']}")
