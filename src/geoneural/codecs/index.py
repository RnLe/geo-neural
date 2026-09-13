"""EATIDX1: a compact page index, because the JSON one is most of the package.

At a 1.0 m error target the manifest plus attribution is 37.9 % of everything a
user downloads. A neural representation that drives payload toward one bit per
sample would be dominated by an index it does not shrink, so the index has to
shrink before any rate claim about a model means anything.

Almost none of the JSON is information. `error_kind` is one 85-character string
repeated 341 times, costing 32 KB with no per-page content. `path`, `x_m`, `z_m`,
`width_m`, `spacing_m`, `level`, `x` and `y` are all functions of the page key
and the lattice. `side` and `raw_bytes` are constants. What is actually
per-page is a hash, a compressed length, a height range and an error estimate.

Bounds are stored as quantum codes and rounded outward: a reconstructed
minimum is never above the true one, a maximum never below, an error estimate
never under. A selector that culls on these bounds therefore stays correct; the
index may cost it a little conservatism, never a missing page. Exact float64
round-tripping is not used: it would cost 8 bytes per field to preserve digits
that came from a float32 reference and have no physical meaning.
"""
from __future__ import annotations
import math
import struct
from pathlib import Path

MAGIC = b"EATIDX1\0"
VERSION = 1
# magic, version, sample_side, intervals, max_level, page_side, raw_bytes
_HEAD = struct.Struct("<8sHIHBHI")
_SCALARS = struct.Struct("<8d")  # spacing, quantum, west, south, east, north, height min/max
_PAGE = struct.Struct("<32sIiiI")
PAGE_BYTES = _PAGE.size


def _text(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<H", len(raw)) + raw


def _read_text(blob: bytes, offset: int) -> tuple[str, int]:
    (length,) = struct.unpack_from("<H", blob, offset)
    offset += 2
    return blob[offset:offset + length].decode("utf-8"), offset + length


def _page_order(manifest: dict) -> list[tuple[int, int, int]]:
    """Level-major, then raster order. Deterministic, and it lets the decoder
    regenerate keys without storing any of them."""
    order = []
    for level in range(manifest["max_level"], -1, -1):
        span = 2 ** (manifest["max_level"] - level)
        for y in range(span):
            for x in range(span):
                order.append((level, x, y))
    return order


def encode(manifest: dict) -> bytes:
    """Pack a published atlas manifest into EATIDX1."""
    pages = manifest["pages"]
    quantum = float(manifest["quantum_m"])
    intervals = int(manifest["page_intervals"])
    side = int(manifest["page_intervals"]) + 1
    sample = next(iter(pages.values()))
    raw_bytes = int(sample["raw_bytes"])
    constants = {"error_kind": sample["error_kind"], "crs": manifest["crs"],
                 "vertical_crs": manifest["vertical_crs"], "filter": manifest["filter"],
                 "source_kind": manifest["source_kind"], "content_id": manifest["content_id"],
                 "algorithm_sha256": manifest["algorithm_sha256"], "name": manifest["name"]}
    for key, entry in pages.items():
        if int(entry["side"]) != side or int(entry["raw_bytes"]) != raw_bytes:
            raise ValueError(f"page {key} has a nonuniform side or raw size; EATIDX1 assumes one page shape")
        if entry["error_kind"] != constants["error_kind"]:
            raise ValueError(f"page {key} declares a different error kind; it can no longer be hoisted")

    west, south, east, north = manifest["bounds"]
    out = bytearray()
    out += _HEAD.pack(MAGIC, VERSION, int(manifest["sample_side"]), intervals,
                      int(manifest["max_level"]), side, raw_bytes)
    out += _SCALARS.pack(float(manifest["spacing_m"]), quantum, west, south, east, north,
                         float(manifest["height_range_m"][0]), float(manifest["height_range_m"][1]))
    for name in ("error_kind", "crs", "vertical_crs", "filter", "source_kind",
                 "content_id", "algorithm_sha256", "name"):
        out += _text(constants[name])

    order = _page_order(manifest)
    out += struct.pack("<I", len(order))
    present = bytearray((len(order) + 7) // 8)
    for position, (level, x, y) in enumerate(order):
        if f"{level}/{x}/{y}" in pages:
            present[position // 8] |= 1 << (position % 8)
    out += bytes(present)

    for level, x, y in order:
        entry = pages.get(f"{level}/{x}/{y}")
        if entry is None:
            continue
        out += _PAGE.pack(
            bytes.fromhex(entry["sha256"]),
            int(entry["packed_bytes"]),
            # Outward rounding: the reconstructed range always contains the real one.
            int(math.floor(float(entry["min_m"]) / quantum)),
            int(math.ceil(float(entry["max_m"]) / quantum)),
            int(math.ceil(float(entry["sample_error_m"]) / quantum)),
        )
    return bytes(out)


def decode(blob: bytes) -> dict:
    """Rebuild the manifest fields a runtime needs, deriving everything derivable."""
    try:
        return _decode(blob)
    except (struct.error, IndexError, UnicodeDecodeError) as exc:
        # struct.error is not a ValueError, so without this a truncated index
        # would escape every caller that refuses bad input by catching ValueError.
        raise ValueError(f"Undecodable EATIDX1 index: {exc}") from exc


def _decode(blob: bytes) -> dict:
    if len(blob) < _HEAD.size + _SCALARS.size:
        raise ValueError("Truncated EATIDX1 header")
    magic, version, sample_side, intervals, max_level, side, raw_bytes = _HEAD.unpack_from(blob)
    if magic != MAGIC or version != VERSION:
        raise ValueError("Not an EATIDX1 index")
    if side != intervals + 1 or not 2 <= side <= 513 or max_level > 24:
        raise ValueError("Invalid EATIDX1 lattice header")
    offset = _HEAD.size
    (spacing, quantum, west, south, east, north,
     height_min, height_max) = _SCALARS.unpack_from(blob, offset)
    offset += _SCALARS.size
    if not math.isfinite(quantum) or quantum <= 0 or not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("Invalid EATIDX1 scalars")
    constants = {}
    for name in ("error_kind", "crs", "vertical_crs", "filter", "source_kind",
                 "content_id", "algorithm_sha256", "name"):
        constants[name], offset = _read_text(blob, offset)

    (count,) = struct.unpack_from("<I", blob, offset)
    offset += 4
    expected = sum(4 ** (max_level - level) for level in range(max_level + 1))
    if count != expected:
        raise ValueError("EATIDX1 page count disagrees with the declared pyramid")
    bitmap_bytes = (count + 7) // 8
    if offset + bitmap_bytes > len(blob):
        raise ValueError("Truncated EATIDX1 presence bitmap")
    present = blob[offset:offset + bitmap_bytes]
    offset += bitmap_bytes

    pages = {}
    position = 0
    for level in range(max_level, -1, -1):
        span = 2 ** (max_level - level)
        page_spacing = spacing * 2 ** level
        width = intervals * page_spacing
        for y in range(span):
            for x in range(span):
                if not present[position // 8] >> (position % 8) & 1:
                    position += 1
                    continue
                position += 1
                if offset + _PAGE.size > len(blob):
                    raise ValueError("EATIDX1 page table is shorter than its presence bitmap claims")
                digest, packed, low, high, error = _PAGE.unpack_from(blob, offset)
                offset += _PAGE.size
                pages[f"{level}/{x}/{y}"] = {
                    "level": level, "x": x, "y": y, "side": side,
                    "path": f"pages/{level}/{x}_{y}.eat.gz",
                    "spacing_m": page_spacing, "width_m": width,
                    "x_m": x * width, "z_m": y * width,
                    "packed_bytes": packed, "raw_bytes": raw_bytes,
                    "sha256": digest.hex(),
                    "min_m": low * quantum, "max_m": high * quantum,
                    "sample_error_m": error * quantum,
                    "error_kind": constants["error_kind"],
                }
    if offset != len(blob):
        raise ValueError("Trailing bytes after the EATIDX1 page table")
    return {
        "sample_side": sample_side, "page_intervals": intervals, "max_level": max_level,
        "spacing_m": spacing, "quantum_m": quantum, "bounds": [west, south, east, north],
        "height_range_m": [height_min, height_max], "root": f"{max_level}/0/0",
        "sample_centres": True, "north_up": True, "pages": pages, **constants,
    }


def compare(manifest: dict, blob: bytes) -> dict:
    """How much was saved, and by exactly how much each bound loosened."""
    rebuilt = decode(blob)
    worst_low = worst_high = worst_error = 0.0
    for key, entry in manifest["pages"].items():
        other = rebuilt["pages"][key]
        worst_low = max(worst_low, float(entry["min_m"]) - other["min_m"])
        worst_high = max(worst_high, other["max_m"] - float(entry["max_m"]))
        worst_error = max(worst_error, other["sample_error_m"] - float(entry["sample_error_m"]))
    return {
        "pages": len(manifest["pages"]),
        "compactBytes": len(blob),
        "bytesPerPage": round(len(blob) / max(len(manifest["pages"]), 1), 1),
        "hashShareOfCompact": round(100.0 * 32 * len(manifest["pages"]) / len(blob), 1),
        "minLoosenedByM": worst_low, "maxLoosenedByM": worst_high,
        "errorRaisedByM": worst_error,
        "boundsNeverTightened": worst_low >= 0 and worst_high >= 0 and worst_error >= 0,
    }


def write(manifest_path: Path, target: Path) -> dict:
    import json
    manifest = json.loads(Path(manifest_path).read_text())
    blob = encode(manifest)
    Path(target).write_bytes(blob)
    report = compare(manifest, blob)
    report["jsonBytes"] = Path(manifest_path).stat().st_size
    report["reductionFactor"] = round(report["jsonBytes"] / len(blob), 1)
    return report
