"""Deterministic EAT1 decoder test vectors.

The vectors check that the page format survives a round trip between independent
decoder implementations, and that every decoder refuses the same invalid
payloads. A Python-only test cannot show that: any other decoder reading these
fixtures may use different integer and float handling, a different gzip binding
and a different byte order helper.

The cases are generated rather than committed as opaque blobs so that a reviewer
can see what each one is meant to prove, and regenerate them from source.

The committed bytes are the contract, not this generator. gzip output is not
stable across CPython builds: Python 3.12 and 3.14 produce different bytes for
these fixtures, and both decode identically. The files are pinned by SHA-256 in
manifest.json, so `write()` refuses to overwrite the committed directory unless
explicitly asked, and the tests regenerate into a temporary directory instead.
"""
from __future__ import annotations
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from geoneural.codecs.eat1 import HEADER, encode

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "eat1-golden"
SIDE = 9


def surfaces() -> dict[str, np.ndarray]:
    """Grids chosen so that a sign error, a flip or a transpose changes the result."""
    rows, cols = np.meshgrid(np.arange(SIDE, dtype=np.float64), np.arange(SIDE, dtype=np.float64), indexing="ij")
    return {
        # Straddles zero: catches unsigned decoding of negative elevations.
        "signed_extremes": np.linspace(-251.37, 813.53, SIDE * SIDE).reshape(SIDE, SIDE),
        # Constant: any quantization drift shows up immediately.
        "flat": np.full((SIDE, SIDE), 137.25),
        # Asymmetric ramp: catches a transpose, which a symmetric grid would hide.
        "asymmetric_ramp": 10.0 + 0.5 * rows + 7.0 * cols,
        # Alternating: catches row-order and stride errors.
        "rough": 100.0 + 40.0 * ((-1.0) ** (rows + cols)) + 0.25 * rows,
        # Below the vertical datum: negative heights are legal terrain, not an error.
        "below_datum": np.linspace(-90.5, -0.5, SIDE * SIDE).reshape(SIDE, SIDE),
    }


def valid_cases() -> list[dict]:
    cases = []
    for index, (name, values) in enumerate(sorted(surfaces().items())):
        packed = encode(values, level=index % 5, quantum=0.01)
        cases.append({
            "name": name,
            "file": f"{name}.eat.gz",
            "bytes": packed,
            "level": index % 5,
            "side": SIDE,
            "quantum_m": 0.01,
            "expected_min_m": float(np.round(values.min() / 0.01) * 0.01),
            "expected_max_m": float(np.round(values.max() / 0.01) * 0.01),
            # Every node is checked; the grids are small enough to state exactly.
            "expected_values_m": [float(v) for v in np.round(values / 0.01).astype(np.int64).ravel() * 0.01],
        })
    return cases


def invalid_cases() -> list[dict]:
    """Payloads every decoder must refuse without producing a partial product."""
    base = encode(surfaces()["flat"], level=0, quantum=0.01)
    raw = gzip.decompress(base)
    magic, version, side, level, reserved, quantum, offset = HEADER.unpack(raw[:32])

    def repack(header: bytes, body: bytes) -> bytes:
        return gzip.compress(header + body, compresslevel=6, mtime=0)

    body = raw[32:]
    return [
        {"name": "truncated_gzip", "file": "truncated_gzip.bin", "bytes": base[: len(base) // 2],
         "why": "A short read must not decode to a partial page"},
        {"name": "trailing_bytes", "file": "trailing_bytes.bin", "bytes": base + b"\x00\x01\x02\x03",
         "why": "Concatenated or padded streams are not a valid page"},
        {"name": "not_gzip", "file": "not_gzip.bin", "bytes": b"EAT1" + b"\x00" * 64,
         "why": "Raw header bytes without the gzip frame must be refused"},
        {"name": "wrong_magic", "file": "wrong_magic.bin",
         "bytes": repack(HEADER.pack(b"EAT0", version, side, level, reserved, quantum, offset), body),
         "why": "A different format must not be decoded as EAT1"},
        {"name": "future_version", "file": "future_version.bin",
         "bytes": repack(HEADER.pack(magic, 2, side, level, reserved, quantum, offset), body),
         "why": "An unknown version must refuse rather than guess the layout"},
        {"name": "reserved_set", "file": "reserved_set.bin",
         "bytes": repack(HEADER.pack(magic, version, side, level, 1, quantum, offset), body),
         "why": "Reserved bits carry no agreed meaning; a set bit is unknown framing"},
        {"name": "side_mismatch", "file": "side_mismatch.bin",
         "bytes": repack(HEADER.pack(magic, version, side + 1, level, reserved, quantum, offset), body),
         "why": "A declared side that disagrees with the payload length is corrupt"},
        {"name": "body_short", "file": "body_short.bin",
         "bytes": repack(HEADER.pack(magic, version, side, level, reserved, quantum, offset), body[:-4]),
         "why": "A body shorter than the declared grid must not be zero-extended"},
    ]


def write(directory: Path = FIXTURES, overwrite_committed: bool = False) -> Path:
    """Write every fixture and the manifest.json that describes them.

    Writing into the committed fixture directory needs `overwrite_committed`,
    because those bytes are pinned by SHA-256 in manifest.json and gzip output
    differs between CPython builds. Pass a temporary directory to regenerate for
    comparison; pass the flag only when the vectors are meant to change.
    """
    if directory == FIXTURES and not overwrite_committed and any(FIXTURES.glob("*.bin")):
        raise PermissionError(
            "refusing to rewrite committed EAT1 golden vectors: they are pinned by SHA-256 in "
            "manifest.json and gzip output is not stable across CPython builds. Regenerate into a "
            "temporary directory, or pass overwrite_committed=True when the vectors are meant to change.")
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "geoneural-eat1-golden-v1",
        "side": SIDE,
        "note": ("Decoder test vectors. Values are exact after quantization at the "
                 "stated quantum; a decoder that differs by more than zero has a framing or sign fault."),
        "valid": [],
        "invalid": [],
    }
    raw_bytes = 32 + SIDE * SIDE * 4
    for case in valid_cases():
        (directory / case["file"]).write_bytes(case["bytes"])
        entry = {k: v for k, v in case.items() if k != "bytes"}
        entry["packed_bytes"] = len(case["bytes"])
        entry["raw_bytes"] = raw_bytes
        entry["sha256"] = hashlib.sha256(case["bytes"]).hexdigest()
        # A page entry shaped like a real atlas manifest row, so a decoder can read
        # these vectors through its normal page path rather than a test-only one.
        entry["page"] = {
            "path": case["file"], "level": case["level"], "x": 0, "y": 0, "side": SIDE,
            "x_m": 0.0, "z_m": 0.0, "width_m": float((SIDE - 1) * 10.0), "spacing_m": 10.0,
            "packed_bytes": entry["packed_bytes"], "raw_bytes": raw_bytes,
            "sha256": entry["sha256"], "min_m": entry["expected_min_m"],
            "max_m": entry["expected_max_m"], "sample_error_m": 0.0,
        }
        manifest["valid"].append(entry)
    for case in invalid_cases():
        (directory / case["file"]).write_bytes(case["bytes"])
        digest = hashlib.sha256(case["bytes"]).hexdigest()
        # Correct length and hash on purpose: the payload must be refused by the
        # decoder itself, not rejected earlier by an integrity check.
        manifest["invalid"].append({
            "name": case["name"], "file": case["file"],
            "packed_bytes": len(case["bytes"]), "sha256": digest, "why": case["why"],
            "page": {"path": case["file"], "level": 0, "x": 0, "y": 0, "side": SIDE,
                     "x_m": 0.0, "z_m": 0.0, "width_m": float((SIDE - 1) * 10.0), "spacing_m": 10.0,
                     "packed_bytes": len(case["bytes"]), "raw_bytes": raw_bytes,
                     "sha256": digest, "min_m": 0.0, "max_m": 0.0, "sample_error_m": 0.0},
        })
    manifest["corrupt_hash_case"] = {
        "file": manifest["valid"][0]["file"],
        "why": "A page whose bytes do not match the manifest hash must be refused before decoding",
        "page": dict(manifest["valid"][0]["page"], sha256="0" * 64),
    }
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    print(write())
