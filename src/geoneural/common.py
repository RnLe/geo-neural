"""Paths, reproducible receipts and bounded local writes."""
from __future__ import annotations
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PACKAGE = Path(__file__).resolve().parent
# Downloads, atlases, reports and checkpoints. Large and machine-local, so never in Git.
HOME = Path(os.environ.get("GEONEURAL_HOME", ".data")).expanduser().resolve()
CONFIG = PACKAGE / "configs" / "regions.json"
# Confirmation regions, written by data/cohort.py after the selection rule was committed.
COHORT_CONFIG = PACKAGE / "configs" / "cohort-regions.json"
COHORT_B_CONFIG = PACKAGE / "configs" / "cohort-b-regions.json"


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path, max_bytes: int = 32 * 1024 * 1024) -> Any:
    if path.stat().st_size > max_bytes:
        raise ValueError(f"JSON exceeds {max_bytes} bytes: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def region(name: str) -> dict:
    choices = read_json(CONFIG)
    for extra in (COHORT_CONFIG, COHORT_B_CONFIG):
        if extra.exists():
            choices.update(read_json(extra))
    if name not in choices:
        raise ValueError(f"Unknown preset {name}; choose from {', '.join(choices)}")
    value = dict(choices[name])
    value["preset"] = name
    return value


def environment() -> dict:
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PACKAGE, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    import importlib.metadata as metadata
    versions = {}
    for name in ("numpy", "rasterio", "pyproj", "requests", "torch", "safetensors", "zstandard"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return {"python": sys.version, "platform": platform.platform(), "git": revision, "packages": versions}


def safe_child(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Path escapes the atlas root")
    return path
