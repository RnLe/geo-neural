"""Run identity for every artifact: environment, package freeze, device and filesystem facts.

`common.environment()` records the interpreter, platform, git revision and a short
version probe. That is not enough to reproduce or to compare measurements, which
also need the resolved package set, the device a
result was produced on, and the filesystem kind behind every artifact path,
because WSL ext4, Windows-local NTFS and Windows access through a WSL share have
different IO paths and their differences must never be ascribed to a codec.

Nothing here asserts that a run was clean. `run_record` reports what was observed.
"""
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from geoneural.common import PACKAGE, digest, environment, utc

MAX_FREEZE_ENTRIES = 2048
PROBE_TIMEOUT_S = 20


def _probe(command: list[str], timeout: int = PROBE_TIMEOUT_S) -> str | None:
    """Run a short read-only probe. Absence is recorded as None, never as a failure."""
    try:
        out = subprocess.run(command, capture_output=True, text=True, timeout=timeout, cwd=PACKAGE)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def pip_freeze(executable: str | None = None) -> list[str] | None:
    """Resolved package set of the interpreter that produced an artifact.

    The dependency ranges in pyproject.toml are not a lock, so the actual resolution
    belongs with the experiment. Another interpreter is asked through pip; this one
    is read from its installed distributions, which also works without pip.
    """
    if executable is not None and executable != sys.executable:
        text = _probe([executable, "-m", "pip", "freeze", "--disable-pip-version-check"])
        if text is None:
            return None
        entries = [line.strip() for line in text.splitlines() if line.strip()]
    else:
        import importlib.metadata as metadata
        entries = sorted({f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions()
                          if d.metadata["Name"]}, key=str.lower)
    if len(entries) > MAX_FREEZE_ENTRIES:
        raise ValueError(f"package listing has {len(entries)} entries; refusing an unbounded record")
    return entries


def cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def gpu() -> list[dict] | None:
    """Identify visible NVIDIA devices. A present device is not a used device."""
    text = _probe(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    if text is None:
        return None
    devices = []
    for line in text.splitlines()[:16]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            devices.append({"name": parts[0], "memory_total": parts[1], "driver": parts[2]})
    return devices or None


def _mounts() -> list[tuple[str, str]]:
    try:
        rows = Path("/proc/mounts").read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    table = []
    for row in rows:
        parts = row.split()
        if len(parts) >= 3:
            table.append((parts[1], parts[2]))
    return table


def filesystem(path: Path) -> dict:
    """Mount point and filesystem kind behind a path.

    ext4, 9p/drvfs and NTFS behave differently enough that IO observations taken
    on one must not be pooled with another. `kind` is the kernel's name, not a
    performance claim.
    """
    target = Path(path).resolve()
    best = ("", "unknown")
    for point, kind in _mounts():
        if target == Path(point) or str(target).startswith(point.rstrip("/") + "/"):
            if len(point) >= len(best[0]):
                best = (point, kind)
    return {"path": str(target), "mount": best[0] or "unknown", "kind": best[1]}


def run_record(kind: str, inputs: dict[str, Any] | None = None, paths: dict[str, Path] | None = None) -> dict:
    """Complete identity of one observation.

    `kind` names the experiment family (for example 'codec-tournament'). `inputs`
    holds the content identities the run depends on. `paths` names artifact
    locations whose filesystem kind must be recorded.
    """
    if not kind or len(kind) > 128:
        raise ValueError("Run kind must be a short non-empty label")
    record = {
        "schema": "geoneural-run-v1",
        "kind": kind,
        "created_utc": utc(),
        "pid": os.getpid(),
        "inputs": dict(inputs or {}),
        "environment": environment(),
        "packages_frozen": pip_freeze(),
        "cpu": cpu_model(),
        "cpu_count": os.cpu_count(),
        "gpu": gpu(),
        "filesystems": {name: filesystem(p) for name, p in (paths or {}).items()},
        "qualification": (
            "Observed run identity only. Presence of a device does not mean it was used, "
            "and this record does not establish host isolation or an idle machine."
        ),
    }
    record["run_id"] = digest({k: record[k] for k in ("kind", "created_utc", "pid", "inputs")})
    return record
