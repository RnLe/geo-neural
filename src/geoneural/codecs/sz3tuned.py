"""SZ3 at its best for each field: the smallest of several configurations, each decoded and bound-checked.

The default SZ3 product (interpolation with Lorenzo and SZ3's own tuning) comes from imagecodecs. This module runs
the SZ3 command-line tool of the same release over a grid of configurations (algorithm, interpolation order and
direction, level-wise error-bound factors) and keeps the smallest stream that holds the bound. Choosing per field
uses the encoder's knowledge of the field, which is allowed; the SZ3 stream carries its configuration, so the
decoder needs nothing else. This is the strongest SZ3 comparator we could set up, not a typical setting.

The tool is not a Python dependency: point `GEONEURAL_SZ3` at an `sz3` binary built from the SZ3 sources (the
release is recorded with every result).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

CONFIGS = {
    "interp-lorenzo": {"CmprAlgo": "ALGO_INTERP_LORENZO"},
    "interp-cubic": {"CmprAlgo": "ALGO_INTERP", "InterpolationAlgo": "INTERP_ALGO_CUBIC"},
    "interp-linear": {"CmprAlgo": "ALGO_INTERP", "InterpolationAlgo": "INTERP_ALGO_LINEAR"},
    "interp-cubic-dir1": {"CmprAlgo": "ALGO_INTERP", "InterpolationAlgo": "INTERP_ALGO_CUBIC",
                          "InterpolationDirection": "1"},
    "lorenzo-reg": {"CmprAlgo": "ALGO_LORENZO_REG"},
    "lorenzo2-reg2": {"CmprAlgo": "ALGO_LORENZO_REG", "Lorenzo2ndOrder": "YES", "Regression2ndOrder": "YES"},
}
for alpha in ("1.25", "1.5", "1.75", "2"):
    for beta in ("2", "4"):
        CONFIGS[f"interp-cubic-a{alpha}-b{beta}"] = {"CmprAlgo": "ALGO_INTERP", "InterpolationAlgo": "INTERP_ALGO_CUBIC",
                                                     "InterpolationAlpha": alpha, "InterpolationBeta": beta}


def binary() -> str | None:
    path = os.environ.get("GEONEURAL_SZ3") or shutil.which("sz3")
    return path if path and Path(path).exists() else None


def version() -> str | None:
    exe = binary()
    if not exe:
        return None
    out = subprocess.run([exe, "-v"], capture_output=True, text=True)
    return (out.stdout or out.stderr).strip().splitlines()[0] if (out.stdout or out.stderr) else None


def run(field: np.ndarray, bound_m: float, config: dict) -> tuple[bytes, np.ndarray]:
    exe = binary()
    if exe is None:
        raise RuntimeError("set GEONEURAL_SZ3 to an sz3 binary")
    a = np.ascontiguousarray(field, np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "in.f32").write_bytes(a.tobytes())
        cfg = tmp / "sz3.config"
        cfg.write_text("[GlobalSettings]\n" + "".join(f"{k} = {v}\n" for k, v in config.items()
                                                       if k in ("CmprAlgo",))
                       + "[AlgoSettings]\n" + "".join(f"{k} = {v}\n" for k, v in config.items() if k != "CmprAlgo"))
        cmd = [exe, "-f", "-i", str(tmp / "in.f32"), "-z", str(tmp / "c.sz"), "-o", str(tmp / "out.f32"),
               "-2", str(a.shape[1]), str(a.shape[0]), "-c", str(cfg), "-M", "ABS", repr(float(bound_m))]
        subprocess.run(cmd, check=True, capture_output=True)
        return (tmp / "c.sz").read_bytes(), np.fromfile(tmp / "out.f32", np.float32).reshape(a.shape)


def best(field: np.ndarray, bound_m: float, configs=CONFIGS) -> dict:
    """Smallest stream over the configurations that keeps |error| <= bound (one float32 ulp allowed)."""
    ref = np.asarray(field, np.float64)
    ulp = np.spacing(np.abs(np.asarray(field, np.float32))).astype(np.float64)
    tried, best_row = {}, None
    for name, cfg in configs.items():
        try:
            blob, dec = run(field, bound_m, cfg)
        except subprocess.CalledProcessError as exc:
            tried[name] = {"error": (exc.stderr or b"")[:200].decode(errors="replace")}
            continue
        ok = bool((np.abs(dec.astype(np.float64) - ref) <= bound_m + ulp).all())
        tried[name] = {"bytes": len(blob), "holdsBound": ok}
        if ok and (best_row is None or len(blob) < len(best_row["blob"])):
            best_row = {"config": name, "blob": blob, "decoded": dec.astype(np.float64)}
    return {"best": best_row, "tried": tried}
