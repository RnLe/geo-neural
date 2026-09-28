"""Experiment records shared by every v2 report.

A record says what was run (task, product, recipe, split, seeds), on which source revision, what was measured,
which gates passed with which thresholds, and what failed. Evidence states:

* exploratory: development data, settings still moving; never a headline.
* selected: chosen on development data by a declared rule; still development evidence.
* confirmed: frozen recipe run once on the untouched confirmation cohort.
* invalid: kept for inspection, excluded from every claim (the reason is in `failures`).
* historical: a v1 report kept for continuity; its comparisons used an incomplete byte contract.

`validate` refuses records that would let a claim through without the information needed to check it.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from geoneural.common import digest, environment, utc, write_json

SCHEMA = "geoneural-experiment-v2"
TASKS = ("compression", "reconstruction", "dynamics", "inverse", "audit")
EVIDENCE = ("exploratory", "selected", "confirmed", "invalid", "historical")
GATE_STATES = ("pass", "fail", "not-run")


def gate(state: str, detail: str = "", **thresholds: Any) -> dict:
    """One gate outcome with the thresholds it was judged against."""
    if state not in GATE_STATES:
        raise ValueError(f"gate state {state!r} not in {GATE_STATES}")
    return {"state": state, "detail": detail, "thresholds": thresholds}


def record(task: str, name: str, evidence: str, *, results: Any, recipe: dict | None = None,
           split: dict | None = None, seeds: list[int] | None = None, product: dict | None = None,
           gates: dict[str, dict] | None = None, failures: list[dict] | None = None,
           timing: dict | None = None, notes: str = "") -> dict:
    env = environment()
    rec = {
        "schema": SCHEMA,
        "task": task,
        "name": name,
        "evidence": evidence,
        "createdUtc": utc(),
        "sourceRevision": env.get("git", {}).get("revision") if isinstance(env.get("git"), dict) else env.get("git"),
        "environment": env,
        "recipe": recipe or {},
        "recipeHash": digest(recipe or {}),
        "split": split or {},
        "splitHash": digest(split or {}),
        "seeds": seeds or [],
        "product": product,
        "results": results,
        "gates": gates or {},
        "timing": timing or {"qualified": False, "note": "not measured"},
        "failures": failures or [],
        "notes": notes,
    }
    validate(rec)
    return rec


def validate(rec: dict) -> None:
    if rec.get("schema") != SCHEMA:
        raise ValueError("not a v2 experiment record")
    if rec.get("task") not in TASKS:
        raise ValueError(f"task {rec.get('task')!r} not in {TASKS}")
    if rec.get("evidence") not in EVIDENCE:
        raise ValueError(f"evidence {rec.get('evidence')!r} not in {EVIDENCE}")
    for key, g in (rec.get("gates") or {}).items():
        if g.get("state") not in GATE_STATES:
            raise ValueError(f"gate {key} has no valid state")
    if rec["evidence"] == "confirmed":
        if not rec.get("seeds"):
            raise ValueError("a confirmed record needs its seeds")
        if not rec.get("split"):
            raise ValueError("a confirmed record needs its split")
        if any(g["state"] == "not-run" for g in rec.get("gates", {}).values()):
            raise ValueError("a confirmed record cannot carry gates that were not run")
    _finite(rec.get("results"), "results")


def _finite(value: Any, where: str) -> None:
    """NaN and infinity never stand in for a failed or missing measurement: use None and say why."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite number at {where}; record None and a failure instead")
    if isinstance(value, dict):
        for k, v in value.items():
            _finite(v, f"{where}.{k}")
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            _finite(v, f"{where}[{i}]")


def write(rec: dict, path: Path) -> Path:
    validate(rec)
    write_json(path, rec)
    return path
