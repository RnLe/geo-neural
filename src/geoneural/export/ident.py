"""Compact data for the browser's identifiability view, from the inverse ridge-profile report."""
from __future__ import annotations

import json
from pathlib import Path

from geoneural.common import read_json

DESIGNS = ("terminal", "fractional", "fixed-lag-20k", "fixed-lag-50k")
VARIANTS = ("rescaledDt", "fixedDtCap")


def export(report: Path, out: Path) -> Path:
    """Expected log-likelihood difference against the common rate factor c, per landscape age, time-step
    handling and survey design, with the 2-log-unit interval where the report has it."""
    r = read_json(report)["results"]
    regimes = {}
    for name, truth in r["truths"].items():
        variants = {}
        for v in VARIANTS:
            block = truth["variants"][v]
            variants[v] = {"c": block["cGrid"],
                           "designs": {d: {"delta": [round(x, 3) for x in block["designs"][d]["deltaLogLikExpected"]],
                                           "interval": (block["designs"][d].get("interval2Expected") or {}).get("interval")}
                                       for d in DESIGNS if d in block["designs"]}}
        regimes[name] = {"years": truth["years"], "reliefM": round(truth["reliefM"], 1), "variants": variants}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"schema": "geoneural-ident-v1", "lagsYears": r["setup"]["lagsYears"],
                               "regimes": regimes}, separators=(",", ":")))
    return out
