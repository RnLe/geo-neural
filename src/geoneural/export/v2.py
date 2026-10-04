"""v2 results: merge the campaign reports, apply the frozen decision rules, write the summary and figures.

Decision rules are those of docs/protocol-v2.md. They are applied here mechanically, so a reader can check a
verdict against the numbers next to it.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from geoneural.common import read_json, write_json
from geoneural.evaluation.statistics import paired_log_ratio

PRIMARY = (0.05, 0.1, 0.25, 0.5, 1.0)


def _mean_by_region(rows, pred, key="bytes"):
    vals: dict[str, list[float]] = {}
    for r in rows:
        if pred(r):
            vals.setdefault(r["region"], []).append(r[key])
    return {k: float(np.mean(v)) for k, v in vals.items()}


def _f1(r):
    return (r.get("drainage") or {}).get("tolerantF1")


def best_conventional(rows, bound, region):
    cands = [r for r in rows if r.get("family") == "conventional" and r["boundM"] == bound and r["region"] == region
             and r.get("product", "standalone") == "standalone" and r.get("boundViolations", 1) == 0]
    return min(cands, key=lambda r: r["bytes"]) if cands else None


def h1_verdict(rows: list[dict], bounds=PRIMARY, threshold: float = 0.90, upper: float = 0.95,
               f1_drop: float = 0.01) -> dict:
    """Per bound: learned standalone against the best conventional product and against cubic-ctx."""
    out = {"perBound": {}, "rule": {"maxRatio": threshold, "bootstrapUpperBelow": upper, "allRegionsSmaller": True,
                                    "maxTolerantF1Drop": f1_drop}}
    passes = []
    for b in bounds:
        regions = sorted({r["region"] for r in rows if r["boundM"] == b})
        best = {g: best_conventional(rows, b, g) for g in regions}
        conv = {g: r["bytes"] for g, r in best.items() if r}
        learned = _mean_by_region(rows, lambda r: r["boundM"] == b and r.get("coder") == "learned"
                                  and r.get("product") == "standalone")
        corpus = _mean_by_region(rows, lambda r: r["boundM"] == b and r.get("coder") == "learned"
                                 and r.get("product") == "corpus")
        ctx = _mean_by_region(rows, lambda r: r["boundM"] == b and r.get("coder") == "cubic-ctx")
        f1_learned = _mean_by_region(rows, lambda r: r["boundM"] == b and r.get("coder") == "learned"
                                     and r.get("product") == "standalone" and _f1(r) is not None, key="f1")
        f1_conv = {g: _f1(r) for g, r in best.items() if r and _f1(r) is not None}
        vs_conv = paired_log_ratio(learned, conv)
        verdict = (vs_conv.get("n", 0) > 0 and vs_conv["geometricMeanRatio"] <= threshold
                   and vs_conv["bootstrap95"][1] < upper and vs_conv["allRegionsSmaller"]
                   and all(f1_learned.get(g, 0) >= f1_conv[g] - f1_drop for g in f1_conv))
        passes.append(verdict)
        out["perBound"][str(b)] = {
            "bestConventionalCoder": {g: r["coder"] for g, r in best.items() if r},
            "learnedStandaloneVsBestConventional": vs_conv,
            "learnedCorpusVsBestConventional": paired_log_ratio(corpus, conv),
            "learnedStandaloneVsCubicCtx": paired_log_ratio(learned, ctx),
            "cubicCtxVsBestConventional": paired_log_ratio(ctx, conv),
            "tolerantF1": {"learned": f1_learned, "bestConventional": f1_conv},
            "passes": bool(verdict)}
    out["supported"] = bool(passes and all(passes))
    return out


def _with_f1(rows):
    for r in rows:
        if r.get("drainage") is not None:
            r["f1"] = r["drainage"].get("tolerantF1")
    return rows


def collect(root: Path) -> dict:
    """Every v2 report under root (the data root's results/v2), by name."""
    out = {}
    for path in sorted(root.rglob("*.json")):
        if path.parent.name == "models":
            continue
        try:
            rec = read_json(path)
        except Exception:  # noqa: BLE001 - a broken file is reported, not skipped silently
            out[str(path.relative_to(root))] = {"unreadable": True}
            continue
        if isinstance(rec, dict) and rec.get("schema") == "geoneural-experiment-v2":
            out[str(path.relative_to(root))] = rec
    return out


def codec_summary(reports: dict) -> dict:
    h1 = reports.get("codec/h1.json")
    if not h1:
        return {}
    rows = _with_f1(list(h1["results"]["rows"]))
    best_sz3 = reports.get("codec/sz3-best.json")
    if best_sz3:
        rows += _with_f1([r for r in best_sz3["results"]["rows"] if "bytes" in r])
    out = {"h1": h1_verdict(rows), "h1All": h1_verdict(rows, bounds=sorted({r["boundM"] for r in rows}))}
    return out


def build(data_root: Path, out_dir: Path) -> Path:
    reports = collect(data_root)
    summary = {"schema": "geoneural-v2-summary", "reports": sorted(reports), "codec": codec_summary(reports)}
    write_json(out_dir / "summary.json", summary)
    return out_dir / "summary.json"


def fmt_ratio(r: float) -> str:
    return f"{100 * (r - 1):+.1f}%" if r is not None and math.isfinite(r) else "n/a"


NICE = {"essen-ruhr": "Essen", "muensterland-plain": "Muensterland", "lower-rhine": "Lower Rhine",
        "teutoburg-forest": "Teutoburg Forest", "bergisches-land": "Bergisches Land",
        "rothaar-sauerland": "Rothaar"}


def h1_figure(rows: list[dict], out: Path, title: str) -> Path:
    """Learned standalone bytes over the best conventional product, per region, against the bound."""
    from geoneural.export.figures import line_svg
    regions = sorted({r["region"] for r in rows})
    bounds = sorted({r["boundM"] for r in rows})
    series = []
    for g in regions:
        pts = []
        for b in bounds:
            best = best_conventional(rows, b, g)
            learned = [r["bytes"] for r in rows if r["region"] == g and r["boundM"] == b
                       and r.get("coder") == "learned" and r.get("product") == "standalone"]
            if best and learned:
                pts.append((b, float(np.mean(learned)) / best["bytes"]))
        series.append((NICE.get(g, g), pts, ""))
    pts = []
    for b in bounds:
        vals = []
        for g in regions:
            best = best_conventional(rows, b, g)
            ctx = [r["bytes"] for r in rows if r["region"] == g and r["boundM"] == b and r.get("coder") == "cubic-ctx"]
            if best and ctx:
                vals.append(np.log(ctx[0] / best["bytes"]))
        if vals:
            pts.append((b, float(np.exp(np.mean(vals)))))
    series.append(("cubic-ctx (mean)", pts, "5 4"))
    svg = line_svg(series, title, "maximum error bound (m)", "bytes / best conventional",
                   refs=((1.0, "equal"), (0.9, "10% smaller")), xticks=(0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg)
    return out
