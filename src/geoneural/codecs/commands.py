"""CLI commands of the multilevel coder and its campaigns."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from geoneural.common import HOME


def summarise_h1(rows: list[dict]) -> dict:
    """Per bound: learned (corpus and standalone, mean over seeds) against the cubic control, SZ3 and the smallest
    conventional product that kept the bound on every node."""
    from geoneural.evaluation.statistics import paired_log_ratio
    out = {}
    for b in sorted({r["boundM"] for r in rows}):
        at = [r for r in rows if r["boundM"] == b]

        def per_region(pred):
            vals: dict[str, list[float]] = {}
            for r in at:
                if pred(r):
                    vals.setdefault(r["region"], []).append(r["bytes"])
            return {k: float(np.mean(v)) for k, v in vals.items()}

        conv = {}
        for r in at:
            if r["family"] == "conventional" and r["boundViolations"] == 0:
                conv[r["region"]] = min(conv.get(r["region"], np.inf), r["bytes"])
        best_name = {}
        for r in at:
            if r["family"] == "conventional" and r["boundViolations"] == 0 and r["bytes"] == conv.get(r["region"]):
                best_name[r["region"]] = r["coder"]
        learned_c = per_region(lambda r: r["coder"] == "learned" and r["product"] == "corpus")
        learned_s = per_region(lambda r: r["coder"] == "learned" and r["product"] == "standalone")
        ctx = per_region(lambda r: r["coder"] == "cubic-ctx")
        sz3 = per_region(lambda r: r["coder"] == "sz3" and r["boundViolations"] == 0)
        out[str(b)] = {"bestConventionalCoder": best_name,
                       "learnedCorpusVsBestConventional": paired_log_ratio(learned_c, conv),
                       "learnedStandaloneVsBestConventional": paired_log_ratio(learned_s, conv),
                       "learnedCorpusVsCubicCtx": paired_log_ratio(learned_c, ctx),
                       "cubicCtxVsBestConventional": paired_log_ratio(ctx, conv),
                       "learnedCorpusVsSz3": paired_log_ratio(learned_c, sz3)}
    return out


def cmd_codec_h1(a):
    from geoneural.codecs import campaign, foreign
    from geoneural.evaluation import protocol
    out = a.out or HOME / "results" / "v2" / "codec"
    rows = campaign.h1(out, tuple(a.regions), tuple(a.bounds), tuple(a.seeds))
    rec = protocol.record(
        "compression", "h1-learned-multilevel", "exploratory",
        results={"rows": rows, "summary": summarise_h1(rows)},
        recipe={"predictor": {"widths": [32, 32], "steps": 5000, "rounds": 2, "loss": "Laplace code length"},
                "lattice_m": 0.001, "bounds": list(a.bounds), "codecVersions": foreign.versions(),
                "product": "standalone 1025 x 1025 raster; learned also as corpus with the model counted once"},
        split={"scheme": "leave one region out", "regions": list(a.regions)}, seeds=list(a.seeds),
        timing={"qualified": False, "note": "shared machine; encode/decode seconds are indicative only"})
    print(protocol.write(rec, out / "h1.json"))


DECISION_BOUNDS = (0.05, 0.1, 0.25, 0.5, 1.0)


def _standalone(r: dict, free: bool = False) -> float:
    """Bytes of the standalone file: the corpus product plus its model and the model's 9-byte directory entry.
    With `free`, the context raster is taken as already at the decoder."""
    return r["bytes"] + r["sharedModelBytes"] + 9 - (r["contextBytes"] if free else 0)


def summarise_h2(rows: list[dict]) -> dict:
    """Per bound: standalone bytes of each context arm against the matched no-context arm, regions as units, and
    the protocol's H2 test. Operational form of "gain absent within seed spread" (fixed before the results): the
    seed spread s is the mean over regions of the standard deviation over seeds of log(real / none); a control
    shows no gain when its geometric-mean ratio is at least exp(-s)."""
    from geoneural.evaluation.statistics import paired_log_ratio
    out = {}
    for b in sorted({r["boundM"] for r in rows}):
        at = [r for r in rows if r["boundM"] == b]

        def per_region(arm, free=False, seed=None):
            vals: dict[str, list[float]] = {}
            for r in at:
                if r["arm"] == arm and (seed is None or r["seed"] == seed):
                    vals.setdefault(r["region"], []).append(_standalone(r, free))
            return {k: float(np.mean(v)) for k, v in vals.items()}

        seeds = sorted({r["seed"] for r in at})
        spread = []
        for region in sorted({r["region"] for r in at}):
            logs = [np.log(per_region("real", seed=s)[region] / per_region("none", seed=s)[region]) for s in seeds
                    if region in per_region("real", seed=s) and region in per_region("none", seed=s)]
            if len(logs) > 1:
                spread.append(float(np.std(logs, ddof=1)))
        s = float(np.mean(spread)) if spread else 0.0
        entry = {"seedSpreadLog": s}
        for free, key in ((False, "charged"), (True, "mapAtDecoder")):
            none = per_region("none", free)
            ratios = {arm: paired_log_ratio(per_region(arm, free), none) for arm in
                      ("constant", "real", "shifted", "shuffled", "wrong-region")}
            real = ratios["real"].get("geometricMeanRatio")
            controls = [ratios[a].get("geometricMeanRatio") for a in ("shifted", "shuffled")]
            passed = (real is not None and real <= 0.99 and all(c is not None and c >= np.exp(-s) for c in controls))
            entry[key] = {**ratios, "pass": bool(passed)}
        out[str(b)] = entry
    return out


def summarise_h4(rows: list[dict]) -> dict:
    """The protocol's H4 test per bound: process-real at least 2% below both real-extra and procedural-real, with
    the bootstrap interval excluding no change. All arms share the model size, so corpus bytes decide."""
    vs = {ref: summarise_arms(rows, ref, ("process-real", ref)) for ref in ("real-extra", "procedural-real")}
    out = {}
    for b in sorted({r["boundM"] for r in rows}):
        tests = [vs[ref][str(b)]["process-real"] for ref in vs]
        ok = all(t.get("n") and t["geometricMeanRatio"] <= 0.98 and t["bootstrap95"][1] < 1.0 for t in tests)
        out[str(b)] = {"vsRealExtra": tests[0], "vsProceduralReal": tests[1], "pass": bool(ok)}
    return out


def _merge_parts(paths) -> list[dict]:
    import json
    rows = []
    for path in paths:
        rows += json.loads(Path(path).read_text())["rows"]
    return rows


def _write_part(rows: list[dict], path: Path) -> None:
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"rows": rows}))
    print(f"part with {len(rows)} rows: {path}")


def cmd_codec_h2(a):
    from geoneural.codecs import campaign
    from geoneural.evaluation import protocol
    out = a.out or HOME / "results" / "v2" / "codec"
    if a.merge:
        rows = _merge_parts(a.merge)
    else:
        rows = campaign.h2(out, tuple(a.regions), tuple(a.bounds), tuple(a.seeds), held_out=a.held)
        if a.part:
            return _write_part(rows, a.part)
    summary = summarise_h2(rows)
    decided = [summary[str(b)]["charged"]["pass"] for b in DECISION_BOUNDS if str(b) in summary]
    rec = protocol.record(
        "compression", "h2-geology", "selected", results={"rows": rows, "summary": summary},
        recipe={"start": "H1 leave-one-region-out model of the same fold and seed",
                "extraTraining": "one closed-loop round, 3000 steps, identical for every arm",
                "context": "GK100 material classes, 40 m raster, nearest node, 4-dimensional embedding",
                "charged": sorted(campaign.CHARGED),
                "bytes": "standalone: corpus file plus model plus its 9-byte directory entry"},
        split={"scheme": "leave one region out", "regions": list(a.regions)}, seeds=sorted({r["seed"] for r in rows}),
        gates={"h2": protocol.gate("pass" if decided and all(decided) else "fail", "protocol-v2 H2 rule, development",
                                   maxRatio=0.99, controls="ratio >= exp(-seed spread)")})
    print(protocol.write(rec, out / "h2.json"))


def summarise_levels(rows: list[dict], conventional: list[dict]) -> dict:
    """Level-wise bounds against the best conventional product of the same region and bound (H1 development rows
    plus sz3-best). Each arm is reported on its own, and as a leave-one-region-out choice: for each region the arm is
    picked on the other regions (smallest worst-region stream-F1 gap among arms whose bytes there are at most 0.90 of
    the best conventional, uniform if none), then scored on the held region and judged by the frozen H1 rule."""
    from geoneural.evaluation.statistics import paired_log_ratio
    f1 = lambda r: r["drainage"]["tolerantF1"] or 0.0
    out = {}
    for b in sorted({r["boundM"] for r in rows}):
        best = {}
        for r in conventional:
            if r["boundM"] == b and r.get("family") == "conventional" and r.get("boundViolations") == 0:
                if r["region"] not in best or r["bytes"] < best[r["region"]]["bytes"]:
                    best[r["region"]] = r
        entry = {}
        for coder in sorted({r["coder"] for r in rows}):
            arms = sorted({r["arm"] for r in rows if r["coder"] == coder})
            cell = {}
            for arm in arms:
                sel = [r for r in rows if r["coder"] == coder and r["arm"] == arm and r["boundM"] == b]
                regions = sorted({r["region"] for r in sel} & set(best))
                by = {g: float(np.mean([r["bytes"] for r in sel if r["region"] == g])) for g in regions}
                gap = {g: float(np.mean([f1(r) for r in sel if r["region"] == g])) - f1(best[g]) for g in regions}
                rmse = {g: float(np.mean([r["rmseM"] for r in sel if r["region"] == g])) / best[g]["rmseM"]
                        for g in regions}
                cell[arm] = {"bytes": by, "f1Gap": gap, "rmseRatio": rmse,
                             "vsBest": paired_log_ratio(by, {g: best[g]["bytes"] for g in regions})}
            chosen, held_bytes, held_gap = {}, {}, {}
            for g in sorted(set.intersection(*[set(c["bytes"]) for c in cell.values()])):
                others = [h for h in cell["uniform"]["bytes"] if h != g]
                ok = []
                for arm, c in cell.items():
                    ratio = float(np.exp(np.mean([np.log(c["bytes"][h] / best[h]["bytes"]) for h in others])))
                    if ratio <= 0.90:
                        ok.append((min(c["f1Gap"][h] for h in others), arm))
                arm = max(ok)[1] if ok else "uniform"
                chosen[g], held_bytes[g], held_gap[g] = arm, cell[arm]["bytes"][g], cell[arm]["f1Gap"][g]
            stat = paired_log_ratio(held_bytes, {g: best[g]["bytes"] for g in held_bytes})
            passed = bool(stat.get("n") and stat["geometricMeanRatio"] <= 0.90 and stat["bootstrap95"][1] < 0.95
                          and stat["allRegionsSmaller"] and min(held_gap.values()) >= -0.01)
            entry[coder] = {"arms": cell, "leaveOneRegionOut": {"chosen": chosen, "vsBest": stat, "f1Gap": held_gap,
                                                                "worstF1Gap": min(held_gap.values()),
                                                                "h1RulePass": passed}}
        out[str(b)] = entry
    return out


def cmd_codec_levels(a):
    from geoneural.codecs import campaign
    from geoneural.common import read_json
    from geoneural.evaluation import protocol
    out = a.out or HOME / "results" / "v2" / "codec"
    if a.merge:
        rows = _merge_parts(a.merge)
    else:
        rows = campaign.levels(out, tuple(a.regions), seeds=tuple(a.seeds), held_out=a.held)
        if a.part:
            return _write_part(rows, a.part)
    conventional = read_json(out / "h1.json")["results"]["rows"]
    if (out / "sz3-best.json").exists():
        conventional += read_json(out / "sz3-best.json")["results"]["rows"]
    rec = protocol.record(
        "compression", "levels", "exploratory", results={"rows": rows, "summary": summarise_levels(rows, conventional)},
        recipe={"rules": campaign.LEVEL_RULES, "bounds": list(campaign.LEVEL_BOUNDS),
                "learned": "H1 leave-one-region-out models, standalone files with the model inside",
                "selection": "leave one region out: smallest worst-region F1 gap among arms at <= 0.90 bytes"},
        split={"scheme": "leave one region out", "regions": list(a.regions)}, seeds=list(a.seeds),
        notes="Exploratory and development only: a variant designed after the H1 confirmation needs a new cohort.")
    print(protocol.write(rec, out / "levels.json"))


def choose_levels(summary: dict) -> dict:
    """The frozen choice per bound, on all development regions with the leave-one-region-out criterion: the arm
    with the smallest worst-region stream-F1 gap among arms whose bytes are at most 0.90 of the best conventional
    product (geometric mean over regions); uniform if none qualifies."""
    out = {}
    for b, entry in summary.items():
        ok = [(min(c["f1Gap"].values()), arm) for arm, c in entry["learned"]["arms"].items()
              if c["vsBest"].get("n") and c["vsBest"]["geometricMeanRatio"] <= 0.90]
        out[b] = max(ok)[1] if ok else "uniform"
    return out


def cmd_codec_freeze_levels(a):
    from geoneural.codecs import campaign
    from geoneural.common import read_json, utc, write_json
    out = a.out or HOME / "results" / "v2" / "codec"
    summary = read_json(out / "levels.json")["results"]["summary"]
    chosen = choose_levels(summary)
    rules = {b: campaign.LEVEL_RULES.get(arm) for b, arm in chosen.items()}
    models = {s: campaign.Predictor.from_bytes((out / "models" / f"final-s{s}.gnm").read_bytes()).sha256()
              for s in (0, 1, 2)}
    recipe = {"schema": "geoneural-recipe-freeze-v1", "hypothesis": "H1b (level-wise bounds)", "frozenUtc": utc(),
              "chosenArm": chosen, "rules": rules, "models": models,
              "rule": "docs/protocol-v2.md, H1 rule unchanged (bounds 0.05 to 1 m)",
              "cohort": "second cohort, data/cohort.py RULE_B, drawn after this freeze",
              "command": "geoneural codec-confirm-levels --cohort-config <cohort B presets>"}
    write_json(a.recipe, recipe)
    print(a.recipe, chosen)


def cmd_codec_confirm_levels(a):
    from geoneural.codecs import campaign, foreign, sz3tuned
    from geoneural.common import read_json
    from geoneural.evaluation import protocol
    from geoneural.export.v2 import h1_verdict
    out = a.out or HOME / "results" / "v2" / "codec"
    recipe = read_json(a.recipe)
    rules = {float(b): r for b, r in recipe["rules"].items()}
    for s, sha in recipe["models"].items():
        if campaign.Predictor.from_bytes((out / "models" / f"final-s{s}.gnm").read_bytes()).sha256() != sha:
            raise RuntimeError(f"model final-s{s} differs from the frozen recipe")
    cohort = list(read_json(a.cohort_config))
    rows = campaign.confirm_levels(out, cohort, rules)
    for r in rows:
        if r.get("drainage"):
            r["f1"] = r["drainage"].get("tolerantF1")
    verdict = h1_verdict(rows)
    rec = protocol.record(
        "compression", "h1b-levels-confirmation", "confirmed", results={"rows": rows, "verdict": verdict},
        recipe={"frozen": recipe, "codecVersions": foreign.versions(), "sz3": sz3tuned.version()},
        split={"scheme": "frozen models and rules from the six development regions", "cohort": cohort},
        seeds=[0, 1, 2],
        gates={"h1b": protocol.gate("pass" if verdict["supported"] else "fail", "protocol-v2 H1 rule, level-wise bounds",
                                    maxRatio=0.90, bootstrapUpperBelow=0.95, maxTolerantF1Drop=0.01)})
    print(protocol.write(rec, out / "h1b-levels-confirmation.json"))


def cmd_cohort_select(a):
    from geoneural.common import write_json
    from geoneural.data import cohort
    rule = {"a": cohort.RULE, "b": cohort.RULE_B}[a.rule]
    result = cohort.select(a.out, rule, max_probes=a.max_probes)
    write_json(a.presets, cohort.presets(result))
    print(a.presets, [c["name"] for c in result["selected"]], result["quotaLeft"])


def summarise_frontier(rows: list[dict], h1_rows: list[dict]) -> dict:
    """C1 (configuration chosen per region on the other regions), C3 paired with its C1 configuration, C2 against
    the uniform curve of the same codec at equal bytes, and the per-tile portfolio of the learned coder and the
    best conventional product (a one-byte selection flag charged), reported apart from the fixed neural mode."""
    from geoneural.evaluation.statistics import paired_log_ratio
    f1 = lambda r: (r.get("drainage") or {}).get("tolerantF1") or 0.0
    out = {}
    for b in sorted({r["boundM"] for r in rows if r.get("family") != "uniform"}):
        best = {}
        for r in h1_rows:
            if r["boundM"] == b and r.get("family") == "conventional" and r.get("boundViolations") == 0:
                if r["region"] not in best or r["bytes"] < best[r["region"]]["bytes"]:
                    best[r["region"]] = r
        def per_region(pred):
            vals: dict[str, list[float]] = {}
            for r in h1_rows:
                if r["boundM"] == b and pred(r):
                    vals.setdefault(r["region"], []).append(r["bytes"])
            return {g: float(np.mean(v)) for g, v in vals.items()}
        learned = per_region(lambda r: r.get("coder") == "learned" and r.get("product") == "standalone")
        ctx = per_region(lambda r: r.get("coder") == "cubic-ctx")
        conv = {g: r["bytes"] for g, r in best.items()}
        at = {(r["region"], r["arm"]): r for r in rows if r["boundM"] == b}
        regions = sorted({g for g, _ in at})
        c1_arms = sorted({a for _, a in at if a.startswith("c1-")})
        gm = lambda arm, gs: float(np.exp(np.mean([np.log(at[(g, arm)]["bytes"] / conv[g]) for g in gs])))
        chosen = {g: min(c1_arms, key=lambda a: gm(a, [h for h in regions if h != g])) for g in regions}
        c1 = {g: at[(g, chosen[g])]["bytes"] for g in regions}
        c1_best_fixed = min(c1_arms, key=lambda a: gm(a, regions))
        pair = "c1-f4-m1.0-sz3"
        c3 = {arm: paired_log_ratio({g: at[(g, arm)]["bytes"] for g in regions},
                                    {g: at[(g, pair)]["bytes"] for g in regions})
              for arm in ("c3-morph", "c3-geology", "c3-geology-shifted")}
        c2 = {}
        for arm in sorted({a for _, a in at if a.startswith("c2-")}):
            codec = arm.split("-")[1]
            deltas, same, overhead = {}, {}, {}
            for g in regions:
                r = at[(g, arm)]
                uni = sorted((u for u in rows if u["region"] == g and u["arm"] == f"uniform-{codec}"),
                             key=lambda u: u["bytes"])
                xs = np.log([u["bytes"] for u in uni])
                x = np.log(r["bytes"])
                if xs[0] <= x <= xs[-1]:
                    deltas[g] = f1(r) - float(np.interp(x, xs, [f1(u) for u in uni]))
                base = next(u for u in uni if u["boundM"] == b)
                same[g] = f1(r) - f1(base)
                overhead[g] = r["bytes"] / base["bytes"] - 1
            c2[arm] = {"deltaF1AtEqualBytes": deltas, "meanDeltaAtEqualBytes": float(np.mean(list(deltas.values())))
                       if deltas else None, "deltaF1SameBound": same, "byteOverhead": overhead}
        portfolio = {g: min(learned[g], conv[g]) + 1 for g in learned if g in conv}
        out[str(b)] = {
            "c1LeaveOneRegionOut": {"chosen": chosen, "vsBestConventional": paired_log_ratio(c1, conv),
                                    "vsCubicCtx": paired_log_ratio(c1, ctx), "vsLearned": paired_log_ratio(c1, learned)},
            "c1BestFixed": {"arm": c1_best_fixed, "ratio": gm(c1_best_fixed, regions)},
            "c3VsC1": c3,
            "c2": c2,
            "portfolio": {"vsBestConventional": paired_log_ratio(portfolio, conv),
                          "vsLearned": paired_log_ratio(portfolio, learned),
                          "conventionalChosen": sorted(g for g in portfolio if conv[g] < learned[g])},
        }
    return out


def cmd_codec_frontier(a):
    from geoneural.codecs import campaign
    from geoneural.common import read_json
    from geoneural.evaluation import protocol
    out = a.out or HOME / "results" / "v2" / "codec"
    if a.merge:
        rows = _merge_parts(a.merge)
    else:
        rows = campaign.frontier(out, tuple(a.regions), held_out=a.held)
        if a.part:
            return _write_part(rows, a.part)
    h1_rows = read_json(out / "h1.json")["results"]["rows"]
    if (out / "sz3-best.json").exists():
        h1_rows += [r for r in read_json(out / "sz3-best.json")["results"]["rows"] if "bytes" in r]
    rec = protocol.record(
        "compression", "conventional-frontier", "selected",
        results={"rows": rows, "summary": summarise_frontier(rows, h1_rows)},
        recipe={"c1Grid": campaign.C1_GRID, "c3": "factor 4, base bound = target, ridge 1e-3 n, sz3 residual",
                "c2": "uniform product plus corrections to bound/4 on stream band 0, band 1 or small-margin cells",
                "uniformGrid": list(campaign.UNIFORM_GRID)},
        split={"scheme": "leave one region out for the C1 choice", "regions": list(a.regions)})
    print(protocol.write(rec, out / "frontier.json"))


def summarise_sensitivity(rows: list[dict]) -> dict:
    """Worst-region and mean stream-F1 gap (learned minus best conventional, learned averaged over seeds) per
    cohort, boundary policy, stream threshold and bound; the crop dependence of the reference streams; and the
    development gap with core-only routing against routing on the wider domain."""
    out = {"cohort": {}, "wideReference": {}, "wide": {}, "domains": {}}
    for r in rows:
        if r["study"] == "wide-reference":
            out["wideReference"][r["region"]] = r["f1"]
        if r["study"] == "wide-domain":
            out["domains"][r["region"]] = {k: r[k] for k in ("holeNodes", "ringMaxDifferenceM")}
    cohort_rows = [r for r in rows if r["study"] == "cohort"]
    for cohort in sorted({r["cohort"] for r in cohort_rows}):
        for policy in sorted({r["policy"] for r in cohort_rows}):
            for area in sorted({a for r in cohort_rows for a in r["f1"]}, key=float):
                for b in sorted({r["boundM"] for r in cohort_rows}):
                    sel = [r for r in cohort_rows if r["cohort"] == cohort and r["policy"] == policy
                           and r["boundM"] == b]
                    gaps = {}
                    for g in sorted({r["region"] for r in sel}):
                        learned = [r["f1"][area] for r in sel if r["region"] == g and r["product"].startswith("learned")]
                        conv = [r["f1"][area] for r in sel if r["region"] == g and r["product"].startswith("conventional")]
                        if learned and conv and None not in learned + conv:
                            gaps[g] = float(np.mean(learned)) - conv[0]
                    if gaps:
                        out["cohort"].setdefault(cohort, {}).setdefault(policy, {}).setdefault(area, {})[str(b)] = {
                            "worstGap": min(gaps.values()), "meanGap": float(np.mean(list(gaps.values()))),
                            "pass": min(gaps.values()) >= -0.01}
    wide = [r for r in rows if r["study"] == "wide"]
    for b in sorted({r["boundM"] for r in wide}):
        for name in ("learned:s0", "learned-levels:s0"):
            for domain in ("core", "wide"):
                gaps = []
                for g in sorted({r["region"] for r in wide}):
                    sel = {r["product"].split(":")[0]: r for r in wide if r["region"] == g and r["boundM"] == b}
                    if name.split(":")[0] in sel and "conventional" in sel:
                        gaps.append(sel[name.split(":")[0]][domain]["50000"] - sel["conventional"][domain]["50000"])
                if gaps:
                    out["wide"].setdefault(str(b), {}).setdefault(name, {})[domain] = {
                        "worstGap": min(gaps), "meanGap": float(np.mean(gaps)), "pass": min(gaps) >= -0.01}
    return out


def cmd_drainage_sensitivity(a):
    from geoneural.codecs import campaign
    from geoneural.evaluation import protocol
    out = a.out or HOME / "results" / "v2" / "codec"
    rows = campaign.drainage_sensitivity(out)
    rec = protocol.record(
        "compression", "drainage-sensitivity", "selected",
        results={"rows": rows, "summary": summarise_sensitivity(rows)},
        recipe={"areasM2": list(campaign.SENS_AREAS), "policies": list(campaign.POLICIES),
                "wideMarginNodes": campaign.WIDE_MARGIN,
                "products": "learned with the frozen models (H1b with its frozen rules) and the best conventional "
                            "product recorded by each confirmation run; development: LORO model seed 0"},
        split={"cohorts": ["A", "B"], "development": list(campaign.REGIONS)})
    print(protocol.write(rec, out / "drainage-sensitivity.json"))


def summarise_geology_sweep(rows: list[dict], h2_rows: list[dict]) -> dict:
    """Each perturbed map against the matched no-map model and the real map of the same seed (H2 rows), with the
    map charged and with the map at the decoder, regions as units."""
    from geoneural.evaluation.statistics import paired_log_ratio
    seed = rows[0]["seed"] if rows else 0
    ref = [r for r in h2_rows if r["seed"] == seed and r["arm"] in ("none", "real")]
    out = {}
    for b in sorted({r["boundM"] for r in rows}):
        def per(src, arm, free):
            return {r["region"]: _standalone(r, free) for r in src if r["boundM"] == b and r["arm"] == arm}
        entry = {}
        for free, key in ((False, "charged"), (True, "mapAtDecoder")):
            none, real = per(ref, "none", free), per(ref, "real", free)
            entry[key] = {arm: {"vsNone": paired_log_ratio(per(rows, arm, free), none),
                                "vsReal": paired_log_ratio(per(rows, arm, free), real)}
                          for arm in sorted({r["arm"] for r in rows})}
            entry[key]["real"] = {"vsNone": paired_log_ratio(real, none)}
        out[str(b)] = entry
    return out


def cmd_codec_geology_sweep(a):
    from geoneural.codecs import campaign
    from geoneural.common import read_json
    from geoneural.evaluation import protocol
    out = a.out or HOME / "results" / "v2" / "codec"
    if a.merge:
        rows = _merge_parts(a.merge)
    else:
        rows = campaign.geology_sweep(out, tuple(a.regions), seed=a.seed, held_out=a.held)
        if a.part:
            return _write_part(rows, a.part)
    h2_rows = read_json(out / "h2.json")["results"]["rows"]
    rec = protocol.record(
        "compression", "geology-sweep", "exploratory",
        results={"rows": rows, "summary": summarise_geology_sweep(rows, h2_rows)},
        recipe={"arms": list(campaign.GEO_PERTURB), "start": "H2 recipe, same seed",
                "resolutions": "10, 40 and 160 m class rasters", "shifts": "3, 8 and 25 cells of 40 m",
                "blur": "most frequent class in 3 x 3 and 5 x 5 windows", "unknown": "a quarter of the 1 km blocks"},
        split={"scheme": "leave one region out", "regions": list(a.regions)}, seeds=[a.seed])
    print(protocol.write(rec, out / "geology-sweep.json"))


def amortisation(rows: list[dict], sizes=(1, 2, 5, 10, 100, 1000)) -> dict:
    """Per bound: bytes per field of the learned coder when N fields share one model (corpus bytes plus model / N)
    over the best conventional product, geometric mean over regions, and each region's break-even N (the smallest
    corpus size at which the learned coder is smaller; none when it is not smaller even with the model free)."""
    out = {}
    for b in sorted({r["boundM"] for r in rows}):
        best, corpus, model = {}, {}, {}
        for r in rows:
            if r["boundM"] != b:
                continue
            if r.get("family") == "conventional" and r.get("boundViolations") == 0:
                best[r["region"]] = min(best.get(r["region"], np.inf), r["bytes"])
            if r.get("coder") == "learned" and r.get("product") == "corpus":
                corpus.setdefault(r["region"], []).append(r["bytes"])
                model[r["region"]] = r["sharedModelBytes"] + 9
        regions = sorted(set(best) & set(corpus))
        per = {n: float(np.exp(np.mean([np.log((np.mean(corpus[g]) + model[g] / n) / best[g]) for g in regions])))
               for n in sizes}
        free = float(np.exp(np.mean([np.log(np.mean(corpus[g]) / best[g]) for g in regions])))
        breakeven = {}
        for g in regions:
            gain = best[g] - np.mean(corpus[g])
            breakeven[g] = int(np.ceil(model[g] / gain)) if gain > 0 else None
        out[str(b)] = {"ratioAtCorpusSize": {str(n): v for n, v in per.items()}, "ratioModelFree": free,
                       "breakEvenFields": breakeven}
    return out


def cmd_codec_amortisation(a):
    from geoneural.common import read_json, write_json
    out = a.out or HOME / "results" / "v2" / "codec"
    dev = read_json(out / "h1.json")["results"]["rows"]
    if (out / "sz3-best.json").exists():
        dev += [r for r in read_json(out / "sz3-best.json")["results"]["rows"] if "bytes" in r]
    conf = read_json(out / "h1-confirmation.json")["results"]["rows"]
    result = {"schema": "geoneural-amortisation-v1", "development": amortisation(dev), "confirmation": amortisation(conf),
              "note": "Bytes only. The model is trained once (training time in the compute profile)."}
    write_json(out / "amortisation.json", result)
    print(out / "amortisation.json")


def cmd_bench_v2(a):
    from geoneural.bench import v2
    from geoneural.codecs import campaign
    from geoneural.common import write_json
    model = a.model or (HOME / "results" / "v2" / "codec" / "models" / "final-s0.gnm")
    out = a.out or HOME / "results" / "v2" / "bench"
    if a.part == "queries":
        res = v2.query_workload(a.region, a.bound, a.repeats, model)
    elif a.part == "scaling":
        regions = list(campaign.REGIONS) + [r for r in __import__("json").loads(
            (Path(__file__).resolve().parents[1] / "configs" / "cohort-regions.json").read_text())]
        res = v2.scaling(regions, model_path=model)
    else:
        res = v2.profile(a.region, a.bound, model, train=not a.no_train)
    write_json(out / f"{a.part}.json", {"schema": "geoneural-bench-v2", "part": a.part, **res})
    print(out / f"{a.part}.json")


def cmd_codec_h3(a):
    from geoneural.codecs import campaign
    from geoneural.evaluation import protocol
    out = a.out or HOME / "results" / "v2" / "codec"
    rows = campaign.h3(out, tuple(a.regions))
    rec = protocol.record(
        "compression", "h3-allocation", "exploratory",
        results={"rows": rows, "atEqualBytes": campaign.at_equal_bytes(rows, campaign.h3_metric)},
        recipe={"rules": campaign.H3_RULES, "bounds": list(campaign.H3_BOUNDS),
                "learnedModel": "H1 leave-one-region-out model, seed 0"},
        split={"scheme": "leave one region out", "regions": list(a.regions)}, seeds=[0])
    print(protocol.write(rec, out / "h3.json"))


def cmd_codec_paged(a):
    from geoneural.codecs import campaign
    from geoneural.evaluation import protocol
    out = a.out or HOME / "results" / "v2" / "codec"
    rows = campaign.paged_campaign(out, tuple(a.regions))
    rec = protocol.record(
        "compression", "paged-product", "exploratory", results={"rows": rows},
        recipe={"page": campaign.PAGE + 1, "directoryBytesPerPage": 8, "learnedModel": "H1 model, seed 0"},
        split={"scheme": "leave one region out", "regions": list(a.regions)}, seeds=[0])
    print(protocol.write(rec, out / "paged.json"))


def cmd_codec_sz3best(a):
    from geoneural.codecs import campaign, sz3tuned
    from geoneural.evaluation import protocol
    out = a.out or HOME / "results" / "v2" / "codec"
    rows = campaign.sz3_best_rows(tuple(a.regions))
    rec = protocol.record("compression", "sz3-best", "exploratory", results={"rows": rows},
                          recipe={"configs": sz3tuned.CONFIGS, "sz3": sz3tuned.version(),
                                  "rule": "smallest stream holding the bound, chosen per field and bound"},
                          split={"regions": list(a.regions)})
    print(protocol.write(rec, out / "sz3-best.json"))


def summarise_arms(rows: list[dict], reference: str, arms) -> dict:
    from geoneural.evaluation.statistics import paired_log_ratio
    out = {}
    for b in sorted({r["boundM"] for r in rows}):
        def per_region(arm):
            vals: dict[str, list[float]] = {}
            for r in rows:
                if r["boundM"] == b and r["arm"] == arm:
                    vals.setdefault(r["region"], []).append(r["bytes"])
            return {k: float(np.mean(v)) for k, v in vals.items()}
        ref = per_region(reference)
        out[str(b)] = {arm: paired_log_ratio(per_region(arm), ref) for arm in arms if arm != reference}
    return out


def cmd_codec_h4(a):
    from geoneural.codecs import campaign
    from geoneural.evaluation import protocol
    out = a.out or HOME / "results" / "v2" / "codec"
    if a.merge:
        rows = _merge_parts(a.merge)
    else:
        held = [] if a.pretrain_only else a.held
        rows = campaign.h4(out, tuple(a.regions), seeds=tuple(a.seeds), held_out=held)
        if a.part or a.pretrain_only:
            return _write_part(rows, a.part) if a.part else None
    summary = summarise_h4(rows)
    decided = [summary[str(b)]["pass"] for b in DECISION_BOUNDS if str(b) in summary]
    rec = protocol.record(
        "compression", "h4-process-prior", "selected",
        results={"rows": rows, "vsRealExtra": summarise_arms(rows, "real-extra", campaign.H4_ARMS),
                 "vsProceduralReal": summarise_arms(rows, "procedural-real", campaign.H4_ARMS), "test": summary},
        recipe={"pretraining": "two closed-loop rounds on 48 synthetic 513 x 513 fields per kind",
                "fineTuning": "two closed-loop rounds on the five training regions",
                "synthetic": "see .data/synthetic/*/manifest.json"},
        split={"scheme": "leave one region out", "regions": list(a.regions)}, seeds=sorted({r["seed"] for r in rows}),
        gates={"h4": protocol.gate("pass" if decided and all(decided) else "fail", "protocol-v2 H4 rule, development",
                                   maxRatio=0.98, bootstrapUpperBelow=1.0)})
    print(protocol.write(rec, out / "h4.json"))


def cmd_codec_final_models(a):
    from geoneural.codecs import campaign
    out = a.out or HOME / "results" / "v2" / "codec"
    for seed in a.seeds:
        m = campaign.final_model(seed, out)
        print(f"final-s{seed}.gnm sha256 {m.sha256()}")


def cmd_codec_confirm_h1(a):
    import json
    from geoneural.codecs import campaign, foreign, sz3tuned
    from geoneural.common import COHORT_CONFIG
    from geoneural.evaluation import protocol
    out = a.out or HOME / "results" / "v2" / "codec"
    cohort = list(json.loads(COHORT_CONFIG.read_text()))
    rows = campaign.confirm_h1(out, cohort, seeds=tuple(a.seeds))
    from geoneural.export.v2 import h1_verdict
    for r in rows:
        if r.get("drainage"):
            r["f1"] = r["drainage"].get("tolerantF1")
    models = {s: campaign.Predictor.from_bytes((out / "models" / f"final-s{s}.gnm").read_bytes()).sha256()
              for s in a.seeds}
    rec = protocol.record(
        "compression", "h1-confirmation", "confirmed",
        results={"rows": rows, "verdict": h1_verdict(rows)},
        recipe={"models": models, "codecVersions": foreign.versions(), "sz3": sz3tuned.version()},
        split={"scheme": "frozen models trained on the six development regions", "cohort": cohort},
        seeds=list(a.seeds),
        gates={"h1": protocol.gate("pass" if h1_verdict(rows)["supported"] else "fail", "protocol-v2 H1 rule",
                                   maxRatio=0.90, bootstrapUpperBelow=0.95, maxTolerantF1Drop=0.01)})
    print(protocol.write(rec, out / "h1-confirmation.json"))


def cmd_encode(a):
    from geoneural.codecs import package
    from geoneural.codecs.predictor import Predictor
    from geoneural.common import read_json
    z = np.load(a.atlas / "reference.npy")
    model = Predictor.from_bytes(a.model.read_bytes()) if a.model else None
    blob, recon, info = package.encode(z, a.bound, a.coder, model, embed_model=not a.corpus,
                                       spatial=package.spatial_from_atlas(read_json(a.atlas / "atlas.json")))
    a.out.write_bytes(blob)
    print(f"{a.out}: {len(blob)} bytes, max error {info['maxErrorM']:.6f} m, {info['breakdown']}")


def cmd_decode(a):
    from geoneural.codecs import package
    from geoneural.codecs.predictor import Predictor
    model = Predictor.from_bytes(a.model.read_bytes()) if a.model else None
    field = package.decode(a.product.read_bytes(), model)
    np.save(a.out, field.astype(np.float32) if a.float32 else field)
    print(a.out)


def cmd_train_predictor(a):
    from geoneural.codecs import campaign
    from geoneural.codecs.predictor import fit
    model = fit([campaign.load(r)[0] for r in a.regions], tuple(a.bounds), rounds=a.rounds,
                widths=tuple(a.widths), steps=a.steps, seed=a.seed)
    a.out.write_bytes(model.to_bytes())
    print(f"{a.out}: {model.nbytes()} bytes, sha256 {model.sha256()[:16]}, {model.meta}")


def register(add):
    from geoneural.codecs import campaign
    s = add("codec-h1", cmd_codec_h1, "Learned multilevel coder against conventional codecs, leave one region out")
    s.add_argument("--regions", nargs="+", default=list(campaign.REGIONS))
    s.add_argument("--bounds", nargs="+", type=float, default=list(campaign.BOUNDS))
    s.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    s.add_argument("--out", type=Path)
    s = add("codec-h2", cmd_codec_h2, "Geology context inside the learned coder, with matched controls")
    s.add_argument("--regions", nargs="+", default=list(campaign.REGIONS))
    s.add_argument("--bounds", nargs="+", type=float, default=list(campaign.BOUNDS))
    s.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    s.add_argument("--out", type=Path)
    s.add_argument("--held", nargs="+", help="Run only these held-out folds")
    s.add_argument("--part", type=Path, help="Write the rows of this run to a part file instead of the report")
    s.add_argument("--merge", nargs="+", type=Path, help="Merge part files into the report")
    s = add("codec-h3", cmd_codec_h3, "Bound allocation near the decoded drainage network and on gentle slopes")
    s.add_argument("--regions", nargs="+", default=list(campaign.REGIONS))
    s.add_argument("--out", type=Path)
    s = add("codec-paged", cmd_codec_paged, "The random-access product: 257 x 257 pages coded alone")
    s.add_argument("--regions", nargs="+", default=list(campaign.REGIONS))
    s.add_argument("--out", type=Path)
    s = add("codec-sz3best", cmd_codec_sz3best, "SZ3 at its best configuration per field (needs GEONEURAL_SZ3)")
    s.add_argument("--regions", nargs="+", default=list(campaign.REGIONS))
    s.add_argument("--out", type=Path)
    s = add("codec-h4", cmd_codec_h4, "Process-prior pretraining of the shared predictor, with matched controls")
    s.add_argument("--regions", nargs="+", default=list(campaign.REGIONS))
    s.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    s.add_argument("--out", type=Path)
    s.add_argument("--held", nargs="+", help="Run only these held-out folds")
    s.add_argument("--pretrain-only", action="store_true", help="Only train the shared pretrained models")
    s.add_argument("--part", type=Path, help="Write the rows of this run to a part file instead of the report")
    s.add_argument("--merge", nargs="+", type=Path, help="Merge part files into the report")
    s = add("codec-levels", cmd_codec_levels, "Level-wise bounds: a lower typical error at the same largest error")
    s.add_argument("--regions", nargs="+", default=list(campaign.REGIONS))
    s.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    s.add_argument("--out", type=Path)
    s.add_argument("--held", nargs="+", help="Run only these held-out folds")
    s.add_argument("--part", type=Path, help="Write the rows of this run to a part file instead of the report")
    s.add_argument("--merge", nargs="+", type=Path, help="Merge part files into the report")
    s = add("codec-frontier", cmd_codec_frontier, "Conventional arms C1 to C3 and task corrections, development")
    s.add_argument("--regions", nargs="+", default=list(campaign.REGIONS))
    s.add_argument("--out", type=Path)
    s.add_argument("--held", nargs="+", help="Run only these regions")
    s.add_argument("--part", type=Path, help="Write the rows of this run to a part file instead of the report")
    s.add_argument("--merge", nargs="+", type=Path, help="Merge part files into the report")
    s = add("codec-geology-sweep", cmd_codec_geology_sweep, "Geology map resolution, shift, blur and unknown classes")
    s.add_argument("--regions", nargs="+", default=list(campaign.REGIONS))
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", type=Path)
    s.add_argument("--held", nargs="+", help="Run only these held-out folds")
    s.add_argument("--part", type=Path, help="Write the rows of this run to a part file instead of the report")
    s.add_argument("--merge", nargs="+", type=Path, help="Merge part files into the report")
    s = add("codec-amortisation", cmd_codec_amortisation, "Bytes per field when a corpus shares the learned model")
    s.add_argument("--out", type=Path)
    s = add("bench-v2", cmd_bench_v2, "Query workload, parallel encode scaling or compute profile of the v2 products")
    s.add_argument("part", choices=["queries", "scaling", "profile"])
    s.add_argument("--region", default="essen-ruhr")
    s.add_argument("--bound", type=float, default=0.25)
    s.add_argument("--repeats", type=int, default=7)
    s.add_argument("--model", type=Path)
    s.add_argument("--no-train", action="store_true")
    s.add_argument("--out", type=Path)
    s = add("drainage-sensitivity", cmd_drainage_sensitivity, "Stream threshold, boundary policy and routing domain")
    s.add_argument("--out", type=Path)
    s = add("codec-freeze-levels", cmd_codec_freeze_levels, "Freeze the level-wise rule per bound from the study")
    s.add_argument("--recipe", type=Path, required=True)
    s.add_argument("--out", type=Path)
    s = add("codec-confirm-levels", cmd_codec_confirm_levels, "Run the frozen level-wise recipe once on a cohort")
    s.add_argument("--recipe", type=Path, required=True)
    s.add_argument("--cohort-config", type=Path, required=True)
    s.add_argument("--out", type=Path)
    s = add("cohort-select", cmd_cohort_select, "Draw a confirmation cohort by its committed rule")
    s.add_argument("--rule", choices=["a", "b"], required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--presets", type=Path, required=True)
    s.add_argument("--max-probes", type=int, default=120)
    s = add("codec-final-models", cmd_codec_final_models, "Train the frozen H1 predictor on all development regions")
    s.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    s.add_argument("--out", type=Path)
    s = add("codec-confirm-h1", cmd_codec_confirm_h1, "Run the frozen H1 arms once on the confirmation cohort")
    s.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    s.add_argument("--out", type=Path)
    s = add("encode", cmd_encode, "Encode an atlas reference into a .gnc product")
    s.add_argument("--atlas", type=Path, required=True)
    s.add_argument("--bound", type=float, required=True, help="Maximum absolute error in metres")
    s.add_argument("--coder", choices=["cubic-order0", "cubic-ctx", "learned"], default="cubic-ctx")
    s.add_argument("--model", type=Path, help="Shared predictor (.gnm) for the learned coder")
    s.add_argument("--corpus", action="store_true", help="Reference the shared model instead of embedding it")
    s.add_argument("--out", type=Path, required=True)
    s = add("decode", cmd_decode, "Decode a .gnc product using only the file (and a shared model if it names one)")
    s.add_argument("product", type=Path)
    s.add_argument("--model", type=Path)
    s.add_argument("--float32", action="store_true")
    s.add_argument("--out", type=Path, required=True)
    s = add("train-predictor", cmd_train_predictor, "Train the shared predictor of the multilevel coder")
    s.add_argument("--regions", nargs="+", required=True)
    s.add_argument("--bounds", nargs="+", type=float, default=list(campaign.BOUNDS))
    s.add_argument("--widths", nargs=2, type=int, default=[32, 32])
    s.add_argument("--steps", type=int, default=5000)
    s.add_argument("--rounds", type=int, default=2)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", type=Path, required=True)
