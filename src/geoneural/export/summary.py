"""The numbers the README and the docs quote, read from the reports.

Every figure written in prose comes from here, so a rerun that changes a number changes it
everywhere it is quoted. Each entry names its report and evidence state.
"""
from __future__ import annotations

import statistics
from pathlib import Path

from geoneural.common import read_json

SCHEMA = "geoneural-summary-v1"


def _load(root, name: str):
    """A report from the first root that has it, preferring reproduced over recovered runs."""
    for base in (root if isinstance(root, (list, tuple)) else [root]):
        for state in ("reproduced", "recovered"):
            path = Path(base) / state / name
            if path.exists():
                return read_json(path), {"report": name, "evidence": state}
    return None, None


def _round(value, digits=4):
    return None if value is None else round(float(value), digits)


def tournament(root):
    report, source = _load(root, "tournament-essen.json")
    if not report:
        return None
    at = {}
    for result in report["results"]:
        if not result.get("available"):
            continue
        for point in result["observations"]:
            key = f"{point['target_m']:g}" if point.get("target_m") is not None else "lossless"
            at.setdefault(key, {})[result["codec"]] = {
                "payloadBytes": point["payload_bytes"], "totalBytes": point["total_deployment_bytes"],
                "maeM": _round(point["mae_m"]), "maxM": _round(point["max_error_m"]),
                "meetsTarget": point.get("meets_target")}
    return {"source": source, "indexBytes": report["index_bytes_shared_by_every_codec"],
            "samplesAcrossLevels": report["samples_across_levels"], "pages": report["pages_in_atlas"],
            "byTarget": at}


def drainage(root):
    report, source = _load(root, "drainage-essen.json")
    if not report:
        return None
    return {"source": source, "streamThresholdCells": report["streamThresholdCells"],
            "byTarget": {f"{t['targetMaxErrorM']:g}": {
                "streamJaccard": _round(t["streamJaccard"]), "streamRecall": _round(t["streamRecall"]),
                "referenceBasins": t["referenceBasins"], "reconstructedBasins": t["reconstructedBasins"],
                "maeM": _round(t["elevationMaeM"])} for t in report["byTarget"]}}


def corrections(root, table):
    report, source = _load(root, "corrections-essen.json")
    if not report:
        return None
    rows = {r["id"]: r for r in table["candidates"]}
    conventional = [r for r in table["candidates"] if r["family"] == "conventional"
                    and r["package"] == "finest-per-page" and r["streamJaccard"] is not None]
    bands = []
    for band in report["byBandRadius"]:
        row = rows.get(f"corrected-r{band['bandRadiusCells']}@{report['targetMaxErrorM']:g}")
        reach = sorted((c for c in conventional if c["streamJaccard"] >= band["streamJaccard"]),
                       key=lambda c: c["bytes"])
        below = sorted((c for c in conventional if c["streamJaccard"] < band["streamJaccard"]),
                       key=lambda c: -c["streamJaccard"])
        bands.append({
            "radiusCells": band["bandRadiusCells"], "correctionBytes": band["correctionBytes"],
            "totalBytes": row and row["bytes"], "streamJaccard": _round(band["streamJaccard"]),
            "streamJaccardAtOtherThresholds": {k: _round(v) for k, v in band["streamJaccardAtOtherThresholds"].items()},
            "maeM": _round(band.get("elevationMaeM")),
            "cheapestUniformReaching": reach and {"id": reach[0]["id"], "bytes": reach[0]["bytes"],
                                                  "streamJaccard": _round(reach[0]["streamJaccard"])},
            "bestUniformBelow": below and {"id": below[0]["id"], "bytes": below[0]["bytes"],
                                           "streamJaccard": _round(below[0]["streamJaccard"])}})
    return {"source": source, "targetM": report["targetMaxErrorM"],
            "baseline": {"streamJaccard": _round(report["baseline"]["streamJaccard"]),
                         "atOtherThresholds": {k: _round(v) for k, v in report["baselineAtOtherThresholds"].items()}},
            "uniformControls": [{"targetM": u["targetMaxErrorM"], "streamJaccard": _round(u["streamJaccard"]),
                                 "atOtherThresholds": {k: _round(v) for k, v in u["streamJaccardAtOtherThresholds"].items()}}
                                for u in report["uniformControls"]],
            "bands": bands}


def representations(table):
    rows = table["candidates"]
    learned = [r for r in rows if r["family"] in ("neural", "hybrid")]
    undominated = [r for r in learned if not r["dominatedBy"]]
    conventional = [r for r in rows if r["family"] == "conventional" and r["package"] == "finest-per-page"]
    drainage_rows = []
    for r in learned:
        if r["streamJaccard"] is None:
            continue
        pool = [c for c in conventional if c["streamJaccard"] is not None and c["bytes"] <= r["bytes"]]
        best = max(pool, key=lambda c: c["streamJaccard"]) if pool else None
        drainage_rows.append({"id": r["id"], "bytes": r["bytes"], "streamJaccard": _round(r["streamJaccard"]),
                              "bestConventional": best and {"id": best["id"], "bytes": best["bytes"],
                                                            "streamJaccard": _round(best["streamJaccard"])}})
    by_id = {r["id"]: r for r in rows}

    def pick(cid):
        r = by_id.get(cid)
        return r and {k: r.get(k) for k in ("id", "label", "bytes", "bytesByPart", "maeM", "maxM", "streamJaccard",
                                        "dominatedBy", "bestConventional", "evidence")}

    gains = [r["bestConventional"]["maeGainPercent"] for r in undominated if r.get("bestConventional")]
    return {
        "learnedRows": len(learned), "conventionalRows": len(conventional),
        "undominatedLearned": [{"id": r["id"], "bytes": r["bytes"], "maeM": _round(r["maeM"]),
                                "maxM": _round(r["maxM"], 2), "bestConventional": r.get("bestConventional")}
                               for r in undominated],
        "largestMeanErrorGainPercent": _round(max(gains), 1) if gains else None,
        "drainageAtMatchedBytes": drainage_rows,
        "learnedRowsWithLowerStreamOverlap": sum(
            1 for d in drainage_rows if d["bestConventional"] and d["streamJaccard"] < d["bestConventional"]["streamJaccard"]),
        "examples": {cid: pick(cid) for cid in ("checkpoint/siren-codec-fit", "hybrid-t0/int4-qat", "q32dz-level1@0.5",
                                                "q32dz-level1@0.3", "q32dz-full@1", "fit/siren-t17", "constant-mean")},
    }


def closure(root):
    seeds, source = _load(root, "flux-closure-seeds.json")
    if not seeds:
        return None
    return {"source": source, "seeds": seeds["seeds"], "penaltyWeights": seeds["penaltyWeights"],
            "linearDiffusionMae": _round(seeds["runs"][0]["linearDiffusionMae"], 5),
            "byArm": {arm: {"maeMedian": _round(s["maeVsTeacher"]["median"], 5),
                            "maeRange": [_round(s["maeVsTeacher"]["min"], 5), _round(s["maeVsTeacher"]["max"], 5)],
                            "residualMedian": float(f"{s['conservationResidualRelative']['median']:.3g}"),
                            "residualRange": [float(f"{s['conservationResidualRelative']['min']:.3g}"),
                                              float(f"{s['conservationResidualRelative']['max']:.3g}")],
                            "finiteRollouts": s["finiteRollouts"]}
                      for arm, s in seeds["summary"].items()}}


def geology(root):
    out = {}
    for name in ("geology-film", "geology-concat", "geology-residual-film", "geology-residual-concat"):
        report, source = _load(root, f"{name}.json")
        if not report:
            continue
        verdict = report.get("verdict", {})
        # Paired by seed: each seed trains every arm, so the per-seed difference is the
        # effect of the raster alone. The report's own verdict compares ranges instead.
        test = {row["arm"]: [run["metrics"]["test"]["mae_m"] for run in row["perSeed"]] for row in report["rows"]}
        paired = {other: [_round(o - r) for r, o in zip(test["real"], test[other])]
                  for other in ("misaligned", "none", "generic") if other in test}
        out[name] = {"source": source, "gainOverMisalignedM": _round(verdict.get("gainOverMisalignedM")),
                     "seedSpreadM": _round(verdict.get("seedSpreadM")),
                     "exceedsSeedSpread": verdict.get("exceedsSeedSpread"),
                     "extraBytesOverNone": verdict.get("extraBytesOverNone"),
                     "testMaeBySeed": {arm: [_round(v) for v in values] for arm, values in test.items()},
                     "pairedGainOfRealM": paired,
                     "note": "pairedGainOfRealM[x] is x minus real per seed on the held-out test pages; "
                             "positive means the real map helped."}
    return out


def registration(root):
    hfp, source = _load(root, "height-benchmarks-essen.json")
    dlm, dlm_source = _load(root, "dlm-essen.json")
    out = {}
    if hfp:
        reg = hfp["registration"]
        out["heightBenchmarks"] = {
            "source": source, "inAtlas": hfp["benchmarksInAtlas"], "inCore": reg["benchmarksInCore"],
            "shiftM": [_round(v, 2) for v in reg["shiftM"]], "spreadM": _round(reg["spreadAtZeroM"], 2),
            "spreadIfFlippedM": _round(reg["spreadIfFlippedNorthSouthM"], 1),
            "spreadIfTransposedM": _round(reg["spreadIfTransposedM"], 1),
            "medianResidualM": _round(hfp["residualPublishedMinusAtlasM"]["median"], 2)}
    if dlm:
        flow = dlm["orientation"]["flowAlongDigitisation"]
        out["landscapeModel"] = {
            "source": dlm_source, "waterLevelToleranceM": dlm["waterLevels"]["toleranceM"],
            "waterLevelLargestDifferenceM": _round(max(abs(p["differenceM"]) for p in dlm["waterLevels"]["points"]), 2),
            "axesDescending": flow["descending"], "axesAscending": flow["ascending"],
            "valleyBelowBanks": _round(dlm["valleyFloor"]["atlas"]["belowBothBanks"], 3),
            "valleyBelowBanksMirrored": _round(dlm["valleyFloor"]["mirroredNorthSouth"]["belowBothBanks"], 3)}
    return out


def recovered_only(root):
    out = {}
    latency, source = _load(root, "decode-latency.json")
    if latency:
        by_size = {}
        for row in latency["rows"]:
            for size in row["byQuerySize"]:
                by_size.setdefault(size["requestedSamples"], []).append(size["ratioMedian"])
        out["decodeLatency"] = {"source": source, "medianNeuralOverConventional": {
            str(k): _round(statistics.median(v), 1) for k, v in sorted(by_size.items())},
            "peakDeviceBytes": [r.get("peakDeviceBytes") for r in latency["rows"]]}
    sr, source = _load(root, "superres-neural.json")
    if sr:
        out["superResolution"] = {"source": source, "arms": {
            row["arm"]: {"bytes": row["deployedBytes"],
                         "withinRegion": {"neuralMaeM": _round(row["withinRegion"]["neural"]["maeM"]),
                                          "bicubicBackProjectedMaeM": _round(row["withinRegion"]["bicubicBackProjected"]["maeM"])},
                         "heldOut": {region: {"neuralMaeM": _round(v["neural"]["maeM"]),
                                              "bicubicBackProjectedMaeM": _round(v["bicubicBackProjected"]["maeM"])}
                                     for region, v in row["heldOutGeography"].items()}}
            for row in sr["rows"]}}
    emulator, source = _load(root, "emulator.json")
    if emulator:
        out["emulator"] = {"source": source, "gates": {row["kind"]: row["evaluation"]["gates"] for row in emulator["rows"]}}
    distillation, source = _load(root, "distillation.json")
    if distillation:
        v = distillation["verdict"]
        out["distillation"] = {"source": source, "physicsGainM": _round(v["physicsGainM"]),
                               "seedSpreadM": _round(v["seedSpread"]), "informative": v["informative"]}
    event, source = _load(root, "event-field.json")
    if event:
        out["eventField"] = {"source": source,
                             "boreholeMaeM": {row["arm"]: _round(row["borehole"]["maeM"], 2) for row in event["rows"]}}
    return out


def build(root: Path, table: dict, local_root: Path | None = None) -> dict:
    """`root` holds the published reports; `local_root` adds reports kept only on this machine."""
    root = Path(root)
    audit, audit_source = _load(root, "teacher-audit.json")
    return {"schema": SCHEMA, "tournament": tournament(root), "drainage": drainage(root),
            "corrections": corrections(root, table), "representations": representations(table),
            "closure": closure(root), "teacherAudit": audit and {"source": audit_source, "verdict": audit["verdict"]},
            "geology": geology(root), "registration": registration(root),
            "otherExperiments": recovered_only([root] + ([Path(local_root)] if local_root else []))}
