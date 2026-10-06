"""Command-line entry points of the physics track (registered by `geoneural.cli`).

Every experiment command writes one v2 record to `results/v2/physics/<name>.json`
under the data root, with gates, failures and negative results kept.
"""
from __future__ import annotations

import json
from pathlib import Path

from geoneural.common import HOME

RESULTS = HOME / "results" / "v2" / "physics"


def gate_from(passed, detail: str = "", **thresholds) -> dict:
    """A protocol gate from a boolean (None means the gate could not be evaluated)."""
    from geoneural.evaluation import protocol
    state = "not-run" if passed is None else ("pass" if passed else "fail")
    return protocol.gate(state, detail, **thresholds)


def write_record(task: str, name: str, evidence: str, results, *, gates=None, failures=None,
                 recipe=None, split=None, seeds=None, notes: str = "", timing=None,
                 out: Path | None = None) -> Path:
    """Build, validate and write one v2 record; print where it went."""
    from geoneural.evaluation import protocol
    rec = protocol.record(task, name, evidence, results=results, recipe=recipe or {}, split=split or {},
                          seeds=seeds or [], gates=gates or {}, failures=failures or [], notes=notes,
                          timing=timing)
    path = protocol.write(rec, (out or RESULTS) / f"{name}.json")
    print(json.dumps({"record": str(path.relative_to(HOME)) if path.is_relative_to(HOME) else path.name,
                      "gates": {k: g["state"] for k, g in (gates or {}).items()},
                      "failures": len(failures or [])}, indent=1))
    return path


def failures_from(gates: dict) -> list[dict]:
    return [{"gate": key, "detail": g["detail"], "thresholds": g["thresholds"]}
            for key, g in gates.items() if g["state"] == "fail"]


def cmd_synthetic_terrain(a):
    from geoneural.physics import synthetic
    manifests = synthetic.generate(a.out, count=a.count, seed=a.seed, workers=a.workers)
    print(json.dumps({name: {"count": m["count"], "wallSeconds": round(m["wallSeconds"], 1)}
                      for name, m in manifests.items()}, indent=1))


def cmd_teacher_audit_v2(a):
    from geoneural.physics import teacher_audit
    report = teacher_audit.run(include_fastscape=not a.skip_fastscape)
    v = report["verdict"]
    timestep, budget = report["timestepRefinement"], report["budget"]
    gates = {
        "timestepOrder": gate_from(v["timestepConverges"], "finest observed order and monotone error",
                                   minimumOrder=0.8),
        "nonDivisibleTime": gate_from(v["nonDivisibleExact"], "exact realised time; error within 1.5x "
                                      "of the nearest divisible duration", maxErrorRatio=1.5),
        "gridExponent": gate_from(v["exponentSurvivesTheMesh"], "concavity converges and is recovered",
                                  maxConcavityError=0.15),
        "conservation": gate_from(v["booksClose"], "ledger closes, diffusion volume zero", tolerance=1e-9),
        "persistentSteadyState": gate_from(v["reachesPersistentSteadyState"],
                                           "rate below threshold for the persistence window",
                                           threshold=report["steadyState"]["thresholdNormalisedRate"],
                                           persistenceYears=report["steadyState"]["persistenceYears"]),
        "analyticIncision": gate_from(v["analyticIncision"], "steady state on the final network",
                                      maxRelativeToRelief=1e-3),
        "manufacturedDiffusion": gate_from(v["manufacturedDiffusion"], "second order in dx", minimumOrder=1.7),
        "quarterTurn": gate_from(v["quarterTurnCommutes"], "rot90 commutes with the teacher", tolerance=1e-9),
        "fastscape": gate_from(v["agreesWithFastscape"] if report["fastscape"].get("available") else None,
                               "differences shrink with dt and stay below 2 % of the change; steady mean "
                               "elevation within 2 % of relief", maxRelative=0.02),
        "dynamicsBudget": gate_from(v["dynamicsBudget"], "teacher error at dt 400 below 10 % of a budget "
                                    "of 10 % of the mean change", maxRatio=0.1),
        "inverseBudget": gate_from(v["inverseBudget"], "teacher error at dt 400 below 10 % of the "
                                   "observation noise", maxRatio=0.1, noiseM=budget["inverse"]["noiseM"]),
    }
    write_record("audit", "teacher-audit-v2", "exploratory", report, gates=gates,
                 failures=failures_from(gates),
                 recipe={"parameters": report["parameters"], "solver": report["solver"],
                         "referenceDtYears": timestep["referenceDtYears"]},
                 timing={"qualified": False, "seconds": report["seconds"], "note": "shared host"},
                 notes="Teacher v2 audit: exact elapsed time and receiver-drop limiter. Thresholds are "
                       "declared in the audit module; the budget criterion is the research plan's "
                       "proposed 10 % rule.")


def _torch(device: str, threads: int = 4, gpu_fraction: float = 0.2):
    """torch with a bounded CPU thread count and, on CUDA, a bounded share of the shared GPU."""
    import torch
    torch.set_num_threads(threads)
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(gpu_fraction)
    return torch


def cmd_closure_study(a):
    from geoneural.physics import hybrid
    torch = _torch(a.device)
    result = hybrid.closure_study(torch, seeds=tuple(a.seeds), steps=a.steps, device=a.device,
                                  material=not a.skip_material)
    summary, selection = result["summary"], result["selection"]
    accepted = selection["accepted"]
    base = summary["baselines"]
    gates = {
        "acceptedArmStructural": gate_from(bool(accepted) and summary[accepted]["structuralPass"],
                                           "flat surface exactly still, offset invariant, conservative, "
                                           "bounded", offsetTolerance=1e-9, conservationTolerance=1e-12),
        "acceptedBeatsLinear": gate_from(
            bool(accepted) and summary[accepted]["in-range"]["meanRmseM"]["max"]
            < base["linear"]["in-range"]["meanRmseM"],
            "worst seed of the accepted arm below analytic linear diffusion, in-range rollouts"),
        "acceptedDissipates": gate_from(
            bool(accepted) and summary[accepted]["in-range"]["maxEnergyIncreaseRelative"] <= 1e-12,
            "sum of squared deviations never increases over the rollout", tolerance=1e-12),
    }
    failures = failures_from(gates)
    for arm, entry in summary.items():
        if arm != "baselines" and not entry.get("structuralPass", True):
            failures.append({"arm": arm, "detail": "fails a structural gate (recorded, not selectable)",
                             "structure": entry["structure"]})
    write_record("dynamics", "closure-study", "selected", result, gates=gates, failures=failures,
                 seeds=list(a.seeds), recipe=result["settings"],
                 split={"train": "fresh rough surfaces from each seed's generator",
                        "evaluation": "seed 31337 surfaces, held-out relief and spectrum, rot90, ridges"},
                 timing={"qualified": False, "seconds": result["seconds"], "note": "shared host and GPU"},
                 notes="Structured closure against linear diffusion, the teacher, a conservative face "
                       "diffusivity, the free flux arm, the penalty arm and the K-field arm, selected on "
                       "rollouts at matched physical times.")


def cmd_distillation_diagnostic(a):
    """Re-label the legacy distillation report as a diagnostic in a v2 record (no rerun)."""
    from geoneural.common import read_json
    legacy = read_json(a.legacy)
    verdict = legacy.get("verdict", {})
    results = {"classification": "diagnostic", "notACodec": True,
               "legacySchema": legacy.get("schema"), "legacyReport": a.legacy.name,
               "rows": [{k: r.get(k) for k in ("arm", "heldOutMaeM", "totalBytes", "sideChannelBytes", "parameters")}
                        for r in legacy.get("rows", [])],
               "verdict": verdict,
               "why": "The teacher and generic arms are conditioned on a raster built from the full target "
                      "field (its own D8 paths and slope-area fit). No standalone decoder could build it, and "
                      "the 24 bytes charged for three scalars do not price the paths. The numbers say whether "
                      "slope-area structure is informative to a small coordinate network at equal capacity; "
                      "they are not compression results."}
    gates = {"isACodec": gate_from(False, "conditioning depends on the full target; not priced")}
    write_record("compression", "distillation-diagnostic", "exploratory", results, gates=gates,
                 failures=[{"gate": "isACodec", "detail": "diagnostic only; a codec rebuild from a priced, "
                            "decoder-available base was not attempted"}],
                 notes="Re-labelled from the legacy report without rerunning.")


def cmd_pilot_ensemble(a):
    from geoneural.physics import ensemble
    import numpy as np
    manifest = ensemble.generate(a.out, count=a.count, side=a.side, spacing_m=a.spacing_m, years=a.years,
                                 seed=a.seed, workers=a.workers, frames=a.frames, split_rule="block")
    sims = manifest["simulations"]
    worst = max(r["worstClosureResidualRelative"] for r in sims)
    clipped = sum(r["cellsIncisionLimitedTotal"] for r in sims)
    timed = max(abs(r["realisedYears"] - r["job"]["years"]) for r in sims)
    results = {"manifest": str(Path(a.out).relative_to(HOME)) if Path(a.out).is_relative_to(HOME) else Path(a.out).name,
               "count": manifest["count"], "side": manifest["side"], "spacingM": manifest["spacingM"],
               "years": manifest["years"], "solver": manifest["solver"], "solverHash": manifest["solverHash"],
               "split": {k: len(v) for k, v in manifest["split"].items() if k.endswith("Ids")},
               "splitRule": manifest["split"]["rule"], "seconds": manifest["seconds"],
               "worstClosureResidualRelative": worst, "cellsIncisionLimitedTotal": clipped,
               "maxRealisedTimeErrorYears": timed,
               "stepsMedian": float(np.median([r["steps"] for r in sims])),
               "dtYearsRange": [min(r["job"]["dtYears"] for r in sims), max(r["job"]["dtYears"] for r in sims)],
               "finalReliefM": {"median": float(np.median([r["finalReliefM"] for r in sims])),
                                "min": float(min(r["finalReliefM"] for r in sims)),
                                "max": float(max(r["finalReliefM"] for r in sims))}}
    gates = {"ledgerCloses": gate_from(worst < 1e-9, "worst per-interval closure residual", tolerance=1e-9),
             "exactTime": gate_from(timed < 1e-6, "realised minus requested years", toleranceYears=1e-6),
             "underBudget": gate_from(manifest["seconds"] < 1800, "wall seconds", maxSeconds=1800)}
    write_record("dynamics", "pilot-ensemble", "exploratory", results, gates=gates, failures=failures_from(gates),
                 seeds=[a.seed], split=results["split"] | {"rule": results["splitRule"]},
                 recipe={"count": a.count, "side": a.side, "spacingM": a.spacing_m, "years": a.years,
                         "frames": a.frames, "courantMax": ensemble.COURANT_MAX,
                         "logFluvialRange": list(ensemble.LOG_FLUVIAL_RANGE),
                         "logPecletRange": list(ensemble.LOG_PECLET_RANGE), "reliefScaleM": ensemble.RELIEF_SCALE_M},
                 timing={"qualified": False, "seconds": manifest["seconds"], "note": "shared host"})


def cmd_emulator_v2(a):
    from geoneural.physics import emulator_train
    torch = _torch(a.device)
    result = emulator_train.campaign(a.ensemble, torch, kinds=tuple(a.kinds), steps=a.steps, batch=a.batch,
                                     lr=a.lr, seed=a.seed, device=a.device)
    gates, failures = {}, []
    for row in result["rows"]:
        for split, score in row["evaluation"].items():
            for key, value in score["gates"].items():
                if row["contract"] == "absolute" and key == "edgeExact":
                    continue
                gates[f"{row['arm']}:{split}:{key}"] = gate_from(bool(value), key)
    failures = failures_from(gates)
    write_record("dynamics", "emulator-v2", "exploratory", result, gates=gates, failures=failures, seeds=[a.seed],
                 split=result["split"], recipe={"kinds": a.kinds, "steps": a.steps, "batch": a.batch, "lr": a.lr,
                                                "contract": "identity plus increment, pinned edges; legacy absolute U-Net"},
                 timing={"qualified": False, "note": "shared host and GPU",
                         "seconds": sum(r["seconds"] for r in result["rows"])})


INVERSE_PARTS = ("profile", "nuisance", "reference", "coverage", "estimators", "misspecification")


def cmd_inverse_study(a):
    """The inverse experiments, one v2 record each. Coverage feeds the estimator comparison."""
    from geoneural.physics import inverse
    setup = inverse.setup_record()
    check = inverse.discretisation_check(workers=4)
    parts = set(a.parts)
    try:
        if "profile" in parts:
            profile = inverse.ridge_profile(workers=a.workers, draws=a.draws)
            flat = profile["flatWithinRounding"]
            worst_flat = max(v for truth in flat.values() for v in truth["rescaledDt"].values())
            worst_cap = min(v for truth in flat.values() for v in truth["fixedDtCap"].values())
            lag = profile["truths"]["transient"]["variants"]["rescaledDt"]["designs"]["fixed-lag-50k"]
            eq = profile["truths"]["near-equilibrium"]["variants"]["rescaledDt"]["designs"]["fixed-lag-50k"]
            gates = {"terminalAndFractionalFlat": gate_from(worst_flat < 1e-6, "expected log-likelihood range along "
                                                            "the ridge with consistently rescaled dt", tolerance=1e-6),
                     "fixedLagInformativeWhenTransient": gate_from(lag["rangeExpected"] > 10.0,
                                                                   "expected range over c in [0.5, 2]", minimum=10.0),
                     "fixedLagFlatNearEquilibrium": gate_from(eq["rangeExpected"] < 2.0, "expected range", maximum=2.0),
                     "fixedCapMasqueradeVisible": gate_from(worst_cap > 1e-3, "a dt held fixed across c creates a "
                                                            "spurious likelihood slope", minimum=1e-3)}
            write_record("inverse", "inverse-ridge-profile", "exploratory",
                         {"setup": setup, "discretisation": check, **profile},
                         gates=gates, failures=failures_from(gates), seeds=[20261004],
                         notes="c-profile along (cU, cK, cD, t/c) with dt/c; terminal, fractional (control) and "
                               "fixed-lag designs; dt-cap and refined-dt controls.")
        if "nuisance" in parts:
            nuisance = inverse.nuisance_profile(workers=a.workers)
            gates = {"stillInformativeWhenProfiled": gate_from(not nuisance["interval2Profile"]["hitsGridEdge"],
                                                               "the 2-log-unit profile interval for c stays inside "
                                                               "[0.5, 2] with both ratios profiled out")}
            write_record("inverse", "inverse-nuisance-profile", "exploratory", {"setup": setup, **nuisance},
                         gates=gates, failures=failures_from(gates),
                         notes="Fixed-lag (50 kyr) profile for c with log10 U/K and log10 D/K maximised at each c, "
                               "against the slice through the truth.")
        coverage = None
        if "reference" in parts:
            reference = inverse.ratio_reference(workers=a.workers)
            mcmc_ok = all(reference[d]["mcmc"]["gate"]["passes"] for d in ("1d", "2d"))
            agree = max(v["maxQuantileDifferenceInSd"] for d in ("1d", "2d")
                        for v in reference[d]["gridVersusMcmc"].values())
            gates = {"mcmcDiagnostics": gate_from(mcmc_ok, "rank-normalised split R-hat <= 1.01, bulk and tail ESS "
                                                  ">= 400", rhatMax=1.01, essMin=400),
                     "gridMatchesMcmc": gate_from(agree <= 0.5, "largest quantile difference in grid sd",
                                                  maxSd=0.5)}
            write_record("inverse", "inverse-ratio-reference", "exploratory", {"setup": setup, **reference},
                         gates=gates, failures=failures_from(gates), seeds=[7],
                         notes="One- and two-ratio posteriors (log10 U/K, log10 D/K) on a likelihood grid, "
                               "checked by random-walk Metropolis with the new diagnostics.")
        if "coverage" in parts or "estimators" in parts:
            coverage = inverse.coverage(workers=a.workers, truths_1d=a.truths_1d, truths_2d=a.truths_2d)
            summary = {k: v["summary"] for k, v in coverage["rows"].items()}
            gates = {}
            for dims, block in summary.items():
                for name in inverse.RATIO_NAMES[:int(dims[0])]:
                    c95 = block[name]["coverage95"]
                    n = block["truths"]
                    tolerance = 2.0 * (0.95 * 0.05 / n) ** 0.5
                    gates[f"{dims}:{name}:coverage95"] = gate_from(abs(c95 - 0.95) <= tolerance,
                                                                   "95 % interval coverage within two binomial sd",
                                                                   nominal=0.95, tolerance=tolerance)
            if "coverage" in parts:
                write_record("inverse", "inverse-coverage", "exploratory",
                             {"setup": setup, **inverse.strip_private(coverage)},
                             gates=gates, failures=failures_from(gates), seeds=[11],
                             notes="Repeated synthetic truths on likelihood grids, correct model.")
        if "estimators" in parts:
            torch = _torch(a.device)
            del torch
            estimates = inverse.estimators(coverage["rows"]["2d"]["truths"], workers=a.workers, device=a.device)
            res = estimates["results"]["inRange"]
            gates = {f"{arm}:{name}:withinTwiceReference": gate_from(
                res[arm][name]["rmseLog10"] <= 2.0 * res["referencePosterior"][name]["rmseLog10"],
                "rmse in log10 at most twice the reference posterior mean's")
                for arm in ("cnn", "featureRegressor") for name in inverse.RATIO_NAMES}
            write_record("inverse", "inverse-estimators", "exploratory", {"setup": setup, **estimates},
                         gates=gates, failures=failures_from(gates), seeds=[17],
                         notes="CNN (Gaussian head) and slope/curvature/drainage feature regressor (split-conformal "
                               "intervals) against the reference posterior on the same noisy data.")
        if "misspecification" in parts:
            mis = inverse.misspecification(workers=a.workers, truths=a.truths_mis)
            gates = {name: gate_from(abs(block["rows"]["1d"]["summary"]["logUpliftOverIncision"]["coverage95"] - 0.95)
                                     <= 0.1, "95 % coverage within 0.1 of nominal under this misspecification",
                                     tolerance=0.1)
                     for name, block in mis.items()}
            write_record("inverse", "inverse-misspecification", "exploratory",
                         {"setup": setup, "cases": inverse.MISSPECIFIED, **inverse.strip_private(mis)},
                         gates=gates, failures=failures_from(gates), seeds=[13],
                         notes="Data from another dt, another solver (fastscapelib), an omitted uplift gradient, "
                               "perturbed boundary values or another initial surface; inference keeps the nominal "
                               "teacher. A failing gate here is the expected finding, recorded as such.")
    finally:
        inverse.close_pools()


def cmd_closure_material_ablation(a):
    from geoneural.physics import hybrid
    torch = _torch(a.device)
    result = hybrid.material_face_loss_ablation(torch, seeds=tuple(a.seeds), device=a.device)
    gentle = result["summary"]["learnedRatioAtGradient0.02"]["median"]
    gates = {"recoversRatio": gate_from(abs(gentle - result["ratio"]) <= 0.05,
                                        "learned class-1 over class-0 conductance at |g| = 0.02", tolerance=0.05)}
    write_record("dynamics", "closure-material-ablation", "exploratory", result, gates=gates,
                 failures=failures_from(gates), seeds=list(a.seeds),
                 notes="Material-aware conductance trained on oracle face conductances instead of tendencies.")


def cmd_export_ident(a):
    from geoneural.export import ident
    print(ident.export(a.report, a.out))


def register(add):
    s = add("synthetic-terrain", cmd_synthetic_terrain,
            "Matched process-made and procedural 513 x 513 terrain at 10 m for the codec campaign")
    s.add_argument("--out", type=Path, default=HOME / "synthetic")
    s.add_argument("--count", type=int, default=48)
    s.add_argument("--seed", type=int, default=20261004)
    s.add_argument("--workers", type=int, default=20)
    s = add("teacher-audit-v2", cmd_teacher_audit_v2,
            "Audit the landscape teacher: refinement, exact time, analytic and manufactured cases, fastscapelib")
    s.add_argument("--skip-fastscape", action="store_true")
    s = add("closure-study", cmd_closure_study,
            "Bounded conductance closure against its baselines, selected on rollouts at matched times")
    s.add_argument("--seeds", type=int, nargs="+", default=[1729, 2, 3, 4, 5])
    s.add_argument("--steps", type=int, default=1500)
    s.add_argument("--device", default="cuda")
    s.add_argument("--skip-material", action="store_true")
    s = add("distillation-diagnostic", cmd_distillation_diagnostic,
            "Re-label the legacy distillation report as a diagnostic that is not a codec")
    s.add_argument("--legacy", type=Path, default=Path("results") / "reproduced" / "distillation.json")
    s = add("pilot-ensemble", cmd_pilot_ensemble,
            "Pilot teacher ensemble: exact times, per-interval balances, block split, solver hash")
    s.add_argument("--out", type=Path, default=HOME / "ensemble" / "pilot-v2")
    s.add_argument("--count", type=int, default=300)
    s.add_argument("--side", type=int, default=64)
    s.add_argument("--spacing-m", type=float, default=100.0)
    s.add_argument("--years", type=float, default=2_000_000.0)
    s.add_argument("--frames", type=int, default=9)
    s.add_argument("--seed", type=int, default=20261004)
    s.add_argument("--workers", type=int, default=20)
    s = add("emulator-v2", cmd_emulator_v2,
            "Increment emulators (U-Net, FNO) with pinned edges against persistence, at physical times")
    s.add_argument("--ensemble", type=Path, default=HOME / "ensemble" / "pilot-v2")
    s.add_argument("--kinds", nargs="+", default=["unet", "fno"])
    s.add_argument("--steps", type=int, default=1500)
    s.add_argument("--batch", type=int, default=8)
    s.add_argument("--lr", type=float, default=1e-3)
    s.add_argument("--seed", type=int, default=1729)
    s.add_argument("--device", default="cuda")
    s = add("inverse-study", cmd_inverse_study,
            "Ridge profile, ratio reference with MCMC diagnostics, coverage, estimators and misspecification")
    s.add_argument("--parts", nargs="+", default=list(INVERSE_PARTS), choices=INVERSE_PARTS)
    s.add_argument("--workers", type=int, default=20)
    s.add_argument("--draws", type=int, default=300)
    s.add_argument("--truths-1d", type=int, default=100)
    s.add_argument("--truths-2d", type=int, default=30)
    s.add_argument("--truths-mis", type=int, default=40)
    s.add_argument("--device", default="cuda")
    s = add("closure-material-ablation", cmd_closure_material_ablation,
            "Material-aware conductance trained on face conductances: does it recover the known ratio?")
    s.add_argument("--seeds", type=int, nargs="+", default=[1729, 2, 3, 4, 5])
    s.add_argument("--device", default="cuda")
    s = add("export-ident", cmd_export_ident, "Write the browser's identifiability data from the ridge profile")
    s.add_argument("--report", type=Path, default=Path("results/v2/physics/inverse-ridge-profile.json"))
    s.add_argument("--out", type=Path, default=Path("web/public/bundle/ident.json"))
