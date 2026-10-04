"""GeoNeural command line. Every expensive step is explicit and writes one report.

Data lives under GEONEURAL_HOME (default ./.data). Commands that need optional
dependencies (torch, numba, zstandard, ...) import them only when they run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from geoneural.common import HOME, read_json, region, utc, write_json

ATLAS = HOME / "atlases" / "essen-ruhr" / "atlas.json"
STAMP = lambda: utc().replace(":", "-")  # noqa: E731
DELEGATED = {"import-geotiff": "geoneural.data.import_geotiff",
             "geology-fields": "geoneural.data.geology_fields"}


def _atlas(p, required=False):
    if required:
        p.add_argument("--atlas", type=Path, required=True)
    else:
        p.add_argument("--atlas", type=Path, default=ATLAS)


def _run(kind, inputs=None, paths=None):
    from geoneural.provenance import run_record
    return run_record(kind, inputs or {}, paths or {})


def _write(report, out, kind, inputs=None, paths=None):
    report["run"] = _run(kind, inputs, paths)
    write_json(out, report)
    print(out)


# Data acquisition and preparation

def cmd_fetch(a):
    from geoneural.data.acquire import fetch_terrain
    print(fetch_terrain(region(a.preset), a.out or HOME / "raw" / a.preset, a.max_mib, a.source_spacing, a.tile_m))


def cmd_geology(a):
    from geoneural.data.acquire import fetch_geology
    print(fetch_geology(region(a.preset), a.out or HOME / "geology" / a.preset, a.max_mib, a.feature_type,
                        a.page_size))


def cmd_prepare(a):
    from geoneural.data.build import prepare
    print(prepare(a.input or HOME / "raw" / a.preset / "input.json", a.out or HOME / "atlases" / a.preset,
                  a.max_mib, a.edge_trim))


def cmd_inspect(a):
    m = read_json(a.atlas)
    print(json.dumps({k: v for k, v in m.items() if k not in ("pages", "provenance")}, indent=2))
    print(f"Pages: {len(m['pages'])}")


def cmd_source_control(a):
    from geoneural.data.source_control import compare
    print(compare(a.fine_input, a.atlas, a.out or HOME / "results" / f"source-control-{STAMP()}.json"))


def cmd_verify_lattice(a):
    from geoneural.data.geodesy import verify
    report = verify(a.atlas, a.landmarks, a.allow_unconfirmed)
    report["run"] = _run("lattice-verification", {"atlas": str(a.atlas)}, {"atlas": a.atlas.parent})
    if a.out:
        write_json(a.out, report)
        print(a.out)
    else:
        print(json.dumps({k: v for k, v in report.items() if k != "run"}, indent=2))
    return 0 if report["passed_structural_checks"] else 1


def cmd_landmarks_hfp(a):
    from geoneural.data import hfp
    report = hfp.run(a.atlas, a.hfp_dir)
    _write(report, a.out, "hfp-landmarks", {"atlas": str(a.atlas)}, {"atlas": a.atlas.parent})
    print(json.dumps(report["verdict"]))


def cmd_landmarks_dlm(a):
    import numpy as np
    from geoneural.data import dlm
    manifest = read_json(a.atlas)
    folder = a.dlm_dir or HOME / "landmarks" / "nrw-basis-dlm" / a.atlas.parent.name
    if not (folder / "receipt.json").exists():
        dlm.fetch(manifest["bounds"], folder)
    reference = np.load(a.atlas.parent / "reference.npy").astype(np.float64)
    report = dlm.inspect(reference, manifest, dlm.parse(folder))
    report.update(atlasContentId=manifest["content_id"], receipt=read_json(folder / "receipt.json"))
    _write(report, a.out, "dlm-inspection", {"atlas": str(a.atlas)}, {"atlas": a.atlas.parent})


def cmd_epochs(a):
    from geoneural.data import epochs
    receipts = read_json(a.atlas.parent / "ATTRIBUTION.json").get("files", [])
    files = receipts.values() if isinstance(receipts, dict) else receipts
    stamps = sorted(f["retrieved_utc"] for f in files if "retrieved_utc" in f)
    if not stamps:
        raise ValueError("the atlas receipts carry no retrieval time")
    manifest = read_json(a.atlas)
    report = epochs.epochs(a.meta, a.index, manifest["bounds"], stamps[0])
    report["atlasContentId"] = manifest["content_id"]
    write_json(a.out, report)
    print(a.out)
    print(json.dumps(report["surveyEpochs"]))


# Codecs and accounting

def cmd_tournament(a):
    from geoneural.codecs.tournament import run
    print(run(a.atlas, a.out or HOME / "results" / f"tournament-{STAMP()}.json", page_limit=a.page_limit,
              only=a.codec))


def cmd_account(a):
    from geoneural.codecs.accounting import package_bytes
    report = package_bytes(a.package)
    if a.out:
        write_json(a.out, report)
        print(a.out)
    else:
        print(json.dumps({k: v for k, v in report.items() if k != "files"}, indent=2))


def cmd_compact_index(a):
    from geoneural.codecs.index import write
    target = a.out or a.atlas.parent / "index.eatidx"
    print(json.dumps(write(a.atlas, target), indent=2))
    print(target)


def cmd_certify_seams(a):
    from geoneural.codecs.bounds import certify
    target = a.out or HOME / "results" / f"seams-{STAMP()}.json"
    report = certify(a.atlas, target, a.samples)
    keys = ("perLevelWorst", "verification", "pagesCertified", "pagesWhereBoundExceedsDeclaredSampleError")
    print(json.dumps({k: report[k] for k in keys}, indent=2))
    print(target)


def cmd_envelope(a):
    from geoneural.codecs import envelope
    report = envelope.sweep(a.atlas, streams=not a.no_streams, stream_cells=a.stream_cells)
    _write(report, a.out or HOME / "results" / f"dense-envelope-{STAMP()}.json", "dense-envelope",
           {"atlas": str(a.atlas)}, {"atlas": a.atlas.parent})


# Drainage diagnostics

def cmd_drainage(a):
    import numpy as np
    from geoneural.codecs.tournament import DEFAULT_TARGETS
    from geoneural.metrics.hydrology import against_targets
    m = read_json(a.atlas)
    reference = np.load(a.atlas.parent / "reference.npy").astype(np.float64)
    report = against_targets(reference, m["spacing_m"], DEFAULT_TARGETS, m["quantum_m"], a.stream_cells)
    report.update(atlas=a.atlas.parent.name, atlasContentId=m["content_id"])
    _write(report, a.out or HOME / "results" / f"drainage-{STAMP()}.json", "drainage-preservation",
           {"atlas": str(a.atlas)}, {"atlas": a.atlas.parent})


def cmd_corrections(a):
    import numpy as np
    from geoneural.metrics.corrections import sweep
    m = read_json(a.atlas)
    reference = np.load(a.atlas.parent / "reference.npy").astype(np.float64)
    report = sweep(reference, m["spacing_m"], a.target_m, tuple(a.radii), m["quantum_m"], a.stream_cells)
    report.update(atlas=a.atlas.parent.name, atlasContentId=m["content_id"])
    _write(report, a.out or HOME / "results" / f"corrections-{STAMP()}.json", "drainage-corrections",
           {"atlas": str(a.atlas)}, {"atlas": a.atlas.parent})


# Neural representations

def _problem(a, scope):
    from geoneural.neural import search as arch
    return arch, arch.Problem(a.atlas, a.device, normalisation=scope)


def _finish_neural(report, a, problem, kind):
    report["atlas"] = a.atlas.parent.name
    report["atlasContentId"] = problem.manifest["content_id"]
    _write(report, a.out, kind, {"atlas": str(a.atlas)}, {"atlas": a.atlas.parent})


def cmd_train(a):
    if not a.ack_experimental:
        raise ValueError("train is the early single-model control; pass --ack-experimental to run it")
    from geoneural.neural.learning import train
    print(train(a.atlas, a.out, a.model, a.steps, a.batch, a.width, a.depth, a.seed, a.device,
                a.checkpoint_every, a.extrapolation_fraction))


def cmd_evaluate_checkpoint(a):
    import numpy as np
    from geoneural.metrics import hydrology
    from geoneural.neural.learning import predict
    m = read_json(a.atlas)
    reference = np.load(a.atlas.parent / "reference.npy").astype(np.float64)
    field, meta, seconds = predict(a.atlas, a.model, a.device)
    errors = np.abs(field.astype(np.float64) - reference)
    drainage = hydrology.compare(reference, field.astype(np.float64), m["spacing_m"], a.stream_cells)
    weights = a.model.parent / "weights.safetensors"
    report = {
        "schema": "geoneural-checkpoint-evaluation-v1", "model": a.model.parent.name, "config": meta["model"],
        "stepsCompleted": meta["steps_completed"], "seed": meta["seed"], "split": meta["split"].get("mode", "holdout"),
        "weightsBytes": weights.stat().st_size, "deployedBytes": weights.stat().st_size,
        "bytesNote": "float32 safetensors file; the two normalisation scalars are inside model.json and "
                     "add 16 bytes of information",
        "fullReference": {"mae_m": float(errors.mean()), "rmse_m": float(np.sqrt((errors ** 2).mean())),
                          "p95_m": float(np.quantile(errors, 0.95)), "p99_m": float(np.quantile(errors, 0.99)),
                          "max_m": float(errors.max())},
        "drainage": {k: drainage[k] for k in ("streamJaccard", "streamRecall", "receiverAgreementFraction",
                                              "basinAgreementFraction", "referenceBasins", "reconstructedBasins")},
        "decodeSeconds": seconds, "device": a.device, "atlas": a.atlas.parent.name}
    if a.save_field:
        a.save_field.parent.mkdir(parents=True, exist_ok=True)
        np.save(a.save_field, field.astype(np.float32), allow_pickle=False)
        report["savedField"] = a.save_field.name
    _write(report, a.out, "checkpoint-evaluation", {"atlas": str(a.atlas), "model": str(a.model)},
           {"atlas": a.atlas.parent})


def cmd_bake(a):
    from geoneural.neural.learning import bake
    print(bake(a.atlas, a.model, a.out, a.device))


def cmd_finalists(a):
    from geoneural.neural import search as arch
    chosen = arch.finalists(read_json(a.report), a.per_family, a.limit)
    write_json(a.out, chosen)
    print(f"{a.out}: {len(chosen)} finalists")


def cmd_decode_latency(a):
    from geoneural.neural import query_cost
    arch, problem = _problem(a, "all")
    report = query_cost.run_configs(read_json(a.chosen), problem, a.trials, a.steps, a.batch, a.seed,
                                    not a.no_verify_pages, a.cache_state, a.qualified, "float16",
                                    a.engine, a.data_path, a.sampling)
    _finish_neural(report, a, problem, "decode-latency")


def cmd_quantised_ladder(a):
    from geoneural.neural import quantise
    arch, problem = _problem(a, "all")
    report = quantise.ladder(read_json(a.chosen), problem, a.steps, a.batch, a.finetune_steps,
                             tuple(a.widths), a.seed, a.engine, a.data_path, a.sampling)
    _finish_neural(report, a, problem, "quantised-ladder")


def cmd_equivalence(a):
    arch, problem = _problem(a, "train")
    report = arch.equivalence(problem, a.steps, a.batch, tuple(a.seeds))
    _finish_neural(report, a, problem, "data-path-equivalence")


def cmd_multiregion(a):
    from geoneural.neural import multiregion, search as arch
    report = multiregion.amortisation(
        [str(x) for x in a.atlas], {"kind": "shared", "width": a.width, "depth": a.depth, "latent": a.latent},
        arch.Recipe(steps=a.steps, batch=a.batch, device=a.device), a.device, a.store_precision,
        a.conventional_target_m, a.transfer_region, tuple(a.seeds), a.objective)
    _write(report, a.out, "multiregion-amortisation", {"atlases": [str(x) for x in a.atlas]},
           {"atlas": a.atlas[0].parent})


def cmd_study(a):
    """screen, search, confirm, codec-fit, ladder and context-ablation share one problem setup.

    Codec work may normalise with whole-reference statistics: they are two stored scalars of
    side information. A predictive holdout may not, because a withheld page must not reach the
    model even through a mean.
    """
    if a.normalisation:
        scope = a.normalisation
    elif a.command in ("codec-fit", "ladder"):
        scope = "all"
    elif a.command == "context-ablation":
        scope = "train"
    else:
        scope = "train" if getattr(a, "mode", "holdout") == "holdout" else "all"
    arch, problem = _problem(a, scope)
    if a.command == "screen":
        report = arch.screen(problem, tuple(a.family) if a.family else arch.FAMILIES, a.steps, a.batch, a.seed,
                             None, a.store_precision)
    elif a.command == "search":
        report = _search(a, arch, problem)
    elif a.command == "confirm":
        report = arch.confirm(read_json(a.chosen), problem, tuple(a.seeds), a.steps, a.batch,
                              store_precision=a.store_precision)
    elif a.command == "ladder":
        report = arch.ladder(read_json(a.chosen), problem, a.seed, a.steps, a.batch)
    elif a.command == "context-ablation":
        import glob
        from geoneural.data import geology
        units = geology.parse_units(glob.glob(str(a.geology / "*.gml")))
        cfg = read_json(a.geology / "geology.json")["config"]
        raster = geology.rasterise(units, cfg["bbox"], problem.side, cfg["spacing_m"])
        arms = geology.ablation_contexts(raster["classes"], raster["classCount"], len(units))
        report = arch.context_ablation(
            problem, arms,
            {"kind": "context", "width": a.width, "depth": a.depth, "embedding": a.embedding,
             "mode": a.conditioning},
            arch.Recipe(steps=a.steps, batch=a.batch, device=a.device), tuple(a.seeds), a.store_precision,
            None if a.base_level is None else (a.base_level, a.base_target_m))
        report["geology"] = {k: v for k, v in raster.items() if k != "classes"}
    else:
        report = arch.codec_fit(read_json(a.chosen), problem, tuple(a.seeds), a.steps, a.batch, a.stream_cells,
                                store_precision=a.store_precision, fields_dir=a.save_fields)
    report["split"] = arch.splits.summary(problem.split)
    report["normalisation"] = {
        "scope": problem.normalisation, "meanM": problem.mean, "scaleM": problem.scale,
        "note": 'Two stored scalars. "all" uses the whole reference and is allowed for codec encoding as '
                'stored side information; "train" uses the training pages only and is required for any '
                'predictive holdout claim.'}
    # Every report that names a neural rate carries the full conventional envelope, read on the
    # split the study scored, including downsample-and-upsample, the strongest competitor at the
    # rates a small network occupies.
    split = "selection" if (a.command == "screen" or getattr(a, "mode", "") == "holdout") else "all"
    report["conventional"] = arch.conventional_envelope(problem, split=split)
    arch.annotate_residual_fronts(report, problem, split=split)
    arch.annotate_convergence(report, problem, split=split)
    report["rateDenominators"] = arch.rate_denominators(problem)
    report["createdUtc"] = utc()
    _finish_neural(report, a, problem, "architecture-" + a.command)


def _search(a, arch, problem):
    """Per-family search, written out after every family so an interruption loses at most one."""
    report = {"schema": arch.SCHEMA, "mode": "per-family architecture search", "objective": a.mode,
              "storePrecision": a.store_precision,
              "compute": {"dataPath": a.data_path, "sampling": a.sampling, "engine": a.engine},
              "complete": False, "familiesRequested": list(a.family), "byFamily": {}}
    # The comparator comes first so a partial report can be read on its own.
    report["conventional"] = arch.conventional_envelope(
        problem, split="selection" if a.mode == "holdout" else "all")
    report["rateDenominators"] = arch.rate_denominators(problem)
    report["split"] = arch.splits.summary(problem.split)
    partial = a.out.with_suffix(".partial.json")
    if a.resume and partial.exists():
        prior = read_json(partial)
        if (prior.get("objective") == a.mode and prior.get("storePrecision") == a.store_precision
                and prior.get("familiesRequested") == list(a.family)):
            report["byFamily"].update(prior.get("byFamily", {}))
            print(f"resuming: {', '.join(report['byFamily'])} already done", flush=True)
        else:
            print("a partial report exists for a different study; ignoring it", flush=True)
    for family in a.family:
        if family in report["byFamily"]:
            continue
        report["byFamily"][family] = arch.study(family, problem, a.trials, a.steps, a.batch, a.seed,
                                                a.byte_ceiling, a.sampler_seed, True, a.mode,
                                                a.store_precision, a.data_path, a.sampling, a.engine)
        report["familiesCompleted"] = list(report["byFamily"])
        write_json(partial, report)
        print(f"partial report written after {family}: {partial}", flush=True)
    report["complete"] = True
    partial.unlink(missing_ok=True)
    return report


# Super-resolution

def cmd_fine_reference(a):
    """The 1 m reference of a region and its 10 m observation through the declared operator."""
    import numpy as np
    from geoneural.superres import superres
    folder = a.out or HOME / "fine" / a.preset
    folder.mkdir(parents=True, exist_ok=True)
    report = superres.fine_reference(a.input or HOME / "raw" / f"{a.preset}-1m" / "input.json",
                                     folder / "reference_1m.npy")
    fine = np.load(folder / "reference_1m.npy", mmap_mode="r")
    np.save(folder / "coarse_from_operator.npy", superres.observe(fine), allow_pickle=False)
    write_json(folder / "fine-reference.json", report)
    print(folder)


def cmd_superres_classical(a):
    import numpy as np
    from geoneural.superres import superres
    report = {"schema": "geoneural-superres-classical-v1", "byRegion": {}}
    for name in a.regions:
        folder = a.root / name
        coarse = np.load(folder / "coarse_from_operator.npy")
        report["byRegion"][name] = superres.classical_baselines(folder / "reference_1m.npy", coarse)
    _write(report, a.out, "superres-classical", {"regions": a.regions})


def cmd_superres_neural(a):
    import torch
    from geoneural.superres import superres_train
    report = superres_train.campaign(
        a.arms, a.train_region, a.test_regions, torch, steps=a.steps, patch=a.patch, queries=a.queries,
        batch=a.batch, lr=a.lr, seed=a.seed, device=a.device, width=a.width, blocks=a.blocks,
        hidden=a.hidden, depth=a.depth, evaluation_tiles=a.evaluation_tiles, root=str(a.root),
        drainage_window=a.drainage_window, drainage_origin=a.drainage_origin, stream_cells=a.stream_cells)
    _write(report, a.out, "superres-neural", {"trainRegion": a.train_region})


# Landscape physics

def cmd_teacher_audit(a):
    from geoneural.physics import teacher_audit
    _write(teacher_audit.run(), a.out, "teacher-audit")


def cmd_ensemble(a):
    from geoneural.physics import ensemble
    report = ensemble.generate(a.out, count=a.count, side=a.side, spacing_m=a.spacing_m, years=a.years,
                               seed=a.seed, workers=a.workers, frames=a.frames)
    print(json.dumps({k: v for k, v in report.items() if not isinstance(v, (list, dict))}, indent=1))


def cmd_emulator(a):
    import torch
    from geoneural.physics import emulator_train
    report = emulator_train.campaign(a.ensemble, torch, kinds=tuple(a.kinds), steps=a.steps, batch=a.batch,
                                     lr=a.lr, seed=a.seed, device=a.device)
    _write(report, a.out, "emulator", {"ensemble": str(a.ensemble)})


def cmd_flux_closure(a):
    import torch
    from geoneural.physics import hybrid
    report = hybrid.campaign(torch, arms=tuple(a.arms), steps=a.steps, side=a.side, spacing_m=a.spacing_m,
                             device=a.device, seed=a.seed)
    _write(report, a.out, "flux-closure")


def cmd_flux_closure_seeds(a):
    import torch
    from geoneural.physics import hybrid
    report = hybrid.seed_study(torch, seeds=tuple(a.seeds), penalty_weights=tuple(a.penalty_weights),
                               steps=a.steps, side=a.side, spacing_m=a.spacing_m, device=a.device)
    _write(report, a.out, "flux-closure-seeds")


def cmd_event_field(a):
    import torch
    from geoneural.physics import structure
    report = structure.campaign(torch, side=a.side, depth=a.depth, steps=a.steps, device=a.device, seed=a.seed)
    _write(report, a.out, "event-field")


def cmd_inverse_history(a):
    from geoneural.physics import inverse
    report = inverse.campaign(side=a.side, chains=a.chains, draws=a.draws, burn=a.burn, seed=a.seed,
                              workers=a.workers)
    _write(report, a.out, "inverse-history")


def cmd_distillation(a):
    import torch
    from geoneural.physics import distillation
    report = distillation.campaign(a.coarse, torch, spacing_m=a.spacing_m, side=a.side, steps=a.steps,
                                   device=a.device, seeds=tuple(a.seeds))
    _write(report, a.out, "distillation")


# Results export

def cmd_candidates(a):
    from geoneural.export.candidates import build
    table = build(a.results)
    if not table["sources"]:
        raise ValueError(f"no reports under {a.results}/reproduced or {a.results}/recovered; run publish first")
    write_json(a.out, table)
    print(f"{a.out}: {len(table['candidates'])} candidates")


def _portable(value, replacements):
    """The same JSON value with machine-specific path prefixes replaced."""
    if isinstance(value, str):
        for old, new in replacements:
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [_portable(v, replacements) for v in value]
    if isinstance(value, dict):
        return {k: _portable(v, replacements) for k, v in value.items()}
    return value


def cmd_publish(a):
    """Copy reports into the repository with local paths replaced, so they can be shared."""
    replacements = [(str(HOME) + "/", "$GEONEURAL_HOME/"), (str(HOME), "$GEONEURAL_HOME"),
                    (str(Path.cwd()) + "/", ""), ("file://" + str(Path.cwd()), "file://."),
                    (str(Path.home()), "~")]
    a.to.mkdir(parents=True, exist_ok=True)
    names = a.names or sorted(p.name for p in a.source.glob("*.json"))
    for name in names:
        report = _portable(read_json(a.source / name), replacements)
        text = json.dumps(report)
        machine = (str(Path.home()), "/" + "mnt/", "/" + "home/", "C:" + "\\")
        if any(marker in text for marker in machine):
            raise ValueError(f"{name} still contains a machine path after rewriting")
        write_json(a.to / name, report)
        print(a.to / name)


def cmd_summary(a):
    from geoneural.export.summary import build
    write_json(a.out, build(a.results, read_json(a.candidates), a.local))
    print(a.out)


def cmd_figures(a):
    from geoneural.export.figures import build
    for path in build(a.atlas, read_json(a.candidates), a.out):
        print(path)


def cmd_export_web(a):
    from geoneural.export.web import export
    geology = a.geology or next(p for p in (HOME / "geology" / "essen-ruhr", SAMPLE / "geology") if p.exists())
    print(export(a.out, read_json(a.candidates), a.atlas, a.fields, a.lab, geology))


# IO benchmarks

def cmd_benchmark(a):
    from geoneural.bench.benchmark import run
    print(run(a.atlas, a.out or HOME / "results" / f"benchmark-{STAMP()}.json", a.repeats, a.limit))


def cmd_io_bench(a):
    from geoneural.bench.bench_io import run
    print(run(a.atlas, a.out or HOME / "results" / f"io-{STAMP()}.json", a.repeats, a.extra_root))


def cmd_paired_io(a):
    from geoneural.bench.paired import run
    print(run(a.atlas, a.out or HOME / "results" / f"paired-io-{STAMP()}.json", a.trials, a.extra_root))


def cmd_io_qualify(a):
    from geoneural.bench import io_qualify
    print(io_qualify.run(a.atlas, a.out, a.extra_root, a.processes))


SAMPLE = Path(__file__).resolve().parents[2] / "data" / "sample" / "essen-ruhr"
# gzip output differs between Python builds, so the check uses codecs whose bytes are fixed by the lock.
SAMPLE_CODECS = ("q32-delta-zstd", "zfp-accuracy", "sz3-absolute")


def sample_summary(atlas: Path, results: Path) -> dict:
    """The numbers the sample reproduction is checked on."""
    tournament = read_json(results / "tournament.json")
    drainage = read_json(results / "drainage.json")
    corrections = read_json(results / "corrections.json")
    return {
        "referenceSha256": read_json(atlas)["reference_sha256"],
        "payloadBytes": {f"{r['codec']}@{o['target_m']:g}": o["payload_bytes"]
                         for r in tournament["results"] if r.get("available") for o in r["observations"]},
        "streamJaccard": {f"{t['targetMaxErrorM']:g}": t["streamJaccard"] for t in drainage["byTarget"]},
        "correctionBytes": {str(b["bandRadiusCells"]): b["correctionBytes"] for b in corrections["byBandRadius"]},
        "correctedJaccard": {str(b["bandRadiusCells"]): b["streamJaccard"] for b in corrections["byBandRadius"]},
    }


def cmd_reproduce(a):
    """Rebuild the Essen reference from the shipped tiles and rerun the core comparisons on it."""
    import numpy as np
    from geoneural.codecs.tournament import DEFAULT_TARGETS, run as tournament
    from geoneural.data.build import prepare
    from geoneural.metrics.corrections import sweep
    from geoneural.metrics.hydrology import against_targets
    if not (a.sample / "raw" / "input.json").exists():
        raise ValueError(f"no sample at {a.sample}; run from a source checkout or pass --sample")
    out = a.out or HOME / "results" / f"sample-{STAMP()}"
    out.mkdir(parents=True, exist_ok=False)
    atlas = prepare(a.sample / "raw" / "input.json", out / "atlas", 512, 2)
    expected = read_json(a.sample / "expected.json") if (a.sample / "expected.json").exists() else None
    if expected and not a.record and read_json(atlas)["reference_sha256"] != expected["referenceSha256"]:
        raise RuntimeError("the rebuilt reference differs from the recorded one; stopping before any comparison")
    m = read_json(atlas)
    reference = np.load(atlas.parent / "reference.npy").astype(np.float64)
    tournament(atlas, out / "tournament.json", only=list(SAMPLE_CODECS))
    write_json(out / "drainage.json", against_targets(reference, m["spacing_m"], DEFAULT_TARGETS, m["quantum_m"]))
    write_json(out / "corrections.json", sweep(reference, m["spacing_m"], 1.0, fine_quantum=m["quantum_m"]))
    summary = sample_summary(atlas, out)
    write_json(out / "summary.json", summary)
    if a.record:
        write_json(a.sample / "expected.json", summary)
        print(f"recorded {a.sample / 'expected.json'}")
        return 0
    if expected is None:
        raise RuntimeError("no expected.json next to the sample; run once with --record")
    mismatches = [f"{group}/{key}: expected {value}, got {summary[group].get(key)}"
                  for group, values in expected.items() if isinstance(values, dict)
                  for key, value in values.items() if summary[group].get(key) != value]
    print(json.dumps({"output": str(out), "checked": sum(len(v) for v in expected.values() if isinstance(v, dict)),
                      "mismatches": mismatches}, indent=1))
    return 1 if mismatches else 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="geoneural", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True, metavar="command")

    def add(name, func, help_text):
        s = sub.add_parser(name, help=help_text, description=help_text)
        s.set_defaults(func=func)
        return s

    # Data
    s = add("fetch", cmd_fetch, "Download a bounded NRW DGM1 region from the provider's WCS")
    s.add_argument("--preset", default="essen-ruhr")
    s.add_argument("--out", type=Path)
    s.add_argument("--max-mib", type=int, default=512)
    s.add_argument("--source-spacing", type=float)
    s.add_argument("--tile-m", type=float, help="Ground size of one request; reduce it for finer source spacing")
    s = add("geology", cmd_geology, "Download bounded GK100 geology vectors and attribution")
    s.add_argument("--preset", default="essen-ruhr")
    s.add_argument("--out", type=Path)
    s.add_argument("--max-mib", type=int, default=128)
    s.add_argument("--feature-type")
    s.add_argument("--page-size", type=int, help="Features per WFS page (default 250)")
    s = add("prepare", cmd_prepare, "Build the reference lattice and the compressed page pyramid")
    s.add_argument("--preset", default="essen-ruhr")
    s.add_argument("--input", type=Path)
    s.add_argument("--out", type=Path)
    s.add_argument("--max-mib", type=int, default=512)
    s.add_argument("--edge-trim", type=int, default=2, help="Discard this many reprojected samples at each tile edge")
    s = add("inspect", cmd_inspect, "Show atlas metadata without reading terrain payloads")
    _atlas(s)
    s = add("source-control", cmd_source_control, "Compare the 10 m service response with the 1 m source")
    s.add_argument("--fine-input", type=Path, default=HOME / "raw" / "essen-ruhr-1m" / "input.json")
    _atlas(s)
    s.add_argument("--out", type=Path)
    s = add("verify-lattice", cmd_verify_lattice, "Check axis order, datum, orientation and codec agreement")
    _atlas(s)
    s.add_argument("--landmarks", type=Path, help="Published elevations supplied by the operator")
    s.add_argument("--allow-unconfirmed", action="store_true", help="Report unconfirmed landmarks as provisional")
    s.add_argument("--out", type=Path)
    s = add("landmarks-hfp", cmd_landmarks_hfp, "Registration and datum against NRW official height benchmarks")
    _atlas(s)
    s.add_argument("--hfp-dir", type=Path, default=HOME / "landmarks" / "nrw-hfp",
                   help="hfp_pl.csv and hfp_plzf.csv from opengeodata.nrw.de")
    s.add_argument("--out", type=Path, required=True)
    s = add("landmarks-dlm", cmd_landmarks_dlm,
            "Terrain features against the ATKIS Basis-DLM (water levels, flow, valleys, crossings)")
    _atlas(s)
    s.add_argument("--dlm-dir", type=Path, help="Stored WFS pages; fetched for the atlas extent when absent")
    s.add_argument("--out", type=Path, required=True)
    s = add("epochs", cmd_epochs, "Provider survey epochs of the tiles under an atlas")
    _atlas(s)
    s.add_argument("--meta", type=Path, default=HOME / "epochs" / "nrw-dgm1" / "dgm1_meta.zip")
    s.add_argument("--index", type=Path, default=HOME / "epochs" / "nrw-dgm1" / "dgm1_tiff_index.xml")
    s.add_argument("--out", type=Path, required=True)
    sub.add_parser("import-geotiff", help="Import local GeoTIFFs as a terrain input (own options)")
    sub.add_parser("geology-fields", help="Inventory and rasterise one GK100 attribute (own options)")

    # Codecs
    s = add("tournament", cmd_tournament, "Conventional codec rate-distortion tournament")
    _atlas(s)
    s.add_argument("--out", type=Path)
    s.add_argument("--page-limit", type=int)
    s.add_argument("--codec", action="append", help="Restrict to named codecs; repeatable")
    s = add("account", cmd_account, "Account for every shipped byte of a published package")
    s.add_argument("--package", type=Path, default=ATLAS.parent)
    s.add_argument("--out", type=Path)
    s = add("compact-index", cmd_compact_index, "Pack the JSON page index into EATIDX1 and report the saving")
    _atlas(s)
    s.add_argument("--out", type=Path)
    s = add("certify-seams", cmd_certify_seams, "Conservative bound on mixed-level seams between pyramid pages")
    _atlas(s)
    s.add_argument("--samples", type=int, default=200000)
    s.add_argument("--out", type=Path)

    s = add("envelope", cmd_envelope, "Dense conventional envelope: every pyramid level at many error targets")
    _atlas(s)
    s.add_argument("--stream-cells", type=int, default=500)
    s.add_argument("--no-streams", action="store_true", help="Skip the drainage comparison per point")
    s.add_argument("--out", type=Path)

    # Drainage
    s = add("drainage", cmd_drainage, "Drainage preservation under bounded height error")
    _atlas(s)
    s.add_argument("--stream-cells", type=int, default=500)
    s.add_argument("--out", type=Path)
    s = add("corrections", cmd_corrections, "Sparse exact corrections around streams at a coarse error target")
    _atlas(s)
    s.add_argument("--target-m", type=float, default=1.0)
    s.add_argument("--radii", type=int, nargs="+", default=[0, 1, 2, 4])
    s.add_argument("--stream-cells", type=int, default=500)
    s.add_argument("--out", type=Path)

    # Neural
    s = add("train", cmd_train, "Early single-model control (SIREN, Fourier or shared)")
    _atlas(s)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--model", choices=["siren", "fourier", "shared"], default="siren")
    s.add_argument("--steps", type=int, default=2000)
    s.add_argument("--batch", type=int, default=8192)
    s.add_argument("--width", type=int, default=128)
    s.add_argument("--depth", type=int, default=3)
    s.add_argument("--device", default="cpu")
    s.add_argument("--seed", type=int, default=1729)
    s.add_argument("--checkpoint-every", type=int, default=500, help="Save every N steps; 0 disables")
    s.add_argument("--extrapolation-fraction", type=float, default=0.25,
                   help="Share of pages withheld as one contiguous unseen block")
    s.add_argument("--ack-experimental", action="store_true")
    s = add("evaluate-checkpoint", cmd_evaluate_checkpoint, "Decode a saved model and score it as a codec, drainage included")
    _atlas(s)
    s.add_argument("--model", type=Path, required=True, help="model.json next to weights.safetensors")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--device", default="cuda")
    s.add_argument("--stream-cells", type=int, default=500)
    s.add_argument("--save-field", type=Path, help="Also save the decoded heights as a .npy file")
    s = add("bake", cmd_bake, "Decode a trained model into atlas pages")
    _atlas(s)
    s.add_argument("--model", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--device", default="cpu")

    gpu = dict(device="cuda", steps=5000, batch=8192)

    def neural(name, func, help_text, chosen=False, seeds=None, store=None):
        s = add(name, func, help_text)
        _atlas(s)
        s.add_argument("--out", type=Path, required=True)
        s.add_argument("--device", default=gpu["device"])
        s.add_argument("--steps", type=int, default=gpu["steps"])
        s.add_argument("--batch", type=int, default=gpu["batch"])
        if chosen:
            s.add_argument("--chosen", type=Path, required=True, help="JSON mapping name -> {config, recipe}")
        if seeds:
            s.add_argument("--seeds", type=int, nargs="+", default=seeds)
        else:
            s.add_argument("--seed", type=int, default=1729)
        if store:
            s.add_argument("--store-precision", choices=["float64", "float32", "float16"], default=store,
                           help="Width the stored tensors ship at; the study prices and measures at it")
        s.add_argument("--normalisation", choices=["all", "train"], default=None,
                       help="Where the two normalisation scalars come from; 'train' is required for holdout claims")
        s.add_argument("--data-path", choices=("host", "device"), default="device",
                       help="Keep the lattice on the GPU ('device', faster, different random stream) or not")
        s.add_argument("--sampling", choices=("without", "with", "reshuffle"), default=None,
                       help="Batch draw policy; defaults to 'without' on the host path and 'with' on the device path")
        s.add_argument("--engine", choices=("eager", "cudagraph"), default="cudagraph",
                       help="Replay the training step as a CUDA graph; execution detail, not study identity")
        return s

    s = neural("screen", cmd_study, "Every architecture family at one matched recipe", store="float16")
    s.add_argument("--family", action="append", help="Restrict to these families; repeatable")
    s = neural("search", cmd_study, "Rate-distortion architecture search per family", store="float16")
    s.add_argument("--family", action="append", required=True, help="Family to search; repeatable")
    s.add_argument("--trials", type=int, default=30)
    s.add_argument("--byte-ceiling", type=int, default=1200000)
    s.add_argument("--sampler-seed", type=int, default=20260912)
    s.add_argument("--mode", choices=["holdout", "codec"], default="holdout",
                   help="holdout: train on training pages and score held-out pages (generalisation). "
                        "codec: train and score every page (rate-distortion). Different questions.")
    s.add_argument("--resume", action="store_true",
                   help="Reuse finished families from a matching .partial.json")
    neural("confirm", cmd_study, "Retrain chosen configurations across seeds and score the test pages",
           chosen=True, seeds=[1729, 20260912, 31337], store="float16")
    s = neural("codec-fit", cmd_study, "Train on every page and judge as a codec, drainage included",
               chosen=True, seeds=[1729], store="float32")
    s.add_argument("--stream-cells", type=int, default=500)
    s.add_argument("--save-fields", type=Path, help="Directory for the decoded heights of every model")
    neural("ladder", cmd_study, "Storage-precision ladder, code coverage and amortisation per candidate",
           chosen=True)
    s = neural("context-ablation", cmd_study, "Geology conditioning: none, real, misaligned, shuffled, generic",
               seeds=[1729, 20260912, 31337], store="float16")
    s.add_argument("--geology", type=Path, required=True)
    s.add_argument("--width", type=int, default=128)
    s.add_argument("--depth", type=int, default=3)
    s.add_argument("--embedding", type=int, default=8)
    s.add_argument("--conditioning", default="film", choices=("film", "concat"))
    s.add_argument("--base-level", type=int, default=None,
                   help="Condition a correction to this conventional pyramid level instead of the whole field")
    s.add_argument("--base-target-m", type=float, default=1.0,
                   help="Max-error target the coarse base is re-encoded at; only used with --base-level")
    s = neural("quantised-ladder", cmd_quantised_ladder,
               "Price networks like a codec: integer weights, real entropy coding, error after quantisation",
               chosen=True)
    s.add_argument("--finetune-steps", type=int, default=1500)
    s.add_argument("--widths", type=int, nargs="+", default=[8, 6, 4])
    s = neural("decode-latency", cmd_decode_latency, "Decode cost of chosen models at renderer query sizes",
               chosen=True)
    s.add_argument("--trials", type=int, default=7)
    s.add_argument("--cache-state", default="warm", choices=("cold-process", "cold-code", "cold-source", "warm"))
    s.add_argument("--qualified", action="store_true", help="Refuse to run unless the host is serial and idle")
    s.add_argument("--no-verify-pages", action="store_true",
                   help="Skip the page hash check (biases the comparison toward the conventional arm)")
    neural("equivalence", cmd_equivalence, "Check that the device data path is the same experiment as the host path",
           seeds=[1729, 20260912, 31337])
    s = add("finalists", cmd_finalists, "Turn a search report into a chosen-configuration file by a stated rule")
    s.add_argument("--report", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--per-family", type=int, default=2)
    s.add_argument("--limit", type=int, default=8)
    s = add("multiregion", cmd_multiregion, "Does a decoder shared across regions amortise its bytes?")
    s.add_argument("--atlas", type=Path, action="append", required=True, help="Atlas manifest; repeatable, two or more")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--device", default="cuda")
    s.add_argument("--steps", type=int, default=5000)
    s.add_argument("--batch", type=int, default=8192)
    s.add_argument("--width", type=int, default=128)
    s.add_argument("--depth", type=int, default=3)
    s.add_argument("--latent", type=int, default=16)
    s.add_argument("--seeds", type=int, nargs="+", default=[1729])
    s.add_argument("--store-precision", default="float16")
    s.add_argument("--conventional-target-m", type=float, default=1.0)
    s.add_argument("--objective", default="codec", choices=("codec", "holdout"),
                   help="codec: train every page of the training regions and hold out a region")
    s.add_argument("--transfer-region", default=None, help="Withhold this region from the backbone, then fit its codes")

    # Super-resolution
    s = add("fine-reference", cmd_fine_reference, "Mosaic a 1 m download and observe it at 10 m with the declared operator")
    s.add_argument("--preset", default="essen-ruhr")
    s.add_argument("--input", type=Path, help="input.json of the 1 m download (default raw/<preset>-1m)")
    s.add_argument("--out", type=Path)
    s = add("superres-classical", cmd_superres_classical, "Classical 10 m to 1 m interpolation baselines")
    s.add_argument("--root", type=Path, default=HOME / "fine")
    s.add_argument("--regions", nargs="+", default=["essen-ruhr", "muensterland-plain", "rothaar-sauerland"])
    s.add_argument("--out", type=Path, required=True)
    s = add("superres-neural", cmd_superres_neural, "Neural super-resolution arms against the classical bar")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--root", type=Path, default=HOME / "fine")
    s.add_argument("--train-region", default="essen-ruhr")
    s.add_argument("--test-regions", nargs="+", default=["muensterland-plain", "rothaar-sauerland"])
    s.add_argument("--arms", nargs="+", default=["liif", "residual-siren", "edsr"])
    for flag, kind, default in (("--steps", int, 3000), ("--patch", int, 17), ("--queries", int, 4096),
                                ("--batch", int, 8), ("--lr", float, 3e-4), ("--seed", int, 1729),
                                ("--width", int, 64), ("--blocks", int, 4), ("--hidden", int, 256),
                                ("--depth", int, 4), ("--evaluation-tiles", int, 512),
                                ("--drainage-window", int, 205), ("--drainage-origin", int, 100),
                                ("--stream-cells", int, 50000)):
        s.add_argument(flag, type=kind, default=default)
    s.add_argument("--device", default="cuda")

    # Physics
    s = add("teacher-audit", cmd_teacher_audit, "Convergence, conservation and slope-area audit of the landscape teacher")
    s.add_argument("--out", type=Path, required=True)
    s = add("ensemble", cmd_ensemble, "Synthetic landscape ensemble for emulator training")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--count", type=int, default=400)
    s.add_argument("--side", type=int, default=128)
    s.add_argument("--spacing-m", type=float, default=50.0)
    s.add_argument("--years", type=float, default=2_000_000.0)
    s.add_argument("--seed", type=int, default=20260914)
    s.add_argument("--workers", type=int, default=0)
    s.add_argument("--frames", type=int, default=9)
    s = add("emulator", cmd_emulator, "Train the U-Net and FNO landscape emulators and gate them")
    s.add_argument("--ensemble", type=Path, default=HOME / "ensemble" / "main")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--kinds", nargs="+", default=["unet", "fno"])
    s.add_argument("--steps", type=int, default=2000)
    s.add_argument("--batch", type=int, default=8)
    s.add_argument("--lr", type=float, default=1e-3)
    s.add_argument("--seed", type=int, default=1729)
    s.add_argument("--device", default="cuda")
    s = add("flux-closure", cmd_flux_closure, "Learned hillslope closure that conserves mass by construction")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--arms", nargs="+", default=["flux", "kfield", "penalty"])
    s.add_argument("--steps", type=int, default=1500)
    s.add_argument("--side", type=int, default=48)
    s.add_argument("--spacing-m", type=float, default=50.0)
    s.add_argument("--seed", type=int, default=1729)
    s.add_argument("--device", default="cuda")
    s = add("flux-closure-seeds", cmd_flux_closure_seeds,
            "Flux closure against unconstrained and penalised arms over several seeds and penalty weights")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--seeds", type=int, nargs="+", default=[1729, 2, 3, 4, 5])
    s.add_argument("--penalty-weights", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0])
    s.add_argument("--steps", type=int, default=1500)
    s.add_argument("--side", type=int, default=48)
    s.add_argument("--spacing-m", type=float, default=50.0)
    s.add_argument("--device", default="cuda")
    s = add("event-field", cmd_event_field, "A fault as a coordinate transform, with the ablations that should fail")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--side", type=int, default=40)
    s.add_argument("--depth", type=int, default=20)
    s.add_argument("--steps", type=int, default=4000)
    s.add_argument("--seed", type=int, default=1729)
    s.add_argument("--device", default="cuda")
    s = add("inverse-history", cmd_inverse_history, "What a present-day surface can and cannot identify")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--side", type=int, default=24)
    s.add_argument("--chains", type=int, default=4)
    s.add_argument("--draws", type=int, default=400)
    s.add_argument("--burn", type=int, default=150)
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--seed", type=int, default=1729)
    s = add("distillation", cmd_distillation, "Does a physics prior help at equal bytes and equal capacity?")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--coarse", type=Path, default=HOME / "fine" / "rothaar-sauerland" / "coarse_from_operator.npy")
    s.add_argument("--side", type=int, default=257)
    s.add_argument("--spacing-m", type=float, default=10.0)
    s.add_argument("--steps", type=int, default=3000)
    s.add_argument("--device", default="cuda")
    s.add_argument("--seeds", type=int, nargs="+", default=[1729, 20260914, 31337])

    # Export
    s = add("candidates", cmd_candidates, "Collect every measured representation into one candidate table")
    s.add_argument("--results", type=Path, default=Path("results"),
                   help="Folder with reproduced/ and recovered/ report subfolders")
    s.add_argument("--out", type=Path, default=Path("results") / "candidates.json")
    s = add("summary", cmd_summary, "Collect the numbers the documentation quotes into one file")
    s.add_argument("--results", type=Path, default=Path("results"))
    s.add_argument("--candidates", type=Path, default=Path("results") / "candidates.json")
    s.add_argument("--local", type=Path, default=HOME / "results", help="Reports that are not published")
    s.add_argument("--out", type=Path, default=Path("results") / "summary.json")
    s = add("figures", cmd_figures, "Draw the README figures from the candidate table")
    _atlas(s)
    s.add_argument("--candidates", type=Path, default=Path("results") / "candidates.json")
    s.add_argument("--out", type=Path, default=Path("docs") / "figures")
    s = add("publish", cmd_publish, "Copy reports into the repository with machine paths replaced")
    s.add_argument("--source", type=Path, default=HOME / "results" / "reproduced")
    s.add_argument("--to", type=Path, default=Path("results") / "reproduced")
    s.add_argument("names", nargs="*", help="Report file names; default all")
    s = add("export-web", cmd_export_web, "Write the data bundle the browser viewer and lab read")
    _atlas(s)
    s.add_argument("--out", type=Path, default=Path("web") / "public" / "bundle")
    s.add_argument("--candidates", type=Path, default=Path("results") / "candidates.json")
    s.add_argument("--fields", type=Path, default=HOME / "fields", help="Saved decoded fields of learned candidates")
    s.add_argument("--lab", type=Path, default=Path("native") / "fixtures", help="closure.json and weights")
    s.add_argument("--geology", type=Path, default=None,
                   help="GK100 extract; defaults to GEONEURAL_HOME/geology/essen-ruhr, then the shipped sample")

    # Reproduction
    s = add("reproduce", cmd_reproduce, "Rebuild the Essen sample and check the core results against the record")
    s.add_argument("--sample", type=Path, default=SAMPLE)
    s.add_argument("--out", type=Path)
    s.add_argument("--record", action="store_true", help="Write expected.json instead of checking against it")

    # IO
    s = add("benchmark", cmd_benchmark, "Local disk, decode and codec benchmark")
    _atlas(s)
    s.add_argument("--out", type=Path)
    s.add_argument("--repeats", type=int, default=3)
    s.add_argument("--limit", type=int, default=128)
    s = add("io-bench", cmd_io_bench, "Random-access IO and read amplification")
    _atlas(s)
    s.add_argument("--repeats", type=int, default=3)
    s.add_argument("--extra-root", type=Path, help="Second filesystem to stage the package on")
    s.add_argument("--out", type=Path)
    s = add("paired-io", cmd_paired_io, "Layout comparison robust to a contended host")
    _atlas(s)
    s.add_argument("--trials", type=int, default=15)
    s.add_argument("--extra-root", type=Path)
    s.add_argument("--out", type=Path)
    s = add("io-qualify", cmd_io_qualify, "Qualified cold and warm random-access IO and decode")
    _atlas(s)
    s.add_argument("--out", type=Path, required=True, help="New directory for process records and the summary")
    s.add_argument("--extra-root", type=Path, help="Second filesystem to stage level-0 pages on")
    s.add_argument("--processes", type=int, default=3)
    # v2 tracks register their own commands.
    import importlib
    for module in V2_COMMANDS:
        try:
            importlib.import_module(module).register(add)
        except ModuleNotFoundError as exc:
            if not module.startswith(exc.name or "\0"):
                raise
    return p


V2_COMMANDS = ("geoneural.codecs.commands", "geoneural.codecs.fixtures", "geoneural.physics.commands",
               "geoneural.recon.commands")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in DELEGATED:
        import importlib
        sys.argv = [f"geoneural {argv[0]}"] + argv[1:]
        importlib.import_module(DELEGATED[argv[0]]).main()
        return 0
    args = parser().parse_args(argv)
    if getattr(args, "sampling", "unset") is None:
        args.sampling = "with" if args.data_path == "device" else "without"
    try:
        return args.func(args) or 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"geoneural {args.command}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
