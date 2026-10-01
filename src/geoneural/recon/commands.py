"""Command-line entry points of the reconstruction track.

Each experiment command runs one or more folds and writes a part per fold under
`HOME/results/v2/recon/parts/<task>/`; the matching report command merges the parts into a v2 record at
`HOME/results/v2/recon/<name>.json`. Folds are independent processes, so they can be run in parallel.

Confirmation: pass `--test-regions` (new regions under the same data layout) and `--frozen-recipe` (written by
the development report as `<name>-frozen-recipe.json`); every development region then trains, nothing is
selected, and the test regions are only scored.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from geoneural.common import HOME, read_json, write_json

RESULTS = HOME / "results" / "v2" / "recon"
MODELS = HOME / "models" / "recon"


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _folds(a):
    from geoneural.recon import fields
    folds = fields.folds(a.regions, a.test_regions)
    if a.only:
        folds = [f for f in folds if set(f["test"]) & set(a.only)]
    return folds


def _common(s, regions=None):
    from geoneural.recon import fields
    s.add_argument("--regions", nargs="+", default=list(regions or fields.REGIONS),
                   help="Development regions (train and validate; each is tested once unless --test-regions)")
    s.add_argument("--test-regions", nargs="+", help="Confirmation regions: train on all --regions with a frozen "
                                                     "recipe and score only these")
    s.add_argument("--frozen-recipe", type=Path, help="Recipe JSON written by the development report")
    s.add_argument("--only", nargs="+", help="Run only the folds whose test region is listed")
    s.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    s.add_argument("--device", default="cuda")
    s.add_argument("--parts", type=Path, help="Directory for per-fold parts")
    s.add_argument("--gpu-gb", type=float, default=1.1, help="Allocator cap for this process on a shared GPU")
    s.add_argument("--lr-grid", nargs="+", type=float, help="Learning rates the selection rule chooses among")
    s.add_argument("--geology-seeds", type=int, help="Train the geology variant on only the first N seeds")


def _limit_gpu(a) -> None:
    """Cap this process's share of the shared GPU (allocator memory, the CUDA context comes on top)."""
    if str(a.device).startswith("cuda"):
        import torch
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        torch.cuda.set_per_process_memory_fraction(min(a.gpu_gb / total, 1.0))


def _frozen(a):
    if a.test_regions and not a.frozen_recipe:
        raise ValueError("a confirmation run needs --frozen-recipe; nothing may be selected on test regions")
    return read_json(a.frozen_recipe)["recipes"] if a.frozen_recipe else None


def _write_part(folder: Path, fold: dict, part: dict) -> Path:
    path = folder / ("-".join(fold["test"]) + ".json")
    write_json(path, part)
    return path


def _jsonable(value):
    import numpy as np
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and value != value:
        return None
    return value


def cmd_provider_kernel(a):
    from geoneural.recon import fields, operators
    from geoneural.evaluation import protocol
    import numpy as np
    pairs = [(r, fields.fine(r), np.load(HOME / "atlases" / r / "reference.npy", mmap_mode="r"))
             for r in a.regions]
    fit = operators.fit_provider_style(pairs)
    rec = protocol.record(
        "reconstruction", a.name, "exploratory", results=_jsonable(fit),
        recipe={"family": "Gaussian average with a registration offset, separable, edge-renormalised",
                "candidates": "sigma 0.40-0.70 coarse cells, offsets -0.40..0.20 cells per axis in 0.05 steps",
                "criterion": "pooled mean absolute difference to the provider 10 m grid on fine nodes "
                             "2000..6000 of each region, interior coarse nodes",
                "declaredInCode": operators.PROVIDER_STYLE},
        split={"regions": list(a.regions), "note": "operator calibration only; heights of these regions are "
                                                   "also targets in the coarse-to-fine tasks"},
        notes="The provider's 10 m DGM is not the trapezoidal average of its 1 m DGM. A Gaussian of about half "
              "a coarse cell displaced roughly 2.5 m north explains most of the systematic difference; what "
              "remains (byRegion.fittedMaeM) is not a linear shift-invariant coarsening.")
    path = protocol.write(rec, a.out or RESULTS / f"{a.name}.json")
    _log(f"wrote {path}")


def cmd_coarse(a):
    _limit_gpu(a)
    from geoneural.recon import coarse
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    frozen = _frozen(a)
    parts = a.parts or RESULTS / "parts" / "coarse"
    for fold in _folds(a):
        _log(f"fold {fold}")
        earlier = parts / ("-".join(fold["test"]) + ".json")
        reuse = read_json(earlier, max_bytes=256 * 1024 * 1024) if a.rescore else None
        part = coarse.run_fold(fold, a.seeds, a.device, MODELS / "coarse", frozen, _log, a.max_steps,
                               a.checkpoints, a.lr_grid or coarse.LR_GRID, a.geology_seeds, reuse)
        _log(f"wrote {_write_part(parts, fold, _jsonable(part))}")


def cmd_blocks(a):
    _limit_gpu(a)
    from geoneural.recon import blocks
    frozen = _frozen(a)
    parts = a.parts or RESULTS / "parts" / "blocks"
    for fold in _folds(a):
        _log(f"fold {fold}")
        part = blocks.run_fold(fold, a.seeds, a.device, MODELS / "blocks", frozen, _log, a.max_steps, a.checkpoints,
                               a.lr_grid or blocks.LR_GRID, a.geology_seeds)
        _log(f"wrote {_write_part(parts, fold, _jsonable(part))}")


def cmd_sparse(a):
    _limit_gpu(a)
    from geoneural.recon import sparse
    frozen = _frozen(a)
    parts = a.parts or RESULTS / "parts" / "sparse"
    for fold in _folds(a):
        _log(f"fold {fold}")
        part = sparse.run_fold(fold, a.seeds, a.device, MODELS / "sparse", frozen, _log, a.max_steps, a.checkpoints,
                               a.lr_grid or sparse.LR_GRID, a.workers)
        _log(f"wrote {_write_part(parts, fold, _jsonable(part))}")


def cmd_fine(a):
    _limit_gpu(a)
    from geoneural.recon import fine
    frozen = _frozen(a)
    parts = a.parts or RESULTS / "parts" / "fine"
    folds = fine.folds(a.regions, a.test_regions)
    if a.only:
        folds = [f for f in folds if set(f["test"]) & set(a.only)]
    for fold in folds:
        _log(f"fold {fold}")
        part = fine.run_fold(fold, a.seeds, a.device, MODELS / "fine", frozen, _log, a.max_steps, a.checkpoints,
                             a.lr_grid or fine.LR_GRID)
        _log(f"wrote {_write_part(parts, fold, _jsonable(part))}")


def cmd_ambiguity(a):
    from geoneural.evaluation import protocol
    from geoneural.recon import ambiguity
    parts = a.parts or RESULTS / "parts" / "coarse"
    results = _jsonable(ambiguity.run(a.test_regions, a.device, MODELS / "coarse", parts))
    rec = protocol.record(
        "reconstruction", a.name, "exploratory", results=results,
        recipe={"task": "ambiguity at 40 m -> 10 m under the declared trapezoid", "crop": ambiguity.CROP,
                "gapThresholdM": ambiguity.GAP_M, "constructions": ["detailSwap", "channelShift"],
                "models": "the coarse-task models (no geology) of the fold that held each region out",
                "intervals": "central 90 % of the single model's Laplace, the seed mixture, and regression-kriging"},
        split={"testRegions": list(a.test_regions)},
        notes=ambiguity.__doc__.strip().split("\n\n")[0])
    _log(f"wrote {protocol.write(rec, a.out or RESULTS / f'{a.name}.json')}")


def cmd_report(a):
    from geoneural.evaluation import protocol
    module = __import__(f"geoneural.recon.{a.task}", fromlist=["summarise"])
    folder = a.parts or RESULTS / "parts" / a.task
    paths = sorted(folder.glob("*.json"))
    if not paths:
        raise ValueError(f"no parts under {folder}")
    parts = [read_json(p, max_bytes=256 * 1024 * 1024) for p in paths]
    confirmation = all(not p["fold"]["validation"] for p in parts)
    if a.evidence == "confirmed" and not confirmation:
        raise ValueError("development folds cannot be reported as confirmed")
    results, failures = module.summarise(parts)
    gates = module.gates(results) if hasattr(module, "gates") else {}
    if hasattr(module, "tiling_check"):
        results["tilingCheck"] = module.tiling_check(parts, MODELS / a.task)
        worst = max(v["maxDifferenceM"] for v in results["tilingCheck"].values())
        gates["tiledEqualsWholeField"] = protocol.gate(
            "pass" if worst <= 1e-4 else "fail", f"largest tiled minus whole-window difference {worst:.2e} m "
            "(float32 on the host)", toleranceM=1e-4)
    recipe, split = module.describe(parts)
    rec = protocol.record("reconstruction", a.name, a.evidence, results=_jsonable(results), recipe=recipe,
                          split=split, seeds=sorted({s for p in parts for s in p.get("seeds", [])}),
                          gates=gates, failures=failures,
                          timing={"qualified": False, "foldSeconds": {"-".join(p["fold"]["test"]): p["seconds"]
                                                                      for p in parts}},
                          notes=getattr(module, "NOTES", ""))
    path = protocol.write(rec, a.out or RESULTS / f"{a.name}.json")
    if hasattr(module, "frozen_recipe") and not confirmation:
        write_json(path.with_name(f"{a.name}-frozen-recipe.json"), module.frozen_recipe(parts))
    _log(f"wrote {path}")


def register(add):
    s = add("recon-provider-kernel", cmd_provider_kernel,
            "Fit the provider-style observation operator from 1 m and provider 10 m data")
    s.add_argument("--regions", nargs="+", default=["essen-ruhr", "muensterland-plain", "rothaar-sauerland"])
    s.add_argument("--name", default="provider-kernel")
    s.add_argument("--out", type=Path)

    s = add("recon-coarse", cmd_coarse, "Coarse-to-fine 40 m to 10 m reconstruction, leave one region out")
    _common(s)
    s.add_argument("--max-steps", type=int, default=5000)
    s.add_argument("--checkpoints", nargs="+", type=int, default=[500, 1000, 1500, 2000, 3000, 4000, 5000])
    s.add_argument("--rescore", action="store_true", help="Score again with the saved models of an earlier part")

    s = add("recon-blocks", cmd_blocks, "Missing blocks of 8, 32 and 128 cells, leave one region out")
    _common(s)
    s.add_argument("--max-steps", type=int, default=3000)
    s.add_argument("--checkpoints", nargs="+", type=int, default=[500, 1000, 1500, 2000, 3000])

    s = add("recon-sparse", cmd_sparse, "Sparse noisy samples (1 and 5 percent of nodes), leave one region out")
    _common(s)
    s.add_argument("--max-steps", type=int, default=3000)
    s.add_argument("--checkpoints", nargs="+", type=int, default=[500, 1000, 1500, 2000, 3000])
    s.add_argument("--workers", type=int, default=4, help="Processes for the thin-plate fits")

    from geoneural.recon import fields
    s = add("recon-fine", cmd_fine, "Coarse-to-fine 10 m to 1 m on the regions with 1 m data, leave one region out")
    _common(s, fields.FINE_REGIONS)
    s.add_argument("--max-steps", type=int, default=3000)
    s.add_argument("--checkpoints", nargs="+", type=int, default=[500, 1000, 1500, 2000, 3000])

    s = add("recon-ambiguity", cmd_ambiguity, "Two fine terrains with one coarse observation: does the model know?")
    s.add_argument("--test-regions", nargs="+", default=["essen-ruhr", "rothaar-sauerland"])
    s.add_argument("--device", default="cuda")
    s.add_argument("--parts", type=Path, help="Coarse-task parts directory")
    s.add_argument("--name", default="ambiguity")
    s.add_argument("--out", type=Path)

    s = add("recon-report", cmd_report, "Merge reconstruction fold parts into one v2 record")
    s.add_argument("--task", required=True, choices=["coarse", "fine", "blocks", "sparse"])
    s.add_argument("--name", required=True)
    s.add_argument("--evidence", default="selected", choices=["exploratory", "selected", "confirmed"],
                   help="confirmed only for a frozen-recipe run on confirmation regions")
    s.add_argument("--parts", type=Path)
    s.add_argument("--out", type=Path)


if __name__ == "__main__":  # pragma: no cover
    sys.exit("use: geoneural recon-...")
