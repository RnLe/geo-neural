"""Is a neural decoder fast at the sizes a renderer asks for?

A network that answers a million points quickly can still be slow for a level-of-detail
selector that asks for 289. Large-batch throughput does not justify slow small requests,
so latency is measured per query size and never averaged with throughput.

The comparison with conventional page decode is unfair in both directions, and both are
stated rather than corrected away. The page path must read and decompress whole pages, so a
one-sample request costs a 65 x 65 page. The network evaluates exactly the points asked for,
which is its real structural advantage. On the other hand the network returns only point
values, while a page carries a hash, a declared height range and an error figure. Equal
latency is not an equal product.

Measurement follows the paired design in `paired.py`: both arms run back to back on the same
query with alternating order, and the statistic is the within-pair ratio. Absolute times on a
shared host are not qualified.
"""
from __future__ import annotations
import statistics
import time
from pathlib import Path

import numpy as np

from geoneural.bench import bench_io
from geoneural.common import read_json

SCHEMA = "geoneural-query-cost-v1"
QUERY_SAMPLES = (1, 289, 4225, 65536, 1_000_000)
ARM_TARGET_MS = 40.0
MAX_INNER = 20000


def _calibrate(run_once, target_ms: float = ARM_TARGET_MS) -> int:
    start = time.perf_counter()
    run_once()
    once = (time.perf_counter() - start) * 1e3
    return max(1, min(MAX_INNER, int(target_ms / once))) if once > 0 else MAX_INNER


def neural_query(model, features_fn, side: int, intervals: int, shared: bool,
                 count: int, device: str, torch, synchronise: bool = True):
    """One decoder call for `count` samples, including the transfer it needs.

    GPU synchronisation is inside the timed region on purpose. A renderer that
    needs the heights this frame must wait for them, so an unsynchronised launch
    time would be a number nobody can use.
    """
    indexes = np.arange(count, dtype=np.int64)

    def run():
        coords, tiles = features_fn(indexes, side, intervals, shared)
        with torch.inference_mode():
            out = model(torch.from_numpy(coords).to(device), torch.from_numpy(tiles).to(device))
            result = out.squeeze(-1).cpu().numpy()
        if synchronise and str(device).startswith("cuda"):
            torch.cuda.synchronize()
        return result

    return run


def conventional_query(atlas_dir: Path, manifest: dict, count: int, verify: bool = True):
    """The same request answered from pages, whole pages at a time.

    `verify` controls the per-page SHA-256 check. Turning it off makes the
    conventional arm faster than a runtime that actually validates its input,
    so it biases the comparison *towards* conventional; both settings are
    measured rather than one being chosen.
    """
    side = manifest["sample_side"]
    window = bench_io.window_for(count, side)

    def run():
        return bench_io.measure_separate(atlas_dir, manifest, window, verify=verify)

    return run


def _paired(neural_run, conventional_run, trials: int, inner_neural: int,
            inner_conventional: int) -> dict:
    pairs = []
    for trial in range(trials):
        neural_first = trial % 2 == 0

        def arm(run, inner):
            start = time.perf_counter()
            for _ in range(inner):
                run()
            return (time.perf_counter() - start) * 1e3 / inner

        if neural_first:
            neural_ms = arm(neural_run, inner_neural)
            conventional_ms = arm(conventional_run, inner_conventional)
        else:
            conventional_ms = arm(conventional_run, inner_conventional)
            neural_ms = arm(neural_run, inner_neural)
        pairs.append({"firstArm": "neural" if neural_first else "conventional",
                      "neuralMs": neural_ms, "conventionalMs": conventional_ms,
                      "neuralOverConventional": neural_ms / conventional_ms})
    ratios = [p["neuralOverConventional"] for p in pairs]
    return {
        "trials": trials,
        "neuralMsMedian": statistics.median(p["neuralMs"] for p in pairs),
        "conventionalMsMedian": statistics.median(p["conventionalMs"] for p in pairs),
        "ratioMedian": statistics.median(ratios),
        "ratioMin": min(ratios), "ratioMax": max(ratios),
        "pairs": pairs,
    }


def run(atlas_path: Path, model_path: Path, device: str = "cuda", trials: int = 7,
        verify_pages: bool = True) -> dict:
    import torch
    from safetensors.torch import load_file
    from geoneural.neural.learning import features
    from geoneural.neural.models import make_model

    atlas_path = Path(atlas_path)
    manifest = read_json(atlas_path)
    meta = read_json(Path(model_path))
    weights = Path(model_path).parent / "weights.safetensors"
    model = make_model(meta["model"]).to(device)
    model.load_state_dict(load_file(str(weights)))
    model.eval()
    shared = meta["model"]["kind"] == "shared"
    side, intervals = manifest["sample_side"], manifest["page_intervals"]

    rows = []
    for count in QUERY_SAMPLES:
        neural = neural_query(model, features, side, intervals, shared, count, device, torch)
        conventional = conventional_query(atlas_path.parent, manifest, count, verify_pages)
        neural(); conventional()  # warm both before calibrating either
        result = _paired(neural, conventional, trials,
                         _calibrate(neural), _calibrate(conventional))
        pages = len(bench_io.pages_for(bench_io.window_for(count, side), intervals))
        rows.append({
            "requestedSamples": count,
            "conventionalPagesRead": pages,
            "conventionalSamplesDecoded": pages * (intervals + 1) ** 2,
            "neuralSamplesEvaluated": count,
            "neuralUsPerSample": result["neuralMsMedian"] * 1e3 / count,
            **{k: v for k, v in result.items() if k != "pairs"},
        })
    return {
        "schema": SCHEMA,
        "atlas": str(atlas_path), "atlasContentId": manifest["content_id"],
        "model": str(model_path), "device": device, "pagesVerified": verify_pages,
        "modelKind": meta["model"]["kind"], "weightsBytes": meta["weights_bytes"],
        "byQuerySize": rows,
        "fairness": {
            "conventionalAdvantage": "The page path returns a payload carrying a hash, a declared height "
                                     "range and an error figure. The network returns point values and "
                                     "nothing else, so equal latency is not equal product. Whether the "
                                     "hash is actually checked is the `pagesVerified` flag: with it off "
                                     "the conventional arm is faster than any runtime that validates its "
                                     "input, which biases the comparison towards conventional.",
            "neuralAdvantage": "The page path must read and decompress whole pages for any query, so a "
                               "one-sample request costs a 65x65 page. The network evaluates exactly the "
                               "points asked for.",
            "synchronisation": "CUDA synchronisation is inside the timed region, because a renderer that "
                               "needs the heights this frame must wait for them.",
        },
        "qualification": "Paired ratios on a shared host, following paired.py. Absolute milliseconds are "
                         "not qualified and no frame-rate conclusion follows. This measures decode cost "
                         "for point queries only, not whole-renderer behaviour.",
    }


CONFIG_SCHEMA = "geoneural-decode-latency-v1"


def run_configs(chosen: dict, problem, trials: int = 7, steps: int = 5000,
                batch: int = 8192, seed: int = 1729, verify_pages: bool = True,
                cache_state: str = "warm", qualified: bool = False, store_precision: str = "float16",
                engine_mode: str = "eager", data_path: str = "device",
                sampling: str = "with") -> dict:
    """Decode cost of each chosen configuration at renderer query sizes.

    `run` reads one saved checkpoint. This trains every chosen configuration and
    measures the model that would ship, at the storage precision it would ship at,
    so decode latency is compared on the same candidates as bytes and error.
    """
    import torch

    from geoneural.neural import search as arch

    from geoneural.bench import timing

    from geoneural.neural import training
    from geoneural.neural.learning import features

    if cache_state != "warm":
        raise ValueError("decode latency measures warm queries only; both arms are warmed before timing")
    guard = None
    if qualified:
        guard = timing.require_quiet(1)
    busy_before = timing.host_busy_fraction()

    side, intervals = problem.side, problem.intervals
    rows = []
    for name in sorted(chosen):
        entry = chosen[name]
        config = dict(entry["config"])
        recipe = training.with_overrides(
            arch.Recipe(**{**entry.get("recipe", {}), "steps": steps, "batch": batch,
                           "device": problem.device, "seed": seed}),
            data_path=data_path, sampling=sampling)
        result = arch.measure(config, recipe, problem, evaluate_on=("all",), train_on="all",
                              store_precision=store_precision, limit=None,
                              engine_mode=engine_mode)
        model = result["model"]
        model.eval()
        if problem.device.startswith("cuda"):
            # Count the memory decoding needs, not what training left behind.
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        shared = config["kind"] == "shared" or (
            isinstance(config.get("inner"), dict)
            and config["inner"].get("kind") == "shared")

        sizes = []
        for count in QUERY_SAMPLES:
            neural = neural_query(model, features, side, intervals, shared, count,
                                  problem.device, torch)
            conventional = conventional_query(
                problem.atlas_path.parent, problem.manifest, count, verify_pages)
            neural(); conventional()
            paired = _paired(neural, conventional, trials,
                             _calibrate(neural), _calibrate(conventional))
            pages = len(bench_io.pages_for(bench_io.window_for(count, side), intervals))
            sizes.append({
                "requestedSamples": count,
                "conventionalPagesRead": pages,
                "conventionalSamplesDecoded": pages * (intervals + 1) ** 2,
                "readAmplification": pages * (intervals + 1) ** 2 / max(count, 1),
                "neuralUsPerSample": paired["neuralMsMedian"] * 1e3 / count,
                **{k: v for k, v in paired.items() if k != "pairs"}})
        peak = (int(torch.cuda.max_memory_allocated()) if problem.device.startswith("cuda")
                else None)
        rows.append({"name": name, "config": arch.describe(config),
                     "deployedBytes": int(result["deployedBytes"]),
                     "maeM": result["metrics"]["all"]["mae_m"],
                     "maxM": result["metrics"]["all"]["max_m"],
                     "peakDeviceBytes": peak,
                     "byQuerySize": sizes})
        del result, model
        if problem.device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    from geoneural.provenance import gpu as gpu_record
    busy_after = timing.host_busy_fraction()
    return {
        "schema": CONFIG_SCHEMA, "rows": rows,
        "timing": timing.block("qualified" if qualified and busy_after <= 0.15 else "throughput",
                               cache_state,
                               workers=1, engine={"engineMode": engine_mode,
                                                  "dataPath": data_path},
                               gpu=gpu_record(),
                               busy_before=busy_before,
                               busy_after=busy_after),
        "guard": guard,
        "fairness": {
            "conventionalAdvantage": "The page path returns a payload carrying a hash, a declared "
                                     "height range and an error figure; the network returns point "
                                     "values and nothing else, so equal latency is not equal "
                                     "product.",
            "neuralAdvantage": "The page path reads and decompresses whole pages for any query, so "
                               "a one-sample request costs a 65x65 page. The network evaluates "
                               "exactly the points asked for, which is the whole structural case "
                               "for a coordinate representation and is what readAmplification "
                               "quantifies.",
        },
        "qualification": "Decode cost for point queries. Not mesh construction, not upload, not "
                         "whole-frame behaviour, and not the cold-start cost of loading "
                         "weights unless cacheState says so.",
    }
