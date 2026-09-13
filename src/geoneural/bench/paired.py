"""Measure a layout ratio on a busy host.

On a shared machine, waiting for the host to go quiet is not reliable: other
work can start at any time during a measurement. Whether one storage layout is
faster than another is a comparison, and a comparison does not need a quiet
host. It needs both arms to experience the same interference. Measuring arm A
for a minute and then arm B for a minute fails that: the load changes in
between, and the difference between arms is confounded with the difference
between minutes.

This module measures them adjacently instead. Each pair runs both layouts back to
back on the same query, so whatever the host was doing, it was doing it to both.
The statistic is the distribution of within-pair ratios, not the ratio of two
separately pooled timings.

Three further controls, because adjacency alone is not enough:

Order alternates every trial. Running A then B always would hand B a page cache
that A just warmed. Alternating cancels that in the median, and the per-pair
record keeps `firstArm` so the effect can be checked rather than assumed.

Contention is recorded per pair from /proc/stat, so each ratio carries the load
it was measured under. If the ratio were an artifact of interference it would
move with that load, and the reported rank correlation would show it. A large
effect does not bound its own interference, but a ratio shown to be independent
of measured interference does.

CPU time is recorded beside wall time. For IO-bound work the gap between them
is the waiting, which is the quantity of interest; a ratio driven by CPU
starvation rather than IO would show up as CPU time moving with it.

This does not make the numbers a qualified benchmark. It makes the comparison
defensible without exclusive use of the machine.
"""
from __future__ import annotations
import json
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from geoneural.bench import bench_io
from geoneural.common import read_json, write_json
from geoneural.provenance import run_record

SCHEMA = "geoneural-paired-io-v1"
DEFAULT_TRIALS = 15


def host_busy() -> tuple[int, int]:
    """Busy and idle jiffies across all CPUs."""
    with open("/proc/stat", "rb") as handle:
        parts = handle.readline().split()
    values = [int(v) for v in parts[1:]]
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return sum(values) - idle, idle


def _busy_fraction(before: tuple[int, int], after: tuple[int, int]) -> float:
    busy = after[0] - before[0]
    idle = after[1] - before[1]
    total = busy + idle
    return busy / total if total else 0.0


# /proc/stat counts in 10 ms jiffies, and a small query finishes in microseconds.
# Timing such a query once measures the clock, and sampling the host across it
# measures nothing at all. Each arm therefore repeats the query until it spans a
# long enough interval for both to mean something.
ARM_TARGET_MS = 40.0
MAX_INNER = 20000


def calibrate(run_once, target_ms: float = ARM_TARGET_MS) -> int:
    """How many repetitions make one arm last `target_ms`."""
    start = time.perf_counter()
    run_once()
    once_ms = (time.perf_counter() - start) * 1e3
    if once_ms <= 0:
        return MAX_INNER
    return max(1, min(MAX_INNER, int(target_ms / once_ms)))


def paired_trial(atlas_dir: Path, manifest: dict, archive: dict, window,
                 verify: bool, separate_first: bool, inner: int) -> dict:
    """One query measured both ways, back to back, with the load it saw."""
    before = host_busy()

    def arm(kind: str) -> dict:
        wall = time.perf_counter()
        cpu = time.process_time()
        for _ in range(inner):
            result = (bench_io.measure_separate(atlas_dir, manifest, window, verify) if kind == "separate"
                      else bench_io.measure_packed(archive, manifest, window, verify))
        return {**result, "innerRepeats": inner,
                "wallMs": (time.perf_counter() - wall) * 1e3 / inner,
                "cpuMs": (time.process_time() - cpu) * 1e3 / inner}

    order = ["separate", "packed"] if separate_first else ["packed", "separate"]
    started = time.perf_counter()
    arms = {kind: arm(kind) for kind in order}
    elapsed_ms = (time.perf_counter() - started) * 1e3
    after = host_busy()
    return {
        "firstArm": order[0], "pairElapsedMs": elapsed_ms,
        # Below a few jiffies the load figure is quantisation noise, so say so
        # rather than correlating against it.
        "loadResolved": elapsed_ms >= 50.0,
        "hostBusyFraction": _busy_fraction(before, after),
        "separate": arms["separate"], "packed": arms["packed"],
        "wallRatio": arms["separate"]["wallMs"] / arms["packed"]["wallMs"],
        "cpuRatio": (arms["separate"]["cpuMs"] / arms["packed"]["cpuMs"]
                     if arms["packed"]["cpuMs"] > 0 else None),
    }


def _rank(values: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged, as Spearman is actually defined.

    Double `argsort` is the usual shortcut and is wrong here: it breaks ties by
    position, so a constant input comes back as 0, 1, 2, ... with full variance.
    The load fraction can be constant across pairs, and the shortcut would then
    report a perfect contention correlation from a perfectly steady host.
    """
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(ordered):
        stop = start
        while stop + 1 < len(ordered) and ordered[stop + 1] == ordered[start]:
            stop += 1
        ranks[order[start:stop + 1]] = (start + stop) / 2.0
        start = stop + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    """Rank correlation without scipy, which is not a dependency here."""
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    rank_x = _rank(np.asarray(xs, dtype=float))
    rank_y = _rank(np.asarray(ys, dtype=float))
    # No variance means no relationship can be supported, not a perfect one.
    if rank_x.std() == 0 or rank_y.std() == 0:
        return None
    return float(np.corrcoef(rank_x, rank_y)[0, 1])


def sweep(atlas_path: Path, archive_path: Path, trials: int = DEFAULT_TRIALS,
          verify: bool = True, seed: int = 20260912) -> list[dict]:
    atlas_path = Path(atlas_path)
    atlas_dir = atlas_path.parent
    manifest = read_json(atlas_path)
    archive = (bench_io.build_archive(atlas_dir, manifest, Path(archive_path))
               if not Path(archive_path).exists() else bench_io._open_archive(Path(archive_path)))
    rng = random.Random(seed)
    rows = []
    for samples in bench_io.QUERY_SAMPLES:
        window = bench_io.window_for(samples, manifest["sample_side"])
        inner = calibrate(lambda: bench_io.measure_separate(atlas_dir, manifest, window, verify))
        pairs = []
        for trial in range(trials):
            # Alternate deterministically, then jitter the last one so a periodic
            # background job cannot stay aligned with one arm.
            separate_first = (trial % 2 == 0) if trial < trials - 1 else rng.random() < 0.5
            pairs.append(paired_trial(atlas_dir, manifest, archive, window, verify,
                                      separate_first, inner))
        wall = [p["wallRatio"] for p in pairs]
        resolved = [p for p in pairs if p["loadResolved"]]
        load = [p["hostBusyFraction"] for p in resolved]
        rows.append({
            "requestedSamples": samples,
            "pagesTouched": pairs[0]["separate"]["fileOpens"],
            "trials": trials, "innerRepeatsPerArm": inner,
            "pairsWithResolvedLoad": len(resolved),
            "wallRatioMedian": statistics.median(wall),
            "wallRatioMin": min(wall), "wallRatioMax": max(wall),
            "wallRatioIqr": (statistics.quantiles(wall, n=4)[2] - statistics.quantiles(wall, n=4)[0]
                             if len(wall) >= 4 else None),
            "cpuRatioMedian": statistics.median([p["cpuRatio"] for p in pairs if p["cpuRatio"]]),
            "hostBusyFractionMedian": statistics.median(load) if load else None,
            "hostBusyFractionMin": min(load) if load else None,
            "hostBusyFractionMax": max(load) if load else None,
            # If interference drove the ratio, it would track the load.
            "ratioVersusLoadSpearman": _spearman(load, [p["wallRatio"] for p in resolved]),
            "loadNote": None if load else "pairs too short for /proc/stat resolution; load not correlated",
            "medianRatioFirstArmSeparate": statistics.median(
                [p["wallRatio"] for p in pairs if p["firstArm"] == "separate"] or [float("nan")]),
            "medianRatioFirstArmPacked": statistics.median(
                [p["wallRatio"] for p in pairs if p["firstArm"] == "packed"] or [float("nan")]),
            "pairs": pairs,
        })
    return rows


def run(atlas_path: Path, out: Path, trials: int = DEFAULT_TRIALS,
        extra_root: Path | None = None) -> Path:
    atlas_path = Path(atlas_path).resolve()
    locations = [{"name": "primary", "atlas": atlas_path,
                  "filesystem": bench_io.filesystem_of(atlas_path)}]
    if extra_root is not None:
        staged = bench_io._stage(atlas_path, Path(extra_root))
        locations.append({"name": "secondary", "atlas": staged,
                          "filesystem": bench_io.filesystem_of(staged)})

    results = []
    for location in locations:
        archive = location["atlas"].parent / "pages.eatpack"
        for row in sweep(location["atlas"], archive, trials):
            results.append({"filesystem": location["filesystem"],
                            "location": location["name"], **row})
        if archive.exists():
            archive.unlink()

    report = {
        "schema": SCHEMA,
        "atlas": str(atlas_path),
        "atlasContentId": read_json(atlas_path)["content_id"],
        "design": {
            "why": "A shared host is rarely quiet, and other work can start at any time during a "
                   "measurement. A layout comparison does not need a quiet host; it needs both arms "
                   "to see the same interference.",
            "pairing": "Both layouts run back to back on the same query, and the statistic is the "
                       "distribution of within-pair ratios, not the ratio of separately pooled timings.",
            "orderControl": "Arm order alternates every trial so neither arm systematically inherits a "
                            "page cache the other warmed; per-arm medians are reported so the residual "
                            "effect is visible rather than assumed away.",
            "loadControl": "Host busy fraction is sampled per pair from /proc/stat. A ratio produced by "
                           "interference would move with load; the reported Spearman correlation between "
                           "load and ratio is the check.",
            "cpuControl": "CPU time is recorded beside wall time, so a ratio driven by CPU starvation "
                          "rather than by IO would be visible.",
        },
        "results": [{k: v for k, v in row.items() if k != "pairs"} for row in results],
        "pairs": {f"{row['filesystem']}/{row['requestedSamples']}": row["pairs"] for row in results},
        "qualification": "Not a qualified benchmark. Absolute milliseconds here should not be cited. "
                         "What this design supports is the comparison between layouts, measured under "
                         "interference rather than in spite of it. Whole-renderer cost is a separate "
                         "question and is not measured here.",
        "cacheState": "Warm. Each arm repeats its query many times to reach a measurable interval, so "
                      "after the first iteration every page is in the OS cache. These are steady-state "
                      "repeated-access ratios, not first-read-from-disk. On ext4 that leaves syscall "
                      "overhead and decompression as the work being compared; a cold-disk first read is "
                      "a different question and is not answered here.",
        "supersedes": "The sequential IO benchmark (bench_io) measures one layout after the other, so "
                      "the second arm inherits a page cache the first has just warmed, which inflates "
                      "the packed arm. Its layout ratios should not be cited; use these paired "
                      "ratios instead.",
    }
    report["run"] = run_record("paired-io-benchmark",
                               {"atlas": str(atlas_path), "trials": trials},
                               {"atlas": atlas_path.parent})
    write_json(Path(out), report)
    return Path(out)


if __name__ == "__main__":
    print(json.dumps(sweep(Path(sys.argv[1]), Path(sys.argv[2]))))
