"""What random access to this atlas costs.

The codec comparison counts bytes on disk. It does not measure what happens when
something asks for a few samples, which is what a renderer does. Those are
different questions: a page is the smallest readable unit, so a one-sample query
reads a whole page and discards almost all of it.

The decision this informs is whether pages should be batched into one archive.
That trades file-open overhead against read amplification, so both are measured
on the same queries.

Filesystems are measured separately and never pooled. On a WSL host that is ext4
(the WSL root) and 9p (the Windows drive through WSL). Windows reading its own
NTFS natively is not measured here, because that requires running under Windows
rather than WSL. It is reported as unmeasured, not folded into the 9p number.

The OS page cache is not flushed, because that needs root. Every cache label
states what it means, and "cold" never means "warm".
"""
from __future__ import annotations
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from geoneural.codecs import eat1 as codec
from geoneural.common import read_json, utc, write_json
from geoneural.provenance import run_record

SCHEMA = "geoneural-io-v1"
# 1 sample, then 17x17, 65x65, 257x257 and roughly a megasample. These bracket a
# point probe, a small patch, exactly one page, a cut of several pages and a
# whole-region request.
QUERY_SAMPLES = (1, 289, 4225, 65536, 1_000_000)
ARCHIVE_MAGIC = b"EATPACK1"
STORED_BYTES_PER_SAMPLE = 4


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_archive(atlas_dir: Path, manifest: dict, target: Path) -> dict:
    """One file holding every page, plus the offsets needed to find them.

    The index is counted as deployed bytes. An archive that pretends its own
    directory is free would win the comparison by not being measured.
    """
    entries, blobs, offset = [], [], 0
    header_keys = sorted(manifest["pages"])
    for key in header_keys:
        entry = manifest["pages"][key]
        blob = (atlas_dir / entry["path"]).read_bytes()
        blobs.append(blob)
        entries.append({"key": key, "offset": offset, "length": len(blob)})
        offset += len(blob)
    index = json.dumps({"magic": "EATPACK1", "entries": entries}, separators=(",", ":")).encode()
    with target.open("wb") as handle:
        handle.write(ARCHIVE_MAGIC)
        handle.write(struct.pack("<Q", len(index)))
        handle.write(index)
        for blob in blobs:
            handle.write(blob)
    body_start = len(ARCHIVE_MAGIC) + 8 + len(index)
    return {"path": target, "bodyStart": body_start, "indexBytes": len(index),
            "entries": {e["key"]: (e["offset"], e["length"]) for e in entries},
            "totalBytes": target.stat().st_size}


def window_for(samples: int, side: int) -> tuple[int, int, int, int]:
    """A square window of at least `samples` nodes, clamped to the lattice."""
    want = min(int(math.ceil(math.sqrt(samples))), side)
    return 0, want - 1, 0, want - 1


def _page_span(low: int, high: int, intervals: int) -> range:
    """Minimal page range covering nodes `low..high` inclusive.

    Pages repeat their shared boundary node, so a node at an exact multiple of
    `intervals` is available from either neighbour. Taking `high // intervals`
    naively pulls in a page for a single already-covered column: a 65x65 query
    reads four pages instead of one, quadrupling its IO. A selector that gets
    this wrong still produces correct images, so only the IO shows the error.
    """
    last = (high - 1) // intervals if high > low and high % intervals == 0 else high // intervals
    return range(low // intervals, max(last, low // intervals) + 1)


def pages_for(window: tuple[int, int, int, int], intervals: int) -> list[tuple[int, int]]:
    r0, r1, c0, c1 = window
    return [(x, y)
            for y in _page_span(r0, r1, intervals)
            for x in _page_span(c0, c1, intervals)]


def pages_for_naive(window: tuple[int, int, int, int], intervals: int) -> list[tuple[int, int]]:
    """What an obvious implementation asks for; kept to measure the penalty."""
    r0, r1, c0, c1 = window
    return [(x, y)
            for y in range(r0 // intervals, r1 // intervals + 1)
            for x in range(c0 // intervals, c1 // intervals + 1)]


def _extract(values: np.ndarray, page_xy: tuple[int, int], window, intervals: int) -> int:
    """Count how many of the query's samples this page supplies."""
    r0, r1, c0, c1 = window
    x, y = page_xy
    rows = range(max(r0, y * intervals), min(r1, y * intervals + intervals) + 1)
    cols = range(max(c0, x * intervals), min(c1, x * intervals + intervals) + 1)
    if not rows or not cols:
        return 0
    # Touch the data so the measurement includes the access, not just the decode.
    patch = values[rows.start - y * intervals:rows.stop - y * intervals,
                   cols.start - x * intervals:cols.stop - x * intervals]
    return int(patch.size)


def measure_separate(atlas_dir: Path, manifest: dict, window, verify: bool) -> dict:
    intervals = manifest["page_intervals"]
    opens = read_bytes = raw_bytes = useful = 0
    t_io = t_hash = t_decode = 0.0
    for xy in pages_for(window, intervals):
        entry = manifest["pages"][f"0/{xy[0]}/{xy[1]}"]
        start = time.perf_counter()
        blob = (atlas_dir / entry["path"]).read_bytes()
        t_io += time.perf_counter() - start
        opens += 1
        read_bytes += len(blob)
        if verify:
            start = time.perf_counter()
            if _sha256(blob) != entry["sha256"]:
                raise ValueError(f"page {xy} failed its manifest hash")
            t_hash += time.perf_counter() - start
        start = time.perf_counter()
        values, _ = codec.decode(blob, entry["raw_bytes"])
        t_decode += time.perf_counter() - start
        raw_bytes += entry["raw_bytes"]
        useful += _extract(values, xy, window, intervals)
    return {"fileOpens": opens, "bytesRead": read_bytes, "bytesDecompressed": raw_bytes,
            "usefulSamples": useful, "ioMs": t_io * 1e3, "hashMs": t_hash * 1e3,
            "decodeMs": t_decode * 1e3}


def measure_packed(archive: dict, manifest: dict, window, verify: bool) -> dict:
    intervals = manifest["page_intervals"]
    read_bytes = raw_bytes = useful = 0
    t_io = t_hash = t_decode = 0.0
    start = time.perf_counter()
    handle = open(archive["path"], "rb")
    t_io += time.perf_counter() - start
    try:
        for xy in pages_for(window, intervals):
            key = f"0/{xy[0]}/{xy[1]}"
            entry = manifest["pages"][key]
            offset, length = archive["entries"][key]
            start = time.perf_counter()
            handle.seek(archive["bodyStart"] + offset)
            blob = handle.read(length)
            t_io += time.perf_counter() - start
            read_bytes += len(blob)
            if verify:
                start = time.perf_counter()
                if _sha256(blob) != entry["sha256"]:
                    raise ValueError(f"archived page {key} failed its manifest hash")
                t_hash += time.perf_counter() - start
            start = time.perf_counter()
            values, _ = codec.decode(blob, entry["raw_bytes"])
            t_decode += time.perf_counter() - start
            raw_bytes += entry["raw_bytes"]
            useful += _extract(values, xy, window, intervals)
    finally:
        handle.close()
    # One open for the whole query, however many pages it touches. That is what
    # the packed layout is for.
    return {"fileOpens": 1, "bytesRead": read_bytes, "bytesDecompressed": raw_bytes,
            "usefulSamples": useful, "ioMs": t_io * 1e3, "hashMs": t_hash * 1e3,
            "decodeMs": t_decode * 1e3}


def sweep(atlas_path: Path, archive_path: Path | None, verify: bool = True) -> list[dict]:
    """One process's worth of observations across every query size and layout."""
    atlas_path = Path(atlas_path)
    atlas_dir = atlas_path.parent
    manifest = read_json(atlas_path)
    side = manifest["sample_side"]
    archive = None
    if archive_path is not None:
        archive = build_archive(atlas_dir, manifest, Path(archive_path)) \
            if not Path(archive_path).exists() else _open_archive(Path(archive_path))
    rows = []
    for samples in QUERY_SAMPLES:
        window = window_for(samples, side)
        requested = (window[1] - window[0] + 1) * (window[3] - window[2] + 1)
        for layout, result in (("separate", measure_separate(atlas_dir, manifest, window, verify)),
                               ("packed", measure_packed(archive, manifest, window, verify) if archive else None)):
            if result is None:
                continue
            useful_bytes = result["usefulSamples"] * STORED_BYTES_PER_SAMPLE
            rows.append({
                "layout": layout, "requestedSamples": requested,
                "windowSide": window[1] - window[0] + 1,
                "pagesTouched": len(pages_for(window, manifest["page_intervals"])),
                "pagesTouchedNaive": len(pages_for_naive(window, manifest["page_intervals"])),
                "usefulBytes": useful_bytes,
                # Two separate ratios. Disk cost per useful sample can fall
                # below the stored width because pages are compressed; that is
                # not the same as reading efficiently. The second ratio is the
                # actual waste: how many samples had to be decoded for each one
                # the query used.
                "diskBytesPerUsefulSample": result["bytesRead"] / max(result["usefulSamples"], 1),
                "samplesDecodedPerUsefulSample":
                    (result["bytesDecompressed"] - codec.HEADER.size * result["fileOpens"])
                    / STORED_BYTES_PER_SAMPLE / max(result["usefulSamples"], 1),
                **result,
            })
    return rows


def _open_archive(path: Path) -> dict:
    with path.open("rb") as handle:
        if handle.read(len(ARCHIVE_MAGIC)) != ARCHIVE_MAGIC:
            raise ValueError("Not an EATPACK1 archive")
        length = struct.unpack("<Q", handle.read(8))[0]
        index = json.loads(handle.read(length))
    return {"path": path, "bodyStart": len(ARCHIVE_MAGIC) + 8 + length, "indexBytes": length,
            "entries": {e["key"]: (e["offset"], e["length"]) for e in index["entries"]},
            "totalBytes": path.stat().st_size}


def filesystem_of(path: Path) -> str:
    try:
        out = subprocess.run(["df", "-T", str(path)], capture_output=True, text=True, timeout=10).stdout
        return out.strip().splitlines()[-1].split()[1]
    except (OSError, IndexError, subprocess.SubprocessError):
        return "unknown"


def run(atlas_path: Path, out: Path, repeats: int = 3, extra_root: Path | None = None) -> Path:
    """Three fresh processes per filesystem, because a warm interpreter and a
    warm application cache are not the state a first page read happens in."""
    atlas_path = Path(atlas_path).resolve()
    locations = [{"name": "primary", "atlas": atlas_path, "filesystem": filesystem_of(atlas_path)}]
    if extra_root is not None:
        copied = _stage(atlas_path, Path(extra_root))
        locations.append({"name": "secondary", "atlas": copied, "filesystem": filesystem_of(copied)})

    observations = []
    for location in locations:
        archive = location["atlas"].parent / "pages.eatpack"
        for index in range(repeats):
            child = subprocess.run(
                [sys.executable, "-m", "geoneural.bench.bench_io", str(location["atlas"]), str(archive)],
                capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1]),
                env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])}, timeout=900)
            if child.returncode != 0:
                raise RuntimeError(f"IO sweep failed on {location['name']}: {child.stderr[-800:]}")
            for row in json.loads(child.stdout):
                observations.append({"location": location["name"], "filesystem": location["filesystem"],
                                     "process": index, "processState": "cold" if index == 0 else "repeat",
                                     **row})
        if archive.exists():
            archive.unlink()

    report = {
        "schema": SCHEMA,
        "atlas": str(atlas_path),
        "atlasContentId": read_json(atlas_path)["content_id"],
        "querySamples": list(QUERY_SAMPLES),
        "repeatsPerLocation": repeats,
        "locations": [{k: str(v) for k, v in loc.items()} for loc in locations],
        "observations": observations,
        "summary": _summarise(observations),
        "layoutDecision": _layout_decision(_summarise(observations)),
        "granularity": {
            "smallestReadableUnit": "one page of (pageIntervals+1)^2 samples",
            "note": "A point query decodes a whole page. That is a property of page-granular storage, "
                    "not of the codec, and no codec change removes it. Small-request cost and "
                    "large-batch throughput are therefore separate results.",
        },
        "cacheState": {
            "processCold": "each observation is a fresh interpreter, so no application or decoder state carries over",
            "osCacheFlushed": False,
            "osCacheNote": "flushing the page cache needs root and was not done. The first process on each "
                           "filesystem still reads through a warm OS cache from staging, so these are "
                           "warm-OS-cache numbers and must not be cited as first-read-from-disk latency.",
            "ntfsNative": "not measured; Windows reading its own NTFS requires running outside WSL",
        },
        "hostContention": "Other work on the host is not controlled or recorded here, so absolute "
                          "latencies are unqualified. Cross-filesystem and cross-layout comparisons are "
                          "reported as ratios for that reason.",
        "qualification": "Random-access IO and decode cost on one host. Not whole-renderer evidence, not a "
                         "frame-time claim, and not a cold-disk measurement. Latency figures are "
                         "host-specific.",
    }
    report["run"] = run_record("io-benchmark", {"atlas": str(atlas_path), "repeats": repeats},
                               {"atlas": atlas_path.parent,
                                **({"secondary": locations[1]["atlas"].parent} if len(locations) > 1 else {})})
    write_json(Path(out), report)
    return Path(out)


def _layout_decision(summary: list[dict]) -> dict:
    """Whether to batch pages into an archive. The answer depends on the
    filesystem, which is why filesystems are never pooled."""
    per_filesystem = {}
    for row in summary:
        per_filesystem.setdefault(row["filesystem"], {}).setdefault(row["requestedSamples"], {})[row["layout"]] = row
    out = {}
    for filesystem, sizes in per_filesystem.items():
        gains = []
        for samples in sorted(sizes):
            pair = sizes[samples]
            if "packed" not in pair or "separate" not in pair:
                continue
            separate, packed = pair["separate"]["ioMsMedian"], pair["packed"]["ioMsMedian"]
            gains.append({"requestedSamples": samples,
                          "separateFileOpens": pair["separate"]["fileOpens"],
                          "separateIoMsMedian": separate, "packedIoMsMedian": packed,
                          "ioSpeedup": round(separate / packed, 2) if packed > 0 else None})
        largest = gains[-1] if gains else None
        out[filesystem] = {
            "byQuerySize": gains,
            "largestQuerySpeedup": largest["ioSpeedup"] if largest else None,
            "verdict": _verdict(largest),
            "status": "provisional pilot from a shared host; not a settled performance recommendation",
        }
    return out


def _verdict(largest: dict | None) -> str:
    if largest is None or largest["ioSpeedup"] is None:
        return "not determined"
    # Provisional, and per filesystem profile. An offset index keeps per-page
    # addressing, cancellation and eviction, so those are not a cost of packing;
    # what one container costs is replacing a page without rewriting it.
    if largest["ioSpeedup"] >= 3.0:
        return ("provisional: per-file open dominates on this filesystem profile, so one archive plus "
                "offsets removes most of the IO cost at large cuts without changing a byte of payload")
    if largest["ioSpeedup"] >= 1.5:
        return ("provisional: packing gains something on this filesystem profile but not an order of "
                "magnitude, and a shared-host pilot cannot bound the interference it was measured under")
    return "provisional: packing buys no useful IO on this filesystem profile"


def _stage(atlas_path: Path, root: Path) -> Path:
    """Copy the runtime package (manifest and pages only) to another
    filesystem. The research reference arrays are deliberately left behind."""
    root.mkdir(parents=True, exist_ok=True)
    target = root / atlas_path.parent.name
    if target.exists():
        shutil.rmtree(target)
    (target / "pages").mkdir(parents=True)
    shutil.copy2(atlas_path, target / atlas_path.name)
    shutil.copytree(atlas_path.parent / "pages", target / "pages", dirs_exist_ok=True)
    return target / atlas_path.name


def _summarise(observations: list[dict]) -> list[dict]:
    """Median per condition. Percentiles across filesystems are never pooled,
    so the grouping key keeps them apart."""
    groups: dict[tuple, list[dict]] = {}
    for row in observations:
        groups.setdefault((row["filesystem"], row["layout"], row["requestedSamples"]), []).append(row)
    summary = []
    for (filesystem, layout, samples), rows in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        total = [r["ioMs"] + r["hashMs"] + r["decodeMs"] for r in rows]
        summary.append({
            "filesystem": filesystem, "layout": layout, "requestedSamples": samples,
            "observations": len(rows),
            "pagesTouched": rows[0]["pagesTouched"],
            "pagesTouchedNaive": rows[0]["pagesTouchedNaive"], "fileOpens": rows[0]["fileOpens"],
            "usefulBytes": rows[0]["usefulBytes"], "bytesRead": rows[0]["bytesRead"],
            "diskBytesPerUsefulSample": round(rows[0]["diskBytesPerUsefulSample"], 2),
            "samplesDecodedPerUsefulSample": round(rows[0]["samplesDecodedPerUsefulSample"], 1),
            "ioMsMedian": round(float(np.median([r["ioMs"] for r in rows])), 3),
            "hashMsMedian": round(float(np.median([r["hashMs"] for r in rows])), 3),
            "decodeMsMedian": round(float(np.median([r["decodeMs"] for r in rows])), 3),
            "totalMsMedian": round(float(np.median(total)), 3),
            "totalMsMin": round(min(total), 3), "totalMsMax": round(max(total), 3),
        })
    return summary


if __name__ == "__main__":
    archive = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    print(json.dumps(sweep(Path(sys.argv[1]), archive)))
