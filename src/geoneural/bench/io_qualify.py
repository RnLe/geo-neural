"""Qualified IO cost of a page request, cold and warm.

Cold is produced per file with posix_fadvise(DONTNEED), which needs no root, and is then
verified with mincore. A trial whose files are still resident is labelled cold-unverified
and never summarised as cold. Both page codecs are timed on identical heights: the EAT1
pages and a lossless q32-delta-zstd transcode of the same pages. Each process records the
host's busy fraction before and after its sweep, and the run is qualified only if every
process saw an idle host.

Scope: level-0 pages, corner-anchored windows, one thread, the manifest or archive index
already in memory. Directory and inode caches and the host's own caches (the WSL2 disk image
on NTFS, the Windows cache behind 9p) are outside this kernel's page cache and are not
controlled: "cold" means "not resident in this kernel's page cache" and nothing more.
"""
from __future__ import annotations

import ctypes
import hashlib
import mmap
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

from geoneural.codecs import eat1 as codec
from geoneural.bench.bench_io import QUERY_SAMPLES, _extract, _open_archive, build_archive, pages_for, window_for
from geoneural.codecs.codecs import registry
from geoneural.common import read_json, write_json
from geoneural.provenance import run_record
from geoneural.bench.timing import host_busy_fraction

SCHEMA = "geoneural-io-qualified-v1"
CODECS = ("eat1", "q32dz")
LAYOUTS = ("separate", "packed")
BUSY_CEILING = 0.12  # host busy fraction above which a process is not qualified

_libc = ctypes.CDLL(None, use_errno=True)
_libc.mmap.restype = ctypes.c_void_p
_libc.mmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                       ctypes.c_int, ctypes.c_long)
_libc.munmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
_libc.mincore.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte))
_MAP_FAILED = ctypes.c_void_p(-1).value


def resident_pages(path: Path) -> tuple[int, int]:
    """(resident, total) memory pages of a file in this kernel's page cache.

    mincore reports residency without faulting anything in, so asking does not
    change the answer.
    """
    size = path.stat().st_size
    if size == 0:
        return 0, 0
    fd = os.open(path, os.O_RDONLY)
    try:
        address = _libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
        if address in (None, _MAP_FAILED):
            raise OSError(ctypes.get_errno(), f"mmap failed for {path}")
        try:
            count = (size + mmap.PAGESIZE - 1) // mmap.PAGESIZE
            vector = (ctypes.c_ubyte * count)()
            if _libc.mincore(address, size, vector) != 0:
                raise OSError(ctypes.get_errno(), f"mincore failed for {path}")
            return sum(v & 1 for v in vector), count
        finally:
            _libc.munmap(address, size)
    finally:
        os.close(fd)


def evict(paths) -> bool:
    """Drop each file from the page cache, then check. True only if verified;
    a filesystem that cannot answer mincore is unverified, never assumed cold."""
    for path in paths:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    try:
        return all(resident_pages(Path(p))[0] == 0 for p in paths)
    except OSError:
        return False


_Q32DZ = None


def _q32dz():
    # Built once: rebuilding the registry per page would bill this codec for it.
    global _Q32DZ
    if _Q32DZ is None:
        _Q32DZ = registry()["q32-delta-zstd"]
    return _Q32DZ


def stage(atlas_path: Path, root: Path) -> dict:
    """Level-0 pages of both codecs, separate and packed, synced to disk.

    The q32-delta-zstd pages re-encode the decoded EAT1 heights at the same
    quantum, so both codecs serve identical terrain; the transcode is checked,
    not assumed. Files are fsynced because the kernel will not evict dirty pages.
    """
    manifest = read_json(atlas_path)
    source = atlas_path.parent
    quantum = manifest["quantum_m"]
    target = root / manifest["content_id"][:16]
    profiles = {}
    for name in CODECS:
        (target / name / "pages").mkdir(parents=True, exist_ok=True)
        profiles[name] = {"pages": {}}
    worst = 0.0
    for key, entry in sorted(manifest["pages"].items()):
        if not key.startswith("0/"):
            continue
        blob = (source / entry["path"]).read_bytes()
        values, _ = codec.decode(blob, entry["raw_bytes"])
        q32 = _q32dz().encode(values, quantum / 2)
        worst = max(worst, float(abs(_q32dz().decode(q32).astype("f8") - values).max()))
        stem = key[2:].replace("/", "_")
        for name, data, suffix in (("eat1", blob, ".eat.gz"), ("q32dz", q32, ".q32dz")):
            path = target / name / "pages" / (stem + suffix)
            with path.open("wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            profiles[name]["pages"][key] = {"path": f"pages/{stem}{suffix}",
                                            "sha256": hashlib.sha256(data).hexdigest(),
                                            "raw_bytes": entry["raw_bytes"]}
    if worst > 1e-5:  # one float32 ulp at these elevations is 7.6e-6 m
        raise ValueError(f"q32dz transcode is not lossless: {worst} m")
    for name in CODECS:
        pack = target / name / "pages.eatpack"
        build_archive(target / name, profiles[name], pack)
        with pack.open("rb") as handle:
            os.fsync(handle.fileno())
        write_json(target / name / "pages.json", profiles[name])
    return {"root": str(target), "intervals": manifest["page_intervals"],
            "side": manifest["sample_side"], "contentId": manifest["content_id"],
            "transcodeMaxErrorM": worst}


def _load(staged: dict) -> dict:
    root = Path(staged["root"])
    loaded = {}
    for name in CODECS:
        pages = read_json(root / name / "pages.json")["pages"]
        loaded[name] = {"dir": root / name, "pages": pages,
                        "archive": _open_archive(root / name / "pages.eatpack")}
    return loaded


def _decode(name: str, blob: bytes, raw_bytes: int):
    return codec.decode(blob, raw_bytes)[0] if name == "eat1" else _q32dz().decode(blob)


def files_for(profile: dict, layout: str, keys: list[str]) -> list[Path]:
    if layout == "packed":
        return [Path(profile["archive"]["path"])]
    return [profile["dir"] / profile["pages"][key]["path"] for key in keys]


def query(name: str, profile: dict, layout: str, window, intervals: int) -> dict:
    """One request, timed whole: open, read, verify, decode and extract."""
    covers = pages_for(window, intervals)
    wall = time.perf_counter()
    cpu = time.process_time()
    t_io = t_hash = t_decode = 0.0
    read = useful = decoded = 0
    handle = None
    if layout == "packed":
        start = time.perf_counter()
        handle = open(profile["archive"]["path"], "rb")
        t_io += time.perf_counter() - start
    try:
        for xy in covers:
            key = f"0/{xy[0]}/{xy[1]}"
            entry = profile["pages"][key]
            start = time.perf_counter()
            if handle is None:
                blob = (profile["dir"] / entry["path"]).read_bytes()
            else:
                offset, length = profile["archive"]["entries"][key]
                handle.seek(profile["archive"]["bodyStart"] + offset)
                blob = handle.read(length)
            t_io += time.perf_counter() - start
            read += len(blob)
            start = time.perf_counter()
            if hashlib.sha256(blob).hexdigest() != entry["sha256"]:
                raise ValueError(f"{name} page {key} failed its hash")
            t_hash += time.perf_counter() - start
            start = time.perf_counter()
            values = _decode(name, blob, entry["raw_bytes"])
            t_decode += time.perf_counter() - start
            decoded += values.size
            useful += _extract(values, xy, window, intervals)
    finally:
        if handle is not None:
            handle.close()
    return {"wallMs": (time.perf_counter() - wall) * 1e3, "cpuMs": (time.process_time() - cpu) * 1e3,
            "ioMs": t_io * 1e3, "hashMs": t_hash * 1e3, "decodeMs": t_decode * 1e3,
            "pagesTouched": len(covers), "fileOpens": 1 if handle is not None else len(covers),
            "bytesRead": read, "samplesDecoded": decoded, "usefulSamples": useful}


def page_cache_backed(path: Path) -> bool:
    """Whether reads of this file go through this kernel's page cache at all.

    WSL's 9p mount of the Windows drive does not: a file just read is still not
    resident, so "evicted and verified" would be vacuously true while the
    Windows cache behind it stays warm. Such a location can never be cold here.
    """
    path.read_bytes()
    try:
        return resident_pages(path)[0] > 0
    except OSError:
        return False


def sweep(locations: list[dict], warm_trials: int, cold_trials: int) -> list[dict]:
    """One process's observations. Condition order rotates every trial so no
    codec or layout systematically runs first."""
    conditions = [(c, l) for c in CODECS for l in LAYOUTS]
    rows = []
    for location in locations:
        profiles = _load(location["staged"])
        backed = page_cache_backed(Path(profiles["eat1"]["archive"]["path"]))
        intervals, side = location["staged"]["intervals"], location["staged"]["side"]
        for samples in QUERY_SAMPLES:
            window = window_for(samples, side)
            keys = [f"0/{x}/{y}" for x, y in pages_for(window, intervals)]
            for cache, trials in (("cold", cold_trials), ("warm", warm_trials)):
                if cache == "warm":
                    for name, layout in conditions:
                        query(name, profiles[name], layout, window, intervals)
                for trial in range(trials):
                    shift = trial % len(conditions)
                    for name, layout in conditions[shift:] + conditions[:shift]:
                        verified = None
                        if cache == "cold":
                            verified = evict(files_for(profiles[name], layout, keys)) and backed
                        result = query(name, profiles[name], layout, window, intervals)
                        rows.append({"filesystem": location["filesystem"], "codec": name,
                                     "layout": layout, "requestedSamples": samples,
                                     "cache": cache if verified in (None, True) else "cold-unverified",
                                     "trial": trial, **result})
    return rows


def _child(config_path: Path, out: Path) -> None:
    config = read_json(config_path)
    record = {"busyBefore": host_busy_fraction()}
    rows = sweep(config["locations"], config["warmTrials"], config["coldTrials"])
    record.update(busyAfter=host_busy_fraction(), rows=rows)
    write_json(out, record)


def run(atlas_path: Path, out_dir: Path, extra_root: Path | None, processes: int = 3,
        warm_trials: int = 11, cold_trials: int = 7) -> Path:
    """Stage, then run `processes` fresh interpreters one after another."""
    from geoneural.bench.bench_io import filesystem_of
    atlas_path = Path(atlas_path).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    # The primary copy lives inside the run, on the output directory's filesystem.
    roots = [out_dir / "staged"] + ([Path(extra_root)] if extra_root else [])
    locations = []
    for root in roots:
        staged = stage(atlas_path, root)
        locations.append({"filesystem": filesystem_of(Path(staged["root"])), "staged": staged})
    config = {"locations": locations, "warmTrials": warm_trials, "coldTrials": cold_trials}
    write_json(out_dir / "config.json", config)
    for index in range(processes):
        child = subprocess.run([sys.executable, "-m", "geoneural.bench.io_qualify",
                                str(out_dir / "config.json"), str(out_dir / f"process-{index}.json")],
                               capture_output=True, text=True, timeout=3000)
        if child.returncode != 0:
            raise RuntimeError(f"process {index} failed: {child.stderr[-800:]}")
    report = summarise(out_dir)
    report["run"] = run_record("io-qualified", {"atlas": str(atlas_path), "processes": processes},
                               {"out": out_dir})
    write_json(out_dir / "summary.json", report)
    return out_dir / "summary.json"


def qualification(processes: list[dict]) -> dict:
    """Qualified only if every process saw an idle host before and after. Each reason is kept."""
    reasons = []
    if not processes:
        reasons.append("no process records")
    for index, record in enumerate(processes):
        for moment in ("busyBefore", "busyAfter"):
            busy = record.get(moment)
            if busy is None or busy > BUSY_CEILING:
                reasons.append(f"process {index}: host busy {busy} at {moment}")
    return {"qualified": not reasons, "reasons": reasons, "busyCeiling": BUSY_CEILING}


def _quantiles(values: list[float]) -> dict:
    ordered = sorted(values)
    # statistics.quantiles needs two points before Python 3.13; one sample is its own quartiles.
    q = statistics.quantiles(ordered, n=4, method="inclusive") if len(ordered) > 1 else ordered * 3
    return {"median": statistics.median(ordered), "p25": q[0], "p75": q[2],
            "min": ordered[0], "max": ordered[-1], "n": len(ordered)}


def summarise(out_dir: Path) -> dict:
    processes = [read_json(p) for p in sorted(out_dir.glob("process-*.json"))]
    groups: dict[tuple, list[tuple[int, dict]]] = {}
    for index, record in enumerate(processes):
        for row in record["rows"]:
            key = (row["filesystem"], row["codec"], row["layout"], row["requestedSamples"], row["cache"])
            groups.setdefault(key, []).append((index, row))
    table = []
    for key, members in sorted(groups.items()):
        rows = [row for _, row in members]
        per_process = [statistics.median(r["wallMs"] for i, r in members if i == index)
                       for index in sorted({i for i, _ in members})]
        first = rows[0]
        table.append({"filesystem": key[0], "codec": key[1], "layout": key[2],
                      "requestedSamples": key[3], "cache": key[4],
                      "wallMs": _quantiles([r["wallMs"] for r in rows]),
                      "cpuMs": _quantiles([r["cpuMs"] for r in rows]),
                      "phaseMedianMs": {p: statistics.median(r[p] for r in rows)
                                        for p in ("ioMs", "hashMs", "decodeMs")},
                      "processMedianSpread": max(per_process) / max(min(per_process), 1e-9),
                      **{k: first[k] for k in ("pagesTouched", "fileOpens", "bytesRead", "usefulSamples")},
                      "samplesDecodedPerUseful": first["samplesDecoded"] / max(first["usefulSamples"], 1)})
    return {"schema": SCHEMA, "qualification": qualification(processes), "conditions": table,
            "cacheScope": "cold = evicted from this kernel's page cache with posix_fadvise(DONTNEED) and "
                          "verified absent with mincore before the timed request; directory/inode caches and "
                          "host-side caches (WSL2 disk image, Windows cache behind 9p) not controlled. "
                          "warm = every request of the block was primed once, untimed, and its files stay "
                          "resident; cold-unverified trials are kept under their own label.",
            "scope": "Level-0 pages, corner-anchored windows, serial, manifest/archive index in memory. "
                     "Not renderer frame time, not Windows reading NTFS natively."}


if __name__ == "__main__":
    _child(Path(sys.argv[1]), Path(sys.argv[2]))
