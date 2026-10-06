"""Benchmarks for the v2 products: a common query workload, parallel encode scaling and a compute profile.

Every timing carries the host's busy fraction before and after, and is labelled qualified only when the host was
idle (below `IDLE`) at both ends of a serial measurement. Answers to queries are checked against the full decode.
"""
from __future__ import annotations

import cProfile
import json
import os
import pstats
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from geoneural.bench.timing import host_busy_fraction
from geoneural.codecs import campaign, foreign, package
from geoneural.codecs.predictor import Predictor

IDLE = 0.10
PAGE = campaign.PAGE


def _busy() -> float:
    return round(host_busy_fraction(0.5), 3)


def _gpu_busy() -> float | None:
    """GPU utilisation in percent as nvidia-smi reports it (other processes included), or None."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return float(out.split()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def workloads(side: int = 1025, seed: int = 0) -> dict:
    """The same query arrays for every product: random points, clustered points, 16 windows of 128 x 128 nodes,
    the full raster, and an overview at every eighth node."""
    rng = np.random.default_rng(seed)
    centres = rng.integers(60, side - 60, (10, 2))
    clustered = np.concatenate([np.clip(c + rng.normal(0, 25, (100, 2)), 0, side - 1) for c in centres]).astype(int)
    windows = [(int(r), int(c)) for r, c in rng.integers(0, side - 128, (16, 2))]
    return {"random": rng.integers(0, side, (1000, 2)), "clustered": clustered, "windows": windows,
            "full": None, "overview": 8}


def _answer(field: np.ndarray, kind: str, q) -> np.ndarray:
    if kind in ("random", "clustered"):
        return field[q[:, 0], q[:, 1]]
    if kind == "windows":
        return np.concatenate([field[r:r + 128, c:c + 128].ravel() for r, c in q])
    if kind == "overview":
        return field[::q, ::q].ravel()
    return field.ravel()


def _pages_for(kind: str, q, side: int) -> list[tuple[int, int]]:
    """Pages (origins on the PAGE grid) a query touches."""
    starts = [o for o in range(0, side - 1, PAGE)]
    def page_of(i):
        return min(int(i) // PAGE, len(starts) - 1) * PAGE
    if kind in ("random", "clustered"):
        return sorted({(page_of(r), page_of(c)) for r, c in q})
    if kind == "windows":
        out = set()
        for r, c in q:
            for rr in {page_of(r), page_of(r + 127)}:
                for cc in {page_of(c), page_of(c + 127)}:
                    out.add((rr, cc))
        return sorted(out)
    return [(r, c) for r in starts for c in starts]


def products(region: str, bound: float, model: Predictor) -> dict:
    """Whole-field products (SZ3, SPERR, zfp, learned with its model inside) and paged products (SZ3 and learned,
    one file per page, the learned model stored once)."""
    z, atlas = campaign.load(region)
    spatial = package.spatial_from_atlas(atlas)
    out = {}
    for codec in ("sz3", "sperr", "zfp"):
        blob = foreign.encode(codec, np.ascontiguousarray(z, np.float32), bound)
        out[codec] = {"kind": "whole", "codec": codec, "blob": package.wrap_foreign(codec, blob, z.shape, bound, spatial)}
    out["learned"] = {"kind": "whole", "codec": "learned",
                      "blob": package.encode(z, bound, "learned", model, embed_model=True, spatial=spatial)[0]}
    for codec in ("sz3", "learned"):
        pages = {}
        for r, c in campaign.pages(z.shape[0]):
            tile = np.ascontiguousarray(z[r:r + PAGE + 1, c:c + PAGE + 1])
            if codec == "learned":
                pages[(r, c)] = package.encode(tile, bound, "learned", model, embed_model=False)[0]
            else:
                pages[(r, c)] = package.wrap_foreign(codec, foreign.encode(codec, np.asarray(tile, np.float32), bound),
                                                     tile.shape, bound)
        out[f"{codec}-paged"] = {"kind": "paged", "codec": codec, "pages": pages}
    return out


def _decode_whole(p: dict, model: Predictor) -> np.ndarray:
    blob = p["blob"]
    if p["codec"] == "learned":
        return package.decode(blob)
    prod = package.read(blob)
    return foreign.decode(p["codec"], prod.components["foreign"], (prod.rows, prod.cols)).astype(np.float64)


def _decode_pages(p: dict, model: Predictor, which, side: int) -> np.ndarray:
    field = np.full((side, side), np.nan)
    for r, c in which:
        blob = p["pages"][(r, c)]
        if p["codec"] == "learned":
            tile = package.decode(blob, model)
        else:
            prod = package.read(blob)
            tile = foreign.decode(p["codec"], prod.components["foreign"], (prod.rows, prod.cols)).astype(np.float64)
        field[r:r + tile.shape[0], c:c + tile.shape[1]] = tile
    return field


def query_workload(region: str = "essen-ruhr", bound: float = 0.25, repeats: int = 7, model_path: Path | None = None,
                   workdir: Path | None = None, log=print) -> dict:
    model = Predictor.from_bytes(Path(model_path).read_bytes())
    prods = products(region, bound, model)
    side = 1025
    wl = workloads(side)
    workdir = Path(workdir or (campaign.HOME / "bench" / "queries"))
    workdir.mkdir(parents=True, exist_ok=True)
    np.save(workdir / "model.npy", np.frombuffer(model.to_bytes(), np.uint8))
    rows = []
    full = {name: (_decode_whole(p, model) if p["kind"] == "whole" else _decode_pages(p, model, campaign.pages(side), side))
            for name, p in prods.items()}
    for name, p in prods.items():
        size = len(p["blob"]) if p["kind"] == "whole" else sum(len(b) for b in p["pages"].values()) + 8 * len(p["pages"])
        if p["kind"] == "paged" and p["codec"] == "learned":
            size += model.nbytes() + 9
        path = workdir / f"{name}.bin"
        if p["kind"] == "whole":
            path.write_bytes(p["blob"])
        else:
            import pickle
            path.write_bytes(pickle.dumps({k: v for k, v in p["pages"].items()}))
        for kind, q in wl.items():
            which = _pages_for(kind, q, side) if p["kind"] == "paged" else None
            busy0 = _busy()
            warm = []
            for _ in range(repeats):
                t = time.perf_counter()
                field = _decode_whole(p, model) if p["kind"] == "whole" else _decode_pages(p, model, which, side)
                ans = _answer(field, kind, q)
                warm.append(time.perf_counter() - t)
            if not np.array_equal(ans, _answer(full[name], kind, q)):
                raise RuntimeError(f"{name} {kind}: partial decode differs from the full decode")
            t = time.perf_counter()
            for _ in range(repeats):
                _answer(full[name], kind, q)
            cached = (time.perf_counter() - t) / repeats
            cold = _cold(path, p, kind, region, bound, workdir)
            busy1 = _busy()
            rows.append({"product": name, "query": kind, "bytes": size, "pagesDecoded": len(which) if which else None,
                         "warmP50S": float(np.median(warm)), "warmP95S": float(np.quantile(warm, 0.95)),
                         "cachedS": cached, **cold, "busyBefore": busy0, "busyAfter": busy1,
                         "qualified": bool(busy0 < IDLE and busy1 < IDLE)})
            log(f"queries {name} {kind}: warm {np.median(warm):.3f} s, cold {cold['coldProcessS']:.2f} s")
    return {"region": region, "boundM": bound, "repeats": repeats, "rows": rows,
            "host": _host(), "workloads": {"random": 1000, "clustered": 1000, "windows": "16 x 128^2",
                                           "full": "1025^2", "overview": "every 8th node"}}


_COLD = r"""
import sys, time, json, pickle, resource
t0 = time.perf_counter()
from pathlib import Path
import numpy as np
from geoneural.bench import v2
from geoneural.codecs import package
from geoneural.codecs.predictor import Predictor
path, kind_p, codec, kind, region, bound, workdir = sys.argv[1:8]
wl = v2.workloads(1025)
q = wl[kind]
t1 = time.perf_counter()
model = Predictor.from_bytes(np.load(Path(workdir) / "model.npy").tobytes())
data = Path(path).read_bytes()
if kind_p == "whole":
    field = v2._decode_whole({"codec": codec, "blob": data}, model)
else:
    pages = pickle.loads(data)
    field = v2._decode_pages({"codec": codec, "pages": pages}, model, v2._pages_for(kind, q, 1025), 1025)
ans = v2._answer(field, kind, q)
t2 = time.perf_counter()
print(json.dumps({"coldProcessS": t2 - t0, "coldImportS": t1 - t0, "coldDecodeS": t2 - t1,
                  "peakRssMiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}))
"""


def _cold(path, p, kind, region, bound, workdir) -> dict:
    env = dict(os.environ, OMP_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
    t = time.perf_counter()
    res = subprocess.run([sys.executable, "-c", _COLD, str(path), p["kind"], p["codec"], kind, region, str(bound),
                          str(workdir)], capture_output=True, text=True, env=env, check=True)
    out = json.loads(res.stdout.strip().splitlines()[-1])
    out["coldWallS"] = time.perf_counter() - t
    return out


def _host() -> dict:
    import platform
    info = {"python": platform.python_version(), "machine": platform.machine(), "cpus": os.cpu_count()}
    try:
        info["cpu"] = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except (OSError, StopIteration):
        pass
    return info


def _encode_task(args):
    region, bound, model_bytes, out_dir = args
    model = Predictor.from_bytes(model_bytes)
    z, atlas = campaign.load(region)
    t = time.perf_counter()
    blob, _, _ = package.encode(z, bound, "learned", model, embed_model=True,
                                spatial=package.spatial_from_atlas(atlas))
    Path(out_dir, f"{region}-{bound}.gnc").write_bytes(blob)
    return time.perf_counter() - t


def scaling(regions, bounds=(0.1, 0.5), workers=(1, 2, 4, 8, 16), model_path: Path | None = None,
            out_dir: Path | None = None, repeats: int = 3, per_worker: int = 4, log=print) -> dict:
    """Independent tile encodes as a job array. Strong scaling: the same task list on 1 to 16 single-threaded
    workers. Weak scaling: `per_worker` tasks per worker. Each setting runs `repeats` times in a warm pool; the
    median wall time is reported."""
    import multiprocessing as mp
    model_bytes = Path(model_path).read_bytes()
    out_dir = Path(out_dir or (campaign.HOME / "bench" / "scaling"))
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.update(OMP_NUM_THREADS="1", NUMBA_NUM_THREADS="1", MKL_NUM_THREADS="1")
    tasks = [(r, b, model_bytes, str(out_dir)) for r in regions for b in bounds]
    ctx = mp.get_context("spawn")
    with ctx.Pool(1) as pool:  # compile numba and fill caches once before timing
        pool.map(_encode_task, tasks[:1])
    rows = []
    for mode in ("strong", "weak"):
        for n in workers:
            todo = tasks if mode == "strong" else [tasks[i % len(tasks)] for i in range(per_worker * n)]
            time.sleep(2)  # let the previous pool's processes exit before probing the host
            busy0 = _busy()
            walls, per = [], []
            with ctx.Pool(n) as pool:
                pool.map(_encode_task, todo[:n], chunksize=1)  # warm each worker
                for _ in range(repeats):
                    t = time.perf_counter()
                    per += pool.map(_encode_task, todo, chunksize=1)
                    walls.append(time.perf_counter() - t)
                busy1 = _busy()
            wall = float(np.median(walls))
            rows.append({"mode": mode, "workers": n, "tasks": len(todo), "wallS": wall, "wallsS": walls,
                         "tasksPerS": len(todo) / wall, "meanTaskS": float(np.mean(per)),
                         "busyBefore": busy0, "busyAfter": busy1,
                         "qualified": bool(busy0 < IDLE and busy1 < IDLE)})
            log(f"scaling {mode} {n}: {wall:.2f} s for {len(todo)} tasks")
    for mode in ("strong", "weak"):
        base = next(r for r in rows if r["mode"] == mode and r["workers"] == 1)
        for r in rows:
            if r["mode"] == mode:
                r["efficiency"] = (base["wallS"] / (r["workers"] * r["wallS"]) if mode == "strong"
                                   else base["wallS"] / r["wallS"])
    return {"rows": rows, "host": _host(), "boundsM": list(bounds), "regions": list(regions), "repeats": repeats,
            "tasksPerWorkerWeak": per_worker}


def profile(region: str = "essen-ruhr", bound: float = 0.25, model_path: Path | None = None, train: bool = True,
            log=print) -> dict:
    """Where the time goes: stage timings for loading, routing, SZ3, and the learned encode and decode, the
    cProfile split of the learned decode, and the cost of training one shared model (collection on the CPU,
    optimisation on the GPU)."""
    from geoneural.codecs.predictor import collect, train as fit_train
    from geoneural.metrics import drainage
    model = Predictor.from_bytes(Path(model_path).read_bytes())
    stages = {}
    t = time.perf_counter(); z, atlas = campaign.load(region); stages["loadReferenceS"] = time.perf_counter() - t
    t = time.perf_counter(); drainage.route(z, campaign.SPACING_M); stages["routeS"] = time.perf_counter() - t
    z32 = np.ascontiguousarray(z, np.float32)
    t = time.perf_counter(); blob = foreign.encode("sz3", z32, bound); stages["sz3EncodeS"] = time.perf_counter() - t
    t = time.perf_counter(); foreign.decode("sz3", blob, z.shape); stages["sz3DecodeS"] = time.perf_counter() - t
    package.encode(z, bound, "learned", model)  # compile
    t = time.perf_counter(); gblob, _, _ = package.encode(z, bound, "learned", model); stages["learnedEncodeS"] = time.perf_counter() - t
    package.decode(gblob, model)
    t = time.perf_counter(); package.decode(gblob, model); stages["learnedDecodeS"] = time.perf_counter() - t
    prof = cProfile.Profile()
    prof.enable(); package.decode(gblob, model); prof.disable()
    st = pstats.Stats(prof)
    top = sorted(((v[3], f"{k[2]} ({Path(k[0]).name}:{k[1]})") for k, v in st.stats.items()), reverse=True)[:12]
    out = {"region": region, "boundM": bound, "stages": stages, "busy": _busy(), "host": _host(),
           "decodeProfile": [{"cumulativeS": round(c, 4), "function": f} for c, f in top]}
    if train:
        import torch
        fields = [campaign.load(r)[0] for r in campaign.REGIONS]
        gpu0, busy0 = _gpu_busy(), _busy()
        rounds, fitted = [], None
        for r in range(2):  # the frozen recipe (campaign.final_model): two rounds, 5000 steps each
            t = time.perf_counter()
            data = collect(fields, campaign.BOUNDS, fitted, seed=r)
            collect_s = time.perf_counter() - t
            t = time.perf_counter()
            fitted = fit_train(data, (32, 32), 5000, seed=1000 * r, init=fitted)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            rounds.append({"collectS": collect_s, "trainS": time.perf_counter() - t, "samples": int(len(data["x"]))})
            log(f"training round {r}: collect {collect_s:.0f} s, train {rounds[-1]['trainS']:.0f} s")
        out["training"] = {"rounds": rounds, "totalS": sum(x["collectS"] + x["trainS"] for x in rounds),
                           "recipe": "campaign.final_model: six regions, all bounds, two rounds of 5000 steps",
                           "device": "cuda" if torch.cuda.is_available() else "cpu",
                           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                           "busyBefore": busy0, "busyAfter": _busy(),
                           "gpuBusyBeforePct": gpu0, "gpuBusyAfterPct": _gpu_busy()}
    return out
