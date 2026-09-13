"""Label every timing number with how it was taken, and refuse to take one that cannot be cited.

There are two kinds of clock reading and they must not be pooled. A throughput run shares
the GPU with other studies, so its wall-clock fields only give ratios between arms measured
back to back. A qualified run is serial on a host that was checked to be idle, and is the only
kind whose absolute times may be quoted.

Cold process, cold code, cold source and warm steady state are separate measurements. Software
and GPU rendering, ext4 and Windows-to-WSL IO are never pooled, compilation and model load are
recorded as inside or outside the measurement, and overlapping CPU and GPU spans are not added
as if they were serial.
"""
from __future__ import annotations

import time

#: What a measurement can be cold with respect to. A run answers exactly one of these.
CACHE_STATES = ("cold-process", "cold-code", "cold-source", "warm")


def host_busy_fraction(window_s: float = 0.4) -> float:
    """Fraction of CPU time that was not idle over a short window."""
    def snapshot():
        with open("/proc/stat", "r", encoding="ascii") as handle:
            fields = [float(v) for v in handle.readline().split()[1:]]
        return sum(fields), fields[3] + (fields[4] if len(fields) > 4 else 0.0)

    total_before, idle_before = snapshot()
    time.sleep(window_s)
    total_after, idle_after = snapshot()
    spent = total_after - total_before
    if spent <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - (idle_after - idle_before) / spent))


def block(mode: str, cache_state: str, workers: int = 1, engine: dict | None = None,
          gpu: dict | None = None, warmup_iterations: int = 0, busy_before: float | None = None,
          busy_after: float | None = None) -> dict:
    """The record every timed report carries."""
    if cache_state not in CACHE_STATES:
        raise ValueError(f"Unknown cache state: {cache_state}")
    qualified = bool(mode == "qualified" and workers == 1)
    return {
        "mode": mode, "qualified": qualified, "contended": bool(workers > 1),
        "cacheState": cache_state,
        "concurrency": {"workers": int(workers),
                        "topology": "serial" if workers == 1 else "threads",
                                                "otherGpuProcessesObserved": None},
        "hostBusyFraction": {"before": busy_before, "after": busy_after},
        "engine": engine,
        "gpu": gpu,
        "warmupIterations": int(warmup_iterations),
        "synchronisation": "torch.cuda.synchronize() inside the timed region on GPU runs",
        "clock": "time.perf_counter",
        "qualification": (
            "Qualified: serial run on a host checked idle before and after. Absolute times hold "
            "for this hardware, this cache state and this query profile only." if qualified else
            "Not qualified. Wall-clock fields are inflated by whatever else was running and only "
            "give a ratio between arms measured back to back. Do not quote absolute times."),
    }


def require_quiet(workers: int, busy_ceiling: float = 0.15) -> dict:
    """Refuse to start a qualified measurement on a host that cannot give one."""
    if workers != 1:
        raise RuntimeError(
            f"a qualified timing run is serial; {workers} workers were requested")
    busy = host_busy_fraction()
    if busy > busy_ceiling:
        raise RuntimeError(
            f"host is {busy:.0%} busy, above the {busy_ceiling:.0%} ceiling; a timing run here "
            "would measure the other work")
    return {"hostBusyFraction": busy}
