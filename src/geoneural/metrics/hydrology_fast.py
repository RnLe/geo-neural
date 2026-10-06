"""A compiled priority-flood and accumulation, bit-identical to `hydrology`.

Ensemble and inverse runs call the router thousands of times per simulation,
and the two Python loops in `hydrology` (the priority-flood heap and the
accumulation walk) dominate the run time. Compiling them makes them roughly two
orders of magnitude faster.

Bit-identical, not equivalent. This module exists only to be faster. A fast
router that disagreed with the reference anywhere would make every drainage
number depend on which router produced it. The tie-breaking is therefore
replicated exactly rather than approximately:

* Python's `heapq` orders tuples `(level, row, column)` lexicographically, so
  cells at equal elevation pop in row-then-column order. The heap here compares
  the same triple in the same order. An `<` on elevation alone would be a correct
  priority-flood but a different one.
* `flow_accumulation` walks `argsort(kind="stable")[::-1]`. Reversing a stable
  ascending sort is not a stable descending sort (equal elevations come out in
  reverse index order), and when one such cell drains into another the totals
  differ. The order is taken from numpy unchanged rather than recomputed.

`bit_identical_to_reference` is the check. Callers opt in by passing this
module as a router; the drainage metric itself always uses `hydrology`.
"""
from __future__ import annotations

import numpy as np

from geoneural.metrics import hydrology

try:
    import numba
    AVAILABLE = True
except ImportError:  # pragma: no cover - recorded, not worked around
    numba = None
    AVAILABLE = False

_OFFSETS = np.array([(-1, -1), (-1, 0), (-1, 1), (0, -1),
                     (0, 1), (1, -1), (1, 0), (1, 1)], dtype=np.int64)


if AVAILABLE:
    @numba.njit(cache=True)
    def _sift_up(level, rows_, cols_, size):
        child = size - 1
        while child > 0:
            parent = (child - 1) // 2
            if (level[parent] > level[child] or
                (level[parent] == level[child] and
                 (rows_[parent] > rows_[child] or
                  (rows_[parent] == rows_[child] and cols_[parent] > cols_[child])))):
                level[parent], level[child] = level[child], level[parent]
                rows_[parent], rows_[child] = rows_[child], rows_[parent]
                cols_[parent], cols_[child] = cols_[child], cols_[parent]
                child = parent
            else:
                break

    @numba.njit(cache=True)
    def _sift_down(level, rows_, cols_, size):
        parent = 0
        while True:
            left, right, best = 2 * parent + 1, 2 * parent + 2, parent
            if left < size and (
                    level[left] < level[best] or
                    (level[left] == level[best] and
                     (rows_[left] < rows_[best] or
                      (rows_[left] == rows_[best] and cols_[left] < cols_[best])))):
                best = left
            if right < size and (
                    level[right] < level[best] or
                    (level[right] == level[best] and
                     (rows_[right] < rows_[best] or
                      (rows_[right] == rows_[best] and cols_[right] < cols_[best])))):
                best = right
            if best == parent:
                break
            level[parent], level[best] = level[best], level[parent]
            rows_[parent], rows_[best] = rows_[best], rows_[parent]
            cols_[parent], cols_[best] = cols_[best], cols_[parent]
            parent = best

    @numba.njit(cache=True)
    def _fill_seeded(filled, epsilon, offsets, seeds):
        """Priority flood from the given seed cells only (the outlets), instead of every edge cell."""
        rows, cols = filled.shape
        capacity = rows * cols + 8
        level = np.empty(capacity, dtype=np.float64)
        heap_r = np.empty(capacity, dtype=np.int64)
        heap_c = np.empty(capacity, dtype=np.int64)
        visited = np.zeros(filled.shape, dtype=np.bool_)
        size = 0
        for i in range(seeds.shape[0]):
            r, c = seeds[i, 0], seeds[i, 1]
            level[size] = filled[r, c]
            heap_r[size] = r
            heap_c[size] = c
            visited[r, c] = True
            size += 1
            _sift_up(level, heap_r, heap_c, size)
        while size > 0:
            top, tr, tc = level[0], heap_r[0], heap_c[0]
            size -= 1
            level[0], heap_r[0], heap_c[0] = level[size], heap_r[size], heap_c[size]
            if size > 0:
                _sift_down(level, heap_r, heap_c, size)
            for k in range(offsets.shape[0]):
                nr = tr + offsets[k, 0]
                nc = tc + offsets[k, 1]
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols or visited[nr, nc]:
                    continue
                visited[nr, nc] = True
                if filled[nr, nc] <= top:
                    filled[nr, nc] = top + epsilon
                level[size] = filled[nr, nc]
                heap_r[size] = nr
                heap_c[size] = nc
                size += 1
                _sift_up(level, heap_r, heap_c, size)
        return filled

    @numba.njit(cache=True)
    def _fill(filled, epsilon, offsets):
        rows, cols = filled.shape
        capacity = rows * cols + 8
        level = np.empty(capacity, dtype=np.float64)
        heap_r = np.empty(capacity, dtype=np.int64)
        heap_c = np.empty(capacity, dtype=np.int64)
        visited = np.zeros(filled.shape, dtype=np.bool_)
        size = 0
        for r in range(rows):
            for c in (0, cols - 1):
                level[size] = filled[r, c]
                heap_r[size] = r
                heap_c[size] = c
                visited[r, c] = True
                size += 1
                _sift_up(level, heap_r, heap_c, size)
        for c in range(1, cols - 1):
            for r in (0, rows - 1):
                level[size] = filled[r, c]
                heap_r[size] = r
                heap_c[size] = c
                visited[r, c] = True
                size += 1
                _sift_up(level, heap_r, heap_c, size)
        while size > 0:
            top, tr, tc = level[0], heap_r[0], heap_c[0]
            size -= 1
            level[0], heap_r[0], heap_c[0] = level[size], heap_r[size], heap_c[size]
            if size > 0:
                _sift_down(level, heap_r, heap_c, size)
            for k in range(offsets.shape[0]):
                nr = tr + offsets[k, 0]
                nc = tc + offsets[k, 1]
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols or visited[nr, nc]:
                    continue
                visited[nr, nc] = True
                if filled[nr, nc] <= top:
                    filled[nr, nc] = top + epsilon
                level[size] = filled[nr, nc]
                heap_r[size] = nr
                heap_c[size] = nc
                size += 1
                _sift_up(level, heap_r, heap_c, size)
        return filled

    @numba.njit(cache=True)
    def _accumulate(order, flat_receiver, no_receiver, total):
        accumulation = np.ones(total, dtype=np.int64)
        for i in range(order.shape[0]):
            index = order[i]
            target = flat_receiver[index]
            if target != no_receiver:
                accumulation[target] += accumulation[index]
        return accumulation


def fill_depressions(surface, epsilon: float = hydrology.FILL_EPSILON_M, seeds=None):
    """Compiled priority-flood from every edge cell, or from `seeds` (an array of (row, col) outlets) only.
    Falls back to the reference when numba is absent (edge seeding only)."""
    filled = np.array(surface, dtype=np.float64, copy=True)
    if seeds is not None:
        return _fill_seeded(filled, float(epsilon), _OFFSETS, np.asarray(seeds, np.int64).reshape(-1, 2))
    if not AVAILABLE:
        return hydrology.fill_depressions(surface, epsilon)
    return _fill(filled, float(epsilon), _OFFSETS)


def flow_accumulation(filled, receiver):
    """Compiled accumulation walk, over numpy's own ordering."""
    if not AVAILABLE:
        return hydrology.flow_accumulation(filled, receiver)
    order = np.argsort(np.asarray(filled, dtype=np.float64).ravel(),
                       kind="stable")[::-1]
    accumulation = _accumulate(np.ascontiguousarray(order),
                               np.ascontiguousarray(receiver.ravel()),
                               hydrology.NO_RECEIVER, int(np.asarray(filled).size))
    return accumulation.reshape(np.asarray(filled).shape)


def bit_identical_to_reference(sides=(17, 33, 64), seed: int = 20260914) -> dict:
    """Compare against `hydrology` on synthetic surfaces. Any disagreement
    anywhere means this module must not be used."""
    rng = np.random.default_rng(seed)
    rows = []
    for side in sides:
        for kind in ("noise", "ramp-with-pits", "plateau"):
            if kind == "noise":
                surface = rng.normal(0.0, 5.0, (side, side))
            elif kind == "ramp-with-pits":
                surface = np.add.outer(np.linspace(0.0, 40.0, side),
                                       np.zeros(side))
                surface += rng.normal(0.0, 0.5, (side, side))
                surface[side // 3, side // 2] -= 20.0
                surface[2 * side // 3, side // 4] -= 15.0
            else:
                surface = np.where(np.add.outer(np.arange(side), np.zeros(side, dtype=int))
                                   < side // 2, 10.0, 0.0)
            reference_filled = hydrology.fill_depressions(surface)
            fast_filled = fill_depressions(surface)
            receiver = hydrology.d8_receivers(reference_filled, 10.0)
            reference_flow = hydrology.flow_accumulation(reference_filled, receiver)
            fast_flow = flow_accumulation(fast_filled, receiver)
            rows.append({
                "side": side, "surface": kind,
                "filledIdentical": bool(np.array_equal(reference_filled, fast_filled)),
                "accumulationIdentical": bool(np.array_equal(reference_flow, fast_flow)),
                "maxFilledDifferenceM": float(np.abs(reference_filled - fast_filled).max()),
            })
    return {"available": AVAILABLE, "rows": rows,
            "identical": all(r["filledIdentical"] and r["accumulationIdentical"]
                             for r in rows),
            "note": "Bit-identity, not agreement to a tolerance. A fast router that "
                    "differs anywhere would make every drainage number depend on "
                    "which router produced it."}
