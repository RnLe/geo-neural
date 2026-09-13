"""Does a reconstruction still drain the way the measured terrain drains?

This is the drainage-preservation metric. It is defined without reference to
any model's output, because a metric chosen after seeing a model's output tends
to flatter that model.

The failure it guards against is easy to miss: a codec can lower mean error
everywhere while raising a single ridge by a metre, and a ridge in the wrong
place sends a river down the wrong valley. Elevation error and drainage error
are different quantities, and the rate-distortion tables of the codec
comparison measure only the first.

Three things are compared, always with the same algorithm and parameters on
both fields, because comparing two fields through two differently configured
hydrologies measures the configuration:

- where each cell sends its water (D8 receiver agreement),
- which cells carry a stream at a fixed contributing-area threshold,
- which outlet each cell ultimately drains to (basin membership).

Depressions are filled first. D8 is undefined in a pit (every neighbour is
higher, so the cell has no receiver), and a quantized field invents pits that
the source did not have, so leaving them unfilled would measure the pits rather
than the drainage. Priority-flood [Barnes et al.] raises each pit to its spill
elevation and adds a tiny monotone increment so flow stays defined across the
filled flat. The increment is recorded in the output: it is a modelling choice,
and the same one is applied to both fields.

What this does not do: it is not a hydrological simulation. It carries no
discharge, no infiltration and no time, and a DTM's bare surface is not a
riverbed under water. It answers one question: whether two surfaces route water
the same way.
"""
from __future__ import annotations
import heapq
from pathlib import Path

import numpy as np

SCHEMA = "geoneural-hydrology-v1"
# Raised over the spill elevation per step across a filled flat, so that a
# filled depression still has a defined downhill direction. Far below the
# 0.005 m quantum, so it cannot move a comparison it is meant to enable.
FILL_EPSILON_M = 1e-6
# Eight neighbours, in (row, col) offsets, with their planar distances.
_OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
_DISTANCE = tuple(float(np.hypot(dr, dc)) for dr, dc in _OFFSETS)
NO_RECEIVER = -1


def fill_depressions(surface: np.ndarray, epsilon: float = FILL_EPSILON_M) -> np.ndarray:
    """Priority-flood: raise every pit to the lowest elevation it can spill over.

    Cells on the domain edge are the initial outlets. Working outward from the
    lowest of them, a neighbour is raised to just above the level the water
    reached getting there, which is exactly its spill elevation.
    """
    filled = np.array(surface, dtype=np.float64, copy=True)
    rows, cols = filled.shape
    visited = np.zeros(filled.shape, dtype=bool)
    heap: list[tuple[float, int, int]] = []
    for r in range(rows):
        for c in (0, cols - 1):
            heap.append((filled[r, c], r, c))
            visited[r, c] = True
    for c in range(1, cols - 1):
        for r in (0, rows - 1):
            heap.append((filled[r, c], r, c))
            visited[r, c] = True
    heapq.heapify(heap)

    while heap:
        level, r, c = heapq.heappop(heap)
        for dr, dc in _OFFSETS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or visited[nr, nc]:
                continue
            visited[nr, nc] = True
            if filled[nr, nc] <= level:
                filled[nr, nc] = level + epsilon
            heapq.heappush(heap, (filled[nr, nc], nr, nc))
    return filled


def d8_receivers(filled: np.ndarray, spacing_m: float) -> np.ndarray:
    """Index of the neighbour each cell drains to, by steepest descent.

    Returns a flat receiver index per cell, or NO_RECEIVER where the cell
    leaves the domain. Slope is per metre of ground, so diagonals are not
    unfairly favoured by being longer.
    """
    rows, cols = filled.shape
    best_slope = np.full(filled.shape, -np.inf)
    receiver = np.full(filled.shape, NO_RECEIVER, dtype=np.int64)
    flat = np.arange(rows * cols, dtype=np.int64).reshape(rows, cols)

    for (dr, dc), distance in zip(_OFFSETS, _DISTANCE):
        # Overlapping slices of the source and shifted grids, without padding.
        src = (slice(max(0, -dr), rows - max(0, dr)), slice(max(0, -dc), cols - max(0, dc)))
        dst = (slice(max(0, dr), rows - max(0, -dr)), slice(max(0, dc), cols - max(0, -dc)))
        drop = (filled[src] - filled[dst]) / (distance * spacing_m)
        better = drop > best_slope[src]
        best_slope[src] = np.where(better, drop, best_slope[src])
        receiver[src] = np.where(better, flat[dst], receiver[src])

    # A cell on the edge with no lower inside neighbour drains out of the domain.
    edge = np.zeros(filled.shape, dtype=bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    receiver[(best_slope <= 0.0) & edge] = NO_RECEIVER
    # Interior cells cannot be pits after filling, but guard rather than assume.
    receiver[(best_slope <= 0.0) & ~edge] = NO_RECEIVER
    return receiver


def flow_accumulation(filled: np.ndarray, receiver: np.ndarray) -> np.ndarray:
    """Cells draining through each cell, itself included.

    Accumulated in order of descending filled elevation: a cell's own total is
    complete before it is handed downstream, because water only moves down.
    """
    flat_receiver = receiver.ravel()
    accumulation = np.ones(filled.size, dtype=np.int64)
    order = np.argsort(filled.ravel(), kind="stable")[::-1]
    for index in order:
        target = flat_receiver[index]
        if target != NO_RECEIVER:
            accumulation[target] += accumulation[index]
    return accumulation.reshape(filled.shape)


def basins(receiver: np.ndarray) -> np.ndarray:
    """The outlet each cell ultimately reaches, by pointer jumping.

    Repeated squaring rather than a walk per cell: the chains are long and a
    naive walk on a megacell grid is quadratic in the worst case.
    """
    flat = receiver.ravel().copy()
    terminal = flat == NO_RECEIVER
    outlet = np.where(terminal, np.arange(flat.size, dtype=np.int64), flat)
    for _ in range(64):
        nxt = outlet[outlet]
        if np.array_equal(nxt, outlet):
            break
        outlet = nxt
    return outlet.reshape(receiver.shape)


# Drainage on near-flat ground is ill-conditioned: where the surface barely
# tilts, which way a cell drains can turn on a millimetre, and the "correct"
# answer is not well determined by the data either. One pooled agreement number
# would blame the codec for ambiguity that belongs to the terrain, so cells are
# stratified by reference slope to tell the two apart.
SLOPE_CLASSES = ((0.0, 0.005), (0.005, 0.02), (0.02, 0.05), (0.05, 0.15), (0.15, float("inf")))


def reference_slope(surface: np.ndarray, spacing_m: float) -> np.ndarray:
    """Gradient magnitude, dimensionless rise over run."""
    dy, dx = np.gradient(surface, spacing_m)
    return np.hypot(dx, dy)


def stratify(reference: np.ndarray, spacing_m: float, receiver_same: np.ndarray,
             basin_same: np.ndarray, interior: np.ndarray) -> list[dict]:
    slope = reference_slope(reference, spacing_m)
    rows = []
    for low, high in SLOPE_CLASSES:
        band = (slope >= low) & (slope < high) & interior
        count = int(np.count_nonzero(band))
        if count == 0:
            continue
        rows.append({
            "slopeFrom": low, "slopeTo": None if high == float("inf") else high,
            "cells": count, "shareOfDomain": count / int(np.count_nonzero(interior)),
            "receiverAgreementFraction": float(np.count_nonzero(receiver_same & band) / count),
            "basinAgreementFraction": float(np.count_nonzero(basin_same & band) / count),
        })
    return rows


def analyse(surface: np.ndarray, spacing_m: float, stream_cells: int) -> dict:
    filled = fill_depressions(surface)
    receiver = d8_receivers(filled, spacing_m)
    accumulation = flow_accumulation(filled, receiver)
    return {
        "filled": filled, "receiver": receiver, "accumulation": accumulation,
        "basin": basins(receiver),
        "stream": accumulation >= stream_cells,
        "filledCells": int(np.count_nonzero(filled > surface)),
        "maxFillM": float((filled - surface).max()),
    }


def compare(reference: np.ndarray, reconstructed: np.ndarray, spacing_m: float,
            stream_cells: int = 500, reference_analysis: dict | None = None) -> dict:
    """Route both fields identically and report where they disagree.

    `reference_analysis` may pass in `analyse(reference, ...)` computed once for many
    reconstructions of the same reference.
    """
    if reference.shape != reconstructed.shape:
        raise ValueError("hydrological comparison needs identical grids")
    left = reference_analysis or analyse(reference, spacing_m, stream_cells)
    right = analyse(reconstructed, spacing_m, stream_cells)

    interior = np.ones(reference.shape, dtype=bool)
    interior[0, :] = interior[-1, :] = interior[:, 0] = interior[:, -1] = False
    receiver_same = (left["receiver"] == right["receiver"]) & interior
    both_stream = left["stream"] & right["stream"]
    either_stream = left["stream"] | right["stream"]
    basin_same = left["basin"] == right["basin"]

    # Where a stream existed in the reference, did it survive?
    reference_stream = int(np.count_nonzero(left["stream"]))
    kept = int(np.count_nonzero(both_stream))
    return {
        "streamThresholdCells": stream_cells,
        "streamThresholdKm2": stream_cells * (spacing_m ** 2) / 1e6,
        "receiverAgreementFraction": float(np.count_nonzero(receiver_same) / np.count_nonzero(interior)),
        "streamJaccard": float(kept / max(int(np.count_nonzero(either_stream)), 1)),
        "referenceStreamCells": reference_stream,
        "reconstructedStreamCells": int(np.count_nonzero(right["stream"])),
        "streamCellsLost": reference_stream - kept,
        "streamRecall": float(kept / max(reference_stream, 1)),
        "basinAgreementFraction": float(np.count_nonzero(basin_same) / basin_same.size),
        "referenceBasins": int(np.unique(left["basin"]).size),
        "reconstructedBasins": int(np.unique(right["basin"]).size),
        "referenceFilledCells": left["filledCells"],
        "reconstructedFilledCells": right["filledCells"],
        "spuriousPitVolumeM": right["maxFillM"] - left["maxFillM"],
        "elevationMaeM": float(np.abs(reference - reconstructed).mean()),
        "elevationMaxM": float(np.abs(reference - reconstructed).max()),
        "bySlopeClass": stratify(reference, spacing_m, receiver_same, basin_same, interior),
    }


def against_targets(reference: np.ndarray, spacing_m: float, targets, quantum_m: float,
                    stream_cells: int = 500) -> dict:
    """Route the reference and each error-bounded reconstruction identically.

    A max-error target of `t` is met by a uniform quantizer of step `2t`, which
    is what the quantized codecs in the codec comparison do. Comparing
    by target rather than by codec is deliberate: the entropy coder changes the
    bytes, not the reconstructed heights, so drainage damage is a property of
    the error budget and every codec meeting the same target inherits it.
    """
    rows = []
    for target in targets:
        step = 2.0 * float(target)
        reconstructed = np.round(reference / step) * step
        rows.append({"targetMaxErrorM": float(target), "quantizerStepM": step,
                     **compare(reference, reconstructed, spacing_m, stream_cells)})
    return {
        "schema": SCHEMA,
        "spacingM": spacing_m, "atlasQuantumM": quantum_m,
        "streamThresholdCells": stream_cells,
        "fillEpsilonM": FILL_EPSILON_M,
        "method": "priority-flood depression filling, D8 steepest descent by ground slope, "
                  "contributing-area accumulation, basin membership by pointer jumping; "
                  "identical algorithm and parameters on both fields",
        "byTarget": rows,
        "qualification": "A routing comparison between two surfaces, not a hydrological simulation. "
                         "No discharge, infiltration or time. A bare-earth DTM is not a riverbed "
                         "under water, so an absent channel here is not an absent river on the "
                         "ground. Reconstructions are uniform quantizations meeting each target, "
                         "which is what the quantized codecs produce; a codec with a different "
                         "error distribution can differ.",
    }
