"""Sparse drainage corrections: exact heights where drainage needs them, and their bytes.

The cheapest point in the rate table destroys the stream network: at a 1.0 m
error target the drainage check finds only 22 % of it intact, and local routing
fails even on steep ground. Spending the bytes on a finer target everywhere
pays for accuracy across the whole domain to fix a failure concentrated on a
thin set of cells.

The alternative is to identify the cells that carry the drainage, store their
heights exactly as a separate sparse correction, and leave the rest coarse.
This module measures whether that works and what it costs, in encoded bytes
rather than in cell counts.

Routing is decided by a cell against its neighbours, so an exact channel
surrounded by coarsely quantized banks can still leak: the water finds a
quantization step leading out of the channel. The protected set is therefore
the stream network dilated by a band, and the band width is swept rather than
assumed, because whether the banks need protecting is the question being
measured.

Bytes are counted the way they would ship: sorted indices delta-coded as
varints, heights delta-coded against the coarse reconstruction at the fine
quantum, both compressed. No cell is assumed to be free; the correction is
measured as an encoded payload.
"""
from __future__ import annotations
import gzip

import numpy as np

from geoneural.metrics import hydrology

SCHEMA = "geoneural-corrections-v1"


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Grow a mask by `radius` cells, separably, without scipy."""
    if radius <= 0:
        return mask.copy()
    grown = mask.copy()
    for _ in range(radius):
        padded = np.pad(grown, 1, mode="edge")
        grown = (padded[1:-1, 1:-1] | padded[:-2, 1:-1] | padded[2:, 1:-1]
                 | padded[1:-1, :-2] | padded[1:-1, 2:]
                 | padded[:-2, :-2] | padded[:-2, 2:] | padded[2:, :-2] | padded[2:, 2:])
    return grown


def _varint(values: np.ndarray) -> bytes:
    """Unsigned LEB128. Gaps between sorted indices are small and skewed, so a
    fixed 4-byte index would triple the cost of the cheapest part of this."""
    out = bytearray()
    for value in values.tolist():
        while True:
            byte = value & 0x7F
            value >>= 7
            if value:
                out.append(byte | 0x80)
            else:
                out.append(byte)
                break
    return bytes(out)


def _zigzag(values: np.ndarray) -> np.ndarray:
    """Map signed to unsigned so small negatives stay short as varints."""
    return (values << 1) ^ (values >> 63)


def encode_corrections(indexes: np.ndarray, residual_codes: np.ndarray) -> dict:
    """Pack a sparse correction the way it would actually be deployed."""
    order = np.argsort(indexes)
    sorted_indexes = indexes[order].astype(np.int64)
    codes = residual_codes[order].astype(np.int64)
    gaps = np.diff(sorted_indexes, prepend=np.int64(-1)) - 1
    index_blob = _varint(gaps.astype(np.uint64))
    value_blob = _varint(_zigzag(codes).astype(np.uint64))
    raw = index_blob + value_blob
    packed = gzip.compress(raw, compresslevel=9, mtime=0)
    try:
        import zstandard
        packed_zstd = zstandard.ZstdCompressor(level=19).compress(raw)
    except ImportError:
        packed_zstd = None
    return {
        "cells": int(sorted_indexes.size),
        "indexVarintBytes": len(index_blob), "valueVarintBytes": len(value_blob),
        "rawBytes": len(raw), "gzipBytes": len(packed),
        "zstdBytes": None if packed_zstd is None else len(packed_zstd),
        "deployedBytes": len(packed) if packed_zstd is None else min(len(packed), len(packed_zstd)),
        "bytesPerCorrectedCell": (len(packed) if packed_zstd is None
                                  else min(len(packed), len(packed_zstd))) / max(sorted_indexes.size, 1),
    }


def apply_corrections(coarse: np.ndarray, reference: np.ndarray, protect: np.ndarray,
                      fine_quantum: float) -> tuple[np.ndarray, dict]:
    """Replace protected cells with the reference quantized at the fine step.

    The correction is stored as a residual against the coarse reconstruction, so
    a protected cell whose coarse value already rounds correctly costs almost
    nothing.
    """
    corrected = coarse.copy()
    exact = np.round(reference / fine_quantum) * fine_quantum
    corrected[protect] = exact[protect]
    indexes = np.flatnonzero(protect.reshape(-1))
    residual_codes = np.rint((exact - coarse).reshape(-1)[indexes] / fine_quantum).astype(np.int64)
    return corrected, encode_corrections(indexes, residual_codes)


def sweep(reference: np.ndarray, spacing_m: float, target_m: float,
          radii=(0, 1, 2, 4), fine_quantum: float = 0.01,
          stream_cells: int = 500, check_cells=(250, 1000, 2000),
          uniform_targets=(0.5,)) -> dict:
    """How much drainage each band width buys back, and what it costs.

    The protected set comes from the reference stream network, which the
    encoder has. This is an encoder-side choice, not information a decoder must
    infer.
    """
    step = 2.0 * target_m
    coarse = np.round(reference / step) * step
    baseline = hydrology.compare(reference, coarse, spacing_m, stream_cells)

    analysis = hydrology.analyse(reference, spacing_m, stream_cells)
    stream = analysis["stream"]

    # The protected band is chosen at `stream_cells`, so scoring only at that threshold would
    # grade the method on its own selection rule. Every field is also scored at other thresholds.
    checks = {cells: hydrology.analyse(reference, spacing_m, cells) for cells in check_cells}

    def elsewhere(field: np.ndarray) -> dict:
        return {str(cells): hydrology.compare(reference, field, spacing_m, cells,
                                              reference_analysis=checks[cells])["streamJaccard"]
                for cells in check_cells}

    rows = []
    for radius in radii:
        protect = dilate(stream, radius)
        corrected, cost = apply_corrections(coarse, reference, protect, fine_quantum)
        result = hydrology.compare(reference, corrected, spacing_m, stream_cells,
                                   reference_analysis=analysis)
        rows.append({
            "bandRadiusCells": radius,
            "protectedCells": int(np.count_nonzero(protect)),
            "protectedFraction": float(np.count_nonzero(protect) / protect.size),
            "correctionBytes": cost["deployedBytes"],
            "bytesPerCorrectedCell": cost["bytesPerCorrectedCell"],
            "streamJaccard": result["streamJaccard"],
            "streamRecall": result["streamRecall"],
            "receiverAgreementFraction": result["receiverAgreementFraction"],
            "basinAgreementFraction": result["basinAgreementFraction"],
            "elevationMaxM": result["elevationMaxM"],
            "elevationMaeM": result["elevationMaeM"],
            "streamJaccardAtOtherThresholds": elsewhere(corrected),
            "encoding": cost,
        })
    uniform = []
    for target in uniform_targets:
        quantised = np.round(reference / (2.0 * target)) * (2.0 * target)
        uniform.append({"targetMaxErrorM": target,
                        "streamJaccard": hydrology.compare(reference, quantised, spacing_m, stream_cells,
                                                           reference_analysis=analysis)["streamJaccard"],
                        "streamJaccardAtOtherThresholds": elsewhere(quantised)})
    return {
        "schema": SCHEMA,
        "targetMaxErrorM": target_m, "quantizerStepM": step, "fineQuantumM": fine_quantum,
        "spacingM": spacing_m, "streamThresholdCells": stream_cells,
        "referenceStreamCells": int(np.count_nonzero(stream)),
        "baseline": {k: baseline[k] for k in
                     ("streamJaccard", "streamRecall", "receiverAgreementFraction",
                      "basinAgreementFraction", "elevationMaeM", "elevationMaxM")},
        "byBandRadius": rows,
        "baselineAtOtherThresholds": elsewhere(coarse),
        "uniformControls": uniform,
        "checkThresholdsCells": list(check_cells),
        "qualification": "Protected cells are chosen from the reference stream network, which the "
                         "encoder has; no decoder infers them. Corrections restore the fine-quantum "
                         "value, so the reconstruction is no longer uniformly bounded by the coarse "
                         "target: it is bounded by the coarse target off the protected set and by "
                         "the fine quantum on it. That is a different error contract, and a consumer "
                         "must be told which cells carry which bound.",
    }
