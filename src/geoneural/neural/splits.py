"""Geographic holdouts for the neural codec, because random pixels are not a holdout.

Pixel-wise random holdouts in one spatially correlated tile do not measure
geographic generalisation. A terrain field is strongly autocorrelated at 10 m
spacing, so a pixel withheld from training sits between eight neighbours that
were not. Predicting it is interpolation between known values, and a model can
score well on it while having learned nothing that transfers.

Two different questions need two different splits, and they are reported apart
because a model can pass one and fail the other:

Interpolation withholds whole pages in a checkerboard. Each held-out page is
surrounded by trained pages, so this asks whether the model fills a gap whose
boundary it has seen. It is the best case, and the relevant one for a codec
that will only ever be asked about the region it encoded.

Extrapolation withholds one contiguous corner block. Nothing inside it was
seen and most of it is far from anything that was. This asks whether the model
learned terrain structure rather than this terrain, which is the question that
matters before any claim that the approach transfers to unseen geography.

Splits are page-aligned so that a held-out region is a whole number of atlas
pages. That keeps the evaluation commensurate with what the codec ships and
stops a page from being half-trained.

A third question appears as soon as anything is tuned: model size and frequency
bands should be chosen with held-out selection criteria. A search that picks its
architecture on the same pages it then reports is not reporting a held-out
number: it has fitted the search to that region, and with enough trials it will
find the configuration that happens to suit it. So `selection_split` halves the
interpolation pages into a selection half, which a search may look at as often
as it likes, and a test half, which is read once after everything is decided.

The two halves are adjacent geography and are therefore not independent: a
configuration chosen on one is expected to score similarly on the other. That is
the intended strength. The test half exists to catch a search that has fitted
the noise of a particular page set, not to prove transfer to different terrain.
The extrapolation block is the only split that bears on transfer, and nothing is
ever selected on it.
"""
from __future__ import annotations

import numpy as np

SCHEMA = "geoneural-split-v1"


def _page_grid(side: int, intervals: int) -> int:
    if (side - 1) % intervals:
        raise ValueError(f"lattice side {side} is not a whole number of {intervals}-interval pages")
    return (side - 1) // intervals


def page_mask_to_samples(page_mask: np.ndarray, side: int, intervals: int) -> np.ndarray:
    """Expand a per-page flag to per-node, over the interior of each page.

    Page boundaries are shared between neighbours, so a boundary node belongs to
    two pages and cannot be cleanly assigned. Those nodes are left out of both
    sets rather than leaked into training, which is why the reported fractions
    do not sum to one.
    """
    samples = np.zeros((side, side), dtype=bool)
    for (py, px) in zip(*np.nonzero(page_mask)):
        r0, c0 = py * intervals + 1, px * intervals + 1
        samples[r0:r0 + intervals - 1, c0:c0 + intervals - 1] = True
    return samples


def checkerboard_pages(side: int, intervals: int, parity: int = 1) -> np.ndarray:
    """Held-out pages surrounded by trained ones: the interpolation split."""
    pages = _page_grid(side, intervals)
    ys, xs = np.mgrid[0:pages, 0:pages]
    return ((ys + xs) % 2) == parity


def corner_block_pages(side: int, intervals: int, fraction: float = 0.25) -> np.ndarray:
    """One contiguous unseen region: the extrapolation split.

    The block is square and anchored at the south-east corner, so most of its
    area is far from any trained page rather than hugging the boundary.
    """
    pages = _page_grid(side, intervals)
    if not 0.0 < fraction < 1.0:
        raise ValueError("holdout fraction must lie strictly between 0 and 1")
    span = max(1, int(round(pages * np.sqrt(fraction))))
    if span >= pages:
        raise ValueError("holdout block would consume the whole domain")
    mask = np.zeros((pages, pages), dtype=bool)
    mask[pages - span:, pages - span:] = True
    return mask


def split_interpolation(page_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Halve the held-out checkerboard into selection and test pages.

    The rule is page-row parity, which is deterministic, needs no seed, and
    leaves both halves spread over the whole domain rather than banding them. A
    search tuned on the north half and tested on the south would be measuring a
    geographic gradient instead of selection overfit.
    """
    rows = np.arange(page_mask.shape[0])[:, None]
    selection = page_mask & (rows % 2 == 0)
    test = page_mask & (rows % 2 == 1)
    return selection, test


def build(side: int, intervals: int, extrapolation_fraction: float = 0.25,
          selection_split: bool = False) -> dict:
    """Train / interpolate / extrapolate node masks that never overlap.

    The extrapolation block is removed from training first and takes precedence
    over the checkerboard, so no node is counted as both.

    `extrapolation_fraction = 0` is the codec-fit control: train on every page
    and withhold nothing. A codec's job is to reproduce the data it encoded, so
    its accuracy on the encoded region is the right measure of it; a
    generalization question needs a different experiment. Both are run and
    labelled apart so that one number is not read as answering the other
    question.
    """
    pages = _page_grid(side, intervals)
    if extrapolation_fraction == 0:
        everything = np.ones((pages, pages), dtype=bool)
        train = page_mask_to_samples(everything, side, intervals)
        empty = np.zeros((side, side), dtype=bool)
        total = side * side
        return {
            "schema": SCHEMA, "side": side, "pageIntervals": intervals,
            "mode": "codec-fit control: no holdout",
            "pages": int(everything.size), "trainPages": int(everything.sum()),
            "interpolationPages": 0, "extrapolationPages": 0,
            "trainMask": train, "interpolationMask": empty, "extrapolationMask": empty,
            "trainFraction": float(train.sum() / total),
            "interpolationFraction": 0.0, "extrapolationFraction": 0.0,
            "unassignedFraction": float(1.0 - train.sum() / total),
            "note": "Every page is trained on. Reported error is codec fit on the encoded region and "
                    "is NOT evidence of generalization to anything, including the withheld pages of "
                    "the split runs.",
            "interpretation": "Use this to ask whether the model is a good codec of what it encoded. "
                              "Use the holdout split to ask whether it learned terrain. They are "
                              "different questions and one cannot answer the other.",
        }
    extrapolation_pages = corner_block_pages(side, intervals, extrapolation_fraction)
    interpolation_pages = checkerboard_pages(side, intervals) & ~extrapolation_pages
    train_pages = ~(extrapolation_pages | interpolation_pages)

    train = page_mask_to_samples(train_pages, side, intervals)
    interpolation = page_mask_to_samples(interpolation_pages, side, intervals)
    extrapolation = page_mask_to_samples(extrapolation_pages, side, intervals)
    selection_pages, test_pages = split_interpolation(interpolation_pages)
    if (train & interpolation).any() or (train & extrapolation).any() \
            or (interpolation & extrapolation).any():
        raise AssertionError("split masks overlap; a held-out node would be trained on")

    total = side * side
    return {
        "schema": SCHEMA,
        "side": side, "pageIntervals": intervals,
        "pages": int(train_pages.size),
        "trainPages": int(train_pages.sum()),
        "interpolationPages": int(interpolation_pages.sum()),
        "extrapolationPages": int(extrapolation_pages.sum()),
        "selectionPages": int(selection_pages.sum()),
        "testPages": int(test_pages.sum()),
        "trainMask": train, "interpolationMask": interpolation, "extrapolationMask": extrapolation,
        "selectionMask": page_mask_to_samples(selection_pages, side, intervals) if selection_split
                         else np.zeros((side, side), dtype=bool),
        "testMask": page_mask_to_samples(test_pages, side, intervals) if selection_split
                    else np.zeros((side, side), dtype=bool),
        "selectionSplit": bool(selection_split),
        "trainFraction": float(train.sum() / total),
        "interpolationFraction": float(interpolation.sum() / total),
        "extrapolationFraction": float(extrapolation.sum() / total),
        "unassignedFraction": float(1.0 - (train.sum() + interpolation.sum() + extrapolation.sum()) / total),
        "note": "Page-boundary nodes belong to two pages and are assigned to neither set, so the "
                "fractions do not sum to one. They are excluded from training rather than leaked.",
        "interpretation": "Interpolation is the best case for a codec of an encoded region. "
                          "Extrapolation is the only one of the two that bears on transfer to unseen "
                          "geography, and neither is evidence about a different landscape.",
        "selectionNote": "selectionMask and testMask halve interpolationMask by page-row parity and are "
                         "empty unless selection_split is set. A search may read the selection half "
                         "freely; the test half is read once, after every choice is fixed. They are "
                         "adjacent geography and are not independent regions: the test half catches a "
                         "search fitted to one page set, and says nothing about different terrain.",
    }


def summary(split: dict) -> dict:
    """The JSON-safe part, for a checkpoint or report."""
    return {k: v for k, v in split.items() if not isinstance(v, np.ndarray)}
