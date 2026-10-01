"""Regions, references, geology and terrain strata for the reconstruction track.

Every array here is read from `geoneural.common.HOME`. The 10 m references are 1025^2 float32 DTMs; the 1 m
references are 10241^2 float32 and only ever memory-mapped. Geology is GK100 surface material rasterised onto
the 10 m lattice with one class dictionary shared by all regions, built from the material labels; code 0 is
unknown. A region whose geology could not be read is all-unknown rather than missing, so a model with a geology
input runs everywhere and the record says where the input carried nothing.
"""
from __future__ import annotations

import functools

import numpy as np

from geoneural.common import HOME, read_json

REGIONS = ("essen-ruhr", "muensterland-plain", "lower-rhine", "teutoburg-forest", "bergisches-land",
           "rothaar-sauerland")
FINE_REGIONS = ("essen-ruhr", "muensterland-plain", "rothaar-sauerland")
#: Regions whose geology is treated as unavailable, with the reason (none of the development regions).
GEOLOGY_UNAVAILABLE: dict[str, str] = {}
SPACING_M = 10.0
#: Slope classes (percent) used to read calibration by terrain regime.
SLOPE_CLASSES_PERCENT = (0.0, 2.0, 5.0, 15.0, np.inf)


def reference(region: str) -> np.ndarray:
    return np.load(HOME / "atlases" / region / "reference.npy").astype(np.float64)


def fine(region: str) -> np.ndarray:
    """The 1 m reference, memory-mapped (10241^2 float32, about 420 MB)."""
    return np.load(HOME / "fine" / region / "reference_1m.npy", mmap_mode="r")


def bounds(region: str) -> list[float]:
    return [float(v) for v in read_json(HOME / "atlases" / region / "atlas.json")["bounds"]]


@functools.lru_cache(maxsize=16)
def _units(region: str) -> tuple:
    from geoneural.data import geology
    if region in GEOLOGY_UNAVAILABLE:
        return ()
    files = sorted((HOME / "geology" / region).glob("geology-00-*.gml"))
    if not files:
        return ()
    return tuple(geology.parse_units(files))


def material_labels(region: str) -> set[str]:
    return {(u.get("material") or "").strip() for u in _units(region)} - {""}


def class_dictionary(regions=REGIONS) -> dict[str, int]:
    """One dictionary for every region: sorted material labels, codes from 1, 0 reserved for unknown."""
    labels = sorted(set().union(*(material_labels(r) for r in regions)))
    return {label: index + 1 for index, label in enumerate(labels)}


def geology(region: str, dictionary: dict[str, int], side: int = 1025) -> np.ndarray:
    """Shared-code class raster on the 10 m lattice (int64, 0 = unknown or unavailable)."""
    from geoneural.data import geology as gk
    units = _units(region)
    if not units:
        return np.zeros((side, side), dtype=np.int64)
    raster = gk.rasterise(list(units), bounds(region), side, SPACING_M, attribute="material")
    lookup = np.zeros(max(raster["legend"].values()) + 1, dtype=np.int64)
    for label, code in raster["legend"].items():
        lookup[code] = dictionary.get(label, 0)
    return lookup[raster["classes"]]


def fold_classes(train_regions, dictionary: dict[str, int], min_fraction: float = 0.005) -> np.ndarray:
    """Map shared codes to model inputs using the training regions only.

    A class covering less than `min_fraction` of the training cells has no weights worth the name, so it is
    folded into unknown; a class that only the test region has is unknown to the model by construction.
    Returns an int array mapping shared code to input slot (slot 0 is unknown).
    """
    counts = np.zeros(len(dictionary) + 1)
    total = 0
    for region in train_regions:
        codes = geology(region, dictionary)
        counts += np.bincount(codes.ravel(), minlength=counts.size)
        total += codes.size
    keep = [code for code in range(1, counts.size) if counts[code] >= min_fraction * total]
    mapping = np.zeros(counts.size, dtype=np.int64)
    for slot, code in enumerate(keep, start=1):
        mapping[code] = slot
    return mapping


def slope_percent(z: np.ndarray, spacing_m: float = SPACING_M) -> np.ndarray:
    gy, gx = np.gradient(np.asarray(z, dtype=np.float64), spacing_m)
    return 100.0 * np.hypot(gx, gy)


def slope_class(z: np.ndarray, spacing_m: float = SPACING_M) -> np.ndarray:
    return np.digitize(slope_percent(z, spacing_m), SLOPE_CLASSES_PERCENT[1:-1])


def folds(regions=REGIONS, test_regions=None) -> list[dict]:
    """Leave-one-region-out: each region is tested once, the next one in the list validates, the rest train.

    With explicit `test_regions` (a confirmation cohort) there is one fold: every listed development region
    trains, nothing validates (the recipe is frozen), and the test regions are only scored.
    """
    regions = list(regions)
    if test_regions:
        return [{"test": list(test_regions), "validation": [], "train": regions}]
    out = []
    for k, region in enumerate(regions):
        validation = regions[(k + 1) % len(regions)]
        out.append({"test": [region], "validation": [validation],
                    "train": [r for r in regions if r not in (region, validation)]})
    return out
