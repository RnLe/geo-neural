"""Representing geological context without claiming causality.

Each property below is enforced in code, not only documented.

A categorical identifier is not a number. GK100 unit codes are labels. Feeding
one to a network as a float asserts that unit 7 lies between 6 and 8 and is twice
unit 3.5, none of which means anything. Classes therefore reach a model only as
embedding lookups, and `class_features` refuses a non-integer code.

Unknown is a class, not a zero. An unmapped cell is not a cell of unit zero
and is not a cell of the most common unit. Code 0 is reserved for unknown, it has
its own embedding, and `unknown_mask` keeps it addressable so a downstream result
can be reported with and without it.

Unoriented strike is not an angle. A fault striking 30 degrees and one
striking 210 degrees are the same fault. Representing strike as sin and cos of
the angle would make those two maximally different; sin and cos of twice the
angle makes them identical, which is the standard axial representation and the
only one consistent with what the source measures.

Context is payload. Any raster consulted at decode time is deployed bytes and
is charged, because a better height loss bought with a larger archive is a net
loss. `context_bytes` counts it as an encoded payload, not a cell count.

Nothing here asserts that a mapped unit has a known erodibility, a known age in
years, or any physical parameter at all. A 1:100,000 interpretive map is not a
subsurface ground truth, and this module only represents what the source says.
"""
from __future__ import annotations
import gzip

import pathlib

import numpy as np

SCHEMA = "geoneural-geology-context-v1"
UNKNOWN_CLASS = 0


def strike_features(strike_degrees: np.ndarray) -> np.ndarray:
    """Axial encoding of an unoriented strike: sin(2t), cos(2t).

    A strike and its reverse describe one line. Doubling the angle before taking
    the sine and cosine maps them to the same point, which is what makes this an
    axial rather than a directional feature.
    """
    radians = np.deg2rad(np.asarray(strike_degrees, dtype=np.float64)) * 2.0
    return np.stack([np.sin(radians), np.cos(radians)], axis=-1)


def unknown_mask(classes: np.ndarray) -> np.ndarray:
    """Where the map says nothing. Never silently merged into a real class."""
    return np.asarray(classes) == UNKNOWN_CLASS


def class_features(classes: np.ndarray, class_count: int) -> np.ndarray:
    """Validate a class raster for embedding lookup.

    Refuses floats outright. A float class raster is the signature of somebody
    having interpolated, averaged or normalised categorical codes, all of which
    destroy the labels and none of which raise on their own.
    """
    array = np.asarray(classes)
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(
            "class codes must be integers; a float raster means the labels were interpolated, "
            "averaged or normalised, which is not a meaningful operation on categories")
    if array.min() < 0 or array.max() >= class_count:
        raise ValueError(f"class codes must lie in 0..{class_count - 1}, got "
                         f"{int(array.min())}..{int(array.max())}")
    return array.astype(np.int64)


def distance_to_boundary(classes: np.ndarray, spacing_m: float,
                         max_cells: int = 64) -> np.ndarray:
    """Distance in metres to the nearest cell of a different class.

    Chebyshev distance by iterative dilation, so it needs no scipy and matches
    the eight-neighbour convention the hydrology in this package already uses.
    Capped at `max_cells`, and the cap is returned rather than infinity so the
    field stays finite for a network. A saturated value means the boundary is
    farther than the cap, not that it does not exist.
    """
    array = np.asarray(classes)
    boundary = np.zeros(array.shape, dtype=bool)
    boundary[:, :-1] |= array[:, :-1] != array[:, 1:]
    boundary[:, 1:] |= array[:, :-1] != array[:, 1:]
    boundary[:-1, :] |= array[:-1, :] != array[1:, :]
    boundary[1:, :] |= array[:-1, :] != array[1:, :]
    distance = np.where(boundary, 0, max_cells).astype(np.int64)
    reached = boundary.copy()
    for step in range(1, max_cells + 1):
        padded = np.pad(reached, 1, mode="constant", constant_values=False)
        grown = (padded[1:-1, 1:-1] | padded[:-2, 1:-1] | padded[2:, 1:-1]
                 | padded[1:-1, :-2] | padded[1:-1, 2:]
                 | padded[:-2, :-2] | padded[:-2, 2:] | padded[2:, :-2] | padded[2:, 2:])
        new = grown & ~reached
        if not new.any():
            break
        distance[new] = step
        reached = grown
    return distance.astype(np.float64) * spacing_m


def context_bytes(classes: np.ndarray, extras: dict[str, np.ndarray] | None = None) -> dict:
    """What a context raster actually costs to ship.

    Classes are small integers and compress well; a derived float field does not,
    and is counted at the precision it would ship in rather than at float64.
    Derived fields are listed separately because a decoder can recompute them
    from the classes and need not carry them. That is a deployment choice with a
    byte cost, so both totals are reported.
    """
    codes = np.asarray(classes).astype(np.int32)
    packed = gzip.compress(codes.tobytes(order="C"), compresslevel=9, mtime=0)
    rows = {"classRasterBytes": len(packed), "cells": int(codes.size),
            "classes": int(codes.max()) + 1}
    derived = {}
    for name, field in (extras or {}).items():
        blob = gzip.compress(np.asarray(field, dtype=np.float16).tobytes(order="C"),
                             compresslevel=9, mtime=0)
        derived[name] = len(blob)
    rows["derivedFieldBytes"] = derived
    rows["totalIfAllShipped"] = rows["classRasterBytes"] + sum(derived.values())
    rows["totalIfDerivedRecomputed"] = rows["classRasterBytes"]
    rows["note"] = ("Derived fields are recomputable from the class raster, so shipping them is a "
                    "latency-for-bytes choice rather than a requirement. Both totals are given because "
                    "a conditioning gain that moves more bytes into context than it saves is not a gain.")
    return rows


def summary(classes: np.ndarray, spacing_m: float, strike_degrees: np.ndarray | None = None) -> dict:
    """A JSON-safe description of one context raster, including what is unknown."""
    array = np.asarray(classes)
    unknown = unknown_mask(array)
    distance = distance_to_boundary(array, spacing_m)
    extras = {"distanceToBoundary": distance}
    if strike_degrees is not None:
        features = strike_features(strike_degrees)
        extras["strikeSin2T"] = features[..., 0]
        extras["strikeCos2T"] = features[..., 1]
    return {
        "schema": SCHEMA,
        "cells": int(array.size),
        "classes": int(array.max()) + 1,
        "unknownCells": int(unknown.sum()),
        "unknownFraction": float(unknown.mean()),
        "meanDistanceToBoundaryM": float(distance.mean()),
        "bytes": context_bytes(array, extras),
        "qualification": "A representation of what a 1:100,000 interpretive map states, not rock "
                         "occupancy, not erodibility and not an age in years. Class 0 is unknown and is "
                         "kept addressable; it is never merged into a mapped unit. Strike is axial, so a "
                         "fault and its reverse are the same feature.",
    }


# --- GK100 GML to the atlas lattice ----------------------------------------
#
# What this produces is a raster of mapped surface geological units at
# 1:100,000, and that is not the same object as rock occupancy at a 10 m node.
# Three gaps separate them and every record written here states all three:
#
#  * Positional generalisation. A 1:100,000 line is drawn to roughly 0.5 mm on
#    the sheet, which is ~50 m on the ground and can be far more where a contact
#    is inferred. The lattice is 10 m. A unit boundary is therefore uncertain by
#    several cells, and a model that learns to predict height from the exact
#    boundary position is learning the cartography, not the geology.
#  * Surface units, not volumes. The map says what outcrops or lies beneath
#    Quaternary cover, with no depth extent.
#  * Attribute, not colour. `material_label` and `OLDERNAMEDAGE` are INSPIRE
#    codelist values carried in the GML. Cartographic colours are never
#    rasterised as lithology; nothing here reads a colour.

GEOLOGY_SCHEMA = "geoneural-geology-raster-v1"
UNKNOWN_CLASS = 0


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _rings(polygon, kind: str) -> list[list[tuple[float, float]]]:
    out = []
    for side in polygon:
        if _local(side.tag) != kind:
            continue
        for ring in side.iter():
            if _local(ring.tag) != "posList":
                continue
            values = [float(v) for v in (ring.text or "").split()]
            if len(values) < 8 or len(values) % 2:
                continue
            out.append(list(zip(values[0::2], values[1::2])))
    return out


def parse_units(paths) -> list[dict]:
    """Read GK100 GeologicUnit features and their geometry, attributes intact.

    Coordinates are read easting-first and checked against the ETRS89 / UTM32N
    magnitudes rather than trusted from the srsName: the GML declares
    `urn:ogc:def:crs:EPSG::25832`, whose formal axis order is northing-first,
    while the file is written easting-first. Guessing wrong transposes the map
    onto the terrain and would corrupt every ablation downstream silently, so it
    is asserted from the numbers.
    """
    import xml.etree.ElementTree as ET
    units = []
    for path in sorted(pathlib.Path(p) for p in paths):
        root = ET.parse(path).getroot()
        for feature in root.iter():
            if _local(feature.tag) != "GE.GeologicUnit":
                continue
            attributes = {_local(child.tag): (child.text or "").strip()
                          for child in feature if child.text and child.text.strip()}
            polygons = []
            for polygon in feature.iter():
                if _local(polygon.tag) != "Polygon":
                    continue
                exterior = _rings(polygon, "exterior")
                interior = _rings(polygon, "interior")
                if exterior:
                    polygons.append([exterior[0]] + interior)
            if not polygons:
                continue
            sample = polygons[0][0][0]
            if not (100_000.0 < sample[0] < 1_000_000.0 and 5_000_000.0 < sample[1] < 6_500_000.0):
                raise ValueError(
                    f"{path.name}: first coordinate {sample} is not ETRS89/UTM32N easting-northing; "
                    "the axis order assumption does not hold for this file")
            units.append({
                "material": attributes.get("material_label") or attributes.get("MATERIAL") or "",
                "age": attributes.get("OLDERNAMEDAGE", ""),
                "name": attributes.get("featurename", ""),
                "localId": attributes.get("id_localid", ""),
                "geometry": {"type": "MultiPolygon",
                             "coordinates": [[list(ring) for ring in polygon]
                                             for polygon in polygons]}})
    return units


def parse_faults(paths) -> list[dict]:
    """Read GK100 `GE.GeologicFault` traces: map-view polylines and nothing more.

    Each feature's geometry is a `MultiCurve` of `LineString` traces (the
    fault's intersection with the mapped surface), with a `faulttype_label`.
    The acquired GK100 response carries 34 such features, all with
    `geologichistory_void=2`.

    Faults are parsed separately from units because a trace supports less. It
    gives a map-view distance and an unoriented strike, both usable as context
    channels. It does not supply dip, throw, displacement, depth extent or slip
    history: those attributes are absent from the source, not merely unparsed.
    So a fault here can condition a surface decoder but cannot support a
    structural or kinematic claim; a model branch named "tectonics" is not
    evidence of learned tectonics.

    Axis order is asserted from coordinate magnitudes for the same reason
    `parse_units` asserts it.
    """
    import xml.etree.ElementTree as ET
    faults = []
    for path in sorted(pathlib.Path(p) for p in paths):
        root = ET.parse(path).getroot()
        for feature in root.iter():
            if _local(feature.tag) != "GE.GeologicFault":
                continue
            attributes = {_local(child.tag): (child.text or "").strip()
                          for child in feature if child.text and child.text.strip()}
            lines = []
            for element in feature.iter():
                if _local(element.tag) != "posList":
                    continue
                values = [float(v) for v in (element.text or "").split()]
                if len(values) < 4:
                    continue
                lines.append(list(zip(values[0::2], values[1::2])))
            if not lines:
                continue
            sample = lines[0][0]
            if not (100_000.0 < sample[0] < 1_000_000.0 and 5_000_000.0 < sample[1] < 6_500_000.0):
                raise ValueError(
                    f"{path.name}: fault coordinate {sample} is not ETRS89/UTM32N "
                    "easting-northing; the axis order assumption does not hold")
            faults.append({
                "faultType": attributes.get("faulttype_label", ""),
                "localId": attributes.get("id_localid", ""),
                "historyVoid": attributes.get("geologichistory_void", "") == "2",
                "geometry": {"type": "MultiLineString",
                             "coordinates": [[list(point) for point in line] for line in lines]}})
    return faults


def fault_rasters(faults: list[dict], bbox, side: int, spacing_m: float,
                  max_cells: int = 64) -> dict:
    """Distance-to-nearest-trace and axial strike, on the atlas lattice.

    Two channels, for the two things a map-view trace actually says:

    * `distanceM`: Chebyshev distance to the nearest burnt trace cell, by the
      same iterative dilation `distance_to_boundary` uses, so the two context
      distances are the same measurement. Saturates at `max_cells`, and the
      saturation value is reported rather than left to be inferred.
    * `strikeSinCos`: the local trace direction as sin(2t), cos(2t) carried
      outward from each trace cell to the cells nearest it. Axial, because a
      fault striking 30 degrees and one striking 210 degrees are the same fault;
      see `strike_features`.

    Far from every trace the strike channel is meaningless, so it is returned
    multiplied by a decay in distance and accompanied by `strikeValidMask`. A
    model handed a strike for a cell 3 km from any fault would be fitting the
    decay, not the structure.
    """
    from rasterio.features import rasterize as _rasterize
    from rasterio.transform import from_origin
    west, south, east, north = (float(v) for v in bbox)
    transform = from_origin(west - spacing_m / 2.0, north + spacing_m / 2.0, spacing_m, spacing_m)
    shapes = [(fault["geometry"], index + 1) for index, fault in enumerate(faults)]
    if shapes:
        burnt = _rasterize(shapes, out_shape=(side, side), transform=transform,
                           fill=0, dtype="int32", all_touched=True)
    else:
        burnt = np.zeros((side, side), dtype=np.int32)
    on_trace = burnt > 0

    # Segment azimuths, burnt the same way so strike and distance register.
    angle = np.zeros((side, side), dtype=np.float64)
    for index, fault in enumerate(faults):
        for line in fault["geometry"]["coordinates"]:
            for (x0, y0), (x1, y1) in zip(line[:-1], line[1:]):
                if x0 == x1 and y0 == y1:
                    continue
                segment = {"type": "LineString", "coordinates": [[x0, y0], [x1, y1]]}
                mask = _rasterize([(segment, 1)], out_shape=(side, side), transform=transform,
                                  fill=0, dtype="uint8", all_touched=True).astype(bool)
                angle[mask] = np.arctan2(y1 - y0, x1 - x0)

    reached = on_trace.copy()
    distance = np.where(on_trace, 0.0, np.inf)
    nearest = angle.copy()
    for step in range(1, int(max_cells) + 1):
        grown = reached.copy()
        for shift, axis in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
            grown |= np.roll(reached, shift, axis=axis)
        # Edges must not wrap: a fault at the north edge is not near the south one.
        if True:
            grown[0, :] |= reached[0, :]
            grown[-1, :] |= reached[-1, :]
        fresh = grown & ~reached
        if not fresh.any():
            break
        distance[fresh] = step * float(spacing_m)
        for shift, axis in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
            donor = np.roll(reached, shift, axis=axis) & fresh
            nearest[donor] = np.roll(nearest, shift, axis=axis)[donor]
        reached = grown
    saturation = float(max_cells) * float(spacing_m)
    distance[~np.isfinite(distance)] = saturation
    decay = np.exp(-distance / max(saturation / 3.0, 1e-9))
    strike = np.stack([np.sin(2.0 * nearest), np.cos(2.0 * nearest)], axis=-1) * decay[..., None]
    return {"schema": GEOLOGY_SCHEMA, "faultsRead": len(faults),
            "traceCells": int(on_trace.sum()),
            "distanceM": distance, "strikeSinCos": strike,
            "strikeValidMask": reached,
            "saturationM": saturation,
            "reachedFraction": float(reached.mean()),
            "qualification": (
                "Map-view fault traces at 1:100,000, rasterised to a 10 m lattice. Distance "
                "saturates at saturationM and strike decays with distance; outside "
                "strikeValidMask neither channel carries information. The source supplies no "
                "dip, throw, displacement, depth extent or slip history, so these channels can "
                "condition a surface decoder and cannot support any structural or kinematic "
                "claim."),
            "omits": ["dip and dip direction", "throw and displacement",
                      "depth extent", "slip history and event ordering",
                      "which side is the hanging wall"]}


def rasterise(units: list[dict], bbox, side: int, spacing_m: float,
              attribute: str = "material") -> dict:
    """Burn the mapped units onto the atlas lattice, node-centred.

    The transform puts pixel centres on lattice nodes, because the terrain
    array is a grid of samples rather than a grid of areas. Offsetting by half a
    cell would shift the geology half a cell against the heights it conditions,
    a silent misregistration that would blur the comparison with the ablation
    controls.

    Cells no polygon covers keep `UNKNOWN_CLASS`. They are not a class; they are
    absence of mapping, and `unknown_mask` exists so a decoder can be told so.
    """
    from rasterio.features import rasterize as _rasterize
    from rasterio.transform import from_origin
    west, south, east, north = (float(v) for v in bbox)
    transform = from_origin(west - spacing_m / 2.0, north + spacing_m / 2.0, spacing_m, spacing_m)
    values = sorted({(unit.get(attribute) or "").strip() for unit in units} - {""})
    legend = {name: index + 1 for index, name in enumerate(values)}
    shapes = [(unit["geometry"], legend[(unit.get(attribute) or "").strip()])
              for unit in units if (unit.get(attribute) or "").strip()]
    classes = _rasterize(shapes, out_shape=(side, side), transform=transform,
                         fill=UNKNOWN_CLASS, dtype="int32", all_touched=False)
    unknown = int((classes == UNKNOWN_CLASS).sum())
    return {"schema": GEOLOGY_SCHEMA, "attribute": attribute, "classes": classes,
            "legend": legend, "classCount": len(legend) + 1,
            "unknownCells": unknown, "unknownFraction": unknown / float(classes.size),
            "unitsRasterised": len(shapes), "unitsRead": len(units),
            "bbox": [west, south, east, north], "side": int(side), "spacingM": float(spacing_m),
            "registration": "pixel centres on lattice nodes",
            "sourceScale": "1:100,000",
            "positionalUncertaintyM": 50.0,
            "qualification": (
                "Mapped surface geological units at 1:100,000, rasterised to a 10 m lattice. "
                "Boundary positions are uncertain by roughly 50 m (several cells), so a model "
                "that sharpens a contact is fitting the cartography. These are surface units with "
                "no depth extent, not rock occupancy. Class codes come from INSPIRE attributes "
                "(material_label, OLDERNAMEDAGE); no cartographic colour is read."),
            "omits": ["fault displacement, dip and depth extent (traces themselves are "
                      "parsed by parse_faults)", "depth extent and dip",
                      "Quaternary cover thickness", "per-unit mapping confidence"]}


def misalign(classes: np.ndarray, shift: tuple[int, int]) -> np.ndarray:
    """Roll the raster. The strongest of the geology ablation controls.

    Everything about the map survives (unit shapes, sizes, adjacency, class
    frequencies, compressed size). Only the correspondence with the terrain is
    broken. A model that improves on real context but not on this has used
    geology; a model that improves on both has used the statistics of a blocky
    categorical field, which any such field would provide.

    This is a better control than shuffling cells, which destroys the spatial
    structure and is therefore easy to beat for reasons that say nothing about
    geology.
    """
    return np.roll(np.asarray(classes), shift=shift, axis=(0, 1))


def shuffle_cells(classes: np.ndarray, seed: int = 0) -> np.ndarray:
    """Permute cells, destroying spatial structure but keeping class frequencies.

    The weaker control, retained because the two fail differently: a decoder that
    still gains here is reading per-node class identity alone, with no use for
    the field's geometry at all.
    """
    flat = np.asarray(classes).reshape(-1).copy()
    np.random.default_rng(seed).shuffle(flat)
    return flat.reshape(np.asarray(classes).shape)


def generic_context(side: int, regions: int, class_count: int, seed: int = 0) -> np.ndarray:
    """A blocky categorical field of the same capacity carrying no geology.

    Nearest-seed-point regions: same lattice, same number of classes, comparable
    region sizes and comparable compressed size, and no relation to anything.
    This is the ablation's equal-capacity generic arm. A decoder given any extra
    per-node input gains some freedom, and that gain must be subtracted before a
    geological claim is made.
    """
    rng = np.random.default_rng(seed)
    points = rng.integers(0, side, size=(regions, 2))
    labels = rng.integers(1, max(class_count, 2), size=regions)
    rows = np.arange(side)[:, None, None]
    columns = np.arange(side)[None, :, None]
    distance = (rows - points[None, None, :, 0]) ** 2 + (columns - points[None, None, :, 1]) ** 2
    return labels[np.argmin(distance, axis=2)].astype(np.int32)


def ablation_contexts(classes: np.ndarray, class_count: int, regions: int,
                      seed: int = 0) -> dict:
    """The geology ablation arms, with their byte costs attached.

    `none` is a single-class raster rather than an absent one so that every arm
    runs the identical code path and differs only in what the context says. An
    arm that changed the architecture as well as the information would confound
    the two.

    The arms are not all byte-matched. On the Essen raster: real 32,489 B,
    misaligned 32,622 B (+0.4%), generic 33,258 B (+2.4%), shuffled 419,095 B
    (thirteen times real), because permuting cells destroys the spatial
    structure that made the field compressible. So `shuffled` is a diagnostic
    for whether a decoder reads per-node class identity alone; it is not an
    equal-capacity control, and a rate comparison against it is meaningless. A
    conclusion about geology should rest on `misaligned`.
    """
    classes = np.asarray(classes)
    side = classes.shape[0]
    arms = {
        "none": np.zeros_like(classes),
        "real": classes,
        "misaligned": misalign(classes, (side // 3, side // 4)),
        "shuffled": shuffle_cells(classes, seed),
        "generic": generic_context(side, regions, class_count, seed),
    }
    return {name: {"classes": field, "bytes": context_bytes(field)["classRasterBytes"],
                   "distinctClasses": int(np.unique(field).size)}
            for name, field in arms.items()}
