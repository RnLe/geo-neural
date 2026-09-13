"""WCS 2.0.1 schema negotiation against the provider's own metadata.

A coverage id of `nw_dgm`, subset axis labels `x` and `y`, and the SCALEFACTOR
scaling parameter are common but not guaranteed: a service may namespace-qualify
its coverage id, label its axes `E`/`N` or `Lon`/`Lat` following the CRS axis
order, and advertise SCALEAXESBYFACTOR instead of SCALEFACTOR. A request built
on the wrong assumption returns a provider exception, not terrain, and the
correct response is to read the metadata rather than retry a malformed request.

Everything here parses XML that has already been downloaded and receipted. The
negotiated choice is recorded in `input.json` so a later reader can tell which
request form actually produced the bytes.
"""
from __future__ import annotations
import xml.etree.ElementTree as ET
from pathlib import Path

#: Axis labels seen in the wild for a projected easting/northing pair, lowercased.
EASTING_LABELS = ("x", "e", "easting", "lon", "long", "longitude")
NORTHING_LABELS = ("y", "n", "northing", "lat", "latitude")


def local(tag: str) -> str:
    """Strip the XML namespace from a tag."""
    return tag.split("}")[-1]


def coverage_ids(capabilities: ET.Element) -> list[str]:
    return [v.text for v in capabilities.iter() if local(v.tag) == "CoverageId" and v.text]


def resolve_coverage(available: list[str], wanted: str) -> str:
    """Match a wanted coverage id against what the service actually advertises.

    An exact match wins. Otherwise a service that qualifies its ids (for example
    `nw_dgm` published as `geobasis__nw_dgm` or `ns:nw_dgm`) is matched on the
    final segment. Anything ambiguous raises rather than guessing.
    """
    if wanted in available:
        return wanted
    def tail(name: str) -> str:
        return name.rsplit(":", 1)[-1].rsplit("__", 1)[-1]
    candidates = [name for name in available if tail(name) == wanted]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(f"Coverage {wanted!r} is ambiguous among {candidates}; name it explicitly")
    raise ValueError(
        f"Coverage {wanted!r} is absent from this service. Inspect the retained capabilities; "
        f"available: {available[:40]}"
    )


def describe_axes(description: ET.Element) -> dict:
    """Read axis labels, CRS and grid geometry from DescribeCoverage.

    Returns what the document states. Missing elements are reported as None rather
    than filled with a plausible default.
    """
    labels: list[str] = []
    srs: str | None = None
    origin: list[float] | None = None
    offsets: list[list[float]] = []
    low: list[int] | None = None
    high: list[int] | None = None
    for node in description.iter():
        name = local(node.tag)
        if name in ("Envelope", "EnvelopeWithTimePeriod") and not labels:
            raw = node.attrib.get("axisLabels")
            if raw:
                labels = raw.split()
            srs = srs or node.attrib.get("srsName")
        elif name == "GridEnvelope":
            for child in node:
                if local(child.tag) == "low" and child.text:
                    low = [int(v) for v in child.text.split()]
                elif local(child.tag) == "high" and child.text:
                    high = [int(v) for v in child.text.split()]
        elif name == "origin" or name == "Point":
            for child in node.iter():
                if local(child.tag) == "pos" and child.text:
                    origin = [float(v) for v in child.text.split()]
                    break
        elif name == "offsetVector" and node.text:
            offsets.append([float(v) for v in node.text.split()])
    size = None
    if low is not None and high is not None and len(low) == len(high):
        size = [h - l + 1 for l, h in zip(low, high)]
    return {"axis_labels": labels or None, "srs_name": srs, "origin": origin,
            "offset_vectors": offsets or None, "grid_size": size}


def subset_axes(axis_labels: list[str] | None) -> tuple[str, str]:
    """Choose the easting and northing subset labels the service expects.

    Falls back to `x`/`y` only when the description carries no axis labels at
    all.
    """
    if not axis_labels:
        return "x", "y"
    if len(axis_labels) < 2:
        raise ValueError(f"Coverage declares too few axes to subset: {axis_labels}")
    lowered = [label.lower() for label in axis_labels[:2]]
    easting = next((axis_labels[i] for i, label in enumerate(lowered) if label in EASTING_LABELS), None)
    northing = next((axis_labels[i] for i, label in enumerate(lowered) if label in NORTHING_LABELS), None)
    if easting is None or northing is None or easting == northing:
        raise ValueError(
            f"Cannot identify an easting/northing pair in axis labels {axis_labels}. "
            "Inspect the retained DescribeCoverage before requesting terrain."
        )
    return easting, northing


def scaling_forms(easting: str, northing: str, factor: float) -> list[list[tuple[str, str]]]:
    """Candidate scaling parameters, most widely supported first.

    WCS 2.0 puts scaling in an extension, and services differ in which spelling
    they accept. These are tried in order and the one that returns imagery is
    recorded; none is assumed to work.
    """
    return [
        [("SCALEFACTOR", repr(factor))],
        [("SCALEAXESBYFACTOR", f"{easting}({factor}),{northing}({factor})")],
        [],
    ]


def scaling_label(form: list[tuple[str, str]]) -> str:
    return form[0][0] if form else "none (native resolution)"


def is_exception(path: Path) -> str | None:
    """Return the provider's exception text if this response is an error document."""
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return None
    if "Exception" in local(root.tag) or "Exception" in root.tag:
        return " ".join(root.itertext())[:1200]
    return None
