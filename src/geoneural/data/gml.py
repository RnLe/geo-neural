"""Bounded INSPIRE GML inventory without a compiled vector dependency.

The GK100 layers, attributes, references and coverage are inventoried before any
categorical conditioning is chosen. That is a schema question, not a geometry
question, so it is answered by reading the retained GML directly rather than by
binding a GDAL vector driver.

This matters beyond convenience. INSPIRE feature types carry linked properties
expressed as `xlink:href` references to external vocabularies; a conventional
vector reader presents those as empty or opaque fields, which is precisely how a
missing property gets mistaken for a rock class. Here they are counted and
surfaced as references, so a curator can see what still has to be resolved.

No attribute is nominated as lithology. Selecting one is a separate curation
decision, made against this inventory.
"""
from __future__ import annotations
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

XLINK = "{http://www.w3.org/1999/xlink}href"
GML_GEOMETRIES = ("Polygon", "MultiSurface", "Surface", "LineString", "MultiCurve", "Curve", "Point", "MultiPoint")
MAX_GML_BYTES = 256 * 1024 * 1024
MAX_FEATURES = 100_000
MAX_SAMPLE_VALUES = 12


def local(tag: str) -> str:
    return tag.split("}")[-1]


def _members(root: ET.Element) -> list[ET.Element]:
    return [child for child in root if local(child.tag) in ("member", "featureMember")]


def inventory_file(path: Path, max_features: int = MAX_FEATURES) -> dict:
    """Describe one GML response: its feature type, fields, references and geometry."""
    if path.stat().st_size > MAX_GML_BYTES:
        raise ValueError(f"GML exceeds {MAX_GML_BYTES} bytes: {path}")
    root = ET.parse(path).getroot()
    if "Exception" in local(root.tag):
        raise ValueError("Provider exception document: " + " ".join(root.itertext())[:600])

    feature_types: Counter[str] = Counter()
    fields: Counter[str] = Counter()
    references: Counter[str] = Counter()
    geometries: Counter[str] = Counter()
    srs_names: Counter[str] = Counter()
    samples: dict[str, list[str]] = {}
    identifiers = 0
    scanned = 0

    for member in _members(root)[:max_features]:
        feature = next(iter(member), None)
        if feature is None:
            continue
        scanned += 1
        feature_types[local(feature.tag)] += 1
        for node in feature.iter():
            name = local(node.tag)
            if name in GML_GEOMETRIES:
                geometries[name] += 1
            srs = node.attrib.get("srsName")
            if srs:
                srs_names[srs] += 1
        for child in feature:
            name = local(child.tag)
            if name in GML_GEOMETRIES or any(local(g.tag) in GML_GEOMETRIES for g in child):
                continue
            fields[name] += 1
            href = child.attrib.get(XLINK)
            if href:
                references[name] += 1
                bucket = samples.setdefault(name, [])
                if href not in bucket and len(bucket) < MAX_SAMPLE_VALUES:
                    bucket.append(href)
                continue
            text = (child.text or "").strip()
            if not text:
                # A nested element may still carry the value (INSPIRE often wraps it).
                nested = [(t.text or "").strip() for t in child.iter() if (t.text or "").strip()]
                text = nested[0] if nested else ""
            if text:
                bucket = samples.setdefault(name, [])
                if text not in bucket and len(bucket) < MAX_SAMPLE_VALUES:
                    bucket.append(text[:200])
            if name in ("identifier", "inspireId", "localId"):
                identifiers += 1

    return {
        "file": path.name,
        "features_scanned": scanned,
        "number_returned": root.attrib.get("numberReturned"),
        "number_matched": root.attrib.get("numberMatched"),
        "feature_types": dict(feature_types),
        "fields": dict(fields.most_common()),
        "linked_reference_fields": dict(references.most_common()),
        "geometry_types": dict(geometries),
        "srs_names": dict(srs_names),
        "sample_values": samples,
        "identifier_elements": identifiers,
    }


def inventory(manifest_path: Path, max_features: int = MAX_FEATURES) -> dict:
    """Inventory every GML file named by a geology acquisition manifest."""
    import json

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("schema") != "geoneural-geology-v1":
        raise ValueError("Expected a geoneural-geology-v1 acquisition manifest")
    directory = Path(manifest_path).parent
    layers = []
    for record in manifest.get("files", []):
        path = (directory / record["path"]).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("GML file escapes the acquisition directory")
        entry = inventory_file(path, max_features)
        entry["declared_type"] = record.get("type")
        entry["declared_count"] = record.get("count")
        layers.append(entry)
    combined_fields: Counter[str] = Counter()
    combined_refs: Counter[str] = Counter()
    for layer in layers:
        combined_fields.update(layer["fields"])
        combined_refs.update(layer["linked_reference_fields"])
    return {
        "schema": "geoneural-geology-inventory-v1",
        "source_manifest": str(manifest_path),
        "complete_acquisition": manifest.get("complete"),
        "provider": manifest.get("provider"),
        "license": manifest.get("license"),
        "attribution": manifest.get("attribution"),
        "scale_denominator": manifest.get("scale_denominator"),
        "layers": layers,
        "fields_across_layers": dict(combined_fields.most_common()),
        "linked_reference_fields_across_layers": dict(combined_refs.most_common()),
        "warning": (
            "Schema inventory only. No attribute is nominated as lithology, and a field reached "
            "through an xlink reference is not resolved here. A 1:100,000 interpretive map is not "
            "exact subsurface occupancy, and map scale is not a positional accuracy."
        ),
    }
