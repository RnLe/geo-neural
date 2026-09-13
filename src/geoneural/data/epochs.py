"""When the terrain was surveyed, as distinct from when it was retrieved.

The WCS receipts record retrieval times only. NRW publishes per-tile metadata for
the DGM1 it serves (`dgm1_meta.zip` beside the 1 km GeoTIFF tiles): currency
date, acquisition and update method codes, stated accuracy, horizontal and
vertical reference and quasigeoid. The tile index gives each tile's publication
timestamp.

An atlas retrieved at time T inherits the epochs of the tiles current at T. A
tile republished after T may have been served in an older state, so it is
flagged and its epoch reported as unknown rather than read from newer metadata.
Method codes are reported as the provider writes them; their meaning is defined
by the AdV DGM standard the file cites, not asserted here.
"""
from __future__ import annotations

import collections
import hashlib
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

TILE = re.compile(r"dgm1_32_(\d+)_(\d+)_1_nw_(\d{4})$")
SOURCE = "https://www.opengeodata.nrw.de/produkte/geobasis/hm/dgm1_tiff/dgm1_tiff/"


def read_metadata(meta_zip: Path) -> tuple[dict, list[dict]]:
    """Header key/values and one row per tile from `dgm1_nw.csv`."""
    with zipfile.ZipFile(meta_zip) as archive:
        lines = archive.read("dgm1_nw.csv").decode("utf-8-sig").splitlines()
    header, rows, columns = {}, [], None
    for line in lines:
        cells = line.split(";")
        if columns is None:
            if cells[0] == "Kachelname":
                columns = cells
            elif len(cells) == 2:
                header[cells[0]] = cells[1]
            continue
        if line.strip():
            rows.append(dict(zip(columns, cells)))
    if columns is None:
        raise ValueError("dgm1_nw.csv has no Kachelname header row")
    return header, rows


def publication_times(index_xml: Path) -> dict[str, str]:
    return {f.get("name").removesuffix(".tif"): f.get("timestamp")
            for f in ET.parse(index_xml).iter("file") if f.get("name", "").endswith(".tif")}


def tiles_under(rows: list[dict], bounds: list[float]) -> list[dict]:
    """1 km tiles (named by their south-west corner in km) meeting the bounds."""
    west, south, east, north = bounds
    chosen = []
    for row in rows:
        match = TILE.match(row["Kachelname"])
        if not match:
            continue
        e, n = int(match.group(1)) * 1000, int(match.group(2)) * 1000
        if e < east and e + 1000 > west and n < north and n + 1000 > south:
            chosen.append(row)
    return chosen


def epochs(meta_zip: Path, index_xml: Path, bounds: list[float], retrieved_utc: str) -> dict:
    header, rows = read_metadata(meta_zip)
    published = publication_times(index_xml)
    tiles = tiles_under(rows, bounds)
    if not tiles:
        raise ValueError("no DGM1 tile meets these bounds")
    later = sorted(t["Kachelname"] for t in tiles if published.get(t["Kachelname"], "") > retrieved_utc)
    unknown = sorted(t["Kachelname"] for t in tiles if t["Kachelname"] not in published)
    known = [t for t in tiles if t["Kachelname"] not in later and t["Kachelname"] not in unknown]
    count = lambda field: dict(collections.Counter(t[field] for t in known).most_common())  # noqa: E731
    return {
        "schema": "geoneural-epochs-v1",
        "source": {"url": SOURCE, "metadataCurrency": header.get("Aktualitaet_Kachelinformationen"),
                   "standardVersion": header.get("Version_Standard"),
                   "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (meta_zip, index_xml)}},
        "retrievedUtc": retrieved_utc,
        "tiles": len(tiles),
        "republishedAfterRetrieval": later,
        "absentFromIndex": unknown,
        "surveyEpochs": count("Aktualitaet"),
        "updateDates": count("Fortfuehrung"),
        "acquisitionMethodCodes": count("Erfassungsmethode"),
        "statedAccuracyM": count("Genauigkeit"),
        "horizontalReference": count("Koordinatenreferenzsystem_Lage"),
        "verticalReference": count("Koordinatenreferenzsystem_Hoehe"),
        "heightAnomalyModel": count("Hoehenanomalie"),
        "scope": "Epochs of the tiles current at retrieval, from the provider's tile metadata. Assumes the "
                 "WCS serves the same DGM1 as the tile product. Tiles republished after retrieval carry no "
                 "epoch here.",
    }
