"""Bounded, restartable acquisition of public NRW terrain and geology.

Completed, hash-verified downloads are reused. Partial files are never inputs.
No source/model data is committed to the repository. WCS/WFS schemas are checked
at runtime because providers can change them independently of this code.
"""
from __future__ import annotations
import hashlib
import math
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlencode
import requests
from geoneural.data import wcs as wcs_schema
from geoneural.common import digest, read_json, sha_file, utc, write_json

WCS = "https://www.wcs.nrw.de/geobasis/wcs_nw_dgm"
WFS = "https://www.wfs.nrw.de/gd/wfs_nw_inspire-gk100"
DGM_DOC = "https://www.bezreg-koeln.nrw.de/geobasis-nrw/produkte-und-dienste/hoehenmodelle/digitale-gelaendemodelle/digitales-gelaendemodell"
GK_DOC = "https://www.gd.nrw.de/pr_kd_geologische-karte-100000.php"

#: WFS paging bounds. A short page ends a collection; the limit bounds one run.
GEOLOGY_PAGE_SIZE = 250
GEOLOGY_PAGE_LIMIT = 40


def url(base: str, params: list[tuple[str, str]]) -> str:
    return base + "?" + urlencode(params)


def download(address: str, path: Path, max_bytes: int, kind: str) -> dict:
    receipt_path = path.with_suffix(path.suffix + ".receipt.json")
    if path.exists() and receipt_path.exists():
        previous = read_json(receipt_path)
        if previous.get("url") == address and previous.get("sha256") == sha_file(path):
            # A completed, hash-verified file is already spent work. The remaining
            # allowance shrinks as a run progresses, so re-checking it here would
            # make an interrupted acquisition impossible to resume.
            return previous
        raise ValueError(f"Existing input identity differs; choose a fresh output directory: {path}")
    if path.exists():
        raise ValueError(f"Unreceipted input exists; inspect it or choose a fresh directory: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with requests.get(address, stream=True, timeout=(15, 180), headers={"User-Agent": "geoneural/0.1 (bounded research download)"}) as response:
                if response.status_code >= 400:
                    # An OWS service explains a refusal in the body. Discarding it
                    # forces a second request just to learn why the first failed.
                    body = response.raw.read(16384, decode_content=True) or b""
                    rejected = path.with_suffix(path.suffix + ".rejected")
                    rejected.parent.mkdir(parents=True, exist_ok=True)
                    rejected.write_bytes(body)
                    text = body.decode("utf-8", "replace").strip()
                    raise ValueError(
                        f"Provider returned HTTP {response.status_code} for {address}; body retained at "
                        f"{rejected}: {text[:600]}")
                response.raise_for_status()
                size = int(response.headers.get("Content-Length", 0))
                if size > max_bytes:
                    raise ValueError(f"Provider response {size} exceeds byte allowance {max_bytes}")
                h, received = hashlib.sha256(), 0
                with temporary.open("wb") as target:
                    for chunk in response.iter_content(1024 * 1024):
                        received += len(chunk)
                        if received > max_bytes:
                            raise ValueError("Download exceeded its byte allowance")
                        target.write(chunk)
                        h.update(chunk)
                with temporary.open("rb") as source:
                    prefix = source.read(256)
                if kind == "tiff" and prefix[:4] not in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
                    # Keep the body: a provider exception document is the diagnosis,
                    # and discarding it forces a second request to learn the reason.
                    rejected = path.with_suffix(path.suffix + ".rejected")
                    os.replace(temporary, rejected)
                    raise ValueError(f"Expected GeoTIFF, received {prefix[:180]!r}; body retained at {rejected}")
                receipt = {"url": address, "sha256": h.hexdigest(), "bytes": received, "retrieved_utc": utc(), "etag": response.headers.get("ETag"), "last_modified": response.headers.get("Last-Modified"), "content_type": response.headers.get("Content-Type")}
                os.replace(temporary, path)
                write_json(receipt_path, receipt)
                return receipt
        except (requests.RequestException, OSError, ValueError) as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)
            if isinstance(exc, ValueError):
                break
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Acquisition failed for {path}: {last_error}") from last_error


def xml_root(path: Path) -> ET.Element:
    root = ET.parse(path).getroot()
    if "Exception" in root.tag:
        raise ValueError("Provider exception: " + " ".join(root.itertext())[:1200])
    return root


def coverage_params(coverage: str, easting_axis: str, northing_axis: str,
                    box: tuple[float, float, float, float], scaling: list[tuple[str, str]]) -> list[tuple[str, str]]:
    x0, x1, y0, y1 = box
    return [("SERVICE", "WCS"), ("VERSION", "2.0.1"), ("REQUEST", "GetCoverage"), ("COVERAGEID", coverage),
            ("FORMAT", "image/tiff"), ("SUBSETTINGCRS", "EPSG:25832"), ("OUTPUTCRS", "EPSG:25832"),
            ("SUBSET", f"{easting_axis}({x0},{x1})"), ("SUBSET", f"{northing_axis}({y0},{y1})")] + scaling


def probe_scaling(out: Path, coverage: str, easting_axis: str, northing_axis: str,
                  spacing: float, box: tuple[float, float, float, float]) -> dict:
    """Find a request form this service actually answers, using one small request.

    A malformed request must not be retried in a loop over hundreds of tiles. One
    bounded probe establishes which scaling spelling works, and every refusal is
    retained with its provider text so a schema change is diagnosable later.
    """
    attempts = []
    for index, form in enumerate(wcs_schema.scaling_forms(easting_axis, northing_axis, 1.0 / spacing)):
        label = wcs_schema.scaling_label(form)
        path = out / f"wcs-probe-{index}.tif"
        address = url(WCS, coverage_params(coverage, easting_axis, northing_axis, box, form))
        try:
            receipt = download(address, path, 32 * 1024 * 1024, "tiff")
        except (ValueError, RuntimeError, requests.RequestException) as exc:
            rejected = path.with_suffix(path.suffix + ".rejected")
            detail = wcs_schema.is_exception(rejected) if rejected.exists() else None
            attempts.append({"scaling": label, "accepted": False, "error": str(exc)[:400],
                             "provider_exception": (detail or "")[:600] or None})
            continue
        observed = {}
        try:
            import rasterio
            with rasterio.open(path) as dataset:
                observed = {"width": dataset.width, "height": dataset.height,
                            "pixel_size_m": [abs(dataset.transform.a), abs(dataset.transform.e)],
                            "crs": dataset.crs.to_string() if dataset.crs else None,
                            "nodata": str(dataset.nodata)}
        except Exception as exc:  # noqa: BLE001 - a readable probe is informative, a failure is not fatal
            observed = {"unreadable": str(exc)[:200]}
        attempts.append({"scaling": label, "accepted": True, "probe": path.name,
                         "bytes": receipt["bytes"], "observed": observed})
        return {"scaling_form": form, "scaling": label, "attempts": attempts, "probe_observed": observed}
    raise ValueError(
        "No supported WCS request form produced imagery. Inspect the retained capabilities, description "
        "and rejected bodies in " + str(out) + " before retrying; do not substitute synthetic terrain."
    )


def fetch_terrain(config: dict, out: Path, max_mib: int = 512, source_spacing: float | None = None,
                  tile_m: float | None = None) -> Path:
    spacing = float(source_spacing or config["source_spacing_m"])
    if not 1 <= spacing <= 100:
        raise ValueError("Source spacing must be 1..100 metres")
    if max_mib <= 0:
        raise ValueError("A positive download allowance is required")
    out.mkdir(parents=True, exist_ok=True)
    caps = out / "wcs-capabilities.xml"
    download(url(WCS, [("SERVICE", "WCS"), ("VERSION", "2.0.1"), ("REQUEST", "GetCapabilities")]), caps, 4 * 1024 * 1024, "xml")
    available = wcs_schema.coverage_ids(xml_root(caps))
    coverage = wcs_schema.resolve_coverage(available, config.get("coverage_id", "nw_dgm"))
    desc = out / "wcs-description.xml"
    download(url(WCS, [("SERVICE", "WCS"), ("VERSION", "2.0.1"), ("REQUEST", "DescribeCoverage"), ("COVERAGEID", coverage)]), desc, 4 * 1024 * 1024, "xml")
    axes = wcs_schema.describe_axes(xml_root(desc))
    easting_axis, northing_axis = wcs_schema.subset_axes(axes["axis_labels"])
    west, south, east, north = config["bbox"]
    # A service caps the pixels it will return per request, so a finer source
    # spacing needs a smaller ground tile. The preset value suits the preset
    # spacing; an explicit override is required when that changes.
    tile = float(tile_m or config["download_tile_m"])
    halo = max(20.0, 2 * spacing)
    negotiation = probe_scaling(out, coverage, easting_axis, northing_axis, spacing,
                                (west, west + 200.0, south, south + 200.0))
    scaling = negotiation["scaling_form"]
    nx, ny = math.ceil((east-west)/tile), math.ceil((north-south)/tile)
    if nx * ny > 256:
        raise ValueError("This acquisition is capped at 256 source tiles; split the region")
    estimate = sum((math.ceil((min(tile,east-west-i*tile)+2*halo)/spacing)+2)*(math.ceil((min(tile,north-south-j*tile)+2*halo)/spacing)+2)*4 for j in range(ny) for i in range(nx))
    allowance = max_mib * 1024 * 1024
    if estimate > allowance:
        raise ValueError(f"Estimated float32 payload {estimate/2**20:.1f} MiB exceeds --max-mib {max_mib}")
    receipts, total = [], 0
    for j in range(ny):
        for i in range(nx):
            x0, x1 = west+i*tile-halo, min(east,west+(i+1)*tile)+halo
            y0, y1 = south+j*tile-halo, min(north,south+(j+1)*tile)+halo
            params = coverage_params(coverage, easting_axis, northing_axis, (x0, x1, y0, y1), scaling)
            filename = f"dgm-{i:03d}-{j:03d}.tif"
            item = download(url(WCS, params), out/filename, allowance-total, "tiff")
            total += item["bytes"]
            receipts.append({"path":filename, **item})
            print(f"terrain {len(receipts)}/{nx*ny}: {filename}, {item['bytes']:,} bytes", flush=True)
            time.sleep(0.2)
    manifest = {"schema":"geoneural-input-v1", "config":config, "config_sha256":digest(config), "source_kind":"observed-derived-dtm", "source_spacing_requested_m":spacing, "download_tile_m":tile, "provider_resampling":f"WCS {negotiation['scaling']}; implementation-dependent, recorded rather than assumed area averaging", "coverage_id":coverage, "coverage_ids_available":available[:40], "axes":axes, "subset_axes":{"easting":easting_axis,"northing":northing_axis}, "negotiation":negotiation["attempts"], "probe_observed":negotiation["probe_observed"], "horizontal_crs":"EPSG:25832", "vertical_crs":"EPSG:7837", "provider":"Geobasis NRW", "documentation":DGM_DOC, "license":"DL-DE-Zero-2.0", "license_url":"https://www.govdata.de/dl-de/zero-2-0", "files":receipts, "bytes":total, "capabilities_sha256":sha_file(caps), "description_sha256":sha_file(desc)}
    write_json(out/"input.json", manifest)
    return out/"input.json"


def fetch_geology(config: dict, out: Path, max_mib: int = 128, feature_type: str | None = None) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    caps = out/"wfs-capabilities.xml"
    download(url(WFS,[("SERVICE","WFS"),("VERSION","2.0.0"),("REQUEST","GetCapabilities")]),caps,4*1024*1024,"xml")
    root = xml_root(caps)
    names = [c.text for f in root.iter() if f.tag.split("}")[-1]=="FeatureType" for c in f if c.tag.split("}")[-1]=="Name" and c.text]
    chosen = [feature_type] if feature_type else [n for n in names if any(x in n.lower() for x in ("mappedfeature","geologicunit","geologicfault"))]
    if not chosen or any(n not in names for n in chosen):
        raise ValueError(f"Specify --feature-type from this service's capabilities: {names}")
    total, records, truncated = 0, [], []
    for index, name in enumerate(chosen):
        start = 0
        for page in range(GEOLOGY_PAGE_LIMIT):
            params=[("SERVICE","WFS"),("VERSION","2.0.0"),("REQUEST","GetFeature"),("TYPENAMES",name),("SRSNAME","EPSG:25832"),("BBOX",','.join(map(str,config['bbox']))+",EPSG:25832"),("COUNT",str(GEOLOGY_PAGE_SIZE)),("STARTINDEX",str(start))]
            filename=f"geology-{index:02d}-{page:03d}.gml"
            receipt=download(url(WFS,params),out/filename,max_mib*1024*1024-total,"xml")
            tree=xml_root(out/filename)
            members=[e for e in tree if e.tag.split('}')[-1] in ('member','featureMember')]
            count=int(tree.attrib.get('numberReturned',len(members)))
            if count != len(members):
                raise ValueError("Unsupported WFS member framing; inspect GML before continuing")
            total += receipt['bytes']
            records.append({"path":filename,"type":name,"count":count,**receipt})
            print(f"geology {name} offset {start}: {count} features",flush=True)
            if count == 0:
                break
            start += count
            matched=tree.attrib.get('numberMatched','unknown')
            # WFS 2.0 may legitimately answer 'unknown'. A short page is then the
            # only reliable end-of-collection signal, and completed pages are kept
            # either way: discarding real downloaded features to signal a cap is
            # worse than reporting the acquisition as incomplete.
            if count < GEOLOGY_PAGE_SIZE:
                break
            if matched.isdigit() and start>=int(matched):
                break
            time.sleep(0.2)
        else:
            truncated.append({"type":name,"retrieved":start,"page_limit":GEOLOGY_PAGE_LIMIT})
            print(f"geology {name}: page limit reached after {start} features; recorded as incomplete",flush=True)
    manifest={"schema":"geoneural-geology-v1","config":config,"provider":"Geologischer Dienst NRW","source_kind":"interpreted-geological-map","scale_denominator":100000,"documentation":GK_DOC,"license":"DL-DE-BY-2.0","license_url":"https://www.govdata.de/dl-de/by-2-0","attribution":f"IS GK100 ({WFS}), Geologischer Dienst NRW, retrieved {utc()}; DL-DE-BY-2.0","files":records,"complete":not truncated,"truncated":truncated,"bytes":total,"notes":"INSPIRE feature schema retained without assuming a geological attribute name. Not a 3D voxel or exact subsurface model. When complete is false the catalogue is partial and must not be used as a coverage statement."}
    write_json(out/"geology.json",manifest)
    return out/"geology.json"
