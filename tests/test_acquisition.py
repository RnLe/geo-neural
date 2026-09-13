"""Acquisition contracts: provider schema negotiation and refusal behaviour.

These run offline against fixtures shaped like retained responses. They cannot
establish that the live NRW service answers any particular request (only a real
acquisition does that), but they pin the behaviour that keeps acquisition robust:
no hard-coded coverage id or axis labels, no discarded complete download, and no
paginated collection thrown away because the server said 'unknown'.
"""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from geoneural.data import wcs
from geoneural.data.acquire import download
from geoneural.data.build import nodata_report
from geoneural.common import sha_file, write_json

DESCRIBE = """<?xml version="1.0"?>
<CoverageDescriptions xmlns="http://www.opengis.net/wcs/2.0" xmlns:gml="http://www.opengis.net/gml/3.2">
  <CoverageDescription gml:id="nw_dgm">
    <gml:boundedBy>
      <gml:Envelope srsName="http://www.opengis.net/def/crs/EPSG/0/25832" axisLabels="E N" uomLabels="m m">
        <gml:lowerCorner>280000 5560000</gml:lowerCorner>
        <gml:upperCorner>540000 5820000</gml:upperCorner>
      </gml:Envelope>
    </gml:boundedBy>
    <gml:domainSet>
      <gml:RectifiedGrid dimension="2">
        <gml:limits><gml:GridEnvelope><gml:low>0 0</gml:low><gml:high>259999 259999</gml:high></gml:GridEnvelope></gml:limits>
        <gml:origin><gml:Point gml:id="p"><gml:pos>280000.5 5819999.5</gml:pos></gml:Point></gml:origin>
        <gml:offsetVector>1 0</gml:offsetVector>
        <gml:offsetVector>0 -1</gml:offsetVector>
      </gml:RectifiedGrid>
    </gml:domainSet>
  </CoverageDescription>
</CoverageDescriptions>
"""

EXCEPTION = """<?xml version="1.0"?>
<ows:ExceptionReport xmlns:ows="http://www.opengis.net/ows/2.0" version="2.0.0">
  <ows:Exception exceptionCode="InvalidAxisLabel"><ows:ExceptionText>Invalid axis label: x</ows:ExceptionText></ows:Exception>
</ows:ExceptionReport>
"""


class CoverageResolution(unittest.TestCase):
    def test_exact_and_namespace_qualified_ids_both_resolve(self) -> None:
        self.assertEqual(wcs.resolve_coverage(["nw_dgm", "nw_dom"], "nw_dgm"), "nw_dgm")
        self.assertEqual(wcs.resolve_coverage(["geobasis__nw_dgm"], "nw_dgm"), "geobasis__nw_dgm")
        self.assertEqual(wcs.resolve_coverage(["ns:nw_dgm"], "nw_dgm"), "ns:nw_dgm")

    def test_ambiguous_or_absent_coverage_refuses_rather_than_guessing(self) -> None:
        with self.assertRaises(ValueError):
            wcs.resolve_coverage(["a__nw_dgm", "b__nw_dgm"], "nw_dgm")
        with self.assertRaises(ValueError) as raised:
            wcs.resolve_coverage(["nw_dom"], "nw_dgm")
        self.assertIn("absent", str(raised.exception))


class AxisNegotiation(unittest.TestCase):
    def test_description_supplies_axis_labels_and_grid_geometry(self) -> None:
        axes = wcs.describe_axes(ET.fromstring(DESCRIBE))
        self.assertEqual(axes["axis_labels"], ["E", "N"])
        self.assertIn("25832", axes["srs_name"])
        self.assertEqual(axes["origin"], [280000.5, 5819999.5])
        self.assertEqual(axes["offset_vectors"], [[1.0, 0.0], [0.0, -1.0]])
        self.assertEqual(axes["grid_size"], [260000, 260000])

    def test_subset_labels_follow_the_service_rather_than_the_old_assumption(self) -> None:
        self.assertEqual(wcs.subset_axes(["E", "N"]), ("E", "N"))
        self.assertEqual(wcs.subset_axes(["Long", "Lat"]), ("Long", "Lat"))
        # Only a description with no labels at all falls back to x/y.
        self.assertEqual(wcs.subset_axes(None), ("x", "y"))
        with self.assertRaises(ValueError):
            wcs.subset_axes(["time", "depth"])

    def test_scaling_forms_are_ordered_alternatives_not_a_single_assumption(self) -> None:
        forms = wcs.scaling_forms("E", "N", 0.1)
        self.assertEqual(wcs.scaling_label(forms[0]), "SCALEFACTOR")
        self.assertEqual(wcs.scaling_label(forms[1]), "SCALEAXESBYFACTOR")
        self.assertIn("E(0.1)", forms[1][0][1])
        self.assertEqual(forms[-1], [])

    def test_provider_exception_body_is_recognised(self) -> None:
        temporary = Path(tempfile.mkdtemp(prefix="geoneural-acq-"))
        self.addCleanup(shutil.rmtree, temporary, ignore_errors=True)
        path = temporary / "rejected.xml"
        path.write_text(EXCEPTION)
        self.assertIn("Invalid axis label", wcs.is_exception(path))
        plain = temporary / "plain.xml"
        plain.write_text("<ok/>")
        self.assertIsNone(wcs.is_exception(plain))


class ResumeBehaviour(unittest.TestCase):
    def test_completed_receipted_file_survives_a_shrunken_allowance(self) -> None:
        """The remaining allowance shrinks as a run progresses.

        Re-checking a finished file against it would make an interrupted
        acquisition impossible to resume: the files that make a restart cheap
        would be the ones rejected.
        """
        temporary = Path(tempfile.mkdtemp(prefix="geoneural-resume-"))
        self.addCleanup(shutil.rmtree, temporary, ignore_errors=True)
        path = temporary / "dgm-000-000.tif"
        path.write_bytes(b"II*\x00" + b"0" * 4096)
        address = "https://example.invalid/coverage"
        write_json(path.with_suffix(path.suffix + ".receipt.json"),
                   {"url": address, "sha256": sha_file(path), "bytes": path.stat().st_size})
        # No network call is made: a verified file returns its receipt directly.
        receipt = download(address, path, max_bytes=16, kind="tiff")
        self.assertEqual(receipt["url"], address)

    def test_differing_identity_still_refuses(self) -> None:
        temporary = Path(tempfile.mkdtemp(prefix="geoneural-resume-"))
        self.addCleanup(shutil.rmtree, temporary, ignore_errors=True)
        path = temporary / "dgm-000-000.tif"
        path.write_bytes(b"II*\x00")
        write_json(path.with_suffix(path.suffix + ".receipt.json"),
                   {"url": "https://example.invalid/other", "sha256": "0" * 64, "bytes": 4})
        with self.assertRaises(ValueError):
            download("https://example.invalid/coverage", path, max_bytes=1 << 20, kind="tiff")


class NodataDiagnosis(unittest.TestCase):
    def test_refused_lattice_records_where_the_holes_are(self) -> None:
        temporary = Path(tempfile.mkdtemp(prefix="geoneural-nodata-"))
        self.addCleanup(shutil.rmtree, temporary, ignore_errors=True)
        config = {"bbox": [356000.5, 5694000.5, 356160.5, 5694160.5], "spacing_m": 10.0,
                  "page_intervals": 8, "quantum_m": 0.01, "crs": "EPSG:25832",
                  "vertical_crs": "EPSG:7837", "title": "fixture"}
        merged = np.zeros((17, 17), dtype=np.float32)
        merged[2:5, 3:6] = np.nan
        path = nodata_report(merged, config, temporary)
        report = json.loads(path.read_text())
        self.assertEqual(report["missing_samples"], 9)
        self.assertEqual(report["extent"]["row_range"], [2, 4])
        self.assertEqual(report["extent"]["col_range"], [3, 5])
        # Row 2 is nearer the north edge, so its northing is the larger value.
        north = config["bbox"][3]
        self.assertAlmostEqual(report["extent"]["northing_range_m"][1], north - 2 * 10.0)
        self.assertTrue((temporary / "nodata-mask.npy").exists())
        self.assertEqual(int(np.load(temporary / "nodata-mask.npy").sum()), 9)


if __name__ == "__main__":
    unittest.main()
