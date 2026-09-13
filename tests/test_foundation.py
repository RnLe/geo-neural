"""Foundation checks: byte accounting, uncertainty channels and lattice verification.

These build a tiny real atlas through the production `build.pack` path rather than
a hand-written fixture, so the conventions under test are the ones the pipeline
actually writes.
"""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from geoneural.codecs import accounting

from geoneural.data import geodesy
from geoneural.data.build import pack
from geoneural.codecs.contracts import DeploymentBytes, Uncertainty, unquantified

#: 17x17 nodes, 8-interval pages, two leaves per side, one parent level.
TINY = {
    "title": "Synthetic foundation fixture",
    "bbox": [356000.5, 5694000.5, 356160.5, 5694160.5],
    "crs": "EPSG:25832",
    "vertical_crs": "EPSG:7837",
    "spacing_m": 10.0,
    "page_intervals": 8,
    "quantum_m": 0.01,
    "preset": "synthetic-foundation-fixture",
}


def tiny_reference() -> np.ndarray:
    """A tilted, asymmetric surface: a transpose or flip changes it detectably."""
    rows, cols = np.meshgrid(np.arange(17.0), np.arange(17.0), indexing="ij")
    return (100.0 + 0.5 * rows + 3.0 * cols + 0.01 * rows * cols).astype(np.float32)


class Foundation(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = Path(tempfile.mkdtemp(prefix="geoneural-foundation-"))
        self.addCleanup(shutil.rmtree, self.temporary, ignore_errors=True)
        self.atlas_dir = self.temporary / "atlas"
        provenance = {"source": {"source_kind": "synthetic-not-earth"}, "note": "unit fixture"}
        pack(tiny_reference(), TINY, self.atlas_dir, provenance)
        self.manifest = json.loads((self.atlas_dir / "atlas.json").read_text())

    # -- accounting ---------------------------------------------------------

    def test_every_shipped_byte_is_accounted_and_research_bytes_stay_separate(self) -> None:
        report = accounting.package_bytes(self.atlas_dir)
        on_disk = sum(p.stat().st_size for p in self.atlas_dir.rglob("*") if p.is_file())
        self.assertEqual(report["deployment_total_bytes"] + report["research_only_bytes"], on_disk)
        # Pages are field payload; the manifest and attribution are index/metadata.
        self.assertEqual(report["deployment"]["field_payload"], self.manifest["runtime_page_bytes"])
        self.assertGreater(report["deployment"]["metadata_and_index"], 0)
        # reference.npy and reference.tif are storage cost, never deployment bytes.
        self.assertGreater(report["research_only_bytes"], 0)
        self.assertEqual(report["deployment"]["shared_weights"], 0)

    def test_unclassified_file_refuses_rather_than_silently_dropping_bytes(self) -> None:
        (self.atlas_dir / "surprise.bin").write_bytes(b"x" * 32)
        with self.assertRaises(ValueError) as raised:
            accounting.package_bytes(self.atlas_dir)
        self.assertIn("surprise.bin", str(raised.exception))

    def test_new_payload_component_is_counted_without_breaking_prior_accounting(self) -> None:
        self.assertEqual(DeploymentBytes(1, 2, 3, 4, 5, 6).total(), 21)
        self.assertEqual(DeploymentBytes(1, 2, 3, 4, 5, 6, 7).total(), 28)
        with self.assertRaises(ValueError):
            DeploymentBytes(0, 0, 0, 0, 0, 0, -1).total()

    # -- uncertainty channels ----------------------------------------------

    def test_uncertainty_channels_stay_distinct_and_absence_is_not_zero(self) -> None:
        absent = unquantified("observation", "provider publishes no calibrated vertical accuracy")
        self.assertIsNone(absent.value_metres)
        self.assertEqual(absent.kind, "observation")
        with self.assertRaises(ValueError):
            Uncertainty(kind="rounding", basis="invented channel")
        with self.assertRaises(ValueError):
            Uncertainty(kind="codec", basis="")
        with self.assertRaises(ValueError):
            Uncertainty(kind="codec", basis="quantization", value_metres=float("nan"))

    # -- lattice, orientation and codec agreement --------------------------

    def test_prepared_atlas_passes_structural_verification(self) -> None:
        report = geodesy.verify(self.atlas_dir / "atlas.json")
        self.assertEqual(report["findings"], [])
        self.assertTrue(report["passed_structural_checks"])
        self.assertLess(report["roundtrip"]["max_roundtrip_error_m"], 1e-6)
        # Decoded pages must match the reference to within the quantization half-step.
        agreement = report["runtime_agreement"]
        self.assertLessEqual(agreement["max_difference_m"], agreement["quantization_half_step_m"] + 1e-9)

    def test_sample_centre_span_is_checked_against_pixel_edge_bounds(self) -> None:
        edges = dict(self.manifest)
        west, south, east, north = edges["bounds"]
        half = edges["spacing_m"] / 2
        edges["bounds"] = [west - half, south - half, east + half, north + half]
        findings = geodesy.lattice_check(edges)["findings"]
        self.assertTrue(any("sample-centre endpoints" in f for f in findings))

    def test_vertically_flipped_reference_is_detected(self) -> None:
        """A flip still decodes and still looks like terrain; only this catches it."""
        reference = np.load(self.atlas_dir / "reference.npy").astype(np.float64)
        flipped = np.flipud(reference)
        agreement = geodesy.runtime_agreement(self.atlas_dir, self.manifest, flipped)
        self.assertTrue(agreement["findings"])
        self.assertGreater(agreement["max_difference_m"], agreement["quantization_half_step_m"])

    def test_transposed_reference_is_detected(self) -> None:
        reference = np.load(self.atlas_dir / "reference.npy").astype(np.float64)
        agreement = geodesy.runtime_agreement(self.atlas_dir, self.manifest, reference.T)
        self.assertTrue(agreement["findings"])

    def test_node_addressing_places_row_zero_at_the_northern_edge(self) -> None:
        west, south, east, north = self.manifest["bounds"]
        row, col = geodesy.node_of(self.manifest, west, north)
        self.assertAlmostEqual(row, 0.0)
        self.assertAlmostEqual(col, 0.0)
        row, col = geodesy.node_of(self.manifest, east, south)
        self.assertAlmostEqual(row, self.manifest["sample_side"] - 1)
        self.assertAlmostEqual(col, self.manifest["sample_side"] - 1)

    def test_landmark_file_refuses_unconfirmed_entries_by_default(self) -> None:
        reference = np.load(self.atlas_dir / "reference.npy").astype(np.float64)
        path = self.temporary / "landmarks.json"
        path.write_text(json.dumps({
            "schema": "geoneural-landmarks-v1",
            "landmarks": [{
                "name": "unchecked", "lat": 51.4, "lon": 7.0, "elevation_m": 120.0,
                "vertical_crs": "EPSG:7837", "tolerance_m": 1.0, "source": "none", "confirmed": False,
            }],
        }))
        skipped = geodesy.landmark_check(self.manifest, reference, path, allow_unconfirmed=False)
        self.assertEqual(skipped["results"][0]["status"], "skipped-unconfirmed")
        self.assertEqual(skipped["unconfirmed_entries"], 1)
        self.assertEqual(skipped["findings"], [])


if __name__ == "__main__":
    unittest.main()
