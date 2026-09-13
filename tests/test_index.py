"""EATIDX1 must lose nothing a runtime needs and must never tighten a bound.

The compact index drops bytes by deriving what is derivable and by rounding the
bounds outward. Both halves are risky in the same way: a derivation that is
subtly wrong, or a rounding that goes inward, produces an index that looks fine
and quietly culls real terrain. These tests check the reconstruction field by
field against a real manifest built through the production path.
"""
from __future__ import annotations
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]

from geoneural.data import build

from geoneural.codecs import index

INTERVALS = 16
DERIVED = ("level", "x", "y", "side", "path", "spacing_m", "width_m",
           "x_m", "z_m", "raw_bytes", "sha256", "packed_bytes", "error_kind")


def tiny_atlas(root: Path) -> Path:
    side = 4 * INTERVALS + 1
    rows = np.linspace(-12.0, 61.0, side, dtype=np.float64)[:, None]
    cols = np.cos(np.linspace(0.0, 6.0, side, dtype=np.float64))[None, :] * 7.0
    config = {"bbox": [0.0, 0.0, (side - 1) * 10.0, (side - 1) * 10.0], "crs": "EPSG:25832",
              "vertical_crs": "EPSG:7837", "spacing_m": 10.0, "page_intervals": INTERVALS,
              "quantum_m": 0.01, "preset": "index-test", "title": "index test",
              "notes": "synthetic", "source_spacing_m": 10}
    provenance = {"source": {"source_kind": "synthetic-not-earth"}, "note": "index fixture"}
    build.pack(rows + cols, config, root / "atlas", provenance)
    return root / "atlas" / "atlas.json"


class CompactIndex(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = tiny_atlas(Path(cls.tmp.name))
        cls.manifest = json.loads(cls.path.read_text())
        cls.blob = index.encode(cls.manifest)
        cls.rebuilt = index.decode(cls.blob)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_every_derived_field_is_reconstructed_exactly(self):
        self.assertEqual(set(self.rebuilt["pages"]), set(self.manifest["pages"]))
        for key, entry in self.manifest["pages"].items():
            other = self.rebuilt["pages"][key]
            for field in DERIVED:
                with self.subTest(page=key, field=field):
                    self.assertEqual(other[field], entry[field])

    def test_bounds_loosen_outward_and_never_inward(self):
        quantum = self.manifest["quantum_m"]
        for key, entry in self.manifest["pages"].items():
            other = self.rebuilt["pages"][key]
            with self.subTest(page=key):
                # A culler reading these must never exclude ground the page holds.
                self.assertLessEqual(other["min_m"], entry["min_m"] + 1e-9)
                self.assertGreaterEqual(other["max_m"], entry["max_m"] - 1e-9)
                self.assertGreaterEqual(other["sample_error_m"], entry["sample_error_m"] - 1e-9)
                # And the price of that safety is bounded at one quantum.
                self.assertLess(entry["min_m"] - other["min_m"], quantum + 1e-9)
                self.assertLess(other["max_m"] - entry["max_m"], quantum + 1e-9)

    def test_lattice_and_identity_survive(self):
        for field in ("sample_side", "page_intervals", "max_level", "spacing_m",
                      "quantum_m", "bounds", "root", "crs", "vertical_crs",
                      "content_id", "algorithm_sha256", "filter", "source_kind", "name"):
            with self.subTest(field=field):
                self.assertEqual(self.rebuilt[field], self.manifest[field])

    def test_it_is_actually_smaller(self):
        self.assertLess(len(self.blob), len(json.dumps(self.manifest).encode()) // 4)
        report = index.compare(self.manifest, self.blob)
        self.assertTrue(report["boundsNeverTightened"])
        self.assertEqual(report["pages"], len(self.manifest["pages"]))

    def test_a_missing_page_is_recorded_rather_than_invented(self):
        # A sparse atlas must not come back with a fabricated entry at full size.
        thinned = json.loads(json.dumps(self.manifest))
        victim = f"0/{2 ** self.manifest['max_level'] - 1}/0"
        del thinned["pages"][victim]
        rebuilt = index.decode(index.encode(thinned))
        self.assertNotIn(victim, rebuilt["pages"])
        self.assertEqual(len(rebuilt["pages"]), len(thinned["pages"]))

    def test_foreign_and_damaged_indexes_are_refused(self):
        with self.assertRaises(ValueError):
            index.decode(b"NOTIDX01" + self.blob[8:])
        with self.assertRaises(ValueError):
            index.decode(self.blob + b"\x00")
        with self.assertRaises(ValueError):
            index.decode(self.blob[:-1])

    def test_a_nonuniform_page_shape_is_refused_not_silently_hoisted(self):
        # side and raw_bytes are stored once. If an atlas ever breaks that
        # assumption the index must refuse rather than relabel every page.
        mutated = json.loads(json.dumps(self.manifest))
        key = next(iter(mutated["pages"]))
        mutated["pages"][key]["side"] = mutated["pages"][key]["side"] + 2
        with self.assertRaises(ValueError):
            index.encode(mutated)
        mutated = json.loads(json.dumps(self.manifest))
        mutated["pages"][key]["error_kind"] = "something else entirely"
        with self.assertRaises(ValueError):
            index.encode(mutated)


if __name__ == "__main__":
    unittest.main()
