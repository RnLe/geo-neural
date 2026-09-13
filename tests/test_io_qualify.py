"""The qualified IO harness: what must hold before any of its milliseconds are read.

A cold label on a resident file, two codecs serving different terrain, or a
qualification granted on a busy host would each corrupt every number measured
on top of them, silently and in the favourable direction.
"""
from __future__ import annotations
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from geoneural.data import build

from geoneural.codecs import eat1 as codec

from geoneural.bench import io_qualify

INTERVALS = 64


def _atlas(root: Path) -> Path:
    side = 2 * INTERVALS + 1
    heights = (np.linspace(0.0, 40.0, side)[:, None] + np.linspace(-3.0, 25.0, side)[None, :])
    config = {"bbox": [0.0, 0.0, (side - 1) * 10.0, (side - 1) * 10.0], "crs": "EPSG:25832",
              "vertical_crs": "EPSG:7837", "spacing_m": 10.0, "page_intervals": INTERVALS,
              "quantum_m": 0.01, "preset": "io-qualify-test", "title": "io qualify test",
              "notes": "synthetic", "source_spacing_m": 10}
    build.pack(heights, config, root / "atlas",
               {"source": {"source_kind": "synthetic-not-earth"}, "note": "IO harness fixture"})
    return root / "atlas" / "atlas.json"


class Staging(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.staged = io_qualify.stage(_atlas(root), root / "staged")
        self.profiles = io_qualify._load(self.staged)

    def tearDown(self):
        self.tmp.cleanup()

    def test_both_codecs_serve_identical_heights(self):
        eat1, q32 = self.profiles["eat1"], self.profiles["q32dz"]
        self.assertEqual(sorted(eat1["pages"]), sorted(q32["pages"]))
        for key, entry in eat1["pages"].items():
            a = codec.decode((eat1["dir"] / entry["path"]).read_bytes(), entry["raw_bytes"])[0]
            b = io_qualify._q32dz().decode((q32["dir"] / q32["pages"][key]["path"]).read_bytes())
            self.assertLessEqual(float(np.abs(a.astype("f8") - b).max()), 1e-5)

    def test_every_codec_and_layout_answers_the_same_request(self):
        window = io_qualify.window_for(4225, self.staged["side"])
        results = [io_qualify.query(name, self.profiles[name], layout, window, INTERVALS)
                   for name in io_qualify.CODECS for layout in io_qualify.LAYOUTS]
        self.assertEqual({r["usefulSamples"] for r in results}, {4225})
        self.assertEqual({r["pagesTouched"] for r in results}, {1})
        packed = [r for r, (_, layout) in zip(results, [(c, l) for c in io_qualify.CODECS
                                                        for l in io_qualify.LAYOUTS]) if layout == "packed"]
        self.assertEqual({r["fileOpens"] for r in packed}, {1})

    def test_a_resident_file_is_never_labelled_cold(self):
        location = {"filesystem": "test", "staged": self.staged}
        with mock.patch.object(io_qualify, "evict", return_value=False), \
                mock.patch.object(io_qualify, "QUERY_SAMPLES", (1,)):
            rows = io_qualify.sweep([location], warm_trials=1, cold_trials=1)
        labels = {row["cache"] for row in rows}
        self.assertNotIn("cold", labels)
        self.assertIn("cold-unverified", labels)

    def test_a_filesystem_outside_the_page_cache_is_never_cold(self):
        # WSL's 9p never holds file data in this kernel's page cache, so an
        # eviction there "verifies" trivially while Windows' own cache stays warm.
        location = {"filesystem": "9p", "staged": self.staged}
        with mock.patch.object(io_qualify, "resident_pages", return_value=(0, 1)), \
                mock.patch.object(io_qualify, "QUERY_SAMPLES", (1,)):
            rows = io_qualify.sweep([location], warm_trials=1, cold_trials=1)
        self.assertNotIn("cold", {row["cache"] for row in rows})

    def test_the_eviction_claim_matches_residency(self):
        # Whatever this filesystem can do, evict() may only say cold when it is.
        path = Path(self.staged["root"]) / "eat1" / "pages.eatpack"
        path.read_bytes()
        claimed = io_qualify.evict([path])
        self.assertEqual(claimed, io_qualify.resident_pages(path)[0] == 0)


class Run(unittest.TestCase):
    def test_a_relative_output_directory_reaches_the_child_processes(self):
        # Children are fresh interpreters, so a relative --out must be resolved first.
        import os
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            atlas = _atlas(root)
            cwd = os.getcwd()
            os.chdir(root)
            try:  # the child is a fresh interpreter: nothing here can be mocked into it
                summary = io_qualify.run(atlas, Path("out"), None, processes=1,
                                         warm_trials=1, cold_trials=1)
            finally:
                os.chdir(cwd)
            self.assertTrue(summary.is_absolute() and summary.exists())
            verdict = io_qualify.read_json(summary)["qualification"]
            self.assertIn("qualified", verdict)
            self.assertIsInstance(verdict["reasons"], list)


class Quantiles(unittest.TestCase):
    def test_one_sample_is_its_own_quartiles(self):
        # Python 3.12's statistics.quantiles refuses a single point; a one-trial run must still summarise.
        self.assertEqual(io_qualify._quantiles([2.5]),
                         {"median": 2.5, "p25": 2.5, "p75": 2.5, "min": 2.5, "max": 2.5, "n": 1})


class Qualification(unittest.TestCase):
    def process(self, **change):
        return {"busyBefore": 0.01, "busyAfter": 0.02, **change}

    def test_an_idle_host_qualifies(self):
        verdict = io_qualify.qualification([self.process()] * 3)
        self.assertTrue(verdict["qualified"], verdict["reasons"])

    def test_any_missing_or_busy_record_refuses(self):
        cases = {
            "no processes": [],
            "busy before": [self.process(busyBefore=0.3)],
            "busy after": [self.process(), self.process(busyAfter=0.5)],
            "not recorded": [self.process(busyAfter=None)],
        }
        for name, processes in cases.items():
            with self.subTest(name):
                verdict = io_qualify.qualification(processes)
                self.assertFalse(verdict["qualified"])
                self.assertTrue(verdict["reasons"])


if __name__ == "__main__":
    unittest.main()
