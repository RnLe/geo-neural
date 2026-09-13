"""The seam certificate must be conservative, and provably so.

A bound that the data can exceed is worse than no bound, because a consumer will
size its LOD error budget with it. Each piece of the derivation is checked
against a case whose answer is known analytically, then the assembled bound is
checked against densely evaluated points on the actual rendered surfaces.
"""
from __future__ import annotations
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]

from geoneural.codecs import bounds


def plane(side: int, a: float = 3.0, b: float = -1.5, c: float = 7.0) -> np.ndarray:
    ys, xs = np.mgrid[0:side, 0:side]
    return a * xs + b * ys + c


class Twist(unittest.TestCase):
    def test_a_plane_has_no_twist_and_no_slack(self):
        # Bilinear and triangulated surfaces coincide exactly on a plane, so a
        # nonzero slack here would mean the formula invents error.
        grid = plane(9)
        self.assertEqual(float(bounds.twist(grid).max()), 0.0)
        self.assertEqual(bounds.triangulation_slack(grid), 0.0)

    def test_the_unit_saddle_has_the_analytic_slack(self):
        # h = x*y over one cell: twist is 1, so the bilinear centre (0.25) and
        # the diagonal (0.5) differ by exactly 1/4.
        cell = np.array([[0.0, 0.0], [0.0, 1.0]])
        self.assertAlmostEqual(float(bounds.twist(cell).max()), 1.0, places=12)
        self.assertAlmostEqual(bounds.triangulation_slack(cell), 0.25, places=12)

    def test_slack_matches_a_direct_search_over_the_cell(self):
        rng = np.random.default_rng(7)
        for _ in range(8):
            cell = rng.normal(0.0, 5.0, (2, 2))
            ys, xs = np.mgrid[0:1:201j, 0:1:201j]
            tri = bounds._sample_triangulated(cell, ys.ravel(), xs.ravel())
            h00, h01, h10, h11 = cell[0, 0], cell[0, 1], cell[1, 0], cell[1, 1]
            bil = (h00 * (1 - xs) * (1 - ys) + h01 * xs * (1 - ys)
                   + h10 * (1 - xs) * ys + h11 * xs * ys).ravel()
            observed = float(np.abs(tri - bil).max())
            self.assertLessEqual(observed, bounds.triangulation_slack(cell) + 1e-9)
            self.assertAlmostEqual(observed, bounds.triangulation_slack(cell), places=3)


class PageBound(unittest.TestCase):
    def test_a_plane_refines_to_itself_with_a_zero_bound(self):
        # Decimating and re-expanding a plane is lossless, so any positive bound
        # would be the certificate manufacturing error out of nothing.
        fine = plane(17)
        coarse = fine[::2, ::2].copy()  # the same plane, every other node
        certificate = bounds.page_bound(coarse, fine)
        self.assertAlmostEqual(certificate["triangleBoundM"], 0.0, places=9)

    def test_a_mismatched_child_region_is_refused(self):
        with self.assertRaises(ValueError):
            bounds.page_bound(plane(9), plane(9))

    def test_the_bound_holds_on_rough_synthetic_terrain(self):
        rng = np.random.default_rng(11)
        for trial in range(5):
            fine = np.cumsum(np.cumsum(rng.normal(0, 1.0, (17, 17)), axis=0), axis=1)
            coarse = fine[::2, ::2].copy()
            certificate = bounds.page_bound(coarse, fine)
            result = bounds.verify_bound(coarse, fine, certificate, samples=60000, seed=trial)
            with self.subTest(trial=trial):
                self.assertTrue(result["holds"],
                                f"observed {result['observedMaxM']} exceeded bound {result['boundM']}")
                self.assertGreaterEqual(result["headroomM"], 0.0)

    def test_verification_rejects_a_bound_that_is_too_small(self):
        rng = np.random.default_rng(3)
        fine = np.cumsum(np.cumsum(rng.normal(0, 1.0, (17, 17)), axis=0), axis=1)
        coarse = fine[::2, ::2].copy()
        honest = bounds.page_bound(coarse, fine)
        self.assertTrue(bounds.verify_bound(coarse, fine, honest, samples=40000)["holds"])
        # Halve it: the check must fail rather than wave it through.
        understated = {**honest, "triangleBoundM": honest["triangleBoundM"] / 2.0}
        self.assertFalse(bounds.verify_bound(coarse, fine, understated, samples=40000)["holds"])


class TriangulatedSampler(unittest.TestCase):
    def test_the_sampler_reproduces_node_heights(self):
        rng = np.random.default_rng(5)
        grid = rng.normal(0, 3.0, (5, 5))
        rows, cols = np.mgrid[0:5, 0:5]
        got = bounds._sample_triangulated(grid, rows.ravel().astype(float), cols.ravel().astype(float))
        # Interpolation must pass through the data it interpolates.
        self.assertTrue(np.allclose(got, grid.ravel(), atol=1e-9))

    def test_the_sampler_is_exact_on_a_plane(self):
        grid = plane(6)
        rng = np.random.default_rng(9)
        ys, xs = rng.uniform(0, 5, 500), rng.uniform(0, 5, 500)
        expected = 3.0 * xs - 1.5 * ys + 7.0
        self.assertTrue(np.allclose(bounds._sample_triangulated(grid, ys, xs), expected, atol=1e-9))


if __name__ == "__main__":
    unittest.main()
