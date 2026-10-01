"""The observation operator defines the experiment, so it is pinned here."""
from __future__ import annotations

import unittest

import numpy as np

from geoneural.superres import superres


class TheOperatorIsUnbiased(unittest.TestCase):
    """A biased operator does not look like an operator error in the results.
    It looks like uniform model error, everywhere."""

    def test_an_interior_cell_is_centred_on_its_node(self):
        weights = superres._axis_weights(101, 10)
        positions = np.arange(101)
        for node in range(1, 10):
            centroid = float((weights[node] * positions).sum())
            self.assertAlmostEqual(centroid, node * 10, places=9)

    def test_weights_are_a_partition(self):
        for side, factor in ((101, 10), (65, 8), (1025, 64)):
            weights = superres._axis_weights(side, factor)
            self.assertTrue(np.allclose(weights.sum(axis=1), 1.0), f"{side}/{factor}")

    def test_a_linear_field_survives_the_operator_in_the_interior(self):
        """The property that makes it unbiased. Both obvious implementations
        (the block starting at the node, and the block ending at it) fail this
        by half a coarse cell in opposite directions."""
        y, x = np.mgrid[0:101, 0:101]
        field = 0.3 * x + 0.7 * y + 5.0
        coarse = superres.observe(field, 10)
        interior = np.abs(coarse - field[::10, ::10])[1:-1, 1:-1]
        self.assertLess(float(interior.max()), 1e-9)

    def test_edge_nodes_average_the_half_cell_that_exists(self):
        """Not a defect: a half-cell average of a ramp is not the ramp at the
        edge node, and pretending otherwise would invent data outside the tile."""
        y, x = np.mgrid[0:101, 0:101]
        field = 0.3 * x + 0.7 * y + 5.0
        coarse = superres.observe(field, 10)
        self.assertGreater(float(np.abs(coarse - field[::10, ::10])[0, 0]), 0.1)

    def test_a_constant_field_is_exact_everywhere_including_edges(self):
        coarse = superres.observe(np.full((101, 101), 42.0), 10)
        self.assertTrue(np.allclose(coarse, 42.0))


class ClassicalControlsAreRealCompetitors(unittest.TestCase):
    """These are what a super-resolution claim must beat. Bilinear is the atlas's
    own declared reconstruction, so it is a floor rather than a strawman."""

    def field(self):
        y, x = np.mgrid[0:101, 0:101]
        return 100.0 + 8.0 * np.sin(2 * np.pi * x / 60) * np.cos(2 * np.pi * y / 80) + 0.02 * x

    def test_every_method_returns_the_fine_lattice(self):
        coarse = superres.observe(self.field(), 10)
        for method in ("bilinear", "bicubic", "lanczos", "cubic_spline"):
            self.assertEqual(superres.classical(coarse, 10, method).shape, (101, 101), method)

    def test_back_projection_improves_operator_consistency_for_all_of_them(self):
        """Applied to classical methods too, deliberately: enforcing the
        constraint only for the network would favour it."""
        fine = self.field()
        coarse = superres.observe(fine, 10)
        for method in ("bilinear", "bicubic", "lanczos"):
            raw = superres.classical(coarse, 10, method)
            fixed, report = superres.back_project(raw, coarse, 10)
            before = superres.operator_consistency(raw, coarse, 10)["maeM"]
            after = superres.operator_consistency(fixed, coarse, 10)["maeM"]
            self.assertLess(after, before, method)
            self.assertTrue(report["converged"], method)

    def test_an_unknown_method_is_refused(self):
        with self.assertRaises(KeyError):
            superres.classical(np.zeros((11, 11)), 10, "magic")


if __name__ == "__main__":
    unittest.main()
