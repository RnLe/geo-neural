"""Lattice and split properties.

These pin the geometry (operator alignment, the node lattice, disjoint tiles,
the held-out volume split) rather than the accuracy.
"""
from __future__ import annotations

import unittest

import numpy as np

from geoneural.physics import structure

from geoneural.superres import superres

from geoneural.superres import superres_models

from geoneural.superres import superres_train

try:
    import torch
    MISSING = False
except ImportError:  # pragma: no cover
    MISSING = True


class TheOperatorIsUnbiased(unittest.TestCase):
    """Area-average 1 m -> 10 m, node-centred, trapezoid rule."""

    def test_axis_weights_are_half_at_the_endpoints(self):
        weights = superres._axis_weights(21, 10)
        row = weights[1]
        support = np.flatnonzero(row)
        self.assertAlmostEqual(float(row[support[0]]), float(row[support[-1]]), places=12)
        self.assertAlmostEqual(float(row[support[0]]) * 2.0,
                               float(row[support[len(support) // 2]]), places=12)

    def test_weights_sum_to_one_on_every_node(self):
        for side, factor in ((11, 10), (21, 10), (17, 4)):
            weights = superres._axis_weights(side, factor)
            self.assertTrue(np.allclose(weights.sum(axis=1), 1.0), f"{side}/{factor}")

    def test_a_linear_ramp_is_reproduced_exactly_in_the_interior(self):
        # A half-cell offset in the operator shows up on a ramp and nowhere
        # else.
        factor, side = 10, 11
        fine_side = (side - 1) * factor + 1
        ramp = np.add.outer(np.arange(fine_side, dtype=np.float64),
                            np.zeros(fine_side))
        observed = superres.observe(ramp, factor)
        expected = np.add.outer(np.arange(side, dtype=np.float64) * factor,
                                np.zeros(side))
        interior = (slice(1, -1), slice(1, -1))
        self.assertLess(float(np.abs(observed[interior] - expected[interior]).max()), 1e-9)

    def test_back_projection_reaches_its_stated_tolerance_and_says_so(self):
        """A fixed iteration count is not a projection; the achieved residual is the claim."""
        rng = np.random.default_rng(5)
        coarse = np.cumsum(rng.normal(0.0, 1.0, (13, 13)), axis=0)
        estimate = superres.classical(coarse, 10, "bilinear")
        projected, report = superres.back_project(estimate, coarse, 10, tolerance=1e-6)
        achieved = superres.operator_consistency(projected, coarse, 10)["maxM"]
        self.assertTrue(report["converged"])
        self.assertLessEqual(achieved, 1e-6)
        self.assertAlmostEqual(achieved, report["achievedMaxM"], places=12)
        self.assertGreater(report["iterations"], 0)
        _, capped = superres.back_project(estimate, coarse, 10, tolerance=1e-12, max_iterations=2)
        self.assertFalse(capped["converged"])
        self.assertEqual(capped["iterations"], 2)


class TheOperatorIdentityIsWhatItDoes(unittest.TestCase):
    """The fingerprint hashes weights and code. Editing the label alone must not change it,
    and editing one weight must."""

    def test_a_changed_weight_changes_the_fingerprint(self):
        original = superres._axis_weights
        before = superres.operator_fingerprint(10)

        def nudged(side, factor):
            weights = original(side, factor)
            weights[1, 10] += 1e-6
            return weights

        superres._axis_weights = nudged
        try:
            after = superres.operator_fingerprint(10)
        finally:
            superres._axis_weights = original
        self.assertNotEqual(before, after)
        self.assertEqual(before, superres.operator_fingerprint(10))

    def test_the_label_does_not_enter_the_fingerprint(self):
        before = superres.operator_fingerprint(10)
        label = superres.OPERATOR
        superres.OPERATOR = "rasterio Resampling.average"
        try:
            self.assertEqual(before, superres.operator_fingerprint(10))
        finally:
            superres.OPERATOR = label

    def test_the_label_describes_trapezoidal_node_averaging(self):
        self.assertIn("trapezoid", superres.OPERATOR)
        self.assertNotIn("rasterio", superres.OPERATOR)
        schema = superres.operator_schema(10)
        self.assertEqual(schema["interiorOffsets"], list(range(-5, 6)))
        self.assertAlmostEqual(schema["interiorWeights"][0] * 2, schema["interiorWeights"][1])


class TilesAndPatchesSeeTheWholeField(unittest.TestCase):
    """Training patches, evaluation tiles and drainage windows are crops of one normalised field."""

    def setUp(self):
        if MISSING:
            self.skipTest("torch is not installed")
        rng = np.random.default_rng(11)
        self.coarse = np.cumsum(np.cumsum(rng.normal(0.0, 1.0, (41, 37)), 0), 1) * 0.3 + 100.0
        self.features = superres_train.coarse_inputs(self.coarse)

    def test_tiled_reconstruction_equals_the_whole_domain(self):
        torch.manual_seed(0)
        for arm in ("edsr", "liif"):
            config = {"arm": arm, "width": 8, "blocks": 2, "hidden": 16, "depth": 2,
                      "factor": 4, "channels": superres_train.INPUT_CHANNELS}
            model = superres_models.make_model(config, torch).eval()
            inputs, scale = self.features
            full = superres_train.reconstruct_window(model, config, inputs, scale, torch, 4, "cpu")
            tiled = superres_train.reconstruct_tiled(model, config, self.features, torch, 4, "cpu",
                                                     tile=8)
            self.assertEqual(full.shape, tiled.shape)
            self.assertLess(float(np.abs(full - tiled).max()), 1e-4 * float(np.abs(full).max()) + 1e-6,
                            arm)

    def test_patch_inputs_do_not_depend_on_the_fine_labels(self):
        fine = np.zeros(((41 - 1) * 4 + 1, (37 - 1) * 4 + 1))
        base = np.zeros_like(fine)
        a = superres_train._patch(self.features, self.coarse, fine, base, 5, 7, 9, 4)
        b = superres_train._patch(self.features, self.coarse, fine + 50.0, base, 5, 7, 9, 4)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[0], self.features[0][:, 5:14, 7:16])


class EveryArmAnswersOnTheNodeLattice(unittest.TestCase):
    """A pixel-shuffle head would put EDSR half a coarse cell off the other arms."""

    def setUp(self):
        if MISSING:
            self.skipTest("torch is not installed")

    def test_edsr_output_side_matches_the_node_count(self):
        model = superres_models.make_model(
            {"arm": "edsr", "width": 8, "blocks": 1, "factor": 10}, torch)
        for side in (9, 17):
            out = model(torch.zeros(1, 1, side, side))
            self.assertEqual(out.shape[-1], (side - 1) * 10 + 1, f"side {side}")

    def test_query_arms_accept_any_window(self):
        for arm in ("liif", "residual-siren"):
            model = superres_models.make_model(
                {"arm": arm, "width": 8, "blocks": 1, "hidden": 16, "depth": 2}, torch)
            out = model(torch.zeros(1, 1, 12, 12),
                        torch.zeros(1, 20, dtype=torch.long),
                        torch.zeros(1, 20, 2))
            self.assertEqual(tuple(out.shape), (1, 20))

    def test_deployed_bytes_counts_every_tensor(self):
        model = superset = superres_models.make_model(
            {"arm": "edsr", "width": 8, "blocks": 1}, torch)
        counted = superres_models.deployed_bytes(model, torch, "float16")
        expected = sum(int(np.prod(v.shape)) for v in model.state_dict().values()) * 2
        self.assertEqual(counted, expected)


class PatchesDoNotLeak(unittest.TestCase):
    """Training and evaluation tiles must share no fine sample."""

    def test_tiles_are_disjoint_in_the_coarse_grid(self):
        sampler = superres_train.PatchSampler(129, 17, 10, seed=3)
        self.assertEqual(set(sampler.training) & set(sampler.evaluation), set())

    def test_a_patch_larger_than_the_grid_is_refused(self):
        with self.assertRaises(ValueError):
            superres_train.PatchSampler(17, 17, 10, seed=1)


class TheVolumeAndItsSplit(unittest.TestCase):
    """Held-out boreholes and block in the synthetic volume, and its fault."""

    def test_the_fault_displaces_the_stratigraphy(self):
        volume = structure.synthetic_volume(side=24, depth=12, throw_m=180.0)
        hanging = volume["hangingWall"]
        self.assertGreater(hanging.mean(), 0.2)
        self.assertLess(hanging.mean(), 0.8)
        near = volume["distanceToFault"] < 0.02
        self.assertTrue(near.any(), "no samples adjacent to the fault")

    def test_all_units_are_present(self):
        volume = structure.synthetic_volume(side=24, depth=12)
        self.assertEqual(len(set(volume["unit"].tolist())), structure.UNITS)

    def test_held_out_boreholes_and_block_do_not_overlap_training(self):
        volume = structure.synthetic_volume(side=24, depth=12)
        split = structure.borehole_split(volume, boreholes=60, held_out=15, seed=9)
        train = set(split["trainIndex"].tolist())
        for name in ("boreholeTestIndex", "blockTestIndex"):
            self.assertEqual(train & set(split[name].tolist()), set(), name)

    def test_the_block_is_contiguous_in_space(self):
        volume = structure.synthetic_volume(side=24, depth=12)
        split = structure.borehole_split(volume, boreholes=60, held_out=15,
                                         block=(0.55, 0.85), seed=9)
        coordinates = volume["coordinates"][split["blockTestIndex"]]
        self.assertGreaterEqual(float(coordinates[:, 0].min()), 0.55 - 1e-9)
        self.assertLessEqual(float(coordinates[:, 0].max()), 0.85 + 1e-9)
