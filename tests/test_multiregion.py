"""Multi-region decoder: the index space, the normalisation and the freeze that transfer rests on."""
import unittest

import numpy as np

from geoneural.neural import multiregion


class GlobalIndexSpace(unittest.TestCase):
    """`region * side**2 + local`, unpacked without threading region through fit."""

    SIDE = 129
    INTERVALS = 64

    def test_the_region_falls_out_of_the_index(self):
        area = self.SIDE ** 2
        indexes = np.array([0, area - 1, area, 3 * area + 7], dtype=np.int64)
        _, tiles = multiregion.features(indexes, self.SIDE, self.INTERVALS, True)
        per_region = ((self.SIDE - 1) // self.INTERVALS) ** 2
        self.assertEqual(int(tiles[0]) // per_region, 0)
        self.assertEqual(int(tiles[1]) // per_region, 0)
        self.assertEqual(int(tiles[2]) // per_region, 1)
        self.assertEqual(int(tiles[3]) // per_region, 3)

    def test_the_same_place_in_two_regions_gets_different_codes(self):
        """This is the mechanism of a shared decoder. If these collided, every
        region would be forced through one code and the arm would measure the
        pointwise mean of the panel."""
        area = self.SIDE ** 2
        _, tiles = multiregion.features(np.array([5, area + 5], dtype=np.int64),
                                        self.SIDE, self.INTERVALS, True)
        self.assertNotEqual(int(tiles[0]), int(tiles[1]))

    def test_the_same_place_in_two_regions_gets_the_same_coordinate(self):
        """Region-local coordinates are what make the control work: without a
        code, a decoder sees contradictory targets at identical inputs."""
        area = self.SIDE ** 2
        coords, _ = multiregion.features(np.array([5, area + 5], dtype=np.int64),
                                         self.SIDE, self.INTERVALS, True)
        self.assertTrue(np.allclose(coords[0], coords[1]))

    def test_it_agrees_with_the_single_region_unpacking_on_region_zero(self):
        """Region zero must be the ordinary lattice, or the multi-region numbers
        are not comparable with the single-region numbers they are read against."""
        from geoneural.neural.learning import features as lattice
        indexes = np.array([0, 1, 64, 130, self.SIDE ** 2 - 1], dtype=np.int64)
        for shared in (True, False):
            mine = multiregion.features(indexes, self.SIDE, self.INTERVALS, shared)
            theirs = lattice(indexes, self.SIDE, self.INTERVALS, shared)
            self.assertTrue(np.array_equal(mine[0], theirs[0]), f"coords differ (shared={shared})")
            self.assertTrue(np.array_equal(mine[1], theirs[1]), f"tiles differ (shared={shared})")


class PerRegionNormalisationIsCharged(unittest.TestCase):
    """A panel spanning 15 m to 818 m must not let region identity be the signal."""

    class _Stub:
        """Only what `metres_by_region` and `normalisation_bytes` read."""
        def __init__(self):
            self.area = 4
            self.regions = 2
            self.means = [100.0, 500.0]
            self.scales = [2.0, 50.0]
            self.reference_metres = np.array([100.0, 102.0, 98.0, 100.0,
                                              500.0, 550.0, 450.0, 500.0])
        region_of = multiregion.MultiRegionProblem.region_of
        metres_by_region = multiregion.MultiRegionProblem.metres_by_region
        normalisation_bytes = multiregion.MultiRegionProblem.normalisation_bytes

    def test_each_region_is_returned_to_its_own_metres(self):
        stub = self._Stub()
        # One normalised unit above each region's mean is 2 m in the first region
        # and 50 m in the second. A single scale cannot produce both, which is
        # why `training.evaluate` cannot be used on this problem.
        errors = stub.metres_by_region(np.array([1.0, 1.0]), np.array([0, 4], dtype=np.int64))
        self.assertAlmostEqual(errors[0], 2.0, places=6)
        self.assertAlmostEqual(errors[1], 50.0, places=6)

    def test_the_stored_scalars_are_counted_not_free(self):
        self.assertEqual(self._Stub().normalisation_bytes("float16"), 2 * 2 * 2)
        self.assertEqual(self._Stub().normalisation_bytes("float32"), 2 * 2 * 4)


class FreezingTheBackboneIsReal(unittest.TestCase):
    """Transfer means the backbone does not move. A drifting one measures joint
    training on N regions and calls it transfer."""

    def model(self):
        import torch  # noqa: F401
        from geoneural.neural.models import make_model
        return make_model({"kind": "shared", "tiles": 8, "width": 16, "depth": 2, "latent": 4})

    def test_only_the_code_table_stays_trainable(self):
        import torch
        model = self.model()
        counts = multiregion._freeze_backbone(model, torch)
        self.assertEqual(counts["trainableParameters"], 8 * 4)
        self.assertGreater(counts["frozenParameters"], 0)
        for name, parameter in model.named_parameters():
            self.assertEqual(parameter.requires_grad, name.startswith("codes"), name)

    def test_a_model_with_no_codes_is_refused_rather_than_silently_frozen(self):
        """Freezing everything would train nothing and report a finished run."""
        import torch
        from geoneural.neural.models import make_model
        with self.assertRaises(ValueError):
            multiregion._freeze_backbone(
                make_model({"kind": "siren", "width": 16, "depth": 2}), torch)


class ByteVerdictsAreGatedOnMatchedError(unittest.TestCase):
    """An arm 45x cheaper and sixty times worse must not read as a win."""

    def report(self, worst_max):
        return {"arms": {
            "conventional": {"totalBytes": 380263, "targetM": 1.0,
                             "perRegion": [{"region": "a", "bytes": 285776}]},
            "independent": {"deployedBytes": 13068},
            "shared": {"deployedBytes": 8586,
                       "evaluation": {"selection": {"byRegion": {"a": {"max_m": worst_max}}}}},
        }}

    def test_an_unmatched_arm_is_flagged_and_not_called_comparable(self):
        verdict = multiregion._verdict(self.report(63.8))
        self.assertFalse(verdict["errorsAreMatched"])
        self.assertFalse(verdict["comparableWithConventional"])
        self.assertIn("readThisFirst", verdict)

    def test_a_matched_arm_carries_no_warning(self):
        verdict = multiregion._verdict(self.report(0.8))
        self.assertTrue(verdict["errorsAreMatched"])
        self.assertNotIn("readThisFirst", verdict)

    def test_the_cheapest_conventional_point_at_the_same_error_is_found(self):
        """Could this error have been bought more cheaply conventionally? A curve
        that stopped short of where the neural arm sits would manufacture a win."""
        report = self.report(6.0)
        report["arms"]["conventionalCurve"] = {"targets": [
            {"targetM": 1.0, "totalBytes": 380263, "worstMaxErrorM": 1.0},
            {"targetM": 8.0, "totalBytes": 40000, "worstMaxErrorM": 8.0},
            {"targetM": 4.0, "totalBytes": 60000, "worstMaxErrorM": 4.0},
        ]}
        found = multiregion._verdict(report)["cheapestConventionalAtThisError"]
        # 8.0 guarantees worse than the 6.0 the shared arm achieved, so it is not
        # eligible however cheap it is; 4.0 is the cheapest that qualifies.
        self.assertEqual(found["targetM"], 4.0)
        self.assertEqual(found["totalBytes"], 60000)
        self.assertFalse(found["conventionalIsCheaper"])

    def test_an_arm_far_worse_than_every_conventional_point_still_gets_a_comparison(self):
        """At 500 m error every conventional point qualifies, so the cheapest is
        picked and the arm is cheaper in bytes, which is the situation
        `errorsAreMatched` exists to stop anyone quoting."""
        report = self.report(500.0)
        report["arms"]["conventionalCurve"] = {"targets": [
            {"targetM": 1.0, "totalBytes": 380263, "worstMaxErrorM": 1.0}]}
        verdict = multiregion._verdict(report)
        found = verdict["cheapestConventionalAtThisError"]
        self.assertEqual(found["targetM"], 1.0)
        self.assertFalse(found["conventionalIsCheaper"])
        self.assertGreater(found["timesCheaper"], 0.0)
        self.assertFalse(verdict["errorsAreMatched"])
        self.assertIn("readThisFirst", verdict)

    def test_a_cheaper_transfer_arm_that_misses_the_target_is_not_called_cheaper(self):
        report = self.report(0.5)
        report["arms"]["transfer"] = {
            "marginalBytesForANewRegion": 2052,
            "conventionalBytesForThatRegion": 285776,
            "evaluation": {"byRegion": {"a": {"max_m": 64.2}}}}
        marginal = multiregion._verdict(report)["marginalVsConventionalForThatRegion"]
        self.assertFalse(marginal["atMatchedError"])
        self.assertFalse(marginal["marginalIsCheaper"])
        self.assertGreater(marginal["ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
