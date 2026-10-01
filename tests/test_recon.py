"""Properties the reconstruction results depend on: operator identity, observation consistency, tiled inference
equal to whole-field inference, windows equal to crops, and no path from hidden heights into a prediction."""
from __future__ import annotations

import unittest

import numpy as np

from geoneural.recon import baselines, fields, operators
from geoneural.superres import superres

try:
    import torch

    from geoneural.recon import blocks, fine, models, train
    MISSING = False
except ImportError:  # pragma: no cover
    MISSING = True


def _terrain(side, seed=3):
    rng = np.random.default_rng(seed)
    return np.cumsum(np.cumsum(rng.normal(0.0, 0.3, (side, side)), 0), 1) + 100.0


class OperatorsAreWhatTheySay(unittest.TestCase):

    def test_the_trapezoid_is_the_declared_superres_operator(self):
        z = _terrain(41)
        np.testing.assert_allclose(operators.make("trapezoid", 4).observe(z), superres.observe(z, 4), atol=1e-12)

    def test_a_changed_weight_changes_the_fingerprint(self):
        op = operators.make("gaussian-0.5", 4)

        def nudged(side, factor, sigma=0.5, shift=0.0):
            w = operators.gaussian_axis(side, factor, sigma, shift)
            w[1, 4] += 1e-6
            return w

        other = operators.Operator(op.name, op.factor, nudged, op.params)
        self.assertNotEqual(op.fingerprint(), other.fingerprint())
        self.assertEqual(op.fingerprint(), operators.make("gaussian-0.5", 4).fingerprint())

    def test_back_projection_reports_what_it_achieved_under_every_operator(self):
        """Converged where the operator allows it; where it does not (a Gaussian a full cell wide nearly erases
        the coarse Nyquist band), the report says so and its residual is the true one."""
        z = _terrain(65)
        for name in operators.EVALUATION:
            op = operators.make(name, 4)
            y = op.observe(z)
            fixed, report = op.project(operators.upsample(y, 4, "bspline"), y, tolerance=1e-5)
            achieved = float(np.abs(op.observe(fixed) - y).max())
            self.assertAlmostEqual(achieved, report["achievedMaxM"], places=12)
            self.assertEqual(report["converged"], achieved <= 1e-5, name)
            if name != "gaussian-1.0":
                self.assertTrue(report["converged"], name)


class WindowsEqualCrops(unittest.TestCase):

    def test_interpolation_windows(self):
        y = _terrain(30)
        for kind in ("keys", "bspline", "bilinear"):
            full = operators.upsample(y, 10, kind)
            part = operators.upsample_window(operators.coefficients(y, kind), 10, kind, slice(41, 200), slice(7, 151))
            np.testing.assert_allclose(part, full[41:200, 7:151], atol=1e-10, err_msg=kind)

    def test_phase_kernels_reproduce_their_definition(self):
        y = _terrain(20)
        kernels = np.random.default_rng(0).normal(size=(4, 4, baselines.K, baselines.K))
        out = baselines.apply_phase_kernels(y, kernels, 4)
        padded = np.pad(y, baselines.K, mode="reflect")
        o = np.arange(-baselines.LO, baselines.K - baselines.LO)
        j, k = 7, 11
        window = padded[np.ix_(j + baselines.K + o, k + baselines.K + o)]
        self.assertAlmostEqual(out[4 * j + 1, 4 * k + 3], float((kernels[1, 3] * window).sum()), places=9)

    @unittest.skipIf(MISSING, "torch is not installed")
    def test_device_spline_windows_equal_the_whole_field_spline(self):
        y = _terrain(40)
        windows = fine.SplineWindows(51, "cpu")
        padded = torch.tensor(np.pad(operators.coefficients(y, "bspline"), 2, mode="reflect")[None],
                              dtype=torch.float32)
        got = windows(padded, np.array([[0, 7, 3], [0, 0, 0]]))
        full = operators.upsample(y, 10, "bspline")
        np.testing.assert_allclose(got[0].numpy(), full[70:121, 30:81], atol=1e-3)
        np.testing.assert_allclose(got[1].numpy(), full[0:51, 0:51], atol=1e-3)


@unittest.skipIf(MISSING, "torch is not installed")
class TilesEqualTheWholeField(unittest.TestCase):

    def test_tiled_prediction_equals_one_window(self):
        torch.manual_seed(0)
        spec = models.Features(scale_sigma=4.0, highpass_sigma=2.0, unit=4.0, factor=4, margin=16)
        model = models.UNet(spec.base_channels(), 8).eval()
        for p in model.parameters():
            p.data.normal_(0.0, 0.1)
        base = _terrain(96)
        whole, _ = train.predict(model, spec, base, tile=96, halo=64, device="cpu")
        tiled, _ = train.predict(model, spec, base, tile=32, halo=64, device="cpu")
        self.assertLess(float(np.abs(whole - tiled).max()), 1e-4 * float(np.abs(whole).max()) + 1e-6)


class NothingHiddenReachesAPrediction(unittest.TestCase):

    def test_folds_keep_test_and_validation_out_of_training(self):
        for fold in fields.folds():
            self.assertFalse(set(fold["test"]) & set(fold["train"]))
            self.assertFalse(set(fold["validation"]) & set(fold["train"]))
            self.assertFalse(set(fold["validation"]) & set(fold["test"]))

    def test_classical_hole_fills_never_read_the_hole(self):
        z = _terrain(80)
        poisoned = z.copy()
        poisoned[30:62, 20:52] = 1e6
        for kind in ("harmonic", "biharmonic"):
            a = baselines.fill_hole(z[28:64, 18:54], 2, 2, 32, kind)
            b = baselines.fill_hole(poisoned[28:64, 18:54], 2, 2, 32, kind)
            np.testing.assert_array_equal(a, b)
        model = {"family": "matern32", "sill": 10.0, "rangeCells": 20.0, "nugget": 0.0}
        np.testing.assert_array_equal(blocks.kriging_fill(z, 30, 20, 32, model)[0],
                                      blocks.kriging_fill(poisoned, 30, 20, 32, model)[0])

    @unittest.skipIf(MISSING, "torch is not installed")
    def test_neural_hole_fill_never_reads_the_hole(self):
        torch.manual_seed(0)
        model = models.UNet(blocks.SPEC.base_channels() + 2, 8).eval()
        z = _terrain(300)
        poisoned = z.copy()
        poisoned[134:166, 134:166] = -5e5
        a, _ = blocks.neural_fill(model, z, [(134, 134)], 32, None, "cpu")
        b, _ = blocks.neural_fill(model, poisoned, [(134, 134)], 32, None, "cpu")
        np.testing.assert_array_equal(a[0], b[0])

    @unittest.skipIf(MISSING, "torch is not installed")
    def test_fine_training_windows_stay_out_of_the_validation_band(self):
        span = fine.CORE + 2 * fine.SPEC.margin + 1
        top = (fine.VALIDATION_ROW - fine.BUFFER - span) // fine.FACTOR
        self.assertLessEqual(top * fine.FACTOR + span, fine.VALIDATION_ROW - fine.BUFFER)
        self.assertGreaterEqual(min(r for r, _ in fine.VALIDATION_WINDOWS), fine.VALIDATION_ROW)


if __name__ == "__main__":
    unittest.main()
