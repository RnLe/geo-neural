"""The network must be charged the way the conventional codec is charged."""
from __future__ import annotations

import unittest

import numpy as np

from geoneural.neural import quantise


class PackingIsHonest(unittest.TestCase):
    """A 6-bit tensor stored one value per byte is an 8-bit tensor."""

    def test_every_width_occupies_exactly_its_bits(self):
        for bits in (8, 6, 4, 3, 2, 1):
            values = (np.arange(37) % (1 << bits)).astype(np.uint8)
            self.assertEqual(len(quantise._pack(values, bits)),
                             -(-37 * bits // 8), f"{bits} bits")

    def test_a_width_outside_one_to_eight_is_refused(self):
        with self.assertRaises(ValueError):
            quantise._pack(np.zeros(4, dtype=np.uint8), 9)


class QuantisationRoundTrips(unittest.TestCase):
    def test_the_error_is_bounded_by_half_a_level(self):
        rng = np.random.default_rng(5)
        values = rng.normal(0.0, 0.3, 2048)
        for bits in (8, 6, 4):
            out = quantise.quantise_tensor(values, bits)
            # Symmetric uniform quantisation puts the scale at max|v| / (levels//2),
            # so no value can be further than half that from its grid point.
            half_level = float(np.abs(values).max()) / (((1 << bits) - 1) // 2) / 2.0
            self.assertLessEqual(out["maxAbsError"], half_level * (1.0 + 1e-9),
                                 f"{bits} bits")

    def test_more_bits_never_costs_more_error(self):
        rng = np.random.default_rng(11)
        values = rng.normal(0.0, 1.0, 4096)
        errors = [quantise.quantise_tensor(values, bits)["maxAbsError"] for bits in (4, 6, 8)]
        self.assertTrue(errors[0] >= errors[1] >= errors[2], errors)


class TheBoundIsNotAFileSize(unittest.TestCase):
    """An entropy estimate without a coder is a bound, not a file size."""

    def state(self):
        rng = np.random.default_rng(3)
        return {"a": rng.normal(0, 0.2, (64, 32)), "b": rng.normal(0, 0.5, (32,))}

    def test_the_coded_length_comes_from_a_real_coder(self):
        encoded = quantise.encode_state(self.state(), 8)
        self.assertGreater(encoded["codedBytes"], 0)
        self.assertIn("zstd", encoded["coder"])
        self.assertIn("NOT a file size", encoded["note"])

    def test_side_information_is_counted_into_the_deployed_total(self):
        """Scales and shapes are payload: small, but not zero."""
        encoded = quantise.encode_state(self.state(), 8)
        self.assertGreater(encoded["sideInformationBytes"], 0)
        self.assertEqual(encoded["deployedBytes"],
                         encoded["codedBytes"] + encoded["sideInformationBytes"])

    def test_fewer_bits_produce_a_smaller_packed_stream(self):
        state = self.state()
        wide = quantise.encode_state(state, 8)["packedBytes"]
        narrow = quantise.encode_state(state, 4)["packedBytes"]
        self.assertAlmostEqual(narrow / wide, 0.5, places=2)


class StraightThroughLetsTheGradientPast(unittest.TestCase):
    """Rounding has zero derivative almost everywhere; without the straight-through
    trick, attaching the quantiser would stop learning."""

    def model(self):
        import torch
        return torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.GELU(),
                                   torch.nn.Linear(8, 1))

    def test_weights_still_receive_gradient_through_the_quantiser(self):
        import torch
        model = self.model()
        attached = quantise.attach(model, 8, torch)
        self.assertEqual(attached, 2)
        out = model(torch.randn(16, 4)).sum()
        out.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(any(float(g.abs().sum()) > 0 for g in grads))

    def test_the_forward_pass_really_is_quantised(self):
        """If the forward pass saw raw weights, fine-tuning would optimise a
        model that is not the one that ships."""
        import torch
        model = self.model()
        with torch.no_grad():
            model[0].weight.fill_(0.0)
            model[0].weight[0, 0] = 1.0
            model[0].weight[0, 1] = 1e-6      # far below one quantisation level
        quantise.attach(model, 4, torch)
        effective = model[0].weight.detach()
        self.assertAlmostEqual(float(effective[0, 1]), 0.0, places=6)

    def test_detaching_bakes_the_quantised_values_in(self):
        import torch
        model = self.model()
        quantise.attach(model, 4, torch)
        quantised = model[0].weight.detach().clone()
        quantise.detach_all(model, torch)
        self.assertTrue(torch.allclose(model[0].weight.detach(), quantised))
        self.assertFalse(hasattr(model[0], "parametrizations"))


if __name__ == "__main__":
    unittest.main()


class LadderHandlesASlopeTerm(unittest.TestCase):
    """A finalist with `slope_weight > 0` must survive the fine-tune.

    The quantisation-aware rung re-trains the model, and a slope term needs a
    mask saying which neighbour pairs lie inside the region being fitted. The
    fine-tune fits the whole field, so the mask is all-true; both the host and
    device paths refuse `None`.
    """

    def test_a_slope_weighted_finalist_reaches_every_width(self):
        from test_search import MISSING, _StubProblem
        if MISSING:
            self.skipTest("torch is not installed")
        problem = _StubProblem(side=65, intervals=32)
        chosen = {"siren-with-slope": {
            "config": {"kind": "siren", "width": 16, "depth": 2,
                       "omega": 6.0, "hidden_omega": 6.0},
            "recipe": {"slope_weight": 0.05, "lr": 1e-3, "loss": "mse"}}}
        report = quantise.ladder(chosen, problem, steps=6, batch=64,
                                 finetune_steps=3, widths=(8, 4),
                                 engine_mode="eager", data_path="host",
                                 sampling="without")
        rows = report["rows"]
        self.assertEqual(len(rows), 1)
        widths = {(row["width"], row["method"]) for row in rows[0]["byWidth"]}
        for bits in (8, 4):
            self.assertIn((f"int{bits}", "quantisation-aware"), widths)
            self.assertIn((f"int{bits}", "post-training"), widths)


class ASurvivorMustActuallyWin(unittest.TestCase):
    """Unbeaten is not the same as winning."""

    def test_a_model_worse_than_the_mean_never_survives(self):
        from test_search import MISSING, _StubProblem
        if MISSING:
            self.skipTest("torch is not installed")
        problem = _StubProblem(side=65, intervals=32)
        floor = quantise.constant_floor(problem)
        self.assertGreater(floor, 0.0)
        # A model cheaper than every conventional point has an empty comparison
        # set, so it must still clear the constant-predictor floor to survive.
        self.assertEqual(
            quantise._survival((False, "worse than the constant predictor")),
            {"survives": False, "whyNot": "worse than the constant predictor"})

    def test_a_residual_finalist_is_charged_for_its_conventional_base(self):
        from test_search import MISSING, _StubProblem
        if MISSING:
            self.skipTest("torch is not installed")
        problem = _StubProblem(side=65, intervals=32)
        chosen = {"residual-t0": {
            "config": {"kind": "residual", "base_level": 1, "base_target_m": 0.5,
                       "inner": {"kind": "siren", "width": 16, "depth": 2,
                                 "omega": 6.0, "hidden_omega": 6.0}},
            "recipe": {"lr": 1e-3, "loss": "mse"}}}
        report = quantise.ladder(chosen, problem, steps=6, batch=64,
                                 finetune_steps=3, widths=(8,),
                                 engine_mode="eager", data_path="host",
                                 sampling="without")
        for row in report["rows"][0]["byWidth"]:
            self.assertGreater(row["baseBytes"], 0, row["width"])
            if "weightBytes" in row:
                self.assertEqual(row["deployedBytes"],
                                 row["weightBytes"] + row["baseBytes"], row["width"])
