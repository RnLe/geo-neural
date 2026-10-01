"""Emulator architectures and the properties they must have."""
from __future__ import annotations

import unittest

from geoneural.physics import emulator


class ArchitecturesBuildAtTheReportsSizes(unittest.TestCase):
    """These configurations span the architecture search bounds; a module that
    cannot instantiate them cannot run that search."""

    CONFIGS = (
        {"kind": "unet", "stages": 3, "base": 16, "blocks": 1},
        {"kind": "unet", "stages": 4, "base": 64, "blocks": 2},
        {"kind": "fno", "blocks": 4, "width": 32, "modes": 8},
        {"kind": "fno", "blocks": 6, "width": 64, "modes": 24},
    )

    def fields(self, torch, batch=2, side=64):
        height = torch.randn(batch, side, side)
        boundary = torch.zeros(batch, side, side)
        boundary[:, 0, :] = boundary[:, -1, :] = 1.0
        return emulator.input_channels(height, boundary, torch)

    def test_each_returns_one_channel_at_the_input_resolution(self):
        import torch
        scalars = torch.zeros(2, len(emulator.CONDITIONERS))
        for config in self.CONFIGS:
            model = emulator.make_emulator(config, torch)
            out = model(self.fields(torch), scalars)
            self.assertEqual(tuple(out.shape), (2, 1, 64, 64), config)

    def test_an_unknown_kind_is_refused(self):
        import torch
        with self.assertRaises(ValueError):
            emulator.make_emulator({"kind": "transformer"}, torch)


class ConditioningIsQuietAndReal(unittest.TestCase):
    def test_zero_conditioning_leaves_the_network_near_its_unconditioned_state(self):
        """An ablation against 'no context' only means something if handing the
        network no context reproduces the network without the channel."""
        import torch
        torch.manual_seed(5)
        model = emulator.make_emulator({"kind": "fno", "blocks": 4, "width": 16, "modes": 8}, torch)
        height = torch.randn(1, 32, 32)
        fields = emulator.input_channels(height, torch.zeros(1, 32, 32), torch)
        out = model(fields, torch.zeros(1, len(emulator.CONDITIONERS)))
        self.assertLess(float(out.detach().abs().mean()), 0.05)

    def test_the_conditioning_actually_changes_the_output(self):
        """A FiLM path that did nothing would pass every structural test."""
        import torch
        torch.manual_seed(7)
        model = emulator.make_emulator({"kind": "unet", "stages": 2, "base": 16}, torch)
        with torch.no_grad():
            for parameter in model.conditioner.body[-1].parameters():
                parameter.uniform_(-1.0, 1.0)
        fields = ArchitecturesBuildAtTheReportsSizes().fields(torch, 1, 32)
        low = model(fields, torch.tensor([[-2.0, -2.0, 0.1]]))
        high = model(fields, torch.tensor([[2.0, 2.0, 0.5]]))
        self.assertGreater(float((low - high).detach().abs().mean()), 1e-6)

    def test_only_the_three_dimensionless_groups_are_conditioned_on(self):
        """The teacher's similarity law collapses four parameters to three.
        Feeding the raw four asks the network to rediscover a symmetry we
        already know and lets it fit noise along the redundant direction."""
        self.assertEqual(len(emulator.CONDITIONERS), 3)
        self.assertIn("logFluvialNumber", emulator.CONDITIONERS)
        self.assertIn("logHillslopeNumber", emulator.CONDITIONERS)


class TheSpectralStackHandlesANonPeriodicDomain(unittest.TestCase):
    def test_the_operator_accepts_a_resolution_it_was_not_built_for(self):
        """Resolution independence is the main property of an FNO; without it
        the operator is only a convolution."""
        import torch
        model = emulator.make_emulator({"kind": "fno", "blocks": 2, "width": 16, "modes": 8}, torch)
        scalars = torch.zeros(1, len(emulator.CONDITIONERS))
        for side in (32, 48, 96):
            fields = emulator.input_channels(torch.randn(1, side, side),
                                             torch.zeros(1, side, side), torch)
            self.assertEqual(tuple(model(fields, scalars).shape), (1, 1, side, side))

    def test_coordinates_and_a_boundary_mask_are_supplied(self):
        """A padded spectral stack has no other way to know where the domain edge
        is, and base level is the single most important thing about a landscape
        run: without one the teacher decays to a rising plane."""
        import torch
        fields = emulator.input_channels(torch.randn(1, 16, 16), torch.ones(1, 16, 16), torch)
        self.assertEqual(fields.shape[1], 4)
        self.assertTrue(bool((fields[:, 1] == 1.0).all()))
        self.assertAlmostEqual(float(fields[0, 2, 0, 0]), -1.0, places=5)
        self.assertAlmostEqual(float(fields[0, 2, 0, -1]), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()


class IdentityPlusIncrement(unittest.TestCase):
    """The contract the training and evaluation code rely on."""

    def test_an_untrained_increment_emulator_is_exact_persistence(self):
        import torch
        for config in ({"kind": "unet", "stages": 2, "base": 16}, {"kind": "fno", "blocks": 2, "width": 16, "modes": 8}):
            model = emulator.make_increment(config, torch)
            height = torch.randn(2, 32, 32)
            mask = torch.zeros(2, 32, 32)
            mask[:, 0, :] = mask[:, -1, :] = mask[:, :, 0] = mask[:, :, -1] = 1.0
            with torch.no_grad():
                out = model(emulator.input_channels(height, mask, torch), torch.randn(2, 3))
            self.assertTrue(torch.equal(out[:, 0], height), config)

    def test_fixed_edges_are_exact_for_any_weights(self):
        import torch
        torch.manual_seed(3)
        model = emulator.make_increment({"kind": "unet", "stages": 2, "base": 16}, torch)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.uniform_(-0.5, 0.5)
        height = torch.randn(1, 32, 32)
        mask = torch.zeros(1, 32, 32)
        mask[:, 0, :] = mask[:, -1, :] = mask[:, :, 0] = mask[:, :, -1] = 1.0
        with torch.no_grad():
            out = model(emulator.input_channels(height, mask, torch), torch.randn(1, 3))[:, 0]
        ring = mask > 0.5
        self.assertTrue(torch.equal(out[ring], height[ring]))
        self.assertGreater(float((out[~ring] - height[~ring]).abs().mean()), 1e-4)
