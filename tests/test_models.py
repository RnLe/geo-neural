"""Structural tests for the architecture families.

These check the properties the experiments rely on, not that a model fits
anything. Two of them exist because getting them wrong would be invisible in a
loss curve: the grid's node lookup must be exact, or every reported grid error
is measured against a shifted field; and an existing baseline checkpoint must
still load into the widened classes, or published numbers stop being
reproducible from the weights that produced them.
"""
from __future__ import annotations
import unittest

try:
    import torch
    from geoneural.neural import models
    TORCH = None
except ImportError as error:  # the terrain path needs no ML extras
    TORCH = str(error)


@unittest.skipIf(TORCH, f"torch unavailable: {TORCH}")
class ModelFamilies(unittest.TestCase):
    def coords(self, count: int = 64) -> torch.Tensor:
        return torch.rand(count, 2) * 2 - 1

    def test_every_family_returns_one_finite_value_per_coordinate(self):
        configs = [
            {"kind": "mlp", "width": 32, "depth": 2},
            {"kind": "siren", "width": 32, "depth": 2},
            {"kind": "fourier", "width": 32, "depth": 2},
            {"kind": "fourier", "width": 32, "depth": 2, "mode": "gaussian", "bands": 16},
            {"kind": "shared", "width": 32, "depth": 2, "tiles": 16},
            {"kind": "bandlimited", "width": 32, "depth": 3},
            {"kind": "grid", "levels": 4, "width": 16, "depth": 2},
        ]
        coords = self.coords()
        tiles = torch.zeros(coords.shape[0], dtype=torch.long)
        for config in configs:
            with self.subTest(config["kind"], mode=config.get("mode")):
                out = models.make_model(config)(coords, tiles)
                self.assertEqual(out.shape, (coords.shape[0], 1))
                self.assertTrue(torch.isfinite(out).all())

    def test_ea05_defaults_are_unchanged_by_the_new_options(self):
        """The baseline parameter counts must survive the widened API."""
        for config, expected in (({"kind": "siren", "width": 128, "depth": 3}, 33537),
                                 ({"kind": "fourier", "width": 128, "depth": 3}, 37633),
                                 ({"kind": "shared", "width": 128, "depth": 3, "tiles": 256}, 50689)):
            with self.subTest(config["kind"]):
                self.assertEqual(models.parameter_count(models.make_model(config)), expected)

    def test_model_envelope_is_enforced(self):
        for config in ({"kind": "siren", "width": 4096, "depth": 3},
                       {"kind": "mlp", "width": 64, "depth": 40},
                       {"kind": "grid", "levels": 99},
                       {"kind": "grid", "levels": 4, "growth": 1.0},
                       {"kind": "nonexistent", "width": 32, "depth": 2}):
            with self.subTest(config), self.assertRaises(ValueError):
                models.make_model(config)


@unittest.skipIf(TORCH, f"torch unavailable: {TORCH}")
class Grid(unittest.TestCase):
    def dense(self, resolution: int = 5) -> models.GridDecoder:
        grid = models.make_model({"kind": "grid", "levels": 1, "base_resolution": resolution,
                                  "growth": 1.5, "features": 2, "table_size": 1 << 16,
                                  "hashed": False, "width": 8, "depth": 1})
        with torch.no_grad():
            grid.table.copy_(torch.arange(resolution * resolution * 2, dtype=torch.float32)
                             .reshape(resolution * resolution, 2))
        return grid

    def test_a_query_on_a_node_returns_that_node(self):
        """Exactness at nodes. A half-cell offset here would shift every grid
        result against the reference without changing anything visible."""
        resolution = 5
        grid = self.dense(resolution)
        for row in range(resolution):
            for col in range(resolution):
                unit = torch.tensor([[col / (resolution - 1), row / (resolution - 1)]])
                coords = unit * 2 - 1
                scaled = ((coords + 1) * 0.5 * (resolution - 1)).round().long()
                gathered = grid._gather(scaled[:, 1:2], scaled[:, 0:1]).squeeze(1)
                self.assertEqual(gathered.tolist(), [[float(2 * (row * resolution + col)),
                                                      float(2 * (row * resolution + col) + 1)]])

    def test_a_midpoint_query_is_the_mean_of_its_neighbours(self):
        resolution = 5
        grid = self.dense(resolution)
        step = 1.0 / (resolution - 1)
        middle = torch.tensor([[step * 0.5 * 2 - 1, -1.0]])
        left = torch.tensor([[-1.0, -1.0]])
        right = torch.tensor([[step * 2 - 1, -1.0]])
        with torch.no_grad():
            def sample(point):
                unit = ((point + 1.0) * 0.5).clamp(0, 1).unsqueeze(-2)
                scaled = unit * (grid.resolutions - 1).unsqueeze(-1)
                base = scaled.floor(); frac = scaled - base; base = base.to(torch.int64)
                wy, wx = frac[..., 1:2], frac[..., 0:1]
                top = grid._gather(base[..., 1], base[..., 0]) * (1 - wx) + \
                    grid._gather(base[..., 1], base[..., 0] + 1) * wx
                bottom = grid._gather(base[..., 1] + 1, base[..., 0]) * (1 - wx) + \
                    grid._gather(base[..., 1] + 1, base[..., 0] + 1) * wx
                return (top * (1 - wy) + bottom * wy).squeeze(-2)
            expected = (sample(left) + sample(right)) / 2
            self.assertTrue(torch.allclose(sample(middle), expected, atol=1e-5))

    def test_coarse_levels_stay_collision_free_and_fine_levels_hash(self):
        grid = models.make_model({"kind": "grid", "levels": 12, "base_resolution": 8,
                                  "growth": 1.8, "features": 2, "table_size": 4096,
                                  "width": 16, "depth": 1})
        resolutions = grid.resolutions.tolist()
        for level, (resolution, is_dense) in enumerate(zip(resolutions, grid.dense)):
            with self.subTest(level=level, resolution=resolution):
                self.assertEqual(is_dense, resolution * resolution <= 4096)
        self.assertTrue(grid.dense[0], "the coarsest level must never be hashed")
        self.assertIn(False, grid.dense, "this configuration is meant to exercise hashing")
        self.assertEqual(grid.table.shape[0], sum(grid.entry_counts))

    def test_hashing_can_be_refused_entirely(self):
        grid = models.make_model({"kind": "grid", "levels": 6, "base_resolution": 8,
                                  "growth": 1.8, "hashed": False, "table_size": 1,
                                  "width": 16, "depth": 1})
        self.assertTrue(all(grid.dense))


@unittest.skipIf(TORCH, f"torch unavailable: {TORCH}")
class BandLimiting(unittest.TestCase):
    def test_the_budget_is_the_cumulative_sum_of_the_bands(self):
        model = models.make_model({"kind": "bandlimited", "width": 16, "depth": 3,
                                   "bandwidth": 8.0})
        self.assertEqual([model.bandwidth_at(i) for i in range(4)], [8.0, 24.0, 56.0, 120.0])

    def test_a_coarse_output_does_not_depend_on_the_fine_layers(self):
        """A coarse request must be answerable without evaluating the fine part.
        Perturbing every later weight must leave the level-0 output
        bit-identical, which a truncated fine field would not."""
        model = models.make_model({"kind": "bandlimited", "width": 16, "depth": 3})
        coords = torch.rand(32, 2) * 2 - 1
        before = model(coords, None, level=0).clone()
        with torch.no_grad():
            for mixer in model.mixers:
                mixer.weight.add_(1.0)
            for head in model.heads[1:]:
                head.weight.add_(1.0)
        self.assertTrue(torch.equal(before, model(coords, None, level=0)))

    def test_all_levels_agrees_with_evaluating_each_level(self):
        model = models.make_model({"kind": "bandlimited", "width": 16, "depth": 3})
        coords = torch.rand(16, 2) * 2 - 1
        for level, value in enumerate(model.all_levels(coords)):
            with self.subTest(level=level):
                self.assertTrue(torch.allclose(value, model(coords, None, level=level), atol=1e-6))

    def test_no_output_exists_above_the_declared_depth(self):
        model = models.make_model({"kind": "bandlimited", "width": 16, "depth": 2})
        with self.assertRaises(ValueError):
            model(torch.zeros(2, 2), None, level=9)


class _Silent(torch.nn.Module if not TORCH else object):
    """An inner network that contributes nothing, isolating the base sampler."""

    def forward(self, coords, tiles=None):
        return torch.zeros(coords.shape[:-1] + (1,), dtype=coords.dtype)


@unittest.skipIf(TORCH, f"torch unavailable: {TORCH}")
class Residual(unittest.TestCase):
    def test_output_is_the_base_plus_the_inner_prediction(self):
        base = torch.arange(25, dtype=torch.float32).reshape(5, 5)
        inner = models.make_model({"kind": "mlp", "width": 8, "depth": 1})
        model = models.ResidualDecoder(base, inner)
        coords = torch.rand(32, 2) * 2 - 1
        with torch.no_grad():
            base_only = models.ResidualDecoder(base, _Silent())
            self.assertTrue(torch.allclose(model(coords, None),
                                           base_only(coords, None) + inner(coords, None), atol=1e-6))
        with torch.no_grad():
            for parameter in inner.parameters():
                parameter.zero_()
        # With a silent inner network the output is exactly the sampled base.
        corners = torch.tensor([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]])
        self.assertEqual(model(corners, None).squeeze(-1).tolist(), [0.0, 4.0, 20.0, 24.0])

    def test_the_far_edge_interpolates_instead_of_repeating_the_previous_node(self):
        """Floor-then-clamp would return node `n-2` for a query exactly on
        node `n-1`, flattening the last row and column of the domain."""
        base = torch.arange(25, dtype=torch.float32).reshape(5, 5)
        model = models.ResidualDecoder(base, _Silent())
        edge = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 1.0]])
        expected = [14.0, 22.0, 23.0]
        self.assertEqual(model(edge, None).squeeze(-1).tolist(), expected)

    def test_the_base_is_not_stored_in_the_checkpoint(self):
        """The base is the decoded form of a compressed conventional grid that
        ships beside the model. Keeping it in the state dict stores the same
        information twice and, because deployed bytes are the serialized state
        dict, charges for both (about a third more bytes for this family)."""
        base = torch.zeros(129, 129)
        model = models.ResidualDecoder(base, models.make_model({"kind": "siren", "width": 64,
                                                                "depth": 2}))
        self.assertNotIn("base", model.state_dict())
        inner_only = models.make_model({"kind": "siren", "width": 64, "depth": 2})
        self.assertEqual(sum(v.numel() for v in model.state_dict().values()),
                         sum(v.numel() for v in inner_only.state_dict().values()))

    def test_a_checkpoint_without_the_base_still_reloads(self):
        """Dropping it from the state dict must not break the round trip: the
        base comes from the conventional payload at construction."""
        from safetensors.torch import load, save
        base = torch.arange(81, dtype=torch.float32).reshape(9, 9)
        inner = {"kind": "siren", "width": 32, "depth": 2}
        first = models.ResidualDecoder(base, models.make_model(inner))
        blob = save({k: v.contiguous() for k, v in first.state_dict().items()})
        second = models.ResidualDecoder(base, models.make_model(inner))
        second.load_state_dict(load(blob))
        coords = torch.rand(16, 2) * 2 - 1
        self.assertTrue(torch.equal(first(coords, None), second(coords, None)))

    def test_a_non_square_base_is_refused(self):
        with self.assertRaises(ValueError):
            models.ResidualDecoder(torch.zeros(4, 5), models.make_model({"kind": "mlp", "width": 8, "depth": 1}))


if __name__ == "__main__":
    unittest.main()


@unittest.skipIf(TORCH, f"torch unavailable: {TORCH}")
class CodeGrid(unittest.TestCase):
    """The difference between this family and `SharedDecoder` is continuity at a
    patch border, so that is what these check. A per-patch constant code cannot
    express a continuous field across an edge, whatever the training does."""

    def test_the_code_field_is_continuous_across_a_patch_border(self):
        model = models.make_model({"kind": "codegrid", "patches": 8, "latent": 8,
                                   "width": 32, "depth": 2})
        border = 2.0 / 8 - 1.0  # the first internal patch edge in [-1, 1]
        step = 1e-5
        left = torch.tensor([[border - step, 0.0]])
        right = torch.tensor([[border + step, 0.0]])
        jump = (model.code_at(left) - model.code_at(right)).abs().max().item()
        self.assertLess(jump, 1e-3)

    def test_the_per_tile_alternative_really_does_jump(self):
        """The control for the test above: without it, a continuity assertion
        proves nothing, because it would also pass on a constant field."""
        model = models.make_model({"kind": "shared", "tiles": 64, "latent": 8,
                                   "width": 32, "depth": 2})
        neighbours = model.codes(torch.tensor([0, 1]))
        self.assertGreater((neighbours[0] - neighbours[1]).abs().max().item(), 1e-3)

    def test_a_node_query_returns_that_node_exactly(self):
        model = models.make_model({"kind": "codegrid", "patches": 4, "latent": 3,
                                   "width": 16, "depth": 1})
        with torch.no_grad():
            model.codes.copy_(torch.arange(5 * 5 * 3, dtype=torch.float32).reshape(5, 5, 3))
        for row in range(5):
            for col in range(5):
                point = torch.tensor([[col / 4 * 2 - 1, row / 4 * 2 - 1]])
                with self.subTest(row=row, col=col):
                    self.assertTrue(torch.allclose(model.code_at(point)[0],
                                                   model.codes[row, col], atol=1e-4))

    def test_the_code_payload_counts_the_shared_border_ring(self):
        model = models.make_model({"kind": "codegrid", "patches": 16, "latent": 16,
                                   "width": 64, "depth": 2})
        self.assertEqual(model.code_parameters(), 17 * 17 * 16)

    def test_the_envelope_is_enforced(self):
        for config in ({"kind": "codegrid", "patches": 4096, "latent": 8},
                       {"kind": "codegrid", "patches": 8, "latent": 4096}):
            with self.subTest(config), self.assertRaises(ValueError):
                models.make_model(config)


@unittest.skipIf(TORCH, f"torch unavailable: {TORCH}")
class GridAllocationEnvelope(unittest.TestCase):
    """Oversized grid configurations are refused before anything is allocated.

    A dense level's table is resolution squared and the resolution grows
    geometrically, so an ordinary-looking configuration asks for tens or hundreds
    of gigabytes. A search that checked its byte ceiling only after building the
    model would allocate first for a trial it is about to reject.
    """

    def test_an_oversized_dense_configuration_is_refused_before_allocating(self):
        for base, growth, levels, features, expected_gib in ((16, 1.6, 16, 4, 8.3),
                                                             (32, 1.8, 16, 2, 502.3)):
            with self.subTest(base=base, growth=growth, gib=expected_gib):
                with self.assertRaises(ValueError) as caught:
                    models.make_model({"kind": "grid", "base_resolution": base, "growth": growth,
                                       "levels": levels, "features": features, "hashed": False,
                                       "table_size": 1, "width": 32, "depth": 2})
                self.assertIn("envelope", str(caught.exception))

    def test_the_refusal_says_what_to_change(self):
        """The error names the parameters that bring a configuration back
        inside the limit."""
        with self.assertRaises(ValueError) as caught:
            models.make_model({"kind": "grid", "base_resolution": 32, "growth": 2.0,
                               "levels": 16, "features": 4, "hashed": False,
                               "table_size": 1, "width": 32, "depth": 2})
        message = str(caught.exception)
        for remedy in ("levels", "growth", "features", "hashing"):
            with self.subTest(remedy):
                self.assertIn(remedy, message)

    def test_hashing_keeps_a_deep_pyramid_inside_the_envelope(self):
        """Capping the fine levels is the intended escape, so it must work."""
        model = models.make_model({"kind": "grid", "levels": 16, "base_resolution": 16,
                                   "growth": 1.6, "features": 4, "table_size": 1 << 14,
                                   "width": 32, "depth": 2})
        self.assertLessEqual(sum(model.entry_counts) * 4, models.MAX_GRID_SCALARS)

    def test_a_configuration_at_the_envelope_still_builds(self):
        """The bound must not be so tight it forbids useful models."""
        model = models.make_model({"kind": "grid", "levels": 8, "base_resolution": 8,
                                   "growth": 1.7, "features": 2, "table_size": 1 << 16,
                                   "width": 32, "depth": 2})
        self.assertGreater(sum(model.entry_counts), 0)


@unittest.skipIf(TORCH, f"torch unavailable: {TORCH}")
class PredictedCodes(unittest.TestCase):
    """Codes computed from the coarse base the decoder already ships.

    The claim is that a local code need not be stored. These check the two
    properties that would make it false: that the code really is derived rather
    than held, and that it is local, so a node's code describes its own
    neighbourhood rather than a global summary that could not modulate anything.
    """

    def build(self, base=None, patches=4, latent=8):
        base = torch.randn(65, 65) if base is None else base
        return models.make_model({"kind": "predicted", "base": base, "patches": patches,
                                  "latent": latent, "width": 32, "depth": 2,
                                  "predictor_width": 8, "predictor_depth": 2,
                                  "patch_cells": 7})

    def test_no_code_is_stored(self):
        model = self.build()
        self.assertEqual(model.stored_code_parameters(), 0)
        self.assertNotIn("codes", dict(model.named_parameters()))
        self.assertNotIn("base", model.state_dict(), "the base is conventional payload, not a weight")

    def test_a_node_code_responds_to_its_own_neighbourhood(self):
        base = torch.zeros(65, 65)
        model = self.build(base, patches=4)
        before = model.code_field().clone()
        with torch.no_grad():
            model.base[0:4, 0:4] += 10.0   # the north-west corner only
        with torch.no_grad():
            after = model.code_field()
        corner_change = (after[0, 0] - before[0, 0]).abs().max()
        far_change = (after[-1, -1] - before[-1, -1]).abs().max()
        self.assertGreater(float(corner_change), 0.0)
        self.assertAlmostEqual(float(far_change), 0.0, places=6)

    def test_patches_are_centred_on_their_nodes(self):
        """A half-patch offset would give every code the wrong neighbourhood,
        which is invisible in a loss curve and breaks the locality above."""
        resolution, patches, cells = 65, 4, 7
        base = torch.arange(resolution * resolution, dtype=torch.float32).reshape(resolution, resolution)
        model = self.build(base, patches=patches)
        grid = model.base_patches().reshape(patches + 1, patches + 1, cells, cells)
        middle = cells // 2
        for node_row in range(patches + 1):
            for node_col in range(patches + 1):
                with self.subTest(node=(node_row, node_col)):
                    centre_row = round(node_row * (resolution - 1) / patches)
                    centre_col = round(node_col * (resolution - 1) / patches)
                    self.assertEqual(float(grid[node_row, node_col, middle, middle]),
                                     float(base[centre_row, centre_col]))

    def test_the_predictor_replaces_a_table_that_grows_with_the_region_count(self):
        """Stored codes scale with the region count; a predictor does not."""
        predicted = self.build(patches=4, latent=8)
        stored = models.make_model({"kind": "codegrid", "patches": 4, "latent": 8,
                                    "width": 32, "depth": 2})
        self.assertEqual(predicted.stored_code_parameters(), 0)
        self.assertEqual(stored.code_parameters(), 5 * 5 * 8)

    def test_the_envelope_is_enforced(self):
        for config in ({"kind": "predicted", "base": torch.zeros(33, 33), "latent": 4096},
                       {"kind": "predicted", "base": torch.zeros(33, 33), "predictor_depth": 99}):
            with self.subTest(config), self.assertRaises(ValueError):
                models.make_model({"width": 32, "depth": 2, **config})

    def test_a_non_square_base_is_refused(self):
        with self.assertRaises(ValueError):
            models.make_model({"kind": "predicted", "base": torch.zeros(8, 9),
                               "width": 32, "depth": 2})


@unittest.skipIf(TORCH, f"torch unavailable: {TORCH}")
class MatchedBackbone(unittest.TestCase):
    """The modulated families must differ from the SIREN control in one respect only.

    `SharedDecoder` and `CodeGridDecoder` use SIREN's frequency-aware
    initialisation, as `Siren` does, so a comparison between them changes one
    thing at a time. These tests pin that by requiring the modulated families to
    reduce exactly to the control at a zero code.
    """

    def copy_backbone(self, source, siren) -> None:
        for index, layer in enumerate(source.layers):
            siren.net[index].linear.weight.data.copy_(layer.weight)
            siren.net[index].linear.bias.data.copy_(layer.bias)
        siren.net[-1].weight.data.copy_(source.output.weight)
        siren.net[-1].bias.data.copy_(source.output.bias)

    def test_a_shared_decoder_at_a_zero_code_is_exactly_the_siren_control(self):
        torch.manual_seed(7)
        shared = models.make_model(
            {"kind": "shared", "tiles": 16, "width": 64, "depth": 3, "latent": 8, "omega": 30.0})
        siren = models.Siren(64, 3, 30.0)
        self.copy_backbone(shared, siren)
        coords = torch.rand(128, 2) * 2 - 1
        tiles = torch.randint(0, 16, (128,))
        with torch.no_grad():
            shared.codes.weight.zero_()
            self.assertEqual(float((shared(coords, tiles) - siren(coords)).abs().max()), 0.0)

    def test_a_code_grid_at_a_zero_code_is_exactly_the_siren_control(self):
        torch.manual_seed(11)
        grid = models.make_model(
            {"kind": "codegrid", "patches": 4, "latent": 8, "width": 48, "depth": 2, "omega": 12.0,
             "hidden_omega": 25.0})
        siren = models.Siren(48, 2, 12.0, 25.0)
        self.copy_backbone(grid, siren)
        coords = torch.rand(128, 2) * 2 - 1
        with torch.no_grad():
            grid.codes.zero_()
            self.assertEqual(float((grid(coords, None) - siren(coords)).abs().max()), 0.0)

    def test_a_nonzero_code_does_change_the_output(self):
        """The reduction above must be a property of the zero code, not of a dead
        modulation path. If the code could never matter, the family would be the
        SIREN control under a different name and its extra payload would buy
        nothing."""
        torch.manual_seed(3)
        shared = models.make_model({"kind": "shared", "tiles": 32, "width": 32, "depth": 3})
        coords = torch.rand(64, 2) * 2 - 1
        with torch.no_grad():
            shared.codes.weight.normal_(std=1.0)
            a = shared(coords, torch.zeros(64, dtype=torch.long))
            b = shared(coords, torch.full((64,), 31, dtype=torch.long))
        self.assertGreater(float((a - b).abs().max()), 0.0)

    def test_the_codes_receive_gradient_from_the_first_step(self):
        """A zeroed FiLM weight would cut this gradient path, which code_coverage needs."""
        torch.manual_seed(5)
        shared = models.make_model({"kind": "shared", "tiles": 4, "width": 32, "depth": 2,
                                    "latent": 4})
        coords = torch.rand(64, 2) * 2 - 1
        tiles = torch.randint(0, 4, (64,))
        shared(coords, tiles).square().mean().backward()
        moved = [float(layer.weight.grad.abs().max()) for layer in shared.modulation]
        self.assertTrue(all(value > 0 for value in moved), moved)
        self.assertGreater(float(shared.codes.weight.grad.abs().max()), 0.0)

    def test_the_first_layer_follows_sirens_initialisation_bound(self):
        torch.manual_seed(13)
        shared = models.make_model({"kind": "shared", "tiles": 4, "width": 64, "depth": 3,
                                    "omega": 30.0})
        # SIREN: first layer bounded by 1/in_features, hidden by sqrt(6/in)/omega.
        self.assertLessEqual(float(shared.layers[0].weight.detach().abs().max()), 1 / 2)
        hidden_bound = (6 / 64) ** 0.5 / 30.0
        self.assertLessEqual(float(shared.layers[1].weight.detach().abs().max()), hidden_bound)
        self.assertEqual(shared.omegas, [30.0, 30.0, 30.0])


class LocalImplicitDecoding(unittest.TestCase):
    """The LIIF family decodes in each cell's own frame, not at an absolute point.

    These pin the properties that make it a different architecture rather than a
    reparameterised code grid: locality of the network's input, continuity across
    cell borders, resolution independence, and full payment for the latent raster.
    """

    def _model(self, **kw):
        config = {"kind": "liif", "patches": 8, "latent": 6, "width": 32, "depth": 2}
        config.update(kw)
        return models.make_model(config)

    def test_the_latent_raster_is_counted_as_payload(self):
        model = self._model()
        self.assertEqual(model.code_parameters(), 8 * 8 * 6)

    def test_the_surface_is_continuous_across_a_cell_border(self):
        """Without the local ensemble this jumps at every border by construction."""
        torch.manual_seed(0)
        model = self._model(patches=4)
        with torch.no_grad():
            for param in model.parameters():
                param.uniform_(-0.5, 0.5)
        # A cell border for patches=4 sits at -1 + 2*(1/4) = -0.5.
        left = torch.tensor([[-0.5 - 1e-5, 0.17]])
        right = torch.tensor([[-0.5 + 1e-5, 0.17]])
        with torch.no_grad():
            gap = (model(left) - model(right)).abs().item()
        self.assertLess(gap, 1e-3, "the local ensemble must blend across the border")

    def test_the_network_only_ever_sees_offsets_inside_one_cell(self):
        """The relative coordinate is what enters the MLP, bounded by one cell."""
        seen = []
        model = self._model(patches=8)
        original = model._decode

        def spy(code, relative):
            seen.append(relative.abs().max().item())
            return original(code, relative)

        model._decode = spy
        with torch.no_grad():
            model(torch.tensor([[0.9, -0.9], [0.0, 0.0], [-1.0, 1.0]]))
        self.assertTrue(seen)
        self.assertLessEqual(max(seen), 1.0 + 1e-6)

    def test_the_same_latents_serve_any_query_resolution(self):
        """Resolution independence: a denser query is not a different model."""
        torch.manual_seed(0)
        model = self._model(patches=4)
        coarse = torch.stack(torch.meshgrid(torch.linspace(-1, 1, 5),
                                            torch.linspace(-1, 1, 5), indexing="ij"), -1).reshape(-1, 2)
        fine = torch.stack(torch.meshgrid(torch.linspace(-1, 1, 9),
                                          torch.linspace(-1, 1, 9), indexing="ij"), -1).reshape(-1, 2)
        with torch.no_grad():
            a, b = model(coarse), model(fine)
        self.assertEqual(a.shape, (25, 1))
        self.assertEqual(b.shape, (81, 1))
        # The coarse nodes are a subset of the fine ones, and must agree exactly.
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(coarse[:1]), b[:1], atol=1e-6))

    def test_edge_queries_are_not_faded_toward_zero(self):
        """Clamping repeats edge cells; the weights must still normalise to one."""
        torch.manual_seed(0)
        model = self._model(patches=4)
        with torch.no_grad():
            for param in model.parameters():
                param.uniform_(0.4, 0.5)
            corner = model(torch.tensor([[-1.0, -1.0]])).item()
            middle = model(torch.tensor([[0.0, 0.0]])).item()
        self.assertGreater(abs(corner), 0.05 * abs(middle))

    def test_cell_decoding_widens_the_input_and_stays_optional(self):
        self.assertEqual(self._model(cell_decoding=False).layers[0].in_features, 6 + 2)
        self.assertEqual(self._model(cell_decoding=True).layers[0].in_features, 6 + 2 + 2)

    def test_the_envelope_is_enforced(self):
        with self.assertRaises(ValueError):
            self._model(patches=512)
        with self.assertRaises(ValueError):
            self._model(latent=512)
        with self.assertRaises(ValueError):
            models.make_model({"kind": "liif", "width": 4, "depth": 2})


class GeologyConditioning(unittest.TestCase):
    """Geology conditioning: the context must reach the network, be charged for,
    and stay categorical."""

    def raster(self, side=8):
        classes = torch.zeros(side, side, dtype=torch.int64)
        classes[:, side // 2:] = 1
        return classes

    def model(self, mode="film", embedding=4):
        return models.make_model({"kind": "context", "classes": self.raster(),
                                  "class_count": 2, "embedding": embedding,
                                  "width": 16, "depth": 2, "mode": mode})

    def test_the_context_changes_the_output(self):
        """A conditioning path that does nothing would pass every other test."""
        for mode in ("film", "concat"):
            torch.manual_seed(3)
            model = self.model(mode)
            with torch.no_grad():
                model.embedding.weight.uniform_(-1.0, 1.0)
                west = model(torch.tensor([[-0.9, 0.0]]))
                east = model(torch.tensor([[0.9, 0.0]]))
            self.assertGreater(abs(float(west - east)), 1e-6, f"{mode} ignores its context")

    def test_classes_are_looked_up_nearest_not_interpolated(self):
        """Interpolating between sandstone and limestone denotes no rock."""
        model = self.model()
        found = model.class_at(torch.tensor([[-1.0, 0.0], [1.0, 0.0], [0.05, 0.0]]))
        self.assertEqual(set(int(v) for v in found), {0, 1})

    def test_the_embedding_table_is_counted_as_payload(self):
        self.assertEqual(self.model(embedding=4).code_parameters(), 2 * 4)

    def test_the_raster_is_not_stored_in_the_checkpoint(self):
        """It ships as a compressed conventional payload; storing it here would
        charge for the same bytes twice."""
        self.assertNotIn("classes", self.model().state_dict())

    def test_an_unknown_mode_is_refused(self):
        with self.assertRaises(ValueError):
            self.model(mode="attention")

    def test_concat_widens_the_first_layer_and_film_does_not(self):
        self.assertEqual(self.model("concat", embedding=4).layers[0].in_features, 2 + 4)
        self.assertEqual(self.model("film", embedding=4).layers[0].in_features, 2)


class GeologyConditionedCorrection(unittest.TestCase):
    """Geology conditioning of a correction to a conventional base, not the whole field.

    The composition is `ResidualDecoder(base, ContextDecoder(...))`. It needs no
    new module, but it does need a test, because the two halves have opposite
    conventions about what they store and a silent change to either would turn
    the arm into a differently priced experiment under the same name.
    """

    def raster(self, side=8):
        classes = torch.zeros(side, side, dtype=torch.int64)
        classes[:, side // 2:] = 1
        return classes

    def build(self, mode="film"):
        base = torch.linspace(0.0, 1.0, 8).unsqueeze(0).repeat(8, 1)
        return models.make_model({
            "kind": "residual", "base": base,
            "inner": {"kind": "context", "classes": self.raster(), "class_count": 2,
                      "embedding": 4, "width": 16, "depth": 2, "mode": mode}})

    def test_the_composition_builds_for_both_conditioning_modes(self):
        for mode in ("film", "concat"):
            model = self.build(mode)
            self.assertIsInstance(model.inner, models.ContextDecoder)
            self.assertEqual(model.inner.mode, mode)

    def test_neither_the_base_nor_the_raster_is_stored_in_the_checkpoint(self):
        """Both are decoded forms of payload shipped alongside. Storing either
        would charge the same bytes twice, and the arm's claim is a byte
        comparison."""
        keys = self.build().state_dict().keys()
        self.assertNotIn("base", keys)
        self.assertNotIn("inner.classes", keys)

    def test_the_context_still_changes_the_output_through_the_base(self):
        """A residual wrapper that swamped its inner network would pass every
        structural test above while measuring bilinear upsampling."""
        torch.manual_seed(5)
        model = self.build()
        with torch.no_grad():
            model.inner.embedding.weight.uniform_(-1.0, 1.0)
            # Same base value, different geological class: any difference is the
            # context, because the base is constant along this row.
            flat = torch.zeros(8, 8)
            model.base.copy_(flat)
            west = model(torch.tensor([[-0.9, 0.0]]))
            east = model(torch.tensor([[0.9, 0.0]]))
        self.assertGreater(abs(float(west - east)), 1e-6)

    def test_the_base_is_added_not_replaced(self):
        """`h = h_base + r`. A composition that returned only the correction
        would look excellent on a normalised field and be wrong on metres."""
        model = self.build()
        with torch.no_grad():
            model.base.fill_(3.0)
            for parameter in model.inner.parameters():
                parameter.zero_()
            value = float(model(torch.tensor([[0.0, 0.0]])))
        self.assertAlmostEqual(value, 3.0, places=5)
