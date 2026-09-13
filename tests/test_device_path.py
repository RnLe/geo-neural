"""The GPU-resident path must be the same experiment, measured differently."""
from __future__ import annotations

import unittest

import numpy as np

from geoneural.neural import device_data



class _StubProblem:
    """The surface `DeviceTables` reads. CPU, so this runs without a GPU."""

    def __init__(self, side=17, intervals=8):
        import torch
        self.torch = torch
        self.side = side
        self.intervals = intervals
        self.device = "cpu"
        rng = np.random.default_rng(7)
        self.flat = rng.normal(120.0, 10.0, side * side)
        self.mean = float(self.flat.mean())
        self.scale = float(self.flat.std())
        every = np.arange(self.flat.size, dtype=np.int64)
        self.indexes = {"train": every[::2], "selection": every[1::2], "all": every}


class TablesAgreeWithTheHostUnpacker(unittest.TestCase):
    """The tables are built by the host unpacker, so drift is impossible by
    construction, but only while that stays true, which is what this pins."""

    def test_coordinates_and_tiles_are_bit_equal(self):
        import torch  # noqa: F401
        from geoneural.neural.learning import features
        problem = _StubProblem()
        for shared in (False, True):
            tables = device_data.DeviceTables(problem, shared, problem.torch)
            index = np.arange(problem.flat.size, dtype=np.int64)
            coords, tiles = features(index, problem.side, problem.intervals, shared)
            self.assertTrue(np.array_equal(tables.coords.numpy(), coords), f"shared={shared}")
            self.assertTrue(np.array_equal(tables.tiles.numpy(), tiles), f"shared={shared}")

    def test_heights_are_kept_in_metres_and_in_normalised_units(self):
        """Errors are reported in metres on a 900 m field to three decimals;
        float32 metres would not carry that."""
        problem = _StubProblem()
        tables = device_data.DeviceTables(problem, False, problem.torch)
        self.assertEqual(tables.reference_m.dtype, problem.torch.float64)
        self.assertEqual(tables.normalised.dtype, problem.torch.float32)
        self.assertTrue(np.allclose(tables.reference_m.numpy(), problem.flat))


class DeviceEvaluateMatchesTheHostOne(unittest.TestCase):
    def test_every_metric_agrees_to_a_nanometre(self):
        import torch
        from geoneural.neural import training
        from geoneural.neural.learning import features
        from geoneural.neural.models import make_model
        problem = _StubProblem()
        tables = device_data.DeviceTables(problem, False, problem.torch)
        torch.manual_seed(3)
        model = make_model({"kind": "siren", "width": 16, "depth": 2, "omega": 30.0})
        picked = problem.indexes["selection"]
        host = training.evaluate(model, features, picked, problem.flat, problem.side,
                                 problem.intervals, False, problem.mean, problem.scale,
                                 "cpu", torch, None)
        device = device_data.evaluate(model, tables, torch.from_numpy(picked), torch)
        for key in ("mae_m", "rmse_m", "p95_m", "p99_m", "max_m"):
            self.assertAlmostEqual(host[key], device[key], delta=1e-9, msg=key)

    def test_a_non_finite_prediction_raises_rather_than_scoring(self):
        """A NaN on a Pareto front is not a bad result, it is an absent one."""
        import torch
        problem = _StubProblem()
        tables = device_data.DeviceTables(problem, False, problem.torch)

        class _Nan(torch.nn.Module):
            def forward(self, coords, tiles=None):
                return torch.full(coords.shape[:-1] + (1,), float("nan"))

        with self.assertRaises(FloatingPointError):
            device_data.evaluate(_Nan(), tables, torch.from_numpy(problem.indexes["all"]), torch)


class SamplingPoliciesAreDeclaredNotAssumed(unittest.TestCase):
    def batcher(self, sampling, seed=1729, batch=4):
        import torch
        pool = torch.arange(20)
        return device_data.Batcher(pool, batch, seed, torch, "cpu", sampling)

    def test_without_replacement_never_repeats_inside_a_batch(self):
        picked = self.batcher("without").next()
        self.assertEqual(len(set(picked.tolist())), picked.numel())

    def test_the_same_seed_gives_the_same_stream(self):
        self.assertEqual(self.batcher("with").next().tolist(),
                         self.batcher("with").next().tolist())

    def test_an_unknown_policy_is_refused(self):
        with self.assertRaises(ValueError):
            self.batcher("magic")

    def test_a_batch_larger_than_the_pool_is_refused(self):
        """Silently shrinking would make two trials with different batch sizes
        report the same one."""
        with self.assertRaises(ValueError):
            self.batcher("with", batch=999)

    def test_only_with_replacement_can_fill_a_static_batch(self):
        """A permutation allocates, so a captured graph would replay a stale
        batch forever. Refusing is better than training on one batch."""
        import torch
        out = torch.zeros(4, dtype=torch.int64)
        self.assertIs(self.batcher("with").fill(out), out)
        with self.assertRaises(ValueError):
            self.batcher("without").fill(out)


if __name__ == "__main__":
    unittest.main()
