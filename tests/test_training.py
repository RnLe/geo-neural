"""Tests for the shared training loop.

The first test guards the most important property: a slope penalty reads two
heights, and if the second one may sit outside the training mask the model
trains on held-out data. That failure improves holdout numbers, so it would
not otherwise be noticed.
"""
from __future__ import annotations
import math
import unittest
import pathlib
import tempfile

import numpy as np

from geoneural.neural import splits

try:
    import torch
    from geoneural.neural import training
    from geoneural.neural.learning import features
    from geoneural.neural.models import make_model
    TORCH = None
except ImportError as error:
    TORCH = str(error)


@unittest.skipIf(TORCH, f"torch unavailable: {TORCH}")
class SlopePairs(unittest.TestCase):
    def test_no_pair_reaches_outside_the_training_region(self):
        split = splits.build(257, 64, 0.25, selection_split=True)
        left, right = training.training_pairs(split["trainMask"])
        self.assertGreater(left.size, 0)
        train = split["trainMask"].reshape(-1)
        for name, side in (("left", left), ("right", right)):
            with self.subTest(name):
                self.assertTrue(train[side].all(), "a slope pair reached a held-out node")

    def test_pairs_are_actual_lattice_neighbours(self):
        split = splits.build(129, 64, 0.25)
        left, right = training.training_pairs(split["trainMask"])
        delta = right - left
        self.assertTrue(np.isin(delta, [1, 129]).all(), "pairs must be east or south neighbours")

    def test_a_slope_term_without_a_mask_is_refused(self):
        model = make_model({"kind": "mlp", "width": 8, "depth": 1})
        recipe = training.Recipe(steps=1, batch=4, device="cpu", slope_weight=0.5)
        with self.assertRaises(ValueError):
            training.fit(model, features, np.zeros(129 * 129), 129, 64, False,
                         np.arange(100), 0.0, 1.0, recipe, torch)


@unittest.skipIf(TORCH, f"torch unavailable: {TORCH}")
class Schedule(unittest.TestCase):
    def test_cosine_decays_to_zero_and_starts_at_the_declared_rate(self):
        recipe = training.Recipe(steps=100, lr=0.01, schedule="cosine", warmup=0)
        self.assertAlmostEqual(training._schedule(recipe, 0), 0.01)
        self.assertLess(training._schedule(recipe, 99), 0.01 * 0.001)
        values = [training._schedule(recipe, step) for step in range(100)]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_warmup_ramps_before_the_schedule_starts(self):
        recipe = training.Recipe(steps=100, lr=0.01, schedule="cosine", warmup=10)
        self.assertAlmostEqual(training._schedule(recipe, 0), 0.001)
        self.assertAlmostEqual(training._schedule(recipe, 9), 0.01)

    def test_none_holds_the_rate_flat(self):
        recipe = training.Recipe(steps=50, lr=0.003, schedule="none", warmup=0)
        self.assertEqual({training._schedule(recipe, s) for s in range(50)}, {0.003})


@unittest.skipIf(TORCH, f"torch unavailable: {TORCH}")
class Fit(unittest.TestCase):
    def problem(self, side: int = 129):
        rows = np.linspace(0.0, 1.0, side)
        field = (np.sin(3 * rows)[:, None] + np.cos(2 * rows)[None, :]) * 10.0
        split = splits.build(side, 64, 0.25, selection_split=True)
        return field.reshape(-1), split

    def test_training_reduces_the_loss(self):
        flat, split = self.problem()
        model = make_model({"kind": "siren", "width": 32, "depth": 2})
        recipe = training.Recipe(steps=200, batch=512, lr=3e-3, device="cpu", seed=7)
        result = training.fit(model, features, flat, 129, 64, False,
                              np.flatnonzero(split["trainMask"].reshape(-1)),
                              float(flat.mean()), max(float(flat.std()), 1.0), recipe, torch,
                              train_mask=split["trainMask"])
        self.assertLess(result["history"][-1]["loss"], result["history"][0]["loss"])

    def test_divergence_is_raised_rather_than_reported_as_a_result(self):
        """A NaN loss must stop the run. It would otherwise reach the metrics as
        a NaN error, and `write_json` refuses NaN, so the failure would surface
        only when results are written, after every trial had run."""
        flat, split = self.problem()
        model = make_model({"kind": "mlp", "width": 32, "depth": 2})
        recipe = training.Recipe(steps=200, batch=512, lr=1e6, device="cpu", seed=7)
        with self.assertRaises(FloatingPointError):
            training.fit(model, features, flat, 129, 64, False,
                         np.flatnonzero(split["trainMask"].reshape(-1)),
                         0.0, 1.0, recipe, torch, train_mask=split["trainMask"])

    def test_a_bounded_activation_can_fail_without_ever_going_non_finite(self):
        """The guard catches NaN, not badness. A SIREN's activations are bounded,
        so the same absurd learning rate leaves it finite and useless: it scores
        badly and the Pareto front drops it, which is the correct outcome and the
        reason the guard is not treated as a completeness check."""
        flat, split = self.problem()
        model = make_model({"kind": "siren", "width": 32, "depth": 2})
        recipe = training.Recipe(steps=100, batch=512, lr=1e6, device="cpu", seed=7)
        result = training.fit(model, features, flat, 129, 64, False,
                              np.flatnonzero(split["trainMask"].reshape(-1)),
                              0.0, 1.0, recipe, torch, train_mask=split["trainMask"])
        final = result["history"][-1]["loss"]
        self.assertTrue(math.isfinite(final))
        self.assertGreater(final, 1e3)

    def test_an_unsupported_loss_is_refused(self):
        flat, split = self.problem()
        model = make_model({"kind": "mlp", "width": 8, "depth": 1})
        recipe = training.Recipe(steps=2, batch=64, device="cpu", loss="mystery")
        with self.assertRaises(ValueError):
            training.fit(model, features, flat, 129, 64, False,
                         np.flatnonzero(split["trainMask"].reshape(-1)), 0.0, 1.0,
                         recipe, torch)

    def test_a_reporter_may_abandon_a_run_without_losing_the_weights(self):
        flat, split = self.problem()
        model = make_model({"kind": "mlp", "width": 16, "depth": 2})
        recipe = training.Recipe(steps=500, batch=256, device="cpu", eval_every=10)

        class Abandon(Exception):
            pass

        def stop(step, record):
            if step >= 20:
                raise Abandon()

        with self.assertRaises(Abandon):
            training.fit(model, features, flat, 129, 64, False,
                         np.flatnonzero(split["trainMask"].reshape(-1)), 0.0, 1.0,
                         recipe, torch, on_report=stop)
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))


@unittest.skipIf(TORCH, f"torch unavailable: {TORCH}")
class Evaluate(unittest.TestCase):
    def test_errors_are_ordered_and_in_physical_units(self):
        side = 129
        flat = (np.arange(side * side, dtype=np.float64) % 97) * 1.5
        model = make_model({"kind": "mlp", "width": 8, "depth": 1})
        result = training.evaluate(model, features, np.arange(0, side * side, 7), flat,
                                   side, 64, False, 50.0, 20.0, "cpu", torch)
        self.assertLessEqual(result["mae_m"], result["p95_m"])
        self.assertLessEqual(result["p95_m"], result["p99_m"])
        self.assertLessEqual(result["p99_m"], result["max_m"])
        self.assertGreater(result["max_m"], 1.0)

    def test_an_empty_index_set_reports_no_samples_rather_than_a_number(self):
        model = make_model({"kind": "mlp", "width": 8, "depth": 1})
        result = training.evaluate(model, features, np.array([], dtype=np.int64),
                                   np.zeros(16), 4, 2, False, 0.0, 1.0, "cpu", torch)
        self.assertEqual(result, {"samples": 0})

    def test_a_limit_subsamples_without_changing_the_units(self):
        side = 129
        flat = np.linspace(0.0, 100.0, side * side)
        model = make_model({"kind": "mlp", "width": 8, "depth": 1})
        result = training.evaluate(model, features, np.arange(side * side), flat, side, 64,
                                   False, 0.0, 1.0, "cpu", torch, limit=1000,
                                   rng=np.random.default_rng(3))
        self.assertEqual(result["samples"], 1000)

    def test_evaluation_restores_training_mode(self):
        model = make_model({"kind": "mlp", "width": 8, "depth": 1}).train()
        training.evaluate(model, features, np.arange(16), np.zeros(16), 4, 2, False,
                          0.0, 1.0, "cpu", torch)
        self.assertTrue(model.training)


@unittest.skipIf(TORCH, f"torch unavailable: {TORCH}")
class DeployedBytes(unittest.TestCase):
    def test_reported_bytes_match_a_written_file(self):
        import tempfile
        from pathlib import Path
        from safetensors.torch import save_file
        model = make_model({"kind": "siren", "width": 64, "depth": 3})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.safetensors"
            save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
                      str(path))
            self.assertEqual(training.deployed_bytes(model, torch), path.stat().st_size)

    def test_framing_is_included_rather_than_assumed_away(self):
        """A parameter count times four is not the deployed size; a width-128,
        depth-3 SIREN's file carries 640 bytes of framing on top of its tensors."""
        from geoneural.neural.models import parameter_count
        model = make_model({"kind": "siren", "width": 128, "depth": 3})
        self.assertGreater(training.deployed_bytes(model, torch), parameter_count(model) * 4)


if __name__ == "__main__":
    unittest.main()


@unittest.skipIf(TORCH, f"torch unavailable: {TORCH}")
class StoragePrecision(unittest.TestCase):
    """A model must be measured at the width it is priced at.

    These tests pin both halves: the byte count follows the stored width, and the
    weights are actually rounded to it.
    """

    def model(self):
        return make_model({"kind": "siren", "width": 64, "depth": 3})

    def test_float16_storage_is_about_half_of_float32(self):
        model = self.model()
        wide = training.deployed_bytes(model, torch, "float32")
        half = training.deployed_bytes(model, torch, "float16")
        self.assertLess(half, wide)
        # Framing is not halved, so the ratio is near a half rather than exactly.
        self.assertAlmostEqual(half / wide, 0.5, delta=0.02)

    def test_float64_storage_is_about_double_and_is_the_control(self):
        model = self.model()
        wide = training.deployed_bytes(model, torch, "float32")
        self.assertAlmostEqual(training.deployed_bytes(model, torch, "float64") / wide,
                               2.0, delta=0.02)

    def test_an_unsupported_precision_is_refused_rather_than_guessed(self):
        with self.assertRaises(ValueError):
            training.deployed_bytes(self.model(), torch, "float8_e4m3fn")

    def test_rounding_changes_the_weights_and_hands_back_the_originals(self):
        model = self.model()
        original = training.round_to_storage(model, torch, "float16")
        after = dict(model.state_dict())
        moved = max(float((original[k] - after[k]).abs().max()) for k in original)
        self.assertGreater(moved, 0.0, "float16 rounding changed nothing, so it was not applied")
        # Every rounded value must be exactly representable in float16.
        for key, tensor in after.items():
            if tensor.is_floating_point():
                self.assertEqual(float((tensor.to(torch.float16).to(tensor.dtype) - tensor)
                                       .abs().max()), 0.0, key)

    def test_rounding_to_float32_is_a_no_op_on_float32_weights(self):
        model = self.model()
        original = training.round_to_storage(model, torch, "float32")
        for key, tensor in model.state_dict().items():
            self.assertEqual(float((original[key] - tensor).abs().max()), 0.0, key)

    def test_store_state_reports_the_dtype_it_claims(self):
        state = training.store_state(self.model(), torch, "float16")
        self.assertTrue(all(v.dtype is torch.float16 for v in state.values()
                            if v.is_floating_point()))


@unittest.skipIf(TORCH, f"torch unavailable: {TORCH}")
class NonFiniteEvaluation(unittest.TestCase):
    """A NaN metric is an absent result, not a bad one, and must not reach a front.

    `fit` checks the training loss for finiteness, but only at reporting steps and
    only on training batches. A model can finish with a finite loss and still
    produce non-finite predictions on the evaluated split. Returning those would
    put NaN into a study objective, where Optuna treats it as a value rather than
    a failure; raising makes it a pruned trial carrying its reason.
    """

    class _Poison(torch.nn.Module if not TORCH else object):
        def forward(self, coords, tiles=None):
            return torch.full(coords.shape[:-1] + (1,), float("nan"), dtype=coords.dtype)

    def test_non_finite_predictions_raise_instead_of_being_averaged(self):
        side, intervals = 65, 32
        flat = np.linspace(0.0, 100.0, side * side)
        indexes = np.arange(flat.size, dtype=np.int64)
        with self.assertRaises(FloatingPointError) as caught:
            training.evaluate(self._Poison(), features, indexes, flat, side, intervals,
                              False, 0.0, 1.0, "cpu", torch)
        self.assertIn("not finite", str(caught.exception))

    def test_a_finite_model_still_evaluates(self):
        """The control: the guard must not reject ordinary results."""
        side, intervals = 65, 32
        flat = np.linspace(0.0, 100.0, side * side)
        indexes = np.arange(flat.size, dtype=np.int64)
        metrics = training.evaluate(make_model({"kind": "siren", "width": 16, "depth": 2}),
                                    features, indexes, flat, side, intervals, False,
                                    0.0, 1.0, "cpu", torch)
        self.assertTrue(math.isfinite(metrics["mae_m"]))
        self.assertEqual(metrics["samples"], flat.size)


@unittest.skipIf(TORCH, f"torch unavailable: {TORCH}")
class ResumedTrainingIsContinuedTraining(unittest.TestCase):
    """A resumed run must be indistinguishable from the uninterrupted one.

    Weights alone are not training state. Adam carries per-parameter moments, the
    schedule is a function of the step counter, and three random streams pick the
    batches, the slope pairs and the initialisation. Reloading `state_dict()` and
    calling `fit` again restarts the optimiser cold, rewinds the learning rate to
    warmup and redraws batches the first run already used.
    """

    def problem(self, side: int = 129):
        rows = np.linspace(0.0, 1.0, side)
        field = (np.sin(3 * rows)[:, None] + np.cos(2 * rows)[None, :]) * 10.0
        split = splits.build(side, 64, 0.25, selection_split=True)
        return field.reshape(-1), split

    # Every run below plans the same number of steps. An interruption does not
    # change the plan, and the cosine schedule is a function of the planned total,
    # so a run that asks for fewer steps is a different experiment, not a
    # truncated one, and would not match however well the state was restored.
    TOTAL = 60

    def run_fit(self, model, flat, split, stop_at=None, **kw):
        recipe = training.Recipe(steps=self.TOTAL, batch=256, lr=3e-3, device="cpu", seed=11,
                                 schedule="cosine", warmup=5, eval_every=10)

        def abort(step, record):
            if stop_at is not None and step >= stop_at:
                raise KeyboardInterrupt("simulated kill")

        try:
            return training.fit(model, features, flat, 129, 64, False,
                                np.flatnonzero(split["trainMask"].reshape(-1)), float(flat.mean()),
                                max(float(flat.std()), 1.0), recipe, torch,
                                train_mask=split["trainMask"], on_report=abort, **kw)
        except KeyboardInterrupt:
            return None

    def fresh_model(self):
        torch.manual_seed(4242)
        return make_model({"kind": "siren", "width": 24, "depth": 2})

    def test_a_resumed_run_matches_an_uninterrupted_one_exactly(self):
        flat, split = self.problem()
        straight = self.fresh_model()
        self.run_fit(straight, flat, split)

        path = pathlib.Path(tempfile.mkdtemp()) / "ckpt.pt"
        self.run_fit(self.fresh_model(), flat, split, stop_at=30,
                     checkpoint_path=path, checkpoint_every=10)
        self.assertTrue(path.exists(), "a checkpoint must have been written before the kill")

        resumed = self.fresh_model()
        result = self.run_fit(resumed, flat, split,
                              resume=training.load_checkpoint(path, torch))
        self.assertEqual(result["resumedFromStep"], 30)
        for (name, a), (_, b) in zip(sorted(straight.state_dict().items()),
                                     sorted(resumed.state_dict().items())):
            self.assertTrue(torch.equal(a, b), f"{name} diverged after resuming")

    def test_reloading_weights_alone_is_not_a_continuation(self):
        """The control: without optimiser and RNG state the run does not match."""
        flat, split = self.problem()
        straight = self.fresh_model()
        self.run_fit(straight, flat, split)

        path = pathlib.Path(tempfile.mkdtemp()) / "ckpt.pt"
        broken = self.fresh_model()
        self.run_fit(broken, flat, split, stop_at=30, checkpoint_path=path, checkpoint_every=10)

        naive = self.fresh_model()
        naive.load_state_dict(training.load_checkpoint(path, torch)["model"])
        self.run_fit(naive, flat, split)              # weights only, cold optimiser
        differs = any(not torch.equal(a, b) for (_, a), (_, b)
                      in zip(sorted(straight.state_dict().items()),
                             sorted(naive.state_dict().items())))
        self.assertTrue(differs, "weights-only reload should NOT reproduce the run")

    def test_the_history_survives_the_interruption(self):
        flat, split = self.problem()
        path = pathlib.Path(tempfile.mkdtemp()) / "ckpt.pt"
        self.run_fit(self.fresh_model(), flat, split, stop_at=30,
                     checkpoint_path=path, checkpoint_every=10)
        result = self.run_fit(self.fresh_model(), flat, split,
                              resume=training.load_checkpoint(path, torch))
        steps = [record["step"] for record in result["history"]]
        self.assertEqual(steps, sorted(steps))
        self.assertLess(min(steps), 30, "records from before the interruption must survive")

    def test_resuming_with_a_different_experiment_is_refused(self):
        flat, split = self.problem()
        path = pathlib.Path(tempfile.mkdtemp()) / "ckpt.pt"
        self.run_fit(self.fresh_model(), flat, split, stop_at=30,
                     checkpoint_path=path, checkpoint_every=10)
        state = training.load_checkpoint(path, torch)
        state["recipe"]["batch"] = 999
        with self.assertRaises(ValueError):
            self.run_fit(self.fresh_model(), flat, split, resume=state)

    def test_an_unknown_checkpoint_schema_is_refused(self):
        with self.assertRaises(ValueError):
            training.restore_state({"schema": "something-else"}, self.fresh_model(),
                                   torch.optim.Adam(self.fresh_model().parameters()),
                                   np.random.default_rng(0), np.random.default_rng(1), torch)
