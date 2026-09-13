"""Tests for the architecture search.

The search's job is to be a measurement, so the properties worth testing are the
ones that decide whether its output means anything: that every family's space
yields a model that can actually be built, that the byte objective is the real
serialized size including any conventional base, that a configuration carrying a
tensor never reaches a JSON report, and that the conventional comparator is
computed the way it is documented rather than the way that flatters the network.

A stub stands in for `Problem` so these run without an atlas on disk. It supplies
exactly the attributes the functions under test read, which also keeps that
surface visible: anything the search starts depending on has to be added here.
"""
from __future__ import annotations
import unittest
import unittest.mock

import numpy as np

try:
    import optuna
    import torch
    from geoneural.neural import search
    MISSING = None
except ImportError as error:
    MISSING = str(error)


class MachineLearningExtras(unittest.TestCase):
    @unittest.skipIf(MISSING, f"ML extras unavailable: {MISSING}")
    def test_the_search_extras_import(self):
        self.assertIsNotNone(search)


def load_tests(loader, tests, pattern):
    """Every class here needs the names imported above. Without the extras the
    module yields a single visible skip instead of NameErrors, under discovery
    and when loaded by name alike."""
    if MISSING:
        return loader.loadTestsFromTestCase(MachineLearningExtras)
    return tests


class _StubProblem:
    """The smallest thing the functions under test will accept."""

    def __init__(self, side: int = 129, intervals: int = 64, max_level: int = 2,
                 normalisation: str = "all"):
        rows = np.linspace(0.0, 60.0, side)
        self.reference = (np.sin(rows / 7.0)[:, None] * 20.0 + rows[None, :] * 0.5)
        self.side = side
        self.intervals = intervals
        self.flat = self.reference.reshape(-1)
        self.manifest = {"max_level": max_level, "page_intervals": intervals}
        self.tiles = ((side - 1) // intervals) ** 2
        self.device = "cpu"
        self.spacing_m = 10.0
        self.torch = torch if not MISSING else None
        from geoneural.neural import splits
        self.split = splits.build(side, intervals, 0.25, selection_split=True)
        self.indexes = {name: np.flatnonzero(self.split[key].reshape(-1)) for name, key in (
            ("train", "trainMask"), ("selection", "selectionMask"),
            ("test", "testMask"), ("extrapolation", "extrapolationMask"))}
        self.indexes["all"] = np.arange(self.flat.size, dtype=np.int64)
        if normalisation not in ("all", "train"):
            raise ValueError(f"Unknown normalisation scope: {normalisation}")
        self.normalisation = normalisation
        source = self.flat if normalisation == "all" else self.flat[self.indexes["train"]]
        self.mean = float(np.mean(source))
        self.scale = max(float(np.std(source)), 1.0)

    def level_grid(self, level: int) -> np.ndarray:
        step = 2 ** level
        return np.ascontiguousarray(self.reference[::step, ::step])

    def base(self, level: int, target_m: float):
        """Paged bytes second, exactly as the real Problem.base returns them.

        The two figures differ on purpose: paging a grid into independently
        decodable units costs more than one stream, and a stub that returned the
        same number for both would hide a caller confusing them (which makes
        randomAccessOverheadFraction zero by construction).
        """
        grid = self.level_grid(level)
        return grid, 1234, {"level": level, "targetM": target_m, "side": int(grid.shape[0]),
                            "bytes": 1234, "monolithicBytes": 1000,
                            "maxErrorM": target_m, "codec": "stub"}


@unittest.skipIf(MISSING, f"ML extras unavailable: {MISSING}")
class SearchSpaces(unittest.TestCase):
    def test_every_family_samples_a_configuration_that_builds(self):
        """Some sampled configurations are legitimately refused (a dense grid
        pyramid can ask for hundreds of gigabytes), so the refusal is converted
        to a pruned trial here exactly as `study` does. What must hold is that
        every family produces at least one model that builds and evaluates."""
        problem = _StubProblem()
        for family in search.FAMILIES:
            with self.subTest(family):
                built, refused = [], []

                def objective(trial, family=family, built=built, refused=refused):
                    config = search.sample_model(trial, family)
                    try:
                        model, extra = search.build_model(config, problem)
                    except (ValueError, KeyError) as error:
                        refused.append(str(error))
                        raise optuna.TrialPruned(str(error))
                    out = model(torch.rand(8, 2) * 2 - 1, torch.zeros(8, dtype=torch.long))
                    built.append((config, extra, out.shape))
                    return float(search.training.deployed_bytes(model, torch))

                study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=5))
                optuna.logging.set_verbosity(optuna.logging.WARNING)
                study.optimize(objective, n_trials=8, catch=())
                self.assertGreater(len(built), 0, f"no {family} configuration built; refused: {refused}")
                self.assertEqual(len(built) + len(refused), 8)
                for config, _, shape in built:
                    self.assertEqual(shape, (8, 1))
                    # An arm name is not always a model kind: "hybrid" is the
                    # preferred design and builds as a `residual`.
                    self.assertEqual(config["kind"], search.ARM_KIND.get(family, family))

    def test_a_recipe_is_sampled_with_the_study_constants_fixed(self):
        """Steps and batch must not be searchable; otherwise the Pareto front is
        a compute front and the rate axis means nothing."""
        def objective(trial):
            recipe = search.sample_recipe(trial, steps=321, batch=64, device="cpu", seed=11)
            self.assertEqual((recipe.steps, recipe.batch, recipe.seed), (321, 64, 11))
            self.assertGreaterEqual(recipe.lr, 1e-4)
            self.assertIn(recipe.loss, ("mse", "huber", "l1"))
            return recipe.lr

        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=2))
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=6)

    def test_every_family_has_a_screening_default(self):
        """FAMILIES and screen()'s defaults table are two lists that must agree.
        A mismatch raises KeyError only after the atlas has been loaded."""
        import inspect
        source = inspect.getsource(search.screen)
        for family in search.FAMILIES:
            with self.subTest(family):
                self.assertIn(f'"{family}"', source)

    def test_an_unknown_family_is_refused(self):
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=1))
        with self.assertRaises(ValueError):
            study.optimize(lambda trial: search.sample_model(trial, "telepathy") and 0.0,
                           n_trials=1, catch=())


@unittest.skipIf(MISSING, f"ML extras unavailable: {MISSING}")
class Accounting(unittest.TestCase):
    def test_a_hybrid_is_charged_for_its_conventional_base(self):
        problem = _StubProblem()
        config = {"kind": "residual", "base_level": 1, "base_target_m": 1.0,
                  "inner": {"kind": "mlp", "width": 16, "depth": 2}}
        model, extra = search.build_model(config, problem)
        self.assertEqual(extra["baseBytes"], 1234)
        plain, _ = search.build_model({"kind": "mlp", "width": 16, "depth": 2}, problem)
        weights = search.training.deployed_bytes(plain, torch)
        result = search.measure(config, search.training.Recipe(steps=2, batch=32, device="cpu"),
                                problem)
        self.assertEqual(result["deployedBytes"], result["weightsBytes"] + 1234)
        self.assertGreater(result["deployedBytes"], weights,
                           "the base must add to the deployed total, not vanish from it")

    def test_bits_per_sample_uses_the_full_lattice(self):
        problem = _StubProblem()
        result = search.measure({"kind": "mlp", "width": 16, "depth": 2},
                                search.training.Recipe(steps=2, batch=32, device="cpu"), problem)
        self.assertAlmostEqual(result["bitsPerSample"],
                               8.0 * result["deployedBytes"] / problem.flat.size)

    def test_a_description_carries_no_tensor_into_a_report(self):
        import json
        config = {"kind": "residual", "base": torch.zeros(5, 5), "base_level": 2,
                  "base_target_m": 0.5, "inner": {"kind": "grid", "levels": 3, "width": 8}}
        described = search.describe(config)
        self.assertNotIn("base", described)
        self.assertEqual(described["inner"]["kind"], "grid")
        json.dumps(described)  # must not raise


@unittest.skipIf(MISSING, f"ML extras unavailable: {MISSING}")
class ConventionalComparator(unittest.TestCase):
    def setUp(self):
        self.curve = search.conventional_curve(_StubProblem(), targets=(0.1, 1.0))

    def test_random_access_costs_more_than_one_monolithic_stream(self):
        """Independently compressed pages are the like-for-like product for a
        field that answers any coordinate on its own. They are also dearer, and
        quoting the monolithic figure would credit the codec with a random
        access it cannot perform."""
        for row in self.curve["rows"]:
            with self.subTest(row["targetMaxErrorM"]):
                self.assertGreater(row["perPageFinestBytes"], row["monolithicFinestBytes"])
                self.assertGreater(row["randomAccessOverheadFraction"], 0.0)

    def test_the_pyramid_costs_more_than_its_finest_level(self):
        for row in self.curve["rows"]:
            with self.subTest(row["targetMaxErrorM"]):
                self.assertGreater(row["pyramidBytes"], row["perPageFinestBytes"])

    def test_a_looser_target_is_cheaper_and_less_accurate(self):
        tight, loose = self.curve["rows"]
        self.assertGreater(tight["perPageFinestBytes"], loose["perPageFinestBytes"])
        self.assertLess(tight["maeM"], loose["maeM"])

    def test_every_row_meets_its_declared_bound(self):
        """Within one float32 ULP at these elevations. The conventional codec
        comparison records the same excess under `float32_representation_ulp_m`: the quantizer is
        exact in integers and the reconstruction is stored as float32, so the
        bound holds up to the representation, not beyond it."""
        ulp = float(np.spacing(np.float32(np.abs(self.curve and 100.0))))
        for row in self.curve["rows"]:
            with self.subTest(row["targetMaxErrorM"]):
                self.assertLessEqual(row["maxErrorM"], row["targetMaxErrorM"] + ulp)

    def test_the_primary_comparator_is_named_in_the_artifact(self):
        self.assertEqual(self.curve["primaryComparator"], "perPageFinestBytes")
        self.assertIn("390,343", self.curve["supersedes"])


if __name__ == "__main__":
    unittest.main()


@unittest.skipIf(MISSING, f"ML extras unavailable: {MISSING}")
class CodeCoverage(unittest.TestCase):
    """The diagnostic that decides whether a local-code family's holdout number
    means anything: the fraction of stored codes that training reaches."""

    def setUp(self):
        self.problem = _StubProblem(side=257, intervals=64)

    def test_per_tile_codes_are_only_partly_reached_by_training(self):
        model, _ = search.build_model({"kind": "shared", "width": 16, "depth": 2, "latent": 4},
                                      self.problem)
        coverage = search.code_coverage(model, self.problem, shared=True)
        self.assertLess(coverage["coverageFraction"], 1.0)
        self.assertGreater(coverage["coverageFraction"], 0.0)
        self.assertEqual(coverage["codeEntries"], self.problem.tiles)

    def test_an_interpolated_grid_reaches_more_of_its_payload(self):
        """Shared border nodes are the mechanism: a node on the edge of a
        held-out patch is also on the edge of a trained one."""
        tiles, _ = search.build_model({"kind": "shared", "width": 16, "depth": 2, "latent": 4},
                                      self.problem)
        grid, _ = search.build_model({"kind": "codegrid", "patches": 4, "latent": 4,
                                      "width": 16, "depth": 2}, self.problem)
        by_tile = search.code_coverage(tiles, self.problem, shared=True)
        by_grid = search.code_coverage(grid, self.problem, shared=False)
        self.assertGreater(by_grid["coverageFraction"], by_tile["coverageFraction"])

    def test_a_grid_entry_is_a_node_not_a_row(self):
        grid, _ = search.build_model({"kind": "codegrid", "patches": 4, "latent": 4,
                                      "width": 16, "depth": 2}, self.problem)
        coverage = search.code_coverage(grid, self.problem, shared=False)
        self.assertEqual(coverage["codeEntries"], 5 * 5)

    def test_a_family_without_codes_reports_nothing_rather_than_a_perfect_score(self):
        model, _ = search.build_model({"kind": "siren", "width": 16, "depth": 2}, self.problem)
        self.assertIsNone(search.code_coverage(model, self.problem, shared=False))


@unittest.skipIf(MISSING, f"ML extras unavailable: {MISSING}")
class Precision(unittest.TestCase):
    """The storage ladder. float64 is the control that makes the rest readable:
    widening float32-trained weights is lossless, so a nonzero change there would
    mean the sweep is measuring something other than storage."""

    @classmethod
    def setUpClass(cls):
        cls.problem = _StubProblem(side=129, intervals=64)
        model, _ = search.build_model({"kind": "siren", "width": 32, "depth": 2}, cls.problem)
        cls.sweep = search.precision_sweep(model, cls.problem, shared=False)
        cls.by_name = {row["precision"]: row for row in cls.sweep["rows"]}
        cls.model = model

    def test_widening_to_float64_changes_nothing_and_costs_double(self):
        """Zero change is the assertion that matters; the size ratio is bounded
        rather than exact because safetensors framing is a fixed cost that does
        not scale with dtype. It is negligible on a 134 kB checkpoint (1.995x)
        and visible on this deliberately tiny one."""
        wide, base = self.by_name["float64"], self.by_name["float32"]
        self.assertEqual(wide["mae_m"], base["mae_m"])
        self.assertEqual(wide["max_m"], base["max_m"])
        self.assertGreater(wide["bytesRelativeToFloat32"], 1.85)
        self.assertLessEqual(wide["bytesRelativeToFloat32"], 2.0)

    def test_the_two_byte_formats_are_the_same_size(self):
        self.assertAlmostEqual(self.by_name["float16"]["weightsBytes"]
                               / self.by_name["bfloat16"]["weightsBytes"], 1.0, places=2)
        self.assertGreaterEqual(self.by_name["float16"]["bytesRelativeToFloat32"], 0.5)
        self.assertLess(self.by_name["float16"]["bytesRelativeToFloat32"], 0.58)

    def test_float16_damages_less_than_bfloat16_at_equal_size(self):
        """Weights occupy a narrow band around zero, so mantissa bits matter and
        exponent range does not. Equal storage, unequal damage."""
        self.assertLess(abs(self.by_name["float16"]["maeChangeFromFloat32M"]),
                        abs(self.by_name["bfloat16"]["maeChangeFromFloat32M"]))

    def test_the_sweep_leaves_the_model_exactly_as_it_found_it(self):
        """A sweep that quietly left the model at the last precision would make
        every measurement after it wrong."""
        after = search.training.evaluate(
            self.model, __import__("geoneural.neural.learning", fromlist=["features"]).features,
            self.problem.indexes["train"][:256], self.problem.flat, self.problem.side,
            self.problem.intervals, False, self.problem.mean, self.problem.scale,
            "cpu", torch)
        again = search.precision_sweep(self.model, self.problem, shared=False,
                                       precisions=("float32",))
        self.assertAlmostEqual(again["rows"][0]["mae_m"],
                               self.by_name["float32"]["mae_m"], places=9)
        self.assertTrue(np.isfinite(after["mae_m"]))

    def test_int8_is_excluded_with_a_stated_reason(self):
        self.assertIn("scales", self.sweep["int8"])

    def test_eight_bit_formats_are_half_the_size_of_sixteen(self):
        self.assertLess(self.by_name["float8_e4m3fn"]["tensorBytes"],
                        self.by_name["float16"]["weightsBytes"] * 0.6)

    def test_more_mantissa_beats_more_exponent_at_every_width(self):
        """e4m3 against e5m2 at eight bits, float16 against bfloat16 at sixteen.
        Weights occupy a narrow band around zero in both cases, so the format
        spending bits on mantissa wins both times."""
        self.assertLess(abs(self.by_name["float8_e4m3fn"]["maeChangeFromFloat32M"]),
                        abs(self.by_name["float8_e5m2"]["maeChangeFromFloat32M"]))
        self.assertLess(abs(self.by_name["float16"]["maeChangeFromFloat32M"]),
                        abs(self.by_name["bfloat16"]["maeChangeFromFloat32M"]))

    def test_a_per_tensor_scale_is_charged_as_payload(self):
        """Four bytes per scaled tensor. Small, and not zero, and a scaled row
        that reported the same size as an unscaled one would be hiding it."""
        scaled = self.by_name["float8_e4m3fn-scaled"]
        plain = self.by_name["float8_e4m3fn"]
        self.assertTrue(scaled["perTensorScale"])
        self.assertFalse(plain["perTensorScale"])
        self.assertGreater(scaled["scaleBytes"], 0)
        self.assertEqual(scaled["weightsBytes"], scaled["tensorBytes"] + scaled["scaleBytes"])
        self.assertEqual(plain["scaleBytes"], 0)


@unittest.skipIf(MISSING, f"ML extras unavailable: {MISSING}")
class EncodeTimeCodeFitting(unittest.TestCase):
    """Local codes on a held-out region are never trained, so a family that
    stores them cannot be read on that region until an encoder fits them.
    These check that the fitting does what it says and only what it says."""

    def setUp(self):
        self.problem = _StubProblem(side=129, intervals=64)
        self.recipe = search.training.Recipe(steps=20, batch=256, lr=1e-2,
                                             schedule="cosine", device="cpu", seed=4)

    def build(self, kind="shared"):
        config = ({"kind": "shared", "width": 16, "depth": 2, "latent": 4} if kind == "shared"
                  else {"kind": "codegrid", "patches": 2, "latent": 4, "width": 16, "depth": 2})
        model, _ = search.build_model(config, self.problem)
        return model

    def test_fitting_moves_the_codes_and_nothing_else(self):
        """A decoder that adapted per region would no longer be shared and its
        bytes would no longer amortise, so everything but the codes is frozen."""
        model = self.build()
        name, _ = search.code_parameter(model)
        others = {k: v.detach().clone() for k, v in model.named_parameters() if k != name}
        result = search.fit_codes(model, self.problem, "selection", self.recipe, shared=True)
        self.assertGreater(result["entriesMoved"], 0)
        for key, value in model.named_parameters():
            if key != name:
                with self.subTest(key):
                    self.assertTrue(torch.equal(value.detach(), others[key]),
                                    "the frozen decoder must not have moved")

    def test_requires_grad_is_restored_afterwards(self):
        """Leaving the decoder frozen would silently break the next training run
        that reused this model."""
        model = self.build()
        search.fit_codes(model, self.problem, "selection", self.recipe, shared=True)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_fitting_reduces_error_on_the_region_it_was_given(self):
        from geoneural.neural.learning import features
        model = self.build()
        args = (model, features, self.problem.indexes["selection"], self.problem.flat,
                self.problem.side, self.problem.intervals, True, self.problem.mean,
                self.problem.scale, "cpu", torch)
        before = search.training.evaluate(*args)["mae_m"]
        search.fit_codes(model, self.problem, "selection",
                         search.training.Recipe(steps=120, batch=512, lr=3e-2,
                                                device="cpu", seed=4), shared=True)
        after = search.training.evaluate(*args)["mae_m"]
        self.assertLess(after, before)

    def test_a_model_without_codes_is_refused_rather_than_silently_skipped(self):
        model, _ = search.build_model({"kind": "siren", "width": 16, "depth": 2}, self.problem)
        with self.assertRaises(ValueError):
            search.fit_codes(model, self.problem, "selection", self.recipe, shared=False)

    def test_the_result_denies_being_a_generalization_claim(self):
        """The number this produces is a codec result, not transfer to unseen
        geography. The split exists to keep the two apart, so the output says so
        itself."""
        model = self.build()
        result = search.fit_codes(model, self.problem, "selection", self.recipe, shared=True)
        self.assertIn("NOT evidence of generalization", result["note"])

    def test_an_interpolated_grid_moves_the_nodes_its_region_touches_and_no_others(self):
        """Fitting on one region moves only the code nodes that region reaches:
        4 of 9 here, because the selection half is a checkerboard subset rather
        than the whole domain. Full coverage would mean the fit had leaked
        outside the region it was given."""
        model = self.build("codegrid")
        result = search.fit_codes(model, self.problem, "selection", self.recipe, shared=False)
        self.assertEqual(result["codeEntries"], 3 * 3)
        self.assertGreater(result["entriesMoved"], 0)
        self.assertLess(result["coverageAfterFitting"], 1.0)

    def test_fitting_the_whole_domain_reaches_more_nodes_than_one_region(self):
        """The control for the test above: the shortfall is the region's extent,
        not a failure of the fitting."""
        part = search.fit_codes(self.build("codegrid"), self.problem, "selection",
                                self.recipe, shared=False)
        whole = search.fit_codes(self.build("codegrid"), self.problem, "all",
                                 self.recipe, shared=False)
        self.assertGreater(whole["entriesMoved"], part["entriesMoved"])


@unittest.skipIf(MISSING, f"ML extras unavailable: {MISSING}")
class Amortisation(unittest.TestCase):
    """Where sharing is actually decided. A single region's total is not the rate
    for a family whose costs split into shared and per-region parts."""

    def test_a_shared_cost_is_paid_once_and_a_per_region_cost_every_time(self):
        result = search.amortisation(1000, 10, regions=(1, 10, 100))
        totals = [row["totalBytes"] for row in result["byRegionCount"]]
        self.assertEqual(totals, [1010, 1100, 2000])

    def test_per_region_bytes_of_zero_stay_flat_forever(self):
        result = search.amortisation(50_000, 0, regions=(1, 4096))
        self.assertEqual({row["totalBytes"] for row in result["byRegionCount"]}, {50_000})

    def test_the_conventional_base_is_charged_per_region(self):
        """A hybrid omitting its base from the corpus total would appear to
        amortise something it does not."""
        without = search.amortisation(1000, 0, 0, regions=(16,))
        with_base = search.amortisation(1000, 0, 500, regions=(16,))
        self.assertEqual(with_base["byRegionCount"][0]["totalBytes"]
                         - without["byRegionCount"][0]["totalBytes"], 16 * 500)

    def test_the_crossover_is_where_the_totals_meet(self):
        first = search.amortisation(2000, 0)
        second = search.amortisation(1000, 100)
        crossing = search.amortisation_crossover(first, second)
        self.assertTrue(crossing["crosses"])
        self.assertAlmostEqual(crossing["crossoverRegions"], 10.0)
        at = int(crossing["crossoverRegions"])
        self.assertEqual(2000 + at * 0, 1000 + at * 100)

    def test_equal_per_region_costs_never_cross(self):
        """Reported as never, rather than as a number beyond which nobody looks."""
        crossing = search.amortisation_crossover(search.amortisation(500, 7),
                                                 search.amortisation(900, 7))
        self.assertFalse(crossing["crosses"])
        self.assertEqual(crossing["cheaperAtEveryCount"], "first")

    def test_a_crossover_below_one_region_is_not_reported_as_a_crossover(self):
        """If one arrangement wins from the first region there is no corpus size
        at which the other is preferable, and saying 'crosses at 0.65' would
        invite someone to plan around it."""
        crossing = search.amortisation_crossover(search.amortisation(1010, 0),
                                                 search.amortisation(1000, 100))
        self.assertFalse(crossing["crosses"])
        self.assertIn("outside", crossing["reason"])


@unittest.skipIf(MISSING, f"ML extras unavailable: {MISSING}")
class DeployedPrecision(unittest.TestCase):
    """The search must price and measure at the same stored width.

    The conventional controls are priced as they ship, so the neural side is
    priced at its stored width too; a front drawn at float32 bytes sits at
    roughly twice the rate a float16 deployment pays. Pricing at float16 without
    evaluating at float16 would be the opposite error, so both are tested.
    """

    CONFIG = {"kind": "siren", "width": 32, "depth": 2, "omega": 12.0}

    def recipe(self, **overrides):
        return search.Recipe(steps=8, batch=256, device="cpu", seed=4, lr=1e-3, **overrides)

    def test_the_reported_bytes_follow_the_stored_width(self):
        problem = _StubProblem()
        half = search.measure(self.CONFIG, self.recipe(), problem, store_precision="float16")
        self.assertEqual(half["storePrecision"], "float16")
        # Strictly above one half: the safetensors header names the same tensors
        # at either width, so framing does not halve with the payload. On this
        # deliberately tiny model that overhead is visible (~0.54); on a
        # deployable model it is not. It is charged either way.
        ratio = half["weightsBytes"] / half["weightsBytesFloat32"]
        self.assertLess(half["weightsBytes"], half["weightsBytesFloat32"])
        self.assertGreater(ratio, 0.5)
        self.assertLess(ratio, 0.6)

    def test_the_model_handed_back_is_the_one_that_was_priced(self):
        """Every stored tensor must already be representable at the stored width."""
        problem = _StubProblem()
        result = search.measure(self.CONFIG, self.recipe(), problem, store_precision="float16")
        for key, tensor in result["model"].state_dict().items():
            if tensor.is_floating_point():
                round_trip = tensor.to(torch.float16).to(tensor.dtype)
                self.assertEqual(float((round_trip - tensor).abs().max()), 0.0, key)

    def test_rounding_actually_moves_the_error_it_is_charged_for(self):
        """float16 is close to free, but it is not free, and it is not unmeasured."""
        problem = _StubProblem()
        wide = search.measure(self.CONFIG, self.recipe(), problem, store_precision="float32")
        half = search.measure(self.CONFIG, self.recipe(), problem, store_precision="float16")
        self.assertNotEqual(wide["metrics"]["selection"]["mae_m"],
                            half["metrics"]["selection"]["mae_m"])

    def test_the_base_is_not_discounted_by_the_weight_precision(self):
        """A hybrid's conventional half is conventional bytes and does not halve."""
        problem = _StubProblem()
        config = {"kind": "residual", "base_level": 1, "base_target_m": 1.0,
                  "inner": dict(self.CONFIG)}
        result = search.measure(config, self.recipe(), problem, store_precision="float16")
        self.assertEqual(result["baseBytes"], 1234)
        self.assertEqual(result["deployedBytes"], result["weightsBytes"] + 1234)
        self.assertEqual(result["deployedBytesFloat32"], result["weightsBytesFloat32"] + 1234)


@unittest.skipIf(MISSING, f"ML extras unavailable: {MISSING}")
class StudyModes(unittest.TestCase):
    """Generalisation and codec fit are different questions with different answers.

    Holdout mode optimises held-out interpolation; codec mode optimises fit on
    every page, which is the question a rate-distortion comparison asks. Each
    study records which mode it ran, so neither can be read as the other.
    """

    def test_the_two_modes_train_and_score_on_different_pages(self):
        self.assertEqual(search.MODES["holdout"], {"trainOn": "train", "scoreOn": "selection"})
        self.assertEqual(search.MODES["codec"], {"trainOn": "all", "scoreOn": "all"})

    def test_an_unknown_mode_is_refused_before_any_training(self):
        with self.assertRaises(ValueError):
            search.study("siren", _StubProblem(), trials=1, mode="rate-distortion")

    def test_a_codec_study_records_that_it_trained_on_every_page(self):
        report = search.study("siren", _StubProblem(), trials=2, steps=4, batch=256,
                              progress=False, mode="codec")
        self.assertEqual(report["mode"], "codec")
        self.assertEqual(report["trainedOn"], "all")
        self.assertEqual(report["scoredOn"], "all")
        self.assertEqual(report["studyConstants"]["objectives"], ["deployedBytes", "scoreMaeM"])
        self.assertEqual(report["studyConstants"]["storePrecision"], "float16")
        for row in report["paretoFront"]:
            self.assertIn("scoreMaeM", row)

    def test_a_holdout_study_never_trains_on_the_pages_it_scores(self):
        problem = _StubProblem()
        report = search.study("siren", problem, trials=2, steps=4, batch=256,
                              progress=False, mode="holdout")
        self.assertEqual(report["trainedOn"], "train")
        self.assertEqual(report["scoredOn"], "selection")
        overlap = np.intersect1d(problem.indexes["train"], problem.indexes["selection"])
        self.assertEqual(overlap.size, 0)


@unittest.skipIf(MISSING, f"ML extras unavailable: {MISSING}")
class ScreeningScope(unittest.TestCase):
    def test_screening_does_not_read_the_test_half(self):
        """Screening decides which families get searched, so it is selection.

        It must not evaluate the test half or the extrapolation block: `confirm`
        is documented as the only function that reads them.
        """
        import inspect
        default = inspect.signature(search.screen).parameters["evaluate_on"].default
        self.assertEqual(tuple(default), ("train", "selection"))


@unittest.skipIf(MISSING, f"ML extras unavailable: {MISSING}")
class FinalistSelection(unittest.TestCase):
    """Choosing what to confirm must be a rule, not a judgement call.

    A manual step between a search and its multi-seed confirmation is where a
    favourable point gets picked by eye. The rule is the two ends of each
    family's front: the cheapest point and the most accurate one.
    """

    def report(self, front, trials, family="siren"):
        return {"byFamily": {family: {
            "mode": "holdout", "paretoFront": front,
            "trials": [{"trial": t, "config": {"kind": family}, "recipe": {"lr": 1e-3},
                        "deployedBytes": b} for t, b in trials]}}}

    def test_it_takes_the_cheapest_and_the_most_accurate_point(self):
        front = [{"trial": 1, "deployedBytes": 1000, "scoreMaeM": 5.0},
                 {"trial": 2, "deployedBytes": 9000, "scoreMaeM": 2.0},
                 {"trial": 3, "deployedBytes": 4000, "scoreMaeM": 3.0}]
        chosen = search.finalists(self.report(front, ((1, 1000), (2, 9000), (3, 4000))))
        self.assertEqual(sorted(chosen), ["siren-t1", "siren-t2"])

    def test_a_single_point_front_yields_one_finalist_not_a_duplicate(self):
        front = [{"trial": 4, "deployedBytes": 700, "scoreMaeM": 1.0}]
        chosen = search.finalists(self.report(front, ((4, 700),)))
        self.assertEqual(list(chosen), ["siren-t4"])

    def test_the_cap_keeps_the_cheap_end_the_conventional_control_contests(self):
        report = {"byFamily": {}}
        for index, family in enumerate(("a", "b", "c", "d", "e")):
            trial = index * 10
            report["byFamily"][family] = {
                "mode": "holdout",
                "paretoFront": [{"trial": trial, "deployedBytes": 1000 * (index + 1),
                                 "scoreMaeM": 1.0}],
                "trials": [{"trial": trial, "config": {"kind": family}, "recipe": {},
                            "deployedBytes": 1000 * (index + 1)}]}
        chosen = search.finalists(report, per_family=1, limit=3)
        self.assertEqual(len(chosen), 3)
        sizes = [v["fromStudy"]["deployedBytes"] for v in chosen.values()]
        self.assertEqual(sizes, [1000, 2000, 3000])

    def test_the_cap_never_drops_a_family_while_another_holds_two_slots(self):
        """A cap that can reduce eight families to two does not give a diverse panel.

        Every family gets its first pick before any family gets a second, so the
        confirmation panel spans the architectures rather than whichever family
        happens to occupy the small end of the byte axis.
        """
        report = {"byFamily": {}}
        for index, family in enumerate(("tiny", "big", "mid")):
            base = (index + 1) * 10_000
            report["byFamily"][family] = {
                "mode": "holdout",
                "paretoFront": [
                    {"trial": index * 10, "deployedBytes": base, "scoreMaeM": 5.0},
                    {"trial": index * 10 + 1, "deployedBytes": base + 500, "scoreMaeM": 1.0}],
                "trials": [
                    {"trial": index * 10, "config": {"kind": family}, "recipe": {},
                     "deployedBytes": base},
                    {"trial": index * 10 + 1, "config": {"kind": family}, "recipe": {},
                     "deployedBytes": base + 500}]}
        chosen = search.finalists(report, per_family=2, limit=3)
        families = {entry["fromStudy"]["family"] for entry in chosen.values()}
        self.assertEqual(families, {"tiny", "big", "mid"})
        # Cheapest-first within the first rung, so the small end still wins ties.
        self.assertEqual([v["fromStudy"]["deployedBytes"] for v in chosen.values()],
                         [10_000, 20_000, 30_000])

    def test_the_output_is_the_shape_confirm_reads(self):
        front = [{"trial": 1, "deployedBytes": 1000, "scoreMaeM": 5.0}]
        chosen = search.finalists(self.report(front, ((1, 1000),)))
        entry = chosen["siren-t1"]
        self.assertIn("config", entry)
        self.assertIn("recipe", entry)
        self.assertEqual(entry["fromStudy"]["mode"], "holdout")

    def test_an_empty_front_contributes_no_finalist(self):
        self.assertEqual(search.finalists(self.report([], ())), {})


@unittest.skipIf(MISSING, f"ML extras unavailable: {MISSING}")
class Reproducibility(unittest.TestCase):
    """A trial's result must not depend on which trials ran before it.

    `fit` seeds torch at the top of the training loop, after the model has been
    constructed, so construction needs its own seed. Without it the initial
    weights come from whatever state the previous trial left in the global
    generator, and the first model built in a process comes from an unseeded one.
    """

    CONFIG = {"kind": "mlp", "width": 32, "depth": 2}

    def recipe(self, seed=4):
        return search.Recipe(steps=8, batch=256, device="cpu", seed=seed, lr=1e-3)

    def test_the_same_trial_gives_the_same_answer_after_unrelated_work(self):
        problem = _StubProblem()
        first = search.measure(self.CONFIG, self.recipe(), problem)
        torch.rand(5000)  # whatever another trial would have consumed
        second = search.measure(self.CONFIG, self.recipe(), problem)
        self.assertEqual(first["metrics"]["selection"]["mae_m"],
                         second["metrics"]["selection"]["mae_m"])

    def test_a_different_seed_still_gives_a_different_model(self):
        """Seeding construction must not pin initialisation to one value for every seed."""
        problem = _StubProblem()
        a = search.measure(self.CONFIG, self.recipe(seed=4), problem)
        b = search.measure(self.CONFIG, self.recipe(seed=5), problem)
        self.assertNotEqual(a["metrics"]["selection"]["mae_m"],
                            b["metrics"]["selection"]["mae_m"])

    def test_the_byte_precheck_does_not_shift_the_trial_it_precedes(self):
        """`study` builds a throwaway model to price it before training one."""
        problem = _StubProblem()
        clean = search.measure(self.CONFIG, self.recipe(), problem)
        search.build_model(dict(self.CONFIG), problem)  # the pre-check's throwaway
        after = search.measure(self.CONFIG, self.recipe(), problem)
        self.assertEqual(clean["metrics"]["selection"]["mae_m"],
                         after["metrics"]["selection"]["mae_m"])


@unittest.skipIf(MISSING, f"ML extras unavailable: {MISSING}")
class Ladder(unittest.TestCase):
    """The precision ladder must be re-runnable, not inherited from one checkpoint.

    float16 storage decides the neural codec baseline's verdict, so the ladder
    that measures it has to run on whatever models the current search chose.
    """

    def named(self):
        return {"siren": {"config": {"kind": "siren", "width": 16, "depth": 2},
                          "recipe": {"lr": 1e-3}},
                "codegrid": {"config": {"kind": "codegrid", "patches": 2, "latent": 4,
                                        "width": 16, "depth": 2},
                             "recipe": {"lr": 1e-3}}}

    def report(self):
        return search.ladder(self.named(), _StubProblem(), steps=6, batch=128,
                             precisions=("float32", "float16"), regions=(1, 16))

    def test_it_runs_the_ladder_on_an_unrounded_model(self):
        """A float32 row taken from an already-rounded model would hide the rounding error."""
        report = self.report()
        row = next(r for r in report["rows"] if r["name"] == "siren")
        ladder = {entry["precision"]: entry for entry in row["precisionLadder"]["rows"]}
        # Strictly above one half because the safetensors header does not shrink
        # with the payload; on a model this small that overhead is a fifth of the
        # file. What matters is that the bytes move and the error moves with them,
        # which is only true if the ladder started from unrounded weights.
        ratio = ladder["float16"]["weightsBytes"] / ladder["float32"]["weightsBytes"]
        self.assertGreater(ratio, 0.5)
        self.assertLess(ratio, 0.7)
        self.assertNotEqual(ladder["float16"]["mae_m"], ladder["float32"]["mae_m"])

    def test_only_a_family_with_codes_pays_per_region(self):
        report = {r["name"]: r for r in self.report()["rows"]}
        plain = report["siren"]["amortisation"]["byRegionCount"]
        coded = report["codegrid"]["amortisation"]["byRegionCount"]
        self.assertEqual(report["siren"]["storedCodeParameters"], 0)
        self.assertEqual(plain[0]["totalBytes"], plain[-1]["totalBytes"])
        self.assertGreater(report["codegrid"]["storedCodeParameters"], 0)
        self.assertGreater(coded[-1]["totalBytes"], coded[0]["totalBytes"])

    def test_code_coverage_is_reported_for_the_family_that_needs_it(self):
        report = {r["name"]: r for r in self.report()["rows"]}
        self.assertIsNone(report["siren"]["codeCoverage"])
        self.assertIsNotNone(report["codegrid"]["codeCoverage"])
        self.assertIn("coverageFraction", report["codegrid"]["codeCoverage"])


@unittest.skipIf(MISSING, f"ML extras unavailable: {MISSING}")
class ConfirmationChain(unittest.TestCase):
    """search -> finalists -> confirm, end to end.

    `confirm` runs last, after the expensive search, so a failure there wastes
    the whole run.
    """

    @classmethod
    def setUpClass(cls):
        problem = _StubProblem()
        cls.report = {"byFamily": {"siren": search.study(
            "siren", problem, trials=3, steps=6, batch=256, progress=False)}}
        cls.chosen = search.finalists(cls.report, per_family=2)

    def test_the_search_output_feeds_the_finalist_rule_without_editing(self):
        self.assertGreaterEqual(len(self.chosen), 1)
        for entry in self.chosen.values():
            self.assertIn("kind", entry["config"])
            self.assertIn("lr", entry["recipe"])

    def test_confirm_reports_a_spread_across_seeds(self):
        report = search.confirm(self.chosen, _StubProblem(), seeds=(1, 2, 3),
                                steps=6, batch=256)
        row = report["rows"][0]
        self.assertEqual(len(row["perSeed"]), 3)
        self.assertEqual(row["seeds"], [1, 2, 3])
        self.assertIn("selection.mae_m", row["summary"])
        self.assertIn("test.mae_m", row["summary"])
        spread = row["summary"]["selection.mae_m"]
        self.assertGreaterEqual(spread["max"], spread["median"])
        self.assertGreaterEqual(spread["median"], spread["min"])
        self.assertGreaterEqual(spread["spreadFraction"], 0.0)

    def test_three_seeds_do_not_all_give_the_same_answer(self):
        """If they did, the seed panel would be measuring nothing."""
        report = search.confirm(self.chosen, _StubProblem(), seeds=(1, 2, 3),
                                steps=6, batch=256)
        values = [run["metrics"]["selection"]["mae_m"]
                  for run in report["rows"][0]["perSeed"]]
        self.assertEqual(len(set(values)), 3, values)

    def test_confirm_can_be_told_not_to_read_the_test_half(self):
        report = search.confirm(self.chosen, _StubProblem(), seeds=(1,), steps=6,
                                batch=256, report_test=False)
        self.assertNotIn("test.mae_m", report["rows"][0]["summary"])
        self.assertIn("selection.mae_m", report["rows"][0]["summary"])

    def test_confirm_records_the_width_it_priced(self):
        report = search.confirm(self.chosen, _StubProblem(), seeds=(1,), steps=6, batch=256)
        self.assertEqual(report["storePrecision"], "float16")
        row = report["rows"][0]
        self.assertLess(row["deployedBytes"], row["deployedBytesFloat32"])


@unittest.skipIf(MISSING, f"ML extras unavailable: {MISSING}")
class CodecFitAndEnvelope(unittest.TestCase):
    """The two functions that produce the neural against conventional comparison.

    `codec_fit` is the only mode whose numbers belong beside the conventional
    rate-distortion curve, and `conventional_envelope` is what they are compared
    against. Both are tested for the properties that would silently invalidate
    the comparison: that codec fit trains on every page,
    that it reports the float16 rate as well as the float32 one, and that the
    envelope's maximum-error front never claims a bound for a representation that
    has none.
    """

    def test_codec_fit_trains_on_every_page_and_says_so(self):
        chosen = {"siren": {"config": {"kind": "siren", "width": 16, "depth": 2},
                            "recipe": {"lr": 1e-3}}}
        report = search.codec_fit(chosen, _StubProblem(), seeds=(1,), steps=6, batch=128,
                                  stream_cells=20)
        self.assertIn("codec fit", report["mode"])
        self.assertIn("NOT evidence of generalization", report["qualification"])
        run = report["rows"][0]["runs"][0]
        # Full reference, not a sample: every node of the lattice.
        self.assertIn("fullReference", run)
        self.assertIn("drainage", run)

    def test_codec_fit_reports_both_storage_rates(self):
        chosen = {"siren": {"config": {"kind": "siren", "width": 16, "depth": 2},
                            "recipe": {"lr": 1e-3}}}
        run = search.codec_fit(chosen, _StubProblem(), seeds=(1,), steps=6, batch=128,
                               stream_cells=20)["rows"][0]["runs"][0]
        self.assertLess(run["halfPrecision"]["deployedBytes"], run["deployedBytes"])
        self.assertIn("maeChangeM", run["halfPrecision"])

    def test_the_envelope_never_claims_a_bound_a_representation_does_not_have(self):
        envelope = search.conventional_envelope(_StubProblem(), targets=(1.0, 0.1),
                                                levels=(1, 2))
        for point in envelope["paretoByMax"]:
            if point["kind"] == "downsampled-bilinear":
                self.assertFalse(point["boundGuaranteed"])
        self.assertTrue(any(p["boundGuaranteed"] for p in envelope["points"]))
        self.assertTrue(any(not p["boundGuaranteed"] for p in envelope["points"]))

    def test_the_envelope_front_is_monotone_in_bytes_and_error(self):
        envelope = search.conventional_envelope(_StubProblem(), targets=(1.0, 0.1),
                                                levels=(1, 2))
        for name in ("paretoByMae", "paretoByMax"):
            front = envelope[name]
            metric = "maeM" if name == "paretoByMae" else "maxM"
            for earlier, later in zip(front, front[1:]):
                self.assertLess(earlier["deployedBytes"], later["deployedBytes"], name)
                self.assertLess(later[metric], earlier[metric], name)


@unittest.skipIf(MISSING, f"ML extras unavailable: {MISSING}")
class RandomAccessOverhead(unittest.TestCase):
    """Paging a grid into independently decodable units costs bytes, and the
    number must be able to say so.

    `randomAccessOverheadFraction` is the paged byte count over the monolithic
    one, minus one; dividing the paged figure by itself would report zero in
    every row. The overhead is why the same grid costs 81,974 bytes paged and
    65,235 in a single stream, and a front that used the smaller number on one
    side would be comparing paged against unpaged.
    """

    def test_paging_a_grid_costs_more_than_one_stream(self):
        problem = _StubProblem()
        curve = search.base_only_curve(problem, levels=(1,), targets=(1.0,))
        row = curve["rows"][0]
        self.assertGreater(row["deployedBytes"], row["monolithicBytes"])
        self.assertGreater(row["randomAccessOverheadFraction"], 0.0)

    def test_the_overhead_matches_the_two_figures_it_is_derived_from(self):
        row = search.base_only_curve(_StubProblem(), levels=(1,), targets=(1.0,))["rows"][0]
        self.assertAlmostEqual(
            row["randomAccessOverheadFraction"],
            row["deployedBytes"] / row["monolithicBytes"] - 1.0, places=9)


@unittest.skipIf(MISSING, f"ML extras unavailable: {MISSING}")
class RateDenominators(unittest.TestCase):
    """Two sample counts, and they are not interchangeable.

    The conventional codec comparison divides by every sample it stores across
    the pyramid; a coordinate network stores no pyramid, so its natural
    denominator is the unique samples of the finest domain. On the Essen-Ruhr
    atlas the two differ by 1.371, so a neural bits-per-sample set beside a
    conventional bits-per-sample without that conversion understates the
    conventional side by 37 per cent.
    """

    def test_the_multiscale_count_matches_the_tournament_it_must_be_compared_to(self):
        """Computed from the lattice, checked against the conventional comparison.

        1,440,725 is the `samples_across_levels` the conventional codec
        comparison records for the 1025-node, 64-interval atlas. Deriving it
        independently is what justifies the conversion factor.
        """
        problem = _StubProblem(side=1025, intervals=64)
        counts = search.rate_denominators(problem)
        self.assertEqual(counts["uniqueFinestSamples"], 1025 * 1025)
        self.assertEqual(counts["samplesAcrossLevels"], 1_440_725)
        self.assertAlmostEqual(counts["multiscaleOverPrimary"], 1.3713, places=4)

    def test_the_multiscale_count_always_exceeds_the_finest_one(self):
        """Shared page edges and stored coarse levels both only ever add samples."""
        for side, intervals in ((129, 64), (257, 64), (513, 128)):
            with self.subTest(side=side, intervals=intervals):
                counts = search.rate_denominators(_StubProblem(side=side, intervals=intervals))
                self.assertGreater(counts["samplesAcrossLevels"], counts["uniqueFinestSamples"])
                self.assertGreater(counts["multiscaleOverPrimary"], 1.0)

    def test_the_primary_denominator_is_named_so_a_report_cannot_be_ambiguous(self):
        counts = search.rate_denominators(_StubProblem())
        self.assertEqual(counts["primary"], "uniqueFinestSamples")
        self.assertIn("multiscaleOverPrimary", counts["note"])


@unittest.skipIf(MISSING, f"ML extras unavailable: {MISSING}")
class ScoredSplitIsComplete(unittest.TestCase):
    """A study's objective must be measured on every node it claims to cover.

    `measure` subsamples to 200,000 nodes by default, which has no effect on the
    selection half (190,512 nodes, so it fits) and is wrong for codec mode, which
    scores all 1,050,625. A sampled maximum recorded as `scoreMaxM` would be
    mislabelled: no sampled maximum may be called a global error bound, and the
    drainage acceptance check reads the maximum before the mean.
    """

    def test_both_modes_evaluate_every_node_of_the_split_they_score(self):
        problem = _StubProblem()
        for mode, key in (("holdout", "selection"), ("codec", "all")):
            with self.subTest(mode):
                report = search.study("siren", problem, trials=1, steps=4, batch=128,
                                      progress=False, mode=mode)
                record = report["trials"][0]
                self.assertEqual(record["metrics"][key]["samples"],
                                 int(problem.indexes[key].size))

    def test_the_codec_split_is_larger_than_the_default_sampling_limit_would_allow(self):
        """The control: if it were not, the test above would pass for free."""
        problem = _StubProblem(side=1025, intervals=64)
        self.assertGreater(problem.indexes["all"].size, 200_000)


@unittest.skipIf(MISSING, f"ML extras unavailable: {MISSING}")
class NormalisationScope(unittest.TestCase):
    """Where the two normalisation scalars come from is part of the contract.

    Whole-reference mean and standard deviation leak withheld target information
    into a holdout experiment's preprocessing. The rule differs by experiment:
    full-target statistics are allowed for codec encoding provided they are
    stored, and are not allowed for a predictive holdout.
    """

    def test_the_two_scopes_give_different_statistics(self):
        """The control. If the training pages happened to have the same mean as
        the whole region, every other test here would pass for free."""
        whole = _StubProblem(normalisation="all")
        train = _StubProblem(normalisation="train")
        self.assertNotEqual(whole.mean, train.mean)

    def test_training_scope_touches_no_page_outside_the_training_region(self):
        problem = _StubProblem(normalisation="train")
        inside = problem.flat[problem.indexes["train"]]
        self.assertAlmostEqual(problem.mean, float(np.mean(inside)), places=12)
        self.assertAlmostEqual(problem.scale, max(float(np.std(inside)), 1.0), places=12)

    def test_an_unknown_scope_is_refused(self):
        with self.assertRaises(ValueError):
            _StubProblem(normalisation="selection")

    def test_a_model_can_be_trained_and_scored_under_either_scope(self):
        for scope in ("all", "train"):
            with self.subTest(scope):
                result = search.measure({"kind": "siren", "width": 16, "depth": 2},
                                        search.Recipe(steps=6, batch=128, device="cpu",
                                                      seed=3, lr=1e-3),
                                        _StubProblem(normalisation=scope))
                self.assertTrue(np.isfinite(result["metrics"]["selection"]["mae_m"]))


@unittest.skipIf(MISSING, f"ML extras unavailable: {MISSING}")
class EnvelopeSplit(unittest.TestCase):
    """A neural score and the front it is compared against must read the same pages.

    The selection half is measurably easier terrain than the whole field:
    conventional error there is 0.07 to 0.50 m lower on the Essen-Ruhr atlas. A
    holdout search scores the selection half, so comparing it against a
    whole-field conventional front credits the network with a property of the
    terrain rather than of the representation, in its favour.
    """

    def test_the_front_can_be_read_on_either_split_and_says_which(self):
        problem = _StubProblem()
        for split in ("all", "selection"):
            with self.subTest(split):
                envelope = search.conventional_envelope(problem, targets=(1.0,), levels=(1,),
                                                        split=split)
                self.assertEqual(envelope["split"], split)
                self.assertTrue(envelope["points"])

    def test_an_unknown_split_is_refused_rather_than_silently_falling_back(self):
        with self.assertRaises(ValueError):
            search.conventional_envelope(_StubProblem(), targets=(1.0,), levels=(1,),
                                         split="holdout")

    def test_both_arms_of_the_front_follow_the_requested_split(self):
        """Not just the downsampled arm: the quantised arm is also scored per
        split rather than taking the whole-field figure from `codecs.measure`."""
        problem = _StubProblem()
        whole = search.conventional_envelope(problem, targets=(1.0,), levels=(1,), split="all")
        part = search.conventional_envelope(problem, targets=(1.0,), levels=(1,),
                                            split="selection")
        for kind in ("full-resolution-quantized", "downsampled-bilinear"):
            a = next(p for p in whole["points"] if p["kind"] == kind)
            b = next(p for p in part["points"] if p["kind"] == kind)
            self.assertEqual(a["deployedBytes"], b["deployedBytes"], kind)
            self.assertNotEqual(a["maeM"], b["maeM"], kind)

    def test_the_guaranteed_bound_stays_a_whole_field_statement(self):
        """A per-split maximum is an observation on a subset and is never larger
        than the target the codec actually guarantees."""
        curve = search.conventional_curve(_StubProblem(), targets=(1.0,))
        row = curve["rows"][0]
        self.assertLessEqual(row["metrics"]["selection"]["max_m"], row["maxErrorM"] + 1e-9)
        self.assertAlmostEqual(row["metrics"]["all"]["max_m"], row["maxErrorM"], places=6)


@unittest.skipIf(MISSING, f"ML extras unavailable: {MISSING}")
class ContainerShare(unittest.TestCase):
    """The one accounting asymmetry that runs against the network.

    A neural row is charged its safetensors header; the conventional index, which
    names and locates 256 pages, is deliberately charged to neither side. The
    header is under one per cent of a deployable model and about forty per cent
    of a very small one, so it matters only at the cheap end of a front. It is
    charged, and reported separately.
    """

    def test_the_container_is_charged_and_reported_separately(self):
        result = search.measure({"kind": "siren", "width": 16, "depth": 2},
                                search.Recipe(steps=6, batch=128, device="cpu", seed=3, lr=1e-3),
                                _StubProblem())
        self.assertEqual(result["weightsBytes"],
                         result["tensorBytes"] + result["containerBytes"])
        self.assertGreater(result["containerBytes"], 0)

    def test_the_container_share_falls_as_the_model_grows(self):
        """Which is why it can only distort the cheap end of a front."""
        recipe = search.Recipe(steps=6, batch=128, device="cpu", seed=3, lr=1e-3)
        problem = _StubProblem()
        small = search.measure({"kind": "siren", "width": 16, "depth": 2}, recipe, problem)
        large = search.measure({"kind": "siren", "width": 128, "depth": 3}, recipe, problem)
        self.assertGreater(small["containerBytes"] / small["weightsBytes"],
                           large["containerBytes"] / large["weightsBytes"])
        self.assertLess(large["containerBytes"] / large["weightsBytes"], 0.02)


@unittest.skipIf(MISSING, f"ML extras unavailable: {MISSING}")
class PreferredArchitectureIsReachable(unittest.TestCase):
    """The preferred design must be somewhere in the search space.

    The preferred design is a coarse base plus a shared modulated decoder with
    local codes. `residual` samples its inner model from siren, grid and mlp,
    none of which carries codes, so a separate `hybrid` arm samples it. It is a
    separate arm rather than a fourth `inner` option so that `residual` trials
    keep sampling from the same space.
    """

    def test_the_hybrid_arm_builds_a_coarse_base_carrying_a_code_grid(self):
        problem = _StubProblem()
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=5))
        built = 0
        for _ in range(12):
            trial = study.ask()
            config = search.sample_model(trial, "hybrid")
            self.assertEqual(config["kind"], "residual")
            self.assertEqual(config["inner"]["kind"], "codegrid")
            model, extra = search.build_model(config, problem)
            self.assertGreater(extra["baseBytes"], 0)
            self.assertGreater(search.code_parameter(model)[1].numel(), 0)
            built += 1
            study.tell(trial, 0.0)
        self.assertEqual(built, 12)

    def test_no_residual_inner_family_carries_local_codes(self):
        """The control: if one did, the hybrid arm would be redundant."""
        problem = _StubProblem()
        for inner in search.RESIDUAL_INNER:
            with self.subTest(inner):
                config = {"kind": "residual", "base_level": 1, "base_target_m": 1.0,
                          "inner": {"kind": inner} if inner != "grid" else
                          {"kind": "grid", "levels": 3, "width": 16, "depth": 2}}
                model, _ = search.build_model(config, problem)
                self.assertIsNone(search.code_parameter(model), inner)

    def test_the_screening_default_matches_the_reports_stated_configuration(self):
        """The preferred configuration: width 128, three layers, 32-dimensional
        codes on 128-interval patches. At 64 intervals per page on a 1025-node lattice,
        8 patches a side is 128 sample intervals."""
        import inspect
        source = inspect.getsource(search.screen)
        self.assertIn('"hybrid"', source)
        problem = _StubProblem()
        defaults = {"kind": "residual", "base_level": 1, "base_target_m": 1.0,
                    "inner": {"kind": "codegrid", "patches": 8, "latent": 32,
                              "width": 128, "depth": 3}}
        model, extra = search.build_model(defaults, problem)
        self.assertEqual(search.code_parameter(model)[1].shape[-1], 32)


@unittest.skipIf(MISSING, f"ML extras unavailable: {MISSING}")
class WrappedCodesAreFound(unittest.TestCase):
    """A code payload inside a wrapper is still a code payload.

    In a code grid carried by a residual base (the preferred design) the
    parameter is `inner.codes`, not a top-level "codes" or "codes.weight". If
    `code_parameter` missed it, `code_coverage` would report nothing for that
    architecture, silently disabling the coverage diagnostic on the model where
    it matters most.
    """

    def test_codes_are_found_through_a_residual_wrapper(self):
        problem = _StubProblem()
        config = {"kind": "residual", "base_level": 1, "base_target_m": 1.0,
                  "inner": {"kind": "codegrid", "patches": 4, "latent": 8,
                            "width": 32, "depth": 2}}
        model, _ = search.build_model(config, problem)
        found = search.code_parameter(model)
        self.assertIsNotNone(found)
        self.assertTrue(found[0].endswith("codes"), found[0])
        self.assertEqual(found[1].shape[-1], 8)

    def test_coverage_is_reported_for_a_wrapped_code_grid(self):
        problem = _StubProblem()
        config = {"kind": "residual", "base_level": 1, "base_target_m": 1.0,
                  "inner": {"kind": "codegrid", "patches": 4, "latent": 8,
                            "width": 32, "depth": 2}}
        model, _ = search.build_model(config, problem)
        coverage = search.code_coverage(model, problem, shared=False)
        self.assertIsNotNone(coverage)
        self.assertGreater(coverage["coverageFraction"], 0.0)

    def test_a_wrapper_without_codes_still_reports_none(self):
        problem = _StubProblem()
        model, _ = search.build_model(
            {"kind": "residual", "base_level": 1, "base_target_m": 1.0,
             "inner": {"kind": "siren", "width": 16, "depth": 2}}, problem)
        self.assertIsNone(search.code_parameter(model))


@unittest.skipIf(MISSING, f"ML extras unavailable: {MISSING}")
class HybridSurvivesTheRoundTrip(unittest.TestCase):
    """The hybrid arm is the most nested config, so it is the one likeliest to
    fail between a search report and the confirmation that reads it back.

    `build_model` pops `base_level` and `base_target_m` from its own copy and
    `describe` strips the base tensor; if either operated on the caller's dict,
    the config written into a report would be missing the fields needed to
    rebuild it, and confirm would fail hours after the search finished.
    """

    def test_a_hybrid_config_survives_json_and_rebuilds(self):
        import json
        problem = _StubProblem()
        config = {"kind": "residual", "base_level": 1, "base_target_m": 1.0,
                  "inner": {"kind": "codegrid", "patches": 4, "latent": 8,
                            "width": 32, "depth": 2, "omega": 14.0}}
        model, extra = search.build_model(config, problem)
        described = search.describe(config)
        # The original must be untouched by building from it.
        self.assertIn("base_level", config)
        self.assertIn("base_level", described)
        self.assertNotIn("base", described)
        revived = json.loads(json.dumps(described))
        again, extra_again = search.build_model(revived, problem)
        self.assertEqual(extra["baseBytes"], extra_again["baseBytes"])
        self.assertEqual(search.code_parameter(again)[1].shape,
                         search.code_parameter(model)[1].shape)

    def test_a_hybrid_measurement_reports_a_base_and_a_code_payload(self):
        result = search.measure(
            {"kind": "residual", "base_level": 1, "base_target_m": 1.0,
             "inner": {"kind": "codegrid", "patches": 4, "latent": 8, "width": 32, "depth": 2}},
            search.Recipe(steps=6, batch=128, device="cpu", seed=3, lr=1e-3), _StubProblem())
        self.assertGreater(result["baseBytes"], 0)
        self.assertEqual(result["deployedBytes"],
                         result["weightsBytes"] + result["baseBytes"])
        coverage = search.code_coverage(result["model"], _StubProblem(), result["shared"])
        self.assertIsNotNone(coverage)


class ComparatorAgreesWithAnIndependentRoute(unittest.TestCase):
    """The conventional envelope is what every neural claim is measured against,
    so it is worth computing twice by different means.

    This recomputes one envelope point without `search.py`: it reads the atlas's
    own page data, encodes with the registry codec directly, upsamples with a
    hand-written bilinear rather than `ResidualDecoder`, and masks with the split
    module. Agreement to four decimals on both the selection half and the whole
    field means the comparator is not an artefact of the code path that produces
    it. The gap between the two numbers is why a neural score and its comparator
    must be read on the same split.

    The reference point comes from the published quantised-ladder report. Skipped
    when the prepared atlas is absent (build it with `geoneural prepare`); a
    synthetic stand-in would verify nothing.
    """

    LEVEL, TARGET = 2, 0.5

    @classmethod
    def setUpClass(cls):
        from pathlib import Path
        from geoneural.common import HOME
        cls.atlas = Path(HOME) / "atlases/essen-ruhr/atlas.json"
        cls.report = Path(__file__).resolve().parents[1] / "results" / "reproduced" / "quantised-ladder.json"
        if not cls.atlas.exists():
            raise unittest.SkipTest(f"prepared atlas absent: {cls.atlas}")

    def recompute(self):
        from geoneural.codecs import bounds
        from geoneural.codecs import codecs
        from geoneural.neural import splits
        from geoneural.common import read_json
        manifest = read_json(self.atlas)
        reference = np.load(self.atlas.parent / "reference.npy",
                            allow_pickle=False).astype(np.float64)
        side = reference.shape[0]
        split = splits.build(side, int(manifest["page_intervals"]), 0.25, selection_split=True)
        grid = bounds.level_grid(self.atlas.parent, manifest, self.LEVEL).astype(np.float32)
        codec = codecs.registry()["q32-delta-zstd"]
        decoded = codec.decode(codec.encode(grid, self.TARGET)).astype(np.float64)
        step = (side - 1) // (decoded.shape[0] - 1)
        rows = np.arange(side) / step
        r0 = np.clip(np.floor(rows).astype(int), 0, decoded.shape[0] - 2)
        weight = (rows - r0)[:, None]
        c0, wc = r0, weight.T
        top = decoded[r0][:, c0] * (1 - wc) + decoded[r0][:, c0 + 1] * wc
        bottom = decoded[r0 + 1][:, c0] * (1 - wc) + decoded[r0 + 1][:, c0 + 1] * wc
        error = np.abs(((top * (1 - weight) + bottom * weight)).reshape(-1)
                       - reference.reshape(-1))
        return error, split["selectionMask"].reshape(-1)

    def test_the_two_routes_agree_on_both_splits(self):
        import json
        if not self.report.exists():
            self.skipTest(f"no screening report to compare against: {self.report}")
        error, selection = self.recompute()
        row = next(r for r in json.loads(self.report.read_text())["conventional"]["coarse"]["rows"]
                   if r["baseLevel"] == self.LEVEL and r["baseTargetM"] == self.TARGET)
        self.assertAlmostEqual(float(error[selection].mean()),
                               row["metrics"]["selection"]["mae_m"], places=4)
        self.assertAlmostEqual(float(error.mean()), row["metrics"]["all"]["mae_m"], places=4)

    def test_the_selection_half_really_is_the_easier_terrain(self):
        """The control: if these agreed, reading the comparator per split would not matter."""
        error, selection = self.recompute()
        self.assertLess(float(error[selection].mean()), float(error.mean()))


class BaseContributionIsAttached(unittest.TestCase):
    """A residual row must carry the control that decides whether it is a win.

    `residual` and `hybrid` stand on a conventional coarse grid. In one measured
    multi-seed panel both ranked three times better than every pure network, yet
    the 8,723-byte base alone scored 1.346 m on selection against the combined
    model's 1.770 m: the network made its base worse by 0.42 m for 67,754 extra
    bytes. These tests keep each row joined to its base control.
    """

    def test_a_network_that_helps_is_marked_as_earning_its_bytes(self):
        row = {"deployedBytes": 76_477,
               "summary": {"selection.mae_m": {"median": 0.9},
                           "test.mae_m": {"median": 1.1}}}
        control = {"baseLevel": 3, "baseTargetM": 1.0, "deployedBytes": 8_723,
                   "metrics": {"selection": {"mae_m": 1.346}, "test": {"mae_m": 1.571}}}
        verdict = search.base_contribution(row, control)
        self.assertTrue(verdict["networkEarnsItsBytes"])
        self.assertAlmostEqual(verdict["bySplit"]["selection"]["deltaM"], -0.446, places=3)
        self.assertEqual(verdict["networkBytes"], 67_754)

    def test_the_measured_panel_row_is_reported_as_a_loss(self):
        """Measured numbers from a real panel row."""
        row = {"deployedBytes": 76_477,
               "summary": {"selection.mae_m": {"median": 1.770},
                           "test.mae_m": {"median": 1.906}}}
        control = {"baseLevel": 3, "baseTargetM": 1.0, "deployedBytes": 8_723,
                   "metrics": {"selection": {"mae_m": 1.346}, "test": {"mae_m": 1.571}}}
        verdict = search.base_contribution(row, control)
        self.assertFalse(verdict["networkEarnsItsBytes"])
        self.assertGreater(verdict["bySplit"]["selection"]["deltaM"], 0.4)
        self.assertGreater(verdict["bySplit"]["test"]["deltaM"], 0.3)

    def test_a_split_that_was_not_evaluated_is_simply_absent(self):
        row = {"deployedBytes": 1000, "summary": {"selection.mae_m": {"median": 1.0}}}
        control = {"baseLevel": 3, "baseTargetM": 1.0, "deployedBytes": 100,
                   "metrics": {"selection": {"mae_m": 2.0}, "test": {"mae_m": 2.0}}}
        verdict = search.base_contribution(row, control)
        self.assertEqual(set(verdict["bySplit"]), {"selection"})
        self.assertTrue(verdict["networkEarnsItsBytes"])

    def test_a_row_with_no_evaluated_split_never_claims_a_win(self):
        row = {"deployedBytes": 1000, "summary": {}}
        control = {"baseLevel": 3, "baseTargetM": 1.0, "deployedBytes": 100,
                   "metrics": {"selection": {"mae_m": 2.0}, "test": {"mae_m": 2.0}}}
        self.assertFalse(search.base_contribution(row, control)["networkEarnsItsBytes"])


class ResidualFrontsCarryTheirBase(unittest.TestCase):
    """Each residual Pareto point stands on its own base, and must say which."""

    def _report(self):
        return {"byFamily": {
            "siren": {"paretoFront": [{"deployedBytes": 1000, "scoreMaeM": 1.0,
                                       "params": {"width": 32}}]},
            "residual": {"paretoFront": [
                {"deployedBytes": 76477, "scoreMaeM": 1.770,
                 "params": {"base_level": 3, "base_target_m": 1.0}},
                {"deployedBytes": 40000, "scoreMaeM": 0.500,
                 "params": {"base_level": 2, "base_target_m": 0.5}}]}}}

    def test_points_are_keyed_on_their_own_base_not_a_shared_default(self):
        calls = []

        class _Problem:
            pass

        def fake_base_row(problem, level, target):
            calls.append((level, target))
            mae = {3: 1.346, 2: 0.682}[level]
            return {"baseLevel": level, "baseTargetM": target, "deployedBytes": 8723,
                    "metrics": {"all": {"mae_m": mae, "max_m": 18.41}}}

        report = self._report()
        with unittest.mock.patch.object(search, "base_row", fake_base_row):
            search.annotate_residual_fronts(report, _Problem())
        self.assertEqual(calls, [(3, 1.0), (2, 0.5)])
        front = report["byFamily"]["residual"]["paretoFront"]
        self.assertFalse(front[0]["baseContribution"]["networkEarnsItsBytes"])
        self.assertTrue(front[1]["baseContribution"]["networkEarnsItsBytes"])
        self.assertAlmostEqual(front[0]["baseContribution"]["deltaM"], 0.424, places=3)

    def test_a_base_is_evaluated_once_per_distinct_level_and_target(self):
        calls = []

        def fake_base_row(problem, level, target):
            calls.append((level, target))
            return {"baseLevel": level, "baseTargetM": target, "deployedBytes": 8723,
                    "metrics": {"all": {"mae_m": 1.346, "max_m": 18.41}}}

        report = self._report()
        report["byFamily"]["residual"]["paretoFront"].append(
            {"deployedBytes": 90000, "scoreMaeM": 2.0,
             "params": {"base_level": 3, "base_target_m": 1.0}})
        with unittest.mock.patch.object(search, "base_row", fake_base_row):
            search.annotate_residual_fronts(report, object())
        self.assertEqual(len(calls), 2, "the repeated (3, 1.0) base must be cached")

    def test_a_family_without_a_base_is_left_alone(self):
        report = self._report()
        with unittest.mock.patch.object(search, "base_row", lambda *a: self.fail("no base here")):
            search.annotate_residual_fronts(
                {"byFamily": {"siren": report["byFamily"]["siren"]}}, object())


class ConvergenceAgainstTheConstantPredictor(unittest.TestCase):
    """A row at the field mean did not train, and must not be ranked as if it had.

    A measured `shared` row scored 23.282 m on selection, where the constant
    predictor scores 23.323 m. That is an optimisation that never started, not an
    architecture in last place.
    """

    def test_the_measured_shared_row_is_not_trained(self):
        verdict = search.convergence(23.282, 23.323)
        self.assertFalse(verdict["trained"])
        self.assertLess(verdict["skill"], 0.01)

    def test_a_real_row_is_trained(self):
        self.assertTrue(search.convergence(1.770, 23.323)["trained"])
        self.assertGreater(search.convergence(1.770, 23.323)["skill"], 0.9)

    def test_worse_than_the_constant_is_negative_skill_and_untrained(self):
        verdict = search.convergence(30.0, 23.323)
        self.assertLess(verdict["skill"], 0.0)
        self.assertFalse(verdict["trained"])

    def test_the_floor_is_a_liveness_check_not_a_quality_bar(self):
        """Two per cent of the constant's error is enough to count as trained."""
        self.assertTrue(search.convergence(23.323 * 0.97, 23.323)["trained"])
        self.assertFalse(search.convergence(23.323 * 0.99, 23.323)["trained"])

    def test_collapsed_trials_are_counted_per_family_not_dropped(self):
        report = {"byFamily": {"shared": {"trials": [
            {"metrics": {"all": {"mae_m": 23.50}}},
            {"metrics": {"all": {"mae_m": 0.915}}},
            {"metrics": {"all": {"mae_m": 23.49}}},
            {"params": {}}]}}}

        class _Problem:
            flat = np.zeros(100)
            indexes = {"all": np.arange(100)}

        problem = _Problem()
        problem.flat = np.concatenate([np.full(50, -23.5), np.full(50, 23.5)])
        search.annotate_convergence(report, problem, split="all")
        family = report["byFamily"]["shared"]
        self.assertEqual(family["scoredTrials"], 3)
        self.assertEqual(family["nonConvergentTrials"], 2)
        self.assertEqual(report["convergenceByFamily"]["shared"],
                         {"scored": 3, "nonConvergent": 2})

    def test_the_baseline_is_reported_per_split(self):
        class _Problem:
            pass

        problem = _Problem()
        problem.flat = np.arange(100, dtype=float)
        problem.indexes = {"selection": np.arange(0, 50), "all": np.arange(100)}
        floor = search.constant_baseline(problem, ("selection", "all"))
        self.assertAlmostEqual(floor["constantM"], 49.5)
        self.assertEqual(floor["bySplit"]["selection"]["samples"], 50)
        self.assertGreater(floor["bySplit"]["all"]["maeM"],
                           floor["bySplit"]["selection"]["maeM"] * 0.0)


class DominanceNotRatio(unittest.TestCase):
    """A neural claim is dominance against the whole front, not a ratio at its rate.

    The conventional front is a staircase: within a pyramid level, 3.1x the bytes
    buys 3% of the error (level 3 runs 8,723 B / 1.567 m to 27,360 B / 1.516 m).
    A network landing between levels shows a flattering ratio while beating
    nothing. These are measured points.
    """

    FRONT = [(8_723, 1.567, 21.07, "L3@1.0"), (20_020, 1.059, 14.69, "L2@2.0"),
             (27_388, 0.835, 13.61, "L2@1.0"), (61_728, 0.730, 13.71, "L2@0.1"),
             (81_974, 0.496, 8.94, "L1@1.0"), (111_219, 0.358, 8.94, "L1@0.5")]

    def test_the_fourier_point_is_dominated_once_cheap_targets_exist(self):
        verdict = search.dominance(21_608, 1.368, self.FRONT, max_m=19.20)
        self.assertTrue(verdict["dominated"])
        self.assertEqual(verdict["dominatedBy"]["label"], "L2@2.0")

    def test_the_same_point_survives_a_front_that_stops_at_one_metre(self):
        """A front capped at 1 m makes the same point look undominated."""
        capped = [p for p in self.FRONT if p[3] != "L2@2.0"]
        self.assertFalse(search.dominance(21_608, 1.368, capped, max_m=19.20)["dominated"])

    def test_the_siren_point_survives_and_states_both_neighbours(self):
        verdict = search.dominance(68_458, 0.526, self.FRONT)
        self.assertFalse(verdict["dominated"])
        self.assertEqual(verdict["cheaperThanNearestBetter"]["label"], "L1@1.0")
        self.assertAlmostEqual(verdict["cheaperThanNearestBetter"]["fraction"], 0.165, places=2)
        self.assertEqual(verdict["betterThanBestCheaper"]["label"], "L2@0.1")
        self.assertAlmostEqual(verdict["betterThanBestCheaper"]["fraction"], 0.279, places=2)

    def test_a_dominator_must_also_win_the_maximum_when_one_is_given(self):
        """The drainage acceptance check reads the maximum, so a mean-only
        dominator is not one."""
        front = [(10_000, 1.0, 99.0, "cheap-but-spiky")]
        self.assertFalse(search.dominance(20_000, 1.5, front, max_m=20.0)["dominated"])
        self.assertTrue(search.dominance(20_000, 1.5, front)["dominated"])

    def test_a_point_cheaper_than_everything_is_never_dominated(self):
        verdict = search.dominance(1_714, 7.485, self.FRONT)
        self.assertFalse(verdict["dominated"])
        self.assertIsNone(verdict.get("betterThanBestCheaper"))


class CoarseAxisIsSweptBeyondTheAccuracyTargets(unittest.TestCase):
    """The downsample axis is a rate knob, not a guarantee, and needs coarse rates."""

    def test_coarse_targets_reach_further_than_the_declared_accuracy_targets(self):
        self.assertGreater(max(search.COARSE_TARGETS), max(search.CONVENTIONAL_TARGETS))
        self.assertGreaterEqual(max(search.COARSE_TARGETS), 16.0)

    def test_the_accuracy_targets_are_left_alone(self):
        """The declared accuracy guarantees must not change when the coarse axis does."""
        self.assertEqual(search.CONVENTIONAL_TARGETS, (0.01, 0.05, 0.1, 0.5, 1.0))


class SurvivingOnTheMaximumIsItsOwnClaim(unittest.TestCase):
    """Two of siren's measured points are dearer and less accurate than a coarse
    grid, and stay on the joint front only because their maximum is lower. That is
    a real result under the drainage acceptance check, but it is not the result
    "cheaper than conventional", and the verdict has to say which one it is."""

    FRONT = [(2_729, 2.888, 28.91, "L4@1.0"), (11_235, 1.533, 22.19, "L3@0.5"),
             (27_388, 0.835, 13.61, "L2@1.0"), (61_728, 0.730, 13.71, "L2@0.1")]

    def test_a_point_kept_alive_by_its_maximum_is_labelled_as_such(self):
        verdict = search.dominance(2_742, 3.208, self.FRONT, max_m=26.52)
        self.assertFalse(verdict["dominated"])
        self.assertTrue(verdict["dominatedOnMeanAndRate"])
        self.assertTrue(verdict["survivesOnMaximumOnly"])
        self.assertEqual(verdict["dominatedOnMeanBy"]["label"], "L4@1.0")

    def test_it_never_claims_to_be_cheaper_than_what_beats_it(self):
        verdict = search.dominance(2_742, 3.208, self.FRONT, max_m=26.52)
        self.assertNotIn("cheaperThanNearestBetter", verdict)
        self.assertNotIn("betterThanBestCheaper", verdict)

    def test_an_outright_survivor_still_states_both_neighbours(self):
        verdict = search.dominance(13_834, 1.186, self.FRONT, max_m=20.76)
        self.assertFalse(verdict["dominated"])
        self.assertFalse(verdict["dominatedOnMeanAndRate"])
        self.assertFalse(verdict["survivesOnMaximumOnly"])
        self.assertEqual(verdict["cheaperThanNearestBetter"]["label"], "L2@1.0")
        self.assertAlmostEqual(verdict["cheaperThanNearestBetter"]["fraction"], 0.495, places=2)

    def test_a_point_beaten_on_every_column_is_plainly_dominated(self):
        verdict = search.dominance(21_608, 1.368, self.FRONT + [(20_020, 1.059, 14.69, "L2@2.0")],
                                   max_m=19.20)
        self.assertTrue(verdict["dominated"])
        self.assertTrue(verdict["dominatedOnMeanAndRate"])
        self.assertFalse(verdict["survivesOnMaximumOnly"])


class OneFinalistPerFamilyKeepsTheCheapEnd(unittest.TestCase):
    """Pins a behaviour that is easy to miss.

    `finalists` builds [cheapest, most accurate] and then truncates, so a single
    slot is always the cheap end. On a measured codec front that confirms siren
    at 2,742 B / 3.208 m and never reaches 68,458 B / 0.526 m, the point that
    matters.
    """

    REPORT = {"byFamily": {"siren": {
        "paretoFront": [{"trial": 0, "deployedBytes": 2742, "scoreMaeM": 3.208},
                        {"trial": 5, "deployedBytes": 68458, "scoreMaeM": 0.526}],
        "trials": [{"trial": 0, "config": {"kind": "siren"}, "recipe": {}, "deployedBytes": 2742},
                   {"trial": 5, "config": {"kind": "siren"}, "recipe": {}, "deployedBytes": 68458}]}}}

    def test_one_slot_takes_the_cheap_end_and_drops_the_accurate_one(self):
        chosen = search.finalists(self.REPORT, per_family=1, limit=9)
        self.assertEqual(list(chosen), ["siren-t0"])
        self.assertEqual(chosen["siren-t0"]["fromStudy"]["deployedBytes"], 2742)

    def test_two_slots_keep_both_ends(self):
        chosen = search.finalists(self.REPORT, per_family=2, limit=9)
        self.assertEqual(sorted(chosen), ["siren-t0", "siren-t5"])
        self.assertEqual({e["fromStudy"]["deployedBytes"] for e in chosen.values()},
                         {2742, 68458})

    def test_a_limit_below_the_family_count_still_reaches_every_family(self):
        report = {"byFamily": {
            name: {"paretoFront": [{"trial": 0, "deployedBytes": 100 * i, "scoreMaeM": 1.0}],
                   "trials": [{"trial": 0, "config": {}, "recipe": {}, "deployedBytes": 100 * i}]}
            for i, name in enumerate(("a", "b", "c"), start=1)}}
        chosen = search.finalists(report, per_family=2, limit=3)
        self.assertEqual(len({e["fromStudy"]["family"] for e in chosen.values()}), 3)


class BothAxesReachTheCoarseEnd(unittest.TestCase):
    """The envelope must be swept where the network actually sits.

    Capping either axis at 1.0 m leaves the cheap end of the conventional front
    empty, so a neural point there looks undominated: on the coarse axis this
    produced a spurious fourier win, on the full-resolution axis a spurious
    hybrid win on maximum error.
    """

    def test_the_envelope_sweeps_past_the_declared_guarantees(self):
        self.assertGreaterEqual(max(search.ENVELOPE_TARGETS), 16.0)
        self.assertGreaterEqual(max(search.COARSE_TARGETS), 16.0)

    def test_the_declared_accuracy_targets_are_still_a_subset(self):
        """The declared accuracy targets must survive intact inside the wider sweep."""
        for target in search.CONVENTIONAL_TARGETS:
            self.assertIn(target, search.ENVELOPE_TARGETS)

    def test_ea03_targets_themselves_are_unchanged(self):
        self.assertEqual(search.CONVENTIONAL_TARGETS, (0.01, 0.05, 0.1, 0.5, 1.0))

    def test_the_coarse_guarantee_that_beat_the_hybrid_point_is_in_range(self):
        """q32 at an 8 m target (73,961 B) is the conventional point that beats the hybrid point."""
        self.assertIn(8.0, search.ENVELOPE_TARGETS)


class FittingAndExtrapolatingAreDifferentVerdicts(unittest.TestCase):
    """In a measured run every family fits its pages, and only the two carrying a
    conventional base stay bounded outside them. siren reaches 1.93 m on train and
    29.55 m on the extrapolation block, against a 26.61 m constant baseline;
    bandlimited reaches 393.74 m. Folding that into `trained` would report eight
    converged models as failed optimisations."""

    def test_a_model_that_fits_but_diverges_is_still_trained(self):
        verdicts = {s: search.convergence(m, b) for s, m, b in
                    (("train", 1.933, 22.554), ("selection", 6.582, 23.323),
                     ("test", 6.784, 21.423), ("extrapolation", 29.554, 26.613))}
        fitted = all(verdicts[s]["trained"] for s in ("train", "selection", "test"))
        self.assertTrue(fitted, "siren fits its pages")
        self.assertFalse(verdicts["extrapolation"]["trained"], "and diverges outside them")

    def test_the_based_families_clear_both(self):
        for mae in (4.004, 5.390):
            self.assertTrue(search.convergence(mae, 26.613)["trained"])

    def test_divergence_is_reported_as_negative_skill_not_clipped(self):
        verdict = search.convergence(393.740, 26.613)
        self.assertLess(verdict["skill"], -13.0)
        self.assertFalse(verdict["trained"])


class TheBytesCouldHaveBoughtABiggerGrid(unittest.TestCase):
    """The economic control.

    `networkEarnsItsBytes` is a strict inequality and passes on noise: hybrid-t12
    spends 29,160 bytes to improve its base by 0.0003 m and is marked a win. What
    a deployment compares against is the best grid the same total bytes could buy.
    """

    # (bytes, selection mae, label) for the base-only curve.
    BASES = [(2_729, 2.3881, "L4@1.0"), (4_762, 1.9655, "L3@4.0"),
             (73_951, 0.6464, "L2@0.05"), (81_974, 0.4872, "L1@1.0"),
             (111_219, 0.3383, "L1@0.5"), (248_295, 0.2603, "L1@0.05")]

    def test_residual_t8_would_have_been_better_off_as_a_grid(self):
        row = {"deployedBytes": 149_881, "summary": {"selection.mae_m": {"median": 1.0102}}}
        verdict = search.spent_better_on_the_base(row, {}, self.BASES)
        self.assertFalse(verdict["bytesWereWellSpent"])
        self.assertEqual(verdict["bestAffordableBase"]["label"], "L1@0.5")
        self.assertLess(verdict["bestAffordableBase"]["maeM"], 0.34)

    def test_hybrid_t5_would_have_been_better_off_as_a_grid(self):
        row = {"deployedBytes": 10_404, "summary": {"selection.mae_m": {"median": 2.8835}}}
        verdict = search.spent_better_on_the_base(row, {}, self.BASES)
        self.assertFalse(verdict["bytesWereWellSpent"])
        self.assertEqual(verdict["bestAffordableBase"]["label"], "L3@4.0")

    def test_hybrid_t12_survives_because_nothing_affordable_beats_it(self):
        row = {"deployedBytes": 277_455, "summary": {"selection.mae_m": {"median": 0.2601}}}
        self.assertTrue(search.spent_better_on_the_base(row, {}, self.BASES)["bytesWereWellSpent"])

    def test_a_base_that_is_cheaper_but_worse_does_not_count(self):
        row = {"deployedBytes": 5_000, "summary": {"selection.mae_m": {"median": 1.5}}}
        self.assertTrue(search.spent_better_on_the_base(row, {}, self.BASES)["bytesWereWellSpent"])


class DescribeStripsArraysByValueNotByName(unittest.TestCase):
    """A config carrying a bulk array must never reach a JSON report.

    Arrays are found by value, not by key name: a name list would miss the
    `context` family's geology raster, which would then fail only at
    `write_json`, after training had already run.
    """

    def test_an_array_under_an_unknown_key_is_stripped(self):
        described = search.describe({"kind": "context", "classes": np.zeros((4, 4)), "width": 8})
        self.assertNotIn("classes", described)
        self.assertEqual(described["classesShape"], [4, 4])
        self.assertEqual(described["width"], 8)

    def test_the_base_tensor_is_still_stripped(self):
        described = search.describe({"kind": "residual", "base": np.zeros((3, 3)),
                                     "inner": {"kind": "siren", "width": 4}})
        self.assertNotIn("base", described)
        self.assertEqual(described["inner"]["width"], 4)

    def test_ordinary_values_survive(self):
        described = search.describe({"kind": "grid", "levels": 8, "growth": 1.7,
                                     "hashed": True, "name": "x"})
        self.assertEqual(described, {"kind": "grid", "levels": 8, "growth": 1.7,
                                     "hashed": True, "name": "x"})

    def test_the_result_is_json_serialisable(self):
        import json
        json.dumps(search.describe({"kind": "context", "classes": np.zeros((2, 2)),
                                    "inner": {"base": np.zeros((2, 2)), "depth": 1}}))
