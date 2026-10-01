"""Is the process teacher valid, before anything learns from it?

Imitating the teacher is not sufficient on its own: an emulator that reproduces
a flawed discretisation accurately is still a flawed physical model. So the
teacher is checked against cases whose answer is known independently of the
code, not against itself.

Each term is isolated first, because a three-term model that is right overall
can be right by cancellation. Then the scheme's own properties (conservation,
refinement, stability) are checked, since those are what a later emulator
would silently inherit.
"""
from __future__ import annotations
import functools
import math
import os
import unittest

import numpy as np

from geoneural.physics import units
from geoneural.physics.landscape import (Parameters, drainage_area, evolve, laplacian,
                                         slope_area, steady_state_report,
                                         stable_timestep, steepest_slope)

SLOW = unittest.skipUnless(os.environ.get("GEONEURAL_SLOW"), "long solver run; set GEONEURAL_SLOW=1")


class Diffusion(unittest.TestCase):
    """Hillslope term alone, against the analytic decay of a sinusoid."""

    def field(self, side: int, wavelength_cells: float) -> np.ndarray:
        cols = np.arange(side)
        return np.tile(np.sin(2 * np.pi * cols / wavelength_cells), (side, 1))

    def only_diffusion(self, spacing: float, diffusivity: float = 1e-2) -> Parameters:
        return Parameters(uplift_m_per_year=0.0, k_incision=0.0,
                          diffusivity_m2_per_year=diffusivity, spacing_m=spacing)

    def test_a_sinusoid_decays_towards_the_analytic_rate_as_the_grid_refines(self):
        """A(t) = A0 exp(-D k^2 t) in the continuum. The five-point Laplacian has
        a slightly different discrete eigenvalue, so the test is refinement: the
        discrepancy must shrink as the wavelength is better resolved."""
        errors = []
        for cells_per_wave in (8, 16, 32):
            spacing = 400.0 / cells_per_wave  # hold the physical wavelength fixed
            parameters = self.only_diffusion(spacing)
            height = self.field(cells_per_wave * 2, cells_per_wave)
            years = 2000.0
            evolved, _ = evolve(height, parameters, years, dt_years=stable_timestep(parameters) / 4)
            wavenumber = 2 * math.pi / 400.0
            expected = math.exp(-parameters.diffusivity_m2_per_year * wavenumber ** 2 * years)
            middle = evolved.shape[0] // 2
            observed = float(np.abs(evolved[middle]).max())
            errors.append(abs(observed - expected))
        self.assertLess(errors[-1], errors[0],
                        "refining the grid must bring the decay closer to the analytic rate")
        self.assertLess(errors[-1], 0.02)

    def test_diffusion_conserves_total_height_with_reflective_edges(self):
        """Zero-flux boundaries move nothing across the edge, so the sum is
        invariant. A conservation error here would appear in any emulator trained
        on this teacher as an invented source or sink."""
        parameters = self.only_diffusion(100.0)
        rng = np.random.default_rng(11)
        height = rng.normal(50.0, 5.0, (32, 32))
        evolved, _ = evolve(height, parameters, 50_000.0,
                            dt_years=stable_timestep(parameters) / 2)
        self.assertAlmostEqual(float(evolved.sum()), float(height.sum()), places=6)

    def test_diffusion_never_creates_a_new_extremum(self):
        """A maximum principle: heat flows downhill, so the range cannot widen."""
        parameters = self.only_diffusion(100.0)
        rng = np.random.default_rng(5)
        height = rng.normal(0.0, 3.0, (24, 24))
        evolved, _ = evolve(height, parameters, 20_000.0,
                            dt_years=stable_timestep(parameters) / 2)
        self.assertLessEqual(evolved.max(), height.max() + 1e-9)
        self.assertGreaterEqual(evolved.min(), height.min() - 1e-9)

    def test_the_laplacian_of_a_plane_is_zero_inside_and_not_on_the_edge(self):
        """Zero in the interior, but not on the edge.

        Edge-replicated padding sets the ghost cell equal to the edge cell rather
        than continuing the plane, so a tilted plane has non-zero curvature at the
        boundary. That is not a bug: it is what zero-flux means, and it is the
        same property that makes the conservation test above exact. A plane is
        therefore not a steady state of the diffusion term near the edge, which
        anything learning from this teacher will inherit and should be told."""
        rows, cols = np.mgrid[0:16, 0:16]
        plane = 3.0 * cols - 2.0 * rows + 7.0
        curvature = laplacian(plane, 10.0)
        self.assertLess(float(np.abs(curvature[1:-1, 1:-1]).max()), 1e-9)
        self.assertGreater(float(np.abs(curvature).max()), 1e-3)


class Uplift(unittest.TestCase):
    def test_uplift_alone_is_exact(self):
        """No approximation is involved, so this must be exact to rounding. It
        catches a timestep applied twice or not at all."""
        parameters = Parameters(uplift_m_per_year=1e-3, k_incision=0.0,
                                diffusivity_m2_per_year=0.0, spacing_m=100.0)
        height = np.zeros((16, 16))
        evolved, record = evolve(height, parameters, 10_000.0, dt_years=100.0)
        self.assertTrue(np.allclose(evolved, 10.0, atol=1e-9))
        self.assertEqual(record["steps"], 100)

    def test_a_spatially_varying_uplift_field_is_applied_per_cell(self):
        parameters = Parameters(uplift_m_per_year=0.0, k_incision=0.0,
                                diffusivity_m2_per_year=0.0, spacing_m=100.0)
        field = np.linspace(0.0, 1e-3, 16)[:, None] * np.ones((16, 16))
        evolved, _ = evolve(np.zeros((16, 16)), parameters, 1000.0, dt_years=100.0,
                            uplift=field)
        self.assertTrue(np.allclose(evolved, field * 1000.0, atol=1e-9))


class Incision(unittest.TestCase):
    def test_a_flat_surface_incises_only_at_the_flat_resolution_epsilon(self):
        """Not exactly zero: about 3e-07 m.

        `hydrology.fill_depressions` raises each step across a filled flat by
        FILL_EPSILON_M = 1e-6 so that flow stays defined, and that artificial
        gradient drives an artificial incision of order
        `K * sqrt(A) * (epsilon / spacing) * t`. It is negligible here and it is
        systematic, growing with drainage area, so it is bounded and documented
        rather than rounded away; an emulator trained on this teacher would
        otherwise learn a small unexplained erosion of flats as if it were
        physics."""
        parameters = Parameters(uplift_m_per_year=0.0, diffusivity_m2_per_year=0.0,
                                k_incision=1e-4, spacing_m=100.0)
        flat = np.full((16, 16), 10.0)
        evolved, _ = evolve(flat, parameters, 1000.0, dt_years=100.0)
        drift = float(np.abs(evolved - flat).max())
        self.assertGreater(drift, 0.0)
        self.assertLess(drift, 1e-5)
        self.assertLessEqual(float((evolved - flat).max()), 0.0)

    def test_the_flat_epsilon_artefact_scales_with_the_incision_constant(self):
        """The control confirming the drift above is the epsilon and not noise:
        it is proportional to K, so ten times the constant gives ten times the
        drift."""
        flat = np.full((16, 16), 10.0)
        drifts = []
        for k in (1e-4, 1e-3):
            parameters = Parameters(uplift_m_per_year=0.0, diffusivity_m2_per_year=0.0,
                                    k_incision=k, spacing_m=100.0)
            evolved, _ = evolve(flat, parameters, 1000.0, dt_years=100.0)
            drifts.append(float(np.abs(evolved - flat).max()))
        self.assertAlmostEqual(drifts[1] / drifts[0], 10.0, delta=0.5)

    def test_incision_lowers_and_never_raises(self):
        parameters = Parameters(uplift_m_per_year=0.0, diffusivity_m2_per_year=0.0,
                                k_incision=1e-5, spacing_m=100.0)
        rows = np.linspace(100.0, 0.0, 24)
        height = np.tile(rows[:, None], (1, 24))
        evolved, _ = evolve(height, parameters, 5000.0, dt_years=500.0)
        self.assertLessEqual(float((evolved - height).max()), 1e-9)
        self.assertLess(float(evolved.sum()), float(height.sum()))

    def test_drainage_area_totals_the_domain_at_the_outlet(self):
        """Every cell drains somewhere, so the accumulated area leaving the
        domain must be the domain's own area. This is the water budget of the
        routing the incision term depends on."""
        rows = np.linspace(50.0, 0.0, 20)
        height = np.tile(rows[:, None], (1, 20)) + np.linspace(0, 0.1, 20)
        area = drainage_area(height, 100.0)
        self.assertAlmostEqual(float(area.max()) / (100.0 ** 2), 20 * 20, delta=20 * 20 * 0.5)

    def test_steepest_slope_matches_a_known_ramp(self):
        rows = np.linspace(0.0, 190.0, 20)[::-1]
        height = np.tile(rows[:, None], (1, 20))
        slope = steepest_slope(height, 100.0)
        interior = slope[1:-1, 1:-1]
        self.assertAlmostEqual(float(np.median(interior)), 10.0 / 100.0, places=6)


class Scheme(unittest.TestCase):
    def test_an_unstable_timestep_is_refused_rather_than_reduced(self):
        """A silently reduced step would make a run mean something other than it
        says. A smooth, plausible, wrong field is the failure being prevented."""
        parameters = Parameters(diffusivity_m2_per_year=1.0, spacing_m=10.0)
        with self.assertRaises(ValueError):
            evolve(np.zeros((8, 8)), parameters, 100.0,
                   dt_years=stable_timestep(parameters) * 4)

    def test_the_stability_limit_is_the_two_dimensional_diffusion_condition(self):
        parameters = Parameters(diffusivity_m2_per_year=0.5, spacing_m=20.0)
        self.assertAlmostEqual(stable_timestep(parameters, safety=1.0),
                               20.0 ** 2 / (4.0 * 0.5))

    def test_halving_the_timestep_halves_the_error_of_an_explicit_scheme(self):
        """First-order accuracy in time. If refinement did not converge at the
        expected rate the integrator would be wrong in a way no single run
        reveals."""
        parameters = Parameters(uplift_m_per_year=1e-4, k_incision=0.0,
                                diffusivity_m2_per_year=1e-2, spacing_m=100.0)
        rng = np.random.default_rng(2)
        height = rng.normal(20.0, 2.0, (24, 24))
        years = 40_000.0
        reference, _ = evolve(height, parameters, years, dt_years=years / 6400)
        coarse, _ = evolve(height, parameters, years, dt_years=years / 100)
        fine, _ = evolve(height, parameters, years, dt_years=years / 200)
        coarse_error = float(np.abs(coarse - reference).mean())
        fine_error = float(np.abs(fine - reference).mean())
        self.assertLess(fine_error, coarse_error)
        self.assertGreater(coarse_error / max(fine_error, 1e-18), 1.5)

    def test_a_run_records_what_it_leaves_out(self):
        parameters = Parameters()
        _, record = evolve(np.zeros((8, 8)), parameters, 100.0, dt_years=50.0)
        for omitted in ("sediment transport and deposition", "human alteration"):
            self.assertIn(omitted, record["omits"])
        self.assertIn("not discharge", record["qualification"])

    def test_the_incision_limiter_reports_when_it_engages(self):
        """The limiter prevents the explicit scheme inverting the gradient that
        drives it. That is an intervention, so a run says how often it acted
        rather than hiding it."""
        parameters = Parameters(uplift_m_per_year=0.0, diffusivity_m2_per_year=0.0,
                                k_incision=5e-2, spacing_m=100.0)
        rows = np.linspace(500.0, 0.0, 24)
        height = np.tile(rows[:, None], (1, 24))
        _, record = evolve(height, parameters, 20_000.0, dt_years=2000.0)
        self.assertGreater(record["cellsIncisionLimitedTotal"], 0)


if __name__ == "__main__":
    unittest.main()


class ExactTimeAndReceiverLimiter(unittest.TestCase):
    """The two counterexamples that version 1 of the teacher failed."""

    def test_a_non_divisible_duration_is_integrated_exactly(self):
        """1000 years at dt 400: two full steps and one 200-year step, not round(2.5) = 2."""
        parameters = Parameters(uplift_m_per_year=1.0, k_incision=0.0,
                                diffusivity_m2_per_year=0.0)
        for dt in (400.0, 300.0, 600.0):
            surface, record = evolve(np.zeros((8, 8)), parameters, 1000.0, dt_years=dt)
            self.assertAlmostEqual(float(surface.mean()), 1000.0, places=9)
            self.assertAlmostEqual(record["realisedYears"], 1000.0, places=9)
            self.assertEqual(record["partialSteps"], 1)

    def test_a_peak_is_cut_no_lower_than_its_receiver(self):
        """Extreme incision on a 3 x 3 peak, receiver cardinal and then diagonal."""
        parameters = Parameters(uplift_m_per_year=0.0, k_incision=1e3,
                                diffusivity_m2_per_year=0.0)
        cardinal = np.zeros((3, 3))
        cardinal[1, 1] = 1.0
        diagonal = np.full((3, 3), 0.9)
        diagonal[1, 1], diagonal[0, 0] = 1.0, 0.0
        for start in (cardinal, diagonal):
            surface, record = evolve(start, parameters, 1.0, dt_years=1.0)
            self.assertEqual(float(surface[1, 1]), 0.0)
            self.assertGreater(record["cellsIncisionLimitedTotal"], 0)


class NonDimensionalGroups(unittest.TestCase):
    """`units` holds the only definition of the groups; these pin its algebra."""

    def test_the_groups_match_the_algebra(self):
        scales = units.Scales(6400.0, 200.0, 2e6)
        groups = units.groups(1e-4, 1e-5, 1e-2, scales, 0.5, 1.0)
        self.assertAlmostEqual(groups["logPiU"], math.log10(1e-4 * 2e6 / 200.0), places=12)
        # m = 1/2, n = 1: L^(2m-n) = 1 and H^(n-1) = 1, so Pi_K = K T.
        self.assertAlmostEqual(groups["logPiK"], math.log10(1e-5 * 2e6), places=12)
        self.assertAlmostEqual(groups["logPiD"], math.log10(1e-2 * 2e6 / 6400.0 ** 2), places=12)
        uplift = units.uplift_groups(1e-4, 1e-5, 1e-2, 6400.0, 200.0)
        self.assertAlmostEqual(uplift["logFluvialNumber"], math.log10(1e-5 * 200.0 / 1e-4), places=12)
        self.assertAlmostEqual(uplift["logHillslopeNumber"],
                               math.log10(1e-2 * 200.0 / (1e-4 * 6400.0 ** 2)), places=12)

    def test_rescaling_length_relief_and_time_leaves_the_groups_unchanged(self):
        """(L, H, T) -> (a L, b H, c T) with the rates of `units.rescale`, for n != 1 too."""
        for m, n in ((0.5, 1.0), (0.4, 1.3)):
            base = units.groups(2e-4, 3e-6, 5e-3, units.Scales(3000.0, 150.0, 1e6), m, n)
            a, b, c = 2.5, 0.3, 7.0
            rates = units.rescale(2e-4, 3e-6, 5e-3, a, b, c, m, n)
            moved = units.groups(rates["uplift"], rates["kIncision"], rates["diffusivity"],
                                 units.Scales(3000.0 * a, 150.0 * b, 1e6 * c), m, n)
            for key in units.GROUPS:
                self.assertAlmostEqual(base[key], moved[key], places=12, msg=(m, n, key))

    def test_the_solver_obeys_the_same_rescaling(self):
        """Spacing a dx, heights b z, the rescaled rates and duration c t: the run is the
        original drawn at another scale. Only the 1e-6 m fill epsilon does not scale."""
        rows = np.linspace(0.0, 1.0, 32)
        start = 40.0 * np.sin(2.5 * rows)[:, None] + 25.0 * np.cos(1.7 * rows)[None, :] + 100.0
        base = Parameters(uplift_m_per_year=1e-4, k_incision=1e-5,
                          diffusivity_m2_per_year=1e-2, spacing_m=100.0)
        a, b, c = 2.0, 3.0, 0.5
        rates = units.rescale(1e-4, 1e-5, 1e-2, a, b, c)
        moved = Parameters(uplift_m_per_year=rates["uplift"], k_incision=rates["kIncision"],
                           diffusivity_m2_per_year=rates["diffusivity"], spacing_m=100.0 * a)
        one, _ = evolve(start, base, 20_000.0, dt_years=100.0, base_level="fixed-edges")
        two, _ = evolve(b * start, moved, 20_000.0 * c, dt_years=100.0 * c, base_level="fixed-edges")
        self.assertLess(float(np.abs(two / b - one).max()), 1e-5)
        self.assertGreater(float(np.abs(one - start).max()), 0.1)

    def test_degenerate_scales_are_refused(self):
        for bad in ((0.0, 200.0, 1.0), (6400.0, 0.0, 1.0), (-1.0, 200.0, 1.0)):
            with self.assertRaises(ValueError):
                units.Scales(*bad)
        with self.assertRaises(ValueError):
            units.uplift_scales(6400.0, 200.0, 0.0)


class SimilarityHoldsInTheSolver(unittest.TestCase):
    """The strongest check available on the teacher, and it is nearly free.

    Scaling U, K and D by the same factor multiplies the whole right-hand side by
    that factor, so the surface at time `t/lambda` must equal the original at `t`.
    A solver that fails this is not integrating the equation it documents.
    Because the check compares a run against another run rather than against an
    analytic answer, it catches errors in the incision limiter, the depression
    filling and the edge handling that a single-term test would not.
    """

    def landscape(self, side: int = 48, spacing: float = 100.0):
        rows = np.linspace(0.0, 1.0, side)
        field = 40.0 * np.sin(2.5 * rows)[:, None] + 25.0 * np.cos(1.7 * rows)[None, :]
        return field + 100.0

    def test_scaling_every_rate_only_rescales_time(self):
        start = self.landscape()
        slow = Parameters(uplift_m_per_year=1e-4, k_incision=1e-5, area_exponent=0.5,
                          slope_exponent=1.0, diffusivity_m2_per_year=1e-2, spacing_m=100.0)
        fast = Parameters(uplift_m_per_year=2e-4, k_incision=2e-5, area_exponent=0.5,
                          slope_exponent=1.0, diffusivity_m2_per_year=2e-2, spacing_m=100.0)
        years, dt = 20_000.0, 100.0
        a, _ = evolve(start, slow, years, dt_years=dt)
        b, _ = evolve(start, fast, years / 2.0, dt_years=dt / 2.0)
        difference = float(np.abs(a - b).max())
        relief = float(a.max() - a.min())
        self.assertLess(difference, 1e-6 * max(relief, 1.0),
                        f"similarity violated by {difference} m over {relief} m of relief")

    def test_the_control_shows_the_check_can_fail(self):
        """Scaling only uplift must NOT be a pure time rescaling."""
        start = self.landscape()
        slow = Parameters(uplift_m_per_year=1e-4, k_incision=1e-5, diffusivity_m2_per_year=1e-2)
        wrong = Parameters(uplift_m_per_year=2e-4, k_incision=1e-5, diffusivity_m2_per_year=1e-2)
        a, _ = evolve(start, slow, 20_000.0, dt_years=100.0)
        b, _ = evolve(start, wrong, 10_000.0, dt_years=50.0)
        self.assertGreater(float(np.abs(a - b).max()), 1.0)


@functools.lru_cache(maxsize=None)
def _evolved(diffusivity: float, side: int = 64):
    """One 600 000-year integration per diffusivity, shared between the tests that
    assert on the same run so it is computed only once. The surface is returned
    read-only, so no test can alter what another sees."""
    rows = np.linspace(0.0, 1.0, side)
    start = 100.0 + 30.0 * np.sin(3.0 * rows)[:, None] + 20.0 * np.cos(2.0 * rows)[None, :]
    parameters = Parameters(uplift_m_per_year=5e-4, k_incision=4e-5, area_exponent=0.5,
                            slope_exponent=1.0, diffusivity_m2_per_year=diffusivity,
                            spacing_m=100.0)
    surface, record = evolve(start, parameters, 600_000.0, dt_years=200.0,
                             base_level="fixed-edges")
    surface.flags.writeable = False
    return surface, parameters, record


class SlopeAreaLaw(unittest.TestCase):
    """`S = (U/K)^(1/n) A^(-m/n)` is a prediction with no free parameter, and the
    only thing that separates this solver from plausible-looking terrain."""

    def evolved(self, diffusivity: float, side: int = 64):
        return _evolved(diffusivity, side)

    def test_a_flat_field_yields_no_usable_fit_rather_than_a_number(self):
        result = slope_area(np.zeros((32, 32)), 100.0)
        self.assertFalse(result["sufficient"])
        self.assertNotIn("concavity", result)

    def test_the_steady_state_matches_the_closed_form_in_both_exponent_and_coefficient(self):
        surface, parameters, _ = self.evolved(1e-4)
        report = steady_state_report(surface, parameters)
        self.assertTrue(report["sufficient"])
        self.assertGreaterEqual(report["binsCompared"], 4)
        self.assertTrue(report["agrees"],
                        f"worst bin off by {report['worstLogRatio']:.3f} in log; "
                        f"ratios {report['binRatios']}")
        for ratio in report["binRatios"]:
            self.assertLess(abs(math.log(ratio)), math.log(1.10))

    @SLOW
    def test_it_holds_across_an_order_of_magnitude_of_diffusivity(self):
        for diffusivity in (5e-3, 1e-4):
            surface, parameters, _ = self.evolved(diffusivity)
            self.assertTrue(steady_state_report(surface, parameters)["agrees"],
                            f"failed at D={diffusivity}")

    @SLOW
    def test_without_a_base_level_relief_collapses_and_the_law_does_not_apply(self):
        """The control, and the reason `base_level` exists. A closed domain rises
        as a plane: relief decays to nothing and there is no steady state to test."""
        rows = np.linspace(0.0, 1.0, 64)
        start = 100.0 + 30.0 * np.sin(3.0 * rows)[:, None] + 20.0 * np.cos(2.0 * rows)[None, :]
        parameters = Parameters(uplift_m_per_year=5e-4, k_incision=4e-5, area_exponent=0.5,
                                slope_exponent=1.0, diffusivity_m2_per_year=5e-3, spacing_m=100.0)
        closed, closed_record = evolve(start, parameters, 1_200_000.0, dt_years=200.0)
        open_, open_record = evolve(start, parameters, 1_200_000.0, dt_years=200.0,
                                    base_level="fixed-edges")
        self.assertLess(closed_record["reliefM"], 1.0, "a closed domain flattens")
        self.assertGreater(open_record["reliefM"], 50.0, "a base level sustains relief")
        self.assertFalse(steady_state_report(closed, parameters)["agrees"])

    def test_an_unsupported_base_level_is_refused(self):
        with self.assertRaises(ValueError):
            evolve(np.zeros((16, 16)), Parameters(), 100.0, dt_years=10.0, base_level="sea")
