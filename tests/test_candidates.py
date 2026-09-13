"""Joint dominance and the conventional comparator: the logic behind every comparison claim."""
from __future__ import annotations

import unittest

from geoneural.export.candidates import _row, best_conventional, dominated_by


def row(cid, family, size, mae, peak, package="finest-per-page"):
    return _row(id=cid, family=family, package=package, bytes=size, maeM=mae, maxM=peak)


class JointDominance(unittest.TestCase):
    def test_two_rows_that_each_win_one_axis_do_not_dominate_together(self):
        candidate = row("n", "neural", 100, 1.0, 10.0)
        cheap_mean = row("a", "conventional", 90, 0.9, 12.0)
        cheap_max = row("b", "conventional", 90, 1.1, 8.0)
        self.assertIsNone(dominated_by(candidate, [candidate, cheap_mean, cheap_max]))

    def test_one_row_better_everywhere_dominates(self):
        candidate = row("n", "neural", 100, 1.0, 10.0)
        better = row("c", "conventional", 100, 0.9, 10.0)
        self.assertEqual(dominated_by(candidate, [candidate, better]), "c")

    def test_an_exact_tie_is_not_dominance(self):
        candidate = row("n", "neural", 100, 1.0, 10.0)
        twin = row("t", "conventional", 100, 1.0, 10.0)
        self.assertIsNone(dominated_by(candidate, [candidate, twin]))

    def test_a_missing_value_is_not_zero(self):
        candidate = row("n", "neural", 100, 1.0, 10.0)
        unmeasured = row("u", "conventional", 50, None, 1.0)
        self.assertIsNone(dominated_by(candidate, [candidate, unmeasured]))
        self.assertIsNone(dominated_by(unmeasured, [candidate, unmeasured]))

    def test_package_conventions_are_never_compared(self):
        candidate = row("n", "neural", 100, 1.0, 10.0)
        other = row("p", "conventional", 10, 0.1, 1.0, package="pyramid")
        self.assertIsNone(dominated_by(candidate, [candidate, other]))

    def test_an_empty_comparison_set(self):
        candidate = row("n", "neural", 100, 1.0, 10.0)
        self.assertIsNone(dominated_by(candidate, [candidate]))
        self.assertIsNone(best_conventional(candidate, [candidate]))


class BestConventional(unittest.TestCase):
    def test_picks_lowest_mean_error_at_no_more_bytes_and_signs_the_margins(self):
        candidate = row("n", "neural", 100, 1.0, 10.0)
        rows = [candidate, row("a", "conventional", 100, 1.2, 4.0), row("b", "conventional", 80, 0.8, 12.0),
                row("c", "conventional", 120, 0.1, 0.1)]
        best = best_conventional(candidate, rows)
        self.assertEqual(best["id"], "b")
        self.assertAlmostEqual(best["maeGainPercent"], -25.0)
        self.assertAlmostEqual(best["maxDifferenceM"], -2.0)


if __name__ == "__main__":
    unittest.main()
