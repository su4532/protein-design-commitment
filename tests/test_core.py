from __future__ import annotations

import unittest

import numpy as np

from analysis.action.objective import center_by_root, objective
from analysis.clock_remap import closed_form
from analysis.commitment.variance import decompose
from analysis.opportunity import available_opportunity
from analysis.pooling import t_half


class CoreEstimatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = np.array([0, 0, 1, 1])
        self.child = np.array([0, 1, 0, 1])
        self.y = np.array([
            [-14.0, -12.0],
            [-8.0, -6.0],
            [6.0, 8.0],
            [12.0, 14.0],
        ])

    def test_nested_anova_known_components(self) -> None:
        result = decompose(self.y, self.root, self.child)
        self.assertAlmostEqual(result.sigma2_root, 191.0)
        self.assertAlmostEqual(result.sigma2_child, 17.0)
        self.assertAlmostEqual(result.sigma2_within, 2.0)
        self.assertAlmostEqual(result.commitment, 208.0 / 210.0)
        self.assertEqual((result.n_roots, result.n_children, result.k), (2, 2, 2))

    def test_available_opportunity_matches_component_ratio(self) -> None:
        cube = self.y.reshape(2, 2, 2)
        self.assertAlmostEqual(float(available_opportunity(cube)), 17.0 / 210.0)

    def test_parent_centering_is_separate_by_root(self) -> None:
        values = np.array([1.0, 3.0, 10.0, 14.0])
        centered = center_by_root(values, self.root)
        np.testing.assert_allclose(centered, [-1.0, 1.0, -2.0, 2.0])
        for group in np.unique(self.root):
            self.assertAlmostEqual(float(centered[self.root == group].mean()), 0.0)

    def test_action_objective_obeys_branch_share_cap(self) -> None:
        result = objective(1.0, sigma2_root=8.0, sigma2_child=1.0, sigma2_within=1.0, k=2)
        self.assertTrue(result["capped"])
        self.assertAlmostEqual(result["J"], result["child_share"])
        self.assertLessEqual(result["J"], 0.1)

    def test_half_commitment_crossing_interpolates(self) -> None:
        self.assertAlmostEqual(t_half([10, 50, 100], [0.2, 0.4, 0.8]), 62.5)
        self.assertIsNone(t_half([10, 50], [0.1, 0.4]))

    def test_clock_remaps_are_monotone_and_normalized(self) -> None:
        for mode in ("log", "power"):
            schedule = closed_form(mode, 2.0, 400)
            self.assertEqual(len(schedule), 401)
            self.assertAlmostEqual(float(schedule[0]), 0.0)
            self.assertAlmostEqual(float(schedule[-1]), 1.0)
            self.assertTrue(np.all(np.diff(schedule) > 0))


if __name__ == "__main__":
    unittest.main()
