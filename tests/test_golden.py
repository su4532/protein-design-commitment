from __future__ import annotations

import importlib
import json
import tempfile
import unittest
from pathlib import Path

from analysis.golden import FIGURE_NAMES, TABLE_NAMES, collect
from analysis.manuscript_tables import build_tables, write_tables

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures"


class FrozenOutputTests(unittest.TestCase):
    def test_compact_results_match_frozen_golden(self) -> None:
        expected = json.loads((FIXTURES / "golden_results.json").read_text())
        self.assertEqual(collect(FIXTURES), expected)

    def test_all_manuscript_tables_regenerate(self) -> None:
        expected = json.loads((FIXTURES / "manuscript_tables.json").read_text())
        observed = build_tables(FIXTURES)
        self.assertEqual(observed, expected)
        self.assertEqual(sorted(observed), sorted(TABLE_NAMES))
        with tempfile.TemporaryDirectory() as tmp:
            write_tables(observed, Path(tmp))
            self.assertEqual(
                sorted(path.stem for path in Path(tmp).glob("*.csv")),
                sorted(TABLE_NAMES),
            )

    def test_expected_manuscript_figure_set(self) -> None:
        self.assertEqual(len(FIGURE_NAMES), 9)
        self.assertEqual(len(set(FIGURE_NAMES)), 9)

    def test_cpu_analysis_modules_import(self) -> None:
        modules = [
            "analysis.commitment.boot",
            "analysis.commitment.grid",
            "analysis.commitment.metrics",
            "analysis.commitment.nested",
            "analysis.commitment.probes",
            "analysis.commitment.variance",
            "analysis.action.compare_grids",
            "analysis.action.features",
            "analysis.action.objective",
            "analysis.action.pool_gain",
            "analysis.action.ridge",
            "analysis.action.score_curves",
            "analysis.scoring.compare",
            "analysis.scoring.correct_target",
            "analysis.scoring.nested_split",
            "analysis.scoring.pool_quant",
            "analysis.scoring.quantisation",
            "analysis.scoring.readability_corrected",
            "analysis.scoring.recompute_objective",
            "analysis.scoring.ruler",
            "analysis.cell_analysis",
            "analysis.clock_remap",
            "analysis.designability",
            "analysis.fold_clustering",
            "analysis.golden",
            "analysis.grid_comparison",
            "analysis.joint_bootstrap",
            "analysis.manuscript_figures",
            "analysis.manuscript_tables",
            "analysis.mlp_retune",
            "analysis.opportunity",
            "analysis.order_simulations",
            "analysis.pooling",
            "analysis.spread",
            "analysis.spread_raw",
            "analysis.transfer",
        ]
        for name in modules:
            with self.subTest(module=name):
                importlib.import_module(name)


if __name__ == "__main__":
    unittest.main()
