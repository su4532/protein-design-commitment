"""Compact deterministic readback of the frozen source-analysis outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FIGURE_NAMES = [
    "fig1_schedule", "fig2_readout", "figA1_commitment", "figA2_consequence",
    "figA3_grid", "figA4_cv", "fig2_readability", "fig3_robustness", "fig4_schedule",
]
TABLE_NAMES = ["clocks", "order-null", "gap-scale", "cv", "pairwise", "oa", "thresholds", "boot"]


def collect(fixtures: Path) -> dict[str, Any]:
    fig = json.loads((fixtures / "figure_data.json").read_text())
    action = json.loads((fixtures / "action_grid_comparison.json").read_text())
    adaptive = json.loads((fixtures / "adaptive_opportunity.json").read_text())
    joint = json.loads((fixtures / "joint_opportunity.json").read_text())
    order = json.loads((fixtures / "order_power.json").read_text())
    tables = json.loads((fixtures / "manuscript_tables.json").read_text())
    interface = action["by_quality"]["interface_quality"]["ten"]["adaptive_vs_deployed"]
    primary = adaptive["primary"]
    noise_cells = fig["noise"]["estimator_cells"]
    j = joint["measures"]["interface_quality"]
    return {
        "capture-core": {
            "cells": fig["instrument_control_parent"]["n_cells"],
            "root_share_min": fig["instrument_control_parent"]["root_share_min"],
            "published_capture_steps": len(fig["design"]["published_capture_steps"]),
            "published_fold_readability_peak": fig["fig2_readability_published"]["fold_identity"]["families"]["linear_activation"]["peak_mean"],
        },
        "action-seven": {
            "rotations": fig["consequence"]["n_rotations"],
            "winner_curse_difference": fig["consequence"]["winners_curse"]["difference"]["mean"],
            "same_group_gain": fig["consequence"]["winners_curse"]["same_group"]["mean"],
        },
        "scoring-sensitivity": {
            "estimator_cells": len(noise_cells),
            "cells_excluding_zero": sum(bool(row["excludes_zero"]) for row in noise_cells),
            "continuous_minus_categorical_cv": fig["quantisation"]["cv_difference"]["point"],
        },
        "capture-grid": {
            "cells": fig["design"]["n_cells"],
            "kendall_w": fig["ordering"]["kendall_w"],
            "distinct_orders": fig["ordering"]["n_distinct_orders"],
            "cluster_cv_gap": fig["spread_cluster_unit"]["extreme_resolved"]["difference"],
        },
        "paper-assembly": {
            "figure_count": len(FIGURE_NAMES),
            "table_count": len(TABLE_NAMES),
            "table_rows": sum(len(tables[name]) for name in TABLE_NAMES),
            "order_null_observed_w": order["observed"]["kendall_w"],
        },
        "action-ten": {
            "gain_mean": interface["mean"], "gain_lo": interface["lo"], "gain_hi": interface["hi"],
            "criterion_met": action["decision"]["criterion_met"],
        },
        "adaptive-selection": {
            "cluster_O": primary["interface_quality"]["O"]["mean"],
            "cluster_A": primary["interface_quality"]["A"]["mean"],
            "nested_O": primary["nested_split_design_sensitivity"]["O"]["mean"],
            "nested_A": primary["nested_split_design_sensitivity"]["A"]["mean"],
            "assessment": primary["act_iv_change"],
            "joint_O": j["O"]["joint_cluster_rotation_design"]["gain_adaptive_vs_deployed"]["mean"],
            "joint_A": j["A"]["joint_cluster_rotation_design"]["gain_adaptive_vs_deployed"]["mean"],
        },
    }
