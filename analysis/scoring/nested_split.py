#!/usr/bin/env python
"""Where in the nesting does the ruler's error sit?

AlphaFold's error is a fixed function of the complex it reads, not a draw per
measurement. Two completions of the same child at a late capture step are nearly the
same complex, so their errors are nearly the same number: the error is shared at the
child level rather than independent within it. Only the part that is genuinely
per-completion belongs in the within-child term of the variance decomposition.

This module measures that split. The nested rater sample scores 4 roots x 2 children
x 2 completions per cell under all five ensemble members, so the recorded member's
deviation from consensus,

    e_i = v_i0 - mean_m v_im

can be put through the same three-level ANOVA as the outcome. ``e`` is a shrunk copy
of the true deviation ``gamma_i0`` -- both terms of ``e`` are functions of structure
``i``, so the shrinkage is a single factor ``(M-1)/M`` on every component and the
fractions are unaffected.

Fractions are pooled across targets at each capture step, because 16 structures in
one cell is thin for a three-level split while 19 cells at one step is not. The
expectation, worth stating before looking: the within-child fraction should fall as
the capture step rises, because completions of a child become more alike.

    nested_split.py <raters_nested_dir> <out_json>
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np


from ..commitment.variance import decompose  # noqa: E402

QUALITY_KEYS = {
    "interface_quality": "af2folding_i_pae",
    "structural_confidence": "af2folding_plddt_log",
    "iptm": "af2folding_i_ptm_log",
}
STEPS = (10, 25, 50, 75, 100, 200, 300)
TAG_RE = re.compile(r"root(\d+)_child(\d+)_comp(\d+)")


def load(directory: str) -> dict:
    """(target, step) -> {quality: (e[N,k], root[N], child[N])}."""
    cells: dict[tuple[str, int], dict] = {}
    files = sorted(f for f in glob.glob(os.path.join(directory, "*.json"))
                   if not os.path.basename(f).startswith("spec_"))
    for f in files:
        with open(f) as fh:
            payload = json.load(fh)
        names = list(payload["spec"]["passes"])
        rows: dict[tuple[str, int], list] = {}
        for row in payload["results"]:
            m = TAG_RE.fullmatch(row["tag"])
            if not m:
                continue
            r, c, j = (int(g) for g in m.groups())
            vals = {}
            ok = True
            for q, key in QUALITY_KEYS.items():
                v = np.array([float(row["stored"][key])]
                             + [float(row["passes"][p][key]) for p in names], float)
                if not np.isfinite(v).all():
                    ok = False
                vals[q] = v
            if ok:
                rows.setdefault((row["target"], int(row["step"])), []).append((r, c, j, vals))
        for key, items in rows.items():
            cells[key] = items
    print(f"loaded {len(files)} files, {len(cells)} cells")
    return cells


def cell_components(items: list, quality: str) -> dict | None:
    """Three-level components of the ruler deviation for one cell."""
    M = len(items[0][3][quality])
    scale = M / (M - 1)
    by_state: dict[tuple[int, int], dict[int, float]] = {}
    for r, c, j, vals in items:
        v = vals[quality]
        e = float(v[0] - v.mean())
        by_state.setdefault((r, c), {})[j] = e
    comps = sorted({j for d in by_state.values() for j in d})
    states = [rc for rc, d in by_state.items() if all(j in d for j in comps)]
    if len(states) < 4 or len(comps) < 2:
        return None
    states.sort()
    y = np.array([[by_state[rc][j] for j in comps] for rc in states], float)
    root = np.array([rc[0] for rc in states])
    child = np.array([rc[1] for rc in states])
    if len(np.unique(root)) < 2 or len(np.unique(child)) < 2:
        return None
    try:
        d = decompose(y, root, child)
    except ValueError:
        return None
    # Raw components are kept unclipped for the pooled estimate. A single cell has
    # 3, 4 and 8 degrees of freedom for the root, child and within terms, so a
    # component whose true value is zero comes out negative about half the time.
    # Clipping first and then averaging fractions would push the root and child
    # shares up; summing raw components across the 19 cells at a step and clipping
    # once at the end does not.
    ra = d.sigma2_root * scale
    rb = d.sigma2_child * scale
    re_ = d.sigma2_within * scale
    pa = max(ra, 0.0)
    pb = max(rb, 0.0)
    pe = max(re_, 0.0)
    tot = pa + pb + pe
    if tot <= 0:
        return None
    return {
        "sigma2_root": pa, "sigma2_child": pb, "sigma2_within": pe, "total": tot,
        "raw_root": float(ra), "raw_child": float(rb), "raw_within": float(re_),
        "frac_root": pa / tot, "frac_child": pb / tot, "frac_within": pe / tot,
        "n_states": int(len(states)), "k": int(len(comps)),
    }


def main() -> None:
    directory, out_path = sys.argv[1], sys.argv[2]
    cells = load(directory)
    out: dict = {"quality_keys": QUALITY_KEYS, "per_quality": {}}

    for q in QUALITY_KEYS:
        per_step: dict[str, dict] = {}
        per_cell: dict[str, dict] = {}
        for step in STEPS:
            rows = []
            for (t, s), items in cells.items():
                if s != step:
                    continue
                c = cell_components(items, q)
                if c:
                    rows.append((t, c))
                    per_cell[f"{t}|{step}"] = c
            if not rows:
                continue
            fw = np.array([c["frac_within"] for _, c in rows])
            fc = np.array([c["frac_child"] for _, c in rows])
            fr = np.array([c["frac_root"] for _, c in rows])
            tot = np.array([c["total"] for _, c in rows])
            # pooled: sum the unclipped components over cells, clip once, normalise
            sa = max(float(np.sum([c["raw_root"] for _, c in rows])), 0.0)
            sb = max(float(np.sum([c["raw_child"] for _, c in rows])), 0.0)
            se = max(float(np.sum([c["raw_within"] for _, c in rows])), 0.0)
            stot = sa + sb + se
            per_step[str(step)] = {
                "n_cells": len(rows),
                "frac_within_mean": float(fw.mean()), "frac_within_sd": float(fw.std(ddof=1)),
                "frac_child_mean": float(fc.mean()),
                "frac_root_mean": float(fr.mean()),
                "frac_within_pooled": float(se / stot) if stot > 0 else float("nan"),
                "frac_child_pooled": float(sb / stot) if stot > 0 else float("nan"),
                "frac_root_pooled": float(sa / stot) if stot > 0 else float("nan"),
                "total_var_mean": float(tot.mean()),
                "within_var_mean": float(np.mean([c["sigma2_within"] for _, c in rows])),
            }
        out["per_quality"][q] = {"by_step": per_step, "by_cell": per_cell}
        print(f"\n{q}")
        for s, v in per_step.items():
            print(f"  step{s:>4s} n={v['n_cells']:2d}  pooled within {v['frac_within_pooled']:.4f}"
                  f"  child {v['frac_child_pooled']:.4f}  root {v['frac_root_pooled']:.4f}"
                  f"   | mean-of-cells within {v['frac_within_mean']:.4f}"
                  f" +- {v['frac_within_sd']:.4f}   total var {v['total_var_mean']:.6f}")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
