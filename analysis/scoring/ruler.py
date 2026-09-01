#!/usr/bin/env python
"""What part of a recorded score is the ruler, and which part of that attenuates commitment.

Each completion in the parent harvest was read once. Replaying that read is
bit-identical (Stage 0), so there is no scoring lottery. What remains is that the
reading came from one member of AlphaFold's five-weight multimer ensemble, and a
different arbitrary choice of member would have given a different number.

**The harvest recorded one member, not an ensemble average.** ColabDesign's option
default is ``num_models = 1`` and the reward model calls ``run`` with
``sample_models=False``, so ``_get_model_nums`` returns ``models[:1]`` -- member 0,
``model_1_multimer_v3`` -- deterministically. Two replays confirm it: the default
call path (``models=[0,1,2,3,4]``) and an explicitly single-member call both
reproduce every stored value bit-for-bit, which a five-member average could not do
because members 1-4 demonstrably disagree with member 0.

That matters for what to subtract. Write the five readings of structure ``s`` as

    v[s, m] = alpha_s + beta_m + gamma_sm

``alpha_s`` is the consensus reading, ``beta_m`` a member's constant offset, and
``gamma_sm`` the structure-by-member interaction. The harvest recorded
``v[s, 0]``. Inside one cell ``beta_0`` is the same additive constant for all 1024
completions, so it cancels out of every variance component and cannot attenuate
commitment at all. Only ``gamma_s0`` varies from completion to completion and
behaves like the independent per-completion error the correction assumes.

So the two are reported separately and only the interaction is subtracted.

**The level dependence is not monotone.** The plan expected the ruler to get noisier
toward the poor-confidence end. What the members actually do is agree tightly at
*both* extremes and disagree in the middle: a complex that is confidently good or
confidently bad is read the same way by all five weights, while an ambiguous one is
not. A single scalar per quality would therefore be wrong in both directions, so the
error is modelled against level.

The correction for a cell follows the law of total variance over that cell's own
1024 recorded levels,

    Var(ruler error) = E_x[ Var(error | level x) ] + Var_x( E[error | level x] )

reported as two terms: ``interaction`` and ``offset_variation``. A member's
*constant* offset appears in neither, which is the point.

Two binning schemes are provided because they answer slightly different questions.
``recorded`` bins on the value the harvest stored, which is also the variable the
sample was stratified on and the variable the correction is applied at, so the law of
total variance above is exact. ``consensus`` bins on the five-member mean, which
describes the ruler's behaviour without conditioning on one member's own error.
Both are computed; the corrected results are reported under one and checked under the
other.

Framing: with the choice of member treated as arbitrary, the interaction is an
**upper bound** on the per-completion scoring noise, not a seed reliability. The five
members share an architecture and a training set, so an error common to all of them
is invisible here and stays in the completion-level term.

    ruler.py <raters_dir> <out_json> [--per-target2 DIR]
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

#: reported quality -> the harvest metric key it is computed from
QUALITY_KEYS = {
    "interface_quality": "af2folding_i_pae",
    "structural_confidence": "af2folding_plddt_log",
    "iptm": "af2folding_i_ptm_log",
}
N_BINS = 12
RECORDED_MEMBER = 0
STEPS = (10, 25, 50, 75, 100, 200, 300)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #


def load_raters(raters_dir: str) -> dict[str, dict[str, np.ndarray]]:
    """Per quality: the [S, 5] reading matrix with the recorded member first."""
    rows: dict[str, list] = {q: [] for q in QUALITY_KEYS}
    meta: list[tuple[str, int, str]] = []
    files = sorted(
        f for f in glob.glob(os.path.join(raters_dir, "*.json"))
        if not os.path.basename(f).startswith("spec_")
    )
    for f in files:
        with open(f) as fh:
            payload = json.load(fh)
        pass_names = list(payload["spec"]["passes"])
        for row in payload["results"]:
            vals, ok = {}, True
            for q, key in QUALITY_KEYS.items():
                v = np.array([float(row["stored"][key])]
                             + [float(row["passes"][p][key]) for p in pass_names], dtype=float)
                if not np.isfinite(v).all():
                    ok = False
                vals[q] = v
            if not ok:
                continue
            for q in QUALITY_KEYS:
                rows[q].append(vals[q])
            meta.append((row["target"], int(row["step"]), row["tag"]))
    out = {}
    for q in QUALITY_KEYS:
        out[q] = {
            "V": np.asarray(rows[q], dtype=float),
            "target": np.asarray([m[0] for m in meta]),
            "step": np.asarray([m[1] for m in meta], int),
        }
    n_m = out[next(iter(QUALITY_KEYS))]["V"].shape[1] if meta else 0
    print(f"loaded {len(files)} target files, {len(meta)} structures, {n_m} members each")
    return out


def load_designability_pairs(per_target2_dir: str) -> dict[str, np.ndarray]:
    """The parent's repeat-measure control: the same state measured twice.

    ProteinMPNN redesign and the refold are stochastic, so designability is the one
    quality with a genuine repeat measurement already in hand. Both passes use the
    same 17-sequence protocol with an independent seed, so they are exchangeable.
    """
    p1_all, p2_all, tgt, stp = [], [], [], []
    for path in sorted(glob.glob(os.path.join(per_target2_dir, "*_step*.json"))):
        base = os.path.basename(path)
        if base.endswith("_boot.npz"):
            continue
        with open(path) as fh:
            j = json.load(fh)
        rep = j.get("cell", {}).get("designability", {}).get("repeat_measure", {})
        a = np.asarray(rep.get("pass1", []), float)
        b = np.asarray(rep.get("pass2", []), float)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() == 0:
            continue
        p1_all.append(a[ok])
        p2_all.append(b[ok])
        tgt.extend([j["target"]] * int(ok.sum()))
        stp.extend([j["step"]] * int(ok.sum()))
    if not p1_all:
        return {}
    return {
        "p1": np.concatenate(p1_all), "p2": np.concatenate(p2_all),
        "target": np.asarray(tgt), "step": np.asarray(stp, int),
    }


# --------------------------------------------------------------------------- #
# the two-way split
# --------------------------------------------------------------------------- #


def two_way(V: np.ndarray) -> dict:
    """Member main effect and structure-by-member interaction for a [S, M] matrix."""
    S, M = V.shape
    row = V.mean(axis=1, keepdims=True)
    col = V.mean(axis=0, keepdims=True)
    grand = V.mean()
    gamma = V - row - col + grand
    ms_ab = float((gamma ** 2).sum() / ((S - 1) * (M - 1)))
    beta = (col - grand).ravel()
    var_beta_raw = float(np.var(beta, ddof=1))
    # E[var of the estimated offsets] = true offset variance + sigma2_gamma / S
    var_beta = max(var_beta_raw - ms_ab / S, 0.0)
    return {
        "n_structures": int(S),
        "n_members": int(M),
        "member_offsets": [float(b) for b in beta],
        "member_offset_sd_raw": float(np.sqrt(var_beta_raw)),
        "member_offset_sd": float(np.sqrt(var_beta)),
        "recorded_member_offset": float(beta[RECORDED_MEMBER]),
        "recorded_minus_min_member": float(beta[RECORDED_MEMBER] - beta.min()),
        "recorded_minus_max_member": float(beta[RECORDED_MEMBER] - beta.max()),
        "interaction_var_unbinned": float(ms_ab),
        "interaction_sd_unbinned": float(np.sqrt(ms_ab)),
        "structure_sd": float(np.std(row.ravel(), ddof=1)),
    }


def level_model(V: np.ndarray, n_bins: int = N_BINS, bin_on: str = "recorded") -> dict:
    """Ruler error against level, as interpolation knots in recorded-value units.

    ``d = v_recorded - v_mean_over_members`` estimates the recorded member's deviation
    from consensus, shrunk because the consensus mean contains that member. Writing the
    recorded member's deviation as ``e_0`` and the others' as ``e_j``,
    ``d = ((M-1)/M) e_0 - (1/M) sum_{j != 0} e_j``. Taking the conditional mean and
    assuming the other members' conditional mean deviation is about zero leaves
    ``E[d] = ((M-1)/M) E[e_0]``, so the mean is scaled once by ``M/(M-1)``. Taking the
    conditional variance with independent deviations of common variance ``s2`` gives
    ``Var(d) = ((M-1)/M) s2``, so the variance is also scaled once by ``M/(M-1)``.
    Scaling the mean by the squared factor would inflate the offset-variation term by
    ``(M/(M-1))**2`` again, which is wrong.
    """
    S, M = V.shape
    rec = V[:, RECORDED_MEMBER]
    cons = V.mean(axis=1)
    d = rec - cons
    x = rec if bin_on == "recorded" else cons
    f_var = M / (M - 1)
    f_mean = M / (M - 1)

    order = np.argsort(x)
    edges = np.linspace(0, S, n_bins + 1).round().astype(int)
    centres, offs, inter, ns = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < 8:
            continue
        idx = order[a:b]
        centres.append(float(np.mean(rec[idx])))
        offs.append(float(np.mean(d[idx]) * f_mean))
        inter.append(float(np.var(d[idx], ddof=1) * f_var))
        ns.append(int(b - a))
    o = np.argsort(centres)
    return {
        "bin_on": bin_on,
        "centres": [centres[i] for i in o],
        "recorded_offset": [offs[i] for i in o],
        "interaction_var": [inter[i] for i in o],
        "n": [ns[i] for i in o],
    }


def pair_level_model(p1: np.ndarray, p2: np.ndarray, n_bins: int = N_BINS,
                     bin_on: str = "recorded") -> dict:
    """Same thing for a quality with two exchangeable repeat measurements.

    ``Var(p1 - p2) = 2 sigma^2``, and by symmetry there is no member offset to
    separate, so the offset term is zero at every level.
    """
    d = p1 - p2
    x = p1 if bin_on == "recorded" else 0.5 * (p1 + p2)
    order = np.argsort(x)
    n = len(x)
    edges = np.linspace(0, n, n_bins + 1).round().astype(int)
    centres, inter, ns = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < 8:
            continue
        idx = order[a:b]
        centres.append(float(np.mean(p1[idx])))
        inter.append(float(np.var(d[idx], ddof=1) / 2.0))
        ns.append(int(b - a))
    o = np.argsort(centres)
    return {
        "bin_on": bin_on,
        "centres": [centres[i] for i in o],
        "recorded_offset": [0.0 for _ in o],
        "interaction_var": [inter[i] for i in o],
        "n": [ns[i] for i in o],
    }


def _interp(model: dict, field: str, x: np.ndarray) -> np.ndarray:
    c = np.asarray(model["centres"], float)
    v = np.asarray(model[field], float)
    if c.size == 0:
        return np.zeros_like(np.asarray(x, float))
    return np.interp(np.asarray(x, float), c, v, left=v[0], right=v[-1])


def cell_sigma2(model: dict, levels: np.ndarray) -> dict:
    """Per-completion ruler variance for one cell, by the law of total variance.

    ``levels`` are the cell's own recorded values, all of them, so the estimate is
    weighted by the cell's real distribution of scores rather than by the stratified
    sample the model was fitted on.
    """
    x = np.asarray(levels, float).ravel()
    x = x[np.isfinite(x)]
    inter = float(np.mean(_interp(model, "interaction_var", x)))
    b = _interp(model, "recorded_offset", x)
    off = float(np.var(b, ddof=1)) if b.size > 1 else 0.0
    return {"interaction": inter, "offset_variation": off, "total": inter + off}


# --------------------------------------------------------------------------- #


def summarize_af2(q: str, V: np.ndarray) -> dict:
    glob_ = two_way(V)
    rec = V[:, RECORDED_MEMBER]
    models = {b: level_model(V, bin_on=b) for b in ("recorded", "consensus")}
    lm = models["recorded"]
    iv = np.asarray(lm["interaction_var"], float)
    peak = int(np.argmax(iv)) if iv.size else 0
    return {
        "recorded_member": RECORDED_MEMBER,
        "global": glob_,
        "level_model": lm,
        "level_model_consensus_binned": models["consensus"],
        "level_min": float(rec.min()),
        "level_max": float(rec.max()),
        "interaction_sd_mean_over_sample": float(np.sqrt(np.mean(_interp(lm, "interaction_var", rec)))),
        "interaction_sd_peak": float(np.sqrt(iv[peak])) if iv.size else float("nan"),
        "interaction_peak_at_level": float(lm["centres"][peak]) if iv.size else float("nan"),
        "interaction_sd_at_lowest_bin": float(np.sqrt(iv[0])) if iv.size else float("nan"),
        "interaction_sd_at_highest_bin": float(np.sqrt(iv[-1])) if iv.size else float("nan"),
        "offset_range_over_levels": float(np.ptp(lm["recorded_offset"])) if lm["recorded_offset"] else 0.0,
        "spread_across_sampled_completions_sd": float(np.std(rec, ddof=1)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raters_dir")
    ap.add_argument("out_json")
    ap.add_argument("--per-target2", default=os.environ.get("PER_TARGET2", ""))
    args = ap.parse_args()

    packed = load_raters(args.raters_dir)
    out = {
        "n_bins": N_BINS,
        "recorded_member": RECORDED_MEMBER,
        "recorded_member_name": "model_1_multimer_v3",
        "harvest_recorded": "single ensemble member (num_models=1, sample_models=False)",
        "quality_keys": QUALITY_KEYS,
        "per_quality": {},
    }
    for q, d in packed.items():
        s = summarize_af2(q, d["V"])
        out["per_quality"][q] = s
        g = s["global"]
        print(
            f"\n{q}\n"
            f"  structures                 {g['n_structures']}\n"
            f"  member offsets             " + ", ".join(f"{b:+.5f}" for b in g["member_offsets"]) + "\n"
            f"  member offset sd           {g['member_offset_sd']:.5f}"
            f"   (recorded member {g['recorded_member_offset']:+.5f})\n"
            f"  interaction sd, mean       {s['interaction_sd_mean_over_sample']:.5f}\n"
            f"  interaction sd, peak       {s['interaction_sd_peak']:.5f} at level "
            f"{s['interaction_peak_at_level']:.4f}\n"
            f"  interaction sd, extremes   {s['interaction_sd_at_lowest_bin']:.5f} (low) / "
            f"{s['interaction_sd_at_highest_bin']:.5f} (high)\n"
            f"  spread across completions  {s['spread_across_sampled_completions_sd']:.5f}"
        )

    if args.per_target2:
        pairs = load_designability_pairs(args.per_target2)
        if pairs:
            p1, p2 = pairs["p1"], pairs["p2"]
            lm = pair_level_model(p1, p2, bin_on="recorded")
            lmc = pair_level_model(p1, p2, bin_on="consensus")
            iv = np.asarray(lm["interaction_var"], float)
            out["per_quality"]["designability"] = {
                "source": "parent repeat-measure control (independent ProteinMPNN redesign + refold)",
                "global": {
                    "n_pairs": int(p1.size),
                    "interaction_var_unbinned": float(np.var(p1 - p2, ddof=1) / 2.0),
                    "interaction_sd_unbinned": float(np.sqrt(np.var(p1 - p2, ddof=1) / 2.0)),
                    "mean_abs_diff": float(np.mean(np.abs(p1 - p2))),
                    "r": float(np.corrcoef(p1, p2)[0, 1]),
                    "member_offsets": [],
                    "member_offset_sd": 0.0,
                    "recorded_member_offset": 0.0,
                },
                "level_model": lm,
                "level_model_consensus_binned": lmc,
                "interaction_sd_mean_over_sample": float(np.sqrt(np.mean(_interp(lm, "interaction_var", p1)))),
                "interaction_sd_peak": float(np.sqrt(iv.max())) if iv.size else float("nan"),
                "interaction_peak_at_level": float(lm["centres"][int(np.argmax(iv))]) if iv.size else float("nan"),
                "spread_across_sampled_completions_sd": float(np.std(p1, ddof=1)),
            }
            d = out["per_quality"]["designability"]
            print(f"\ndesignability (repeat-measure control)\n"
                  f"  pairs                      {d['global']['n_pairs']}\n"
                  f"  repeat r                   {d['global']['r']:.4f}   "
                  f"mean abs diff {d['global']['mean_abs_diff']:.4f} A\n"
                  f"  noise sd, unbinned         {d['global']['interaction_sd_unbinned']:.5f} A\n"
                  f"  noise sd, mean over sample {d['interaction_sd_mean_over_sample']:.5f} A\n"
                  f"  spread across completions  {d['spread_across_sampled_completions_sd']:.5f} A")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n-> {args.out_json}")


if __name__ == "__main__":
    main()
