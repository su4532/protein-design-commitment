#!/usr/bin/env python3
"""Draw the manuscript figures from results/figure_data.json.

Vector PDF for LaTeX, PNG twin at 300 dpi so the figures can be inspected
visually before they ship.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Stable metadata for anonymized, byte-reproducible builds.
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "fixtures" / "figure_data.json"
FIGDIR = ROOT / "generated_figures"

QUALITIES = ["fold_identity", "structural_confidence", "designability", "interface_quality"]
SHORT = {
    "fold_identity": "Fold identity",
    "structural_confidence": "Structural confidence",
    "designability": "Designability",
    "interface_quality": "Interface quality",
}
# Interface quality is the object of study: it is both the widest-spread and the
# least readable quality, so it carries the emphasis colour in every figure.
COLOR = {
    "fold_identity": "#1f6f9c",
    "structural_confidence": "#4c9f70",
    "designability": "#8e6bab",
    "interface_quality": "#c4650d",
}
FAMILY_LABEL = {
    "linear_activation": "Linear, activations",
    "geometry": "Decoded partial structure",
    "mlp": "MLP, activations",
}
FAMILY_STYLE = {
    "linear_activation": dict(hatch=None, alpha=1.0),
    "geometry": dict(hatch="///", alpha=0.55),
    "mlp": dict(hatch="...", alpha=0.75),
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 7.2,
        "axes.labelsize": 7.6,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 6.8,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def save(fig, name: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / f"{name}.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(FIGDIR / f"{name}.png", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


# ---------------------------------------------------------------------------
# Main Figure 1: pooled commitment beside the per-target settling step, on one
# shared 0-400 axis. Rendered at the manuscript's own \textwidth (5.5 in) so the
# type sizes below are the sizes a reader sees; \includegraphics does not scale.
# ---------------------------------------------------------------------------
PAPER_TEXTWIDTH_IN = 5.5
XTICKS_400 = [0, 100, 200, 300, 400]
GUIDES_400 = [100, 200, 300]
GUIDE_KW = dict(color="#cfcfc9", lw=0.55, ls=(0, (3, 2.4)), zorder=0)

# Type for the two at-scale main figures, in points as rendered.
ATSCALE = {
    "font.size": 6.8,
    "axes.labelsize": 7.0,
    "axes.titlesize": 7.2,
    "xtick.labelsize": 6.6,
    "ytick.labelsize": 6.6,
    "legend.fontsize": 6.0,
}


def _fit_titles(fig, axes, sizes=(7.2, 7.0, 6.8, 6.6, 6.4, 6.2)) -> float:
    """Largest title size whose widest title still fits inside its own panel."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.dpi_scale_trans.inverted()

    def w(a):
        return a.get_window_extent(renderer=rend).transformed(inv).width

    panel_w = min(w(ax) for ax in axes)
    for cand in sizes:
        for ax in axes:
            ax.title.set_fontsize(cand)
        fig.canvas.draw()
        if max(w(ax.title) for ax in axes) <= panel_w - 0.03:
            return cand
    raise AssertionError("no title size fits the panel width")


def figure1_schedule(d: dict) -> None:
    steps = d["design"]["capture_steps"]
    com = d["commitment"]
    rows = d["fig1_settling_steps"]["rows"]
    floor = d["fig1_settling_steps"]["floor_step"]
    cl = d["spread_cluster_unit"]

    with plt.rc_context(ATSCALE):
        fig, (axA, axB) = plt.subplots(
            1, 2, figsize=(PAPER_TEXTWIDTH_IN, 1.72), sharex=True
        )
        fig.subplots_adjust(left=0.072, right=0.988, top=0.882, bottom=0.235, wspace=0.075)

        # ---- (a) pooled commitment trajectories ---------------------------
        for s in GUIDES_400:
            axA.axvline(s, **GUIDE_KW)
        for q in QUALITIES:
            mean = [com[q]["by_step"][str(s)] for s in steps]
            lo = [com[q]["by_step_lo"][str(s)] for s in steps]
            hi = [com[q]["by_step_hi"][str(s)] for s in steps]
            axA.fill_between(steps, lo, hi, color=COLOR[q], alpha=0.13, lw=0, zorder=2)
            axA.plot(steps, mean, "-", lw=1.1, color=COLOR[q], zorder=3, label=SHORT[q])
            axA.plot(steps, mean, "o", ms=2.1, color=COLOR[q], mew=0, zorder=4)
        axA.axhline(0.5, color="#8a8a85", lw=0.6, ls=(0, (3, 2)), zorder=1)
        axA.text(396, 0.515, "half commitment", fontsize=5.6, color="#6b6b66",
                 va="bottom", ha="right")
        axA.set_ylim(0, 1.03)
        axA.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        axA.set_ylabel("Commitment")
        axA.set_title("a  Pooled over targets", loc="left", fontweight="bold", pad=3.0)
        axA.legend(loc="lower right", ncol=2, handlelength=1.1, handletextpad=0.4,
                   labelspacing=0.24, columnspacing=0.9, borderaxespad=0.12,
                   frameon=True, facecolor="white", edgecolor="none", framealpha=1.0,
                   borderpad=0.22)

        # ---- (b) per-target settling step ---------------------------------
        for s in GUIDES_400:
            axB.plot([s, s], [-0.44, 3.45], **GUIDE_KW)
        rng = np.random.default_rng(42)
        for i, q in enumerate(QUALITIES):
            y0 = len(QUALITIES) - 1 - i
            jitter = rng.uniform(-0.19, 0.19, size=len(rows))
            for j, r in enumerate(rows):
                v = r["values"][q]
                y = y0 + jitter[j]
                if v["censored"]:
                    axB.plot(v["t_half"], y, marker="<", ms=3.6, color=COLOR[q],
                             mfc="white", mew=0.8, zorder=4, clip_on=False)
                else:
                    axB.plot([v["t_half"] - v["se"], v["t_half"] + v["se"]], [y, y],
                             color=COLOR[q], lw=0.65, alpha=0.38, zorder=3,
                             solid_capstyle="butt")
                    axB.plot(v["t_half"], y, "o", ms=2.7, color=COLOR[q], mew=0, zorder=4)
            med = cl["per_metric"][q]["median_t_half"]
            axB.plot([med, med], [y0 - 0.34, y0 + 0.34], color="#1d272a", lw=1.3, zorder=5)
        axB.axvline(floor, color="#8a8a85", lw=0.6, ls=(0, (3, 2)), zorder=1)
        axB.set_yticks([])
        axB.set_ylim(-0.98, 3.45)
        rowlab = blended_transform_factory(axB.transAxes, axB.transData)
        row_labels = {}
        for i, q in enumerate(QUALITIES):
            row_labels[q] = axB.text(
                0.992, len(QUALITIES) - 1 - i, SHORT[q], transform=rowlab, ha="right",
                va="center", fontsize=6.0, color=COLOR[q], zorder=6,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.8),
            )
        axB.set_title("b  One point per target", loc="left", fontweight="bold", pad=3.0)
        key = [
            Line2D([], [], marker="o", ms=2.7, ls="", color="#4a4a46", label="target ±1 s.e."),
            Line2D([], [], marker="<", ms=3.6, ls="", color="#4a4a46", mfc="white", mew=0.8,
                   label="at floor: upper bound"),
            Line2D([], [], color="#1d272a", lw=1.3, label="cluster median"),
        ]
        axB.legend(handles=key, loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=3,
                   fontsize=5.5, handlelength=0.9, handletextpad=0.35, columnspacing=0.9,
                   borderaxespad=0.0, frameon=True, facecolor="white", edgecolor="none",
                   framealpha=1.0, borderpad=0.15)

        for ax in (axA, axB):
            ax.set_xlim(0, 400)
            ax.set_xticks(XTICKS_400)
        fig.supxlabel("Denoising step (of 400)", fontsize=7.0, x=0.53, y=0.035)

        size = _fit_titles(fig, (axA, axB))
        # every in-panel row label must clear the furthest target in its own row
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        clear = {}
        for q, art in row_labels.items():
            left = axB.transData.inverted().transform(
                (art.get_window_extent(renderer=rend).x0, 0))[0]
            furthest = max(v["t_half"] + (0.0 if v["censored"] else v["se"])
                           for v in (r["values"][q] for r in rows))
            clear[q] = (round(float(left), 1), round(float(furthest), 1),
                        bool(left - 8.0 > furthest))
        assert all(c[2] for c in clear.values()), f"row label hits data: {clear}"
        print(f"  fig1_schedule: titles {size} pt, row-label clearance {clear}")
        save(fig, "fig1_schedule")


# ---------------------------------------------------------------------------
# Main Figure 2: the activation readout against the decoded partial structure
# ---------------------------------------------------------------------------
def figure2_readout(d: dict) -> None:
    f2 = d["fig2_readability"]
    fams = ["linear_activation", "geometry"]

    with plt.rc_context(ATSCALE):
        fig, (axL, axR) = plt.subplots(
            1, 2, figsize=(PAPER_TEXTWIDTH_IN, 1.40),
            gridspec_kw={"width_ratios": [1.62, 1.0], "wspace": 0.10},
        )
        h = 0.30
        for i, q in enumerate(QUALITIES):
            base = len(QUALITIES) - 1 - i
            for k, fam in enumerate(fams):
                e = f2[q]["families"][fam]
                y = base + (0.5 - k) * h
                st = FAMILY_STYLE[fam]
                axL.barh(y, e["peak_mean"], height=h * 0.86, color=COLOR[q],
                         alpha=st["alpha"], hatch=st["hatch"], edgecolor="white",
                         linewidth=0.4, zorder=2)
                if fam == "linear_activation":
                    axL.plot([e["peak_lo"], e["peak_hi"]], [y, y], color="#1d272a",
                             lw=0.7, alpha=0.8, zorder=3, solid_capstyle="butt")
        axL.set_yticks(range(len(QUALITIES)))
        axL.set_yticklabels([SHORT[q] for q in reversed(QUALITIES)])
        for lab, col in zip(axL.get_yticklabels(), [COLOR[q] for q in reversed(QUALITIES)]):
            lab.set_color(col)
        axL.set_ylim(-0.62, len(QUALITIES) - 0.32)
        axL.set_xlim(0, 0.90)
        axL.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
        axL.set_xlabel("Peak noise-corrected $R^2$")
        axL.set_title("a  What the readout sees", loc="left", fontweight="bold", pad=3.0)
        hL = [
            plt.Rectangle((0, 0), 1, 1, facecolor="#4a4a46", alpha=FAMILY_STYLE[f]["alpha"],
                          hatch=FAMILY_STYLE[f]["hatch"], edgecolor="white", linewidth=0.4,
                          label=FAMILY_LABEL[f])
            for f in fams
        ]
        axL.legend(handles=hL, loc="lower right", ncol=1, frameon=True, facecolor="white",
                   edgecolor="none", framealpha=1.0, handlelength=1.2, handletextpad=0.45,
                   labelspacing=0.25, borderaxespad=0.15, borderpad=0.2, fontsize=5.8)

        for i, q in enumerate(QUALITIES):
            y = len(QUALITIES) - 1 - i
            e = f2[q]
            axR.barh(y, e["ceiling_share_at_peak"], height=0.46, color=COLOR[q], zorder=2)
            axR.plot([e["ceiling_share_at_peak_lo"], e["ceiling_share_at_peak_hi"]], [y, y],
                     color="#1d272a", lw=0.7, alpha=0.8, zorder=3, solid_capstyle="butt")
        axR.axvline(1.0, color="#8a8a85", lw=0.6, ls=(0, (3, 2)), zorder=1)
        axR.text(0.985, len(QUALITIES) - 0.5, "all of it", fontsize=5.6, color="#6b6b66",
                 va="center", ha="right")
        axR.set_yticks(range(len(QUALITIES)))
        axR.set_yticklabels([])
        axR.set_ylim(-0.62, len(QUALITIES) - 0.32)
        axR.set_xlim(0, 1.08)
        axR.set_xticks([0.0, 0.5, 1.0])
        axR.set_xlabel("Share of the ceiling")
        axR.set_title("b  Against what is there", loc="left", fontweight="bold", pad=3.0)

        size = _fit_titles(fig, (axL, axR))
        print(f"  fig2_readout: titles {size} pt")
        save(fig, "fig2_readout")


# ---------------------------------------------------------------------------
# Appendix figure: the corrected coefficient of variation at both units, the
# panel lifted out of the previous main Figure 1.
# ---------------------------------------------------------------------------
def figure_a4(d: dict) -> None:
    cl, tg = d["spread_cluster_unit"], d["spread_target_unit"]
    fig, ax = plt.subplots(figsize=(3.5, 2.05))
    ypos = np.arange(len(QUALITIES))[::-1]
    off = 0.17
    for i, q in enumerate(QUALITIES):
        y = ypos[i]
        for blk, dy, filled in ((cl, +off, True), (tg, -off, False)):
            m = blk["per_metric"][q]
            lo, hi = m["cv_corrected_ci95"]
            ax.plot([lo, hi], [y + dy, y + dy], color=COLOR[q], lw=1.15,
                    alpha=1.0 if filled else 0.55, solid_capstyle="butt", zorder=2)
            ax.plot(m["cv_corrected"], y + dy, "o", ms=4.0, color=COLOR[q],
                    mfc=COLOR[q] if filled else "white", mew=0.9, zorder=3)
    ax.axvline(0, color="#8a8a85", lw=0.6, zorder=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([SHORT[q] for q in QUALITIES])
    for lab, col in zip(ax.get_yticklabels(), [COLOR[q] for q in QUALITIES]):
        lab.set_color(col)
    ax.set_ylim(-0.62, len(QUALITIES) - 0.25)
    ax.set_xlim(-0.03, 1.30)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Corrected coefficient of variation of the settling step")
    hK = [
        Line2D([], [], marker="o", ms=4.0, ls="", color="#4a4a46", label="12 structural clusters"),
        Line2D([], [], marker="o", ms=4.0, ls="", color="#4a4a46", mfc="white", mew=0.9,
               label="19 targets (optimistic)"),
    ]
    # upper right is the only quadrant free of an interval: fold identity, the
    # tightest quality, sits at the top and stops well short of 0.5.
    ax.legend(handles=hK, loc="upper right", frameon=False, handletextpad=0.4,
              labelspacing=0.28, borderaxespad=0.2)
    save(fig, "figA4_cv")


# ---------------------------------------------------------------------------
# Superseded main Figure 1, kept as the source of the appendix CV panel
# ---------------------------------------------------------------------------
def figure1(d: dict) -> None:
    rows = d["fig1_settling_steps"]["rows"]
    floor = d["fig1_settling_steps"]["floor_step"]
    cl = d["spread_cluster_unit"]
    tg = d["spread_target_unit"]

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(6.6, 2.32), gridspec_kw={"width_ratios": [1.75, 1.0], "wspace": 0.32}
    )

    rng = np.random.default_rng(42)
    for i, q in enumerate(QUALITIES):
        y0 = len(QUALITIES) - 1 - i
        vals = [r["values"][q] for r in rows]
        jitter = rng.uniform(-0.19, 0.19, size=len(vals))
        for j, v in enumerate(vals):
            y = y0 + jitter[j]
            if v["censored"]:
                axL.plot(
                    v["t_half"], y, marker="<", ms=4.2, color=COLOR[q],
                    mfc="white", mew=0.9, zorder=3, clip_on=False,
                )
            else:
                axL.plot(
                    [v["t_half"] - v["se"], v["t_half"] + v["se"]], [y, y],
                    color=COLOR[q], lw=0.7, alpha=0.38, zorder=2, solid_capstyle="butt",
                )
                axL.plot(v["t_half"], y, "o", ms=3.1, color=COLOR[q], mew=0, zorder=3)
        med = cl["per_metric"][q]["median_t_half"]
        axL.plot([med, med], [y0 - 0.34, y0 + 0.34], color="#1d272a", lw=1.5, zorder=4)

    axL.axvline(floor, color="#8a8a85", lw=0.6, ls=(0, (3, 2)), zorder=1)
    axL.text(
        floor + 4, -0.40, "earliest capture step (10)",
        fontsize=6.2, color="#6b6b66", va="center", ha="left",
    )
    axL.set_yticks(range(len(QUALITIES)))
    axL.set_yticklabels([SHORT[q] for q in reversed(QUALITIES)])
    axL.set_ylim(-0.62, len(QUALITIES) - 0.25)
    axL.set_xlim(0, 285)
    axL.set_xlabel("Settling step (denoising step at half commitment, of 400)")
    axL.set_title("a  Per-target settling step", loc="left", fontweight="bold")
    for lab, col in zip(axL.get_yticklabels(), [COLOR[q] for q in reversed(QUALITIES)]):
        lab.set_color(col)

    handles = [
        Line2D([], [], marker="o", ms=3.1, ls="", color="#4a4a46", label="one target (±1 s.e.)"),
        Line2D([], [], marker="<", ms=4.2, ls="", color="#4a4a46", mfc="white", mew=0.9,
               label="at the floor: an upper bound"),
        Line2D([], [], color="#1d272a", lw=1.5, label="median over 12 clusters"),
    ]
    axL.legend(handles=handles, loc="upper right", frameon=False, handletextpad=0.5,
               borderaxespad=0.2, labelspacing=0.32)

    # right panel: corrected coefficient of variation at both units
    ypos = np.arange(len(QUALITIES))[::-1]
    off = 0.17
    for i, q in enumerate(QUALITIES):
        y = ypos[i]
        for blk, dy, filled in ((cl, +off, True), (tg, -off, False)):
            m = blk["per_metric"][q]
            lo, hi = m["cv_corrected_ci95"]
            axR.plot([lo, hi], [y + dy, y + dy], color=COLOR[q], lw=1.15,
                     alpha=1.0 if filled else 0.55, solid_capstyle="butt", zorder=2)
            axR.plot(
                m["cv_corrected"], y + dy, "o", ms=4.0, color=COLOR[q],
                mfc=COLOR[q] if filled else "white", mew=0.9, zorder=3,
            )
    axR.axvline(0, color="#8a8a85", lw=0.6, zorder=1)
    axR.set_yticks(ypos)
    axR.set_yticklabels([])
    axR.set_ylim(-0.62, len(QUALITIES) - 0.25)
    axR.set_xlim(-0.03, 1.28)
    axR.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axR.set_xlabel("Corrected coefficient of variation")
    axR.set_title("b  Spread of the settling step", loc="left", fontweight="bold")
    # Label the two inference units directly on the top row instead of adding a
    # legend box, which would land on the fold-identity interval.
    top = ypos[0]
    hi_top = cl["per_metric"][QUALITIES[0]]["cv_corrected_ci95"][1]
    axR.text(hi_top + 0.05, top + off, "12 structural clusters", fontsize=6.2,
             color="#4a4a46", va="center", ha="left")
    hi_top_t = tg["per_metric"][QUALITIES[0]]["cv_corrected_ci95"][1]
    axR.text(hi_top_t + 0.05, top - off, "19 targets (optimistic)", fontsize=6.2,
             color="#8a8a85", va="center", ha="left")

    save(fig, "fig1_settling")


# ---------------------------------------------------------------------------
# Figure 2: how far a readout of the denoiser's own activations gets
# ---------------------------------------------------------------------------
def figure2(d: dict) -> None:
    f2 = d["fig2_readability"]
    fams = ["linear_activation", "geometry", "mlp"]

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(6.6, 2.15), gridspec_kw={"width_ratios": [1.55, 1.0], "wspace": 0.30}
    )

    h = 0.24
    for i, q in enumerate(QUALITIES):
        base = len(QUALITIES) - 1 - i
        for k, fam in enumerate(fams):
            e = f2[q]["families"][fam]
            y = base + (1 - k) * h
            st = FAMILY_STYLE[fam]
            axL.barh(
                y, e["peak_mean"], height=h * 0.88, color=COLOR[q], alpha=st["alpha"],
                hatch=st["hatch"], edgecolor="white", linewidth=0.45, zorder=2,
            )
            axL.plot([e["peak_lo"], e["peak_hi"]], [y, y], color="#1d272a", lw=0.75,
                     alpha=0.8, zorder=3, solid_capstyle="butt")
            if fam == "linear_activation":
                axL.text(e["peak_hi"] + 0.015, y, f"peak at step {e['peak_step']}", fontsize=6.1,
                         va="center", ha="left", color="#4a4a46")

    axL.set_yticks(range(len(QUALITIES)))
    axL.set_yticklabels([SHORT[q] for q in reversed(QUALITIES)])
    for lab, col in zip(axL.get_yticklabels(), [COLOR[q] for q in reversed(QUALITIES)]):
        lab.set_color(col)
    axL.set_ylim(-0.5, len(QUALITIES) - 0.25)
    axL.set_xlim(0, 1.02)
    axL.set_xlabel("Peak noise-corrected $R^2$ over capture steps")
    axL.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
    axL.set_title("a  Readout family", loc="left", fontweight="bold")
    hL = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#4a4a46", alpha=FAMILY_STYLE[f]["alpha"],
                      hatch=FAMILY_STYLE[f]["hatch"], edgecolor="white", linewidth=0.45,
                      label=FAMILY_LABEL[f])
        for f in fams
    ]
    axL.legend(handles=hL, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=3,
               frameon=False, handlelength=1.4, handletextpad=0.5, columnspacing=1.4,
               borderaxespad=0.0)

    for i, q in enumerate(QUALITIES):
        y = len(QUALITIES) - 1 - i
        e = f2[q]
        axR.barh(y, e["ceiling_share_at_peak"], height=0.5, color=COLOR[q], zorder=2)
        axR.plot([e["ceiling_share_at_peak_lo"], e["ceiling_share_at_peak_hi"]], [y, y],
                 color="#1d272a", lw=0.75, alpha=0.8, zorder=3, solid_capstyle="butt")
    axR.axvline(1.0, color="#8a8a85", lw=0.6, ls=(0, (3, 2)), zorder=1)
    axR.text(0.985, len(QUALITIES) - 0.42, "all of it", fontsize=6.2, color="#6b6b66",
             va="center", ha="right")
    axR.set_yticks(range(len(QUALITIES)))
    axR.set_yticklabels([])
    axR.set_ylim(-0.5, len(QUALITIES) - 0.25)
    axR.set_xlim(0, 1.08)
    axR.set_xlabel("Share of the commitment ceiling recovered")
    axR.set_title("b  Against what is there to recover", loc="left", fontweight="bold")

    save(fig, "fig2_readability")


# ---------------------------------------------------------------------------
# Appendix figure: pooled commitment curves
# ---------------------------------------------------------------------------
def figure_a1(d: dict) -> None:
    steps = d["design"]["capture_steps"]
    com = d["commitment"]
    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    for q in QUALITIES:
        m = [com[q]["by_step"][str(s)] for s in steps]
        lo = [com[q]["by_step_lo"][str(s)] for s in steps]
        hi = [com[q]["by_step_hi"][str(s)] for s in steps]
        ax.fill_between(steps, lo, hi, color=COLOR[q], alpha=0.13, lw=0)
        ax.plot(steps, m, "-o", ms=2.8, lw=1.2, color=COLOR[q], label=SHORT[q])
    ax.axhline(0.5, color="#8a8a85", lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.text(11, 0.525, "half commitment", fontsize=6.1, color="#6b6b66")
    ax.set_xscale("log")
    # Labelling all ten capture steps crowds the top of the decade into an
    # unreadable run, so label a legible subset and keep an unlabelled tick at
    # every captured step.
    ax.set_xticks(steps)
    labelled = {10, 25, 50, 100, 200, 350}
    ax.set_xticklabels([str(s) if s in labelled else "" for s in steps])
    ax.minorticks_off()
    ax.set_xlabel("Capture step (log)")
    ax.set_ylabel("Commitment")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", frameon=False, labelspacing=0.3, handletextpad=0.5,
              borderaxespad=0.2)
    save(fig, "figA1_commitment")


# ---------------------------------------------------------------------------
# Figure 3: does the central comparison survive the ways it could be wrong
# ---------------------------------------------------------------------------
TREATMENT_LABEL = {
    ("observed_cv", "published"): "as published",
    ("observed_cv", "noise_corrected_only"): "scorer disagreement removed",
    ("observed_cv", "continuous_ruler_only"): "continuous fold ruler",
    ("observed_cv", "both"): "both together",
    ("spread_estimator", "published"): "as published",
    ("spread_estimator", "corrected"): "all disagreement charged as error",
    ("spread_estimator", "within_only"): "within-cell error only",
}
FAMILY_TITLE = {
    "observed_cv": "Observed spread",
    "spread_estimator": "Noise-corrected spread",
}
UNIT_LABEL = {"by_cluster": "12 clusters", "by_target": "19 targets"}


def figure3(d: dict) -> None:
    cells = d["noise"]["estimator_cells"]
    quant = d["quantisation"]

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(6.9, 2.85), gridspec_kw={"width_ratios": [1.72, 1.0], "wspace": 0.46}
    )

    # Order rows so the two estimator families are contiguous and each unit is
    # paired, which is how the count of 12 of 14 is read off the panel.
    order = [
        (fam, unit, tr)
        for fam in ("observed_cv", "spread_estimator")
        for tr in (
            ["published", "noise_corrected_only", "continuous_ruler_only", "both"]
            if fam == "observed_cv"
            else ["published", "corrected", "within_only"]
        )
        for unit in ("by_cluster", "by_target")
    ]
    rows = []
    for fam, unit, tr in order:
        rows.append(next(c for c in cells if (c["family"], c["unit"], c["treatment"]) == (fam, unit, tr)))

    labels = []
    for i, c in enumerate(rows):
        y = len(rows) - 1 - i
        good = c["excludes_zero"]
        col = "#1f6f9c" if good else "#c4650d"
        axL.plot(c["ci95"], [y, y], color=col, lw=1.25, solid_capstyle="butt",
                 alpha=1.0 if good else 0.95, zorder=2)
        axL.plot(c["point"], y, "o", ms=4.0, color=col, mew=0, zorder=3)
        lab = TREATMENT_LABEL[(c["family"], c["treatment"])]
        labels.append(f"{lab}, {UNIT_LABEL[c['unit']]}")
        if not good:
            axL.text(c["ci95"][1] + 0.02, y, "includes zero", fontsize=6.0, color="#c4650d",
                     va="center", ha="left")

    axL.axvline(0, color="#8a8a85", lw=0.6, zorder=1)
    axL.set_yticks(range(len(rows)))
    axL.set_yticklabels(labels[::-1])
    axL.tick_params(axis="y", length=0)
    axL.set_ylim(-0.6, len(rows) - 0.4)
    axL.set_xlim(-0.55, 1.30)
    axL.set_xlabel("Spread gap: interface quality $-$ fold identity")
    axL.set_title(
        f"a  The gap across {d['noise']['n_estimator_cells']} estimators, treatments and units",
        loc="left", fontweight="bold",
    )
    # Separator between the two estimator families, and their names.
    split = len(rows) - 8 - 0.5
    axL.axhline(split, color="#c9c9c4", lw=0.6, zorder=1)
    axL.text(-0.53, len(rows) - 0.55, FAMILY_TITLE["observed_cv"], fontsize=6.4,
             color="#6b6b66", va="top", ha="left", style="italic")
    axL.text(-0.53, split - 0.2, FAMILY_TITLE["spread_estimator"], fontsize=6.4,
             color="#6b6b66", va="top", ha="left", style="italic")

    # right panel: the two fold-identity rulers
    entries = [
        ("Categorical\n(cluster label)", quant["cv_observed"]["categorical"],
         quant["cv_ci"]["categorical"], "#1f6f9c"),
        ("Continuous\n(pairwise TM)", quant["cv_observed"]["continuous_pairwise"],
         quant["cv_ci"]["continuous_pairwise"], "#4c9f70"),
        ("Difference", quant["cv_difference"]["point"], quant["cv_difference"]["ci"], "#4a4a46"),
    ]
    for i, (lab, pt, ci, col) in enumerate(entries):
        y = len(entries) - 1 - i
        axR.plot(ci, [y, y], color=col, lw=1.25, solid_capstyle="butt", zorder=2)
        axR.plot(pt, y, "o", ms=4.4, color=col, mew=0, zorder=3)
    axR.axvline(0, color="#8a8a85", lw=0.6, zorder=1)
    axR.set_yticks(range(len(entries)))
    axR.set_yticklabels([e[0] for e in entries][::-1])
    axR.tick_params(axis="y", length=0)
    axR.set_ylim(-0.55, len(entries) - 0.45)
    axR.set_xlim(-0.12, 0.46)
    axR.set_xlabel("Spread of fold identity's settling step")
    axR.set_title("b  Two fold-identity rulers", loc="left", fontweight="bold")
    axR.text(
        quant["cv_difference"]["ci"][1] + 0.02, 0,
        f"$p$ = {quant['cv_difference']['p_two_sided_sign']:.3f}",
        fontsize=6.2, color="#4a4a46", va="center", ha="left",
    )

    save(fig, "fig3_robustness")


# ---------------------------------------------------------------------------
# Figure 4: the same crossing read on Complexa's own two denoising clocks
# ---------------------------------------------------------------------------
CLOCK_COLOR = {"bb_ca": "#1f6f9c", "local_latents": "#8e6bab"}
AXIS_LABEL = {
    "step_index": "Capture step",
    "time_bb_ca": "Backbone clock",
    "time_local_latents": "Latent clock",
    "time_native_assumed": "Assumed own clock",
}


def figure4(d: dict) -> None:
    s = d["schedule"]
    nsteps = s["nsteps"]
    caps = s["capture_steps"]

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(6.9, 2.55), gridspec_kw={"width_ratios": [1.0, 1.30], "wspace": 0.26}
    )

    # (a) the two schedules, with the capture steps marked on each
    grid = np.linspace(0, nsteps, len(s["clocks"]["clocks"]["bb_ca"]["t"]))
    axL.plot([0, nsteps], [0, 1], color="#8a8a85", lw=0.7, ls=(0, (3, 2)), zorder=1)
    axL.text(nsteps * 0.66, 0.70, "step $/$ 400", fontsize=6.0, color="#8a8a85",
             rotation=30, va="bottom", ha="center")
    for mode in ("bb_ca", "local_latents"):
        clk = s["clocks"]["clocks"][mode]
        axL.plot(grid, clk["t"], color=CLOCK_COLOR[mode], lw=1.5, zorder=3,
                 label=clk["label"])
        axL.plot(caps, s["time_at_capture_steps"][mode], "o", ms=3.0,
                 color=CLOCK_COLOR[mode], mew=0, zorder=4)
    axL.set_xlim(0, nsteps)
    axL.set_ylim(-0.02, 1.02)
    axL.set_xticks([0, 100, 200, 300, 400])
    axL.set_xlabel("Sampler step")
    axL.set_ylabel("Flow-matching time $t$")
    axL.legend(loc="lower right", frameon=False, handlelength=1.4, borderpad=0.1,
               bbox_to_anchor=(1.0, -0.02))
    axL.set_title("a  One step index, two denoising clocks", loc="left", fontweight="bold")

    # (b) the pre-registered gap on each axis, at both inference units
    order = [
        ("step_index", "cluster"), ("step_index", "target"),
        ("time_bb_ca", "cluster"), ("time_bb_ca", "target"),
        ("time_local_latents", "cluster"), ("time_local_latents", "target"),
        ("time_native_assumed", "cluster"), ("time_native_assumed", "target"),
    ]
    rows = [next(r for r in s["gap_rows"] if (r["axis"], r["unit"]) == k) for k in order]
    unit_label = {"cluster": "12 clusters", "target": "19 targets"}
    xlo, xhi = -0.12, 2.02
    labels = []
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        assumed = r["assumed"]
        col = "#8a8a85" if assumed else "#1f6f9c"
        axR.plot(r["ci95"], [y, y], color=col, lw=1.25, solid_capstyle="butt", zorder=2)
        axR.plot(r["gap"], y, "s" if assumed else "o", ms=3.4 if assumed else 4.0,
                 color=col, mew=0, zorder=3)
        labels.append(unit_label[r["unit"]])
        if i % 2 == 0:  # the axis name once per pair, above its two rows
            axR.text(xlo + 0.02, y + 0.42, AXIS_LABEL[r["axis"]], fontsize=6.6,
                     color="#6b6b66", va="center", ha="left", style="italic", zorder=4)
    axR.axvline(0, color="#8a8a85", lw=0.6, zorder=1)
    axR.set_yticks(range(len(rows)))
    axR.set_yticklabels(labels[::-1])
    axR.tick_params(axis="y", length=0)
    axR.set_ylim(-1.05, len(rows) - 0.05)
    axR.set_xlim(xlo, xhi)
    axR.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    axR.set_xlabel("Spread gap: interface quality $-$ fold identity")
    axR.set_title("b  The gap on each clock", loc="left", fontweight="bold")
    axR.axhline(1.72, color="#c9c9c4", lw=0.6, zorder=1)
    axR.text(xhi - 0.02, -0.62, "assumed per-quality loading: a sensitivity, not the headline",
             fontsize=6.0, color="#8a8a85", va="center", ha="right", style="italic")

    save(fig, "fig4_schedule")


# ---------------------------------------------------------------------------
# Appendix figure: what refining the capture grid moved
# ---------------------------------------------------------------------------
def figure_a3(d: dict) -> None:
    g = d["grid"]
    shifts = g["t_half_shift_summary"]

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(6.6, 2.05), gridspec_kw={"width_ratios": [1.0, 1.12], "wspace": 0.34}
    )

    comps = [("published_vs_seven", "Re-clustering\n(same 7 steps)"),
             ("seven_vs_ten", "Added capture steps\n(same cells)")]
    width = 0.34
    for k, (comp, _) in enumerate(comps):
        for i, q in enumerate(QUALITIES):
            y = len(QUALITIES) - 1 - i + (0.5 - k) * width
            v = shifts[q][comp]["mean_abs"]
            axL.barh(y, v, height=width * 0.86, color=COLOR[q],
                     alpha=1.0 if k else 0.45, edgecolor="white", lw=0.4, zorder=2)
            if v > 0:
                axL.text(v + 0.25, y, f"{v:.2f}", fontsize=6.0, va="center", ha="left",
                         color="#4a4a46")
    axL.set_yticks(range(len(QUALITIES)))
    axL.set_yticklabels([SHORT[q] for q in reversed(QUALITIES)])
    for lab, col in zip(axL.get_yticklabels(), [COLOR[q] for q in reversed(QUALITIES)]):
        lab.set_color(col)
    axL.set_ylim(-0.5, len(QUALITIES) - 0.4)
    axL.set_xlim(0, 7.6)
    axL.set_xlabel("Mean absolute shift in settling step")
    axL.set_title("a  What each change moved", loc="left", fontweight="bold")
    hL = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#4a4a46", alpha=a, edgecolor="white",
                      lw=0.4, label=lab)
        for a, (_, lab) in zip((0.45, 1.0), comps)
    ]
    axL.legend(handles=hL, loc="upper right", frameon=True, framealpha=1.0,
               facecolor="white", edgecolor="none", handlelength=1.1,
               handletextpad=0.4, labelspacing=0.35, borderpad=0.3,
               borderaxespad=0.25, fontsize=6.0)

    # right: the pre-registered gap under the three derivations
    pv = g["pre_registered_verdict"]
    derivs = [("published", "Published\n7 steps"), ("seven", "Re-derived\n7 steps"),
              ("ten", "Refined\n10 steps")]
    for j, (unit, marker, off) in enumerate((("cluster", "o", +0.15), ("target", "s", -0.15))):
        for i, (key, _) in enumerate(derivs):
            point, ci = pv[unit][f"gap_{key}"], pv[unit][f"ci95_{key}"]
            y = len(derivs) - 1 - i + off
            col = "#c4650d" if key == "ten" else "#6b6b66"
            axR.plot(ci, [y, y], color=col, lw=1.2, solid_capstyle="butt",
                     alpha=1.0 if j == 0 else 0.5, zorder=2)
            axR.plot(point, y, marker, ms=4.0, color=col,
                     mfc=col if j == 0 else "white", mew=0.9, zorder=3)
    axR.axvline(0, color="#8a8a85", lw=0.6, zorder=1)
    axR.set_yticks(range(len(derivs)))
    axR.set_yticklabels([lab for _, lab in derivs][::-1])
    axR.tick_params(axis="y", length=0)
    axR.set_ylim(-0.5, len(derivs) - 0.5)
    axR.set_xlim(-0.05, 1.0)
    axR.set_xlabel("Spread gap: interface quality $-$ fold identity")
    axR.set_title("b  The pre-registered gap under all three", loc="left", fontweight="bold")
    hR = [
        Line2D([], [], marker="o", ms=4.0, ls="", color="#4a4a46", label="12 clusters"),
        Line2D([], [], marker="s", ms=4.0, ls="", color="#4a4a46", mfc="white", mew=0.9,
               label="19 targets"),
    ]
    axR.legend(handles=hR, loc="lower right", frameon=False, handletextpad=0.5,
               labelspacing=0.3, borderaxespad=0.3)

    save(fig, "figA3_grid")


# ---------------------------------------------------------------------------
# Appendix figure: the consequence test, and where its width comes from
# ---------------------------------------------------------------------------
def figure_a2(d: dict) -> None:
    c = d["consequence"]
    a = d["action_grid"]
    fig, (axL, axM, axR) = plt.subplots(
        1, 3, figsize=(7.1, 2.15),
        gridspec_kw={"width_ratios": [1.28, 0.90, 0.80], "wspace": 0.95},
    )

    # -- a: adaptive minus best deployed, on both capture grids ---------------
    # The ten-step estimate is the result of record; the corrected seven-step
    # estimate is drawn beside it so the effect of adding steps is visible.
    order = ["interface_quality", "fold_identity", "structural_confidence", "designability"]
    primary = a["by_quality"]["interface_quality"]["ten"]
    axL.axvspan(primary["lo"], primary["hi"], color="#c9c9c4", alpha=0.32, lw=0, zorder=0)
    for i, q in enumerate(order):
        y = len(order) - 1 - i
        for v, dy, col, lw in (
            (a["by_quality"][q]["ten"], +0.17, COLOR[q], 1.4),
            (a["by_quality"][q]["seven"], -0.17, "#9a9a95", 1.0),
        ):
            axL.plot([v["lo"], v["hi"]], [y + dy, y + dy], color=col, lw=lw,
                     solid_capstyle="butt", zorder=2)
            axL.plot(v["mean"], y + dy, "o", ms=4.0 if dy > 0 else 3.0, color=col, zorder=3)
    axL.axvline(0, color="#8a8a85", lw=0.6, zorder=1)
    axL.set_yticks(range(len(order)))
    axL.set_yticklabels([SHORT[q] for q in order][::-1])
    axL.set_ylim(-0.55, len(order) - 0.42)
    axL.set_xlabel("Gain in $J$ over the best deployed action point")
    axL.set_title("a  Per-target action timing", loc="left", fontweight="bold")

    # -- b: the settling step is not the step worth acting at -----------------
    rows = a["settling_vs_peak_rows"]
    sv = a["settling_vs_peak"]
    axM.plot([0, 400], [0, 400], color="#8a8a85", lw=0.6, ls="--", zorder=1)
    axM.plot([r["t_half"] for r in rows], [r["objective_peak"] for r in rows], "o",
             ms=4.0, color=COLOR["interface_quality"], alpha=0.8, mew=0, zorder=3)
    axM.set_xlim(0, 400)
    axM.set_ylim(0, 400)
    axM.set_xticks([0, 100, 200, 300, 400])
    axM.set_yticks([0, 100, 200, 300, 400])
    axM.set_xlabel("Settling step")
    axM.set_ylabel("Step of maximum $J$")
    axM.set_title("b  Settling step vs best step", loc="left", fontweight="bold")
    axM.annotate(
        f"median offset {sv['median_abs_offset']:.4g} steps\n"
        f"Spearman {sv['spearman_by_cluster']:.4f}, "
        f"$p$ = {sv['spearman_permutation_p_two_sided']:.4f}",
        xy=(0.97, 0.04), xycoords="axes fraction", ha="right", va="bottom",
        fontsize=6.4, color="#4a4a46",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.0),
    )

    # -- c: the winner's curse, on the seven-step grid where it was run -------
    arms = [
        ("Scored roots", c["winners_curse"]["same_group"]),
        ("Other roots", c["winners_curse"]["held_out"]),
    ]
    for i, (lab, v) in enumerate(arms):
        y = len(arms) - 1 - i
        col = "#6b6b66" if i == 0 else "#c4650d"
        axR.plot([v["lo"], v["hi"]], [y, y], color=col, lw=1.3, solid_capstyle="butt", zorder=2)
        axR.plot(v["mean"], y, "o", ms=4.6, color=col, zorder=3)
    axR.axvline(0, color="#8a8a85", lw=0.6, zorder=1)
    axR.set_yticks(range(len(arms)))
    axR.set_yticklabels([lab for lab, _ in arms][::-1])
    axR.set_ylim(-0.5, len(arms) - 0.5)
    axR.set_xlabel("Gain in objective $J$")
    axR.set_title("c  Where the step was chosen", loc="left", fontweight="bold")

    save(fig, "figA2_consequence")


def main(argv: list[str] | None = None) -> None:
    global DATA, FIGDIR
    parser = argparse.ArgumentParser(description="Regenerate all manuscript figures")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=FIGDIR)
    args = parser.parse_args(argv)
    DATA = args.data.resolve()
    FIGDIR = args.out.resolve()
    with DATA.open() as fh:
        d = json.load(fh)
    print("drawing figures:")
    figure1_schedule(d)
    figure2_readout(d)
    figure_a4(d)
    figure2(d)
    figure3(d)
    figure4(d)
    figure_a1(d)
    figure_a2(d)
    figure_a3(d)

    # console check of what each figure asserts
    cl = d["spread_cluster_unit"]
    print("\nfig1 check — corrected CV, cluster unit:")
    for q in QUALITIES:
        m = cl["per_metric"][q]
        print(f"  {SHORT[q]:24s} {m['cv_corrected']:.4f} "
              f"[{m['cv_corrected_ci95'][0]:.4f}, {m['cv_corrected_ci95'][1]:.4f}] "
              f"median {m['median_t_half']:.2f} range {m['min_t_half']:.2f}-{m['max_t_half']:.2f}")
    n_cens = sum(v["censored"] for r in d["fig1_settling_steps"]["rows"] for v in r["values"].values())
    print(f"  censored points drawn as bounds: {n_cens}")
    print("\nfig2 check — peak R2 by family:")
    for q in QUALITIES:
        fam = d["fig2_readability"][q]["families"]
        print(f"  {SHORT[q]:24s} " + "  ".join(
            f"{FAMILY_LABEL[f]}={fam[f]['peak_mean']:.4f}@{fam[f]['peak_step']}" for f in fam))
    print("\nfig3 check — estimator cells:")
    for c in d["noise"]["estimator_cells"]:
        print(f"  {c['family']:17s} {c['unit']:10s} {c['treatment']:22s} "
              f"{c['point']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] "
              f"{'excludes 0' if c['excludes_zero'] else 'INCLUDES 0'}")
    q = d["quantisation"]
    print(f"  rulers: categorical {q['cv_observed']['categorical']:.4f} "
          f"{[round(x,4) for x in q['cv_ci']['categorical']]}, continuous "
          f"{q['cv_observed']['continuous_pairwise']:.4f} "
          f"{[round(x,4) for x in q['cv_ci']['continuous_pairwise']]}, difference "
          f"{q['cv_difference']['point']:+.5f} {[round(x,5) for x in q['cv_difference']['ci']]}")
    print("\nfig4 check — gap on each axis (both units):")
    for r in d["schedule"]["gap_rows"]:
        print(f"  {r['label']:34s} {r['gap']:+.4f} "
              f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]"
              f"{'   ASSUMED' if r['assumed'] else ''}")
    for mode in ("bb_ca", "local_latents"):
        tt = d["schedule"]["time_at_capture_steps"][mode]
        print(f"  t at capture steps, {mode:14s} " + " ".join(f"{v:.4g}" for v in tt))

    print("\nfigA3 check — mean absolute settling-step shift:")
    for q_ in QUALITIES:
        s = d["grid"]["t_half_shift_summary"][q_]
        print(f"  {SHORT[q_]:24s} re-clustering {s['published_vs_seven']['mean_abs']:.4f}"
              f"   added steps {s['seven_vs_ten']['mean_abs']:.4f}"
              f"   ratio to panel spread {s['grid_error_vs_panel_spread']['ratio']:.5f}")


if __name__ == "__main__":
    main()
