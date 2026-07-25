#!/usr/bin/env python3
"""
Build the intrinsic-grub paper as LaTeX and compile with tectonic.

  python3 -m lol_kills.research.grubs_intrinsic_pdf

Outputs (professional basename):
  output/pdf/void_grubs_scrap_value_and_contest_rationality.tex
  output/pdf/void_grubs_scrap_value_and_contest_rationality.pdf
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lol_kills.etl.paths import MODELS_DIR, ROOT
from lol_kills.research.grubs_intrinsic_value import (
    contest_certainty_atlas,
    contest_ev_terminal_states,
    delta_pp,
)

JSON_PATH = MODELS_DIR / "grubs_intrinsic_value.json"
RANKED_JSON_PATH = MODELS_DIR / "grubs_ranked_contest_proof.json"
OUT_DIR = ROOT / "output" / "pdf"
BASENAME = "void_grubs_scrap_value_and_contest_rationality"
TEX_PATH = OUT_DIR / f"{BASENAME}.tex"
PDF_PATH = OUT_DIR / f"{BASENAME}.pdf"
TMP = ROOT / "tmp" / "pdfs"


def _pp(x: float) -> str:
    """Always signed pp to two decimals (e.g. +1.75, -7.49)."""
    v = round(float(x), 2)
    return f"{v:+.2f}"


def _prob(x: float) -> str:
    """Probability / p* display (no forced +)."""
    return f"{round(float(x), 2):.2f}"


def _load() -> dict:
    return json.loads(JSON_PATH.read_text())


def _style_ax(ax) -> None:
    ax.set_facecolor("#F4F5F7")
    ax.tick_params(colors="#5C606A", labelsize=7.5)
    for spine in ax.spines.values():
        spine.set_color("#C8CAD0")


def _fig_bounds(d: dict, path: Path) -> None:
    b = d["intrinsic_bounds_pp_at_even"]
    c = d["contaminated_association_for_contrast"]
    intervals = d.get("sampling_intervals") or {}
    labels = [
        "90g cash\n(paid locally)",
        "Cash + 8s Touch\n(pre-26.11 ceiling)",
        "Cash + 8s Touch\n(26.11+ ceiling)",
        "Cash + 195 XP\n(team-total calibration)",
        "Ends 3-0\n(non-target association)",
    ]
    vals = [
        b["lower_gold_only_pp"],
        b["preferred_gold_plus_brief_burn_pp"],
        b["post_26_11_cash_plus_brief_pressure_ceiling_pp"],
        b["central_gold_plus_xp_joint_pp"],
        c["unique_dpp"],
    ]
    ci_keys = [
        "cash_90g",
        "cash_plus_pre26_11_brief_pressure_ceiling",
        "cash_plus_post26_11_brief_pressure_ceiling",
        "joint_cash_xp",
        None,
    ]
    colors = ["#3D4A5C"] * 4 + ["#6B3A32"]
    fig, ax = plt.subplots(figsize=(6.7, 3.25), dpi=200, facecolor="#F4F5F7")
    _style_ax(ax)
    y = np.arange(len(labels))[::-1]
    ax.hlines(y, 0, vals, color="#C8CAD0", lw=1.0, zorder=1)
    ax.scatter(vals, y, color=colors, s=58, zorder=4, edgecolors="#F4F5F7", linewidths=0.9)
    for yi, v, key in zip(y, vals, ci_keys):
        ci = intervals.get(key) if key else None
        if ci and ci.get("ci95_low_pp") is not None:
            lo, hi = float(ci["ci95_low_pp"]), float(ci["ci95_high_pp"])
            ax.hlines(yi, lo, hi, color="#3D4A5C", lw=2.6, zorder=3)
            ax.vlines([lo, hi], yi - 0.10, yi + 0.10, color="#3D4A5C", lw=1.1, zorder=3)
    for yi, v in zip(y, vals):
        ax.text(
            v + 0.10,
            yi,
            _pp(v),
            ha="left",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color="#1C1E24",
        )
    ax.axvline(0, color="#5C606A", lw=0.8, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color="#1C1E24", fontsize=8.2, fontfamily="sans-serif")
    ax.set_xlim(0, max(vals) + 0.85)
    ax.set_xlabel("Associational map win-rate probability difference (pp)", color="#1C1E24", fontsize=9)
    ax.text(
        0.01, -0.23,
        "Steel whiskers: conditional 95% Wald interval for the fitted association. Red point: excluded take-regime association.",
        transform=ax.transAxes, fontsize=7.1, color="#5C606A", ha="left", va="top",
    )
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    fig.tight_layout(pad=0.45)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _fig_edge(d: dict, path: Path) -> None:
    """Single-purpose refusal threshold chart with no space-hungry legend."""
    key = d.get("headline_key") or "preferred_farm_2kills_sym"
    block = d["contest_ev"].get(key) or d["contest_ev"]["preferred_farm_2kills_sym"]
    curve = block["curve"]
    xs = np.array([r["p_win_fight"] * 100 for r in curve], dtype=float)
    ys = np.array([r["edge_contest_minus_leave_pp"] for r in curve], dtype=float)
    pstar = float(block["breakeven_p_win_fight"]) * 100
    e25 = float(next(r["edge_contest_minus_leave_pp"] for r in curve if r["p_win_fight"] == 0.25))

    ink, mute, leave_c, contest_c = "#1C1E24", "#5C606A", "#8B4518", "#2F4A5C"
    fig, ax = plt.subplots(figsize=(7.0, 3.75), dpi=220, facecolor="#F4F5F7")
    _style_ax(ax)

    ax.fill_between(
        xs, ys, 0, where=(ys < 0), color=leave_c, alpha=0.15, interpolate=True,
        zorder=1,
    )
    ax.fill_between(
        xs, ys, 0, where=(ys >= 0), color=contest_c, alpha=0.13, interpolate=True,
        zorder=1,
    )
    ax.axhline(0, color=mute, lw=1.0, ls="-", alpha=0.65, zorder=2)
    ax.axvline(pstar, color=contest_c, lw=1.25, ls=":", zorder=2)

    ax.plot(
        xs, ys, color=ink, lw=2.1, marker="o", ms=4.5, zorder=4,
        markerfacecolor="#F4F5F7", markeredgecolor=ink, markeredgewidth=1.15,
    )
    ax.scatter(
        [25], [e25], color=leave_c, s=75, zorder=6,
        edgecolors="#F4F5F7", linewidths=1.3,
    )

    ymin, ymax = float(np.nanmin(ys)), float(np.nanmax(ys))
    pad = (ymax - ymin) * 0.10
    ax.set_ylim(ymin - pad * 1.15, ymax + pad * 1.25)
    ax.set_xlim(10, 80)
    ax.set_xticks([10, 20, 30, 40, 50, 60, 70, 80])

    # One threshold annotation, parked below the decision line.
    ax.annotate(
        f"break-even\n{pstar:.0f}%",
        xy=(pstar, 0),
        xytext=(pstar + 1.8, ymin + pad * 0.15),
        ha="left",
        va="bottom",
        fontsize=8,
        color=contest_c,
        fontfamily="sans-serif",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F4F5F7", edgecolor=contest_c, linewidth=0.85),
        arrowprops=dict(arrowstyle="->", color=contest_c, lw=0.85),
    )

    ax.set_xlabel("Chance you win the river fight (%)", color=ink, fontsize=9, fontfamily="sans-serif")
    ax.xaxis.set_label_coords(0.5, -0.16)
    ax.set_ylabel("Contest − leave  (map WR pp)", color=ink, fontsize=9, fontfamily="sans-serif")
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily("sans-serif")

    ax.set_title(
        "Reference scenario: contest minus leave",
        loc="left", pad=11, color=ink, fontsize=11, fontfamily="sans-serif", fontweight="bold",
    )
    fig.tight_layout(pad=0.7)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _fig_certainty_atlas(d: dict, path: Path) -> None:
    """Direct decision map across fight-win probability and outside-option states."""
    atlas = d["certainty_atlas"]
    leaves = atlas["leave_states"]
    panels = list(atlas["packages"].values())
    y_rows = np.arange(len(leaves), dtype=float)
    y_dense = np.linspace(y_rows.min(), y_rows.max(), 320)
    row_labels = [
        "Concede only",
        "One average wave",
        "Two average waves",
        "Two waves + one plate",
        "Three waves + one plate",
    ]
    ink, mute = "#1C1E24", "#5C606A"
    concede_fill, contest_fill = "#F3E9E6", "#E6EBEF"
    concede_ink, contest_ink = "#6B3A32", "#3D4A5C"

    fig, axes = plt.subplots(
        1, 2, figsize=(7.35, 3.82), dpi=220, sharex=True, sharey=True,
        facecolor="#FFFFFF",
    )
    for ax, panel in zip(axes, panels):
        _style_ax(ax)
        ax.set_facecolor("#FFFFFF")
        thresholds = np.asarray(
            [float(cell["breakeven_p_win_fight"]) * 100.0 for cell in panel["cells"]],
            dtype=float,
        )
        boundary = np.interp(y_dense, y_rows, thresholds)
        ax.fill_betweenx(
            y_dense, 0.0, boundary, color=concede_fill, edgecolor="none", zorder=0
        )
        ax.fill_betweenx(
            y_dense, boundary, 100.0, color=contest_fill, edgecolor="none", zorder=0
        )
        for y in y_rows:
            ax.axhline(y, color="#D7D9DE", lw=0.65, zorder=1)
        ax.plot(boundary, y_dense, color=ink, lw=1.8, zorder=3)
        ax.scatter(
            thresholds,
            y_rows,
            s=30,
            facecolor="#FFFFFF",
            edgecolor=ink,
            linewidth=1.1,
            zorder=4,
        )
        for threshold, y in zip(thresholds, y_rows):
            align_right = threshold > 76.0
            ax.annotate(
                f"{threshold:.1f}%",
                xy=(threshold, y),
                xytext=(-6 if align_right else 6, 0),
                textcoords="offset points",
                ha="right" if align_right else "left",
                va="center",
                fontsize=7.1,
                color=ink,
                fontfamily="sans-serif",
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.16",
                    facecolor="#FFFFFF",
                    edgecolor="none",
                    alpha=0.88,
                ),
                zorder=5,
            )
        ax.set_title(
            panel["label"], loc="left", pad=10, color=ink,
            fontsize=9.4, fontfamily="sans-serif", fontweight="bold",
        )
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.28, len(leaves) - 0.55)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.text(
            3.0,
            len(leaves) - 0.72,
            "CONCEDE WITHOUT FIGHTING",
            ha="left",
            va="top",
            fontsize=6.5,
            color=concede_ink,
            fontfamily="sans-serif",
            fontweight="bold",
        )
        ax.text(
            97.0,
            len(leaves) - 0.72,
            "CONTEST",
            ha="right",
            va="top",
            fontsize=6.8,
            color=contest_ink,
            fontfamily="sans-serif",
            fontweight="bold",
        )
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#C8CAD0")
        ax.spines["bottom"].set_color("#C8CAD0")
    axes[0].set_yticks(y_rows)
    axes[0].set_yticklabels(
        row_labels, color=mute, fontsize=7.2, fontfamily="sans-serif",
    )
    axes[0].set_ylabel(
        "Farm preserved by conceding",
        color=ink,
        fontsize=8.7,
        fontfamily="sans-serif",
    )
    for ax in axes:
        for lbl in ax.get_xticklabels():
            lbl.set_fontfamily("sans-serif")
    fig.text(
        0.535,
        0.105,
        "Estimated chance of winning the river fight, p (%)",
        ha="center",
        color=ink,
        fontsize=8.5,
        fontfamily="sans-serif",
    )
    fig.text(
        0.16,
        0.018,
        "Read each row horizontally: below the labelled p* → concede without fighting; above p* → contest.",
        color=mute,
        fontsize=7.2,
        fontfamily="sans-serif",
    )
    fig.subplots_adjust(left=0.17, right=0.985, bottom=0.22, top=0.88, wspace=0.10)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _fig_ledger(d: dict, path: Path) -> None:
    """Horizontal magnitude ledger — scrap vs leave farm vs deaths vs edge."""
    b = d["intrinsic_bounds_pp_at_even"]
    ls = d["ls_furia_scenario"]
    farm = float(ls.get("farm_pp") or d["opportunity_gold"]["packages"]["preferred_waves_plus_camp"]["wr_pp_at_even"])
    kill = abs(float(d["opportunity_gold"]["median_tf_assumption"]["lose_pp"]))
    edge = float(ls["at_p_025"]["edge_contest_minus_leave_pp"])
    farm_lab = "Leave farm\n(wiki waves±plate)"
    labels = [
        "T_pref scrap",
        farm_lab,
        "−2 kills",
        "Edge @ 25%",
    ]
    vals = [
        b["preferred_gold_plus_brief_burn_pp"],
        farm,
        -kill,
        edge,
    ]
    colors = ["#2F4A5C", "#8A8E96", "#8B4518", "#8B4518"]
    ink, mute = "#1C1E24", "#5C606A"

    fig, ax = plt.subplots(figsize=(6.4, 2.55), dpi=200, facecolor="#F4F5F7")
    _style_ax(ax)
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, color=colors, height=0.62, edgecolor="none")
    ax.axvline(0, color=mute, lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9, fontfamily="sans-serif", color=ink)
    ax.set_xlabel("Map win-rate effect (pp)", color=ink, fontsize=9, fontfamily="sans-serif")
    for yi, v in zip(y, vals):
        ax.text(
            v + (0.35 if v >= 0 else -0.35),
            yi,
            _pp(v),
            ha="left" if v >= 0 else "right",
            va="center",
            fontsize=9,
            color=ink,
            fontfamily="sans-serif",
            fontweight="bold",
        )
    lo, hi = min(vals), max(vals)
    span = hi - lo
    ax.set_xlim(lo - span * 0.22, hi + span * 0.28)
    for lbl in ax.get_xticklabels():
        lbl.set_fontfamily("sans-serif")
    fig.tight_layout(pad=0.4)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _ensure_outcome_matrix(d: dict) -> dict:
    if "river_outcome_matrix" in d:
        return d["river_outcome_matrix"]
    from lol_kills.research.grubs_intrinsic_value import river_outcome_matrix

    scrap = d["intrinsic_bounds_pp_at_even"]["preferred_gold_plus_brief_burn_pp"]
    farm = d["opportunity_gold"]["packages"]["preferred_waves_plus_camp"]["wr_pp_at_even"]
    kw = d["opportunity_gold"]["median_tf_assumption"]["win_pp"]
    kl = d["opportunity_gold"]["median_tf_assumption"]["lose_pp"]
    return river_outcome_matrix(scrap, farm, kw, kl)


def _fig_pd_matrix(d: dict, path: Path) -> None:
    """Decision-payoff board from terminal states, not an additive pp shortcut."""
    m = _ensure_outcome_matrix(d)
    mat = m["matrix"]
    vsl = m["vs_leave"]
    leave = float(m["outside_option_leave"]["pp"])

    cells = [
        [float(mat["lose_tf_no_grubs_pp"]), float(mat["lose_tf_but_you_get_grubs_pp"])],
        [float(mat["win_tf_no_grubs_pp"]), float(mat["win_tf_and_grubs_pp"])],
    ]
    tags = [
        ["usual collapse", "rare (secure then die)"],
        ["smite loss", "usual contest win"],
    ]
    ink, mute = "#1C1E24", "#5C606A"
    leave_c, win_c = "#8B4518", "#2F4A5C"

    fig = plt.figure(figsize=(6.5, 4.15), dpi=200, facecolor="#F4F5F7")
    # margins: left gutter for row labels, top for outside option + col headers
    ax = fig.add_axes([0.20, 0.14, 0.74, 0.62])
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("#F4F5F7")

    # Outside option banner (figure-level, above axes)
    fig.patches.append(
        plt.Rectangle(
            (0.20, 0.90), 0.74, 0.065,
            transform=fig.transFigure, facecolor="#E8EAEE",
            edgecolor="#C8CAD0", lw=0.9, clip_on=False,
        )
    )
    fig.text(
        0.57, 0.932,
        f"Outside option  ·  Leave river:  {_pp(leave)} pp",
        ha="center", va="center", fontsize=8.5, color=ink, fontfamily="sans-serif",
    )

    # Column headers
    fig.text(0.385, 0.855, "You do NOT get grubs", ha="center", va="bottom",
             fontsize=8.5, color=ink, fontfamily="sans-serif")
    fig.text(0.755, 0.855, "You GET grubs", ha="center", va="bottom",
             fontsize=8.5, color=ink, fontfamily="sans-serif")

    # Row labels (horizontal, left gutter — not rotated)
    fig.text(0.175, 0.14 + 0.62 * 0.75, "Lose\nriver TF", ha="right", va="center",
             fontsize=8.5, color=leave_c, fontfamily="sans-serif", fontweight="bold",
             linespacing=1.25)
    fig.text(0.175, 0.14 + 0.62 * 0.25, "Win\nriver TF", ha="right", va="center",
             fontsize=8.5, color=win_c, fontfamily="sans-serif", fontweight="bold",
             linespacing=1.25)

    for i in range(2):
        for j in range(2):
            x0, y0 = j, 1 - i
            v = cells[i][j]
            is_neg = v < 0
            face = "#F3E9E6" if is_neg else "#E6EBEF"
            edge = leave_c if is_neg else win_c
            ax.add_patch(
                plt.Rectangle(
                    (x0 + 0.04, y0 + 0.04), 0.92, 0.92,
                    facecolor=face, edgecolor=edge, lw=1.5, zorder=2,
                )
            )
            ax.text(
                x0 + 0.5, y0 + 0.58, _pp(v),
                ha="center", va="center", fontsize=18, color=edge,
                fontweight="bold", fontfamily="sans-serif", zorder=3,
            )
            ax.text(
                x0 + 0.5, y0 + 0.28, tags[i][j],
                ha="center", va="center", fontsize=7.5, color=mute,
                fontfamily="sans-serif", zorder=3,
            )

    # Footer: vs-leave deltas (kept in-figure so LaTeX caption can stay short)
    fig.text(
        0.57, 0.06,
        f"Vs leave:  win+grubs {_pp(vsl['win_tf_and_grubs_pp'])}   ·   "
        f"win / no grubs {_pp(vsl['win_tf_no_grubs_pp'])}   ·   "
        f"lose+gift {_pp(vsl['lose_tf_no_grubs_pp'])}",
        ha="center", va="center", fontsize=7.5, color=mute, fontfamily="sans-serif",
    )
    fig.text(
        0.57, 0.025,
        "Payoffs = terminal-state map WR pp vs even   ·   columns = your camp secure",
        ha="center", va="center", fontsize=7, color=mute, fontfamily="sans-serif",
    )
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _fig_resolved_payoffs(d: dict, path: Path) -> None:
    """Reference-branch leave-relative endpoints plus the expected-value line."""
    outcomes = _ensure_outcome_matrix(d)["vs_leave"]
    # Reference branch (s_W,s_D)=(1,0): only these two terminals enter Panel B.
    labels = [
        "Lose fight; opponent secures",
        "Win fight; own team secures",
    ]
    values = np.asarray([
        outcomes["lose_tf_no_grubs_pp"],
        outcomes["win_tf_and_grubs_pp"],
    ], dtype=float)
    loss_value = float(values[0])
    win_value = float(values[1])
    p_star = -loss_value / (win_value - loss_value)
    ink, muted, rule = "#1C1E24", "#5C606A", "#C8CAD0"
    steel, rust, paper = "#3D4A5C", "#6B3A32", "#F4F5F7"

    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(7.35, 3.35), dpi=230,
        gridspec_kw={"width_ratios": [1.02, 1.0], "wspace": 0.34},
        facecolor=paper,
    )
    for ax in (ax0, ax1):
        _style_ax(ax)
        ax.grid(False)

    y = np.arange(len(labels))[::-1]
    colors = [rust, steel]
    ax0.axvline(0, color=ink, lw=1.0, zorder=1)
    ax0.hlines(y, 0, values, color=rule, lw=1.1, zorder=1)
    for yi, value, color in zip(y, values, colors):
        ax0.scatter(
            [value], [yi], s=52, color=color if value > 0 else paper,
            edgecolor=color, linewidth=1.5, zorder=3,
        )
        # Labels above markers so they never collide with long y-tick text.
        ax0.text(
            value, yi + 0.28, _pp(value),
            ha="center", va="bottom",
            fontsize=8.0, color=color, fontweight="bold", fontfamily="sans-serif",
        )
    ax0.set_yticks(y)
    ax0.set_yticklabels(labels, fontsize=7.2, color=ink, fontfamily="sans-serif")
    ax0.tick_params(axis="y", length=0, pad=4)
    ax0.set_xlim(-16.6, 12.2)
    ax0.set_ylim(-0.72, len(labels) - 0.25)
    ax0.set_xticks([-15, -10, -5, 0, 5, 10])
    ax0.set_xlabel("Map-win difference versus conceding (pp)", fontsize=8.0, color=ink)
    ax0.set_title("A  Reference outcomes only", loc="left", fontsize=9.3, fontweight="bold", color=ink)
    ax0.text(
        0.5, -0.22,
        "concede without fighting = 0; other capture corners in Fig. 2 bands",
        transform=ax0.transAxes, ha="center", va="top",
        fontsize=6.4, color=muted, fontfamily="sans-serif",
    )
    ax0.spines["left"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax0.spines["top"].set_visible(False)

    p = np.linspace(0.0, 1.0, 201)
    edge = loss_value + (win_value - loss_value) * p
    ax1.axhline(0, color=ink, lw=1.0, zorder=1)
    ax1.plot(p * 100, edge, color=ink, lw=2.0, zorder=2)
    anchors = [(0.50, "50%", rust), (p_star, f"p* = {p_star*100:.1f}%", ink), (0.70, "70%", ink)]
    for probability, label, color in anchors:
        value = loss_value + (win_value - loss_value) * probability
        ax1.scatter(
            [probability * 100], [value], s=42,
            facecolor=paper if probability != p_star else color,
            edgecolor=color, linewidth=1.4, zorder=4,
        )
        # Keep both lines of each label fully off the diagonal.
        # 50%: below-center; p*: above-left; 70%: below-right.
        if probability == p_star:
            xytext, ha, va = (-16, 16), "right", "bottom"
        elif probability < p_star:
            xytext, ha, va = (0, -26), "center", "top"
        else:
            xytext, ha, va = (22, -24), "left", "top"
        ax1.annotate(
            f"{label}\n{_pp(value)} pp",
            xy=(probability * 100, value), xytext=xytext,
            textcoords="offset points", ha=ha, va=va,
            fontsize=7.2, color=color, fontweight="bold", fontfamily="sans-serif",
            clip_on=False,
        )
    ax1.set_xlim(0, 100)
    ax1.set_ylim(loss_value - 3.2, win_value + 3.8)
    ax1.set_xlabel("Pre-fight probability of winning the fight, p (%)", fontsize=8.0, color=ink)
    ax1.set_ylabel("Contest minus concede (map-WR pp)", fontsize=8.0, color=ink)
    ax1.set_title("B  Expected value before the fight", loc="left", fontsize=9.3, fontweight="bold", color=ink)
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    for ax in (ax0, ax1):
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontfamily("sans-serif")
    fig.subplots_adjust(left=0.22, right=0.985, bottom=0.28, top=0.90)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _fig_threshold_ladder(d: dict, path: Path) -> None:
    """Ordered opportunity-cost thresholds with capture corners as ranges."""
    packages = d["certainty_atlas"]["packages"]
    cash = packages["cash_only"]
    touch = packages["cash_plus_touch"]
    reference = touch["cells"]
    cash_reference = cash["cells"]
    branches = touch["capture_branches"]
    labels = [
        "No farm recovered (0g)",
        "One average wave (120.67g)",
        "Two average waves (241.33g)",
        "Two waves + one plate (361.33g)",
        "Three waves + one plate (482g)",
    ]
    ref = np.asarray([cell["breakeven_p_win_fight"] * 100 for cell in reference], dtype=float)
    cash_ref = np.asarray([cell["breakeven_p_win_fight"] * 100 for cell in cash_reference], dtype=float)
    corner_values = np.asarray([
        [branch["cells"][i]["breakeven_p_win_fight"] * 100 for branch in branches]
        for i in range(len(reference))
    ], dtype=float)
    lo = corner_values.min(axis=1)
    hi = corner_values.max(axis=1)
    ink, muted, rule = "#1C1E24", "#5C606A", "#C8CAD0"
    steel, rust, paper = "#3D4A5C", "#6B3A32", "#F4F5F7"
    fig, ax = plt.subplots(figsize=(7.2, 3.25), dpi=230, facecolor=paper)
    _style_ax(ax)
    y = np.arange(len(labels))[::-1]
    for yi, low, high, cash_value, ref_value in zip(y, lo, hi, cash_ref, ref):
        ax.hlines(yi, low, high, color=rule, lw=5.0, zorder=1)
        ax.vlines([low, high], yi - 0.10, yi + 0.10, color=muted, lw=1.0, zorder=2)
        ax.scatter([cash_value], [yi], s=35, facecolor=paper, edgecolor=muted, linewidth=1.2, zorder=3)
        ax.scatter([ref_value], [yi], s=50, facecolor=steel, edgecolor=paper, linewidth=0.9, zorder=4)
        # Place the main label past the interval so it cannot overlap bound digits.
        ax.text(
            high + 1.4, yi, f"{ref_value:.1f}%",
            ha="left", va="center", fontsize=8.0, color=ink,
            fontweight="bold", fontfamily="sans-serif",
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.7, color=ink, fontfamily="sans-serif")
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(28, 98)
    ax.set_xticks([30, 40, 50, 60, 70, 80, 90])
    ax.set_xlabel("Required pre-fight probability of winning, p* (%)", fontsize=8.3, color=ink)
    ax.set_title("Leave-farm opportunity cost sets the required fight-win probability", loc="left", fontsize=10.0, fontweight="bold", color=ink, pad=13)
    ax.text(
        0.0, 1.03,
        "Filled: reference secure-if-win   Hollow: cash only   Band: capture-corner range",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=6.8, color=muted, fontfamily="sans-serif",
    )
    ax.grid(axis="x", color=rule, linewidth=0.55, alpha=0.8)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    for tick in ax.get_xticklabels():
        tick.set_fontfamily("sans-serif")
    fig.subplots_adjust(left=0.29, right=0.985, bottom=0.18, top=0.82)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _fig_probability_vs_hurdle(fight: dict, path: Path) -> None:
    """Compare the ranked pilot forecast with the professional decision hurdle."""
    grid = fight["reference_decision_comparison"]["grid"]
    x = np.asarray([row["gold_lead"] for row in grid], dtype=float)
    p_hat = np.asarray([row["p_hat_fight_win"] * 100 for row in grid], dtype=float)
    p_star = np.asarray([row["p_star_reference"] * 100 for row in grid], dtype=float)
    ink, muted, rule = "#1C1E24", "#5C606A", "#C8CAD0"
    steel, rust, paper = "#3D4A5C", "#6B3A32", "#F4F5F7"
    fig, ax = plt.subplots(figsize=(7.1, 3.2), dpi=230, facecolor=paper)
    _style_ax(ax)
    ax.fill_between(x, p_hat, p_star, color="#E3E5E9", zorder=1)
    ax.plot(x, p_star, color=rust, lw=2.0, zorder=3)
    ax.plot(x, p_hat, color=steel, lw=2.0, zorder=4)
    ax.axvline(0, color=rule, lw=0.9, zorder=2)
    parity = int(np.argmin(np.abs(x)))
    ax.scatter([0, 0], [p_hat[parity], p_star[parity]], s=44, color=[steel, rust], edgecolor=paper, linewidth=0.9, zorder=5)
    ax.text(100, p_star[parity] + 0.7, f"required p*  {p_star[parity]:.1f}%", color=rust, fontsize=8.0, fontweight="bold", va="bottom")
    ax.text(100, p_hat[parity] - 0.7, f"ranked pilot p-hat  {p_hat[parity]:.1f}%", color=steel, fontsize=8.0, fontweight="bold", va="top")
    ax.text(
        -1900, 68.2,
        "Shaded gap: forecast does not clear the reference hurdle",
        ha="left", va="top", fontsize=7.0, color=muted, fontfamily="sans-serif",
    )
    ax.set_xlim(-2000, 2000)
    ax.set_ylim(34, 70)
    ax.set_xticks([-2000, -1000, 0, 1000, 2000])
    ax.set_xticklabels(["-2,000", "-1,000", "Parity", "+1,000", "+2,000"])
    ax.set_ylabel("Fight-win probability (%)", fontsize=8.3, color=ink)
    ax.set_xlabel("Focal team's pre-fight gold difference", fontsize=8.3, color=ink)
    ax.set_title("A weak gold-only forecast remains below the reference contest threshold", loc="left", fontsize=9.8, fontweight="bold", color=ink, pad=10)
    ax.grid(axis="y", color=rule, linewidth=0.55)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontfamily("sans-serif")
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.20, top=0.86)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _fig_fight_probability_pilot(fight: dict, path: Path) -> None:
    """Show label sensitivity and the narrow conditional probability forecast."""
    sensitivity = fight["descriptive"]["radius_sensitivity"]
    grid = fight["reference_decision_comparison"]["grid"]
    validation = fight["primary_model"]["validation"]
    x = np.asarray([row["gold_lead"] for row in grid], dtype=float)
    p_hat = np.asarray([row["p_hat_fight_win"] * 100 for row in grid], dtype=float)
    ink, muted, rule = "#1C1E24", "#5C606A", "#C8CAD0"
    steel, rust, paper = "#3D4A5C", "#6B3A32", "#F4F5F7"
    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(7.35, 3.25), dpi=230,
        gridspec_kw={"width_ratios": [0.88, 1.12], "wspace": 0.32},
        facecolor=paper,
    )
    for ax in (ax0, ax1):
        _style_ax(ax)
        ax.grid(False)

    radii = np.arange(len(sensitivity))
    local = np.asarray([row["episodes_with_local_kill"] for row in sensitivity])
    decisive = np.asarray([row["decisive_local_exchanges"] for row in sensitivity])
    width = 0.33
    ax0.bar(radii - width / 2, local, width=width, color="#AEB3BB", edgecolor="none", label="any local kill")
    ax0.bar(radii + width / 2, decisive, width=width, color=steel, edgecolor="none", label="decisive exchange")
    for xi, value in zip(radii - width / 2, local):
        ax0.text(xi, value + 11, f"{value}", ha="center", va="bottom", fontsize=7.2, color=muted, fontweight="bold")
    for xi, value in zip(radii + width / 2, decisive):
        ax0.text(xi, value + 11, f"{value}", ha="center", va="bottom", fontsize=7.2, color=ink, fontweight="bold")
    ax0.set_xticks(radii)
    ax0.set_xticklabels([f"{row['radius']:,}" for row in sensitivity])
    ax0.set_ylim(0, max(local) * 1.22)
    ax0.set_ylabel("Episodes", fontsize=8.0, color=ink)
    ax0.set_xlabel("Pit radius (map units)", fontsize=8.0, color=ink)
    ax0.set_title("A  The label depends on radius", loc="left", fontsize=9.2, fontweight="bold", color=ink)
    leg = ax0.legend(
        loc="upper left", frameon=False, fontsize=6.8, handlelength=1.0, handletextpad=0.4,
        labelcolor=muted,
    )
    for text in leg.get_texts():
        text.set_fontfamily("sans-serif")
    ax0.spines["right"].set_visible(False)
    ax0.spines["top"].set_visible(False)

    ax1.axhline(50, color=rule, lw=0.9)
    ax1.plot(x, p_hat, color=steel, lw=2.1)
    for gold in (-1000, 0, 1000):
        row = next(item for item in grid if int(item["gold_lead"]) == gold)
        value = row["p_hat_fight_win"] * 100
        ax1.scatter([gold], [value], s=42, color=steel, edgecolor=paper, linewidth=0.9, zorder=4)
        label = "parity" if gold == 0 else f"{gold:+,}g"
        ax1.annotate(
            f"{label}\n{value:.1f}%", xy=(gold, value),
            xytext=(0, 9 if gold >= 0 else -11), textcoords="offset points",
            ha="center", va="bottom" if gold >= 0 else "top",
            fontsize=7.1, color=ink, fontweight="bold",
        )
    ax1.set_xlim(-2000, 2000)
    ax1.set_ylim(37, 63)
    ax1.set_xticks([-2000, -1000, 0, 1000, 2000])
    ax1.set_xticklabels(["-2,000", "-1,000", "0", "+1,000", "+2,000"])
    ax1.set_ylabel("Conditional fight-win forecast (%)", fontsize=8.0, color=ink)
    ax1.set_xlabel("Pre-fight gold difference", fontsize=8.0, color=ink)
    ax1.set_title("B  p-hat only among decisive exchanges", loc="left", fontsize=9.2, fontweight="bold", color=ink)
    ax1.text(
        0.02, 0.95,
        f"Grouped 10-fold AUC {validation['auc']:.3f}   Brier {validation['brier']:.3f} (null {validation['null_brier']:.3f})",
        transform=ax1.transAxes, ha="left", va="top", fontsize=6.7, color=muted,
    )
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    for ax in (ax0, ax1):
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontfamily("sans-serif")
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.20, top=0.88)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _tex_escape(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def _item_pace_tex(d: dict) -> str:
    """Short LaTeX block for item-completion / tempo gap."""
    ip = d.get("item_pace") or {}
    if not ip:
        return (
            "Item-pace layer unavailable --- re-enrich "
            "\\texttt{grubs\\_intrinsic\\_value.json}."
        )
    gift = ip["gift_path_no_tf"]["relative_leave_minus_take"]
    fight = ip["fight_path_both_contest"]["relative_win_minus_lose"]
    cst = ip["constants"]
    rows = []
    for h in ip.get("horizons_after_window_min") or []:
        rows.append(
            f"{h['horizon_min']:.0f} & {h['gold']:.0f} & "
            f"{h['frac_modal_pickaxe_875']:.2f} & "
            f"{h['minutes_solo_laner_gpm']:.2f} & "
            f"{_pp(h['wr_pp_at_even'])} \\\\"
        )
    horizon_tex = "\n".join(rows)
    return (
        f"Modal next component = Pickaxe ${cst['modal_next_component_g']:.0f}$g "
        f"(basket median ${cst['median_next_component_g']:.0f}$g); "
        f"early laner GPM $\\approx {cst['laner_gpm_early']:.1f}$g/min "
        f"(wiki wave every ${cst['wave_period_s']:.0f}$s). "
        f"Gift path relative leave$-$take $\\approx {gift['gold']:.0f}$g "
        f"({gift['frac_modal_pickaxe_875']:.2f}$\\times$ Pickaxe; "
        f"{gift['minutes_solo_laner_gpm']:.1f} solo-laner minutes; "
        f"{_pp(gift['wr_pp_at_even'])}\\,pp) --- own farm plus opponent missed waves "
        f"minus scrap. Fight win$-$lose item gap $\\approx {fight['gold']:.0f}$g "
        f"({fight['frac_modal_pickaxe_875']:.2f}$\\times$ Pickaxe) when both contest "
        f"(CS cancels). Dual-tempo leave EV is sensitivity only.\n"
        r"\begin{center}" "\n"
        r"\small" "\n"
        r"\begin{tabular}{rrrrr}" "\n"
        r"\toprule" "\n"
        r"Horizon (min) & Rel.\ gold (g) & $\times$ Pickaxe & Solo-min & WR pp \\" "\n"
        r"\midrule" "\n"
        f"{horizon_tex}\n"
        r"\bottomrule" "\n"
        r"\end{tabular}\\[0.25em]" "\n"
        r"{\footnotesize\textit{Table 2b. Leave$-$take gold gap after the window, "
        r"while the river side is delayed ($\sim$1.25\,min). "
        r"Headline leave EV still counts own farm only.}}" "\n"
        r"\end{center}"
    )


def write_tex(d: dict, fight: dict, fig1: Path, fig2: Path, fig3: Path, fig4: Path) -> str:
    if not RANKED_JSON_PATH.exists():
        raise FileNotFoundError(
            f"Diamond+ control artifact is required: {RANKED_JSON_PATH}"
        )
    ranked = json.loads(RANKED_JSON_PATH.read_text())
    sq = ranked["gold10_calibration"]
    sq_d = ranked["by_rank_bucket"]["diamond"]
    sq_m = ranked["by_rank_bucket"]["masters_plus"]
    sq_d_cal = sq_d["gold10_calibration"]
    sq_m_cal = sq_m["gold10_calibration"]

    b = d["intrinsic_bounds_pp_at_even"]
    g = d["component_wr_pp"]["gold_only"]
    c = d["contaminated_association_for_contrast"]
    ls = d["ls_furia_scenario"]
    opp = d["opportunity_gold"]
    leave_farm = d.get("leave_farm") or opp.get("leave_farm") or {}
    head_key = d.get("headline_key") or "preferred_farm_2kills_sym"
    head = d["contest_ev"].get(head_key) or d["contest_ev"]["preferred_farm_2kills_sym"]
    mix = (
        d["contest_ev"].get("headline_leave_farm_2kills_mixture")
        or d["contest_ev"].get("headline_nofarm_2kills_mixture")
        or {}
    )
    base = d["contest_ev"].get("baseline_2deaths_gift_no_farm") or d["contest_ev"]["baseline_gift_no_farm_no_kills"]
    burn = d["burn"]
    logits = d["logits"]["gold10"]
    joint_logits = d["logits"]["joint_gold_xp"]
    diagnostics = d["model_diagnostics"]["gold10_10fold"]
    farm_pp = float(ls.get("farm_pp", head.get("farm_pp", 0.0)))
    om = _ensure_outcome_matrix(d)
    mat = om["matrix"]
    vsl = om["vs_leave"]
    leave_pp = om["outside_option_leave"]["pp"]
    recon = d.get("sister_study_reconciliation", {})

    def rows_curve(curve):
        keep = {0.15, 0.25, 0.50, 0.60, 0.70, 0.75}
        lines = []
        for r in curve:
            if r["p_win_fight"] not in keep:
                continue
            verdict = "Leave" if r["edge_contest_minus_leave_pp"] < 0 else "Contest"
            win_pct = f"{r['p_win_fight']*100:.0f}\\%"
            lines.append(
                f"{win_pct} & {_pp(r['ev_contest_pp'])} & {_pp(r['edge_contest_minus_leave_pp'])} & {verdict} \\\\"
            )
        return "\n".join(lines)

    # Wiki leave-farm scenarios (1/2/3 laners ± plate)
    scen = leave_farm.get("scenarios") or {}
    show_keys = [
        "one_laner_one_wave",
        "two_laners_one_wave",
        "three_laners_one_wave",
        "two_laners_wave_plate_p25",
        "three_laners_wave_plate_p25",
        "two_laners_wave_plate_p50",
        "two_laners_nocannon_wave",
    ]
    farm_row_bits = []
    for k in show_keys:
        if k not in scen:
            continue
        v = scen[k]
        mark = r" \textbf{(headline)}" if k == leave_farm.get("headline_key") else ""
        farm_row_bits.append(
            f"{_tex_escape(v['label'])}{mark} & {v['gold']:.1f} & {_pp(v['wr_pp_at_even'])} \\\\"
        )
    farm_rows = "\n".join(farm_row_bits)
    kill_rows = "\n".join(
        f"{r['net_kills_for_contester']:+d} & {r['gold']:.0f} & {_pp(r['wr_pp_at_even'])} \\\\"
        for r in opp["kill_net_table"]
        if r["net_kills_for_contester"] in (-2, 0, 2)
    )

    def _sens_line(label: str, sc: dict) -> str:
        e25 = next(x["edge_contest_minus_leave_pp"] for x in sc["curve"] if x["p_win_fight"] == 0.25)
        pstar = sc["breakeven_p_win_fight"]
        if pstar is None:
            pstar_s = "n/a"
        elif float(pstar) > 1.0:
            pstar_s = r"$>$100\%"
        else:
            pstar_s = f"{float(pstar)*100:.0f}\\%"
        return (
            f"{label} & {_pp(sc['delta_obj_pp'])} & {_pp(sc['farm_pp'])} & "
            f"{_pp(sc['fight_win_extra_pp'])} & {_pp(sc['fight_loss_pp'])} & "
            f"{pstar_s} & {_pp(e25)} \\\\"
        )

    sens_lines = [
        _sens_line(r"Ignore farm and kills", base),
        _sens_line(r"1 laner $\times$ wiki wave, $\pm 2$ kills", d["contest_ev"]["leave_one_laner_one_wave_2kills"]),
        _sens_line(r"2 laners $\times$ wiki wave, $\pm 2$ kills", d["contest_ev"]["leave_two_laners_one_wave_2kills"]),
        _sens_line(r"3 laners $\times$ wiki wave, $\pm 2$ kills", d["contest_ev"]["leave_three_laners_one_wave_2kills"]),
        _sens_line(r"Headline: 2 laners + 25\% plate", head),
    ]
    if mix:
        sens_lines.append(_sens_line(r"Headline + $2\times 2$ mixture", mix))
    sens_lines.append(
        _sens_line(r"Contrast: zero leave-farm", d["contest_ev"]["contrast_nofarm_2kills"])
    )
    dual = d["contest_ev"].get("leave_dual_tempo_own_plus_opp_miss_2kills")
    if dual:
        sens_lines.append(
            _sens_line(r"Dual tempo: own farm + opp miss", dual)
        )
    sens_tex = "\n".join(sens_lines)

    miss = d["sample"].get("missingness_note") or (
        f"Complete-case gold@10 fit $n={logits['n']:,}$ "
        f"(era $n={d['sample']['n_era_3camp']:,}$; $|$golddiff$|\\le 3000$)."
    )
    miss_tex = _tex_escape(miss).replace("n_gold=", "$n_{\\mathrm{gold}}=$").replace(
        "n_xp=", "$n_{\\mathrm{xp}}=$"
    ).replace("n_joint=", "$n_{\\mathrm{joint}}=$")

    if recon.get("available"):
        sister = (
            f"OE take-regime contest EV ({_tex_escape(str(recon.get('source')))}): "
            f"at $p=0.25$ edge vs leave\\_mix $\\approx {_pp(recon.get('at_p_025_edge_vs_leave_mix_pp', 0))}$\\,pp "
            f"({_tex_escape(str(recon.get('at_p_025_verdict_vs_leave')))}); "
            f"$p^{{\\star}}_{{\\mathrm{{leave\\_mix}}}}\\approx {_prob(recon.get('breakeven_p_vs_leave_mix', 0))}$. "
            f"{_tex_escape(str(recon.get('note', '')))}"
        )
    else:
        sister = "Sister OE contest numbers unavailable."

    mix25 = ls.get("mixture_at_p_025") or (
        next(x for x in mix["curve"] if x["p_win_fight"] == 0.25) if mix else None
    )
    farm25 = ls.get("at_p_025")  # headline IS farm now
    nofarm25 = ls.get("nofarm_contrast_at_p_025")
    mix_be = ls.get("mixture_breakeven_p", mix.get("breakeven_p_win_fight") if mix else None)

    wiki_note = (
        leave_farm.get("constants", {}).get("note")
        or "Grub-era E[wave] 120.67g at 10:00; outer plate 120g local."
    )

    intervals = d.get("sampling_intervals") or {}
    cash_ci = intervals.get("cash_90g") or {}
    pref_ci = intervals.get("cash_plus_pre26_11_brief_pressure_ceiling") or {}
    pref_post_ci = intervals.get("cash_plus_post26_11_brief_pressure_ceiling") or {}
    joint_ci = intervals.get("joint_cash_xp") or {}

    def ci_value(ci: dict, key: str) -> str:
        value = ci.get(key)
        return _pp(value) if value is not None else "n/a"

    def scenario_row(label: str, assumption: str, scenario: dict) -> str:
        row25 = next(
            r for r in scenario["curve"] if abs(float(r["p_win_fight"]) - 0.25) < 1e-9
        )
        pstar = scenario.get("breakeven_p_win_fight")
        pstar_s = "n/a" if pstar is None else f"{float(pstar) * 100:.0f}\\%"
        return (
            f"{label} & {assumption} & {pstar_s} & "
            f"{_pp(row25['edge_contest_minus_leave_pp'])}\\,pp \\\\"
        )

    scenario_rows = [
        scenario_row(
            "No-farm contrast",
            "No counterfactual farm credited to leave; $\\pm2$ kill prior.",
            d["contest_ev"]["contrast_nofarm_2kills"],
        ),
        scenario_row(
            "One-lane leave",
            "One average early wave; $\\pm2$ kill prior.",
            d["contest_ev"]["leave_one_laner_one_wave_2kills"],
        ),
        scenario_row(
            "Reference scenario",
            "Two average early waves plus 25\\% plate conversion; $\\pm2$ kill prior.",
            head,
        ),
    ]
    dual = d["contest_ev"].get("leave_dual_tempo_own_plus_opp_miss_2kills")
    if dual:
        scenario_rows.append(
            scenario_row(
                "Dual-tempo",
                "Reference farm plus opponent missed waves; $\\pm2$ kill prior.",
                dual,
            )
        )
    scenario_tex = "\n".join(scenario_rows)

    atlas = d["certainty_atlas"]

    def atlas_pct(value) -> str:
        if value is None:
            return "n/a"
        value = float(value)
        return r"$>$100\%" if value > 1.0 else f"{value * 100:.1f}\\%"

    atlas_packages = atlas["packages"]
    cash_cells = atlas_packages["cash_only"]["cells"]
    touch_cells = atlas_packages["cash_plus_touch"]["cells"]
    touch_branches = {
        branch["key"]: branch
        for branch in atlas_packages["cash_plus_touch"]["capture_branches"]
    }
    certainty_rows = "\n".join(
        (
            f"{_tex_escape(ref['label'])} & "
            f"{atlas_pct(ref['breakeven_p_win_fight'])} & "
            f"{atlas_pct(always['breakeven_p_win_fight'])} & "
            f"{atlas_pct(never['breakeven_p_win_fight'])} & "
            f"{atlas_pct(inverse['breakeven_p_win_fight'])} \\\\"
        )
        for ref, always, never, inverse in zip(
            touch_branches["secure_if_win"]["cells"],
            touch_branches["always_secure"]["cells"],
            touch_branches["never_secure"]["cells"],
            touch_branches["secure_if_lose"]["cells"],
        )
    )
    certainty_values = [
        float(cell["breakeven_p_win_fight"])
        for package in atlas_packages.values()
        for branch in package["capture_branches"]
        for cell in branch["cells"]
        if cell["breakeven_p_win_fight"] is not None
    ]
    certainty_min = min(certainty_values) * 100.0
    certainty_max = max(certainty_values) * 100.0
    reference_values = [
        float(cell["breakeven_p_win_fight"])
        for package in atlas_packages.values()
        for cell in package["cells"]
        if cell["breakeven_p_win_fight"] is not None
    ]
    reference_min = min(reference_values) * 100.0
    reference_max = max(reference_values) * 100.0
    central_pstar_ci = touch_cells[2]["map_level_sampling_interval"]
    fight_model = fight["primary_model"]
    fight_validation = fight_model["validation"]
    fight_presence_validation = fight["candidate_models"]["gold_plus_presence"]
    fight_grid = {
        int(row["gold_lead"]): row
        for row in fight["reference_decision_comparison"]["grid"]
    }
    fight_gold_rows = "\n".join(
        (
            f"{('Parity' if gold == 0 else f'{gold:+,}g')} & "
            f"{fight_grid[gold]['p_hat_fight_win'] * 100:.1f}\\% \\\\"
        )
        for gold in (-1000, 0, 1000)
    )
    camp_ownership_pp = float(vsl["win_tf_and_grubs_pp"]) - float(vsl["win_tf_no_grubs_pp"])
    fight_result_pp = float(vsl["win_tf_and_grubs_pp"]) - float(vsl["lose_tf_but_you_get_grubs_pp"])
    reference_loss = float(vsl["lose_tf_no_grubs_pp"])
    reference_win = float(vsl["win_tf_and_grubs_pp"])
    reference_slope = reference_win - reference_loss

    # Pre-contest deficits for the reference branch: cash plus brief Touch,
    # two average waves preserved by conceding, secure if the fight is won.
    # These are structural thresholds, not observed fight-win frequencies.
    deficit_rows = []
    for baseline_gold in (0.0, -500.0, -1000.0, -2000.0):
        _, pstar, _ = contest_ev_terminal_states(
            float(logits["intercept"]),
            float(logits["coef"]),
            baseline_gold=baseline_gold,
            objective_gold=float(atlas_packages["cash_plus_touch"]["objective_gold"]),
            leave_farm_gold=float(touch_cells[2]["leave_gold"]),
            win_kill_gold=600.0,
            loss_kill_gold=-600.0,
            p_secure_if_win=1.0,
            p_secure_if_lose=0.0,
        )
        if pstar is None:
            raise RuntimeError("Undefined deficit threshold in reference branch")
        label = "Parity" if baseline_gold == 0 else f"{baseline_gold:,.0f}g"
        deficit_rows.append(
            f"{label} & {pstar * 100:.1f}\\% & {(1.0 - pstar) * 100:.1f}\\% \\\\"
        )
    deficit_rows_tex = "\n".join(deficit_rows)

    siege = d["mechanical_package"]["worked_siege_example"]
    siege_rows_tex = "\n".join(
        (
            f"{int(row['stacks'])} & {int(row['tick_damage'])} & "
            f"{float(row['maintained_true_dps']):.0f} & "
            f"{float(row['destruction_time_s']):.2f} & "
            f"{int(row['attacks'])} & "
            f"{float(row['touch_true_damage']):.0f} & "
            f"{float(row['zaahen_without_touch_damage']):.0f} & "
            f"{float(row['touch_vs_zaahen_without_touch_pct']):.1f}\\% & "
            f"{float(row['time_saved_s']):.2f} \\\\"
        )
        for row in siege["rows"]
    )

    pro_cash_ci = d["sampling_intervals"]["cash_90g"]
    sq_cash_ci = sq["cash_90g_wald_95_ci"]
    sq_90g_neutral = float(delta_pp(sq["intercept"], sq["coef_per_gold"], 0.0, 90.0))
    pro_minus_sq = float(pro_cash_ci["estimate_pp"] - sq_90g_neutral)
    pro_minus_sq_se = math.sqrt(
        float(pro_cash_ci["se_pp"]) ** 2 + float(sq_cash_ci["se_pp"]) ** 2
    )
    pro_minus_sq_lo = pro_minus_sq - 1.96 * pro_minus_sq_se
    pro_minus_sq_hi = pro_minus_sq + 1.96 * pro_minus_sq_se
    pro_minus_sq_z = pro_minus_sq / pro_minus_sq_se
    pro_minus_sq_p = math.erfc(abs(pro_minus_sq_z) / math.sqrt(2.0))

    def calibration_cell(cal: dict) -> str:
        ci = cal["cash_90g_wald_95_ci"]
        neutral_estimate = delta_pp(cal["intercept"], cal["coef_per_gold"], 0.0, 90.0)
        return (
            f"{_pp(neutral_estimate)} "
            f"$[{_pp(ci['ci95_low_pp'])},{_pp(ci['ci95_high_pp'])}]$"
        )

    control_population_rows = "\n".join([
        (
            f"Competitive & {d['sample']['n_era_3camp']:,} & "
            f"{d['sample']['n_fit_gold10']:,} & "
            f"{_pp(pro_cash_ci['estimate_pp'])} "
            f"$[{_pp(pro_cash_ci['ci95_low_pp'])},{_pp(pro_cash_ci['ci95_high_pp'])}]$ & "
            f"{d['model_diagnostics']['gold10_10fold']['auc']:.3f} \\\\"
        ),
        (
            f"Diamond & {sq_d['n_matches_with_horde']:,} & "
            f"{sq_d_cal['n_fit']:,} & "
            f"{calibration_cell(sq_d_cal)} & "
            f"{sq_d_cal['diagnostics_10fold']['auc']:.3f} \\\\"
        ),
        (
            f"Master+ & {sq_m['n_matches_with_horde']:,} & "
            f"{sq_m_cal['n_fit']:,} & "
            f"{calibration_cell(sq_m_cal)} & "
            f"{sq_m_cal['diagnostics_10fold']['auc']:.3f} \\\\"
        ),
        (
            f"Equal-quota pooled anchors & {ranked['n_matches_with_horde']:,} & "
            f"{sq['n_fit']:,} & "
            f"{calibration_cell(sq)} & "
            f"{sq['diagnostics_10fold']['auc']:.3f} \\\\"
        ),
    ])

    pro_cash_cells = d["certainty_atlas"]["packages"]["cash_only"]["cells"]
    sq_neutral_atlas = contest_certainty_atlas(
        float(sq["intercept"]),
        float(sq["coef_per_gold"]),
        touch_gold=(192.0 / 900.0) * 120.0,
    )
    sq_cash_cells = sq_neutral_atlas["packages"]["cash_only"]["cells"]

    def signed_tenth_pp(value: float) -> str:
        value = float(value)
        if abs(value) < 0.05:
            value = 0.0
        return f"{value:+.1f} pp"

    pstar_gaps_pp = [
        (sq_cell["breakeven_p_win_fight"] - pro_cell["breakeven_p_win_fight"]) * 100.0
        for pro_cell, sq_cell in zip(pro_cash_cells, sq_cash_cells)
    ]
    max_abs_dpstar_pp = max(abs(gap) for gap in pstar_gaps_pp) if pstar_gaps_pp else 0.0
    control_threshold_rows = "\n".join(
        (
            f"{_tex_escape(pro_cell['label'])} & "
            f"{pro_cell['breakeven_p_win_fight']*100:.1f}\\% & "
            f"{sq_cell['breakeven_p_win_fight']*100:.1f}\\% & "
            f"{signed_tenth_pp(gap)} \\\\"
        )
        for (pro_cell, sq_cell), gap in zip(
            zip(pro_cash_cells, sq_cash_cells), pstar_gaps_pp
        )
    )

    template = Path(__file__).with_name("grubs_intrinsic_value.tex.tpl")
    if not template.exists():
        raise FileNotFoundError(template)
    text = template.read_text()
    repl = {
        "N_RAW_2026": f"{d['sample']['n_raw_2026_maps']:,}",
        "N_MAPS": f"{d['sample']['n_era_3camp']:,}",
        "N_EXACT_THREE": f"{d['sample']['n_exactly_three_grubs']:,}",
        "N_FEWER_THREE": f"{d['sample']['n_fewer_than_three_grubs']:,}",
        "N_LEAGUES": f"{d['sample']['n_leagues']:,}",
        "N_FIT_LEAGUES": f"{d['sample']['n_fit_leagues']:,}",
        "N_GOLD_MISSING": f"{d['sample']['n_gold10_missing']:,}",
        "MISSING_GOLD_LEAGUES": _tex_escape(
            ", ".join(d["sample"]["gold10_missing_leagues"])
        ),
        "N_OUTCOME_MISSING": f"{d['sample']['n_outcome_missing_after_gold']:,}",
        "N_GOLD_OUTCAP": f"{d['sample']['n_gold10_outside_cap']:,}",
        "DATE_MIN": _tex_escape(str(d["sample"]["date_min"])),
        "DATE_MAX": _tex_escape(str(d["sample"]["date_max"])),
        "PP_GOLD": _pp(g["at_even"]),
        "PP_PER_100": _pp(g["pp_per_100g_at_even"]),
        "PP_PREF": _pp(b["preferred_gold_plus_brief_burn_pp"]),
        "PP_PREF_POST": _pp(b["post_26_11_cash_plus_brief_pressure_ceiling_pp"]),
        "PP_JOINT": _pp(b["central_gold_plus_xp_joint_pp"]),
        "PP_UPPER": _pp(b["upper_joint_plus_wiki_burn_20s_pp"]),
        "PP_LS_MID": _pp(b["ls_style_mid_turret_burn_alone_pp"]),
        "PP_ASSOC": _pp(c["unique_dpp"]),
        "N_ASSOC": f"{int(c['n']):,}",
        "PP_KILL": _pp(opp["median_tf_assumption"]["win_pp"]),
        "PP_EDGE25": _pp(ls["at_p_025"]["edge_contest_minus_leave_pp"]),
        "PSTAR": _prob(ls["breakeven_p"]),
        "PSTAR_PCT": f"{float(ls['breakeven_p'])*100:.0f}",
        "CERTAINTY_ROWS": certainty_rows,
        "CERTAINTY_MIN_PCT": f"{certainty_min:.1f}",
        "CERTAINTY_MAX_PCT": f"{certainty_max:.1f}",
        "REFERENCE_MIN_PCT": f"{reference_min:.1f}",
        "REFERENCE_MAX_PCT": f"{reference_max:.1f}",
        "DEFICIT_ROWS": deficit_rows_tex,
        "SIEGE_ROWS": siege_rows_tex,
        "SIEGE_ARMOR_MULT": f"{float(siege['armor_multiplier']):.3f}",
        "SIEGE_NORMAL_DMG": f"{float(siege['normal_attack_damage']):.2f}",
        "SIEGE_SPELL_DMG": f"{float(siege['spellblade_bonus_damage']):.2f}",
        "SIEGE_ATTACK_PERIOD": f"{float(siege['attack_period_s']):.4f}",
        "SIEGE_PROC_EVERY": f"{int(siege['spellblade_proc_every_attacks'])}",
        "SIEGE_AS": f"{float(siege['attack_speed']):.4f}",
        "SIEGE_TOTAL_AD": f"{float(siege['total_ad']):.2f}",
        "SIEGE_T0": f"{float(siege['zero_stack_time_s']):.2f}",
        "SIEGE_A0": f"{int(siege['zero_stack_attacks'])}",
        "SIEGE_T3": f"{float(siege['three_stack_time_s']):.2f}",
        "SIEGE_A3": f"{int(siege['three_stack_attacks'])}",
        "SIEGE_TSAVE": f"{float(siege['three_stack_time_saved_s']):.2f}",
        "SIEGE_ASAVE": f"{int(siege['three_stack_attacks_saved'])}",
        "SIEGE_TREDUCTION": f"{float(siege['three_stack_time_reduction_pct']):.1f}",
        "SIEGE_TOUCH_DMG3": f"{float(siege['rows'][3]['touch_true_damage']):.0f}",
        "SIEGE_ZAAHEN0": f"{float(siege['rows'][0]['zaahen_without_touch_damage']):.0f}",
        "SIEGE_TOUCH_SHARE3": f"{float(siege['rows'][3]['touch_vs_zaahen_without_touch_pct']):.1f}",
        "CENTRAL_PSTAR_CI_LO": f"{float(central_pstar_ci['ci95_low']) * 100:.1f}",
        "CENTRAL_PSTAR_CI_HI": f"{float(central_pstar_ci['ci95_high']) * 100:.1f}",
        "CASH_FIRST_PCT": f"{float(cash_cells[0]['breakeven_p_win_fight']) * 100:.1f}",
        "CASH_LAST_PCT": f"{float(cash_cells[-1]['breakeven_p_win_fight']) * 100:.1f}",
        "TOUCH_FIRST_PCT": f"{float(touch_cells[0]['breakeven_p_win_fight']) * 100:.1f}",
        "TOUCH_LAST_PCT": f"{float(touch_cells[-1]['breakeven_p_win_fight']) * 100:.1f}",
        "REFUSAL_SLOPE": _pp(float(vsl["win_tf_and_grubs_pp"]) - float(vsl["lose_tf_no_grubs_pp"])),
        "REFUSAL_PCT": f"{float(touch_cells[2]['breakeven_p_win_fight']) * 100:.1f}",
        "EV_AT_50": _pp(reference_loss + 0.50 * reference_slope),
        "EV_AT_70": _pp(reference_loss + 0.70 * reference_slope),
        "EV_PER_10": _pp(0.10 * reference_slope),
        "CAMP_OWNERSHIP_PP": _pp(camp_ownership_pp),
        "FIGHT_RESULT_PP": _pp(fight_result_pp),
        "FIGHT_TO_CAMP_RATIO": f"{fight_result_pp / camp_ownership_pp:.1f}",
        "FIGHT_N": f"{fight_validation['n_engagements']:,}",
        "FIGHT_ROWS": f"{fight_validation['n_oriented_rows']:,}",
        "FIGHT_EPISODES": f"{fight['sample']['valid_first_grub_episodes']:,}",
        "FIGHT_DECISIVE_RATE": f"{fight['descriptive']['decisive_exchange_rate'] * 100:.1f}",
        "FIGHT_BETA": f"{fight_model['beta_per_1000_gold']:.4f}",
        "FIGHT_AUC": f"{fight_validation['auc']:.3f}",
        "FIGHT_BRIER": f"{fight_validation['brier']:.3f}",
        "FIGHT_NULL_BRIER": f"{fight_validation['null_brier']:.3f}",
        "FIGHT_PRESENCE_AUC": f"{fight_presence_validation['auc']:.3f}",
        "FIGHT_GOLD_ROWS": fight_gold_rows,
        "CONTROL_POPULATION_ROWS": control_population_rows,
        "CONTROL_THRESHOLD_ROWS": control_threshold_rows,
        "SQ_ATTEMPTED": f"{ranked['n_match_ids_attempted']:,}",
        "SQ_USABLE": f"{ranked['n_matches_with_horde']:,}",
        "SQ_ATTRITION": f"{ranked['n_match_ids_attempted'] - ranked['n_matches_with_horde']:,}",
        "SQ_DIAMOND_N": f"{sq_d['n_matches_with_horde']:,}",
        "SQ_MASTER_N": f"{sq_m['n_matches_with_horde']:,}",
        "SQ_FIT_N": f"{sq['n_fit']:,}",
        "SQ_CONTEST_RATE": f"{ranked['contest_rate']*100:.1f}",
        "SQ_DIAMOND_CONTEST": f"{sq_d['contest_rate']*100:.1f}",
        "SQ_MASTER_CONTEST": f"{sq_m['contest_rate']*100:.1f}",
        "SQ_ALL3_N": f"{ranked['n_all3']:,}",
        "SQ_CONTESTED_N": f"{ranked['n_contested']:,}",
        "SQ_FREE_N": f"{ranked['n_free']:,}",
        "SQ_CONTESTED_FREE_DPP": _pp(
            ranked["delta_sweeper_wr_contested_vs_free_pp"]
        ),
        "SQ_90G": _pp(sq_90g_neutral),
        "SQ_90G_CI_LO": _pp(sq_cash_ci["ci95_low_pp"]),
        "SQ_90G_CI_HI": _pp(sq_cash_ci["ci95_high_pp"]),
        "SQ_AUC": f"{sq['diagnostics_10fold']['auc']:.3f}",
        "SQ_PRO_AUC": f"{d['model_diagnostics']['gold10_10fold']['auc']:.3f}",
        "SQ_DIAMOND_AUC": f"{sq_d_cal['diagnostics_10fold']['auc']:.3f}",
        "SQ_MASTER_AUC": f"{sq_m_cal['diagnostics_10fold']['auc']:.3f}",
        "SQ_MAX_ABS_DPSTAR": f"{max_abs_dpstar_pp:.1f}",
        "PRO_MINUS_SQ_90G": _pp(pro_minus_sq),
        "PRO_MINUS_SQ_CI_LO": _pp(pro_minus_sq_lo),
        "PRO_MINUS_SQ_CI_HI": _pp(pro_minus_sq_hi),
        "PRO_MINUS_SQ_Z": f"{pro_minus_sq_z:.2f}",
        "PRO_MINUS_SQ_P": f"{pro_minus_sq_p:.3f}",
        "BETA0": f"{logits['intercept']:.4f}",
        "BETA1": f"{logits['coef']:.6f}",
        "N_FIT": f"{logits['n']:,}",
        "BETA_J0": f"{joint_logits['intercept']:.4f}",
        "BETA_JG": f"{joint_logits['coef_gold']:.6f}",
        "BETA_JX": f"{joint_logits['coef_xp']:.6f}",
        "N_JOINT": f"{joint_logits['n']:,}",
        "CV_FOLDS": f"{int(diagnostics['folds'])}",
        "CV_AUC": f"{diagnostics['auc']:.3f}",
        "CV_BRIER": f"{diagnostics['brier']:.3f}",
        "CV_NULL_BRIER": f"{diagnostics['null_brier']:.3f}",
        "CV_LOGLOSS": f"{diagnostics['log_loss']:.3f}",
        "CV_CAL_INTERCEPT": f"{diagnostics['calibration_intercept']:.3f}",
        "CV_CAL_SLOPE": f"{diagnostics['calibration_slope']:.3f}",
        "MISSINGNESS": miss_tex,
        "WIKI_FARM_NOTE": _tex_escape(wiki_note),
        "FIG1": fig1.name,
        "FIG2": fig2.name,
        "FIG3": fig3.name,
        "FIG4": fig4.name,
        "PP_BURN8": _pp(burn["wiki_scenarios"]["pre_26_11_brief_8s"]["wr_pp_via_gold10_logit"]),
        "PP_BURN20": _pp(burn["wiki_scenarios"]["pre_26_11_3stack"]["wr_pp_via_gold10_logit"]),
        "FARM_ROWS": farm_rows,
        "KILL_ROWS": kill_rows,
        "CURVE_ROWS": rows_curve(head["curve"]),
        "SENS_ROWS": sens_tex,
        "HEAD_PSTAR_PCT": f"{float(head['breakeven_p_win_fight'])*100:.0f}",
        "PP_LOSE": _pp(-abs(float(opp["median_tf_assumption"]["lose_pp"]))),
        "PP_EDGE25_SIGNED": _pp(ls["at_p_025"]["edge_contest_minus_leave_pp"]),
        "PP_LOWER": _pp(b["lower_gold_only_pp"]),
        "PD_LEAVE": _pp(leave_pp),
        "PD_LOSE_NO": _pp(mat["lose_tf_no_grubs_pp"]),
        "PD_LOSE_YES": _pp(mat["lose_tf_but_you_get_grubs_pp"]),
        "PD_WIN_NO": _pp(mat["win_tf_no_grubs_pp"]),
        "PD_WIN_YES": _pp(mat["win_tf_and_grubs_pp"]),
        "PD_VS_LEAVE_WIN_NO": _pp(vsl["win_tf_no_grubs_pp"]),
        "PD_VS_LEAVE_WIN_YES": _pp(vsl["win_tf_and_grubs_pp"]),
        "PD_VS_LEAVE_LOSE_NO": _pp(vsl["lose_tf_no_grubs_pp"]),
        "PD_VS_LEAVE_LOSE_YES": _pp(vsl["lose_tf_but_you_get_grubs_pp"]),
        "PP_MIX25": _pp(mix25["edge_contest_minus_leave_pp"]) if mix25 else "n/a",
        "PMIX_PCT": f"{float(mix_be)*100:.0f}" if mix_be is not None else "n/a",
        "PP_FARM25": _pp(farm25["edge_contest_minus_leave_pp"]) if farm25 else "n/a",
        "PP_NOFARM25": _pp(nofarm25["edge_contest_minus_leave_pp"]) if nofarm25 else "n/a",
        "SISTER_NOTE": sister,
        "PP_FARM": _pp(farm_pp),
        "FARM_GOLD": f"{float(ls.get('farm_gold', 0)):.1f}",
        "FARM_LABEL": _tex_escape(str(ls.get("farm_label", "leave farm"))),
        "ITEM_PACE_BLOCK": _item_pace_tex(d),
        "PP_DUAL25": (
            _pp(ls["dual_tempo_at_p_025"]["edge_contest_minus_leave_pp"])
            if ls.get("dual_tempo_at_p_025")
            else "n/a"
        ),
        "PDUAL_PCT": (
            f"{float(ls['dual_tempo_breakeven_p'])*100:.0f}"
            if ls.get("dual_tempo_breakeven_p") is not None
            else "n/a"
        ),
        "CI_CASH_LO": ci_value(cash_ci, "ci95_low_pp"),
        "CI_CASH_HI": ci_value(cash_ci, "ci95_high_pp"),
        "CI_PREF_LO": ci_value(pref_ci, "ci95_low_pp"),
        "CI_PREF_HI": ci_value(pref_ci, "ci95_high_pp"),
        "CI_PREF_POST_LO": ci_value(pref_post_ci, "ci95_low_pp"),
        "CI_PREF_POST_HI": ci_value(pref_post_ci, "ci95_high_pp"),
        "CI_JOINT_LO": ci_value(joint_ci, "ci95_low_pp"),
        "CI_JOINT_HI": ci_value(joint_ci, "ci95_high_pp"),
        "SCENARIO_ROWS": scenario_tex,
    }
    for k, v in repl.items():
        text = text.replace("<<" + k + ">>", str(v))
    if "<<" in text:
        bad = [w for w in text.split() if "<<" in w][:10]
        raise RuntimeError(f"unreplaced placeholders: {bad}")
    return text


def build_pdf() -> Path:
    from lol_kills.research.grubs_intrinsic_value import build_report, enrich_report_v4
    from lol_kills.research.grubs_fight_probability import build_report as build_fight_report

    # Rebuild from the OE source on every PDF invocation.  Reusing an existing
    # JSON cache here previously allowed corrected era parsing to be hidden by a
    # stale PDF.
    print("[pdf] rebuilding intrinsic JSON from current OE snapshot…")
    d = enrich_report_v4(build_report())
    fight = build_fight_report()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(d, indent=2))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)

    fig1 = OUT_DIR / f"{BASENAME}_fig1_resolved_payoffs.png"
    fig2 = OUT_DIR / f"{BASENAME}_fig2_threshold_ladder.png"
    fig3 = OUT_DIR / f"{BASENAME}_fig3_probability_hurdle.png"
    fig4 = OUT_DIR / f"{BASENAME}_fig4_outcome_matrix.png"
    _fig_resolved_payoffs(d, fig1)
    _fig_threshold_ladder(d, fig2)
    _fig_fight_probability_pilot(fight, fig3)
    _fig_pd_matrix(d, fig4)

    tex = write_tex(d, fight, fig1, fig2, fig3, fig4)
    TEX_PATH.write_text(tex)
    JSON_PATH.write_text(json.dumps(d, indent=2))

    tectonic = shutil.which("tectonic")
    if not tectonic:
        raise RuntimeError("tectonic not found; brew install tectonic")
    print(f"[pdf] compiling {TEX_PATH} …")
    subprocess.run(
        [tectonic, "-o", str(OUT_DIR), str(TEX_PATH)],
        check=True,
        cwd=str(OUT_DIR),
    )
    if not PDF_PATH.exists():
        raise RuntimeError(f"PDF not produced at {PDF_PATH}")
    print(f"[pdf] wrote {PDF_PATH}")
    return PDF_PATH
def main() -> None:
    path = build_pdf()
    desk = Path.home() / "Desktop" / PDF_PATH.name
    shutil.copy2(path, desk)
    legacy = Path.home() / "Desktop" / "grubs_intrinsic_value.pdf"
    if legacy.exists() and legacy.resolve() != desk.resolve():
        legacy.unlink()
    print(f"[pdf] copied {desk}")


if __name__ == "__main__":
    main()
