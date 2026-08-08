#!/usr/bin/env python3
"""Generate the figures and numeric appendix for the elemental-drakes paper."""

from __future__ import annotations

import json
import math
import sys
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lol_kills.draft_archetypes import ARCHETYPE_NAMES, champ_tags
from lol_kills.research.elemental_drake_explorer_model import (
    ELEMENTS,
    MAX_STACKS,
    _StandardizedRuntimeScorer,
)


ROOT = REPO_ROOT
PAPER_DIR = Path(__file__).resolve().parent
FIGURE_DIR = PAPER_DIR / "figures"
STUDY_PATH = ROOT / "apps" / "elemental-drakes" / "src" / "data" / "drake-study.json"

INK = "#1C1E24"
MUTED = "#5C606A"
STEEL = "#3D4A5C"
RUST = "#6B3A32"
RULE = "#C8CAD0"
CANVAS = "#F4F5F7"

ELEMENT_LABELS = {
    "infernal": "Infernal",
    "mountain": "Mountain",
    "ocean": "Ocean",
    "cloud": "Cloud",
    "hextech": "Hextech",
    "chemtech": "Chemtech",
}


def _load() -> dict:
    return json.loads(STUDY_PATH.read_text())


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(CANVAS)
    ax.tick_params(colors=MUTED, labelsize=7.4)
    ax.grid(axis="x", color=RULE, linewidth=0.55, alpha=0.75)
    for spine in ax.spines.values():
        spine.set_color(RULE)
        spine.set_linewidth(0.7)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(
        path,
        dpi=240,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(fig)


def _figure_stage_rankings(study: dict) -> None:
    rankings = study["explorerModel"]["models"]["jointState"][
        "overallElementRankings"
    ]["rankings"]
    metrics = (
        ("firstCapturePp", "A  First global dragon", "stage 1 · median 7:08"),
        ("secondCapturePp", "B  Second global dragon", "stage 2 · median 13:15"),
        (
            "mapPhaseCapturePp",
            "C  Map-phase dragon",
            "capture 3 onward · soul included",
        ),
    )
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.25, 3.15),
        dpi=240,
        facecolor=CANVAS,
        sharex=True,
    )
    for ax, (metric, title, subtitle) in zip(axes, metrics):
        _style_axis(ax)
        ordered = sorted(
            rankings,
            key=lambda row: float(row[metric]),
            reverse=True,
        )
        labels = [ELEMENT_LABELS[row["element"]] for row in ordered][::-1]
        values = np.array([float(row[metric]) for row in ordered][::-1])
        y = np.arange(len(labels))
        ax.hlines(y, 8.7, values, color=RULE, linewidth=1.05, zorder=1)
        ax.scatter(
            values,
            y,
            s=42,
            color=STEEL,
            edgecolor=CANVAS,
            linewidth=0.9,
            zorder=3,
        )
        for yi, value in zip(y, values):
            ax.text(
                value + 0.07,
                yi,
                f"{value:.2f}",
                va="center",
                ha="left",
                fontsize=7.2,
                color=INK,
                fontweight="bold",
                fontfamily="sans-serif",
            )
        ax.set_yticks(y)
        ax.set_yticklabels(
            labels,
            color=INK,
            fontsize=7.25,
            fontfamily="sans-serif",
        )
        ax.set_xlim(8.7, 12.65)
        ax.set_title(
            title,
            loc="left",
            color=INK,
            fontsize=9.2,
            fontweight="bold",
            fontfamily="sans-serif",
            pad=18,
        )
        ax.text(
            0,
            1.035,
            subtitle,
            transform=ax.transAxes,
            color=MUTED,
            fontsize=6.9,
            fontfamily="sans-serif",
            ha="left",
            va="bottom",
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
    axes[1].set_xlabel(
        "Adjusted change in modeled map-win probability (pp)",
        color=INK,
        fontsize=8.0,
        fontfamily="sans-serif",
        labelpad=8,
    )
    fig.subplots_adjust(left=0.105, right=0.985, top=0.82, bottom=0.20, wspace=0.36)
    _save(fig, FIGURE_DIR / "fig1_stage_rankings.png")


def _composition_row(team: dict, opponent: dict) -> dict:
    own = [champion["champion"] for champion in team["composition"]]
    opp = [champion["champion"] for champion in opponent["composition"]]
    row: dict[str, object] = {
        "own_champions": own,
        "opp_champions": opp,
    }
    for tag in ARCHETYPE_NAMES:
        row[f"own_{tag}"] = sum(tag in champ_tags(champion) for champion in own)
        row[f"opp_{tag}"] = sum(tag in champ_tags(champion) for champion in opp)
    return row


def _scorer_for_teams(runtime: dict, focal: dict, opponent: dict) -> _StandardizedRuntimeScorer:
    focal_rows = pd.DataFrame([_composition_row(focal, opponent)])
    mirror_rows = pd.DataFrame([_composition_row(opponent, focal)])
    return _StandardizedRuntimeScorer(runtime, focal_rows, mirror_rows)


def _marginal_rankings(
    scorer: _StandardizedRuntimeScorer,
    stage_seconds: dict[int, int],
) -> list[dict[str, float | str]]:
    zero = {element: 0 for element in ELEMENTS}
    first_minute = stage_seconds[1] / 60.0
    second_minute = stage_seconds[2] / 60.0
    first_pre = scorer.probability(
        minute=first_minute,
        own_inventory=zero,
        opp_inventory=zero,
    )
    rows: list[dict[str, float | str]] = []
    for element in ELEMENTS:
        own_first = dict(zero)
        own_first[element] = 1
        first = (
            scorer.probability(
                minute=first_minute,
                own_inventory=own_first,
                opp_inventory=zero,
            )
            - first_pre
        )

        second_deltas: list[np.ndarray] = []
        for prior in (candidate for candidate in ELEMENTS if candidate != element):
            for focal_owned_first in (False, True):
                own_pre = dict(zero)
                opp_pre = dict(zero)
                (own_pre if focal_owned_first else opp_pre)[prior] = 1
                own_post = dict(own_pre)
                own_post[element] += 1
                second_deltas.append(
                    scorer.probability(
                        minute=second_minute,
                        own_inventory=own_post,
                        opp_inventory=opp_pre,
                    )
                    - scorer.probability(
                        minute=second_minute,
                        own_inventory=own_pre,
                        opp_inventory=opp_pre,
                    )
                )

        path_deltas: list[np.ndarray] = []
        openings = [candidate for candidate in ELEMENTS if candidate != element]
        for left, right in combinations(openings, 2):
            for left_focal in (False, True):
                for right_focal in (False, True):
                    own = dict(zero)
                    opp = dict(zero)
                    (own if left_focal else opp)[left] = 1
                    (own if right_focal else opp)[right] = 1
                    increments: list[np.ndarray] = []
                    capture_index = 0
                    while sum(own.values()) < MAX_STACKS:
                        capture_index += 1
                        stage = 2 + capture_index
                        minute = stage_seconds[stage] / 60.0
                        post = dict(own)
                        post[element] += 1
                        soul = element if sum(post.values()) == MAX_STACKS else None
                        increments.append(
                            scorer.probability(
                                minute=minute,
                                own_inventory=post,
                                opp_inventory=opp,
                                own_soul=soul,
                            )
                            - scorer.probability(
                                minute=minute,
                                own_inventory=own,
                                opp_inventory=opp,
                            )
                        )
                        own = post
                    path_deltas.append(np.mean(np.vstack(increments), axis=0))
        rows.append(
            {
                "element": element,
                "first": float(np.mean(first) * 100),
                "second": float(np.mean(np.vstack(second_deltas)) * 100),
                "map": float(np.mean(np.vstack(path_deltas)) * 100),
            }
        )
    return rows


def _figure_composition(study: dict) -> dict:
    pilot = next(
        game
        for game in study["pilotGames"]
        if game["competitionLevel"] == "tier1" and game["league"] == "LCK"
    )
    blue = next(team for team in pilot["teams"] if team["side"] == "blue")
    red = next(team for team in pilot["teams"] if team["side"] == "red")
    runtime = study["explorerModel"]["models"]["jointState"]["runtime"]
    stage_seconds = {
        int(row["stage"]): int(row["medianSeconds"])
        for row in study["explorerModel"]["stageReference"]
    }
    blue_rankings = _marginal_rankings(
        _scorer_for_teams(runtime, blue, red),
        stage_seconds,
    )
    red_rankings = _marginal_rankings(
        _scorer_for_teams(runtime, red, blue),
        stage_seconds,
    )
    lookups = {
        "blue": {row["element"]: row for row in blue_rankings},
        "red": {row["element"]: row for row in red_rankings},
    }
    panels = (
        ("first", "A  First-dragon comparison", "zero inventory · 7:08"),
        (
            "map",
            "B  Map-phase comparison",
            "legal capture-3-to-soul paths",
        ),
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.25, 3.18),
        dpi=240,
        facecolor=CANVAS,
    )
    order = [row["element"] for row in sorted(blue_rankings, key=lambda x: x["map"])]
    y = np.arange(len(order))
    for ax, (metric, title, subtitle) in zip(axes, panels):
        _style_axis(ax)
        blue_values = np.array([float(lookups["blue"][element][metric]) for element in order])
        red_values = np.array([float(lookups["red"][element][metric]) for element in order])
        for yi, left, right in zip(y, blue_values, red_values):
            ax.plot(
                [left, right],
                [yi, yi],
                color=RULE,
                linewidth=1.1,
                zorder=1,
            )
        ax.scatter(
            blue_values,
            y,
            s=39,
            color=STEEL,
            edgecolor=CANVAS,
            linewidth=0.8,
            zorder=3,
            label=blue["name"],
        )
        ax.scatter(
            red_values,
            y,
            s=39,
            color=RUST,
            edgecolor=CANVAS,
            linewidth=0.8,
            zorder=3,
            label=red["name"],
        )
        low = min(float(blue_values.min()), float(red_values.min()))
        high = max(float(blue_values.max()), float(red_values.max()))
        span = max(high - low, 1.0)
        ax.set_xlim(low - 0.12 * span, high + 0.18 * span)
        for yi, value in zip(y, blue_values):
            ax.annotate(
                f"{value:.2f}",
                (value, yi),
                xytext=(4, -6),
                textcoords="offset points",
                color=STEEL,
                fontsize=6.2,
                fontweight="bold",
                fontfamily="sans-serif",
            )
        for yi, value in zip(y, red_values):
            ax.annotate(
                f"{value:.2f}",
                (value, yi),
                xytext=(4, 4),
                textcoords="offset points",
                color=RUST,
                fontsize=6.2,
                fontweight="bold",
                fontfamily="sans-serif",
            )
        ax.set_yticks(y)
        ax.set_yticklabels(
            [ELEMENT_LABELS[element] for element in order],
            color=INK,
            fontsize=7.3,
            fontfamily="sans-serif",
        )
        ax.set_title(
            title,
            loc="left",
            color=INK,
            fontsize=9.3,
            fontweight="bold",
            fontfamily="sans-serif",
            pad=18,
        )
        ax.text(
            0,
            1.035,
            subtitle,
            transform=ax.transAxes,
            color=MUTED,
            fontsize=6.9,
            fontfamily="sans-serif",
            ha="left",
            va="bottom",
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.035),
        ncol=2,
        frameon=False,
        fontsize=7.0,
        labelcolor=INK,
        handletextpad=0.35,
        columnspacing=1.4,
    )
    fig.text(
        0.5,
        0.13,
        "Adjusted change in modeled map-win probability (pp)",
        ha="center",
        color=INK,
        fontsize=8.0,
        fontfamily="sans-serif",
    )
    fig.subplots_adjust(left=0.14, right=0.985, top=0.82, bottom=0.27, wspace=0.27)
    _save(fig, FIGURE_DIR / "fig2_composition_example.png")
    return {
        "game": pilot["tournament"],
        "blue": blue["name"],
        "red": red["name"],
        "blueChampions": [row["champion"] for row in blue["composition"]],
        "redChampions": [row["champion"] for row in red["composition"]],
        "blueRankings": blue_rankings,
        "redRankings": red_rankings,
    }


def _figure_residual_validation(study: dict) -> None:
    joint = study["explorerModel"]["models"]["jointState"]
    residual = joint["championResidual"]
    eligible = pd.DataFrame(residual["eligibleCells"])
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.25, 3.25),
        dpi=240,
        facecolor=CANVAS,
    )

    ax = axes[0]
    _style_axis(ax)
    data = [
        eligible.loc[eligible["element"] == element, "gatedCoefficient"].to_numpy(dtype=float)
        for element in ELEMENTS
    ]
    positions = np.arange(len(ELEMENTS))
    parts = ax.violinplot(
        data,
        positions=positions,
        widths=0.75,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )
    for body in parts["bodies"]:
        body.set_facecolor(STEEL)
        body.set_edgecolor(STEEL)
        body.set_alpha(0.38)
    parts["cmedians"].set_color(INK)
    parts["cmedians"].set_linewidth(1.1)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [ELEMENT_LABELS[element] for element in ELEMENTS],
        rotation=32,
        ha="right",
        fontsize=6.8,
        color=INK,
        fontfamily="sans-serif",
    )
    ax.set_ylabel(
        "Ridge-shrunk residual logit coefficient",
        fontsize=7.7,
        color=INK,
        fontfamily="sans-serif",
    )
    ax.set_title(
        "A  Champion residuals",
        loc="left",
        color=INK,
        fontsize=9.3,
        fontweight="bold",
        fontfamily="sans-serif",
        pad=10,
    )
    ax.grid(axis="y", color=RULE, linewidth=0.55, alpha=0.75)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    _style_axis(ax)
    locked = residual["diagnostics"]["lockedHoldout"]
    july = residual["diagnostics"]["publicationExpansionAudit"]
    labels = ["Mar–Apr locked holdout", "July expansion audit"]
    centers = [
        float(locked["deltaBrier"]),
        float(july["deltaBrier"]),
    ]
    intervals = [
        locked["deltaBrierSeriesBootstrap95"],
        july["deltaBrierSeriesBootstrap95"],
    ]
    y = np.arange(2)[::-1]
    for yi, center, interval in zip(y, centers, intervals):
        low = float(interval["lower"])
        high = float(interval["upper"])
        ax.hlines(yi, low, high, color=STEEL, linewidth=2.4, zorder=2)
        ax.vlines([low, high], yi - 0.10, yi + 0.10, color=STEEL, linewidth=1.0)
        ax.scatter(
            [center],
            [yi],
            s=46,
            color=RUST if center > 0 else STEEL,
            edgecolor=CANVAS,
            linewidth=0.8,
            zorder=3,
        )
        ax.text(
            high + 0.00010,
            yi,
            f"{center:+.5f}",
            va="center",
            ha="left",
            fontsize=7.2,
            color=INK,
            fontweight="bold",
            fontfamily="sans-serif",
        )
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.axvline(
        0.0025,
        color=RUST,
        linewidth=0.8,
        linestyle=":",
        alpha=0.9,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(
        labels,
        fontsize=7.2,
        color=INK,
        fontfamily="sans-serif",
    )
    ax.set_xlim(-0.0022, 0.0031)
    ax.set_xlabel(
        "Augmented minus base Brier score",
        fontsize=7.8,
        color=INK,
        fontfamily="sans-serif",
    )
    ax.set_title(
        "B  Family-level validation",
        loc="left",
        color=INK,
        fontsize=9.3,
        fontweight="bold",
        fontfamily="sans-serif",
        pad=10,
    )
    ax.grid(axis="x", color=RULE, linewidth=0.55, alpha=0.75)
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.12, right=0.985, top=0.87, bottom=0.24, wspace=0.42)
    _save(fig, FIGURE_DIR / "fig3_residual_validation.png")


def _write_numbers(study: dict, composition: dict) -> None:
    joint = study["explorerModel"]["models"]["jointState"]
    allocation = study["explorerModel"]["models"]["captureAllocation"]
    rankings = joint["overallElementRankings"]["rankings"]
    by_element = {row["element"]: row for row in rankings}
    residual = joint["championResidual"]
    audit = study["audit"]
    numbers = {
        "generatedAt": study["metadata"]["generatedAt"],
        "dateMin": study["explorerModel"]["cohort"]["dateMin"],
        "dateMax": study["explorerModel"]["cohort"]["dateMax"],
        "completeGames": study["cohort"]["completeGames"],
        "modeledGames": study["explorerModel"]["cohort"]["modeledGames"],
        "series": study["explorerModel"]["cohort"]["series"],
        "captures": study["explorerModel"]["cohort"]["captures"],
        "tierOneGames": study["competitionCoverage"]["tierOneGames"],
        "internationalGames": study["competitionCoverage"]["internationalGames"],
        "otherProGames": study["competitionCoverage"]["otherProGames"],
        "medianFirstSeconds": study["cohort"]["medianFirstDrakeSeconds"],
        "jointDiagnostics": joint["diagnostics"],
        "allocationDiagnostics": allocation["diagnostics"],
        "rankings": rankings,
        "rankingsByElement": by_element,
        "rankingSupport": joint["overallElementRankings"]["support"],
        "championResidual": {
            "publicationCells": residual["vocabularies"]["publication"]["cells"],
            "publicationChampions": residual["vocabularies"]["publication"]["champions"],
            "selectedMinGames": residual["selectedMinGames"],
            "selectedLambda": residual["selectedLambda"],
            "lockedHoldout": residual["diagnostics"]["lockedHoldout"],
            "julyAudit": residual["diagnostics"]["publicationExpansionAudit"],
        },
        "rawAudit": audit,
        "compositionExample": composition,
    }
    (PAPER_DIR / "paper_numbers.json").write_text(
        json.dumps(numbers, indent=2, sort_keys=True) + "\n"
    )


def main() -> int:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    study = _load()
    _figure_stage_rankings(study)
    composition = _figure_composition(study)
    _figure_residual_validation(study)
    _write_numbers(study, composition)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
