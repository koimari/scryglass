#!/usr/bin/env python3
"""Directed action graph for Void Grub leave / contest decisions.

Two currencies on every load-bearing edge:
  - map WR (pp) from the trailing@10 OE decision table
  - gold (ΔL or mean gold@10→@15 path) where identified

Also attaches the article opportunity-cost p* ladder (gold@10 associational
logit) and an optional FURIA vs G2 @8:21 path overlay.

This is a policy graph over coarse actions, not a full MDP over champ kits.
Thin vs full commit are not separately identified in OE — both share the
contest outcome table; they differ only by the implied fight-win probability.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from lol_kills.etl.paths import MODELS_DIR, ROOT, WAREHOUSE_DIR
from lol_kills.research.grubs_intrinsic_value import contest_ev_terminal_states

OUT_JSON = MODELS_DIR / "grubs_action_graph.json"
OUT_FIG = ROOT / "output" / "pdf" / "grubs_action_graph.png"
FUR_CASE = WAREHOUSE_DIR / "esports_events" / "426848"

# Article reference knobs (Void_Grubs_koimari.pdf)
GOLD10_INTERCEPT = 0.1611182873782888
GOLD10_COEF = 0.000666860223609559
BRIEF_TOUCH_O = 115.6
TWO_WAVE_F = 241.33
FIGHT_SWING = 600.0


def _pp(x: float | None, digits: int = 2) -> float | None:
    if x is None:
        return None
    return round(float(x), digits)


def _load_decision() -> dict[str, Any]:
    path = MODELS_DIR / "grubs_decision_numbers.json"
    return json.loads(path.read_text())


def _pstar(*, baseline_gold: float, leave_farm_gold: float, objective_gold: float = BRIEF_TOUCH_O) -> float | None:
    _, be, _ = contest_ev_terminal_states(
        GOLD10_INTERCEPT,
        GOLD10_COEF,
        baseline_gold=baseline_gold,
        objective_gold=objective_gold,
        leave_farm_gold=leave_farm_gold,
        win_kill_gold=FIGHT_SWING,
        loss_kill_gold=-FIGHT_SWING,
        p_secure_if_win=1.0,
        p_secure_if_lose=0.0,
    )
    return None if be is None else float(be)


def _article_ladder() -> dict[str, Any]:
    behind = [0.0, -500.0, -1000.0, -2000.0]
    ahead = [500.0, 1000.0, 1183.0, 1200.0]
    rows = []
    for b in behind + ahead:
        p = _pstar(baseline_gold=b, leave_farm_gold=TWO_WAVE_F)
        rows.append({
            "B_gold": b,
            "leave_farm_gold": TWO_WAVE_F,
            "objective_gold": BRIEF_TOUCH_O,
            "p_star": _pp(p, 4) if p is not None else None,
            "p_star_pct": _pp(100.0 * p, 1) if p is not None else None,
        })
    farm_rows = []
    for f, label in [
        (0.0, "no_farm"),
        (120.67, "one_wave"),
        (241.33, "two_waves"),
        (350.0, "fur_realistic"),
        (432.0, "fur_optimistic"),
    ]:
        p0 = _pstar(baseline_gold=0.0, leave_farm_gold=f)
        p_fur = _pstar(baseline_gold=1183.0, leave_farm_gold=f)
        farm_rows.append({
            "label": label,
            "leave_farm_gold": f,
            "p_star_at_parity": _pp(p0, 4) if p0 is not None else None,
            "p_star_at_parity_pct": _pp(100.0 * p0, 1) if p0 is not None else None,
            "p_star_at_B_plus_1183": _pp(p_fur, 4) if p_fur is not None else None,
            "p_star_at_B_plus_1183_pct": _pp(100.0 * p_fur, 1) if p_fur is not None else None,
        })
    return {
        "units": (
            "p* = min fight-win probability making contest EV = concede EV "
            "under gold@10 side-neutral logit; two-wave reference unless noted"
        ),
        "reference_knobs": {
            "objective_gold_brief_touch": BRIEF_TOUCH_O,
            "two_wave_leave_farm": TWO_WAVE_F,
            "fight_swing_gold": FIGHT_SWING,
            "secure_if_win": 1.0,
            "secure_if_lose": 0.0,
        },
        "by_precontest_gold_B_two_wave_leave": rows,
        "by_leave_farm_F": farm_rows,
        "estimand_note": (
            "Article opportunity-cost hurdle. Distinct from OE trailing-team "
            "leave-mix breakeven (~24%)."
        ),
    }


def build_graph(decision: dict[str, Any] | None = None) -> dict[str, Any]:
    d = decision or _load_decision()
    oc = d["outcomes"]
    leave_wr = oc["leave_mix_no_all3"]["wr"]
    win_wr = oc["contest_and_win"]["wr"]
    lose_wr = oc["contest_and_lose_or_gift"]["wr"]
    split_wr = oc["split_no_sweep"]["wr"]

    def edge(
        src: str,
        dst: str,
        *,
        action: str | None = None,
        wr_pp: float | None = None,
        gold: float | None = None,
        gold_kind: str | None = None,
        note: str | None = None,
        highlighted: bool = False,
    ) -> dict[str, Any]:
        return {
            "from": src,
            "to": dst,
            "action": action,
            "wr_pp": wr_pp,
            "gold": gold,
            "gold_kind": gold_kind,
            "note": note,
            "highlighted": highlighted,
        }

    nodes = [
        {"id": "pre_grub_state", "label": "Pre-grub state", "kind": "state"},
        {"id": "leave", "label": "Leave", "kind": "action"},
        {"id": "thin_contest", "label": "Thin contest", "kind": "action"},
        {"id": "full_commit", "label": "Full commit", "kind": "action"},
        {"id": "leave_mix", "label": "Leave-mix (no all-3)", "kind": "outcome"},
        {"id": "got_camp", "label": "Got camp (all 3)", "kind": "outcome"},
        {"id": "gifted_camp", "label": "Gifted camp", "kind": "outcome"},
        {"id": "split", "label": "Split (no sweep)", "kind": "outcome"},
        {"id": "map_wr", "label": "Map WR", "kind": "terminal"},
    ]

    # Policy edges: OE trailing@10 sample. Thin/full share contest outcomes.
    edges = [
        edge("pre_grub_state", "leave", action="leave"),
        edge(
            "pre_grub_state",
            "thin_contest",
            action="thin_contest",
            note="Not separately IDd in OE; same outcome table as full commit",
        ),
        edge(
            "pre_grub_state",
            "full_commit",
            action="full_commit",
            note="Not separately IDd in OE; same outcome table as thin contest",
        ),
        edge(
            "leave",
            "leave_mix",
            wr_pp=_pp(100.0 * leave_wr),
            gold=_pp(oc["leave_mix_no_all3"]["mean_gold_path_10_to_15"]),
            gold_kind="mean_gold_path_10_to_15_trailing",
            note=f"n={oc['leave_mix_no_all3']['n']}",
        ),
        edge(
            "thin_contest",
            "got_camp",
            wr_pp=_pp(100.0 * win_wr),
            gold=_pp(oc["contest_and_win"]["mean_gold_path_10_to_15"]),
            gold_kind="mean_gold_path_10_to_15_trailing",
            note=f"n={oc['contest_and_win']['n']}; conditional on trail_all3",
        ),
        edge(
            "thin_contest",
            "gifted_camp",
            wr_pp=_pp(100.0 * lose_wr),
            gold=_pp(oc["contest_and_lose_or_gift"]["mean_gold_path_10_to_15"]),
            gold_kind="mean_gold_path_10_to_15_trailing",
            note=f"n={oc['contest_and_lose_or_gift']['n']}",
        ),
        edge(
            "thin_contest",
            "split",
            wr_pp=_pp(100.0 * split_wr),
            gold=_pp(oc["split_no_sweep"]["mean_gold_path_10_to_15"]),
            gold_kind="mean_gold_path_10_to_15_trailing",
            note=f"n={oc['split_no_sweep']['n']}",
        ),
        edge(
            "full_commit",
            "got_camp",
            wr_pp=_pp(100.0 * win_wr),
            gold=_pp(oc["contest_and_win"]["mean_gold_path_10_to_15"]),
            gold_kind="mean_gold_path_10_to_15_trailing",
            note="Same OE outcome rates as thin_contest",
        ),
        edge(
            "full_commit",
            "gifted_camp",
            wr_pp=_pp(100.0 * lose_wr),
            gold=_pp(oc["contest_and_lose_or_gift"]["mean_gold_path_10_to_15"]),
            gold_kind="mean_gold_path_10_to_15_trailing",
            note="Same OE outcome rates as thin_contest",
        ),
        edge(
            "full_commit",
            "split",
            wr_pp=_pp(100.0 * split_wr),
            gold=_pp(oc["split_no_sweep"]["mean_gold_path_10_to_15"]),
            gold_kind="mean_gold_path_10_to_15_trailing",
            note="Same OE outcome rates as thin_contest",
        ),
        # Terminal: outcome WR already on prior edge; map_wr is sink label.
        edge("leave_mix", "map_wr", wr_pp=_pp(100.0 * leave_wr), note="terminal WR"),
        edge("got_camp", "map_wr", wr_pp=_pp(100.0 * win_wr), note="terminal WR"),
        edge("gifted_camp", "map_wr", wr_pp=_pp(100.0 * lose_wr), note="terminal WR"),
        edge("split", "map_wr", wr_pp=_pp(100.0 * split_wr), note="terminal WR"),
    ]

    deltas = d.get("deltas_pp") or {}
    return {
        "version": 1,
        "title": "Void Grubs action graph (leave / thin / commit)",
        "currencies": {
            "wr": "map WR % or Δpp on trailing@10 OE outcomes",
            "gold": "mean gold@10→@15 path (OE) or ΔL (case overlay)",
        },
        "sample": d.get("sample"),
        "breakeven_p_win_fight_oe": d.get("breakeven_p_win_fight"),
        "deltas_pp_oe": deltas,
        "nodes": nodes,
        "edges": edges,
        "article_pstar_ladder": _article_ladder(),
        "notes": [
            "Thin vs full commit share OE contest outcomes; not separately identified.",
            "OE leave-mix breakeven (~24%) ≠ article two-wave p* (~58.9% at parity).",
            "Article p(map win) = side-neutral gold@10 associational logit, not causal WR.",
        ],
    }


def load_fur_overlay(case_dir: Path = FUR_CASE) -> dict[str, Any] | None:
    gc_path = case_dir / "gold_curve_composite.json"
    if not gc_path.exists():
        return None
    gc = json.loads(gc_path.read_text())
    ev = gc.get("ev_in_gold_lead_units") or {}
    b = float(ev.get("at_8_21_L") or 1183)
    leave_net = float(ev.get("leave_net_if_G2_gets_scrap") or 260)
    contest_dl = float(ev.get("contest_delta_L_8_21_to_8_43") or -1134)
    leave_farm = 350.0
    p_ref = _pstar(baseline_gold=b, leave_farm_gold=TWO_WAVE_F)
    p_real = _pstar(baseline_gold=b, leave_farm_gold=leave_farm)
    return {
        "case": "FURIA vs G2 · gameID 426848 · decision ~8:21",
        "B_gold": b,
        "path_taken": ["pre_grub_state", "full_commit", "gifted_camp", "map_wr"],
        "counterfactual_leave": ["pre_grub_state", "leave", "leave_mix", "map_wr"],
        "edges": [
            {
                "from": "pre_grub_state",
                "to": "leave",
                "gold": leave_net,
                "gold_kind": "counterfactual_delta_L_vs_scrap",
                "wr_pp": None,
                "note": "Realistic leave package ≈350g − 90g scrap → ΔL ≈ +260",
                "highlighted": True,
                "counterfactual": True,
            },
            {
                "from": "pre_grub_state",
                "to": "full_commit",
                "gold": contest_dl,
                "gold_kind": "observed_delta_L_8_21_to_first_grub",
                "wr_pp": None,
                "note": "Observed contest path; G2 took 3 grubs",
                "highlighted": True,
                "counterfactual": False,
            },
        ],
        "opportunity_cost_delta_L": _pp(leave_net - contest_dl),
        "article_p_star": {
            "two_wave_leave_pct": _pp(100.0 * p_ref, 1) if p_ref else None,
            "realistic_leave_350_pct": _pp(100.0 * p_real, 1) if p_real else None,
        },
        "read": (
            f"Ahead B≈+{b:.0f}g with leave up. Article p* ≈ "
            f"{100.0 * p_ref:.1f}% (2-wave) / {100.0 * p_real:.1f}% (≈350g leave). "
            f"Observed ΔL {contest_dl:.0f} vs leave ΔL +{leave_net:.0f}."
        ),
    }


def attach_fur_path(graph: dict[str, Any], overlay: dict[str, Any] | None) -> dict[str, Any]:
    out = dict(graph)
    out["fur_overlay"] = overlay
    if not overlay:
        return out
    taken = set(zip(overlay["path_taken"], overlay["path_taken"][1:]))
    new_edges = []
    for e in out["edges"]:
        ee = dict(e)
        if (e["from"], e["to"]) in taken:
            ee["highlighted"] = True
            ee["fur_realized"] = True
        new_edges.append(ee)
    out["edges"] = new_edges
    return out


def render_figure(graph: dict[str, Any], out_path: Path) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    ink = "#1a1a1a"
    mute = "#5c5c5c"
    paper = "#fbfaf8"
    accent = "#3d5a80"
    realized = "#9a3412"
    cf = "#4a5568"

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.2), dpi=160, gridspec_kw={"width_ratios": [1.35, 1.0]})
    fig.patch.set_facecolor(paper)
    for ax in axes:
        ax.set_facecolor(paper)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    ax = axes[0]
    ax.set_title(
        "Policy graph · trailing@10 OE outcomes",
        loc="left",
        fontsize=11,
        fontweight="bold",
        color=ink,
        pad=8,
    )

    pos = {
        "pre_grub_state": (0.10, 0.55),
        "leave": (0.34, 0.82),
        "thin_contest": (0.34, 0.55),
        "full_commit": (0.34, 0.28),
        "leave_mix": (0.64, 0.82),
        "got_camp": (0.64, 0.58),
        "gifted_camp": (0.64, 0.38),
        "split": (0.64, 0.18),
        "map_wr": (0.90, 0.50),
    }
    labels = {n["id"]: n["label"] for n in graph["nodes"]}
    edge_lookup = {(e["from"], e["to"]): e for e in graph["edges"]}

    def box(xy, text, *, color=ink, lw=1.0):
        x, y = xy
        w, h = 0.17, 0.09
        ax.add_patch(
            FancyBboxPatch(
                (x - w / 2, y - h / 2),
                w,
                h,
                boxstyle="round,pad=0.012,rounding_size=0.02",
                facecolor=paper,
                edgecolor=color,
                linewidth=lw,
            )
        )
        ax.text(x, y, text, ha="center", va="center", fontsize=7.0, color=color)

    for nid, xy in pos.items():
        box(xy, labels.get(nid, nid))

    # Draw action fan-out + leave outcome + thin outcomes + sinks (skip full→outcome dups)
    draw_edges = [
        ("pre_grub_state", "leave", accent),
        ("pre_grub_state", "thin_contest", mute),
        ("pre_grub_state", "full_commit", mute),
        ("leave", "leave_mix", accent),
        ("thin_contest", "got_camp", mute),
        ("thin_contest", "gifted_camp", mute),
        ("thin_contest", "split", mute),
        ("leave_mix", "map_wr", accent),
        ("got_camp", "map_wr", mute),
        ("gifted_camp", "map_wr", mute),
        ("split", "map_wr", mute),
    ]
    # Dashed note that full_commit shares thin outcomes
    ax.annotate(
        "",
        xy=pos["got_camp"],
        xytext=pos["full_commit"],
        arrowprops=dict(arrowstyle="-|>", color=mute, lw=0.7, ls="--", mutation_scale=8),
    )
    ax.text(0.46, 0.33, "same OE rates", fontsize=5.8, color=mute, rotation=18)

    for frm, to, color in draw_edges:
        x1, y1 = pos[frm]
        x2, y2 = pos[to]
        ax.annotate(
            "",
            xy=(x2 - 0.09, y2),
            xytext=(x1 + 0.09, y1),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=0.95, mutation_scale=9),
        )

    for to in ("leave_mix", "got_camp", "gifted_camp", "split"):
        src = "leave" if to == "leave_mix" else "thin_contest"
        e = edge_lookup.get((src, to))
        if not e:
            continue
        x, y = pos[to]
        ax.text(
            x,
            y - 0.068,
            f"WR {e['wr_pp']:.1f}% · Δg {e['gold']:+.0f}",
            fontsize=6.0,
            color=mute,
            ha="center",
            va="top",
        )

    ax.text(
        0.02,
        0.02,
        "Thin and full commit share OE rates (not separately identified).\n"
        "WR = map win % · Δg = mean gold path @10→@15 (trailing side).",
        fontsize=6.5,
        color=mute,
        va="bottom",
        transform=ax.transAxes,
    )

    # Right panel: article p* + FUR overlay
    ax2 = axes[1]
    ax2.set_title(
        "Hurdles + FUR@8:21 overlay",
        loc="left",
        fontsize=11,
        fontweight="bold",
        color=ink,
        pad=8,
    )

    ladder = graph.get("article_pstar_ladder") or {}
    by_b = ladder.get("by_precontest_gold_B_two_wave_leave") or []
    y0 = 0.92
    ax2.text(0.04, y0, "Article p* (2-wave leave, brief-Touch O)", fontsize=8.5, fontweight="bold", color=ink)
    y = y0 - 0.06
    ax2.text(0.04, y, "B", fontsize=7, color=mute, fontweight="bold")
    ax2.text(0.40, y, "p*", fontsize=7, color=mute, fontweight="bold")
    for row in by_b:
        if row["B_gold"] not in (0, -1000, -2000, 1183, 1200):
            continue
        y -= 0.045
        b = row["B_gold"]
        label = "Parity" if b == 0 else f"{b:+.0f}g"
        ax2.text(0.04, y, label, fontsize=7.5, color=ink)
        ax2.text(0.40, y, f"{row['p_star_pct']:.1f}%", fontsize=7.5, color=ink)

    y -= 0.08
    ax2.text(0.04, y, "Leave-farm F at B=+1,183g", fontsize=8.5, fontweight="bold", color=ink)
    y -= 0.05
    for row in ladder.get("by_leave_farm_F") or []:
        if row["label"] not in ("two_waves", "fur_realistic", "fur_optimistic"):
            continue
        y -= 0.045
        ax2.text(0.04, y, f"{row['label']} ({row['leave_farm_gold']:.0f}g)", fontsize=7.3, color=ink)
        ax2.text(0.72, y, f"{row['p_star_at_B_plus_1183_pct']:.1f}%", fontsize=7.3, color=ink, ha="right")

    fur = graph.get("fur_overlay")
    y -= 0.10
    ax2.text(0.04, y, "FURIA vs G2 @8:21", fontsize=8.5, fontweight="bold", color=ink)
    if fur:
        y -= 0.055
        ax2.text(0.04, y, f"B = +{fur['B_gold']:.0f}g", fontsize=7.5, color=ink)
        y -= 0.05
        ax2.text(0.04, y, "Leave (CF)", fontsize=7.5, color=cf)
        ax2.text(0.55, y, f"ΔL ≈ +{fur['edges'][0]['gold']:.0f}g", fontsize=7.5, color=cf)
        y -= 0.05
        ax2.text(0.04, y, "Full commit (obs.)", fontsize=7.5, color=realized)
        ax2.text(0.55, y, f"ΔL = {fur['edges'][1]['gold']:+.0f}g", fontsize=7.5, color=realized)
        y -= 0.05
        ax2.text(0.04, y, "Opportunity cost", fontsize=7.5, color=ink)
        ax2.text(0.55, y, f"ΔL ≈ +{fur['opportunity_cost_delta_L']:.0f}g", fontsize=7.5, color=ink)
        y -= 0.06
        ps = fur.get("article_p_star") or {}
        ax2.text(
            0.04,
            y,
            f"p* ≈ {ps.get('two_wave_leave_pct')}% (2-wave) · "
            f"{ps.get('realistic_leave_350_pct')}% (≈350g leave)",
            fontsize=7.0,
            color=mute,
        )
        y -= 0.08
        ax2.text(0.04, y, fur.get("read", ""), fontsize=6.6, color=mute, wrap=True)

    y = 0.06
    ax2.text(
        0.04,
        y,
        "OE leave-mix breakeven ≈24%  ≠  article 2-wave p* ≈58.9%.\n"
        "p(map win) here = gold@10 associational conversion, not draft-true WR.",
        fontsize=6.4,
        color=mute,
        va="bottom",
    )

    fig.suptitle(
        "Void Grubs · directed action graph (WR + gold)",
        fontsize=12.5,
        fontweight="bold",
        color=ink,
        y=0.98,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, facecolor=paper, edgecolor="none")
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-fur", action="store_true", help="Skip FURIA vs G2 overlay")
    ap.add_argument("--json-out", type=Path, default=OUT_JSON)
    ap.add_argument("--fig-out", type=Path, default=OUT_FIG)
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args(argv)

    graph = build_graph()
    overlay = None if args.no_fur else load_fur_overlay()
    graph = attach_fur_path(graph, overlay)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(graph, indent=2) + "\n")
    print(f"[grubs_action_graph] wrote {args.json_out}")

    if not args.no_fig:
        path = render_figure(graph, args.fig_out)
        print(f"[grubs_action_graph] wrote {path}")

    # Smoke checks
    ladder = graph["article_pstar_ladder"]["by_precontest_gold_B_two_wave_leave"]
    parity = next(r for r in ladder if r["B_gold"] == 0.0)
    assert abs(parity["p_star_pct"] - 58.9) < 0.2, parity
    if overlay:
        assert overlay["B_gold"] == 1183
        assert overlay["opportunity_cost_delta_L"] == 1394.0
    print("[grubs_action_graph] smoke OK")


if __name__ == "__main__":
    main()
