#!/usr/bin/env python3
"""Compute trailing@10 contest decision numbers + write IMLS decision report."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lol_kills.etl.paths import MODELS_DIR
from lol_kills.research.grubs_contest_study import (
    attach_beatdown,
    blend_fight_proxy,
    block_a,
    engineer_v2,
    load_drafts,
    load_maps,
)


def main() -> None:
    print("[decision] loading…")
    raw = load_maps()
    df = engineer_v2(raw)
    df = attach_beatdown(df, load_drafts())
    df = blend_fight_proxy(df)
    era = df[df.era_3grub & (df.grub_sum == 3)].copy()
    a = block_a(era)

    m = era[era.gold10.notna() & ((era.gold10 <= -500) | (era.gold10 >= 500))].copy()
    blue_trail = m.gold10 <= -500
    m["trail_won"] = np.where(blue_trail, m.y_blue_win, 1 - m.y_blue_win)
    m["trail_all3"] = np.where(blue_trail, m.blue_all3, m.red_all3)
    m["lead_all3"] = np.where(blue_trail, m.red_all3, m.blue_all3)
    m["trail_g10"] = np.where(blue_trail, m.gold10, -m.gold10)
    m["trail_dg"] = np.where(blue_trail, m.gold15 - m.gold10, -(m.gold15 - m.gold10))
    m["split"] = ((m.trail_all3 == 0) & (m.lead_all3 == 0)).astype(float)

    win = m[m.trail_all3 == 1]
    lose = m[m.lead_all3 == 1]
    split = m[m.split == 1]
    leave = m[m.trail_all3 == 0]

    def st(g: pd.DataFrame) -> dict:
        return {
            "n": int(len(g)),
            "wr": float(g.trail_won.mean()) if len(g) else None,
            "mean_gold10": float(g.trail_g10.mean()) if len(g) else None,
            "mean_gold_path_10_to_15": float(g.trail_dg.mean()) if len(g) else None,
        }

    W, L, S, Lv = st(win), st(lose), st(split), st(leave)
    wr_w, wr_l, wr_s, wr_lv = W["wr"], L["wr"], S["wr"], Lv["wr"]
    assert wr_w is not None and wr_l is not None and wr_s is not None and wr_lv is not None

    curve = []
    for p in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]:
        ev = p * wr_w + (1 - p) * wr_l
        curve.append(
            {
                "p_win_fight": p,
                "expected_wr_if_contest": round(ev, 4),
                "edge_vs_leave_mix_pp": round((ev - wr_lv) * 100, 2),
                "edge_vs_split_pp": round((ev - wr_s) * 100, 2),
                "edge_vs_sure_gift_pp": round((ev - wr_l) * 100, 2),
                "verdict_vs_leave": (
                    "CONTEST"
                    if ev > wr_lv + 0.005
                    else ("AVOID" if ev < wr_lv - 0.005 else "TOSS-UP")
                ),
            }
        )

    be_leave = (wr_lv - wr_l) / (wr_w - wr_l)
    be_split = (wr_s - wr_l) / (wr_w - wr_l)

    strata = []
    for name, lo, hi in [
        ("behind_0.5_to_1k", -1000, -500),
        ("behind_1_to_2k", -2000, -1000),
        ("behind_2k_plus", -1e9, -2000),
    ]:
        g = m[(m.trail_g10 >= lo) & (m.trail_g10 < hi)]
        if len(g) < 80:
            continue
        w, l, s = g[g.trail_all3 == 1], g[g.lead_all3 == 1], g[g.split == 1]
        strata.append(
            {
                "bin": name,
                "n": int(len(g)),
                "p_get_all3": round(float(g.trail_all3.mean()), 4),
                "p_leader_gets_all3": round(float(g.lead_all3.mean()), 4),
                "wr_if_get_all3": round(float(w.trail_won.mean()), 4) if len(w) >= 30 else None,
                "wr_if_leader_gets_all3": round(float(l.trail_won.mean()), 4) if len(l) >= 30 else None,
                "wr_if_split": round(float(s.trail_won.mean()), 4) if len(s) >= 30 else None,
            }
        )

    out = {
        "version": 1,
        "language": "fight underdog = trailing @10 by >=500g",
        "sample": {
            "n_era_3camp": int(len(era)),
            "n_trailing_maps": int(len(m)),
            "date_min": str(era.date.min().date()),
            "date_max": str(era.date.max().date()),
            "filter": "patch>=15.09 or 2026; grub_sum==3; |golddiff@10|>=500",
        },
        "association": {
            "headline_dpp_given_gold10": a.get("headline_dpp"),
            "matched_gold10_dpp": (a.get("matched_gold10") or {}).get("dpp"),
            "mediator_residual_dpp": a.get("mediator_residual_dpp"),
            "lr_p_headline": a.get("headline_lr_p"),
        },
        "rates": {
            "p_trailing_gets_all3": round(float(m.trail_all3.mean()), 4),
            "p_leader_gets_all3": round(float(m.lead_all3.mean()), 4),
            "p_split": round(float(m.split.mean()), 4),
        },
        "outcomes": {
            "contest_and_win": W,
            "contest_and_lose_or_gift": L,
            "split_no_sweep": S,
            "leave_mix_no_all3": Lv,
        },
        "deltas_pp": {
            "win_minus_lose": round((wr_w - wr_l) * 100, 2),
            "win_minus_split": round((wr_w - wr_s) * 100, 2),
            "lose_minus_split": round((wr_l - wr_s) * 100, 2),
            "win_minus_leave_mix": round((wr_w - wr_lv) * 100, 2),
        },
        "breakeven_p_win_fight": {
            "vs_leave_mix": round(float(be_leave), 4),
            "vs_split": round(float(be_split), 4),
        },
        "ev_curve": curve,
        "strata_how_far_behind": strata,
        "answers": {
            "should_likely_losers_contest": (
                "Default no — do not auto-contest. Soft breakeven vs leave-mix ≈ "
                f"{be_leave:.0%}; honest breakeven vs split ≈ {be_split:.0%}. "
                "Only force when you truly believe you win the river fight."
            ),
            "is_win_vs_lose_delta_worth": (
                f"Win vs lose ≈ {(wr_w-wr_l)*100:.1f}pp map WR. "
                f"Win vs split only ≈ {(wr_w-wr_s)*100:.1f}pp; lose vs split ≈ {(wr_l-wr_s)*100:.1f}pp. "
                "Gift pain > steal jackpot relative to a quiet split."
            ),
        },
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    num_path = MODELS_DIR / "grubs_decision_numbers.json"
    num_path.write_text(json.dumps(out, indent=2))
    print(f"[decision] wrote {num_path}")

    # Human report
    report = f"""# Should the trailing team contest void grubs?

**Audience:** clear decision report (IMLS-ready language)  
**Sample:** 3-camp era (`patch ≥ 15.09` / 2026) · `grub_sum = 3` · maps with a clear lead @10 (`|gold| ≥ 500`)  
**n maps:** {out['sample']['n_trailing_maps']:,} trailing situations · full era n={out['sample']['n_era_3camp']:,}  
**Dates:** {out['sample']['date_min']} → {out['sample']['date_max']}

---

## Terms (simple)

| Term | Meaning |
|------|---------|
| **Trailing @10** | Team behind by ≥500 gold at minute 10 (fight **underdog**) |
| **Leading @10** | Team ahead by ≥500 gold (fight **favorite**) |
| **Contest & win** | Trailing team still ends with all 3 grubs (steal / won the take) |
| **Contest & lose / gift** | Leading team ends with all 3 (favorite got the camp) |
| **Split** | Neither side swept all 3 |
| **Leave-mix** | Trailing team did **not** get all 3 (failed take + gifts + splits) |
| **Map WR** | Chance that side eventually wins the map |

Grubs spawn ~**8:00**. Gold @10 is the nearest clean checkpoint after that window. We do **not** see the river fight on tape in Oracle’s Elixir — we see **who got the camp** and **what happened next**.

---

## Two questions, direct answers

### 1) Should teams who most likely lose the fight contest?

**Usually no.**

If you are trailing @10, contesting only beats “not getting the camp” when your true fight-win chance is at least about **{be_leave:.0%}**.

Below that: expected map WR from contesting is **worse** than the leave-mix baseline.

### 2) If they contest and win vs contest and lose — is the delta worth it?

**The win-vs-lose gap is real (~{out['deltas_pp']['win_minus_lose']:.1f}pp of map WR), but it is not a jackpot.**

| Outcome for trailing team | n | Map WR | Gold path @10→@15 |
|---------------------------|---|--------|-------------------|
| Contest & win (got all 3) | {W['n']:,} | **{100*wr_w:.1f}%** | {W['mean_gold_path_10_to_15']:+.0f}g |
| Contest & lose / gift | {L['n']:,} | **{100*wr_l:.1f}%** | {L['mean_gold_path_10_to_15']:+.0f}g |
| Split (no sweep) | {S['n']:,} | **{100*wr_s:.1f}%** | {S['mean_gold_path_10_to_15']:+.0f}g |
| Leave-mix (no all-3) | {Lv['n']:,} | **{100*wr_lv:.1f}%** | {Lv['mean_gold_path_10_to_15']:+.0f}g |

- **Win vs lose:** +{out['deltas_pp']['win_minus_lose']:.1f}pp  
- **Win vs split:** +{out['deltas_pp']['win_minus_split']:.1f}pp  
- **Lose vs split:** {out['deltas_pp']['lose_minus_split']:+.1f}pp  

**Read:** most of the pain is **losing the fight and gifting** (you end worse than a quiet split). Winning the steal only barely beats “nobody swept.”

---

## Decision table (use this in callouts)

Assume: if you contest and **win**, you get WR={100*wr_w:.1f}%; if you contest and **lose**, WR={100*wr_l:.1f}%.  
Expected WR if you contest = `p × win + (1−p) × lose`.

| Your fight-win chance (p) | Expected map WR if you contest | vs leave-mix ({100*wr_lv:.1f}%) | Call |
|---------------------------|--------------------------------|--------------------------------|------|
"""
    for c in curve:
        report += (
            f"| {c['p_win_fight']:.0%} | {100*c['expected_wr_if_contest']:.1f}% | "
            f"{c['edge_vs_leave_mix_pp']:+.1f}pp | **{c['verdict_vs_leave']}** |\n"
        )

    report += f"""
**Breakeven:** need fight-win% ≥ **{be_leave:.0%}** vs leave-mix · ≥ **{be_split:.0%}** vs pure split.

---

## Background: do grubs matter at all?

Yes — but mostly as a **tempo path**, not a magic permanent buff.

| Estimand | Δ map WR | Notes |
|----------|----------|-------|
| Given gold/kills/xp **@10** | **+{a.get('headline_dpp'):.1f}pp** | Headline association |
| Matched gold@10 | **+{(a.get('matched_gold10') or {}).get('dpp'):.1f}pp** | Same story |
| After gold@15 + Herald + first tower | **{a.get('mediator_residual_dpp'):.1f}pp** | Residual ≈ 0 → value ran through plates/tempo |

So: taking grubs matters for **how the next minutes go**. After those minutes are already in the scoreboard, little unique WR is left.

---

## How often does the trailing side still get all 3?

- Trailing team gets all 3: **{100*out['rates']['p_trailing_gets_all3']:.1f}%**
- Leader gets all 3: **{100*out['rates']['p_leader_gets_all3']:.1f}%**
- Split: **{100*out['rates']['p_split']:.1f}%**

Trailing teams **do** take the camp sometimes — but that outcome is rare relative to the leader taking it, and when they force it their map WR is still low ({100*wr_w:.1f}%).

---

## Limits (read this)

1. Oracle’s Elixir has **no fight log**. “Contest & win” = trailing @10 **and** ended with all 3 — a **proxy**, not a filmed contest.  
2. Personal Riot keys cannot download pro `LOLTMNT*` timelines; ranked HORDE proof only shows the **parser** works.  
3. Successful steals are **selected** (you only see takes that finished). Failed contests that never secured the camp sit inside leave-mix/split.  
4. This is **association + decision framing**, not lab causality.

---

## Bottom line

1. **Grubs matter through gold/plates/tempo** (~+3–4pp @10-conditional).  
2. **Trailing teams should not auto-contest.**  
3. **Win-vs-lose delta ≈ {out['deltas_pp']['win_minus_lose']:.0f}pp** — real, but the bigger lesson is: **losing the take hurts more than winning it helps** relative to a split.  
4. Rule of thumb: contest from behind only if you truly believe you win the fight **≥ ~{be_leave:.0%}**.

---

*Generated from `lol_kills.research.grubs_decision_report` · numbers in `grubs_decision_numbers.json`.*
"""
    brief = MODELS_DIR / "grubs_decision_report.md"
    brief.write_text(report)
    print(f"[decision] wrote {brief}")
    print("[decision] answers:")
    print(" ", out["answers"]["should_likely_losers_contest"])
    print(" ", out["answers"]["is_win_vs_lose_delta_worth"])


if __name__ == "__main__":
    main()
