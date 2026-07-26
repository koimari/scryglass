# Should the trailing team contest void grubs?

**Audience:** clear decision report (IMLS-ready language)  
**Sample:** 3-camp era (`patch ≥ 15.09` / 2026) · `grub_sum = 3` · maps with a clear lead @10 (`|gold| ≥ 500`)  
**n maps:** 7,846 trailing situations · full era n=12,011  
**Dates:** 2025-05-02 → 2026-07-18

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

If you are trailing @10, contesting only beats “not getting the camp” when your true fight-win chance is at least about **24%**.

Below that: expected map WR from contesting is **worse** than the leave-mix baseline.

### 2) If they contest and win vs contest and lose — is the delta worth it?

**The win-vs-lose gap is real (~7.5pp of map WR), but it is not a jackpot.**

| Outcome for trailing team | n | Map WR | Gold path @10→@15 |
|---------------------------|---|--------|-------------------|
| Contest & win (got all 3) | 1,992 | **30.0%** | -622g |
| Contest & lose / gift | 3,750 | **22.5%** | -1098g |
| Split (no sweep) | 2,104 | **27.6%** | -759g |
| Leave-mix (no all-3) | 5,854 | **24.3%** | -976g |

- **Win vs lose:** +7.5pp  
- **Win vs split:** +2.4pp  
- **Lose vs split:** -5.1pp  

**Read:** most of the pain is **losing the fight and gifting** (you end worse than a quiet split). Winning the steal only barely beats “nobody swept.”

---

## Decision table (use this in callouts)

Assume: if you contest and **win**, you get WR=30.0%; if you contest and **lose**, WR=22.5%.  
Expected WR if you contest = `p × win + (1−p) × lose`.

| Your fight-win chance (p) | Expected map WR if you contest | vs leave-mix (24.3%) | Call |
|---------------------------|--------------------------------|--------------------------------|------|
| 15% | 23.6% | -0.7pp | **AVOID** |
| 20% | 24.0% | -0.3pp | **TOSS-UP** |
| 25% | 24.4% | +0.0pp | **TOSS-UP** |
| 30% | 24.7% | +0.4pp | **TOSS-UP** |
| 35% | 25.1% | +0.8pp | **CONTEST** |
| 40% | 25.5% | +1.2pp | **CONTEST** |
| 45% | 25.9% | +1.6pp | **CONTEST** |
| 50% | 26.2% | +1.9pp | **CONTEST** |
| 55% | 26.6% | +2.3pp | **CONTEST** |
| 60% | 27.0% | +2.7pp | **CONTEST** |
| 70% | 27.8% | +3.4pp | **CONTEST** |

**Breakeven:** need fight-win% ≥ **24%** vs leave-mix · ≥ **68%** vs pure split.

---

## Background: do grubs matter at all?

Yes — but mostly as a **tempo path**, not a magic permanent buff.

| Estimand | Δ map WR | Notes |
|----------|----------|-------|
| Given gold/kills/xp **@10** | **+3.6pp** | Headline association |
| Matched gold@10 | **+4.3pp** | Same story |
| After gold@15 + Herald + first tower | **-0.2pp** | Residual ≈ 0 → value ran through plates/tempo |

So: taking grubs matters for **how the next minutes go**. After those minutes are already in the scoreboard, little unique WR is left.

---

## How often does the trailing side still get all 3?

- Trailing team gets all 3: **25.4%**
- Leader gets all 3: **47.8%**
- Split: **26.8%**

Trailing teams **do** take the camp sometimes — but that outcome is rare relative to the leader taking it, and when they force it their map WR is still low (30.0%).

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
3. **Win-vs-lose delta ≈ 8pp** — real, but the bigger lesson is: **losing the take hurts more than winning it helps** relative to a split.  
4. Rule of thumb: contest from behind only if you truly believe you win the fight **≥ ~24%**.

---

*Generated from `lol_kills.research.grubs_decision_report` · numbers in `grubs_decision_numbers.json`.*
