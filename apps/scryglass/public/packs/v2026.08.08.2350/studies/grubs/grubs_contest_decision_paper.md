# Contesting Void Grubs from Behind: Take Regimes, Mediation, and Decision Thresholds in the 3-Camp Era

**Working paper** · Empirical esports analytics · Not peer-reviewed  
**Data:** Oracle’s Elixir professional maps · 3-camp void-grub era (`patch ≥ 15.09` or year 2026)  
**Companion artifacts:** `grubs_decision_numbers.json` · `grubs_isolation_study.json` · ranked HORDE parser proof  

---

## Abstract

We study whether a team that is **trailing at minute 10** should contest **void grubs** (max 3 per side in the current camp economy). Using Oracle’s Elixir (OE) professional maps with `grub_sum = 3` (\(n = 11{,}168\) era maps; \(n = 7{,}295\) with \(|\mathrm{golddiff}_{10}| \ge 500\)), we (i) estimate the map-win association of a 3–0 grub sweep after early-state controls, (ii) decompose that association into a **tempo/mediation path** through gold@15 and early objectives, and (iii) frame a **decision rule** for fight underdogs using take-regime outcomes.

**Findings.** Conditional on gold/kills/xp @10, blue 3–0 associates with **+3.73pp** map WR (LR \(p = 9.6 \times 10^{-4}\); gold@10-matched **+4.42pp**). After gold@15 + first herald + first tower—plausible **post-treatment mediators**—the residual is **−0.13pp** (LR \(p = 0.92\)). Among trailing teams, ending with all 3 (“contest & win” *proxy*) yields map WR **29.74%** [95% Wilson CI 27.72–31.84]; leader taking all 3 (“gift”) yields **22.49%** [21.14–23.91]; split yields **27.76%** [25.81–29.80]. Thus win−lose = **+7.25pp**, but win−split = **+1.98pp** and lose−split = **−5.26pp**. Expected-WR breakeven fight-win probability is **25.92%** vs leave-mix (soft) vs **72.65%** vs split (preferred baseline). **Recommendation:** do not auto-contest from behind; treat ~73% as the honest fight-win bar against a quiet split.

---

## 1. Introduction

Void grubs spawn near minute 8 and feed wave pressure into the plate window (~14:00). Public debate often treats “take the grubs” as intrinsically valuable even from behind. Two questions matter for strategy and for betting models:

1. **Should teams that are likely to lose the river fight contest?**  
2. **Conditional on contesting, is the map-WR gap between winning the take and losing it large enough to justify the risk?**

OE does not record fight participants or contest flags. We therefore use **observable take regimes** anchored at gold@10 (nearest post-spawn checkpoint) and conversion through gold@15.

---

## 2. Data and sample

| Item | Value |
|------|-------|
| Source | Oracle’s Elixir team rows (raw CSVs), map-pivoted |
| Era filter | `patch ≥ 15.09` or `oe_year ≥ 2026` (6→3 camp change) |
| Allocation filter | `grub_sum = 3` |
| Era \(n\) | 11,168 maps |
| Decision subsample | \(\|\mathrm{golddiff}_{10}\| \ge 500\) → 7,295 maps |
| Date range | 2025-05-02 → 2026-07-18 |
| Unit | One professional map |

**Trailing @10** = side with golddiff ≤ −500 from its perspective.  
**Leading @10** = opposite side.

---

## 3. Methods

### 3.1 Association / mediation ladder (map-level, blue 3–0)

Nested logistic models for \(Y = \mathbf{1}\{\text{blue wins}\}\) with treatment \(T = \mathbf{1}\{\text{blue void\_grubs}=3,\ \text{red}=0\}\):

1. Raw association  
2. **Headline:** controls = gold@10, kills@10 diff, xp@10  
3. **Mediator residual:** + gold@15, first herald, first tower  

Report unique Δpp at mean covariates, partial correlation, likelihood-ratio \(p\), and 1:1 gold@10 matching (±750g).

### 3.2 Take regimes (decision subsample)

| Regime label | Definition (trailing side) |
|--------------|----------------------------|
| Contest & win *(proxy)* | Trailing side ends with all 3 grubs |
| Contest & lose / gift | Leading side ends with all 3 |
| Split | Neither all-3 |
| Leave-mix | Trailing side does not get all 3 |

**Important:** gold@10 is *post-spawn*; “contest & win” is an **outcome proxy**, not a filmed contest. Successful steals are selected (mean trail gold@10 −1385g vs −1585g on gifts).

### 3.3 Expected WR under contest

Let \(p\) = fight-win probability for the trailing side, \(w = 0.2974\), \(\ell = 0.2249\):

\[
\mathbb{E}[\mathrm{WR}\mid \text{contest}] = p\, w + (1-p)\, \ell.
\]

Breakeven vs baseline \(b\): \(p^\star = (b-\ell)/(w-\ell)\).

---

## 4. Results

### 4.1 Do grubs associate with map wins?

| Estimand | Δ map WR (pp) | LR \(p\) | Interpretation |
|----------|---------------|---------|----------------|
| Given gold/kills/xp @10 | **+3.7317** | 0.000959 | Headline association |
| Matched gold@10 (±750g), \(n_{\mathrm{pairs}}=4296\) | **+4.4227** | — | Robust to matching |
| + gold@15 + FH + FT | **−0.1306** | 0.919 | Mediator residual ≈ 0 |

**Figure 1 intuition:** sweeping all 3 co-moves with early lead conversion. Once that path is conditioned on, little unique WR remains.

### 4.2 Outcome distribution when trailing @10

| Outcome | \(n\) | Map WR | 95% Wilson CI | Mean gold@10 (trail) | Mean Δgold 10→15 |
|---------|------:|-------:|---------------|---------------------:|-----------------:|
| Contest & win | 1,883 | 29.7398% | [27.72, 31.84] | −1385.0 | −638.3 |
| Split | 1,931 | 27.7576% | [25.81, 29.80] | −1376.9 | −782.2 |
| Leave-mix | 5,412 | 24.3718% | [23.25, 25.53] | −1510.7 | −993.7 |
| Contest & lose / gift | 3,481 | 22.4935% | [21.14, 23.91] | −1585.0 | −1111.0 |

**Allocation rates:** trailing all-3 25.81% · leader all-3 47.72% · split 26.47%.

### 4.3 Win vs lose vs split (the decision-relevant gaps)

| Contrast | Δ map WR (pp) |
|----------|-------------:|
| Win − lose | **+7.2464** |
| Win − split | **+1.9821** |
| Lose − split | **−5.2641** |
| Win − leave-mix | **+5.3680** |

**Interpretation.** Relative to a quiet split, **gifting after a lost take (−5.26pp) hurts more than stealing helps (+1.98pp).**

### 4.4 Decision thresholds

| Baseline | Breakeven fight-win \(p^\star\) | Role |
|----------|-------------------------------:|------|
| Leave-mix (WR 24.3718%) | **0.2592 (25.92%)** | Soft / easy on contesting |
| Split (WR 27.7576%) | **0.7265 (72.65%)** | **Preferred honest bar** |

At \(p = 0.30\) (common “underdog fight” belief):  
\(\mathbb{E}[\mathrm{WR}] = 24.67\%\) → **+0.30pp** vs leave-mix, **−3.09pp** vs split → **avoid** under the honest bar.

### 4.5 Heterogeneity by deficit size

| Trail gold@10 bin | \(n\) | P(get all 3) | WR if get all 3 | WR if gift | WR if split |
|-------------------|------:|-------------:|----------------:|-----------:|------------:|
| −0.5k to −1k | 2,664 | 28.98% | 37.95% | 32.23% | 37.56% |
| −1k to −2k | 3,030 | 24.88% | 28.51% | 23.62% | 25.49% |
| ≤ −2k | 1,598 | 22.34% | 14.57% | 8.60% | 10.78% |

In the mild-deficit band, steal WR ≈ split WR (37.95% vs 37.56%)—little jackpot.

---

## 5. Answering the two operational questions

### Q1. Should teams who most likely lose the fight contest?

**Default: no.** Auto-contesting from behind is not supported.  
Use **dual thresholds:** soft \(p^\star \approx 25.9\%\) (leave-mix) must not be cited alone; **honest \(p^\star \approx 72.7\%\)** (vs split) is the decision-relevant bar for “we didn’t force the river.”

### Q2. Is the win-vs-lose delta worth it?

**Partially.** Winning vs losing the take is **+7.25pp** map WR—material. But winning vs **split** is only **+1.98pp**, while losing vs split is **−5.26pp**. The risk is asymmetric: **downside of gifting dominates upside of stealing** relative to not sweeping.

---

## 6. Limitations

1. **No fight logs** in OE; regimes are outcome proxies.  
2. **Post-spawn gold@10** already embeds early grub/fight outcomes.  
3. **Selection** on successful all-3 takes.  
4. Leave-mix **contaminates** failed contests into the soft baseline.  
5. Pro Match-V5 `LOLTMNT*` timelines are inaccessible with personal Riot keys; ranked HORDE proof validates parsers only.  
6. Results are **associational** decision framing, not experimental causal estimates.

---

## 7. Conclusion

In the 3-camp professional era, void-grub sweeps retain a **~+3.7–4.4pp** map-WR association after early gold controls, largely **mediated** by the subsequent tempo path. For trailing teams, contesting is **not a default**. The honest fight-win requirement against a split baseline is approximately **73%**; the commonly cited soft 26% bar is an artifact of a mixed leave baseline. Relative to splits, **lost gifts cost more map WR than successful steals return.**

---

## Reproducibility

```bash
python3 -m lol_kills.research.grubs_contest_study
python3 -m lol_kills.research.grubs_decision_report
```

Primary outputs: `data/lol/models/grubs_decision_numbers.json`, `grubs_decision_report.md`, this paper.

---

*Version 2026-07-19 · Analytic codebase: `lol_kills`*
