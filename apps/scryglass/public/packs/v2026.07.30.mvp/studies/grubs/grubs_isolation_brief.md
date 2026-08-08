# Void Grubs v2 — isolation + measurable contest EV (beatdown)

**Sample:** 3-camp era (`patch ≥ 15.09` or 2026) AND `grub_sum==3` · n=11168 · 2025-05-02 → 2026-07-18

## Limits

OE has no fight logs. **Contest EV here is measurable under Flores roles + gold@10 + who ended with 3**, with P̂_fight = P(beatdown gets first tower | state). Riot Match-V5 HORDE timelines refine contested vs free takes when `RIOT_API_KEY` fills `data/lol/warehouse/timelines/` (coverage now: {'n_cached_in_sample': 0, 'n_sample': 11168, 'frac': 0.0, 'hint': 'export RIOT_API_KEY && python3 -m lol_kills.etl.riot_timelines --from-oe --year-min 2025 --fetch --limit 500'}).

## 1. Raw ladder (blue grub count)

| Blue grubs | n | Blue WR | mean gold@10 |
|------------|---|---------|--------------|
| 0 | 2617 | 44.02% | -368 |
| 1 | 1383 | 46.71% | -184 |
| 2 | 1433 | 53.31% | 164 |
| 3 | 4828 | 59.76% | 414 |

## 2. Controlled association (correct estimands)

| Step | Estimand | Δpp | partial r | LR p | n |
|------|----------|-----|-----------|------|---|
| 1_raw | raw association | 12.60 | — | — | 10261 |
| 2_pre_gold10 | total assoc | pre/near-treatment @10 | 3.73 | 0.0369 | 0.0010 | 10261 |
| 3_post_mediators | residual after gold@15+FH+FT (mediators — NOT unique causal value) | -0.13 | -0.0017 | 0.9193 | 10261 |

**Headline (ship this):** total assoc | pre/near-treatment @10 → **3.73pp** (partial r=0.03693460651352206, LR p=0.0009591890792921523).
**Mediator residual (do not call 'unique value'):** -0.13pp (p=0.9193271014568971). Path check: all3 → +203g @15 | gold@10; FT OR=1.87.
Matched gold@10 ±750.0g: **4.42pp** (n_pairs=4296).

### Gold@10 strata

| Bin | n | WR all3 | WR not | Δpp |
|-----|---|---------|--------|-----|
| behind2k | 685 | 14.2% | 10.1% | +4.0 |
| behind | 2692 | 35.6% | 31.0% | +4.6 |
| even | 2968 | 52.5% | 48.6% | +3.9 |
| ahead | 3002 | 73.8% | 70.2% | +3.5 |
| ahead2k | 914 | 92.7% | 86.1% | +6.7 |

## 3. Measurable contest EV (Flores beatdown)

Clear roles (early_gap≥0.35): n=8520. Beatdown sweeps 38.4%; control sweeps 34.6%.
Control still steals all3 while beatdown ahead@10: **25.8%** (n=3386).
Beatdown still takes all3 while behind@10: **26.7%** (n=2679).

| Outcome | n | WR beatdown | WR control |
|---------|---|-------------|------------|
| Beatdown all3 | 3269 | 58.5% | 41.5% |
| Control all3 | 2946 | 41.9% | 58.1% |
| Neither all3 | 2305 | 50.5% | 49.5% |

**V_control (steal vs gift)** = 16.67pp · **V_beatdown (take vs neither)** = 8.06pp · **V_gift_cost** = 8.60pp.

### Control contest EV vs gifting to beatdown

| P̂_fight beatdown | P̂_fight control | EV pp | verdict |
|------------------|-----------------|-------|---------|
| 30% | 70% | 11.67 | +EV |
| 40% | 60% | 10.00 | +EV |
| 50% | 50% | 8.33 | +EV |
| 60% | 40% | 6.67 | +EV |
| 70% | 30% | 5.00 | +EV |

*Control EV is an **upper bound** on camp value (p×ΔWR steal-vs-gift). Losing a contest may cost extra deaths/tempo — fill Riot timelines to subtract that.*

### Beatdown force EV (behind / contested)

| P̂_fight beatdown | EV pp | verdict |
|------------------|-------|---------|
| 30% | -3.60 | −EV |
| 40% | -1.94 | −EV |
| 50% | -0.27 | ~0 |
| 60% | 1.40 | +EV |
| 70% | 3.06 | +EV |

**One-liner:** Clear Flores roles: control steal-vs-gift ΔWR=16.67pp (camp value only — death cost of losing a contest needs timelines). Control contest upper-bound EV = p_control×16.67: at 30% fight≈5.00pp, at 50%≈8.33pp. Beatdown force: get-vs-neither=8.06pp, gift-cost=8.60pp; at 30% fight EV≈-3.60pp (−EV to force while dog).

## What we claim / don't

- Claim: @10-conditional association ~ multi-pp; residual after gold@15+FT+Herald ≈ 0 = **mediation**, not worthlessness.
- Claim: contest EV is defined for Flores roles with measurable WR branches + FT-calibrated P̂_fight.
- Don't: call residual-after-mediators the 'unique grub value'.
- Don't: claim P̂_fight is true combat win% without timelines.
- Don't: use mixed favorite-sweep controls for underdog Δ.