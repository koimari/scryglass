# Void Grubs scrap proxy and contest sensitivity

**Version 7** · Associational unit conversion · OE professional maps

> Source of truth: `grubs_intrinsic_value.json`. This file is regenerated from JSON.
> Plate constants: **120g / plate**, first outer plate **900 HP** (wiki).

---

## Abstract

We separate (i) an **associational scrap proxy** for mechanical bumps (90g, optional XP, Touch→plate gold) mapped through gold@10 / xp@10 logits from (ii) the **OE take-regime association** of ending 3–0 after early controls.
Era sample n=1,378; gold@10 fit n=904.
+90g at even → **1.49pp**. **T_pref** (90g + 8s Touch) → **1.91pp**; joint gold+XP → **3.55pp**; take-regime contrast → **0.14pp** (not scrap).
Headline leave-farm: at p=0.25 edge **-7.90pp**, p*≈**0.59**. Item-pace / dual-tempo and 2×2 mixture are sensitivities.

## Estimand (honest)

Associational calibration of the 90g cash reward and conditional Touch pressure ceiling via a gold@10 logit, versus leave-window farm: N∈{1,2,3} laners × grub-era average wave gold (120.67g) plus E[plate]=p×120g. Headline leave = 2 laners + 25% plate (own farm only). Item-pace layer: median/modal early component costs, relative leave−take gold (own farm + opp missed waves − scrap), fight win−lose item gap, and 1/2/3 min horizons while river side is delayed. Dual-tempo leave is sensitivity only.

**T_full** = {90g, 195 XP radius 2000, Touch@3 + Hunger mite}.
**T_pref** (headline) = {90g, brief Touch} — XP is in joint/upper only.

## Item completion pace

Gift relative gap ≈ 393g (0.45× Pickaxe, 1.6 solo-laner minutes). Fight win−lose ≈ 1380g (1.58× Pickaxe). Take/contest side is behind on item pace even when they 'win' free scrap.

## Mechanical package

| Component | Value |
|-----------|------:|
| Gold (3 × 30) | 90 |
| XP (3 × 65) | 195 |
| XP toward level 7→8 (880) | 195/880 = 0.222 |
| Touch tick melee @3 (pre-26.11) | 12 true / 0.5s |
| Hunger at 3 stacks | True |
| Plate gold / first-plate HP | 120 / 900 |

## Results

| Bound | pp | Contents |
|-------|---:|----------|
| Lower | **1.49** | 90g only |
| T_pref | **1.91** | 90g + 8s Touch |
| Joint | **3.55** | 90g + 195 XP |
| Upper | **4.18** | joint + 20s Touch |
| Take-regime | **0.14** | blue 3–0 | controls@10 |

### Headline contest sensitivity

- Δ T_pref = **1.91pp**; farm headline = **0**
- Loss prior (2 deaths + gift) = **11.72pp**; win +2 kills = **9.81pp**
- At p=0.25: edge **-7.90pp**; p*≈**0.59**

### Sister OE contest study (different estimand)

Sister study at p=0.25 is near toss-up vs leave_mix (~0pp / p*~0.26). This note's structural scrap lottery can show large negative edges under death+gift priors. Different objects — report both; do not collapse.
At p=0.25 vs leave_mix: **0.04pp** (TOSS-UP); p*_leave_mix≈**0.2447**.

## Answers

**what_is_the_scrap.** T_pref ≈ +1.91pp (90g + 8s Touch→plate). Full T includes 195 XP in joint/upper bounds.

**leave_farm_hypothesis.** If the giving team farms 1–3 laner-waves (wiki early wave gold) during the window and has a chance at a plate (120g), what map-WR pp does that buy vs contesting the river? Wave = grub-era E[wave] at 10:00 = 102 + 56/3 = 120.6̅g (pre-14 composition; cannon gold 50+⌊t/90⌋). Plate = 120g local outer plating (plates persist past 14:00). 1 laner(s): 120.67g → +2.00pp; edge@25%=-5.90pp; p*≈0.50 | 2 laner(s): 241.33g → +3.99pp; edge@25%=-7.90pp; p*≈0.59 | 3 laner(s): 362.00g → +5.97pp; edge@25%=-9.89pp; p*≈0.67

**how_large_is_intrinsic_grub_value.** T_pref +1.91pp; gold alone +1.49pp; joint +3.55pp.

**leave_farm_gold.** Headline: 2 laners × 1 grub-era average wave each = 241.33g → +3.99pp (grub-era wave 120.67g × 2 + 0.25×120g plate).

**tf_kill_gold.** Kill bounty 300g prior. +2/−2 → +9.81pp / -9.81pp.

**should_you_take_25_75_to_deny.** Headline leave-farm (241g): at p=25% edge = -7.90pp → LEAVE; p*≈0.59. Mixture: -7.90pp. If somehow zero farm: -3.90pp (p*≈0.42). Dual-tempo (own+opp miss): -11.87pp (p*≈0.76).

**item_pace.** Gift relative gap ≈ 393g (0.45× Pickaxe, 1.6 solo-laner minutes). Fight win−lose ≈ 1380g (1.58× Pickaxe). Take/contest side is behind on item pace even when they 'win' free scrap.

**is_ls_right.** On Furia 25/75 under headline leave-farm (wiki waves ± plate): leave still preferred. T_pref ≈ +1.91pp is not literal 0.10pp. Item-pace: gifting scrap can still leave the take side behind on next-item completion for ~1–3 min because they missed waves.

**is_it_decimal_like_ls.** T_pref ≈ +1.91pp — not 0.10. Take-regime +0.14pp is a different estimand.

**why_not_plus_3_7pp.** OE 3–0 association +0.14pp is selection+tempo, not scrap.

**certainty_disclaimer.** Leave farm uses wiki gold constants + gold@10 associational map; p_plate and N laners are scenario inputs; p(fight) is exogenous. Item costs and contest delay are wiki/structural priors, not OE-logged recalls.

## Limitations

- Gold/XP→WR maps are associational, not experimental grants; gold@10 is post-spawn.
- T_pref excludes XP by definition; do not quote it as the price of T_full.
- Farm and kill nets are structural priors; assist gold and smite are stylized in the mixture.
- Exogenous p; one-objective / one-fight lottery.
- Do not coach from this note or the sister OE contest EV alone.

Reproduce:
```bash
python3 -m lol_kills.research.grubs_intrinsic_value
python3 -m lol_kills.research.grubs_intrinsic_pdf
```
