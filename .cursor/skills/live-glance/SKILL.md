---
name: live-glance
description: >-
  Fast live map-WR glance from a pasted live state (clock, kills, gold, dragons,
  grubs, towers) plus optional stake/odds/cashout. Prints state checklist,
  p(win) + fair ML, per-objective ablation Δpp chips, separate grubs research
  row when ≤14:45, and HOLD/CASHOUT. Use when the user pastes LIVE updates,
  cashout questions, or asks "what's my live WR" / objective Δpp — not for
  full draft boards (use bet-slip-view / build_board for that).
---

# Live glance → HOLD/CASHOUT

When the user pastes a **live** update (composer block, HUD numbers, cashout): run this path. **Do not** reload a full `build_board` unless `p_pre` is missing and draft is present.

## Workspace

`~/parlay-risk-sim`. User never runs CLI — you run Shell.

## Before numbers — checklist (always)

Verify and list (ask if ambiguous; do not guess):

1. Clock (mm:ss)
2. Kill score (left–right = blue–red unless user says otherwise)
3. Gold totals + lead
4. Towers
5. **Dragon icons** — count each side (3 ≠ 2)
6. Grubs / Herald / Baron — **grubs ≠ dragons**; grub shields ≠ drakes
7. Champs only from clear draft / user list — user draft wins over OCR

## Run (fast)

Prefer helper with last board `p_pre` (or `pre 37%` from paste):

```bash
cd /Users/river/parlay-risk-sim && python3 -m lol_kills.skills.live_glance \
  --team "Movistar KOI" --opp "G2 Esports" --league LEC \
  --p-pre 0.37 --team-is-blue true \
  --clock 18:00 --kills 4 --opp-kills 8 \
  --gold 31.8 --opp-gold 36.3 \
  --dragons 0 --opp-dragons 3 \
  --grubs 1 --opp-grubs 2 \
  --towers 0 --opp-towers 1 \
  --blue "Gnar,Xin Zhao,Orianna,Syndra,Tahm Kench" \
  --red "Rumble,Vi,Yone,Lucian,Leona" \
  --stake 1.81 --odds 5.70 --cashout 1.40
```

`--gold` / `--opp-gold`: pass `k` values (`31.8`) or absolute.

Fallback API (same numbers):

```python
from lol_kills.live_win import objective_delta_pp_breakdown
from lol_kills.skills.live_glance import format_glance
br = objective_delta_pp_breakdown(...)
print(format_glance(br, team="Movistar KOI", opp="G2 Esports"))
```

## Hard rules

1. **&lt;2s path** — reuse session/`pre` `p_pre`; no SHAP; no warehouse refresh.
2. **Grubs research** row (win−leave_mix ≈ +5.69pp) only when clock ≤14:45 — labeled **contest research**, never mixed into live `p_win` or cashout EV.
3. Baron/Herald: collect if present; say **not in WR model**.
4. Book decimal &gt; fair ⇒ +EV; cashout: take if offered ≥ fair cashout (or model says CASHOUT).

## Output shape

```markdown
**State check**
1. Clock: …
…

### Live glance · {TEAM}

**{TEAM} {p}%** · fair **{odds}** · vs pre **{±pp}** · {phase} @ {m}'

**Δpp (ablation):**
- gold: …
- dragons: …

**Grubs research** (contest, not in WR): …   ← only if ≤14:45

**Ticket:** fair cashout ≈ R$X
**HOLD** / **CASHOUT** — …
```

O/U ladder only if kills market mentioned in the paste.

## Composer twin

Same math in-browser: Slip Composer Live WR glance (`web/composer/` + `live_win.js` + `coefs/draft_live_coefs.json`). Chat should match Python within softcap≈OE gap when `draft_edge` omitted in JS.
