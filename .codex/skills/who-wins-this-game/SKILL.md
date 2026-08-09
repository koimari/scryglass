---
name: who-wins-this-game
description: Predict who wins a professional League of Legends map in under ten seconds using Scryglass Draft Score, team and lineup strength, a calibrated composite expectation, fair decimal odds, and a disclosed minimum-bettable-odds policy. Use for “who wins this game?”, “which draft is better?”, post-draft pregame win expectations, fair odds, value checks, or drafts that specify teams, sides, rosters, substitutes, players, or bookmaker odds.
---

# Who Wins This Game

Return the map winner first. Use one local scoring command and do not browse unless the user explicitly requests fresh external verification.

## Fast path

1. Parse each team in role order: `top,jng,mid,bot,sup`.
2. Require the teams and their actual blue/red sides for a match-win prediction.
3. Resolve each named team’s checked-in roster automatically. Apply explicit player/substitute flags as role overrides.
4. Keep the three estimands distinct:
   - `draft_score`: champion strength, role evidence, allied synergy, enemy counters, same-role evidence, and player–champion comfort.
   - `winning_expectation`: the calibrated team + exact-lineup Elo prior before draft information.
   - `composite`: the model’s calibrated map-win expectation after adding Draft Score evidence to the strength prior.
5. Run:

```bash
/Users/river/scryglass/apps/scryglass/node_modules/.bin/tsx \
  /Users/river/.codex/skills/who-wins-this-game/scripts/who_wins_game.ts \
  --blue "CHAMP1,CHAMP2,CHAMP3,CHAMP4,CHAMP5" \
  --red "CHAMP1,CHAMP2,CHAMP3,CHAMP4,CHAMP5" \
  --league LCK \
  --blue-name "BLUE TEAM" \
  --red-name "RED TEAM" \
  --blue-odds 2.82 \
  --red-odds 1.52
```

Omit player flags that were not supplied. Supported player flags are `--blue-top-player`, `--blue-jng-player`, `--blue-mid-player`, `--blue-bot-player`, `--blue-sup-player`, and red-side equivalents. Use `--repo PATH` only if Scryglass is not at `/Users/river/scryglass`.
Add `--details` only when debugging; the default compact output is designed for the ten-second response budget.

## Response contract

Keep the answer below 220 words unless the user asks for detail.

- First line: `TEAM is the model favorite at X%.`
- Report `Draft Score`, `Strength-only expectation`, and `Composite map expectation` as separate labeled lines.
- Report fair decimal odds and minimum bettable odds for both teams.
- When market odds are supplied, report model edge in percentage points, expected return, and `bettable: yes/no`.
- Explain the main draft driver in at most two bullets. Player–champion comfort is an attribution inside Draft Score, never a separate score.
- End with one uncertainty sentence including `runtime_as_of`.

Use these edge labels:

- under 2 points: `dead even`
- 2–5 points: `slight`
- 5–10 points: `clear`
- over 10 points: `strong`

`edge_points` is the difference between the two displayed percentages. `wr_bump_pp` is the isolation-controlled probability bump and must retain its `pp` unit.

## Fair-odds policy

- Fair decimal odds are `1 / composite_probability`.
- The minimum bettable price must satisfy both policy buffers:
  - at least `+5%` model expected return;
  - at least `+3 percentage points` between model probability and the offered break-even probability.
- Therefore `minimum_bettable_odds = max(1.05 / p, 1 / (p - 0.03))`.
- This threshold is a conservative decision policy, not a learned or backtested calibration result. Say so when the user asks how it was derived.
- `bettable: yes` means only that the offered decimal odds clear this model policy. It is not a guarantee or stake-size recommendation.

## Guardrails

- Never show separate “composition-only” and “with player comfort” scores. Player–champion comfort is one component of the primary Draft Score.
- Resolve ordinary roster seats from `context.json`; let explicit substitutions override the matching role.
- Team and exact-lineup Elo belong only to `winning_expectation`; do not double-count them inside Draft Score.
- Fail closed when either team or any lineup role is unidentified: return Draft Score, but set strength, composite, fair odds, minimum odds, and betting classification to unavailable with the reason.
- Only classify a completed-draft, pregame, single-map winner market. For live/in-game or series odds, do not call the price bettable.
- Treat input odds as decimal. If the format is unclear, report fair and minimum decimal odds without classifying the market.
- Do not silently map patch `16.x` to `26.x`. State `runtime_as_of`; only state a competitive patch when separately verified.
- Treat `same_role` near zero as unavailable/unelected serving evidence, not proof that lanes do not matter.
- Flag thin role evidence from `role_evidence`; in particular, do not hide an off-role bot carry behind high overall champion coverage.
- Base explanations on `components`, `top_synergies`, and `top_counters`. Do not reduce the verdict to one lane matchup.
- For a new champion whose mechanics are uncertain, report the learned contribution without inventing kit details.
- If the helper fails, give a qualitative answer immediately and say the local score was unavailable; do not spend the ten-second budget debugging.
