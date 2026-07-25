---
name: bet-slip-view
description: >-
  One-view fair-odds board from a pasted bet slip, odds screenshot, or Simples
  ticket (map winner, kills O/U, first inhib). Reports favorite + model %, fair
  odds, kills bucket distribution + median, if-X-then-Y ladders, and GRADE vs
  book. Use when the user pastes a bet slip, "Simples", "Vencedor do mapa",
  "Kills Totais", book odds, or asks for implied/fair odds / decision ladder.
---

# Bet slip → 1-view

When the user pastes a slip, odds, or screenshot of markets: **do not** only grade the locked ticket. Always emit the **1-view** below so they can decide while odds move.

## Workspace

Work in `~/parlay-risk-sim`. If not already there, call `move_agent_to_root` first. User never runs CLI — you run Shell.

## Inputs (parse from paste / images / chat)

1. **Markets + book odds** (map winner, kills lines, first inhib, race, series…)
2. **Stake** if present (informational)
3. **Draft** — 10 champs + who is blue/red. Prefer chat/composer paste over OCR.
4. **League** (default `EWC` if series context says so)
5. **Sides** — `team1_is_blue` must match who is actually blue

If draft missing → ask for blue 5 + red 5 before scoring. Do not invent champs.

## Run

Prefer the helper (prints the 1-view):

```bash
cd /Users/river/parlay-risk-sim && python3 -m lol_kills.skills.slip_view \
  --team1 "Karmine Corp" --team2 "Dplus Kia" --league EWC --map 2 \
  --blue "Gnar,Jarvan IV,Cassiopeia,Caitlyn,Lux" \
  --red "Olaf,Naafiri,Anivia,Ezreal,Bard" \
  --team1-is-blue false \
  --ml "Karmine Corp:2.40,Dplus Kia:1.88" \
  --kills "29.5:1.87/1.87,30.5:2.20/1.62" \
  --locked "Karmine Corp:2.40:10"
```

`--kills` format: `line:over/under` (Mais de / Menos de).  
`--locked`: `selection:odds:stake` for the map-winner ticket.

Fallback: `lol_kills.board.build_board` + `grade_bet` / `p_under_normal` with the same numbers.

## Output — always this shape (1-view)

Keep it scannable. No essay. Start with the favorite line.

```markdown
### 1-view · {match} · Map {n} · {league}

**Favorite:** {TEAM} **{p}%** · fair **{odds}** · book **{odds or —}** · edge **{±Xpp}** · **{GRADE}**
**Underdog:** {TEAM} **{p}%** · fair **{odds}**

#### Kills
μ **{mu}** · median ≈ **{med}** · sd **{sd}**
Buckets: {≤26: a% · 27–29: b% · 30–32: c% · 33+: d%}
Main line {L}: Under fair **{fu}** · Over fair **{fo}**

#### Ladder (odds move — use this)
**Map {fav}:** skip &lt; {fair} · small ≥ {fair*1.05≈} · **yes ≥ {EV10%}** · punch ≥ {EV20%}
**Under {L}:** skip &lt; {fu} · **yes ≥ {fu_yes}** · punch ≥ {fu_punch}
**Over {L}:** skip &lt; {fo} · **yes ≥ {fo_yes}**

#### vs book now
| Market | Book | Fair | Edge | Grade | Action |
|--------|------|------|------|-------|--------|
| … | … | … | … | … | TAKE / SKIP |

**Action:** one sentence.
```

### Ladder math (fixed)

For a selection with model `p`:

| Band | Odds threshold | Meaning |
|------|----------------|---------|
| skip | `< 1/p` | −EV |
| small | `≥ (1.05)/p` | thin +EV |
| **yes** | `≥ (1.10)/p` | main stake (B zone if p≥55%) |
| punch | `≥ (1.20)/p` | size up |

State ladders as: *If {side} odds ≥ X → yes.*

### Rules

- Fair-first: never invent book odds; only use what user pasted.
- First inhib GBM can be wild — if p≥90% or ≤10%, label **soft** and show win-tied prior `0.85*p_win+0.075` instead of leaning hard.
- Locked ticket: still print 1-view, then one line “Locked @ odds — grade X”.
- Odds update only (“became 1.85”): refresh ladder + vs-book row; don’t re-ask draft.
- Portuguese slips (`Vencedor do mapa`, `Mais de`/`Menos de`, `Simples`) are first-class.

## Example trigger

User pastes:

```
Simples · R$10 · 2.40 · Karmine Corp · Vencedor do mapa (Mapa 2)
```

→ 1-view with KC favorite %, fair 1.76, ladder yes≥1.94, locked grade — not just “good ticket”.
