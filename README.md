# Parlay Risk Simulation

Exact Kelly / system-bet allocation for a fixed R$10 stake across a 30-leg Betano esports parlay.

## Run (parlay Kelly)

```bash
pip install -r requirements.txt
python -m sim.run
```

Writes `results/allocation.json`.

## LoL total-kills props (LCK / LEC / LCS)

Leaguepedia-backed team form + H2H kill models for map total O/U betting.

```bash
# Refresh from Leaguepedia (major leagues + MSI/Worlds for H2H)
python -m lol_kills.fetch
python -m lol_kills.build

# List team pace stats
python -m lol_kills.teams
python -m lol_kills.teams --league LCK

# Score a board (LINE:OVER/UNDER decimal odds)
python -m lol_kills.recommend T1 Gen.G --lines "28.5:1.39/2.87,29.5:1.47/2.55,34.5:2.25/1.60"

# Live game: P(Under | minute, kills) + cashout vs hold
python -m lol_kills.live --minute 12 --kills 14 --line 34.5 --stake 7 --odds 1.87 --cashout 1.89

# Draft-conditioned (refresh drafts first if stale)
python -m lol_kills.fetch_drafts --min-per-region 120
python -m lol_kills.draft_model
python -m lol_kills.enrich_games
python -m lol_kills.markets_model

# Full post-draft market sheet (winner, FB, inhib, kills, race)
python -m lol_kills.predict_markets T1 Gen.G --league LCK \
  --blue "Renekton,Nidalee,Azir,Varus,Nautilus" \
  --red "Ornn,Xin Zhao,Ahri,Kai'Sa,Milio" \
  --lines "25.5:1.19/4.35,34.5:1.87/1.87"
```

`--blue` = team1 side, `--red` = team2 side. Pick ΔWR% is the normalized marginal win-rate contribution of each champion.

Data files:

- `data/lol/games_raw.json` — cleaned games
- `data/lol/kill_models.json` — per-team attack/defense + H2H fits

Model: domestic form (kills for/against) × league pace, blended with H2H when enough maps exist. Recommend by EV and by hit-rate among +EV sides.

## Structure

- `data/legs.json` — 30 legs (OCR odds + slip-calibrated odds)
- `sim/probs.py` — fair probs, edge sweeps, Poisson-binomial FFT
- `sim/tickets.py` — 30/29/28-fold + optional 27-fold ladder wealth
- `sim/optimize.py` — exact metrics + SLSQP Kelly / Pareto
- `sim/run.py` — CLI entrypoint
- `lol_kills/` — Leaguepedia fetch / build / recommend for kill totals

## Constraint

Recommended allocation keeps **≥ R$1** on the full 30-fold (user: all games in a parlay) while maximizing `E[log(1+W)]` over covering system tickets.
