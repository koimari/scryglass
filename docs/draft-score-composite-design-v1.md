# Composite Draft Score — 10-Layer Design (v1)

Goal: a per-team draft win probability and per-pick contribution that is a
function of the actual teams, players, champions, and their atomized mechanics —
so that "the best Syndra player picking Syndra" contributes more draft win % than
the worst Syndra player picking it, grounded in player ratings AND atomization,
not just past pick win rates.

Guiding constraints (non-negotiable):
- Strictly-prior computation: nothing from the target game or later may influence
  any feature, model, or calibrator (one global date-ordered pass).
- Every empirical layer shrinks toward its parent layer (small-sample honesty).
- The composite is validated on the fixed 4-window protocol; the promotion gate
  contract is unchanged; no leakage-based selection (the round-3 lesson).
- Each layer ships with tests + an evaluation checkpoint on real OE data.

## The 10 layers

1. **Champion atom prior** — per-(role, champion) draft coefficient; picks with
   no historical support get an atom-based estimate (ridge: 18-dim atom aggregate
   → coefficients). [FOUNDATION — in progress, PR pending]
2. **Champion role WR% base** — historical pick win rate per (role, champion),
   count-shrunk toward the atom prior. The "how good is this champion here" layer.
3. **Player × champion mastery** — strictly-prior EWMA of THIS player's results
   with THIS champion, shrunk toward the champion base (L2). The first
   player-specific layer.
4. **Team × champion affinity** — strictly-prior results of THIS team with THIS
   champion (comfort picks), shrunk toward the champion base.
5. **Player rating** — the player's global Scryglass rating (strictly prior) as a
   per-pick factor. Separates "great player, mediocre matchup" from "great
   player, great pick."
6. **Team rating** — team strength (dual Elo) as the game-level baseline.
7. **Atom × player proficiency** — the player's atom profile (which mechanic
   families they execute best, decomposed from their performance history) applied
   to the champion's atoms: an engage specialist on an engage-heavy champion gets
   a bonus. The novel "atomized player profile."
8. **Intra-team synergy** — role-pair / champion-pair results within the team
   (bot duo, jg-mid), shrunk; plus the team's combined atom profile.
9. **Matchup / counter layer** — player-vs-player and champion-vs-champion lane
   matchups (strictly prior, shrunk) + atom-relation counters (your CC vs their
   tenacity).
10. **Composite calibration + validation** — one calibrated model over all layer
    features (logistic or CatBoost, in-fold calibration), the fixed 4-window
    evaluation, the gate contract, per-team draft win % headline, per-pick
    marginal contribution in % points.

## Evaluation protocol (unchanged)
Same 4 chronological windows, same seed, strictly-prior features; report
Brier / log-loss / AUC pooled + per window; gate contract unchanged.

## Delivery order
- L1: finish atom-prior (code + tests + PR) — in flight.
- L2–L4: champion/player/team empirical layers (one implementation round).
- L5–L7: ratings + atomized player profile (one round).
- L8–L9: synergy + matchups (one round).
- L10: composite model + UI (% per team, per-pick % contributions) + validation.
