# Atomized Random Forest composite

Status: research only. Public and production authority are false.

## Product boundaries

The public descriptive Draft Score stays composition-only. This study does not edit it.

The pre-match composite estimates map outcome from information available before a map starts. The live-state composite is a separate future model. It can use observed state after the map starts.

## Layer A feature authority

| Group | Exact inputs | Time rule |
| --- | --- | --- |
| Team rating | Prior team logit and scaled rating difference | Before map |
| Player rating | Prior lineup logit, scaled lineup difference, lineup coverage | Before map |
| Player and champion history | Separate gold, XP, CS, K/A/D checkpoint fields; damage, vision, jungle, and economy fields; base-probability result residual | Strictly prior games |
| Ally and enemy pairs | Each historical field for directed player/champion ally and enemy pairs | Strictly prior games |
| Phase forecast | Gold and XP forecasts at 10, 15, 20, and 25; interval slopes; peak checkpoint | Current checkpoints are targets only |
| Parity history | Each historical field when player gold and XP differences are both within 250 at a checkpoint | Strictly prior games |
| Momentum | Seven-map mean of outcome minus prior base probability, scaled by 80 | Hash-bound research receipt |
| Patch history | Each field by patch/player/champion and patch/champion | Same patch, strictly prior games |

Every historical estimate has a value, support count, and missing flag. Shrinkage uses five source rows of the prior global mean. Equal timestamps update together.

The locked matrix has 1,700 maps. The refreshed OE source resolves 133 identities through canonical game ID, side, role, and champion. The accepted raw Drive revision resolves the remaining rows through timestamp, league, canonical team, side, role, and champion. Final coverage is 1,700 of 1,700.

## Layer B mechanics ledger

Mechanics must use a versioned game ledger. A complete trusted snapshot seeds stable atom IDs for champions, spells, passives, items, runes, objectives, buffs, debuffs, effects, triggers, targets, formula terms, cooldowns, costs, ranges, durations, stacks, resets, and state transitions.

Each patch supplies a hash-bound delta. The event names the prior ledger hash, changed atoms, source receipts, delta hash, and result hash. Unchanged atoms carry forward through this chain.

Patch 26.15 has a full seed corpus at LCC commit `f0718a98c29dcf5559ffa98c46a487cd52d9c9e3`. It has 173 champion files and 6,017 atoms.

The refreshed archive has 646 patch 16.15 maps. Only 83 intersect the locked full-composite matrix, all in its late test period. This does not supply separate within-patch train, validation, and holdout periods. A mechanics-only fit would omit the required rating and history groups. The harness withholds that fit.

Patch 26.16 has no valid mechanics ledger. The current aggregate combines 16.16 Wiki input with a 16.15 binary atom corpus. It has no explicit 26.16 delta event. The harness rejects it.

Pre-match mechanics can use champion-native atoms and strictly prior build and rune distributions. Observed items, runes, buffs, debuffs, and state belong to the live model.

## Evaluation rules

Random Forest is the primary estimator. The search uses expanding chronological whole-series folds. A bounded first stage uses 120 trees for 12 configurations. Four survivors use their full tree counts on validation. Selection uses validation log loss and the matched baseline AUC floor. The test set does not enter selection.

Calibration uses forward-only whole-series predictions. Region, patch, sparse evidence, phase error, label shuffle, group ablation, and series-cluster bootstrap checks are part of the receipt. Full identity coverage allowed one consumed-test evaluation.

## Measured Layer A result

The frozen validation candidate used 600 trees, unlimited depth, a minimum leaf size of 20, 25 percent of features per split, bootstrap sampling, and no class weighting.

Validation AUC increased from 0.70515 to 0.71544. Brier score changed from 0.21684 to 0.21550. Log loss changed from 0.62282 to 0.62153.

The consumed test failed. AUC was 0.61266, compared with 0.63365 for the matched baseline. Brier score was 0.24176, compared with 0.23216. Log loss was 0.67807, compared with 0.65469. ECE was 0.08798, compared with 0.07369.

The series bootstrap AUC difference was -0.02037. Its 95 percent interval was -0.06612 to +0.02282. Brier and log-loss differences also favored the baseline at their medians. Their intervals included zero.

This candidate fails promotion. The test result is consumed. A later prospective holdout remains required for any new candidate.
