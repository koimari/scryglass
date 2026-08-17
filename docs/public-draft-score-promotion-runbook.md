# Public Draft Score promotion runbook

## Current frozen candidate

- Candidate: `public-draft-score-selective-candidate-v34.json`
- Development AUC: `0.7108585859`
- Selected development games: `1,482`
- Development coverage: `87.02%`
- Candidate receipt: `f5e895bb60946c9e6c2fcb528245597236e15960c1cf83ca835fadb56e750757`
- Protocol: `public-draft-score-promotion-protocol-v39.json`
- Protocol file SHA-256: `6a95f63d5a00b229967379a13be2eb4e213aa2184d87d5b3ce9ae3f2bdda52bf`
- Public probability: unavailable
- Public recommendation: unavailable

The current protocol binds the candidate, model code, batch preparation, sealing, inventory, final evaluation, promotion verification, and public result builder. A change to a bound file creates a new protocol hash. It also starts a new holdout.

## Holdout gates

Open the outcomes one time after all inventory gates pass:

- At least `100` selected recommendations.
- At least `75%` coverage.
- At least three leagues with `20` selected recommendations each.

The final evaluation must then pass these checks:

- AUC above `0.710`.
- Brier score no worse than the same-row quantum voter.
- Log loss no worse than the same-row quantum voter.
- Ten-bin expected calibration error at or below `0.08`.
- Series-cluster bootstrap median AUC above `0.710`.

## Prepare one blind batch

Use an accepted Oracle's Elixir source and a pre-match feature matrix. Keep the result and checkpoint targets in the private source. The preparer removes them before inference.

```zsh
python3 -m lol_kills.research.prepare_selective_draft_holdout_sources \
  --feature-matrix "$FEATURE_MATRIX" \
  --players "$OE_PLAYER_ROWS" \
  --batch-start "$BATCH_START" \
  --batch-end-exclusive "$BATCH_END" \
  --feature-output "$BLIND_FEATURES" \
  --player-output "$BLIND_PLAYERS" \
  --receipt-output "$SOURCE_RECEIPT"
```

The player output contains only game ID, date, side, role, champion, player, team, and league. The feature output contains no outcome, target, observed, or final fields.

## Fit the frozen voters

```zsh
python3 -m lol_kills.research.selective_draft_constituents \
  --training-matrix "$TRAINING_MATRIX" \
  --training-matrix-sha256 "$TRAINING_SHA256" \
  --quantum-training-matrix "$QUANTUM_TRAINING_MATRIX" \
  --quantum-training-matrix-sha256 "$QUANTUM_TRAINING_SHA256" \
  --evaluation-features "$BLIND_FEATURES" \
  --evaluation-features-sha256 "$BLIND_FEATURES_SHA256" \
  --players "$BLIND_PLAYERS" \
  --players-sha256 "$BLIND_PLAYERS_SHA256" \
  --v24-protocol data/lol/v2/evaluation/public-draft-score-promotion-protocol-v24.json \
  --inner-start 2026-06-01T00:00:00Z \
  --evaluation-start 2026-08-16T00:00:00Z \
  --cache-dir "$VOTER_CACHE" \
  --predictions-output "$VOTER_OUTPUT" \
  --receipt-output "$VOTER_RECEIPT"
```

The training matrices can contain historical outcomes. The evaluation feature and player files cannot contain them.

## Seal the batch

```zsh
python3 -m lol_kills.research.seal_selective_draft_holdout \
  --protocol data/lol/v2/evaluation/public-draft-score-promotion-protocol-v39.json \
  --protocol-sha256 "$PROTOCOL_SHA256" \
  --candidate data/lol/v2/evaluation/public-draft-score-selective-candidate-v34.json \
  --candidate-sha256 "$CANDIDATE_SHA256" \
  --features "$BLIND_FEATURES" \
  --features-sha256 "$BLIND_FEATURES_SHA256" \
  --voters "$VOTER_OUTPUT" \
  --voters-sha256 "$VOTER_OUTPUT_SHA256" \
  --voter-receipt "$VOTER_RECEIPT" \
  --voter-receipt-sha256 "$VOTER_RECEIPT_FILE_SHA256" \
  --batch-start "$BATCH_START" \
  --batch-end-exclusive "$BATCH_END" \
  --output "$SEALED_OUTPUT" \
  --receipt-output "$SEALED_RECEIPT"
```

Never replace a sealed output. Use a new path for each batch.

## Check inventory

Pass every sealed receipt in chronological order.

```zsh
python3 -m lol_kills.research.selective_draft_holdout_inventory \
  --receipt "$SEALED_RECEIPT_1" \
  --receipt "$SEALED_RECEIPT_2"
```

Continue collection while `outcomes_may_be_joined` is `false`.

## Evaluate one time

Create a minimal outcome file with exactly `game_uid` and binary `y`. Do this only after the inventory says that outcomes can be joined.

```zsh
python3 -m lol_kills.research.evaluate_selective_draft_holdout \
  --protocol data/lol/v2/evaluation/public-draft-score-promotion-protocol-v39.json \
  --protocol-sha256 "$PROTOCOL_SHA256" \
  --candidate data/lol/v2/evaluation/public-draft-score-selective-candidate-v34.json \
  --candidate-sha256 "$CANDIDATE_SHA256" \
  --receipt "$SEALED_RECEIPT_1" \
  --sealed "$SEALED_OUTPUT_1" \
  --paired-receipt "$PAIRED_RECEIPT_1" \
  --paired-sealed "$PAIRED_OUTPUT_1" \
  --outcomes "$MINIMAL_OUTCOMES" \
  --outcomes-sha256 "$OUTCOMES_SHA256" \
  --output "$EVALUATION_RECEIPT"
```

Repeat each `--receipt` and `--sealed` pair in the same chronological order.

## Independent decision

An independent reviewer creates `scryglass:selective-draft-promotion-decision:v1`. The decision binds:

- Evaluation file and receipt hashes.
- Candidate receipt hash.
- Outcome file hash.
- Reviewer identity and issue time.
- Exact public fields.
- A false betting, odds, expected value, and stake flag.

Verify the decision:

```zsh
python3 -m lol_kills.research.verify_selective_draft_promotion \
  --evaluation "$EVALUATION_RECEIPT" \
  --evaluation-sha256 "$EVALUATION_FILE_SHA256" \
  --decision "$INDEPENDENT_DECISION" \
  --decision-sha256 "$DECISION_FILE_SHA256" \
  --output "$PROMOTION_RECEIPT"
```

Only `scryglass:public-draft-score-promotion-receipt:v1` can unlock the public result builder. A hash-shaped placeholder fails closed.

## Current blind inventory

The August 16 batch is the first valid batch.

- Eligible games: `14`
- Selected recommendations: `13`
- Coverage: `92.86%`
- Selected by league: LPL `7`, LEC `4`, LCS `2`
- Sealed receipt: `468b220f2650c7876b2a6d80ee1e6dfbdedad1a3901de538e071bf2c7ea832fe`
- Inventory receipt: `e407bfdce6a8daa53c3f090dc643c6e1949a9fe135b774b2cab27ac3b4f89716`
- Paired intervention receipt: `10f5531f9ff06ae6396ccb7dfc8315fe862f39354ac7ecc25fabad61af202758`
- Outcomes opened: no

## Public contract

The promoted result keeps two outputs separate.

`match_win_probability` uses the full model. It includes team strength, player strength, momentum, and atomized draft evidence.

`controlled_draft_score` describes the composition contribution with strength controls held fixed. Its model units and percentage-point edge use the same direction.

`side_recommendation` names the side with the higher match win probability.

Betting, odds, expected value, and stake fields stay unavailable.

## Controlled Draft contribution

The controlled Draft value uses two pre-match predictions for the same map.

1. Score the observed ten-player draft.
2. Exchange the Blue and Red champions within each role.
3. Keep the teams, players, roles, date, league, side, ratings, uncertainty,
   momentum, and match context fixed.
4. Score the exchanged draft before reading the result.

`validate_role_matched_champion_swap` proves that the paired rows changed only
the role-matched champions. It binds the two inputs and the fixed controls in a
SHA-256 receipt.

Let `L_observed` and `L_swapped` be the two Blue win logits.

```text
fixed_strength_logit = (L_observed + L_swapped) / 2
controlled_draft_logit = (L_observed - L_swapped) / 2
```

The public percentage-point Draft edge is:

```text
100 * (sigmoid(controlled_draft_logit) - 0.5)
```

This operation removes every effect that is shared by the two predictions.
It preserves champion atoms, player-champion atoms, ally and enemy atom
interactions, patch atom history, and the pre-match phase curve.

The implementation is in:

- `lol_kills/research/controlled_draft_contribution.py`
- `lol_kills/export/paired_public_draft_score.py`

Protocol v39 binds both files, the atomized counterfactual feature builder,
the swap preparer, and the paired intervention sealer. The first August 16
batch was resealed under v39 before any outcome was opened. Its predictions,
selection, and game IDs stayed unchanged. Use only the v39 receipt for the
promotion inventory.

The paired v39 seal contains 14 observed and swapped predictions. Thirteen
rows pass the frozen selection rule in each direction. The controlled Draft
edge ranges from `-5.64` to `+6.00` percentage points. This is outcome-blind
evidence. It does not grant public authority.
