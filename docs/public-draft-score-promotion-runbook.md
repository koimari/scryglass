# Public Draft Score promotion runbook

## Current frozen candidate

- Candidate: `public-draft-score-selective-candidate-v34.json`
- Development AUC: `0.7108585859`
- Selected development games: `1,482`
- Development coverage: `87.02%`
- Candidate receipt: `f5e895bb60946c9e6c2fcb528245597236e15960c1cf83ca835fadb56e750757`
- Protocol file SHA-256: `740e46a6a8c0e0030543ef7de3c72247ae1d6d57e88292c889a1c8b4286637e8`
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
  --protocol data/lol/v2/evaluation/public-draft-score-promotion-protocol-v34.json \
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
  --protocol data/lol/v2/evaluation/public-draft-score-promotion-protocol-v34.json \
  --protocol-sha256 "$PROTOCOL_SHA256" \
  --candidate data/lol/v2/evaluation/public-draft-score-selective-candidate-v34.json \
  --candidate-sha256 "$CANDIDATE_SHA256" \
  --receipt "$SEALED_RECEIPT_1" \
  --sealed "$SEALED_OUTPUT_1" \
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
- Sealed receipt: `ea9cce89d8cf54ad3dc6acd5e62ace1616398c832a09e8afabea894615076634`
- Inventory receipt: `1ac00d3d1b3a1e0c029729c933a895bf938d1bbc22ce58bbfa3c870ebf540aa5`
- Outcomes opened: no

## Public contract

The promoted result keeps two outputs separate.

`match_win_probability` uses the full model. It includes team strength, player strength, momentum, and atomized draft evidence.

`controlled_draft_score` describes the composition contribution with strength controls held fixed. Its model units and percentage-point edge use the same direction.

`side_recommendation` names the side with the higher match win probability.

Betting, odds, expected value, and stake fields stay unavailable.
