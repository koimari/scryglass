# Draft Score autoresearch report — 2026-07-31

## Decision

The requested `>=80%` accuracy threshold was not reached. The run is stopped at a measured ceiling rather than promoting a model on the basis of the training block.

The best validated candidate was a contextual winner model that combines the existing Draft Score components with strictly pre-map historical team, player, form, series, and league ratings:

- chronological validation: **133 / 199 = 66.8342%**;
- untouched final holdout: **126 / 200 = 63.00%**;
- fixed production Draft Score on the full revealed ledger: **633 / 997 = 63.49%**.

This candidate is not a pure draft-composition model. It must not be presented as evidence that the draft itself predicts 66.83% of winners.

## Frozen source run

The run is `data/lol/v2/experiments/leaguepedia/manual-run-2026-07-31/`.

- 997 unique maps from 2026-07-01 through 2026-07-30.
- 37 competition labels, including major regional, international, academy/challenger, and tier-2 circuits returned by Leaguepedia.
- 9,970 player/draft rows captured separately from 997 outcome rows.
- 7,355 January–June maps captured as a pre-July seed for historical ratings.
- draft pages contain no `WinTeam`, `Team1Kills`, `Team2Kills`, or `Gamelength_Number` fields.
- the source is a retrospective retrieval made on 2026-07-31; it is not a proof of strict live pre-event availability.

The processing boundary was:

1. capture and hash the schedule, draft, and outcome payloads;
2. freeze a result-free ledger;
3. score every frozen map with the deterministic local engine;
4. seal the scored ledger;
5. open outcomes and reveal the labels;
6. run autoresearch only against the revealed labels, with the final time block untouched during selection.

The fixed runtime is dated 2026-07-18, so the score run is explicitly retrospective. The runner records unavailable player context instead of silently substituting a current roster; 412 maps had complete context and 585 used the deterministic champion-only fallback.

## Search

The autoresearch harness tested 101 candidates in the final pass, including:

- production Draft Score and component recalibrations;
- online historical team/player/league ratings with seven update-rate runs;
- series-state and form features;
- champion, interaction, team-identity, player-identity, and team/player categorical models;
- fixed-strength and composite fallbacks.

Selection used only the first 60% of maps for fitting and the next 20% for validation. The final 20% was used once, after selection, for the holdout result.

Rolling checks for the selected online candidate were:

| Fit through map | Next 199 maps | Accuracy |
| --- | ---: | ---: |
| 40% | 199 | 64.8241% |
| 50% | 199 | 59.7990% |
| 60% | 199 | 66.8342% |
| 70% | 199 | 62.3116% |
| 80% | 199 | 62.8141% |

No candidate produced a validated path to 80%. The identity features did not improve the selected result, and changing the online update rate did not produce a stable improvement.

## Integrity and replay evidence

- `capture-manifest.json` SHA-256: `8ec0c4aec5578c43cf7239272bc6f7e52d233e23d0db985d2f3393892de493e9`
- `frozen-ledger.jsonl` SHA-256: `6577378c8222b965911c9ea0e4d8722c2840d399df66129562253ecfb918d7b3`
- `scored-ledger.jsonl` SHA-256: `fea8b6867490bec17251c729e8d8fc80f7281b76f52acd0da624fc631884ff4`
- `revealed-ledger.jsonl` SHA-256: `09e98d85dd6b103f7906cfc71ac5e21d4bf736d82ce499f3a7f25e0b341c397c`
- autoresearch result SHA-256: `e034ab7dd220a0012ccece1f60782b367edce2edf220fdbb5a1a597e96365c4f`
- autoresearch journal SHA-256: `89599514dce28b87b71b3c14d9883850993be548091d47a5e23a974e3da8ffe4`
- 997 frozen, 997 scored, and 997 revealed ledgers verified successfully.
- the autoresearch rerun reproduced both result files byte-for-byte.
- automated tests: **9 passed**.

## What this means for the build

The build is no longer conceptually blocked. The ingestion, provenance boundary, deterministic scoring, delayed outcome reveal, roster-context fallback, and research loop are implemented.

The `80%` expectation is the part that is not supported by this mixed all-league population. More tuning on this same retrospective ledger would mainly increase selection risk. The next meaningful upgrade is better pre-event evidence: a runtime snapshot dated before each match, exact announced starters, and a timestamped substitution/leave feed such as the Inspired/Armao change. That would improve the validity of the experiment, even if it does not guarantee 80% accuracy.
