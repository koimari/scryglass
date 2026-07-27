# Team-strength-offset draft residual tournament

Status: research-only design, executable test harness, corrected-pack
empirical run, paired dependence-block inference, and calibration-transfer
audit. The run is not a promotion decision, and no production model or public
probability surface was changed.

## Question and estimand

The tournament asks a narrow predictive question:

> After conditioning on a strictly pre-event dynamic team-strength forecast,
> do role-aware champion-composition terms improve out-of-sample map-win
> probabilities?

For map \(i\), the candidate is

\[
\operatorname{logit} P(Y_i=1)
= o_i + a_{\ell(i)} + x_i^\top\beta,
\]

where:

- \(o_i\) is the caller-supplied blue-minus-red team-strength logit. It is a
  fixed offset, not a fitted champion coefficient.
- \(a_{\ell(i)}\) contains the global blue-side and league-specific blue-side
  nuisance terms.
- \(x_i^\top\beta\) is the signed, role-aware composition residual.

The research-facing raw composition estimand is only \(x_i^\top\beta\). Its
neutral-context probability is
\(\operatorname{logit}^{-1}(x_i^\top\beta)\), with team strength and
league/side context set to zero. Organization, player, and roster identities
are not accepted as composition features. Treating team strength as dynamic is
consistent with the time-varying pairwise-comparison framing of Maystre et al.
(2019), although this module does not prescribe the upstream team model.

## Input grain and fail-closed contract

The API accepts already-prepared map rows plus a one-to-one map of pre-event
team logits. It performs no data download.

Each map must have:

- one unique event identifier and one nonblank dependence identifier, normally
  a verified containing series; inferred identifiers must be explicitly
  labeled and may be used only for dependence grouping;
- an event timestamp, league, patch, and binary completed-map outcome;
- exactly one canonical champion for each of top, jungle, mid, bot, and support
  on both sides;
- five unique champions within each side and an explicit draft mode. Ordinary
  tournament drafts require ten globally unique champions. An explicitly
  identified blind-pick map may contain legal cross-side mirrors. Missing,
  same-side duplicate, placeholder, malformed-role, and undeclared
  cross-side-duplicate states are rejected.

Each team offset must:

- match exactly one event;
- be finite natural log-odds oriented blue minus red;
- assert dynamic team-strength-only scope, excluding both side and draft terms;
- retain a model version and source provenance;
- have a data cutoff strictly earlier than the event timestamp.

These are interface assertions. They do not independently prove that an
upstream team model honored its declared feature scope.

## Candidate families and fitting

All champion features are signed blue minus red.

- Additive: role-by-champion main effects.
- Synergy: additive effects plus role-aware within-side champion pairs.
- Opposition: additive and synergy effects plus role-aware cross-side pairs.
- Optional patch deviations: patch-by-role-by-champion deviations around the
  global role/champion term.

Cross-side pairs use one canonical key and an orientation sign. Swapping the two
five-champion sides therefore negates the same accumulated floating-point value
instead of evaluating an unrelated representation. The raw logit is checked
for exact, not approximate, side-swap antisymmetry. A same-role mirrored
champion in an explicit blind-pick map has zero antisymmetric opposition
contribution by construction.

The implementation builds a CSR sparse design and minimizes

\[
\sum_i \left[\log(1+\exp(\eta_i))-y_i\eta_i\right]
+ \tfrac12\beta^\top\Lambda\beta,
\qquad
\eta_i=o_i+X_i\beta.
\]

The analytic gradient is
\(X^\top[\sigma(o+X\beta)-y]+\Lambda\beta\). An independent symbolic check
confirmed this gradient and the signed-score identity. Positive ridge
penalties make sparse composition terms estimable under collinearity; patch
deviations receive at least as much shrinkage as their global terms. The
penalty is a fixed prior precision against the summed likelihood, so the same
predeclared value is retained when a winner is refit on more pre-final data.
Each fit records the iteration count and gradient infinity norm and fails if
the latter exceeds the configured acceptance bound.

Feature vocabularies are built from the relevant training partition only.
Unseen champions contribute zero and are listed on every affected prediction.
An unseen or explicitly unknown patch falls back to global composition terms
and is labeled as such. No future vocabulary is used to construct an earlier
fit.

## Four untouched chronological gates

The splitter keeps both UTC calendar dates and dependence clusters intact.

1. Fit every predeclared hyperparameter specification on train; choose one per
   feature family on validation.
2. Refit only those family winners on train plus validation; select one
   composition candidate on selection.
3. Freeze the candidate identity before reading final outcomes.
4. Refit that candidate on all pre-final rows; score final exactly once.

Changing final labels cannot change validation scores, family winners,
selection scores, the selected candidate, or the final fitted coefficients.
Changing selection labels cannot change the validation gate. Every fitted
prediction records the frozen tournament configuration, training-event digest,
offset digest, offset model version, cutoff, and source provenance, and it must
be scored strictly after the last fitted event.

After the composition candidate is selected, calibration follows a separate
nested development protocol:

1. Fit identity and antisymmetric Platt candidates on the selected
   composition model's validation out-of-sample predictions.
2. Select the calibration form on selection by the predeclared proper score,
   resolving practical ties toward identity.
3. Refit only the chosen form on the combined validation and selection
   out-of-sample prediction ledgers.
4. Freeze it before final and report raw-to-calibrated transfer once.

The Platt form is
\(\operatorname{logit}(p_\text{cal})=s a+b\operatorname{logit}(p)\), where
\(s\) changes sign under a side swap. Its intercept therefore remains a side
nuisance rather than entering the neutral raw-composition estimand. Changing
final labels cannot change the selected calibration method, its coefficients,
or any raw or calibrated final probability.

## Baselines and evidence

Every scored gate contains:

- exact offset-only baseline:
  \(\operatorname{logit}^{-1}(o_i)\), without recalibration or refitting;
- offset plus league/side baseline:
  \(\operatorname{logit}^{-1}(o_i+a_{\ell(i)})\);
- the applicable residual composition candidate.

Log loss and Brier score are the proper-score comparisons, following the
forecast-evaluation principle in Gneiting and Raftery (2007). Fixed-bin ECE is
reported as descriptive calibration evidence, never as the selection
criterion. Lower scores are better.

The final paired ledger has one aligned row per map with event time,
dependence identifier, outcome, all three probabilities, both event-level
proper scores, and candidate-minus-baseline deltas. Negative deltas favor the
candidate.

The default dependence analysis uses 5,000 paired percentile circular
moving-block replicates, a fixed seed of `20260727`, and blocks of 12
consecutive caller-supplied clusters in earliest-event order. The block wraps
circularly and is capped only when fewer than 12 clusters exist. Each selected
cluster contributes all of its maps, so the estimand remains the map-weighted
mean paired loss difference. The same sampled blocks are used for both
baselines and both proper scores. The module never reconstructs a series for
inference. Choe and Ramdas (2021) provide a stronger sequential-forecast
comparison route when its assumptions and score-boundedness choices are
predeclared.

## Corrected-pack empirical run

The tournament was run on 2026-07-27 from the local `v2026.07.26`
partitions, as permitted for the immutable `v2026.07.26.2343` population. The
local manifest name alone is not treated as proof that the directory is the
immutable remote object; the exact partition hashes used are recorded below.

### Population reconstruction and integrity gates

Canonical `oe_year` filtering to 2025 or 2026 produced:

- 32,668 team rows, exactly two per map;
- 163,340 player rows, exactly ten per map;
- 16,334 distinct OE maps from 2025-01-11 through 2026-07-18;
- complete, unique top/jungle/mid/bot/support role rows for both sides; and
- zero exclusions from the dynamic team-strength run.

The unfiltered local partitions contained 33,394 team rows and 166,970 player
rows because transport partitions also contained canonical `oe_year=2024`
records. The filter used canonical `oe_year`, never the folder name.

The 6,196-row published map-feature subset was joined back to the reconstructed
population. Date, league, map number, canonical two-decimal patch, blue and red
team keys, and blue-win outcome had zero mismatches.

Ten maps initially appeared to violate the ten-unique-champion rule. All ten
were legal mirrors in Demacia Cup decider games. The format record states that
game 3 of a Bo3 and game 5 of a Bo5 used no-ban blind pick; the individual-game
record independently shows the mirrored picks. Four additional deciders had no
mirror, for 14 explicitly labeled blind-pick maps in total.
[Liquipedia format record](https://liquipedia.net/leagueoflegends/Demacia_Cup/2025/Group_Stage);
[Leaguepedia individual games](https://lol.fandom.com/wiki/Demacia_Cup_2025/Picks_and_Bans/Individual_Games).

Canonical scheduled-series provenance was unavailable. Source map-number
resets with an 18-hour same-scope gap bound produced 7,805
`inferred-unverified:` dependence clusters. All 7,805 remained quarantined
from rating-series eligibility. These identifiers were used only to preserve
dependence grouping; no scheduled format, completion, or series result was
inferred for the model.

| Split | Inclusive date range | Maps | Inferred, unverified clusters |
|---|---|---:|---:|
| Train | 2025-01-11–2025-09-16 | 8,987 | 4,366 |
| Validation | 2025-09-17–2026-02-18 | 2,425 | 1,251 |
| Selection | 2026-02-19–2026-04-29 | 2,445 | 1,164 |
| Final | 2026-04-30–2026-07-18 | 2,477 | 1,024 |

No UTC date or inferred dependence cluster crossed a gate.

### Strictly pre-event team-strength offset

The upstream offset used the predeclared `balanced` candidate from
`lol_kills.ratings.dynamic_bt`:

```text
team_prior_sd=0.90
team_variance_per_day=0.003
mean_reversion_half_life_days=365
context_prior_sd=0.65
context_variance_per_day=0.001
blue_side_prior_logit=0.08
blue_side_prior_sd=0.30
side_variance_per_day=0
min_variance=0.00001
max_team_variance=4
max_context_variance=3
max_side_variance=1
max_abs_mean=8
probability_floor=0.000001
bridge thresholds: 3 maps, 2 teams/context, 1 competition
unsupported_bridge_variance=2
```

Every exact-timestamp batch was predicted before any outcome in that batch was
assimilated. The supplied offset was the raw latent team difference
`blue_team_mean - red_team_mean`, not `latent_logit` or `logit(p_blue)`, so it
excluded the learned side term, context term, predictive-uncertainty
compression, and every draft feature. Its information cutoff was the previous
distinct event timestamp; the first map used the initial prior with a cutoff
one nanosecond before the event. All 16,334 cutoffs were strictly pre-event.

The raw offsets ranged from -3.3366780637 to +3.3239452861 natural log-odds,
with mean -0.0069924645. The immutable derived offset model identifier was
`dynamic-bt-balanced-raw-team:30a7655ee1c9bd75f970`; the checked
`dynamic_bt.py` SHA-256 was
`01f23726430a20216fc5ee652cc903d0e567859420d8394e3df387e86f09756e`.

### Frozen gates and exact metrics

The candidate grid was fixed before the run: additive, synergy, and opposition
families; ridge penalties 10 and 40; minimum feature support 3; and global-only
or patch-main-deviation variants. Log loss was the sole selection score.
League/side ridge precision was 20. The gradient acceptance bound remained
`1e-4`.

Validation selected the `l2=40`, patch-deviation variant within every family:

| Validation family winner | Log loss | Brier | 10-bin ECE |
|---|---:|---:|---:|
| Additive | 0.6616054774 | 0.2346856235 | 0.0521023386 |
| Synergy | 0.6697117077 | 0.2379688719 | 0.0598400185 |
| Opposition | 0.6726021018 | 0.2389033738 | 0.0694302899 |
| Exact raw-team offset | 0.6653050903 | 0.2364444883 | 0.0361540788 |
| Raw-team offset + league/side | 0.6684333919 | 0.2378829935 | 0.0510602921 |

After refitting the three family winners on train plus validation, the
selection gate froze the additive `l2=40`, patch-deviation candidate:

| Selection model | Log loss | Brier | 10-bin ECE |
|---|---:|---:|---:|
| Additive | 0.6056698401 | 0.2092913988 | 0.0142096028 |
| Raw-team offset + league/side | 0.6074020613 | 0.2099361037 | 0.0147731746 |
| Exact raw-team offset | 0.6084924625 | 0.2104099430 | 0.0308489608 |
| Synergy | 0.6102376617 | 0.2111978446 | 0.0133977132 |
| Opposition | 0.6167522725 | 0.2138785268 | 0.0226280748 |

The selected model was then refit on all 13,857 pre-final maps and evaluated
once on the untouched final gate:

| Final model | Log loss | Brier | 10-bin ECE | Mean probability |
|---|---:|---:|---:|---:|
| Additive residual candidate | 0.6300491137 | 0.2196713225 | 0.0363597359 | 0.5486511101 |
| Exact raw-team offset | 0.6337423231 | 0.2213277535 | 0.0253090780 | 0.5155418125 |
| Raw-team offset + league/side | 0.6333253243 | 0.2211261812 | 0.0215047626 | 0.5495475743 |

The final blue-win rate was 0.5361324182. Paired
candidate-minus-baseline deltas were:

| Baseline | Log-loss delta | Brier delta | ECE delta |
|---|---:|---:|---:|
| Exact raw-team offset | -0.0036932094 | -0.0016564310 | +0.0110506579 |
| Raw-team offset + league/side | -0.0032762107 | -0.0014548588 | +0.0148549733 |

Negative proper-score deltas descriptively favor the candidate; positive ECE
deltas show worse fixed-bin calibration. The paired ledger contains all 2,477
final maps and 1,024 explicitly inferred, unverified clusters.

### Paired dependence-block inference

The predeclared circular moving-block procedure was run for 5,000 replicates
with seed `20260727` and 12 consecutive clusters per block. Clusters were
ordered by earliest event timestamp and then dependence identifier; each
replicate sampled 1,024 clusters with replacement, retaining every map from
each sampled cluster. Identical sampled blocks were used for both baselines
and both proper scores.

| Baseline | Score | Point delta | Bootstrap mean | Bootstrap SE | Percentile 95% interval |
|---|---|---:|---:|---:|---:|
| Exact raw-team offset | Log loss | -0.0036932093890190585 | -0.0036459532687792176 | 0.003040346693082188 | [-0.009561255527747923, 0.002359093191013108] |
| Exact raw-team offset | Brier | -0.0016564309975880876 | -0.001635418964568099 | 0.001365671118259187 | [-0.004312912057823497, 0.0010400139947709841] |
| Raw-team offset + league/side | Log loss | -0.0032762106524579157 | -0.003202240715436524 | 0.0026697333973592 | [-0.008399634121618555, 0.002016542160021439] |
| Raw-team offset + league/side | Brier | -0.0014548587771188093 | -0.00142227083271706 | 0.001161433593319013 | [-0.0036493556749968995, 0.000823415099407948] |

The deterministic bootstrap-distribution SHA-256 values, in table order, are
`bce0e75439d123b6c7bc9ca4c68a13e2028fc199416638fad879de5459c2c0f0`,
`056ead5f9943892fa660d67b13aedd5d3ba89eb0f6bf9dfcc0185e6a90cdb143`,
`af04d973845c153602bce834b1fa0532a5bd1695e0a3153f6e53a33e370917a6`,
and
`1dd7518a8634e46a71981f0306abdbfd3e2d67ed4f4da197e210e9cc1adccb26`.

Every interval crosses zero. The dependence-aware point-estimate comparison is
therefore inconclusive. More importantly, the 1,024 identifiers are inferred,
not verified scheduled-series identifiers. This run uses them only as
dependence blocks, so even its inconclusive intervals are sensitivity evidence
rather than verified-series inference. It does not establish superiority or
support promotion.

### Calibration selection and untouched-final transfer

Calibration used no final label. Identity and bounded antisymmetric Platt forms
were fitted on 2,425 validation out-of-sample predictions, then compared on
2,445 selection predictions:

| Calibration form | Validation-fitted intercept | Validation-fitted slope | Selection log loss | Selection Brier | Selection ECE |
|---|---:|---:|---:|---:|---:|
| Identity | 0 | 1 | 0.6056698401127236 | 0.20929139882513176 | 0.014209602820055728 |
| Platt | -0.08377708543215111 | 0.7222428387874306 | 0.6136676739251842 | 0.2124174774467876 | 0.050845193738973594 |

Identity won the predeclared selection log loss. It was refitted and frozen on
the combined 4,870 validation-plus-selection out-of-sample ledger, whose last
event was `2026-04-29T22:37:59Z`, under calibration model version
`52ce91433d82ea6541d1`. Untouched-final transfer therefore left every candidate
probability unchanged: final log loss remained `0.630049113673521`, Brier
remained `0.2196713224663047`, ECE remained `0.03635973592096003`, and all
raw-minus-calibrated deltas were exactly zero. A final-label counterfactual
regression check confirms that final outcomes cannot alter the selected
calibration form, frozen coefficients, or raw and calibrated final
probabilities.

The final model has 4,178 composition coefficients and 50 nuisance
coefficients. Its sparse fit took 61 recorded optimizer iterations and ended
at gradient infinity norm `2.7991825897e-6`. Exact side-swap antisymmetry,
offset identity, probability bounds, no-future-leakage, sparse-design, and
explicit-unknown-state checks all passed.

Only 168 final maps used a fitted patch deviation; 2,309 maps were on patches
unseen before the final gate and explicitly fell back to global composition
terms. Forty-two final maps contained at least one role/champion identity not
seen in pre-final fitting, and 62 contained at least one seen but
support-thresholded role/champion identity. Their corresponding terms remained
zero and were exposed in prediction metadata. This is a material limitation,
not evidence that those champions have neutral real effects.

### Numerical preflight and immutable source evidence

The first attempt stopped before emitting any gate score: L-BFGS relative-loss
convergence left gradient norm `9.66183e-4`, above `1e-4`. Tightening only the
solver relative-loss tolerance from `1e-12` to `1e-15` completed validation
and selection but stopped the final refit at `1.18382e-4`. A deterministic
Newton-CG polish from the unchanged convex objective and L-BFGS solution then
met the original `1e-4` bound. No split, candidate, outcome, feature,
regularization value, or acceptance bound changed.

| Partition | SHA-256 |
|---|---|
| `team_games/year=2025/part.parquet` | `72716d0be606d29a4b601d74bd0ef1a63fa80c4a4cd1a81083d50928bbda6f74` |
| `team_games/year=2026/part.parquet` | `ed1ff86d3edcec0f976c5ce89c09b1b3f2c8216f99534727b00a56c7a861a459` |
| `player_games/year=2025/part.parquet` | `1ffe0f3454b503f6e04cd7447b07c4222afada6fbe7376a2eaadd751f926a8fe` |
| `player_games/year=2026/part.parquet` | `7a2ababb0241e76b22178dae9c31fa9f5e59035e20a6de99646a7cf5113b37bc` |
| `maps/year=2025/part.parquet` | `6e9a7be4b421726ab192a3fd985e22b93d505abde6b517eba3e8b36f2385f707` |
| `maps/year=2026/part.parquet` | `151d7f29640e533301946c0e19039788a2acf10d0dfd0fb002e6e4346fd99acb` |

Relevant methodological anchors are:

- Gneiting and Raftery, “Strictly Proper Scoring Rules, Prediction, and
  Estimation” (2007), DOI
  [10.1198/016214506000001437](https://doi.org/10.1198/016214506000001437).
- Choe and Ramdas, “Comparing Sequential Forecasters” (2021),
  [arXiv:2110.00115](https://arxiv.org/abs/2110.00115).
- Maystre, Kristof, and Grossglauser, “Pairwise Comparisons with Flexible
  Time-Dynamics” (2019),
  [arXiv:1903.07746](https://arxiv.org/abs/1903.07746).

These references support proper forecast comparison, chronological forecasting
evidence, and time-varying team strength. They do not validate this particular
League of Legends feature set; only untouched population evidence can do that.

## What this design does not establish

- A residual association is not a causal champion effect. Draft choices remain
  endogenous to teams, players, opponents, patch strategy, and tournament
  context.
- A fixed offset prevents the optimizer from directly reallocating fitted team
  strength to champion coefficients, but an omitted, misoriented,
  miscalibrated, or noisy team offset can leave residual confounding.
- The raw neutral-context score is not a claim about a specific roster or
  organization and must not be labeled as a standalone real-world win
  probability.
- Champion, patch, league, draft-mode, and dependence identities are assumed
  to have been canonically resolved upstream. This module rejects malformed
  states but does not decide aliases or reconstruct series.
- Sparse ridge coefficients do not provide post-selection uncertainty.
  Selection-aware intervals, stability checks, coefficient ablations, and
  future-patch replication remain required.
- ECE depends on binning and sample size. It is a descriptive companion to log
  loss and Brier score, not proof of calibration.
- Unknown champions and patches use an explicit zero-deviation fallback; the
  model does not extrapolate their mechanics.
- The circular block intervals use inferred, unverified clusters only as
  dependence blocks. Verified scheduled-series identifiers, alternative
  predeclared block lengths, and/or a valid sequential comparison are still
  required for stronger inference.
- The synthetic regression tests establish software invariants, not empirical
  model validity.
- The corrected-pack tournament supplies descriptive untouched-gate evidence,
  not a superiority claim. No production artifact, API integration, or public
  wording change is included here.
