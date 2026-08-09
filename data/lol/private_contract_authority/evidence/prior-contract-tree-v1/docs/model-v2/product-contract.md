# Product contract

## Product position

Scryglass is an authored League of Legends research publication with analytical
tools. Essays establish the questions and methods; ratings, match exploration,
Draft Score, the sandbox, and tier lists let readers inspect the evidence.

The publication should feel confident, clear, and restrained. The main views
show the result, interval, date, and compact labels. Explanation belongs in
details and Methodology, not in repeated instructional paragraphs.

## Audiences and jobs

| Audience | Primary job |
|---|---|
| Fan | Understand which players, exact rosters, and drafts the model rates highly |
| Journalist | Verify a claim, date, conditioning set, and reproducible source |
| Analyst | Compare rosters, inspect composition structure, test draft continuations |
| Coach | Explore flex sets, future responses, and roster-specific draft fit |
| Author | Publish a versioned argument whose live figures can update without changing its claim |

The editorial and analytical halves have equal status. Public essays are
written by a selected author group; there is no public article submission.

## Initial public surfaces

1. **Articles** — selected-author essays with dated, versioned quantitative
   inserts and a stable claim record.
2. **Player ratings** — role, league scope, rating, 95% interval, status, and
   last update.
3. **Team ratings** — exact active roster, league scope, rating, 95% interval,
   status, and last update.
4. **Match explorer** — immutable forecast, actual outcome, optional separately
   labeled hindsight, Team Rating comparison, and Draft Score.
5. **Draft sandbox** — neutral or identity-selected partial draft with explicit
   flex role sets, signed strategic-response adjustment, and ranked legal
   continuations.
6. **Tier lists** — one role, one league, and that league's current patch;
   refreshed after each completed eligible match and limited to champions
   already played in that cell.
7. **Methodology and reproduce** — estimands, sources, version manifest,
   evaluation, and only the artifacts approved by the publication matrix.

An authenticated product may expose additional approved analyses after the
post-C4 user decision. Authentication never changes an estimand and does not
by itself authorize exposure of private weights, licensed rows, or source data.

## Primary output semantics

| Public label | One-sentence interpretation |
|---|---|
| Player Rating | Posterior-mean role-adjusted individual contribution in the visibly labeled league or bridged-global scope, estimated from information available at `as_of`; League Rating is never included as player skill. |
| Team Rating | Current neutral-draft strength of the exact five-player main roster, built from player states in the same visibly selected league or global scope, estimated at `as_of`. |
| League Rating | Cross-league adjustment shared by a tier-1 circuit, learned from international bridges and partial pooling. |
| Draft Score | Out of 100, the model-estimated map-win probability for side A under this draft after equalizing baseline roster and league strength and neutralizing in-game side advantage. |
| 95% range | The model's 95% interval for the named estimand; it is not a range of possible match scores. |
| Evidence | Separate posterior displacement, precision, and source/context coverage diagnostics for the terms used here; not games played or a correctness probability. |
| Reliability | How well this model version calibrated on comparable unseen cases. |
| Settled | The rating is precise and stable at the model's demonstrated resolution. |
| Provisional | The output relies materially on broad priors or has not met the settled rule. |
| As of | No information after this time was allowed into the output. |
| Forecast | Stored before the event began. |
| Current analysis | Ad-hoc as-of analysis with no claim that it was sealed before an event. |
| State snapshot | Current/as-of rating or tier-list artifact, not an event forecast. |
| Hindsight | Recomputed after the event and not comparable to the stored forecast. |

Public copy must not use “bet,” “odds,” “market,” “under,” “over,” “lock,”
“edge” as a gambling cue, “confidence” as an unexplained scalar, or internal
module/function names. “Advantage” and “probability” are permitted.

## Neutral, contextual, match forecast, and hindsight

These outputs are different and must never be merged under one label.

### Neutral Draft Score

Conditions on champions, legal roles or role sets, one competition scope,
patch, draft protocol, and action order. It excludes all team and player
identity. This is the diagnostic used to compare composition structure for
exchangeable rosters.

### Contextual Draft Score

Adds player-champion-condition response and team policy for the selected exact
rosters. It removes the rosters' draft-independent strength difference before
scoring. Therefore a stronger team does not automatically “win draft.”

When valid team/player identity is supplied, **contextual Draft Score is the
principal Draft Score** and neutral Draft Score is a labeled comparison. When
identity is absent, the sandbox is explicitly **Neutral**.

Neutral is available only when identity was intentionally omitted. If a user
selected identity and its required roster, player, policy, or freshness
contract fails, contextual Draft Score is unavailable; the service must not
quietly replace it with neutral.

At an international event, Draft Score uses one named event/meta competition
scope shared by both sides (for example MSI or EWC). It never chooses one
team's domestic league as the draft environment.

### Partial Draft Score

For a nonterminal draft, Draft Score is the projected standardized map-win
probability under the exact registered future-response policy, after accounting
for committed picks and remaining options and applying the same baseline
strength/side standardization as the terminal score. Its observed evaluation
target is eventual map outcome, not a binary “draft advantage” label.

Probability wording is approved separately by prefix/slot stratum for the exact
served search policy, temperature, approximation, and transform. Terminal
calibration does not approve partial probability wording. A failed prefix gate
returns unavailable and cannot expose a probability-like index or
probability-worded recommendations.

Historical prefix replay validates only the observed behavior policy. A served
policy that changes future responses—such as soft minimax—needs prospective
on-policy evidence or a valid sequential off-policy evaluation with adequate
overlap and diagnostics. Otherwise its Partial Draft Score is unavailable,
even if its terminal model is calibrated.

For a zero-play/new champion without logged-policy support, the sandbox may
offer a separate **Archetype extrapolation** research view: ordinal candidates
and wide uncertainty only. It is not Partial Draft Score and shows no 0–100,
probability/advantage, or Reliability label. It may be Neutral or use a selected
exact fresh roster; the contextual form may use general player strength and
broad archetype fit, never a fabricated exact player-champion residual.

### Match forecast

A separate pre-match probability may combine Team Rating, League Rating,
contextual draft fit, and any other registered pre-match features. It is
labeled **Match forecast**, not Draft Score. In-game side advantage may enter
this forecast if validated; it may not enter the pure draft estimand.

### Hindsight

Hindsight may resolve actual roles, corrected rosters, patch metadata, or use a
later model snapshot. It must have its own `as_of`, model version, and lineage.
It must not overwrite, backfill, or masquerade as the immutable forecast.
Post-draft game events never enter Draft Score, even in hindsight.

## Display rules

- Show one principal number, its interval/status, scope, and `as_of`.
- Use exactly two decimals for probability-point differences in research
  figures; use fewer digits on compact public cards where resolution warrants.
- Sort equal posterior means by lower posterior uncertainty, then stable entity
  ID. Never manufacture precision to break a tie.
- Show active roster names with Team Rating.
- Show competition scope and patch with Draft Score; show league and patch with
  tier lists.
- Show `Neutral` or `Contextual` beside every Draft Score.
- A missing or stale required input produces an unavailable state, not a
  neutral-looking number.
- Draft Score receives its probability wording only when the exact transform
  used by the serving path passed the calibration and estimator-identity gates.
  An unpromoted transform is unavailable, not a probability-like index.
- Partial Draft Score additionally requires approval for the applicable
  prefix/slot stratum and exact search policy/temperature/transform.
- “SOTA” is absent until the benchmark promotion rule is met and documented.

## Article self-healing

An article consists of immutable prose claims plus versioned typed inserts.
Quantitative inserts and source-backed patch-mechanics inserts use separate
schemas. A mechanics insert identifies patch, entity/mechanic, value or rule,
units where applicable, authoritative source snapshot, availability time, and
a semantic signature. An insert may update automatically only if:

1. its estimand and conditioning set are unchanged;
2. the replacement artifact passes all promotion gates;
3. the visible date, model version, and revision note change;
4. the original cited result remains available in article history; and
5. the update does not reverse the prose claim beyond its registered tolerance;
   and
6. for mechanics, the semantic signature is unchanged—not merely the displayed
   value or source timestamp.

If any condition fails, the insert freezes and the author is asked for a new
article revision. A changed item rule, interaction, timing, unit, or patch
meaning always requires author review. Self-healing never silently rewrites
prose or updates a mechanically incompatible claim.

## Access and publication

Method definitions and benchmark summaries are public candidates, subject to
review. Source rows, code, features, weights, and artifacts require both an
allowing source-by-source publication-matrix decision and explicit user
approval after the C4 costed preview. Credentials, tokens, licensed private
feeds, user records, and any artifact from which those can be recovered are
always private. Public results and authenticated access never imply public
weights or licensed training rows.

The post-C4 decision packet reports measured training time, runtime latency,
storage, refresh compute, hosting cost, publication/licensing constraints, and
analyst need. The user then chooses refresh budget/cadence, public versus
authenticated advanced scope, and whether any code, data, or weights may be
open. Until that choice, advanced/auth/open-source scope remains blocked rather
than inferred.

Known teams in the sandbox always use their exact active rosters. A custom five,
if later user-approved, is labeled **What-if roster context** and never called
Team Rating.

Live scoring is excluded only from the initial cycle. A future live phase would
require a new contract, data-availability proof, calibration, costs, and
explicit user approval; this contract does not permanently prohibit it.
