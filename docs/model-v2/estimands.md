# Estimands

All public numbers are posterior summaries of explicitly named quantities.
They are descriptive/predictive estimates, not causal effects.

Let

\[
\sigma(x)=\frac{1}{1+e^{-x}},\qquad
c_E=\frac{400}{\log 10}.
\]

An Elo-scale difference \(\Delta R\) maps to an expected result by

\[
\Pr(A\text{ wins})=\sigma(\Delta R/c_E)
=\frac{1}{1+10^{-\Delta R/400}}.
\]

The display anchor is 1500. The anchor changes presentation, not probability.
For uncertain ratings, this Elo curve is the registered **plug-in display
mapping at fixed rating values**. A predictive match probability integrates
over the joint rating posterior,
\(\mathbb E[\sigma((R_A-R_B)/c_E)\mid\mathcal D]\), and generally is not equal
to the sigmoid of the posterior-mean difference.

## Player Rating

Player Rating has two distinct, visibly labeled scopes. Neither adds League
Rating to a player.

For player \(p\), role \(r\), league \(L\), and time \(t\), let
\(u^{league}_{p,r,L,t}\) be the player's latent contribution and let
\(w^{ref}_{r,L,t}>0\) be the frozen reference-policy role weight. The identified
display contribution is the role-specific reference intervention
\[
s^{league}_{p,r,L,t}
=\kappa w^{ref}_{r,L,t}
\left(u^{league}_{p,r,L,t}-u^{ref}_{r,L,t}\right).
\]
This scaling makes one displayed player-logit point equal one reference-roster
replacement-logit point for every role. Define:

\[
R^{player,league}_{p,r,L,t}
=1500+c_Es^{league}_{p,r,L,t}.
\]

For a player on the active main roster of a tier-1 circuit that is
**structurally globally eligible** at \(t\), let \(s^{global}_{p,r,t}\) be the
analogously reference-scaled contribution on the bridged common player scale:

\[
R^{player,global}_{p,r,t}=1500+c_Es^{global}_{p,r,t}.
\]

The global player scale is identified jointly through international
competition, time-safe player transfers, opponent/roster adjustment, frozen
centering/reference constraints, and hierarchical partial pooling. International
results alone identify only total roster-plus-league strength; mobility,
auxiliary channels, and design-rank/reference sensitivity must support the split
between player skill and League Rating. League Rating is a separate
nuisance/environment term in the outcome model; it is never added to,
subtracted from, or relabeled as individual skill. Weak bridge or decomposition
support widens the global Player Rating interval or makes the global output
unavailable. Tier-2/tier-3 players remain league-scoped unless their active
roster later becomes structurally eligible.

**Public number:** the posterior mean of the selected rating estimand. There is
no separately writable display `rating`; the posterior mean is the rating.

**League literal interpretation:** within the displayed league scope, replacing
the same-role reference player with this player changes an otherwise identical
reference roster's neutral-draft log-odds by
\((R-1500)\log(10)/400\), before a specific team's policy adjustment.

**Global literal interpretation:** on the bridged eligible-player scale, the
same replacement interpretation holds against the displayed global reference
population, with league environment effects controlled separately.

**Conditions on:** selected scope, role, observation time, opponent and roster
strength, champion/draft context, role-normalized performance channels,
resource allocation only where the registered identified joint
resource-to-performance channel is selected, bridge support for global scope,
and the registered dynamic prior.

**Does not condition on:** future matches, current-map post-match statistics,
public popularity, or raw champion game volume as “comfort.”

The scalar rating represents general individual contribution in its named
scope. A separate player-champion conditional response describes how that
contribution changes under ally/enemy draft conditions.

The 1500 anchor and 400-point curve give ratings the same expected-result
meaning as chess Elo: a 400-point difference maps to the registered logistic
expected-result probability. Scryglass does not use or claim FIDE's update
rules, game-count coefficients, or federation pool semantics.

## Team Rating

For exact active roster
\(\rho_t=(p_{top},p_{jng},p_{mid},p_{bot},p_{sup})\), define scope-specific
roster strength

\[
T^q_{\rho,t}
=A^q(s^q_{\rho,t},w_{\rho,t})+\gamma^q_{\rho,t},
\]

where \(q=league(L)\) or \(global\), \(A^q\) is the registered
policy-weighted aggregation of the five player states in that same scope, and
\(\gamma^q_{\rho,t}\) is a shrunken lineup-synergy term. The aggregation uses
the same role-specific reference scaling as Player Rating and is centered so a
reference roster has value zero. Under the candidate weighted parameterization,
\[
A^q(s^q_{\rho,t},w_{\rho,t})
=\sum_r \frac{w_{\rho,r,t}}{w^{ref,q}_{r,t}}s^q_{p(r),r,t},
\]
so replacing one player under the reference policy changes \(A^q\) by exactly
the displayed player-logit difference. This reference intervention holds
lineup synergy at zero; a realized new exact roster receives its own uncertain
\(\gamma^q\).

Within a league:

\[
R^{team,regional}_{\rho,L,t}=1500+c_ET^{league(L)}_{\rho,t}.
\]

For a tier-1 global comparison in league \(L\):

\[
R^{team,global}_{\rho,L,t}
=1500+c_E(T^{global}_{\rho,t}+\lambda_{L,t}),
\]

where \(\lambda_{L,t}\) is League Rating on the logit scale.
Thus the roster term in a global Team Rating is built from the five global
Player Rating states; a regional roster state is not silently carried into the
global equation before adding \(\lambda_L\).

**Public number:** posterior mean for the selected scope. A globally eligible
tier-1 roster uses the global definition; a league view uses the regional
definition. The scope is always visible.

**Literal interpretation:** the rating difference between two exact rosters in
the same displayed scope maps through the Elo formula to a plug-in expected
result on a neutral draft and neutral in-game side. The Match Forecast
integrates rating uncertainty rather than treating posterior means as known.

**Conditions on:** the five active main-roster players, their roles and
histories, learned team policy, lineup synergy, time, and—only for global
scope—League Rating.

**Does not condition on:** the organization's previous rosters, the actual
match draft, in-game state, or tier-2/tier-3 international extrapolation.

New rosters are immediately estimable from player histories and broad priors;
they are normally provisional because lineup synergy and policy are uncertain.

## League Rating

For tier-1 league \(L\),

\[
R^{league}_{L,t}=1500+c_E\lambda_{L,t}.
\]

**Literal interpretation:** relative to a league rated 1500, this is the shared
cross-league adjustment applied when otherwise equal rosters meet on a neutral
draft.

It is identified through international/interregional bridges plus hierarchical
partial pooling. It is a league environment effect, not a team's international
component and not proof that every team in the league is equally strong.

Tier-1 eligibility is structural: the circuit can send teams to a designated
international event. Statistical connectivity is separate and may be weak,
which widens intervals. Tier-2 and tier-3 teams are not globally ranked.

## Terminal Draft Score

Let \(x\) be a completed legal draft state with champions, roles, patch, league,
protocol, and action order. Let \(\eta_D^{neutral}(x)\) be the identity-free
draft component of a map-outcome logit after setting baseline roster and league
strength differences and in-game side advantage to zero.
Let \(g_T\!:\mathbb R\to(0,1)\) be the exact registered terminal calibration
transform evaluated and served for this model version. Strictly open support is
required for any terminal transform consumed by partial search, so the
calibrated logit below is always finite. The transform obeys
\(g_T(-z)=1-g_T(z)\) and is monotone nondecreasing, so a larger raw draft logit
cannot receive a lower calibrated probability.

\[
p^{neutral}_{std,A}(x)=g_T(\eta_D^{neutral}(x)),\qquad
DS^{neutral}_A(x)=100\,p^{neutral}_{std,A}(x),\qquad
DS^{neutral}_B(x)=100-DS^{neutral}_A(x).
\]

For exact rosters \(\rho_A,\rho_B\), let \(f_D\) be roster-specific champion and
composition fit after setting draft-independent roster strength difference to
zero:

\[
\eta_D^{context}(x,\rho_A,\rho_B)
=\eta_D^{neutral}(x)+f_D(x,\rho_A,\rho_B).
\]

\[
p^{context}_{std,A}=g_T(\eta_D^{context}),\qquad
DS^{context}_A=100\,p^{context}_{std,A}.
\]

**Literal interpretation:** out of 100, the model-estimated map-win probability
for side A under this draft after equalizing baseline roster and league
strength and neutralizing in-game side advantage.

The observed event is the map outcome \(y\), not a latent binary event that a
draft “has the advantage.” Draft Score is a standardized conditional
prediction. It is not a causal effect of choosing the draft.

**Neutral conditions on:** draft contents, legal role assignment, league,
patch, protocol, and draft order.
**Contextual additionally conditions on:** exact players, their conditional
champion responses, and team policy.
**Does not condition on:** baseline Team Rating difference, in-game side
advantage, post-draft events, or future outcomes.

The contextual model may use player-champion response, team policy, and
identity-specific interaction structure, but its standardized scoring
calculation sets the two rosters' draft-independent Team Rating difference and
the League Rating difference to exactly zero. A separate Match Forecast is the
only output allowed to add those differences back.

The 0–100 probability interpretation is available only after L2 verifies the
exact estimator and score transform served by L10, including calibration on
untouched holdouts. If that gate fails, canonical Draft Score is unavailable;
serving the uncalibrated transform as a probability-like index is prohibited.
Neutral score is used only when identity is intentionally omitted. Missing or
stale requested identity/context fails closed and cannot trigger neutral.

For an international event, both sides use one named competition/meta scope
for that event and patch (for example MSI or EWC), not either team's domestic
league. The scope and its reference distribution are stored once on the draft
record and apply symmetrically to both sides. Domestic league remains roster
provenance and may enter a separate Match Forecast, never the international
Draft Score standardization.

An empirical average draft-order coefficient may be learned only where protocol
variation identifies it separately from game side. Perfectly collinear
draft-order/game-side data make that coefficient unavailable and force its
served contribution to zero by standardization convention, not as a finding of
no effect. If bounding the admissible split of the identified combined
side/order effect materially changes the score or calibration, the affected
Draft Score endpoint is unavailable. Structural action-tree value from the legal
action tree remains valid as a different quantity. Side-swap plus transformed
order must complement the score exactly.

## Partial Draft Score and strategic response

Let \(s\) be a legal partial state and \(\mathcal C(s)\) its legal terminal
completions. Let \(U(c)\) be the standardized terminal map-win logit for
completion \(c\):
\(U(c)=\operatorname{logit}(g_T(\eta_D(c)))\). Because \(g_T\) maps to
\((0,1)\), \(U(c)\) is finite.

The **committed value** is the reference-policy standardized value

\[
M(s)=\mathbb E_{c\sim q_{\pi_{ref}}(\cdot\mid s)}[U(c)],
\]

where \(q_{\pi_{ref}}\) is the completion rollout induced by the same
registered, time-safe baseline policy \(\pi_{ref}\) at every remaining
decision, fitted without using the evaluated outcome.

The **strategic value** \(V(s)\) is the registered soft-minimax recursion over
future legal actions. The **strategic-response adjustment** is the signed
difference

\[
A_{resp}(s)=V(s)-M(s).
\]

It is not a nonnegative amount of “optionality.” It can be negative when the
opponent controls the next decision or the served response policy is harsher
than the baseline rollout. The acting side, baseline policy, and policy version
are part of its literal label. Flex value below remains a separate controlled
role-set comparison.

Let \(k(s)\) be the registered prefix/slot stratum and
\(g_{P,k}\!:\mathbb R\to(0,1)\) the exact prefix transform calibrated with the
same response policy, search method, temperature, and approximation settings
served at that stratum. It must obey
\(g_{P,k}(-v)=1-g_{P,k}(v)\) and be monotone nondecreasing. The canonical
partial score is

\[
PDS_A(s)=100\,g_{P,k(s)}(M(s)+A_{resp}(s))
=100\,g_{P,k(s)}(V(s)).
\]

**Literal interpretation:** out of 100, the projected standardized map-win
probability for side A under the registered future-response policy, after
accounting for committed picks and remaining options, equalizing baseline
roster and league strength, and neutralizing in-game side advantage.

The observed event used to evaluate this probability is the eventual map
outcome, not an unobserved binary “draft advantage” event. Terminal calibration
alone does not authorize this prefix probability. L2 must approve the exact
served policy/temperature/transform for \(k(s)\); otherwise Partial Draft Score
is unavailable in that stratum and neither the score nor recommendations may
use calibrated-probability wording.

Historical prefix replay is direct evidence only for the observed behavior
policy. If the registered served policy differs, this literal probability
requires prospective on-policy evaluation or a valid sequential off-policy
evaluation with defensible consistency, exchangeability, positivity, and
behavior-policy assumptions plus effective-sample-size/weight and sensitivity
diagnostics. Without that evidence, the counterfactual policy cannot inherit
probability wording from observed continuations.

Zero-play/new-champion actions generally have no logged-policy positivity.
Archetype transfer may therefore power a separately labeled
**Archetype extrapolation** sandbox view with ordinal candidates and wide model
uncertainty, but never canonical Partial Draft Score, a 0–100 value,
probability/advantage wording, or a Reliability label for that unsupported
policy stratum. It is a research view, not a successful partial-score payload.
It may be neutral or contextual. Contextual extrapolation requires the exact
fresh rosters and may use general player strength plus broad team/player
archetype-fit priors, but never an unsupported exact player-champion residual.

At every legal terminal state \(c\), the partial evaluator delegates to the
canonical terminal transform:
\[
PDS_A(c)=DS_A(c)
\]
for identical inputs and `model_version`, before and after display rounding.

Flex roles are explicit sets. Their value is the difference between \(V(s)\)
with the declared legal role set and the policy-weighted value after fixing the
pick to one role. Flex is never inferred from champion popularity alone.

Recommendations rank legal actions by the posterior distribution of
\(V(s\to a)-V(s)\), report downside as well as mean change, and must not call an
action better when its interval cannot distinguish it at the model's resolution.

## Tier Value and counterability

For champion \(c\), role \(r\), league \(L\), current league patch \(P\), and
as-of time \(t\), let \(z\) denote allied/context state excluding the opponent's
next response and \(a\) a registered plausible opponent response. Define the
response-specific standardized replacement value

\[
\Delta_{c}(z,a)
=100\left[p_D(c,r,z,a)-p_D(c_{ref},r,z,a)\right].
\]

The model-standardized incremental value is

\[
IV_{c,r,L,P,t}
=\mathbb E_{z\sim G_{L,P,t},\,
a\sim R_{ref}(\cdot\mid z)}[\Delta_c(z,a)],
\]

where \(G\) is the registered time-safe distribution of allied compositions,
exact-roster conditional fit, and draft positions, \(R_{ref}\) is the registered
reference response policy, and \(c_{ref}\) is the role-patch reference mixture.
Within one league-patch-role artifact, \(G\), \(R_{ref}\), and the
role-reference mixture are frozen and common across champions. Every
replacement pair is restricted to common legal support: no duplicate or banned
champion and no role/protocol-invalid composition may enter either side of the
comparison.
Draft-independent Team Rating is held equal inside this standardization; it is
included only as a nuisance offset when fitting against observed outcomes. This
is a probability-point difference, not raw win rate and not a causal effect.

Define counterability as plausible-response regret:

\[
C_{c,r,L,P,t}
=\mathbb E_{z\sim G_{L,P,t}}\left[
\left(
\mathbb E_{a\sim R_{ref}(\cdot\mid z)}[\Delta_c(z,a)]
-Q_{\alpha}\!\left(
\Delta_c(z,a)\mid a\sim R_{plaus}(\cdot\mid z)
\right)
\right)_+
\right],
\]

where \(Q_\alpha\) is the registered lower-tail quantile,
\((x)_+=\max(x,0)\), and the tail level and plausible-response policy are chosen
inside nested validation and frozen in the model manifest. Thus
counterability is explicitly nonnegative and conditions a response-specific
quantity rather than the already-marginal \(IV\).

Primary **Tier Value** is

\[
TV=IV-\lambda_C C,
\]

where \(\lambda_C\ge0\) is selected only if it improves registered out-of-sample
ranking/prediction performance; otherwise \(\lambda_C=0\) and counterability is
shown separately. No hand-tuned tax is permitted.

Only champions with at least one verified completed eligible appearance in the
exact league-patch-role cell can appear. Archetype transfer can power sandbox
recommendations for zero-play champions, but cannot place them on a tier list.

## Evidence diagnostics

Evidence is a structured object, not one permanent scalar or a synonym for
games played. It keeps four concepts separate:

1. **Posterior displacement/information** — how the prediction-relevant
   posterior differs from its registered prior. Candidate diagnostics may
   include unnormalized KL divergence, Wasserstein distance, or standardized
   posterior displacement; none is called a fraction of uncertainty removed.
2. **Precision/interval contraction** — how posterior dispersion or interval
   width changed relative to the registered prior/reference, reported
   separately from displacement.
3. **Source/context coverage** — which required source families, interaction
   contexts, identity terms, bridge paths, and fallback levels actually support
   this estimate.
4. **Heldout reliability** — proper-score, calibration, interval-coverage,
   resolved-cluster, and OOD diagnostics from unseen data. This remains the
   separate Reliability object below.

R-20 and L2 select and validate the first three methods by output and stratum
before compact labels are allowed. No formula, normalization, or aggregation
weight is hard-coded by this contract. Game count, pick rate, popularity, or
champion volume may be descriptive provenance but cannot define evidence.

**Literal interpretation:** which parts of this estimate moved beyond their
priors, how precise they became, and which relevant sources/contexts support
them. It is not a correctness probability.

## Predictive reliability

Reliability is a validation object, not an impressionistic scalar. The manifest
contains a frozen, hashed, total mapping from output context and OOD flags to
one validation stratum or `unrated`; runtime cannot choose a friendlier
“nearest” stratum. Every rated label records that mapping hash and selected
stratum, log-loss and Brier skill versus named baselines, calibration intercept
and slope, empirical interval coverage and nominal level, effective resolved
cluster support under the registered dependence design, and out-of-distribution
state. No mapped match returns `unrated`. `high` cannot be emitted with missing
diagnostics, zero effective cluster support, an unapproved transform, or an OOD
state.

If a compact public label is required, `high`, `moderate`, `limited`, and
`unrated` boundaries must be derived and frozen from baseline-relative
validation before the final holdout is opened. The detailed object remains
available.

## 95% intervals

Ratings and draft logits use central 95% posterior intervals transformed to
their display scale. Empirical coverage is tested on simulation and untouched
forecast cells as specified in [evaluation-contract.md](evaluation-contract.md).
An interval is never described as exact unless the reported validation supports
nominal coverage for the applicable stratum.

## Stability and settled status

For each model version, L2 estimates a **rating resolution**
\(\delta_{res}\): the smallest Elo-scale difference for which the lower bound
of the registered series-preserving, higher-level dependence-aware
out-of-sample pairwise discrimination exceeds its matched production baseline.
The derivation, uncertainty, and scope are recorded in the manifest.

A rating is `settled=true` only when all are true:

1. posterior precision at the registered resolution is strictly greater than
   95%:
   \(\Pr(|R-\bar R|\le\delta_{res})>0.95\);
2. its central 95% interval width is at most \(2\delta_{res}\), with
   `lower <= posterior_mean <= upper`;
3. over the registered stability window, posterior probability
   \(\Pr(|R_t-R_{t-w}|\le\delta_{res})>0.95\);
4. the entity is active, current for the displayed scope, and—when global
   scope is requested—structurally globally eligible;
5. every required input is complete and fresh;
6. no material fallback (including F5) or out-of-distribution flag applies;
   and
7. interval coverage for its validation stratum passed the release gate.

The stability window is evaluated in [research-register.md](research-register.md)
and should correspond to meaningful competition exposure, not a fixed arbitrary
day count. Missing one regional tournament does not by itself make a roster
inactive.

Posterior-mean ties are broken by lower posterior uncertainty, then stable ID;
the tie-break never changes the rating itself.
