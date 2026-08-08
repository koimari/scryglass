# Mathematical contract

This document specifies the candidate family and the invariants every candidate
must satisfy. It does not predeclare a winner. Hyperparameters, inference
engine, decay law, set-function family, and search policy are research choices
selected under [evaluation-contract.md](evaluation-contract.md).

## 1. Joint probabilistic frame

For completed map \(g\) beginning at \(t_g\), let \(y_g=1\) when side A wins.
All features are snapshots with availability time strictly before \(t_g\) for
a forecast fit.

\[
y_g\sim \operatorname{Bernoulli}(\sigma(\eta_g))
\]

\[
\eta_g =
\underbrace{T^{q_g}_{\rho_A,t_g}-T^{q_g}_{\rho_B,t_g}}_{\text{exact rosters in one scope}}
+\underbrace{\lambda_{L_A,t_g}-\lambda_{L_B,t_g}}_{\text{league bridge}}
+\underbrace{\eta_D^{context}(d_g,\rho_A,\rho_B)}_{\text{draft fit}}
+\underbrace{b_{game}(g)}_{\text{registered non-draft pre-match terms}}.
\]

Here \(q_g=league(L)\) for a regional contest and \(q_g=global\) for a
cross-league/global contest. A global roster term uses global player states
before \(\lambda_L\) is added; the model never inserts a league-relative roster
state into the global equation. The league difference is zero for a same-league
regional contest.

This outcome equation trains and evaluates the pieces jointly or by a
cross-fitted modular approximation. It does not define Draft Score: Draft Score
sets exact-roster strength, League Rating difference, and in-game-side terms to
zero. A Match Forecast may use the complete equation.

Every observation stores the prediction-time snapshot. Post-map measurements
may update states for later maps but never the forecast for the same map.

## 2. Dynamic Player Rating

Each player has a time-varying raw latent general contribution \(u\) on a
league-relative scale and, when structurally eligible and statistically
bridged, on a common global player scale. League environment effects remain
separate nuisance terms and are never relabeled as player skill.

Candidate dynamics include a continuous-time random walk, a mean-reverting
state model, and preregistered season/calendar-boundary shock or reset variants:

\[
u_{p,r,t+\Delta}=m_r+
\phi_r(\Delta)(u_{p,r,t}-m_r)+
\epsilon,\qquad
\epsilon\sim\mathcal N(0,q_r(\Delta)).
\]

A random walk is the special case \(\phi=1\). A half-life may be displayed only
for a promoted mean-reverting model and must be the fitted implication of
\(\phi\), not a hand-written recency setting. `season_id` and derived
`calendar_year` are distinct inputs. No-reset/carry-over is the operational
default until L2 compares it with boundary-shock/reset candidates; neither a
January reset nor permanent carry-over may be inferred from the public year
filter.

### Individual-performance channels

Team outcomes alone weakly identify individuals. The candidate therefore uses
predeclared role-normalized auxiliary outcomes \(z_{g,p,k}\), for example lane
economy, damage conversion, participation, vision, roaming or objective
involvement when source coverage permits:

\[
z_{g,p,k}\sim F_k\!\left(
\alpha_{r,k}
+\beta_{k}u_{p,r,t_g}
+h_{c,r,k}
+u_{draft,k}
+u_{opp,k}
+u_{resource,k},
\theta_k\right).
\]

The source, availability, transformation, missingness, and role normalization
of every channel are preregistered. Outcomes from map \(g\) update the state
only after its forecast is stored.

Same-map resource allocation is endogenous: player strength, team policy,
draft, game state, and realized performance can all affect it. The model never
gives a causal “holding resources equal” interpretation or inserts one aggregate
resource total both as a player control and an independent policy outcome.
R-03 compares a temporally ordered joint resource-to-performance measurement
model with no-resource, lagged/pre-map-policy, and sensitivity variants. When
event ordering is unavailable or the player/policy split is weak, the resource
channel is excluded from Player Rating or allowed to update policy only. Team
outcome remains the anchor. Supports are assessed with role-appropriate
non-resource channels and the team result; low farm cannot directly lower
their latent skill.

For scope \(q\), let \(u^{ref}_{r,q,t}\) and
\(w^{ref}_{r,q,t}>0\) be the frozen time-safe reference player and
reference-policy role weight. The public display contribution is

\[
s^q_{p,r,t}=
\kappa w^{ref}_{r,q,t}
\left(u^q_{p,r,t}-u^{ref}_{r,q,t}\right),\qquad
R^q_{p,r,t}=1500+\frac{400}{\log 10}s^q_{p,r,t}.
\]

Thus one displayed player-logit point has the same reference-roster
replacement meaning in every role; the public player scale is not the raw
\(u\) coordinate. A league Player Rating is centered on its league-role
reference. A global Player Rating is centered on the bridged eligible-player
population and requires structural global eligibility plus bridge diagnostics.
Role-specific diagnostics can remain internal; any public role adjustment must
preserve this literal replacement interpretation and the definitions in
[estimands.md](estimands.md).

## 3. Exact-roster Team Rating and team policy

For exact roster \(\rho\), define a time-varying policy simplex

\[
w_{\rho,t}=\operatorname{softmax}(\alpha^{policy}
+u^{team}_{\rho,t}),
\qquad \sum_r w_{\rho,r,t}=1.
\]

Policy is learned from time-safe resource allocation, role-normalized impact,
and the degree to which role-level deviations explain later team results.
Resource share alone cannot set \(w\). Team-specific deviations shrink toward
the league/role policy prior.

A candidate roster aggregation, using the same display scaling, is

\[
T^q_{\rho,t}
=\sum_{r}\frac{w_{\rho,r,t}}{w^{ref}_{r,q,t}}
s^q_{p(r),r,t}
+\gamma^q_{\rho,t},
\]

equivalently
\(\kappa\sum_r w_{\rho,r,t}
(u^q_{p(r),r,t}-u^{ref}_{r,q,t})+\gamma^q_{\rho,t}\).
Here \(\kappa\) anchors the public Elo interpretation and
\(\gamma^q_{\rho,t}\) is a zero-centered, strongly shrunken exact-lineup synergy
state. Under the reference policy, replacing one player changes the aggregation
\(A^q\) by exactly that player's displayed logit difference. The literal Player
Rating intervention holds lineup synergy at its zero reference; it does not
claim that a realized new lineup's \(\gamma^q\) is unchanged. An equivalent
identified parameterization is allowed only if it produces this same
reference-roster intervention and passes the invariants.

Roster identity is the ordered five-tuple of stable player IDs by role. No
organization-only residual may survive a roster change. A new five immediately
inherits the five player posteriors; its policy and lineup-synergy terms fall
back to broader priors and widen uncertainty.

Policy weights and lineup synergy are separately interpreted only after a
registered identification audit. Policy is anchored by time-safe resource and
non-resource auxiliary channels; \(\gamma^q\) is zero-centered and projected
off the policy-weighted player span. L2 checks within-roster policy variation,
design rank/conditioning, posterior dependence, source removal, and separate
ablations. When the split is weak, the system uses the validated simpler
fallback (for example pooled/equal policy with \(\gamma=0\)) or reports only the
identified total \(T^q\); it may not expose unstable policy and synergy drivers
as separate facts.

### League Rating

Tier-1 league effects \(\lambda_{L,t}\) share a hierarchical dynamic prior and
are anchored by a sum-to-zero constraint or a registered reference league.
International series create likelihood bridges. Structural global eligibility
comes from competition rules; the amount and age of statistical bridge
connection controls uncertainty or availability, not eligibility. The same
rule gates global Player and Team Rating, while League Rating remains a
separate model component.

League Rating is never stored as a team field named `meta`, `international`, or
similar. Regional and global Team Ratings expose the included components
explicitly.

### Player/league identification

International results identify only opposing **total** roster-plus-league
strength unless the player/league split receives additional identifying
structure. The global model therefore freezes all of the following before
evaluation:

- a time-safe bridged-player reference population and role weights that center
  global player states within role;
- a sum-to-zero or fixed-reference constraint for League Rating;
- continuity of the same player's global state through a league transfer,
  while the applicable league environment changes;
- no unconstrained league-by-role intercept inside displayed global player
  skill; auxiliary-channel league/role intercepts remain nuisance terms; and
- a connected-design rank, bridge-strength, mobility, and source-removal
  diagnostic for the player-versus-league decomposition.

Hierarchical shrinkage alone is not identification. If international
matchups, player mobility, and auxiliary channels do not separate the global
player distribution from \(\lambda_L\), global Player and Team Ratings widen
or become unavailable. The system may still publish properly identified
league-scoped Player and regional Team Ratings.

## 4. Champion representation and archetype transfer

Each champion-role-patch representation is

\[
e_{c,r,P}=A_{r,P}z_c+u_{c,r,P}.
\]

- \(z_c\) is a versioned, interpretable multi-label ontology (damage profile,
  range, engage, peel, mobility, wave control, scaling, target access, and
  other validated dimensions).
- \(A_{r,P}z_c\) is the archetype-derived prior contribution.
- \(u_{c,r,P}\) is a champion-specific residual with hierarchical shrinkage.

New or zero-play champions receive an ontology-based posterior with
\(u\) centered at zero and wide uncertainty. This permits sandbox exploration
such as an artillery/poke substitute. It does not permit a tier-list row until
the champion has a verified appearance in that league-patch-role cell.

Ontology labels are not outcomes. They are reviewed against official patch/kit
sources and validated by predictive ablation. The model may learn continuous
embeddings, but every such embedding must be linked to the interpretable prior
and tested against an ontology-free baseline.

## 5. Neutral full-composition model

Let \(C_A=\{(c_i,r_i)\}_{i=1}^5\) and \(C_B\) be valid role-assigned sets.
The identity-free logit is

\[
\eta_D^{neutral}
=o(protocol,actions)
+F(C_A,\Omega,P)-F(C_B,\Omega,P)
+G(C_A,C_B,\Omega,P),
\]

with

\[
F(C)=
\sum_i\phi(c_i,r_i)
+\sum_{i<j}S((c_i,r_i),(c_j,r_j))
+H(C),
\]

\[
G(C_A,C_B)=
\sum_{i=1}^5\sum_{j=1}^5
O((c_i,r_i),(c'_j,r'_j))
+K(C_A,C_B).
\]

Required symmetry:

\[
S(a,b)=S(b,a),\quad
O(a,b)=-O(b,a),\quad
K(A,B)=-K(B,A).
\]

\(H\) and \(K\) are permutation-invariant within role-labeled sets. Candidate
families include a factorization-machine sparse-interaction baseline, low-rank
bilinear terms, a Deep Sets residual, or a Set Transformer residual. Complex
residuals are promoted only when their out-of-sample gain survives the
registered ablations and their uncertainty can be propagated.

The model covers all four allies of every champion, all five opponents, and
whole-composition structure. Same-role opposition is one of 25 cross-team
relations, not the entire explanation.

### Patch and league structure

Champion and interaction terms use hierarchical global → role → league → patch
deviations. Patch effects are dynamic and partially pooled; an unseen patch
falls back through the explicit hierarchy. “Current patch” never means a
hard-coded coefficient from a different patch.

The conditioning scope is one `competition_scope_id`. A regional draft uses
one league-patch scope. An international draft uses one named event/meta scope
(for example MSI or EWC) shared by both sides; it never selects either team's
domestic league as the draft environment.

### Sparse effects

Sparse champion residuals and interactions use hierarchical zero-centered
priors. A regularized horseshoe or validated multilevel Gaussian/low-rank
alternative is admissible. Prior scales and expected effective complexity are
chosen on training/inner-validation data and stored in the manifest. No
minimum-game cutoff may turn an unobserved interaction into a fixed empirical
effect.

### Decomposition identification

For every model version, L2 freezes a time-safe reference distribution over
champions, roles, legal compositions, rosters, and policies. The composition
decomposition uses functional-ANOVA-style constraints under that distribution:
main effects are centered; every pair term has zero weighted marginal in each
argument; \(H\) is orthogonal to the main and ally-pair spans; and \(K\) is
orthogonal to all lower-order cross-team spans. Equivalent explicit projection
constraints are allowed.

These constraints fix the algebraic representation (the parameter gauge); they
do not by themselves create statistical identification on restricted legal
support. L2 additionally checks the legal-support design rank and condition,
posterior dependence, co-occurrence structure, and source/patch-removal
sensitivity. If, for example, two champions always co-occur, their main and pair
effects cannot be reported as distinct learned facts merely because priors
separate them.

When component identification is weak, the model collapses affected terms into
the smallest supported joint residual or exposes only total composition value,
with widened uncertainty and an `unresolved_collinear` interpretation status.
Ledger reconciliation and relation coverage remain exact, but unsupported
main/synergy/counter labels do not. Exact Shapley/Owen allocation may allocate
only an identified \(H\) or \(K\) residual; it cannot manufacture
identification.

## 6. Contextual draft fit

For exact rosters:

\[
\eta_D^{context}
=\eta_D^{neutral}
+\sum_{i\in A}h(p_i,c_i,r_i,C_A,C_B)
-\sum_{j\in B}h(p_j,c'_j,r'_j,C_B,C_A)
+q(policy_A,C_A,C_B)-q(policy_B,C_B,C_A).
\]

The player-champion term is a shrunken conditional response under allied and
enemy draft structure. Raw volume, pick rate, and an unconditioned
player-champion win rate are not valid “comfort” features.

Under the same frozen reference distribution,
\(h\) is centered for each player-role over eligible champion/composition
contexts and \(q\) is centered over eligible roster policies/compositions.
Both are projected off the span of general player/roster strength, League
Rating, and the neutral composition terms. A frozen ordered/hierarchical
orthogonalization or explicit cross-projection also separates \(h\) from \(q\);
L2 checks their joint design rank, posterior dependence, deterministic
player-champion/policy overlap, and source-removal sensitivity. Otherwise a
team policy that always selects one player's champion could exchange the same
direction between \(h\) and \(q\). A weak split exposes only total contextual
fit, with widened uncertainty, not separate player-champion and policy
drivers. A candidate unable to satisfy the broader constraints may be evaluated
as a joint forecast but cannot expose contextual fit as a separately
interpreted Draft Score driver.

For Draft Score, the standardized prediction sets
\(T^q_{\rho_A}-T^q_{\rho_B}=0\),
\(\lambda_{L_A}-\lambda_{L_B}=0\), and the in-game side term to zero. The
contextual terms may say one roster fits this draft better; they may not reward
that roster merely for being stronger. The observed target remains the map
outcome; there is no observed binary “draft advantage” label and no causal
interpretation.

If identity was requested, all exact-roster and context freshness requirements
are mandatory. The neutral model is not a missing-context fallback.

## 7. Draft order and side

`protocol` defines legal actions, bans, pick order, and side transformation.
Every observation also stores, as separate fields, canonical analytical side
(`A`/`B`), actual game side (`blue`/`red`), draft order (`first`/`second`), the
protocol mapping among them, and mapping source/availability. These variables
may coincide in one competition but may not be collapsed into one `side`
column.
An empirical population-average draft-order term \(o_{emp}\) is admissible only
when registered protocol variation or another preregistered source supplies
support that separates draft order from game side. A zero-centered prior does
not identify collinear effects.

Before fitting \(o_{emp}\), L2 must publish:

- support/positivity counts for every protocol × draft-order × game-side cell;
- the relevant design-matrix rank and condition number;
- posterior correlation or equivalent confounding diagnostics; and
- sensitivity under removal of each identifying protocol/source.

If draft order and game side are perfectly collinear, only their combined
population effect is identified. The Draft Score convention sets
\(o_{emp}=0\), records `unavailable_collinear`, and assigns no empirical
advantage to draft order; this is a standardization convention, not evidence
that the true order effect is zero. L2 must bound Draft Score and calibration
under the preregistered admissible decompositions of that combined effect. If
the resulting change is material at the public decision resolution, Draft
Score probability wording—and the affected endpoint—are unavailable for that
protocol. A prior cannot rescue identification. The structural action-tree value
induced by legal pick/ban order is still computed from the action tree and
terminal standardized map-win logits; it is not evidence of a separately
identified population-average order effect.

In-game side features—map geometry, first move, objective access, or historical
blue-side win rate—are prohibited from neutral/contextual Draft Score. They may
enter a separately registered Match Forecast.

For side-swap operator \(\mathcal S\), which swaps rosters, compositions, and
transforms action order under the protocol:

\[
\eta_D(\mathcal Sx)=-\eta_D(x),\qquad
DS_A(\mathcal Sx)=100-DS_A(x).
\]

## 8. Partial-draft graph

A state contains the protocol, canonical/actual-side and draft-order mapping,
ordered pick/ban actions, acting canonical side, available champions, current
role constraints for every picked champion, an append-only role-constraint
revision history, identity mode, competition scope, patch, and as-of snapshot.
A revision records the pick ID, previous and new role set, effective action
sequence, reason/source, and availability. Legal widening, narrowing, and
reassignment are supported without rewriting the original pick—e.g. Swain may begin
`{support,top}`, narrow to `{support}`, and later move to `{mid}` when subsequent
picks make that legal. Edges are legal actions or role-constraint revisions.
Transpositions share a canonical node ID only when current role constraints and
all other legality-relevant state agree; their histories remain auditable.

`terminal` is derived by the protocol validator, not hard-coded false. A
terminal state has a complete legal action sequence and singleton final role
assignment for five champions per side. For terminal \(s\),
\(V(s)=U(s)=\operatorname{logit}(g_T(\eta_D(s)))\), the finite standardized
terminal map-win logit. Every terminal transform consumed by search maps to
\((0,1)\), and the exact prefix transform \(g_{P,k}\) also maps to \((0,1)\).
Both are monotone nondecreasing and obey complement symmetry. For nonterminal
state with legal actions
\(\mathcal A(s)\), a candidate soft-minimax recursion is

\[
V(s)=
\begin{cases}
\tau\log\sum_{a\in\mathcal A(s)}
\pi_0(a\mid s)\exp(V(s_a)/\tau),&A\text{ acts}\\
-\tau\log\sum_{a\in\mathcal A(s)}
\pi_0(a\mid s)\exp(-V(s_a)/\tau),&B\text{ acts}.
\end{cases}
\]

\(\pi_0(\cdot\mid s)=\pi_{ref}(\cdot\mid s)\) is the time-safe fitted baseline
policy whose rollout defines committed value. It is normalized to
sum to one and strictly positive for every legal action included in the soft
recursion; \(\tau\) is selected in nested validation. Hard minimax,
empirical-policy expectation, and risk-aware variants are registered
alternatives, not silent runtime switches. Model-prior support is not a
substitute for logged-policy positivity in off-policy evaluation.

The committed value \(M\), signed strategic-response adjustment
\(A_{resp}=V-M\), flex value, and search coverage follow
[estimands.md](estimands.md). \(A_{resp}\) is acting-side and baseline-policy
dependent and may be negative; it is never labeled as a nonnegative quantity
of optionality. A pruned or sampled search reports the explored mass,
error/bound method, and deterministic seed. It cannot label an approximation as
an exact solution.

Prefix probabilities are calibrated and approved by slot/stratum using the
exact served response policy, temperature, search approximation, and
transform. Terminal calibration is insufficient. If the applicable prefix gate
fails, the partial endpoint fails closed for that stratum. At a terminal state,
the partial evaluator must call the canonical terminal path and return the
identical Draft Score for the same inputs and model version.

Historical prefix replay directly evaluates only the observed behavior policy.
If the served soft-minimax, risk-aware, or recommendation policy differs from
that behavior policy, ordinary outcome calibration is invalid. Probability
wording then requires either prospective/on-policy evaluation or a
preregistered sequential off-policy estimator with defensible consistency,
sequential exchangeability, positivity, and behavior-policy estimation
assumptions. L2 reports effective sample size, weight concentration/truncation,
doubly robust or equivalent sensitivity, and failure by prefix stratum. An
assumption or diagnostic failure makes Partial Draft Score unavailable there;
historical completion replay alone cannot validate the counterfactual served
policy.

For zero-play/new-champion actions without logged-policy positivity, archetype
transfer may produce a separate research-only ordinal **Archetype
extrapolation** view with wide uncertainty. That view is outside the canonical
Partial Draft Score response and may not display 0–100, probability,
advantage, or Reliability wording. It may be neutral or contextual; contextual
research requires exact fresh rosters and may use general player strength plus
broad archetype-fit priors, but no unsupported exact player-champion residual.

Canonical recommendations in approved strata maximize the selected side's
posterior strategic value subject to legal action and role constraints. The API
also returns downside, option diversity, and the best opponent responses.
“Avoids one-dimensional drafts” means it retains distinct high-value legal
continuation classes under the registered diversity measure, not that an
archetype label sounds flexible.

## 9. Uncertainty, calibration, and contribution ledger

Posterior draws or a validated approximation propagate through ratings,
composition terms, contextual fit, and the served score transform. A diagonal
coefficient variance that ignores material covariance is not sufficient unless
L2 demonstrates interval coverage.

Calibration is trained only within nested calibration partitions. The exact
serialized transform used by serving is hashed into the model manifest. If the
served transform differs from the evaluated transform, promotion fails. Every
transform used inside partial search or to publish a partial probability must
have open support \((0,1)\), be monotone nondecreasing, and obey
\(g(-z)=1-g(z)\); otherwise ordering, finite-logit, or side-swap
complementarity fails. A stepwise/isotonic candidate that can emit exactly zero
or one is ineligible unless a finite boundary treatment is itself manifested,
benchmarked, and included in parity tests.

### Exact ledger

Every terminal draft response includes a signed logit ledger:

- protocol/order;
- champion-role main effects;
- all ally-pair effects, split equally across the two participants;
- all 25 enemy interactions, split equally across the two participants;
- whole-team and cross-team residuals;
- contextual player-champion effects;
- contextual policy effects; and
- calibration transform as a separate non-additive presentation step.

When a component split fails its legal-support identification gate, the ledger
uses one signed `unresolved_joint_residual` covering the affected entities and
relations rather than inventing separate main/synergy/counter or
player-champion/policy claims. Coverage metadata still proves that all four
allies and five enemies were modeled, and the grouped ledger still reconciles
exactly.

Analytic terms use analytic allocation. \(H\) and \(K\) use exact Shapley/Owen
allocation over the at-most-ten entities against the registered neutral
baseline, unless the promoted residual has an equally exact additive
decomposition. Approximate attribution is prohibited on the public principal
score.

Before calibration:

\[
\sum_k contribution_k=\eta_D.
\]

The numerical reconciliation tolerance is
\(64\epsilon_{machine}\max(1,\sum_k|contribution_k|)\), avoiding a hand-tuned
semantic tolerance. Side swap negates every signed ledger family and preserves
absolute evidence metadata.

## 10. Fallback hierarchy

Fallbacks are prior levels, not alternate engines. Every used level is emitted
in provenance.

| Level | Available evidence | Required behavior |
|---|---|---|
| F0 | Identified exact player/champion/interaction term | Use posterior exact residual plus hierarchy |
| F1 | Sparse exact term | Shrink toward role-archetype and broader interaction prior |
| F2 | No exact champion evidence | Use role-patch-competition-scope archetype prior |
| F3 | New patch/competition-scope cell | Use pooled role/archetype temporal prior with wider interval |
| F4 | New player or lineup | Use role/league player prior and exact known-player histories; mark provisional |
| F5 | No champion ontology term | Use role-patch neutral prior; evidence coverage reflects the gap |

Fail closed instead of falling back when a required stable ID, exact requested
roster, event time, patch, protocol, model/calibration artifact, or freshness
check is absent. Requested contextual identity may not fall back to neutral.

## 11. Required invariants

All model families must satisfy:

1. one canonical promoted estimator per output/version;
2. every `status=ok` output has complete required inputs, fresh checks, and no
   missing/stale/conflict provenance;
3. strict forecast-time feature availability;
4. exact-roster identity and no sticky organization residual;
5. executable semantic validation of five distinct roster players, one per
   role, and legal terminal draft identities/assignments;
6. required Elo-scale result mapping and 1500/400 display contract in rating
   manifests and artifacts;
7. neutral draft independence from player/team identity;
8. contextual draft baseline-strength equalization and a required labeled
   neutral comparison;
9. no in-game side term in Draft Score;
10. exact side-swap antisymmetry and role-input invariance;
11. all-four-ally and all-five-enemy coverage;
12. hierarchical shrinkage for sparse effects;
13. calibrated exact served transform before probability wording;
14. executable equality/order checks for rating posterior mean,
    score/probability transforms, complements, intervals, and ledger
    reconciliation;
15. propagated uncertainty and tested 95% coverage;
16. explicit fallback provenance and fail-closed required inputs; and
17. reference/centering/orthogonality constraints plus legal-support
    rank/dependence diagnostics for player, league, roster, composition, and
    contextual components;
18. on-policy or valid sequential off-policy evidence for any partial-draft
    probability under a policy different from observed behavior; and
19. identical Python/artifact/API/TypeScript predictions within declared
    floating-point tolerance.

## 12. Research choices

L2 may compare, but builders may not settle by preference:

- random walk versus mean reversion, boundary shock/reset, and decay scale;
- inference engine and posterior approximation;
- auxiliary player-performance channels;
- team-policy parameterization;
- regularized horseshoe versus multilevel Gaussian/low-rank shrinkage;
- pairwise hypergraph versus Deep Sets/Set Transformer residual;
- calibration family;
- posterior-displacement, precision, and source/context evidence diagnostics;
- reference populations and admissible identified parameterizations within the
  required intervention/centering constraints;
- hard, soft, empirical, or risk-aware partial-draft policy;
- tier-list counterability tail and weight; and
- public compact reliability bands.

Each choice resolves through the research register, benchmark evidence, and
manifested decision record.
