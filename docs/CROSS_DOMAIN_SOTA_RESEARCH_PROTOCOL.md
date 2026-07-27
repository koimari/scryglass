# Scryglass cross-domain state-of-the-art research protocol

Version: 1.0
Date: 2026-07-27
Status: governing research and model-tournament protocol

## Purpose

Scryglass will not define state of the art as “the most complicated League of
Legends model” or as “the newest paper.” The selected model must answer the
declared product estimand, outperform or safely tie the strongest eligible
benchmarks on untouched future data, preserve calibration and uncertainty, and
remain reproducible and interpretable under external challenge.

The evidence universe therefore includes:

- pairwise-comparison and sports-rating research;
- Bayesian state-space and dynamic latent-trait models;
- financial econometrics for nonstationarity, shocks, volatility, covariance,
  shrinkage, and predictive-density combinations;
- psychometrics for repeated latent ability;
- adjusted plus-minus and hierarchical contribution models;
- proper scoring rules, forecast comparison, and calibration;
- factorization and sparse-interaction models;
- game theory, search, offline reinforcement learning, and safe policy
  improvement; and
- causal inference and off-policy evaluation.

The transfer is mathematical, not metaphorical. A team is not a stock, a
champion is not an asset, and a draft recommendation is not a trade. Methods
cross the boundary only when their likelihood, state, dependence, and
identifiability assumptions match the Scryglass estimand.

## Non-negotiable epistemic rules

1. **Prediction is not explanation and neither is causation.** A model may
   predict map wins without identifying why a player or champion caused them.
2. **No model self-certifies.** Model choice is frozen before the final test and
   evaluated from an immutable prediction ledger.
3. **Complexity loses ties.** If a more complex model is not demonstrably
   superior, the simpler eligible model remains the production choice.
4. **The raw draft estimand stays pure.** Team, player, roster, and organization
   state cannot enter the five-versus-five composition estimate.
5. **Current context is a separate forecast layer.** Contextualized forecasts
   may combine raw composition with time-safe team/player state, but the two
   outputs and their uncertainty remain visible separately.
6. **Unidentified quantities are not ranked.** Shrinkage can stabilize an
   estimand; it cannot create information absent from the likelihood.
7. **Distribution shift invalidates guarantees.** Patch, roster, tournament,
   league, and data-source shifts require explicit evaluation or a fail-closed
   result.
8. **Offline policy value is not observed policy value.** Sandbox policies
   require behavior-support diagnostics and off-policy uncertainty.

## Cross-domain evidence map

### Dynamic latent strength

The current production alternatives must include:

| Family | What it tests | Scryglass use |
|---|---|---|
| Static logistic Bradley–Terry | Whether dynamics add value at all | Mandatory simple baseline |
| Sequential Elo and uncertainty-aware Elo/Glicko variants | Whether low-cost online updates are sufficient | Operational baseline |
| Whole-History Rating | Whether retrospective dynamic MAP improves forecasting | Benchmark |
| Dynamic Bradley–Terry with Gaussian state evolution | Smooth ability drift | Core candidate |
| Heavy-tailed or change-point state evolution | Abrupt roster, patch, coaching, or organizational shocks | Core candidate |
| Kernel/nonparametric dynamic Bradley–Terry | Misspecification-resistant time variation | Core candidate |
| Gaussian-process latent strength | Flexible smooth/discontinuous trajectories | Research candidate |
| Spectral dynamic ranking | Nonparametric ranking and alternative uncertainty | Independent benchmark |
| Dynamic predictive-density combination | Time-varying ensemble weights without declaring one permanent winner | Candidate only after component validation |
| Intransitive Bradley–Terry/Hodge decomposition | Cyclic matchup structure not reducible to one scalar ladder | Diagnostic and optional extension |

Relevant primary work includes:

- Glickman and Jones, *Models and Rating Systems for Head-to-Head
  Competition*: <https://doi.org/10.1146/annurev-statistics-040722-061813>;
- Coulom, *Whole-History Rating*:
  <https://www.remi-coulom.fr/WHR/WHR.pdf>;
- Bong et al., *Nonparametric Estimation in the Dynamic Bradley–Terry
  Model*: <https://arxiv.org/abs/2003.00083>;
- Duffield, Power, and Rimella, *A State-Space Perspective on Modelling and
  Inference for Online Skill Rating*: <https://arxiv.org/abs/2308.02414>;
- Krese and Štrumbelj, *A Bayesian approach to time-varying latent strengths in
  pairwise comparisons*: <https://doi.org/10.1371/journal.pone.0251945>;
- Tian et al., *A Spectral Approach for the Dynamic Bradley–Terry Model*:
  <https://arxiv.org/abs/2307.16642>; and
- Wang, Berger, and Burdick, *Dynamic Item Response Models*:
  <https://arxiv.org/abs/1304.4441>.

The production winner is selected on frozen pre-event predictions, not on
in-sample ladder plausibility.

### Financial econometrics transfer

Financial modelling contributes useful machinery for changing latent state and
uncertainty:

| Financial concept | Legitimate transfer | Prohibited shortcut |
|---|---|---|
| Latent expected return | Latent team/player strength that evolves through time | Treating win/loss observations as Gaussian returns |
| Stochastic volatility | Time-varying uncertainty or innovation scale | Calling high uncertainty “volatility” without a fitted state model |
| Heavy-tailed state innovations | Rare abrupt roster/patch/organizational shocks | Letting every upset cause an unrestricted rating jump |
| Dynamic shrinkage | Most state changes near zero, occasional large changes | Post-hoc tuning around famous matches |
| Factor covariance | Low-rank shared interaction structure among champions/roles | Calling champion covariance portfolio diversification |
| Regime switching | Explicit patch, format, roster, or competition states | Inferring unnamed regimes and narrating them as causes |
| Predictive-density combination | Time-varying ensemble of validated forecasts | Blending models on the final test or by webpage appearance |
| Stress indicators | Detecting model/data shift and uncertainty inflation | Treating stress as a forecast of who wins |

Candidate references include:

- Gruber and West, *Bayesian forecasting and scalable multivariate volatility
  analysis using simultaneous graphical dynamic models*:
  <https://arxiv.org/abs/1606.08291>;
- Huber, Kastner, and Pfarrhofer, *Introducing shrinkage in heavy-tailed state
  space models to predict equity excess returns*:
  <https://arxiv.org/abs/1805.12217>; and
- Cho and Matteson, *Smoothing Variances Across Time: Adaptive Stochastic
  Volatility*: <https://arxiv.org/abs/2408.11315>.

GARCH is not a default match-outcome model. Conditional heteroscedasticity may
motivate a state-innovation model, but a binary observation still requires a
binary likelihood.

### Player contribution and uncertain ranks

Player-rating candidates form an estimand ladder:

1. shared lineup strength only;
2. ridge regularized adjusted plus-minus from valid player-specific outcomes;
3. role-aware hierarchical adjusted plus-minus;
4. offense/defense or phase-specific latent contributions;
5. dynamic hierarchical player states with roster and opponent context;
6. interaction/factor extensions only after main effects are identified.

Every candidate must report:

- design-matrix rank, effective rank, and condition diagnostics;
- posterior or sampling covariance, not only marginal standard errors;
- sensitivity to priors and regularization;
- synthetic recovery with known player effects;
- fixed-lineup negative controls;
- role, league, era, and roster-movement slices;
- out-of-sample lineup and match prediction;
- uncertain ordering or tie groups instead of forced ordinal ranks; and
- complementarity sensitivity.

Useful references include:

- Macdonald, *Adjusted Plus-Minus for NHL Players using Ridge Regression*:
  <https://arxiv.org/abs/1201.0317>;
- Matano et al., *Augmenting Adjusted Plus-Minus in Soccer with FIFA Ratings*:
  <https://arxiv.org/abs/1810.08032>;
- Barrientos et al., *Bayesian Inferences on Uncertain Ranks and Orderings*:
  <https://arxiv.org/abs/1907.04842>; and
- Ghimire, Ehrlich, and Sanders, a complementarity critique of regularized
  adjusted plus-minus: <https://doi.org/10.1371/journal.pone.0237920>.
- De Bois, Parmentier, and Puget, *PandaSkill — Player Performance and Skill
  Rating in Esports: Application to League of Legends*:
  <https://arxiv.org/abs/2501.10049>; and
- Morgan et al., *The SIDO Performance Model for League of Legends*:
  <https://arxiv.org/abs/2403.04873>.

The current team-outcome player update is a lineup signal, not a candidate
individual model. Performance-statistic models are a separate descriptive
estimand and must not use post-map combat or economy statistics to claim a
pre-map skill forecast without a second chronological validation layer.

### Complete-draft composition and match probability

Eligible complete-draft candidates include:

- side/league/patch intercept-only and champion-main-effect baselines;
- role-aware penalized logistic models;
- hierarchical main effects with patch/league partial pooling;
- within-team synergy and cross-team opposition with sparse shrinkage;
- low-rank factorization machines for interaction sharing;
- set- or graph-based models with permutation/role constraints;
- tree/boosting models over frozen pre-match features;
- calibrated ensembles of independently validated candidates.

Required invariants:

- swapping blue and red maps probability to `1 - p` within numerical tolerance;
- permuting players without changing role assignment changes nothing;
- duplicate champion states are rejected as illegal;
- complete-draft raw output contains no team/player identifiers;
- patch and league terms use only data available at prediction time;
- every interaction contribution reconciles exactly to the total logit;
- probabilities are finite and bounded;
- low-evidence interactions shrink toward neutral;
- final-test events never influence calibration or hyperparameters.

Sparse factorization is a benchmark, not an automatic upgrade. Relevant general
interaction work includes regularized factorization machines:
<https://arxiv.org/abs/2010.09225>.

### Probability evaluation and calibration

Primary model comparison uses a strictly proper score. Accuracy, AUC, and
headline upset counts are secondary diagnostics.

Each frozen test artifact must include:

- event-level log loss and Brier score;
- paired candidate-minus-baseline differences;
- dependence-aware uncertainty over those differences;
- CORP/PAV reproducible reliability diagnostics;
- discrimination, miscalibration, and uncertainty components;
- league, patch, side, roster-change, evidence, and probability-band slices;
- confidence intervals with denominators;
- stability under reasonable perturbations;
- an always-valid or predeclared fixed-horizon comparison policy;
- raw prediction rows for independent recomputation.

Core references:

- Dimitriadis, Gneiting, and Jordan, *Evaluating Probabilistic Classifiers:
  Reliability Diagrams and Score Decompositions Revisited*:
  <https://arxiv.org/abs/2008.03033>;
- Gneiting et al., *Evaluating Probabilistic Classifiers: The Triptych*:
  <https://arxiv.org/abs/2301.10803>;
- Choe and Ramdas, *Comparing Sequential Forecasters*:
  <https://arxiv.org/abs/2110.00115>; and
- Heiser, Allikivi, and Kull, *Shift Happens: Adjusting Classifiers*:
  <https://arxiv.org/abs/2111.02529>.

Conformal prediction is not advertised as distribution-free protection after a
patch or roster shift. Standard exchangeability guarantees do not survive an
uncontrolled distribution change.

### Sandbox as a sequential adversarial policy

The Sandbox state is:

`S = (ruleset, side, turn, picks, bans, open roles, legal champions, patch,
league scope, optional roster context, evidence state)`.

The action space is the legal champion-role assignments at that turn. The
terminal utility is the calibrated complete-draft map-win distribution, with a
separate contextual layer when requested. A recommendation is an action value
under an explicit policy over all later own actions and opponent responses.

Mandatory baselines:

1. pro pick-frequency by role and state;
2. one-step greedy composition delta;
3. depth-limited expectimax against an empirical opponent policy;
4. minimax/robust search over plausible opponent responses;
5. beam or Monte Carlo tree search;
6. behavior cloning from legal professional draft trajectories;
7. counterfactual regret or self-play where the abstraction is valid;
8. conservative offline RL candidates only when behavior support is adequate.

The production policy may not recommend an action solely because an
out-of-distribution value model assigns it a high value. Required controls:

- legal-action masking at every node;
- role-feasibility propagation, including flex picks;
- exact draft-order and ban rules by tournament;
- behavior-policy support and effective sample size;
- uncertainty-aware fallback to empirical/search baselines;
- held-out future-patch and future-tournament trajectories;
- top-k recall of professional actions as a descriptive diagnostic, not the
  utility objective;
- terminal composition forecast quality;
- off-policy evaluation with multiple estimators and stress tests;
- no claim of policy improvement when confidence intervals overlap or support
  is inadequate.

The present OE/GRID draft rows do not include the probability with which the
professional drafting policy selected each legal action. They also provide
extremely sparse support relative to the champion-role slate space. Therefore
ordinary inverse-propensity off-policy evaluation is not currently identified.
The app must restrict recommendations to supported action/role regions, label
model extrapolation, and abstain from any “better than pro” or “optimal policy”
claim until a defensible behavior policy and effective-support analysis exist.

Relevant methods:

- JueWuDraft: <https://arxiv.org/abs/2012.10171>;
- DraftRec: <https://arxiv.org/abs/2204.12750>;
- DraftComPromise:
  <https://doi.org/10.1109/GEM61861.2024.10585636>;
- BPCoach: <https://arxiv.org/abs/2311.05912>;
- Off-policy Bandits with Deficient Support:
  <https://arxiv.org/abs/2006.09438>;
- Off-Policy Evaluation in Embedded Spaces:
  <https://arxiv.org/abs/2203.02807>;
- Off-Policy Evaluation for Large Action Spaces via Conjunct Effect Modeling:
  <https://arxiv.org/abs/2305.08062>;
- Off-Policy Evaluation of Slate Bandit Policies via Optimizing Abstraction:
  <https://arxiv.org/abs/2402.02171>;
- Distributional Off-Policy Evaluation for Slate Recommendations:
  <https://arxiv.org/abs/2308.14165>;
- Conservative Q-Learning: <https://arxiv.org/abs/2006.04779>;
- Implicit Q-Learning: <https://arxiv.org/abs/2110.06169>;
- Safe Policy Improvement with Baseline Bootstrapping:
  <https://arxiv.org/abs/1712.06924>; and
- Conformal Off-Policy Evaluation:
  <https://arxiv.org/abs/2304.02574>.

Offline reinforcement learning is a late benchmark, not the first production
implementation. Sparse logged pro drafts and deterministic team preferences
create severe support and confounding problems. Search over a validated
terminal model and an empirical opponent policy is the safer first serious
candidate.

## Frozen model-tournament design

### Dataset partitions

Every model family uses four chronological layers:

1. **development train:** parameter fitting;
2. **development validation:** hyperparameters, architecture, and candidate
   pruning;
3. **selection holdout:** one comparison among frozen finalists;
4. **final test:** one untouched evaluation used for the release decision.

Series, tournament, and roster intervals cannot cross layers. An embargo covers
features whose publication or aggregation time could leak the boundary.
Patch-forward and league-forward tests are additional targets, not substitutes
for the main chronological test.

### Candidate admission

A candidate enters selection only when:

- its estimand and grain match the tournament;
- its prediction ledger passes temporal/provenance validation;
- training is deterministic under a declared seed or has a declared
  multi-seed distribution;
- hyperparameters were chosen without selection/final-test access;
- required invariants pass;
- calibration uses development data only;
- all baseline predictions cover the identical event set;
- resource use is compatible with pack generation and request latency;
- a model card and reproducible artifact exist.

### Decision rule

For lower-is-better proper score `L`, compare candidate `c` with baseline `b`
using paired event losses:

`d_i = L(y_i, p_ci) - L(y_i, p_bi)`.

Dependence is preserved by aggregating maps within canonical series and using a
predeclared moving-block or cluster-aware interval. A complex candidate is
accepted only when the interval supports superiority. An indistinguishable
complex model does not replace the simpler model. Calibration, subgroup
stability, and uncertainty gates can reject a score winner.

Sequential monitoring uses confidence sequences or a frozen stopping rule. The
release team may not repeatedly inspect a conventional confidence interval and
stop when it first becomes favorable.

## Executable evidence contract

`lol_kills.model_tournament` now supplies:

- a required prediction-ledger schema;
- exact checks for duplicate predictions, probability/outcome validity,
  cross-model event consistency, and
  `data_as_of <= prediction_time <= event_time`;
- Brier and log-loss event contributions;
- deterministic PAV calibration with an exact Brier decomposition;
- paired circular moving-block model comparison over canonical-series means;
- frozen tournament specifications, random seeds, confidence levels, and
  noninferiority margins.

This infrastructure is necessary but not sufficient. It becomes release
evidence only after canonical series and temporal source ledgers replace the
current pseudo-series and appearance-derived state.

## Rejected shortcuts

- Do not choose a model because it produces familiar rankings.
- Do not call arbitrary rating spread “volatility.”
- Do not use a final-test ensemble selected after seeing final-test scores.
- Do not use current roster or rating state in historical predictions.
- Do not infer individual skill from team outcomes without player-specific
  identifying variation.
- Do not treat posterior shrinkage as observed evidence.
- Do not use fixed-width ECE bins as the sole calibration diagnostic.
- Do not use accuracy as the primary probability score.
- Do not interpret feature attribution as causal effect.
- Do not deploy offline RL from logged drafts without support diagnostics and
  policy-value uncertainty.
- Do not call top-k imitation accuracy optimal drafting.
- Do not hide abstention, missing provenance, or model fallback.

## External challenge package

The package provided to an external reviewer must contain:

1. estimand, grain, population, and time-policy registry;
2. canonical identity, series, roster, membership, patch, and tournament
   ledgers;
3. immutable train/validation/selection/final split IDs;
4. candidate registry and frozen tournament specifications;
5. complete event-level prediction ledgers;
6. proper-score, calibration, and uncertainty recomputation;
7. synthetic recovery, identifiability, and invariant tests;
8. model cards, ablations, failures, and rejected alternatives;
9. Sandbox legal-state corpus, behavior support, policy evaluation, and
   counterexamples;
10. clean-room reproduction commands and environment lock;
11. source, calculation, and public-claim ledgers;
12. exact pack and artifact hashes.

An expert should be able to disagree with a modelling choice while still
confirming exactly what was calculated and why the public wording does or does
not follow.
