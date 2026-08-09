# Acceptance gates

Every checked item must link to machine output, an artifact hash, or a reviewed
diff. “Looks right” is not evidence.

## Gate A — contracts (S0)

- [ ] All eleven prose files and required schemas exist and link correctly.
- [ ] Every public number has a literal interpretation and conditioning set.
- [ ] Neutral/contextual and state-snapshot/current-analysis/forecast/
      forecast-simulation/hindsight meanings are disjoint.
- [ ] Contextual Draft Score equalizes baseline Team/League strength.
- [ ] Probability wording depends on exact served-transform calibration.
- [ ] Requested stale context cannot fall back to neutral.
- [ ] Required invariants and research choices are separated.
- [ ] Initial-cycle runtime contains no live scoring and no grubs 24% scope;
      future live work is neither implied nor permanently prohibited.
- [ ] Schemas pass Draft 2020-12 meta-validation.

## Gate B — foundations (C1)

### L1

- [ ] Stable IDs and aliases are separate; collisions are audited.
- [ ] Availability time is populated or conservatively falls back to observed
      time.
- [ ] Exact active roster resolves one player per role or fails closed.
- [ ] Series use authoritative IDs; no fixed-time grouping.
- [ ] Unresolved-series maps are excluded from primary inference/bootstrap and
      are never treated as independent singleton clusters.
- [ ] League tier/connectivity and patch rules are versioned.
- [ ] `season_id` is authoritative, `calendar_year` is derived from UTC event
      date, and neither implies a state reset.
- [ ] Draft data stores canonical A/B, actual blue/red, first/second draft
      order, protocol mapping, and mapping source/availability separately.
- [ ] Source/training snapshots are immutable and content-addressed.
- [ ] Every candidate has a canonical source allowlist and matching
      `source_tree_sha256`; candidate `code_commit` may be absent/null.
- [ ] Public/private matrix covers every source and artifact class.
- [ ] Credentials, private URLs, and recoverable secrets are absent from public
      candidates, logs, fixtures, and examples.
- [ ] Removed grubs/market fields are absent from v2 snapshots.
- [ ] Known team selection resolves the exact active roster; a custom five is
      never Team Rating or a ladder row.

### L2

- [ ] Split plan is frozen before sealed results.
- [ ] Same-series maps stay together.
- [ ] Temporal, future-patch, league, international, and sparse holdouts exist
      for neutral Draft Score; roster-change holdouts are required only when
      player or exact-roster identity terms are modeled.
- [ ] Every sealed suite is single-use, and every data-informed feature,
      state, prior, reference, policy, threshold, weight, and calibration step
      is refit inside each outer fold.
- [ ] Leakage sentinels fail deliberately contaminated fixtures.
- [ ] All five successful-output schemas reject stale/missing/conflict required
      provenance and false required freshness checks.
- [ ] Every successful output requires nonempty Reliability proper-score,
      calibration, interval-coverage, resolved-cluster, and OOD diagnostics;
      `high` fails when any are missing/failed, support is zero, the transform
      is unapproved, or the artifact is OOD.
- [ ] Reliability stratum selection replays a frozen hashed total
      context/OOD mapping; no match is `unrated`, never a hand-selected nearby
      stratum.
- [ ] Structural negative mutations and semantic counterexamples fail for all
      five canonical examples.
- [ ] Series remain indivisible inner blocks within a preregistered coarsest
      defensible or multiway participant/team plus tournament/time dependence
      design; small-cluster correction and higher-level robustness pass.
- [ ] Any coarser unresolved-series sensitivity is preregistered and
      deterministic.
- [ ] Baselines are exact, versioned, and replayable.
- [ ] Sealed evaluation runs the frozen end-to-end pipeline from raw snapshot
      through feature/state reconstruction, calibration, serialization, and
      serving; isolated component passes cannot promote.
- [ ] Partial probabilities are calibrated by prefix/slot using the exact
      served search policy, temperature, approximation, and transform.
- [ ] A served partial policy different from observed behavior has prospective
      on-policy evidence or valid sequential OPE with exchangeability,
      positivity, behavior-policy, effective-sample-size/weight, and sensitivity
      diagnostics; naive prefix replay is rejected.
- [ ] Promotion report cannot be generated with a changed metric/split hash.

### L3

- [ ] Ontology dimensions are interpretable, patch-versioned, and sourced.
- [ ] Multi-label role-specific archetype priors validate.
- [ ] New/zero-play champion gets a wide prior without fabricated residual.
- [ ] Masked-champion evaluation exists.
- [ ] Zero-play champion is explicitly ineligible for tier-list membership.

## Gate C — mathematical cores (C2/C3)

### L4 Player Rating

- [ ] Pre-event state replay equals stored forecast state.
- [ ] Future deletion/shuffle leaves earlier states unchanged.
- [ ] League-scoped and bridged-global Player Ratings are distinct, labeled,
      and evaluated on their own reference populations.
- [ ] Global Player Rating requires current structural global eligibility and
      bridge support; League Rating is never added as individual skill.
- [ ] The posterior mean is the rating; any compatibility duplicate is
      semantically equality-checked.
- [ ] Rating posterior mean and 95% interval map through the required 1500/400
      Elo expected-result display contract, not FIDE update rules.
- [ ] Role normalization and the frozen reference policy are documented, and a
      one-player replacement under that policy changes the roster aggregation
      by exactly the displayed player-logit difference in every role, with
      lineup synergy held at its zero reference.
- [ ] Plug-in expected result at posterior-mean ratings is labeled separately
      from posterior-predictive probability integrated over joint uncertainty.
- [ ] Support channels include non-resource impact or use broad equal prior.
- [ ] Endogenous resource channels pass temporal-order, joint-measurement versus
      no-resource/lagged-policy, and player-policy double-count/collider
      sensitivities; no causal holding-resources-equal interpretation appears.
- [ ] Player transfer carries history without sticky team effect.
- [ ] Decay/mean-reversion label matches the selected fitted model.
- [ ] Evidence is posterior information, not sample count.
- [ ] Evidence keeps displacement, precision, and source/context coverage
      separate; R-20/L2 selects methods and no KL normalization is hard-coded.

### L5 Team/League Rating

- [ ] Roster ID is the exact ordered five; public output names all five.
- [ ] New roster derives from player posteriors with wide policy/synergy prior.
- [ ] Regional Team Rating uses league-scoped player states; global Team Rating
      uses global player states before League Rating is added.
- [ ] Organization-only residual is absent after roster change.
- [ ] Policy uses resource and non-resource signals.
- [ ] A weak resource channel is excluded from Player Rating or updates policy
      only; one aggregate cannot independently credit/debit both player and
      policy.
- [ ] Team-policy versus lineup-synergy separation passes within-roster
      variation, rank/conditioning, posterior-dependence, source-removal, and
      ablation checks, or only identified total strength/the simpler fallback
      is exposed.
- [ ] League Rating is explicit and not called a team international component.
- [ ] Global player/League Rating separation passes frozen centering,
      transfer-continuity, mobility/bridge design-rank, posterior-dependence,
      and reference/source-removal sensitivity checks; shrinkage alone does not
      count as identification.
- [ ] Tier-2/3 `structurally_globally_eligible` is false; statistical bridge
      strength is a separate diagnostic.
- [ ] Posterior-mean ties use lower uncertainty then stable ID.
- [ ] Settled requires strictly greater-than-95% precision and stability,
      interval width within resolution, active/current eligibility, complete
      fresh inputs, no material fallback/OOD, and passed coverage.
- [ ] Duplicate-player exact-five fixtures fail semantic validation.
- [ ] Hypothetical/custom rosters cannot pass Team Rating validation.

### L6 composition

- [ ] Role-preserving input permutation leaves output unchanged.
- [ ] Side swap negates the raw logit.
- [ ] Each champion includes four ally and five enemy relations.
- [ ] Ally/cross terms have required symmetry.
- [ ] Sparse terms shrink through manifested priors.
- [ ] New patch/champion uses one recorded hierarchy, not another engine.
- [ ] Residual complexity survives ablation.
- [ ] Main, pair, whole-team, and cross-team terms satisfy the frozen
      functional-ANOVA centering/orthogonality constraints and legal-support
      rank/conditioning, co-occurrence, posterior-dependence, and
      source-removal checks; Shapley allocation is not identification.
- [ ] Weak component splits collapse to a supported joint residual or total
      composition value with widened uncertainty and no invented
      main/synergy/counter claim.
- [ ] Posterior covariance is propagated or coverage validates approximation.

### L7 terminal Draft Score

- [ ] Exactly one estimator and transform produces the served result.
- [ ] Neutral path contains no identity fields.
- [ ] Contextual path uses conditional fit and sets baseline strength difference
      exactly to zero.
- [ ] Context request with stale/missing roster/policy is unavailable.
- [ ] Contextual success includes its labeled neutral comparison; neutral mode
      forbids a redundant self-comparison.
- [ ] International Draft Score uses one named event/meta
      `competition_scope_id` shared by both sides and is invariant to either
      team's domestic-league provenance.
- [ ] No in-game side advantage is present.
- [ ] Draft-order/game-side support, positivity, rank, conditioning, and
      confounding diagnostics pass before any empirical order term is nonzero.
- [ ] Perfect collinearity forces the empirical order coefficient to zero and
      records it unavailable as a convention; structural action-tree value
      stays separate.
- [ ] Under collinearity, admissible combined side/order decompositions are
      sensitivity-bounded; material score or calibration movement makes the
      affected Draft Score endpoint unavailable.
- [ ] Swap yields exact pre-rounding complement.
- [ ] Contribution ledger reconciles at floating-point-derived tolerance.
- [ ] Semantic fixtures reject 99/99 complements, score/probability mismatch,
      transform mismatch, reversed intervals, and raw-logit/ledger mismatch.
- [ ] Terminal semantic validation enforces five assignments per side, exactly
      one of each role, globally unique champions, and legal protocol/actions.
- [ ] All 25 enemy relations and all ally relations appear in ledger coverage.
- [ ] Player-champion versus policy drivers pass frozen cross-orthogonalization,
      joint-rank/dependence, deterministic-overlap, and source-removal checks;
      otherwise only total contextual fit is exposed.
- [ ] Unidentified component families use a reconciled
      `unresolved_joint_residual` and interpretation status while retaining
      complete ally/enemy coverage.
- [ ] 0–100 equals 100 times the calibrated standardized map-win probability.
- [ ] Every serving transform is open-support, complement-symmetric, and
      monotone nondecreasing.
- [ ] L2 approved the exact serialized serving transform.
- [ ] Match Forecast, if present, is a different schema/label.

## Gate D — derived structures and serving (C4)

### L8 partial draft

- [ ] State graph accepts only legal protocol actions.
- [ ] Canonical transpositions and deterministic replay agree.
- [ ] Current role constraints and append-only role-set revision/reassignment
      history are both present; legal sets can widen, narrow, or change without
      rewriting the original pick.
- [ ] Swain-style flex revisions and reassignment fixtures replay correctly.
- [ ] The signed strategic-response adjustment equals strategic minus committed
      value under the same registered baseline policy, including negative
      opponent-to-act fixtures.
- [ ] Flex value compares declared set to fixed-role alternatives.
- [ ] Search policy/temperature is manifested.
- [ ] The soft-search reference policy is normalized and strictly positive on
      every legal action included in its recursion; model-prior support is not
      mistaken for logged-policy positivity.
- [ ] Historical replay is labeled on-policy only for the observed behavior
      policy; a different served policy meets the prospective/sequential-OPE
      gate before any probability wording.
- [ ] Search artifact, finite open-support transform, prefix-calibration report,
      and approved prefix strata are manifested and repeated in provenance.
- [ ] Terminal calibration alone cannot approve partial probability wording.
- [ ] A failed prefix/slot gate returns unavailable; no probability-like index
      or calibrated-probability recommendation is served.
- [ ] Zero-play/new-champion actions without logged-policy positivity can enter
      only a separate ordinal Archetype extrapolation research view—never
      Partial Draft Score, 0–100, probability/advantage, or Reliability.
- [ ] Contextual Archetype extrapolation requires exact fresh rosters and uses
      only general player/broad archetype-fit priors, never an unsupported exact
      player-champion residual.
- [ ] Every transform consumed by search or partial probability maps to
      `(0,1)` and preserves complement symmetry.
- [ ] Approximate search reports coverage/bounds/seed.
- [ ] Every terminal graph state equals canonical Terminal Draft Score for the
      same inputs/model version, including display rounding.
- [ ] Recommendations show value change, uncertainty, downside, and opponent
      responses.

### L9 tier lists

- [ ] Artifact key is exactly one league, current patch, and role.
- [ ] Refresh occurs after each completed eligible match.
- [ ] Membership contains only verified played champions in that exact cell.
- [ ] Value is model-standardized incremental probability points, not raw WR.
- [ ] Prospective tier evaluation uses observed future outcomes through the
      proper-score adapter; model-derived latent residuals are not ground truth.
- [ ] The nested adapter removes the evaluated champion's ordinary contribution,
      directly inserts its pre-event \(TV=IV-\lambda_C C\) row, keeps other
      time-safe strength/draft terms as offsets, and compares \(\lambda_C=0\)
      against candidates by future proper score/calibration.
- [ ] Roster/strength nuisance is controlled time-safely.
- [ ] Counterability rule and weight are validated and manifested; weight is
      zero if it adds no out-of-sample value.
- [ ] Counterability is computed from response-specific
      \(\Delta_c(z,a)\), uses the registered lower-tail plausible-response
      regret, and is nonnegative before weighting.
- [ ] Allied-context, response, and reference-champion distributions are common
      across champions in the cell, and every replacement respects champion
      uniqueness, bans, roles, and protocol.
- [ ] Inadequate legal overlap/support or effective clusters forces
      \(\lambda_C=0\) and leaves counterability descriptive.
- [ ] Uncertainty and evidence remain visible for one-game entries.
- [ ] Patch conflict or no played champions returns unavailable/empty with
      reason, not another patch.

### L10 serving

- [ ] Registry points to one promoted manifest/output.
- [ ] Rating manifests/artifacts require the 1500 anchor and 400-point
      expected-result scale.
- [ ] Artifact, calibration, evaluation, and schema hashes agree.
- [ ] Python, artifact, API, and TypeScript golden replay pass.
- [ ] Generated status union prevents score access on unavailable output.
- [ ] Shared semantic validator runs in Python, artifact compilation, API, and
      TypeScript; schema validity alone cannot promote.
- [ ] Cache key separates model/as-of/identity/authorization.
- [ ] Neutral cache cannot answer contextual request.
- [ ] All fail-closed error codes have fixtures.
- [ ] Runtime has no secret, private rows, training-only objects, or hidden
      fallback.
- [ ] Legacy estimators are inaccessible from v2 serving.

## Gate E — product and access (C5)

### L11 public surfaces

- [ ] Player Rating, Team Rating, and Draft Score are the principal numbers.
- [ ] Exact roster, scope, patch, mode, status, and as-of are visible where
      applicable.
- [ ] Main views are concise and explanation lives in details/Methodology.
- [ ] Recorded fan/journalist/analyst comprehension checks show readers can
      identify the principal number, scope, uncertainty/status, and freshness
      from the visual without inline explanatory paragraphs.
- [ ] Recorded analyst/pro-player review covers rating face validity, draft
      interaction usefulness, and disagreement cases without overriding
      predictive evaluation.
- [ ] Approved editorial typography avoids generic AI/dashboard defaults;
      hierarchy, density, and responsive layouts pass an anti-clutter review.
- [ ] No wagering terms, bare internal math names, AI-style filler, or SOTA
      wording without benchmark approval.
- [ ] State snapshot, current analysis, forecast, historical forecast
      simulation, actual outcome, and hindsight labels are correct and distinct.
- [ ] Neutral/contextual labels never disappear at responsive breakpoints.
- [ ] Unavailable state never shows zero, 50, prior value, or stale cache.
- [ ] Keyboard, focus, contrast, reduced-motion, screen-reader, mobile, and
      desktop checks pass.

### L12 article/auth/access

- [ ] Only selected authors can create/publish articles.
- [ ] Quantitative inserts pin estimand/model/as-of/revision.
- [ ] Typed patch-mechanics inserts pin mechanic semantics, patch, source
      snapshot, availability, and revision as well as quantitative inserts.
- [ ] Self-healing freezes on numerical claim, mechanics-semantic, or prose
      conflict and requires author review; it never silently rewrites prose.
- [ ] Historical article figures remain recoverable.
- [ ] Public/authenticated access changes breadth, not number meaning.
- [ ] Publication matrix decisions are enforced server-side.
- [ ] Post-C4 U1 records measured training/runtime/storage/refresh/hosting cost
      and explicit user decisions on cadence, access, custom what-if rosters,
      and any code/data/weight publication.
- [ ] Authentication alone exposes neither private weights nor licensed rows.
- [ ] Private credentials never enter public artifacts/client/logs.
- [ ] Authorization, object-level access, cache separation, and revocation tests
      pass.

## Gate F — reproducibility

- [ ] Clean environment rebuilds every public artifact from the declared
      snapshot, canonical source tree, config, and lockfile.
- [ ] Candidate top-level and lineage source-tree digests match; promoted
      manifests additionally require a non-null 40-hex commit containing that
      exact tree.
- [ ] Rebuild hashes match or documented nondeterminism is bounded and tested.
- [ ] Random seeds and approximate search/inference settings are recorded.
- [ ] Public reproduce page contains only matrix-approved essentials.
- [ ] Source attribution and Riot legal boilerplate are present where required.
- [ ] Evaluation report names data window, baselines, clusters, metrics, and
      limitations.
- [ ] Promoted status requires sealed holdout opened, decision pass, all gates,
      and numerical/semantic parity evidence.
- [ ] The promoted artifact is the unchanged frozen end-to-end pipeline that
      consumed the sealed raw snapshot; no post-opening component substitution
      occurred.
- [ ] Overall and every critical-stratum non-inferiority claim uses a frozen
      one-sided bound and multiplicity procedure; low power or failure to rule
      out the harm margin blocks promotion.
- [ ] SOTA wording is false for candidate/failed/noninferiority-only models and
      true only for promoted, preregistered meaningful superiority with
      uncertainty.
- [ ] No empty validation block or `--no-validate` artifact is promotable.
- [ ] “Latest” pointer targets the exact promoted immutable version.

## Gate G — clean-worktree release allowlist

- [ ] Release starts from a clean worktree at the intended upstream commit.
- [ ] The release commit's canonical source-tree digest matches the promoted
      manifest and source allowlist.
- [ ] User-owned dirty files in the working checkout are not copied implicitly.
- [ ] One explicit path allowlist is reviewed against each owner boundary.
- [ ] Generated artifacts have source manifests and are individually listed.
- [ ] `git diff --name-status` against upstream contains only approved paths.
- [ ] `git diff --check`, tests, build, schema, artifact, browser, and access
      checks pass on the exact staged tree.
- [ ] No broad staging command such as `git add -A` is used.
- [ ] Commit, push, PR, migration, and deployment each have explicit user
      authorization.
- [ ] Production root/settings remain compatible with the existing
      `apps/lol-atlas` Vercel project.
- [ ] Rollback manifest and previous registry pointer are recorded before
      deploy.

## Final S∞ review

- [ ] Contract hash and owner allowlists match.
- [ ] Every C0–C5 checkpoint has evidence.
- [ ] Sealed L2 report satisfies the benchmark-driven promotion rule.
- [ ] U1 user decision exists and every publication/access choice stays within
      both it and the source matrix.
- [ ] Public probability wording is supported by exact serving calibration.
- [ ] No unresolved research item is disguised as settled production semantics.
- [ ] Source publication and authentication boundaries hold.
- [ ] Current-repo conflicts are either migrated or explicitly blocked from v2.
- [ ] Outcome is `ACCEPT`, bounded `REMAND`, or `BLOCK` with owner/evidence.
