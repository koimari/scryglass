# External methodology review: Machine LoLing

Reviewed 2026-07-31 from the eight linked articles. The local research packet
contains the retrieved HTML copies, the original figure URLs, 111 downloaded
figure assets, and contact sheets under
`data/lol/v2/experiments/leaguepedia/manual-run-2026-07-31/external-methodology/machineloling/`.
The packet is a provenance aid and methods reference, not a republication of
the articles.

## What appears reusable

### 1. Champion identity should be represented by an interaction profile

The 2026 topology study trains a neural network on role-aware drafts to predict
multiple outcomes over time: win likelihood, game duration, damage dealt, and
damage taken. It removes the representation direction most associated with raw
win rate before visualizing the remaining champion identity. It then uses 3D
plots, similarity maps, and dendrograms.

The important separation for Scryglass is:

- ordinary champion strength is one signal;
- identity is the shape of matchup, synergy, time, and output responses;
- a visual 3D projection is a display of a higher-dimensional representation,
  not the model itself.

We should therefore fit role- and patch-conditioned response profiles and use
their low-dimensional basis for shrinkage. UMAP is suitable for inspection, but
not as the deterministic serving transform: its coordinates can rotate or
change with the fit population.

### 2. Correct interaction deltas for ordinary champion strength

The bot-lane studies, the champion-class encyclopedia, the synergy visualizer,
and the blindability study all use a version of the same core idea: estimate
whether a pair performs above or below what the individual champions would
predict. The encyclopedia describes win-rate-corrected matchup and synergy
deltas, empirical-Bayes correction for sample size, correlation-based profile
similarity, and Ward/SNN clustering. The synergy visualizer also warns that a
trio can be good even when one of its pair edges is bad because the other edges
compensate.

This supports a hierarchical feature family:

1. role/champion strength;
2. corrected ally and enemy pair residuals;
3. corrected anchored triple and higher-order residuals;
4. a backoff to the lower-order parent when the higher-order cell is sparse.

The model should not treat raw pair win rate as synergy. A pair effect is a
residual after accounting for both members' ordinary strength and the relevant
role/league/patch context.

### 3. Similarity is role-specific and can change with game phase

The encyclopedia uses all-role interaction profiles but finds substantially
cleaner classes for support, ADC, mid, and jungle than for top. The bot-lane
series shows that classes can shift when interactions with top, jungle, and mid
are added, which is a useful warning against one universal champion taxonomy.

For our model, a champion should not have one global point. It should have a
family of points or response vectors, at least:

- role × patch/meta window;
- ally-synergy profile;
- enemy-response profile;
- early/mid/late or state-conditioned profile when the data supports it.

Top-lane “unique” or boundary champions should remain mixed/uncertain rather
than being forced into an archetype for presentation convenience.

### 4. Blindability is dispersion, not strength

The blindability study defines synergy blindability as the spread of a
champion's ally interactions and matchup blindability as the spread of its
enemy interactions. It applies a binomial Method-of-Moments correction because
low-support cells appear artificially volatile. It also notes that the measure
does not yet condition on player mastery.

That maps cleanly to a diagnostic for the player-aware model:

- raw interaction spread is not enough;
- support-adjusted spread is required;
- player/champion experience can be used as a separate conditional layer;
- “blindable” must not be presented as “strong.”

### 5. Scaling curves need a state caveat

The scaling study uses gold/XP parity as a proxy for a neutral game state and
explicitly notes missing objectives and late-game state information. That is a
useful diagnostic idea, but not a causal scaling label. A champion can look
strong at parity because it usually reached parity after a lead, or weak at
parity because it is recovering from an earlier deficit.

For Scryglass, state-conditioned response curves should be descriptive and
should use objective/state controls when available. They should not be inserted
into the terminal draft probability without a pre-event state contract.

## What we should redo ourselves

The linked work uses mostly Diamond+ solo-queue data from ranked regions and
patch windows such as 16.1–16.4 or 2025. Our target is professional regional
and international play, where pick policy, role assignment, player identity,
patch, and tournament format differ. We should reproduce the method on our own
Leaguepedia/OE population rather than import the published classes or deltas.

The proposed reproducible study is:

1. Build a pre-event role-aware champion interaction ledger.
2. Fit ordinary strength and corrected ally/enemy residuals with empirical-Bayes
   support shrinkage.
3. Build role-specific response vectors from the residual profile, with power
   removed or separately projected.
4. Fit deterministic low-rank bases using a fixed seeded SVD/factorization;
   reserve UMAP, Ward, and SNN for exploratory visual diagnostics.
5. Add player baseline, player–champion experience, player–champion residual,
   and player-conditioned interaction residuals only when the exact pre-event
   roster packet is available.
6. Compare identity-only, interaction-only, and combined models on rolling
   temporal folds, reporting accuracy, Brier, log loss, calibration, support,
   and side-swap replay.

Higher-order terms are generated as an anchored hierarchy: for a picked
champion/player, order (k) is its response to every (k)-subset of the other
nine picks, from (k=1) through (k=9). This is a mathematical feature map,
not nine independent free coefficients for every observed combination. Order
and support determine shrinkage; an unsupported order backs off to its parent.

## Current experiment consequence

The July anchored order-1-to-order-9 probe generated 48,547 support-gated
terms through order 4. Orders 5–9 had no independently supported terms under
the declared thresholds. The raw higher-order candidate reached 64.1926%
accuracy but had Brier 0.290099 and log loss 1.185920, so it is rejected as a
serving upgrade for now. Its overconfidence is direct evidence that topology
and hyperedges need hierarchical shrinkage/calibration, not a reason to add
more unconstrained weights.

The current player-aware runtime remains development-only. The score file is
written before outcome evaluation, and missing pre-event roster evidence keeps
contextual output unavailable in strict mode.

## Mechanics-first follow-up

The external topology work is useful as a representation study, but it does
not justify treating a champion identity or a low-dimensional coordinate as a
mechanics value. The next layer is therefore a patch-aware rules packet:

- the League Wiki is mirrored as a revisioned, hash-addressed Obsidian source
  vault for semantics, patch history, and interaction notes;
- CommunityDragon client data is captured separately for patch-pinned numeric
  records, including champion bins, item data, cooldown arrays, targeters,
  base stats, and raw calculation graphs;
- a narrow evaluator executes only implemented calculation primitives and
  fails closed on unknown stat codes or formula types;
- statistical champion/player residuals are added only after the mechanics
  baseline has been frozen and tested.

The article namespace inventory contains 21,922 entries and the local mirror
completed with zero retrieval errors. The broader selected source catalog
contains 311,560 entries, including File-page metadata; the text checkpoint is
resumable and currently includes the article/project pages plus 4,445
templates. The available CommunityDragon 16.8
fixture contains 191 champion records and 695 item records with zero missing
champion bins. The requested tournament patch 26.13 was not available from
that source at capture time; no 16.8 value is being substituted into the July
forecast. This is the correct blocker to expose, not hide.

The first source-backed micro-tests evaluate Aatrox Q rank 1 at 100 total AD
to 70 damage, W rank 5 at 100 total AD to 110 damage, and the Q edge modifier
to 119 damage from the captured raw formula graph. Those numbers are parser
fixtures, not a claim about the July patch.

## Sources

- [We trained a Neural Network to discover the Topology behind Champion Identities](https://machineloling.com/2026/05/05/we-used-ai-to-discover-the-hidden-topology-behind-champion-identities/)
- [Champion Pool Designer](https://machineloling.com/2026/04/28/champion-pool-designer/)
- [The Definitive Season 16 Champion Class Encyclopedia](https://machineloling.com/2026/04/13/the-definitive-season-16-champion-class-encyclopedia/)
- [Synergy Visualizer](https://machineloling.com/2026/02/05/synergy-visualizer/)
- [Don’t worry, we scale: a vignette on champion scaling](https://machineloling.com/2025/07/22/dont-worry-we-scale-a-vignette-on-champion-scaling/)
- [The Bot Lane Ecosystem Part 1](https://machineloling.com/2024/07/28/bot-lane-ecosystem-part1/)
- [The Bot Lane Ecosystem Part 2](https://machineloling.com/2024/08/06/the-bot-lane-ecosystem-part-2/)
- [Blindability](https://machineloling.com/2024/12/17/blindability/)
