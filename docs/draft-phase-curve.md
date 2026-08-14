# Draft phase curve

Status: development input. The phase curve has no public authority.

Scryglass keeps three estimands separate:

- Static Draft Score describes composition interactions before the game.
- The pre-match phase curve forecasts gold differences at minutes 10, 15, 20,
  and 25 from draft, strength, lineup, competition, region, patch, and roster
  context.
- The live state model may use an observed timeline with gold, objectives, and
  clock. The phase curve does not require a live provider.

OE gold checkpoints are training targets for the pre-match forecast. They are
never pre-match features. Missing gold at 25 minutes stays censored. OE has no
gold-at-30 field. A true 30-minute live checkpoint needs a timeline source
with a 30-minute observation.

Patch identity stays explicit. The raw OE token `16.15` remains in provenance.
The current post-release tournament rows stay on that token because the source
has no per-game realm field. A derived `16.16` token requires explicit live
realm evidence. The public Riot label for that client patch is `26.16`. The
26.16 LCC bridge is staged and cannot change a 26.15 fit.

The phase artifact uses the Scryglass Pi round-9 atom path. It includes the
depth-2, depth-3, and depth-4 composition descriptors, prior state-space team
features, and per-pick atom contributions. The bridge also carries LCC family
and ontology features. These are development features, not causal claims.

The reference AUC gate is `0.70681`. A candidate below that value remains
unavailable. A candidate at or above it still needs chronological, regional,
patch, roster, missingness, sparse-data, scaling, snowball, and comeback gates.
An independent hash-bound promotion receipt is required before any numeric
phase result can be served.

The checked-in artifact is
`data/lol/models/draft_phase_curve.json`. Its unavailable fields are deliberate.
