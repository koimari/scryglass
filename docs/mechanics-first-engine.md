# Mechanics-first League engine

Status: internal foundation implemented; not a serving or predictive-authority claim
As of: 2026-07-31

## Implementation checkpoint

The first bounded implementation is materialized locally:

- the Wiki vault validates 311,560 catalog entries, with text bodies for all
  cataloged text namespaces and File pages retained as metadata only;
  three legacy latest-checkpoint rows are preserved and reported rather than
  silently deleted;
- one explicit Wiki packet and one exact client-source packet exist for each
  tracked 2026 patch, 26.01 through 26.16; the client packets provide
  patch-pinned base-stat cells, while abilities, items, runes, and game-system
  execution remain semantic-only or blocked;
- the typed transition kernel, deterministic event ordering, state hashes,
  execution traces, parity receipts, temporal roster intervals, GRID
  checkpoint receipts, and order-1 through order-9 interaction keys are
  implemented and covered by targeted tests;
- the current 997-game readiness run now has a revision-backed roster receipt
  path: 610 fixtures have a confirmed exact five-role lineup from a strictly
  pre-event page revision, while 387 remain explicitly unavailable because the
  historical page did not resolve to one unambiguous five-role roster;
  predictions remain unavailable for every map because the frozen ledger does
  not contain a pre-event patch binding, full mechanics execution is not yet
  available, and 387 roster receipts are unresolved. The primary gate
  therefore reports 0.0 accuracy with 0% coverage; this is a
  coverage/authority result, not a trained-model accuracy claim.

The exact-patch CommunityDragon bridge maps each public 2026 label to its
same-minor client namespace (`26.13` → `16.13`) and records both labels in
the packet. It rejects a payload when the requested client namespace does not
match, emits exact cells only for extracted base stats, and keeps ability
formula graphs and item payloads semantic-only until execution semantics are
implemented and tested. The packet index is maintained in the local ignored
cache at `data/lol/knowledge/patch-packets/cdragon/matrix-manifest.json`.
The tracked 26.16 source receipt is the authority record for this candidate.
This makes patch source available, but does not make the full game mechanics
executable.

The 26.16 CommunityDragon capture is recorded by
`data/lol/v2/champions/lcc-atom-refresh-26.16-receipt.json`. Its raw packet is
an ignored local cache. The receipt binds its manifest and source hashes to the
26.16 public label, the 16.16 client namespace, and the atom bridge used for
the Scryglass refresh.

## Current authority artifacts

The roster capture is resumable and preserves both the revision-history JSON
and the exact selected revision's wikitext plus rendered HTML. It lives at
`data/lol/v2/experiments/leaguepedia/manual-run-2026-07-31/roster-receipts-v1/`.
The receipt manifest is hash-bound to 997 fixture rows and 279 team histories:

- manifest SHA-256: `b27611f2381b1f8612568263bb1d6031c20cecac349c5c3db597c71ad0d04d24`;
- lineup receipt file SHA-256: `25c6c91fb97c86f5e0f69423d459a60038016a6ddea9b565ba003d02d9b26992`;
- confirmed fixture receipts: 610;
- unavailable fixture receipts: 387.

The exact-source client probe manifest is hash-bound separately in the local
packet cache. The tracked candidate receipt binds the 26.16 public label to
the 16.16 client namespace and the atom bridge. Ability, item, rune, and
game-system execution remain blocked or semantic-only.

The result-free Leaguepedia patch capture is at
`data/lol/v2/experiments/leaguepedia/manual-run-2026-07-31/leaguepedia-patch-receipts-v1/`:

- manifest SHA-256: `c41fc2be8b22ddeb5b6e6a1658167af3f0421ddfd272766bb1dddda91bd4c985`;
- 988 of 997 fixtures have an exact `ScoreboardGames.Patch` value;
- 450 are on 26.13 and 538 are on 26.14; nine have no patch field;
- the capture requests no winner, kill, length, or other result field;
- 0 are pre-event-authorized because the capture was made after the fixture
  cutoffs.

This is now the preferred retrospective patch crosswalk. The GRID crosswalk
below remains useful as an independent identity check for the 245 fixtures it
can match exactly.

The historical patch-revision recovery is at
`data/lol/v2/experiments/leaguepedia/manual-run-2026-07-31/leaguepedia-patch-revisions-v1/`:

- manifest SHA-256: `80bf8a2c99bf2f792faf07d7b94bb6343928ea1265f4b14e09ac40212524401a`;
- 794 fixtures have a `SetPatch` value from a Data-page revision strictly
  before the fixture cutoff;
- 143 have a blank pre-event patch directive;
- 60 have a pre-event revision that conflicts with the later retrospective
  scoreboard patch and remain blocked;
- only the extracted patch value and evidence hash enter the engine; the full
  revision payload is retained for audit.

The readiness runner now binds those 794 patch receipts to the exact client
packets. The remaining 203 continue to carry explicit patch blockers.

The GRID identity crosswalk is kept separate from predictive inputs at
`data/lol/v2/experiments/leaguepedia/manual-run-2026-07-31/grid-patch-receipts-v1/`:

- manifest SHA-256: `e437cbd99880762d4d0ea299abdd1e397972cb97023b0c3ffdc3787f3c2e4f87`;
- 245 fixtures have exact retrospective team/time/champion/player identity;
- 752 are unavailable (751 without an exact identity and one ambiguous match);
- 0 are pre-event-authorized because the local GRID catalog was captured on
  July 28, after the evaluation window began;
- winner and final-state fields are excluded from emitted receipts.

This crosswalk can validate identity and patch joins, but it cannot authorize a
prediction after the fact. A GRID capture made before each fixture cutoff, or
another independently timestamped pre-start patch receipt, is required for
strict evaluation.

The substitution path is deliberately conservative. For LYON's July 25
fixtures, the selected historical team-page revision contains both Inspired
and Armao as jungle candidates. The receipt records
`active_roster_role_jungle_arity_2` and remains unavailable rather than using
the retrospective draft row to decide who was known before the game. A
pre-start scheduled-lineup or GRID identity receipt is the next authority
needed for that map.

The readiness report consumes both artifacts at
`data/lol/v2/experiments/leaguepedia/manual-run-2026-07-31/mechanics-engine-v1/evaluation.json`.
The 80% search is intentionally not started while full mechanics execution is
blocked, the roster denominator is incomplete, the patch binding is not
pre-event-authorized, and no prediction is available.

## The conceptual correction

The word “champion” currently hides several different objects:

1. a patch-versioned ruleset (stats, abilities, costs, cooldowns, tags,
   transformations, and exceptions);
2. a legal role assignment in a draft;
3. an item/rune-dependent state transition system;
4. a player’s observed ability to execute that system;
5. a statistical residual learned from outcomes under a particular population.

These must not be collapsed into one champion identity coefficient. The
serving engine should consume a patch snapshot and compose mechanics. The
statistical layer should estimate the parts the mechanics cannot identify,
with support and uncertainty. A low-dimensional topology may summarize the
mechanical or residual space for inspection, but it is not the source of
truth and cannot replace the underlying features.

## Source precedence

For a requested patch, the source contract is:

1. patch-pinned client data for exact numeric values and machine-readable
   identifiers;
2. the League Wiki for human-readable semantics, interaction notes, patch
   history, and citations;
3. observed match data for what was actually drafted and what happened;
4. statistical estimation for residual execution/context effects.

If two sources disagree, the cell is not silently averaged. It becomes a
reconciliation record with both payload hashes, timestamps, and a blocked or
reviewed status. A Wiki page is not allowed to manufacture a number that the
patch data does not establish.

## Proposed data layers

### 1. Immutable patch packet

One packet per patch and region/language where relevant:

- champion base stats and level-growth rules;
- spell/passive definitions and rank values;
- item and rune stats, costs, recipes, tags, and effect formulas;
- game-system modifiers (damage types, penetration, healing, shields,
  crowd-control rules, terrain, objectives, and map rules);
- source URLs, retrieval timestamps, exact bytes, hashes, and source status.

### 2. Typed mechanics kernel

Represent effects as typed operations rather than prose labels:

```text
Damage(amount, type, scaling, target_filter, timing)
ApplyCC(kind, duration, tenacity_rule, target_filter)
ModifyStat(stat, formula, duration, stack_rule)
Move(distance, direction, collision_rule)
CreateZone(shape, duration, effects)
Mark(target_filter, duration, consume_effect)
Summon(entity, duration, behavior)
Transform(state, trigger, reversible)
```

The kernel should be deterministic and testable against hand-written
micro-scenarios: level 1/6/11/16 stat checks, one spell against one target,
penetration order, shields, healing reduction, crowd-control duration, and
multi-target interactions. It is not a full game emulator at first. It is a
typed partial evaluator with explicit `unknown` results when a rule is not
implemented.

### 3. Composition evaluator

Evaluate a draft as a set of legal interactions over typed mechanics:

- lane and role pressure;
- ally enablement and protection;
- access, peel, zone, and target restrictions;
- damage-channel coverage and resistance pressure;
- objective and wave-control capacity;
- timing windows and scaling;
- counter interactions whose target filters actually overlap.

This is where order-1 through order-9 terms belong. They should be generated
from typed effects and factorized attributes, with empirical-Bayes backoff:

```text
exact 9-way mechanic pattern
  -> exact lower-order typed pattern
  -> role/patch mechanic class
  -> neutral prior
```

An unsupported nine-way interaction therefore contributes no invented value;
it falls back and reports its support state.

### 4. Contextual residuals

Only after the mechanic-first baseline is frozen should we add:

- player × champion execution residual;
- player × role residual;
- player-specific ally and enemy residuals;
- patch/league/team residuals;
- roster-change intervals and substitute identity.

These are contextual observations, not mechanics, and must remain time-safe.

### 5. Outcome and calibration layer

The model target remains a descriptive pre-map association. Evaluation must be
chronological and include calibration, log loss, Brier score, support, patch
stress, league stress, and replay parity. Accuracy alone cannot promote a
mechanics claim or a public probability.

## Obsidian vault role

The League Wiki vault is a source and review layer. It should contain
revisioned Markdown/wikitext pages, not be imported wholesale into the
serving bundle. The ingestion tool at
`lol_kills/knowledge/league_wiki_vault.py` creates a resumable catalog and
stores page/revision hashes. The first run should inventory all selected
namespaces; the mechanics extractor should then materialize only the pages
needed for the current patch packet while preserving the full catalog for
auditability.

The vault is deliberately separate from the patch packet: a page can explain
an interaction without providing an executable, patch-specific formula.

The current local capture has a validated 311,560-entry catalog across the
selected source namespaces. Text bodies are complete for namespaces 0, 4, 10,
12, 14, 110, and 828; File pages remain cataloged as metadata rather than
copied as hundreds of thousands of media documents. The checkpoint is
explicit in `snapshot-manifest.json`, and the validation report records three
legacy latest rows that predate or sit outside the catalog. No source bytes
are discarded to make the count appear cleaner.

## First build slice

The next meaningful vertical slice is:

1. ingest the current champion/item/ability JSON from a pinned client-data
   snapshot;
2. ingest champion and patch-history Wiki pages into the vault;
3. normalize one champion (Aatrox is the initial fixture) into typed stats,
   spell coefficients, tags, and unresolved exceptions;
4. write micro-scenario tests for the normalized kernel;
5. compare the mechanics feature output against the existing neutral draft
   baseline on a chronology-safe holdout;
6. retain statistical residuals only where the mechanics layer cannot explain
   observed performance.

This is the path from “champion identity” to a falsifiable rules-and-residuals
engine. It does not assume that an 80% map forecast is attainable; it creates
auditable reasons when the model cannot know.
