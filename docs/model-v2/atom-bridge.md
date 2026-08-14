# Atom bridge: League Combat Calculator mechanics -> Scryglass v2

Status: **development infrastructure, version 1 — no authority granted**
Schema: `scryglass.lcc-atom-bridge.v1` | Builder: `lol_kills/v2/champions/atoms/**`
Artifact: `data/lol/v2/champions/lcc-atom-bridge-v1.json`

## Purpose

Scryglass v2 models champions through an interpretable, reviewed semantic
ontology (L3: 12 dimensions, label probabilities, uncertainty, human review
trail). The League Combat Calculator (LCC) models champions through
*mechanistic behavior atoms* — typed facts per game object with dual
provenance (League Wiki page + game binary). This bridge is the pinned,
provenance-bearing connection between the two layers: it imports LCC atom
data read-only, maps atoms onto ontology dimensions, and exposes champion
atom profiles that downstream Scryglass models (draft interactions L6,
terminal Draft Score L7, tier lists L9, team ratings L5) may consume as
structured priors.

Nothing in this bridge is a measurement, an outcome calibration, or a
publication claim. Atom-derived values are **prior generators**: consumers
must keep them distinguishable from empirical residuals and review-accepted
ontology values, and must fail closed when the artifact or its pinned sources
are unavailable.

## Patch identity and the 26.16 refresh

The public Riot patch label is `26.16`. The client-source namespace is
`16.16`. Scryglass keeps both labels in every patch receipt. It never uses the
client label as a public patch name.

The 26.16 source receipt is
`data/lol/v2/champions/lcc-atom-refresh-26.16-receipt.json`. It binds the
CommunityDragon manifest, the 16.16.1 LCC source, the 173-champion atom bridge,
and the LCC commit used to build it. The raw packet remains a local ignored
cache. The receipt therefore proves the source capture and bridge input, not
that the raw packet is part of the repository.

The R9 depth-2, depth-3, depth-4, and state-space production features from the
Scryglass Pi research sessions remain in the production scorer. Their current
aggregate files are separate certified development artifacts. They are not
silently relabelled as a 26.16 depth refresh until the full numeric corpus is
rebuilt and evaluated against the 26.16 receipt.

## Architecture

```text
League Combat Calculator (external repo, read-only)
  data/atoms/atom-summary.json        champion x family presence index (173)
  data/atoms/classification-report.json  classifier totals, damage-type mix
  data/wiki-atoms/atom-relations.json    directed mechanic-interaction graph
  data/atoms/<champion>.atoms.json       per-champion atom detail (26.16: 173)
  data/champions.json                    identity, positions, roles, ratings
          |
          |  lol_kills/v2/champions/atoms/lcc_sources.py (pin by sha256)
          v
  Bridge builder (bridge_v1.py)
    - champion identity crosswalk  (LCC key -> riot:champion:<ddragon id>)
    - atom profiles                (family presence + counts, damage-type mix,
                                    relations, LCC positions/roles/ratings)
    - mapping table                (55 curated atom -> dimension/label rows)
    - family fallback              (used only when a source row lacks atom detail)
    - soft ontology prior          (weighted atom evidence per dimension,
                                    normalized, with evidence-derived
                                    uncertainty; unavailable when no evidence)
          |
          v
  data/lol/v2/champions/lcc-atom-bridge-v1.json
    schema_id, version, generated_at, artifact_sha256 (canonical),
    provenance (LCC repo path, git commit, per-file sha256)
          |
          +-- consume.py (fail-closed reader) --> L6 draft interactions,
                                                  L7 terminal Draft Score,
                                                  L9 tier lists, L5 team rating
```

## The two-layer champion model

| Layer | Owner | Content | Provenance | Status semantics |
|---|---|---|---|---|
| Mechanistic atoms | LCC | typed behavior facts (family, behavior, trigger, target_policy, parameters, relations) | wiki page + game binary | classified, sanity-checked (19/19) |
| Semantic ontology | Scryglass L3 | 12 dimensions x labels with probabilities + uncertainty | human review log + sources | review-accepted or prior |

The bridge only flows **atoms -> ontology prior** and **atoms -> structured
features**. The reverse direction (ontology -> LCC module hints) is a
coordination offer to the LCC thread, not implemented here.

## Atom -> ontology mapping (v1)

55 rows; every row = `(atom_id, dimension, label, weight, evidence_note)`.
Weights: 1.0 strong defining signal, 0.5 clear supporting signal,
0.25 weak/contextual. Examples:

| Atom | Dimension | Label | Weight | Note |
|---|---|---|---|---|
| crowd-control-mobility.stun | crowd_control | stun | 0.7 | defining atom |
| crowd-control-mobility.root | crowd_control | root | 0.7 | defining atom |
| crowd-control-mobility.knockback | peel | forced | 0.5 | pushes threats off allies |
| heal-shield.heal | sustain | self_heal | 0.7 | defining atom |
| heal-shield.shield | sustain | shield | 0.7 | defining atom |
| damage.execute | damage_profile | burst | 0.7 | finishing burst tool |
| damage.dot | damage_profile | poke | 0.6 | ranged attrition |
| damage.aoe | damage_profile | teamfight | 0.6 | clumped-fight value |
| stack-transform-summon-resource.stack | scaling | late | 0.5 | stacks scale late |
| vision-economy.stealth | target_access | reposition | 0.5 | reposition w/o vision |

Negative labels (`none`, `low`, `squishy`, ...) are never driven by atom
presence; absence of evidence yields `status: unavailable` per dimension, not
a measured zero (same rule as L5 lineup synergy).

## Fail-closed rules

1. Artifact must pass canonical-sha256 validation on every load (consume.py).
2. Pinned LCC file hashes must match; a changed LCC file invalidates the
   artifact until the bridge is rebuilt and re-pinned.
3. A champion with no atom evidence gets `unavailable` priors, never zeros.
4. Family-only champions (no atom detail) get only family-fallback priors and
   are labeled `profile_status: family_only`.
5. Consumers must not present atom-derived values as empirical residuals,
   review-accepted ontology, or outcome-calibrated numbers.

## Current coverage

- 173/173 champions profiled (identity crosswalk complete).
- 173/173 champions carry 26.16 atom detail from the refreshed LCC binary
  extraction. This is mechanistic source data, not a reviewed game emulator.
- 55 mapping rows, 7 family fallbacks, 20 relation edges (from
  atom-relations.json).
- Damage-type coverage remains source-dependent. Missing values stay null.

## Regeneration

```bash
python3 -m lol_kills.v2.champions.atoms.bridge_v1
# or with an alternate LCC checkout:
SCRYGLASS_LCC_REPO=/path/to/league-combat-calculator \
  python3 -m lol_kills.v2.champions.atoms.bridge_v1
```

Rebuild whenever LCC data changes (patch day), then re-run
`tests/model_v2/champions/test_atom_bridge.py` and update this doc's coverage
numbers if they moved.

## Draft-score evidence (R-22, development only)

`lol_kills/v2/draft/terminal/development_evaluation_v4_atoms.py` runs the
frozen clustered cohort through two atom-aware candidates in the same
chronological folds as the existing role-additive candidate:

| Fold | m0 (role-additive) test LL | m3 (+ atom presence) | m4 (atoms only) |
|---|---|---|---|
| outer-00 | 0.69819 | 0.69664 (-0.00155) | 0.70204 (+0.00384) |
| outer-01 | 0.75501 | 0.75438 (-0.00063) | 0.75548 (+0.00047) |
| outer-02 | 0.66031 | 0.66095 (+0.00064) | 0.66270 (+0.00239) |

Interpretation (adaptive development diagnostics, no authority):

- Atom family presence adds a small incremental signal on top of champion
  main effects (m3 improves 2/3 folds by ~0.001 log loss); it does not
  replace identity.
- Atom features alone (m4) are close but lose to role main effects.
- The structural zero-play transfer check passes: a champion with zero cohort
  appearances (Master Yi) receives a nonzero composition value from atom
  presence by construction, which role main effects structurally cannot do.
  This is the L3 "archetype transfer structurally possible without an
  empirical residual" definition of done, now demonstrated with mechanistic
  atoms.
- Artifact: `data/lol/v2/models/draft-terminal/development-evaluation-summary-v4-atoms.json`
  (sha256 `d4e107fc9910f67cfa237e65fa5f251fb7bf6d1461a9504e5aafce5f74016ea5`).
- Decision rule (R-22): select only if paired deltas are nonpositive on
  independent evaluation. **Result (2026-08-07):** deltas are tiny and not
  consistently signed (−0.0016/−0.0006/+0.0007 log loss), and atom-only (m4)
  loses on every fold — **not selected**. Atoms remain descriptive priors.

## Ontology seeding (v1)

The bridge is the sanctioned zero-play prior path. `lol_kills/v2/champions/atoms/seed_ontology_v1.py`
emits the full 173-champion ontology seed
(`data/lol/v2/champions/champion-ontology-seed-26.16.json`) at patch
**26.16**, one role profile per legal role. The frozen
`champion-ontology-seed.json` remains unchanged because the C1 trust root
binds it by hash.

- available atom dimensions → mapped label probabilities with the bridge's
  dimension uncertainty (per label); unavailable dimensions → explicit
  uniform prior with maximum uncertainty (1.0) — honest ignorance, never
  fabricated zeros.
- the four hand-authored champions (Ziggs/Xerath/Vel'Koz/Neeko) keep their
  existing 26.14 profiles untouched; bridge profiles are additive.
- every profile references the new `atom_bridge` source kind
  (`source:lcc-atom-bridge-v1`, private pending review) declared in
  `champion-ontology-sources-26.16.json`.
- reproducible: `python3 -m lol_kills.v2.champions.atoms.seed_ontology_v1`
  regenerates the identical seed; `tests/model_v2/champions/test_atom_seed_v1.py`
  enforces coverage, fail-closed priors, idempotent sources, and CLI determinism.

Ontology coverage is now 173/173 champions at the current patch, which
unblocks role-legal archetype priors for tier lists and interactions across
every league scope (previously only 4 champions had priors).

## L5 policy / lineup-synergy estimand opener (v1)

`lol_kills/v2/ratings/team/estimands_v1.py` is the first downstream consumer
that treats the bridge as a champion-composition channel (not just a prior):

- **policy weights**: role-normalized time-safe resource deviations, shrunk by
  kappa toward the reference policy (mathematical-contract §Team Rating);
- **lineup synergy** `gamma^q`: composition residual `psi = phi - proj_span(phi)`
  (atom composition projected off the policy-weighted player span through the
  roster composition matrix), with a normal-normal strong-shrinkage update;
- **identification audit**: within-roster policy variation, orthogonalization
  residual ratio, posterior dependence, source removal jackknife, design rank
  — verdict strong only when every gate passes;
- fail closed: when the audit is weak (or inputs are inadmissible) the caller
  keeps the null-with-blocker fallback; no separate policy/synergy facts are
  ever fabricated. `aggregate_team_rating(..., estimand_inputs=...)` exposes
  components only in the strong case; claim ceilings and rank eligibility stay
  false.
- tests: `tests/model_v2/ratings/team/test_estimands_v1.py` (strong opens,
  weak fails closed, unknown champion fails closed, end-to-end wiring).

## Open coordination items (LCC thread)

1. Regenerate the full 173-champion `*.atoms.json` set (needs `data/bin`
   game binaries via `scripts/decompose_binaries.py` + `scripts/extract_atoms.py`)
   so every champion gets atom-level detail. LCC thread will ping when done.
2. **Resolved 2026-08-13**: canonical data patch is **26.16**. The public
   label is pinned separately from the `16.16` client namespace in the refresh
   receipt and bridge provenance.
3. Reciprocal mapping table in LCC for `analyze-champion`/`atomizer`.
