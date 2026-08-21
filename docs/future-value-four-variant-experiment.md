# Future-value four-variant experiment

## Purpose

This experiment measures how future player form and composition scaling change Scryglass ratings, Draft Score, Tier Lists, ranks, and public-pack records.

All variants use one accepted Oracle's Elixir census. Every fold uses strict prior data. Public authority stays unavailable until the selected variant passes the release gates.

## Frozen baseline

- Release: `v2026.08.20.210112`
- Source as of: `2026-08-20T14:51:29Z`
- Accepted maps: 17,764
- Source identity: `591820cb87bcb847da449af11349c9f75f4993a9295998cd46db17e1535c5cfb`
- Model-eligible maps: 16,553
- Local freeze: `/private/tmp/scryglass-four-variant-freeze-20260820T145129`

The freeze contains the accepted census, annual source exports, OE bridge files, normalized map/player/team parquets, current rating receipt, full public pack, and production Tier List artifacts. The freeze manifest binds every file by byte count and SHA-256.

## Variants

| ID | Rating inputs | Draft Score inputs |
| --- | --- | --- |
| V1 | Current team and player ratings | Fixed atomized composition plus current rating controls |
| V2 | V1 plus future player form | V1 plus the cross-fitted future player-form component |
| V3 | V1 plus expected gold and XP curves | V1 plus signed curve shape and curve-by-atom interactions |
| V4 | V2 and V3 | V2 and V3 together |

Static champion, role, ally-synergy, enemy-counter, same-role, and archetype terms stay fixed across the four variants. This rule isolates the two new feature families.

## Draft Score component contract

Each scored map must expose these parts separately:

- current team strength
- current five-player roster strength
- future player form
- expected scaling curve
- curve-by-role, curve-by-synergy, and curve-by-counter interactions
- fixed atomized composition terms

The fitted component sum must reconstruct the independent model logit within `1e-12`. Curve fields that stay unchanged after a side swap are diagnostics or gates. They cannot enter an antisymmetric score by themselves.

## Evaluation

The evaluator uses non-overlapping chronological validation intervals and whole-series boundaries. Representation fitting, imputation, phase fitting, atom fitting, regularization, and combination weights use training rows only.

All four variants must share the same validation game IDs. The report includes log loss, Brier score, AUC, calibration, side-swap error, coverage, missingness, patch transfer, regional transfer, roster-change slices, and series-boundary evidence.

The downstream report measures:

- player and team value changes
- weekly rank movement and top-list turnover
- champion Tier List score, rank, and tier movement
- Draft Score component changes and sign changes
- match, profile, manifest, and public-pack row parity

## Promotion boundary

Research results cannot grant probability, odds, expected-value, recommendation, or betting authority. A selected variant can reach production only through `python3 -m lol_kills.public_refresh --once --force` after provenance, coverage, calibration, parity, CI, review, and rollback gates pass.
