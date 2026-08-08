# Elemental Drakes

A standalone Scryglass worksheet for comparing legal elemental-dragon
inventories across two five-champion teams.

The default browser opens a Tier 1 LCK example. It also includes one curated
game from each canonical Tier 1 regional league plus international play. Each
example loads both role-ordered compositions and the complete observed capture
history. Manual selectors support arbitrary 5v5s; role-aware randomization uses
observed Leaguepedia professional role appearances. The three legal spawn
elements can also be randomized without changing capture ownership.

## What the worksheet estimates

- The joint-state curve compares each legal inventory prefix with the same ten
  champions, stage, reference time, and standardized neutral state at 0/0
  inventory.
- Team totals are computed by scoring both mirrored perspectives, reconciling
  their logits, and applying the logistic transform.
- Champion lines show only differences above the common team effect. The
  common component appears once per team instead of being divided by five and
  repeated on every row. Supported champion-element estimates are regularized
  and partially pooled toward the pooled and archetype terms; tagged cells
  below the release threshold use the archetype prior, while unsupported
  untagged cells remain at zero differential. An unassigned remainder keeps the
  ledger reconciled to the team total.
- The six-element matrix reports comparable marginal capture estimates: first
  global dragon, second global dragon averaged over legal first-spawn states,
  and one map-phase capture averaged from global capture three through soul.
- The latest-capture recipient comparison shows Team A’s adjusted map-win
  estimate under two explicit scenarios: Team A receives the selected dragon,
  or Team B receives it at the same modeled pre-capture state. It is
  associational, not a causal contest-or-leave policy.

All five champions receive every elemental stack. A zero champion differential
means the model has no supported reason to distinguish that champion from the
team-common effect; it does not mean the champion receives no buff.

Champion-element residuals are admitted by a fixed, outcome-independent
exposure rule and constrained to be deviations from the pooled and archetype
model. The family is tuned chronologically, its expanded June vocabulary is
checked on July games, and it fails closed if it materially worsens either
evaluation. Cells that cross the support threshold only in the final refit are
labeled separately and are not claimed to have individual holdout validation.
Champion-specific soul, capture-stage, exact allied-pair, and exact
opponent-pair terms remain disabled because the current sample cannot support
them. Generic Draft Score coefficients are not added because they estimate a
different pre-match composition association.

The lineup curve uses the same median stage time with an all-zero inventory as
an explicit synthetic reference. The stage-ranking matrix instead compares
each capture with its legal pre-capture inventory. Stage 0 is unscored.

## Evidence scope

The audited cohort contains 6,504 completed professional games. Eligibility and
legal-path checks leave 6,382 modeled games and 27,689 captures. Tier 1
regional, international, and other professional coverage is reported
separately. No region or competition tier has 6,000 games, so the publication
does not call any subgroup independently normalized.

The public app contains derived model runtimes, support metadata, aggregate
coverage, archetype priors, role counts, and seven curated games. GRID
provider archives, stable provider identifiers, and credentials remain outside
the deployable directory.

The normalized warehouse keeps only the projected game, draft, objective, and
pre-capture columns: 3,296 resumable series files average about 6.4 KB each,
the consolidated Parquet inputs total about 2.0 MB, and the self-contained
deployable JSON is about 2.2 MB. Source archives are discarded after projection.

## App

```bash
cd apps/elemental-drakes
npm ci
npm run typecheck
npm run lint
npm run build
```

The Vercel project root is this directory. It is separate from the main
Scryglass publication.

## Dragon images

The six locally vendored 128×128 dragon icons are Riot Games assets sourced
through the corresponding League of Legends Wiki file pages. The public footer
links every source page and carries Riot’s required fan-project notice:

> Scryglass was created under Riot Games' "Legal Jibber Jabber" policy using
> assets owned by Riot Games. Riot Games does not endorse or sponsor this
> project.
