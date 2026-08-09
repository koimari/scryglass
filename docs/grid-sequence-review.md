# Deterministic GRID sequence review

This private workflow turns one Riot LiveStats file and one closed request into
the complete first-pass answer for a professional-game macro sequence. It is
designed to remove the manual arithmetic and scope changes that made the
MKOI–FNC review take about an hour.

## Fast path

Once the GRID file is cached and the request is written:

```bash
python3 -m lol_kills.research.grid_sequence_review \
  --request data/lol/warehouse/private_grid/sequence_review/v1/requests/series_2966877_game_3.json
```

For MKOI–FNC game 3, this replays the 105 MB source in about 1–2 seconds of
analysis time and checks 53 declared results before writing the report. A
changed timestamp, resource total, plate allocation, damage result, camp value,
turret threshold, mechanics receipt, or action-graph hash makes the command
fail instead of silently changing the answer.

## Atomic tool calls

The normal command is now only an orchestrator. Its analysis is made of 13
mandatory actions, followed by three separately callable acceptance/rendering
actions. No action can run before its declared dependencies, and every action
records a version plus deterministic input and output hashes.

List the exact contracts instead of reconstructing the workflow from memory:

```bash
python3 -m lol_kills.research.grid_sequence_actions list
```

Run through one specific action:

```bash
python3 -m lol_kills.research.grid_sequence_actions run \
  --request data/lol/warehouse/private_grid/sequence_review/v1/requests/series_2966877_game_3.json \
  --action objective_resources
```

Replace `objective_resources` with `resource_ledgers`, `siege_damage`,
`delayed_camps`, `turret_health`, `wave_counterfactual`,
`named_farm_comparison`, `verify_expected_results`, `render_evidence`, or
`render_public`. The runner executes only the required ordered prefix and
returns the requested action's typed JSON envelope and action receipt.

The full ordered path is:

1. `verify_source`
2. `verify_catalog`
3. `parse_game`
4. `verify_mechanics`
5. `locate_objectives`
6. `objective_resources`
7. `resource_ledgers`
8. `siege_damage`
9. `delayed_camps`
10. `turret_health`
11. `wave_counterfactual`
12. `named_farm_comparison`
13. `assemble_report`
14. `verify_expected_results`
15. `render_evidence`
16. `render_public`

Aliases and equivalent clocks are canonicalized before hashing: for example,
`wolves` and `wolf`, or `10:45` and `10:45.000`, produce the same analytical
input. The complete report stores the first 13 receipts; the closed request
pins their graph hash as its 53rd acceptance check.

The closed request records:

- exact series, game, provider-game identity, and source hash;
- decision, wave-comparison, and turret-siege windows;
- the explicitly involved champions and public team labels;
- analyst-supplied farm alternatives such as Gromp and Wolves;
- any observer-only turret-health estimate and its uncertainty range; and
- acceptance values for the facts already reviewed.

## Dragon-buff valuation sidecar

When the review asks for the stat-equivalent value of the drake itself, run the
separate deterministic sidecar after the Wiki effect lookup:

```bash
python3 -m lol_kills.research.dragon_gold_equivalent \
  ocean states.json --stacks 1 --missing-health 0.5 --duration 35
```

`states.json` is a closed array of champion snapshots from the declared
checkpoint. The sidecar reads item anchors from the selected CDragon packet,
fills only missing level-based base values from the fastpack, and records
unpriced components as `null`. Its output is a stat-equivalent, not direct
objective gold, observed gold, or a causal value.

The first retrieval of a new game still requires the explicit `--download`
flag. The long-form command accepts the same windows and options directly; the
cached request is the replay and regression format, not a hidden download
authority.

For a new case, the first run can save that closed request automatically:

```bash
python3 -m lol_kills.research.grid_sequence_review \
  --series SERIES_ID --game GAME_NUMBER --download \
  --sequence-start DECISION_START --sequence-end REVIEW_END \
  --resource-start WAVE_START --resource-end WAVE_END \
  --siege-start TURRET_START --siege-end TURRET_END \
  --involved ChampionA,ChampionB,ChampionC \
  --team-label 100=TEAM_A --team-label 200=TEAM_B \
  --save-request /private/path/series_GAME.json
```

Optional delayed camps and observer-health inputs can be added on that same
run. The generated request includes a larger automatically derived acceptance
set covering every table subtotal and difference, not just the headline
numbers.

## What one run emits

1. Exact objective and plate timestamps.
2. Immediate objective gold and event-adjacent XP.
3. Touch-compatible true damage, other true damage, champion building damage,
   and the explicitly conditional tower-time estimate.
4. An involved-player table whose subtotals contain only visible rows.
5. A separate five-player team table.
6. Plate gold split by recipient, with the enforced identity
   `gold without plates + plate gold = total gold`.
7. Later same-game values for explicitly named skipped camps, kept separate
   from actual totals so delayed farm cannot be subtracted twice.
8. Observer-health and fixed-state no-Touch calculations using the real 9,000
   outer-turret HP and 8,100/6,750/4,950/2,700/0 plate thresholds.
9. Named wave counterfactuals, clearly separated from actual scoreboard gains.

## Fail-closed rules learned from this case

- `−1 CS` is never displayed beside another player's actual `+29 CS` without
  labelling one as a counterfactual. Actual gains and missed-farm estimates are
  different tables.
- A selected-player subtotal is never called a full-team total. Each total
  stores and prints its included player count.
- Plate gold is a component of LiveStats total gold, not a second gain to add
  on top.
- LiveStats champion building damage is not total turret damage. It excludes
  direct minion and Voidmite attacks and cannot be subtracted from turret max
  HP to recover exact structure health.
- An observer estimate is stored as an estimate/range. Without one, exact
  turret health is unavailable.
- At 2,560 remaining HP, an outer turret has crossed four plate thresholds,
  not two. The threshold calculation is tested.
- “Skipped camps” is an analyst-supplied counterfactual. GRID verifies the
  later same-game clear and exact reward but does not invent camp availability.
- “One extra auto in ten seconds” is boundary- and passive-state-sensitive.
  The mechanics calculator reports continuous rate, two static passive states,
  two discrete timer conventions, and an explicitly idealized ramp from zero.

## Static auto/DPS follow-up

The Jax follow-up now has its own patch-pinned command:

```bash
python3 -m lol_kills.knowledge.basic_attack_profile \
  --champion Jax \
  --level 14 \
  --items 'Trinity Force,Sundered Sky' \
  --adaptive-shards 1 \
  --seconds 10 \
  --format markdown
```

It reads champion base AD, attack-speed ratio/growth, item stats, the adaptive
shard, and Jax's passive formula from hashed patch-26.15 inputs. It prints both
zero-stack and full-stack states plus the ideal uninterrupted ramp from zero.
Item procs, W/R, crits, armor, resets, and animation timing remain excluded
rather than guessed.

## Verification target

The exact-game regression must reproduce:

- dragon at 8:29.981 and Grubs at 8:31.884 / 8:41.668 / 8:45.075;
- four top plates and 480g;
- 1,840 Touch-compatible damage and 13.521 seconds under the constant-DPS
  counterfactual;
- shown-player totals of 3,312g/4,014 XP versus 2,971g/3,900 XP;
- full-team difference of MKOI +382g/−87 XP;
- Gromp plus Wolves at 185g/460 XP, or 75g/283 XP above Xin's Grub resources;
  and
- observer estimate 720 HP, fixed-state no-Touch lower bound 2,560 HP, and
  four crossed plate thresholds.

Raw GRID data, credentials, and signed URLs stay private.

## Recorded validation

On 2026-08-02, two consecutive closed-request runs took 1.289s and 1.265s of
analysis time. Both passed all 53 acceptance checks and produced the same
analysis hash:

`ee1df4f04b943a9ec2861a05be331c3d8c32a994f05dabfd68a5d8e8a39154f6`

The pinned 13-action analysis graph hash is:

`98d686d5afe940e9662c0f15449b47d058824e1f6d48a77d67c9ab2a427e58f9`

Two independent 16-action `render_public` runs produced byte-identical JSON
envelopes with SHA-256:

`cd36e7dc566b5d9aad09ce974fd32852b6333480b4fa7ed8f3468c63b3dff673`

The relevant regression and mechanics suite passed 49/49 tests. Runtime is
deliberately excluded from the analysis hash; source identity, catalog and Wiki
receipts, action inputs/outputs, windows, extracted facts, mechanics, and
modeled outputs are included.
