# Public Draft Score 0.710 handover

## Goal state

The active goal is:

> Build and publicly promote a leakage-safe Scryglass Draft Score model with
> pooled frozen-evaluation AUC above 0.710, calibrated win probability, a
> separate controlled Draft Score contribution, a public side recommendation,
> and no betting, odds, expected value, or stake outputs.

The goal is active. It is not complete. The current model has passed the
development threshold. Its prospective holdout has 13 selected games. The
frozen gate needs 100 selected games and enough league coverage before outcomes
can be opened.

## Checkout

- Worktree: `/private/tmp/scryglass-atomized-public.UN76kO`
- Branch: `codex/draft-score-0710`
- Last implementation head before the handover commit: `7d12d3e8e6e68adb3c196b5a77160fa00c683be4`
- Worktree state: clean
- Remote work: none
- Pull request: none
- Deployment: none
- Supabase change: none
- Merge policy: manual merge only after required checks pass

Do not continue in `/Users/river/Projects/scryglass`. That checkout contains
other work. Continue in this isolated worktree or make a new isolated worktree
from this branch.

## Frozen model

Candidate:

`data/lol/v2/evaluation/public-draft-score-selective-candidate-v34.json`

Candidate file SHA-256:

`b0de3d4d0e683a3cdbaa71fe7282817e95538f5f1dbd0f992a646f64cedb2b10`

Candidate receipt:

`f5e895bb60946c9e6c2fcb528245597236e15960c1cf83ca835fadb56e750757`

Frozen voter weights:

- Quantum random forest: `0.40`
- Roster model: `0.20`
- Identity model: `0.10`
- All-atom identity logit: `0.30`

The last slot retains the historical field name `development_composite` in the
sealed table. It uses the repository all-atom identity logit. It is not R9E.
R9E has no public authority.

Development evidence:

- Eligible maps: `1,703`
- Selected maps: `1,482`
- Coverage: `0.870229`
- AUC: `0.7108585859`
- Brier score: `0.2148938742`
- Log loss: `0.6197182809`
- Ten-bin ECE: `0.0173955`
- Same-row quantum AUC: `0.70885668`
- Series bootstrap median AUC: `0.71173275`

These are development results. They do not grant public authority.

## Active frozen protocol

Protocol:

`data/lol/v2/evaluation/public-draft-score-promotion-protocol-v39.json`

Protocol file SHA-256:

`6a95f63d5a00b229967379a13be2eb4e213aa2184d87d5b3ce9ae3f2bdda52bf`

Version 39 binds:

- the candidate and every model implementation;
- source preparation, sealing, inventory, and final evaluation;
- the role-matched champion swap;
- the counterfactual atom feature build;
- the paired Draft intervention seal;
- independent decision verification;
- the final promotion receipt and public result builders.

Do not modify this protocol. A bound code change requires a new protocol and a
new seal before outcomes are opened.

## Promotion gates

Inventory must pass before any outcome join:

- At least `100` selected games.
- At least `75%` selected coverage.
- At least three leagues with `20` selected games each.

The one-time evaluation must then pass:

- AUC above `0.710`.
- Brier score no worse than the same-row quantum voter.
- Log loss no worse than the same-row quantum voter.
- Ten-bin ECE at or below `0.08`.
- Series-cluster bootstrap median AUC above `0.710`.

If any gate fails, public probability and recommendation remain unavailable.

## First prospective batch

Official Oracle's Elixir Drive file:

- File ID: `1hnpbrUpBMS1TZI7IovfpKeZfWJH1Aptm`
- Title: `2026_LoL_esports_match_data_from_OraclesElixir.csv`
- Last checked size: `60,726,444` bytes
- Last checked modification: `2026-08-17T01:06:47.570Z`
- Source SHA-256: `a797763fbb1f1a8e95df43580458d52d0abe0a179e97c316e809e0c31dc7b82e`

The source contains 14 target-league games from August 16:

- LPL: `8`
- LEC: `4`
- LCS: `2`

Source patch `16.16` maps to public patch `26.16`.

### Observed seal

Observed output:

`/private/tmp/scryglass-v39-holdout-batch-20260816-sealed-v1.parquet`

Observed output SHA-256:

`5ff3c9cf6d9b38806f78fd8d796ca5c8bacf87f36ff9237302ed88851b1b964a`

Observed receipt:

`/private/tmp/scryglass-v39-holdout-batch-20260816-sealed-v1.receipt.json`

Observed receipt file SHA-256:

`b6fb662d272210aa37f45e0a3934e8071a9df818317d4c5877802cdfa278d6ef`

Observed internal receipt:

`468b220f2650c7876b2a6d80ee1e6dfbdedad1a3901de538e071bf2c7ea832fe`

### Paired Draft seal

Paired output:

`/private/tmp/scryglass-v39-holdout-batch-20260816-paired-sealed-v1.parquet`

Paired output SHA-256:

`9de0d1f437ad87e3478946d9bd2270af6cacdff33a928d5dbe7663119ff2a3f5`

Paired receipt:

`/private/tmp/scryglass-v39-holdout-batch-20260816-paired-sealed-v1.receipt.json`

Paired receipt file SHA-256:

`68cbbe130a7458a77c9ae0e177c4d741fbcc8e7393e3d4758ab4eb4423a3b409`

Paired internal receipt:

`10f5531f9ff06ae6396ccb7dfc8315fe862f39354ac7ecc25fabad61af202758`

### Inventory

- Eligible games: `14`
- Selected games: `13`
- Coverage: `92.86%`
- Selected LPL games: `7`
- Selected LEC games: `4`
- Selected LCS games: `2`
- Inventory receipt: `e407bfdce6a8daa53c3f090dc643c6e1949a9fe135b774b2cab27ac3b4f89716`
- Outcomes opened: no
- Outcomes may be joined: no

The observed and swapped passes each select 13 games. The blind controlled
Draft edges range from `-5.64` percentage points toward Red to `+6.00`
percentage points toward Blue. This statement uses predictions only.

## Controlled Draft estimand

The public result keeps the full match prediction and Draft contribution
separate.

For each map:

1. Score the observed ten-player draft.
2. Exchange the Blue and Red champions within each role.
3. Keep players, teams, roles, side, date, league, ratings, uncertainty,
   momentum, and match context fixed.
4. Build current-map atom features from the swapped champions.
5. Keep historical state updates on the observed lineups.
6. Score the swapped draft before outcomes are available.

Let `L_observed` and `L_swapped` be Blue win logits.

```text
fixed_strength_logit = (L_observed + L_swapped) / 2
controlled_draft_logit = (L_observed - L_swapped) / 2
draft_edge_pp = 100 * (sigmoid(controlled_draft_logit) - 0.5)
```

The full observed probability drives the side recommendation. The controlled
logit drives the Draft Score. A shared strength shift cannot change the
controlled Draft value.

## Important source artifacts

- Blind observed features:
  `/private/tmp/scryglass-v34-holdout-batch-20260816-blinded-v3.parquet`
  SHA `3b59195b648ef534e5f4099f4b671e7cdef731b3fee71efc2053c2a685c0ab86`
- Blind observed players:
  `/private/tmp/scryglass-v34-holdout-draft-source-through-20260816-blinded-v3.parquet`
  SHA `b8034e98eda78c17416678769df100c8a894312bf2f69d94523c82e33ad2ea49`
- Observed voters:
  `/private/tmp/scryglass-v34-holdout-batch-20260816-voters-v3.parquet`
  SHA `0a1e5314cef9a646d5a452e32c9b1a982c95a7e328a1025108c2a9153221b430`
- Observed voter receipt file:
  `/private/tmp/scryglass-v34-holdout-batch-20260816-voters-v3.receipt.json`
  SHA `ee9a12fb00318128a606a28095fe6e423e340586011f29d8c746d968836c89f6`
- Role swap batch:
  `/private/tmp/scryglass-v35-holdout-batch-20260816-role-swaps-v2.json`
  SHA `0af66b2625b3f1f788d1bf2671004fd4b27a5b86ba836ae2a74a2eb2b9bc1382`
- Swapped blind features:
  `/private/tmp/scryglass-v35-holdout-batch-20260816-swapped-blinded-v1.parquet`
  SHA `61d7ccf90cebe8042ba39a1f6bb7bb9216cf20a62b72c91ffae78f702272837f`
- Swapped voters:
  `/private/tmp/scryglass-v35-holdout-batch-20260816-swapped-voters-v1.parquet`
  SHA `48bfed2962d49e15ffb2dc76a1d987ec70ad887b73a4bd056baa2a6c9c6371c2`
- Swapped voter receipt file:
  `/private/tmp/scryglass-v35-holdout-batch-20260816-swapped-voters-v1.receipt.json`
  SHA `35cc4296b4bf9408b766fcb0054784e57fdd1821c1ea6861eb461cbae11616cd`

## Verification completed

- Focused Python model and authority suite: `103 passed`
- Latest protocol and paired authority suite: `81 passed`
- Scryglass app unit tests: `155 passed`
- TypeScript: passed
- ESLint: zero errors, one existing unused request warning
- Production build: passed
- Public boundary audit: passed
- Python compileall: passed
- Git diff check: passed

The production build emits two known font fallback warnings for Atkinson
Hyperlegible Mono and Atkinson Hyperlegible Next.

## Resume procedure

1. Read `docs/public-draft-score-promotion-runbook.md`.
2. Check the official Drive file metadata. Download only a newer revision.
3. Accept new source rows through the normal OE identity and patch checks.
4. Prepare a new date-bounded blind feature and player batch.
5. Fit the frozen voters. Do not refit the candidate or change weights.
6. Create the role-matched swap batch before any outcome read.
7. Build counterfactual features with observed historical updates and swapped
   current-map champion features.
8. Seal observed predictions under protocol v39.
9. Seal paired interventions under protocol v39.
10. Add the new observed receipt to the inventory in chronological order.
11. Keep outcomes closed until every inventory gate passes.

When inventory passes, create one minimal outcome file with only `game_uid` and
binary `y`. Run the evaluation once. Pass both observed and paired path pairs:

```zsh
python3 -m lol_kills.research.evaluate_selective_draft_holdout \
  --protocol data/lol/v2/evaluation/public-draft-score-promotion-protocol-v39.json \
  --protocol-sha256 6a95f63d5a00b229967379a13be2eb4e213aa2184d87d5b3ce9ae3f2bdda52bf \
  --candidate data/lol/v2/evaluation/public-draft-score-selective-candidate-v34.json \
  --candidate-sha256 b0de3d4d0e683a3cdbaa71fe7282817e95538f5f1dbd0f992a646f64cedb2b10 \
  --receipt "$OBSERVED_RECEIPT_1" \
  --sealed "$OBSERVED_OUTPUT_1" \
  --paired-receipt "$PAIRED_RECEIPT_1" \
  --paired-sealed "$PAIRED_OUTPUT_1" \
  --outcomes "$MINIMAL_OUTCOMES" \
  --outcomes-sha256 "$OUTCOMES_SHA256" \
  --output "$EVALUATION_RECEIPT"
```

Repeat all four batch arguments for each chronological batch.

## Public release work still required

The exact three-state release design is in
`docs/public-draft-score-promoted-release-contract.md`, added by commit
`3df747f1`. Use that contract for the publisher, migration, RPC, Storage, and
web implementation.

The research and receipt chain is ready. The current publication boundary still
rejects `status: promoted` by design:

- `lol_kills/export/public_pack.py` emits descriptive or unavailable authority.
- `lol_kills/export/supabase_publication.py` rejects promoted authority.
- `supabase/migrations/20260815060000_descriptive_draft_authority.sql` accepts
  only unavailable or descriptive authority.
- `supabase/migrations/20260815060001_descriptive_draft_query_api.sql` exposes
  descriptive authority only.
- `apps/scryglass/src/lib/pack.ts` makes
  `hasPromotedDraftAuthority()` return `false` for every manifest.

Do not open the browser gate alone. Implement the promoted lane after a valid
promotion receipt exists, or in a separate fail-closed slice that requires that
exact receipt schema and hashes.

The promoted release slice must:

- verify the complete independent promotion receipt at pack build time;
- bind its candidate, protocol, evaluation, outcomes, paired interventions,
  decision, model version, and release ID;
- publish only the fixed public probability, controlled Draft Score, and side
  recommendation fields;
- update the final Supabase activation and RPC projections;
- let the web client accept promoted authority only from the sanitized active
  manifest and release-bound result;
- keep betting, odds, expected value, and stake fields absent;
- add zero-state migration, publisher, rollback, corruption, and browser tests.

## Branch commits

```text
73fbdc0e research: freeze selective draft probability candidate
fa54e69e research: seal post-freeze draft holdouts
f93387fb research: remove holdout result dependency
dbe32f4f research: prepare blind draft holdout batches
ea0aa8af research: gate one-time draft holdout evaluation
2df1903a research: verify independent draft promotion
8b2ad636 research: bind public draft output to promotion
783203a6 research: bind holdout batches to frozen protocol
149b71d3 docs: add public draft promotion runbook
719a73df research: define controlled draft contribution
ed59e372 research: bind paired draft intervention
3ed94d9f research: derive public draft score from paired inference
96c1bd20 docs: define controlled draft intervention
4147b160 research: bind paired public result in holdout seal
89a936d3 research: freeze paired public draft protocol
224a3512 test: make paired protocol the active freeze
2248e727 research: seal outcome-blind draft interventions
b8360c3e research: bind controlled draft inference protocol
c6749078 docs: record paired draft holdout seal
a1b5dcce research: bind paired interventions to final evaluation
bd908394 docs: record evaluation-bound paired holdout
40d356db research: bind paired evidence to promotion receipt
7d12d3e8 docs: record promotion-bound paired holdout
```

## Process cleanup

At handover time there were no active goal-related Python model processes.
Check before resuming:

```zsh
ps -axo pid,pcpu,pmem,command \
  | rg 'python|Python' \
  | rg 'selective_draft|draft_score|scryglass'
```

Terminate only processes that belong to this goal. Do not terminate MCP,
LiteLLM, or unrelated Python services.
