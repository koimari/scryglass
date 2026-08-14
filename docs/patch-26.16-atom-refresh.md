# Scryglass patch 26.16 atom refresh

Status: development input. This file grants no model or publication authority.

## Patch identity

Scryglass uses the public Riot label `26.16`. CommunityDragon uses the client
namespace `16.16`. The source receipt also records `16.16.1` when the client
build includes a suffix. The conversion is handled by
`lol_kills/v2/patch_identity.py`.

The refresh is bound by
`data/lol/v2/champions/lcc-atom-refresh-26.16-receipt.json`. The receipt covers
the CommunityDragon manifest, the captured entities, the bridge artifact, and
the LCC source commit. The raw packet stays in the ignored local cache.

The bridge has profiles for all 173 champions. The current ontology output is
`data/lol/v2/champions/champion-ontology-seed-26.16.json`. Its source metadata
is `data/lol/v2/champions/champion-ontology-sources-26.16.json`.
The bridge is stored at
`data/lol/v2/champions/lcc-atom-bridge-26.16.json`. The live tier refresh keeps
the audited 26.15 bridge at
`data/lol/v2/champions/lcc-atom-bridge-v1.json` until 16.16 games enter the
accepted source window.

The frozen C1 files keep their original bytes:

- `data/lol/v2/champions/champion-ontology-seed.json`
- `data/lol/v2/champions/champion-ontology-sources.json`

This separation prevents a patch refresh from changing the hash-bound
research foundation.

## Scryglass Pi research context

The production composition scorer keeps the Scryglass Pi research path:

- depth-2 atom descriptors;
- depth-3 state and cycle descriptors;
- depth-4 interaction descriptors;
- strictly prior team state-space features;
- atomized per-pick contributions.

The Pi round-9 same-data comparison recorded Brier `0.21708`, log-loss
`0.62330`, and AUC `0.70681` for the depth-4 plus state-space variant. The
result passed the composition gate at bootstrap 100. This is research evidence
for the scorer path. It is not a 26.16 holdout result or a public promotion
receipt.

The Pi expansion work separates the Scryglass bridge from full ability-level
atom research. The bridge currently covers 173 champions, six mechanic
families, 20 atom relations, and 12 ontology dimensions. The scorer adds
depth-2 descriptors, depth-3 state and cycle descriptors, depth-4 interaction
descriptors, strictly prior state-space features, and per-pick contributions.
These layers preserve the provenance of each pick contribution and keep the
pre-game boundary explicit.

The round-4 expansion tests covered the full semantic dimension set, the
relations graph, composition geometry, role-specific profiles, embeddings,
cold-start champions, combined features, and a regularisation floor check.
The full semantic set, graph, geometry, and PCA variants stayed close to the
21-dimension baseline. Pure atom features and role-specific profiles were
weaker. Cold-start AUC was 0.477. Atom features became useful when injected
into draft space, where all four evaluation windows improved. The best flexible
floor remained in the approximate Brier range `0.215` to `0.220` for that
dataset. These findings guide the next 26.16 evaluation. They do not grant
production authority.

The deeper ability-level corpus from the separate LCC research line is not
part of this Scryglass release. Scryglass keeps the 26.16 bridge as a staged
development input until the accepted 16.16 game rows, the chronological
holdouts, and the hash-bound promotion record are complete.

The 26.16 bridge refresh supplies new mechanistic source data. The existing
numeric aggregate files remain separate certified development artifacts until
the complete 26.16 corpus is rebuilt and evaluated. This avoids a false patch
label on a model that has not passed its chronological holdouts.

## Regional refresh boundary

A regional tier refresh was run after the latest OE download.
Its receipt is kept in the worker runtime at
`data/lol/v2/tierlists/refresh-receipts/tierlist-live-refresh-20260814T052013765685Z-7427f9957fe5fc9b.json`.
It replayed 17,503 maps, used a 1,182-map live window, and built 195 cells
with 7,546 rows. The bundle contains 1,100 regional views across 39 scopes,
covering CBLOL, LCK, LCP, LCS, LEC, LJL, LPL, PCS, TCL, VCS, and international
events. The latest accepted game data is still public patch `26.15`, and every
scope label uses the Riot namespace.

The refresh was deferred from publication. It used the audited 26.15 bridge
because the accepted OE source has no 16.16 game rows yet. The latest OE file
ends at `2026-08-14T03:34:42Z`. It contains 133 maps after the 26.16 public
release boundary, all reported as source patch `16.15`. LCK and LPL match
reports provide secondary corroboration, so a date-only rewrite would be
unsafe. The ingest now keeps the raw token and requires explicit live-realm
evidence before it derives `16.16`.

The public release boundary comes from [Riot's 26.16 patch notes](https://www.leagueoflegends.com/en-us/news/game-updates/league-of-legends-patch-26-16-notes/).
The primary per-game check is [Riot's official LoL Esports live-stats feed](https://feed.lolesports.com/livestats/v1/window/115548147900619042). It
publishes `gameMetadata.patchVersion` for a game window. Three LCK games from
the 12 to 14 August post-release window were checked directly:

The checks below were retrieved at `2026-08-14T15:06:20Z`. The live-stats
window is a rolling response, so the hash is a transport receipt for that
retrieval and is not treated as a timeless fixture.

| Official game ID | Feed patch | Public patch | Raw response SHA-256 |
| --- | --- | --- | --- |
| `115548147900684590` | `16.15.800.4844` | `26.15` | `d2892677097990caa0d1910c344a40adbdcde551250d80321a90ffeed958734f` |
| `115548147900750226` | `16.15.800.4844` | `26.15` | `9021add057f851f937d99ac484a6a38d730647009fef4e245cb8b74b94cefa26` |
| `115548147900619042` | `16.15.800.4844` | `26.15` | `6e608fb3044c65080c14387f3003ed7797426e74ed80c272029443285692032a` |

The source receipts are game-bound and hash-bound. They contain no winner or
model fields. A future accepted OE row can use the same receipt path when its
game identity is crosswalked to an official game ID. GRID is not required for
this correction. The worker accepts an optional JSON catalog at
`$SCRYGLASS_RUN_ROOT/riot-patch-receipts.json` and passes it to the OE
importer. A row with no exact Riot receipt keeps the OE token. The catalog is
never built from dates alone.

The production index remains on canonical `26.15` patch options and scope
labels. The 26.16 bridge is staged for the first accepted 16.16 ingest. It
must not be mixed into 26.15 regional ratings.

## Verification

Run:

```bash
python3 -m pytest -q \
  tests/test_patch_identity.py \
  tests/test_lcc_atom_refresh_receipt.py \
  tests/model_v2/champions/test_atom_bridge.py \
  tests/model_v2/champions/test_atom_seed_v1.py
```

The seed builder writes the versioned 26.16 files by default. It never writes
the frozen C1 seed.
