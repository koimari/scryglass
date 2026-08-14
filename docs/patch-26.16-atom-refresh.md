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

A regional tier refresh was run after the label fix and the latest OE download.
Its receipt is kept in the worker runtime at
`data/lol/v2/tierlists/refresh-receipts/tierlist-live-refresh-20260814T052013765685Z-7427f9957fe5fc9b.json`.
It replayed 17,503 maps, used a 1,182-map live window, and built 195 cells
with 7,546 rows. The bundle contains 1,100 regional views across 39 scopes,
covering CBLOL, LCK, LCP, LCS, LEC, LJL, LPL, PCS, TCL, VCS, and international
events. The latest accepted game data is still public patch `26.15`, and every
scope label uses the Riot namespace.

The refresh was deferred from publication. It used the audited 26.15 bridge
because the accepted OE source has no 16.16 game rows yet. The latest OE file
ends at `2026-08-13T20:27:53Z` and contains zero 16.16 rows. The production
index was rebuilt with canonical `26.15` patch options and scope labels. Its
index raw digest is `e8c5aab7bd365ec440c531b44b4a18c7b500218c3c69a132abc1f2ea2f7d8954`
and its embedded artifact digest is
`db4dc5b457d0f81a8e15ca003c105df69e1170c53e83ade8ca24fa3fba008592`.
The 26.16 bridge is staged for the first complete 16.16 ingest. It must not be
mixed into 26.15 regional ratings.

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
