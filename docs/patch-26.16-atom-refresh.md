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

The 26.16 bridge refresh supplies new mechanistic source data. The existing
numeric aggregate files remain separate certified development artifacts until
the complete 26.16 corpus is rebuilt and evaluated. This avoids a false patch
label on a model that has not passed its chronological holdouts.

## Regional refresh boundary

A regional tier refresh was run after the label fix. Its receipt is kept in the
worker runtime at
`data/lol/v2/tierlists/refresh-receipts/tierlist-live-refresh-20260814T042036200243Z-898104ca0bcf88e7.json`.
It replayed 17,483 maps, used a 1,162-map live window, and built 195 cells
with 7,544 rows. The bundle contains 970 regional views across 39 scopes.
The latest accepted game data is still public patch `26.15`, and every scope
label uses the Riot namespace.

The refresh was deferred from publication. It used the audited 26.15 bridge
because the accepted OE source has no 16.16 game rows yet. The production index
was rebuilt with canonical `26.15` patch options and scope labels. Its
production index artifact is `7a9e1248761d11f3262e761b6839add43b2e3b917e2a948ef4c0011bc57c1d08`.
The 26.16
bridge is staged for the next complete 16.16 ingest and must not be mixed into
26.15 regional ratings.

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
