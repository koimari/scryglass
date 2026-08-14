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

The 26.16 bridge refresh supplies new mechanistic source data. The existing
numeric aggregate files remain separate certified development artifacts until
the complete 26.16 corpus is rebuilt and evaluated. This avoids a false patch
label on a model that has not passed its chronological holdouts.

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
