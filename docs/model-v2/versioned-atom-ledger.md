# Versioned League atom ledger

## Purpose

The ledger stores League mechanics as stable atoms. A patch update is a sparse set of field changes. Every atom that has no delta keeps its prior record and record hash.

The first base contains the complete six-domain League Combat Calculator 26.15 source corpus. It has 19,852 atoms. The compressed snapshot binds the LCC manifest and each domain file with full SHA-256 values. Its authority status is `exact_patch_bound_source_corpus`.

This vertical slice has measured records in these atom categories:

`champion`, `spell`, `passive`, `item`, `rune`, `objective`, `buff`, `debuff`, `effect`, `trigger`, `target`, `formula`, `cooldown`, `cost`, `range`, `duration`, `stack`, `reset`, and `state-transition`.

These category counts describe the imported LCC slice. They do not state complete Wiki or game coverage. The base marks coverage as `measured_partial`. Missing domains include maps, minions, neutral monsters, structures, summoner spells, global rules, complete objective state machines, and complete cross-entity interactions. Full Wiki coverage remains the campaign target.

## Identity and fields

An atom ID uses the stable source identity fields below. Numeric values, formula text, and behavior do not enter the ID hash.

- Domain
- Entity
- Source atom ID
- Source locator
- Per-locator source slot

Behavior is mutable record content. A behavior or formula change keeps the atom ID and changes the record hash.

Each numeric field stores its value, unit, source, confidence, authority, and missing flag. Each atom also has an explicit missing mask. An atom with no numeric value has one missing field. This preserves the difference between an unknown value and zero.

The record hash covers the current field values and all evidence. The snapshot hash covers every record and its source binding.

## Patch replay

A delta event names its base patch and target patch. It binds the previous snapshot hash and the previous event hash. Operations use one of these forms:

- `add` introduces one new stable atom.
- `change` replaces named fields after an exact prior record hash check.
- `deactivate` keeps the historical record and marks it inactive.

The replay rejects a missing base, a broken hash chain, duplicate events, duplicate operations, two operations for the same atom in one event, an unexpected prior record hash, and a future patch. It also rejects source evidence that is later than the caller's knowledge cutoff.

The replay receipt gives the base, events, final snapshot, active count, changed count, and unchanged count. A compact per-field index gives every unchanged field the status `unchanged_with_prior_authority`. Changed fields have an explicit `refreshed_by_delta` override. This keeps unchanged record hashes intact.

Each delta declares its coverage status, candidate page-change count, parsed count, unsupported count, and unsupported list. The model-ready resolver accepts only a complete delta.

## 26.16 pilot

The pilot contains 12 explicit numeric changes from the League Wiki `V26.16` page, revision 4052214 at `2026-08-16T01:01:11Z`. The event covers selected fields for Caitlyn, Kog'Maw, Poppy, Berserker's Greaves, Black Cleaver, Tiamat, and Sunfire Aegis.

The Wiki page has 118 candidate change entries under the pilot counting rule. The pilot parses 12 entries and lists 106 unparsed or unsupported entries by entity. Its authority status is `partial_delta_pilot`, and `model_ready` is false.

The event does not refresh other 26.16 fields. Those fields retain their 26.15 record hashes and prior authority. The model-ready resolver rejects this event as a complete 26.16 ledger.

## Model use

Prematch features can use champion-native atoms from a model-ready patch snapshot within its declared coverage scope. Item and rune inputs must come from build and rune distributions that were available before the match cutoff. The replay knowledge cutoff must also precede that match.

Live features can use observed items and state from the current game. The ledger supplies the patch-specific mechanics for those observed entities. The live state must name its source time. Missing observed state remains missing.

Build and rune choices can change the model weights that activate atoms. A patch delta changes only the affected atom fields. The remaining imported mechanics graph carries forward.

## Artifacts

- Base: `lol_kills/v2/mechanics/atom_ledger/snapshots/lcc-26.15-base.json.gz`
- Delta: `lol_kills/v2/mechanics/atom_ledger/deltas/26.16-wiki-pilot.json`
- Loader and builder: `lol_kills/v2/mechanics/atom_ledger/base.py`
- Replay: `lol_kills/v2/mechanics/atom_ledger/replay.py`
