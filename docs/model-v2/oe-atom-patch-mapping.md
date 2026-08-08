# OE to official and atom patch mapping

Status: audited sidecar, version 1.

Artifact: `data/lol/v2/champions/oe-atom-patch-map-v1.json`

## Scope

The sidecar covers Oracle's Elixir source rows from 2025-01-01 through the
current OE source watermark. The current watermark is
`2026-08-08T12:13:56Z`. It contains 17,209 unique maps and 39 observed OE
patch tokens.

OE can write a patch with one decimal place. For example, `15.1` means
`15.10`. The resolver keeps the raw token in `source_tokens` and uses the
two-decimal token as its key.

## Evidence

The audit uses these inputs:

- The current OE-only player-game parquet. The resolver binds its SHA-256 hash.
  It deduplicates rows by `gameid` and groups the exact `patch` value.
- The OE live-source receipt. It binds the source watermark and the OE-only
  source mode.
- Riot patch-note pages. Each mapping row records the page locator, page hash,
  JSON-LD publication time, and page label.
- The LCC atom bridge. Its raw hash, bridge artifact hash, canonical data patch,
  and LCC commit are recorded in the snapshot registry.

The official mapping matches the season and minor number. The audit checks that
the first OE observation occurs after the matching Riot patch-note publication.
This test catches a wrong season prefix and a shifted minor number. It uses the
source patch token. It does not replace an older competition patch with the
current live client patch.

Professional matches can remain on an older patch after a live release. The
OE interval can cross a later Riot release. This is expected for a competition
source and is recorded in the sidecar.

## Resolver rules

`lol_kills.v2.tierlists.patch_mapping` exposes three functions:

```python
resolve_oe_patch("16.15", "2026-08-01T00:00:00Z")
resolve_official_patch("15.1", "2025-05-20T00:00:00Z")
resolve_atom_snapshot_patch("16.15", "2026-08-01T00:00:00Z")
```

The resolver verifies the sidecar hash and local source hashes before it uses a
row. It requires an `as_of` timestamp. The timestamp must be inside the
hash-bound OE interval and after the official release time.

The official resolver returns `26.15` for `16.15` and `26.14` for `16.14`.
The atom resolver returns `26.15` only for the current `16.15` row. The bridge
contains one atom snapshot. Older rows have exact official evidence and an
explicit `atom_snapshot_unavailable` status.

Unknown tokens, malformed tokens, changed hashes, missing evidence, missing
timestamps, and unavailable atom snapshots return `None` or an unavailable
resolution. There is no nearest-patch fallback.

## Source boundary

This sidecar provides patch identity and patch-state provenance. It does not
estimate champion effects. It does not grant prediction, publication, or
betting authority.
