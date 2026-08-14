# OE to official and atom patch mapping

Status: audited sidecar, version 1.

Artifact: `data/lol/v2/champions/oe-atom-patch-map-v1.json`

## Scope

The sidecar covers Oracle's Elixir source rows from 2025-01-01 through the
audited OE source snapshot. The snapshot watermark is
`2026-08-08T12:13:56Z`. It contains 17,209 unique maps and 39 observed OE
patch tokens. A refresh binds the current OE live watermark and intervals to
the same audited rows. The current live parquet and receipt are mutable.

OE can write a patch with one decimal place. For example, `15.1` means
`15.10`. The resolver keeps the raw token in `source_tokens` and uses the
two-decimal token as its key.

## Evidence

The audit uses these inputs:

- An OE-only player-game parquet snapshot. Its SHA-256 hash is retained as the
  audit evidence. A refresh verifies the current parquet schema, game count,
  patch tokens, source watermark, and raw hash in its refresh receipt.
- An OE live-source receipt. It binds the current source watermark and the
  OE-only source mode.
- Riot patch-note pages. Each mapping row records the page locator, page hash,
  JSON-LD publication time, and page label.
- The LCC atom bridge. Its semantic hash is audited. Its generated artifact
  hash, source commit, and generation time are recorded for each refresh.

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

The resolver verifies the sidecar hash and local source bindings before it uses
a row. Static sources require their exact SHA-256 hash. Mutable OE sources
require a valid OE-only receipt and a parquet interval audit. The mutable atom
bridge requires its exact semantic hash and a valid canonical artifact hash.
The resolver requires an `as_of` timestamp. The timestamp must be inside the
current source interval and after the official release time.

The official resolver returns `26.15` for `16.15` and `26.14` for `16.14`.
The sidecar's registered atom snapshot returns `26.15` only for the current
`16.15` row. The refreshed bridge source receipt now records public patch
`26.16` and client source `16.16`, but the sidecar does not register an OE
`16.16` atom interval yet. `resolve_atom_snapshot_patch("16.16", ...)`
therefore stays unavailable until a current OE interval and official patch
evidence are captured. Older rows have exact official evidence and an explicit
`atom_snapshot_unavailable` status.

Unknown tokens, malformed tokens, changed static hashes, missing evidence,
missing timestamps, and unavailable atom snapshots return `None` or an
unavailable resolution. An OE token that has no audited row stops the refresh.
There is no nearest-patch fallback.

## Source boundary

This sidecar provides patch identity and patch-state provenance. It does not
estimate champion effects. It does not grant prediction, publication, or
betting authority.
