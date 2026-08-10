# Six-hour completed-match and tier refresh

Public Scryglass updates from Oracle's Elixir only.

```text
OE annual CSV files
              ↓
cached six-hour remote-file check
              ↓
new canonical game IDs
              ↓
identity, role, and statistic checks
              ↓
ratings, player grades, profiles, match pages, and tier authority
              ↓
immutable object upload, cache invalidation, and smoke checks
```

The worker keeps validated annual files and normalized parquet in its local
warehouse. A later cycle checks the public file signature again. A changed file
passes the same structural and continuity gates before publication.

## Acceptance gate

Each accepted map needs one canonical game ID, two named teams with opposite
results, ten unique named players, five canonical roles on each side, and the
complete postgame statistics used by player grades. A failed map stays outside
the public source set.

Before publication, the worker compares the candidate canonical IDs with the
exact source set behind the current pack. Every completed published map must
remain present. File sizes and SHA-256 digests must also match the new manifest.
The current pack remains active when a check fails.

The website reads the object-store pointer at runtime and caches it for six
hours. Refreshes do not rebuild or redeploy the website. Only a code change can
start a Vercel build.

## Schedule

The systemd timer in `ops/systemd` runs the complete sequence at minute 0 every
six hours. Run the same path manually with:

```bash
/srv/scryglass/venv/bin/python -m lol_kills.public_refresh \
  --root /srv/scryglass/current \
  --public-root /srv/scryglass-data/public-packs \
  --once
```

The runner verifies immutable pack files before it replaces the current
manifest. It then clears the cached ratings, matches, and tier pages. A smoke
failure restores the previous ratings pointer.

## Private GRID modules

GRID event and checkpoint ingestion remains available for optional private
historical research. It has no public route, schedule, pack dependency, build
step, or deployment requirement.
