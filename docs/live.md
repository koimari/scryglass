# Six-hour completed-match refresh

Public Scryglass updates from Oracle's Elixir only.

```text
OE annual files + OE API bridge
              ↓
cached six-hour discovery
              ↓
new canonical game IDs
              ↓
identity, role, and statistic checks
              ↓
ratings, player grades, profiles, match pages, and tier lists
              ↓
immutable object upload and atomic pointer update
```

The API bridge discovers completed maps that are waiting for the next annual
file. It keeps discovery receipts and completed game details in the local
warehouse. A later cycle requests details again when a map is incomplete.

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

The systemd timer in `ops/systemd` runs at minute 0 every six hours. Run the
same path manually with:

```bash
python3 -m lol_kills.postgame_sync \
  --root . \
  --public-root apps/scryglass/public/packs \
  --once
```

The worker needs `ORACLES_ELIXIR_API_KEY`. It does not read `GRID_API_KEY`.

After the pack and tier display pass their checks, publish them without a site
build:

```bash
cd apps/scryglass
SCRYGLASS_DATA_PUBLISH_TOKEN=<secret> npm run publish:data -- \
  --pack-dir ../../output/public_pack/<pack-id> \
  --tierlists public/rankings/tierlists.json
```

Use the generated public display path when it differs from the example. The
maintenance endpoint issues a short-lived Blob token for approved JSON paths.
The publisher verifies immutable pack files before it replaces the current
manifest. It then clears the cached ratings and match pages.

## Private GRID modules

GRID event and checkpoint ingestion remains available for optional private
historical research. It has no public route, schedule, pack dependency, build
step, or deployment requirement.
