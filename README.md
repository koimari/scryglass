# Scryglass

Independent League of Legends ratings publication by Koi. The public site
contains team ratings, player ratings, completed matches, patch-wide champion
tier lists, and their methods.

## Public site

```bash
cd apps/scryglass
npm install
npm run dev
```

Open http://localhost:3000.

## Surfaces

- `/elo` — team and player ratings
- `/matches` — completed maps with lineups, KDA, and player grades
- `/tiers` — patch-wide champion tier lists by role
- `/methodology` — rating and tier-list methods

## Pack

Oracle's Elixir annual files provide the historical baseline. The Oracle's
Elixir API bridge discovers completed maps between annual-file updates. A
six-hour refresh caches discovery results and requests details only for maps
that are new or incomplete.

Accepted packs go to immutable object storage. The site reads the current pack
pointer there and caches it for six hours. A data refresh does not run a site
build or create a deployment.

Run a complete OE-only ratings refresh with:

```bash
python3 -m lol_kills.postgame_sync \
  --root . \
  --public-root apps/scryglass/public/packs \
  --once \
  --force
```

Set `ORACLES_ELIXIR_API_KEY` in the worker environment. The refresh accepts a
map after it has canonical identities, two teams, ten players, five roles per
side, and complete public statistics. The current pack remains active when a
map is incomplete or when a completed map disappears from the source set.

Publish the accepted pack and tier display with `npm run publish:data` from
`apps/scryglass`. Set `SCRYGLASS_DATA_PUBLISH_TOKEN` in the local worker and
the production project. The publisher verifies Blob readback before it moves
the stable pack pointer.

## Private GRID research

GRID ingestion and event modules remain in the repository for optional private
historical research. Public refresh commands do not read `GRID_API_KEY` and do
not load local GRID rows. The sibling `lol-strength-analysis/.env` belongs to
that project and must stay unchanged.

For an already-current OE warehouse:

```bash
python3 -m lol_kills.export.public_pack --years 2025,2026
python3 -m lol_kills.export.upload_pack
```

## Build budget

Data refreshes use zero Vercel build minutes. Run a Vercel build only for a
code release. Validate locally before the push, create one production build
after merge, and keep the daily Scryglass build budget below one hour.
