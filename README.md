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

Run the complete local control loop with:

```bash
SCRYGLASS_PUBLIC_RELEASE=1 python3 -m lol_kills.public_refresh \
  --root . \
  --public-root apps/scryglass/public/packs \
  --once \
  --force
```

Copy `ops/systemd/postgame-sync.env.example` to
`/etc/scryglass/postgame-sync.env` and set `ORACLES_ELIXIR_API_KEY`. Copy
`ops/systemd/public-refresh.env.example` to
`/etc/scryglass/public-refresh.env` and set `BLOB_READ_WRITE_TOKEN`,
`LIVE_BLOB_BASE_URL`, `SCRYGLASS_DATA_PUBLISH_TOKEN`, and
`SCRYGLASS_ALERT_WEBHOOK_URL` in the worker environment. The runner performs
OE discovery, ratings, tier authority,
publication, cache invalidation, and live smoke checks in one locked cycle.

The refresh accepts a map after it has canonical identities, two teams, ten
players, five roles per side, and complete public statistics. The current pack
and tier pointer remain active when a map is incomplete or a stage fails. A
successful ratings publication rolls back when its public smoke check fails.

Install `ops/systemd/scryglass-ratings-sync.service` and
`ops/systemd/scryglass-ratings-sync.timer` on the worker host. Install the
matching `scryglass-public-refresh-alert@.service`,
`scryglass-public-refresh-watchdog.service`, and watchdog timer. The refresh
timer runs every six hours. The watchdog checks the health file every hour. The
worker does not run a site build or a deployment.

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
