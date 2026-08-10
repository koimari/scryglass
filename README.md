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

Oracle's Elixir publishes public annual CSV files. Every six-hour refresh checks
the small remote file signature first. It downloads a file only when its size
or modification time changes. A validated local copy remains active when the
source is unchanged or temporarily unavailable.

Accepted page data goes to Supabase. Postgres holds ratings, profiles, match
records, and the active release. Supabase Storage holds the larger tier-list
matrix. The worker makes the complete release visible in one database
transaction. The site caches it for six hours. A data refresh does not run a
site build or create a deployment.

Run the complete local control loop with:

```bash
SCRYGLASS_PUBLIC_RELEASE=1 python3 -m lol_kills.public_refresh \
  --root . \
  --public-root apps/scryglass/public/packs \
  --once \
  --force
```

Copy `ops/systemd/public-refresh.env.example` to
`/etc/scryglass/public-refresh.env`. Set the Supabase project URL, the dedicated
worker secret key, and `SCRYGLASS_DATA_PUBLISH_TOKEN`. The alert URL is
optional. The runner performs OE CSV refresh, ratings, tier authority,
publication, cache invalidation, and live smoke checks in one locked cycle.

The website needs `SCRYGLASS_SUPABASE_URL` and
`SCRYGLASS_SUPABASE_PUBLISHABLE_KEY` in Vercel. The publishable key can read
only the active public release. It cannot upload or activate a release.

The refresh accepts a map after it has canonical identities, two teams, ten
players, five roles per side, and complete public statistics. The current pack
and tier pointer remain active when a map is incomplete or a stage fails. A
successful ratings publication rolls back when its public smoke check fails.

Install `ops/systemd/scryglass-ratings-sync.service` and
`ops/systemd/scryglass-ratings-sync.timer` on the worker host. The alert
service is optional. The watchdog service and timer record stale state every
hour. They send a webhook only when one is configured. The refresh timer runs
every six hours. The worker does not run a site build or a deployment.

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
