# Six-hour ratings sync

One small Linux host can update the public ratings data. GitHub and website
builds are not part of the data path. The current deployment reads accepted
files from public object storage.

```text
Oracle's Elixir API
        ↓
six-hour discovery cache
        ↓
new canonical game IDs
        ↓
complete map, team, and player validation
        ↓
team ratings and player ratings
        ↓
seven JSON files in an immutable object pack
        ↓
atomic object-store manifest replacement
```

The site continues to read the previous pack during a refresh. A malformed game or failed checksum stops publication.

The worker binds the current pack to its exact canonical game-ID set before it
requests new OE details. Every ID in that set must remain in the next accepted
source. This prevents an incomplete annual file or API response from removing
completed maps, ratings history, or profile history.

## Host paths

- Code: `/srv/scryglass/current`
- Published packs: `/srv/scryglass-data/public-packs`
- Worker state and health: `/srv/scryglass/current/data/lol/runtime`
- Source and warehouse cache: `/srv/scryglass/current/data/lol/warehouse`

Set `SCRYGLASS_PACK_MANIFEST_URL` when the public pointer uses another object
store. The deployed server reads the stable manifest and the selected
immutable pack from that store. It caches the pointer for six hours. Local
files remain an outage and development fallback.

Keep `ORACLES_ELIXIR_API_KEY` in `/etc/scryglass/postgame-sync.env`. Give the service account read access only.
The service does not use `GRID_API_KEY` or local GRID rows.

## Publication checks

Each new game must have one map row, two named teams with complementary results, and ten unique named players. Both sides need one player in each canonical role. The pack must contain the exact seven public rating JSON files. Every file size and SHA-256 digest must match its manifest entry.

The manifest also stores a count and SHA-256 digest for the sorted canonical game IDs. This proves that the rating export used the same complete source set that passed validation.

Tier lists use a separate authority gate. A ratings refresh never promotes a tier list.

## Service setup

Install the service and timer from `ops/systemd`. The timer runs at minute 0 every six hours and catches a missed run after reboot.

Run a manual check with:

```bash
/srv/scryglass/venv/bin/python -m lol_kills.postgame_sync \
  --root /srv/scryglass/current \
  --public-root /srv/scryglass-data/public-packs \
  --once
```

Read `data/lol/runtime/postgame-sync-health.json` for the last local result. It stays outside the public website.

The six-hour timer must not call a deployment command. One merged code release
can run one production build. The daily Scryglass build CPU budget is 60
minutes.
