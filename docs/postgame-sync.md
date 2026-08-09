# Six-hour ratings sync

One small Linux host can update and serve the public ratings data. GitHub and Vercel are not part of the data path.

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
seven JSON files in an immutable pack
        ↓
atomic manifest.json replacement
```

The site continues to read the previous pack during a refresh. A malformed game or failed checksum stops publication.

## Host paths

- Code: `/srv/scryglass/current`
- Published packs: `/srv/scryglass-data/public-packs`
- Worker state and health: `/srv/scryglass/current/data/lol/runtime`
- Source and warehouse cache: `/srv/scryglass/current/data/lol/warehouse`

Set `SCRYGLASS_PACK_ROOT=/srv/scryglass-data/public-packs` for the Next.js server. The server reads the root manifest and the selected immutable pack from that directory. It caches reads for six hours.

Keep `ORACLES_ELIXIR_API_KEY` in `/etc/scryglass/postgame-sync.env`. Give the service account read access only.

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
