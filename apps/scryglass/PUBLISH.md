# Public rankings publish checklist

Canonical public site: https://scryglass.xyz

The public payload contains nine rating and profile files plus one accepted tier-list file.

1. Install the systemd refresh service, alert service, refresh timer, watchdog
   service, and watchdog timer from `ops/systemd`.
2. Put the OE key, Blob write key, Blob root, data-publish secret, and alert webhook in `/etc/scryglass/public-refresh.env`. Keep all secrets outside the repository.
3. Start one controlled run:

   ```bash
   SCRYGLASS_PUBLIC_RELEASE=1 python3 -m lol_kills.public_refresh \
     --root . \
     --public-root /srv/scryglass-data/public-packs \
     --once
   ```

4. The runner performs source discovery, ratings, patch-wide tier authority,
   Blob publication, cache invalidation, and checks for `/elo`, `/matches`,
   and `/tiers`.
5. Read `data/lol/runtime/public-refresh-health.json`. A failed map stays
   pending. A failed release keeps the previous pointer. A failed service
   sends the health payload to `SCRYGLASS_ALERT_WEBHOOK_URL`.
6. The watchdog sends one stale-refresh alert after the configured twelve-hour
   freshness window. It does not publish data.

## Manual refresh

Oracle's Elixir annual files and the OE API bridge are the only public match
sources. Incomplete maps stay pending for the next cycle. GRID remains
available only to private research modules.

GitHub checks validate code changes. They are not part of the data refresh
path. No production workflow or deployment requires `GRID_API_KEY`.

The worker uses `BLOB_READ_WRITE_TOKEN` for immutable packs and tier releases.
It uses `SCRYGLASS_DATA_PUBLISH_TOKEN` only for cache invalidation. The
short-lived upload token limits each write to the public data paths.

## Build budget

- A six-hour data refresh must not run `vercel deploy`.
- A code release gets one production build after merge.
- Use the same successful deployment for production verification.
- Keep Scryglass below 60 Vercel build CPU minutes per day.
