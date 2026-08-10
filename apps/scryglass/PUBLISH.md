# Public rankings publish checklist

Canonical public site: https://scryglass.xyz

The public payload contains nine rating and profile files plus one accepted tier-list file.

1. Install the systemd refresh service, alert service, refresh timer, watchdog
   service, and watchdog timer from `ops/systemd`.
2. Copy `ops/systemd/public-refresh.env.example` to
   `/etc/scryglass/public-refresh.env` and set the Supabase project URL, worker
   secret key, and data-publish secret. Keep all secrets outside the
   repository. Alerts are optional.
3. Start one controlled run:

   ```bash
   SCRYGLASS_PUBLIC_RELEASE=1 python3 -m lol_kills.public_refresh \
     --root . \
     --public-root /srv/scryglass-data/public-packs \
     --once
   ```

4. The runner performs source discovery, ratings, patch-wide tier authority,
   atomic Supabase publication, cache invalidation, and checks for `/elo`,
   `/matches`, and `/tiers`.
5. Read `data/lol/runtime/public-refresh-health.json`. A failed map stays
   pending. A failed release keeps the previous pointer.
6. The watchdog records stale state after the configured twelve-hour freshness
   window. It does not publish data.

## Manual refresh

Oracle's Elixir annual files are the public match source. Incomplete maps stay
pending for the next cycle. GRID remains available only to private research
modules.

GitHub checks validate code changes. They are not part of the data refresh
path. No production workflow or deployment requires `GRID_API_KEY`.

The worker uses `SCRYGLASS_SUPABASE_SECRET_KEY` to stage and activate releases.
The website uses a separate publishable key with read-only policies.
`SCRYGLASS_DATA_PUBLISH_TOKEN` clears the website cache after activation.

## Build budget

- A six-hour data refresh must not run `vercel deploy`.
- A code release gets one production build after merge.
- Use the same successful deployment for production verification.
- Keep Scryglass below 60 Vercel build CPU minutes per day.
