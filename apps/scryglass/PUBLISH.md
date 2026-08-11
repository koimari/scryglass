# Public rankings publish checklist

Canonical public site: https://scryglass.xyz

The public payload contains rating, profile, match, schedule, and tier-list files.

1. Install the refresh and database-backup agents from `ops/launchd`.
2. Store the Supabase worker key and data-publish token in the login Keychain.
   Log in with the pinned Supabase CLI for short-lived backup access. The
   launch scripts contain no secrets.
3. Start one controlled run with the Mac launchd script. It acquires the 2026
   OE file once and creates the required source and import receipts.

   ```bash
   /Users/river/Library/Application\ Support/Scryglass\ Worker/run-public-refresh.sh
   ```

4. The runner performs source discovery, ratings, patch-wide tier authority,
   atomic Supabase publication, cache invalidation, and checks for `/elo`,
   `/matches`, and `/tiers`.
5. Read the runtime `public-refresh-health.json`. A failed game stays
   pending. A failed release keeps the previous pointer.
6. The health endpoint reports stale state after the configured twelve-hour
   freshness window.

## Manual refresh

Oracle's Elixir annual files are the public match source. Incomplete games stay
pending for the next cycle. GRID remains available only to private research
modules.

GitHub checks validate code changes. They are not part of the data refresh
path. No production workflow or deployment requires `GRID_API_KEY`.

The worker uses `SCRYGLASS_SUPABASE_SECRET_KEY` to stage and activate releases.
The website uses a separate publishable key with read-only policies.
`SCRYGLASS_DATA_PUBLISH_TOKEN` clears the website cache after activation.
The cache endpoint requires the activated release ID and confirms that Vercel
serves that same release.

## Build budget

- A six-hour data refresh must not run `vercel deploy`.
- A code release gets one production build after merge.
- Use the same successful deployment for production verification.
- Keep Scryglass below 60 Vercel build CPU minutes per day.
