# Public rankings publish checklist

Canonical public site: https://scryglass.xyz

The public payload contains nine rating and profile files plus one accepted tier-list file.

1. Run `python3 -m lol_kills.postgame_sync --root . --public-root apps/scryglass/public/packs --once --force` with `ORACLES_ELIXIR_API_KEY` available and `GRID_API_KEY` absent.
2. Confirm that the current production source IDs are a subset of the candidate OE source IDs. The refresh stops when a completed map disappears.
3. Run the OE-only patch-wide tier refresh. Use `python3 -m lol_kills.v2.tierlists.live_refresh --root . --expected-live-as-of <UTC timestamp> --source-mode oe_only --skip-annual-oe --skip-atom-bridge`.
4. Run the descriptive evaluation and authority checks. Then run `python3 -m lol_kills.v2.tierlists.production_bundle --root .` to update the cached patch-wide display file.
5. Set `SCRYGLASS_DATA_PUBLISH_TOKEN` to the same secret used by the production maintenance endpoint. From `apps/scryglass`, run `npm run publish:data -- --pack-dir <pack directory> --tierlists <tierlists.json>`. The command uploads immutable files, verifies each checksum by readback, and moves the stable pointers last.
6. Run tests, lint, the production build, pack checksum checks, and the public-boundary audit. Deploy the merged `main` commit once. Verify `/elo`, `/matches`, player and team profiles, `/tiers`, and `/methodology` on `https://scryglass.xyz`.

## Manual refresh

Run the local refresh every six hours. Oracle's Elixir annual files and the OE
API bridge are the only public match sources. Incomplete maps stay pending for
the next cycle. GRID remains available only to private research modules.

GitHub checks validate code changes. They are not part of the data refresh
path. No production workflow or deployment requires `GRID_API_KEY`.

The production project stores `BLOB_READ_WRITE_TOKEN` and
`SCRYGLASS_DATA_PUBLISH_TOKEN`. The local worker only needs the second value.
The short-lived Blob token limits each write to public pack or tier-list
paths. Keep both values outside the repository.

## Build budget

- A six-hour data refresh must not run `vercel deploy`.
- A code release gets one production build after merge.
- Use the same successful deployment for production verification.
- Keep Scryglass below 60 Vercel build CPU minutes per day.
