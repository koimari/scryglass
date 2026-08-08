# Scryglass

Independent League of Legends research publication by Koi. 
Authored essays lead; hierarchical team ratings with a sequential Dual Elo benchmark,
match explorer, and head-to-head boards support the claims with reproducible Oracle’s Elixir packs
(2025–2026).

## Public site

```bash
cd apps/scryglass
npm install
npm run dev
```

Open http://localhost:3000.

## Surfaces

- `/` — latest article
- `/articles` — research notes index
- `/articles/void-grubs-contest-or-leave` — void-grubs contest-or-leave essay (+ charts)
- `/elo` — Dual Elo ratings
- `/browse` — match explorer
- `/browse/head-to-head` — head-to-head
- `/live` — verified live game state and conditional model estimate
- `/methodology` — estimands
- `/reproduce` — pack download / reproduction

`/grubs` permanently redirects to the void-grubs article.

## Pack

For a current pack, refresh the reconciled OE baseline, pull completed recent
pro games from GRID, rebuild ratings, and publish:

```bash
python3 -m lol_kills.update_public_pack \
  --years 2025,2026 \
  --download-oe \
  --download-grid \
  --grid-required \
  --publish
```

The local refresh can read `GRID_API_KEY` from the environment, this repo's
`.env`, or the sibling `lol-strength-analysis/.env`. Never commit that key.
The scheduled GitHub Actions workflow uses repository secrets named
`GRID_API_KEY` and `BLOB_READ_WRITE_TOKEN`; add the same GRID credential there
before enabling the schedule.

## Live-feed experiment

The low-latency proof of concept is a server-side worker, not a browser
connection. Discover currently started professional series with:

```bash
python3 -m lol_kills.etl.grid_series_events --discover-live
```

Then connect to a discovered series and request full state:

```bash
python3 -m lol_kills.etl.grid_series_events \
  --series-id SERIES_ID --seconds 60 --full-state \
  --out data/lol/warehouse/raw_grid/live_events.jsonl
```

The WebSocket preserves GRID transactions and can feed `lol_kills.live_model`
for a preliminary live-game estimate. It does not mutate official Dual Elo:
the public rating changes only after a completed game passes the normal
provenance and OE/GRID reconciliation path. The current live coefficient
artifact is intentionally limited to approximately the 8:00–20:00 window;
outside it, state may be shown but the probability is withheld until the
model is calibrated on that horizon.

The persistent snapshot worker and `/live` deployment contract are documented
in [`docs/live.md`](docs/live.md). The worker writes immutable snapshots plus
short-lived latest/index pointers to Blob; the public page reads only those
snapshots.

For an already-built local pack:

```bash
python3 -m lol_kills.export.public_pack --years 2025,2026
python3 -m lol_kills.export.upload_pack --local-only
```
