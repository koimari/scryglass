# Scryglass live feed

The live surface is deliberately split into a server-side worker and a
read-only public page:

```text
GRID Series Events → live_worker → verified Blob snapshots → /live
```

The browser never receives `GRID_API_KEY` and never opens a GRID connection.
The worker does not update official Dual Elo while a game is in progress.

## Local proof with a captured Series State

The worker can publish one captured state without opening GRID. This is useful
for schema and page checks:

```bash
python3 -m lol_kills.live_worker \
  --series-id 2970137 \
  --state-file /path/to/series-state.json \
  --sequence 12 \
  --local-root apps/scryglass/public/live
```

The resulting files are:

```text
apps/scryglass/public/live/index.json
apps/scryglass/public/live/health.json
apps/scryglass/public/live/series/{series_id}/latest.json
apps/scryglass/public/live/series/{series_id}/snapshots/{sequence}.json
```

These are generated runtime artifacts. Do not commit a real live snapshot to
the public repository.

## GRID worker

Discover active professional series and keep the worker running in a separate
long-lived process or container:

```bash
python3 -m lol_kills.live_worker \
  --discovery-seconds 30 \
  --series-seconds 3600
```

The worker resolves `GRID_API_KEY` using the same environment / `.env` lookup as
the existing GRID ingestion bridge. It writes to local `apps/scryglass/public/live`
when no Blob token is present. In production provide:

```text
GRID_API_KEY=...
BLOB_READ_WRITE_TOKEN=...
```

The public Next.js app needs only the Blob read prefix:

```text
LIVE_BLOB_BASE_URL=https://<blob-store-root>
```

The app reads `live/index.json` and the referenced immutable snapshots with
`no-store` fetches. Pointer objects are short-lived; numbered snapshots are
immutable.

## Model boundary

The current live coefficient artifact is calibrated for approximately the
8:00–20:00 window. The state board remains useful outside that interval, but
the probability is shown as withheld until a later-horizon calibration artifact
is trained and validated. This prevents a late-game number from appearing more
certain than the evidence supports.

The “Compare another model” panel accepts either a JSON object containing
`p_blue` / `p_red` or a simple pair such as `57/43`. It is an audit surface:
input mismatches and calibration differences remain visible rather than being
collapsed into a claim that one model is correct.
