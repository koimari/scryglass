# Tier list production readiness

The tier-list API has a production bundle. It reads the approved index from
public/v2/tierlists/production/index-v1.json or from the URL in
SCRYGLASS_TIERLIST_INDEX_URL.

The API returns 503 when the index is absent, stale, malformed, or fails a
hash check. A valid production bundle returns source-bound champion ranks and
movement. It does not open outcome-calibrated probability, causal, or betting
claims.

## Data window

The current bundle covers 2025-01-01T00:00:00Z through the latest completed
game in the OE bridge, 2026-08-08T12:13:56Z. The live review window starts at
2026-07-18T00:00:00Z.

The current source records:

- 38,510 complete maps across the available source years;
- 17,209 complete maps from 2025 onward;
- 888 maps after the July 18 review start;
- 836 accepted games from the current OE API bridge;
- 525 full-detail games after the annual OE file watermark.

The annual OE file ends at 2026-07-28T23:45:16Z. The API bridge supplies the
completed games after that watermark. The source receipt records the OE
watermark and the requested wall-clock end separately.

Oracle’s Elixir publishes downloadable files once per day. The official
source page is [Oracle’s Elixir downloads](https://master.d36liwrx5rvjnc.amplifyapp.com/tools/downloads).

The source mode is oe_only. It uses the annual OE file and the current OE API.
It skips GRID. The tier ladder uses the final map result, both sides, role,
champion, team, league, patch, and game date. The full-detail API rows also
supply player names for the team and player rating refresh.

## Elo ladder

Each champion and role starts at 1500. The replay runs in chronological order.
It uses the pre-map team Elo as the team-strength control. It updates the
champion and opposing-champion states after the map result.

The bundle contains 285 role and scope cells. The scope set contains 57 league
or international scopes, 44 league labels, six international event labels,
five competition bands, and all five roles.

Each champion row contains:

- champion identity and image URL;
- tier value and tier bucket;
- rating and rating change;
- current rank and prior rank;
- rank change and movement label;
- verified map count;
- counterability status.

The ladder is a rolling 2025-to-live view for each scope. The patch field marks
the latest observed patch in that scope. It does not create a separate
patch-specific replay. A positive rank change means a climb. A missing prior
rank produces new.

Counterability is unavailable in the current bundle. The API keeps this state
explicit. The tier value remains eligible for descriptive rank display.

The candidate replay remains at:

~~~text
data/lol/v2/tierlists/champion-elo-candidate-v1.json
~~~

The candidate stays development-only. Its digest binds the source, champion
identity map, replay settings, and row movement fields.

## Evidence and authority

The promotion records are:

~~~text
data/lol/v2/tierlists/prospective-evaluation-v1.json
data/lol/v2/tierlists/independent-l2-authority-v1.json
data/lol/v2/tierlists/production-manifest-v1.json
~~~

The prospective evaluation replays the candidate in time order. It freezes
the pre-cutoff state and observes 888 later map outcomes. The diagnostic
decision is descriptive_pass.

The forward diagnostic records these controlled scores:

- champion ladder log loss: 0.6511345662;
- team-only baseline log loss: 0.6324651837;
- champion ladder Brier score: 0.2282398404;
- team-only baseline Brier score: 0.2204296376.

These scores are diagnostic evidence. The champion ladder scores worse than
the team-only baseline in this holdout. The record keeps predictive authority
closed. It also keeps outcome-calibrated probability closed.

The independent authority record repeats the raw-source, schema, identity,
movement, and time-order checks. It approves descriptive rank publication.
The record has no external human signoff. Its authority is limited to the
source-bound descriptive tier surface.

The private terminal Draft Score model has a separate authority boundary. Its
development status and blocked future prediction ledger remain visible in the
readiness report. They do not block the descriptive champion tier API.

## Production bundle

The canonical bundle is:

~~~text
data/lol/v2/tierlists/production/index-v1.json
data/lol/v2/tierlists/production/cells/
~~~

The app mirror is:

~~~text
apps/scryglass/public/v2/tierlists/production/index-v1.json
apps/scryglass/public/v2/tierlists/production/cells/
~~~

The current index contains 285 production cells. Every scope has top, jungle,
mid, bot, and support cells. Every cell has an independent raw SHA-256 digest.
The canonical index raw SHA-256 is:

~~~text
86b9f439cfd426beb02ac4dbea6f1a3090939fb1e9248fd798943687cda05170
~~~

The promotion manifest binds the candidate, prospective evaluation,
independent authority, source tree, commit, index, and rollback fields.
Rollback starts with an empty pointer because this is the first approved
production bundle in the local checkout.

Run the bundle check from the repository root:

~~~text
python3 -m lol_kills.v2.tierlists.production_bundle --verify
~~~

Run the complete readiness audit:

~~~text
python3 -m lol_kills.v2.tierlists.production_readiness --root . --strict
~~~

The current audit returns ready_for_promotion_review, with no tier-list
blockers. It reports 285 production cells. The terminal Draft Score boundary
remains closed and does not block this result.

## Public API

The app server reads the production index and verifies the canonical digest,
cell bytes, status flags, and five-role coverage before it serves data.

The endpoint is:

~~~text
/api/v2/tierlist
~~~

It accepts region, league, international event, competition tier, role, patch,
and minimum played-map filters. A successful response includes the as-of
watermark, scope metadata, source mode, index digest, champion image URLs,
movement fields, and claim ceiling.

The tier page uses the same rows. It shows all five roles and all available
league, event, tier, and patch choices from the production index.

## Live refresh

apps/scryglass/vercel.json schedules:

~~~text
GET /api/cron/tierlist-refresh
~~~

The route requires Authorization: Bearer $CRON_SECRET. It sends the worker the
OE window and the worker-received time. It requires
SCRYGLASS_TIERLIST_INGEST_TOKEN when a worker URL is configured. It accepts
only an HTTPS worker URL and aborts the request after 15 seconds.

The durable worker runs this command with --promote:

~~~text
python3 -m lol_kills.v2.tierlists.live_refresh \
  --expected-live-as-of <worker-received-time> \
  --source-mode oe_only \
  --promote
~~~

The command refreshes OE, fetches full game detail, builds one deduplicated
source, replays the champion, team, and player ladders, writes weekly rank
snapshots, runs the forward evaluation, runs the independent authority check,
and writes the production bundle. It then publishes immutable production cells
and an immutable release index to Vercel Blob. It replaces the stable pointer
only after those writes pass the retention guard. The pointer passes an exact
readback check before the receipt reports production promotion. A failed step
leaves the prior published artifact in place.

The weekly baseline is Sunday 00:00 UTC. Team, player, and champion movement
use the same rank convention. One run can process several completed games.
The published as-of value is the last completed game in the ordered batch.

Vercel functions do not write the deployed repository. The worker needs
durable storage for source receipts, immutable cells, the index, and the
serving pointer. The cron route only starts the refresh request.

The poll schedule can run more often than OE publishes its annual file. The
OE API bridge adds a completed game after OE exposes it. The worker needs
ORACLES_ELIXIR_API_KEY as an environment value. The poll does not invent game
rows.

### External worker contract

The Vercel route does not run Python and cannot rewrite the deployed app. A
durable HTTPS worker endpoint is required. It must accept the route's POST
request with `Authorization: Bearer <SCRYGLASS_TIERLIST_INGEST_TOKEN>`.
It must run from a checkout that contains the Scryglass repository and keep
the following values in its private environment:

~~~text
ORACLES_ELIXIR_API_KEY=...
BLOB_READ_WRITE_TOKEN=...
SCRYGLASS_TIERLIST_BLOB_BASE_URL=https://<store>.public.blob.vercel-storage.com
~~~

`SCRYGLASS_TIERLIST_BLOB_BASE_URL` can use the existing Blob public root. The
worker also accepts `LIVE_BLOB_BASE_URL` as a fallback. The token and public
root must identify the same Blob store. `BLOB_STORE_ID` is optional because
the worker checks the store identity encoded in the token and URL.

The worker publishes the stable index at:

~~~text
https://<store>.public.blob.vercel-storage.com/tierlists/index-v1.json
~~~

The Vercel app must have this production variable:

~~~text
SCRYGLASS_TIERLIST_INDEX_URL=https://<store>.public.blob.vercel-storage.com/tierlists/index-v1.json
~~~

The Vercel Production environment also needs:

~~~text
CRON_SECRET=...
SCRYGLASS_TIERLIST_INGEST_URL=https://<worker-host>/tierlist-refresh
SCRYGLASS_TIERLIST_INGEST_TOKEN=...
~~~

The secret values must be generated outside the repository. The current
Vercel Production environment audit found `LIVE_BLOB_BASE_URL` and
`BLOB_READ_WRITE_TOKEN`. It did not find the cron secret, worker URL, worker
token, or tier-list index URL. No external worker or paid service was created
by this change.

The Blob publication uses the existing fail-closed retention guard. The
current hard stop is 850,000,000 retained bytes. Immutable release files use
`tierlists/releases/<source-index-sha256>/`. The stable pointer is written
last. A missing or malformed existing pointer blocks replacement.

The `*/5 * * * *` schedule needs a Vercel plan that supports five-minute cron
intervals. A daily-only plan cannot provide after-game polling at this rate.

## Release boundary

The exact local production build passed with Next 16.2.11. The production
server returned status: available, development_only: false, and
publication_eligible: true for LEC, LCS, EWC, tier 2, and all five LEC role
queries. Each response contained champion image URLs. The browser check found
the tier page, five-role sections, filters, and rank cards. It found no console
errors or framework error overlay. The cron route returned 401 without its
secret.

The public scryglass.xyz deployment has not changed in this task. A deploy is a
separate release action. It needs explicit authorization and a production
environment check for the cron secret, OE API key, worker URL, and artifact
storage.

The live Vercel logs show the five-minute cron invoking the route. Each observed
invocation returned 401 because `CRON_SECRET` is absent. No worker refresh has
therefore started in Production.
