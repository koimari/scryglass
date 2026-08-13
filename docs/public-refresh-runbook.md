# Public refresh runbook

The Mac owns the six-hour data refresh. The checked-out Git commit stays clean.
Runtime data lives under `~/Library/Application Support/Scryglass Worker/runtime`.

## One cycle

1. The launch agent obtains an exclusive lock.
2. Brave Origin downloads the 2026 Oracle's Elixir file once.
3. The worker validates the file and writes a hash-bound source receipt.
4. The ingest step verifies that receipt. It adds new game versions and keeps corrected versions immutable.
5. The local Parquet cache must match the accepted Supabase game identity digest.
6. Ratings, player grades, profiles, matches, and patch-wide tier lists use that cache.
7. Supabase receives one staging release. The worker reads every database row and Storage object back before activation.
8. The worker activates the release, clears the website cache, and checks production.

A launchd retry in the same six-hour cycle reuses the accepted source and import receipts. The worker commit and both receipt bindings must match. A later scheduled cycle acquires a new 2026 file.

## Database migration gate

Run these commands from the merged Git commit before that commit becomes the worker version:

```sh
npm ci
npx supabase login
npx supabase link --project-ref uytblwbtkwuukbbrugdi
npx supabase migration list --linked
```

If the live schema contains the first seven repository migrations and the remote history is empty, repair only these known timestamps:

```sh
for version in \
  20260810153000 \
  20260810161500 \
  20260811121932 \
  20260811140238 \
  20260811142611 \
  20260811144542 \
  20260811165241
do
  npx supabase migration repair --linked --status applied "${version}"
done
npx supabase db push --linked --dry-run
npx supabase db push --linked
npx supabase migration list --linked
```

The current worker needs migrations `20260811174735` and `20260811193000`. Keep the old worker commit active until both appear in the remote list.
After those historical migrations, use the phase-specific commands below. A
general database push from a checkout that contains more than one cutover phase
is prohibited.

## Worker update

1. Stop the launch agent.
2. Copy the existing 2025 raw file and Parquet baseline from the old worker checkout into the runtime directory once.
3. Update the detached worker repository to the tested merge commit.
4. Copy `ops/launchd/run-public-refresh.sh` into the worker root.
5. Start the launch agent.
6. Run one changed-source cycle.
7. Run the same cycle again to prove the no-change path.

The old active Supabase release and previous Vercel deployment remain rollback targets during both checks.

## Backup

The database backup launch agent runs at 03:20 local time. The linked Supabase
CLI creates a short-lived database login for each run. Homebrew `libpq` supplies
`pg_dump` and `pg_restore`. The job keeps seven daily dumps and four Sunday
dumps. Each dump must pass `pg_restore --list` before it replaces its temporary
file.

Restore a backup into a temporary database before calling the backup lane ready. Compare table counts and canonical game identity digests with production.

## Bounded query and private Storage cutover

Use three merge commits. Each commit contains only the migrations named for its
phase. Never run one database push from a checkout that contains a later phase.

The checkpoints are:

1. Apply additive migrations `20260813010000` and `20260814010000`.
2. Deploy the compatible web build. It can read the active inline release and
   the bounded query API.
3. Apply private Storage migration `20260814020000`.
4. Install the tested publisher and create two Storage-only, query-complete
   releases.
5. Deploy the strict web build, then apply strict migration `20260814030000`.

Keep the prior verified data release and Vercel deployment as rollback targets
through all five checkpoints.

Set the release variables from the tested merge and current production state:

```sh
export SCRYGLASS_RELEASE_COMMIT="$(git rev-parse --verify HEAD)"
export SCRYGLASS_DATA_PUBLISH_TOKEN="$(security find-generic-password -a scryglass-public-worker -s scryglass-data-publish-token -w)"
export SCRYGLASS_DIAGNOSTIC_TOKEN="$(security find-generic-password -a scryglass-public-worker -s scryglass-diagnostic-token -w)"
export SCRYGLASS_LEGACY_RELEASE="$(curl -fsS https://scryglass.xyz/api/health \
  -H "Authorization: Bearer ${SCRYGLASS_DIAGNOSTIC_TOKEN}" | \
  jq -r '(.diagnostics.release_id // .pack_id)')"
test -n "${SCRYGLASS_LEGACY_RELEASE}"
test -z "$(git status --porcelain=v1 --untracked-files=normal)"
```

### Phase 1: additive schema and compatible web

Check out the merged Phase 1 commit. Its pending migration list must contain
only `20260813010000` and `20260814010000`:

```sh
npx supabase migration list --linked
npx supabase db push --linked --dry-run
```

Stop when the dry run names any later migration. When the list is exact, apply
and test Phase 1:

```sh
npx supabase db push --linked
npx supabase migration list --linked
npx supabase test db --linked
```

Deploy the compatible web commit through its pull request. Wait for every
required check, merge it manually, and verify the production deployment. The
active legacy release must still load through the parsed inline compatibility
RPC. Check `/elo`, `/matches`, `/tiers`, `/chat`, one player profile, and one
team profile.

### Phase 2: private Storage and clean releases

Check out the merged Phase 2 migration commit. Its dry run must contain only
`20260814020000`:

```sh
npx supabase migration list --linked
npx supabase db push --linked --dry-run
npx supabase db push --linked
npx supabase migration list --linked
npx supabase test db --linked
```

The compatible web build must pass the route checks again after the bucket
becomes private. Then install the publisher on the detached worker. Replace the
launch-agent commit marker and keep the strict web build out of production:

```sh
launchctl bootout "gui/$(id -u)/xyz.scryglass.public-refresh" 2>/dev/null || true
git -C "${HOME}/Library/Application Support/Scryglass Worker/repo" fetch origin
git -C "${HOME}/Library/Application Support/Scryglass Worker/repo" checkout --detach "${SCRYGLASS_RELEASE_COMMIT}"
test -z "$(git -C "${HOME}/Library/Application Support/Scryglass Worker/repo" status --porcelain=v1 --untracked-files=normal)"
sed "s/__TESTED_WORKER_COMMIT__/${SCRYGLASS_RELEASE_COMMIT}/" \
  ops/launchd/xyz.scryglass.public-refresh.plist.template \
  > "${HOME}/Library/LaunchAgents/xyz.scryglass.public-refresh.plist"
install -m 700 ops/launchd/run-public-refresh.sh \
  "${HOME}/Library/Application Support/Scryglass Worker/run-public-refresh.sh"
```

Run two forced all-Storage publications. The first release becomes a compatible
rollback target. The second becomes the web cutover release. The script gets
the service key and cache token from Keychain:

```sh
SCRYGLASS_WORKER_COMMIT="${SCRYGLASS_RELEASE_COMMIT}" \
  "${HOME}/Library/Application Support/Scryglass Worker/run-public-refresh.sh" --force
export SCRYGLASS_PREVIOUS_RELEASE="$(curl -fsS https://scryglass.xyz/api/health \
  -H "Authorization: Bearer ${SCRYGLASS_DIAGNOSTIC_TOKEN}" | \
  jq -r '(.diagnostics.release_id // .pack_id)')"
test "${SCRYGLASS_PREVIOUS_RELEASE}" != "${SCRYGLASS_LEGACY_RELEASE}"
SCRYGLASS_WORKER_COMMIT="${SCRYGLASS_RELEASE_COMMIT}" \
  "${HOME}/Library/Application Support/Scryglass Worker/run-public-refresh.sh" --force
export SCRYGLASS_NEW_RELEASE="$(curl -fsS https://scryglass.xyz/api/health \
  -H "Authorization: Bearer ${SCRYGLASS_DIAGNOSTIC_TOKEN}" | \
  jq -r '(.diagnostics.release_id // .pack_id)')"
test "${SCRYGLASS_NEW_RELEASE}" != "${SCRYGLASS_PREVIOUS_RELEASE}"
```

Prove that the active manifest and every database asset agree. This query uses
the linked project database:

```sh
export SCRYGLASS_DATABASE_URL='<short-lived Supabase direct connection URI>'
psql "${SCRYGLASS_DATABASE_URL}" -v SCRYGLASS_NEW_RELEASE="${SCRYGLASS_NEW_RELEASE}" <<'SQL'
select release_id,
       status,
       manifest ->> 'pack_id' as manifest_release_id,
       manifest #>> '{release,release_id}' as bound_release_id,
       manifest #>> '{draft_authority,status}' as draft_authority,
       jsonb_array_length(manifest -> 'files') as manifest_files
from public.scryglass_public_releases
where status = 'active';

select count(*) filter (where body is not null) as inline_assets,
       count(*) filter (
         where storage_path is distinct from release_id || '/' || path
            or content_type is distinct from 'application/json'
            or sha256 !~ '^[0-9a-f]{64}$'
       ) as invalid_assets,
       count(*) filter (where path = 'features/draft_records.json') as draft_assets,
       count(*) as storage_assets
from public.scryglass_public_assets
where release_id = :'SCRYGLASS_NEW_RELEASE';
SQL
```

Require `inline_assets = 0`, `invalid_assets = 0`, `draft_assets = 0`, and
`draft_authority = unavailable`. One active release ID must appear in all three
identity columns. Check the application families before web deploy:

```sh
for path in \
  /elo \
  /matches \
  /tiers \
  /chat \
  /api/public-data/tierlists \
  '/api/chat/leaderboards?category=rating&limit=1' \
  '/api/chat/leaderboards?category=teams_draft&limit=1' \
  '/api/chat/tier?limit=1' \
  '/api/chat/methodology?topic=ratings'
do
  curl -fsS "https://scryglass.xyz${path}" >/dev/null
done
curl -fsS https://scryglass.xyz/api/health \
  -H "Authorization: Bearer ${SCRYGLASS_DIAGNOSTIC_TOKEN}" | jq -e \
  --arg release "${SCRYGLASS_NEW_RELEASE}" \
  '.status == "ok" and (.diagnostics.release_id // .pack_id) == $release and (.diagnostics.refresh_status // .refresh_status) == "idle" and .stale == false'
```

### Phase 3: strict query cutover

Deploy the strict web commit after the Phase 2 probes pass. Use the normal pull
request and manual merge flow. The strict build removes the inline RPC and all
large-asset page and chat fallbacks. Confirm the production deployment and
repeat the family probes.

Check out the merged Phase 3 migration commit. Its dry run must contain only
`20260814030000`. Apply it and run the final database tests:

```sh
npx supabase migration list --linked
npx supabase db push --linked --dry-run
npx supabase db push --linked
npx supabase migration list --linked
npx supabase test db --linked
```

Repeat the route probes. An inactive object must return `404` through the site
asset route:

```sh
curl -fsS https://scryglass.xyz/api/health \
  -H "Authorization: Bearer ${SCRYGLASS_DIAGNOSTIC_TOKEN}" | jq -e \
  --arg release "${SCRYGLASS_NEW_RELEASE}" \
  '(.diagnostics.release_id // .pack_id) == $release'
test "$(curl -sS -o /dev/null -w '%{http_code}' \
  "https://scryglass.xyz/api/assets/${SCRYGLASS_PREVIOUS_RELEASE}/features/ratings_snapshot.json")" = 404
```

Rollback the data release when any pre-deploy or post-deploy probe fails. Load
the worker credentials from Keychain, restore the prior release, invalidate the
site cache, and save the returned receipt:

```sh
export SCRYGLASS_SUPABASE_SECRET_KEY="$(security find-generic-password -a scryglass-public-worker -s scryglass-supabase-secret -w)"
python -c 'import json, os; from lol_kills.export.supabase_publication import restore_release; print(json.dumps(restore_release(os.environ["SCRYGLASS_PREVIOUS_RELEASE"], project_url="https://uytblwbtkwuukbbrugdi.supabase.co", secret_key=os.environ["SCRYGLASS_SUPABASE_SECRET_KEY"]), sort_keys=True))'
export SCRYGLASS_DATA_PUBLISH_TOKEN="$(security find-generic-password -a scryglass-public-worker -s scryglass-data-publish-token -w)"
curl -fsS -X POST https://scryglass.xyz/api/data-published \
  -H "Authorization: Bearer ${SCRYGLASS_DATA_PUBLISH_TOKEN}" \
  -H 'Content-Type: application/json' \
  --data "{\"release_id\":\"${SCRYGLASS_PREVIOUS_RELEASE}\"}" | tee /tmp/scryglass-rollback-receipt.json
curl -fsS https://scryglass.xyz/api/health \
  -H "Authorization: Bearer ${SCRYGLASS_DIAGNOSTIC_TOKEN}" | jq -e \
  --arg release "${SCRYGLASS_PREVIOUS_RELEASE}" \
  '(.diagnostics.release_id // .pack_id) == $release'
```

Promote the previous READY Vercel deployment when the web deployment itself
caused the fault. Retain the failed release, rollback receipt, Vercel deployment
IDs, application commit, worker commit, and active asset digest inventory in the
release evidence packet.
