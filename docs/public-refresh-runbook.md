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
npx supabase db push --linked
npx supabase migration list --linked
```

The current worker needs migrations `20260811174735` and `20260811193000`. Keep the old worker commit active until both appear in the remote list.

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
