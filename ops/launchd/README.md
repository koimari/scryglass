# macOS public refresh worker

The launch agent runs the Oracle's Elixir public refresh at 00:00, 06:00,
12:00, and 18:00 local time. macOS starts one missed calendar run after the
computer wakes.

The worker reads the Supabase publisher key and website cache token from the
login Keychain. The plist and logs contain no credentials.

Each run opens the current Oracle's Elixir annual file in Brave Origin. Brave
must save downloads automatically to the worker `oe-inbox` directory. The
worker waits for the download, validates it, archives changed source bytes, and
stops if a fresh browser download does not arrive. It does not publish from an
older annual-file cache after a browser download failure.

After validation, the worker groups the annual file by canonical game ID. It
adds new games to the private Supabase OE tables and records corrected versions
without replacing history. Previously stored games must remain present. The
local normalized cache updates from the same accepted groups, so the public
refresh does not parse the annual file a second time.

The installed runtime uses these paths:

- `~/Library/Application Support/Scryglass Worker/repo`
- `~/Library/Application Support/Scryglass Worker/venv`
- `~/Library/Application Support/Scryglass Worker/public-packs`
- `~/Library/Application Support/Scryglass Worker/runtime`
- `~/Library/Application Support/Scryglass Worker/logs`
- `~/Library/Application Support/Scryglass Worker/backups/postgres`
- `~/Library/Application Support/Scryglass Worker/oe-inbox`
- `~/Library/LaunchAgents/xyz.scryglass.public-refresh.plist`

Use `ops/launchd/run-public-refresh.sh` for manual runs.

The launcher pins itself. Every run starts by fetching `origin/main` and, when
the checkout is clean and behind, resetting the worker checkout onto it. It
then reinstalls itself from `ops/launchd/run-public-refresh.sh` in the synced
commit if the installed copy differs, re-executes once, and derives
`SCRYGLASS_WORKER_COMMIT` from the resulting HEAD. The launch agent therefore
carries no commit and needs no re-render when main moves.

This is safe because main is the tested branch: merging into it requires green
required checks and resolved review threads, so nothing reaches main untested.
The binding is not weakened either. `refresh_ledger.worker_commit` still
verifies the environment against the real HEAD and still refuses a dirty
checkout; deriving the value simply makes that check hold by construction
instead of by an operator remembering to re-render the launch agent. Every
receipt and ledger entry stays bound to the exact commit that produced it.

Failure handling is deliberately asymmetric. A failed fetch is fail-open: the
run logs a warning and continues on the commit already checked out, because a
network blip must not cancel a refresh. Everything about consistency is
fail-closed: a dirty checkout is never reset and never synced over, and it
still stops the run with `The worker checkout contains uncommitted files`.

Export `SCRYGLASS_WORKER_COMMIT` to pin a manual run to an exact commit. The
launcher then requires the real HEAD to equal it and stops otherwise, which is
the behaviour to use when testing a commit that is not on main yet.

A run that is killed leaves `runtime/data/lol/runtime/public-refresh-worker.lock`
behind. `shlock` is documented to reclaim such a lock, but the macOS build gates
the takeover on the lock file's ctime and refuses when that ctime is not
strictly older than the invocation, so a leftover lock could block every later
run until it was deleted by hand. The launcher now reads the owning pid first
and removes the lock only when that process is gone. A lock held by a live
process is never removed and still refuses the run.

Because the checkout syncs before `ops/verify-public-refresh-env.sh` runs, a
commit that changes `requirements.lock` stops the run with
`The worker environment does not match requirements.lock` until an operator
reinstalls the worker venv. That is intended: the worker must not publish from
an environment that does not match the commit it is attesting to.

The launch script locks the complete cycle before it opens Brave. The accepted
source receipt and import receipt bind all later stages to the same 2026 file.
The 2025 baseline remains in the runtime cache. Generated data does not enter
the worker Git checkout.

Install `xyz.scryglass.database-backup.plist.template` as a separate launch
agent after the pinned Supabase CLI is logged in and the worker checkout is
linked to the project. Install Homebrew `libpq` for `pg_dump` and `pg_restore`.
The backup job uses a short-lived database login. It keeps seven daily dumps
and four Sunday dumps. It verifies each dump before retention runs.
