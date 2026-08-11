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

Use `ops/launchd/run-public-refresh.sh` for manual runs. The repository remains
detached at a tested production commit until an operator updates it.

The launch script locks the complete cycle before it opens Brave. The accepted
source receipt and import receipt bind all later stages to the same 2026 file.
The 2025 baseline remains in the runtime cache. Generated data does not enter
the worker Git checkout.

Install `xyz.scryglass.database-backup.plist.template` as a separate launch
agent after the pinned Supabase CLI is logged in and the worker checkout is
linked to the project. Install Homebrew `libpq` for `pg_dump` and `pg_restore`.
The backup job uses a short-lived database login. It keeps seven daily dumps
and four Sunday dumps. It verifies each dump before retention runs.
