# Six-hour public refresh

The Mac worker updates public data. GitHub and website builds are not part of
the data path. The current deployment reads the active Supabase release.

```text
Oracle's Elixir annual CSV files
        ↓
six-hour source cache check
        ↓
new canonical game IDs
        ↓
complete game, team, and player validation
        ↓
team ratings and player ratings
        ↓
patch-wide tier candidate and authority checks
        ↓
immutable Supabase release and tier display
        ↓
atomic pointers, cache invalidation, and smoke checks
```

The site continues to read the previous release during a refresh. A malformed game or failed checksum stops publication.

The worker binds the current pack to its exact canonical game-ID set before it
accepts a changed annual file. Every ID in that set must remain in the next
accepted source. This prevents an incomplete file from removing completed games,
ratings history, or profile history.

## Host paths

- Code: `~/Library/Application Support/Scryglass Worker/repo`
- Published packs: `~/Library/Application Support/Scryglass Worker/public-packs`
- Worker state and health: `~/Library/Application Support/Scryglass Worker/runtime/data/lol/runtime`
- Source and warehouse cache: `~/Library/Application Support/Scryglass Worker/runtime/data/lol/warehouse`

The deployed server reads the active Supabase manifest and immutable assets.
Bundled files support builds and local development. Production requests do not
fall back to another data transport.

Copy `ops/systemd/public-refresh.env.example` to
values. Alerts are optional. Keep the file outside the code checkout.

## Publication checks

Each new game must have one game row, two named teams with complementary results, and ten unique named players. Both sides need one player in each canonical role. Every declared public file must match its recorded size and SHA-256 digest.

The manifest also stores a count and SHA-256 digest for the sorted canonical game IDs. This proves that the rating export used the same complete source set that passed validation.

Tier lists use a separate authority gate inside the same public refresh run. A
ratings result can remain active while a tier authority check waits for a later
cycle. The run records that partial state. An optional webhook can notify the
owner.

## Service setup

Install the refresh agent from `ops/launchd`. It runs at minute 0 every six
hours. The local lock prevents overlap. The separate backup agent runs daily.

Run a manual check with:

```bash
~/Library/Application\ Support/Scryglass\ Worker/run-public-refresh.sh
```

Read the runtime health file for the full local result. The public website
receives only the sanitized Supabase health row.

The six-hour timer must not call a deployment command. One merged code release
can run one production build. The daily Scryglass build CPU budget is 60
minutes.
