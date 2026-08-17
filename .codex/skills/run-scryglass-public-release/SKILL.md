---
name: run-scryglass-public-release
description: Run, repair, or verify the Scryglass production data release. Use for Oracle's Elixir refreshes, new patches, ratings, Draft Score, Tier Lists, matches, Supabase activation, cache invalidation, release health, or reports that public tabs show different snapshots.
---

# Run Scryglass Public Release

Use one accepted source census for every public surface.

## Production path

1. Inspect the active worker and repository state. Preserve unrelated changes.
2. Load worker secrets from Keychain without printing them.
3. Run only this production entry point:

```sh
python3 -m lol_kills.public_refresh --once --force
```

4. Let the command ingest OE, freeze accepted game IDs, build ratings, refresh Tier Lists, score Draft, build matches, publish Supabase, invalidate caches, and write health.
5. Do not run a separate production tier, Draft, ratings, or pack command.
6. Require the same `source_as_of`, `source_game_count`, and `source_identity_sha256` in every release receipt.

## Release gates

Require these results:

- Supabase has one active release.
- Public health is `ok`, `idle`, and `stale: false`.
- Manifest, health, HTML, and public queries use one release ID.
- Team Draft covers the full scoreable archive. Player best-available metrics can use the smaller complete-pool set.
- Tier Lists uses the accepted patch census and labels champion counts as appearances.
- Match rows contain canonical league and competition tier values.
- Current match results are available when the accepted census has matching games.
- Supabase security and performance advisors return no findings.
- Production browser console and Vercel runtime checks are clean after cache invalidation.

Run the bundled live verifier after activation:

```sh
python3 .codex/skills/run-scryglass-public-release/scripts/verify_live_release.py \
  --site https://scryglass.xyz
```

## Failure handling

Stop before activation when a census binding differs. Restore the prior active release when a post-activation gate fails. Invalidate caches with the restored release ID. Keep the failed receipt and exact error.

## Patch handling

Keep the OE source token, such as `16.16`, in source evidence. Convert it once to the public label, such as `26.16`, at the display boundary. The release census must carry both values. A patch update changes the accepted source and triggers the full production path.
