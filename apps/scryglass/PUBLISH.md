# Public rankings publish checklist

Canonical public site: https://scryglass.xyz

The public payload contains seven rating files and one accepted tier-list file.

1. Refresh the local source cache. Canonicalize game IDs before joining new rows.
2. Export ratings with `python3 -m lol_kills.export.public_pack --years 2025,2026`.
3. Build one champion board for each patch and role. Pool all eligible competitions in that patch.
4. Run the source, identity, coverage, and authority checks. Keep the previous accepted files when any check fails.
5. Publish the complete accepted files together. Verify `/elo`, `/tiers`, and `/methodology` on `https://scryglass.xyz`.

## Manual refresh

Run the local refresh every six hours. GitHub checks can validate code changes.
They are not part of the data refresh path.
