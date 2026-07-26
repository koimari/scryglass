# Public pack publish checklist

1. Current-data refresh: `python3 -m lol_kills.update_public_pack --years 2025,2026 --download-oe --download-grid --grid-required --publish`
   - OE is the reconciled baseline.
   - GRID is a pro-only freshness bridge for completed games not yet in OE.
   - `GRID_API_KEY` may come from the environment or the sibling `lol-strength-analysis/.env` locally.
2. Offline export, when the warehouse is already current: `python3 -m lol_kills.export.public_pack --years 2025,2026`
3. Publish pack:
   - Dev: `python3 -m lol_kills.export.upload_pack --local-only`
   - Prod CDN: `BLOB_READ_WRITE_TOKEN=… python3 -m lol_kills.export.upload_pack`
4. Vercel: deploy `apps/lol-atlas` (Root Directory `apps/lol-atlas`). The scheduled workflow commits only the two small pack pointers; Git integration should deploy that commit. If Git integration is not active, set the repository variable `VERCEL_DEPLOY_HOOK_URL`.
5. When posting a finding: cite `pack_id` + filters from `/packs/manifest.json`.
6. For void-grubs companions: link `/articles/void-grubs-contest-or-leave` and pin `studies/grubs/grubs_decision_numbers.json`.

Do not ship betting tooling, fair-odds boards, or timelines in the default pack.

## Scheduled refresh

`.github/workflows/refresh-public-pack.yml` runs every 15 minutes and can also
be started manually. Configure these repository secrets first:

- `GRID_API_KEY`
- `BLOB_READ_WRITE_TOKEN`

Each successful run writes a new immutable Blob path and updates
`apps/lol-atlas/public/packs/manifest.json` and `latest.json`. Versioned paths
avoid serving a half-written pack; the site always reads the latest complete
pointer.
