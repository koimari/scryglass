# Public pack publish checklist

Canonical public site: https://scryglass.xyz

Use the canonical domain for post-release verification. The Vercel project
domain is not the publication URL.

1. Current-data refresh: `python3 -m lol_kills.update_public_pack --years 2025,2026 --download-oe --download-grid --grid-required --publish`
   - OE is the reconciled baseline.
   - GRID is a pro-only freshness bridge for completed games not yet in OE.
   - `GRID_API_KEY` may come from the environment or the sibling `lol-strength-analysis/.env` locally.
2. Offline export, when the warehouse is already current: `python3 -m lol_kills.export.public_pack --years 2025,2026`
3. Publish pack:
   - Dev: `python3 -m lol_kills.export.upload_pack --local-only`
   - Prod CDN: `BLOB_READ_WRITE_TOKEN=… python3 -m lol_kills.export.upload_pack`
4. Vercel: deploy `apps/scryglass` (Root Directory `apps/scryglass`). The scheduled workflow commits only the two small pack pointers; Git integration should deploy that commit. If Git integration is not active, set the repository variable `VERCEL_DEPLOY_HOOK_URL`.
5. When posting a finding: cite `pack_id` + filters from `/packs/manifest.json`.
6. For void-grubs companions: link `/articles/void-grubs-contest-or-leave` and pin `studies/grubs/grubs_decision_numbers.json`.

Do not ship betting tooling, fair-odds boards, or timelines in the default pack.

## Manual refresh

`.github/workflows/refresh-public-pack.yml` and
`.github/workflows/reconcile-oe-baseline.yml` are manual-only while the
publication, storage, and cost gates are unresolved. Configure these repository
secrets before an approved run:

- `GRID_API_KEY`
- `BLOB_READ_WRITE_TOKEN`

Each successful approved run writes a new immutable Blob path and updates
`apps/scryglass/public/packs/manifest.json` and `latest.json`. Versioned paths
avoid serving a half-written pack; the site always reads the latest complete
pointer.
