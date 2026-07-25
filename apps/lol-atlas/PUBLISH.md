# Public pack publish checklist

1. Offline: refresh warehouse / Dual Elo as needed.
2. Export: `python3 -m lol_kills.export.public_pack --years 2025,2026`
3. Publish pack:
   - Dev: `python3 -m lol_kills.export.upload_pack --local-only`
   - Prod CDN: `BLOB_READ_WRITE_TOKEN=… python3 -m lol_kills.export.upload_pack`
4. Vercel: deploy `apps/lol-atlas` (Root Directory `apps/lol-atlas`).
5. When posting a finding: cite `pack_id` + filters from `/packs/manifest.json`.
6. For void-grubs companions: link `/articles/void-grubs-contest-or-leave` and pin `studies/grubs/grubs_decision_numbers.json`.

Do not ship betting tooling, fair-odds boards, or timelines in the default pack.
