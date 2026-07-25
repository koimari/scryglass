# Scryglass

Public LoL research publication (Next.js) under `apps/lol-atlas/`.

## Local

```bash
cd apps/lol-atlas
npm install
npm run dev
```

Open http://localhost:3000.

## Pack

```bash
python3 -m lol_kills.export.public_pack --years 2025,2026
python3 -m lol_kills.export.upload_pack --local-only
```

## Surfaces

- `/` — latest article
- `/articles` — research notes
- `/articles/void-grubs-contest-or-leave` — void-grubs essay
- `/elo` — Dual Elo ratings
- `/browse` — match explorer
- `/browse/head-to-head` — head-to-head
- `/reproduce` — pack download
- `/methodology` — estimands

Theme: System (default) / Light / Dark via the header control.
