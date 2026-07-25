# Scryglass

Public LoL research companion (Next.js) under `apps/lol-atlas/`.

## Local

```bash
cd apps/lol-atlas
npm install
npm run dev
```

Open http://localhost:3000.

## Pack

Refresh years 2025–2026 into `public/packs/`:

```bash
python3 -m lol_kills.export.public_pack --years 2025,2026
python3 -m lol_kills.export.upload_pack --local-only
```

## Surfaces

- `/elo` — Dual Elo ladders
- `/grubs` — void-grubs article p*
- `/browse` — one row per OE map
- `/browse/head-to-head` — H2H + Leaguepedia-style scoreboard
- `/browse/match/[gameId]` — single match board
- `/reproduce` — pack download
- `/methodology` — estimands

Theme: System (default) / Light / Dark via the header control.
