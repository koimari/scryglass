# Scryglass

Independent League of Legends research publication by koi. Authored essays lead; Dual Elo ratings,
match explorer, and head-to-head boards support the claims with reproducible Oracle’s Elixir packs
(2025–2026).

## Public site

```bash
cd apps/lol-atlas
npm install
npm run dev
```

Open http://localhost:3000.

## Surfaces

- `/` — latest article
- `/articles` — research notes index
- `/articles/void-grubs-contest-or-leave` — void-grubs contest-or-leave essay (+ charts)
- `/elo` — Dual Elo ratings
- `/browse` — match explorer
- `/browse/head-to-head` — head-to-head
- `/methodology` — estimands
- `/reproduce` — pack download / reproduction

`/grubs` permanently redirects to the void-grubs article.

## Pack

```bash
python3 -m lol_kills.export.public_pack --years 2025,2026
python3 -m lol_kills.export.upload_pack --local-only
```

## Voice

Hedges and full sentences. Plain labels for lay readers (“contest bar”, not bare math nicknames).
Not a betting product.
