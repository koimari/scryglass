# Public reproduction pack

Versioned parquet + calibration for reproducing published LoL research findings.

## Years
2025, 2026

## Contents
- `team_games/` — OE team-row maps (one file per year, zstd parquet)
- `player_games/` — OE player rows (one file per year, zstd parquet)
- `maps/` — wide map table (trimmed columns, per year)
- `features/` — team and Player Dual Elo outputs plus the separately named,
  role-specific 15-minute resource-performance snapshot, metadata, and compact
  chronological validation artifact
- `models/` — pinned calibration / study JSON
- `studies/grubs/` — one versioned, current-mechanics article JSON
- `meta/teams.json` — team aliases for display
- `meta/source_summary.json` — sanitized source counts and dedupe policy
- `manifest.json` — governed public file list, row counts, sha256, schema_version

## Not included
- Riot Match-V5 / Live Stats timelines (~GB)
- Raw Oracle's Elixir CSVs (download from OE; filters documented in manifest)
- Champion tierlists and model CSVs pending a validated replacement
- Private odds / prediction tooling

## Attribution
Rows derive from Oracle's Elixir public match data. Obtain the raw CSVs from Oracle's Elixir; Scryglass canonicalizes identities and competition labels. Ratings, validation, and calibration are Scryglass calculations.

## Reproduce
1. Download this pack (or fetch partitions via the atlas app).
2. Load parquet with DuckDB / pandas / polars.
3. Match filters in the published post to `manifest.json` → `filters`.
4. For void grubs: use only `studies/grubs/grubs_article_contest_ev.json`.
   Its strict schema records current mechanics, the exact estimand, and limits.
