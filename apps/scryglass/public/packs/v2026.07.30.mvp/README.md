# Public reproduction pack

Versioned parquet + calibration for reproducing published LoL research findings.

## Years
2025, 2026

## Contents
- `team_games/` — OE team-row maps (one file per year, zstd parquet)
- `player_games/` — OE player rows (one file per year, zstd parquet)
- `maps/` — wide map table (trimmed columns, per year)
- `features/` — Dual Elo snapshots and map-level history
- `models/` — pinned public calibration and tier-list JSON
- `studies/grubs/` — void-grubs decision numbers, briefs, PDF, key figures
- `meta/teams.json` — team aliases for display
- `manifest.json` — file list, row counts, sha256, schema_version

The exporter refuses to write a publishable manifest when a cited reproduction
file is missing.

## Not included
- Riot Match-V5 / Live Stats timelines (~GB)
- Raw Oracle's Elixir CSVs (download from OE; filters documented in manifest)
- Betting fair-odds / Slip Composer artifacts
- Draft Score calibration and recommendation artifacts while the independent
  serving review is incomplete

## Attribution
Map and player rows are derived from Oracle's Elixir public match data. Obtain raw CSVs from Oracle's Elixir; this pack is a filtered parquet subset for reproducing published research. Hierarchical Bradley–Terry, Dual Elo benchmark, and calibration are our own.

## Reproduce
1. Download this pack (or fetch partitions via the atlas app).
2. Load parquet with DuckDB / pandas / polars.
3. Match filters in the published post to `manifest.json` → `filters`.
4. For void grubs: start at `studies/grubs/grubs_decision_numbers.json` + the PDF;
   do not confuse leave-mix breakeven (~24%) with article p* ladders.
