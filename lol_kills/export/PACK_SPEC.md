# Public reproduction pack specification (2025–2026)

Canonical machine-readable allowlists and defaults live in
[`lol_kills/export/pack_spec.py`](../../lol_kills/export/pack_spec.py).

## Filters

| Field | Value |
|-------|--------|
| Years | **2025, 2026** (inclusive, OE `year` / `oe_year`) |
| Leagues | All leagues present in that year window |
| Schema | `1.0.0` |

## Tables

| Path | Source | Notes |
|------|--------|--------|
| `team_games/year=Y/part.parquet` | `warehouse/parquet/oe_team_games.parquet` | Column allowlist `TEAM_COLS`; `source` is `oe`, `grid`, or `mixed` |
| `player_games/year=Y/part.parquet` | `oe_player_games.parquet` | One player table; no duplicate `players.parquet`; source is retained |
| `maps/year=Y/part.parquet` | `maps.parquet` | Trimmed identity + both-side draft/obj/@10–25; `source_oe` / `source_grid` preserve provenance |
| `features/*_snapshot.*` | Hierarchical team ladder and player Dual Elo snapshots | Parquet + JSON twins for ladders |
| `features/*_history.parquet` | Elo history | Rows whose `game_uid` is in year-filtered maps |
| `models/` | Pinned calibration / tierlist CSV | See `PINNED_MODEL_FILES` |
| `studies/grubs/` | Void-grubs decision + PDF + figures | See `GRUBS_MODEL_FILES` / `GRUBS_PDF_FILES` |

## Excluded

- `warehouse/timelines/` (~1.2GB)
- Raw OE CSVs (`warehouse/raw/`)
- Private betting tooling / joblibs

## Build

```bash
python3 -m lol_kills.update_public_pack --years 2025,2026 --download-oe --download-grid --grid-required --publish
# Or, when the warehouse is already current:
python3 -m lol_kills.export.public_pack --years 2025,2026
python3 -m lol_kills.export.upload_pack --local-only   # or Blob with token
```

Target size: tens of MB compressed (current ~26 MB for 2025–2026).
