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
| `team_games/year=Y/part.parquet` | `warehouse/parquet/oe_team_games.parquet` | Column allowlist `TEAM_COLS` |
| `player_games/year=Y/part.parquet` | `oe_player_games.parquet` | One player table; no duplicate `players.parquet` |
| `maps/year=Y/part.parquet` | `maps.parquet` | Trimmed identity + both-side draft/obj/@10–25 |
| `features/*_snapshot.*` | Dual Elo snapshots | Parquet + JSON twins for ladders |
| `features/*_history.parquet` | Elo history | Rows whose `game_uid` is in year-filtered maps |
| `models/` | Pinned calibration / tierlist CSV | See `PINNED_MODEL_FILES` |
| `studies/grubs/` | Void-grubs decision + PDF + figures | See `GRUBS_MODEL_FILES` / `GRUBS_PDF_FILES` |

## Excluded

- `warehouse/timelines/` (~1.2GB)
- Raw OE CSVs (`warehouse/raw/`)
- Betting fair-odds / Slip Composer / joblibs

## Build

```bash
python3 -m lol_kills.export.public_pack --years 2025,2026
python3 -m lol_kills.export.upload_pack --local-only   # or Blob with token
```

Target size: tens of MB compressed (current ~26 MB for 2025–2026).
