# Public reproduction pack specification (2025–2026)

Canonical machine-readable allowlists and defaults live in
[`lol_kills/export/pack_spec.py`](../../lol_kills/export/pack_spec.py).

## Filters

| Field | Value |
|-------|--------|
| Years | **2025, 2026** (inclusive, OE `year` / `oe_year`) |
| Leagues | All leagues present in that year window |
| Schema | `1.3.0` |

## Tables

| Path | Source | Notes |
|------|--------|--------|
| `team_games/year=Y/part.parquet` | `warehouse/parquet/oe_team_games.parquet` | Column allowlist `TEAM_COLS`; `source` is `oe`, `grid`, or `mixed` |
| `player_games/year=Y/part.parquet` | `oe_player_games.parquet` | One player table; no duplicate `players.parquet`; source is retained |
| `maps/year=Y/part.parquet` | `maps.parquet` | Trimmed identity + both-side draft/obj/@10–25; `source_oe` / `source_grid` preserve provenance |
| `features/*_snapshot.*` | Hierarchical team ladder and player Dual Elo snapshots | Parquet + JSON twins for ladders |
| `features/player_weekly_ranks.json` | Player rank movement | Current rank and change versus the prior Sunday plus 1, 3, and 12 calendar months, by competitive tier |
| `features/match_index.json` | Match archive index | Compact list of every accepted 2025 and 2026 OE game, including competition level |
| `features/match_records_{year}_q{n}.json` | 2025/2026 match details, split per calendar quarter | Ten-player rosters, champions, grades, KDA, farm, gold, damage, vision, team objectives, draft pool, and composition evidence when OE supplies them; quarters keep every object under the ~50 MiB storage limit |
| `features/match_records_2025.json` / `match_records_2026.json` | Legacy single-file year archives (pre-quarter packs) | Superseded by the quarter files; the site falls back to them for legacy releases |
| `features/player_metadata.json` | Player display metadata | Leaguepedia nationality/country code/flag when available |
| `features/team_records.json` | Current team affiliation and ladder scope aggregates | Includes `current_tier`, `by_league`, and `by_tier` for hierarchical filters and win rates |
| `features/*_history.parquet` | Elo history | Rows whose `game_uid` is in year-filtered maps |
| `models/` | Pinned public calibration / tierlist CSV | See `PINNED_MODEL_FILES` |
| `studies/grubs/` | Void-grubs decision + PDF + figures | See `GRUBS_MODEL_FILES` / `GRUBS_PDF_FILES` |

## Excluded

- `warehouse/timelines/` (~1.2GB)
- Raw OE CSVs (`warehouse/raw/`)
- Private betting tooling / joblibs
- Draft Score calibration and recommendation artifacts while the independent
  serving review is incomplete

## Build

```bash
python3 -m lol_kills.update_public_pack --years 2025,2026 --refresh-oe --publish
# Or, when the warehouse is already current:
python3 -m lol_kills.export.public_pack --years 2025,2026
python3 -m lol_kills.export.upload_pack --local-only   # or Blob with token
```

Target size: tens of MB compressed (current ~26 MB for 2025–2026).
