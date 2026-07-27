# Public reproduction pack specification (2025–2026)

Canonical machine-readable allowlists and defaults live in
[`lol_kills/export/pack_spec.py`](../../lol_kills/export/pack_spec.py).

## Filters

| Field | Value |
|-------|--------|
| Years | **2025, 2026** (inclusive, OE `year` / `oe_year`) |
| Leagues | All leagues present in that year window |
| Schema | `1.6.0` |

## Tables

| Path | Source | Notes |
|------|--------|--------|
| `team_games/year=Y/part.parquet` | `warehouse/parquet/oe_team_games.parquet` | Column allowlist `TEAM_COLS`; `source` is `oe`, `grid`, or `mixed`; exact Riot-platform-ID schedule fields preserve match/format provenance without rewriting team identity |
| `player_games/year=Y/part.parquet` | `oe_player_games.parquet` | One player table; no duplicate `players.parquet`; source is retained |
| `maps/year=Y/part.parquet` | `maps.parquet` | Trimmed identity + both-side draft/obj/@10–25; `canonical_map_source` identifies OE inclusion versus verified GRID gap fill, while `map_detail_source` separately identifies optional detail enrichment; canonical completed-series fields are rebuilt and schedule team/date reconciliation status is retained |
| `features/*_snapshot.*` | Hierarchical team ladder and player Dual Elo snapshots | Parquet + JSON twins for ladders |
| `features/player_ratings_meta.json` | Player outcome-signal publication gate | Explicitly withholds individual ordering and weekly rank movement until an individual-skill estimand passes its own validation gate |
| `features/player_metadata.json` | Player display metadata | Leaguepedia nationality/country code/flag when available |
| `features/team_records.json` | Current team affiliation and ladder scope aggregates | Includes `current_tier`, `by_league`, and `by_tier` for hierarchical filters and win rates |
| `features/*_history.parquet` | Elo history | Rows whose `game_uid` is in year-filtered maps |
| `models/` | Pinned calibration and governed validation JSON only | See `PINNED_MODEL_FILES` |
| `studies/grubs/` | Void-grubs decision + PDF + figures | See `GRUBS_MODEL_FILES` / `GRUBS_PDF_FILES` |

## Excluded

- `warehouse/timelines/` (~1.2GB)
- Raw OE CSVs (`warehouse/raw/`)
- Champion tierlists, `champ_oe_lenses`, Blade-Chest artifacts, and model CSVs
- Private odds / prediction tooling and joblibs

Champion tierlists are publication-quarantined. The machine-readable path gate
in `pack_spec.py` rejects these files even if they are accidentally added to a
future allowlist or generated under a renamed model CSV path.

## Build

```bash
python3 -m lol_kills.update_public_pack --years 2025,2026 --download-oe --download-grid --grid-required --publish
# Or, when the warehouse is already current:
python3 -m lol_kills.export.public_pack --years 2025,2026
python3 -m lol_kills.export.upload_pack --local-only   # or Blob with token
```

Target size: tens of MB compressed (current ~26 MB for 2025–2026).
