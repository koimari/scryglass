"""Public pack schema: years, column allowlists, pinned model artifacts.

Default competitive window: 2025–2026 (user lock). Timelines / raw OE CSVs
are out of the default pack.
"""

from __future__ import annotations

SCHEMA_VERSION = "1.4.0"

# Inclusive calendar years on OE `year` / `oe_year`.
DEFAULT_YEARS: tuple[int, ...] = (2025, 2026)

# Optional major-event leagues always kept even if year filter alone would
# drop edge cases (year column is primary; this is documentation + soft hint).
DEFAULT_LEAGUES_NOTE = (
    "All canonical leagues present in OE for years 2025–2026. "
    "Legacy LTA North/South source labels are retained as provenance and mapped "
    "to LCS/CBLOL; unqualified LTA is an Americas cross-region event. "
    "International events remain separate."
)

ATTRIBUTION = (
    "Map and player rows are derived from Oracle's Elixir public match data. "
    "Obtain raw CSVs from Oracle's Elixir; this pack is a filtered parquet subset "
    "for reproducing published research. Hierarchical Bradley–Terry, Dual Elo benchmark, "
    "and calibration are our own."
)

# --- OE team / player game columns (allowlist) ---
# Drop: url, damageshare noise we don't cite, unknown dragon type, duplicate bloat.

TEAM_PLAYER_SHARED_COLS: tuple[str, ...] = (
    "gameid",
    "datacompleteness",
    "league",
    "league_source",
    "tournament",
    "competition_scope",
    "event_kind",
    "is_international",
    "is_interregional",
    "competition_tier",
    "year",
    "split",
    "playoffs",
    "date",
    "game",
    "patch",
    "participantid",
    "side",
    "position",
    "playername",
    "playerid",
    "teamname",
    "team_key",
    "teamid",
    "firstPick",
    "champion",
    "ban1",
    "ban2",
    "ban3",
    "ban4",
    "ban5",
    "pick1",
    "pick2",
    "pick3",
    "pick4",
    "pick5",
    "gamelength",
    "result",
    "kills",
    "deaths",
    "assists",
    "teamkills",
    "teamdeaths",
    "firstblood",
    "ckpm",
    "firstdragon",
    "dragons",
    "opp_dragons",
    "elementaldrakes",
    "opp_elementaldrakes",
    "elders",
    "opp_elders",
    "firstherald",
    "heralds",
    "opp_heralds",
    "void_grubs",
    "opp_void_grubs",
    "firstbaron",
    "barons",
    "opp_barons",
    "firsttower",
    "towers",
    "opp_towers",
    "inhibitors",
    "opp_inhibitors",
    "turretplates",
    "opp_turretplates",
    "damagetochampions",
    "dpm",
    "totalgold",
    "earnedgold",
    "goldspent",
    "gspd",
    "cspm",
    "goldat10",
    "xpat10",
    "csat10",
    "opp_goldat10",
    "opp_xpat10",
    "opp_csat10",
    "golddiffat10",
    "xpdiffat10",
    "csdiffat10",
    "killsat10",
    "assistsat10",
    "deathsat10",
    "opp_killsat10",
    "opp_assistsat10",
    "opp_deathsat10",
    "goldat15",
    "xpat15",
    "csat15",
    "opp_goldat15",
    "opp_xpat15",
    "opp_csat15",
    "golddiffat15",
    "xpdiffat15",
    "csdiffat15",
    "killsat15",
    "assistsat15",
    "deathsat15",
    "opp_killsat15",
    "opp_assistsat15",
    "opp_deathsat15",
    "goldat20",
    "xpat20",
    "csat20",
    "opp_goldat20",
    "opp_xpat20",
    "opp_csat20",
    "golddiffat20",
    "xpdiffat20",
    "csdiffat20",
    "killsat20",
    "assistsat20",
    "deathsat20",
    "opp_killsat20",
    "opp_assistsat20",
    "opp_deathsat20",
    "goldat25",
    "xpat25",
    "csat25",
    "opp_goldat25",
    "opp_xpat25",
    "opp_csat25",
    "golddiffat25",
    "xpdiffat25",
    "csdiffat25",
    "killsat25",
    "assistsat25",
    "deathsat25",
    "opp_killsat25",
    "opp_assistsat25",
    "opp_deathsat25",
    "source",
    "oe_year",
)

TEAM_EXTRA: tuple[str, ...] = (
    "team kpm",
    "firstbloodkill",
    "firstmidtower",
    "firsttothreetowers",
    "atakhans",
    "opp_atakhans",
)

PLAYER_EXTRA: tuple[str, ...] = (
    "damageshare",
    "earnedgoldshare",
    "visionscore",
    "vspm",
    "wardsplaced",
    "wardskilled",
    "controlwardsbought",
    "minionkills",
    "monsterkills",
    "firstbloodkill",
    "firstbloodassist",
    "firstbloodvictim",
)

TEAM_COLS = TEAM_PLAYER_SHARED_COLS + TEAM_EXTRA
PLAYER_COLS = TEAM_PLAYER_SHARED_COLS + PLAYER_EXTRA

# maps.parquet — keep identity + both sides' draft/result/objectives/@10–25 gold
MAPS_IDENTITY: tuple[str, ...] = (
    "oe_gameid",
    "game_uid",
    "league",
    "league_source",
    "competition_scope",
    "event_kind",
    "is_international",
    "is_interregional",
    "competition_tier",
    "year",
    "split",
    "playoffs",
    "date",
    "game",
    "patch",
    "blue_team",
    "red_team",
    "blue_teamname",
    "red_teamname",
    "blue_team_key",
    "red_team_key",
    "total_kills",
    "y_blue_win",
    "y_total_kills",
    "y_blue_firstblood",
    "y_blue_first_inhib",
    "gamelength",
    "length_min",
    "ckpm",
    "source_oe",
    "source_grid",
    "grid_series_id",
    "grid_game_id",
    "grid_game_index",
    "grid_completion_source",
    "oe_year",
    "tournament",
    "lp_matched",
)

MAPS_SIDE_PREFIXES: tuple[str, ...] = ("blue_", "red_")
MAPS_SIDE_FIELDS: tuple[str, ...] = (
    "result",
    "teamkills",
    "teamdeaths",
    "firstblood",
    "dragons",
    "opp_dragons",
    "void_grubs",
    "opp_void_grubs",
    "heralds",
    "barons",
    "towers",
    "opp_towers",
    "inhibitors",
    "pick1",
    "pick2",
    "pick3",
    "pick4",
    "pick5",
    "ban1",
    "ban2",
    "ban3",
    "ban4",
    "ban5",
    "goldat10",
    "golddiffat10",
    "goldat15",
    "golddiffat15",
    "goldat20",
    "golddiffat20",
    "goldat25",
    "golddiffat25",
    "totalgold",
    "earnedgold",
)


def maps_columns(available: list[str] | None = None) -> list[str]:
    cols: list[str] = list(MAPS_IDENTITY)
    for pref in MAPS_SIDE_PREFIXES:
        for field in MAPS_SIDE_FIELDS:
            cols.append(f"{pref}{field}")
    if available is None:
        return cols
    avail = set(available)
    return [c for c in cols if c in avail]


# Features Elo
RATINGS_SNAPSHOT_COLS = (
    "team",
    "team_key",
    "mu_total",
    "mu_regional",
    "mu_meta",
    "sigma",
    "rating_p10",
    "n_series",
    "n_maps",
    "international_series",
    "home_league",
    "model",
)
PLAYER_RATINGS_SNAPSHOT_COLS = (
    "player",
    "mu_total",
    "mu_regional",
    "mu_meta",
    "sigma",
    "n_maps",
    "last_team",
)
RATINGS_HISTORY_COLS = (
    "game_uid",
    "date",
    "blue_team",
    "red_team",
    "mu_blue",
    "mu_red",
    "mu_diff",
    "mu_regional_blue",
    "mu_regional_red",
    "mu_meta_blue",
    "mu_meta_red",
    "sigma_blue",
    "sigma_red",
    "sigma_pair",
    "p_dual_elo",
)
PLAYER_RATINGS_HISTORY_COLS = (
    "game_uid",
    "date",
    "blue_team",
    "red_team",
    "player_mu_blue",
    "player_mu_red",
    "player_mu_diff",
    "player_sigma_blue",
    "player_sigma_red",
    "player_sigma_pair",
    "p_player_elo",
)

# Model / calibration JSON copied into pack/models/
PINNED_MODEL_FILES: tuple[str, ...] = (
    "elo_wr_calibration.json",
    "elo_year_holdup.json",
    "draft_wr_calibration.json",
    "draft_composition.json",
    "blade_chest_role_matchups.json",
)

# Tierlist CSVs if present (small)
TIERLIST_CSV_GLOB = "tierlists_csv/*.csv"

# Void grubs study bundle → studies/grubs/ (for video / article companions)
GRUBS_MODEL_FILES: tuple[str, ...] = (
    "grubs_article_contest_ev.json",
    "grubs_decision_numbers.json",
    "grubs_decision_report.md",
    "grubs_contest_decision_paper.md",
    "grubs_intrinsic_value.json",
    "grubs_intrinsic_value_summary.md",
    "grubs_intrinsic_value_paper.md",
    "grubs_fight_probability.json",
    "grubs_action_graph.json",
    "grubs_isolation_study.json",
    "grubs_isolation_brief.md",
    "grubs_contest_study.json",
    "grubs_contest_brief.md",
    "grubs_ranked_contest_proof.json",
    "grubs_ranked_contest_proof.md",
)

# PDFs / figures under output/pdf (relative to repo root)
GRUBS_PDF_FILES: tuple[str, ...] = (
    "void_grubs_scrap_value_and_contest_rationality.pdf",
    "void_grubs_conceito_ptbr.pdf",
    "grubs_intrinsic_value.pdf",
    "void_grubs_scrap_value_and_contest_rationality_fig1_resolved_payoffs.png",
    "void_grubs_scrap_value_and_contest_rationality_fig2_threshold_ladder.png",
    "void_grubs_scrap_value_and_contest_rationality_fig3_probability_hurdle.png",
    "void_grubs_scrap_value_and_contest_rationality_fig4_outcome_matrix.png",
    "grubs_action_graph.png",
)

GRUBS_STUDY_NOTE = (
    "Article headline: two-wave leave-farm opportunity-cost p* ≈ 58.9% at parity "
    "(50/50 fight still prefers leave by ~2pp). "
    "OE trailing-team leave-mix breakeven (~24%) is a different estimand; do not collapse them. "
    "win−leave_mix ≈ +5.69pp is OE contest research, not live map-WR. "
    "Gold@10→WR is associational, not causal."
)

PACK_README = """# Public reproduction pack

Versioned parquet + calibration for reproducing published LoL research findings.

## Years
{years}

## Contents
- `team_games/` — OE team-row maps (one file per year, zstd parquet)
- `player_games/` — OE player rows (one file per year, zstd parquet)
- `maps/` — wide map table (trimmed columns, per year)
- `features/` — Dual Elo team/player snapshots + map-level history (year-filtered)
- `models/` — pinned calibration / study JSON
- `studies/grubs/` — void-grubs decision numbers, briefs, PDF, key figures
- `meta/teams.json` — team aliases for display
- `manifest.json` — file list, row counts, sha256, schema_version

## Not included
- Riot Match-V5 / Live Stats timelines (~GB)
- Raw Oracle's Elixir CSVs (download from OE; filters documented in manifest)
- Betting fair-odds / Slip Composer artifacts

## Attribution
{attribution}

## Reproduce
1. Download this pack (or fetch partitions via the atlas app).
2. Load parquet with DuckDB / pandas / polars.
3. Match filters in the published post to `manifest.json` → `filters`.
4. For void grubs: start at `studies/grubs/grubs_decision_numbers.json` + the PDF;
   do not confuse leave-mix breakeven (~24%) with article p* ladders.
"""
