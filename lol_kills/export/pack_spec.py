"""Schema for the small public ratings payload."""

from __future__ import annotations

SCHEMA_VERSION = "2.0.0"
DEFAULT_YEARS: tuple[int, ...] = (2025, 2026)

DEFAULT_LEAGUES_NOTE = (
    "All canonical leagues in the selected years. International and "
    "interregional events remain separate scopes."
)

ATTRIBUTION = (
    "Ratings use completed professional match data from Oracle's Elixir, "
    "with an accepted completed-game bridge when the daily export is pending."
)

# These columns are read from the local map table to calculate rating outputs.
# They are never copied into the public payload.
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
    "date",
    "blue_team",
    "red_team",
    "blue_teamname",
    "red_teamname",
    "blue_team_key",
    "red_team_key",
    "blue_result",
    "red_result",
    "y_blue_win",
    "source_oe",
    "source_grid",
    "oe_year",
    "tournament",
    # Draft identity is read for the public profile pool audit. These columns
    # remain in memory and are never copied to raw public tables.
    "patch",
    "blue_firstPick",
    "red_firstPick",
    "blue_ban1",
    "blue_ban2",
    "blue_ban3",
    "blue_ban4",
    "blue_ban5",
    "red_ban1",
    "red_ban2",
    "red_ban3",
    "red_ban4",
    "red_ban5",
    "blue_pick1",
    "blue_pick2",
    "blue_pick3",
    "blue_pick4",
    "blue_pick5",
    "red_pick1",
    "red_pick2",
    "red_pick3",
    "red_pick4",
    "red_pick5",
)


def maps_columns(available: list[str] | None = None) -> list[str]:
    if available is None:
        return list(MAPS_IDENTITY)
    present = set(available)
    return [column for column in MAPS_IDENTITY if column in present]


EVIDENCE_STATE_COLS = (
    "evidence_interval_width",
    "evidence_precision_ratio",
    "evidence_stability",
    "evidence_freshness_days",
    "evidence_support_coverage",
    "evidence_fallback",
    "evidence_active",
    "evidence_disconnected",
    "evidence_ood",
    "evidence_state",
)

RATINGS_SNAPSHOT_COLS = (
    "team",
    "team_key",
    "mu_base_total",
    "mu_total",
    "mu_effective",
    "momentum_residual",
    "mu_regional",
    "mu_meta",
    "sigma",
    "rating_p10",
    "n_series",
    "n_maps",
    "international_series",
    "home_league",
    *EVIDENCE_STATE_COLS,
)

PLAYER_RATINGS_SNAPSHOT_COLS = (
    "player",
    "mu_base_total",
    "mu_total",
    "mu_effective",
    "momentum_residual",
    "mu_regional",
    "mu_meta",
    "sigma",
    "n_maps",
    "last_team",
    "home_league",
    *EVIDENCE_STATE_COLS,
)

PUBLIC_RATING_REQUIRED_FILES: tuple[str, ...] = (
    "features/ratings_snapshot.json",
    "features/player_ratings_snapshot.json",
    "features/team_records.json",
    "features/team_weekly_ranks.json",
    "features/player_records.json",
    "features/player_champion_records.json",
    "features/profile_records.json",
    "features/match_index.json",
    "features/match_records_2025_q1.json",
    "features/match_records_2025_q2.json",
    "features/match_records_2025_q3.json",
    "features/match_records_2025_q4.json",
    "features/match_records_2026_q1.json",
    "features/match_records_2026_q2.json",
    "features/match_records_2026_q3.json",
    "features/match_records_2026_q4.json",
    "features/player_weekly_ranks.json",
    "features/player_metadata.json",
)

# Display-only files can be absent when their external source is unavailable.
# They never take part in the ratings release gate.
OPTIONAL_PUBLIC_FILES: tuple[str, ...] = (
    "features/schedule.json",
    "features/leaderboards.json",
    "features/draft_records.json",
    "features/promoted_draft_results.json",
    "features/player_map_stats.json",
)

MATCH_YEAR_COMPATIBILITY_FILES: tuple[str, ...] = tuple(
    f"features/match_records_{year}.json" for year in DEFAULT_YEARS
)

TIER_PUBLIC_FILES: tuple[str, ...] = (
    "rankings/tierlists.json",
    "rankings/tierlists-latest.json",
)

# PUBLIC_ASSET_ALLOWLIST_V1. Keep this marker beside the one authoritative
# tuple. A cross-language contract test reads each runtime consumer and the
# final SQL constraint so an allowlist change cannot ship on one boundary only.
PUBLIC_ASSET_PATHS: tuple[str, ...] = (
    *PUBLIC_RATING_REQUIRED_FILES,
    *OPTIONAL_PUBLIC_FILES,
    *MATCH_YEAR_COMPATIBILITY_FILES,
    *TIER_PUBLIC_FILES,
)

FORBIDDEN_PUBLIC_MODEL_FILES: tuple[str, ...] = (
    "draft_wr_calibration.json",
    "draft_recommendation.json",
    "blade_chest_role_matchups.json",
    "features/draft_context.json",
    "models/",
    "studies/",
    "team_games/",
    "player_games/",
    "maps/",
)

WITHHELD_PUBLIC_FILES = FORBIDDEN_PUBLIC_MODEL_FILES
