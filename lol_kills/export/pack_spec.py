"""Public pack schema: years, column allowlists, pinned model artifacts.

Default competitive window: 2025–2026 (user lock). Timelines / raw OE CSVs
are out of the default pack.
"""

from __future__ import annotations

SCHEMA_VERSION = "1.6.0"

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

ATTRIBUTION_OE_ONLY = (
    "Rows derive from Oracle's Elixir public match data. Obtain the raw CSVs "
    "from Oracle's Elixir; Scryglass canonicalizes identities and competition "
    "labels. Ratings, validation, and calibration are Scryglass calculations."
)

ATTRIBUTION_OE_GRID_GAP = (
    "Rows combine Oracle's Elixir as the canonical baseline with verified "
    "completed GRID games not yet present in that baseline. Overlap is "
    "deduplicated by canonical game identity with Oracle's Elixir precedence, "
    "and row-level source provenance is retained. Ratings, validation, and "
    "calibration are Scryglass calculations."
)

ATTRIBUTION_OE_GRID_DETAIL = (
    "Canonical map inclusion and results derive from Oracle's Elixir. GRID "
    "supplies verified event detail for explicitly labelled maps; it does not "
    "create an additional result row. Canonical origin and detail enrichment "
    "are published as separate provenance fields. Ratings, validation, and "
    "calibration are Scryglass calculations."
)

ATTRIBUTION_OE_GRID_GAP_AND_DETAIL = (
    "Oracle's Elixir is the canonical baseline. Verified completed GRID maps "
    "may fill a current canonical-result gap, and GRID may separately enrich "
    "event detail for an Oracle's Elixir-backed map. These two uses are "
    "published in separate provenance fields and overlap is deduplicated by "
    "canonical game identity. Ratings, validation, and calibration are "
    "Scryglass calculations."
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
    "teamname_source",
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
    "grid_series_id",
    "grid_game_id",
    "grid_game_index",
    "grid_completion_source",
    "source_series_id",
    "leaguepedia_match_id",
    "leaguepedia_game_id",
    "leaguepedia_game_index",
    "leaguepedia_best_of",
    "leaguepedia_overview_page",
    "leaguepedia_scheduled_at",
    "leaguepedia_team1",
    "leaguepedia_team2",
    "series_schedule_team_pair_status",
    "series_schedule_date_status",
    "series_format",
    "series_format_source",
    "series_format_stage_id",
    "series_format_registry_snapshot_id",
    "series_format_registry_verified",
    "series_format_registry_conflict",
    "best_of",
    "series_completion_status",
    "series_completion_source",
    "completion_source",
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
    "blue_team_source",
    "red_team",
    "red_team_source",
    "blue_teamname",
    "blue_teamname_source",
    "red_teamname",
    "red_teamname_source",
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
    "canonical_map_source",
    "map_detail_source",
    "grid_series_id",
    "grid_game_id",
    "grid_game_index",
    "grid_completion_source",
    "source_series_id",
    "leaguepedia_match_id",
    "leaguepedia_game_id",
    "leaguepedia_game_index",
    "leaguepedia_best_of",
    "leaguepedia_overview_page",
    "leaguepedia_scheduled_at",
    "leaguepedia_team1",
    "leaguepedia_team2",
    "series_schedule_team_pair_status",
    "series_schedule_date_status",
    "series_format",
    "series_format_source",
    "series_format_stage_id",
    "series_format_registry_snapshot_id",
    "series_format_registry_verified",
    "series_format_registry_conflict",
    "canonical_series_id",
    "scheduled_best_of",
    "canonical_game_index",
    "raw_source_game_index",
    "raw_source_game_uid",
    "canonical_series_status",
    "canonical_series_completion_source",
    "series_rating_eligible",
    "canonical_series_winner_team_key",
    "series_quarantine_reasons",
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
    "sigma",
    "rating_p05",
    "n_series",
    "n_maps",
    "international_series",
    "home_league",
    "last_series_at",
    "as_of",
    "sigma_kind",
    "rating_p05_interpretation",
    "comparison_component_id",
    "comparison_component_size",
    "cross_component_rankable",
    "model",
    "model_version",
)
PLAYER_RATINGS_SNAPSHOT_COLS = (
    "player",
    "mu_total",
    "mu_regional",
    "mu_meta",
    "sigma",
    "n_maps",
    "last_team",
    "outcome_exposure_group_id",
    "outcome_exposure_group_size",
    "outcome_separately_identified",
    "outcome_identifiability_label",
    "outcome_identical_players",
    "n_outcome_maps",
    "n_distinct_lineups",
    "n_distinct_teams",
)
PLAYER_PERFORMANCE_SNAPSHOT_COLS = (
    "model_id",
    "model_hash",
    "player_id",
    "player_name",
    "role",
    "last_team_key",
    "last_observed_league",
    "last_observed_date",
    "fit_through",
    "effective_sample_maps",
    "performance_mean",
    "performance_sd",
    "lower_bound",
    "rank",
    "uncertainty_method",
    "estimand",
    "publication_status",
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
    "model_validation_2026-07-27.json",
)

# Champion tierlists remain quarantined until a replacement clears the governed
# chronological/calibration contract. This path gate is intentionally broader
# than the current filenames so a renamed legacy CSV cannot re-enter a pack.
QUARANTINED_PUBLIC_PATH_TOKENS: tuple[str, ...] = (
    "tierlist",
    "champ_oe_lenses",
    "blade_chest",
    "draft_tierlist",
)


def public_path_quarantine_reason(path: str) -> str | None:
    """Return why a generated public-pack path is quarantined, if applicable."""

    normalized = str(path).replace("\\", "/").strip().casefold()
    if any(token in normalized for token in QUARANTINED_PUBLIC_PATH_TOKENS):
        return "champion tierlist artifacts are quarantined"
    if normalized.startswith("models/") and normalized.endswith(".csv"):
        return "model CSV downloads are not an approved public-pack surface"
    return None


def require_publication_paths_allowed(paths: tuple[str, ...] | list[str]) -> None:
    """Fail closed if any proposed pack path belongs to a quarantined surface."""

    blocked = [
        (path, reason)
        for path in paths
        if (reason := public_path_quarantine_reason(path)) is not None
    ]
    if blocked:
        details = ", ".join(f"{path} ({reason})" for path, reason in blocked)
        raise ValueError(f"public pack path quarantine rejected: {details}")

# One fail-closed Void Grubs article artifact. Broader internal studies, OE
# leave-mix outputs, figures, and superseded PDFs are not public-pack inputs.
GRUBS_MODEL_FILES: tuple[str, ...] = (
    "grubs_article_contest_ev.json",
)

# The existing paper cannot be regenerated from the canonical current-mechanics
# article writer and is withheld rather than shipped with unsupported claims.
GRUBS_PDF_FILES: tuple[str, ...] = ()

GRUBS_STUDY_NOTE = (
    "Patch 26.11+ article sensitivity: the 124.13g current-mechanics objective "
    "equivalent gives a two-wave opportunity-cost contest bar of about 58.24% at "
    "parity. The Touch term is an upper-bound plate-progress equivalent, not "
    "guaranteed gold. Gold-at-10 to map-win is associational, and the contest "
    "bar is not an identified action policy."
)

PACK_README = """# Public reproduction pack

Versioned parquet + calibration for reproducing published LoL research findings.

## Years
{years}

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
{attribution}

## Reproduce
1. Download this pack (or fetch partitions via the atlas app).
2. Load parquet with DuckDB / pandas / polars.
3. Match filters in the published post to `manifest.json` → `filters`.
4. For void grubs: use only `studies/grubs/grubs_article_contest_ev.json`.
   Its strict schema records current mechanics, the exact estimand, and limits.
"""
