"""Export a compressed, versioned public reproduction pack (2025–2026 default).

Usage:
  python3 -m lol_kills.export.public_pack
  python3 -m lol_kills.export.public_pack --years 2025,2026 --out output/public_pack
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from lol_kills.export import pack_spec as spec
from lol_kills.export.pack_records import (
    build_current_tournament_membership,
    build_maps_frame_from_team_games,
    build_player_records,
    build_team_records,
    complete_public_map_population,
)
from lol_kills.export.player_metadata import build_player_metadata
from lol_kills.export.player_performance_artifacts import (
    REQUIRED_PUBLIC_YEARS,
    build_player_performance_public_artifacts,
)
from lol_kills.export.upload_pack import validate_pack_id
from lol_kills.etl.competition import TAXONOMY_VERSION, canonicalize_competition_frame
from lol_kills.etl.series_ledger import build_canonical_series_ledger
from lol_kills.etl.series_schedule import annotate_scheduled_series, load_schedule
from lol_kills.etl.tournament_registry import (
    annotate_maps_with_tournament_registry,
    load_tournament_registry,
)
from lol_kills.ratings.player_elo import build_maps_frame_from_players
from lol_kills.ratings.dual_elo import (
    build_dual_ratings,
    lineup_hashes_from_players,
)
from lol_kills.ratings.player_elo import build_player_rating_artifacts
from lol_kills.ratings.series_dynamic_bt import (
    MODEL_ID as SERIES_RATING_MODEL_ID,
    SeriesTournamentSpec,
    run_series_rating_tournament,
)

ROOT = Path(__file__).resolve().parents[2]
WAREHOUSE = ROOT / "data" / "lol" / "warehouse" / "parquet"
FEATURES = ROOT / "data" / "lol" / "features"
MODELS = ROOT / "data" / "lol" / "models"
TEAMS_JSON = ROOT / "web" / "composer" / "teams.json"
DEFAULT_OUT = ROOT / "output" / "public_pack"
VERIFIED_GRID_COMPLETION_SOURCES = frozenset(
    {"events_game_end", "end_state_summary"}
)


def _build_public_series_ratings(
    maps: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Build the frozen, leakage-safe public team-rating release."""

    completion = pd.to_datetime(maps.get("date"), errors="coerce", utc=True)
    if completion.dropna().empty:
        raise RuntimeError("Team-rating input has no valid completion time.")
    cutoff = completion.max().ceil("s").isoformat()
    tournament = run_series_rating_tournament(
        maps,
        spec=SeriesTournamentSpec(
            validation_start="2025-09-01T00:00:00Z",
            test_start="2026-04-01T00:00:00Z",
            data_cutoff=cutoff,
            bootstrap_replicates=5000,
            moving_block_size=8,
            alpha=0.05,
            noninferiority_margin=0.005,
            random_seed=20260726,
            minimum_test_series=500,
        ),
    )
    if not tournament.gate.get("passed"):
        raise RuntimeError(
            "Series Dynamic Bradley-Terry did not pass its frozen "
            "chronological promotion gate."
        )

    ledger = tournament.prediction_ledger
    exposures = pd.concat(
        [
            ledger[
                ["team_a", "n_maps", "international"]
            ].rename(columns={"team_a": "team_key"}),
            ledger[
                ["team_b", "n_maps", "international"]
            ].rename(columns={"team_b": "team_key"}),
        ],
        ignore_index=True,
    )
    exposure_summary = (
        exposures.groupby("team_key", as_index=False)
        .agg(
            n_series=("team_key", "size"),
            n_maps=("n_maps", "sum"),
            international_series=("international", "sum"),
        )
    )
    identity = dict(tournament.metadata["model"])
    ratings = tournament.snapshot.merge(
        exposure_summary,
        on="team_key",
        how="left",
        validate="one_to_one",
    ).rename(columns={"mean": "mu_total"})
    ratings["n_series"] = ratings["n_series"].fillna(0).astype(int)
    ratings["n_maps"] = ratings["n_maps"].fillna(0).astype(int)
    ratings["international_series"] = (
        ratings["international_series"].fillna(0).astype(int)
    )
    ratings["model"] = SERIES_RATING_MODEL_ID
    ratings["model_version"] = identity["model_version"]

    observation_audit = dict(tournament.metadata["observation_audit"])
    ledger_audit = dict(observation_audit.get("series_ledger_audit") or {})
    meta = {
        "model": SERIES_RATING_MODEL_ID,
        **identity,
        "as_of": tournament.metadata["snapshot"]["as_of"],
        "n_series": int(len(ledger)),
        "n_maps": int(ledger["n_maps"].sum()),
        "series_ledger_audit": ledger_audit,
        "input_audit": {
            "ok": bool(ledger_audit.get("ok")),
            "observation_rows_sha256": tournament.metadata[
                "observation_rows_sha256"
            ],
            "cutoff_policy": observation_audit.get("cutoff_policy"),
            "forbidden_predictors": observation_audit.get(
                "forbidden_predictors"
            ),
        },
        "config": tournament.selected_config.__dict__,
        "uncertainty": {
            "field": "rating_p05",
            "z": 1.6448536269514722,
            "formula": "rating_p05 = mu_total - z * sigma",
            "sigma_kind": "diagonal_filter_approximation_sd",
            "coverage_claim": False,
            "interpretation": (
                "Normal-approximation lower quantile for conservative "
                "ordering within a connected comparison component; empirical "
                "coverage has not been established."
            ),
        },
        "comparison_components": {
            "count": int(ratings["comparison_component_id"].nunique()),
            "cross_component_rankable": False,
            "policy": (
                "Ratings may be ordered only within one connected historical "
                "comparison component."
            ),
        },
        "tournament": {
            "spec": tournament.metadata["tournament_spec"],
            "spec_sha256": tournament.metadata["tournament_spec_sha256"],
            "selection": tournament.metadata["selection"],
            "split": tournament.metadata["split"],
            "final_metrics": tournament.final_metrics,
            "comparisons": tournament.comparisons,
            "gate": tournament.gate,
        },
    }
    gate = {
        "model_id": identity["model_id"],
        "model_version": identity["model_version"],
        "model_code_sha256": identity["model_code_sha256"],
        "model_config_sha256": identity["model_config_sha256"],
        "observation_rows_sha256": tournament.metadata[
            "observation_rows_sha256"
        ],
        "estimand": "pre_series_organization_strength_probability",
        "gate_status": "passed",
        "temporal_audit": {
            "ok": True,
            "prediction_time": "verified series start",
            "assimilation_time": "verified series completion",
            "test_labels_used_for_selection": False,
        },
        "final_test": {
            "series": tournament.final_metrics[SERIES_RATING_MODEL_ID]["n"],
            "log_loss": tournament.final_metrics[
                SERIES_RATING_MODEL_ID
            ]["log_loss"],
            "brier": tournament.final_metrics[SERIES_RATING_MODEL_ID][
                "brier"
            ],
            "ece_10_equal_width": tournament.final_metrics[
                SERIES_RATING_MODEL_ID
            ]["ece"],
            "format_stratified_calibration": tournament.final_metrics[
                SERIES_RATING_MODEL_ID
            ]["format_stratified_calibration"],
        },
        "paired_primary_comparison": {
            "baseline_model_id": "rolling_series_elo",
            "primary_score": "log_loss",
            **tournament.comparisons["rolling_series_elo"],
        },
        "secondary_comparison": tournament.comparisons[
            "historical_symmetric_base_rate"
        ],
        "claim_boundary": tournament.gate["claim_boundary"],
    }
    return ratings, meta, gate


def _require_pinned_team_rating_gate(
    generated_gate: dict[str, Any],
    validation_path: Path,
) -> None:
    """Block publication unless the pinned gate is this exact frozen run."""

    try:
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "Pinned model validation artifact is missing or unreadable."
        ) from exc
    pinned = validation.get("team_rating")
    if pinned != generated_gate:
        raise RuntimeError(
            "Pinned team-rating gate does not match the exact fresh-data "
            "tournament; regenerate and review model_validation_2026-07-27.json."
        )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _strict_json(value: Any) -> str:
    """Serialize public artifacts without NaN or implementation-only clocks."""

    def encode(item: Any) -> str:
        if isinstance(item, (datetime, pd.Timestamp)):
            return item.isoformat()
        raise TypeError(
            f"Unsupported public JSON value: {type(item).__name__}"
        )

    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
        default=encode,
    )


def require_pinned_model_files(
    models_root: Path = MODELS,
) -> dict[str, Path]:
    """Resolve every release-pinned model artifact or fail the pack build.

    A missing artifact must never silently change the public model surface.
    """

    proposed_paths = tuple(
        f"models/{name}" for name in spec.PINNED_MODEL_FILES
    )
    spec.require_publication_paths_allowed(proposed_paths)
    resolved = {
        name: models_root / name
        for name in spec.PINNED_MODEL_FILES
    }
    missing = [name for name, path in resolved.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "public pack is missing required pinned model artifact(s): "
            + ", ".join(sorted(missing))
        )
    return resolved


def _present(cols: Sequence[str], available: Iterable[str]) -> list[str]:
    avail = set(available)
    return [c for c in cols if c in avail]


def _source_summary(
    team: pd.DataFrame,
    players: pd.DataFrame,
    maps: pd.DataFrame,
    *,
    data_as_of: str | None,
    grid_completion_gate: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Return a sanitized public provenance ledger and truthful attribution."""

    def source_counts(
        frame: pd.DataFrame,
        *,
        game_column: str,
    ) -> dict[str, dict[str, int]]:
        if frame.empty or "source" not in frame.columns:
            return {}
        working = frame[[game_column, "source"]].copy()
        working["source"] = (
            working["source"].astype(str).str.strip().str.casefold()
        )
        result: dict[str, dict[str, int]] = {}
        for source, group in working.groupby("source", sort=True):
            if not source or source == "nan":
                continue
            result[source] = {
                "rows": int(len(group)),
                "maps": int(group[game_column].dropna().astype(str).nunique()),
            }
        return result

    team_game_column = "gameid" if "gameid" in team.columns else "game_uid"
    player_game_column = (
        "gameid" if "gameid" in players.columns else "game_uid"
    )
    canonical_map_sources: dict[str, dict[str, int]] = {}
    detail_sources: dict[str, dict[str, int]] = {}
    if not maps.empty:
        if "canonical_map_source" in maps.columns:
            values = maps["canonical_map_source"].fillna("").astype(str)
            for source, group in maps.groupby(values, sort=True):
                if not source:
                    continue
                canonical_map_sources[str(source)] = {
                    "rows": int(len(group)),
                    "maps": int(len(group)),
                }
        if "map_detail_source" in maps.columns:
            values = maps["map_detail_source"].fillna("").astype(str)
            for source, group in maps.groupby(values, sort=True):
                if not source:
                    continue
                detail_sources[str(source)] = {
                    "rows": int(len(group)),
                    "maps": int(len(group)),
                }
    team_sources = source_counts(team, game_column=team_game_column)
    player_sources = source_counts(players, game_column=player_game_column)
    grid_gap_fill_maps = canonical_map_sources.get(
        "grid_gap_fill", {}
    ).get("maps", 0)
    grid_detail_maps = sum(
        block.get("maps", 0)
        for source, block in detail_sources.items()
        if source.startswith("grid_")
    )
    if grid_gap_fill_maps and grid_detail_maps:
        attribution = spec.ATTRIBUTION_OE_GRID_GAP_AND_DETAIL
    elif grid_gap_fill_maps:
        attribution = spec.ATTRIBUTION_OE_GRID_GAP
    elif grid_detail_maps:
        attribution = spec.ATTRIBUTION_OE_GRID_DETAIL
    else:
        attribution = spec.ATTRIBUTION_OE_ONLY
    return (
        {
            "schema_version": 2,
            "data_as_of": data_as_of,
            "sources": {
                "team_games": team_sources,
                "player_games": player_sources,
                "canonical_map_inclusion": canonical_map_sources,
                "map_detail_enrichment": detail_sources,
            },
            "canonicalization": {
                "identity_grain": "canonical game identity",
                "overlap_precedence": "oracle_elixir_then_verified_grid_gap_fill",
                "canonical_inclusion_field": "canonical_map_source",
                "detail_enrichment_field": "map_detail_source",
                "completion_requirement": [
                    "events_game_end",
                    "end_state_summary",
                ],
                "source_fields_retained": [
                    "source",
                    "source_oe",
                    "source_grid",
                    "canonical_map_source",
                    "map_detail_source",
                    "grid_completion_source",
                ],
            },
            "completion_gate": grid_completion_gate,
            "attribution": attribution,
        },
        attribution,
    )


def _apply_map_provenance_contract(
    maps: pd.DataFrame,
    team_games: pd.DataFrame,
) -> pd.DataFrame:
    """Separate canonical map inclusion from optional detail enrichment.

    `maps.parquet` may retain GRID event detail for a game whose canonical
    result has since arrived in Oracle's Elixir.  Canonical origin therefore
    comes from the retained two-sided team population, while
    `map_detail_source` records which row supplied the wider fields.
    """

    if maps.empty:
        return maps.copy()
    if team_games.empty or "source" not in team_games.columns:
        raise RuntimeError(
            "map provenance requires the retained team-game source ledger"
        )
    team_id = "gameid" if "gameid" in team_games.columns else "game_uid"
    if team_id not in team_games.columns:
        raise RuntimeError("team-game provenance has no canonical game id")
    working = team_games[[team_id, "source"]].copy()
    working[team_id] = working[team_id].astype(str)
    working["source"] = (
        working["source"].fillna("").astype(str).str.strip().str.casefold()
    )

    def canonical_origin(values: pd.Series) -> str:
        sources = set(values)
        if "oe" in sources:
            return "oe"
        if "grid" in sources:
            return "grid_gap_fill"
        raise RuntimeError(
            f"unsupported canonical map source set: {sorted(sources)}"
        )

    origin = working.groupby(team_id, sort=False)["source"].agg(
        canonical_origin
    )
    out = maps.copy()
    if "oe_gameid" in out.columns:
        map_ids = out["oe_gameid"]
        if "game_uid" in out.columns:
            map_ids = map_ids.where(map_ids.notna(), out["game_uid"])
    elif "game_uid" in out.columns:
        map_ids = out["game_uid"]
    else:
        raise RuntimeError("public maps have no canonical game id")
    out["_canonical_map_id"] = map_ids.astype(str)
    out["canonical_map_source"] = out["_canonical_map_id"].map(origin)
    missing = out["canonical_map_source"].isna()
    if missing.any():
        examples = out.loc[missing, "_canonical_map_id"].head(10).tolist()
        raise RuntimeError(
            "public maps are absent from the retained canonical team ledger: "
            f"{examples}"
        )
    detail = out.get(
        "map_detail_source",
        pd.Series("", index=out.index),
    ).fillna("").astype(str)
    canonical_oe = out["canonical_map_source"].eq("oe")
    canonical_grid = out["canonical_map_source"].eq("grid_gap_fill")
    out["source_oe"] = canonical_oe | detail.str.startswith("oe_")
    out["source_grid"] = canonical_grid | detail.str.startswith("grid_")
    return out.drop(columns=["_canonical_map_id"])


def _ensure_columns(table: pa.Table, columns: Sequence[str]) -> pa.Table:
    """Materialize stable public columns even when an older warehouse omits them."""

    out = table
    for column in columns:
        if column not in out.column_names:
            out = out.append_column(column, pa.nulls(out.num_rows))
    return out


def _canonicalize_year(
    table: pa.Table,
    year_cols: Sequence[str],
) -> pa.Table:
    """Materialize one authoritative ``year`` using precedence order.

    OE's source year is authoritative when present. A transport/partition year
    is only a fallback. Keeping the canonical value in ``year`` ensures the
    filter, output partition, manifest contract, and public row all agree.
    """

    canonical_year = None
    for col in year_cols:
        if col not in table.column_names:
            continue
        arr = table[col]
        # year may be int or string
        try:
            as_int = pc.cast(arr, pa.int64(), safe=False)
        except Exception:
            as_int = pc.cast(pc.utf8_to_int(pc.cast(arr, pa.string())), pa.int64(), safe=False)
        canonical_year = (
            as_int
            if canonical_year is None
            else pc.coalesce(canonical_year, as_int)
        )
    if canonical_year is None:
        return table
    if "year" in table.column_names:
        return table.set_column(
            table.schema.get_field_index("year"),
            "year",
            canonical_year,
        )
    return table.append_column("year", canonical_year)


def _filter_years(table: pa.Table, years: Sequence[int], year_cols: Sequence[str]) -> pa.Table:
    """Filter and normalize by one canonical year in precedence order."""

    years_list = list(years)
    table = _canonicalize_year(table, year_cols)
    if "year" not in table.column_names:
        return table
    mask = pc.is_in(
        table["year"],
        value_set=pa.array(years_list, type=pa.int64()),
    )
    return table.filter(mask)


def _filter_unverified_grid_games(
    team: pa.Table,
) -> tuple[pa.Table, dict[str, Any]]:
    """Remove GRID-only games lacking verified completion provenance.

    The same retained identity set is subsequently applied to player rows and
    maps. A result value alone is not accepted as proof that a GRID file
    reached a verified completed end state.
    """

    if "source" not in team.column_names or "gameid" not in team.column_names:
        return team, {
            "grid_games_seen": 0,
            "grid_games_retained": 0,
            "grid_games_excluded_unverified": 0,
            "excluded_examples": [],
        }
    frame = team.to_pandas()
    grid = frame["source"].fillna("").astype(str).str.casefold().eq("grid")
    if not grid.any():
        return team, {
            "grid_games_seen": 0,
            "grid_games_retained": 0,
            "grid_games_excluded_unverified": 0,
            "excluded_examples": [],
        }
    provenance_columns = [
        column
        for column in (
            "grid_completion_source",
            "completion_source",
            "series_completion_source",
        )
        if column in frame.columns
    ]
    provenance = pd.Series("", index=frame.index, dtype=object)
    for column in provenance_columns:
        values = frame[column].fillna("").astype(str).str.strip()
        provenance = provenance.where(provenance.ne(""), values)
    frame["_verified_grid_completion"] = provenance.isin(
        VERIFIED_GRID_COMPLETION_SOURCES
    )
    grid_games = frame.loc[grid, "gameid"].astype(str)
    valid_by_game = (
        frame.loc[grid]
        .assign(_gameid=grid_games)
        .groupby("_gameid", sort=False)["_verified_grid_completion"]
        .all()
    )
    retained_grid_ids = set(valid_by_game[valid_by_game].index)
    excluded_grid_ids = sorted(
        set(valid_by_game.index) - retained_grid_ids
    )
    keep = ~grid | frame["gameid"].astype(str).isin(retained_grid_ids)
    filtered = frame.loc[keep].drop(
        columns=["_verified_grid_completion"]
    )
    return pa.Table.from_pandas(filtered, preserve_index=False), {
        "grid_games_seen": int(valid_by_game.size),
        "grid_games_retained": int(valid_by_game.sum()),
        "grid_games_excluded_unverified": int(
            len(excluded_grid_ids)
        ),
        "excluded_examples": excluded_grid_ids[:10],
        "accepted_completion_sources": sorted(
            VERIFIED_GRID_COMPLETION_SOURCES
        ),
    }


def _filter_to_game_ids(
    table: pa.Table,
    game_ids: set[str],
    *,
    column: str = "gameid",
) -> pa.Table:
    if column not in table.column_names:
        return table
    values = pc.cast(table[column], pa.string())
    return table.filter(
        pc.is_in(
            values,
            value_set=pa.array(sorted(game_ids), type=pa.string()),
        )
    )


def _write_parquet(table: pa.Table, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        table,
        path,
        compression="zstd",
        compression_level=6,
        use_dictionary=True,
    )
    return {
        "path": str(path.name) if path.parent == path.parent else path.as_posix(),
        "relative": None,  # filled by caller
        "rows": table.num_rows,
        "cols": table.num_columns,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "columns": table.column_names,
    }


def _join_history_years(
    history: pa.Table,
    maps: pa.Table,
    years: Sequence[int],
) -> pa.Table:
    """Keep history rows whose game_uid appears in year-filtered maps."""
    if "game_uid" not in history.column_names or "game_uid" not in maps.column_names:
        return history
    uids = maps.column("game_uid").unique()
    return history.filter(pc.is_in(history["game_uid"], value_set=uids))


def _draft_coverage(maps: pd.DataFrame, players: pd.DataFrame) -> dict[str, Any]:
    """Require every public map to have a verified ten-champion composition.

    OE's map-level pick columns preserve draft order, while GRID and a few OE
    anomalies only carry the final champion lineup on participant rows. Both
    are valid composition sources, but participant fallbacks must contain one
    champion in each canonical role on both sides.
    """

    roles = ("top", "jng", "mid", "bot", "sup")

    def known(value: Any) -> bool:
        if pd.isna(value):
            return False
        return str(value).strip().lower() not in {"", "unknown", "nan", "none", "null"}

    def role(value: Any) -> str | None:
        raw = str(value or "").strip().lower()
        if raw == "top" or raw.startswith("top"):
            return "top"
        if raw in {"jng", "jungle"} or raw.startswith("jungler"):
            return "jng"
        if raw == "mid" or raw.startswith("middle"):
            return "mid"
        if raw in {"bot", "adc"} or raw.startswith("bottom"):
            return "bot"
        if raw in {"sup", "utility"} or raw.startswith("support"):
            return "sup"
        return None

    if maps.empty:
        return {
            "maps": 0,
            "map_pick_rows": 0,
            "participant_fallback_rows": 0,
            "complete_rows": 0,
            "coverage_rate": 1.0,
        }

    pick_columns = [
        f"{side}_pick{index}"
        for side in ("blue", "red")
        for index in range(1, 6)
    ]
    map_pick_complete = pd.Series(False, index=maps.index)
    if all(column in maps.columns for column in pick_columns):
        map_pick_complete = maps[pick_columns].map(known).sum(axis=1).eq(10)

    player_key = "gameid" if "gameid" in players.columns else "game_uid"
    map_keys = (
        maps.get("oe_gameid", pd.Series(index=maps.index, dtype=object))
        .where(lambda values: values.map(known), maps.get("game_uid"))
        .astype(str)
    )
    complete_participant_games: set[str] = set()
    required_player_columns = {player_key, "side", "position", "champion"}
    if not players.empty and required_player_columns.issubset(players.columns):
        relevant = players[
            players["champion"].map(known)
            & players["side"].astype(str).str.title().isin(["Blue", "Red"])
        ].copy()
        relevant["_side"] = relevant["side"].astype(str).str.title()
        relevant["_role"] = relevant["position"].map(role)
        relevant = relevant[relevant["_role"].notna()]
        side_summary = (
            relevant.groupby([player_key, "_side"], sort=False)
            .agg(
                rows=("champion", "size"),
                roles=("_role", "nunique"),
                champions=("champion", "nunique"),
            )
            .reset_index()
        )
        complete_sides = side_summary[
            side_summary["rows"].eq(5)
            & side_summary["roles"].eq(len(roles))
            & side_summary["champions"].eq(5)
        ]
        complete_games = complete_sides.groupby(player_key, sort=False)["_side"].agg(
            lambda sides: set(sides) == {"Blue", "Red"}
        )
        complete_participant_games = {
            str(game_id) for game_id, valid in complete_games.items() if valid
        }

    participant_complete = map_keys.isin(complete_participant_games)
    complete = map_pick_complete | participant_complete
    if not complete.all():
        missing = map_keys[~complete].head(10).tolist()
        raise ValueError(
            "Public pack draft coverage failed: "
            f"{int((~complete).sum())} map(s) have neither ten map picks nor "
            f"five role-aligned participant champions per side; examples={missing}"
        )

    fallback = ~map_pick_complete & participant_complete
    return {
        "maps": int(len(maps)),
        "map_pick_rows": int(map_pick_complete.sum()),
        "participant_fallback_rows": int(fallback.sum()),
        "complete_rows": int(complete.sum()),
        "coverage_rate": float(complete.mean()),
    }


def export_public_pack(
    *,
    years: Sequence[int] | None = None,
    out_root: Path | None = None,
    pack_id: str | None = None,
    warehouse_root: Path | None = None,
) -> dict[str, Any]:
    years = tuple(years or spec.DEFAULT_YEARS)
    # Include UTC time so the 15-minute freshness workflow can publish more
    # than one immutable pack per day without colliding in Blob storage.
    stamp = datetime.now(timezone.utc).strftime("%Y.%m.%d.%H%M")
    pack_id = validate_pack_id(f"v{stamp}" if pack_id is None else pack_id)
    out_root = Path(out_root or DEFAULT_OUT)
    warehouse_root = Path(warehouse_root or WAREHOUSE)
    pack_dir = out_root / pack_id
    if pack_dir.exists():
        shutil.rmtree(pack_dir)
    pack_dir.mkdir(parents=True)

    files_meta: list[dict[str, Any]] = []

    def register(meta: dict[str, Any], rel: str) -> None:
        spec.require_publication_paths_allowed([rel])
        meta = dict(meta)
        meta["relative"] = rel
        meta["path"] = rel
        files_meta.append(meta)

    # --- team games (partition by year) ---
    team_path = warehouse_root / "oe_team_games.parquet"
    team = pq.read_table(team_path)
    team = pa.Table.from_pandas(canonicalize_competition_frame(team.to_pandas()), preserve_index=False)
    team_cols = _present(spec.TEAM_COLS, team.column_names)
    team = team.select(team_cols)
    team = _filter_years(team, years, ("oe_year", "year"))
    team, grid_completion_gate = _filter_unverified_grid_games(team)
    for y in years:
        part = team.filter(pc.equal(team["year"], y))
        dest = pack_dir / "team_games" / f"year={y}" / "part.parquet"
        if part.num_rows == 0:
            continue
        register(_write_parquet(part, dest), f"team_games/year={y}/part.parquet")
    team_for_records = team.to_pandas()
    team_maps_for_ratings = build_maps_frame_from_team_games(team_for_records)

    # --- player games ---
    player_path = warehouse_root / "oe_player_games.parquet"
    player = pq.read_table(player_path)
    player = pa.Table.from_pandas(canonicalize_competition_frame(player.to_pandas()), preserve_index=False)
    player_cols = _present(spec.PLAYER_COLS, player.column_names)
    player = player.select(player_cols)
    player = _filter_years(player, years, ("oe_year", "year"))
    retained_game_ids = set(
        team["gameid"].to_pandas().dropna().astype(str)
    )
    player = _filter_to_game_ids(player, retained_game_ids)
    player_frame = player.to_pandas()
    for y in years:
        part = player.filter(pc.equal(player["year"], y))
        dest = pack_dir / "player_games" / f"year={y}" / "part.parquet"
        if part.num_rows == 0:
            continue
        register(_write_parquet(part, dest), f"player_games/year={y}/part.parquet")

    # --- maps ---
    maps_path = warehouse_root / "maps.parquet"
    maps = pq.read_table(maps_path)
    # Re-apply the canonical map contract at export time as a safety net for
    # packs built from an older local warehouse refresh.
    maps = pa.Table.from_pandas(canonicalize_competition_frame(maps.to_pandas()), preserve_index=False)
    maps = _ensure_columns(maps, spec.maps_columns())
    map_cols = spec.maps_columns(maps.column_names)
    maps = maps.select(map_cols)
    maps = _filter_years(maps, years, ("oe_year", "year"))
    map_identity_column = (
        "oe_gameid"
        if "oe_gameid" in maps.column_names
        else "game_uid"
    )
    maps = _filter_to_game_ids(
        maps,
        retained_game_ids,
        column=map_identity_column,
    )
    maps_for_records, map_population_coverage = complete_public_map_population(
        maps.to_pandas(),
        team_maps_for_ratings,
    )
    maps_for_records = _apply_map_provenance_contract(
        maps_for_records,
        team_for_records,
    )
    draft_coverage = _draft_coverage(maps_for_records, player_frame)
    # The feature-oriented maps table intentionally covers the major/public
    # event slice.  Team ladders need the full OE team-game population so
    # Tier 2 and Tier 3 organizations receive both records and estimates.
    rating_input = team_maps_for_ratings if not team_maps_for_ratings.empty else maps_for_records
    schedule = load_schedule()
    rating_schedule_annotation = annotate_scheduled_series(
        rating_input,
        schedule,
    )
    rating_input = rating_schedule_annotation.rows
    map_schedule_annotation = annotate_scheduled_series(
        maps_for_records,
        schedule,
    )
    maps_for_records = map_schedule_annotation.rows
    membership_checked_at = datetime.now(timezone.utc).isoformat()
    tournament_registry = load_tournament_registry()
    rating_format_annotation = annotate_maps_with_tournament_registry(
        rating_input,
        tournament_registry,
        as_of=membership_checked_at,
    )
    rating_input = rating_format_annotation.maps
    map_format_annotation = annotate_maps_with_tournament_registry(
        maps_for_records,
        tournament_registry,
        as_of=membership_checked_at,
    )
    public_series = build_canonical_series_ledger(
        map_format_annotation.maps
    )
    if not public_series.audit.get("ok"):
        raise RuntimeError(
            "Public pack has no format-verified completed series; team ratings "
            "and series surfaces cannot be published."
        )
    maps_for_records = public_series.maps
    maps = pa.Table.from_pandas(maps_for_records, preserve_index=False)
    maps = _ensure_columns(maps, spec.maps_columns())
    maps = maps.select(spec.maps_columns(maps.column_names))
    (
        public_ratings,
        public_ratings_meta,
        public_team_rating_gate,
    ) = _build_public_series_ratings(rating_input)
    _require_pinned_team_rating_gate(
        public_team_rating_gate,
        MODELS / "model_validation_2026-07-27.json",
    )
    if public_ratings.empty or int(public_ratings_meta.get("n_series") or 0) <= 0:
        raise RuntimeError(
            "Series Dynamic Bradley-Terry release is empty; publication is blocked."
        )
    public_ratings_meta["pack_years"] = list(years)
    public_ratings_meta["rating_window"] = "full canonical OE team-game window as this pack"
    public_dual_history = build_dual_ratings(
        rating_input,
        lineup_by_game=lineup_hashes_from_players(player_frame),
        write=False,
    )
    player_maps = build_maps_frame_from_players(player_frame)
    (
        public_player_history,
        public_player_ratings,
        public_player_ratings_meta,
    ) = build_player_rating_artifacts(player_maps, player_frame)
    for y in years:
        part = maps.filter(pc.equal(maps["year"], y))
        dest = pack_dir / "maps" / f"year={y}" / "part.parquet"
        if part.num_rows == 0:
            continue
        register(_write_parquet(part, dest), f"maps/year={y}/part.parquet")

    # --- features snapshots (team ladder uses the same pack window; player
    # snapshot remains the full roster-history artifact) ---
    feat_dir = pack_dir / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    data_dates = pd.concat(
        [
            pd.to_datetime(rating_input.get("date"), errors="coerce", utc=True)
            if "date" in rating_input.columns
            else pd.Series(dtype="datetime64[ns, UTC]"),
            pd.to_datetime(maps_for_records.get("date"), errors="coerce", utc=True)
            if "date" in maps_for_records.columns
            else pd.Series(dtype="datetime64[ns, UTC]"),
        ],
        ignore_index=True,
    )
    data_as_of = data_dates.max().isoformat() if not data_dates.dropna().empty else None
    current_membership = build_current_tournament_membership(
        maps_for_records,
        as_of=membership_checked_at,
        window_days=90,
        registry=tournament_registry,
    )
    player_records_payload = build_player_records(player_frame, current_membership)

    # Weekly rank movement is a ranking claim, not neutral metadata.  The
    # player outcome model explicitly fails the individual-skill/ordering gate,
    # so no rank or rank-delta artifact is generated.
    if (
        public_player_ratings_meta.get("outcome_ordering_verified") is True
        and public_player_ratings_meta.get("individual_skill_estimand") is True
    ):
        raise RuntimeError(
            "player rank export requires a separately implemented and tested "
            "individual-ordering artifact"
        )
    public_player_ratings_meta["weekly_rank_artifact"] = {
        "status": "withheld",
        "reason": (
            "the published outcome signal does not identify individual skill "
            "or support individual ordering"
        ),
    }

    player_metadata = build_player_metadata(
        player_frame["playername"].dropna().astype(str).unique(),
        player_context={
            player: record.get("current_team")
            for player, record in player_records_payload.items()
        },
        player_identities=player_frame,
    )
    metadata_dest = feat_dir / "player_metadata.json"
    metadata_dest.write_text(json.dumps(player_metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    register(
        {
            "rows": len(player_metadata),
            "cols": None,
            "bytes": metadata_dest.stat().st_size,
            "sha256": _sha256(metadata_dest),
            "columns": None,
        },
        "features/player_metadata.json",
    )

    # These records are intentionally built from the same year-filtered
    # canonical rows that are exported above.  This avoids mixing a full
    # history snapshot with a different pack-window win rate.
    for filename, payload in (
        (
            "team_records.json",
            build_team_records(
                rating_input,
                current_membership,
                tournament_maps=maps_for_records,
            ),
        ),
        ("player_records.json", player_records_payload),
    ):
        dest = feat_dir / filename
        dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        register(
            {
                "rows": len(payload),
                "cols": None,
                "bytes": dest.stat().st_size,
                "sha256": _sha256(dest),
                "columns": None,
            },
            f"features/{filename}",
        )

    # --- narrow descriptive player-performance model ---
    # This is intentionally not Player Dual Elo and is never substituted for
    # it.  The governed candidate is valid only for the locked 2025-2026,
    # complete-OE source contract.  Custom-window packs omit it explicitly.
    player_performance_quality: dict[str, Any]
    if tuple(sorted(years)) == REQUIRED_PUBLIC_YEARS:
        performance_artifacts = build_player_performance_public_artifacts(
            player_frame,
            years=years,
        )
        performance_table = pa.Table.from_pandas(
            performance_artifacts.snapshot,
            preserve_index=False,
        ).select(spec.PLAYER_PERFORMANCE_SNAPSHOT_COLS)
        performance_parquet_dest = (
            feat_dir / "player_performance_snapshot.parquet"
        )
        register(
            _write_parquet(performance_table, performance_parquet_dest),
            "features/player_performance_snapshot.parquet",
        )
        performance_json_dest = feat_dir / "player_performance_snapshot.json"
        performance_json_dest.write_text(
            json.dumps(
                performance_table.to_pylist(),
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        register(
            {
                "rows": performance_table.num_rows,
                "cols": performance_table.num_columns,
                "bytes": performance_json_dest.stat().st_size,
                "sha256": _sha256(performance_json_dest),
                "columns": performance_table.column_names,
            },
            "features/player_performance_snapshot.json",
        )
        for filename, payload in (
            ("player_performance_meta.json", performance_artifacts.meta),
            (
                "player_performance_validation.json",
                performance_artifacts.validation,
            ),
        ):
            dest = feat_dir / filename
            dest.write_text(
                json.dumps(
                    payload,
                    indent=2,
                    ensure_ascii=False,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
            register(
                {
                    "rows": None,
                    "cols": None,
                    "bytes": dest.stat().st_size,
                    "sha256": _sha256(dest),
                    "columns": None,
                },
                f"features/{filename}",
            )
        player_performance_quality = {
            "status": "published_validated_narrow_descriptive_view",
            "model_id": performance_artifacts.meta["model_id"],
            "model_hash": performance_artifacts.meta["model_hash"],
            "fit_through": performance_artifacts.meta["fit_through"],
            "test_gate_passed": performance_artifacts.validation[
                "test_gate_passed"
            ],
            "large_prediction_ledger_exported": False,
        }
    else:
        player_performance_quality = {
            "status": "omitted_incompatible_year_window",
            "required_years": list(REQUIRED_PUBLIC_YEARS),
            "requested_years": list(years),
        }

    for src_name, cols, out_name in (
        ("ratings_snapshot.parquet", spec.RATINGS_SNAPSHOT_COLS, "ratings_snapshot.parquet"),
        (
            "player_ratings_snapshot.parquet",
            spec.PLAYER_RATINGS_SNAPSHOT_COLS,
            "player_ratings_snapshot.parquet",
        ),
    ):
        if src_name == "ratings_snapshot.parquet":
            t = pa.Table.from_pandas(public_ratings, preserve_index=False)
        elif src_name == "player_ratings_snapshot.parquet":
            t = pa.Table.from_pandas(
                public_player_ratings, preserve_index=False
            )
        else:
            continue
        t = t.select(_present(cols, t.column_names))
        dest = feat_dir / out_name
        register(_write_parquet(t, dest), f"features/{out_name}")
        # JSON twin for fast atlas ladders (no WASM)
        if out_name.endswith("_snapshot.parquet"):
            jdest = feat_dir / out_name.replace(".parquet", ".json")
            rows = t.to_pylist()
            jdest.write_text(_strict_json(rows), encoding="utf-8")
            register(
                {
                    "rows": len(rows),
                    "cols": t.num_columns,
                    "bytes": jdest.stat().st_size,
                    "sha256": _sha256(jdest),
                    "columns": t.column_names,
                },
                f"features/{jdest.name}",
            )

    # --- features history year-filtered via maps game_uid ---
    # Rating history may only reference the exact canonical map identities
    # published above. Re-reading the feature-oriented warehouse subset here
    # would silently reintroduce excluded or population-mismatched games.
    maps_all = maps

    for src_name, cols, out_name in (
        ("ratings.parquet", spec.RATINGS_HISTORY_COLS, "ratings_history.parquet"),
        (
            "player_ratings.parquet",
            spec.PLAYER_RATINGS_HISTORY_COLS,
            "player_ratings_history.parquet",
        ),
    ):
        if src_name == "ratings.parquet":
            t = pa.Table.from_pandas(
                public_dual_history, preserve_index=False
            )
        elif src_name == "player_ratings.parquet":
            t = pa.Table.from_pandas(
                public_player_history, preserve_index=False
            )
        else:
            continue
        t = t.select(_present(cols, t.column_names))
        t = _join_history_years(t, maps_all, years)
        dest = feat_dir / out_name
        register(_write_parquet(t, dest), f"features/{out_name}")

    for meta_name in ("ratings_meta.json", "player_ratings_meta.json"):
        if meta_name == "ratings_meta.json":
            dest = feat_dir / meta_name
            dest.write_text(json.dumps(public_ratings_meta, indent=2), encoding="utf-8")
            register(
                {
                    "rows": None,
                    "cols": None,
                    "bytes": dest.stat().st_size,
                    "sha256": _sha256(dest),
                    "columns": None,
                },
                f"features/{meta_name}",
            )
            continue
        if meta_name == "player_ratings_meta.json":
            dest = feat_dir / meta_name
            dest.write_text(
                json.dumps(public_player_ratings_meta, indent=2),
                encoding="utf-8",
            )
            register(
                {
                    "rows": None,
                    "cols": None,
                    "bytes": dest.stat().st_size,
                    "sha256": _sha256(dest),
                    "columns": None,
                },
                f"features/{meta_name}",
            )

    # --- models ---
    models_dir = pack_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    pinned_models = require_pinned_model_files()
    for name in spec.PINNED_MODEL_FILES:
        src = pinned_models[name]
        dest = models_dir / name
        shutil.copy2(src, dest)
        register(
            {
                "rows": None,
                "cols": None,
                "bytes": dest.stat().st_size,
                "sha256": _sha256(dest),
                "columns": None,
            },
            f"models/{name}",
        )

    # --- void grubs study bundle ---
    pdf_root = ROOT / "output" / "pdf"
    grubs_dir = pack_dir / "studies" / "grubs"
    grubs_dir.mkdir(parents=True, exist_ok=True)
    for name in spec.GRUBS_MODEL_FILES:
        src = MODELS / name
        if not src.is_file():
            raise FileNotFoundError(
                f"public pack is missing required grubs article artifact: {name}"
            )
        dest = grubs_dir / name
        shutil.copy2(src, dest)
        register(
            {
                "rows": None,
                "cols": None,
                "bytes": dest.stat().st_size,
                "sha256": _sha256(dest),
                "columns": None,
            },
            f"studies/grubs/{name}",
        )
    for name in spec.GRUBS_PDF_FILES:
        src = pdf_root / name
        if not src.is_file():
            raise FileNotFoundError(
                f"public pack is missing required grubs PDF artifact: {name}"
            )
        dest = grubs_dir / name
        shutil.copy2(src, dest)
        register(
            {
                "rows": None,
                "cols": None,
                "bytes": dest.stat().st_size,
                "sha256": _sha256(dest),
                "columns": None,
            },
            f"studies/grubs/{name}",
        )

    # --- meta ---
    meta_dir = pack_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    if TEAMS_JSON.exists():
        dest = meta_dir / "teams.json"
        shutil.copy2(TEAMS_JSON, dest)
        register(
            {
                "rows": None,
                "cols": None,
                "bytes": dest.stat().st_size,
                "sha256": _sha256(dest),
                "columns": None,
            },
            "meta/teams.json",
        )

    source_summary, attribution = _source_summary(
        team_for_records,
        player_frame,
        maps_for_records,
        data_as_of=data_as_of,
        grid_completion_gate=grid_completion_gate,
    )
    source_summary_path = meta_dir / "source_summary.json"
    source_summary_path.write_text(
        json.dumps(source_summary, indent=2),
        encoding="utf-8",
    )
    register(
        {
            "rows": None,
            "cols": None,
            "bytes": source_summary_path.stat().st_size,
            "sha256": _sha256(source_summary_path),
            "columns": None,
        },
        "meta/source_summary.json",
    )

    readme = pack_dir / "README.md"
    readme.write_text(
        spec.PACK_README.format(
            years=", ".join(str(y) for y in years),
            attribution=attribution,
        ),
        encoding="utf-8",
    )

    total_bytes = sum(f["bytes"] for f in files_meta)
    manifest: dict[str, Any] = {
        "pack_id": pack_id,
        # Model and validation files are copied into this same immutable
        # directory.  Declaring the shared bundle ID lets every consumer prove
        # that data, ratings, calibration, and model evidence use one release
        # clock instead of merely assuming it from relative paths.
        "model_pack_id": pack_id,
        "schema_version": spec.SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_as_of": data_as_of,
        "recent_activity_window_days": 90,
        "current_tournament_as_of": current_membership.get("as_of"),
        "current_tournaments": current_membership.get("leagues", {}),
        "membership_registry": {
            "snapshot_id": current_membership.get("registry_snapshot_id"),
            "authority": current_membership.get("authority"),
            "checked_at": current_membership.get("checked_at"),
            "review_due_at": current_membership.get("review_due_at"),
            "sources": current_membership.get("sources", {}),
            "participants_by_league": current_membership.get(
                "participants_by_league", {}
            ),
            "observation_audit": current_membership.get(
                "observation_audit", {}
            ),
        },
        "membership_note": (
            "Current league membership is the reviewed participant list for "
            "each current Riot Tier 1 regional tournament. Match appearances "
            "are reconciliation evidence only and cannot create membership. "
            "Historical rows and historical affiliations remain available."
        ),
        "filters": {
            "years": list(years),
            "leagues": "all_in_year_window",
            "leagues_note": spec.DEFAULT_LEAGUES_NOTE,
        },
        "identity": {
            "taxonomy_version": TAXONOMY_VERSION,
            "team_key": "one canonical organization identity across regional and international events",
            "league_source": "raw source label retained on rows for auditability",
            "competition_tier": "tier1/tier2/tier3 assigned from the canonical league taxonomy; international and interregional are separate scopes",
            "deprecated_leagues": {
                "LTA": "AMERICAS",
                "LTA N": "LCS",
                "LTA NORTH": "LCS",
                "LTA S": "CBLOL",
                "LTA SOUTH": "CBLOL",
            },
        },
        "attribution": attribution,
        "source_summary": source_summary,
        "quality": {
            "draft_coverage": draft_coverage,
            "grid_completion_gate": grid_completion_gate,
            "map_population_coverage": map_population_coverage,
            "rating_series_format_annotation": rating_format_annotation.audit,
            "rating_series_schedule_annotation": rating_schedule_annotation.audit,
            "public_map_schedule_annotation": map_schedule_annotation.audit,
            "public_map_series_ledger": public_series.audit,
            "player_performance": player_performance_quality,
        },
        "excluded": [
            "warehouse/timelines",
            "warehouse/raw OE CSVs",
            "champion tierlists and model CSVs pending validated replacement",
            "private odds / prediction tooling",
            "joblib models",
        ],
        "studies": {
            "grubs": {
                "path": "studies/grubs/",
                "note": spec.GRUBS_STUDY_NOTE,
                "entrypoints": [
                    f"studies/grubs/{name}"
                    for name in (
                        *spec.GRUBS_MODEL_FILES,
                        *spec.GRUBS_PDF_FILES,
                    )
                ],
            }
        },
        "base_url": None,  # filled by upload / atlas config
        "total_bytes": total_bytes,
        "total_files": len(files_meta),
        "files": files_meta,
    }
    man_path = pack_dir / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # also write latest pointer
    latest = out_root / "latest.json"
    latest.write_text(
        json.dumps({"pack_id": pack_id, "path": pack_id, "created_utc": manifest["created_utc"]}, indent=2),
        encoding="utf-8",
    )

    # atlas-facing copy of manifest (for static deploy before Blob upload)
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Export public LoL research reproduction pack")
    ap.add_argument("--years", default="2025,2026", help="Comma-separated years (default 2025,2026)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output root directory")
    ap.add_argument("--pack-id", default=None, help="Override pack id (default vYYYY.MM.DD)")
    ap.add_argument(
        "--warehouse",
        type=Path,
        default=WAREHOUSE,
        help="Parquet warehouse root (default data/lol/warehouse/parquet)",
    )
    args = ap.parse_args(argv)
    years = tuple(int(x.strip()) for x in args.years.split(",") if x.strip())
    man = export_public_pack(
        years=years,
        out_root=args.out,
        pack_id=args.pack_id,
        warehouse_root=args.warehouse,
    )
    mb = man["total_bytes"] / (1024 * 1024)
    print(f"Wrote pack {man['pack_id']} → {args.out / man['pack_id']}")
    print(f"Files: {man['total_files']}  Size: {mb:.1f} MB  schema={man['schema_version']}")
    for f in man["files"]:
        print(f"  {f['path']}: {f['bytes']/1024:.0f} KB  rows={f.get('rows')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
