"""Governed public artifacts for the narrow player-performance model.

The public snapshot is intentionally separate from Player Dual Elo.  It
publishes a descriptive, role-specific early-resource estimand and the compact
chronological validation evidence needed to interpret it.  The large
player-map prediction ledger remains internal.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import pandas as pd

from lol_kills.ratings.player_performance import (
    CANONICAL_ROLES,
    ESTIMAND,
    NON_ESTIMANDS,
    RESEARCH_ANCHORS,
    PlayerPerformanceConfig,
    PlayerPerformanceDataError,
    PlayerPerformanceTournament,
    run_player_performance_tournament,
)


PUBLIC_ARTIFACT_SCHEMA_VERSION = "1.0.0"
PUBLIC_MODEL_FAMILY = "role_relative_15_minute_resource_performance"
PUBLIC_DISPLAY_NAME = "15-minute resource performance"
PUBLICATION_STATUS = "validated_narrow_descriptive_view"
REQUIRED_PUBLIC_YEARS: tuple[int, int] = (2025, 2026)
REQUIRED_PUBLIC_BOOTSTRAP_REPLICATES = 5_000


@dataclass(frozen=True)
class PlayerPerformancePublicArtifacts:
    """Small public outputs; the tournament prediction ledger is not exported."""

    snapshot: pd.DataFrame
    meta: dict[str, Any]
    validation: dict[str, Any]


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _metrics(value: Any) -> dict[str, Any]:
    return _json_safe(asdict(value))


def select_canonical_player_performance_rows(
    player_rows: pd.DataFrame,
    years: Sequence[int],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Select exactly canonical complete OE player rows in the locked window."""

    normalized_years = tuple(sorted({int(year) for year in years}))
    if normalized_years != REQUIRED_PUBLIC_YEARS:
        raise PlayerPerformanceDataError(
            "the public player-performance artifact is locked to canonical "
            "2025-2026 rows"
        )
    required = {"source", "datacompleteness", "year", "date", "playerid"}
    missing = sorted(required - set(player_rows.columns))
    if missing:
        raise PlayerPerformanceDataError(
            "public player-performance input is missing columns: "
            + ", ".join(missing)
        )

    working = player_rows.copy()
    canonical_year = pd.to_numeric(working["year"], errors="coerce")
    if "oe_year" in working.columns:
        oe_year = pd.to_numeric(working["oe_year"], errors="coerce")
        conflict = oe_year.notna() & canonical_year.notna() & oe_year.ne(canonical_year)
        if conflict.any():
            raise PlayerPerformanceDataError(
                f"{int(conflict.sum())} player rows disagree on oe_year and year"
            )
        canonical_year = oe_year.fillna(canonical_year)
    working["year"] = canonical_year

    year_mask = working["year"].isin(REQUIRED_PUBLIC_YEARS)
    oe_mask = (
        working["source"].astype("string").fillna("").str.strip().str.casefold().eq("oe")
    )
    complete_mask = (
        working["datacompleteness"]
        .astype("string")
        .fillna("")
        .str.strip()
        .str.casefold()
        .eq("complete")
    )
    selected = working.loc[year_mask & oe_mask & complete_mask].copy()
    selected["year"] = selected["year"].astype(int)
    selected = selected.sort_values(
        ["date", "gameid", "side", "position"], kind="mergesort"
    ).reset_index(drop=True)
    if selected.empty:
        raise PlayerPerformanceDataError(
            "no canonical complete OE player rows remain for 2025-2026"
        )
    audit = {
        "input_rows": int(len(working)),
        "selected_complete_oe_rows": int(len(selected)),
        "outside_year_window_rows": int((~year_mask).sum()),
        "non_oe_rows": int((year_mask & ~oe_mask).sum()),
        "incomplete_rows": int((year_mask & oe_mask & ~complete_mask).sum()),
    }
    return selected, audit


def _fit_period_context(
    selected_rows: pd.DataFrame,
    fit_through: str,
) -> pd.DataFrame:
    cutoff = pd.Timestamp(fit_through)
    dates = pd.to_datetime(selected_rows["date"], errors="coerce", utc=True)
    fit_rows = selected_rows.loc[dates.le(cutoff)].copy()
    fit_rows["_date"] = dates.loc[fit_rows.index]
    fit_rows["_player_id"] = (
        fit_rows["playerid"].astype("string").fillna("").str.strip()
    )
    fit_rows["_role"] = (
        fit_rows["position"].astype("string").fillna("").str.strip().str.casefold()
    )
    fit_rows = fit_rows[
        fit_rows["_player_id"].ne("")
        & fit_rows["_role"].isin(CANONICAL_ROLES)
        & fit_rows["_date"].notna()
    ]
    latest = (
        fit_rows.sort_values(["_date", "gameid"], kind="mergesort")
        .groupby(["_player_id", "_role"], sort=True)
        .tail(1)
    )
    return latest.set_index(["_player_id", "_role"])


def render_player_performance_public_artifacts(
    tournament: PlayerPerformanceTournament,
    selected_rows: pd.DataFrame,
    source_selection_audit: Mapping[str, int],
    *,
    years: Sequence[int],
    config: PlayerPerformanceConfig,
) -> PlayerPerformancePublicArtifacts:
    """Render a validated tournament into immutable compact public artifacts."""

    if not tournament.audit.ready:
        raise PlayerPerformanceDataError(
            "player-performance input audit did not pass"
        )
    if not tournament.test_gate_passed:
        raise PlayerPerformanceDataError(
            "player-performance chronological test gate did not pass"
        )
    if (
        config.metric_bootstrap_replicates
        != REQUIRED_PUBLIC_BOOTSTRAP_REPLICATES
        or tournament.player_incremental_test_contrast.bootstrap_replicates
        != REQUIRED_PUBLIC_BOOTSTRAP_REPLICATES
    ):
        raise PlayerPerformanceDataError(
            "public player-performance uncertainty requires exactly 5,000 "
            "calendar-day bootstrap replicates"
        )
    if (
        tournament.player_incremental_test_contrast.resampling_unit
        != "calendar_day"
    ):
        raise PlayerPerformanceDataError(
            "public player-performance uncertainty must use calendar-day blocks"
        )
    if tournament.player_ratings.empty:
        raise PlayerPerformanceDataError(
            "player-performance tournament produced no public ratings"
        )

    fit_through = tournament.split_boundaries["validation_end"]
    context = _fit_period_context(selected_rows, fit_through)
    snapshot = tournament.player_ratings.copy()
    context_leagues: list[str | None] = []
    context_dates: list[str | None] = []
    for row in snapshot.itertuples(index=False):
        key = (str(row.player_id), str(row.role))
        if key not in context.index:
            context_leagues.append(None)
            context_dates.append(None)
            continue
        value = context.loc[key]
        if isinstance(value, pd.DataFrame):
            raise PlayerPerformanceDataError(
                f"non-unique fit-period player-role context for {key}"
            )
        league = str(value.get("league", "")).strip()
        context_leagues.append(league or None)
        context_dates.append(pd.Timestamp(value["_date"]).isoformat())

    snapshot = snapshot.rename(
        columns={
            "maps": "effective_sample_maps",
            "conservative_performance": "lower_bound",
        }
    )
    snapshot["last_observed_league"] = context_leagues
    snapshot["last_observed_date"] = context_dates
    snapshot["fit_through"] = str(fit_through)
    snapshot["rank"] = (
        snapshot.groupby("role", sort=False)["lower_bound"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    snapshot["publication_status"] = PUBLICATION_STATUS
    snapshot = snapshot[
        [
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
        ]
    ].sort_values(
        ["role", "rank", "performance_mean", "player_id"],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    uncertainty_methods = sorted(
        snapshot["uncertainty_method"].dropna().astype(str).unique()
    )
    model_fingerprint_payload = {
        "artifact_schema_version": PUBLIC_ARTIFACT_SCHEMA_VERSION,
        "model_family": PUBLIC_MODEL_FAMILY,
        "estimand": ESTIMAND,
        "non_estimands": NON_ESTIMANDS,
        "years": list(years),
        "split_boundaries": dict(tournament.split_boundaries),
        "selected_base_penalty": tournament.selected_base_penalty,
        "selected_context_base_penalty": tournament.selected_context_base_penalty,
        "config": asdict(config),
        "snapshot": snapshot.to_dict(orient="records"),
        "test_metrics": _metrics(tournament.test_metrics),
        "test_context_baseline_metrics": _metrics(
            tournament.test_context_baseline_metrics
        ),
        "player_incremental_test_contrast": _metrics(
            tournament.player_incremental_test_contrast
        ),
    }
    model_hash = hashlib.sha256(
        _canonical_json_bytes(model_fingerprint_payload)
    ).hexdigest()
    model_id = f"player-performance-v1-{model_hash[:12]}"
    snapshot.insert(0, "model_hash", model_hash)
    snapshot.insert(0, "model_id", model_id)

    validation = {
        "artifact_schema_version": PUBLIC_ARTIFACT_SCHEMA_VERSION,
        "model_id": model_id,
        "model_hash": model_hash,
        "model_hash_scope": (
            "canonical public player-role coefficient snapshot, predeclared "
            "configuration, chronological splits, and frozen-test metrics"
        ),
        "evaluation_target": "held-out role-relative 15-minute resource performance",
        "estimand": ESTIMAND,
        "non_estimands": list(NON_ESTIMANDS),
        "roles": list(CANONICAL_ROLES),
        "effective_sample": {
            "eligible_role_matchups": tournament.audit.eligible_matchups,
            "stable_identity_matchups": tournament.audit.stable_identity_matchups,
            "test_player_rows": tournament.test_metrics.rows,
        },
        "test_gate_passed": True,
        "split_boundaries": dict(tournament.split_boundaries),
        "selected_base_penalty": tournament.selected_base_penalty,
        "selected_context_base_penalty": tournament.selected_context_base_penalty,
        "validation_candidates": _json_safe(
            tournament.validation_candidates.to_dict(orient="records")
        ),
        "test_metrics": _metrics(tournament.test_metrics),
        "test_context_baseline_metrics": _metrics(
            tournament.test_context_baseline_metrics
        ),
        "player_incremental_test_rmse_lift": (
            tournament.player_incremental_test_rmse_lift
        ),
        "player_incremental_test_contrast": _metrics(
            tournament.player_incremental_test_contrast
        ),
        "future_patch_test_metrics": _metrics(
            tournament.future_patch_test_metrics
        ),
        "roster_move_test_metrics": _metrics(
            tournament.roster_move_test_metrics
        ),
        "large_prediction_ledger_exported": False,
    }
    meta = {
        "artifact_schema_version": PUBLIC_ARTIFACT_SCHEMA_VERSION,
        "model_family": PUBLIC_MODEL_FAMILY,
        "display_name": PUBLIC_DISPLAY_NAME,
        "publication_status": PUBLICATION_STATUS,
        "model_id": model_id,
        "model_hash": model_hash,
        "model_hash_scope": (
            "canonical public player-role coefficient snapshot, predeclared "
            "configuration, chronological splits, and frozen-test metrics"
        ),
        "grain": "one stable player ID by canonical role",
        "source_contract": {
            "provider": "Oracle's Elixir",
            "source": "oe",
            "datacompleteness": "complete",
            "years": list(years),
            "identity_key": "provider playerid",
            "name_is_identity": False,
            "selection_audit": dict(source_selection_audit),
        },
        "estimand": ESTIMAND,
        "target_components": [
            "gold differential at 15 minutes",
            "experience differential at 15 minutes",
            "creep-score differential at 15 minutes",
        ],
        "target_combination": (
            "equal-weight mean after training-only robust standardization "
            "within role, league, and patch with declared fallbacks"
        ),
        "roles": list(CANONICAL_ROLES),
        "effective_sample": {
            "eligible_role_matchups": tournament.audit.eligible_matchups,
            "stable_identity_matchups": tournament.audit.stable_identity_matchups,
            "published_player_role_rows": int(len(snapshot)),
            "test_player_rows": tournament.test_metrics.rows,
        },
        "fit_through": str(fit_through),
        "test_window": {
            "start": tournament.split_boundaries["test_start"],
            "end": tournament.split_boundaries["test_end"],
        },
        "uncertainty": {
            "methods": uncertainty_methods,
            "conservative_z": config.conservative_z,
            "lower_bound": (
                f"mean minus {config.conservative_z} times the local "
                "Gaussian ridge standard deviation"
            ),
            "interpretation": (
                "prior-conditioned local coefficient uncertainty; not causal "
                "uncertainty and not a full Bayesian posterior"
            ),
        },
        "test_metrics": _metrics(tournament.test_metrics),
        "context_only_test_metrics": _metrics(
            tournament.test_context_baseline_metrics
        ),
        "player_incremental_test_contrast": _metrics(
            tournament.player_incremental_test_contrast
        ),
        "non_estimands": list(NON_ESTIMANDS),
        "limitations": list(tournament.limitations),
        "research_anchors": list(RESEARCH_ANCHORS),
        "ranking": {
            "scope": "within canonical role",
            "score": "lower_bound",
            "ties": "minimum competition rank on exact unrounded values",
        },
        "validation_artifact": "features/player_performance_validation.json",
    }
    return PlayerPerformancePublicArtifacts(
        snapshot=snapshot,
        meta=_json_safe(meta),
        validation=_json_safe(validation),
    )


def build_player_performance_public_artifacts(
    player_rows: pd.DataFrame,
    *,
    years: Sequence[int],
    config: PlayerPerformanceConfig | None = None,
) -> PlayerPerformancePublicArtifacts:
    """Run the governed tournament and produce only compact public artifacts."""

    cfg = config or PlayerPerformanceConfig()
    selected, selection_audit = select_canonical_player_performance_rows(
        player_rows, years
    )
    tournament = run_player_performance_tournament(selected, cfg)
    return render_player_performance_public_artifacts(
        tournament,
        selected,
        selection_audit,
        years=years,
        config=cfg,
    )
