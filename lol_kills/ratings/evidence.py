"""Validated evidence-state fields for public ratings (issue #46).

The public UI previously labeled a rating ``Settled`` when sigma reached a
fixed floor.  That conflated a low internal variance with validated public
evidence.  This module attaches explicit evidence fields to rating rows and
derives the evidence state from the validated contract:

* ``evidence_interval_width``  — two-sided 95% interval in display points;
* ``evidence_precision_ratio`` — relative posterior information
  (sigma_tightest / sigma)^2 versus the tightest row in the same scope;
* ``evidence_stability``       — per-game (players) or weekly (teams)
  posterior displacement; ``None`` when there is no history to measure;
* ``evidence_freshness_days``  — days between the pack source as-of and the
  row's most recent game;
* ``evidence_support_coverage``— sample support relative to the coverage
  target (players 10 maps, teams 8 series);
* ``evidence_fallback``        — the row rests on a fallback/neutral prior;
* ``evidence_active``          — active/current eligibility;
* ``evidence_disconnected``    — the row has no supported league anchor;
* ``evidence_ood``             — the row lies outside the supported
  distribution.

Sigma and map count stay separate diagnostics.  Settled requires strictly
greater-than-95% relative precision AND known bounded stability AND fresh
inputs AND full support coverage AND active eligibility, with no fallback,
disconnection, or out-of-distribution flag.  Every other row fails closed to
an explicit state.

All thresholds are exact constants shared with the TypeScript mirror
(``apps/scryglass/src/lib/evidence.ts``); changing them requires updating both
sides and the adversarial tests.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import pandas as pd


# Two-sided 95% normal quantile (display-points interval half-width).
Z95 = 1.959963984540054

# Exact contract thresholds (mirrored in apps/scryglass/src/lib/evidence.ts).
FRESH_DAYS = 14          # Settled requires a game within this many days.
ACTIVE_DAYS = 60         # Row is active within this many days.
STALE_DAYS = 90          # Beyond this the row is stale.
WIDE_INTERVAL_WIDTH = 200.0  # Display points; wider rows are wide-interval.
SETTLED_PRECISION_RATIO = 0.95  # Strictly greater than this.
OBSERVED_PRECISION_RATIO = 0.80
SETTLED_STABILITY = 6.0  # Max posterior displacement per game/week.
PLAYER_COVERAGE_TARGET = 10  # Maps.
TEAM_COVERAGE_TARGET = 8     # Series.
# Tightest sigma each model can express (its sigma floor): the precision
# reference.  These mirror PlayerEloConfig.sigma_min and
# HierarchicalBTConfig.min_sigma; keep them in sync.
PLAYER_SIGMA_FLOOR = 28.0
TEAM_SIGMA_FLOOR = 20.0
STATE_ORDER = (
    "unsupported",
    "ood",
    "disconnected",
    "stale",
    "inactive",
    "wide_interval",
    "fallback",
    "settled",
    "observed",
    "thin",
)


def _number(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def _flag(value: Any, name: str) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value in (0, 1) else None


def evidence_state(fields: Mapping[str, Any]) -> str:
    """Derive the fail-closed evidence state from the validated contract."""

    interval_width = _number(fields.get("evidence_interval_width"), "interval_width")
    precision = _number(fields.get("evidence_precision_ratio"), "precision_ratio")
    stability = _number(fields.get("evidence_stability"), "stability")
    freshness = _number(fields.get("evidence_freshness_days"), "freshness_days")
    coverage = _number(fields.get("evidence_support_coverage"), "support_coverage")
    fallback = _flag(fields.get("evidence_fallback"), "fallback")
    active = _flag(fields.get("evidence_active"), "active")
    disconnected = _flag(fields.get("evidence_disconnected"), "disconnected")
    ood = _flag(fields.get("evidence_ood"), "ood")

    # Fail closed on any missing or non-finite required field: an artifact
    # that cannot prove its evidence basis is unsupported, never settled.
    if any(value is None for value in (interval_width, precision, coverage, fallback, active, disconnected, ood)):
        return "unsupported"
    if ood:
        return "ood"
    if disconnected:
        return "disconnected"
    if fallback:
        return "fallback"
    if freshness is None or freshness > STALE_DAYS:
        return "stale"
    if not active:
        return "inactive"
    if interval_width > WIDE_INTERVAL_WIDTH:
        return "wide_interval"
    if (
        precision > SETTLED_PRECISION_RATIO
        and stability is not None
        and stability <= SETTLED_STABILITY
        and freshness <= FRESH_DAYS
        and coverage >= 1.0
        and active == 1
    ):
        return "settled"
    if precision > OBSERVED_PRECISION_RATIO and stability is not None and coverage >= 0.5:
        return "observed"
    return "thin"


def _attach(
    frame: pd.DataFrame,
    *,
    source_as_of: pd.Timestamp,
    coverage_numerator: str,
    coverage_target: int,
    sigma_floor: float,
    stability_by_key: Mapping[str, float] | None = None,
    last_game_by_key: Mapping[str, pd.Timestamp] | None = None,
    fallback_by_key: Mapping[str, int] | None = None,
    disconnected_by_key: Mapping[str, int] | None = None,
    ood_by_key: Mapping[str, int] | None = None,
    key_column: str,
) -> pd.DataFrame:
    """Attach evidence fields to a rating snapshot frame."""

    out = frame.copy()
    if out.empty:
        for column in (
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
        ):
            out[column] = pd.Series(dtype="float64")
        out["evidence_state"] = pd.Series(dtype="object")
        return out
    sigma = pd.to_numeric(out["sigma"], errors="coerce")
    keys = out[key_column].astype(str)
    # Registered precision reference for the descriptive ladder: the median
    # sigma of the same scope, floored at the model's expressible floor.
    # Relative precision is then (sigma_ref / sigma)^2, clamped to [0, 1].
    # The v2 contract defines settled by posterior precision at a registered
    # resolution; this descriptive pack registers the median-tightness
    # reference until an L2-estimated resolution exists.
    median_sigma = float(sigma.median()) if sigma.notna().any() else sigma_floor
    sigma_ref = max(sigma_floor, median_sigma)
    latest = pd.Timestamp(source_as_of)
    if latest.tzinfo is not None:
        latest = latest.tz_convert("UTC").tz_localize(None)

    def by_key(mapping: Mapping[str, Any] | None, default: Any) -> list[Any]:
        if mapping is None:
            return [default] * len(out)
        return [mapping.get(str(key), default) for key in keys]

    counts = pd.to_numeric(out[coverage_numerator], errors="coerce").fillna(0.0)
    last_dates = by_key(last_game_by_key, None)
    stabilities = by_key(stability_by_key, None)
    fallbacks = by_key(fallback_by_key, 0)
    disconnected = by_key(disconnected_by_key, 0)
    oods = by_key(ood_by_key, 0)

    rows: list[dict[str, Any]] = []
    for index, (row_sigma, count, last_date, stability, fb, disc, ood) in enumerate(
        zip(sigma, counts, last_dates, stabilities, fallbacks, disconnected, oods)
    ):
        if row_sigma is None or not math.isfinite(float(row_sigma)) or row_sigma <= 0:
            rows.append({"interval_width": None, "precision": None, "freshness": None, "coverage": None, "fallback": None, "active": None, "disconnected": None, "ood": None, "stability": None, "state": "unsupported"})
            continue
        interval_width = 2.0 * Z95 * float(row_sigma)
        # Relative precision: posterior information versus the registered
        # reference (median tightness of the scope, floored at the model's
        # expressible floor).  The reference alone never settles a row; it is
        # one criterion among the contract's stability/freshness/coverage/
        # active/fallback checks.
        precision = 0.0 if sigma_ref <= 0 else min(1.0, (sigma_ref / float(row_sigma)) ** 2)
        freshness = None
        if last_date is not None:
            try:
                stamp = pd.Timestamp(last_date)
                if stamp.tzinfo is not None:
                    stamp = stamp.tz_convert("UTC").tz_localize(None)
                freshness = float((latest - stamp).total_seconds() / 86400.0)
            except (TypeError, ValueError):
                freshness = None
        coverage = min(1.0, float(count) / max(coverage_target, 1))
        fallback = int(fb or 0)
        active = int(1 if (freshness is not None and freshness <= ACTIVE_DAYS) else 0)
        fields = {
            "interval_width": interval_width,
            "precision": precision,
            "stability": stability,
            "freshness": freshness,
            "coverage": coverage,
            "fallback": fallback,
            "active": active,
            "disconnected": int(disc or 0),
            "ood": int(ood or 0),
        }
        rows.append({**fields, "state": evidence_state({
            "evidence_interval_width": interval_width,
            "evidence_precision_ratio": precision,
            "evidence_stability": stability,
            "evidence_freshness_days": freshness,
            "evidence_support_coverage": coverage,
            "evidence_fallback": fallback,
            "evidence_active": active,
            "evidence_disconnected": int(disc or 0),
            "evidence_ood": int(ood or 0),
        })})
    for field, column in (
        ("interval_width", "evidence_interval_width"),
        ("precision", "evidence_precision_ratio"),
        ("stability", "evidence_stability"),
        ("freshness", "evidence_freshness_days"),
        ("coverage", "evidence_support_coverage"),
        ("fallback", "evidence_fallback"),
        ("active", "evidence_active"),
        ("disconnected", "evidence_disconnected"),
        ("ood", "evidence_ood"),
    ):
        out[column] = [row[field] for row in rows]
    out["evidence_state"] = [row["state"] for row in rows]
    return out


def attach_team_evidence(
    ratings: pd.DataFrame,
    *,
    source_as_of: pd.Timestamp,
    weekly_stability: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Attach evidence fields to the hierarchical team snapshot."""

    key_column = "team_key" if "team_key" in ratings.columns else "team"
    # Weekly movement is keyed by display name in build_team_weekly_ranks;
    # remap onto the identity key used by the snapshot.
    if weekly_stability:
        display_by_key = {
            str(row[key_column]): str(row.get("team") or row[key_column])
            for _, row in ratings.iterrows()
        }
        key_by_display = {display: key for key, display in display_by_key.items()}
        remapped: dict[str, float] = {}
        for name, value in weekly_stability.items():
            target = str(name)
            if target in key_by_display:
                target = key_by_display[target]
            remapped[target] = float(value)
        weekly_stability = remapped
    last_game = {}
    if "last_game_date" in ratings.columns:
        for _, row in ratings.iterrows():
            value = row.get("last_game_date")
            if value is not None and pd.notna(value):
                last_game[str(row[key_column])] = pd.Timestamp(value)
    fallback = None  # the hierarchical fit only emits teams with series
    disconnected = {
        str(row[key_column]): int(
            str(row.get("home_league") or "") == "UNKNOWN"
            and int(pd.to_numeric(row.get("international_series", 0), errors="coerce") or 0) == 0
        )
        for _, row in ratings.iterrows()
    }
    return _attach(
        ratings,
        source_as_of=source_as_of,
        coverage_numerator="n_series",
        coverage_target=TEAM_COVERAGE_TARGET,
        sigma_floor=TEAM_SIGMA_FLOOR,
        stability_by_key=weekly_stability,
        last_game_by_key=last_game,
        fallback_by_key=fallback,
        disconnected_by_key=disconnected,
        key_column=key_column,
    )


def attach_player_evidence(
    snapshot: pd.DataFrame,
    *,
    source_as_of: pd.Timestamp,
) -> pd.DataFrame:
    """Attach evidence fields to the sequential player snapshot."""

    key_column = "player"
    last_game = {}
    stability = {}
    fallback = {}
    disconnected = {}
    ood = {}
    for _, row in snapshot.iterrows():
        key = str(row[key_column])
        if row.get("last_game_date") is not None and pd.notna(row.get("last_game_date")):
            last_game[key] = pd.Timestamp(row["last_game_date"])
        if row.get("evidence_stability") is not None and pd.notna(row.get("evidence_stability")):
            stability[key] = float(row["evidence_stability"])
        n_maps = int(pd.to_numeric(row.get("n_maps", 0), errors="coerce") or 0)
        fallback[key] = int(n_maps == 0)
        team_value = row.get("last_team")
        has_team = bool(
            pd.notna(team_value)
            and str(team_value).strip()
            and str(team_value).casefold() != "nan"
        )
        global_connected = int(pd.to_numeric(row.get("global_connected", 1), errors="coerce") or 0)
        disconnected[key] = int(n_maps > 0 and (not has_team or global_connected != 1))
        league = str(row.get("home_league") or "").strip()
        ood[key] = int(n_maps > 0 and league == "UNKNOWN")
    return _attach(
        snapshot,
        source_as_of=source_as_of,
        coverage_numerator="n_maps",
        coverage_target=PLAYER_COVERAGE_TARGET,
        sigma_floor=PLAYER_SIGMA_FLOOR,
        stability_by_key=stability,
        last_game_by_key=last_game,
        fallback_by_key=fallback,
        disconnected_by_key=disconnected,
        ood_by_key=ood,
        key_column=key_column,
    )
