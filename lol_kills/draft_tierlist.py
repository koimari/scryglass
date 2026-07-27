#!/usr/bin/env python3
"""Validated champion delta-WR artifact reader.

The historical Blade-Chest, side/blind, patch-window, and OE-lens artifacts are
quarantined. They were not evaluated as calibrated, chronological out-of-sample
champion effects and must not be used to alter match-win probabilities.

A future artifact is accepted only when it declares and passes the integrity
contract enforced by :func:`validate_tierlist_artifact`. Even then, this module
returns champion-level conditional delta-WR estimates only. It does not turn a
tierlist into a draft probability or blend one into another model.
"""

from __future__ import annotations

import json
import math
import re
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

from lol_kills.etl.aliases import normalize_champ
from lol_kills.etl.paths import MODELS_DIR

VALIDATED_TIERLIST_PATH = MODELS_DIR / "champ_tierlist_calibrated_delta_wr.json"
LEGACY_TIERLIST_PATHS: tuple[Path, ...] = (
    MODELS_DIR / "champ_tierlist_16_13_blade_chest.json",
    MODELS_DIR / "champ_tierlist_side_blind_counter.json",
    MODELS_DIR / "champ_tierlist_patch_window.json",
    MODELS_DIR / "champ_oe_lenses.json",
)

ARTIFACT_TYPE = "calibrated_champion_delta_wr_tierlist"
MIN_SCHEMA_MAJOR = 2
PATCH_RE = re.compile(r"^[0-9]{1,2}\.[0-9]{1,2}$")
REQUIRED_CONTROLS = (
    "elo",
    "team",
    "full_composition",
)
PROPER_SCORES = ("log_loss", "brier")

# League scopes are not interchangeable. In particular, regional leagues do
# not silently borrow an international board.
LEAGUE_SCOPE = {
    "LEC": "lec",
    "LCS": "lcs",
    "MSI": "msi",
    "EWC": "ewc",
}


class TierlistArtifactError(RuntimeError):
    """Base error for unavailable or invalid champion-tierlist artifacts."""


class TierlistArtifactUnavailableError(TierlistArtifactError):
    """No artifact exists at the governed, validated path."""


class TierlistArtifactContractError(TierlistArtifactError):
    """An artifact does not satisfy the publication/runtime integrity contract."""


class _ValidatedTierlist(dict[str, Any]):
    """Internal marker for a payload normalized by the integrity gate."""


def _require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TierlistArtifactContractError(f"{label} must be an object")
    return value


def _require_bool_true(block: dict[str, Any], key: str, label: str) -> None:
    if block.get(key) is not True:
        raise TierlistArtifactContractError(f"{label}.{key} must be true")


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TierlistArtifactContractError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise TierlistArtifactContractError(f"{label} must be finite")
    return number


def _iso_date(value: Any, label: str) -> date:
    if not isinstance(value, str):
        raise TierlistArtifactContractError(f"{label} must be an ISO date string")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise TierlistArtifactContractError(
            f"{label} must be an ISO date string"
        ) from exc


def _schema_major(value: Any) -> int:
    if not isinstance(value, str):
        raise TierlistArtifactContractError("schema_version must be a string")
    match = re.fullmatch(r"([0-9]+)(?:\.[0-9]+){1,2}", value)
    if not match:
        raise TierlistArtifactContractError(
            "schema_version must use a dotted string such as 2.0.0"
        )
    return int(match.group(1))


def require_patch_string(value: Any, *, label: str = "patch") -> str:
    """Return a canonical Riot patch string or reject lossy numeric storage."""

    if not isinstance(value, str) or not PATCH_RE.fullmatch(value):
        raise TierlistArtifactContractError(
            f"{label} must be a string in major.minor form; numeric patches are forbidden"
        )
    return value


def _validate_patch_values(value: Any, path: str = "$") -> None:
    """Reject numeric or malformed patch declarations anywhere in the payload."""

    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in {"patch", "prefer_patch"}:
                require_patch_string(child, label=child_path)
            elif key == "patches":
                if not isinstance(child, list) or not child:
                    raise TierlistArtifactContractError(
                        f"{child_path} must be a non-empty list of patch strings"
                    )
                for index, patch in enumerate(child):
                    require_patch_string(
                        patch,
                        label=f"{child_path}[{index}]",
                    )
            _validate_patch_values(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_patch_values(child, f"{path}[{index}]")


def _validate_chronological_holdout(validation: dict[str, Any]) -> None:
    holdout = _require_dict(
        validation.get("chronological_holdout"),
        "validation.chronological_holdout",
    )
    _require_bool_true(
        holdout,
        "scored_out_of_sample",
        "validation.chronological_holdout",
    )
    train_end = _iso_date(
        holdout.get("train_end"),
        "validation.chronological_holdout.train_end",
    )
    test_start = _iso_date(
        holdout.get("test_start"),
        "validation.chronological_holdout.test_start",
    )
    test_end = _iso_date(
        holdout.get("test_end"),
        "validation.chronological_holdout.test_end",
    )
    if not train_end < test_start <= test_end:
        raise TierlistArtifactContractError(
            "chronological holdout must start strictly after the training window"
        )
    n_games = holdout.get("n_games")
    if isinstance(n_games, bool) or not isinstance(n_games, int) or n_games <= 0:
        raise TierlistArtifactContractError(
            "validation.chronological_holdout.n_games must be a positive integer"
        )


def _validate_proper_scores(validation: dict[str, Any]) -> None:
    comparison = _require_dict(
        validation.get("proper_score_comparison"),
        "validation.proper_score_comparison",
    )
    _require_bool_true(
        comparison,
        "passed",
        "validation.proper_score_comparison",
    )
    if comparison.get("evaluated_on") != "chronological_holdout":
        raise TierlistArtifactContractError(
            "validation.proper_score_comparison.evaluated_on must be "
            "chronological_holdout"
        )
    baseline = _require_dict(
        comparison.get("baseline"),
        "validation.proper_score_comparison.baseline",
    )
    candidate = _require_dict(
        comparison.get("candidate"),
        "validation.proper_score_comparison.candidate",
    )
    if not str(baseline.get("name") or "").strip():
        raise TierlistArtifactContractError(
            "validation.proper_score_comparison.baseline.name is required"
        )
    for metric in PROPER_SCORES:
        baseline_score = _finite_number(
            baseline.get(metric),
            f"validation.proper_score_comparison.baseline.{metric}",
        )
        candidate_score = _finite_number(
            candidate.get(metric),
            f"validation.proper_score_comparison.candidate.{metric}",
        )
        if candidate_score >= baseline_score:
            raise TierlistArtifactContractError(
                f"candidate {metric} must be lower than the declared baseline"
            )


def _validate_calibration(validation: dict[str, Any]) -> None:
    calibration = _require_dict(
        validation.get("calibration"),
        "validation.calibration",
    )
    if not str(calibration.get("method") or "").strip():
        raise TierlistArtifactContractError(
            "validation.calibration.method is required"
        )
    if calibration.get("evaluated_on") != "chronological_holdout":
        raise TierlistArtifactContractError(
            "validation.calibration.evaluated_on must be chronological_holdout"
        )
    fit_end = _iso_date(
        calibration.get("fit_end"),
        "validation.calibration.fit_end",
    )
    test_start = _iso_date(
        _require_dict(
            validation.get("chronological_holdout"),
            "validation.chronological_holdout",
        ).get("test_start"),
        "validation.chronological_holdout.test_start",
    )
    if fit_end >= test_start:
        raise TierlistArtifactContractError(
            "calibration must be fit before the chronological holdout starts"
        )
    ece = _finite_number(
        calibration.get("ece"),
        "validation.calibration.ece",
    )
    if not 0.0 <= ece <= 1.0:
        raise TierlistArtifactContractError(
            "validation.calibration.ece must be in [0, 1]"
        )


def _validate_leakage_checks(validation: dict[str, Any]) -> None:
    checks = _require_dict(
        validation.get("leakage_checks"),
        "validation.leakage_checks",
    )
    _require_bool_true(checks, "passed", "validation.leakage_checks")
    if checks.get("copied_map_outcomes_detected") is not False:
        raise TierlistArtifactContractError(
            "validation.leakage_checks.copied_map_outcomes_detected must be false"
        )
    if checks.get("post_match_features_detected") is not False:
        raise TierlistArtifactContractError(
            "validation.leakage_checks.post_match_features_detected must be false"
        )


def _flatten_board(
    side_block: dict[str, Any] | None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Convert governed tier buckets into a champion lookup."""

    out: dict[str, dict[str, dict[str, Any]]] = {}
    if not side_block:
        return out
    board = _require_dict(side_block.get("board"), "tierlist side board")
    for tier, rows in board.items():
        if not isinstance(rows, list):
            raise TierlistArtifactContractError(
                f"tier bucket {tier!r} must be a list"
            )
        for row in rows:
            entry = _require_dict(row, f"tier bucket {tier!r} row")
            champ = normalize_champ(str(entry.get("champ") or ""))
            role = str(entry.get("role") or "").strip().lower()
            if not champ or not role:
                raise TierlistArtifactContractError(
                    "every tierlist row requires champ and role"
                )
            delta = _finite_number(
                entry.get("delta_wr_pp"),
                f"{champ}.delta_wr_pp",
            )
            champion_roles = out.setdefault(champ, {})
            if role in champion_roles:
                raise TierlistArtifactContractError(
                    f"duplicate champion-role row in one side board: {champ}/{role}"
                )
            normalized = dict(entry)
            normalized["champ"] = champ
            normalized["role"] = role
            normalized["delta_wr_pp"] = delta
            normalized["tier_label"] = str(tier)
            champion_roles[role] = normalized
    return out


def validate_tierlist_artifact(payload: Any) -> dict[str, Any]:
    """Validate and normalize a future champion delta-WR artifact.

    This is an integrity gate over the artifact's explicit evidence contract.
    It does not independently reproduce the model evaluation.
    """

    artifact = _require_dict(payload, "artifact")
    if artifact.get("artifact_type") != ARTIFACT_TYPE:
        raise TierlistArtifactContractError(
            f"artifact_type must be {ARTIFACT_TYPE!r}"
        )
    if _schema_major(artifact.get("schema_version")) < MIN_SCHEMA_MAJOR:
        raise TierlistArtifactContractError(
            f"schema_version major must be at least {MIN_SCHEMA_MAJOR}"
        )
    if artifact.get("publication_status") != "validated":
        raise TierlistArtifactContractError(
            "publication_status must be validated"
        )

    _validate_patch_values(artifact)
    patch_contract = _require_dict(
        artifact.get("patch_contract"),
        "patch_contract",
    )
    if patch_contract.get("storage") != "string":
        raise TierlistArtifactContractError(
            "patch_contract.storage must be string"
        )
    if patch_contract.get("source_dtype") != "string":
        raise TierlistArtifactContractError(
            "patch_contract.source_dtype must be string"
        )
    if patch_contract.get("numeric_coercion") != "forbidden":
        raise TierlistArtifactContractError(
            "patch_contract.numeric_coercion must be forbidden"
        )
    if patch_contract.get("format") != "major.minor":
        raise TierlistArtifactContractError(
            "patch_contract.format must be major.minor"
        )
    patches = patch_contract.get("patches")
    if not isinstance(patches, list) or not patches:
        raise TierlistArtifactContractError(
            "patch_contract.patches must be a non-empty list"
        )
    if len(set(patches)) != len(patches):
        raise TierlistArtifactContractError(
            "patch_contract.patches must not contain duplicates"
        )

    controls = _require_dict(artifact.get("controls"), "controls")
    for control in REQUIRED_CONTROLS:
        _require_bool_true(controls, control, "controls")

    estimand = _require_dict(artifact.get("estimand"), "estimand")
    if estimand.get("unit") != "percentage_points":
        raise TierlistArtifactContractError(
            "estimand.unit must be percentage_points"
        )
    _require_bool_true(estimand, "calibrated", "estimand")
    if estimand.get("quantity") != "conditional_delta_win_probability":
        raise TierlistArtifactContractError(
            "estimand.quantity must be conditional_delta_win_probability"
        )

    validation = _require_dict(artifact.get("validation"), "validation")
    _validate_chronological_holdout(validation)
    _validate_proper_scores(validation)
    _validate_calibration(validation)
    _validate_leakage_checks(validation)

    by_scope = _require_dict(artifact.get("by_scope"), "by_scope")
    if not by_scope:
        raise TierlistArtifactContractError("by_scope must not be empty")
    normalized_scopes: dict[str, dict[str, Any]] = {}
    declared_patches = set(patches)
    for raw_scope, raw_block in by_scope.items():
        scope = str(raw_scope).strip().lower()
        block = _require_dict(raw_block, f"by_scope.{scope}")
        patch = require_patch_string(
            block.get("patch"),
            label=f"by_scope.{scope}.patch",
        )
        if patch not in declared_patches:
            raise TierlistArtifactContractError(
                f"by_scope.{scope}.patch is absent from patch_contract.patches"
            )
        normalized_scopes[scope] = {
            **block,
            "patch": patch,
            "blue": _flatten_board(
                _require_dict(block.get("blue"), f"by_scope.{scope}.blue")
            ),
            "red": _flatten_board(
                _require_dict(block.get("red"), f"by_scope.{scope}.red")
            ),
        }

    return _ValidatedTierlist(
        {
            **artifact,
            "by_scope": normalized_scopes,
        }
    )


def load_tierlist_artifact(
    path: Path = VALIDATED_TIERLIST_PATH,
) -> dict[str, Any]:
    """Load the governed artifact path and fail closed on all other states."""

    path = Path(path)
    if not path.is_file():
        legacy_present = [p.name for p in LEGACY_TIERLIST_PATHS if p.is_file()]
        suffix = (
            f"; quarantined legacy files present: {', '.join(legacy_present)}"
            if legacy_present
            else ""
        )
        raise TierlistArtifactUnavailableError(
            f"validated champion tierlist artifact is unavailable at {path}{suffix}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TierlistArtifactContractError(
            f"could not read champion tierlist artifact at {path}"
        ) from exc
    return validate_tierlist_artifact(payload)


@lru_cache(maxsize=1)
def _load_artifacts() -> dict[str, Any]:
    return load_tierlist_artifact()


def league_to_scope(league: str | None) -> str:
    league_name = str(league or "").strip().upper()
    return LEAGUE_SCOPE.get(league_name, league_name.casefold())


def _validated_artifact(
    artifact: dict[str, Any] | None,
) -> dict[str, Any]:
    if artifact is None:
        return _load_artifacts()
    if isinstance(artifact, _ValidatedTierlist):
        return artifact
    return validate_tierlist_artifact(artifact)


def lookup_champ(
    champ: str,
    side: str,
    *,
    league: str,
    patch: str,
    role: str | None = None,
    artifact: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Look up one exact-scope, exact-patch calibrated champion estimate."""

    governed = _validated_artifact(artifact)
    scope = league_to_scope(league)
    patch = require_patch_string(patch)
    scope_block = governed["by_scope"].get(scope)
    if not scope_block or scope_block["patch"] != patch:
        return None
    side_key = side.strip().lower()
    if side_key not in {"blue", "red"}:
        raise ValueError("side must be blue or red")
    champion_roles = scope_block[side_key].get(normalize_champ(champ))
    if not champion_roles:
        return None
    if role is not None:
        entry = champion_roles.get(str(role).strip().lower())
    elif len(champion_roles) == 1:
        entry = next(iter(champion_roles.values()))
    else:
        return None
    if entry is None:
        return None
    return {
        "champ": entry["champ"],
        "role": entry["role"],
        "delta_wr_pp": round(float(entry["delta_wr_pp"]), 2),
        "n": int(entry.get("n") or 0),
        "tier_label": entry.get("tier_label"),
        "source": "validated_chronological_artifact",
    }


def score_draft_tierlist(
    blue: list[str],
    red: list[str],
    *,
    league: str,
    patch: str,
    blue_roles: list[str] | None = None,
    red_roles: list[str] | None = None,
    artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a descriptive comparison of validated champion estimates.

    The difference of side means is not a match prediction and is deliberately
    not converted to a probability.
    """

    if len(blue) != 5 or len(red) != 5:
        raise ValueError("champion tierlist comparison requires five champions per side")
    patch = require_patch_string(patch)
    governed = _validated_artifact(artifact)
    scope = league_to_scope(league)
    scope_block = governed["by_scope"].get(scope)
    if scope_block is None or scope_block["patch"] != patch:
        raise TierlistArtifactUnavailableError(
            f"no validated champion tierlist for scope={scope!r}, patch={patch!r}"
        )

    if blue_roles is not None and len(blue_roles) != 5:
        raise ValueError("blue_roles must contain five roles")
    if red_roles is not None and len(red_roles) != 5:
        raise ValueError("red_roles must contain five roles")
    blue_role_values = blue_roles or [None] * 5
    red_role_values = red_roles or [None] * 5

    blue_rows = [
        lookup_champ(
            champ,
            "blue",
            league=league,
            patch=patch,
            role=role,
            artifact=governed,
        )
        for champ, role in zip(blue, blue_role_values)
    ]
    red_rows = [
        lookup_champ(
            champ,
            "red",
            league=league,
            patch=patch,
            role=role,
            artifact=governed,
        )
        for champ, role in zip(red, red_role_values)
    ]

    missing = [
        champ
        for champ, row in zip(blue + red, blue_rows + red_rows)
        if row is None
    ]
    if missing:
        raise TierlistArtifactUnavailableError(
            "validated champion estimates are incomplete for: "
            + ", ".join(missing)
        )

    complete_blue = [row for row in blue_rows if row is not None]
    complete_red = [row for row in red_rows if row is not None]
    blue_mean = sum(row["delta_wr_pp"] for row in complete_blue) / 5.0
    red_mean = sum(row["delta_wr_pp"] for row in complete_red) / 5.0
    return {
        "status": "validated_champion_estimates",
        "scope": scope,
        "league": league,
        "patch": patch,
        "unit": "percentage_points",
        "blue_mean_delta_wr_pp": round(blue_mean, 2),
        "red_mean_delta_wr_pp": round(red_mean, 2),
        "difference_of_means_pp": round(blue_mean - red_mean, 2),
        "blue": complete_blue,
        "red": complete_red,
        "match_win_probability": None,
        "note": (
            "Champion-level conditional delta-WR estimates from a validated "
            "chronological artifact; the side comparison is not a match prediction."
        ),
    }


def blend_win_with_tierlist(
    p_blue: float,
    tier: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    """Do not blend champion tierlists into match-win probabilities."""

    del tier
    return float(p_blue), {
        "applied": False,
        "p_before": float(p_blue),
        "reason": "champion_delta_wr_is_not_a_match_probability_input",
    }


def format_tierlist_report(
    tier: dict[str, Any],
    *,
    team_blue: str = "Blue",
    team_red: str = "Red",
) -> str:
    """Format the non-predictive, validated champion-estimate comparison."""

    lines = [
        "--- Validated champion delta-WR estimates ---",
        (
            f"  Scope={tier.get('scope')}  patch={tier.get('patch')}  "
            f"unit={tier.get('unit')}"
        ),
        (
            f"  {team_blue} mean {tier.get('blue_mean_delta_wr_pp'):+.2f}pp  |  "
            f"{team_red} mean {tier.get('red_mean_delta_wr_pp'):+.2f}pp  |  "
            f"difference {tier.get('difference_of_means_pp'):+.2f}pp"
        ),
        "  Descriptive champion comparison only; no match probability is produced.",
    ]
    for label, rows in (
        (team_blue, tier.get("blue") or []),
        (team_red, tier.get("red") or []),
    ):
        entries = [
            f"{row['champ']} {float(row['delta_wr_pp']):+.2f}pp [{row['role']}]"
            for row in rows
        ]
        lines.append(f"  {label}: " + (", ".join(entries) if entries else "—"))
    return "\n".join(lines)


if __name__ == "__main__":
    _load_artifacts.cache_clear()
    try:
        artifact = _load_artifacts()
    except TierlistArtifactError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        "Validated champion tierlist loaded: "
        f"{len(artifact.get('by_scope', {}))} scope(s)"
    )
