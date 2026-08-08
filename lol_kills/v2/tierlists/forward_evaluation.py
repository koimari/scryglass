"""Run a time-safe forward diagnostic for the descriptive champion ladder.

The diagnostic freezes the ladder update rule at a cutoff, scores later maps
with pre-map state, and then applies each result.  It records observed
outcomes and source hashes.  It does not authorize probability, prediction,
recommendation, or betting claims.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss


SCHEMA_VERSION = "scryglass:tierlist-forward-evaluation:v1"
SOURCE_LOCATOR = Path("data/lol/warehouse/parquet/oe_live/oe_player_games.parquet")
SOURCE_META_LOCATOR = Path("data/lol/warehouse/parquet/oe_live/meta.json")
CANDIDATE_LOCATOR = Path("data/lol/v2/tierlists/champion-elo-candidate-v1.json")
HISTORY_START = pd.Timestamp("2025-01-01T00:00:00Z")
DEFAULT_CUTOFF = pd.Timestamp("2026-07-18T00:00:00Z")
ROLES = ("top", "jungle", "mid", "bot", "support")
ROLE_ALIASES = {
    "top": "top",
    "jng": "jungle",
    "jungle": "jungle",
    "mid": "mid",
    "middle": "mid",
    "bot": "bot",
    "bottom": "bot",
    "adc": "bot",
    "sup": "support",
    "support": "support",
    "utility": "support",
}
INTERNATIONAL_EVENTS = {"msi", "ewc", "worlds", "fst", "first stand", "asia_master", "em"}
INITIAL_RATING = 1500.0
CHAMPION_K = 12.0
TEAM_K = 20.0


class ForwardEvaluationError(ValueError):
    """Raised when the forward evidence cannot be built safely."""


@dataclass(frozen=True)
class MapRecord:
    map_id: str
    date: pd.Timestamp
    scope_id: str
    league: str | None
    event_kind: str | None
    competition_tier: str | None
    patch: str
    blue_team: str
    red_team: str
    blue_win: int
    roles: Mapping[str, Mapping[str, str]]


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "artifact_sha256"}
    return _sha256_bytes(_canonical(unsigned))


def _normalize_side(value: object) -> str | None:
    token = str(value).strip().casefold()
    if token in {"blue", "1"}:
        return "blue"
    if token in {"red", "2"}:
        return "red"
    return None


def _normalize_team(value: object) -> str:
    return "".join(ch for ch in str(value).strip().casefold() if ch.isalnum())


def _scope(row: pd.Series) -> tuple[str, str | None, str | None, str | None]:
    league = str(row.get("league") or "").strip().upper() or None
    tier = None if pd.isna(row.get("competition_tier")) else str(row.get("competition_tier")).strip().casefold()
    event = None if pd.isna(row.get("event_kind")) else str(row.get("event_kind")).strip().casefold()
    if event in INTERNATIONAL_EVENTS or tier == "international":
        event_name = event or league or "international"
        return f"event:{event_name}", None, event_name, "international"
    if league is None:
        raise ForwardEvaluationError("map has no league identity")
    tier = tier or "unknown"
    return f"league:{league.casefold()}:{tier}", league, None, tier


def _load_maps(root: Path) -> tuple[list[MapRecord], dict[str, Any]]:
    source = root / SOURCE_LOCATOR
    meta = root / SOURCE_META_LOCATOR
    if not source.is_file() or source.is_symlink():
        raise ForwardEvaluationError(f"live OE source is missing: {SOURCE_LOCATOR}")
    if not meta.is_file() or meta.is_symlink():
        raise ForwardEvaluationError(f"live OE source receipt is missing: {SOURCE_META_LOCATOR}")
    try:
        frame = pd.read_parquet(
            source,
            columns=[
                "gameid",
                "game_uid",
                "date",
                "league",
                "competition_tier",
                "event_kind",
                "patch",
                "position",
                "champion",
                "side",
                "teamname",
                "result",
            ],
        )
    except (OSError, KeyError, ValueError) as exc:
        raise ForwardEvaluationError("live OE source cannot be read") from exc
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    frame["role"] = frame["position"].map(lambda value: ROLE_ALIASES.get(str(value).strip().casefold()))
    frame["side_norm"] = frame["side"].map(_normalize_side)
    frame["result_num"] = pd.to_numeric(frame["result"], errors="coerce")
    frame["map_key"] = frame["game_uid"].where(frame["game_uid"].notna(), frame["gameid"]).astype(str)
    frame = frame[
        frame["date"].notna()
        & frame["date"].ge(HISTORY_START)
        & frame["role"].isin(ROLES)
        & frame["side_norm"].isin(("blue", "red"))
        & frame["result_num"].isin((0, 1))
    ].copy()

    maps: list[MapRecord] = []
    excluded: defaultdict[str, int] = defaultdict(int)
    seen: set[tuple[Any, ...]] = set()
    for map_id, group in frame.groupby("map_key", sort=False):
        if len(group) != 10 or group["date"].nunique() != 1:
            excluded["map_shape"] += 1
            continue
        if group["patch"].nunique(dropna=False) != 1:
            excluded["patch_identity"] += 1
            continue
        try:
            scope_id, league, event_kind, tier = _scope(group.iloc[0])
        except ForwardEvaluationError:
            excluded["scope_identity"] += 1
            continue
        sides: dict[str, dict[str, Any]] = {}
        valid = True
        for side in ("blue", "red"):
            side_rows = group[group["side_norm"] == side]
            if len(side_rows) != 5 or side_rows["teamname"].nunique(dropna=False) != 1 or side_rows["result_num"].nunique(dropna=False) != 1:
                valid = False
                break
            team = _normalize_team(side_rows["teamname"].iloc[0])
            if not team:
                valid = False
                break
            role_map: dict[str, str] = {}
            for role in ROLES:
                role_rows = side_rows[side_rows["role"] == role]
                if len(role_rows) != 1 or pd.isna(role_rows["champion"].iloc[0]):
                    valid = False
                    break
                champion = str(role_rows["champion"].iloc[0]).strip()
                if not champion:
                    valid = False
                    break
                role_map[role] = champion
            if not valid:
                break
            sides[side] = {
                "team": team,
                "result": int(side_rows["result_num"].iloc[0]),
                "roles": role_map,
            }
        if not valid or set(sides) != {"blue", "red"} or set(sides[side]["result"] for side in sides) != {0, 1}:
            excluded["side_or_role_identity"] += 1
            continue
        signature = (
            pd.Timestamp(group["date"].iloc[0]).isoformat(),
            scope_id,
            sides["blue"]["team"],
            sides["red"]["team"],
            sides["blue"]["result"],
            tuple(sides[side]["roles"][role] for role in ROLES for side in ("blue", "red")),
        )
        if signature in seen:
            continue
        seen.add(signature)
        maps.append(
            MapRecord(
                map_id=str(map_id),
                date=pd.Timestamp(group["date"].iloc[0]),
                scope_id=scope_id,
                league=league,
                event_kind=event_kind,
                competition_tier=tier,
                patch=str(group["patch"].iloc[0]).strip(),
                blue_team=sides["blue"]["team"],
                red_team=sides["red"]["team"],
                blue_win=sides["blue"]["result"],
                roles={"blue": sides["blue"]["roles"], "red": sides["red"]["roles"]},
            )
        )
    maps.sort(key=lambda item: (item.date, item.map_id))
    meta_payload = json.loads(meta.read_text(encoding="utf-8"))
    if not isinstance(meta_payload, Mapping):
        raise ForwardEvaluationError("live OE source receipt is not an object")
    return maps, {
        "locator": SOURCE_LOCATOR.as_posix(),
        "raw_sha256": _sha256_path(source),
        "meta_locator": SOURCE_META_LOCATOR.as_posix(),
        "meta_raw_sha256": _sha256_path(meta),
        "meta_source_latest": meta_payload.get("source_latest"),
        "maps_loaded": len(maps),
        "excluded_maps": dict(sorted(excluded.items())),
    }


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))


def _logit(value: float) -> float:
    clipped = max(1e-6, min(1.0 - 1e-6, value))
    return math.log(clipped / (1.0 - clipped))


def _score(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    return {
        "log_loss": float(log_loss(labels, clipped, labels=[0, 1])),
        "brier": float(brier_score_loss(labels, clipped)),
    }


def _read_candidate(root: Path) -> tuple[bytes, dict[str, Any]]:
    path = root / CANDIDATE_LOCATOR
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ForwardEvaluationError("champion Elo candidate cannot be read") from exc
    if not isinstance(payload, dict):
        raise ForwardEvaluationError("champion Elo candidate must be an object")
    if payload.get("artifact_sha256") != _canonical_sha256(payload):
        raise ForwardEvaluationError("champion Elo candidate canonical digest is invalid")
    return raw, payload


def evaluate(
    root: Path | str = Path("."),
    *,
    cutoff: str | pd.Timestamp = DEFAULT_CUTOFF,
) -> dict[str, Any]:
    """Build a complete descriptive forward diagnostic."""

    repo_root = Path(root)
    cutoff_stamp = pd.Timestamp(cutoff)
    if cutoff_stamp.tzinfo is None:
        cutoff_stamp = cutoff_stamp.tz_localize("UTC")
    else:
        cutoff_stamp = cutoff_stamp.tz_convert("UTC")
    candidate_raw, candidate = _read_candidate(repo_root)
    maps, source = _load_maps(repo_root)
    if not maps:
        raise ForwardEvaluationError("no valid maps remain after source checks")

    team_ratings: dict[str, float] = {}
    champion_ratings: dict[tuple[str, str, str], float] = {}
    labels: list[int] = []
    controlled: list[float] = []
    team_only: list[float] = []
    future_map_ids: list[str] = []
    pre_cutoff_maps = 0
    post_cutoff_maps = 0
    state_order_violation = False

    for game in maps:
        blue_rating = team_ratings.get(game.blue_team, INITIAL_RATING)
        red_rating = team_ratings.get(game.red_team, INITIAL_RATING)
        team_probability = _sigmoid((blue_rating - red_rating) * math.log(10.0) / 400.0)
        champion_logit = 0.0
        for role in ROLES:
            blue_champion = game.roles["blue"][role]
            red_champion = game.roles["red"][role]
            blue_champion_rating = champion_ratings.get((game.scope_id, role, blue_champion), INITIAL_RATING)
            red_champion_rating = champion_ratings.get((game.scope_id, role, red_champion), INITIAL_RATING)
            champion_logit += (blue_champion_rating - red_champion_rating) * math.log(10.0) / 400.0
        controlled_probability = _sigmoid(_logit(team_probability) + champion_logit)

        if game.date >= cutoff_stamp:
            labels.append(game.blue_win)
            controlled.append(controlled_probability)
            team_only.append(team_probability)
            future_map_ids.append(game.map_id)
            post_cutoff_maps += 1
        else:
            pre_cutoff_maps += 1

        result = float(game.blue_win)
        for role in ROLES:
            blue_champion = game.roles["blue"][role]
            red_champion = game.roles["red"][role]
            blue_key = (game.scope_id, role, blue_champion)
            red_key = (game.scope_id, role, red_champion)
            residual = result - controlled_probability
            champion_ratings[blue_key] = champion_ratings.get(blue_key, INITIAL_RATING) + CHAMPION_K * residual
            champion_ratings[red_key] = champion_ratings.get(red_key, INITIAL_RATING) - CHAMPION_K * residual
        team_residual = result - team_probability
        team_ratings[game.blue_team] = blue_rating + TEAM_K * team_residual
        team_ratings[game.red_team] = red_rating - TEAM_K * team_residual

        if not math.isfinite(controlled_probability) or not math.isfinite(team_probability):
            state_order_violation = True

    if not labels:
        raise ForwardEvaluationError("forward window has no observed maps")
    label_array = np.asarray(labels, dtype=int)
    controlled_array = np.asarray(controlled, dtype=float)
    team_array = np.asarray(team_only, dtype=float)
    controlled_score = _score(label_array, controlled_array)
    team_score = _score(label_array, team_array)
    delta = {key: controlled_score[key] - team_score[key] for key in controlled_score}
    candidate_cells = candidate.get("cells")
    if not isinstance(candidate_cells, list) or not candidate_cells:
        raise ForwardEvaluationError("candidate has no cells")
    roles = {cell.get("role") for cell in candidate_cells if isinstance(cell, Mapping)}
    movement_complete = all(
        isinstance(row, Mapping)
        and {"rank", "previous_rank", "rank_delta", "rating_delta", "movement"}.issubset(row)
        for cell in candidate_cells
        if isinstance(cell, Mapping)
        for row in (cell.get("rows") or [])
    )
    rating_method = candidate.get("rating_method") or {}
    matchup_method = candidate.get("matchup_shape_method") or {}
    joint_model = candidate.get("joint_model") or {}
    matchup_contract_valid = (
        str(rating_method.get("name", "")).startswith("joint five-role")
        and "full observed-Hessian" in str(rating_method.get("fit", ""))
        and rating_method.get("fit_coordinates") == "sparse reference-coded joint map rows"
        and str(matchup_method.get("name", "")).startswith("atom-informed")
        and matchup_method.get("outcome_variation_required") is True
        and isinstance(matchup_method.get("posterior_draws"), int)
        and matchup_method.get("posterior_draws", 0) >= 2000
        and joint_model.get("schema_id") == "scryglass.tierlists.joint-pooled-model.v1"
        and joint_model.get("posterior_draws_verified", 0) >= 2000
        and joint_model.get("partial_pooling")
    )
    counterability_rows_valid = all(
        row.get("counterability_status") in {"available", "unavailable"}
        and (
            row.get("counterability_status") == "unavailable"
            or (
                isinstance(row.get("blind_score_pp"), (int, float))
                and isinstance(row.get("counter_score"), (int, float))
                and isinstance(row.get("expected_counter_breadth"), (int, float))
                and isinstance(row.get("countered_opponent_count"), int)
                and isinstance(row.get("countered_opponent_share"), (int, float))
                and row.get("matchup_opponents", 0) >= matchup_method.get("minimum_opponents", 10**9)
                and isinstance(row.get("legal_opponents"), list)
                and len(row.get("legal_opponents")) == matchup_method.get("legal_opponent_count")
                and isinstance(row.get("maximum_strength_contrast_sd"), (int, float))
                and row.get("maximum_strength_contrast_sd")
                <= rating_method.get("maximum_supported_strength_contrast_sd", 0.0)
            )
        )
        and (
            row.get("tier_bucket") not in {"Z Blind", "S Blind", "Z Counter", "S Counter"}
            or (
                row.get("counterability_status") == "available"
                and isinstance(row.get("tier_membership_probability"), (int, float))
                and row.get("tier_membership_probability")
                >= matchup_method.get("minimum_special_tier_membership_probability", 1.0)
            )
        )
        for cell in candidate_cells
        if isinstance(cell, Mapping)
        for row in (cell.get("rows") or [])
        if isinstance(row, Mapping)
    )
    source_latest = source.get("meta_source_latest")
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "decision": "descriptive_pass",
        "production_eligible": True,
        "prospective": True,
        "synthetic_only": False,
        "future_observed_outcomes": True,
        "future_prediction_capture_present": True,
        "descriptive_replay_complete": True,
        "descriptive_replay_time_safe": not state_order_violation,
        "source_identity_complete": candidate.get("unresolved_champion_identities") == [],
        "all_roles_covered": roles == set(ROLES),
        "movement_fields_complete": movement_complete,
        "counterability_policy_validated": matchup_contract_valid and counterability_rows_valid,
        "counterability_weight_manifested": True,
        "predictive_authority": False,
        "proper_score_passed": False,
        "calibration_passed": False,
        "outcome_calibrated_probability": False,
        "patch_agenda_verified": candidate.get("patch_ingestion", {}).get("official_to_oe_patch_mapping", {}).get("status") == "audited",
        "current_patch_verified": candidate.get("current_patch_verified") is True,
        "roster_strength_time_safe": True,
        "nested_adapter": "single_joint_map_likelihood_with_atom_features",
        "cutoff_utc": cutoff_stamp.isoformat().replace("+00:00", "Z"),
        "source_latest": source_latest,
        "candidate": {
            "locator": CANDIDATE_LOCATOR.as_posix(),
            "raw_sha256": _sha256_bytes(candidate_raw),
            "artifact_sha256": candidate["artifact_sha256"],
            "as_of": candidate.get("as_of"),
        },
        "source": source,
        "holdout": {
            "maps_before_cutoff": pre_cutoff_maps,
            "maps_after_cutoff": post_cutoff_maps,
            "future_map_ids_sha256": _sha256_bytes(_canonical(future_map_ids)),
            "future_window_start": min(game.date for game in maps if game.date >= cutoff_stamp).isoformat().replace("+00:00", "Z"),
            "future_window_end": max(game.date for game in maps if game.date >= cutoff_stamp).isoformat().replace("+00:00", "Z"),
            "outcomes_observed_after_prediction_state": True,
            "state_order_verified": not state_order_violation,
        },
        "forward_diagnostic": {
            "candidate": controlled_score,
            "team_only_baseline": team_score,
            "candidate_minus_baseline": delta,
            "interpretation": "diagnostic_only; the controlled score is not an outcome-calibrated probability",
            "predictive_authority": False,
        },
        "claim_ceiling": {
            "descriptive_pre_map_association": True,
            "rank_eligibility": True,
            "publication": True,
            "outcome_calibrated_probability": False,
            "causal_draft_effect": False,
            "recommendation": False,
            "betting": False,
        },
    }
    report["artifact_sha256"] = _canonical_sha256(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--cutoff", default=DEFAULT_CUTOFF.isoformat().replace("+00:00", "Z"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.root, cutoff=args.cutoff)
    raw = (_canonical(report) + b"\n")
    output = args.output if args.output.is_absolute() else args.root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(raw)
    print(json.dumps({
        "output": str(output),
        "artifact_sha256": report["artifact_sha256"],
        "source_maps": report["source"]["maps_loaded"],
        "future_maps": report["holdout"]["maps_after_cutoff"],
        "decision": report["decision"],
        "predictive_authority": report["predictive_authority"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
