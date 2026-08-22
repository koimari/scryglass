"""Build the full, source-bound current-rating research artifact.

This command replays the current sequential team and player ratings over the
complete model-eligible census.  It scores every UTC timestamp batch before it
applies that batch's results.  The output is a research trust root.  It cannot
publish ratings or grant public authority.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import math
import re
import time
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from lol_kills.etl.competition import canonicalize_competition_frame
from lol_kills.research.future_value_rating import (
    CURRENT_RATING_SIGNED_MAP_FEATURES,
    _canonical_json_bytes,
    _map_model_frame,
    _sha256_path,
    _utc_text,
    rating_feature_values_sha256,
    validate_future_value_source_receipt_payload,
)
from lol_kills.research.future_value_rating_ledger import (
    CurrentRatingLedgerError,
    DualEloConfig,
    PlayerEloConfig,
    _ROLE_ORDER,
    _EXPECTED_ROLES,
    _artifact_digest,
    _as_ids,
    _frame_digest,
    _game_ids,
    _identity_text,
    _identity_text as _stable_identity_text,
    _implementation_hash as _ledger_implementation_hash,
    _series_ids,
    _stable_lineup_hashes,
    _stable_player_lineups,
    _stable_team_ids_by_game_side,
    _validate_source_frames,
    _norm_role,
    _is_intl,
    _team_replay,
    _player_replay,
)
from lol_kills.ratings.dual_elo import (
    TeamState,
    _append_momentum as _append_team_momentum,
    _is_intl as _team_is_intl,
    _momentum_residual as _team_momentum_residual,
    expected_score,
    total_mu as team_total_mu,
)
from lol_kills.ratings.player_elo import (
    PlayerState,
    _aggregate,
    _append_momentum as _append_player_momentum,
    is_team_affiliation_league,
    player_attribution_multipliers,
    total_mu as player_total_mu,
)
from lol_kills.v2.tierlists.accepted_census import identity_sha256


LEDGER_SCHEMA_VERSION = "scryglass:future-value-current-rating-ledger:v2"
LEDGER_RECEIPT_SCHEMA_VERSION = (
    "scryglass:future-value-current-rating-ledger-receipt:v2"
)
SNAPSHOT_RECEIPT_SCHEMA_VERSION = "scryglass:current-rating-snapshot-receipt:v1"
IMPLEMENTATION_LOCATOR = "benchmarks/build_full_current_rating_trust.py"
AUTHORITY = {
    "research_only": True,
    "public_player_rating": False,
    "public_team_rating": False,
    "public_probability": False,
    "promotion": False,
    "merge": False,
    "deployment": False,
    "betting": False,
}


class FullCurrentRatingTrustError(ValueError):
    """The full current-rating trust root cannot be proved."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _snapshot_canonical_json_bytes(value: object) -> bytes:
    """Use the canonical JSON form shared by snapshot consumers."""

    return (
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FullCurrentRatingTrustError(f"{label} is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise FullCurrentRatingTrustError(f"{label} cannot be read") from error
    if not isinstance(value, dict):
        raise FullCurrentRatingTrustError(f"{label} must be a JSON object")
    return value


def _write_json(path: Path, value: object) -> None:
    if path.exists() or path.is_symlink():
        raise FullCurrentRatingTrustError(f"output already exists: {path}")
    path.write_bytes(
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
        .encode("utf-8")
        + b"\n"
    )


def _safe_output_root(path: Path) -> Path:
    if path.exists() and path.is_symlink():
        raise FullCurrentRatingTrustError("output root must not be a symlink")
    if path.exists() and not path.is_dir():
        raise FullCurrentRatingTrustError("output root must be a directory")
    path.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or any(path.iterdir()):
        raise FullCurrentRatingTrustError("output root must be safe and empty")
    return path.resolve()


def _read_source(paths: Mapping[str, Path]) -> dict[str, pd.DataFrame]:
    try:
        frames = {label: pd.read_parquet(path) for label, path in paths.items()}
    except Exception as error:
        raise FullCurrentRatingTrustError("source parquet cannot be read") from error
    return frames


def _source_file_path(
    receipt_path: Path,
    source_root: Path,
    label: str,
    record: Mapping[str, Any],
) -> Path:
    """Resolve one receipt file and require its bytes to be present.

    Receipt records can use an absolute ``path`` or a locator relative to the
    receipt directory.  The source parquet records also have a fixed name
    under ``source_root``.  This keeps a valid receipt from being paired with
    an unrelated source directory.
    """

    raw_path = record.get("path")
    raw_locator = record.get("locator")
    if isinstance(raw_path, str) and raw_path.strip():
        candidate = Path(raw_path)
    elif isinstance(raw_locator, str) and raw_locator.strip():
        locator = Path(raw_locator)
        if locator.is_absolute():
            raise FullCurrentRatingTrustError(f"source file locator is absolute: {label}")
        candidate = receipt_path.parent / locator
    else:
        raise FullCurrentRatingTrustError(f"source file locator is missing: {label}")
    if not candidate.parts or ".." in candidate.parts:
        raise FullCurrentRatingTrustError(f"source file locator is unsafe: {label}")
    if candidate.is_symlink() or not candidate.is_file():
        raise FullCurrentRatingTrustError(f"source file is missing or unsafe: {label}")
    resolved = candidate.resolve()
    if resolved.is_symlink() or not resolved.is_file():
        raise FullCurrentRatingTrustError(f"source file is missing or unsafe: {label}")
    if label in {"maps", "players", "teams"}:
        expected_names = {
            "maps": ("maps.parquet",),
            "players": ("oe_player_games.parquet", "players.parquet"),
            "teams": ("oe_team_games.parquet", "teams.parquet"),
        }[label]
        expected = {
            (source_root / name).resolve() for name in expected_names
        }
        if resolved not in expected:
            raise FullCurrentRatingTrustError(
                f"source {label} file is outside the supplied source root"
            )
    declared_bytes = record.get("bytes")
    declared_sha = str(record.get("sha256") or "").lower()
    if (
        isinstance(declared_bytes, bool)
        or not isinstance(declared_bytes, int)
        or declared_bytes <= 0
        or re.fullmatch(r"[0-9a-f]{64}", declared_sha) is None
    ):
        raise FullCurrentRatingTrustError(f"source file binding is invalid: {label}")
    if resolved.stat().st_size != declared_bytes:
        raise FullCurrentRatingTrustError(f"source file bytes changed: {label}")
    if _sha256_path(resolved) != declared_sha:
        raise FullCurrentRatingTrustError(f"source file hash changed: {label}")
    return resolved


def _verify_source_receipt(
    path: Path,
    *,
    source_root: Path,
    expected_file_sha256: str,
    expected_receipt_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    expected_file_sha256 = str(expected_file_sha256 or "").lower()
    if re.fullmatch(r"[0-9a-f]{64}", expected_file_sha256) is None:
        raise FullCurrentRatingTrustError("source receipt file SHA-256 is required")
    actual_file_sha256 = _sha256_path(path)
    if actual_file_sha256 != expected_file_sha256:
        raise FullCurrentRatingTrustError("source receipt file hash changed")
    expected_receipt_sha256 = str(expected_receipt_sha256 or "").lower()
    if re.fullmatch(r"[0-9a-f]{64}", expected_receipt_sha256) is None:
        raise FullCurrentRatingTrustError(
            "independent source receipt hash is required"
        )
    receipt = _load_json(path, "source receipt")
    try:
        validate_future_value_source_receipt_payload(
            receipt,
            expected_receipt_sha256=expected_receipt_sha256,
        )
    except Exception as error:
        raise FullCurrentRatingTrustError(f"source receipt is invalid: {error}") from error
    source_files = receipt.get("source_files")
    if not isinstance(source_files, Mapping):
        raise FullCurrentRatingTrustError("source file bindings are missing")
    verified_paths: dict[str, Path] = {}
    for label, record in source_files.items():
        if not isinstance(label, str) or not isinstance(record, Mapping):
            raise FullCurrentRatingTrustError("source file binding is invalid")
        verified_paths[label] = _source_file_path(path, source_root, label, record)
    if not {"maps", "players", "teams"}.issubset(verified_paths):
        raise FullCurrentRatingTrustError("required source file bindings are missing")
    return receipt, verified_paths


def _normalized_alias(value: Any) -> str:
    """Return one deterministic display alias for fallback identity keys."""

    return " ".join(_identity_text(value).casefold().split())


def _fallback_identity(kind: str, *parts: str) -> str:
    normalized = [str(value).strip() for value in parts]
    if not normalized or any(not value for value in normalized):
        raise FullCurrentRatingTrustError(f"{kind} fallback identity is incomplete")
    digest = _sha256_bytes(_canonical_json_bytes(normalized))
    return f"fallback:{kind}:{digest}"


def _valid_stable_identity(value: Any, prefix: str) -> str:
    text = _identity_text(value)
    if text and not text.startswith(prefix):
        raise FullCurrentRatingTrustError("source contains an invalid stable identity")
    return text


def _prepare_replay_identities(
    players: pd.DataFrame,
    teams: pd.DataFrame,
    eligible_ids: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fill missing replay keys with game-scoped, non-authoritative IDs."""

    player_required = {"side", "position", "playerid", "playername", "teamid", "teamname"}
    team_required = {"side", "teamid", "teamname"}
    if not player_required.issubset(players.columns):
        raise FullCurrentRatingTrustError("stable player and team identity columns are required")
    if not team_required.issubset(teams.columns):
        raise FullCurrentRatingTrustError("stable team identity columns are required")

    player_work = players.copy()
    team_work = teams.copy()
    player_work["__gid"] = _game_ids(player_work, "players").astype(str).to_numpy()
    team_work["__gid"] = _game_ids(team_work, "teams").astype(str).to_numpy()
    eligible = set(eligible_ids)
    player_work = player_work.loc[player_work["__gid"].isin(eligible)].copy()
    team_work = team_work.loc[team_work["__gid"].isin(eligible)].copy()
    player_work["__side"] = player_work["side"].astype("string").str.strip().str.title()
    team_work["__side"] = team_work["side"].astype("string").str.strip().str.title()

    player_names = player_work["playername"].map(_normalized_alias)
    team_names = pd.concat(
        [player_work["teamname"], team_work["teamname"]], ignore_index=True
    ).map(_normalized_alias)
    if player_names.eq("").any() or team_names.eq("").any():
        raise FullCurrentRatingTrustError("display identity is incomplete")

    audit: dict[str, Any] = {
        "policy": "stable_id_or_game_scoped_deterministic_fallback",
        "resolved_scope": "exact_model_eligible_census_only",
        "player_rows_total": int(len(player_work)),
        "team_identity_rows_total": int(len(player_work) + len(team_work)),
        "player_rows_source_stable": 0,
        "player_rows_fallback": 0,
        "team_rows_source_stable": 0,
        "team_rows_resolved_stable": 0,
        "team_rows_fallback": 0,
    }

    resolved_team_by_slot: dict[tuple[str, str], str] = {}
    for (gid, side), player_group in player_work.groupby(["__gid", "__side"], sort=False):
        if side not in {"Blue", "Red"}:
            raise FullCurrentRatingTrustError(f"team side is invalid for {gid}")
        team_group = team_work.loc[
            team_work["__gid"].eq(str(gid)) & team_work["__side"].eq(str(side))
        ]
        if len(team_group) != 1:
            raise FullCurrentRatingTrustError(f"team row identity is incomplete for {gid} {side}")
        source_ids = {
            stable
            for stable in (
                _valid_stable_identity(value, "oe:team:")
                for value in [*player_group["teamid"].tolist(), *team_group["teamid"].tolist()]
            )
            if stable
        }
        if len(source_ids) > 1:
            raise FullCurrentRatingTrustError(f"team stable identities conflict for {gid} {side}")
        aliases = {
            _normalized_alias(value)
            for value in [*player_group["teamname"].tolist(), *team_group["teamname"].tolist()]
        }
        if len(aliases) != 1:
            raise FullCurrentRatingTrustError(f"team display identities conflict for {gid} {side}")
        alias = next(iter(aliases))
        if source_ids:
            resolved = next(iter(source_ids))
        else:
            resolved = _fallback_identity("team", str(gid), str(side), alias)
        resolved_team_by_slot[(str(gid), str(side))] = resolved
        original_values = [
            *player_group["teamid"].tolist(),
            *team_group["teamid"].tolist(),
        ]
        player_work.loc[player_group.index, "teamid"] = resolved
        team_work.loc[team_group.index, "teamid"] = resolved
        source_count = sum(
            bool(_identity_text(value).startswith("oe:team:"))
            for value in original_values
        )
        missing_count = len(original_values) - source_count
        audit["team_rows_source_stable"] += source_count
        audit["team_rows_resolved_stable"] += len(original_values) if source_ids else 0
        audit["team_rows_fallback"] += len(original_values) if not source_ids else 0

    resolved_player_ids: list[str] = []
    for values in player_work.to_dict(orient="records"):
        source_id = _valid_stable_identity(values.get("playerid"), "oe:player:")
        alias = _normalized_alias(values.get("playername"))
        if source_id:
            resolved = source_id
            audit["player_rows_source_stable"] += 1
        else:
            gid = str(values.get("__gid"))
            side = str(values.get("__side"))
            team_id = resolved_team_by_slot.get((gid, side))
            role = str(_norm_role(values.get("position")))
            resolved = _fallback_identity(
                "player", gid, side, role, alias, str(team_id or "")
            )
            audit["player_rows_fallback"] += 1
        resolved_player_ids.append(resolved)
    player_work["playerid"] = resolved_player_ids

    digest_rows = []
    for row in player_work[["__gid", "__side", "position", "playerid", "teamid"]].itertuples(
        index=False, name=None
    ):
        digest_rows.append(
            {
                "game_id": str(row[0]),
                "side": str(row[1]),
                "role": str(_norm_role(row[2])),
                "player_id": str(row[3]),
                "team_id": str(row[4]),
            }
        )
    digest_rows.sort(key=lambda value: (value["game_id"], value["side"], value["role"]))
    audit["resolution_sha256"] = _sha256_bytes(_canonical_json_bytes(digest_rows))
    return (
        player_work.drop(columns=["__gid", "__side"]),
        team_work.drop(columns=["__gid", "__side"]),
        audit,
    )


def _validate_replay_source_frames(
    maps: pd.DataFrame,
    players: pd.DataFrame,
    teams: pd.DataFrame,
    eligible_ids: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply the shared grain checks while retaining labeled fallback keys."""

    validation_players = players.copy()
    validation_teams = teams.copy()
    validation_players["playerid"] = validation_players["playerid"].map(
        lambda value: (
            "oe:player:" + _identity_text(value).removeprefix("fallback:player:")
            if _identity_text(value).startswith("fallback:player:")
            else value
        )
    )
    for frame in (validation_players, validation_teams):
        frame["teamid"] = frame["teamid"].map(
            lambda value: (
                "oe:team:" + _identity_text(value).removeprefix("fallback:team:")
                if _identity_text(value).startswith("fallback:team:")
                else value
            )
        )
    map_frame, _validated_players, _validated_teams = _validate_source_frames(
        maps,
        validation_players,
        validation_teams,
        eligible_ids,
        eligible_ids,
    )
    eligible = set(eligible_ids)
    player_frame = players.copy()
    player_frame["__game_id"] = _game_ids(player_frame, "players").astype(str).to_numpy()
    player_frame = player_frame.loc[player_frame["__game_id"].isin(eligible)].copy()
    team_frame = teams.copy()
    team_frame["__game_id"] = _game_ids(team_frame, "teams").astype(str).to_numpy()
    team_frame = team_frame.loc[team_frame["__game_id"].isin(eligible)].copy()
    return map_frame, player_frame, team_frame


def _series_for_eligible(maps: pd.DataFrame, eligible_ids: tuple[str, ...]) -> pd.Series:
    try:
        values = _series_ids(maps, _game_ids(maps, "maps"))
    except Exception as error:
        raise FullCurrentRatingTrustError(f"series identity cannot be derived: {error}") from error
    by_id = pd.Series(values.to_numpy(), index=_game_ids(maps, "maps").astype(str))
    result = by_id.reindex(list(eligible_ids))
    if result.isna().any() or result.astype(str).str.strip().eq("").any():
        raise FullCurrentRatingTrustError("series identity is incomplete for eligible maps")
    return result.astype(str)


def _player_snapshot_replay(
    maps: pd.DataFrame,
    players: pd.DataFrame,
    *,
    cfg: PlayerEloConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay player states with timestamp-batch updates and stable IDs."""

    source_maps = maps.copy()
    source_players = players.copy()
    frame = canonicalize_competition_frame(source_maps).copy()
    frame["__game_id"] = _game_ids(frame, "maps").astype(str).to_numpy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce").dt.tz_localize(None)
    frame = frame.sort_values(["date", "__game_id"], kind="mergesort").reset_index(drop=True)
    stable_lineups = _stable_player_lineups(source_players)
    team_ids = _stable_team_ids_by_game_side(source_players)
    name_by_id: dict[str, str] = {}
    for row in source_players[["playerid", "playername"]].itertuples(index=False, name=None):
        name_by_id.setdefault(_identity_text(row[0]), _identity_text(row[1]))
    player_work = source_players.copy()
    player_work["__gid"] = _game_ids(player_work, "players").astype(str).to_numpy()
    player_work["__side"] = player_work["side"].astype("string").str.strip().str.title()
    player_work["__role"] = player_work["position"].map(_norm_role)
    player_work["__player_id"] = player_work["playerid"].map(_identity_text)
    stable_by_slot = {
        (str(gid), str(side), str(role)): str(player_id)
        for gid, side, role, player_id in player_work[
            ["__gid", "__side", "__role", "__player_id"]
        ].itertuples(index=False, name=None)
    }
    from lol_kills.ratings.player_elo import _lineups_by_game

    _name_lineups, attribution_metrics = _lineups_by_game(
        source_players, with_metrics=True
    )
    metric_ids: list[str] = []
    for row in attribution_metrics[["_gid", "side", "_role"]].itertuples(index=False, name=None):
        key = (str(row[0]), str(row[1]), str(row[2]))
        player_id = stable_by_slot.get(key)
        if not player_id:
            raise FullCurrentRatingTrustError("player attribution identity is unresolved")
        metric_ids.append(player_id)
    attribution_metrics = attribution_metrics.copy()
    attribution_metrics["_name"] = metric_ids
    attribution, _stats = player_attribution_multipliers(attribution_metrics, cfg)
    states: dict[str, PlayerState] = {}
    rows: list[dict[str, Any]] = []
    recent_mus: dict[str, list[float]] = {}
    for stamp, batch in frame.groupby("date", sort=False, dropna=False):
        seen: set[str] = set()
        for raw in batch.to_dict(orient="records"):
            gid = str(raw["__game_id"])
            blue = stable_lineups[gid]["Blue"]
            red = stable_lineups[gid]["Red"]
            blue_team = team_ids[(gid, "Blue")]
            red_team = team_ids[(gid, "Red")]
            if blue_team == red_team:
                raise FullCurrentRatingTrustError("player lineup has duplicate team identity")
            for player_id, _role in [*blue, *red]:
                if player_id in seen:
                    raise FullCurrentRatingTrustError("player appears in multiple maps at one timestamp")
                seen.add(player_id)
                state = states.setdefault(player_id, PlayerState(sigma=cfg.sigma0))
                if pd.notna(stamp) and state.last_date is not None:
                    months = max((pd.Timestamp(stamp) - state.last_date).days / 30.0, 0.0)
                    state.sigma = min(160.0, state.sigma + cfg.sigma_month_inflate * months)
                team_now = blue_team if any(item[0] == player_id for item in blue) else red_team
                if state.last_team and state.last_team != team_now:
                    state.sigma = min(160.0, state.sigma + cfg.team_switch_sigma_bump)
                states[player_id] = state
        pending: list[tuple[dict[str, Any], str, str, float, float, float, float]] = []
        for raw in batch.to_dict(orient="records"):
            gid = str(raw["__game_id"])
            blue = stable_lineups[gid]["Blue"]
            red = stable_lineups[gid]["Red"]
            blue_team = team_ids[(gid, "Blue")]
            red_team = team_ids[(gid, "Red")]
            base_b, sig_b, _known_b, _ = _aggregate(states, blue, cfg, include_momentum=False)
            base_r, sig_r, _known_r, _ = _aggregate(states, red, cfg, include_momentum=False)
            mu_b, _, _, _ = _aggregate(states, blue, cfg)
            mu_r, _, _, _ = _aggregate(states, red, cfg)
            sig = math.hypot(sig_b, sig_r)
            p_base = expected_score(base_b, base_r)
            p = expected_score(mu_b, mu_r)
            shrink = 1.0 / (1.0 + (sig / 130.0) ** 2)
            p_shrunk = 0.5 + (p - 0.5) * shrink
            rows.append(
                {
                    "game_id": gid,
                    "date": pd.Timestamp(raw["date"], tz="UTC"),
                    "base_player_logit": float(np.log(p_shrunk / (1.0 - p_shrunk))),
                    "player_rating_diff_scaled": float((mu_b - mu_r) / 400.0),
                }
            )
            y = raw.get("y_blue_win")
            if pd.isna(y):
                continue
            y_value = float(y)
            if y_value not in (0.0, 1.0):
                raise FullCurrentRatingTrustError(f"training outcome is invalid: {gid}")
            gdiff = raw.get("blue_golddiffat15")
            if pd.isna(gdiff):
                gdiff = raw.get("blue_golddiffat10")
            length = raw.get("length_min")
            if pd.isna(length):
                length = float(raw["gamelength"]) / 60.0 if pd.notna(raw.get("gamelength")) else 30.0
            mov = 1.0 if pd.isna(gdiff) else 1.0 + cfg.mov_scale * math.tanh(float(gdiff) / (200.0 * max(float(length), 1.0)))
            pending.append((raw, blue_team, red_team, p, p_base, mov, y_value))
        for raw, blue_team, red_team, p, p_base, mov, y_value in pending:
            gid = str(raw["__game_id"])
            blue = stable_lineups[gid]["Blue"]
            red = stable_lineups[gid]["Red"]
            intl = _is_intl(str(raw.get("league") or ""), raw.get("tournament"))
            for player_id, _role in blue:
                state = states[player_id]
                k_scale = state.sigma / cfg.sigma0
                multiplier = attribution.get((gid, "Blue", player_id), 1.0)
                delta = cfg.k_meta if intl else cfg.k_regional
                state.mu_meta += delta * k_scale * mov * (y_value - p) * multiplier if intl else 0.0
                state.mu_regional += delta * k_scale * mov * (y_value - p) * multiplier if not intl else 0.0
                state.sigma = max(cfg.sigma_min, state.sigma * 0.985)
                state.n_maps += 1
                state.last_date = pd.Timestamp(raw["date"]) if pd.notna(raw.get("date")) else state.last_date
                state.last_team = blue_team
                if is_team_affiliation_league(str(raw.get("league") or "")):
                    state.home_league = str(raw.get("league") or "")
                _append_player_momentum(state, y_value - p_base, cfg)
            for player_id, _role in red:
                state = states[player_id]
                k_scale = state.sigma / cfg.sigma0
                multiplier = attribution.get((gid, "Red", player_id), 1.0)
                delta = cfg.k_meta if intl else cfg.k_regional
                red_delta = (1.0 - y_value) - (1.0 - p)
                state.mu_meta += delta * k_scale * mov * red_delta * multiplier if intl else 0.0
                state.mu_regional += delta * k_scale * mov * red_delta * multiplier if not intl else 0.0
                state.sigma = max(cfg.sigma_min, state.sigma * 0.985)
                state.n_maps += 1
                state.last_date = pd.Timestamp(raw["date"]) if pd.notna(raw.get("date")) else state.last_date
                state.last_team = red_team
                if is_team_affiliation_league(str(raw.get("league") or "")):
                    state.home_league = str(raw.get("league") or "")
                _append_player_momentum(
                    state, (1.0 - y_value) - (1.0 - p_base), cfg
                )
            for player_id in [item[0] for item in [*blue, *red]]:
                recent_mus.setdefault(player_id, []).append(player_total_mu(states[player_id]))
                recent_mus[player_id] = recent_mus[player_id][-10:]
    snapshot_rows: list[dict[str, Any]] = []
    for player_id, state in states.items():
        history = recent_mus.get(player_id, [])
        stability = None
        if len(history) > 1:
            stability = float(np.mean(np.abs(np.diff(history))))
        snapshot_rows.append(
            {
                "player": name_by_id.get(player_id, player_id),
                "player_id": player_id,
                "team_id": f"{state.last_team}" if state.last_team else None,
                "mu_base_total": player_total_mu(state),
                "mu_total": player_total_mu(state) + cfg.momentum_scale * (float(np.mean(state.momentum_history[-cfg.momentum_window_games :])) if cfg.momentum_window_games and state.momentum_history else 0.0),
                "mu_effective": player_total_mu(state) + cfg.momentum_scale * (float(np.mean(state.momentum_history[-cfg.momentum_window_games :])) if cfg.momentum_window_games and state.momentum_history else 0.0),
                "momentum_residual": float(np.mean(state.momentum_history[-cfg.momentum_window_games :])) if cfg.momentum_window_games and state.momentum_history else 0.0,
                "mu_regional": state.mu_regional,
                "mu_meta": state.mu_meta,
                "sigma": state.sigma,
                "n_maps": state.n_maps,
                "last_team": state.last_team,
                "home_league": state.home_league,
                "last_game_date": state.last_date.isoformat() if state.last_date is not None else None,
                "evidence_stability": stability,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(snapshot_rows).sort_values("mu_effective", ascending=False, kind="mergesort").reset_index(drop=True)


def _team_snapshot_replay(
    maps: pd.DataFrame,
    players: pd.DataFrame,
    *,
    cfg: DualEloConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay team states with the same timestamp policy as the feature ledger."""

    frame = canonicalize_competition_frame(maps).copy()
    frame["__game_id"] = _game_ids(frame, "maps").astype(str).to_numpy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce").dt.tz_localize(None)
    frame = frame.sort_values(["date", "__game_id"], kind="mergesort").reset_index(drop=True)
    team_ids = _stable_team_ids_by_game_side(players)
    lineups = _stable_lineup_hashes(players)
    states: dict[str, TeamState] = defaultdict(lambda: TeamState(sigma=cfg.sigma0))
    map_counts: dict[str, int] = defaultdict(int)
    home_leagues: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    for stamp, batch in frame.groupby("date", sort=False, dropna=False):
        seen_teams: set[str] = set()
        for raw in batch.to_dict(orient="records"):
            gid = str(raw["__game_id"])
            for side in ("Blue", "Red"):
                team = team_ids[(gid, side)]
                if team in seen_teams:
                    raise FullCurrentRatingTrustError("team appears in multiple maps at one timestamp")
                seen_teams.add(team)
                state = states[team]
                if pd.notna(stamp) and state.last_date is not None:
                    months = max((pd.Timestamp(stamp) - state.last_date).days / 30.0, 0.0)
                    state.sigma = min(150.0, state.sigma + cfg.sigma_month_inflate * months)
                lineup_hash = lineups.get(f"{gid}|{team}")
                if lineup_hash and state.lineup_hash and lineup_hash != state.lineup_hash:
                    state.sigma = min(150.0, state.sigma + cfg.roster_sigma_bump)
                if lineup_hash:
                    state.lineup_hash = lineup_hash
        pending: list[tuple[dict[str, Any], str, str, float, float, float, float]] = []
        for raw in batch.to_dict(orient="records"):
            gid = str(raw["__game_id"])
            blue, red = team_ids[(gid, "Blue")], team_ids[(gid, "Red")]
            sb, sr = states[blue], states[red]
            base_b, base_r = team_total_mu(sb), team_total_mu(sr)
            mb = cfg.momentum_scale * _team_momentum_residual(sb, cfg)
            mr = cfg.momentum_scale * _team_momentum_residual(sr, cfg)
            mu_b, mu_r = base_b + mb, base_r + mr
            sig = math.hypot(sb.sigma, sr.sigma)
            p_base = expected_score(base_b, base_r)
            p = expected_score(mu_b, mu_r)
            shrink = 1.0 / (1.0 + (sig / 120.0) ** 2)
            p_shrunk = 0.5 + (p - 0.5) * shrink
            rows.append({
                "game_id": gid,
                "date": pd.Timestamp(raw["date"], tz="UTC"),
                "base_team_logit": float(np.log(p_shrunk / (1.0 - p_shrunk))),
                "team_rating_diff_scaled": float((mu_b - mu_r) / 400.0),
            })
            y = raw.get("y_blue_win")
            if pd.isna(y):
                continue
            y_value = float(y)
            gdiff = raw.get("blue_golddiffat15")
            if pd.isna(gdiff):
                gdiff = raw.get("blue_golddiffat10")
            length = raw.get("length_min")
            if pd.isna(length):
                length = float(raw["gamelength"]) / 60.0 if pd.notna(raw.get("gamelength")) else 30.0
            mov = 1.0 if pd.isna(gdiff) else 1.0 + cfg.mov_scale * math.tanh(float(gdiff) / (200.0 * max(float(length), 1.0)))
            pending.append((raw, blue, red, p, p_base, mov, y_value))
        for raw, blue, red, p, p_base, mov, y_value in pending:
            sb, sr = states[blue], states[red]
            intl = _team_is_intl(str(raw.get("league") or ""), raw.get("tournament"))
            if intl:
                sb.mu_meta += cfg.k_meta * mov * (y_value - p)
                sr.mu_meta += cfg.k_meta * mov * ((1.0 - y_value) - (1.0 - p))
            else:
                sb.mu_regional += cfg.k_regional * mov * (y_value - p)
                sr.mu_regional += cfg.k_regional * mov * ((1.0 - y_value) - (1.0 - p))
            _append_team_momentum(sb, y_value - p_base, cfg)
            _append_team_momentum(sr, (1.0 - y_value) - (1.0 - p_base), cfg)
            sb.sigma = max(cfg.sigma_min, sb.sigma * 0.98)
            sr.sigma = max(cfg.sigma_min, sr.sigma * 0.98)
            if pd.notna(raw.get("date")):
                sb.last_date = pd.Timestamp(raw["date"])
                sr.last_date = pd.Timestamp(raw["date"])
            map_counts[blue] += 1
            map_counts[red] += 1
            league = str(raw.get("league") or "")
            if is_team_affiliation_league(league):
                home_leagues[blue] = league
                home_leagues[red] = league
    snapshot_rows = []
    for team_id, state in states.items():
        residual = _team_momentum_residual(state, cfg)
        snapshot_rows.append({
            "team": team_id,
            "team_key": team_id,
            "team_id": team_id,
            "mu_base_total": team_total_mu(state),
            "mu_total": team_total_mu(state) + cfg.momentum_scale * residual,
            "mu_effective": team_total_mu(state) + cfg.momentum_scale * residual,
            "momentum_residual": residual,
            "mu_regional": state.mu_regional,
            "mu_meta": state.mu_meta,
            "sigma": state.sigma,
            "rating_p10": team_total_mu(state) + cfg.momentum_scale * residual,
            "n_series": map_counts[team_id],
            "n_maps": map_counts[team_id],
            "international_series": 0,
            "home_league": home_leagues.get(team_id, "UNKNOWN"),
            "last_game_date": state.last_date.isoformat() if state.last_date is not None else None,
            "model": "sequential_timestamp_batch",
        })
    return pd.DataFrame(rows), pd.DataFrame(snapshot_rows).sort_values("mu_effective", ascending=False, kind="mergesort").reset_index(drop=True)


def _snapshot_schema_digest(frame: pd.DataFrame) -> str:
    payload = {
        "columns": [str(c) for c in frame.columns],
        "dtypes": {str(c): str(frame[c].dtype) for c in frame.columns},
    }
    return _sha256_bytes(_snapshot_canonical_json_bytes(payload))


def _snapshot_value_digest(frame: pd.DataFrame, identity_column: str) -> str:
    rows = []
    ordered = frame[[identity_column, "mu_effective"]].copy()
    ordered[identity_column] = ordered[identity_column].astype(str)
    ordered = ordered.sort_values(identity_column, kind="mergesort")
    for identity, value in ordered.itertuples(index=False, name=None):
        number = float(value)
        if not math.isfinite(number):
            raise FullCurrentRatingTrustError("snapshot contains a non-finite rating")
        rows.append({identity_column: str(identity), "mu_effective": number})
    return _sha256_bytes(_snapshot_canonical_json_bytes(rows))


def _snapshot_record(root: Path, frame: pd.DataFrame, relative: str, kind: str) -> dict[str, Any]:
    path = root / relative
    if frame.empty:
        raise FullCurrentRatingTrustError(f"{kind} stable snapshot is empty")
    identity = "player_id" if kind == "player" else "team_id"
    ids = frame[identity].astype("string")
    if ids.isna().any() or ids.str.strip().eq("").any() or ids.duplicated().any():
        raise FullCurrentRatingTrustError(f"{kind} snapshot stable identity is incomplete")
    expected_prefix = "oe:player:" if kind == "player" else "oe:team:"
    if not ids.str.startswith(expected_prefix, na=False).all():
        raise FullCurrentRatingTrustError(f"{kind} snapshot contains fallback identities")
    if kind == "player":
        team_ids = frame["team_id"].astype("string")
        if not team_ids.str.startswith("oe:team:", na=False).all():
            raise FullCurrentRatingTrustError(
                "player snapshot contains fallback team identities"
            )
    values = pd.to_numeric(frame["mu_effective"], errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise FullCurrentRatingTrustError(f"{kind} snapshot values are non-finite")
    return {
        "locator": relative,
        "bytes": int(path.stat().st_size),
        "sha256": _sha256_path(path),
        "rows": int(len(frame)),
        "columns": [str(c) for c in frame.columns],
        "dtypes": {str(c): str(frame[c].dtype) for c in frame.columns},
        "schema_sha256": _snapshot_schema_digest(frame),
        "identity_column": identity,
        "value_column": "mu_effective",
        "value_digest_sha256": _snapshot_value_digest(frame, identity),
        "verified_rows": int(len(frame)),
    }


def _build_receipts(
    *,
    output_root: Path,
    source_receipt: Mapping[str, Any],
    source_receipt_path: Path,
    source_frame_hashes: Mapping[str, str],
    ledger: pd.DataFrame,
    series_by_game: pd.Series,
    elapsed_seconds: float,
    player_snapshot: pd.DataFrame,
    team_snapshot: pd.DataFrame,
    identity_audit: Mapping[str, Any],
) -> dict[str, Any]:
    ledger_path = output_root / "current-rating-ledger.parquet"
    ids = tuple(sorted(ledger["game_id"].astype(str)))
    source_ids = tuple(str(value) for value in source_receipt["model_eligible_game_ids"])
    series_ids = tuple(sorted(set(series_by_game.astype(str))))
    artifact_sha = _sha256_path(ledger_path)
    receipt: dict[str, Any] = {
        "schema_version": LEDGER_RECEIPT_SCHEMA_VERSION,
        "ledger_schema_version": LEDGER_SCHEMA_VERSION,
        "source_receipt_sha256": str(source_receipt["receipt_sha256"]),
        "source_as_of": str(source_receipt["source_as_of"]),
        "source_game_count": int(source_receipt["source_game_count"]),
        "source_identity_sha256": str(source_receipt["source_identity_sha256"]),
        "model_eligible_game_count": len(source_ids),
        "model_eligible_identity_sha256": str(source_receipt["model_eligible_identity_sha256"]),
        "model_eligible_game_ids": list(source_ids),
        "output_game_ids": list(ids),
        "output_game_count": len(ids),
        "output_game_identity_sha256": identity_sha256(ids),
        "train_game_ids": list(ids),
        "train_game_count": len(ids),
        "train_game_identity_sha256": identity_sha256(ids),
        "validation_game_ids": [],
        "validation_game_count": 0,
        "validation_game_identity_sha256": identity_sha256([]),
        "train_series_ids": list(series_ids),
        "train_series_count": len(series_ids),
        "train_series_identity_sha256": identity_sha256(series_ids),
        "validation_series_ids": [],
        "validation_series_count": 0,
        "validation_series_identity_sha256": identity_sha256([]),
        "series_disjoint": True,
        "series_partition_source": "conservative_series_superset",
        "series_partition_key_fields": ["league", "tournament", "unordered_team_pair"],
        "series_partition_receipt_file_sha256": None,
        "strict_prior_timing": "source_bound_current_rating_before_snapshot_as_of",
        "same_timestamp_policy": "score_full_utc_timestamp_batch_before_training_updates",
        "masked_nontraining_map_columns": [],
        "masked_nontraining_player_columns": [],
        "source_frame_sha256": dict(source_frame_hashes),
        "feature_names": list(CURRENT_RATING_SIGNED_MAP_FEATURES),
        "ledger_rows_sha256": _artifact_digest(ledger, CURRENT_RATING_SIGNED_MAP_FEATURES),
        "feature_value_digest": rating_feature_values_sha256(ledger, CURRENT_RATING_SIGNED_MAP_FEATURES),
        "implementation_locator": IMPLEMENTATION_LOCATOR,
        "implementation_sha256": _sha256_path(Path(__file__).resolve()),
        "ledger_implementation_locator": "lol_kills/research/future_value_rating_ledger.py",
        "ledger_implementation_sha256": _ledger_implementation_hash(),
        "artifact": {"path": str(ledger_path.resolve()), "bytes": int(ledger_path.stat().st_size), "sha256": artifact_sha},
        "artifact_path": str(ledger_path.resolve()),
        "artifact_bytes": int(ledger_path.stat().st_size),
        "artifact_sha256": artifact_sha,
        "rows": len(ledger),
        "expected_rows": len(source_ids),
        "complete_exact_five": len(source_ids),
        "elapsed_seconds": elapsed_seconds,
        "fit_window_end": str(source_receipt["source_as_of"]),
        "state_key_policy": {
            "team": "stable_oe_team_id_or_labeled_source_bound_fallback",
            "player": "stable_oe_player_id_or_labeled_source_bound_fallback",
            "fallback_continuity": "game_scoped_only",
            "identity_resolution_scope": "exact_model_eligible_census_only",
            "fallback_snapshot_policy": "excluded_from_player_and_team_rank_snapshots",
            "display_names": "metadata_only",
        },
        "identity_resolution": dict(identity_audit),
        "authority": dict(AUTHORITY),
        "source_receipt_file": {"path": str(source_receipt_path.resolve()), "bytes": source_receipt_path.stat().st_size, "sha256": _sha256_path(source_receipt_path)},
    }
    receipt["receipt_sha256"] = _sha256_bytes(_canonical_json_bytes(receipt))

    snapshots = {
        "player": _snapshot_record(output_root, player_snapshot, "player/player_ratings_snapshot.parquet", "player"),
        "team": _snapshot_record(output_root, team_snapshot, "team/ratings_snapshot.parquet", "team"),
    }
    snapshot_receipt: dict[str, Any] = {
        "schema_version": SNAPSHOT_RECEIPT_SCHEMA_VERSION,
        "source_receipt_sha256": str(source_receipt["receipt_sha256"]),
        "source_identity_sha256": str(source_receipt["source_identity_sha256"]),
        "source_as_of": str(source_receipt["source_as_of"]),
        "source_game_count": int(source_receipt["source_game_count"]),
        "snapshots": snapshots,
        "identity_resolution": dict(identity_audit),
        "snapshot_scope": "stable_verified_oe_identities_only",
        "authority": dict(AUTHORITY),
        "implementation_locator": IMPLEMENTATION_LOCATOR,
        "implementation_sha256": _sha256_path(Path(__file__).resolve()),
    }
    snapshot_receipt["receipt_sha256"] = _sha256_bytes(
        _snapshot_canonical_json_bytes(snapshot_receipt)
    )
    return {"ledger": receipt, "snapshots": snapshot_receipt}


def build_full_current_rating_trust(
    *,
    source_root: Path,
    source_receipt_path: Path,
    source_receipt_file_sha256: str,
    output_root: Path,
    expected_source_receipt_sha256: str,
) -> dict[str, Any]:
    """Build and bind the full model-eligible current rating trust root."""

    started = time.perf_counter()
    source_root = Path(source_root)
    if source_root.is_symlink() or not source_root.is_dir():
        raise FullCurrentRatingTrustError("source root is missing or unsafe")
    output_root = _safe_output_root(Path(output_root))
    source_receipt_path = Path(source_receipt_path)
    if source_receipt_path.is_symlink():
        raise FullCurrentRatingTrustError("source receipt is missing or unsafe")
    source_receipt, source_paths = _verify_source_receipt(
        source_receipt_path,
        source_root=source_root,
        expected_file_sha256=source_receipt_file_sha256,
        expected_receipt_sha256=expected_source_receipt_sha256,
    )
    frames = _read_source(
        {label: source_paths[label] for label in ("maps", "players", "teams")}
    )
    eligible_ids = tuple(str(value) for value in source_receipt["model_eligible_game_ids"])
    if tuple(sorted(eligible_ids)) != eligible_ids:
        raise FullCurrentRatingTrustError("model-eligible IDs are not canonically ordered")
    try:
        maps, players, teams = frames["maps"], frames["players"], frames["teams"]
        map_ids = _game_ids(maps, "maps").astype(str)
        if not set(eligible_ids).issubset(set(map_ids)) or map_ids.duplicated().any():
            raise FullCurrentRatingTrustError("maps do not contain the eligible census exactly")
        replay_players, replay_teams, identity_audit = _prepare_replay_identities(
            players, teams, eligible_ids
        )
        map_frame, player_frame, team_frame = _validate_replay_source_frames(
            maps, replay_players, replay_teams, eligible_ids
        )
    except CurrentRatingLedgerError as error:
        raise FullCurrentRatingTrustError(str(error)) from error
    series_by_game = _series_for_eligible(map_frame, eligible_ids)
    source_cutoff = pd.Timestamp(source_receipt["source_as_of"])
    map_dates = pd.to_datetime(map_frame["date"], utc=True, errors="coerce")
    if map_dates.isna().any() or map_dates.gt(source_cutoff).any():
        raise FullCurrentRatingTrustError(
            "eligible map dates exceed the verified source_as_of boundary"
        )
    map_frame["series_id"] = series_by_game.reindex(map_frame["__game_id"].astype(str)).to_numpy()
    source_hashes = {label: _frame_digest(frames[label], label) for label in ("maps", "players", "teams")}
    team_features, team_snapshot_all = _team_snapshot_replay(
        map_frame, player_frame, cfg=DualEloConfig()
    )
    player_features, player_snapshot_all = _player_snapshot_replay(
        map_frame, player_frame, cfg=PlayerEloConfig()
    )
    player_snapshot = player_snapshot_all.loc[
        player_snapshot_all["player_id"].astype("string").str.startswith(
            "oe:player:", na=False
        )
        & player_snapshot_all["team_id"].astype("string").str.startswith(
            "oe:team:", na=False
        )
    ].copy()
    team_snapshot = team_snapshot_all.loc[
        team_snapshot_all["team_id"].astype("string").str.startswith(
            "oe:team:", na=False
        )
    ].copy()
    identity_audit.update(
        {
            "player_states_total": int(len(player_snapshot_all)),
            "player_states_stable_verified": int(len(player_snapshot)),
            "player_states_fallback_excluded": int(
                len(player_snapshot_all) - len(player_snapshot)
            ),
            "team_states_total": int(len(team_snapshot_all)),
            "team_states_stable_verified": int(len(team_snapshot)),
            "team_states_fallback_excluded": int(
                len(team_snapshot_all) - len(team_snapshot)
            ),
        }
    )
    team_features["series_id"] = team_features["game_id"].astype(str).map(series_by_game)
    player_features["series_id"] = player_features["game_id"].astype(str).map(series_by_game)
    ledger = team_features.merge(player_features, on=["game_id", "date", "series_id"], how="inner", validate="one_to_one")
    ledger = ledger.sort_values(["date", "game_id"], kind="mergesort").reset_index(drop=True)
    if tuple(sorted(ledger["game_id"].astype(str))) != tuple(sorted(eligible_ids)):
        raise FullCurrentRatingTrustError("current rating ledger coverage changed")
    for feature in CURRENT_RATING_SIGNED_MAP_FEATURES:
        values = pd.to_numeric(ledger[feature], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise FullCurrentRatingTrustError(f"current rating feature is non-finite: {feature}")
        ledger[feature] = values
    (output_root / "player").mkdir()
    (output_root / "team").mkdir()
    ledger_path = output_root / "current-rating-ledger.parquet"
    ledger.to_parquet(ledger_path, index=False)
    team_features.to_parquet(output_root / "team/ratings.parquet", index=False)
    team_snapshot.to_parquet(output_root / "team/ratings_snapshot.parquet", index=False)
    team_snapshot.to_parquet(output_root / "team/ratings_dual_snapshot.parquet", index=False)
    player_features.to_parquet(output_root / "player/player_ratings.parquet", index=False)
    player_snapshot.to_parquet(output_root / "player/player_ratings_snapshot.parquet", index=False)
    cache = {"schema_version": "scryglass:current-rating-player-cache:v1", "source_receipt_sha256": source_receipt["receipt_sha256"], "source_identity_sha256": source_receipt["source_identity_sha256"], "player_ids": sorted(player_snapshot["player_id"].astype(str)), "authority": dict(AUTHORITY)}
    _write_json(output_root / "player/player_ratings_cache.json", cache)
    elapsed = time.perf_counter() - started
    receipts = _build_receipts(output_root=output_root, source_receipt=source_receipt, source_receipt_path=source_receipt_path, source_frame_hashes=source_hashes, ledger=ledger, series_by_game=series_by_game, elapsed_seconds=elapsed, player_snapshot=player_snapshot, team_snapshot=team_snapshot, identity_audit=identity_audit)
    _write_json(output_root / "current-rating-ledger-receipt.json", receipts["ledger"])
    _write_json(output_root / "current-rating-snapshot-receipt-v1.json", receipts["snapshots"])
    return {"status": "research_only", "output_root": str(output_root), "rows": len(ledger), "player_rows": len(player_snapshot), "team_rows": len(team_snapshot), "identity_resolution": dict(identity_audit), "ledger_receipt_sha256": receipts["ledger"]["receipt_sha256"], "snapshot_receipt_sha256": receipts["snapshots"]["receipt_sha256"], "authority": dict(AUTHORITY)}


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-receipt", required=True, type=Path)
    parser.add_argument("--source-receipt-file-sha256", "--source-receipt-sha256", dest="source_receipt_file_sha256", required=True)
    parser.add_argument("--expected-source-receipt-sha256", required=True)
    parser.add_argument("--output-root", "--output-dir", dest="output_root", required=True, type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = build_full_current_rating_trust(source_root=args.source_root, source_receipt_path=args.source_receipt, source_receipt_file_sha256=args.source_receipt_file_sha256, expected_source_receipt_sha256=args.expected_source_receipt_sha256, output_root=args.output_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
