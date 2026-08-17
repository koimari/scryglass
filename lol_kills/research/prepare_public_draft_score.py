"""Build the receipt-bound base table for public Draft Score promotion.

The builder uses only information available before each map.  It removes
ambiguous equal-time maps before rating replay.  It also requires exact team,
player, side, role, and champion identities for all ten players.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from collections import defaultdict, deque
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from lol_kills.export.pack_records import build_maps_frame_from_team_games
from lol_kills.ratings.dual_elo import (
    DualEloConfig,
    build_dual_ratings,
    lineup_hashes_from_players,
)
from lol_kills.ratings.player_elo import (
    PlayerEloConfig,
    build_maps_frame_from_players,
    build_player_ratings,
)
from lol_kills.ratings.resolved_rating_source import enrich_rating_frame
from lol_kills.research.atomized_rf_composite import TARGET_LEAGUES, _series_ids


SCHEMA_VERSION = "scryglass:public-draft-score-base:v1"
RATING_CONTEXT_SCHEMA = "scryglass:public-draft-score-rating-context:v1"
MOMENTUM_WINDOW = 40
ROLE_ORDER = ("top", "jungle", "mid", "bot", "support")
RATING_CONTEXT_ROLES = ("top", "jng", "mid", "bot", "sup")
RATING_CONTEXT_FIELDS = (
    "team_sigma_pair_scaled",
    "team_sigma_diff_scaled",
    "player_sigma_pair_scaled",
    "player_sigma_diff_scaled",
    "player_known_fraction_min",
    *tuple(
        f"player_role_{field}_{role}"
        for role in RATING_CONTEXT_ROLES
        for field in (
            "rating_diff_scaled",
            "sigma_pair_scaled",
            "momentum_diff_scaled",
            "rating_available",
        )
    ),
)


class PublicDraftScoreBaseError(ValueError):
    """Raised when the promotion base cannot prove its input authority."""


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _utc_text(value: Any) -> str:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.isoformat().replace("+00:00", "Z")


def _logit(probability: Any) -> float:
    value = float(np.clip(float(probability), 1e-6, 1.0 - 1e-6))
    return float(math.log(value / (1.0 - value)))


def _normalize_role(value: Any) -> str:
    text = str(value or "").strip().casefold()
    return {
        "jng": "jungle",
        "jg": "jungle",
        "adc": "bot",
        "carry": "bot",
        "sup": "support",
        "utility": "support",
    }.get(text, text)


def _normalize_side(value: Any) -> str:
    text = str(value or "").strip().casefold()
    if text in {"blue", "b"}:
        return "blue"
    if text in {"red", "r"}:
        return "red"
    return ""


def _complete_roster_game_ids(players: pd.DataFrame) -> set[str]:
    frame = players.copy()
    frame["gameid"] = frame["gameid"].astype(str)
    frame["side_norm"] = frame["side"].map(_normalize_side)
    frame["role_norm"] = frame["position"].map(_normalize_role)
    frame["stable_player"] = frame["playerid"].astype(str).str.startswith("oe:player:")
    frame["stable_team"] = frame["teamid"].astype(str).str.startswith("oe:team:")
    valid: set[str] = set()
    for game_id, group in frame.groupby("gameid", sort=False):
        if len(group) != 10 or not group["stable_player"].all() or not group["stable_team"].all():
            continue
        assignments = set(zip(group["side_norm"], group["role_norm"]))
        expected = {(side, role) for side in ("blue", "red") for role in ROLE_ORDER}
        if assignments != expected:
            continue
        if group["playerid"].astype(str).nunique() != 10:
            continue
        valid.add(str(game_id))
    return valid


def _lineups(players: pd.DataFrame) -> dict[str, dict[str, list[str]]]:
    frame = players.copy()
    frame["gameid"] = frame["gameid"].astype(str)
    frame["side_norm"] = frame["side"].map(_normalize_side)
    frame["role_norm"] = frame["position"].map(_normalize_role)
    output: dict[str, dict[str, list[str]]] = {}
    for game_id, group in frame.groupby("gameid", sort=False):
        sides: dict[str, list[str]] = {}
        for side in ("blue", "red"):
            current = group[group["side_norm"].eq(side)].set_index("role_norm")
            sides[side] = [str(current.loc[role, "playerid"]) for role in ROLE_ORDER]
        output[str(game_id)] = sides
    return output


def _history_stats(events: Iterable[tuple[float, float]]) -> tuple[float, float, float, float, float]:
    rows = list(events)[-MOMENTUM_WINDOW:]
    if not rows:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    outcomes = np.asarray([row[0] for row in rows], dtype=float)
    residuals = np.asarray([row[1] for row in rows], dtype=float)
    sign = 1.0 if outcomes[-1] >= 0.5 else -1.0
    streak = 0
    for outcome in reversed(outcomes):
        if (outcome >= 0.5) == (sign > 0):
            streak += 1
        else:
            break
    return (
        float(outcomes.mean() - 0.5),
        float(residuals.mean()),
        sign,
        float(sign * streak),
        float(len(rows)),
    )


def _player_history_stats(
    history: Mapping[str, deque[tuple[float, float]]], players: list[str]
) -> tuple[float, float, float, float, float, float]:
    values = [_history_stats(history.get(player, ())) for player in players]
    known = [value for value in values if value[4] > 0]
    if not known:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    matrix = np.asarray(known, dtype=float)
    return (
        float(matrix[:, 0].mean()),
        float(matrix[:, 1].mean()),
        float(matrix[:, 2].mean()),
        float(matrix[:, 3].mean()),
        float(matrix[:, 4].mean()),
        float(len(known) / max(len(players), 1)),
    )


def _add_momentum_columns(
    frame: pd.DataFrame,
    *,
    lineups: Mapping[str, Mapping[str, list[str]]],
) -> pd.DataFrame:
    team_history: dict[str, deque[tuple[float, float]]] = defaultdict(
        lambda: deque(maxlen=MOMENTUM_WINDOW)
    )
    player_history: dict[str, deque[tuple[float, float]]] = defaultdict(
        lambda: deque(maxlen=MOMENTUM_WINDOW)
    )
    rows: list[dict[str, Any]] = []
    ordered = frame.sort_values(["date", "game_uid"], kind="stable")
    for record in ordered.to_dict("records"):
        game_id = str(record["game_uid"])
        blue_team = str(record["blue_team_id"])
        red_team = str(record["red_team_id"])
        players = lineups[game_id]
        blue_team_values = np.asarray(_history_stats(team_history[blue_team]))
        red_team_values = np.asarray(_history_stats(team_history[red_team]))
        team_delta = blue_team_values - red_team_values
        blue_player_values = np.asarray(
            _player_history_stats(player_history, players["blue"])
        )
        red_player_values = np.asarray(
            _player_history_stats(player_history, players["red"])
        )
        player_delta = blue_player_values[:5] - red_player_values[:5]
        record.update(
            {
                "team_wr_diff_g40": float(team_delta[0]),
                "team_residual_diff_g40": float(team_delta[1]),
                "team_last_diff_g40": float(team_delta[2]),
                "team_streak_diff_g40": float(team_delta[3]),
                "team_count_diff_g40": float(team_delta[4]),
                "player_wr_diff_g40": float(player_delta[0]),
                "player_residual_diff_g40": float(player_delta[1]),
                "player_last_diff_g40": float(player_delta[2]),
                "player_streak_diff_g40": float(player_delta[3]),
                "player_count_diff_g40": float(player_delta[4]),
                "player_coverage_g40": float(
                    (blue_player_values[5] + red_player_values[5]) / 2.0
                ),
            }
        )
        rows.append(record)
        outcome = float(record["y"])
        team_probability = float(1.0 / (1.0 + math.exp(-float(record["base_team_logit"]))))
        player_probability = float(
            1.0 / (1.0 + math.exp(-float(record["base_player_logit"])))
        )
        team_history[blue_team].append((outcome, outcome - team_probability))
        team_history[red_team].append((1.0 - outcome, team_probability - outcome))
        for player in players["blue"]:
            player_history[player].append((outcome, outcome - player_probability))
        for player in players["red"]:
            player_history[player].append(
                (1.0 - outcome, player_probability - outcome)
            )
    return pd.DataFrame(rows)


def _add_match_context_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add strictly prior series and head-to-head state."""

    series_wins: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    series_last_winner: dict[str, str] = {}
    head_to_head: dict[tuple[str, str], deque[str]] = defaultdict(
        lambda: deque(maxlen=10)
    )
    rows: list[dict[str, Any]] = []
    for record in frame.sort_values(["date", "game_uid"], kind="stable").to_dict(
        "records"
    ):
        series_id = str(record["series_id"])
        blue_team = str(record["blue_team_id"])
        red_team = str(record["red_team_id"])
        pair = tuple(sorted((blue_team, red_team)))
        series = series_wins[series_id]
        prior_h2h = head_to_head[pair]
        blue_h2h_wins = sum(winner == blue_team for winner in prior_h2h)
        red_h2h_wins = sum(winner == red_team for winner in prior_h2h)
        record.update(
            {
                "series_map_index": float(sum(series.values())),
                "series_score_diff": float(
                    series.get(blue_team, 0) - series.get(red_team, 0)
                ),
                "series_previous_winner_blue": float(
                    1.0
                    if series_last_winner.get(series_id) == blue_team
                    else -1.0
                    if series_last_winner.get(series_id) == red_team
                    else 0.0
                ),
                "series_state_available": float(bool(series)),
                "h2h_win_rate_diff_g10": float(
                    (blue_h2h_wins - red_h2h_wins) / len(prior_h2h)
                    if prior_h2h
                    else 0.0
                ),
                "h2h_count_g10": float(len(prior_h2h)),
                "h2h_available": float(bool(prior_h2h)),
            }
        )
        rows.append(record)
        winner = blue_team if int(record["y"]) == 1 else red_team
        series[winner] += 1
        series_last_winner[series_id] = winner
        prior_h2h.append(winner)
    return pd.DataFrame(rows)


def build_public_draft_score_base(
    *,
    oe_root: Path,
    output: Path,
    start: Any = "2025-01-01T00:00:00Z",
    end: Any | None = None,
    rating_history_csvs: Iterable[Path] = (),
    momentum_window_games: int = 0,
    momentum_scale: float = 0.0,
) -> dict[str, Any]:
    maps_path = oe_root / "maps.parquet"
    players_path = oe_root / "oe_player_games.parquet"
    teams_path = oe_root / "oe_team_games.parquet"
    for path in (maps_path, players_path, teams_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    maps = pd.read_parquet(maps_path)
    players = pd.read_parquet(players_path)
    teams = pd.read_parquet(teams_path)
    maps["date"] = pd.to_datetime(maps["date"], utc=True, errors="raise")
    start_time = pd.Timestamp(start)
    start_time = start_time.tz_localize("UTC") if start_time.tzinfo is None else start_time.tz_convert("UTC")
    end_time = pd.Timestamp(end) if end is not None else maps["date"].max() + pd.Timedelta(seconds=1)
    end_time = end_time.tz_localize("UTC") if end_time.tzinfo is None else end_time.tz_convert("UTC")
    maps = maps[
        maps["league"].isin(TARGET_LEAGUES)
        & maps["date"].ge(start_time)
        & maps["date"].lt(end_time)
        & maps["y_blue_win"].notna()
    ].copy()
    maps["game_uid"] = maps["game_uid"].astype(str)

    timestamp_counts = maps.groupby("date")["game_uid"].nunique()
    ambiguous_timestamps = set(timestamp_counts[timestamp_counts.gt(1)].index)
    ambiguous_time_ids = set(maps.loc[maps["date"].isin(ambiguous_timestamps), "game_uid"])
    maps = maps[~maps["game_uid"].isin(ambiguous_time_ids)].copy()

    players["gameid"] = players["gameid"].astype(str)
    teams["gameid"] = teams["gameid"].astype(str)
    candidate_ids = set(maps["game_uid"])
    players = players[players["gameid"].isin(candidate_ids)].copy()
    teams = teams[teams["gameid"].isin(candidate_ids)].copy()
    complete_ids = _complete_roster_game_ids(players)
    incomplete_identity_ids = candidate_ids - complete_ids
    accepted_ids = candidate_ids & complete_ids
    maps = maps[maps["game_uid"].isin(accepted_ids)].copy()
    players = players[players["gameid"].isin(accepted_ids)].copy()
    teams = teams[teams["gameid"].isin(accepted_ids)].copy()
    if len(maps) < 3000:
        raise PublicDraftScoreBaseError("accepted promotion base is unexpectedly small")

    team_maps = build_maps_frame_from_team_games(teams)
    player_maps = build_maps_frame_from_players(players)
    if set(team_maps["game_uid"].astype(str)) != accepted_ids:
        raise PublicDraftScoreBaseError("team rating map inventory differs from accepted maps")
    if set(player_maps["game_uid"].astype(str)) != accepted_ids:
        raise PublicDraftScoreBaseError("player rating map inventory differs from accepted maps")

    history_paths = [Path(path) for path in rating_history_csvs]
    historical_rows: list[pd.DataFrame] = []
    for path in history_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        history = pd.read_csv(path, low_memory=False)
        history["date"] = pd.to_datetime(history["date"], utc=True, errors="raise")
        history = history[
            history["league"].isin(TARGET_LEAGUES) & history["date"].lt(start_time)
        ].copy()
        historical_rows.append(history)
    if historical_rows:
        historical = pd.concat(historical_rows, ignore_index=True, sort=False)
        historical_teams = historical[
            historical["position"].astype(str).str.casefold().eq("team")
        ].copy()
        historical_players = historical[
            ~historical["position"].astype(str).str.casefold().eq("team")
        ].copy()
        rating_teams = pd.concat(
            [historical_teams, teams], ignore_index=True, sort=False
        )
        rating_players = pd.concat(
            [historical_players, players], ignore_index=True, sort=False
        )
    else:
        rating_teams = teams
        rating_players = players
    rating_team_maps = build_maps_frame_from_team_games(rating_teams)
    rating_player_maps = build_maps_frame_from_players(rating_players)

    team_config = DualEloConfig(
        momentum_window_games=momentum_window_games,
        momentum_scale=momentum_scale,
    )
    player_config = PlayerEloConfig(
        momentum_window_games=momentum_window_games,
        momentum_scale=momentum_scale,
    )
    with tempfile.TemporaryDirectory(prefix="scryglass-public-draft-rating-") as temporary:
        rating_root = Path(temporary)
        team_ratings = build_dual_ratings(
            rating_team_maps,
            team_config,
            lineup_by_game=lineup_hashes_from_players(rating_players),
            output_dir=rating_root,
        )
        player_ratings = build_player_ratings(
            rating_player_maps,
            rating_players,
            player_config,
            output_dir=rating_root,
        )

    rating_values = team_ratings[
        [
            "game_uid",
            "date",
            "mu_diff",
            "p_dual_elo",
            "sigma_blue",
            "sigma_red",
            "sigma_pair",
        ]
    ].merge(
        player_ratings[
            [
                "game_uid",
                "player_mu_diff",
                "p_player_elo",
                "player_sigma_blue",
                "player_sigma_red",
                "player_sigma_pair",
                "player_known_blue",
                "player_known_red",
                *[
                    f"player_role_{field}_{role}"
                    for role in RATING_CONTEXT_ROLES
                    for field in (
                        "mu_diff",
                        "sigma_pair",
                        "momentum_diff",
                        "rating_available",
                    )
                ],
            ]
        ],
        on="game_uid",
        how="inner",
        validate="one_to_one",
    )
    rating_values["game_uid"] = rating_values["game_uid"].astype(str)
    rating_values = rating_values.sort_values(["date", "game_uid"], kind="stable")
    rating_start = pd.to_datetime(rating_values["date"], utc=True).min()
    prior_initialization = rating_start.floor("D") - pd.Timedelta(seconds=1)
    previous_timestamp = prior_initialization
    rating_rows: list[dict[str, Any]] = []
    rating_context_rows: list[dict[str, Any]] = []
    for timestamp, batch in rating_values.groupby("date", sort=False):
        for row in batch[batch["game_uid"].isin(accepted_ids)].to_dict("records"):
            player_complete = float(
                int(row["player_known_blue"]) == 5 and int(row["player_known_red"]) == 5
            )
            rating_rows.append(
                {
                    "game_uid": str(row["game_uid"]),
                    "rating_as_of": _utc_text(previous_timestamp),
                    "base_team_logit": _logit(row["p_dual_elo"]),
                    "team_rating_diff_scaled": float(row["mu_diff"]) / 400.0,
                    "base_player_logit": _logit(row["p_player_elo"]),
                    "player_rating_diff_scaled": float(row["player_mu_diff"]) / 400.0,
                    "player_lineup_complete": player_complete,
                }
            )
            context_values = {
                "team_sigma_pair_scaled": float(row["sigma_pair"]) / 400.0,
                "team_sigma_diff_scaled": (
                    float(row["sigma_blue"]) - float(row["sigma_red"])
                )
                / 400.0,
                "player_sigma_pair_scaled": float(row["player_sigma_pair"])
                / 400.0,
                "player_sigma_diff_scaled": (
                    float(row["player_sigma_blue"])
                    - float(row["player_sigma_red"])
                )
                / 400.0,
                "player_known_fraction_min": min(
                    int(row["player_known_blue"]),
                    int(row["player_known_red"]),
                )
                / 5.0,
            }
            for role in RATING_CONTEXT_ROLES:
                context_values.update(
                    {
                        f"player_role_rating_diff_scaled_{role}": float(
                            row[f"player_role_mu_diff_{role}"]
                        )
                        / 400.0,
                        f"player_role_sigma_pair_scaled_{role}": float(
                            row[f"player_role_sigma_pair_{role}"]
                        )
                        / 400.0,
                        f"player_role_momentum_diff_scaled_{role}": float(
                            row[f"player_role_momentum_diff_{role}"]
                        )
                        / 400.0,
                        f"player_role_rating_available_{role}": float(
                            row[f"player_role_rating_available_{role}"]
                        ),
                    }
                )
            rating_context_rows.append(
                {"game_uid": str(row["game_uid"]), **context_values}
            )
        previous_timestamp = pd.Timestamp(timestamp)

    source_identity_payload = {
        "schema_version": SCHEMA_VERSION,
        "window": {"start": _utc_text(start_time), "end_exclusive": _utc_text(end_time)},
        "target_leagues": list(TARGET_LEAGUES),
        "team_rating_config": asdict(team_config),
        "player_rating_config": asdict(player_config),
        "equal_timestamp_policy": "exclude-all-maps-sharing-an-exact-source-timestamp-v1",
        "prior_initialization": _utc_text(prior_initialization),
        "rating_history_start": _utc_text(rating_start),
    }
    source_identity = canonical_bytes(source_identity_payload).decode("utf-8")
    source_artifact = canonical_bytes(
        {
            **source_identity_payload,
            "source_files": {
                "maps": sha256_path(maps_path),
                "players": sha256_path(players_path),
                "teams": sha256_path(teams_path),
                "builder": sha256_path(Path(__file__)),
                "rating_history": {
                    str(path): sha256_path(path) for path in history_paths
                },
            },
        }
    )
    roster_frame = players[
        ["gameid", "side", "position", "teamid", "playerid", "champion"]
    ].rename(columns={"gameid": "game_uid"})
    receipt_maps = maps[
        ["game_uid", "date", "league", "y_blue_win"]
    ].rename(columns={"y_blue_win": "y"})
    rating_frame = pd.DataFrame(rating_rows)
    receipt_parts: list[pd.DataFrame] = []
    ordered_ids = receipt_maps.sort_values(["date", "game_uid"], kind="stable")[
        "game_uid"
    ].astype(str).tolist()
    for start_index in range(0, len(ordered_ids), 500):
        chunk_ids = set(ordered_ids[start_index : start_index + 500])
        receipt_parts.append(
            enrich_rating_frame(
                receipt_maps[receipt_maps["game_uid"].isin(chunk_ids)],
                roster_frame[roster_frame["game_uid"].isin(chunk_ids)],
                rating_frame[rating_frame["game_uid"].isin(chunk_ids)],
                source_identity=source_identity,
                source_artifact=source_artifact,
                strict=True,
            )
        )
    receipts = pd.concat(receipt_parts, ignore_index=True).sort_values(
        ["date", "game_uid"], kind="stable"
    )
    if not receipts["rating_source_available"].eq(1.0).all():
        raise PublicDraftScoreBaseError("one or more rating receipts failed closed")

    map_context = maps[
        ["game_uid", "date", "league", "y_blue_win", "blue_team", "red_team"]
    ].copy()
    team_ids = (
        players.assign(side_norm=players["side"].map(_normalize_side))
        .groupby(["gameid", "side_norm"])["teamid"]
        .first()
        .unstack()
        .rename(columns={"blue": "blue_team_id", "red": "red_team_id"})
        .reset_index()
        .rename(columns={"gameid": "game_uid"})
    )
    map_context = map_context.merge(team_ids, on="game_uid", validate="one_to_one")
    base = receipts.drop(columns=[column for column in ("date", "league", "y") if column in receipts])
    base = map_context.merge(base, on="game_uid", validate="one_to_one")
    rating_context = pd.DataFrame(rating_context_rows)
    base = base.merge(rating_context, on="game_uid", validate="one_to_one")
    base["rating_context_schema"] = RATING_CONTEXT_SCHEMA
    base["rating_context_available"] = 1.0
    base["rating_context_missing"] = 0.0
    base["rating_context_sha256"] = base.apply(
        lambda row: hashlib.sha256(
            canonical_bytes(
                {
                    "schema_version": RATING_CONTEXT_SCHEMA,
                    "rating_receipt_sha256": str(row["rating_receipt_sha256"]),
                    "values": {
                        field: float(row[field]) for field in RATING_CONTEXT_FIELDS
                    },
                }
            )
        ).hexdigest(),
        axis=1,
    )
    base = base.rename(columns={"y_blue_win": "y"})
    base["y"] = base["y"].astype(int)
    base["blue_side"] = 1.0
    series = _series_ids(maps.copy())
    base["series_id"] = base["game_uid"].map(series)
    base = _add_momentum_columns(base, lineups=_lineups(players))
    base = _add_match_context_columns(base)
    if not base["date"].is_monotonic_increasing:
        raise PublicDraftScoreBaseError("promotion base is not chronological")
    if base["game_uid"].duplicated().any():
        raise PublicDraftScoreBaseError("promotion base has duplicate maps")

    output.parent.mkdir(parents=True, exist_ok=True)
    base.to_parquet(output, index=False, compression="zstd")
    output_sha256 = sha256_path(output)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen_promotion_base",
        "output": {"path": str(output), "sha256": output_sha256, "rows": len(base)},
        "date_min": _utc_text(base["date"].min()),
        "date_max": _utc_text(base["date"].max()),
        "league_counts": {
            str(key): int(value) for key, value in base["league"].value_counts().sort_index().items()
        },
        "exclusions": {
            "equal_timestamp_maps": sorted(ambiguous_time_ids),
            "incomplete_stable_roster_maps": sorted(incomplete_identity_ids),
        },
        "rating_source_identity": source_identity_payload,
        "rating_source_artifact_sha256": hashlib.sha256(source_artifact).hexdigest(),
        "source_files": {
            "maps": {"path": str(maps_path), "sha256": sha256_path(maps_path)},
            "players": {"path": str(players_path), "sha256": sha256_path(players_path)},
            "teams": {"path": str(teams_path), "sha256": sha256_path(teams_path)},
            "rating_history": {
                str(path): sha256_path(path) for path in history_paths
            },
        },
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oe-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", default="2025-01-01T00:00:00Z")
    parser.add_argument("--end")
    parser.add_argument("--rating-history-csv", type=Path, action="append", default=[])
    parser.add_argument("--momentum-window-games", type=int, default=0)
    parser.add_argument("--momentum-scale", type=float, default=0.0)
    args = parser.parse_args()
    print(
        json.dumps(
            build_public_draft_score_base(
                oe_root=args.oe_root,
                output=args.output,
                start=args.start,
                end=args.end,
                rating_history_csvs=args.rating_history_csv,
                momentum_window_games=args.momentum_window_games,
                momentum_scale=args.momentum_scale,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
