"""Outcome-design diagnostics for player ratings.

A map result is observed at team grain.  Two players with identical signed map
exposure have identical outcome-design columns, so team results alone cannot
separate their individual contributions.  This module reports that fact
directly instead of inventing within-lineup precision.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

import pandas as pd


def _clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def build_player_outcome_identifiability(
    players: pd.DataFrame,
) -> pd.DataFrame:
    """Return exact co-exposure groups in the team-outcome design matrix."""

    columns = (
        "player",
        "outcome_exposure_group_id",
        "outcome_exposure_group_size",
        "outcome_separately_identified",
        "outcome_identifiability_label",
        "outcome_identical_players",
        "n_outcome_maps",
        "n_distinct_lineups",
        "n_distinct_teams",
    )
    if players is None or players.empty or "playername" not in players.columns:
        return pd.DataFrame(columns=columns)

    frame = players.copy()
    game_column = "game_uid" if "game_uid" in frame.columns else "gameid"
    if game_column not in frame.columns or "side" not in frame.columns:
        return pd.DataFrame(columns=columns)
    if "position" in frame.columns:
        frame = frame[
            frame["position"].astype(str).str.casefold().ne("team")
        ]
    frame["_player"] = frame["playername"].map(_clean_text)
    frame["_game"] = frame[game_column].map(_clean_text)
    frame["_side"] = frame["side"].astype(str).str.casefold()
    frame = frame[
        frame["_player"].ne("")
        & frame["_game"].ne("")
        & frame["_side"].isin({"blue", "red"})
    ].drop_duplicates(["_game", "_side", "_player"])
    if frame.empty:
        return pd.DataFrame(columns=columns)

    lineup_by_map_side: dict[tuple[str, str], tuple[str, ...]] = {}
    for key, group in frame.groupby(["_game", "_side"], sort=True):
        lineup_by_map_side[(str(key[0]), str(key[1]))] = tuple(
            sorted(group["_player"].astype(str).unique())
        )

    exposures: dict[str, list[str]] = defaultdict(list)
    lineups: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    teams: dict[str, set[str]] = defaultdict(set)
    for _, row in frame.iterrows():
        player = str(row["_player"])
        game = str(row["_game"])
        side = str(row["_side"])
        sign = "+" if side == "blue" else "-"
        exposures[player].append(f"{game}:{sign}")
        lineups[player].add(lineup_by_map_side[(game, side)])
        team = _clean_text(row.get("teamname"))
        if team:
            teams[player].add(team)

    signature_to_players: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for player, values in exposures.items():
        signature_to_players[tuple(sorted(values))].append(player)

    rows: list[dict[str, Any]] = []
    for signature, group_players in signature_to_players.items():
        members = sorted(group_players, key=str.casefold)
        payload = "\x1f".join(signature).encode("utf-8")
        group_id = f"outcome:{hashlib.sha256(payload).hexdigest()[:20]}"
        group_size = len(members)
        for player in members:
            rows.append(
                {
                    "player": player,
                    "outcome_exposure_group_id": group_id,
                    "outcome_exposure_group_size": group_size,
                    "outcome_separately_identified": group_size == 1,
                    "outcome_identifiability_label": (
                        "Separately observed"
                        if group_size == 1
                        else "Shared outcome history"
                    ),
                    "outcome_identical_players": members,
                    "n_outcome_maps": len(signature),
                    "n_distinct_lineups": len(lineups[player]),
                    "n_distinct_teams": len(teams[player]),
                }
            )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["outcome_exposure_group_size", "player"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
