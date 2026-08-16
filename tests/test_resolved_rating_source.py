from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest

from lol_kills.research.atomized_rf_composite import _resolved_roster_sha256
from lol_kills.ratings.resolved_rating_source import (
    SCHEMA_VERSION,
    ResolvedRatingSourceError,
    build_rating_receipt_sha256,
    build_resolved_rating_source,
    build_roster_sha256,
    canonical_sha256,
    enrich_rating_frame,
    sha256_bytes,
)


SOURCE_ARTIFACT = b"resolved-rating-source-fixture-v1"
SOURCE_IDENTITY = {
    "locator": "ratings/pre-game.parquet",
    "revision": "source-revision-1",
}
GAME_TIME = "2026-08-01T12:00:00Z"
RATING_TIME = "2026-08-01T11:59:59Z"


def _roster() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for side, team in (("blue", "oe:team:blue"), ("red", "oe:team:red")):
        for role_index, role in enumerate(("top", "jungle", "mid", "bot", "support")):
            rows.append(
                {
                    "game_uid": "map-1",
                    "side": side,
                    "position": role,
                    "teamid": team,
                    "playerid": f"oe:player:{side}-{role_index}",
                    "champion": f"Champion{role_index}",
                }
            )
    return rows


def _ratings(**updates: object) -> dict[str, object]:
    values: dict[str, object] = {
        "game_uid": "map-1",
        "rating_as_of": RATING_TIME,
        "base_team_logit": 0.2,
        "team_rating_diff_scaled": 0.1,
        "base_player_logit": 0.3,
        "player_rating_diff_scaled": 0.2,
        "player_lineup_complete": 1.0,
    }
    values.update(updates)
    return values


def _receipt(**updates: object) -> dict[str, object]:
    values = _ratings(**updates)
    return build_resolved_rating_source(
        game_id="map-1",
        game_timestamp=GAME_TIME,
        source_identity=SOURCE_IDENTITY,
        source_artifact=SOURCE_ARTIFACT,
        roster_rows=_roster(),
        rating_values=values,
    )


def test_receipt_is_deterministic_and_uses_the_consumer_payload() -> None:
    first = _receipt()
    second = _receipt()
    assert first == second
    assert first["rating_source_available"] == 1.0
    assert first["rating_source_sha256"] == sha256_bytes(SOURCE_ARTIFACT)
    assert first["rating_receipt_sha256"] == build_rating_receipt_sha256(
        rating_source_available=1.0,
        rating_source_sha256=first["rating_source_sha256"],
        rating_roster_sha256=first["rating_roster_sha256"],
    )
    assert first["rating_receipt_schema"] == SCHEMA_VERSION


def test_rf_harness_recomputes_the_same_context_bound_roster_hash() -> None:
    receipt = _receipt()
    assert _resolved_roster_sha256(
        _roster(),
        game_id="map-1",
        timestamp=GAME_TIME,
        source_identity=SOURCE_IDENTITY,
    ) == receipt["rating_roster_sha256"]


@pytest.mark.parametrize("field", ("side", "position", "playerid", "champion"))
def test_roster_hash_changes_when_any_assignment_field_changes(field: str) -> None:
    original = _roster()
    changed = deepcopy(original)
    if field == "side":
        changed[0]["side"], changed[5]["side"] = "red", "blue"
        changed[0]["teamid"], changed[5]["teamid"] = changed[5]["teamid"], changed[0]["teamid"]
    elif field == "position":
        changed[0]["position"], changed[1]["position"] = "jungle", "top"
    elif field == "playerid":
        changed[0][field] = "oe:player:replacement"
    else:
        changed[0][field] = "DifferentChampion"
    assert build_roster_sha256(
        game_id="map-1",
        timestamp=GAME_TIME,
        source_identity=SOURCE_IDENTITY,
        roster_rows=original,
    ) != build_roster_sha256(
        game_id="map-1",
        timestamp=GAME_TIME,
        source_identity=SOURCE_IDENTITY,
        roster_rows=changed,
    )


def test_roster_hash_binds_game_time_and_source_identity() -> None:
    base = build_roster_sha256(
        game_id="map-1",
        timestamp=GAME_TIME,
        source_identity=SOURCE_IDENTITY,
        roster_rows=_roster(),
    )
    assert base != build_roster_sha256(
        game_id="map-2",
        timestamp=GAME_TIME,
        source_identity=SOURCE_IDENTITY,
        roster_rows=_roster(),
    )
    assert base != build_roster_sha256(
        game_id="map-1",
        timestamp="2026-08-01T12:00:00.001Z",
        source_identity=SOURCE_IDENTITY,
        roster_rows=_roster(),
    )
    assert base != build_roster_sha256(
        game_id="map-1",
        timestamp=GAME_TIME,
        source_identity={**SOURCE_IDENTITY, "revision": "source-revision-2"},
        roster_rows=_roster(),
    )
    assert base == build_roster_sha256(
        game_id="map-1",
        timestamp=GAME_TIME,
        source_identity=SOURCE_IDENTITY,
        roster_rows=list(reversed(_roster())),
    )


@pytest.mark.parametrize(
    "mutator",
    (
        lambda rows: rows.__setitem__(0, {**rows[0], "playerid": ""}),
        lambda rows: rows.__setitem__(0, {**rows[0], "playerid": "player-without-stable-prefix"}),
        lambda rows: rows.append(dict(rows[0])),
        lambda rows: rows.__setitem__(1, {**rows[1], "position": rows[0]["position"]}),
    ),
)
def test_missing_ambiguous_or_duplicate_identity_is_unavailable(mutator) -> None:
    rows = _roster()
    mutator(rows)
    receipt = build_resolved_rating_source(
        game_id="map-1",
        game_timestamp=GAME_TIME,
        source_identity=SOURCE_IDENTITY,
        source_artifact=SOURCE_ARTIFACT,
        roster_rows=rows,
        rating_values=_ratings(),
    )
    assert receipt["rating_source_available"] == 0.0
    assert receipt["rating_roster_sha256"] is None
    assert receipt["rating_receipt_sha256"] is None
    with pytest.raises(ResolvedRatingSourceError):
        build_resolved_rating_source(
            game_id="map-1",
            game_timestamp=GAME_TIME,
            source_identity=SOURCE_IDENTITY,
            source_artifact=SOURCE_ARTIFACT,
            roster_rows=rows,
            rating_values=_ratings(),
            strict=True,
        )


def test_equal_or_later_rating_timestamp_fails_closed() -> None:
    for timestamp in (GAME_TIME, "2026-08-01T12:00:01Z"):
        receipt = _receipt(rating_as_of=timestamp)
        assert receipt["rating_source_available"] == 0.0
        assert "pre-game" in receipt["rating_source_reason"]


def test_same_timestamp_maps_are_independent_and_ordered() -> None:
    maps = [
        {"game_uid": "map-2", "date": "2026-08-01T12:00:00Z", "y_blue_win": 1},
        {"game_uid": "map-1", "date": "2026-08-01T12:00:00Z", "y_blue_win": 0},
    ]
    roster = [dict(row, game_uid=game_id) for game_id in ("map-1", "map-2") for row in _roster()]
    ratings = [dict(_ratings(game_uid=game_id)) for game_id in ("map-1", "map-2")]
    result = enrich_rating_frame(
        maps,
        roster,
        ratings,
        source_identity=SOURCE_IDENTITY,
        source_artifact=SOURCE_ARTIFACT,
    )
    assert result["game_uid"].tolist() == ["map-1", "map-2"]
    assert result["rating_source_available"].tolist() == [1.0, 1.0]
    assert result["rating_roster_sha256"].nunique() == 2


def test_outcome_and_current_state_fields_cannot_supply_rating_values() -> None:
    receipt = _receipt(result=1)
    assert receipt["rating_source_available"] == 0.0
    assert "forbidden" in receipt["rating_source_reason"]
    receipt = _receipt(current_gold=1000)
    assert receipt["rating_source_available"] == 0.0


def test_map_outcome_does_not_enter_receipt_or_rating_values() -> None:
    base = [{"game_uid": "map-1", "date": GAME_TIME, "y_blue_win": 0}]
    changed = [{"game_uid": "map-1", "date": GAME_TIME, "y_blue_win": 1}]
    kwargs = {
        "roster_rows": _roster(),
        "rating_rows": [_ratings()],
        "source_identity": SOURCE_IDENTITY,
        "source_artifact": SOURCE_ARTIFACT,
    }
    first = enrich_rating_frame(base, **kwargs).iloc[0]
    second = enrich_rating_frame(changed, **kwargs).iloc[0]
    receipt_fields = [
        "rating_source_available",
        "rating_source_sha256",
        "rating_roster_sha256",
        "rating_receipt_sha256",
        "base_team_logit",
        "base_player_logit",
    ]
    assert first[receipt_fields].to_dict() == second[receipt_fields].to_dict()


def test_neutral_placeholders_have_explicit_missingness() -> None:
    receipt = _receipt(
        base_team_logit=None,
        team_rating_diff_scaled=None,
        base_player_logit=None,
        player_rating_diff_scaled=None,
        player_lineup_complete=0.0,
    )
    assert receipt["rating_source_available"] == 1.0
    assert receipt["rating_values_available"] == 0.0
    assert receipt["team_rating_available"] == 0.0
    assert receipt["player_rating_available"] == 0.0
    assert receipt["base_team_logit"] == 0.0
    assert receipt["base_player_logit"] == 0.0
