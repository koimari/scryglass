from __future__ import annotations

import json
from pathlib import Path

import pytest

from lol_kills.research import grid_sequence_review as review
from lol_kills.research import grid_sequence_actions as actions


def _state(
    participant_id: int,
    *,
    team_id: int = 100,
    champion: str = "Ashe",
    role: str = "Bottom",
    xp: float = 0,
    cs: float = 0,
    minion_gold: float = 0,
    support_gold: float = 0,
    neutral_gold: float = 0,
    true_damage: float = 0,
    building_damage: float = 0,
    total_gold: float | None = None,
) -> dict:
    return {
        "participant_id": participant_id,
        "team_id": team_id,
        "champion": champion,
        "role": role,
        "xp": xp,
        "level": 7,
        "total_gold": (
            minion_gold + support_gold + neutral_gold
            if total_gold is None
            else total_gold
        ),
        "current_gold": 0,
        "x": 0,
        "z": 0,
        "gold": {
            field: (
                minion_gold
                if field == "killMinion"
                else support_gold
                if field == "supportItemMinion"
                else neutral_gold
                if field == "killNeutralMinion"
                else 0
            )
            for field in review.GOLD_STAT_FIELDS
        },
        "stats": {
            field: (
                cs
                if field == "MINIONS_KILLED"
                else true_damage
                if field == "TRUE_DAMAGE_DEALT_PLAYER"
                else building_damage
                if field in {
                    "TOTAL_DAMAGE_DEALT_TO_BUILDINGS",
                    "TOTAL_DAMAGE_DEALT_TO_TURRETS",
                }
                else 0
            )
            for field in review.PLAYER_STAT_FIELDS
        },
    }


def _frame(time_ms: int, states: list[dict]) -> review.Frame:
    return review.Frame(
        time_ms=time_ms,
        sequence_index=time_ms,
        players={row["participant_id"]: row for row in states},
    )


def test_clock_round_trip() -> None:
    assert review.parse_clock("8:29.981") == 509_981
    assert review.format_clock(509_981) == "8:29.981"
    assert review.parse_clock("10:20") == 620_000
    with pytest.raises(review.GridSequenceReviewError):
        review.parse_clock("8:61")


def test_catalog_hash_must_verify(tmp_path: Path) -> None:
    catalog = {
        "catalog_version": "1",
        "capabilities": [
            {"capability": "historical_file_listing", "status": "confirmed"},
            {"capability": "historical_file_download", "status": "not_tested"},
        ],
        "endpoints": [],
    }
    catalog["catalog_sha256"] = review._hash(catalog)
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog))
    assert review.catalog_provenance(path)["catalog_sha256"] == catalog[
        "catalog_sha256"
    ]
    catalog["catalog_version"] = "changed"
    path.write_text(json.dumps(catalog))
    with pytest.raises(review.GridSequenceReviewError):
        review.catalog_provenance(path)


def test_safe_file_metadata_removes_signed_capabilities() -> None:
    safe = review._safe_file_metadata(
        {
            "id": "events-riot-game-3",
            "fullURL": "https://example.invalid/?token=secret",
            "signature": "secret",
            "status": "ready",
        }
    )
    assert safe == {"id": "events-riot-game-3", "status": "ready"}
    assert "secret" not in json.dumps(safe)


def test_touch_classifier_excludes_non_tick_true_damage() -> None:
    start = _frame(
        585_000,
        [
            _state(1, champion="Ashe"),
            _state(2, champion="XinZhao", role="Jungle"),
        ],
    )
    middle = _frame(
        586_000,
        [
            _state(
                1,
                champion="Ashe",
                true_damage=778.1035,
                building_damage=914.9758,
            ),
            _state(
                2,
                champion="XinZhao",
                role="Jungle",
                true_damage=32,
                building_damage=130,
            ),
        ],
    )
    end = _frame(
        587_000,
        [
            _state(
                1,
                champion="Ashe",
                true_damage=794.1035,
                building_damage=986.9758,
            ),
            _state(
                2,
                champion="XinZhao",
                role="Jungle",
                true_damage=64,
                building_damage=260,
            ),
        ],
    )
    roster = (
        {"participant_id": 1, "team_id": 100, "role": "Bottom", "champion": "Ashe", "player": "A"},
        {"participant_id": 2, "team_id": 100, "role": "Jungle", "champion": "XinZhao", "player": "B"},
        {"participant_id": 3, "team_id": 100, "role": "Top", "champion": "Gragas", "player": "C"},
        {"participant_id": 4, "team_id": 100, "role": "Middle", "champion": "Akali", "player": "D"},
        {"participant_id": 5, "team_id": 100, "role": "Support", "champion": "Seraphine", "player": "E"},
        {"participant_id": 6, "team_id": 200, "role": "Top", "champion": "Jax", "player": "F"},
        {"participant_id": 7, "team_id": 200, "role": "Jungle", "champion": "JarvanIV", "player": "G"},
        {"participant_id": 8, "team_id": 200, "role": "Middle", "champion": "Annie", "player": "H"},
        {"participant_id": 9, "team_id": 200, "role": "Bottom", "champion": "Yunara", "player": "I"},
        {"participant_id": 10, "team_id": 200, "role": "Support", "champion": "Lulu", "player": "J"},
    )
    # Fill missing players because window extraction deliberately requires the
    # complete roster at every accepted stats frame.
    frames = []
    for frame in (start, middle, end):
        players = dict(frame.players)
        for row in roster:
            players.setdefault(
                row["participant_id"],
                _state(
                    row["participant_id"],
                    team_id=row["team_id"],
                    champion=row["champion"],
                    role=row["role"],
                ),
            )
        frames.append(review.Frame(frame.time_ms, frame.sequence_index, players))
    game = review.GameData(
        identity={"game_version": "16.15.1"},
        roster=roster,
        frames=tuple(frames),
        events=(),
        completeness={},
    )
    result = review.analyze_siege(
        game,
        start_ms=585_000,
        end_ms=587_000,
        attacking_team_id=100,
        lane="top",
    )
    assert result["touch_compatible_true_damage"] == 80
    assert result["touch_by_champion"] == {"Ashe": 16, "XinZhao": 64}
    assert result["other_true_by_champion"] == {"Ashe": 778.1035}


def test_infer_cannon_gold_uses_same_game_last_hit() -> None:
    states0 = [_state(i, champion=f"C{i}") for i in range(1, 11)]
    states1 = [dict(row) for row in states0]
    states1[0] = _state(1, champion="C1", xp=75, cs=1, minion_gold=54)
    game = review.GameData(
        identity={},
        roster=tuple(
            {
                "participant_id": i,
                "team_id": 100 if i <= 5 else 200,
                "role": "Top",
                "champion": f"C{i}",
                "player": f"P{i}",
            }
            for i in range(1, 11)
        ),
        frames=(_frame(499_000, states0), _frame(500_000, states1)),
        events=(),
        completeness={},
    )
    result = review.infer_cannon_gold(
        game, resource_start_ms=510_000, resource_end_ms=525_000
    )
    assert result["status"] == "verified_from_same_game_stats"
    assert result["value"] == 54


def _ten_player_roster() -> tuple[dict, ...]:
    champions = (
        "Gragas",
        "XinZhao",
        "Akali",
        "Ashe",
        "Seraphine",
        "Jax",
        "JarvanIV",
        "Annie",
        "Yunara",
        "Lulu",
    )
    roles = ("Top", "Jungle", "Middle", "Bottom", "Support") * 2
    return tuple(
        {
            "participant_id": participant_id,
            "team_id": 100 if participant_id <= 5 else 200,
            "role": roles[participant_id - 1],
            "champion": champion,
            "player": f"{'MKOI' if participant_id <= 5 else 'FNC'} P{participant_id}",
        }
        for participant_id, champion in enumerate(champions, start=1)
    )


def test_actual_resource_view_separates_plate_gold_without_double_counting() -> None:
    roster = _ten_player_roster()
    start = [
        _state(
            row["participant_id"],
            team_id=row["team_id"],
            champion=row["champion"],
            role=row["role"],
        )
        for row in roster
    ]
    end = list(start)
    end[1] = _state(
        2,
        champion="XinZhao",
        role="Jungle",
        neutral_gold=712,
        cs=0,
        xp=849,
        total_gold=832,
    )
    end[2] = _state(3, champion="Akali", minion_gold=900, total_gold=900)
    end[8] = _state(
        9,
        team_id=200,
        champion="Yunara",
        minion_gold=873,
        total_gold=993,
    )
    game = review.GameData(
        identity={},
        roster=roster,
        frames=(_frame(466_000, start), _frame(620_000, end)),
        events=(
            {
                "schema": "turret_plate_gold_earned",
                "game_time_ms": 500_000,
                "sequence_index": 1,
                "participantID": 2,
                "teamID": 100,
                "bounty": 120,
            },
            {
                "schema": "turret_plate_gold_earned",
                "game_time_ms": 510_000,
                "sequence_index": 2,
                "participantID": 9,
                "teamID": 200,
                "bounty": 120,
            },
        ),
        completeness={},
    )
    result = review.build_resource_views(
        game,
        start_ms=466_000,
        end_ms=620_000,
        reference_team_id=100,
        involved_champions=("Xin Zhao", "Yunara"),
    )
    selected = result["selected"]
    xin = next(row for row in selected["rows"] if row["champion"] == "XinZhao")
    assert xin["gold_excluding_plates"] == 712
    assert xin["plate_gold"] == 120
    assert xin["total_gold"] == 832
    assert selected["team_totals"][100]["players_included"] == 1
    assert selected["team_totals"][100]["total_gold"] == 832
    assert result["full_teams"]["team_totals"][100]["players_included"] == 5
    assert result["full_teams"]["team_totals"][100]["total_gold"] == 1732


def test_turret_health_counterfactual_uses_9000_hp_plate_thresholds() -> None:
    siege = {"touch_compatible_true_damage": 1840}
    result = review.analyze_turret_health(
        siege,
        {
            "game_time": "10:45",
            "health_estimate": 720,
            "health_low": 700,
            "health_high": 750,
            "source": "observer",
        },
    )
    fixed = result["fixed_state_remove_touch_only"]
    assert fixed["health_estimate"] == 2560
    assert fixed["plate_thresholds_reached"] == 4
    assert result["exact_live_stats_health"]["status"] == "unavailable"
    assert review.analyze_turret_health(siege, None)["status"] == "unavailable"


def test_request_acceptance_fails_closed_on_changed_number() -> None:
    report = {"observed": {"siege": {"plate_gold": 480}}}
    passed = review.verify_request_acceptance(
        report, {"observed.siege.plate_gold": 480}
    )
    assert passed["status"] == "passed"
    with pytest.raises(review.GridSequenceReviewError, match="acceptance checks failed"):
        review.verify_request_acceptance(
            report, {"observed.siege.plate_gold": 600}
        )


def test_action_contracts_form_one_ordered_dependency_graph() -> None:
    contracts = actions.action_contracts()
    assert len(contracts) == 16
    assert contracts[0]["action"] == "verify_source"
    assert contracts[-1]["action"] == "render_public"
    seen: set[str] = set()
    for expected_order, contract in enumerate(contracts, start=1):
        assert contract["order"] == expected_order
        assert set(contract["dependencies"]) <= seen
        assert contract["action"] not in seen
        assert contract["inputs"]
        assert contract["outputs"]
        seen.add(contract["action"])


def test_mkoi_fnc_game_3_regression_if_private_source_is_present() -> None:
    source = Path(
        "data/lol/warehouse/private_grid/sequence_review/v1/raw/"
        "events_2966877_3_riot_"
        "4ef6a826e9bc453589a872aaf8c0c343f271bf7b0b5e821253e6848edf3e4391.jsonl"
    )
    if not source.is_file():
        pytest.skip("private GRID regression source is not installed")
    kwargs = {
        "source_path": source,
        "receipt": {
            "provider_series_id": "2966877",
            "provider_game_id": "a48e0458-3021-4e87-bf97-79ca96847db6",
            "game_index": 3,
        },
        "sequence_start_ms": review.parse_clock("7:46"),
        "sequence_end_ms": review.parse_clock("10:20"),
        "resource_start_ms": review.parse_clock("8:30"),
        "resource_end_ms": review.parse_clock("8:45"),
        "siege_start_ms": review.parse_clock("9:45"),
        "siege_end_ms": review.parse_clock("10:20"),
        "involved_champions": (
            "Gragas",
            "XinZhao",
            "Ashe",
            "Seraphine",
            "Jax",
            "JarvanIV",
            "Yunara",
            "Lulu",
        ),
        "team_labels": {100: "MKOI", 200: "FNC"},
        "delayed_camps": ("gromp", "wolves"),
        "turret_observation": {
            "game_time": "10:45",
            "health_estimate": 720,
            "health_low": 700,
            "health_high": 750,
            "source": "observer",
        },
    }
    first = review.analyze_sequence(**kwargs)
    second = review.analyze_sequence(**kwargs)
    assert first["analysis_sha256"] == second["analysis_sha256"]
    assert first["action_graph"] == second["action_graph"]
    assert len(first["action_graph"]["actions"]) == 13
    assert first["mechanics"]["receipt_status"] == "verified"
    legacy = review._analyze_sequence_monolith(**kwargs)
    assert first["observed"] == legacy["observed"]
    assert first["counterfactual"] == legacy["counterfactual"]
    assert [
        event["game_time"] for event in first["observed"]["grubs"]["events"]
    ] == ["8:31.884", "8:41.668", "8:45.075"]
    siege = first["observed"]["siege"]
    assert siege["touch_compatible_true_damage"] == 1840
    assert siege["plate_gold"] == 480
    assert siege["conditional_time_saved_seconds"] == pytest.approx(13.5210234)
    selected = first["observed"]["resource_views"]["selected"]
    assert selected["team_totals"][100]["total_gold"] == 3312
    assert selected["team_totals"][200]["total_gold"] == 2971
    full_difference = first["observed"]["resource_views"]["full_teams"][
        "comparison"
    ]["reference_minus_opponent"]
    assert full_difference["total_gold"] == 382
    assert full_difference["xp"] == -87
    camps = first["observed"]["delayed_camps"]
    assert camps["later_camp_resources"] == {"gold": 185, "xp": 460}
    assert camps["camps_minus_grubs"] == {"gold": 75, "xp": 283}
    assert first["observed"]["turret_health"]["fixed_state_remove_touch_only"][
        "plate_thresholds_reached"
    ] == 4
    request_path = Path(
        "data/lol/warehouse/private_grid/sequence_review/v1/requests/"
        "series_2966877_game_3.json"
    )
    request = review.load_review_request(request_path)
    assert len(request["expected_observed"]) == 53
    assert review.verify_request_acceptance(
        first, request["expected_observed"]
    )["status"] == "passed"
    public_action = actions.run_analysis_action_graph(
        **kwargs,
        expected_observed=request["expected_observed"],
        stop_after="render_public",
    )
    assert len(public_action["graph"]["actions"]) == 16
    assert public_action["requested_action"] == "render_public"
    assert public_action["output"]["content"] == review.render_public_digest(first)
    generated_request = review.build_review_request_from_report(first)
    assert generated_request["source"]["raw_sha256"] == (
        "4ef6a826e9bc453589a872aaf8c0c343f271bf7b0b5e821253e6848edf3e4391"
    )
    assert len(generated_request["expected_observed"]) >= 40
    assert review.verify_request_acceptance(
        first, generated_request["expected_observed"]
    )["status"] == "passed"
