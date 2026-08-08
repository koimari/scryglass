from __future__ import annotations

import hashlib
import json
import warnings
import zipfile

import numpy as np
import pandas as pd

from lol_kills.research.elemental_drake_model import (
    ELEMENTS,
    _bootstrap_first,
    _design_first,
    _diagnostics,
    _fit,
    build_model_artifact,
    pregame_strengths,
)
from lol_kills.research.elemental_drake_audit import audit_compact_cohort
from lol_kills.research.elemental_drakes import (
    _composition_fit,
    build_artifact,
    load_role_catalog,
    normalize_dragon_type,
    parse_normalized_grid,
    summarize_cohort,
)


def test_normalize_dragon_type_covers_riot_labels() -> None:
    assert normalize_dragon_type("fire") == "infernal"
    assert normalize_dragon_type("EarthDragon") == "mountain"
    assert normalize_dragon_type("OceanDrake") == "ocean"
    assert normalize_dragon_type("air") == "cloud"
    assert normalize_dragon_type("baron") is None


def test_parse_normalized_grid_keeps_state_before_outcome(tmp_path) -> None:
    game_id = "game-1"
    team_blue = {
        "id": "blue",
        "netWorth": 10_500,
        "loadoutValue": 7_800,
        "money": 2_700,
        "objectives": [],
        "players": [{"id": "player-blue", "name": "Carry", "netWorth": 2_500}],
    }
    team_red = {
        "id": "red",
        "netWorth": 9_800,
        "loadoutValue": 7_350,
        "money": 2_450,
        "objectives": [],
        "players": [{"id": "player-red", "name": "Carry", "netWorth": 2_200}],
    }
    pre_state = {
        "games": [
            {
                "id": game_id,
                "clock": {"currentSeconds": 419},
                "titleVersion": {"name": "16.14"},
                "teams": [team_blue, team_red],
            }
        ]
    }
    post_state = {
        "games": [
            {
                "id": game_id,
                "clock": {"currentSeconds": 420},
                "titleVersion": {"name": "16.14"},
                "teams": [
                    {**team_blue, "netWorth": 10_625},
                    team_red,
                ],
            }
        ]
    }
    envelopes = [
        {
            "seriesId": "series-1",
            "occurredAt": "2026-07-27T11:59:00Z",
            "events": [
                {
                    "type": "team-picked-character",
                    "actor": {
                        "type": "team",
                        "id": "blue",
                        "state": {
                            "id": "blue",
                            "name": "Blue Example",
                            "side": "red",
                        },
                    },
                    "target": {"state": {"name": "Syndra"}},
                    "seriesState": pre_state,
                },
            ],
        },
        {
            "seriesId": "series-1",
            "occurredAt": "2026-07-27T12:00:00Z",
            "events": [
                {
                    "type": "player-completed-slayInfernalDrake",
                    "actor": {
                        "type": "player",
                        "id": "jungler",
                        "state": {
                            "id": "jungler",
                            "name": "Jungler",
                            "teamId": "blue",
                            "side": "blue",
                        },
                    },
                    "seriesState": post_state,
                },
            ],
        },
        {
            "seriesId": "series-1",
            "occurredAt": "2026-07-27T12:30:00Z",
            "events": [
                {
                    "type": "team-won-game",
                    "actor": {
                        "type": "team",
                        "id": "blue",
                        "state": {
                            "id": "blue",
                            "name": "Blue Example",
                            "side": "red",
                        },
                    },
                    "seriesState": post_state,
                }
            ],
        },
    ]
    archive = tmp_path / "events_1_grid.jsonl.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(
            "events_1_grid.jsonl",
            "\n".join(json.dumps(row) for row in envelopes),
        )

    games = parse_normalized_grid(archive)

    assert len(games) == 1
    assert games[0]["complete"] is True
    assert games[0]["patch"] == "16.14"
    assert games[0]["winnerTeamId"] == "blue"
    assert games[0]["dragonEvents"] == [
        {
            "element": "infernal",
            "occurredAt": "2026-07-27T12:00:00Z",
            "timeSeconds": 420,
            "ownerTeamId": "blue",
            "ownerSide": "red",
            "stateTiming": "previous-envelope",
            "stateLagSeconds": 60.0,
            "ownerNetWorth": 10_500,
            "opponentNetWorth": 9_800,
            "goldDiff": 700,
            "ownerLoadoutValue": 7_800,
            "opponentLoadoutValue": 7_350,
            "loadoutDiff": 450,
            "ownerUnspentMoney": 2_700,
            "opponentUnspentMoney": 2_450,
            "unspentMoneyDiff": 250,
            "ownerTopPlayerNetWorth": 2_500,
            "opponentTopPlayerNetWorth": 2_200,
            "topPlayerNetWorthDiff": 300,
            "ownerKills": 0,
            "opponentKills": 0,
            "ownerTowers": 0,
            "opponentTowers": 0,
            "globalIndex": 1,
            "ownerStack": 1,
        }
    ]
    assert games[0]["teams"][0]["champions"] == ["Syndra"]
    assert games[0]["teams"][0]["playerIds"] == ["player-blue"]


def test_cohort_summary_does_not_turn_counts_into_effects() -> None:
    summary = summarize_cohort(
        [
                {
                    "complete": True,
                    "winnerTeamId": "blue",
                    "teams": [{"id": "blue"}, {"id": "red"}],
                    "dragonEvents": [
                    {
                        "element": "ocean",
                        "globalIndex": 1,
                        "timeSeconds": 390,
                    },
                    {
                        "element": "hextech",
                        "globalIndex": 2,
                        "timeSeconds": 710,
                    },
                ],
            }
        ]
    )

    assert summary["firstDrakeDistribution"]["ocean"] == 1
    assert summary["medianFirstDrakeSeconds"] == 390
    assert summary["outcomeModel"]["status"] == "gated"


def test_public_artifact_embeds_exact_hashed_explorer_model(tmp_path) -> None:
    model_path = tmp_path / "elemental_drake_explorer_model.json"
    payload = {
        "schemaVersion": "elemental-drake-explorer-v3",
        "status": "ready",
        "cohort": {"completedGames": 6_504},
    }
    model_path.write_text(
        json.dumps(payload, separators=(",", ":")),
        encoding="utf-8",
    )
    raw = model_path.read_bytes()

    artifact = build_artifact(
        raw_dir=tmp_path,
        explorer_model_path=model_path,
        audit_path=tmp_path / "missing-audit.json",
    )

    assert artifact["explorerModel"] == payload
    assert artifact["metadata"]["explorerModelSource"] == {
        "file": model_path.name,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "schemaVersion": "elemental-drake-explorer-v3",
    }


def test_composition_fit_treats_all_five_champions_as_recipients() -> None:
    composition = [
        {"champion": "Ornn"},
        {"champion": "Sejuani"},
        {"champion": "Orianna"},
        {"champion": "Jinx"},
        {"champion": "Lulu"},
    ]

    fit = _composition_fit(composition)

    for annotation in fit.values():
        assert annotation["recipients"] == [
            "Ornn",
            "Sejuani",
            "Orianna",
            "Jinx",
            "Lulu",
        ]
        assert annotation["recipientCount"] == 5
        assert set(annotation["higherConversionCandidates"]) <= set(
            annotation["recipients"]
        )
        assert "directUsers" not in annotation
        assert "all five champions receive the buff" in annotation["basis"]


def test_role_catalog_projects_only_aggregate_role_counts(tmp_path) -> None:
    source = tmp_path / "draft_players.json"
    source.write_text(
        json.dumps(
            {
                "source": "reviewed role source",
                "players": [
                    {
                        "game_id": "private-game-1",
                        "player": "Private Player",
                        "role": "Top",
                        "champion": "Ornn",
                    },
                    {
                        "game_id": "private-game-2",
                        "player": "Another Player",
                        "role": "Top",
                        "champion": "Ornn",
                    },
                    {
                        "game_id": "private-game-1",
                        "player": "Private Player",
                        "role": "Support",
                        "champion": "Lulu",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    catalog = load_role_catalog(source)

    assert catalog["status"] == "ready"
    assert catalog["appearances"] == 3
    assert catalog["games"] == 2
    top = next(row for row in catalog["roles"] if row["role"] == "Top")
    assert top["champions"] == [{"name": "Ornn", "appearances": 2}]
    assert "private-game-1" not in json.dumps(catalog)
    assert "Private Player" not in json.dumps(catalog)


def test_pregame_strength_uses_only_prior_results() -> None:
    games = pd.DataFrame(
        [
            {
                "series_id": "series-1",
                "game_id": "game-1",
                "date": "2026-01-01T12:00:00Z",
                "complete": True,
                "winner_team_id": "a",
                "team_1_id": "a",
                "team_1_name": "Alpha",
                "team_1_players": json.dumps(["a1", "a2", "a3", "a4", "a5"]),
                "team_2_id": "b",
                "team_2_name": "Beta",
                "team_2_players": json.dumps(["b1", "b2", "b3", "b4", "b5"]),
            },
            {
                "series_id": "series-1",
                "game_id": "game-2",
                "date": "2026-01-01T13:00:00Z",
                "complete": True,
                "winner_team_id": "b",
                "team_1_id": "a",
                "team_1_name": "Alpha",
                "team_1_players": json.dumps(["a1", "a2", "a3", "a4", "a5"]),
                "team_2_id": "b",
                "team_2_name": "Beta",
                "team_2_players": json.dumps(["b1", "b2", "b3", "b4", "b5"]),
            },
        ]
    )
    events = pd.DataFrame(
        [
            {
                "series_id": "series-1",
                "game_id": "game-1",
                "occurred_at": "2026-01-01T12:07:00Z",
            },
            {
                "series_id": "series-1",
                "game_id": "game-2",
                "occurred_at": "2026-01-01T13:07:00Z",
            },
        ]
    )

    strengths = pregame_strengths(games, events).set_index("game_id")

    assert strengths.loc["game-1", "team_1_org_elo"] == 1500
    assert strengths.loc["game-1", "team_1_player_elo"] == 1500
    assert strengths.loc["game-2", "team_1_org_elo"] > 1500
    assert strengths.loc["game-2", "team_1_player_elo"] > 1500


def test_first_drake_design_is_finite_and_has_each_element() -> None:
    rows = []
    for index, element in enumerate(ELEMENTS):
        row = {
            "element": element,
            "gold_diff_k": index / 10,
            "loadout_diff_k": index / 20,
            "unspent_money_diff_k": -index / 30,
            "top_player_net_worth_diff_k": index / 40,
            "minute": 7 + index,
            "kill_diff": index % 2,
            "tower_diff": 0,
            "blue": index % 2,
            "org_elo_diff": 0.1,
            "player_elo_diff": -0.1,
            "roster_coverage": 5,
            "league": "LCK",
            "patch": "16.14",
            "year": "2026",
        }
        from lol_kills.draft_archetypes import ARCHETYPE_NAMES

        for tag in ARCHETYPE_NAMES:
            row[f"diff_{tag}"] = 0
        rows.append(row)

    design = _design_first(pd.DataFrame(rows))

    assert design.notna().all().all()
    assert all(f"element_{element}" in design.columns for element in ELEMENTS)


def test_series_bootstrap_executes_with_cluster_multiplicities() -> None:
    from lol_kills.draft_archetypes import ARCHETYPE_NAMES

    rows = []
    for series_index in range(6):
        for element_index, element in enumerate(ELEMENTS):
            row = {
                "series_id": f"series-{series_index}",
                "owner_won": (series_index + element_index) % 2,
                "element": element,
                "gold_diff_k": series_index / 10,
                "loadout_diff_k": element_index / 20,
                "unspent_money_diff_k": 0,
                "top_player_net_worth_diff_k": 0,
                "minute": 7 + element_index / 10,
                "kill_diff": 0,
                "tower_diff": 0,
                "blue": series_index % 2,
                "org_elo_diff": 0,
                "player_elo_diff": 0,
                "roster_coverage": 5,
                "league": "LCK",
                "patch": "16.14",
                "year": "2026",
            }
            for tag in ARCHETYPE_NAMES:
                row[f"diff_{tag}"] = 0
            rows.append(row)
    frame = pd.DataFrame(rows)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        intervals = _bootstrap_first(
            frame,
            frame,
            replicates=2,
            seed=461,
            alpha=0.1,
        )

    assert set(intervals) == set(ELEMENTS)
    assert all(
        values["lowPp"] <= values["highPp"]
        for values in intervals.values()
    )


def test_regularized_logit_prediction_is_finite_for_extreme_counterfactual() -> None:
    design = pd.DataFrame(
        {
            "state": np.linspace(-2, 2, 120),
            "rare_interaction": [0.0] * 114 + [1.0] * 6,
            "constant": [1.0] * 120,
        }
    )
    outcome = np.array([index % 2 for index in range(120)])

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        fit = _fit(design, outcome)
        probability = fit.predict(
            pd.DataFrame(
                {
                    "state": [-1_000_000.0, 1_000_000.0],
                    "rare_interaction": [0.0, 1.0],
                    "constant": [1.0, 1.0],
                }
            )
        )

    assert np.isfinite(probability).all()
    assert ((probability >= 0) & (probability <= 1)).all()


def test_regularized_logit_accepts_cluster_bootstrap_weights() -> None:
    design = pd.DataFrame(
        {
            "state": np.linspace(-1, 1, 80),
            "interaction": [0.0, 1.0] * 40,
        }
    )
    outcome = np.array([index % 2 for index in range(80)])
    weights = np.array([3.0 if index < 20 else 1.0 for index in range(80)])

    fit = _fit(design, outcome, sample_weight=weights)
    probability = fit.predict(design.iloc[:4])

    assert np.isfinite(probability).all()
    assert ((probability >= 0) & (probability <= 1)).all()


def test_public_diagnostics_use_prespecified_unseen_window() -> None:
    rows = []
    for series, date in (
        ("train-1", "2026-01-01T12:00:00Z"),
        ("train-2", "2026-01-15T12:00:00Z"),
        ("train-3", "2026-02-15T12:00:00Z"),
        ("holdout", "2026-03-15T12:00:00Z"),
        ("post", "2026-05-15T12:00:00Z"),
    ):
        for game_index, outcome in enumerate((0, 1)):
            rows.append(
                {
                    "series_id": series,
                    "game_id": f"{series}-{game_index}",
                    "date": pd.Timestamp(date),
                    "owner_won": outcome,
                    "signal": float(outcome),
                    "noise": float(game_index + len(rows)),
                }
            )
    frame = pd.DataFrame(rows)

    diagnostics, _ = _diagnostics(
        frame,
        lambda values: values[["signal", "noise"]],
    )

    assert diagnostics["trainSeries"] == 3
    assert diagnostics["holdoutSeries"] == 1
    assert diagnostics["holdoutStart"].startswith("2026-03-15")
    assert diagnostics["postHoldoutRows"] == 2


def test_model_artifact_remains_gated_below_prespecified_threshold(tmp_path) -> None:
    games_path = tmp_path / "games.parquet"
    events_path = tmp_path / "events.parquet"
    pd.DataFrame(
        [
            {
                "series_id": "series-1",
                "game_id": "game-1",
                "complete": True,
                "winner_team_id": "blue",
                "team_1_id": "blue",
                "team_2_id": "red",
            }
        ]
    ).to_parquet(games_path)
    pd.DataFrame([{"game_id": "game-1"}]).to_parquet(events_path)

    artifact = build_model_artifact(
        games_path=games_path,
        events_path=events_path,
        min_games=6_000,
    )

    assert artifact["status"] == "gated"
    assert artifact["games"] == 1
    assert artifact["requiredGames"] == 6_000


def test_compact_audit_fails_closed_below_launch_threshold(tmp_path) -> None:
    games_path = tmp_path / "games.parquet"
    events_path = tmp_path / "events.parquet"
    pd.DataFrame(
        [
            {
                "series_id": "series-1",
                "game_id": "game-1",
                "patch": "16.14",
                "complete": True,
                "winner_team_id": "blue",
                "team_1_id": "blue",
                "team_1_side": "blue",
                "team_1_champions": json.dumps(["A", "B", "C", "D", "E"]),
                "team_1_players": json.dumps(["1", "2", "3", "4", "5"]),
                "team_1_player_ids": json.dumps(["id1", "id2", "id3", "id4", "id5"]),
                "team_2_id": "red",
                "team_2_side": "red",
                "team_2_champions": json.dumps(["F", "G", "H", "I", "J"]),
                "team_2_players": json.dumps(["6", "7", "8", "9", "10"]),
                "team_2_player_ids": json.dumps(["id6", "id7", "id8", "id9", "id10"]),
            }
        ]
    ).to_parquet(games_path)
    pd.DataFrame(
        [
            {
                "series_id": "series-1",
                "game_id": "game-1",
                "global_index": 1,
                "element": "infernal",
                "owner_team_id": "blue",
                "owner_side": "blue",
                "state_timing": "previous-envelope",
                "state_lag_seconds": 1.0,
                "owner_net_worth": 10_000,
                "opponent_net_worth": 9_500,
            }
        ]
    ).to_parquet(events_path)

    report = audit_compact_cohort(
        games_path,
        events_path,
        required_games=6_000,
    )

    assert report["status"] == "fail"
    assert report["games"]["completed"] == 1
    assert report["events"]["validatedFirstDrakes"] == 1
    assert any("6000 required" in error for error in report["errors"])
