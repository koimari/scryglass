from __future__ import annotations

import json

import pandas as pd
import pytest

from lol_kills.ratings import hierarchical_bt
from lol_kills.ratings.hierarchical_bt import (
    HierarchicalBTConfig,
    fit_hierarchical_bt_research_prediction,
)
from lol_kills.v2.tierlists.accepted_census import identity_sha256


def _map(
    game_id: str,
    date: str,
    blue: str,
    red: str,
    league: str,
    result: int,
) -> dict[str, object]:
    return {
        "game_uid": game_id,
        "date": date,
        "blue_team": blue,
        "red_team": red,
        "league": league,
        "y_blue_win": result,
    }


def _training() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _map("g1", "2026-01-01T10:00:00Z", "A", "B", "LEC", 1),
            _map("g2", "2026-01-02T10:00:00Z", "B", "A", "LEC", 0),
            _map("g3", "2026-01-03T10:00:00Z", "C", "D", "LCK", 1),
            _map("g4", "2026-01-04T10:00:00Z", "D", "C", "LCK", 0),
            _map("g5", "2026-01-05T10:00:00Z", "A", "C", "MSI", 1),
            _map("g6", "2026-01-06T10:00:00Z", "C", "A", "MSI", 0),
            _map("g7", "2026-01-07T10:00:00Z", "A", "C", "MSI", 1),
            _map("g8", "2026-01-08T10:00:00Z", "D", "B", "LEC", 1),
        ]
    )


def _validation() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_uid": "g9",
                "date": "2026-01-09T10:00:00Z",
                "blue_team": "A",
                "red_team": "C",
                "league": "LEC",
            },
            {
                "game_uid": "g10",
                "date": "2026-01-10T10:00:00Z",
                "blue_team": "C",
                "red_team": "A",
                "league": "LEC",
            },
        ]
    )


def _fit(**kwargs: object) -> dict[str, object]:
    train = _training()
    validation = _validation()
    return fit_hierarchical_bt_research_prediction(
        train,
        validation,
        cutoff=pd.Timestamp("2026-01-08T23:59:59Z"),
        source_identity_sha256=identity_sha256([*train.game_uid, *validation.game_uid]),
        **kwargs,
    )


def test_research_prediction_has_exact_receipts_and_digests() -> None:
    report = _fit()
    train_ids = [f"g{i}" for i in range(1, 9)]
    validation_ids = ["g10", "g9"]
    assert report["schema_version"] == "scryglass:hierarchical-bt-map-prediction:v1"
    assert report["authority"] == "research_only"
    assert report["writes_artifacts"] is False
    assert report["strict_cutoff"] is True
    assert report["train_receipt"]["game_ids"] == train_ids
    assert report["validation_receipt"]["game_ids"] == sorted(validation_ids)
    assert report["train_receipt"]["identity_sha256"] == identity_sha256(train_ids)
    assert report["validation_receipt"]["identity_sha256"] == identity_sha256(validation_ids)
    assert report["source_identity_sha256"] == identity_sha256([*train_ids, *validation_ids])
    assert report["config_sha256"] == hierarchical_bt._research_sha256(report["config"])
    assert report["implementation_sha256"] == hierarchical_bt.HIERARCHICAL_IMPLEMENTATION_SHA256
    assert set(report["terms"]["team_logit"]) == {"a", "b", "c", "d"}
    assert set(report["terms"]["league_logit"]) == {"LCK", "LEC"}
    assert isinstance(report["terms"]["side_logit"], float)


def test_research_prediction_uses_side_and_league_terms() -> None:
    train = _training()
    validation = _validation()
    first = fit_hierarchical_bt_research_prediction(
        train,
        validation.iloc[[0]],
        cutoff=pd.Timestamp("2026-01-08T23:59:59Z"),
    )
    second = fit_hierarchical_bt_research_prediction(
        train,
        validation.iloc[[1]],
        cutoff=pd.Timestamp("2026-01-08T23:59:59Z"),
    )
    first_row = first["predictions"][0]
    second_row = second["predictions"][0]
    assert first_row["side_term"] == 1.0
    assert second_row["side_term"] == -1.0
    assert first_row["league_a_known"]
    assert first_row["league_b_known"]
    assert first_row["team_logit"] == pytest.approx(-second_row["team_logit"])
    assert first_row["league_logit"] == pytest.approx(-second_row["league_logit"])
    assert first_row["side_logit"] == pytest.approx(second_row["side_logit"])
    assert first_row["predicted_logit"] == pytest.approx(
        first_row["team_logit"] + first_row["league_logit"] + first_row["side_logit"]
    )


def test_validation_outcomes_are_not_required_or_consumed() -> None:
    validation = _validation()
    without_outcomes = _fit()
    with_outcomes = fit_hierarchical_bt_research_prediction(
        _training(),
        validation.assign(y_blue_win=["not-read", "not-read"]),
        cutoff=pd.Timestamp("2026-01-08T23:59:59Z"),
        source_identity_sha256=identity_sha256([*list(_training().game_uid), *list(validation.game_uid)]),
    )
    assert without_outcomes["predictions"] == with_outcomes["predictions"]
    assert without_outcomes["output_sha256"] == with_outcomes["output_sha256"]
    assert all("y_blue_win" not in row for row in without_outcomes["predictions"])


def test_strict_cutoff_rejects_boundary_and_future_training_rows() -> None:
    validation = _validation()
    with pytest.raises(ValueError, match="at or before the strict cutoff"):
        fit_hierarchical_bt_research_prediction(
            _training(),
            validation.assign(date=["2026-01-08T23:59:59Z", "2026-01-10T10:00:00Z"]),
            cutoff=pd.Timestamp("2026-01-08T23:59:59Z"),
        )
    late_train = _training().copy()
    late_train.loc[0, "date"] = "2026-01-09T00:00:00Z"
    with pytest.raises(ValueError, match="after the strict cutoff"):
        fit_hierarchical_bt_research_prediction(
            late_train,
            validation,
            cutoff=pd.Timestamp("2026-01-08T23:59:59Z"),
        )


def test_missing_validation_ids_are_reported_and_unknown_teams_are_explicit() -> None:
    validation = pd.DataFrame(
        [
            {
                "game_uid": "g9",
                "date": "2026-01-09T10:00:00Z",
                "blue_team": None,
                "red_team": "A",
                "league": "LEC",
            },
            {
                "game_uid": "g10",
                "date": "2026-01-10T10:00:00Z",
                "blue_team": "New Team",
                "red_team": "A",
                "league": "LEC",
            },
        ]
    )
    train = _training()
    report = fit_hierarchical_bt_research_prediction(
        train,
        validation,
        cutoff=pd.Timestamp("2026-01-08T23:59:59Z"),
    )
    assert report["missing_ids"] == ["g9"]
    assert report["validation_receipt"]["missing_game_ids"] == ["g9"]
    assert report["validation_receipt"]["scored_game_ids"] == ["g10"]
    assert report["missing"]["unseen_team_keys"] == ["new-team"]
    assert report["missing"]["unseen_model_game_ids"] == ["g10"]
    assert report["output_row_count"] == 1


def test_output_hash_is_canonical_and_config_changes_are_bound() -> None:
    report = _fit(cfg=HierarchicalBTConfig(side_l2=200.0))
    payload = json.dumps(
        report["predictions"],
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    import hashlib

    assert report["output_sha256"] == hashlib.sha256(payload).hexdigest()
    other = _fit(cfg=HierarchicalBTConfig(side_l2=50.0))
    assert report["config_sha256"] != other["config_sha256"]
