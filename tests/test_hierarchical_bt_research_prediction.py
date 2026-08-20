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
        "grid_series_id": f"series-{game_id}",
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
                "grid_series_id": "series-g9",
            },
            {
                "game_uid": "g10",
                "date": "2026-01-10T10:00:00Z",
                "blue_team": "C",
                "red_team": "A",
                "league": "LEC",
                "grid_series_id": "series-g10",
            },
        ]
    )


def _source_receipt(ids: list[str]) -> dict[str, object]:
    payload: dict[str, object] = {
        "source_as_of": "2026-01-10T23:00:00Z",
        "source_game_count": len(ids),
        "source_identity_sha256": identity_sha256(ids),
        "accepted_game_ids": sorted(ids),
        "model_eligible_game_count": len(ids),
        "model_eligible_identity_sha256": identity_sha256(ids),
        "model_eligible_game_ids": sorted(ids),
    }
    return {
        **payload,
        "receipt_sha256": hierarchical_bt._research_sha256(payload),
    }


def _run(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    **kwargs: object,
) -> dict[str, object]:
    receipt = _source_receipt(
        [*train["game_uid"].astype(str).tolist(), *validation["game_uid"].astype(str).tolist()]
    )
    return fit_hierarchical_bt_research_prediction(
        train,
        validation,
        cutoff=pd.Timestamp("2026-01-08T23:59:59Z"),
        source_receipt=receipt,
        source_identity_sha256=str(receipt["source_identity_sha256"]),
        **kwargs,
    )


def _fit(**kwargs: object) -> dict[str, object]:
    train = _training()
    validation = _validation()
    return _run(train, validation, **kwargs)


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
    assert report["source"]["source_as_of"] == "2026-01-10T23:00:00+00:00"
    assert report["source"]["receipt_sha256"] == _source_receipt(
        [*train_ids, *validation_ids]
    )["receipt_sha256"]
    assert report["config_sha256"] == hierarchical_bt._research_sha256(report["config"])
    assert report["implementation_sha256"] == hierarchical_bt.HIERARCHICAL_IMPLEMENTATION_SHA256
    assert set(report["terms"]["team_logit"]) == {"a", "b", "c", "d"}
    assert set(report["terms"]["league_logit"]) == {"LCK", "LEC"}
    assert isinstance(report["terms"]["side_logit"], float)


def test_research_prediction_uses_side_and_league_terms() -> None:
    train = _training()
    validation = _validation()
    first = _run(train, validation.iloc[[0]])
    second = _run(train, validation.iloc[[1]])
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
    with_outcomes = _run(
        _training(),
        validation.assign(y_blue_win=["not-read", "not-read"]),
    )
    assert without_outcomes["predictions"] == with_outcomes["predictions"]
    assert without_outcomes["output_sha256"] == with_outcomes["output_sha256"]
    assert all("y_blue_win" not in row for row in without_outcomes["predictions"])


def test_strict_cutoff_rejects_boundary_and_future_training_rows() -> None:
    validation = _validation()
    with pytest.raises(ValueError, match="at or before the strict cutoff"):
        _run(
            _training(),
            validation.assign(date=["2026-01-08T23:59:59Z", "2026-01-10T10:00:00Z"]),
        )
    late_train = _training().copy()
    late_train.loc[0, "date"] = "2026-01-09T00:00:00Z"
    with pytest.raises(ValueError, match="after the strict cutoff"):
        _run(late_train, validation)


def test_missing_validation_ids_are_reported_and_unknown_teams_are_explicit() -> None:
    validation = pd.DataFrame(
        [
            {
                "game_uid": "g9",
                "date": "2026-01-09T10:00:00Z",
                "blue_team": "Ood Team",
                "red_team": "A",
                "league": "LEC",
                "grid_series_id": "series-g9",
            },
            {
                "game_uid": "g10",
                "date": "2026-01-10T10:00:00Z",
                "blue_team": "New Team",
                "red_team": "A",
                "league": "LEC",
                "grid_series_id": "series-g10",
            },
        ]
    )
    train = _training()
    report = _run(train, validation)
    assert report["missing_ids"] == ["g10", "g9"]
    assert report["validation_receipt"]["missing_game_ids"] == ["g10", "g9"]
    assert report["validation_receipt"]["scored_game_ids"] == []
    assert report["missing"]["unseen_team_keys"] == ["new-team", "ood-team"]
    assert report["missing"]["unseen_model_game_ids"] == ["g10", "g9"]
    assert report["missing"]["blockers"]
    assert report["output_row_count"] == 0


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


def test_reordering_train_and_validation_rows_keeps_predictions_and_receipts() -> None:
    baseline = _fit()
    train = _training().iloc[::-1].reset_index(drop=True)
    validation = _validation().iloc[::-1].reset_index(drop=True)
    reordered = _run(train, validation)
    assert baseline["predictions"] == reordered["predictions"]
    assert baseline["output_sha256"] == reordered["output_sha256"]
    assert baseline["train_receipt"] == reordered["train_receipt"]
    assert baseline["validation_receipt"] == reordered["validation_receipt"]


def test_explicit_non_grid_series_column_is_bound_as_conservative_proxy() -> None:
    train = _training().drop(columns=["grid_series_id"]).assign(
        series_id=lambda frame: frame["game_uid"].map(lambda value: f"proxy-{value}")
    )
    validation = _validation().drop(columns=["grid_series_id"]).assign(
        series_id=lambda frame: frame["game_uid"].map(lambda value: f"proxy-{value}")
    )
    report = _run(train, validation)
    series_identity = report["series_identity"]
    assert series_identity["column"] == "series_id"
    assert series_identity["source_types"] == ["explicit_series_id"]
    assert series_identity["authoritative"] is False
    assert report["source"]["series_identity"] == series_identity
    assert report["train_receipt"]["series_identity"] == series_identity
    assert report["validation_receipt"]["series_identity"] == series_identity

    reordered = _run(train.iloc[::-1].reset_index(drop=True), validation.iloc[::-1].reset_index(drop=True))
    assert report["predictions"] == reordered["predictions"]
    assert report["output_sha256"] == reordered["output_sha256"]


def test_source_receipt_and_caller_identity_mutations_fail_closed() -> None:
    train = _training()
    validation = _validation()
    ids = [*train["game_uid"].tolist(), *validation["game_uid"].tolist()]
    receipt = _source_receipt(ids)
    bad_hash = dict(receipt)
    bad_hash["receipt_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="receipt hash"):
        fit_hierarchical_bt_research_prediction(
            train,
            validation,
            cutoff=pd.Timestamp("2026-01-08T23:59:59Z"),
            source_receipt=bad_hash,
            source_identity_sha256=str(receipt["source_identity_sha256"]),
        )
    with pytest.raises(ValueError, match="source identity"):
        fit_hierarchical_bt_research_prediction(
            train,
            validation,
            cutoff=pd.Timestamp("2026-01-08T23:59:59Z"),
            source_receipt=receipt,
            source_identity_sha256="0" * 64,
        )
    duplicate_payload = dict(receipt)
    duplicate_payload["model_eligible_game_ids"] = [*ids, ids[0]]
    duplicate_payload["receipt_sha256"] = hierarchical_bt._research_sha256(
        {key: value for key, value in duplicate_payload.items() if key != "receipt_sha256"}
    )
    with pytest.raises(ValueError, match="canonical and unique"):
        fit_hierarchical_bt_research_prediction(
            train,
            validation,
            cutoff=pd.Timestamp("2026-01-08T23:59:59Z"),
            source_receipt=duplicate_payload,
            source_identity_sha256=str(receipt["source_identity_sha256"]),
        )
    outside_payload = dict(receipt)
    outside_payload["accepted_game_ids"] = sorted(train["game_uid"].tolist())
    outside_payload["source_game_count"] = len(train)
    outside_payload["source_identity_sha256"] = identity_sha256(
        outside_payload["accepted_game_ids"]
    )
    outside_payload["receipt_sha256"] = hierarchical_bt._research_sha256(
        {
            key: value
            for key, value in outside_payload.items()
            if key != "receipt_sha256"
        }
    )
    with pytest.raises(ValueError, match="outside the accepted census"):
        fit_hierarchical_bt_research_prediction(
            train,
            validation,
            cutoff=pd.Timestamp("2026-01-08T23:59:59Z"),
            source_receipt=outside_payload,
            source_identity_sha256=str(outside_payload["source_identity_sha256"]),
        )


def test_series_identity_is_required_and_cannot_cross_the_fold() -> None:
    train = _training()
    validation = _validation()
    validation.loc[0, "grid_series_id"] = "series-g1"
    with pytest.raises(ValueError, match="share grid_series_id"):
        _run(train, validation)
    with pytest.raises(ValueError, match="no safe grid_series_id"):
        _run(train.drop(columns=["grid_series_id"]), _validation())
    unsafe_train = train.copy()
    unsafe_train.loc[1, "blue_team"] = "C"
    unsafe_train.loc[1, "grid_series_id"] = "series-g1"
    with pytest.raises(ValueError, match="multiple team pairs"):
        _run(unsafe_train, _validation())
