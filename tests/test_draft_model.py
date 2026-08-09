from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone

import pytest

import lol_kills.draft_model as draft_model


def synthetic_rows(series_count: int = 30) -> list[dict]:
    champion_sets = [
        ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
        ["A", "B", "C", "D", "E", "K", "L", "M", "N", "O"],
        ["F", "G", "H", "I", "J", "K", "L", "M", "N", "O"],
    ]
    rows = []
    for series_index in range(series_count):
        for game_index in range(2):
            champions = champion_sets[(series_index + game_index) % len(champion_sets)]
            rows.append(
                {
                    "game_id": f"series-{series_index}_{game_index + 1}",
                    "series_id": f"series-{series_index}",
                    "date": f"2026-01-{1 + series_index // 2:02d}T{series_index % 2:02d}:00:00",
                    "league": "LCK",
                    "patch": "16.01",
                    "total_kills": 22 + 3 * ("K" in champions) + (series_index % 4),
                    "champs": champions,
                    "roles": [],
                }
            )
    return rows


def supported_payload() -> dict:
    return {
        "meta": {"data_cutoff": "2026-07-25T00:00:00+00:00"},
        "model": {
            "intercept": 25.0,
            "league_effects": {"LCK": 1.0},
            "champion_effects": {champion: 0.0 for champion in "ABCDEFGHIJ"},
            "baseline_mean": 25.0,
            "predictive_sd": 7.0,
        },
        "evaluation": {
            "authority": {"predictive_probability_supported": True},
            "splits": {"test": {"by_patch": {"16.14": 20}}},
            "test": {
                "by_league": {
                    "LCK": {"predictive_probability_supported": True},
                }
            },
            "calibration": {"residuals": [-8.0, -2.0, 0.0, 3.0, 9.0]},
        },
    }


def test_chronological_split_keeps_series_disjoint_and_ordered() -> None:
    split = draft_model.chronological_series_split(synthetic_rows())
    identities = {
        name: {row["series_id"] for row in rows}
        for name, rows in split.items()
    }
    assert identities["train"].isdisjoint(identities["calibration"])
    assert identities["train"].isdisjoint(identities["test"])
    assert identities["calibration"].isdisjoint(identities["test"])
    assert max(row["date"] for row in split["train"]) <= min(
        row["date"] for row in split["calibration"]
    )
    assert max(row["date"] for row in split["calibration"]) <= min(
        row["date"] for row in split["test"]
    )


def test_test_labels_do_not_change_training_or_calibration_fit(monkeypatch) -> None:
    monkeypatch.setattr(draft_model, "MIN_CALIBRATION_GAMES", 5)
    monkeypatch.setattr(draft_model, "MIN_TEST_GAMES", 5)
    monkeypatch.setattr(draft_model, "MIN_LEAGUE_TEST_GAMES", 5)
    rows = synthetic_rows()
    original = draft_model.evaluate_chronological_holdout(
        rows, min_champ_games=1, lam=5.0
    )
    changed = deepcopy(rows)
    test_ids = {
        row["game_id"]
        for row in draft_model.chronological_series_split(rows)["test"]
    }
    for row in changed:
        if row["game_id"] in test_ids:
            row["total_kills"] += 20
    mutated = draft_model.evaluate_chronological_holdout(
        changed, min_champ_games=1, lam=5.0
    )
    assert original["calibration"]["residuals"] == mutated["calibration"]["residuals"]
    assert original["test"]["model"]["rmse"] != mutated["test"]["model"]["rmse"]


def test_pricing_eligibility_fails_closed_for_stale_or_unseen_patch(monkeypatch) -> None:
    monkeypatch.setattr(draft_model, "MIN_LEAGUE_TEST_GAMES", 2)
    payload = supported_payload()
    stale = draft_model.pricing_eligibility(
        payload,
        champions=list("ABCDEFGHIJ"),
        league="LCK",
        patch="16.14",
        as_of=datetime(2026, 8, 20, tzinfo=timezone.utc),
    )
    assert stale["status"] == "unavailable"
    assert "data_stale" in stale["blockers"]

    unseen = draft_model.pricing_eligibility(
        payload,
        champions=list("ABCDEFGHIJ"),
        league="LCK",
        patch="16.15",
        as_of=datetime(2026, 7, 30, tzinfo=timezone.utc),
    )
    assert unseen["status"] == "unavailable"
    assert "exact_patch_holdout_unavailable:16.15" in unseen["blockers"]


def test_price_under_withholds_probability_when_authority_is_missing() -> None:
    payload = supported_payload()
    payload["evaluation"]["authority"]["predictive_probability_supported"] = False
    priced = draft_model.price_under(
        payload,
        champions=list("ABCDEFGHIJ"),
        league="LCK",
        patch="16.14",
        as_of=datetime(2026, 7, 30, tzinfo=timezone.utc),
        line=32.5,
        odds=1.80,
    )
    assert priced["classification"] == "WITHHELD"
    assert priced["under_probability"] is None
    assert priced["edge_pp"] is None


def test_price_under_uses_calibration_residuals_only_after_all_gates(monkeypatch) -> None:
    monkeypatch.setattr(draft_model, "MIN_CALIBRATION_GAMES", 5)
    monkeypatch.setattr(draft_model, "MIN_LEAGUE_TEST_GAMES", 2)
    priced = draft_model.price_under(
        supported_payload(),
        champions=list("ABCDEFGHIJ"),
        league="LCK",
        patch="16.14",
        as_of=datetime(2026, 7, 30, tzinfo=timezone.utc),
        line=32.5,
        odds=1.80,
    )
    assert priced["eligibility"]["status"] == "supported"
    assert priced["under_probability"] is not None
    assert priced["classification"] in {"POSITIVE_MODEL_EV", "NEGATIVE_MODEL_EV"}


@pytest.mark.parametrize("line", [32.0, 32.25])
def test_price_under_rejects_push_or_quarter_lines(line: float) -> None:
    with pytest.raises(ValueError):
        draft_model.price_under(
            supported_payload(),
            champions=list("ABCDEFGHIJ"),
            league="LCK",
            patch="16.14",
            as_of=datetime(2026, 7, 30, tzinfo=timezone.utc),
            line=line,
            odds=1.80,
        )
