from __future__ import annotations

import inspect
from datetime import datetime, timedelta, timezone

from lol_kills.research import composition_signal
from lol_kills.research.public_draft_probability import (
    CandidateConfig,
    PreparedGame,
    _feature_tokens,
    _patch_token,
    fixed_chronological_folds,
)


ROLES = ("top", "jng", "mid", "bot", "sup")


def _item(index: int) -> PreparedGame:
    date = datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(days=index)
    game = {
        "game_uid": f"game-{index}",
        "date": date,
        "y": index % 2,
        "patch": "16.16" if index == 19 else "16.15",
        "blue_team": f"blue-{index}",
        "red_team": f"red-{index}",
        "blue": {
            role: {"champion": f"Blue{role}{index}", "player": f"bp-{role}-{index}"}
            for role in ROLES
        },
        "red": {
            role: {"champion": f"Red{role}{index}", "player": f"rp-{role}-{index}"}
            for role in ROLES
        },
    }
    return PreparedGame(
        game=game,
        region="EMEA",
        scope="TIER1",
        event_kind="DOMESTIC",
        tournament="DOMESTIC",
        comfort_blue=(0.0,) * 5,
        comfort_red=(0.0,) * 5,
        roster_change=False,
    )


def test_patch_identity_preserves_source_and_public_transfer() -> None:
    assert _patch_token("16.15") == "16.15"
    assert _patch_token("16.9") == "16.09"


def test_fixed_folds_keep_final_block_sealed_and_ordered() -> None:
    items = [_item(index) for index in range(20)]
    development, final_holdout, folds = fixed_chronological_folds(items)
    assert development[-1].game["date"] < final_holdout[0].game["date"]
    assert len(folds) == 3
    for train, validation in folds:
        assert train[-1].game["date"] < validation[0].game["date"]
        assert set(item.game["game_uid"] for item in train).isdisjoint(
            item.game["game_uid"] for item in validation
        )


def test_composition_feature_contract_has_no_strength_or_live_state_terms() -> None:
    item = _item(0)
    rows = _feature_tokens(
        [item],
        CandidateConfig(
            "contract",
            atoms=False,
            atom_interactions=False,
            comfort=False,
        ),
        {},
        [],
    )
    keys = {key for row in rows for key, _ in row}
    assert keys
    assert all(
        not set(key.casefold().split("|")).intersection(
            {
                "elo",
                "mu_diff",
                "sigma",
                "rating",
                "momentum",
                "gold",
                "objective",
                "tower",
                "dragon",
                "baron",
                "inhibitor",
                "outcome",
                "r9e",
                "history",
                "form",
            }
        )
        for key in keys
    )


def test_history_candidate_is_attached_to_each_evaluated_window() -> None:
    source = inspect.getsource(composition_signal.evaluate_composition_signal)
    assert 'window_payload["draft_plus_team_history"]' in source
    assert 'windows[-1]["draft_plus_team_history"]' not in source
