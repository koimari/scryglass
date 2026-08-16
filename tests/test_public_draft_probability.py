from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from lol_kills.research.public_draft_probability import (
    AtomRouter,
    CandidateConfig,
    PreparedGame,
    _composition_tokens,
    _context_tokens,
    _fit_predict,
    _group_metrics,
    _patch_token,
    _roster_change_flags,
    _series_cluster_bootstrap,
    _validate_feature_rows,
    candidate_configs,
    canonicalize_player_maps,
    fixed_chronological_folds,
    load_atom_router,
)


ROLES = ("top", "jng", "mid", "bot", "sup")


def _empty_router() -> AtomRouter:
    return AtomRouter(
        vectors_by_source_patch={},
        vector_names=(),
        status_by_source_patch={},
        snapshot_by_source_patch={},
        snapshot_meta={},
    )


def _game(index: int, *, date: datetime | None = None, roster_suffix: str = "") -> dict[str, object]:
    timestamp = date or datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(days=index)
    return {
        "game_uid": f"game-{index}",
        "date": timestamp,
        "y": index % 2,
        "patch": "16.16" if index == 39 else "16.15",
        "league": "LEC",
        "series_id": "",
        "blue_team": "blue-team",
        "red_team": "red-team",
        "blue": {
            role: {
                "champion": f"Blue{role}{index % 2}",
                "player": f"player-blue-{role}{roster_suffix}",
            }
            for role in ROLES
        },
        "red": {
            role: {
                "champion": f"Red{role}{index % 2}",
                "player": f"player-red-{role}{roster_suffix}",
            }
            for role in ROLES
        },
    }


def _item(index: int, *, date: datetime | None = None, roster_suffix: str = "") -> PreparedGame:
    return PreparedGame(
        game=_game(index, date=date, roster_suffix=roster_suffix),
        region="EMEA",
        scope="REGIONAL",
        event_kind="DOMESTIC",
        competition_tier="TIER1",
        tournament="LEC",
        roster_change=False,
        series_cluster=f"series-{index // 2}",
        atom_status="source_patch_unregistered",
        atom_snapshot_patch=None,
    )


def _aggregate(tokens: list[tuple[str, float]]) -> dict[str, float]:
    output: dict[str, float] = {}
    for key, value in tokens:
        output[key] = output.get(key, 0.0) + value
    return output


def test_patch_identity_preserves_source_token() -> None:
    assert _patch_token("16.15") == "16.15"
    assert _patch_token("16.9") == "16.09"


def test_holdout_fraction_is_honored_without_splitting_equal_timestamps() -> None:
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    items = [_item(index, date=start + timedelta(days=index // 2)) for index in range(40)]
    development_10, holdout_10, folds = fixed_chronological_folds(items, 0.10)
    development_30, holdout_30, _ = fixed_chronological_folds(items, 0.30)
    assert len(holdout_10) < len(holdout_30)
    assert abs(len(holdout_10) / len(items) - 0.10) <= 0.05
    assert abs(len(holdout_30) / len(items) - 0.30) <= 0.05
    assert development_10[-1].game["date"] != holdout_10[0].game["date"]
    assert development_30[-1].game["date"] != holdout_30[0].game["date"]
    assert len(folds) == 3
    for train, validation in folds:
        assert train[-1].game["date"] != validation[0].game["date"]


def test_fixed_consumed_holdout_start_overrides_fraction_without_splitting_batch() -> None:
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    items = [_item(index, date=start + timedelta(days=index // 2)) for index in range(40)]
    cutoff = start + timedelta(days=15)
    development, holdout, _ = fixed_chronological_folds(items, 0.10, cutoff)
    assert holdout[0].game["date"] == cutoff
    assert development[-1].game["date"] < cutoff
    assert len(holdout) == 10


def test_draft_features_are_antisymmetric_and_context_stays_separate() -> None:
    item = _item(0)
    swapped_game = dict(item.game)
    swapped_game["blue"] = item.game["red"]
    swapped_game["red"] = item.game["blue"]
    swapped_game["blue_team"] = item.game["red_team"]
    swapped_game["red_team"] = item.game["blue_team"]
    swapped_game["y"] = 1 - int(item.game["y"])
    swapped = replace(item, game=swapped_game)
    config = CandidateConfig("contract", atoms=False, atom_interactions=False)
    original = _aggregate(_composition_tokens(item, config, _empty_router()))
    reversed_side = _aggregate(_composition_tokens(swapped, config, _empty_router()))
    assert original
    assert original.keys() == reversed_side.keys()
    assert all(reversed_side[key] == pytest.approx(-value) for key, value in original.items())
    assert _context_tokens(item, config) == _context_tokens(swapped, config)
    assert all(not key.startswith("CTX|") for key in original)


def test_model_fits_zero_draft_intercept() -> None:
    items = [_item(index) for index in range(60)]
    config = CandidateConfig(
        "zero-intercept",
        ally=False,
        counter=False,
        same_role=False,
        atoms=False,
        atom_interactions=False,
        support=1,
    )
    prediction = _fit_predict(items[:40], items[40:], config, _empty_router())
    assert prediction["draft_intercept"] == 0.0
    assert np.isfinite(prediction["probabilities"]).all()


def test_fitted_draft_logit_flips_sign_while_context_stays_fixed() -> None:
    train = [_item(index) for index in range(40)]
    original = _item(40)
    swapped_game = dict(original.game)
    swapped_game["blue"] = original.game["red"]
    swapped_game["red"] = original.game["blue"]
    swapped_game["blue_team"] = original.game["red_team"]
    swapped_game["red_team"] = original.game["blue_team"]
    swapped = replace(original, game=swapped_game)
    config = CandidateConfig(
        "antisymmetric-logit",
        ally=False,
        counter=False,
        same_role=False,
        atoms=False,
        atom_interactions=False,
        support=1,
    )
    prediction = _fit_predict(train, [original, swapped], config, _empty_router())
    assert prediction["draft_logits"][0] == pytest.approx(-prediction["draft_logits"][1])
    assert prediction["context_logits"][0] == pytest.approx(prediction["context_logits"][1])


def _player_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for side_index, side in enumerate(("Blue", "Red")):
        for role_index, role in enumerate(ROLES):
            rows.append(
                {
                    "game_uid": "oe:game:one",
                    "gameid": "one",
                    "date": "2026-08-15T10:00:00Z",
                    "side": side,
                    "position": role,
                    "playername": f"display-{side}-{role}",
                    "playerid": f"oe:player:{side_index}-{role_index}",
                    "teamname": f"display-{side}",
                    "teamid": f"oe:team:{side_index}",
                    "result": 1 if side == "Blue" else 0,
                    "champion": f"Champion{side_index}{role_index}",
                    "league": "LEC",
                    "patch": "16.16",
                    "oe_patch_token": "16.16",
                }
            )
    return pd.DataFrame(rows)


def test_canonical_map_dedup_uses_stable_player_ids() -> None:
    frame = _player_rows()
    frame = pd.concat((frame, frame.iloc[[0]]), ignore_index=True)
    canonical, metadata = canonicalize_player_maps(frame)
    assert len(canonical) == 10
    assert canonical["game_uid"].nunique() == 1
    assert canonical["playername"].str.startswith("oe:player:").all()
    assert metadata["duplicate_rows_removed"] == 1
    assert metadata["canonical_maps_accepted"] == 1


def test_map_without_stable_player_id_is_excluded() -> None:
    frame = _player_rows()
    frame.loc[0, "playerid"] = None
    canonical, metadata = canonicalize_player_maps(frame)
    assert canonical.empty
    assert metadata["missing_stable_player_id_maps_excluded"] == 1


def test_roster_history_batches_equal_timestamps() -> None:
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    prior = _game(0, date=start)
    changed_first = _game(1, date=start + timedelta(days=1), roster_suffix="-new")
    changed_first["game_uid"] = "a-changed"
    stable_second = _game(2, date=start + timedelta(days=1))
    stable_second["game_uid"] = "z-stable"
    flags = _roster_change_flags([prior, changed_first, stable_second])
    assert flags["a-changed"] is True
    assert flags["z-stable"] is False


def test_atom_router_fails_closed_on_hash_mismatch(tmp_path) -> None:
    supported = tmp_path / "supported.json"
    mismatch = tmp_path / "mismatch.json"
    supported.write_text(json.dumps({"artifact_sha256": "supported", "champions": []}))
    mismatch.write_text(json.dumps({"artifact_sha256": "wrong", "champions": []}))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "atom_snapshots": [
                    {"patch": "26.16", "bridge_artifact_sha256": "supported", "source": {"locator": str(supported)}},
                    {"patch": "26.15", "bridge_artifact_sha256": "expected", "source": {"locator": str(mismatch)}},
                ],
                "mappings": [
                    {"oe_token": "16.16", "ambiguity_status": "none", "atom_snapshot_patch": "26.16"},
                    {"oe_token": "16.15", "ambiguity_status": "none", "atom_snapshot_patch": "26.15"},
                ],
            }
        )
    )
    router = load_atom_router(manifest)
    assert router.status_by_source_patch["16.16"] == "available"
    assert router.snapshot_by_source_patch["16.16"] == "26.16"
    assert router.status_by_source_patch["16.15"] == "snapshot_artifact_hash_mismatch"
    assert "16.15" not in router.vectors_by_source_patch


def test_candidate_grid_has_no_duplicate_configuration() -> None:
    configs = candidate_configs(include_atoms=True)
    signatures = {
        tuple(value for key, value in config.__dict__.items() if key != "name")
        for config in configs
    }
    assert len(signatures) == len(configs)


def test_feature_allowlist_rejects_unknown_and_forbidden_keys() -> None:
    _validate_feature_rows([[("CH|TOP|AATROX", 1.0)]], ("CH|",))
    with pytest.raises(AssertionError, match="outside the allowlist"):
        _validate_feature_rows([[("TEAM|T1", 1.0)]], ("CH|",))
    with pytest.raises(AssertionError, match="forbidden feature"):
        _validate_feature_rows([[("CH|TOP|ELO", 1.0)]], ("CH|",))


def test_series_cluster_bootstrap_reports_series_unit_and_median() -> None:
    items = [_item(index) for index in range(40)]
    probabilities = np.linspace(0.2, 0.8, len(items))
    result = _series_cluster_bootstrap(items, probabilities, reps=20, seed=461)
    assert result["unit"] == "series"
    assert result["clusters"] == 20
    assert result["reps"] == 20
    assert "median" in result["auc"]
    assert "mean" not in result["auc"]


def test_subgroup_labels_do_not_fall_back_across_fields() -> None:
    items = [_item(index) for index in range(4)]
    probabilities = np.asarray([0.2, 0.8, 0.3, 0.7])
    assert set(_group_metrics(items, probabilities, "event_kind")) == {"DOMESTIC"}
    assert set(_group_metrics(items, probabilities, "competition_tier")) == {"TIER1"}
    assert set(_group_metrics(items, probabilities, "scope")) == {"REGIONAL"}
