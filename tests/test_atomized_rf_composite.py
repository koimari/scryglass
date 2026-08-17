from __future__ import annotations

import hashlib
from copy import deepcopy

import pandas as pd
import numpy as np
import pytest
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from lol_kills.research.atomized_rf_composite import (
    CATEGORICAL_CONTEXT_COLUMNS,
    FEATURE_AVAILABILITY_COLUMNS,
    MODEL_COLUMNS,
    RATING_BATCH_POLICY,
    RATING_CONTEXT_FIELDS,
    RATING_CONTEXT_SCHEMA,
    RATING_RECEIPT_SCHEMA,
    AtomizedResearchError,
    RFConfig,
    RunningStat,
    _calibration_outer_audit,
    _categorical_context,
    _expanding_series_folds,
    _cluster_bootstrap_differences,
    _controlled_feature_lineup,
    _emit_role_metric_families,
    _equal_weight_team_forecast,
    _locked_rating_authority,
    _matched_comparison_config,
    _momentum_features,
    _phase_curve_features,
    _rating_batch_receipt_sha256,
    _resolved_roster_sha256,
    _shrunk_metric_mean,
    _strict_canonical_sha256,
    _unique_player_map_support,
    _validate_no_current_state_features,
    _write_json,
    exact_mechanic_keys,
    feature_group_coverage_report,
    layer_a_build_preflight,
    normalize_source_patch,
    phase_coverage_report,
)


def test_layer_a_preflight_binds_all_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = {
        name: tmp_path / name
        for name in ("base.parquet", "maps.parquet", "players.parquet", "teams.parquet", "raw.csv")
    }
    for name, path in paths.items():
        path.write_bytes(f"source:{name}".encode("utf-8"))
    base_sha256 = hashlib.sha256(paths["base.parquet"].read_bytes()).hexdigest()
    raw_sha256 = hashlib.sha256(paths["raw.csv"].read_bytes()).hexdigest()
    monkeypatch.setattr(
        "lol_kills.research.atomized_rf_composite.RAW_2026_IDENTITY_SHA256",
        raw_sha256,
    )

    receipt = layer_a_build_preflight(
        base_dataset=paths["base.parquet"],
        maps_path=paths["maps.parquet"],
        players_path=paths["players.parquet"],
        team_path=paths["teams.parquet"],
        identity_overlay_players_path=paths["players.parquet"],
        raw_identity_overlay_csv=paths["raw.csv"],
        cache_dir=tmp_path / "cache",
        expected_base_sha256=base_sha256,
    )

    assert receipt["status"] == "frozen_inputs_ready_for_matrix_build"
    assert receipt["sources"]["base_dataset"] == base_sha256
    assert receipt["sources"]["raw_identity_overlay_csv"] == raw_sha256
    assert receipt["matrix_path"].endswith(".parquet")
    assert receipt["authority"] == {
        "model_fit": False,
        "public_probability": False,
        "promotion": False,
    }


def test_controlled_feature_lineup_changes_only_role_matched_champions() -> None:
    observed = []
    for side in ("Blue", "Red"):
        for index, role in enumerate(("top", "jng", "mid", "bot", "sup")):
            observed.append(
                {
                    "game_uid": "map-1",
                    "date": "2026-08-17T12:00:00Z",
                    "side": side,
                    "position": role,
                    "champion": f"{side}-champion-{index}",
                    "playername": f"{side}-player-{index}",
                    "teamname": f"{side}-team",
                    "league": "LCK",
                    "playerid": f"oe:player:{side}-{index}",
                    "teamid": f"oe:team:{side}",
                    "golddiffat10": index,
                }
            )
    champions = {
        (row["side"], row["position"]): row["champion"] for row in observed
    }
    swapped = [
        {
            **row,
            "champion": champions[
                ("Red" if row["side"] == "Blue" else "Blue", row["position"])
            ],
        }
        for row in observed
    ]

    feature_rows, receipt = _controlled_feature_lineup(observed, swapped)

    assert receipt["slots"] == 10
    for original, feature in zip(observed, feature_rows):
        other_side = "Red" if original["side"] == "Blue" else "Blue"
        assert feature["champion"] == champions[
            (other_side, original["position"])
        ]
        assert feature["playerid"] == original["playerid"]
        assert feature["teamid"] == original["teamid"]
        assert feature["golddiffat10"] == original["golddiffat10"]


PRODUCER_GAME_TIME = "2026-08-01T12:00:00Z"
PRODUCER_RATING_TIME = "2026-08-01T11:59:59Z"
PRODUCER_SOURCE_IDENTITY = {
    "locator": "ratings/pre-game.parquet",
    "revision": "source-revision-1",
}
PRODUCER_SOURCE_SHA256 = (
    "12b29600078993239cf8fe7508994873d4eaca1ff7559c27f177e3c79313505e"
)
PRODUCER_ROSTER_SHA256 = (
    "d271e2aec25625822d850c96f453e4050c4542f4eac8784f65f8bf78e0f4a1b7"
)
PRODUCER_BATCH_SHA256 = (
    "338fb35ec92e4d7fdbc1bb1cf208b02a72101e0f36b909f07edb529e78d5d220"
)
PRODUCER_FULL_RECEIPT_SHA256 = (
    "a08ce2ae2211ca762e0703d7499d54313fa58fd1bd1bb509690c41b1689cd3ff"
)
PRODUCER_TEAM_ONLY_RECEIPT_SHA256 = (
    "9f3e7959e6dc19dc640d3eab9f96656605a5ceb53203008743fa48cae91cc9e1"
)


def _producer_roster() -> list[dict[str, str]]:
    return [
        {
            "game_uid": "map-1",
            "side": side,
            "position": role,
            "teamid": f"oe:team:{side}",
            "playerid": f"oe:player:{side}-{index}",
            "champion": f"{side.title()}Champion{index}",
        }
        for side in ("blue", "red")
        for index, role in enumerate(("top", "jungle", "mid", "bot", "support"))
    ]


def _categorical_lineup(side: str) -> list[dict[str, str]]:
    return [
        {
            "side": side,
            "position": role,
            "teamid": f"oe:team:{side.casefold()}",
            "playerid": f"oe:player:{side.casefold()}-{role}",
            "champion": f"{side}-{role}-champion",
        }
        for role in ("top", "jng", "mid", "bot", "sup")
    ]


def _producer_receipt(*, team_only: bool = False) -> dict[str, object]:
    values: dict[str, float | None] = {
        "base_team_logit": 0.2,
        "team_rating_diff_scaled": 0.1,
        "base_player_logit": None if team_only else 0.3,
        "player_rating_diff_scaled": None if team_only else 0.2,
        "player_lineup_complete": 0.0 if team_only else 1.0,
    }
    player_available = 0.0 if team_only else 1.0
    return {
        "schema_version": RATING_RECEIPT_SCHEMA,
        "rating_receipt_schema": RATING_RECEIPT_SCHEMA,
        "game_uid": "map-1",
        "rating_timestamp": PRODUCER_RATING_TIME,
        "rating_source_identity": deepcopy(PRODUCER_SOURCE_IDENTITY),
        "rating_source_available": 1.0,
        "rating_source_sha256": PRODUCER_SOURCE_SHA256,
        "rating_roster_sha256": PRODUCER_ROSTER_SHA256,
        "rating_receipt_sha256": (
            PRODUCER_TEAM_ONLY_RECEIPT_SHA256
            if team_only
            else PRODUCER_FULL_RECEIPT_SHA256
        ),
        "rating_values": values.copy(),
        "rating_batch_timestamp": PRODUCER_GAME_TIME,
        "rating_batching_policy": RATING_BATCH_POLICY,
        "rating_batch_receipt_sha256": PRODUCER_BATCH_SHA256,
        "rating_values_available": 1.0,
        "rating_values_missing": 0.0,
        "team_rating_available": 1.0,
        "team_rating_missing": 0.0,
        "player_rating_available": player_available,
        "player_rating_missing": 1.0 - player_available,
        **values,
    }


def _producer_like_receipt(
    *, game_time: str, rating_time: str
) -> dict[str, object]:
    receipt = _producer_receipt()
    receipt["rating_timestamp"] = rating_time
    receipt["rating_batch_timestamp"] = game_time
    roster_sha256 = _resolved_roster_sha256(
        _producer_roster(),
        game_id="map-1",
        timestamp=game_time,
        source_identity=receipt["rating_source_identity"],
    )
    batch_sha256 = _rating_batch_receipt_sha256(
        timestamp=game_time,
        game_ids=["map-1"],
        policy=RATING_BATCH_POLICY,
    )
    receipt["rating_roster_sha256"] = roster_sha256
    receipt["rating_batch_receipt_sha256"] = batch_sha256
    receipt["rating_receipt_sha256"] = _strict_canonical_sha256(
        {
            "schema_version": RATING_RECEIPT_SCHEMA,
            "source_available": receipt["rating_source_available"],
            "source_sha256": receipt["rating_source_sha256"],
            "roster_sha256": roster_sha256,
            "rating_timestamp": rating_time,
            "rating_values": receipt["rating_values"],
            "rating_values_available": receipt["rating_values_available"],
            "team_rating_available": receipt["team_rating_available"],
            "player_rating_available": receipt["player_rating_available"],
            "equal_timestamp_batching": {
                "policy": RATING_BATCH_POLICY,
                "receipt_sha256": batch_sha256,
            },
        }
    )
    return receipt


def _with_rating_context(receipt: dict[str, object]) -> dict[str, object]:
    context_values = {
        field: float(index + 1) / 100.0
        for index, field in enumerate(RATING_CONTEXT_FIELDS)
    }
    receipt.update(context_values)
    receipt["rating_context_schema"] = RATING_CONTEXT_SCHEMA
    receipt["rating_context_available"] = 1.0
    receipt["rating_context_missing"] = 0.0
    receipt["rating_context_sha256"] = _strict_canonical_sha256(
        {
            "schema_version": RATING_CONTEXT_SCHEMA,
            "rating_receipt_sha256": receipt["rating_receipt_sha256"],
            "values": context_values,
        }
    )
    return receipt


def _consume_producer_receipt(
    receipt: dict[str, object], *, game_time: str = PRODUCER_GAME_TIME
) -> dict[str, object]:
    roster_sha256 = _resolved_roster_sha256(
        _producer_roster(),
        game_id="map-1",
        timestamp=game_time,
        source_identity=receipt.get("rating_source_identity"),
    )
    batch_sha256 = _rating_batch_receipt_sha256(
        timestamp=game_time,
        game_ids=["map-1"],
        policy=RATING_BATCH_POLICY,
    )
    return _locked_rating_authority(
        receipt,
        resolved_roster_sha256=roster_sha256,
        map_timestamp=game_time,
        expected_batch_receipt_sha256=batch_sha256,
    )


def test_receipt_writer_serializes_numpy_scalars(tmp_path) -> None:
    path = tmp_path / "receipt.json"
    _write_json(path, {"rows": np.int64(1664)})
    assert '"rows": 1664' in path.read_text(encoding="utf-8")


def test_float_patch_token_uses_date_to_distinguish_16_1_and_16_10() -> None:
    assert normalize_source_patch("16.1", "2026-01-20T00:00:00Z") == "16.1"
    assert normalize_source_patch("16.1", "2026-05-20T00:00:00Z") == "16.10"
    assert normalize_source_patch("16.15", "2026-08-01T00:00:00Z") == "16.15"


def test_model_columns_exclude_current_state_and_targets() -> None:
    _validate_no_current_state_features(MODEL_COLUMNS)
    assert not any(column.startswith("target_") for column in MODEL_COLUMNS)


def test_categorical_context_preserves_exact_prematch_identities() -> None:
    map_row = {
        "league": "LEC",
        "tournament": "LEC 2026 Summer",
        "competition_scope": "regional",
        "event_kind": "playoffs",
        "blue_firstPick": 1,
        "red_firstPick": 0,
        **{
            f"{side}_ban{slot}": f"{side}-ban-{slot}"
            for side in ("blue", "red")
            for slot in range(1, 6)
        },
    }
    output = _categorical_context(
        map_row,
        _categorical_lineup("Blue"),
        _categorical_lineup("Red"),
        source_patch="16.16",
    )

    assert set(output) == set(CATEGORICAL_CONTEXT_COLUMNS)
    assert output["category_source_patch"] == "16.16"
    assert output["category_first_pick_side"] == "blue"
    assert output["category_blue_player_id_mid"] == "oe:player:blue-mid"
    assert output["category_red_champion_sup"] == "Red-sup-champion"
    assert output["category_red_ban_5"] == "red-ban-5"


def test_categorical_context_rejects_incomplete_roles() -> None:
    with pytest.raises(AtomizedResearchError, match="roles are incomplete"):
        _categorical_context(
            {"league": "LEC"},
            _categorical_lineup("Blue")[:-1],
            _categorical_lineup("Red"),
            source_patch="16.16",
        )


def test_categorical_context_uses_lineup_tournament_when_map_omits_it() -> None:
    blue = _categorical_lineup("Blue")
    for row in blue:
        row["tournament"] = "LEC 2026 Summer"

    output = _categorical_context(
        {"league": "LEC"},
        blue,
        _categorical_lineup("Red"),
        source_patch="16.16",
    )

    assert output["category_tournament"] == "LEC 2026 Summer"


def test_expanding_series_folds_are_forward_only() -> None:
    rows = []
    for series in range(40):
        for game in range(10):
            rows.append(
                {
                    "series_id": f"series-{series}",
                    "date": pd.Timestamp("2026-01-01", tz="UTC")
                    + pd.Timedelta(days=series, minutes=game),
                    "y": (series + game) % 2,
                }
            )
    frame = pd.DataFrame(rows)
    folds = _expanding_series_folds(frame)
    assert len(folds) >= 2
    for train_index, validation_index, audit in folds:
        assert frame.iloc[train_index]["date"].max() < frame.iloc[validation_index]["date"].min()
        assert set(frame.iloc[train_index]["series_id"]).isdisjoint(
            set(frame.iloc[validation_index]["series_id"])
        )
        assert audit["whole_series"] is True


def test_shrinkage_uses_only_supplied_prior_state() -> None:
    state = {("player", "champion", "gold_diff_10"): RunningStat(total=400.0, count=2)}
    global_state = {"gold_diff_10": RunningStat(total=100.0, count=2)}
    value, support, missing = _shrunk_metric_mean(
        state,
        global_state,
        [("player", "champion")],
        "gold_diff_10",
    )
    assert value == pytest.approx((400.0 + 5.0 * 50.0) / 7.0)
    assert support == 2
    assert missing == 0


def test_role_metric_families_keep_lineup_roles_separate() -> None:
    roles = ("top", "jng", "mid", "bot", "sup")
    blue = [{"position": role, "playerid": f"blue-{role}"} for role in roles]
    red = [{"position": role, "playerid": f"red-{role}"} for role in roles]
    state: dict[tuple[str, str], RunningStat] = {}
    for index, role in enumerate(roles, start=1):
        state[(f"blue-{role}", "gold_diff_10")] = RunningStat(
            total=float(index * 10), count=1
        )
        state[(f"red-{role}", "gold_diff_10")] = RunningStat(
            total=float(index), count=1
        )
    output: dict[str, float] = {}

    _emit_role_metric_families(
        output,
        prefix="history_player_overall",
        state=state,
        global_state={"gold_diff_10": RunningStat(total=0.0, count=1)},
        blue_rows=blue,
        red_rows=red,
        key_from_row=lambda row: (str(row["playerid"]),),
    )

    assert output["history_player_overall_top_gold_diff_10"] != output[
        "history_player_overall_sup_gold_diff_10"
    ]
    assert output["history_player_overall_top_gold_diff_10_support"] == 1
    assert output["history_player_overall_sup_gold_diff_10_support"] == 1


def test_exact_mechanic_keys_keep_raw_fields() -> None:
    keys = exact_mechanic_keys(
        [
            {
                "atom_id": "damage.packet",
                "behavior": "SpellQ",
                "trigger": "on_cast",
                "target_policy": "enemy",
                "parameters": {"cooldown_seconds": 8.0, "damage_type": "magic"},
                "relations": ["damage.resistance"],
                "states": [{"state": "active"}],
            }
        ]
    )
    assert any("parameter=cooldown_seconds" in key for key in keys)
    assert any("parameter=damage_type|value=magic" in key for key in keys)


def test_broad_mechanics_label_fails_closed() -> None:
    with pytest.raises(AtomizedResearchError, match="broad mechanic labels"):
        exact_mechanic_keys(
            [
                {
                    "atom_id": "teamfight",
                    "behavior": "summary",
                    "trigger": "on_cast",
                    "target_policy": "enemy",
                    "parameters": {},
                }
            ]
        )


def test_numpy_cluster_bootstrap_matches_dataframe_reference() -> None:
    frame = pd.DataFrame(
        {
            "series_id": ["a", "a", "b", "b", "c", "c", "d", "d"],
            "y": [0, 1, 0, 1, 1, 0, 1, 0],
        }
    )
    candidate = np.array([0.2, 0.7, 0.3, 0.8, 0.6, 0.4, 0.75, 0.25])
    baseline = np.array([0.3, 0.6, 0.4, 0.7, 0.55, 0.45, 0.65, 0.35])
    actual = _cluster_bootstrap_differences(
        frame, candidate, baseline, repetitions=25
    )

    work = frame.assign(_candidate=candidate, _baseline=baseline)
    clusters = [group for _, group in work.groupby("series_id", sort=True)]
    rng = np.random.default_rng(461)
    values = {"auc": [], "brier": [], "log_loss": []}
    for _ in range(25):
        sample = pd.concat(
            [clusters[index] for index in rng.integers(0, len(clusters), len(clusters))],
            ignore_index=True,
        )
        values["auc"].append(
            roc_auc_score(sample["y"], sample["_candidate"])
            - roc_auc_score(sample["y"], sample["_baseline"])
        )
        values["brier"].append(
            brier_score_loss(sample["y"], sample["_candidate"])
            - brier_score_loss(sample["y"], sample["_baseline"])
        )
        values["log_loss"].append(
            log_loss(sample["y"], sample["_candidate"], labels=[0, 1])
            - log_loss(sample["y"], sample["_baseline"], labels=[0, 1])
        )
    for metric, samples in values.items():
        assert actual[metric]["median"] == pytest.approx(np.median(samples))
        assert actual[metric]["lower_95"] == pytest.approx(np.quantile(samples, 0.025))
        assert actual[metric]["upper_95"] == pytest.approx(np.quantile(samples, 0.975))


def test_unique_player_map_support_deduplicates_metric_families() -> None:
    state = {
        ("player-1", "Galio"): {"map-1", "map-2"},
        ("player-2", "Ahri"): {"map-3"},
    }
    assert _unique_player_map_support(
        state,
        [("player-1", "Galio"), ("player-1", "Galio"), ("player-2", "Ahri")],
    ) == 3


def test_phase_forecast_weights_each_current_player_once() -> None:
    keys = [(f"player-{index}", "champion", 10, "gold") for index in range(5)]
    state = {
        keys[0]: RunningStat(total=10_000.0, count=100),
        keys[1]: RunningStat(total=20.0, count=1),
        keys[2]: RunningStat(total=60.0, count=2),
        keys[3]: RunningStat(total=120.0, count=3),
        keys[4]: RunningStat(total=200.0, count=4),
    }
    total, support, coverage, missing = _equal_weight_team_forecast(state, keys)
    assert total == pytest.approx(240.0)
    assert support == 1
    assert coverage == 1.0
    assert missing == 0


def test_phase_curve_is_side_swap_antisymmetric() -> None:
    blue = _phase_curve_features(
        [100.0, 200.0, -300.0, -900.0],
        [50.0, 100.0, -200.0, -500.0],
        available=True,
    )
    red = _phase_curve_features(
        [-100.0, -200.0, 300.0, 900.0],
        [-50.0, -100.0, 200.0, 500.0],
        available=True,
    )
    signed = [
        key
        for key in blue
        if key.startswith("forecast_") and key not in {
            "forecast_curve_available",
            "forecast_curve_missing",
        }
    ]
    for key in signed:
        assert red[key] == pytest.approx(-blue[key])


def test_pr281_producer_receipt_is_consumed_exactly() -> None:
    assert _resolved_roster_sha256(
        _producer_roster(),
        game_id="map-1",
        timestamp=PRODUCER_GAME_TIME,
        source_identity=PRODUCER_SOURCE_IDENTITY,
    ) == PRODUCER_ROSTER_SHA256
    assert _rating_batch_receipt_sha256(
        timestamp=PRODUCER_GAME_TIME,
        game_ids=["map-1"],
        policy=RATING_BATCH_POLICY,
    ) == PRODUCER_BATCH_SHA256

    bound = _consume_producer_receipt(_producer_receipt())
    assert bound["team_rating_available"] == 1.0
    assert bound["player_rating_available"] == 1.0
    assert bound["rating_source_receipt_hash_match"] == 1.0
    assert bound["rating_roster_receipt_match"] == 1.0
    assert bound["rating_batch_receipt_match"] == 1.0


def test_pr281_microsecond_timestamps_remain_canonical() -> None:
    game_time = "2026-08-01T12:00:00.123456Z"
    rating_time = "2026-08-01T11:59:59.123456Z"
    receipt = _producer_like_receipt(
        game_time=game_time,
        rating_time=rating_time,
    )
    bound = _consume_producer_receipt(receipt, game_time=game_time)
    assert bound["team_rating_available"] == 1.0
    assert bound["player_rating_available"] == 1.0
    assert bound["rating_source_receipt_hash_match"] == 1.0


@pytest.mark.parametrize("timestamp_field", ("rating", "game", "batch"))
def test_pr281_nanosecond_timestamps_fail_closed(timestamp_field: str) -> None:
    game_time = "2026-08-01T12:00:00.123456Z"
    receipt = _producer_like_receipt(
        game_time=game_time,
        rating_time="2026-08-01T11:59:59.123456Z",
    )
    presented_game_time = game_time
    if timestamp_field == "rating":
        receipt["rating_timestamp"] = "2026-08-01T11:59:59.123456789Z"
    elif timestamp_field == "batch":
        receipt["rating_batch_timestamp"] = "2026-08-01T12:00:00.123456789Z"
    else:
        presented_game_time = "2026-08-01T12:00:00.123456789Z"
        bound = _locked_rating_authority(
            receipt,
            resolved_roster_sha256=str(receipt["rating_roster_sha256"]),
            map_timestamp=presented_game_time,
            expected_batch_receipt_sha256=str(
                receipt["rating_batch_receipt_sha256"]
            ),
        )
        assert bound["rating_source_receipt_available"] == 0.0
        return

    bound = _consume_producer_receipt(receipt, game_time=presented_game_time)
    assert bound["team_rating_available"] == 0.0
    assert bound["player_rating_available"] == 0.0
    assert bound["rating_source_receipt_available"] == 0.0


def test_pr281_nullable_player_group_fails_closed_without_hiding_team_group() -> None:
    bound = _consume_producer_receipt(_producer_receipt(team_only=True))
    assert bound["team_rating_available"] == 1.0
    assert bound["base_team_logit"] == pytest.approx(0.2)
    assert bound["player_rating_available"] == 0.0
    assert bound["player_rating_missing"] == 1.0
    assert bound["base_player_logit"] == 0.0
    assert bound["rating_source_receipt_hash_match"] == 1.0


def test_rating_context_receipt_exposes_role_specific_features() -> None:
    receipt = _with_rating_context(_producer_receipt())
    bound = _consume_producer_receipt(receipt)

    assert bound["team_rating_available"] == 1.0
    assert bound["player_rating_available"] == 1.0
    assert bound["rating_context_available"] == 1.0
    assert bound["rating_context_missing"] == 0.0
    for field in RATING_CONTEXT_FIELDS:
        assert bound[field] == pytest.approx(receipt[field])


def test_rating_context_tampering_closes_only_context_group() -> None:
    receipt = _with_rating_context(_producer_receipt())
    receipt["player_role_rating_diff_scaled_mid"] = 9.0
    bound = _consume_producer_receipt(receipt)

    assert bound["team_rating_available"] == 1.0
    assert bound["player_rating_available"] == 1.0
    assert bound["rating_source_receipt_hash_match"] == 1.0
    assert bound["rating_context_available"] == 0.0
    assert bound["rating_context_missing"] == 1.0
    for field in RATING_CONTEXT_FIELDS:
        assert bound[field] == 0.0


@pytest.mark.parametrize(
    "field",
    (
        "nested_value",
        "top_level_value",
        "rating_timestamp",
        "source_sha256",
        "source_identity",
        "source_availability",
        "batch_receipt",
        "batch_policy",
        "batch_timestamp",
        "availability",
        "missing_flag",
        "receipt_sha256",
    ),
)
def test_pr281_receipt_tampering_fails_closed(field: str) -> None:
    receipt = _producer_receipt()
    if field == "nested_value":
        receipt["rating_values"]["base_team_logit"] = 0.21  # type: ignore[index]
    elif field == "top_level_value":
        receipt["base_team_logit"] = 0.21
    elif field == "rating_timestamp":
        receipt["rating_timestamp"] = "2026-08-01T11:59:58Z"
    elif field == "source_sha256":
        receipt["rating_source_sha256"] = "f" * 64
    elif field == "source_identity":
        receipt["rating_source_identity"] = {
            **PRODUCER_SOURCE_IDENTITY,
            "revision": "tampered-revision",
        }
    elif field == "source_availability":
        receipt["rating_source_available"] = 0.0
    elif field == "batch_receipt":
        receipt["rating_batch_receipt_sha256"] = "f" * 64
    elif field == "batch_policy":
        receipt["rating_batching_policy"] = "legacy-batch-policy"
    elif field == "batch_timestamp":
        receipt["rating_batch_timestamp"] = "2026-08-01T12:00:01Z"
    elif field == "availability":
        receipt["team_rating_available"] = 0.0
    elif field == "missing_flag":
        receipt["player_rating_missing"] = 1.0
    else:
        receipt["rating_receipt_sha256"] = "0" * 64

    bound = _consume_producer_receipt(receipt)
    assert bound["team_rating_available"] == 0.0
    assert bound["player_rating_available"] == 0.0
    assert bound["rating_source_receipt_available"] == 0.0


def test_legacy_four_field_rating_receipt_fails_closed() -> None:
    legacy = _producer_receipt()
    legacy.pop("schema_version")
    legacy.pop("rating_values")
    legacy.pop("rating_batch_receipt_sha256")
    legacy["rating_receipt_sha256"] = "7" * 64
    bound = _consume_producer_receipt(legacy)
    assert bound["team_rating_available"] == 0.0
    assert bound["player_rating_available"] == 0.0


def test_rating_and_momentum_missingness_is_explicit() -> None:
    missing_receipt = _producer_receipt()
    missing_receipt.pop("rating_receipt_schema")
    rating = _consume_producer_receipt(missing_receipt)
    assert rating["team_rating_available"] == 0.0
    assert rating["player_rating_available"] == 0.0
    assert rating["base_player_logit"] == 0.0

    momentum = _momentum_features(
        {}, {}, "blue-team", "red-team", [f"b-{i}" for i in range(5)], [f"r-{i}" for i in range(5)]
    )
    assert momentum["team_momentum_missing"] == 1.0
    assert momentum["player_momentum_missing"] == 1.0
    assert momentum["team_momentum_points_diff"] == 0.0


def test_feature_group_coverage_gates_each_split_and_league() -> None:
    rows = []
    for index in range(40):
        rows.append(
            {
                "date": pd.Timestamp("2026-04-01", tz="UTC")
                if index < 20
                else pd.Timestamp("2026-06-01", tz="UTC"),
                "league": "LEC" if index < 20 else "LCK",
                FEATURE_AVAILABILITY_COLUMNS["team_rating"]: 1.0
                if index < 30
                else 0.0,
            }
        )
    report = feature_group_coverage_report(
        pd.DataFrame(rows), thresholds={"team_rating": 0.8}
    )
    failures = {
        (row["dimension"], row["value"]) for row in report["failures"]
    }
    assert ("split", "validation") in failures
    assert ("league", "LCK") in failures
    assert ("split_league", "validation|LCK") in failures


def test_prospective_coverage_is_report_only() -> None:
    column = FEATURE_AVAILABILITY_COLUMNS["team_rating"]
    frame = pd.DataFrame(
        [
            {"date": pd.Timestamp("2026-04-01", tz="UTC"), "league": "LEC", column: 1.0}
            for _ in range(20)
        ]
        + [
            {"date": pd.Timestamp("2026-08-10", tz="UTC"), "league": "LEC", column: 0.0}
            for _ in range(20)
        ]
    )
    development_report = feature_group_coverage_report(
        frame.iloc[:20].copy(), thresholds={"team_rating": 0.8}
    )
    report = feature_group_coverage_report(
        frame,
        thresholds={"team_rating": 0.8},
        prospective_start="2026-08-09T00:00:00Z",
        prospective_end="2026-09-01T00:00:00Z",
    )
    assert report["passed"] is True
    prospective = [
        row for row in report["rows"] if row["dimension"] == "prospective"
    ]
    assert prospective[0]["coverage"] == 0.0
    assert prospective[0]["gate_role"] == "report_only"
    assert prospective[0]["passed"] is None
    assert (
        report["eligibility_receipt"]
        == development_report["eligibility_receipt"]
    )


def test_phase_coverage_reports_target_and_forecast_by_split_league() -> None:
    rows = []
    for index in range(20):
        row = {
            "date": pd.Timestamp("2026-06-01", tz="UTC"),
            "league": "LCK",
        }
        for checkpoint in (10, 15, 20, 25):
            for metric in ("gold", "xp"):
                row[f"target_{metric}_diff_{checkpoint}"] = (
                    100.0 if index < 15 else np.nan
                )
                row[f"forecast_{metric}_available_{checkpoint}"] = (
                    1.0 if index < 10 else 0.0
                )
        rows.append(row)
    report = phase_coverage_report(pd.DataFrame(rows))
    row = next(
        item
        for item in report["rows"]
        if item["dimension"] == "split_league"
        and item["value"] == "validation|LCK"
        and item["checkpoint"] == 10
        and item["metric"] == "gold"
    )
    assert row["eligible_rows"] == 20
    assert row["target_available"] == 15
    assert row["forecast_available"] == 10
    assert row["joint_available"] == 10


def test_repeated_metric_support_columns_are_not_model_inputs() -> None:
    repeated_prefixes = (
        "history_player_champion_",
        "history_ally_champion_pair_",
        "history_enemy_champion_pair_",
        "parity_player_champion_",
        "patch_player_champion_",
        "patch_champion_",
    )
    assert not any(
        column.endswith("_support") and column.startswith(repeated_prefixes)
        for column in MODEL_COLUMNS
    )
    assert "history_unique_player_maps_min" in MODEL_COLUMNS


def test_calibration_requires_brier_and_log_loss_improvement_in_every_fold() -> None:
    target = np.tile(np.array([0, 1]), 100)
    underconfident = np.where(target == 1, 0.60, 0.40)
    accepted = _calibration_outer_audit(
        [underconfident, underconfident, underconfident],
        [target, target, target],
    )
    assert accepted["accepted"] is True

    reversed_probability = 1.0 - underconfident
    rejected = _calibration_outer_audit(
        [underconfident, reversed_probability], [target, target]
    )
    assert rejected["accepted"] is False


def test_ablation_comparison_keeps_exact_frozen_learner() -> None:
    config = RFConfig(
        n_estimators=600,
        max_depth=None,
        min_samples_leaf=20,
        max_features=0.25,
        class_weight=None,
        bootstrap=True,
        max_samples=None,
    )
    comparison = _matched_comparison_config(config)
    assert comparison == config
    assert comparison.n_estimators == 600
