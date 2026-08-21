from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from lol_kills.research.future_value_rating import (
    CURRENT_RATING_SIGNED_MAP_FEATURES,
    FutureValueSourceError,
    FUTURE_PLAYER_FORM_SIDE_FEATURES,
    RATING_VARIANT_CONFIGS,
    RATING_VARIANT_SCHEMA_VERSION,
    RatingVariant,
    SCALING_CURVE_DERIVED_FEATURES,
    SCALING_CURVE_SIGNED_MAP_FEATURES,
    assert_rating_feature_names,
    assert_rating_variant_features,
    bind_rating_feature_ledger,
    build_rating_variant_matrix,
    classify_rating_feature,
    get_rating_variant_config,
    rating_variant_config_receipt,
    rating_variant_config_sha256,
    rating_variant_registry_receipt,
    validate_rating_feature_ledger,
)
from lol_kills.v2.tierlists.accepted_census import identity_sha256


def _canonical(payload: object) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def test_registry_has_exactly_the_four_frozen_variants() -> None:
    assert tuple(RATING_VARIANT_CONFIGS) == (
        RatingVariant.CURRENT_ONLY,
        RatingVariant.FUTURE_PLAYER_FORM,
        RatingVariant.SCALING_CURVE,
        RatingVariant.BOTH,
    )
    assert {variant.value for variant in RATING_VARIANT_CONFIGS} == {
        "current_only",
        "future_player_form",
        "scaling_curve",
        "both",
    }
    assert get_rating_variant_config("current_only").feature_names == (
        *CURRENT_RATING_SIGNED_MAP_FEATURES,
    )
    assert get_rating_variant_config("future_player_form").side_level_features == (
        *FUTURE_PLAYER_FORM_SIDE_FEATURES,
    )
    assert get_rating_variant_config("scaling_curve").feature_names == (
        *CURRENT_RATING_SIGNED_MAP_FEATURES,
        *SCALING_CURVE_SIGNED_MAP_FEATURES,
    )
    assert get_rating_variant_config("both").feature_names == (
        *CURRENT_RATING_SIGNED_MAP_FEATURES,
        *FUTURE_PLAYER_FORM_SIDE_FEATURES,
        *SCALING_CURVE_SIGNED_MAP_FEATURES,
    )
    assert [get_rating_variant_config(variant).ordinal for variant in RATING_VARIANT_CONFIGS] == [1, 2, 3, 4]
    assert [get_rating_variant_config(variant).label for variant in RATING_VARIANT_CONFIGS] == ["V1", "V2", "V3", "V4"]


def test_registry_and_configs_are_immutable() -> None:
    with pytest.raises(TypeError):
        RATING_VARIANT_CONFIGS["extra"] = RATING_VARIANT_CONFIGS[RatingVariant.CURRENT_ONLY]
    config = get_rating_variant_config(RatingVariant.CURRENT_ONLY)
    with pytest.raises((AttributeError, TypeError)):
        config.feature_names += ("arbitrary",)
    with pytest.raises((AttributeError, TypeError)):
        config.feature_names[0] = "arbitrary"
    with pytest.raises(TypeError):
        config.feature_families["new_family"] = ()


def test_families_are_isolated_and_team_context_is_not_player_form() -> None:
    assert set(CURRENT_RATING_SIGNED_MAP_FEATURES).isdisjoint(
        FUTURE_PLAYER_FORM_SIDE_FEATURES
    )
    assert set(CURRENT_RATING_SIGNED_MAP_FEATURES).isdisjoint(
        SCALING_CURVE_SIGNED_MAP_FEATURES
    )
    assert set(FUTURE_PLAYER_FORM_SIDE_FEATURES).isdisjoint(
        SCALING_CURVE_SIGNED_MAP_FEATURES
    )
    assert "team_prior_win" not in FUTURE_PLAYER_FORM_SIDE_FEATURES
    assert "roster_continuity" not in FUTURE_PLAYER_FORM_SIDE_FEATURES
    assert "team_prior_win_diff" not in get_rating_variant_config(
        "future_player_form"
    ).feature_names
    assert "roster_continuity_diff" not in get_rating_variant_config(
        "future_player_form"
    ).feature_names
    assert all(
        feature not in get_rating_variant_config("both").feature_names
        for feature in SCALING_CURVE_DERIVED_FEATURES
    )
    assert all(
        classify_rating_feature(feature) == "signed_map"
        for feature in SCALING_CURVE_SIGNED_MAP_FEATURES
    )


def test_receipts_and_hashes_are_canonical_and_distinct() -> None:
    receipts = [rating_variant_config_receipt(variant) for variant in RatingVariant]
    hashes = [rating_variant_config_sha256(variant) for variant in RatingVariant]
    assert len(set(hashes)) == len(RatingVariant)
    assert all(len(value) == 64 for value in hashes)
    for receipt in receipts:
        payload = dict(receipt)
        claimed = payload.pop("config_sha256")
        receipt_hash = payload.pop("receipt_sha256")
        config_payload = dict(receipt)
        config_payload.pop("config_sha256")
        config_payload.pop("receipt_sha256")
        assert receipt["schema_version"] == RATING_VARIANT_SCHEMA_VERSION
        assert claimed == hashlib.sha256(_canonical(config_payload)).hexdigest()
        assert receipt_hash == hashlib.sha256(
            _canonical({**config_payload, "config_sha256": claimed})
        ).hexdigest()
    registry = rating_variant_registry_receipt()
    registry_payload = dict(registry)
    claimed_registry = registry_payload.pop("registry_sha256")
    registry_payload.pop("receipt_sha256")
    assert claimed_registry == hashlib.sha256(_canonical(registry_payload)).hexdigest()


def test_feature_classification_is_closed() -> None:
    assert all(
        classify_rating_feature(name) == "signed_map"
        for name in CURRENT_RATING_SIGNED_MAP_FEATURES
    )
    assert all(
        classify_rating_feature(name) == "side_level"
        for name in FUTURE_PLAYER_FORM_SIDE_FEATURES
    )
    assert all(
        classify_rating_feature(name) == "derived"
        for name in SCALING_CURVE_DERIVED_FEATURES
    )
    assert classify_rating_feature("unknown_feature") == "unknown"


@pytest.mark.parametrize(
    "feature_name",
    (
        "target",
        "y_blue_win",
        "observed_result",
        "final_gold",
        "goldat10",
        "gold_diff_15",
        "golddiffat20",
        "forecast_gold_slope_10_15",
        "scaling_index",
    ),
)
def test_forbidden_feature_names_fail_closed(feature_name: str) -> None:
    with pytest.raises(FutureValueSourceError):
        assert_rating_feature_names([feature_name])


def test_unknown_variant_and_arbitrary_feature_list_fail_closed() -> None:
    with pytest.raises(FutureValueSourceError, match="unknown rating variant"):
        get_rating_variant_config("future_player_form_v2")
    with pytest.raises(FutureValueSourceError, match="arbitrary feature list"):
        get_rating_variant_config("current_only", ["team_rating_diff_scaled"])
    with pytest.raises(FutureValueSourceError, match="not registered"):
        assert_rating_variant_features(
            "scaling_curve",
            [*CURRENT_RATING_SIGNED_MAP_FEATURES, "unknown_feature"],
        )


def _source_receipt(game_ids: list[str]) -> dict[str, object]:
    payload: dict[str, object] = {
        "source_as_of": "2026-01-05T00:00:00Z",
        "source_game_count": len(game_ids),
        "source_identity_sha256": identity_sha256(game_ids),
        "accepted_game_ids": sorted(game_ids),
        "model_eligible_game_count": len(game_ids),
        "model_eligible_identity_sha256": identity_sha256(game_ids),
        "model_eligible_game_ids": sorted(game_ids),
        "source_files": {"fixture": {"bytes": 1, "sha256": "0" * 64}},
    }
    payload["receipt_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    return payload


def _variant_design() -> pd.DataFrame:
    frame = pd.DataFrame({"game_id": ["g1", "g2", "g3", "g4"]})
    for position, feature in enumerate(FUTURE_PLAYER_FORM_SIDE_FEATURES, 1):
        frame[f"__blue_{feature}"] = np.arange(4, dtype=float) + position
        frame[f"__red_{feature}"] = np.arange(4, dtype=float) + position / 2
    for position, feature in enumerate((*CURRENT_RATING_SIGNED_MAP_FEATURES, *SCALING_CURVE_SIGNED_MAP_FEATURES), 1):
        frame[feature] = np.arange(4, dtype=float) / 10.0 + position
    return frame


def test_four_variant_matrices_select_different_registered_families() -> None:
    design = _variant_design()
    matrices = {
        variant: build_rating_variant_matrix(design, variant)
        for variant in RatingVariant
    }
    assert matrices[RatingVariant.CURRENT_ONLY].shape[1] == len(CURRENT_RATING_SIGNED_MAP_FEATURES)
    assert matrices[RatingVariant.FUTURE_PLAYER_FORM].shape[1] == len(
        get_rating_variant_config(RatingVariant.FUTURE_PLAYER_FORM).feature_names
    )
    assert matrices[RatingVariant.SCALING_CURVE].shape[1] == len(
        get_rating_variant_config(RatingVariant.SCALING_CURVE).feature_names
    )
    assert matrices[RatingVariant.BOTH].shape[1] == len(
        get_rating_variant_config(RatingVariant.BOTH).feature_names
    )
    assert not np.array_equal(matrices[RatingVariant.CURRENT_ONLY], matrices[RatingVariant.FUTURE_PLAYER_FORM])
    assert not np.array_equal(matrices[RatingVariant.CURRENT_ONLY], matrices[RatingVariant.SCALING_CURVE])
    assert not np.array_equal(matrices[RatingVariant.FUTURE_PLAYER_FORM], matrices[RatingVariant.BOTH])


def test_fold_bound_ledger_receipt_binds_source_cutoff_and_features() -> None:
    game_ids = ["g1", "g2", "g3", "g4"]
    source = _source_receipt(game_ids)
    features = tuple(CURRENT_RATING_SIGNED_MAP_FEATURES)
    raw = pd.DataFrame({"game_id": game_ids, **{name: [1.0, 2.0, 3.0, 4.0] for name in features}})
    ledger = bind_rating_feature_ledger(
        raw,
        source_receipt=source,
        train_game_ids=["g1", "g2"],
        fit_window_end="2026-01-03T00:00:00Z",
        feature_names=features,
        producer={"name": "fixture"},
    )
    bound = validate_rating_feature_ledger(
        ledger,
        feature_names=features,
        model_game_ids=game_ids,
        train_game_ids=["g1", "g2"],
        fit_window_end="2026-01-03T00:00:00Z",
        source_receipt=source,
    )
    assert tuple(bound["game_id"]) == tuple(game_ids)
    assert ledger.attrs["producer_receipt_sha256"] == ledger.attrs["producer_receipt"]["receipt_sha256"]
    with pytest.raises(FutureValueSourceError, match="cutoff"):
        validate_rating_feature_ledger(
            ledger,
            feature_names=features,
            model_game_ids=game_ids,
            train_game_ids=["g1", "g2"],
            fit_window_end="2026-01-04T00:00:00Z",
            source_receipt=source,
        )


def test_signed_variants_reject_missing_or_arbitrary_external_ledger() -> None:
    source = _source_receipt(["g1", "g2"])
    with pytest.raises(FutureValueSourceError, match="required"):
        validate_rating_feature_ledger(
            None,
            feature_names=CURRENT_RATING_SIGNED_MAP_FEATURES,
            model_game_ids=["g1", "g2"],
            train_game_ids=["g1"],
            fit_window_end="2026-01-02T00:00:00Z",
            source_receipt=source,
        )
    with pytest.raises(FutureValueSourceError, match="not registered"):
        bind_rating_feature_ledger(
            pd.DataFrame({"game_id": ["g1"], "unknown_feature": [1.0]}),
            source_receipt=source,
            train_game_ids=["g1"],
            fit_window_end="2026-01-02T00:00:00Z",
            feature_names=("unknown_feature",),
        )
