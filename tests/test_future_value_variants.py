from __future__ import annotations

import hashlib
import json

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
    classify_rating_feature,
    get_rating_variant_config,
    rating_variant_config_receipt,
    rating_variant_config_sha256,
    rating_variant_registry_receipt,
)


def _canonical(payload: object) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def test_registry_has_exactly_the_four_frozen_variants() -> None:
    assert tuple(RATING_VARIANT_CONFIGS) == tuple(RatingVariant)
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
        config_payload = {
            key: receipt[key]
            for key in (
                "schema_version",
                "variant",
                "feature_names",
                "signed_map_features",
                "side_level_features",
                "excluded_features",
            )
        }
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
