from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from benchmarks.build_future_value_final_fit import (
    VARIANTS,
    FinalFitError,
    _bind_scaling_features,
    _bind_current_rating_features,
    _canonical_sha,
    _design_digest,
    _evaluation_blockers,
    _target_digest,
    _variant_dependencies,
    _variant_feature_order,
)
from lol_kills.research.future_value_rating import (
    CURRENT_RATING_SIGNED_MAP_FEATURES,
    FUTURE_PLAYER_FORM_SIDE_FEATURES,
    SCALING_CURVE_SIGNED_MAP_FEATURES,
    RatingVariant,
    rating_feature_values_sha256,
    rating_variant_config,
)
from tests.test_future_value_snapshots import _source_receipt


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _current_frame() -> pd.DataFrame:
    rows = []
    for index, game_id in enumerate(("g1", "g2")):
        row: dict[str, object] = {
            "game_id": game_id,
            "date": pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=index),
            "series_id": f"series-{index}",
        }
        for feature_index, feature in enumerate(CURRENT_RATING_SIGNED_MAP_FEATURES):
            row[feature] = float(index + feature_index / 10.0)
        rows.append(row)
    return pd.DataFrame(rows)


def _write_current_receipt(
    path: Path,
    frame: pd.DataFrame,
    artifact_path: Path,
    source: dict[str, object],
    *,
    include_value_digest: bool = True,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "fixture:current-rating-receipt:v1",
        "source_receipt_sha256": source["receipt_sha256"],
        "source_identity_sha256": source["source_identity_sha256"],
        "artifact_sha256": _sha256_path(artifact_path),
        "rows": len(frame),
    }
    if include_value_digest:
        payload["feature_value_digest"] = rating_feature_values_sha256(
            frame, CURRENT_RATING_SIGNED_MAP_FEATURES
        )
    payload["receipt_sha256"] = _canonical_sha(payload)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return payload


def test_final_fit_imports_source_bound_evaluation_blockers(tmp_path) -> None:
    source = _source_receipt(["g1", "g2"])
    evaluation = {
        "source": {
            "source_as_of": source["source_as_of"],
            "source_game_count": source["source_game_count"],
            "source_identity_sha256": source["source_identity_sha256"],
            "source_receipt_sha256": source["receipt_sha256"],
        },
        "variants": {
            "future_player_form": {
                "blockers": [
                    "nested_inner_feature_ledger_missing_fixed_c_used",
                    "authoritative_series_id_missing_proxy_cluster_used",
                ],
                "evaluation": {
                    "strict_prior_calibration": {
                        "status": "available",
                        "blockers": [],
                    },
                    "support_uncertainty_calibration": {
                        "status": "research_only",
                        "blockers": [],
                        "coverage": {
                            "complete_enough": True,
                            "calibrated_row_fraction": 1.0,
                            "first_fold_without_history": False,
                        },
                    },
                },
            }
        },
    }
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps(evaluation), encoding="utf-8")
    assert _evaluation_blockers(path, source) == (
        "authoritative_series_id_missing_proxy_cluster_used",
        "nested_inner_feature_ledger_missing_fixed_c_used",
    )


def test_final_fit_adds_calibration_blockers_when_evidence_is_missing(tmp_path) -> None:
    source = _source_receipt(["g1", "g2"])
    evaluation = {
        "source": {
            "source_as_of": source["source_as_of"],
            "source_game_count": source["source_game_count"],
            "source_identity_sha256": source["source_identity_sha256"],
            "source_receipt_sha256": source["receipt_sha256"],
        },
        "variants": {"future_player_form": {"blockers": [], "evaluation": {}}},
    }
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps(evaluation), encoding="utf-8")
    assert _evaluation_blockers(path, source) == (
        "final_calibration_receipt_missing",
        "support_uncertainty_proxy_not_calibrated",
    )


def test_final_fit_rejects_co_mutated_current_rows_and_receipt(tmp_path) -> None:
    source = _source_receipt(["g1", "g2"])
    source_path = tmp_path / "source-receipt.json"
    source_path.write_text(json.dumps(source, sort_keys=True), encoding="utf-8")
    artifact_path = tmp_path / "current-rating-ledger.parquet"
    receipt_path = tmp_path / "current-rating-ledger-receipt.json"
    frame = _current_frame()
    frame.to_parquet(artifact_path, index=False)
    receipt = _write_current_receipt(
        receipt_path, frame, artifact_path, source
    )
    expected_receipt_hash = _sha256_path(receipt_path)
    expected_artifact_hash = _sha256_path(artifact_path)

    mutated = frame.copy()
    mutated.loc[0, CURRENT_RATING_SIGNED_MAP_FEATURES[0]] += 9.0
    mutated.to_parquet(artifact_path, index=False)
    mutated_receipt = _write_current_receipt(
        receipt_path, mutated, artifact_path, source
    )

    with pytest.raises(FinalFitError, match="receipt file changed|artifact file changed"):
        _bind_current_rating_features(
            mutated,
            mutated_receipt,
            source_receipt=source,
            source_receipt_path=source_path,
            current_artifact_path=artifact_path,
            current_receipt_path=receipt_path,
            implementation_path=Path(
                "lol_kills/research/future_value_rating_ledger.py"
            ).resolve(),
            fit_game_ids=("g1", "g2"),
            fit_window_end=str(source["source_as_of"]),
            expected_current_receipt_sha256=expected_receipt_hash,
            expected_current_artifact_sha256=expected_artifact_hash,
        )


def test_final_fit_requires_current_feature_value_digest(tmp_path) -> None:
    source = _source_receipt(["g1", "g2"])
    source_path = tmp_path / "source-receipt.json"
    source_path.write_text(json.dumps(source, sort_keys=True), encoding="utf-8")
    artifact_path = tmp_path / "current-rating-ledger.parquet"
    receipt_path = tmp_path / "current-rating-ledger-receipt.json"
    frame = _current_frame()
    frame.to_parquet(artifact_path, index=False)
    receipt = _write_current_receipt(
        receipt_path,
        frame,
        artifact_path,
        source,
        include_value_digest=False,
    )

    with pytest.raises(FinalFitError, match="feature-value digest is required"):
        _bind_current_rating_features(
            frame,
            receipt,
            source_receipt=source,
            source_receipt_path=source_path,
            current_artifact_path=artifact_path,
            current_receipt_path=receipt_path,
            implementation_path=Path(
                "lol_kills/research/future_value_rating_ledger.py"
            ).resolve(),
            fit_game_ids=("g1", "g2"),
            fit_window_end=str(source["source_as_of"]),
            expected_current_receipt_sha256=_sha256_path(receipt_path),
            expected_current_artifact_sha256=_sha256_path(artifact_path),
        )


@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        ("current_only", tuple(CURRENT_RATING_SIGNED_MAP_FEATURES)),
        (
            "future_player_form",
            tuple((*CURRENT_RATING_SIGNED_MAP_FEATURES, *FUTURE_PLAYER_FORM_SIDE_FEATURES)),
        ),
        (
            "scaling_curve",
            tuple((*CURRENT_RATING_SIGNED_MAP_FEATURES, *SCALING_CURVE_SIGNED_MAP_FEATURES)),
        ),
        (
            "both",
            tuple(
                (
                    *CURRENT_RATING_SIGNED_MAP_FEATURES,
                    *FUTURE_PLAYER_FORM_SIDE_FEATURES,
                    *SCALING_CURVE_SIGNED_MAP_FEATURES,
                )
            ),
        ),
    ],
)
def test_final_fit_exposes_each_registered_feature_order(variant, expected) -> None:
    assert VARIANTS == tuple(item.value for item in RatingVariant)
    assert _variant_feature_order(variant) == expected
    assert _variant_feature_order(variant) == rating_variant_config(variant).feature_names


@pytest.mark.parametrize(
    ("variant", "needs_form", "needs_scaling"),
    [
        ("current_only", False, False),
        ("future_player_form", True, False),
        ("scaling_curve", False, True),
        ("both", True, True),
    ],
)
def test_final_fit_dependency_receipt_families_are_explicit(
    variant, needs_form, needs_scaling
) -> None:
    dependencies = _variant_dependencies(variant)
    assert dependencies["current_rating"] is True
    assert dependencies["future_player_form"] is needs_form
    assert dependencies["scaling_curve"] is needs_scaling


def test_final_fit_rejects_cross_variant_evaluation_receipt(tmp_path) -> None:
    source = _source_receipt(["g1", "g2"])
    evaluation = {
        "source": {
            "source_as_of": source["source_as_of"],
            "source_game_count": source["source_game_count"],
            "source_identity_sha256": source["source_identity_sha256"],
            "source_receipt_sha256": source["receipt_sha256"],
        },
        "variants": {
            "future_player_form": {
                "variant": "future_player_form",
                "blockers": [],
                "evaluation": {},
            }
        },
    }
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps(evaluation), encoding="utf-8")
    with pytest.raises(FinalFitError, match="variant evidence is missing"):
        _evaluation_blockers(path, source, RatingVariant.SCALING_CURVE)


def test_final_fit_digests_are_invariant_to_row_order() -> None:
    frame = pd.DataFrame(
        {
            "game_id": ["g2", "g1"],
            "target": [0, 1],
            "base_team_logit": [0.2, -0.4],
            "base_player_logit": [0.1, -0.3],
            "team_rating_diff_scaled": [0.4, -0.5],
            "player_rating_diff_scaled": [0.6, -0.7],
        }
    )
    ordered = frame.sort_values("game_id", kind="stable").reset_index(drop=True)
    feature_names = tuple(CURRENT_RATING_SIGNED_MAP_FEATURES)
    assert _target_digest(frame) == _target_digest(ordered)
    assert _design_digest(frame, feature_names) == _design_digest(ordered, feature_names)


def test_final_fit_family_digest_mutations_are_isolated() -> None:
    feature_names = tuple(
        (
            *CURRENT_RATING_SIGNED_MAP_FEATURES,
            *FUTURE_PLAYER_FORM_SIDE_FEATURES,
            *SCALING_CURVE_SIGNED_MAP_FEATURES,
        )
    )
    frame = pd.DataFrame({"game_id": ["g1", "g2"]})
    for index, name in enumerate(feature_names):
        frame[name] = [float(index + 1), float(index + 2)]
    base = {
        variant: _design_digest(frame, _variant_feature_order(variant))
        for variant in VARIANTS
    }
    form_mutated = frame.copy()
    form_mutated[FUTURE_PLAYER_FORM_SIDE_FEATURES[0]] += 100.0
    assert _design_digest(form_mutated, _variant_feature_order("current_only")) == base[
        "current_only"
    ]
    assert _design_digest(form_mutated, _variant_feature_order("scaling_curve")) == base[
        "scaling_curve"
    ]
    assert _design_digest(form_mutated, _variant_feature_order("future_player_form")) != base[
        "future_player_form"
    ]
    scaling_mutated = frame.copy()
    scaling_mutated[SCALING_CURVE_SIGNED_MAP_FEATURES[0]] += 100.0
    assert _design_digest(scaling_mutated, _variant_feature_order("current_only")) == base[
        "current_only"
    ]
    assert _design_digest(scaling_mutated, _variant_feature_order("future_player_form")) == base[
        "future_player_form"
    ]
    assert _design_digest(scaling_mutated, _variant_feature_order("scaling_curve")) != base[
        "scaling_curve"
    ]


def test_final_fit_requires_independent_scaling_receipt_hash(tmp_path) -> None:
    source = _source_receipt(["g1", "g2"])
    frame = pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            **{feature: [1.0, 2.0] for feature in SCALING_CURVE_SIGNED_MAP_FEATURES},
        }
    )
    artifact_path = tmp_path / "scaling.parquet"
    receipt_path = tmp_path / "scaling-receipt.json"
    frame.to_parquet(artifact_path, index=False)
    receipt_path.write_text("{}", encoding="utf-8")
    with pytest.raises(FinalFitError, match="independent scaling feature receipt hash"):
        _bind_scaling_features(
            frame,
            {},
            source_receipt=source,
            source_receipt_path=tmp_path / "source-receipt.json",
            scaling_artifact_path=artifact_path,
            scaling_receipt_path=receipt_path,
            fit_game_ids=("g1", "g2"),
            fit_window_end=str(source["source_as_of"]),
            expected_scaling_receipt_sha256=None,
            expected_scaling_artifact_sha256=_sha256_path(artifact_path),
        )
