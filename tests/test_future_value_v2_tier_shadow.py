from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from lol_kills.research.future_value_rating import (
    CURRENT_RATING_SIGNED_MAP_FEATURES,
    RatingVariant,
    rating_variant_config,
)
from lol_kills.research.future_value_tierlist import offset_values_sha256
from lol_kills.research.future_value_v2_tier_shadow import (
    AUTHORITY,
    V2TierShadowError,
    canonical_json_bytes,
    load_v2_tier_offset_ledger,
    score_v2_design,
    write_v2_tier_offset_ledger,
)
from lol_kills.research.future_value_tier_shadow import (
    TierShadowError,
    score_variant_design,
)
from lol_kills.v2.tierlists.accepted_census import identity_sha256


def _source_receipt(ids: list[str]) -> dict[str, object]:
    return {
        "source_as_of": "2026-02-03T00:00:00Z",
        "source_game_count": len(ids) + 1,
        "source_identity_sha256": "a" * 64,
        "receipt_sha256": "b" * 64,
        "model_eligible_game_ids": ids,
        "model_eligible_identity_sha256": identity_sha256(ids),
    }


def _parameters() -> dict[str, object]:
    names = rating_variant_config(RatingVariant.FUTURE_PLAYER_FORM).feature_names
    coefficients = {name: 0.0 for name in names}
    coefficients["base_team_logit"] = 1.0
    return {
        "feature_names": list(names),
        "fold_local_side_imputation": {name: 0.0 for name in names},
        "feature_scales": {name: 1.0 for name in names},
        "coefficients": coefficients,
        "intercept": 0.0,
    }


def _design() -> pd.DataFrame:
    names = rating_variant_config(RatingVariant.FUTURE_PLAYER_FORM).feature_names
    rows = []
    for game_id, day, target, base in (
        ("g1", "2026-02-01T00:00:00Z", 1, 0.2),
        ("g2", "2026-02-02T00:00:00Z", 0, -0.4),
    ):
        row: dict[str, object] = {
            "game_id": game_id,
            "date": day,
            "target": target,
        }
        for name in CURRENT_RATING_SIGNED_MAP_FEATURES:
            row[name] = base if name == "base_team_logit" else 0.0
        for name in names:
            if name not in CURRENT_RATING_SIGNED_MAP_FEATURES:
                row[f"__blue_{name}"] = 0.0
                row[f"__red_{name}"] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _bound_file(root: Path, name: str) -> Path:
    path = root / name
    path.write_bytes((name + "\n").encode())
    return path


def test_score_v2_design_reconstructs_exact_model_logit() -> None:
    rows, offsets, matrix_hash = score_v2_design(
        _design(),
        _parameters(),
        expected_game_ids=["g1", "g2"],
    )
    assert offsets == pytest.approx({"g1": 0.2, "g2": -0.4})
    assert [row["target"] for row in rows] == [1, 0]
    assert len(matrix_hash) == 64
    assert offset_values_sha256(offsets) == offset_values_sha256(
        {"g2": -0.4, "g1": 0.2}
    )


def test_score_v2_design_rejects_partial_census_and_bad_parameters() -> None:
    with pytest.raises(V2TierShadowError, match="census"):
        score_v2_design(
            _design().iloc[:1],
            _parameters(),
            expected_game_ids=["g1", "g2"],
        )
    parameters = _parameters()
    parameters["intercept"] = 0.1
    with pytest.raises(V2TierShadowError, match="parameters"):
        score_v2_design(
            _design(),
            parameters,
            expected_game_ids=["g1", "g2"],
        )


def test_source_bound_ledger_round_trip_and_pooled_input(tmp_path: Path) -> None:
    rows, offsets, matrix_hash = score_v2_design(
        _design(), _parameters(), expected_game_ids=["g1", "g2"]
    )
    source = _source_receipt(["g1", "g2"])
    source_receipt_file = _bound_file(tmp_path, "source-receipt.json")
    files = {
        "maps": _bound_file(tmp_path, "maps.parquet"),
        "players": _bound_file(tmp_path, "players.parquet"),
        "teams": _bound_file(tmp_path, "teams.parquet"),
    }
    model = _bound_file(tmp_path, "model.json")
    model_receipt = _bound_file(tmp_path, "model-receipt.json")
    run_receipt = _bound_file(tmp_path, "run-receipt.json")
    current = _bound_file(tmp_path, "current.parquet")
    current_receipt = _bound_file(tmp_path, "current-receipt.json")
    target_hash = hashlib.sha256(
        canonical_json_bytes(
            [
                {"game_id": row["game_id"], "target": row["target"]}
                for row in sorted(rows, key=lambda item: item["game_id"])
            ]
        )
    ).hexdigest()
    result = write_v2_tier_offset_ledger(
        tmp_path / "ledger.json",
        rows=rows,
        offsets=offsets,
        design_matrix_sha256=matrix_hash,
        source_receipt=source,
        source_receipt_file=source_receipt_file,
        source_files=files,
        model_file=model,
        model_receipt_file=model_receipt,
        run_receipt_file=run_receipt,
        current_ledger_file=current,
        current_receipt_file=current_receipt,
        target_rows_sha256=target_hash,
    )
    loaded, provenance = load_v2_tier_offset_ledger(
        result.ledger_path,
        result.receipt_path,
        source_receipt=source,
    )
    assert loaded == pytest.approx(offsets)
    assert provenance["source_game_count"] == 2
    assert provenance["source_identity_sha256"] == identity_sha256(["g1", "g2"])
    assert result.receipt["authority"] == AUTHORITY
    payload = json.loads(result.ledger_path.read_text())
    assert payload["timing"] == {
        "feature_state": "strict_prior_before_each_map",
        "same_timestamp_policy": "batch_exclude_same_timestamp",
        "model_fit_scope": "retrospective_full_model_eligible_census",
        "chronological_evaluation_suitable": False,
    }
    assert payload["source"]["excluded_game_count"] == 1


def test_ledger_rejects_row_mutation_and_wrong_source(tmp_path: Path) -> None:
    rows, offsets, matrix_hash = score_v2_design(
        _design(), _parameters(), expected_game_ids=["g1", "g2"]
    )
    source = _source_receipt(["g1", "g2"])
    inputs = [_bound_file(tmp_path, f"input-{index}") for index in range(8)]
    target_hash = hashlib.sha256(
        canonical_json_bytes(
            [
                {"game_id": row["game_id"], "target": row["target"]}
                for row in sorted(rows, key=lambda item: item["game_id"])
            ]
        )
    ).hexdigest()
    result = write_v2_tier_offset_ledger(
        tmp_path / "ledger.json",
        rows=rows,
        offsets=offsets,
        design_matrix_sha256=matrix_hash,
        source_receipt=source,
        source_receipt_file=inputs[0],
        source_files={"maps": inputs[1], "players": inputs[2], "teams": inputs[3]},
        model_file=inputs[4],
        model_receipt_file=inputs[5],
        run_receipt_file=inputs[6],
        current_ledger_file=inputs[7],
        current_receipt_file=inputs[0],
        target_rows_sha256=target_hash,
    )
    changed = json.loads(result.ledger_path.read_text())
    changed["rows"][0]["v2_offset_logit"] += 1.0
    result.ledger_path.write_bytes(canonical_json_bytes(changed) + b"\n")
    with pytest.raises(V2TierShadowError, match="bytes changed"):
        load_v2_tier_offset_ledger(
            result.ledger_path,
            result.receipt_path,
            source_receipt=source,
        )
    result.ledger_path.write_bytes(
        canonical_json_bytes(
            {
                **changed,
                "rows": rows,
            }
        )
        + b"\n"
    )
    wrong_source = dict(source)
    wrong_source["receipt_sha256"] = "c" * 64
    with pytest.raises(V2TierShadowError, match="source receipt"):
        load_v2_tier_offset_ledger(
            result.ledger_path,
            result.receipt_path,
            source_receipt=wrong_source,
        )


def _variant_parameters(variant: RatingVariant) -> dict[str, object]:
    names = rating_variant_config(variant).feature_names
    return {
        "variant": variant.value,
        "feature_names": list(names),
        "fold_local_side_imputation": {name: 0.0 for name in names},
        "feature_scales": {name: 1.0 for name in names},
        "coefficients": {name: 1.0 for name in names},
        "intercept": 0.0,
    }


def _variant_design(
    variant: RatingVariant,
    *,
    form_shift: float = 0.0,
    scaling_shift: float = 0.0,
) -> pd.DataFrame:
    config = rating_variant_config(variant)
    rows: list[dict[str, object]] = []
    for game_id, day, target, current in (
        ("g1", "2026-02-01T00:00:00Z", 1, 0.2),
        ("g2", "2026-02-02T00:00:00Z", 0, -0.4),
    ):
        row: dict[str, object] = {
            "game_id": game_id,
            "date": day,
            "target": target,
        }
        for name in config.feature_names:
            if name in config.signed_map_features:
                if name in CURRENT_RATING_SIGNED_MAP_FEATURES:
                    row[name] = current
                else:
                    row[name] = scaling_shift
            else:
                row[f"__blue_{name}"] = form_shift
                row[f"__red_{name}"] = 0.0
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.attrs["variant"] = variant.value
    return frame


def test_four_variant_scores_use_one_universe_and_neutral_offsets() -> None:
    scored = {}
    for variant in RatingVariant:
        rows, offsets, _ = score_variant_design(
            _variant_design(variant),
            _variant_parameters(variant),
            variant=variant,
            expected_game_ids=["g1", "g2"],
        )
        scored[variant] = (rows, offsets)
        assert {row["game_id"] for row in rows} == {"g1", "g2"}
        assert {row["target"] for row in rows} == {0, 1}
        assert all("offset_logit" in row for row in rows)
        assert all("v2_offset_logit" not in row for row in rows)
    assert set(scored[RatingVariant.CURRENT_ONLY][1]) == set(
        scored[RatingVariant.BOTH][1]
    )


def test_form_mutation_changes_only_v2_and_v4() -> None:
    baseline = {}
    mutated = {}
    for variant in RatingVariant:
        parameters = _variant_parameters(variant)
        baseline[variant] = score_variant_design(
            _variant_design(variant),
            parameters,
            variant=variant,
            expected_game_ids=["g1", "g2"],
        )[1]
        mutated[variant] = score_variant_design(
            _variant_design(variant, form_shift=7.0),
            parameters,
            variant=variant,
            expected_game_ids=["g1", "g2"],
        )[1]
    assert mutated[RatingVariant.CURRENT_ONLY] == pytest.approx(
        baseline[RatingVariant.CURRENT_ONLY]
    )
    assert mutated[RatingVariant.SCALING_CURVE] == pytest.approx(
        baseline[RatingVariant.SCALING_CURVE]
    )
    assert mutated[RatingVariant.FUTURE_PLAYER_FORM] != pytest.approx(
        baseline[RatingVariant.FUTURE_PLAYER_FORM]
    )
    assert mutated[RatingVariant.BOTH] != pytest.approx(baseline[RatingVariant.BOTH])


def test_scaling_mutation_changes_only_v3_and_v4() -> None:
    baseline = {}
    mutated = {}
    for variant in RatingVariant:
        parameters = _variant_parameters(variant)
        baseline[variant] = score_variant_design(
            _variant_design(variant),
            parameters,
            variant=variant,
            expected_game_ids=["g1", "g2"],
        )[1]
        mutated[variant] = score_variant_design(
            _variant_design(variant, scaling_shift=9.0),
            parameters,
            variant=variant,
            expected_game_ids=["g1", "g2"],
        )[1]
    assert mutated[RatingVariant.CURRENT_ONLY] == pytest.approx(
        baseline[RatingVariant.CURRENT_ONLY]
    )
    assert mutated[RatingVariant.FUTURE_PLAYER_FORM] == pytest.approx(
        baseline[RatingVariant.FUTURE_PLAYER_FORM]
    )
    assert mutated[RatingVariant.SCALING_CURVE] != pytest.approx(
        baseline[RatingVariant.SCALING_CURVE]
    )
    assert mutated[RatingVariant.BOTH] != pytest.approx(baseline[RatingVariant.BOTH])


def test_variant_score_rejects_missing_or_extra_ids() -> None:
    parameters = _variant_parameters(RatingVariant.CURRENT_ONLY)
    with pytest.raises(TierShadowError, match="census"):
        score_variant_design(
            _variant_design(RatingVariant.CURRENT_ONLY).iloc[:1],
            parameters,
            variant=RatingVariant.CURRENT_ONLY,
            expected_game_ids=["g1", "g2"],
        )
    extra = pd.concat(
        [_variant_design(RatingVariant.CURRENT_ONLY), _variant_design(RatingVariant.CURRENT_ONLY).iloc[:1]],
        ignore_index=True,
    )
    extra.loc[2, "game_id"] = "g3"
    with pytest.raises(TierShadowError, match="census"):
        score_variant_design(
            extra,
            parameters,
            variant=RatingVariant.CURRENT_ONLY,
            expected_game_ids=["g1", "g2"],
        )
