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
    FINAL_MODEL_AUTHORITY as V2_FINAL_MODEL_AUTHORITY,
    V2TierShadowError,
    canonical_json_bytes,
    load_v2_tier_offset_ledger,
    score_v2_design,
    verify_target_parity,
    write_v2_tier_offset_ledger,
)
from lol_kills.research.future_value_tier_shadow import (
    FINAL_MODEL_AUTHORITY,
    TierShadowError,
    _verify_authority,
    _verify_current_feature_binding,
    load_tier_offset_ledger,
    score_variant_design,
    verify_final_model,
    write_tier_offset_ledger,
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


def test_tier_authority_is_closed_and_explicitly_non_authoritative() -> None:
    expected_false = {
        "public_tierlist",
        "public_player_rating",
        "public_team_rating",
        "public_probability",
        "promotion",
        "merge",
        "deployment",
        "odds",
        "expected_value",
        "recommendation",
        "betting",
    }
    assert AUTHORITY["research_only"] is True
    assert {key for key, value in AUTHORITY.items() if value is False} == expected_false
    assert V2_FINAL_MODEL_AUTHORITY["research_only"] is True
    assert {
        key for key, value in V2_FINAL_MODEL_AUTHORITY.items() if value is False
    } == expected_false - {"public_tierlist"}
    with pytest.raises(TierShadowError, match="authority"):
        _verify_authority(
            {key: value for key, value in AUTHORITY.items() if key != "odds"},
            "Tier authority",
        )


def test_current_feature_binding_uses_its_closed_receipt_schema() -> None:
    ids = ["g1", "g2"]
    source = _source_receipt(ids)
    binding = {
        "schema_version": "scryglass:future-value-final-current-rating-binding:v1",
        "source_receipt_sha256": source["receipt_sha256"],
        "source_identity_sha256": source["source_identity_sha256"],
        "artifact": {"sha256": "c" * 64},
        "fit_game_ids": ids,
        "fit_game_identity_sha256": identity_sha256(ids),
        "game_identity_sha256": identity_sha256(ids),
        "rows": len(ids),
        "feature_names": list(CURRENT_RATING_SIGNED_MAP_FEATURES),
    }

    _verify_current_feature_binding(
        binding,
        source,
        expected_ledger_sha256="c" * 64,
        expected_game_ids=ids,
    )

    mutated = dict(binding)
    mutated["fit_game_ids"] = ["g1"]
    with pytest.raises(TierShadowError, match="game census"):
        _verify_current_feature_binding(
            mutated,
            source,
            expected_ledger_sha256="c" * 64,
            expected_game_ids=ids,
        )


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


def test_score_v2_design_digest_is_row_order_invariant() -> None:
    forward = score_v2_design(
        _design(), _parameters(), expected_game_ids=["g1", "g2"]
    )
    reverse = score_v2_design(
        _design().iloc[::-1].reset_index(drop=True),
        _parameters(),
        expected_game_ids=["g1", "g2"],
    )
    assert forward[0] == reverse[0]
    assert forward[1] == reverse[1]
    assert forward[2] == reverse[2]


def test_v2_target_parity_requires_exact_target_rows() -> None:
    maps = pd.DataFrame(
        {
            "game_uid": ["g1", "g2"],
            "date": ["2026-02-01T00:00:00Z", "2026-02-02T00:00:00Z"],
            "y_blue_win": [1, 0],
        }
    )
    with pytest.raises(V2TierShadowError, match="full census"):
        verify_target_parity(
            [{"game_id": "g1", "target": 1}],
            maps,
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
    model.write_bytes(b"mutated model\n")
    with pytest.raises(V2TierShadowError, match="bytes changed"):
        load_v2_tier_offset_ledger(
            result.ledger_path,
            result.receipt_path,
            source_receipt=source,
        )


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


def _final_model_source_binding(source: dict[str, object]) -> dict[str, object]:
    eligible = tuple(sorted(str(value) for value in source["model_eligible_game_ids"]))
    return {
        "source_as_of": source["source_as_of"],
        "source_receipt_sha256": source["receipt_sha256"],
        "source_identity_sha256": source["source_identity_sha256"],
        "source_game_count": source["source_game_count"],
        "model_eligible_game_count": len(eligible),
        "model_eligible_game_ids": list(eligible),
        "model_eligible_identity_sha256": identity_sha256(eligible),
    }


def _write_final_model_fixture(
    root: Path,
    source: dict[str, object],
    *,
    omission: str | None = None,
) -> tuple[Path, Path, Path, dict[str, str]]:
    root.mkdir(parents=True, exist_ok=True)
    eligible = tuple(sorted(str(value) for value in source["model_eligible_game_ids"]))
    binding = _final_model_source_binding(source)
    authority = dict(FINAL_MODEL_AUTHORITY)
    if omission == "authority":
        authority.pop("odds")
    receipt: dict[str, object] = {
        "schema_version": "scryglass:future-value-final-fit:v1",
        "status": "research_only_blocked",
        "authority": authority,
        "variant": RatingVariant.CURRENT_ONLY.value,
        "source_binding": binding,
        "fit_game_ids": list(eligible),
        "fit_game_count": len(eligible),
        "fit_game_identity_sha256": identity_sha256(eligible),
        "fit_window_end": source["source_as_of"],
    }
    if omission == "source":
        receipt.pop("source_binding")
    if omission == "fit":
        receipt.pop("fit_game_ids")
    if omission == "timing":
        receipt.pop("fit_window_end")
    receipt["receipt_sha256"] = hashlib.sha256(
        canonical_json_bytes(receipt)
    ).hexdigest()
    model: dict[str, object] = {
        "schema_version": "scryglass:future-value-final-fit:v1",
        "status": "research_only_blocked",
        "authority": authority,
        "variant": RatingVariant.CURRENT_ONLY.value,
        "source": binding,
        "receipt_sha256": receipt["receipt_sha256"],
        "parameters": _variant_parameters(RatingVariant.CURRENT_ONLY),
    }
    if omission == "source":
        model.pop("source")
    model_path = root / "model.json"
    receipt_path = root / "model-receipt.json"
    run_path = root / "run.json"
    model_path.write_bytes(canonical_json_bytes(model) + b"\n")
    receipt_path.write_bytes(canonical_json_bytes(receipt) + b"\n")
    run: dict[str, object] = {
        "schema_version": "scryglass:future-value-final-fit:v1",
        "status": "research_only_blocked",
        "authority": authority,
        "variant": RatingVariant.CURRENT_ONLY.value,
        "source_receipt_sha256": source["receipt_sha256"],
        "source_identity_sha256": source["source_identity_sha256"],
        "fit_game_count": len(eligible),
        "fit_game_identity_sha256": identity_sha256(eligible),
        "eligible_game_ids": list(eligible),
        "eligible_game_identity_sha256": identity_sha256(eligible),
        "fit_window_end": source["source_as_of"],
        "model_receipt_sha256": receipt["receipt_sha256"],
        "model_artifact_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    }
    if omission == "source":
        run.pop("source_receipt_sha256")
    if omission == "fit":
        run.pop("eligible_game_ids")
    if omission == "timing":
        run.pop("fit_window_end")
    run_path.write_bytes(canonical_json_bytes(run) + b"\n")
    hashes = {
        "model": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "receipt": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        "run": hashlib.sha256(run_path.read_bytes()).hexdigest(),
    }
    return model_path, receipt_path, run_path, hashes


@pytest.mark.parametrize(
    ("omission", "message"),
    [
        ("source", "source binding"),
        ("fit", "fit census"),
        ("timing", "fit cutoff"),
        ("authority", "authority"),
    ],
)
def test_variant_model_requires_closed_bindings(
    tmp_path: Path,
    omission: str,
    message: str,
) -> None:
    source = _source_receipt(["g1", "g2"])
    model, receipt, run, hashes = _write_final_model_fixture(
        tmp_path / omission,
        source,
        omission=omission,
    )
    with pytest.raises(TierShadowError, match=message):
        verify_final_model(
            model,
            receipt,
            run,
            variant=RatingVariant.CURRENT_ONLY,
            expected_model_sha256=hashes["model"],
            expected_model_receipt_file_sha256=hashes["receipt"],
            expected_run_receipt_sha256=hashes["run"],
            source_receipt=source,
        )


def test_variant_model_accepts_complete_closed_bindings(tmp_path: Path) -> None:
    source = _source_receipt(["g1", "g2"])
    model, receipt, run, hashes = _write_final_model_fixture(tmp_path, source)
    verified = verify_final_model(
        model,
        receipt,
        run,
        variant=RatingVariant.CURRENT_ONLY,
        expected_model_sha256=hashes["model"],
        expected_model_receipt_file_sha256=hashes["receipt"],
        expected_run_receipt_sha256=hashes["run"],
        source_receipt=source,
    )
    assert verified[0]["variant"] == RatingVariant.CURRENT_ONLY.value


def test_neutral_ledger_reopens_every_recorded_input(tmp_path: Path) -> None:
    variant = RatingVariant.CURRENT_ONLY
    rows, offsets, matrix_hash = score_variant_design(
        _variant_design(variant),
        _variant_parameters(variant),
        variant=variant,
        expected_game_ids=["g1", "g2"],
    )
    source = _source_receipt(["g1", "g2"])
    source_file = _bound_file(tmp_path, "source-receipt.json")
    source_files = {
        label: _bound_file(tmp_path, f"{label}.parquet")
        for label in ("maps", "players", "teams")
    }
    model = _bound_file(tmp_path, "model.json")
    model_receipt = _bound_file(tmp_path, "model-receipt.json")
    run = _bound_file(tmp_path, "run.json")
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
    result = write_tier_offset_ledger(
        tmp_path / "neutral-ledger.json",
        variant=variant,
        rows=rows,
        offsets=offsets,
        design_matrix_sha256=matrix_hash,
        source_receipt=source,
        source_receipt_file=source_file,
        source_files=source_files,
        model_file=model,
        model_receipt_file=model_receipt,
        run_receipt_file=run,
        current_ledger_file=current,
        current_receipt_file=current_receipt,
        target_rows_sha256=target_hash,
    )
    loaded, _ = load_tier_offset_ledger(
        result.ledger_path,
        result.receipt_path,
        source_receipt=source,
        variant=variant,
    )
    assert loaded == pytest.approx(offsets)
    model.write_bytes(b"mutated model\n")
    with pytest.raises(TierShadowError, match="bytes changed"):
        load_tier_offset_ledger(
            result.ledger_path,
            result.receipt_path,
            source_receipt=source,
            variant=variant,
        )
