from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lol_kills.research.future_value_draft_score import (
    DraftScoreProducerBinding,
    FutureValueDraftScoreError,
    validate_producer_binding,
)
from lol_kills.research.future_value_draft_score_adapter import (
    DraftScoreAdapterError,
    adapt_public_crossfit_draft_rows,
    adapt_public_descriptive_draft_records,
    adapt_verified_public_descriptive_draft_records,
    load_source_bound_atom_ledger,
    verify_public_descriptive_authority,
    write_source_bound_atom_ledger,
)
from lol_kills.v2.tierlists.accepted_census import identity_sha256


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_json(path: Path, value: object) -> bytes:
    raw = json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


def _public_files(
    tmp_path: Path,
    *,
    fit_through: str = "2026-02-03T00:00:00Z",
    ids: list[str] | None = None,
    draft_ids: list[str] | None = None,
) -> tuple[Path, Path, Path]:
    ids = ids or ["game-1", "game-2"]
    draft_ids = draft_ids or ids
    source_file = tmp_path / "accepted-census.csv"
    source_raw = b"game-1\ngame-2\n"
    source_file.write_bytes(source_raw)
    source = {
        "schema_version": "scryglass:future-value-rating-source:v1",
        "status": "verified_public_pack_source",
        "source_as_of": "2026-02-03T00:00:00Z",
        "source_game_count": len(ids),
        "source_identity_sha256": identity_sha256(ids),
        "accepted_game_ids": ids,
        "source_files": {
            "accepted_census": {
                "locator": source_file.name,
                "bytes": len(source_raw),
                "sha256": hashlib.sha256(source_raw).hexdigest(),
            }
        },
        "authority": {"research_only": True},
    }
    source["receipt_sha256"] = _sha(source)
    source_path = _write_json(tmp_path / "source-receipt.json", source)
    source_receipt_path = tmp_path / "source-receipt.json"
    assert source_path

    games = {
        game_id: {
            "date": "2026-02-02T00:00:00Z",
            "edge_components": {
                "base": 1.0 if game_id == "game-1" else 2.0,
                "ally_synergy": 2.0,
                "enemy_counter": 3.0,
                "same_role": 4.0,
                "archetype_interactions": 5.0,
                "total": 15.0 if game_id == "game-1" else 16.0,
            },
        }
        for game_id in draft_ids
    }
    draft = {
        "schema_version": "scryglass:draft-records:v1",
        "authority": "descriptive",
        "estimand": "composition_only",
        "model_version": "draft-recommendation-static-v2",
        "fit_through": fit_through,
        "source_identity_sha256": source["source_identity_sha256"],
        "release_id": "v2026.02.03.000001",
        "games": games,
    }
    draft_path = tmp_path / "features" / "draft_records.json"
    draft_raw = _write_json(draft_path, draft)
    manifest = {
        "pack_id": "v2026.02.03.000001",
        "source_identity_sha256": source["source_identity_sha256"],
        "files": [
            {
                "path": "features/draft_records.json",
                "bytes": len(draft_raw),
                "sha256": hashlib.sha256(draft_raw).hexdigest(),
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    _write_json(manifest_path, manifest)
    return draft_path, manifest_path, source_receipt_path


def test_public_descriptive_adapter_converts_real_five_component_edges(tmp_path: Path) -> None:
    draft_path, manifest_path, source_path = _public_files(tmp_path)
    result = adapt_public_descriptive_draft_records(
        draft_path,
        manifest_path,
        source_path,
        output_dir=tmp_path / "adapter",
    )
    assert list(result.frame["game_id"]) == ["game-1", "game-2"]
    assert [column for column in result.frame if column.startswith("composition_")] == [
        "composition_base_logit",
        "composition_ally_synergy_logit",
        "composition_enemy_counter_logit",
        "composition_same_role_logit",
        "composition_archetype_interactions_logit",
    ]
    assert result.frame.loc[0, "composition_enemy_counter_logit"] == 3.0
    assert result.chronological_evaluation_suitable is False
    assert result.static_atom_receipt_path.is_file()
    assert result.static_atom_receipt["source_receipt_sha256"] == json.loads(source_path.read_text())["receipt_sha256"]


def test_public_descriptive_adapter_rejects_manifest_hash_mutation(tmp_path: Path) -> None:
    draft_path, manifest_path, source_path = _public_files(tmp_path)
    draft_path.write_text(draft_path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(DraftScoreAdapterError, match="bytes or SHA-256"):
        adapt_public_descriptive_draft_records(draft_path, manifest_path, source_path)


def test_public_descriptive_adapter_rejects_self_sealed_source_fields(tmp_path: Path) -> None:
    draft_path, manifest_path, source_path = _public_files(tmp_path)
    source = json.loads(source_path.read_text())
    source["forged_by"] = "attacker"
    source["receipt_sha256"] = _sha({key: value for key, value in source.items() if key != "receipt_sha256"})
    _write_json(source_path, source)
    with pytest.raises(DraftScoreAdapterError, match="unknown fields"):
        adapt_public_descriptive_draft_records(draft_path, manifest_path, source_path)

    source.pop("forged_by")
    source.pop("authority")
    source["receipt_sha256"] = _sha({key: value for key, value in source.items() if key != "receipt_sha256"})
    _write_json(source_path, source)
    with pytest.raises(DraftScoreAdapterError, match="incomplete"):
        adapt_public_descriptive_draft_records(draft_path, manifest_path, source_path)


def test_public_descriptive_adapter_rejects_symlinked_source_artifact(tmp_path: Path) -> None:
    draft_path, manifest_path, source_path = _public_files(tmp_path)
    target = tmp_path / "real-census.csv"
    target.write_bytes(b"game-1\ngame-2\n")
    link = tmp_path / "linked-census.csv"
    link.symlink_to(target)
    source = json.loads(source_path.read_text())
    source["source_files"]["accepted_census"]["locator"] = link.name
    source["source_files"]["accepted_census"]["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    source["receipt_sha256"] = _sha({key: value for key, value in source.items() if key != "receipt_sha256"})
    _write_json(source_path, source)
    with pytest.raises(DraftScoreAdapterError, match="unsafe|symlink"):
        adapt_public_descriptive_draft_records(draft_path, manifest_path, source_path)


def test_frozen_public_pack_adapts_as_coverage_scoped_evidence(tmp_path: Path) -> None:
    freeze = Path("/private/tmp/scryglass-four-variant-freeze-20260820T145129")
    draft_path = freeze / "baseline/public-pack/features/draft_records.json"
    manifest_path = freeze / "baseline/public-pack/manifest.json"
    source_path = freeze / "future-value-source-receipt.json"
    if not all(path.is_file() for path in (draft_path, manifest_path, source_path)):
        pytest.skip("frozen public pack is unavailable")
    repository_root = Path(__file__).resolve().parents[1]
    result = adapt_verified_public_descriptive_draft_records(
        draft_path,
        manifest_path,
        source_path,
        authority_path=repository_root
        / "data/lol/v2/evaluation/composition-descriptive-authority.json",
        repository_root=repository_root,
        output_dir=tmp_path / "adapter",
        source_root=freeze,
    )
    assert 0 < len(result.frame) < result.source_receipt["source_game_count"]
    assert result.static_atom_receipt["coverage_game_count"] == len(result.frame)
    assert result.static_atom_receipt["coverage_game_ids"] == list(result.frame["game_id"])
    assert result.chronological_evaluation_suitable is False
    authority = verify_public_descriptive_authority(
        repository_root
        / "data/lol/v2/evaluation/composition-descriptive-authority.json",
        repository_root=repository_root,
    )
    ledger = write_source_bound_atom_ledger(
        result,
        tmp_path / "real-pack-atoms.json",
        authority=authority,
        fold_id="descriptive-public-pack",
        fit_game_ids=[],
        fit_window_end="2026-07-18T16:33:48Z",
    )
    assert ledger.row_count == len(result.frame)
    assert ledger.receipt["evidence_mode"] == "descriptive_only"
    loaded = load_source_bound_atom_ledger(
        ledger.ledger_path,
        ledger.receipt_path,
        source_receipt=result.source_receipt,
        authority=authority,
        expected_fold_id="descriptive-public-pack",
    )
    assert len(loaded) == 2051
    with pytest.raises(DraftScoreAdapterError, match="fit and scored"):
        write_source_bound_atom_ledger(
            result,
            tmp_path / "invalid-overlap-atoms.json",
            authority=authority,
            fold_id="invalid-overlap",
            fit_game_ids=[str(result.frame.iloc[0]["game_id"])],
            fit_window_end="2026-07-18T16:33:48Z",
            fit_game_dates={
                str(result.frame.iloc[0]["game_id"]): "2026-01-01T00:00:00Z"
            },
        )


def test_independent_descriptive_authority_binds_model_recipe_and_scorer() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    authority_path = repository_root / "data/lol/v2/evaluation/composition-descriptive-authority.json"
    result = verify_public_descriptive_authority(
        authority_path,
        repository_root=repository_root,
    )
    assert result.model_sha256 == "3a42542710e8a61f11f740ff85965d7f4541724575c3dc7fd063872b7a0c71fe"
    assert result.recipe_sha256
    assert result.scorer_sha256
    assert result.authority_receipt_sha256


def test_verified_adapter_writes_source_bound_atom_ledger(tmp_path: Path) -> None:
    draft_path, manifest_path, source_path = _public_files(
        tmp_path,
        fit_through="2026-02-01T00:00:00Z",
        ids=["fit-0", "game-1", "game-2"],
        draft_ids=["game-1", "game-2"],
    )
    model_path = tmp_path / "model.json"
    recipe_path = tmp_path / "recipe.json"
    scorer_path = tmp_path / "scorer.py"
    model_raw = b"{\"model\":\"fixture\"}\n"
    recipe_raw = b"{\"recipe\":\"fixture\"}\n"
    scorer_raw = b"# fixture scorer\n"
    model_path.write_bytes(model_raw)
    recipe_path.write_bytes(recipe_raw)
    scorer_path.write_bytes(scorer_raw)
    draft = json.loads(draft_path.read_text())
    draft["artifact_sha256"] = hashlib.sha256(model_raw).hexdigest()
    draft_path.write_bytes(_write_json(draft_path, draft))
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][0]["bytes"] = draft_path.stat().st_size
    manifest["files"][0]["sha256"] = hashlib.sha256(draft_path.read_bytes()).hexdigest()
    _write_json(manifest_path, manifest)
    authority = {
        "schema_version": "scryglass:draft-authority:v1",
        "status": "descriptive",
        "estimand": "composition_only",
        "model_version": "draft-recommendation-static-v2",
        "artifact_path": model_path.name,
        "artifact_sha256": hashlib.sha256(model_raw).hexdigest(),
        "recipe_path": recipe_path.name,
        "recipe_sha256": hashlib.sha256(recipe_raw).hexdigest(),
        "scorer_code_path": scorer_path.name,
        "scorer_code_sha256": hashlib.sha256(scorer_raw).hexdigest(),
        "probability_authority": False,
        "recommendation_authority": False,
        "betting_authority": False,
    }
    authority_path = tmp_path / "authority.json"
    _write_json(authority_path, authority)
    result = adapt_verified_public_descriptive_draft_records(
        draft_path,
        manifest_path,
        source_path,
        authority_path=authority_path,
        repository_root=tmp_path,
        output_dir=tmp_path / "adapter",
    )
    ledger = write_source_bound_atom_ledger(
        result,
        tmp_path / "ledger" / "atoms.json",
        authority=verify_public_descriptive_authority(authority_path, repository_root=tmp_path),
        fold_id="fold-1",
        fit_game_ids=["fit-0"],
        fit_window_start="2026-01-30T00:00:00Z",
        fit_window_end="2026-02-01T00:00:00Z",
        fit_game_dates={"fit-0": "2026-01-31T00:00:00Z"},
    )
    assert ledger.row_count == 2
    assert ledger.ledger_path.is_file()
    assert ledger.receipt_path.is_file()
    assert ledger.producer_receipt_path.is_file()
    assert ledger.receipt["source_identity_sha256"] == result.source_receipt["source_identity_sha256"]
    loaded = load_source_bound_atom_ledger(
        ledger.ledger_path,
        ledger.receipt_path,
        source_receipt=result.source_receipt,
        authority=verify_public_descriptive_authority(authority_path, repository_root=tmp_path),
        expected_fold_id="fold-1",
    )
    assert list(loaded["game_id"]) == ["game-1", "game-2"]


def test_verified_adapter_keeps_overlapping_static_fit_blocked(tmp_path: Path) -> None:
    draft_path, manifest_path, source_path = _public_files(
        tmp_path,
        fit_through="2026-02-03T00:00:00Z",
        ids=["fit-0", "game-1", "game-2"],
        draft_ids=["game-1", "game-2"],
    )
    repository_root = Path(__file__).resolve().parents[1]
    authority_path = (
        repository_root / "data/lol/v2/evaluation/composition-descriptive-authority.json"
    )
    authority = verify_public_descriptive_authority(
        authority_path,
        repository_root=repository_root,
    )
    draft = json.loads(draft_path.read_text())
    draft["artifact_sha256"] = authority.model_sha256
    draft_raw = _write_json(draft_path, draft)
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][0]["bytes"] = len(draft_raw)
    manifest["files"][0]["sha256"] = hashlib.sha256(draft_raw).hexdigest()
    _write_json(manifest_path, manifest)
    result = adapt_verified_public_descriptive_draft_records(
        draft_path,
        manifest_path,
        source_path,
        authority_path=authority_path,
        repository_root=repository_root,
        output_dir=tmp_path / "adapter",
    )
    assert result.chronological_evaluation_suitable is False
    ledger = write_source_bound_atom_ledger(
        result,
        tmp_path / "blocked-atoms.json",
        authority=authority,
        fold_id="fold-blocked",
        fit_game_ids=["fit-0"],
        fit_window_end="2026-02-03T00:00:00Z",
        fit_game_dates={"fit-0": "2026-01-31T00:00:00Z"},
    )
    binding = DraftScoreProducerBinding.create(
        source_receipt=result.source_receipt,
        accepted_game_ids=result.source_receipt["accepted_game_ids"],
        fit_game_ids=["fit-0"],
        fit_window_end="2026-02-03T00:00:00Z",
        fit_game_dates={"fit-0": "2026-01-31T00:00:00Z"},
        fold_id="fold-blocked",
        producer="public_descriptive_draft_records",
        producer_family="static_composition",
        series_safe_evidence={
            "series_safe": True,
            "fit_validation_disjoint": True,
            "source_type": "public_pack",
            "series_column": "game_id",
            "cluster_identity_sha256": result.source_receipt["source_identity_sha256"],
        },
        source_receipt_path=result.source_receipt_path,
        source_root=tmp_path,
        producer_receipt_path=ledger.producer_receipt_path,
    )
    with pytest.raises(FutureValueDraftScoreError, match="strictly prior"):
        validate_producer_binding(result.frame, binding)


def test_crossfit_adapter_maps_public_component_rows(tmp_path: Path) -> None:
    _draft_path, _manifest_path, source_path = _public_files(
        tmp_path,
        fit_through="2026-01-01T00:00:00Z",
        ids=["fit-0", "game-1", "game-2"],
    )
    rows_path = tmp_path / "crossfit-rows.json"
    rows = {
        "rows": [
            {
                "game_uid": "game-1",
                "date": "2026-02-02T00:00:00Z",
                "crossfit_champion_main": 0.1,
                "crossfit_role_champion": 0.2,
                "crossfit_ally_synergy": 0.3,
                "crossfit_archetype_synergy": 0.4,
                "crossfit_enemy_counter": 0.5,
                "crossfit_archetype_counter": 0.6,
                "crossfit_same_role": 0.7,
            },
            {
                "game_uid": "game-2",
                "date": "2026-02-02T00:00:00Z",
                "crossfit_champion_main": 0.1,
                "crossfit_role_champion": 0.2,
                "crossfit_ally_synergy": 0.3,
                "crossfit_archetype_synergy": 0.4,
                "crossfit_enemy_counter": 0.5,
                "crossfit_archetype_counter": 0.6,
                "crossfit_same_role": 0.7,
            },
        ]
    }
    rows_raw = _write_json(rows_path, rows)
    source = json.loads(source_path.read_text())
    receipt = {
        "schema_version": "scryglass:public-crossfit-draft-receipt:v1",
        "source_receipt_sha256": source["receipt_sha256"],
        "source_identity_sha256": source["source_identity_sha256"],
        "accepted_game_ids": ["fit-0", "game-1", "game-2"],
        "fold_id": "fold-1",
        "model_id": "crossfit-v1",
        "fit_game_ids": ["fit-0"],
        "fit_game_identity_sha256": identity_sha256(["fit-0"]),
        "producer_timing": "cross_fitted_pregame",
        "artifact_locator": str(rows_path),
        "artifact_bytes": len(rows_raw),
        "artifact_sha256": hashlib.sha256(rows_raw).hexdigest(),
    }
    receipt["receipt_sha256"] = _sha(receipt)
    receipt_path = tmp_path / "crossfit-receipt.json"
    _write_json(receipt_path, receipt)
    result = adapt_public_crossfit_draft_rows(rows_path, receipt_path, source_path)
    assert result.frame.loc[0, "composition_base_logit"] == pytest.approx(0.3)
    assert result.frame.loc[0, "composition_archetype_interactions_logit"] == pytest.approx(1.0)
    assert result.frame.loc[0, "producer_timing"] == "cross_fitted_pregame"


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing_component", "components are incomplete"),
        ("null_artifact_sha", "artifact_sha256"),
        ("null_artifact_bytes", "byte count"),
        ("fit_overlap", "fit and scored"),
        ("fit_outside", "outside the model fold census"),
    ),
)
def test_crossfit_adapter_rejects_incomplete_or_leaking_evidence(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    _draft_path, _manifest_path, source_path = _public_files(
        tmp_path,
        ids=["fit-0", "game-1"],
    )
    row = {
        "game_uid": "game-1",
        "date": "2026-02-02T00:00:00Z",
        "crossfit_champion_main": 0.1,
        "crossfit_role_champion": 0.2,
        "crossfit_ally_synergy": 0.3,
        "crossfit_archetype_synergy": 0.4,
        "crossfit_enemy_counter": 0.5,
        "crossfit_archetype_counter": 0.6,
        "crossfit_same_role": 0.7,
    }
    if mutation == "missing_component":
        row.pop("crossfit_enemy_counter")
    rows_path = tmp_path / "crossfit-rows.json"
    rows_raw = _write_json(rows_path, {"rows": [row]})
    source = json.loads(source_path.read_text())
    fit_ids = ["fit-0"]
    if mutation == "fit_overlap":
        fit_ids = ["game-1"]
    elif mutation == "fit_outside":
        fit_ids = ["outside"]
    receipt = {
        "schema_version": "scryglass:public-crossfit-draft-receipt:v1",
        "source_receipt_sha256": source["receipt_sha256"],
        "source_identity_sha256": source["source_identity_sha256"],
        "accepted_game_ids": source["accepted_game_ids"],
        "fold_id": "fold-1",
        "model_id": "crossfit-v1",
        "fit_game_ids": fit_ids,
        "fit_game_identity_sha256": identity_sha256(fit_ids),
        "producer_timing": "cross_fitted_pregame",
        "artifact_locator": str(rows_path),
        "artifact_bytes": None if mutation == "null_artifact_bytes" else len(rows_raw),
        "artifact_sha256": None
        if mutation == "null_artifact_sha"
        else hashlib.sha256(rows_raw).hexdigest(),
    }
    receipt["receipt_sha256"] = _sha(receipt)
    receipt_path = tmp_path / "crossfit-receipt.json"
    _write_json(receipt_path, receipt)
    with pytest.raises(DraftScoreAdapterError, match=message):
        adapt_public_crossfit_draft_rows(rows_path, receipt_path, source_path)
