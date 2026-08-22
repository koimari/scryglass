from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd
import pytest

from lol_kills.research.future_value_draft_score_adapter import (
    DEFAULT_TRUST_ROOT_FILE_SHA256,
    DraftScoreAdapterError,
    adapt_public_crossfit_draft_rows,
    adapt_verified_public_descriptive_draft_records,
    load_default_public_draft_trust_root,
    load_source_bound_atom_ledger,
    verify_public_descriptive_authority,
    write_source_bound_atom_ledger,
)


RELEASE_ID = "v2026.08.20.210112"
FREEZE = Path("/private/tmp/scryglass-four-variant-freeze-20260820T145129")


def _sha(value: object) -> str:
    raw = json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, value: object) -> bytes:
    raw = json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def real_result(tmp_path_factory: pytest.TempPathFactory):
    required = (
        FREEZE / "freeze-manifest.json",
        FREEZE / "future-value-source-receipt.json",
        FREEZE / "baseline/public-pack/manifest.json",
        FREEZE / "baseline/public-pack/features/draft_records.json",
        FREEZE / "source/maps.parquet",
    )
    if not all(path.is_file() for path in required):
        pytest.skip("frozen four-variant pack is unavailable")
    return adapt_verified_public_descriptive_draft_records(
        required[3], required[2], required[1],
        manifest_receipt_path=required[0], release_id=RELEASE_ID,
        repository_root=_repository_root(), source_root=FREEZE,
        output_dir=tmp_path_factory.mktemp("draft-adapter"),
    )


def _authority():
    return verify_public_descriptive_authority(repository_root=_repository_root())


def _fit_evidence(result) -> tuple[str, str, str]:
    scored = set(result.frame["game_id"].astype(str))
    maps = pd.read_parquet(FREEZE / "source/maps.parquet", columns=["game_uid", "date"])
    row = maps.loc[~maps["game_uid"].astype(str).isin(scored)].iloc[0]
    game_id = str(row["game_uid"])
    date = pd.Timestamp(row["date"]).isoformat().replace("+00:00", "Z")
    cutoff = (pd.Timestamp(row["date"]) + pd.Timedelta(days=1)).isoformat().replace("+00:00", "Z")
    return game_id, date, cutoff


def _write_fit_ledger(result, tmp_path: Path):
    game_id, date, cutoff = _fit_evidence(result)
    ledger = write_source_bound_atom_ledger(
        result, tmp_path / "atoms.json", authority=_authority(),
        fold_id="descriptive-public-pack", fit_game_ids=[game_id],
        fit_window_end=cutoff, fit_game_dates={game_id: date},
    )
    return ledger, game_id, date, cutoff


def _load(ledger, result):
    return load_source_bound_atom_ledger(
        ledger.ledger_path, ledger.receipt_path,
        source_receipt=result.source_receipt,
        source_receipt_path=result.source_receipt_path,
        source_root=result.source_root, authority=_authority(),
        expected_fold_id="descriptive-public-pack",
    )


def _reseal_ledger(ledger, *, producer_mutation=None, ledger_mutation=None, receipt_mutation=None):
    producer = json.loads(ledger.producer_receipt_path.read_text())
    artifact = json.loads(ledger.ledger_path.read_text())
    receipt = json.loads(ledger.receipt_path.read_text())
    if producer_mutation:
        producer_mutation(producer)
    producer.pop("receipt_sha256", None)
    producer["receipt_sha256"] = _sha(producer)
    producer_raw = _write_json(ledger.producer_receipt_path, producer)
    for payload in (artifact, receipt):
        payload["producer_receipt_sha256"] = producer["receipt_sha256"]
        payload["producer_receipt_bytes"] = len(producer_raw)
        payload["producer_receipt_file_sha256"] = hashlib.sha256(producer_raw).hexdigest()
    if ledger_mutation:
        ledger_mutation(artifact)
    artifact_raw = _write_json(ledger.ledger_path, artifact)
    if receipt_mutation:
        receipt_mutation(receipt)
    receipt["artifact_bytes"] = len(artifact_raw)
    receipt["artifact_sha256"] = hashlib.sha256(artifact_raw).hexdigest()
    receipt.pop("receipt_sha256", None)
    receipt["receipt_sha256"] = _sha(receipt)
    _write_json(ledger.receipt_path, receipt)


def test_checked_in_trust_root_and_authority_are_byte_pinned() -> None:
    root = load_default_public_draft_trust_root(_repository_root())
    assert root.trust_root_file_sha256 == DEFAULT_TRUST_ROOT_FILE_SHA256
    assert hashlib.sha256(root.trust_root_path.read_bytes()).hexdigest() == DEFAULT_TRUST_ROOT_FILE_SHA256
    authority = _authority()
    assert authority.model_version == "draft-recommendation-static-v2"
    assert authority.model_sha256 == "3a42542710e8a61f11f740ff85965d7f4541724575c3dc7fd063872b7a0c71fe"
    assert authority.authority_receipt_sha256 == "7f6e1a538912b15a021fe90425c5efa4fb91dd88b5be1de0bbb12b1230da0ebd"


def test_real_public_pack_adapts_as_descriptive_only_evidence(real_result, tmp_path: Path) -> None:
    result = real_result
    assert len(result.frame) == 2051
    assert result.chronological_evaluation_suitable is False
    ledger = write_source_bound_atom_ledger(
        result, tmp_path / "real-pack-atoms.json", authority=_authority(),
        fold_id="descriptive-public-pack", fit_game_ids=[],
        fit_window_end="2026-07-18T16:33:48Z",
    )
    assert ledger.receipt["evidence_mode"] == "descriptive_only"
    assert len(_load(ledger, result)) == 2051


def test_forged_authority_chain_cannot_replace_code_pinned_trust_root(tmp_path: Path) -> None:
    repository = _repository_root()
    paths = (
        "data/lol/v2/evaluation/composition-descriptive-authority.json",
        "data/lol/v2/evaluation/composition-descriptive-authority-receipt.json",
        "data/lol/v2/evaluation/future-value-draft-score-trust-root.json",
        "data/lol/v2/evaluation/composition-descriptive-recipe.json",
        "data/lol/models/draft_recommendation.json",
        "lol_kills/research/descriptive_draft_score.py",
    )
    for relative in paths:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repository / relative, target)
    authority_path = tmp_path / paths[0]
    authority = json.loads(authority_path.read_text())
    authority["model_version"] = "attacker-model"
    authority_raw = _write_json(authority_path, authority)
    receipt_path = tmp_path / paths[1]
    receipt = json.loads(receipt_path.read_text())
    receipt["model_version"] = "attacker-model"
    receipt["authority_bytes"] = len(authority_raw)
    receipt["authority_sha256"] = hashlib.sha256(authority_raw).hexdigest()
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = _sha(receipt)
    receipt_raw = _write_json(receipt_path, receipt)
    trust_path = tmp_path / paths[2]
    trust = json.loads(trust_path.read_text())
    trust["descriptive_authority"]["authority_bytes"] = len(authority_raw)
    trust["descriptive_authority"]["authority_sha256"] = hashlib.sha256(authority_raw).hexdigest()
    trust["descriptive_authority"]["authority_receipt_bytes"] = len(receipt_raw)
    trust["descriptive_authority"]["authority_receipt_sha256"] = hashlib.sha256(receipt_raw).hexdigest()
    trust.pop("trust_root_sha256")
    trust["trust_root_sha256"] = _sha(trust)
    _write_json(trust_path, trust)
    with pytest.raises(DraftScoreAdapterError, match="trust root file changed"):
        verify_public_descriptive_authority(repository_root=tmp_path)


def test_resealed_draft_manifest_and_freeze_cannot_replace_release_pin(tmp_path: Path) -> None:
    draft_path = tmp_path / "features/draft_records.json"
    manifest_path = tmp_path / "manifest.json"
    freeze_path = tmp_path / "freeze-manifest.json"
    draft_path.parent.mkdir(parents=True)
    shutil.copy2(FREEZE / "baseline/public-pack/features/draft_records.json", draft_path)
    shutil.copy2(FREEZE / "baseline/public-pack/manifest.json", manifest_path)
    shutil.copy2(FREEZE / "freeze-manifest.json", freeze_path)
    for path in (draft_path, manifest_path, freeze_path):
        path.chmod(0o600)
    draft = json.loads(draft_path.read_text())
    first = next(iter(draft["games"].values()))
    first["edge_components"]["base"] += 0.5
    first["edge_components"]["total"] += 0.5
    draft_raw = _write_json(draft_path, draft)
    manifest = json.loads(manifest_path.read_text())
    for record in manifest["files"]:
        if str(record.get("path", "")).endswith("draft_records.json"):
            record["bytes"] = len(draft_raw)
            record["sha256"] = hashlib.sha256(draft_raw).hexdigest()
    manifest_raw = _write_json(manifest_path, manifest)
    freeze = json.loads(freeze_path.read_text())
    for record in freeze["files"]:
        if str(record["path"]).endswith("manifest.json"):
            record["bytes"] = len(manifest_raw)
            record["sha256"] = hashlib.sha256(manifest_raw).hexdigest()
        if str(record["path"]).endswith("draft_records.json"):
            record["bytes"] = len(draft_raw)
            record["sha256"] = hashlib.sha256(draft_raw).hexdigest()
    freeze.pop("freeze_sha256")
    freeze["freeze_sha256"] = _sha(freeze)
    _write_json(freeze_path, freeze)
    with pytest.raises(DraftScoreAdapterError, match="pinned public (manifest|draft records)"):
        adapt_verified_public_descriptive_draft_records(
            draft_path, manifest_path, FREEZE / "future-value-source-receipt.json",
            manifest_receipt_path=freeze_path, release_id=RELEASE_ID,
            repository_root=_repository_root(), source_root=FREEZE,
        )


def test_loader_rejects_producer_ledger_fit_date_disagreement(real_result, tmp_path: Path) -> None:
    ledger, _game_id, _date, cutoff = _write_fit_ledger(real_result, tmp_path)
    forged = (pd.Timestamp(cutoff) - pd.Timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    _reseal_ledger(ledger, producer_mutation=lambda payload: payload["fit_game_dates"].update({next(iter(payload["fit_game_dates"])): forged}))
    with pytest.raises(DraftScoreAdapterError, match="fit_game_dates binding changed"):
        _load(ledger, real_result)


def test_loader_rejects_self_claimed_fit_date_against_frozen_source(real_result, tmp_path: Path) -> None:
    ledger, game_id, date, _cutoff = _write_fit_ledger(real_result, tmp_path)
    forged = (pd.Timestamp(date) + pd.Timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    mutate = lambda payload: payload["fit_game_dates"].update({game_id: forged})
    _reseal_ledger(ledger, producer_mutation=mutate, ledger_mutation=mutate, receipt_mutation=mutate)
    with pytest.raises(DraftScoreAdapterError, match="fit date changed from frozen source"):
        _load(ledger, real_result)


def test_loader_rejects_self_claimed_scored_date_against_frozen_source(real_result, tmp_path: Path) -> None:
    ledger = write_source_bound_atom_ledger(
        real_result, tmp_path / "atoms.json", authority=_authority(),
        fold_id="descriptive-public-pack", fit_game_ids=[],
        fit_window_end="2026-07-18T16:33:48Z",
    )
    game_id = ledger.receipt["game_ids"][0]
    forged = "2026-01-01T00:00:00Z"

    def mutate_artifact(payload):
        payload["scored_game_dates"][game_id] = forged
        payload["scored_dates_sha256"] = _sha(payload["scored_game_dates"])
        payload["rows"][0]["date"] = forged
        payload["row_digest_sha256"] = _sha(payload["rows"])

    def mutate_receipt(payload):
        payload["scored_game_dates"][game_id] = forged
        payload["scored_dates_sha256"] = _sha(payload["scored_game_dates"])
        artifact = json.loads(ledger.ledger_path.read_text())
        payload["row_digest_sha256"] = artifact["row_digest_sha256"]

    _reseal_ledger(ledger, ledger_mutation=mutate_artifact, receipt_mutation=mutate_receipt)
    with pytest.raises(DraftScoreAdapterError, match="scored date changed from frozen source"):
        _load(ledger, real_result)


def test_crossfit_receipt_requires_a_pin_from_the_checked_in_root(tmp_path: Path) -> None:
    rows_path = tmp_path / "rows.json"
    receipt_path = tmp_path / "receipt.json"
    source_path = tmp_path / "source.json"
    _write_json(rows_path, {"rows": []})
    _write_json(receipt_path, {"receipt_sha256": "a" * 64})
    _write_json(source_path, {})
    with pytest.raises(DraftScoreAdapterError, match="receipt is not pinned"):
        adapt_public_crossfit_draft_rows(
            rows_path, receipt_path, source_path,
            repository_root=_repository_root(),
            receipt_pin_id="attacker-self-sealed", source_root=tmp_path,
        )
