from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from lol_kills.v2.provenance import g1_pre_event_receipt as receipt


def _payload() -> dict:
    return json.loads(receipt.RECEIPT_PATH.read_text(encoding="utf-8"))


def test_committed_receipt_exactly_replays_pinned_non_authorizing_input() -> None:
    payload = _payload()
    expected = receipt.build_receipt()

    assert receipt.validate_receipt(payload) == expected
    assert receipt.canonical_receipt_bytes(payload) == receipt.RECEIPT_PATH.read_bytes()
    assert payload["receipt_sha256"] == "94882c6a6bde5ddf181e964f39342043f75fd2a35a196ae2b301893f2612bc04"
    assert payload["final_holdout"] == {"accessed": False, "included": False, "status": "SEALED_UNREAD"}
    assert payload["claim_ceiling"]["prediction"] is False
    assert payload["claim_ceiling"]["private_model_fit_feature_input"] is True


def test_receipt_schema_is_closed_and_validates_committed_artifact() -> None:
    schema = json.loads(receipt.SCHEMA_PATH.read_text(encoding="utf-8"))
    errors = list(Draft202012Validator(schema).iter_errors(_payload()))
    assert errors == []

    altered = deepcopy(_payload())
    altered["unexpected"] = True
    assert list(Draft202012Validator(schema).iter_errors(altered))

    altered = deepcopy(_payload())
    altered["claim_ceiling"]["prediction"] = None
    assert list(Draft202012Validator(schema).iter_errors(altered))


def test_two_fresh_processes_replay_the_same_receipt_digest() -> None:
    code = (
        "import json; from lol_kills.v2.provenance.g1_pre_event_receipt import build_receipt; "
        "print(json.dumps(build_receipt(), sort_keys=True))"
    )
    import subprocess
    import sys

    first = subprocess.run([sys.executable, "-c", code], cwd=receipt.ROOT, check=True, capture_output=True, text=True).stdout
    second = subprocess.run([sys.executable, "-c", code], cwd=receipt.ROOT, check=True, capture_output=True, text=True).stdout
    assert first == second
    assert json.loads(first)["receipt_sha256"] == _payload()["receipt_sha256"]


def test_feature_rows_are_scanned_for_outcome_keys_without_opening_target_rows() -> None:
    rows = receipt._load_feature_rows()
    assert len(rows) == 1226
    assert sum(len(row["picks"]) for row in rows) == 12260
    assert all("target" not in row and "winner" not in row and "outcome" not in row for row in rows)


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        (lambda payload: payload["claim_ceiling"].update({"prediction": True}), "claim ceiling"),
        (lambda payload: payload["final_holdout"].update({"accessed": True}), "final holdout"),
        (lambda payload: payload["row_interface"].update({"target_included": True}), "receipt"),
    ),
)
def test_self_rehashed_receipt_cannot_expand_authority(mutation, match: str) -> None:
    altered = deepcopy(_payload())
    mutation(altered)
    unsigned = dict(altered)
    unsigned.pop("receipt_sha256")
    altered["receipt_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    with pytest.raises(receipt.G1PreEventReceiptError):
        receipt.validate_receipt(altered)


def test_mutated_pinned_manifest_or_rows_fail_closed(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(receipt.BASE_MANIFEST.read_bytes() + b" ")
    with pytest.raises(receipt.G1PreEventReceiptError, match="unsafe|raw sha256 mismatch"):
        receipt._load_manifest(manifest, receipt.BASE_MANIFEST_RAW_SHA256, receipt.BASE_MANIFEST_CANONICAL_SHA256, label="base G1 manifest")

    rows = tmp_path / "rows.jsonl"
    rows.write_bytes(receipt.FEATURE_ROWS.read_bytes() + b"\n")
    with pytest.raises(receipt.G1PreEventReceiptError, match="unsafe|raw sha256 mismatch"):
        receipt._load_feature_rows(rows)


def test_symlink_hardlink_and_traversal_paths_fail_closed(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_bytes(b"{}")
    alias = tmp_path / "alias.json"
    alias.symlink_to(target)
    with pytest.raises(receipt.G1PreEventReceiptError, match="unsafe"):
        receipt._safe_path(alias, label="test")

    hardlink = tmp_path / "hardlink.json"
    os.link(target, hardlink)
    with pytest.raises(receipt.G1PreEventReceiptError, match="unsafe"):
        receipt._safe_path(hardlink, label="test")

    with pytest.raises(receipt.G1PreEventReceiptError, match="unsafe"):
        receipt._safe_path(receipt.ROOT / ".." / "outside.json", label="test")
