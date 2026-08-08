from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from lol_kills.v2.provenance import g1_source_authority as authority


RECEIPT_PATH = (
    authority.REPO_ROOT
    / "data/lol/v2/snapshots/real-v1/g1-source-authority-inventory.json"
)


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_catalog(path: Path, payload: dict) -> Path:
    unsigned = dict(payload)
    unsigned.pop("catalog_sha256", None)
    payload["catalog_sha256"] = _canonical_sha256(unsigned)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def _catalog() -> dict:
    return json.loads(authority.DEFAULT_GRID_CATALOG.read_text(encoding="utf-8"))


def _receipt() -> dict:
    return json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))


def test_committed_receipt_exactly_replays_pinned_sources_and_catalog() -> None:
    payload = _receipt()
    expected = authority.build_g1_source_authority_receipt()

    assert authority.validate_g1_source_authority_receipt(payload) == expected
    assert authority.canonical_receipt_bytes(payload) == RECEIPT_PATH.read_bytes()
    assert payload["receipt_sha256"] == (
        "d85d504e25a23d4853f4e0201849b9e5b953fa0e769458a7d8bb2f810d98de59"
    )


def test_receipt_preserves_exact_claim_ceiling_and_typed_blockers() -> None:
    payload = _receipt()

    assert payload["disposition"] == "SOURCE_INVENTORY_BOUND_AUTHORITY_UNAVAILABLE"
    assert payload["content_addressing_confers_authority"] is False
    assert payload["claim_ceiling"] == {
        "source_inventory_preflight": True,
        "existing_private_fit_authority_expanded": False,
        "current_roster": False,
        "pre_event_roster": False,
        "authoritative_provider_series_crosswalk": False,
        "historical_ingest": False,
        "grid_row_coverage": False,
        "benchmark_source_bound_transition": False,
        "forecast": False,
        "prediction": False,
        "production": False,
        "publication": False,
        "promotion": False,
        "sota": False,
        "final_holdout": False,
    }
    assert {row["code"] for row in payload["typed_blockers"]} == {
        "CURRENT_ROSTER_AUTHORITY_UNAVAILABLE",
        "AUTHORITATIVE_SERIES_CROSSWALK_UNBOUND",
        "HISTORICAL_INGEST_RECEIPT_UNAVAILABLE",
        "GRID_HISTORICAL_PAYLOAD_NOT_AUTHORIZED_OR_DOWNLOADED",
        "G1_UNIFIED_BENCHMARK_AUTHORITY_BUNDLE_UNAVAILABLE",
        "FINAL_HOLDOUT_SEALED",
    }
    assert payload["final_holdout"] == {
        "status": "SEALED_UNREAD",
        "accessed": False,
        "included": False,
    }


def test_catalog_is_reduced_to_provenance_metadata_without_probe_identity() -> None:
    receipt_catalog = _receipt()["grid_catalog_provenance"]
    source_catalog = _catalog()
    probe_series_id = source_catalog["file_listing_probe"]["series_id"]

    assert receipt_catalog["use_boundary"] == (
        "PROVENANCE_METADATA_ONLY_NO_QUERY_NO_DOWNLOAD"
    )
    assert receipt_catalog["file_listing"] == {
        "status": "confirmed",
        "download_attempted": False,
        "signed_urls_retained": False,
        "payload_completeness_authority": False,
    }
    assert "files" not in receipt_catalog
    assert "series_id" not in receipt_catalog
    assert "introspection" not in receipt_catalog
    if isinstance(probe_series_id, str) and probe_series_id:
        assert probe_series_id not in json.dumps(receipt_catalog, sort_keys=True)


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        (
            lambda catalog: next(
                row
                for row in catalog["capabilities"]
                if row["capability"] == "historical_file_download"
            ).update({"status": "confirmed"}),
            "historical_file_download",
        ),
        (
            lambda catalog: catalog["file_listing_probe"].update(
                {"download_attempted": True}
            ),
            "attempted a download",
        ),
        (
            lambda catalog: catalog["scope"].update({"model_authority": True}),
            "model_authority",
        ),
    ),
)
def test_self_consistent_catalog_cannot_expand_authority(
    tmp_path: Path,
    mutation,
    match: str,
) -> None:
    catalog = _catalog()
    mutation(catalog)
    path = _write_catalog(tmp_path / "catalog.json", catalog)

    with pytest.raises(authority.G1SourceAuthorityError, match=match):
        authority.build_g1_source_authority_receipt(catalog_path=path)


def test_catalog_schema_digest_drift_fails_closed(tmp_path: Path) -> None:
    catalog = _catalog()
    catalog["endpoints"][0]["introspection"]["queryType"]["name"] = "DriftedQuery"
    path = _write_catalog(tmp_path / "catalog.json", catalog)

    with pytest.raises(authority.G1SourceAuthorityError, match="schema digest drifted"):
        authority.build_g1_source_authority_receipt(catalog_path=path)


def test_catalog_symlink_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "catalog-target.json"
    target.write_bytes(authority.DEFAULT_GRID_CATALOG.read_bytes())
    alias = tmp_path / "catalog-alias.json"
    alias.symlink_to(target)

    with pytest.raises(authority.G1SourceAuthorityError, match="symlink"):
        authority.build_g1_source_authority_receipt(catalog_path=alias)


def test_pinned_accepted_manifest_drift_fails_before_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = authority._read_repo_bytes
    target = authority._PINNED_REPOSITORY_ARTIFACTS[
        "private_lpl_snapshot_manifest"
    ]["locator"]

    def drifted(root: Path, locator: str) -> bytes:
        raw = original(root, locator)
        return raw + b" " if locator == target else raw

    monkeypatch.setattr(authority, "_read_repo_bytes", drifted)
    with pytest.raises(authority.G1SourceAuthorityError, match="artifact drifted"):
        authority.build_g1_source_authority_receipt()


def test_self_rehashed_receipt_cannot_claim_current_roster_authority() -> None:
    payload = deepcopy(_receipt())
    payload["claim_ceiling"]["current_roster"] = True
    payload.pop("receipt_sha256")
    payload["receipt_sha256"] = _canonical_sha256(payload)

    with pytest.raises(
        authority.G1SourceAuthorityError,
        match="differs from reopened pinned evidence",
    ):
        authority.validate_g1_source_authority_receipt(payload)


def test_source_contract_tree_is_identity_only_and_binds_foundations() -> None:
    tree = _receipt()["source_contract_tree"]

    assert tree["allowlist"] == list(authority.SOURCE_CONTRACT_ALLOWLIST)
    assert tree["authority_effect"] == "NONE_CONTRACT_IDENTITY_ONLY"
    assert {row["locator"] for row in tree["files"]} == {
        "lol_kills/v2/data/rosters.py",
        "lol_kills/v2/data/series.py",
        "lol_kills/v2/data/source_tree.py",
        "lol_kills/v2/provenance/g1_source_authority.py",
    }
    assert {
        row["authority_effect"] for row in tree["files"]
    } == {"NONE_CONTENT_ADDRESSING_ONLY"}
