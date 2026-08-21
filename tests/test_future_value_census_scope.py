from __future__ import annotations

import pandas as pd
import pytest

from benchmarks.future_value_four_variant_bundle import (
    FourVariantBundleError,
    _accepted_map_frame as bundle_accepted_map_frame,
    _scope_series_audit_to_eligible,
)
from lol_kills.research.future_value_series_authority import (
    _crosswalk_summary,
    canonical_sha256,
)
from lol_kills.research.future_value_training import (
    FutureValueTrainingError,
    _accepted_map_frame as training_accepted_map_frame,
    _phase_reference_game_ids,
)
from lol_kills.v2.tierlists.accepted_census import identity_sha256


def _source(ids: tuple[str, ...]) -> dict[str, object]:
    payload: dict[str, object] = {
        "accepted_game_ids": list(ids),
        "source_identity_sha256": identity_sha256(ids),
        "source_game_count": len(ids),
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def _maps(*, include_extras: bool) -> pd.DataFrame:
    ids = ["g1", "g2"]
    if include_extras:
        ids.extend(f"extra-{index}" for index in range(1, 7))
    return pd.DataFrame(
        {
            "game_uid": ids,
            "date": pd.date_range("2026-01-01", periods=len(ids), tz="UTC"),
            "series_id": ["series-a", "series-b", *(["series-a"] * 6 if include_extras else [])],
        }
    )


def test_six_source_extras_cannot_change_series_audit_or_reference_partition() -> None:
    source = _source(("g1", "g2"))
    base = _maps(include_extras=False)
    expanded = _maps(include_extras=True)

    selected_training = training_accepted_map_frame(expanded, source_receipt=source)
    selected_bundle = bundle_accepted_map_frame(expanded, source_receipt=source)
    assert selected_training["game_uid"].tolist() == base["game_uid"].tolist()
    assert selected_bundle["game_uid"].tolist() == base["game_uid"].tolist()

    def audit(frame: pd.DataFrame) -> dict[str, object]:
        model_frame = frame.rename(columns={"game_uid": "game_id"})
        return _scope_series_audit_to_eligible(
            model_frame,
            base_audit={"source": "accepted", "map_count": len(model_frame)},
            assignments=tuple(
                {"oe_game_id": value, "series_id": series}
                for value, series in zip(
                    model_frame["game_id"], model_frame["series_id"]
                )
            ),
        )

    assert audit(selected_training) == audit(base)
    assert _phase_reference_game_ids(
        {
            **source,
            "source_extra_game_ids": {
                "maps": [f"extra-{index}" for index in range(1, 7)]
            },
        }
    ) == ("g1", "g2")


def test_accepted_map_scope_rejects_missing_and_duplicate_ids() -> None:
    source = _source(("g1", "g2"))
    missing = _maps(include_extras=False).iloc[:1]
    duplicate = pd.concat([_maps(include_extras=False), _maps(include_extras=False).iloc[[0]]])

    for helper in (training_accepted_map_frame, bundle_accepted_map_frame):
        error_type = (
            FutureValueTrainingError
            if helper is training_accepted_map_frame
            else FourVariantBundleError
        )
        with pytest.raises(error_type, match="missing"):
            helper(missing, source_receipt=source)
        with pytest.raises(error_type, match="duplicate"):
            helper(duplicate, source_receipt=source)


def _crosswalk_fixture(*, artifact_authoritative: bool, receipt_authoritative: bool):
    ids = ("g1", "g2")
    source = _source(ids)
    artifact: dict[str, object] = {
        "status": "complete_authoritative_coverage",
        "authority": {
            "research_only": True,
            "authoritative_series": artifact_authoritative,
            "public": False,
            "promotion": False,
            "deployment": False,
        },
        "source_binding": {
            "receipt_sha256": source["receipt_sha256"],
            "accepted_game_ids": list(ids),
            "selected_game_ids": list(ids),
            "selected_is_full_accepted_census": True,
        },
        "coverage": {
            "mapped_is_full_accepted_census": True,
            "mapped_game_count": len(ids),
        },
        "assignments": [
            {"oe_game_id": game_id, "outcome_used": False}
            for game_id in ids
        ],
    }
    artifact["crosswalk_sha256"] = canonical_sha256(
        {key: value for key, value in artifact.items() if key != "crosswalk_sha256"}
    )
    artifact_record = {"bytes": 1, "locator": "fixture/crosswalk.json", "sha256": "a" * 64}
    receipt: dict[str, object] = {
        "status": "verified_research_only",
        "authority": {
            "research_only": True,
            "authoritative_series": receipt_authoritative,
            "public": False,
            "promotion": False,
            "deployment": False,
        },
        "source_receipt_sha256": source["receipt_sha256"],
        "source_identity_sha256": source["source_identity_sha256"],
        "accepted_game_count": len(ids),
        "crosswalk_sha256": artifact["crosswalk_sha256"],
        "mapped_game_ids": list(ids),
        "mapped_game_identity_sha256": identity_sha256(ids),
        "artifact": artifact_record,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return source, artifact, receipt, artifact_record, {"bytes": 1, "sha256": "b" * 64}


def test_full_authority_requires_both_authoritative_series_flags() -> None:
    for artifact_authoritative, receipt_authoritative, expected in (
        (False, True, False),
        (True, False, False),
        (True, True, True),
    ):
        source, artifact, receipt, artifact_file, receipt_file = _crosswalk_fixture(
            artifact_authoritative=artifact_authoritative,
            receipt_authoritative=receipt_authoritative,
        )
        summary = _crosswalk_summary(
            artifact,
            receipt,
            source_receipt=source,
            accepted_ids=("g1", "g2"),
            crosswalk_artifact_file=artifact_file,
            crosswalk_receipt_file=receipt_file,
        )
        assert summary["authoritative_for_accepted_census"] is expected


def test_full_authority_keeps_receipt_public_flags_fail_closed() -> None:
    source, artifact, receipt, artifact_file, receipt_file = _crosswalk_fixture(
        artifact_authoritative=True,
        receipt_authoritative=True,
    )
    authority = dict(receipt["authority"])
    authority["public"] = True
    receipt["authority"] = authority
    receipt["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    summary = _crosswalk_summary(
        artifact,
        receipt,
        source_receipt=source,
        accepted_ids=("g1", "g2"),
        crosswalk_artifact_file=artifact_file,
        crosswalk_receipt_file=receipt_file,
    )
    assert summary["receipt_authority_safe"] is False
    assert summary["authoritative_for_accepted_census"] is False
