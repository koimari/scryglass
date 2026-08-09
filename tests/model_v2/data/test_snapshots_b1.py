from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from lol_kills.v2.provenance.snapshots import (
    ChampionPatchRoleAppearanceRow,
    LineageReport,
    SourceSnapshot,
    SourceSnapshotError,
    SourceSnapshotRow,
    SourceSnapshotRowSummary,
    SourceTreeMismatchError,
    TrainingSnapshot,
    TrainingSnapshotError,
    _reject_forbidden_recursive,
    _validate_forbidden_filters,
    make_training_snapshot,
)
from lol_kills.v2.data.common import sha256_canonical_object_hash, sha256_raw_bytes_hash
from lol_kills.v2.data.source_tree import canonical_source_tree_sha256

BASE_DIR = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "lol"
    / "v2"
    / "snapshots"
    / "b1"
)
SOURCE_SNAPSHOT_PATH = BASE_DIR / "source-snapshot-passb1.json"
TRAINING_SNAPSHOT_PATH = BASE_DIR / "training-snapshot-passb1.json"
COMPACT_CAMEL_FORBIDDEN_ALIASES = (
    "wagerOutput",
    "wageringOutput",
    "betOutput",
    "bettingOutput",
    "totalKills",
    "underOver",
    "overUnder",
    "totalkills",
    "underover",
    "overunder",
)



def _source_payload() -> dict:
    return json.loads(SOURCE_SNAPSHOT_PATH.read_text(encoding="utf-8"))



def _training_payload() -> dict:
    return json.loads(TRAINING_SNAPSHOT_PATH.read_text(encoding="utf-8"))



def _decode_snapshot_pair_list(pairs: list[object] | tuple[object, ...]) -> tuple[tuple[str, str], ...]:
    decoded: list[tuple[str, str]] = []
    for pair in pairs:
        if isinstance(pair, (list, tuple)):
            source_id, source_hash = pair
        else:
            source_id = pair["source_snapshot_id"]
            source_hash = pair.get("source_snapshot_content_sha256")
            if source_hash is None:
                source_hash = pair["source_snapshot_sha256"]
        decoded.append((source_id, source_hash))
    return tuple(decoded)



def _source_snapshot_from_payload(payload: dict) -> SourceSnapshot:
    snapshot_id = payload["snapshot_id"]
    row_snapshot_id = snapshot_id or "scryglass:source-snapshot:pending"
    rows_payload = [dict(row) for row in payload["rows"]]
    for row in rows_payload:
        if "source_snapshot_id" not in row:
            row["source_snapshot_id"] = row_snapshot_id
    rows = tuple(SourceSnapshotRow(**row) for row in rows_payload)
    row_lookup = {row.source_content_sha256: row.source_snapshot_row_id for row in rows}

    champion_payload = []
    for row in payload["champion_patch_role_counts"]:
        row_dict = dict(row)
        if "source_snapshot_id" not in row_dict:
            row_dict["source_snapshot_id"] = row_snapshot_id
        if "source_snapshot_row_id" not in row_dict:
            row_dict["source_snapshot_row_id"] = row_lookup[row_dict["source_snapshot_content_sha256"]]
        champion_payload.append(row_dict)

    return SourceSnapshot(
        schema_version=payload["schema_version"],
        model_version=payload["model_version"],
        adapter_version=payload["adapter_version"],
        code_version=payload["code_version"],
        as_of=payload["as_of"],
        snapshot_id=payload["snapshot_id"],
        reviewed_at=payload["reviewed_at"],
        rows=rows,
        source_tree_sha256=payload["source_tree_sha256"],
        source_tree_allowlist=tuple(payload["source_tree_allowlist"]),
        created_at=payload["created_at"],
        contract_tree_sha256=payload["contract_tree_sha256"],
        champion_patch_role_counts=tuple(
            ChampionPatchRoleAppearanceRow(**row) for row in champion_payload
        ),
        status=payload["status"],
    )



def _training_snapshot_from_payload(payload: dict) -> TrainingSnapshot:
    return TrainingSnapshot(
        schema_version=payload["schema_version"],
        model_version=payload["model_version"],
        adapter_version=payload["adapter_version"],
        code_version=payload["code_version"],
        as_of=payload["as_of"],
        train_cutoff=payload["train_cutoff"],
        source_manifest_locator=payload["source_manifest_locator"],
        source_manifest_object_sha256=payload["source_manifest_object_sha256"],
        source_snapshot_pairs=tuple(
            (pair["source_snapshot_id"], pair["source_snapshot_sha256"])
            for pair in payload["source_snapshot_pairs"]
        ),
        source_tree_sha256=payload["source_tree_sha256"],
        source_tree_allowlist=tuple(payload["source_tree_allowlist"]),
        row_count_evidence_locator=payload["row_count_evidence_locator"],
        row_count_evidence_sha256=payload["row_count_evidence_sha256"],
        row_count_by_year=dict(payload["row_count_by_year"]),
        row_count_by_league=dict(payload["row_count_by_league"]),
        row_count_by_tier=dict(payload["row_count_by_tier"]),
        row_count_by_patch=dict(payload["row_count_by_patch"]),
        row_count_by_source=dict(payload["row_count_by_source"]),
        source_rows=tuple(SourceSnapshotRowSummary(**row) for row in payload["source_rows"]),
        created_at=payload["created_at"],
        taxonomy_version=payload["taxonomy_version"],
        crosswalk_version=payload["crosswalk_version"],
        inclusion_filters=tuple(payload["inclusion_filters"]),
        exclusion_filters=tuple(payload["exclusion_filters"]),
        min_event_at=payload["min_event_at"],
        max_event_at=payload["max_event_at"],
        min_available_at=payload["min_available_at"],
        max_available_at=payload["max_available_at"],
        duplicate_count=payload["duplicate_count"],
        correction_count=payload["correction_count"],
        missingness_count=payload["missingness_count"],
        conflict_count=payload["conflict_count"],
        identity_audit_count=payload["identity_audit_count"],
        split_assignment_ids=tuple(payload["split_assignment_ids"]),
        split_assignment_locators=tuple(payload["split_assignment_locators"]),
        split_assignment_sha256s=tuple(payload["split_assignment_sha256s"]),
        environment_lock_sha256=payload.get("environment_lock_sha256"),
        environment_lock_locator=payload.get("environment_lock_locator"),
        candidate_code_commit=payload["candidate_code_commit"],
        code_commit=payload["code_commit"],
        supersession_lines=tuple(payload["supersession_lines"]),
        correction_lines=tuple(payload["correction_lines"]),
        contract_tree_sha256=payload["contract_tree_sha256"],
        row_count=payload["row_count"],
        require_no_stale_required=payload["require_no_stale_required"],
        required_source_snapshot_pairs=_decode_snapshot_pair_list(
            payload["required_source_snapshot_pairs"]
        ),
        optional_source_snapshot_pairs=_decode_snapshot_pair_list(
            payload["optional_source_snapshot_pairs"]
        ),
        status=payload["status"],
        snapshot_id=payload["snapshot_id"],
    )



def _lineage_from_training(
    source_snapshot: SourceSnapshot,
    training_snapshot: TrainingSnapshot,
    *,
    freshness_rows: tuple[SourceSnapshotRowSummary, ...] | None = None,
    status: str = "ok",
    source_tree_match: bool = True,
    missing_required_sources: tuple[str, ...] = (),
    completeness_issues: tuple[str, ...] = (),
    duplicate_count: int = 0,
    correction_count: int = 0,
    missingness_count: int = 0,
    conflict_count: int = 0,
) -> LineageReport:
    return LineageReport(
        manifest_id="lineage-passb1",
        model_version=source_snapshot.model_version,
        source_snapshot_id=source_snapshot.snapshot_id,
        training_snapshot_id=training_snapshot.snapshot_id,
        source_manifest_locator=training_snapshot.source_manifest_locator,
        source_manifest_object_sha256=training_snapshot.source_manifest_object_sha256,
        training_manifest_locator="data/lol/v2/snapshots/b1/training-snapshot-passb1.json",
        training_manifest_object_sha256=training_snapshot.sha256(),
        source_snapshot_tree_sha256=source_snapshot.source_tree_sha256,
        training_snapshot_tree_sha256=source_snapshot.source_tree_sha256,
        source_tree_sha256=source_snapshot.source_tree_sha256,
        source_tree_allowlist=source_snapshot.source_tree_allowlist,
        as_of=training_snapshot.as_of,
        generated_at=training_snapshot.as_of,
        source_snapshot_ids=training_snapshot.source_snapshot_ids,
        source_snapshot_hashes=tuple(hash_ for _, hash_ in training_snapshot.source_snapshot_pairs),
        source_snapshot_pairs=training_snapshot.source_snapshot_pairs,
        source_snapshot_row_count=source_snapshot.row_count,
        training_row_count=training_snapshot.row_count,
        freshness_report=tuple(freshness_rows or tuple()),
        artifact_manifest_id=None,
        source_tree_match=source_tree_match,
        require_tree_match=True,
        duplicate_count=duplicate_count,
        correction_count=correction_count,
        missingness_count=missingness_count,
        conflict_count=conflict_count,
        missing_required_sources=missing_required_sources,
        required_source_snapshot_pairs=training_snapshot.required_source_snapshot_pairs,
        optional_source_snapshot_pairs=training_snapshot.optional_source_snapshot_pairs,
        map_appearance_evidence=source_snapshot.champion_patch_role_counts,
        status=status,
        completeness_issues=completeness_issues,
    )



def test_b1_source_snapshot_round_trip_is_deterministic(tmp_path: Path) -> None:
    snapshot = _source_snapshot_from_payload(_source_payload())
    out_path = tmp_path / "source-snapshot.json"

    snapshot.write(out_path)
    restored = _source_snapshot_from_payload(json.loads(out_path.read_text(encoding="utf-8")))

    assert restored.snapshot_id == snapshot.snapshot_id
    assert restored.sha256() == snapshot.sha256()



def test_b1_source_snapshot_order_canonicalization_still_determines_same_id() -> None:
    base_payload = _source_payload()
    baseline = _source_snapshot_from_payload(base_payload)

    shuffled = deepcopy(base_payload)
    shuffled["rows"] = list(reversed(shuffled["rows"]))
    shuffled["champion_snapshot_counts"] = list(
        reversed(shuffled["champion_patch_role_counts"])
    )
    shuffled["champion_patch_role_counts"] = list(
        reversed(shuffled["champion_patch_role_counts"])
    )

    reordered = _source_snapshot_from_payload(shuffled)
    assert reordered.snapshot_id == baseline.snapshot_id
    assert reordered.sha256() == baseline.sha256()



def test_b1_source_snapshot_rejects_content_hash_mutation() -> None:
    payload = _source_payload()
    payload["rows"][0]["source_content_sha256"] = "0" * 64

    with pytest.raises(SourceSnapshotError, match="source_content_sha256 mismatch"):
        _source_snapshot_from_payload(payload)



def test_b1_source_snapshot_rejects_snapshot_id_mutation() -> None:
    payload = _source_payload()
    payload["snapshot_id"] = "scryglass:source-snapshot:tampered"

    with pytest.raises(SourceSnapshotError, match="snapshot_id must be derived"):
        _source_snapshot_from_payload(payload)



def test_b1_source_snapshot_rejects_missing_source_bytes() -> None:
    payload = _source_payload()
    payload["rows"][0]["source_content_path"] = "data/lol/v2/snapshots/b1/missing-passb1.json"

    with pytest.raises(SourceSnapshotError, match="source content path must reference an existing file"):
        _source_snapshot_from_payload(payload)



def test_b1_source_snapshot_rejects_source_path_escape_and_backslash() -> None:
    payload = _source_payload()

    payload["rows"][0]["source_content_path"] = "/tmp/system-passb1.json"
    with pytest.raises(SourceSnapshotError, match="must be relative"):
        _source_snapshot_from_payload(payload)

    payload = _source_payload()
    payload["rows"][0]["source_content_path"] = ".."
    with pytest.raises(SourceSnapshotError):
        _source_snapshot_from_payload(payload)

    payload = _source_payload()
    payload["rows"][0]["source_content_path"] = "data\\lol\\v2\\snapshots\\b1\\source-passb1.json"
    with pytest.raises(SourceSnapshotError):
        _source_snapshot_from_payload(payload)



def test_b1_source_snapshot_rejects_terminal_symlink_source_path() -> None:
    link = BASE_DIR / "source-passb1-link.json"
    if link.exists():
        link.unlink()

    try:
        link.symlink_to(BASE_DIR / "source-passb1.json")
        payload = _source_payload()
        payload["rows"][0]["source_content_path"] = "data/lol/v2/snapshots/b1/source-passb1-link.json"

        with pytest.raises(SourceSnapshotError, match="cannot include symlinks"):
            _source_snapshot_from_payload(payload)
    finally:
        if link.exists():
            link.unlink()



def test_b1_source_snapshot_rejects_intermediate_symlink_source_path() -> None:
    link_dir = BASE_DIR / "passb1-symlink-dir"
    if link_dir.exists():
        link_dir.unlink()

    try:
        link_dir.symlink_to(BASE_DIR)
        payload = _source_payload()
        payload["rows"][0]["source_content_path"] = "data/lol/v2/snapshots/b1/passb1-symlink-dir/source-passb1.json"

        with pytest.raises(SourceSnapshotError, match="cannot include symlinks"):
            _source_snapshot_from_payload(payload)
    finally:
        if link_dir.exists():
            link_dir.unlink()



def test_b1_training_snapshot_round_trip_is_deterministic(tmp_path: Path) -> None:
    snapshot = _training_snapshot_from_payload(_training_payload())
    out_path = tmp_path / "training-snapshot.json"

    snapshot.write(out_path)
    restored = _training_snapshot_from_payload(json.loads(out_path.read_text(encoding="utf-8")))

    assert restored.snapshot_id == snapshot.snapshot_id
    assert restored.sha256() == snapshot.sha256()



def test_b1_training_snapshot_order_canonicalization_keeps_id_stable() -> None:
    base_payload = _training_payload()
    baseline = _training_snapshot_from_payload(base_payload)

    shuffled = deepcopy(base_payload)
    shuffled["source_rows"] = list(reversed(shuffled["source_rows"]))
    shuffled["source_snapshot_pairs"] = list(reversed(shuffled["source_snapshot_pairs"]))

    rerun = _training_snapshot_from_payload(shuffled)
    assert rerun.snapshot_id == baseline.snapshot_id
    assert rerun.sha256() == baseline.sha256()



def test_b1_training_snapshot_rejects_required_source_missing_then_stale_and_conflict() -> None:
    base_payload = _training_payload()

    missing = deepcopy(base_payload)
    missing["required_source_snapshot_pairs"] = [
        {
            "source_snapshot_id": "scryglass:source-snapshot:missing",
            "source_snapshot_sha256": "0" * 64,
        }
    ]
    with pytest.raises(TrainingSnapshotError, match="must reference declared source_snapshot_pairs"):
        _training_snapshot_from_payload(missing)

    stale = deepcopy(base_payload)
    stale["source_rows"][0]["stale_count"] = 3
    with pytest.raises(TrainingSnapshotError, match="must exactly match summaries"):
        _training_snapshot_from_payload(stale)

    conflict = deepcopy(base_payload)
    conflict["conflict_count"] = 1
    with pytest.raises(TrainingSnapshotError, match="cannot include conflict_count"):
        _training_snapshot_from_payload(conflict)



def test_b1_training_snapshot_rejects_status_ok_without_bounds() -> None:
    payload = _training_payload()
    payload["min_event_at"] = None
    payload["max_event_at"] = None

    with pytest.raises(TrainingSnapshotError, match="requires min/max event bounds"):
        _training_snapshot_from_payload(payload)



def test_b1_training_snapshot_rejects_status_ok_with_bound_exceeds_train_cutoff() -> None:
    payload = _training_payload()
    payload["max_event_at"] = "2026-07-30T00:00:00Z"
    with pytest.raises(TrainingSnapshotError, match="max_event_at must be <= train_cutoff"):
        _training_snapshot_from_payload(payload)



def test_b1_training_snapshot_rejects_offseted_event_bounds_out_of_order() -> None:
    payload = _training_payload()
    payload["train_cutoff"] = "2026-07-21T00:00:00Z"
    payload["min_event_at"] = "2026-07-20T00:00:00-05:00"
    payload["max_event_at"] = "2026-07-20T01:00:00+05:00"

    with pytest.raises(TrainingSnapshotError, match="min_event_at cannot be later than max_event_at"):
        _training_snapshot_from_payload(payload)



def test_b1_training_snapshot_rejects_count_reconciliation_mismatch() -> None:
    mismatch = _training_payload()
    mismatch["row_count_by_year"]["2026"] += 1

    with pytest.raises(TrainingSnapshotError, match="hashed row-membership evidence"):
        _training_snapshot_from_payload(mismatch)



def test_b1_training_snapshot_rejects_incomplete_source_rows_count_reconciliation() -> None:
    mismatch = _training_payload()
    mismatch["source_rows"][0]["row_count"] = 0

    with pytest.raises(TrainingSnapshotError, match="must exactly match summaries"):
        _training_snapshot_from_payload(mismatch)



def test_b1_training_snapshot_rejects_manifest_hash_in_row_content_hash() -> None:
    payload = _training_payload()
    manifest_hash = payload["source_snapshot_pairs"][0]["source_snapshot_sha256"]
    payload["source_rows"][0]["source_snapshot_content_sha256"] = manifest_hash

    with pytest.raises(TrainingSnapshotError, match="must not use source snapshot manifest hash"):
        _training_snapshot_from_payload(payload)



def test_b1_training_snapshot_rejects_forbidden_filters() -> None:
    forbidden = _training_payload()
    forbidden["inclusion_filters"] = ["grubs_24_percent"]

    with pytest.raises(TrainingSnapshotError, match="contains forbidden field"):
        _training_snapshot_from_payload(forbidden)

    forbidden = _training_payload()
    forbidden["exclusion_filters"] = ["market_total_kills"]

    with pytest.raises(TrainingSnapshotError, match="contains forbidden field"):
        _training_snapshot_from_payload(forbidden)



def test_b1_training_snapshot_rejects_fake_commit_hashes() -> None:
    fake = "a1" * 20
    payload = _training_payload()
    payload["code_commit"] = fake

    with pytest.raises(TrainingSnapshotError, match="commit hash cannot be resolved"):
        _training_snapshot_from_payload(payload)

    payload = _training_payload()
    payload["candidate_code_commit"] = fake

    with pytest.raises(TrainingSnapshotError, match="commit hash cannot be resolved"):
        _training_snapshot_from_payload(payload)



def test_b1_training_snapshot_rejects_environment_lock_violations() -> None:
    payload = _training_payload()
    payload["environment_lock_locator"] = "data/lol/v2/snapshots/b1/missing-lock-passb1.txt"
    payload["environment_lock_sha256"] = payload["environment_lock_sha256"]

    with pytest.raises(TrainingSnapshotError, match="environment_lock_locator must reference an existing file"):
        _training_snapshot_from_payload(payload)

    payload = _training_payload()
    payload["environment_lock_locator"] = "data/lol/v2/snapshots/b1/source-passb1.json"
    payload["environment_lock_sha256"] = payload["source_snapshot_pairs"][0]["source_snapshot_sha256"]

    with pytest.raises(TrainingSnapshotError, match="cannot reuse source snapshot manifest hash"):
        _training_snapshot_from_payload(payload)



def test_b1_source_and_training_snapshot_rejects_contract_hash_mismatch() -> None:
    source_payload = _source_payload()
    source_payload["contract_tree_sha256"] = "0" * 64

    with pytest.raises(SourceSnapshotError, match="contract_tree_sha256 mismatch"):
        _source_snapshot_from_payload(source_payload)

    training_payload = _training_payload()
    training_payload["contract_tree_sha256"] = "0" * 64

    with pytest.raises(TrainingSnapshotError, match="contract_tree_sha256 mismatch"):
        _training_snapshot_from_payload(training_payload)



def test_b1_lineage_report_requires_fresh_complete_rows_when_ok() -> None:
    source = _source_snapshot_from_payload(_source_payload())
    training = _training_snapshot_from_payload(_training_payload())

    rows = tuple(
        SourceSnapshotRowSummary(
            source_id=row.source_id,
            source_name=row.source_name,
            row_count=row.row_count,
            stale_count=row.stale_count,
            latest_available_at=row.latest_available_at,
            source_snapshot_id=row.source_snapshot_id,
            source_snapshot_content_sha256=row.source_snapshot_content_sha256,
        )
        for row in training.source_rows
    )

    good = _lineage_from_training(source, training, freshness_rows=rows, status="ok")
    assert good.status == "ok"
    assert good.source_tree_match is True

    stale_rows = (
        SourceSnapshotRowSummary(
            source_id=rows[0].source_id,
            source_name=rows[0].source_name,
            row_count=rows[0].row_count,
            stale_count=1,
            latest_available_at=rows[0].latest_available_at,
            source_snapshot_id=rows[0].source_snapshot_id,
            source_snapshot_content_sha256=rows[0].source_snapshot_content_sha256,
        ),
    )

    with pytest.raises(SourceSnapshotError, match="freshness rows to be complete"):
        _lineage_from_training(source, training, freshness_rows=stale_rows, status="ok")



def test_b1_lineage_report_rejects_incomplete_pairs_and_tree_mismatch() -> None:
    source = _source_snapshot_from_payload(_source_payload())
    training = _training_snapshot_from_payload(_training_payload())

    rows = tuple(
        SourceSnapshotRowSummary(
            source_id=row.source_id,
            source_name=row.source_name,
            row_count=row.row_count,
            stale_count=row.stale_count,
            latest_available_at=row.latest_available_at,
            source_snapshot_id=row.source_snapshot_id,
            source_snapshot_content_sha256=row.source_snapshot_content_sha256,
        )
        for row in training.source_rows
    )

    with pytest.raises(SourceSnapshotError, match="source_tree_match is required"):
        _lineage_from_training(
            source,
            training,
            freshness_rows=rows,
            status="ok",
            source_tree_match=False,
        )

    payload = {
        "source_snapshot_id": "scryglass:source-snapshot:missing",
        "source_snapshot_sha": "0" * 64,
    }
    with pytest.raises(SourceSnapshotError, match="must reference declared"):
        _lineage_from_training(source, training, freshness_rows=rows, status="ok", missing_required_sources=(payload,))


def test_b1_training_resolves_exact_source_manifest_locator_id_and_object_hash() -> None:
    payload = _training_payload()

    nonexistent = deepcopy(payload)
    nonexistent["source_manifest_locator"] = (
        "data/lol/v2/snapshots/b1/missing-source-snapshot.json"
    )
    with pytest.raises(TrainingSnapshotError, match="source_manifest_locator"):
        _training_snapshot_from_payload(nonexistent)

    wrong_object = deepcopy(payload)
    wrong_object["source_manifest_object_sha256"] = "0" * 64
    with pytest.raises(TrainingSnapshotError, match="canonical manifest object"):
        _training_snapshot_from_payload(wrong_object)

    wrong_pair = deepcopy(payload)
    wrong_pair["source_snapshot_pairs"][0]["source_snapshot_id"] = (
        "scryglass:source-snapshot:arbitrary"
    )
    with pytest.raises(TrainingSnapshotError, match="resolved source manifest"):
        _training_snapshot_from_payload(wrong_pair)


def test_b1_training_rejects_missing_or_mutated_environment_and_split_evidence() -> None:
    payload = _training_payload()

    missing_environment = deepcopy(payload)
    missing_environment["environment_lock_locator"] = None
    missing_environment["environment_lock_sha256"] = None
    with pytest.raises(TrainingSnapshotError, match="requires environment lock"):
        _training_snapshot_from_payload(missing_environment)

    mutated_environment = deepcopy(payload)
    mutated_environment["environment_lock_sha256"] = "0" * 64
    with pytest.raises(TrainingSnapshotError, match="must match repository file"):
        _training_snapshot_from_payload(mutated_environment)

    missing_split = deepcopy(payload)
    missing_split["split_assignment_ids"] = []
    missing_split["split_assignment_locators"] = []
    missing_split["split_assignment_sha256s"] = []
    with pytest.raises(TrainingSnapshotError, match="nonempty split assignment evidence"):
        _training_snapshot_from_payload(missing_split)

    mutated_split = deepcopy(payload)
    mutated_split["split_assignment_sha256s"][0] = "0" * 64
    with pytest.raises(TrainingSnapshotError, match="must match repository file raw bytes"):
        _training_snapshot_from_payload(mutated_split)

    missing_split_locator = deepcopy(payload)
    missing_split_locator["split_assignment_locators"][0] = (
        "data/lol/v2/snapshots/b1/missing-split-assignment.json"
    )
    with pytest.raises(TrainingSnapshotError, match="split_assignment_locator"):
        _training_snapshot_from_payload(missing_split_locator)

    label_split = deepcopy(payload)
    label_split["split_assignment_ids"][0] = "split-passb1"
    with pytest.raises(TrainingSnapshotError, match="must be derived"):
        _training_snapshot_from_payload(label_split)


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    (
        ("row_count_by_year", {"2025": 1, "2026": 5}),
        ("row_count_by_league", {"LCK": 1, "LPL": 5}),
        ("row_count_by_tier", {"tier1": 5, "tier2": 1}),
        ("row_count_by_patch", {"26.13": 1, "26.14": 5}),
        (
            "row_count_by_source",
            {"source-json-b1": 3, "source-text-b1": 2, "source-lie": 1},
        ),
    ),
)
def test_b1_training_rejects_total_preserving_count_redistributions(
    field_name: str, replacement: dict[str, int]
) -> None:
    payload = _training_payload()
    payload[field_name] = replacement
    with pytest.raises(TrainingSnapshotError, match="hashed row-membership evidence"):
        _training_snapshot_from_payload(payload)


def test_b1_source_status_ok_requires_positive_finite_freshness_slo() -> None:
    payload = _source_payload()
    payload["snapshot_id"] = ""
    payload["rows"][0].pop("freshness_limit_seconds")
    with pytest.raises(SourceSnapshotError, match="freshness SLO"):
        _source_snapshot_from_payload(payload)

    payload = _source_payload()
    payload["snapshot_id"] = ""
    payload["rows"][0]["freshness_limit_seconds"] = 0
    with pytest.raises(SourceSnapshotError, match="positive and finite"):
        _source_snapshot_from_payload(payload)


@pytest.mark.parametrize(
    "forbidden",
    (
        "derived.feature.GRUBS-24-PERCENT.v2",
        "audit/prefix/MARKET.TOTAL.KILLS/suffix",
        "legacy-market_kills-channel",
    ),
)
def test_b1_forbidden_legacy_aliases_are_case_and_separator_insensitive(
    forbidden: str,
) -> None:
    payload = _training_payload()
    payload["correction_lines"] = [forbidden]
    payload["correction_count"] = 1
    with pytest.raises(TrainingSnapshotError, match="forbidden field"):
        _training_snapshot_from_payload(payload)


@pytest.mark.parametrize(
    "field_name",
    ("inclusion_filters", "exclusion_filters"),
)
@pytest.mark.parametrize(
    "forbidden",
    (
        "wager_total_kills",
        "betting-total-kills",
        "TOTAL.KILLS",
        "under_over_total_kills",
        "over-under-output",
        *COMPACT_CAMEL_FORBIDDEN_ALIASES,
    ),
)
def test_b1_non_betting_aliases_are_rejected_in_training_filters(
    field_name: str, forbidden: str
) -> None:
    payload = _training_payload()
    payload[field_name] = [forbidden]
    with pytest.raises(TrainingSnapshotError, match="forbidden field"):
        _training_snapshot_from_payload(payload)


@pytest.mark.parametrize(
    "benign",
    (
        "overall_damage",
        "coverage_window",
        "better_model",
        "betterOutput",
        "total_assists",
        "solo_kills",
        "over",
        "under",
        "understudy_output",
        "overlap_output",
    ),
)
def test_b1_forbidden_token_guard_preserves_benign_words(benign: str) -> None:
    assert _validate_forbidden_filters((benign,), "positive_control") == (benign,)

    payload = _training_payload()
    payload["snapshot_id"] = ""
    payload["inclusion_filters"] = [benign]
    assert _training_snapshot_from_payload(payload).inclusion_filters == (benign,)


@pytest.mark.parametrize(
    "forbidden",
    (
        "prefix.WAGER.suffix",
        "prefix.WAGERING.suffix",
        "nested/bet/output",
        "feature.total.output.kills",
        "feature/under/derived/over",
        "feature-over-derived-under",
        *COMPACT_CAMEL_FORBIDDEN_ALIASES,
    ),
)
def test_b1_recursive_nested_payload_scan_rejects_non_betting_aliases(
    forbidden: str,
) -> None:
    nested_payload = {
        "safe": [
            {"deeper": {"field_name": forbidden}},
        ]
    }
    with pytest.raises(SourceSnapshotError, match="forbidden field"):
        _reject_forbidden_recursive(nested_payload, "nested payload")


def test_b1_recursive_nested_payload_scan_accepts_benign_over_substrings() -> None:
    _reject_forbidden_recursive(
        {
            "coverage": [
                "overall_damage",
                {"overlap": "understudy_output"},
                {"model": "better_calibration"},
            ]
        },
        "nested payload",
    )


def test_b1_legal_utc_offsets_compare_as_instants() -> None:
    payload = _training_payload()
    payload["snapshot_id"] = ""
    payload["as_of"] = "2026-07-27T10:00:00-03:00"
    payload["created_at"] = "2026-07-27T15:00:00+02:00"
    payload["train_cutoff"] = "2026-07-19T21:00:00-03:00"
    payload["min_event_at"] = "2026-07-20T02:00:00+02:00"
    payload["max_event_at"] = "2026-07-19T19:00:00-05:00"
    payload["min_available_at"] = "2026-07-20T00:00:00Z"
    payload["max_available_at"] = "2026-07-19T21:00:00-03:00"

    snapshot = _training_snapshot_from_payload(payload)
    assert snapshot.snapshot_id.startswith("scryglass:training-snapshot:")


def test_b1_make_training_snapshot_requires_real_environment_and_parsed_bounds() -> None:
    source = _source_snapshot_from_payload(_source_payload())
    payload = _training_payload()
    kwargs = {
        "model_version": source.model_version,
        "as_of": payload["as_of"],
        "source_snapshot": source,
        "source_manifest_locator": payload["source_manifest_locator"],
        "row_count_evidence_locator": payload["row_count_evidence_locator"],
        "environment_lock_locator": payload["environment_lock_locator"],
        "environment_lock_sha256": payload["environment_lock_sha256"],
        "split_assignment_ids": tuple(payload["split_assignment_ids"]),
        "split_assignment_locators": tuple(payload["split_assignment_locators"]),
        "split_assignment_sha256s": tuple(payload["split_assignment_sha256s"]),
        "min_event_at": payload["min_event_at"],
        "max_event_at": payload["max_event_at"],
        "min_available_at": payload["min_available_at"],
        "max_available_at": payload["max_available_at"],
        "row_count_by_year": payload["row_count_by_year"],
        "row_count_by_league": payload["row_count_by_league"],
        "row_count_by_tier": payload["row_count_by_tier"],
        "row_count_by_patch": payload["row_count_by_patch"],
        "row_count_by_source": payload["row_count_by_source"],
        "source_tree_sha256": source.source_tree_sha256,
        "source_tree_allowlist": source.source_tree_allowlist,
        "train_cutoff": payload["train_cutoff"],
        "taxonomy_version": payload["taxonomy_version"],
        "crosswalk_version": payload["crosswalk_version"],
        "inclusion_filters": tuple(payload["inclusion_filters"]),
        "require_no_stale_required": True,
        "adapter_version": payload["adapter_version"],
        "code_version": payload["code_version"],
        "required_source_snapshot_pairs": (
            (source.snapshot_id, source.sha256()),
        ),
    }
    built = make_training_snapshot(**kwargs)
    assert built.environment_lock_locator == payload["environment_lock_locator"]

    kwargs["min_event_at"] = "not-rfc3339"
    with pytest.raises(TrainingSnapshotError, match="min_event_at"):
        make_training_snapshot(**kwargs)


def test_b1_appearance_uses_referenced_source_row_availability_ordering() -> None:
    payload = _source_payload()
    payload["snapshot_id"] = ""
    payload["champion_patch_role_counts"][0]["as_of"] = "2026-07-19T23:59:59Z"
    with pytest.raises(SourceSnapshotError, match="source row available_at"):
        _source_snapshot_from_payload(payload)


def test_b1_lineage_rejects_empty_extra_or_contradictory_redundant_evidence() -> None:
    source = _source_snapshot_from_payload(_source_payload())
    training = _training_snapshot_from_payload(_training_payload())
    good = _lineage_from_training(
        source,
        training,
        freshness_rows=training.source_rows,
        status="ok",
    )

    with pytest.raises(SourceSnapshotError, match="nonempty exact freshness"):
        replace(good, freshness_report=())

    extra = training.source_rows + (
        replace(training.source_rows[0], source_id="unexpected-source"),
    )
    with pytest.raises(SourceSnapshotError, match="with no extras"):
        replace(good, freshness_report=extra)

    with pytest.raises(SourceSnapshotError, match="canonical pairs"):
        replace(good, source_snapshot_hashes=("0" * 64,))

    with pytest.raises(SourceTreeMismatchError, match="every source-tree digest"):
        replace(good, training_snapshot_tree_sha256="0" * 64)

    with pytest.raises(SourceSnapshotError, match="referenced training manifest"):
        replace(good, training_snapshot_id="scryglass:training-snapshot:arbitrary")


def test_b1_golden_hashes_are_independently_recomputed() -> None:
    source_payload = _source_payload()
    training_payload = _training_payload()
    source = _source_snapshot_from_payload(source_payload)
    training = _training_snapshot_from_payload(training_payload)

    assert sha256_canonical_object_hash(source_payload) == (
        training_payload["source_manifest_object_sha256"]
    )
    assert sha256_canonical_object_hash(training_payload) == (
        "8f056c8100ccb9771779338879b05242938b91c5ad0017057a8eed8c570d25c2"
    )
    assert sha256_raw_bytes_hash(
        (BASE_DIR / "environment-lock-passb1.txt").read_bytes()
    ) == training_payload["environment_lock_sha256"]
    assert sha256_raw_bytes_hash(
        (BASE_DIR / "split-assignment-passb1.json").read_bytes()
    ) == training_payload["split_assignment_sha256s"][0]
    assert sha256_raw_bytes_hash(
        (BASE_DIR / "row-count-evidence-passb1.json").read_bytes()
    ) == training_payload["row_count_evidence_sha256"]
    assert training_payload["row_count_by_tier"] == {"tier1": 6}
    assert source.snapshot_id == (
        "scryglass:source-snapshot:"
        + sha256_canonical_object_hash(source._payload_for_id())
    )
    assert training.snapshot_id == (
        "scryglass:training-snapshot:"
        + sha256_canonical_object_hash(training._payload_for_id())
    )
    repo_root = Path(__file__).resolve().parents[3]
    assert canonical_source_tree_sha256(
        repo_root, source.source_tree_allowlist
    ) == source.source_tree_sha256
    for row in source.rows:
        raw_bytes = (repo_root / row.source_content_path).read_bytes()
        assert sha256_raw_bytes_hash(raw_bytes) == row.source_content_sha256
        if row.source_content_object_sha256 is not None:
            assert sha256_canonical_object_hash(
                json.loads(raw_bytes)
            ) == row.source_content_object_sha256
