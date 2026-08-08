from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from lol_kills.v2.draft.terminal.candidate_registry_v3 import (
    CandidateRegistryV3Error,
    DEFAULT_OUTPUT as REGISTRY_PATH,
    _clock as registry_clock,
    validate_candidate_registry_v3,
)
from lol_kills.v2.draft.terminal.development_artifact_v3 import (
    DEFAULT_ARTIFACT,
    DEFAULT_FIXTURE,
)
from lol_kills.v2.draft.terminal.development_evaluation_v3 import (
    DEFAULT_SUMMARY,
    evaluate,
)
from lol_kills.v2.draft.terminal.development_snapshot import (
    DEFAULT_MANIFEST,
    DEFAULT_PAYLOAD,
    DevelopmentSnapshotError,
    load_development_snapshot,
)
from lol_kills.v2.draft.terminal.future_protocol_registry_v1 import (
    validate_registered_future_protocol_v1,
)
from lol_kills.v2.draft.terminal.model import (
    TerminalDraft,
    TerminalModel,
    score_terminal_draft,
)
from lol_kills.v2.ratings.player.multileague_v3_future_protocol import (
    FUTURE_SEALED_START,
)


ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture(scope="module")
def evaluation() -> dict[str, object]:
    return evaluate(ROOT)


def test_frozen_development_snapshot_is_clustered_and_hash_bound() -> None:
    rows, metadata = load_development_snapshot(ROOT)
    assert len(rows) == 6194
    assert len({row.dependence_cluster_id for row in rows}) == 2871
    assert all(
        not row.dependence_cluster_id.startswith("unclustered-game:")
        for row in rows
    )
    assert metadata["development_snapshot_payload_raw_sha256"] == hashlib.sha256(
        (ROOT / DEFAULT_PAYLOAD).read_bytes()
    ).hexdigest()
    assert metadata["availability_status"] == "frozen_retrospective_development_only"


def test_frozen_development_snapshot_rejects_payload_tampering(
    tmp_path: Path,
) -> None:
    manifest_target = tmp_path / DEFAULT_MANIFEST
    payload_target = tmp_path / DEFAULT_PAYLOAD
    manifest_target.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / DEFAULT_MANIFEST, manifest_target)
    shutil.copyfile(ROOT / DEFAULT_PAYLOAD, payload_target)
    raw = payload_target.read_bytes()
    payload_target.write_bytes(raw[:-2] + b"0\n")
    with pytest.raises(
        DevelopmentSnapshotError, match="payload hash does not match"
    ):
        load_development_snapshot(tmp_path)


def test_v3_evaluation_binds_incremental_estimand_and_current_summary(
    evaluation: dict[str, object],
) -> None:
    summary = json.loads((ROOT / DEFAULT_SUMMARY).read_text(encoding="utf-8"))
    serialized = (
        json.dumps(evaluation, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")
    assert summary["run_output_sha256"] == hashlib.sha256(serialized).hexdigest()
    assert evaluation["estimands"]["neutral_probability_calibration_directly_identified"] is False
    selected = evaluation["development_candidate_for_future_freeze"]
    assert selected["variant_id"] == "m0-role-additive@ridge-0.05"
    assert selected["all_validation_folds_nonharmful"] is True
    outer = summary["fold_locked_selected_test"]
    assert sum(
        row["locked_outer_test_incremental_vs_baseline_only"]["passed"]
        for row in outer
    ) == 2
    assert len(outer) == 3
    assert summary["production_eligible"] is False
    assert summary["public_probability_authorized"] is False


def test_v3_model_replays_as_equal_strength_development_index() -> None:
    artifact_raw = (ROOT / DEFAULT_ARTIFACT).read_bytes()
    fixture = json.loads((ROOT / DEFAULT_FIXTURE).read_text(encoding="utf-8"))
    model = TerminalModel.from_artifact_bytes(
        artifact_raw,
        expected_artifact_sha256=hashlib.sha256(artifact_raw).hexdigest(),
    )
    raw = fixture["draft"]
    draft = TerminalDraft.from_sides(
        raw["side_a"],
        raw["side_b"],
        event_start=raw["event_start"],
        source_available_at=raw["source_available_at"],
        source_record_id=raw["source_record_id"],
        source_payload_sha256=raw["source_payload_sha256"],
        source_rights_status=raw["source_rights_status"],
        mode=raw.get("mode", "neutral"),
        actions=raw.get("actions"),
        final_assignments=raw.get("final_assignments"),
    )
    assert score_terminal_draft(draft, model, development=True) == fixture[
        "expected_development"
    ]
    assert fixture["probability_semantics"] == (
        "equal_strength_composition_index_not_directly_outcome_calibrated"
    )


def test_candidate_registry_and_future_protocol_are_locked_but_empty() -> None:
    registry_raw = (ROOT / REGISTRY_PATH).read_bytes()
    registry = validate_candidate_registry_v3(json.loads(registry_raw), root=ROOT)
    assert registry["selected_candidate"]["variant_id"] == (
        "m0-role-additive@ridge-0.05"
    )
    assert registry["authority"]["betting_authority"] is False
    protocol = validate_registered_future_protocol_v1(root=ROOT)
    assert protocol["future_holdout"]["status"] == "EMPTY_NOT_YET_ACQUIRED"
    assert protocol["capture_state"]["eligible_entries"] == 0
    assert protocol["estimands"]["neutral_output_directly_outcome_calibrated"] is False
    assert not any(protocol["authority"].values())


def test_candidate_registry_clock_rejects_the_future_boundary() -> None:
    with pytest.raises(CandidateRegistryV3Error, match="before the future boundary"):
        registry_clock(
            lambda: FUTURE_SEALED_START.replace(tzinfo=timezone.utc)
        )

