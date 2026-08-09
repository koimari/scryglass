"""Lock the corrected terminal Draft Score candidate before future outcomes.

The registry records an adaptive development choice, not an approved model.
It binds exact source-frozen evaluation, model, report, and replay bytes and
samples its own UTC clock before the shared August 3 prospective boundary.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping

from lol_kills.v2.data.common import sha256_canonical_object
from lol_kills.v2.ratings.player.multileague_v3_future_protocol import (
    FUTURE_SEALED_START,
)

from .development_artifact_v3 import (
    DEFAULT_ARTIFACT,
    DEFAULT_FIXTURE,
    DEFAULT_REPORT,
    MODEL_VERSION,
    SELECTED_CANDIDATE_ID,
    SELECTED_RIDGE_STRENGTH,
)
from .development_evaluation_v3 import DEFAULT_SUMMARY, evaluate
from .development_snapshot import DEFAULT_MANIFEST, DEFAULT_PAYLOAD
from .model import TerminalDraft, TerminalModel, score_terminal_draft


SCHEMA_VERSION = "scryglass:draft-terminal-candidate-registry:v3"
RESULT_STATE = "ADAPTIVE_INCREMENTAL_CANDIDATE_LOCKED_FOR_FUTURE_EVALUATION"
DEFAULT_OUTPUT = Path(
    "data/lol/v2/models/draft-terminal/draft-terminal-candidate-registry-v3.json"
)
V2_ARTIFACT = Path(
    "data/lol/v2/models/draft-terminal/terminal-model-neutral-development-v2.json"
)
V2_SUMMARY = Path(
    "data/lol/v2/models/draft-terminal/development-evaluation-summary-v2.json"
)


class CandidateRegistryV3Error(ValueError):
    """Raised when corrected candidate lineage cannot be locked or replayed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clock(clock: Callable[[], datetime]) -> datetime:
    observed = clock()
    if not isinstance(observed, datetime) or observed.tzinfo is None:
        raise CandidateRegistryV3Error("registry clock must be timezone aware")
    observed = observed.astimezone(timezone.utc)
    if observed >= FUTURE_SEALED_START.replace(tzinfo=timezone.utc):
        raise CandidateRegistryV3Error(
            "candidate registry must be locked before the future boundary"
        )
    return observed


def _atomic_write_new(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise CandidateRegistryV3Error(f"refusing to overwrite registry: {path}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise CandidateRegistryV3Error(f"refusing to overwrite registry: {path}")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _validate_replay(root: Path, artifact_raw: bytes) -> str:
    fixture_path = root / DEFAULT_FIXTURE
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    if not isinstance(fixture, Mapping):
        raise CandidateRegistryV3Error("v3 replay fixture is malformed")
    artifact_sha256 = hashlib.sha256(artifact_raw).hexdigest()
    if fixture.get("model_artifact_sha256") != artifact_sha256:
        raise CandidateRegistryV3Error("v3 fixture model binding changed")
    draft_mapping = fixture.get("draft")
    if not isinstance(draft_mapping, Mapping):
        raise CandidateRegistryV3Error("v3 fixture draft is missing")
    draft = TerminalDraft.from_sides(
        draft_mapping["side_a"],
        draft_mapping["side_b"],
        event_start=draft_mapping["event_start"],
        source_available_at=draft_mapping["source_available_at"],
        source_record_id=draft_mapping["source_record_id"],
        source_payload_sha256=draft_mapping["source_payload_sha256"],
        source_rights_status=draft_mapping["source_rights_status"],
        mode=draft_mapping.get("mode", "neutral"),
        actions=draft_mapping.get("actions"),
        final_assignments=draft_mapping.get("final_assignments"),
    )
    model = TerminalModel.from_artifact_bytes(
        artifact_raw, expected_artifact_sha256=artifact_sha256
    )
    if score_terminal_draft(draft, model, development=True) != fixture.get(
        "expected_development"
    ):
        raise CandidateRegistryV3Error("v3 replay fixture does not reproduce")
    return artifact_sha256


def build_candidate_registry_v3(
    *,
    root: Path = Path("."),
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    observed = _clock(clock)
    summary_path = root / DEFAULT_SUMMARY
    summary_raw = summary_path.read_bytes()
    summary = json.loads(summary_raw)
    report = evaluate(root)
    report_raw = (
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")
    if summary.get("run_output_sha256") != hashlib.sha256(report_raw).hexdigest():
        raise CandidateRegistryV3Error("v3 summary no longer binds evaluation")
    candidate = summary.get("development_candidate_for_future_freeze")
    if not isinstance(candidate, Mapping):
        raise CandidateRegistryV3Error("v3 summary has no future candidate")
    if (
        candidate.get("candidate_id") != SELECTED_CANDIDATE_ID
        or candidate.get("ridge_strength") != SELECTED_RIDGE_STRENGTH
        or candidate.get("all_validation_folds_nonharmful") is not True
    ):
        raise CandidateRegistryV3Error("v3 candidate identity changed")
    artifact_path = root / DEFAULT_ARTIFACT
    artifact_raw = artifact_path.read_bytes()
    artifact_sha256 = _validate_replay(root, artifact_raw)
    model = TerminalModel.from_artifact_bytes(
        artifact_raw, expected_artifact_sha256=artifact_sha256
    )
    if model.model_version != MODEL_VERSION:
        raise CandidateRegistryV3Error("v3 model version changed")
    report_path = root / DEFAULT_REPORT
    artifact_report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        artifact_report.get("artifact_raw_sha256") != artifact_sha256
        or artifact_report.get("development_evaluation_summary_raw_sha256")
        != hashlib.sha256(summary_raw).hexdigest()
        or artifact_report.get("neutral_equal_strength_index_directly_outcome_calibrated")
        is not False
    ):
        raise CandidateRegistryV3Error("v3 artifact report lineage changed")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "result_state": RESULT_STATE,
        "locked_at_utc": observed.isoformat(),
        "clock_attestation": {
            "source": "system_utc_clock_sampled_inside_builder",
            "observed_wall_clock_utc": observed.isoformat(),
            "user_supplied_timestamp_allowed": False,
            "future_boundary_utc": FUTURE_SEALED_START.replace(
                tzinfo=timezone.utc
            ).isoformat(),
        },
        "selected_candidate": {
            "candidate_id": SELECTED_CANDIDATE_ID,
            "variant_id": candidate["variant_id"],
            "ridge_strength": SELECTED_RIDGE_STRENGTH,
            "model_version": MODEL_VERSION,
            "model_as_of": model.model_as_of,
            "artifact_locator": str(DEFAULT_ARTIFACT),
            "artifact_raw_sha256": artifact_sha256,
            "selection_target": "incremental_context_plus_draft_vs_same_input_context_only",
            "served_neutral_semantics": "equal_strength_composition_index_not_directly_outcome_calibrated",
            "development_validation_all_folds_nonharmful": True,
            "independent_validation": False,
        },
        "bindings": {
            "development_snapshot_manifest_locator": str(DEFAULT_MANIFEST),
            "development_snapshot_manifest_raw_sha256": _sha256(
                root / DEFAULT_MANIFEST
            ),
            "development_snapshot_payload_locator": str(DEFAULT_PAYLOAD),
            "development_snapshot_payload_raw_sha256": _sha256(root / DEFAULT_PAYLOAD),
            "development_evaluation_summary_locator": str(DEFAULT_SUMMARY),
            "development_evaluation_summary_raw_sha256": hashlib.sha256(
                summary_raw
            ).hexdigest(),
            "development_artifact_report_locator": str(DEFAULT_REPORT),
            "development_artifact_report_raw_sha256": _sha256(report_path),
            "replay_fixture_locator": str(DEFAULT_FIXTURE),
            "replay_fixture_raw_sha256": _sha256(root / DEFAULT_FIXTURE),
        },
        "development_evidence": {
            "mean_validation_log_loss_delta": candidate[
                "mean_validation_log_loss_delta"
            ],
            "mean_validation_brier_delta": candidate[
                "mean_validation_brier_delta"
            ],
            "validation_log_loss_deltas": candidate[
                "validation_log_loss_deltas"
            ],
            "validation_brier_deltas": candidate["validation_brier_deltas"],
            "outer_test_pass_count": sum(
                bool(item["locked_outer_test_incremental_vs_baseline_only"]["passed"])
                for item in summary["fold_locked_selected_test"]
            ),
            "outer_test_fold_count": len(summary["fold_locked_selected_test"]),
            "outer_test_is_candidate_selection_input": False,
            "development_evidence_is_independent": False,
        },
        "supersession": {
            "v2_artifact_locator": str(V2_ARTIFACT),
            "v2_artifact_raw_sha256": _sha256(root / V2_ARTIFACT),
            "v2_summary_locator": str(V2_SUMMARY),
            "v2_summary_raw_sha256": _sha256(root / V2_SUMMARY),
            "reason": "v2 ranked neutral composition logits directly against unequal-team outcomes",
            "v2_authority_rehabilitated": False,
        },
        "future_state": {
            "future_holdout_status": "EMPTY_NOT_YET_ACQUIRED",
            "future_outcomes_present": False,
            "future_outcomes_accessed": False,
            "prediction_capture_present": False,
            "independent_review_present": False,
        },
        "authority": {
            "model_validation_authority": False,
            "neutral_probability_authority": False,
            "contextual_probability_authority": False,
            "recommendation_authority": False,
            "betting_authority": False,
        },
        "claim_ceiling": {
            "adaptive_development_candidate": True,
            "equal_strength_composition_index": True,
            "outcome_calibrated_neutral_probability": False,
            "independent_validation": False,
            "production_probability": False,
            "recommendation": False,
            "betting": False,
        },
    }
    payload["artifact_sha256"] = sha256_canonical_object(payload)
    return validate_candidate_registry_v3(payload, root=root)


def validate_candidate_registry_v3(
    payload: Mapping[str, Any], *, root: Path = Path(".")
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise CandidateRegistryV3Error("candidate registry v3 must be an object")
    value = dict(payload)
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("result_state") != RESULT_STATE
    ):
        raise CandidateRegistryV3Error("candidate registry v3 identity changed")
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if value.get("artifact_sha256") != sha256_canonical_object(unsigned):
        raise CandidateRegistryV3Error("candidate registry v3 canonical hash changed")
    try:
        locked = datetime.fromisoformat(str(value.get("locked_at_utc")))
    except ValueError as exc:
        raise CandidateRegistryV3Error("candidate registry lock time is invalid") from exc
    if locked.tzinfo is None or locked.astimezone(timezone.utc) >= FUTURE_SEALED_START.replace(
        tzinfo=timezone.utc
    ):
        raise CandidateRegistryV3Error("candidate registry lock time is not pre-boundary")
    selected = value.get("selected_candidate") or {}
    if (
        selected.get("candidate_id") != SELECTED_CANDIDATE_ID
        or selected.get("ridge_strength") != SELECTED_RIDGE_STRENGTH
        or selected.get("model_version") != MODEL_VERSION
        or selected.get("independent_validation") is not False
    ):
        raise CandidateRegistryV3Error("candidate registry selection changed")
    bindings = value.get("bindings") or {}
    for locator_field, hash_field in (
        ("development_snapshot_manifest_locator", "development_snapshot_manifest_raw_sha256"),
        ("development_snapshot_payload_locator", "development_snapshot_payload_raw_sha256"),
        ("development_evaluation_summary_locator", "development_evaluation_summary_raw_sha256"),
        ("development_artifact_report_locator", "development_artifact_report_raw_sha256"),
        ("replay_fixture_locator", "replay_fixture_raw_sha256"),
    ):
        locator = bindings.get(locator_field)
        if not isinstance(locator, str) or _sha256(root / locator) != bindings.get(
            hash_field
        ):
            raise CandidateRegistryV3Error(f"candidate registry binding drifted: {locator_field}")
    artifact_locator = selected.get("artifact_locator")
    if (
        not isinstance(artifact_locator, str)
        or _sha256(root / artifact_locator) != selected.get("artifact_raw_sha256")
    ):
        raise CandidateRegistryV3Error("candidate model bytes drifted")
    if any((value.get("authority") or {}).values()):
        raise CandidateRegistryV3Error("candidate registry granted authority")
    if (value.get("future_state") or {}).get("future_outcomes_accessed") is not False:
        raise CandidateRegistryV3Error("candidate registry future state changed")
    return value


def write_candidate_registry_v3(root: Path) -> Path:
    payload = build_candidate_registry_v3(root=root)
    path = root / DEFAULT_OUTPUT
    _atomic_write_new(
        path,
        (json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode(
            "ascii"
        ),
    )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    try:
        path = write_candidate_registry_v3(args.root)
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, CandidateRegistryV3Error) as exc:
        print(str(exc))
        return 2
    print(
        json.dumps(
            {
                "registry": str(path),
                "raw_sha256": _sha256(path),
                "artifact_sha256": payload["artifact_sha256"],
                "locked_at_utc": payload["locked_at_utc"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

