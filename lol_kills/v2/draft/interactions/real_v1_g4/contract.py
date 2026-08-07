"""No-fit G4 repair contract with an independently permit-gated chronology.

This module deliberately contains no target, M0-prediction, or outcome loader.
It can verify the small released G4 result metadata, construct the frozen
52-slot chronology, and produce a pending report.  A separate authenticated
full-2026 support population and a fresh independent permit are required
before an execution module may even be introduced.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Mapping

from lol_kills.v2.draft.interactions import representation_rank_2026_support_gate as support_gate


ROOT = Path(__file__).resolve().parents[5]
NAMESPACE = ROOT / "data/lol/v2/models/draft-interactions/real-v1-g4"
G4_RESULT_PATH = ROOT / "data/lol/warehouse/private_v2/draft-interactions/representation-rank-private-result.json"
G4_RESULT_RAW_SHA256 = "1219c5c94e805f56418cb16c980b25c039c8d182b2b8b650508317a844c16881"
G4_RESULT_ARTIFACT_SHA256 = "6936525d564b819082b5a8bfc8a11599d535d155192797ec890b624b2e23df35"
SUPPORT_GATE_PATH = ROOT / "data/lol/v2/models/draft-interactions/representation-rank-2026-support-gate.json"
SUPPORT_GATE_RAW_SHA256 = "a48a89ac663d4461eeaba6503c561cc34e4c364e1f2f7a562c3f8446fcc1b8c1"
SUPPORT_GATE_ARTIFACT_SHA256 = "89787963d0cbe9aa915db0a9ad1d4ef95d2d700acba1073635dc72b2c5ca2d61"
SUPPORT_PROJECTION_SHA256 = "132ed5b87169b4b937897be0f095af3d70a8001cfb0474dd5b6677ae8f51783f"
G1_PINS = {
    "manifest_sha256": "3af87fffb2b32fd95aeb920409abe0254fa158b3dc7f079650b3472731d4ff72",
    "rows_sha256": "4ed79abb0b2471a666ab5643b91edf33c2fdde19e361c456aa589d2e9a4df846",
    "selected_target_sha256": "4c332fa4e6cb155341bcffd83bd0ee1be2e04f3b5950b8a7745931253dd8bd2d",
    "split_payload_sha256": "1695cee14ad6b4221526ec6187206b8c61a560a00005d2f799f808ed901ee014",
}
INNER_MONTHS = ("2025-04", "2025-05", "2025-06", "2025-07", "2025-08", "2025-09")
PENALTIES = (0.01, 0.1, 1.0)
DEVELOPMENT_MONTHS = ("2026-01", "2026-02", "2026-03")
VALIDATION_MONTHS = ("2026-04", "2026-05")
WIDTHS = (1, 2, 4, 8)
SEEDS = {"inner": 2026072900, "development": 2026072901, "validation": 2026072902}
PERMIT_SCHEMA = "scryglass.representation-rank-runner-review-permit.v1"
OUTPUT_SCHEMA = "scryglass:real-v1-g4-repair-execution-output:v1"
TARGET_AUTHORITY_PINS = {
    "authority": {"locator": "data/lol/v2/models/draft-interactions/oe-private-target-authority.json", "raw_sha256": "b1d0a6e37abb9a74dee8689dc19ab54d30fd15516bd4ee454906a075d8f20788"},
    "split": {"locator": "data/lol/v2/models/draft-interactions/oe-private-split-assignment.json", "raw_sha256": "76717d32a1686348d6a4408d00427af7bf0eb45c86afbfb99521b94ac0f7bc4d", "artifact_sha256": G1_PINS["split_payload_sha256"]},
    "evidence": {"locator": "data/lol/v2/models/draft-interactions/oe-private-target-evidence.json", "raw_sha256": "164e90134f4fe464eed7784314307b48bb44e86de80158b36a6250b6cd2f21aa", "artifact_sha256": "8d96e0fef0883595595b8e962bf14a920b3488bb2189ce3f7ab8fe23221f5304"},
}


class G4RepairBlocked(RuntimeError):
    """Raised if a pending contract is asked to run a protected fit."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, allow_nan=False, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise G4RepairBlocked("noncanonical pending artifact") from error


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _read_verified_g4_result_metadata(path: Path = G4_RESULT_PATH) -> dict[str, Any]:
    """Read result metadata only; it does not open any target/M0/outcome data."""

    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != G4_RESULT_RAW_SHA256:
        raise G4RepairBlocked("G4_RESULT_RAW_SHA256_MISMATCH")
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as error:
        raise G4RepairBlocked("G4_RESULT_NOT_JSON") from error
    required = {
        "schema_id": "scryglass.representation-rank-private-result.v1",
        "run_status": "inconclusive",
        "selected_model": "M0",
        "selected_width": None,
        "reason_code": "coverage_gate_failed",
    }
    if any(result.get(key) != expected for key, expected in required.items()):
        raise G4RepairBlocked("G4_RESULT_SEMANTIC_MISMATCH")
    if result.get("fallback") != "M0":
        raise G4RepairBlocked("G4_RESULT_FALLBACK_MISMATCH")
    if result.get("fit_counts", {}).get("actual") != 0:
        raise G4RepairBlocked("G4_RESULT_FIT_ACTUAL_NOT_ZERO")
    stage = result.get("stage_status", {})
    if stage.get("development", {}).get("status") != "not_run" or stage.get("validation", {}).get("status") != "not_run":
        raise G4RepairBlocked("G4_RESULT_DEVELOPMENT_OR_VALIDATION_NOT_NOT_RUN")
    if result.get("final_target_loaded") is not False:
        raise G4RepairBlocked("G4_RESULT_FINAL_HOLDOUT_ACCESSED")
    if result.get("artifact_sha256") != G4_RESULT_ARTIFACT_SHA256:
        raise G4RepairBlocked("G4_RESULT_ARTIFACT_SHA256_MISMATCH")
    return {key: result[key] for key in (*required, "fit_counts", "stage_status", "final_target_loaded", "fallback", "artifact_sha256")}


def _slots() -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    sequence = 1
    for month in INNER_MONTHS:
        for family in ("ally_penalty", "enemy_penalty"):
            for penalty in PENALTIES:
                slots.append({"sequence": sequence, "stage": "inner", "calendar_month": month, "family": family, "penalty": penalty, "width": 8, "seed": SEEDS["inner"], "execution_status": "pending_permit"})
                sequence += 1
    # October is deliberately a diagnostic/history boundary, not an inner
    # penalty slot and therefore absent from the 52-slot execution ledger.
    for month in DEVELOPMENT_MONTHS:
        for width in WIDTHS:
            slots.append({"sequence": sequence, "stage": "development", "calendar_month": month, "family": "candidate_width", "penalty": None, "width": width, "seed": SEEDS["development"], "execution_status": "pending_permit"})
            sequence += 1
    for month in VALIDATION_MONTHS:
        slots.append({"sequence": sequence, "stage": "validation", "calendar_month": month, "family": "locked_candidate", "penalty": None, "width": None, "seed": SEEDS["validation"], "execution_status": "pending_permit"})
        sequence += 1
        slots.append({"sequence": sequence, "stage": "validation", "calendar_month": month, "family": "M8_comparator", "penalty": None, "width": 8, "seed": SEEDS["validation"], "execution_status": "pending_permit"})
        sequence += 1
    if len(slots) != 52 or [slot["sequence"] for slot in slots] != list(range(1, 53)):
        raise G4RepairBlocked("CHRONOLOGY_SLOT_COUNT_MISMATCH")
    return slots


def _chronology_contract() -> dict[str, Any]:
    slots = _slots()
    return {
        "schema_version": "scryglass:real-v1-g4-repair-chronology:v1",
        "execution_slots": slots,
        "slot_count": 52,
        "chronology": {
            "inner_penalty_months": list(INNER_MONTHS),
            "inner_penalty_family_count": 2,
            "inner_penalty_grid": list(PENALTIES),
            "october_2025": {"status": "diagnostic_history_only", "excluded_from_width_scoring": True, "excluded_from_fit_ledger": True},
            "development_width_months": list(DEVELOPMENT_MONTHS),
            "development_widths": list(WIDTHS),
            "validation_months": list(VALIDATION_MONTHS),
            "validation_members": ["locked_candidate", "M8_comparator"],
            "seeds": dict(SEEDS),
        },
        "execution_policy": {"one_execution_only_after_fresh_permit": True, "three_deterministic_optimization_starts_per_fit": True, "additional_bruteforce_reruns": False, "gate_edits": False, "validation_reuse": False, "target_loader_authorized": False},
    }


def _authenticate_2026_support() -> dict[str, Any]:
    """Replay the registered aggregate-only full-2026 support PASS first."""

    raw = SUPPORT_GATE_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SUPPORT_GATE_RAW_SHA256:
        raise G4RepairBlocked("SUPPORT_GATE_RAW_SHA256_MISMATCH")
    try:
        gate = json.loads(raw)
    except json.JSONDecodeError as error:
        raise G4RepairBlocked("SUPPORT_GATE_NOT_JSON") from error
    if gate.get("artifact_sha256") != SUPPORT_GATE_ARTIFACT_SHA256:
        raise G4RepairBlocked("SUPPORT_GATE_ARTIFACT_SHA256_MISMATCH")
    # The existing module first authenticates the G4 raw aggregate source,
    # projects only outcome-free coverage, and deterministically replays gate
    # arithmetic.  No target/M0/outcome table is read on this path.
    projection = support_gate.load_pinned_source_projection(G4_RESULT_PATH)
    if support_gate.canonical_sha256(projection) != SUPPORT_PROJECTION_SHA256:
        raise G4RepairBlocked("SUPPORT_PROJECTION_SHA256_MISMATCH")
    support_gate.validate_support_gate(gate, json.loads(G4_RESULT_PATH.read_text(encoding="utf-8")))
    if gate.get("terminal_status") != "PASS":
        raise G4RepairBlocked("SUPPORT_GATE_NOT_PASS")
    return {
        "status": "PASS",
        "kind": "separate_registered_full_2026_approved_oe_support",
        "locator": str(SUPPORT_GATE_PATH.relative_to(ROOT)),
        "raw_sha256": SUPPORT_GATE_RAW_SHA256,
        "artifact_sha256": SUPPORT_GATE_ARTIFACT_SHA256,
        "authorized_source_projection_sha256": SUPPORT_PROJECTION_SHA256,
        "outcome_free": True,
        "aggregate_only": True,
        "terminal_status": "PASS",
    }


def _missing_permit(review_core_sha256: str) -> dict[str, Any]:
    return {
        "status": "missing_fresh_independent_permit",
        "required_exact_fields": [
            "approved_action",
            "decision",
            "final_temporal_holdout_sealed",
            "independent_from_runner_and_generator",
            "review_core_sha256",
            "schema_id",
        ],
        "required_value": {
            "approved_action": "private_target_m0_load_and_rank_assay",
            "decision": "PASS",
            "final_temporal_holdout_sealed": True,
            "independent_from_runner_and_generator": True,
            "review_core_sha256": review_core_sha256,
            "schema_id": PERMIT_SCHEMA,
        },
        "reviewer_identity_handling": "Recorded by the independent human approval authority, not as an invented permit field.",
    }


def _source_binding() -> dict[str, Any]:
    g4 = _read_verified_g4_result_metadata()
    return {
        "schema_version": "scryglass:real-v1-g4-repair-source-binding:v1",
        "g4_private_result": {"locator": str(G4_RESULT_PATH.relative_to(ROOT)), "raw_sha256": G4_RESULT_RAW_SHA256, "artifact_sha256": G4_RESULT_ARTIFACT_SHA256, "verified_metadata": g4},
        "g1_lpl_subset_cross_check": {"pins": dict(G1_PINS), "status": "identity_bound_not_support_substitute", "usage": "exact LPL subset binding/cross-check only; not the registered full-2026 OE support population", "origin_ledgers": {"status": "not_used", "reason": "no target or fit loader is allowed in pending state"}},
        "registered_2026_support_population": _authenticate_2026_support(),
        "target_authority_split_evidence_pins": TARGET_AUTHORITY_PINS,
    }


def _raw_file_sha256(path: Path) -> str:
    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise G4RepairBlocked("REVIEW_CORE_EXECUTABLE_PATH_UNSAFE")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verified_raw_file(root_relative: str, expected_sha256: str, *, code: str) -> dict[str, str]:
    path = ROOT / root_relative
    actual = _raw_file_sha256(path)
    if actual != expected_sha256:
        raise G4RepairBlocked(code)
    return {"locator": root_relative, "raw_sha256": actual}


def _review_core(chronology: Mapping[str, Any], source_binding: Mapping[str, Any]) -> dict[str, Any]:
    """Everything a fresh permit must bind, including the executable bytes."""

    subject_paths = {
        "isolated_package_init": Path(__file__).with_name("__init__.py"),
        "isolated_contract": Path(__file__),
        "isolated_runner": Path(__file__).with_name("runner.py"),
        "isolated_result_schema": Path(__file__).with_name("result.py"),
        "isolated_synthetic_fixtures": Path(__file__).with_name("fixtures.py"),
        "isolated_coverage_preflight": Path(__file__).with_name("coverage_preflight.py"),
        "isolated_tests": ROOT / "tests/model_v2/draft/interactions/test_real_v1_g4_runner.py",
    }
    return {
        "schema_version": "scryglass:real-v1-g4-repair-review-core:v1",
        "chronology_contract_sha256": _sha256(chronology),
        "source_binding_sha256": _sha256(source_binding),
        "support_first_loader_sequence": ["verify_review_core", "verify_registered_2026_support_PASS", "verify_fresh_independent_permit", "then_and_only_then_authorize_target_m0_outcome_loaders"],
        "review_subject_bytes": {
            name: {"locator": str(path.relative_to(ROOT)), "raw_sha256": _raw_file_sha256(path)}
            for name, path in subject_paths.items()
        },
        "executables": {
            "fit_primitives": {
                "assay": _verified_raw_file("lol_kills/v2/draft/interactions/representation_rank_assay.py", "2fdd312ecf468d9f6b42dfe47fca3b81d9a1460ad24283b247c03561fba4cc2c", code="FIT_PRIMITIVE_ASSAY_SHA256_MISMATCH"),
                "private_runner": _verified_raw_file("lol_kills/v2/draft/interactions/representation_rank_private_runner.py", "81e232ba4d34af6c4039f4e987466ab1030f990381ba70dffc08c132c83f9ab4", code="FIT_PRIMITIVE_RUNNER_SHA256_MISMATCH"),
            },
        },
        "target_authority_split_evidence_pins": {
            name: _verified_raw_file(item["locator"], item["raw_sha256"], code=f"TARGET_{name.upper()}_SHA256_MISMATCH") | ({"artifact_sha256": item["artifact_sha256"]} if "artifact_sha256" in item else {})
            for name, item in TARGET_AUTHORITY_PINS.items()
        },
        "output_schema": OUTPUT_SCHEMA,
        "claim_ceiling": {"private_execution_only_after_permit": True, "prediction": False, "publication": False, "production": False, "promotion": False, "sota": False, "final_holdout": False},
        "optimization": {"starts_per_fit": 3, "starts": ["fit_only_residual_informed_nonzero_start", "fixed_seed_nonzero_perturbation_1", "fixed_seed_nonzero_perturbation_2"], "extra_bruteforce_reruns": False, "one_execution_only": True},
    }


def dry_run_preflight() -> dict[str, Any]:
    """A metadata-only dry run which proves protected loaders were not called."""

    source_binding = _source_binding()
    chronology = _chronology_contract()
    review_core = _review_core(chronology, source_binding)
    return {
        "schema_version": "scryglass:real-v1-g4-repair-dry-run:v1",
        "run_status": "blocked_before_target_m0_or_outcome_load",
        "target_loader_calls": 0,
        "m0_loader_calls": 0,
        "outcome_loader_calls": 0,
        "fit_execution_calls": 0,
        "slot_count": chronology["slot_count"],
        "review_core_sha256": _sha256(review_core),
        "chronology_contract_sha256": _sha256(chronology),
        "source_binding_sha256": _sha256(source_binding),
        "blocker": _missing_permit(_sha256(review_core)),
    }


def build_pending_artifacts() -> dict[str, dict[str, Any]]:
    """Return only no-fit, reproducible contract and blocker artifacts."""

    chronology = _chronology_contract()
    source_binding = _source_binding()
    review_core = _review_core(chronology, source_binding)
    review_core_sha256 = _sha256(review_core)
    source_binding = {**source_binding, "review_core_sha256": review_core_sha256, "fresh_independent_permit": _missing_permit(review_core_sha256)}
    dry_run = dry_run_preflight()
    review_core["artifact_sha256"] = review_core_sha256
    for value in (chronology, source_binding, dry_run):
        value["artifact_sha256"] = _sha256(value)
    pending_report = {
        "schema_version": "scryglass:real-v1-g4-repair-pending-report:v1",
        "run_status": "blocked_before_target_m0_or_outcome_load",
        "target_loader_calls": 0,
        "m0_loader_calls": 0,
        "outcome_loader_calls": 0,
        "fit_execution_calls": 0,
        "review_core_sha256": review_core_sha256,
        "chronology_contract_sha256": chronology["artifact_sha256"],
        "source_binding_sha256": source_binding["artifact_sha256"],
        "dry_run_sha256": dry_run["artifact_sha256"],
        "missing_approval": _missing_permit(review_core_sha256),
        "claim_ceiling": {"private_contract_preflight": True, "fit_executed": False, "prediction": False, "publication": False, "production": False, "promotion": False, "sota": False, "final_holdout": False},
    }
    pending_report["artifact_sha256"] = _sha256(pending_report)
    return {"chronology-contract.json": chronology, "source-binding-manifest.json": source_binding, "review-core.json": review_core, "dry-run-preflight.json": dry_run, "pending-report.json": pending_report}


def _preflight_lexical_parent_chain(path: Path) -> None:
    absolute = path.absolute()
    current, parts = Path(absolute.anchor), absolute.parts[1:-1]
    for anchor in (ROOT, Path(tempfile.gettempdir())):
        try:
            relative = absolute.relative_to(anchor.absolute())
        except ValueError:
            continue
        current, parts = anchor.resolve(), relative.parts[:-1]
        break
    for part in parts:
        current = current / part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError as error:
            raise G4RepairBlocked("OUTPUT_PARENT_MISSING") from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise G4RepairBlocked("OUTPUT_PARENT_UNSAFE")


def _ensure_safe_directory(directory: Path) -> None:
    # Check every existing lexical parent before creating anything.  In
    # particular, never call mkdir(parents=True) through a symlinked parent.
    _preflight_lexical_parent_chain(directory)
    try:
        metadata = os.lstat(directory)
    except FileNotFoundError:
        directory.mkdir()
        metadata = os.lstat(directory)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise G4RepairBlocked("OUTPUT_DIRECTORY_UNSAFE")


def _safe_atomic_write(path: Path, payload: bytes) -> None:
    _preflight_lexical_parent_chain(path)
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        metadata = None
    if metadata is not None and (stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1):
        raise G4RepairBlocked("OUTPUT_LEAF_UNSAFE")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_pending_artifacts(directory: Path = NAMESPACE) -> dict[str, str]:
    """Write only the pending no-fit artifacts under the isolated namespace."""

    _ensure_safe_directory(directory)
    artifacts = build_pending_artifacts()
    for name, artifact in artifacts.items():
        _safe_atomic_write(directory / name, _canonical_bytes(artifact) + b"\n")
    return {name: artifact["artifact_sha256"] for name, artifact in artifacts.items()}
