from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from lol_kills.v2.draft.interactions.real_v1_g4 import contract as g4
from lol_kills.v2.draft.interactions.real_v1_g4 import runner as g4_runner


ROOT = Path(__file__).parents[4]
ARTIFACT_DIR = ROOT / "data/lol/v2/models/draft-interactions/real-v1-g4"
RAW_SHA256 = {
    "chronology-contract.json": "11b9280375f6f9e404d1aee647ea53aa978498dc3ad4bbd8d6217425782f116a",
    "source-binding-manifest.json": "2542217fbe3e57a2b1f3202cb52f99823111fcf6287899188911cd19a5823d1b",
    "review-core.json": "0594ad8ea014990cba5e6d87f4292109cb22ba5ef59aae8a54e353ead1ffac10",
    "dry-run-preflight.json": "799fb45da3f223f61b5fd6ac1698ae3321c37d65159307c861cf35da0a2e9e51",
    "pending-report.json": "7996352458dc9aec154794159b39db9b11f3a61c0fdc7a16245ce69d97f96b89",
}


def _read(name: str) -> dict:
    return json.loads((ARTIFACT_DIR / name).read_text(encoding="utf-8"))


def test_checked_artifacts_are_canonical_reproducible_and_independently_pinned() -> None:
    rebuilt = g4.build_pending_artifacts()
    for name, expected_raw in RAW_SHA256.items():
        raw = (ARTIFACT_DIR / name).read_bytes()
        parsed = json.loads(raw)
        assert hashlib.sha256(raw).hexdigest() == expected_raw
        assert raw == g4._canonical_bytes(parsed) + b"\n"
        unsigned = dict(parsed)
        claimed = unsigned.pop("artifact_sha256")
        assert claimed == g4._sha256(unsigned)
        assert rebuilt[name] == parsed


def test_exact_52_slot_sequence_stage_month_family_and_width_contract() -> None:
    chronology = _read("chronology-contract.json")
    slots = chronology["execution_slots"]
    assert chronology["slot_count"] == len(slots) == 52
    assert [slot["sequence"] for slot in slots] == list(range(1, 53))
    assert [(slot["stage"], slot["calendar_month"], slot["family"], slot["width"]) for slot in slots[:6]] == [
        ("inner", "2025-04", "ally_penalty", 8),
        ("inner", "2025-04", "ally_penalty", 8),
        ("inner", "2025-04", "ally_penalty", 8),
        ("inner", "2025-04", "enemy_penalty", 8),
        ("inner", "2025-04", "enemy_penalty", 8),
        ("inner", "2025-04", "enemy_penalty", 8),
    ]
    assert len(slots[:36]) == 36 and all(slot["stage"] == "inner" and slot["calendar_month"] in g4.INNER_MONTHS and slot["width"] == 8 for slot in slots[:36])
    assert [(slot["calendar_month"], slot["width"]) for slot in slots[36:48]] == [(month, width) for month in g4.DEVELOPMENT_MONTHS for width in g4.WIDTHS]
    assert [(slot["calendar_month"], slot["family"], slot["width"]) for slot in slots[48:]] == [
        ("2026-04", "locked_candidate", None), ("2026-04", "M8_comparator", 8),
        ("2026-05", "locked_candidate", None), ("2026-05", "M8_comparator", 8),
    ]
    assert all(slot["execution_status"] == "pending_permit" for slot in slots)


def test_october_is_diagnostic_only_and_excluded_from_every_scored_slot() -> None:
    chronology = _read("chronology-contract.json")
    october = chronology["chronology"]["october_2025"]
    assert october == {
        "status": "diagnostic_history_only",
        "excluded_from_width_scoring": True,
        "excluded_from_fit_ledger": True,
    }
    assert all(slot["calendar_month"] != "2025-10" for slot in chronology["execution_slots"])


def test_g1_is_not_silently_substituted_for_missing_full_2026_support() -> None:
    source = _read("source-binding-manifest.json")
    assert source["g1_lpl_subset_cross_check"]["pins"] == g4.G1_PINS
    assert source["g1_lpl_subset_cross_check"]["status"] == "identity_bound_not_support_substitute"
    assert source["g1_lpl_subset_cross_check"]["usage"] == "exact LPL subset binding/cross-check only; not the registered full-2026 OE support population"
    support = source["registered_2026_support_population"]
    assert support["status"] == support["terminal_status"] == "PASS"
    assert support["kind"] == "separate_registered_full_2026_approved_oe_support"
    assert support["outcome_free"] is support["aggregate_only"] is True


def test_dry_run_authenticates_metadata_then_blocks_before_any_target_m0_outcome_or_fit_loader() -> None:
    dry = g4.dry_run_preflight()
    checked = _read("dry-run-preflight.json")
    assert g4.build_pending_artifacts()["dry-run-preflight.json"] == checked
    assert dry == {key: value for key, value in checked.items() if key != "artifact_sha256"}
    assert checked["run_status"] == "blocked_before_target_m0_or_outcome_load"
    assert {key: checked[key] for key in ("target_loader_calls", "m0_loader_calls", "outcome_loader_calls", "fit_execution_calls")} == {
        "target_loader_calls": 0, "m0_loader_calls": 0, "outcome_loader_calls": 0, "fit_execution_calls": 0,
    }
    assert checked["blocker"]["status"] == "missing_fresh_independent_permit"
    assert g4_runner.dry_run()["call_order"] == [
        "verify_review_core", "verify_registered_2026_support_PASS", "verify_fresh_independent_permit", "blocked_missing_permit_before_protected_loaders",
    ]


def test_no_target_m0_outcome_fit_or_final_holdout_loader_exists_in_the_pending_contract_module() -> None:
    forbidden_callables = ("load_target", "load_m0", "load_outcome", "run_fit", "run_private", "open_final_holdout")
    assert all(not hasattr(g4, name) for name in forbidden_callables)
    source = Path(g4.__file__).read_text(encoding="utf-8")
    assert "parquet" not in source.lower()


def test_old_permit_core_cannot_authorize_changed_52_slot_chronology() -> None:
    old = _read_old_permit()
    review_core = _read("review-core.json")
    source = _read("source-binding-manifest.json")
    assert old["review_core_sha256"] != review_core["artifact_sha256"]
    required = source["fresh_independent_permit"]["required_exact_fields"]
    assert required == [
        "approved_action",
        "decision",
        "final_temporal_holdout_sealed",
        "independent_from_runner_and_generator",
        "review_core_sha256",
        "schema_id",
    ]
    assert source["fresh_independent_permit"]["status"] == "missing_fresh_independent_permit"


def _read_old_permit() -> dict:
    return json.loads((ROOT / "data/lol/v2/models/draft-interactions/representation-rank-runner-review-permit.json").read_text(encoding="utf-8"))


def test_result_raw_or_semantic_mutation_fails_before_any_pending_artifact_is_built(tmp_path: Path) -> None:
    raw = g4.G4_RESULT_PATH.read_bytes()
    changed_raw = tmp_path / "changed.json"
    changed_raw.write_bytes(raw + b" ")
    with pytest.raises(g4.G4RepairBlocked, match="G4_RESULT_RAW_SHA256_MISMATCH"):
        g4._read_verified_g4_result_metadata(changed_raw)

    changed_semantics = json.loads(raw)
    changed_semantics["selected_width"] = 1
    changed_semantics["artifact_sha256"] = g4._sha256({key: value for key, value in changed_semantics.items() if key != "artifact_sha256"})
    altered = tmp_path / "rehashed.json"
    altered.write_bytes(g4._canonical_bytes(changed_semantics))
    with pytest.raises(g4.G4RepairBlocked, match="G4_RESULT_RAW_SHA256_MISMATCH"):
        g4._read_verified_g4_result_metadata(altered)


def test_pending_artifact_membership_order_or_final_access_mutation_cannot_self_rehash_into_the_rebuilt_contract() -> None:
    rebuilt = g4.build_pending_artifacts()
    changed = deepcopy(_read("chronology-contract.json"))
    changed["execution_slots"][0], changed["execution_slots"][1] = changed["execution_slots"][1], changed["execution_slots"][0]
    changed["artifact_sha256"] = g4._sha256({key: value for key, value in changed.items() if key != "artifact_sha256"})
    assert changed != rebuilt["chronology-contract.json"]
    changed_pending = deepcopy(_read("pending-report.json"))
    changed_pending["claim_ceiling"]["final_holdout"] = True
    changed_pending["artifact_sha256"] = g4._sha256({key: value for key, value in changed_pending.items() if key != "artifact_sha256"})
    assert changed_pending != rebuilt["pending-report.json"]


def test_pending_contract_is_byte_identical_across_two_fresh_processes() -> None:
    code = (
        "import json; from lol_kills.v2.draft.interactions.real_v1_g4.contract import build_pending_artifacts; "
        "print(json.dumps({k:v['artifact_sha256'] for k,v in build_pending_artifacts().items()},sort_keys=True))"
    )
    command = [sys.executable, "-c", code]
    first = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True).stdout
    second = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True).stdout
    assert first == second
    assert json.loads(first) == {name: _read(name)["artifact_sha256"] for name in RAW_SHA256}


def test_review_core_binds_every_executable_byte_and_rejects_mutated_pin() -> None:
    core = _read("review-core.json")
    executables = core["executables"]
    for item in (*core["review_subject_bytes"].values(), *executables["fit_primitives"].values()):
        assert hashlib.sha256((ROOT / item["locator"]).read_bytes()).hexdigest() == item["raw_sha256"]
    with pytest.raises(g4.G4RepairBlocked, match="FIT_PRIMITIVE_ASSAY_SHA256_MISMATCH"):
        g4._verified_raw_file(
            "lol_kills/v2/draft/interactions/representation_rank_assay.py",
            "0" * 64,
            code="FIT_PRIMITIVE_ASSAY_SHA256_MISMATCH",
        )


def test_runner_is_a_complete_bound_execution_path_not_a_permit_only_no_op() -> None:
    source = Path(g4_runner.__file__).read_text(encoding="utf-8")
    assert "EXECUTION_REQUIRES_SEPARATELY_REVIEWED_TARGET_M0_OUTCOME_LOADER_BINDING" not in source
    assert "execute_once_after_permit" in source
    assert "_fit_once_with_exact_starts" in source


@pytest.mark.parametrize("alias_kind", ["symlink", "hardlink"])
def test_writer_rejects_leaf_aliases_without_touching_backing_file(tmp_path: Path, alias_kind: str) -> None:
    output = tmp_path / "output"
    output.mkdir()
    backing = tmp_path / "backing.json"
    backing.write_text("untouched", encoding="utf-8")
    leaf = output / "chronology-contract.json"
    if alias_kind == "symlink":
        leaf.symlink_to(backing)
    else:
        os.link(backing, leaf)
    with pytest.raises(g4.G4RepairBlocked, match="OUTPUT_LEAF_UNSAFE"):
        g4.write_pending_artifacts(output)
    assert backing.read_text(encoding="utf-8") == "untouched"


def test_writer_rejects_symlinked_ancestor(tmp_path: Path) -> None:
    backing = tmp_path / "backing"
    backing.mkdir()
    safe = tmp_path / "safe"
    safe.mkdir()
    unsafe = safe / "unsafe"
    unsafe.symlink_to(backing, target_is_directory=True)
    with pytest.raises(g4.G4RepairBlocked, match="OUTPUT_PARENT_UNSAFE"):
        g4.write_pending_artifacts(unsafe / "nested")
    assert not (backing / "nested").exists()
