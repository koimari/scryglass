"""Bind the strict-roster July Draft Score replay as adaptive evidence.

The July outcomes were already available during model development, so this
artifact is deliberately not an independent validation record.  Its purpose
is narrower: make the known same-context harm from the current temporal draft
feature family durable, reproducible, and impossible to promote by omission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from lol_kills.etl.roster_receipts import load_receipt_manifest
from lol_kills.v2.data.common import sha256_canonical_object


SCHEMA_VERSION = "scryglass:draft-terminal-adaptive-temporal-diagnostic:v1"
DEFAULT_RUN_DIR = Path(
    "data/lol/v2/experiments/leaguepedia/manual-run-2026-07-31"
)
DEFAULT_REPLAY_DIR = (
    DEFAULT_RUN_DIR / "temporal-runtime-hybrid-strict-roster-receipts-v2-incremental"
)
DEFAULT_ROSTER_MANIFEST = DEFAULT_RUN_DIR / "roster-receipts-v1/receipt-manifest.json"
DEFAULT_OUTPUT = Path(
    "data/lol/v2/models/draft-terminal/adaptive-temporal-diagnostic-v1.json"
)
EXPECTED_MODEL_VERSION = "temporal-hybrid-v1.3.0"
OUTCOME_FIELDS = frozenset(
    {
        "actual_blue_win",
        "blue_win",
        "result",
        "winner",
        "winner_team_id",
        "won",
        "WinTeam",
        "LossTeam",
    }
)


class AdaptiveTemporalDiagnosticError(ValueError):
    """The adaptive temporal diagnostic or one of its bindings is invalid."""


def _raw_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_object(path: Path, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdaptiveTemporalDiagnosticError(f"{field} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise AdaptiveTemporalDiagnosticError(f"{field} must contain a JSON object")
    return value


def _atomic_write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise AdaptiveTemporalDiagnosticError(f"refusing to overwrite existing artifact: {path}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise AdaptiveTemporalDiagnosticError(
                f"refusing to overwrite existing artifact: {path}"
            )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _metric(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AdaptiveTemporalDiagnosticError(f"{field} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise AdaptiveTemporalDiagnosticError(f"{field} is not finite")
    return result


def _validate_source_hashes(
    manifest: Mapping[str, Any],
    *,
    run_dir: Path,
    roster_manifest: Path,
) -> dict[str, str]:
    sources = manifest.get("source_hashes")
    if not isinstance(sources, Mapping):
        raise AdaptiveTemporalDiagnosticError("replay manifest source_hashes are missing")
    paths = {
        "frozen_ledger": run_dir / "frozen-ledger.jsonl",
        "target_outcomes": run_dir / "normalized-outcome-rows.jsonl",
        "prior_outcomes": run_dir
        / "autoresearch/raw/prior-games/normalized-prior-rows.jsonl",
        "prior_drafts": run_dir
        / "autoresearch/raw/prior-drafts/normalized-prior-draft-rows.jsonl",
        "runner": Path("lol_kills/research/temporal_draft_runtime.py"),
    }
    actual: dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise AdaptiveTemporalDiagnosticError(f"bound source is missing: {path}")
        digest = _raw_sha256(path)
        if sources.get(name) != digest:
            raise AdaptiveTemporalDiagnosticError(f"bound source hash changed: {name}")
        actual[name] = digest
    roster_readiness, _ = load_receipt_manifest(roster_manifest)
    roster_manifest_sha = str(roster_readiness["manifest_sha256"])
    roster_file_sha = str(roster_readiness["receipt_file_sha256"])
    if sources.get("roster_receipt_manifest") != roster_manifest_sha:
        raise AdaptiveTemporalDiagnosticError("roster receipt manifest binding changed")
    if sources.get("roster_receipt_file") != roster_file_sha:
        raise AdaptiveTemporalDiagnosticError("roster receipt file binding changed")
    actual["roster_receipt_manifest"] = roster_manifest_sha
    actual["roster_receipt_file"] = roster_file_sha
    return actual


def _ledger_summary(path: Path) -> dict[str, Any]:
    statuses: Counter[str] = Counter()
    fixture_ids: set[str] = set()
    row_count = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AdaptiveTemporalDiagnosticError(
                f"scored ledger line {line_number} is not JSON"
            ) from exc
        if not isinstance(row, dict):
            raise AdaptiveTemporalDiagnosticError(
                f"scored ledger line {line_number} is not an object"
            )
        if OUTCOME_FIELDS.intersection(row):
            raise AdaptiveTemporalDiagnosticError(
                f"outcome field leaked into scored ledger line {line_number}"
            )
        fixture_id = str(row.get("fixture_id") or "")
        if not fixture_id or fixture_id in fixture_ids:
            raise AdaptiveTemporalDiagnosticError(
                f"scored ledger line {line_number} fixture_id is missing or duplicated"
            )
        fixture_ids.add(fixture_id)
        lineup = row.get("lineup")
        score = row.get("score")
        if not isinstance(lineup, Mapping) or not isinstance(score, Mapping):
            raise AdaptiveTemporalDiagnosticError(
                f"scored ledger line {line_number} is incomplete"
            )
        if OUTCOME_FIELDS.intersection(lineup) or OUTCOME_FIELDS.intersection(score):
            raise AdaptiveTemporalDiagnosticError(
                f"outcome field leaked into scored ledger line {line_number}"
            )
        status = str(lineup.get("status") or "")
        if status not in {"verified_preevent", "mismatch", "unavailable"}:
            raise AdaptiveTemporalDiagnosticError(
                f"scored ledger line {line_number} lineup status is invalid"
            )
        if score.get("model_version") != EXPECTED_MODEL_VERSION:
            raise AdaptiveTemporalDiagnosticError(
                f"scored ledger line {line_number} model version changed"
            )
        context_available = score.get("p_blue_context") is not None
        if context_available != (status == "verified_preevent"):
            raise AdaptiveTemporalDiagnosticError(
                f"scored ledger line {line_number} context gate is inconsistent"
            )
        statuses[status] += 1
        row_count += 1
    return {
        "rows": row_count,
        "unique_fixture_ids": len(fixture_ids),
        "lineup_status_counts": dict(sorted(statuses.items())),
        "outcome_fields_present": False,
        "raw_sha256": _raw_sha256(path),
    }


def build_adaptive_temporal_diagnostic(
    *,
    replay_dir: Path = DEFAULT_REPLAY_DIR,
    run_dir: Path = DEFAULT_RUN_DIR,
    roster_manifest: Path = DEFAULT_ROSTER_MANIFEST,
) -> dict[str, Any]:
    manifest_path = replay_dir / "temporal-hybrid-manifest.json"
    evaluation_path = replay_dir / "temporal-hybrid-evaluation.json"
    runtime_path = replay_dir / "temporal-hybrid-runtime.json"
    ledger_path = replay_dir / "temporal-hybrid-scored-ledger.jsonl"
    for path in (manifest_path, evaluation_path, runtime_path, ledger_path):
        if not path.is_file():
            raise AdaptiveTemporalDiagnosticError(f"required replay artifact is missing: {path}")
    manifest = _read_object(manifest_path, "replay manifest")
    if manifest.get("schema_version") != "scryglass:temporal-draft-runtime:v1":
        raise AdaptiveTemporalDiagnosticError("replay manifest schema_version changed")
    if manifest.get("availability_status") != "adaptive_development_replay_not_independent":
        raise AdaptiveTemporalDiagnosticError("replay is not classified as adaptive development")
    declared_manifest_hash = manifest.get("manifest_sha256")
    unsigned_manifest = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    if declared_manifest_hash != sha256_canonical_object(unsigned_manifest):
        raise AdaptiveTemporalDiagnosticError("replay manifest canonical hash does not match")
    claim_ceiling = manifest.get("claim_ceiling")
    if claim_ceiling != {
        "adaptive_development_diagnostic": True,
        "betting": False,
        "independent_validation": False,
        "production_probability": False,
    }:
        raise AdaptiveTemporalDiagnosticError("replay claim ceiling changed")
    source_hashes = _validate_source_hashes(
        manifest, run_dir=run_dir, roster_manifest=roster_manifest
    )

    evaluation = _read_object(evaluation_path, "replay evaluation")
    if evaluation != manifest.get("evaluation"):
        raise AdaptiveTemporalDiagnosticError("evaluation file differs from bound manifest evaluation")
    if evaluation.get("model_version") != EXPECTED_MODEL_VERSION:
        raise AdaptiveTemporalDiagnosticError("replay model_version changed")
    if evaluation.get("strict_roster") is not True:
        raise AdaptiveTemporalDiagnosticError("replay did not require strict roster authority")
    if _raw_sha256(runtime_path) != evaluation.get("artifact_sha256"):
        raise AdaptiveTemporalDiagnosticError("runtime artifact hash does not match evaluation")
    ledger = _ledger_summary(ledger_path)
    if ledger["rows"] != int(evaluation.get("maps", -1)):
        raise AdaptiveTemporalDiagnosticError("scored ledger row count does not match evaluation")
    verified = int(ledger["lineup_status_counts"].get("verified_preevent", 0))
    if verified != int(evaluation.get("contextual_lineup_coverage", -1)):
        raise AdaptiveTemporalDiagnosticError("verified lineup count does not match evaluation")

    contextual = evaluation.get("contextual")
    comparator = evaluation.get("context_without_draft")
    incremental = evaluation.get("incremental_draft_against_same_context")
    if not all(isinstance(item, Mapping) for item in (contextual, comparator, incremental)):
        raise AdaptiveTemporalDiagnosticError("same-context comparison is missing")
    if int(contextual.get("n", -1)) != verified or int(comparator.get("n", -1)) != verified:
        raise AdaptiveTemporalDiagnosticError("same-context comparison denominator changed")
    expected_brier_delta = round(
        _metric(contextual.get("brier"), "contextual.brier")
        - _metric(comparator.get("brier"), "context_without_draft.brier"),
        6,
    )
    expected_logloss_delta = round(
        _metric(contextual.get("logloss"), "contextual.logloss")
        - _metric(comparator.get("logloss"), "context_without_draft.logloss"),
        6,
    )
    if _metric(incremental.get("brier_delta"), "incremental.brier_delta") != expected_brier_delta:
        raise AdaptiveTemporalDiagnosticError("incremental Brier delta does not recompute")
    if _metric(incremental.get("logloss_delta"), "incremental.logloss_delta") != expected_logloss_delta:
        raise AdaptiveTemporalDiagnosticError("incremental log-loss delta does not recompute")
    nonharmful = expected_brier_delta <= 0 and expected_logloss_delta <= 0
    result_state = (
        "ADAPTIVE_NONHARM_SIGNAL_NOT_INDEPENDENT"
        if nonharmful
        else "ADAPTIVE_DRAFT_TERMS_HARM"
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "result_state": result_state,
        "purpose": (
            "bind a known adaptive temporal diagnostic for the current app draft "
            "feature family; never substitute it for terminal-model L2 evaluation"
        ),
        "model_scope": {
            "model_version": EXPECTED_MODEL_VERSION,
            "same_as_terminal_m0_candidate": False,
            "applicability": "known_app_draft_feature_family_diagnostic",
        },
        "inputs": {
            "replay_manifest": str(manifest_path),
            "replay_manifest_raw_sha256": _raw_sha256(manifest_path),
            "replay_manifest_canonical_sha256": declared_manifest_hash,
            "evaluation": str(evaluation_path),
            "evaluation_raw_sha256": _raw_sha256(evaluation_path),
            "runtime": str(runtime_path),
            "runtime_raw_sha256": _raw_sha256(runtime_path),
            "outcome_free_scored_ledger": str(ledger_path),
            "outcome_free_scored_ledger_raw_sha256": ledger["raw_sha256"],
            "source_hashes": source_hashes,
        },
        "population": {
            "maps": ledger["rows"],
            "lineup_status_counts": ledger["lineup_status_counts"],
            "exact_roster_context_maps": verified,
        },
        "metrics": {
            "pure_draft": evaluation.get("pure_draft"),
            "context_without_draft": dict(comparator),
            "context_with_draft": dict(contextual),
            "incremental_draft_against_same_context": {
                "n": verified,
                "brier_delta": expected_brier_delta,
                "logloss_delta": expected_logloss_delta,
                "pass_rule": "both deltas must be nonpositive",
                "passed": nonharmful,
            },
        },
        "decision": {
            "known_app_draft_family_nonharmful": nonharmful,
            "terminal_candidate_selected": False,
            "independent_validation": False,
            "production_eligible": False,
        },
        "claim_ceiling": {
            "adaptive_development_diagnostic": True,
            "terminal_model_reliability": False,
            "independent_validation": False,
            "probability": False,
            "recommendation": False,
            "betting": False,
        },
    }
    payload["artifact_sha256"] = sha256_canonical_object(payload)
    return payload


def validate_adaptive_temporal_diagnostic(
    payload: Mapping[str, Any], *, root: Path = Path(".")
) -> dict[str, Any]:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise AdaptiveTemporalDiagnosticError("diagnostic schema_version changed")
    unsigned = {key: value for key, value in payload.items() if key != "artifact_sha256"}
    if payload.get("artifact_sha256") != sha256_canonical_object(unsigned):
        raise AdaptiveTemporalDiagnosticError("diagnostic artifact hash does not match")
    expected = build_adaptive_temporal_diagnostic(
        replay_dir=root / DEFAULT_REPLAY_DIR,
        run_dir=root / DEFAULT_RUN_DIR,
        roster_manifest=root / DEFAULT_ROSTER_MANIFEST,
    )
    if dict(payload) != expected:
        raise AdaptiveTemporalDiagnosticError("diagnostic does not replay from current bound bytes")
    return dict(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        payload = build_adaptive_temporal_diagnostic(
            replay_dir=args.root / DEFAULT_REPLAY_DIR,
            run_dir=args.root / DEFAULT_RUN_DIR,
            roster_manifest=args.root / DEFAULT_ROSTER_MANIFEST,
        )
        output = args.out if args.out.is_absolute() else args.root / args.out
        _atomic_write_new(
            output,
            (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
                "utf-8"
            ),
        )
    except (OSError, AdaptiveTemporalDiagnosticError, ValueError) as exc:
        print(str(exc))
        return 2
    print(
        json.dumps(
            {
                "output": str(output),
                "artifact_sha256": payload["artifact_sha256"],
                "result_state": payload["result_state"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
