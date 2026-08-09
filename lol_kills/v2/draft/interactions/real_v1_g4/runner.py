"""Complete permit-gated, aggregate-only corrected 52-slot G4 runner.

The runner has no default protected data loader.  A future reviewed adapter
must inject the pinned existing feature/availability/target-M0/fit primitives;
that keeps importing and dry-running this module incapable of reading targets.
With a valid existing-schema permit and conforming injected adapters, the full
inner/development/validation execution path below is runnable exactly once.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import result
from .contract import G1_PINS, G4RepairBlocked, PERMIT_SCHEMA, _chronology_contract, _missing_permit, _review_core, _source_binding, _sha256


FitAdapter = Callable[[Mapping[str, Any], Sequence[Mapping[str, Any]]], Mapping[str, Any]]


def review_core() -> dict[str, Any]:
    return _review_core(_chronology_contract(), _source_binding())


def dry_run() -> dict[str, Any]:
    core = review_core(); digest = _sha256(core)
    return {
        "schema_version": "scryglass:real-v1-g4-repair-runner-dry-run:v2",
        "run_status": "blocked_before_target_m0_or_outcome_load",
        "call_order": ["verify_review_core", "verify_registered_2026_support_PASS", "verify_fresh_independent_permit", "blocked_missing_permit_before_protected_loaders"],
        "review_core_sha256": digest,
        "target_loader_calls": 0, "m0_loader_calls": 0, "outcome_loader_calls": 0,
        "fit_availability_loader_calls": 0, "fit_execution_calls": 0,
        "blocker": _missing_permit(digest),
    }


def validate_fresh_independent_permit(permit: Mapping[str, Any], *, expected_review_core_sha256: str) -> None:
    expected = {
        "approved_action": "private_target_m0_load_and_rank_assay",
        "decision": "PASS",
        "final_temporal_holdout_sealed": True,
        "independent_from_runner_and_generator": True,
        "review_core_sha256": expected_review_core_sha256,
        "schema_id": PERMIT_SCHEMA,
    }
    if dict(permit) != expected:
        raise G4RepairBlocked("FRESH_EXISTING_SCHEMA_PERMIT_MISMATCH")


def _metric(y: Sequence[int], p: Sequence[float]) -> dict[str, float]:
    if not y or len(y) != len(p): raise G4RepairBlocked("FIT_OUTPUT_CARDINALITY_MISMATCH")
    clipped=[]
    for value in p:
        if not isinstance(value, (int,float)) or not math.isfinite(float(value)) or not 0.0 < float(value) < 1.0: raise G4RepairBlocked("FIT_OUTPUT_PROBABILITY_INVALID")
        clipped.append(float(value))
    if any(value not in (0,1) for value in y): raise G4RepairBlocked("TARGET_OUTCOME_INVALID")
    return {"log_loss":sum(-(v*math.log(q)+(1-v)*math.log(1-q)) for v,q in zip(y,clipped))/len(y), "brier":sum((q-v)**2 for v,q in zip(y,clipped))/len(y)}


def _fit_once_with_exact_starts(
    fit: FitAdapter, slot: Mapping[str, Any], subset: Sequence[Mapping[str, Any]]
) -> list[float]:
    """Run one frozen slot once and reject any departure from three starts."""

    response = fit(slot, subset)
    if not isinstance(response, Mapping) or set(response) != {"predictions", "optimization_start_count"}:
        raise G4RepairBlocked("FIT_ADAPTER_RESPONSE_SCHEMA_MISMATCH")
    if response["optimization_start_count"] != 3:
        raise G4RepairBlocked("FIT_ADAPTER_START_BUDGET_MISMATCH")
    predictions = response["predictions"]
    if not isinstance(predictions, Sequence) or isinstance(predictions, (str, bytes)):
        raise G4RepairBlocked("FIT_ADAPTER_PREDICTIONS_SCHEMA_MISMATCH")
    return list(predictions)


def _rows(loaders: Mapping[str, Any]) -> list[dict[str, Any]]:
    # These calls occur only after support and permit verification.  The
    # adapter can directly reuse existing `load_authoritative_features`,
    # `load_fit_availability_domain`, and `load_authoritative_target_m0`.
    features = loaders["load_features"](); availability = loaders["load_fit_availability"](); target = loaders["load_target_m0"]()
    if not isinstance(features, Mapping) or not isinstance(availability, Mapping) or not isinstance(target, Sequence): raise G4RepairBlocked("PROTECTED_ADAPTER_SCHEMA_MISMATCH")
    feature_ids, availability_ids = tuple(features.get("game_ids", ())), tuple(availability.get("game_ids", ()))
    rows=[dict(row) for row in target]
    target_ids=tuple(row.get("game_id") for row in rows)
    if not feature_ids or feature_ids != availability_ids or feature_ids != target_ids or len(set(target_ids)) != len(target_ids): raise G4RepairBlocked("EXACT_IDENTITY_JOIN_MISMATCH")
    for row in rows:
        if row.get("split") not in {"train","development","validation"} or row.get("split")=="final" or not isinstance(row.get("calendar_month"),str): raise G4RepairBlocked("SEALED_OR_UNKNOWN_TARGET_REACHED")
        if row.get("y") not in (0,1) or not isinstance(row.get("m0"),(int,float)) or not 0.0<float(row["m0"])<1.0: raise G4RepairBlocked("TARGET_M0_VALUE_INVALID")
    return rows


def _verify_post_support_g1_subset(loaders: Mapping[str, Any]) -> None:
    """Require the exact LPL subset identity only after support+permit gates."""
    observed = loaders["load_g1_lpl_subset_crosscheck"]()
    if dict(observed) != G1_PINS:
        raise G4RepairBlocked("POST_SUPPORT_G1_LPL_SUBSET_PIN_MISMATCH")


def _select_penalty(records: list[tuple[float,dict[str,float]]]) -> float:
    grouped: dict[float,list[dict[str,float]]]={}
    for penalty, metric in records: grouped.setdefault(penalty,[]).append(metric)
    return min(grouped, key=lambda x:(sum(m['log_loss'] for m in grouped[x])/len(grouped[x]),sum(m['brier'] for m in grouped[x])/len(grouped[x]),-x))


def _slice(rows: Sequence[Mapping[str,Any]], slot: Mapping[str,Any]) -> list[Mapping[str,Any]]:
    selected=[row for row in rows if row["split"] == ("train" if slot["stage"]=="inner" else slot["stage"]) and row["calendar_month"]==slot["calendar_month"]]
    if not selected: raise G4RepairBlocked("CHRONOLOGY_SLOT_HAS_NO_EXACT_ROWS")
    return selected


def execute_once_after_permit(permit: Mapping[str, Any], *, loaders: Mapping[str, Any], result_path: Path | None = None) -> dict[str, Any]:
    """Run all 52 frozen slots once with approved injected protected adapters."""
    core=review_core(); core_sha=_sha256(core)
    # `review_core` has already replayed support PASS before this permit gate.
    validate_fresh_independent_permit(permit, expected_review_core_sha256=core_sha)
    required={"load_g1_lpl_subset_crosscheck","load_features","load_fit_availability","load_target_m0","fit"}
    if set(loaders) != required or not all(callable(loaders[key]) for key in required): raise G4RepairBlocked("PROTECTED_ADAPTER_SET_MISMATCH")
    _verify_post_support_g1_subset(loaders)
    rows=_rows(loaders); slots=_chronology_contract()["execution_slots"]; fit: FitAdapter=loaders["fit"]
    ledger=[]; inner={"ally_penalty":[],"enemy_penalty":[]}; dev: dict[int,dict[str,list[float]]]={width:{"y":[],"candidate":[],"m0":[]} for width in (1,2,4,8)}; val={"y":[],"m0":[],"locked":[],"m8":[]}
    lambda_ally=lambda_enemy=None; selected_width=None
    for slot in slots:
        active_slot=dict(slot)
        if slot["stage"]=="validation" and selected_width is None:
            lambda_ally=_select_penalty(inner["ally_penalty"]); lambda_enemy=_select_penalty(inner["enemy_penalty"])
            candidates=[]
            for width,item in dev.items():
                c,m=_metric(item["y"],item["candidate"]),_metric(item["y"],item["m0"])
                if c["log_loss"] < m["log_loss"] and c["brier"] <= m["brier"]: candidates.append(width)
            selected_width=min(candidates) if candidates else None
        if slot["stage"]=="validation":
            active_slot["lambda_ally"],active_slot["lambda_enemy"]=lambda_ally,lambda_enemy
            active_slot["width"]=selected_width if slot["family"]=="locked_candidate" else 8
        active_slot["optimization_start_count"] = 3
        active_slot["extra_bruteforce_reruns"] = False
        subset=_slice(rows,active_slot); predictions=_fit_once_with_exact_starts(fit,active_slot,subset); y=[int(row["y"]) for row in subset]; m0=[float(row["m0"]) for row in subset]
        candidate_metric=_metric(y,predictions); m0_metric=_metric(y,m0)
        ledger.append({**active_slot,"execution_status":"passed","game_ids":[row["game_id"] for row in subset],"candidate_metrics":candidate_metric,"m0_metrics":m0_metric})
        if slot["stage"]=="inner": inner[slot["family"]].append((float(slot["penalty"]),candidate_metric))
        elif slot["stage"]=="development":
            item=dev[int(slot["width"])]; item["y"].extend(y); item["candidate"].extend(predictions); item["m0"].extend(m0)
        else:
            # Validation order is locked candidate then M8 for each month;
            # width is bound only after all development rows were scored.
            if slot["family"]=="locked_candidate": val["y"].extend(y); val["m0"].extend(m0); val["locked"].extend(predictions)
            else: val["m8"].extend(predictions)
    if lambda_ally is None or lambda_enemy is None: raise G4RepairBlocked("INNER_SELECTION_NOT_REACHED")
    development_metrics={str(width):{"candidate":_metric(item["y"],item["candidate"]),"m0":_metric(item["y"],item["m0"])} for width,item in dev.items()}
    validation_metrics={"locked":_metric(val["y"],val["locked"]),"m8":_metric(val["y"],val["m8"]),"m0":_metric(val["y"],val["m0"])}
    accepted=selected_width is not None and validation_metrics["locked"]["log_loss"] <= validation_metrics["m0"]["log_loss"] and validation_metrics["locked"]["log_loss"] <= validation_metrics["m8"]["log_loss"]
    unsigned={"schema_version":result.SCHEMA,"run_status":"accepted" if accepted else "NO_INCREMENTAL_DRAFT_WINNER","selected_model":"locked_width" if accepted else "M0","selected_width":selected_width if accepted else None,"penalties":{"lambda_ally":lambda_ally,"lambda_enemy":lambda_enemy},"ledger":ledger,"metrics":{"development":development_metrics,"validation":validation_metrics},"review_core_sha256":core_sha,"support_first_verified":True,"final_holdout_loaded":False,"claim_ceiling":{"prediction":False,"publication":False,"production":False,"promotion":False,"sota":False}}
    payload={**unsigned,"artifact_sha256":result.sha256(unsigned)}; result.validate_result(payload)
    if result_path is not None: result.write_result(payload,path=result_path,root=Path.cwd())
    return payload
