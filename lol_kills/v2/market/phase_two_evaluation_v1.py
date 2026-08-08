"""Frozen phase-two market evaluation; contains no outcome-opening permission."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import betano_br_quote_adapter_v2 as quote_v2
from . import match_winner_future_protocol_v1 as protocol_source
from . import phase_one_evaluation_v1 as phase_one
from . import phase_two_stopping_snapshot_v1 as snapshot_source
from .match_winner_future_protocol_registry_v1 import (
    validate_registered_match_winner_future_protocol_v1,
)


ROOT = Path(__file__).resolve().parents[3]
SOURCE_LOCATOR = "lol_kills/v2/market/phase_two_evaluation_v1.py"
OUTCOME_SCHEMA_VERSION = "scryglass:phase-two-sealed-outcome-cohort:v1"
RESULT_SCHEMA_VERSION = "scryglass:phase-two-market-evaluation-result:v1"
OUTCOME_PREFIX = PurePosixPath(
    "data/lol/v2/evaluation/match-winner-market-v1/phase-two/outcomes-v1"
)
OUTCOME_EVIDENCE_PREFIX = PurePosixPath(
    "data/lol/v2/evaluation/match-winner-market-v1/phase-two/outcome-evidence-v1"
)
RESULT_PREFIX = PurePosixPath(
    "data/lol/v2/evaluation/match-winner-market-v1/phase-two/results-v1"
)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_806
CONFIDENCE_INTERVAL = (0.025, 0.975)
ECE_BINS = 10
AUTHORITY = {
    "probability_authority": False,
    "fair_odds_authority": False,
    "expected_value_authority": False,
    "recommendation_authority": False,
    "transaction_authority": False,
    "stake_authority": False,
    "betting_authority": False,
}
CLAIM_CEILING = (
    "One-time evaluation candidate under the frozen phase-two protocol. Even a "
    "pass requires independent result registration and separate market authority; "
    "quoted shadow return is not executable return and no stake is authorized."
)


class PhaseTwoEvaluationError(RuntimeError):
    """The snapshot, outcomes, scoring, bootstrap, or gate reconciliation failed."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PhaseTwoEvaluationError("phase-two evaluation is not canonical") from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _timestamp(value: Any, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise PhaseTwoEvaluationError(f"{field} must be RFC-3339") from exc
    if parsed.tzinfo is None:
        raise PhaseTwoEvaluationError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PhaseTwoEvaluationError(f"{field} must be nonempty")
    return value.strip()


def _sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise PhaseTwoEvaluationError(f"{field} must be a lowercase SHA-256")
    return value


def _snapshot(
    *, root: Path, locator_value: str, environment: Mapping[str, str]
) -> tuple[str, bytes, dict[str, Any]]:
    locator = phase_one._locator(
        locator_value, snapshot_source.OUTPUT_PREFIX, "snapshot_locator"
    )
    raw = phase_one._read_regular(root, locator, "phase-two stopping snapshot")
    try:
        checked = snapshot_source.validate_phase_two_stopping_snapshot_v1(
            phase_one._strict_object(raw, "phase-two stopping snapshot"),
            root=root,
            environment=environment,
        )
    except Exception as exc:
        raise PhaseTwoEvaluationError("phase-two snapshot is invalid") from exc
    if checked["support"]["support_met"] is not True:
        raise PhaseTwoEvaluationError("phase-two snapshot did not meet support")
    return locator, raw, checked


def validate_outcome_cohort_v1(
    payload: Mapping[str, Any], *, snapshot: Mapping[str, Any], root: Path = ROOT
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise PhaseTwoEvaluationError("outcome cohort must be an object")
    value = dict(payload)
    if set(value) != {
        "schema_version", "created_at_utc", "snapshot_artifact_sha256",
        "rows", "artifact_sha256",
    }:
        raise PhaseTwoEvaluationError("outcome cohort structure changed")
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if value.get("artifact_sha256") != _canonical_sha256(unsigned):
        raise PhaseTwoEvaluationError("outcome cohort hash changed")
    if value.get("schema_version") != OUTCOME_SCHEMA_VERSION or value.get("snapshot_artifact_sha256") != snapshot["artifact_sha256"]:
        raise PhaseTwoEvaluationError("outcome cohort identity changed")
    created = _timestamp(value.get("created_at_utc"), "outcomes.created_at")
    expected = {
        (row["event_id"], row["game_number"]): row for row in snapshot["entries"]
    }
    rows = value.get("rows")
    if not isinstance(rows, list):
        raise PhaseTwoEvaluationError("outcome rows must be a list")
    row_keys = {
        "event_id", "series_id", "game_number", "actual_map_start_utc",
        "winning_side", "source_system", "source_record_id",
        "source_revision_id", "source_observed_at_utc", "evidence_locator",
        "evidence_raw_sha256",
    }
    seen: set[tuple[str, int]] = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != row_keys:
            raise PhaseTwoEvaluationError("outcome row structure changed")
        game = row.get("game_number")
        if isinstance(game, bool) or not isinstance(game, int) or game < 1:
            raise PhaseTwoEvaluationError("outcome game number changed")
        identity = (_nonempty(row.get("event_id"), "event_id"), game)
        expected_row = expected.get(identity)
        if (
            expected_row is None
            or identity in seen
            or row.get("series_id") != expected_row["series_id"]
            or row.get("actual_map_start_utc") != expected_row["actual_map_start_utc"]
            or row.get("winning_side") not in {"blue", "red"}
        ):
            raise PhaseTwoEvaluationError("outcome identity or side changed")
        seen.add(identity)
        for field in ("source_system", "source_record_id", "source_revision_id"):
            _nonempty(row.get(field), f"outcome.{field}")
        start = _timestamp(row["actual_map_start_utc"], "actual_map_start")
        observed = _timestamp(row["source_observed_at_utc"], "source_observed_at")
        if observed <= start or created < observed:
            raise PhaseTwoEvaluationError("outcome evidence timing changed")
        evidence_locator = phase_one._locator(
            row.get("evidence_locator"), OUTCOME_EVIDENCE_PREFIX, "evidence_locator"
        )
        evidence_raw = phase_one._read_regular(root, evidence_locator, "outcome evidence")
        if _sha256(evidence_raw) != _sha(row.get("evidence_raw_sha256"), "evidence_raw_sha256"):
            raise PhaseTwoEvaluationError("outcome evidence hash changed")
    if seen != set(expected):
        raise PhaseTwoEvaluationError("outcome cohort is not the exact snapshot cohort")
    ordered = sorted(rows, key=lambda row: (row["actual_map_start_utc"], row["event_id"], row["game_number"]))
    if rows != ordered:
        raise PhaseTwoEvaluationError("outcome rows are not deterministically ordered")
    return value


def _evaluation_rows(
    *, snapshot: Mapping[str, Any], outcomes: Mapping[str, Any], root: Path,
    environment: Mapping[str, str]
) -> list[dict[str, Any]]:
    outcome_by_id = {(row["event_id"], row["game_number"]): row for row in outcomes["rows"]}
    rows: list[dict[str, Any]] = []
    for entry in snapshot["entries"]:
        if not entry["qualified_quote"]:
            continue
        _locator, _raw, completed = snapshot_source._completion(
            root=root,
            locator_value=entry["completion_locator"],
            environment=environment,
        )
        quote_raw = phase_one._read_regular(root, completed["quote_binding"]["locator"], "Betano v2 quote")
        quote = quote_v2.validate_betano_map_winner_quote_v2(
            phase_one._strict_object(quote_raw, "Betano v2 quote"),
            root=root,
            environment=environment,
        )
        _, _, probability = quote_v2._probability(
            root=root,
            locator_value=quote["event_probability_v2_binding"]["locator"],
            environment=environment,
        )
        prices = quote["frozen_v1_transport_quote"]["generic_quote_receipt"]["prices"]
        event = probability["event"]
        blue_odds = float(prices[event["selection"]])
        red_odds = float(prices[event["opposing_selection"]])
        blue_implied = 1.0 / blue_odds
        red_implied = 1.0 / red_odds
        outcome = outcome_by_id[(entry["event_id"], entry["game_number"])]
        rows.append(
            {
                "event_id": entry["event_id"],
                "series_id": entry["series_id"],
                "game_number": entry["game_number"],
                "actual_map_start_utc": entry["actual_map_start_utc"],
                "league": entry["league"],
                "patch": entry["patch"],
                "roster_change": entry["roster_change_stratum"] != "UNCHANGED",
                "sparse_or_new": bool(entry["sparse_or_new_champion_map"]),
                "blue_win": int(outcome["winning_side"] == "blue"),
                "combined": float(probability["probability"]),
                "rating_only": float(probability["calculation"]["rating_only_comparator"]["probability_blue"]),
                "market": blue_implied / (blue_implied + red_implied),
                "blue_odds": blue_odds,
                "red_odds": red_odds,
                "shadow_signal": entry["shadow_signal"],
            }
        )
    return rows


def _primary_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    reports = {
        comparator: {
            metric: phase_one._delta_interval(
                rows,
                candidate_key="combined",
                comparator_key=comparator,
                metric=metric,
                replicates=BOOTSTRAP_REPLICATES,
                seed=phase_one._derived_seed(BOOTSTRAP_SEED, f"primary|{comparator}|{metric}"),
            )
            for metric in ("log_loss", "brier_score")
        }
        for comparator in ("market", "rating_only")
    }
    all_pass = all(
        report["point_delta"] is not None
        and report["point_delta"] <= 0.0
        and report["upper_95"] is not None
        and report["upper_95"] <= 0.0
        for comparator in reports.values()
        for report in comparator.values()
    )
    strict_market = any(
        reports["market"][metric]["upper_95"] is not None
        and reports["market"][metric]["upper_95"] < 0.0
        for metric in ("log_loss", "brier_score")
    )
    return {
        "comparisons": reports,
        "all_point_and_upper_bounds_nonpositive": all_pass,
        "at_least_one_market_upper_bound_strictly_negative": strict_market,
        "passed": all_pass and strict_market,
    }


def _calibration_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    probabilities = np.asarray([row["combined"] for row in rows], dtype=float)
    outcomes = np.asarray([row["blue_win"] for row in rows], dtype=float)
    series = sorted({str(row["series_id"]) for row in rows})
    by_series = {
        series_id: np.asarray([i for i, row in enumerate(rows) if row["series_id"] == series_id], dtype=int)
        for series_id in series
    }
    point_fit = phase_one._calibration_fit(probabilities, outcomes)
    point_ece = phase_one._ece(probabilities, outcomes)
    rng = np.random.default_rng(phase_one._derived_seed(BOOTSTRAP_SEED, "calibration"))
    intercepts: list[float] = []
    slopes: list[float] = []
    eces: list[float] = []
    failures = 0
    for _ in range(BOOTSTRAP_REPLICATES):
        selected = rng.integers(0, len(series), size=len(series))
        indices = np.concatenate([by_series[series[index]] for index in selected])
        fitted = phase_one._calibration_fit(probabilities[indices], outcomes[indices])
        if fitted is None:
            failures += 1
            continue
        intercepts.append(fitted[0]); slopes.append(fitted[1])
        eces.append(phase_one._ece(probabilities[indices], outcomes[indices]))
    complete = point_fit is not None and failures == 0 and len(intercepts) == BOOTSTRAP_REPLICATES
    intercept_interval = [float(np.quantile(intercepts, 0.025)), float(np.quantile(intercepts, 0.975))] if complete else None
    slope_interval = [float(np.quantile(slopes, 0.025)), float(np.quantile(slopes, 0.975))] if complete else None
    ece_upper = float(np.quantile(eces, 0.975)) if complete else None
    strata: dict[str, list[Mapping[str, Any]]] = {
        **{f"league:{league}": [row for row in rows if row["league"] == league] for league in ("LCS", "LEC", "LCK", "LPL")},
        **{f"patch:{patch}": [row for row in rows if row["patch"] == patch] for patch in sorted({row["patch"] for row in rows})},
        "roster_change": [row for row in rows if row["roster_change"]],
        "sparse_or_new": [row for row in rows if row["sparse_or_new"]],
        "international": [row for row in rows if row["league"] in {"MSI", "EWC"}],
    }
    stratum_reports = {
        name: {
            "maps": len(subset),
            "log_loss_point_delta_vs_market": None if not subset else float(np.mean([
                phase_one._loss(row["combined"], row["blue_win"], "log_loss")
                - phase_one._loss(row["market"], row["blue_win"], "log_loss")
                for row in subset
            ])),
        }
        for name, subset in strata.items()
    }
    supported_nonharm = all(
        report["maps"] == 0
        or report["log_loss_point_delta_vs_market"] <= 0.02
        for report in stratum_reports.values()
    )
    passed = (
        complete
        and point_ece <= 0.03
        and ece_upper is not None and ece_upper <= 0.05
        and intercept_interval is not None and intercept_interval[0] <= 0.0 <= intercept_interval[1]
        and slope_interval is not None and slope_interval[0] <= 1.0 <= slope_interval[1]
        and supported_nonharm
    )
    return {
        "equal_frequency_bins": ECE_BINS,
        "bootstrap_failures": failures,
        "point_ece": point_ece,
        "ece_upper_95": ece_upper,
        "calibration_intercept_interval": intercept_interval,
        "calibration_slope_interval": slope_interval,
        "strata": stratum_reports,
        "supported_strata_nonharm_passed": supported_nonharm,
        "passed": passed,
    }


def _capture_report(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    entries = snapshot["entries"]
    latencies = [
        row["prediction_to_response_seconds"]
        for row in entries if row["prediction_to_response_seconds"] is not None
    ]
    p95 = float(np.quantile(latencies, 0.95)) if latencies else None
    failures = snapshot["support"]["failure_codes"]
    extractor_mismatches = failures.get("EXTRACTION_OR_REPLAY_FAILURE", 0)
    binding_mismatches = failures.get("SOURCE_IDENTITY_OR_BINDING_MISMATCH", 0)
    after_start = snapshot["support"]["quote_received_after_map_start_maps"]
    within_five_seconds = snapshot["support"]["quote_response_too_late_maps"]
    passed = (
        snapshot["support"]["quote_coverage"] >= 0.80
        and p95 is not None and p95 <= 30.0
        and after_start == 0
        and extractor_mismatches == 0 and binding_mismatches == 0
    )
    return {
        "quote_coverage": snapshot["support"]["quote_coverage"],
        "prediction_to_quote_response_p95_seconds": p95,
        "quote_received_after_map_start_count": after_start,
        "quote_received_after_or_within_five_seconds_before_start_count": within_five_seconds,
        "extractor_replay_mismatch_count": extractor_mismatches,
        "team_or_map_binding_mismatch_count": binding_mismatches,
        "passed": passed,
    }


def _shadow_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: (item["actual_map_start_utc"], item["event_id"])):
        signal = row["shadow_signal"]
        selection = signal.get("selected_side") if signal else None
        if selection is None:
            continue
        blue_selected = selection == signal["blue_selection"]
        won = bool(row["blue_win"]) if blue_selected else not bool(row["blue_win"])
        odds = signal["blue_decimal_odds"] if blue_selected else signal["red_decimal_odds"]
        raw_profit = float(odds) - 1.0 if won else -1.0
        profit = raw_profit * 0.99 if raw_profit > 0.0 else raw_profit
        selected.append({"series_id": row["series_id"], "profit": profit, "won": won})
    point_roi = float(np.mean([row["profit"] for row in selected])) if selected else None
    series = sorted({row["series_id"] for row in selected})
    sums = np.asarray([sum(row["profit"] for row in selected if row["series_id"] == series_id) for series_id in series], dtype=float)
    counts = np.asarray([sum(row["series_id"] == series_id for row in selected) for series_id in series], dtype=float)
    if series:
        rng = np.random.default_rng(phase_one._derived_seed(BOOTSTRAP_SEED, "shadow-roi"))
        draws = rng.integers(0, len(series), size=(BOOTSTRAP_REPLICATES, len(series)))
        rois = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
        interval = [float(np.quantile(rois, 0.025)), float(np.quantile(rois, 0.975))]
    else:
        interval = None
    bankroll = 0.0; peak = 0.0; maximum_drawdown = 0.0; losing = 0; longest = 0
    for row in selected:
        bankroll += row["profit"]; peak = max(peak, bankroll)
        maximum_drawdown = max(maximum_drawdown, peak - bankroll)
        losing = losing + 1 if row["profit"] < 0 else 0
        longest = max(longest, losing)
    passed = (
        len(selected) >= 100
        and point_roi is not None and point_roi > 0.0
        and interval is not None and interval[0] > 0.0
    )
    return {
        "qualifying_maps": len(selected),
        "execution_haircut_fraction_of_positive_profit": 0.01,
        "point_roi_after_haircut": point_roi,
        "roi_interval_95": interval,
        "maximum_drawdown_units": maximum_drawdown,
        "longest_losing_run": longest,
        "quoted_shadow_return_is_not_executable_return": True,
        "stake_or_bankroll_size_authorized": False,
        "passed": passed,
    }


def evaluate_phase_two_v1(
    *, snapshot_locator: str, outcome_cohort_raw: bytes,
    outcome_cohort_locator: str, opening_authority_binding: Mapping[str, Any],
    run_id: str, root: Path = ROOT,
    environment: Mapping[str, str] = os.environ,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    evaluated = clock()
    if not isinstance(evaluated, datetime) or evaluated.tzinfo is None:
        raise PhaseTwoEvaluationError("evaluation clock must be timezone-aware")
    evaluated = evaluated.astimezone(timezone.utc)
    locator, snapshot_raw, snapshot = _snapshot(
        root=root, locator_value=snapshot_locator, environment=environment
    )
    outcome_locator = phase_one._locator(
        outcome_cohort_locator, OUTCOME_PREFIX, "outcome_cohort_locator"
    )
    outcomes = validate_outcome_cohort_v1(
        phase_one._strict_object(outcome_cohort_raw, "phase-two outcome cohort"),
        snapshot=snapshot,
        root=root,
    )
    if evaluated < _timestamp(outcomes["created_at_utc"], "outcomes.created_at"):
        raise PhaseTwoEvaluationError("evaluation predates the sealed outcome cohort")
    rows = _evaluation_rows(
        snapshot=snapshot, outcomes=outcomes, root=root, environment=environment
    )
    primary = _primary_report(rows)
    calibration = _calibration_report(rows)
    capture = _capture_report(snapshot)
    shadow = _shadow_report(rows)
    passed = primary["passed"] and calibration["passed"] and capture["passed"] and shadow["passed"]
    protocol = validate_registered_match_winner_future_protocol_v1(root=root)
    payload: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "result_state": "PHASE_TWO_MARKET_GATES_PASSED_PENDING_INDEPENDENT_REGISTRATION" if passed else "PHASE_TWO_MARKET_GATE_FAILED_TERMINALLY",
        "run_id": _nonempty(run_id, "run_id"),
        "evaluated_at_utc": evaluated.isoformat(),
        "opening_authority_binding": dict(opening_authority_binding),
        "inputs": {
            "snapshot_locator": locator,
            "snapshot_raw_sha256": _sha256(snapshot_raw),
            "snapshot_artifact_sha256": snapshot["artifact_sha256"],
            "outcome_cohort_locator": outcome_locator,
            "outcome_cohort_raw_sha256": _sha256(outcome_cohort_raw),
            "outcome_cohort_artifact_sha256": outcomes["artifact_sha256"],
            "otherwise_eligible_maps": len(snapshot["entries"]),
            "qualified_quoted_maps": len(rows),
            "series": len({row["series_id"] for row in rows}),
        },
        "evaluation_contract": protocol["evaluation"],
        "bootstrap": {
            "method": "paired_series_cluster_bootstrap",
            "replicates": BOOTSTRAP_REPLICATES,
            "base_seed": BOOTSTRAP_SEED,
            "confidence_level": 0.95,
        },
        "primary_probabilistic_gates": primary,
        "calibration_gates": calibration,
        "capture_gates": capture,
        "shadow_policy_gates": shadow,
        "phase_two_market_gates_passed": passed,
        "independently_registered": False,
        "authority": dict(AUTHORITY),
        "claim_ceiling": CLAIM_CEILING,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return validate_phase_two_evaluation_result_v1(payload)


def validate_phase_two_evaluation_result_v1(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise PhaseTwoEvaluationError("evaluation result must be an object")
    value = dict(payload)
    if set(value) != {
        "schema_version", "result_state", "run_id", "evaluated_at_utc",
        "opening_authority_binding", "inputs", "evaluation_contract", "bootstrap",
        "primary_probabilistic_gates", "calibration_gates", "capture_gates",
        "shadow_policy_gates", "phase_two_market_gates_passed",
        "independently_registered", "authority", "claim_ceiling", "artifact_sha256",
    }:
        raise PhaseTwoEvaluationError("evaluation result structure changed")
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if value.get("artifact_sha256") != _canonical_sha256(unsigned):
        raise PhaseTwoEvaluationError("evaluation result hash changed")
    _nonempty(value.get("run_id"), "run_id"); _timestamp(value.get("evaluated_at_utc"), "evaluated_at")
    binding = value.get("opening_authority_binding")
    if not isinstance(binding, Mapping) or set(binding) != {"authority_id", "authority_raw_sha256", "opening_marker_locator"}:
        raise PhaseTwoEvaluationError("opening authority binding changed")
    _nonempty(binding.get("authority_id"), "authority_id"); _sha(binding.get("authority_raw_sha256"), "authority_raw_sha256")
    marker = PurePosixPath(_nonempty(binding.get("opening_marker_locator"), "opening_marker_locator"))
    marker_prefix = PurePosixPath("data/lol/v2/evaluation/match-winner-market-v1/phase-two/outcome-opening-markers-v1")
    if marker.is_absolute() or tuple(marker.parts[:len(marker_prefix.parts)]) != marker_prefix.parts or marker.suffix != ".json":
        raise PhaseTwoEvaluationError("opening marker locator changed")
    inputs = value.get("inputs") or {}
    if set(inputs) != {
        "snapshot_locator", "snapshot_raw_sha256", "snapshot_artifact_sha256",
        "outcome_cohort_locator", "outcome_cohort_raw_sha256",
        "outcome_cohort_artifact_sha256", "otherwise_eligible_maps",
        "qualified_quoted_maps", "series",
    }:
        raise PhaseTwoEvaluationError("evaluation inputs changed")
    phase_one._locator(inputs["snapshot_locator"], snapshot_source.OUTPUT_PREFIX, "snapshot_locator")
    phase_one._locator(inputs["outcome_cohort_locator"], OUTCOME_PREFIX, "outcome_cohort_locator")
    for key, item in inputs.items():
        if key.endswith("sha256"): _sha(item, f"inputs.{key}")
    if value.get("evaluation_contract") != protocol_source._evaluation_contract():
        raise PhaseTwoEvaluationError("evaluation contract changed")
    if value.get("bootstrap") != {
        "method": "paired_series_cluster_bootstrap", "replicates": BOOTSTRAP_REPLICATES,
        "base_seed": BOOTSTRAP_SEED, "confidence_level": 0.95,
    }:
        raise PhaseTwoEvaluationError("bootstrap contract changed")
    reports = [value.get("primary_probabilistic_gates"), value.get("calibration_gates"), value.get("capture_gates"), value.get("shadow_policy_gates")]
    if any(not isinstance(report, Mapping) for report in reports):
        raise PhaseTwoEvaluationError("evaluation reports are missing")
    passed = all(report.get("passed") is True for report in reports)
    if value.get("phase_two_market_gates_passed") is not passed:
        raise PhaseTwoEvaluationError("evaluation gate result does not reconcile")
    expected_state = "PHASE_TWO_MARKET_GATES_PASSED_PENDING_INDEPENDENT_REGISTRATION" if passed else "PHASE_TWO_MARKET_GATE_FAILED_TERMINALLY"
    if value.get("schema_version") != RESULT_SCHEMA_VERSION or value.get("result_state") != expected_state:
        raise PhaseTwoEvaluationError("evaluation result identity changed")
    if value.get("independently_registered") is not False:
        raise PhaseTwoEvaluationError("evaluation result self-registered")
    if value.get("authority") != AUTHORITY or value.get("claim_ceiling") != CLAIM_CEILING:
        raise PhaseTwoEvaluationError("evaluation result exceeds authority")
    return value


def write_no_clobber(path: Path, payload: Mapping[str, Any]) -> str:
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise PhaseTwoEvaluationError(f"refusing to replace evaluation result: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise PhaseTwoEvaluationError(f"refusing to replace evaluation result: {path}") from exc
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
    return _sha256(raw)


__all__ = [
    "BOOTSTRAP_REPLICATES", "BOOTSTRAP_SEED", "OUTCOME_PREFIX",
    "OUTCOME_SCHEMA_VERSION", "RESULT_PREFIX", "RESULT_SCHEMA_VERSION",
    "SOURCE_LOCATOR", "PhaseTwoEvaluationError", "evaluate_phase_two_v1",
    "validate_outcome_cohort_v1", "validate_phase_two_evaluation_result_v1",
    "write_no_clobber",
]
