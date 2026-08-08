"""Run the locked equal-series adaptive Player/Organization protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import multileague_benchmark as benchmark
from . import multileague_development as adapter
from . import multileague_runner as rating
from . import multileague_v2_protocol as parent_protocol
from . import multileague_v2_protocol_equal_series as protocol
from . import multileague_v2_runner as parent_runner


SCHEMA_VERSION = "scryglass:multileague-rating-v2-adaptive-development:v2"
RESULT_SELECTED = "EQUAL_SERIES_ADAPTIVE_CANDIDATE_SELECTED_SEALED_FINAL_UNOPENED"
RESULT_NO_ELIGIBLE = "EQUAL_SERIES_NO_ELIGIBLE_CANDIDATE_SEALED_FINAL_UNOPENED"
SOURCE_LOCATOR = (
    "lol_kills/v2/ratings/player/multileague_v2_runner_equal_series.py"
)
DEFAULT_PROTOCOL = protocol.DEFAULT_OUTPUT
DEFAULT_OUTPUT = Path(
    "data/lol/v2/models/player/multileague-v2/adaptive-development-artifact-v2.json"
)


class EqualSeriesRunnerError(ValueError):
    """The equal-series adaptive replay or artifact failed closed."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise EqualSeriesRunnerError("artifact value is not canonical") from error


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _artifact_sha256(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("artifact_sha256", None)
    return _sha256(_canonical_bytes(body))


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise EqualSeriesRunnerError(f"{label} must be a lowercase SHA-256")
    return value


def _read_object(path: Path) -> tuple[bytes, dict[str, Any]]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EqualSeriesRunnerError(f"cannot read bound artifact: {path}") from error
    if not isinstance(value, dict):
        raise EqualSeriesRunnerError(f"bound artifact is not an object: {path}")
    return raw, value


def _source_record(root: Path, locator: str, kind: str) -> dict[str, Any]:
    try:
        raw = (root / locator).read_bytes()
    except OSError as error:
        raise EqualSeriesRunnerError(f"bound source is unavailable: {locator}") from error
    return {
        "kind": kind,
        "locator": locator,
        "bytes": len(raw),
        "raw_sha256": _sha256(raw),
    }


def _selected_rows(
    rows: Sequence[Mapping[str, Any]],
    series_ids: set[str],
) -> list[Mapping[str, Any]]:
    return [row for row in rows if str(row["series_id"]) in series_ids]


def _metric_bundle(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return parent_runner._metric_bundle(rows)


def _series_macro(bundle: Mapping[str, Any], metric: str) -> float:
    return parent_runner._series_macro_value(bundle, metric)


def _candidate_report(
    replay: parent_runner.ReplayResult,
    rows: Sequence[Mapping[str, Any]],
    windows: Sequence[Sequence[adapter.DevelopmentSeries]],
    manifests: Sequence[Mapping[str, Any]],
    baseline_windows: Mapping[str, Mapping[str, Any]],
    selection_rule: Mapping[str, Any],
) -> dict[str, Any]:
    reports = []
    regrets = []
    pooled_ids: set[str] = set()
    for window, manifest in zip(windows, manifests):
        identities = {series.series_id for series in window}
        pooled_ids.update(identities)
        metrics = _metric_bundle(_selected_rows(rows, identities))
        log_loss = _series_macro(metrics, "log_loss")
        baseline = baseline_windows[str(manifest["window_id"])]
        better_baseline = min(
            float(baseline["player"]["series_macro"]["log_loss"]),
            float(baseline["organization"]["series_macro"]["log_loss"]),
        )
        regret = log_loss - better_baseline
        regrets.append(regret)
        reports.append(
            {
                **dict(manifest),
                "metrics": metrics,
                "better_baseline_log_loss": better_baseline,
                "log_loss_regret": regret,
            }
        )
    pooled = _metric_bundle(_selected_rows(rows, pooled_ids))
    minimum = int(selection_rule["minimum_series_per_scored_window"])
    enough = all(item["metrics"]["overall"]["series"] >= minimum for item in reports)
    worst_regret = max(regrets)
    eligible = enough and worst_regret <= float(
        selection_rule["maximum_allowed_worst_window_log_loss_regret"]
    )
    spec = replay.candidate
    return {
        "candidate": {
            "candidate_id": spec.candidate_id,
            "player_weight_per_role": spec.player_weight_per_role,
            "player_prior_variance": spec.player_prior_variance,
            "player_process_variance_per_day": spec.player_process_variance_per_day,
            "organization_weight": spec.organization_weight,
            "organization_prior_variance": spec.organization_prior_variance,
            "organization_process_variance_per_day": (
                spec.organization_process_variance_per_day
            ),
            "organization_retention_floor": spec.organization_retention_floor,
        },
        "windows": reports,
        "pooled_adaptive_development": pooled,
        "selection_diagnostics": {
            "finite_psd_replay": True,
            "minimum_series_per_window_met": enough,
            "worst_window_log_loss_regret": worst_regret,
            "maximum_allowed_worst_window_log_loss_regret": selection_rule[
                "maximum_allowed_worst_window_log_loss_regret"
            ],
            "eligible": eligible,
        },
        "replay": {
            "applied_series": replay.applied_series,
            "applied_maps": replay.applied_maps,
            "bridge_diagnostics": replay.bridge_diagnostics,
            "roster_transition_diagnostics": replay.roster_transition_diagnostics,
            "posterior_psd": replay.state.assert_psd(),
        },
    }


def build_equal_series_adaptive_artifact(
    root: Path | str = Path("."),
    *,
    built_at: str,
) -> dict[str, Any]:
    repo_root = Path(root)
    protocol_raw, protocol_payload = _read_object(repo_root / DEFAULT_PROTOCOL)
    try:
        protocol_payload = protocol.validate_equal_series_protocol_lock(
            protocol_payload,
            root=repo_root,
        )
    except protocol.EqualSeriesProtocolError as error:
        raise EqualSeriesRunnerError("equal-series protocol lock is invalid") from error
    boundary = protocol_payload["information_boundary"]
    if boundary["sealed_final_targets_accessed"] is not False:
        raise EqualSeriesRunnerError("sealed-final isolation is not intact")
    binding = protocol_payload["input_binding"]
    input_data = adapter.load_multileague_development_input(
        expected_maps_sha256=str(binding["maps_sha256"]),
        expected_players_sha256=str(binding["players_sha256"]),
    )
    windows = protocol.equal_series_windows(input_data)
    manifests = protocol.window_manifests(windows)
    if manifests != protocol_payload["adaptive_development"]["windows"]:
        raise EqualSeriesRunnerError("equal-series membership no longer matches lock")

    player_replay = rating._replay(
        input_data,
        next(
            item
            for item in rating.CANDIDATES
            if item.candidate_id == "random_walk_no_reset"
        ),
    )
    organization_replay = benchmark._organization_replay(
        input_data,
        next(
            item
            for item in benchmark.ORGANIZATION_CANDIDATES
            if item.candidate_id == "organization_random_walk_no_reset"
        ),
    )
    player_rows = benchmark._attach_roster_strata(
        player_replay.predictions,
        organization_replay.predictions,
    )
    organization_rows = organization_replay.predictions
    baseline_windows: dict[str, dict[str, Any]] = {}
    pooled_ids: set[str] = set()
    for window, manifest in zip(windows, manifests):
        identities = {series.series_id for series in window}
        pooled_ids.update(identities)
        baseline_windows[str(manifest["window_id"])] = {
            "player": rating._metric_payload(_selected_rows(player_rows, identities)),
            "organization": rating._metric_payload(
                _selected_rows(organization_rows, identities)
            ),
        }

    selection_rule = protocol_payload["adaptive_development"]["selection_rule"]
    reports = []
    replays: dict[str, parent_runner.ReplayResult] = {}
    specs = [
        parent_runner.CandidateSpec.from_payload(item)
        for item in protocol_payload["candidate_family"]["candidates"]
    ]
    for spec in specs:
        replay = parent_runner.replay_candidate(input_data, spec)
        rows = parent_runner._attach_roster_strata(
            replay.predictions,
            organization_replay.predictions,
        )
        reports.append(
            _candidate_report(
                replay,
                rows,
                windows,
                manifests,
                baseline_windows,
                selection_rule,
            )
        )
        replays[spec.candidate_id] = replay

    eligible = [
        item
        for item in reports
        if item["selection_diagnostics"]["eligible"] is True
    ]
    selected = (
        min(
            eligible,
            key=lambda item: (
                item["selection_diagnostics"]["worst_window_log_loss_regret"],
                _series_macro(item["pooled_adaptive_development"], "log_loss"),
                _series_macro(item["pooled_adaptive_development"], "brier"),
                item["candidate"]["candidate_id"],
            ),
        )
        if eligible
        else None
    )
    selected_id = None if selected is None else selected["candidate"]["candidate_id"]
    posterior = (
        None
        if selected_id is None
        else parent_runner._posterior_payload(replays[selected_id])
    )

    source_locks = [
        _source_record(repo_root, DEFAULT_PROTOCOL.as_posix(), "equal_series_protocol"),
        _source_record(
            repo_root,
            parent_runner.DEFAULT_OUTPUT.as_posix(),
            "failed_parent_adaptive_replay",
        ),
        _source_record(repo_root, adapter.DEFAULT_MAPS_LOCATOR, "warehouse_maps"),
        _source_record(repo_root, adapter.DEFAULT_PLAYERS_LOCATOR, "warehouse_players"),
        _source_record(
            repo_root,
            "lol_kills/v2/ratings/player/multileague_development.py",
            "input_adapter_source",
        ),
        _source_record(repo_root, parent_runner.SOURCE_LOCATOR, "hierarchical_runner_source"),
        _source_record(repo_root, protocol.SOURCE_LOCATOR, "equal_series_protocol_source"),
        _source_record(repo_root, SOURCE_LOCATOR, "equal_series_runner_source"),
    ]
    if source_locks[0]["raw_sha256"] != _sha256(protocol_raw):
        raise EqualSeriesRunnerError("protocol byte binding is inconsistent")

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "built_at": built_at,
        "result_state": RESULT_SELECTED if selected is not None else RESULT_NO_ELIGIBLE,
        "protocol": {
            "locator": DEFAULT_PROTOCOL.as_posix(),
            "raw_sha256": _sha256(protocol_raw),
            "artifact_sha256": protocol_payload["artifact_sha256"],
            "adaptation_disclosure_status": protocol_payload[
                "adaptation_disclosure"
            ]["status"],
        },
        "input": {
            "maps_sha256": input_data.maps_sha256,
            "players_sha256": input_data.players_sha256,
            "cluster_partition_sha256": input_data.cluster_partition_sha256,
            "sealed_selected_metadata_sha256": (
                input_data.sealed_selected_metadata_sha256
            ),
            "sealed_metadata_series": input_data.coverage["sealed_metadata_series"],
            "sealed_metadata_maps": input_data.coverage["sealed_metadata_maps"],
            "sealed_final_targets_accessed": False,
        },
        "window_manifests": manifests,
        "baselines": {
            "windows": [
                {
                    **dict(manifest),
                    **baseline_windows[str(manifest["window_id"])],
                }
                for manifest in manifests
            ],
            "pooled_adaptive_development": {
                "player": _metric_bundle(_selected_rows(player_rows, pooled_ids)),
                "organization": _metric_bundle(
                    _selected_rows(organization_rows, pooled_ids)
                ),
            },
        },
        "candidate_results": reports,
        "selection": {
            "selection_is_adaptive_not_independent_validation": True,
            "eligible_candidate_ids": [
                item["candidate"]["candidate_id"] for item in eligible
            ],
            "selected_candidate_id": selected_id,
            "selection_rank": selection_rule["stage_3_rank"],
            "sealed_final_opened": False,
            "candidate_eligible_for_independently_approved_sealed_evaluation": (
                selected is not None
            ),
        },
        "adaptive_posterior": posterior,
        "sealed_final": {
            "opened": False,
            "targets_accessed": False,
            "series": input_data.coverage["sealed_metadata_series"],
            "maps": input_data.coverage["sealed_metadata_maps"],
            "opening_authority_present": False,
            "gate_passed": False,
        },
        "source_locks": source_locks,
        "claim_ceiling": dict(protocol_payload["claim_ceiling"]),
        "decision_outputs": {
            "match_probability": None,
            "fair_odds": None,
            "expected_value": None,
            "bet_recommendation": None,
        },
    }
    payload["artifact_sha256"] = _artifact_sha256(payload)
    return validate_equal_series_adaptive_artifact(payload, root=repo_root)


def validate_equal_series_adaptive_artifact(
    payload: Mapping[str, Any],
    *,
    root: Path | str = Path("."),
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise EqualSeriesRunnerError("equal-series artifact must be an object")
    value = dict(payload)
    if value.get("schema_version") != SCHEMA_VERSION:
        raise EqualSeriesRunnerError("equal-series artifact schema is unsupported")
    if value.get("result_state") not in {RESULT_SELECTED, RESULT_NO_ELIGIBLE}:
        raise EqualSeriesRunnerError("equal-series artifact result state is invalid")
    declared = _require_sha256(value.get("artifact_sha256"), "artifact_sha256")
    if declared != _artifact_sha256(value):
        raise EqualSeriesRunnerError("equal-series artifact digest mismatch")
    if (
        (value.get("input") or {}).get("sealed_final_targets_accessed") is not False
        or (value.get("sealed_final") or {}).get("opened") is not False
        or (value.get("sealed_final") or {}).get("targets_accessed") is not False
    ):
        raise EqualSeriesRunnerError("equal-series artifact opened sealed outcomes")
    manifests = value.get("window_manifests")
    if (
        not isinstance(manifests, list)
        or len(manifests) != 3
        or [item.get("series") for item in manifests] != [165, 164, 164]
    ):
        raise EqualSeriesRunnerError("equal-series window support changed")
    outputs = value.get("decision_outputs") or {}
    if any(item is not None for item in outputs.values()):
        raise EqualSeriesRunnerError("equal-series artifact contains decision outputs")
    results = value.get("candidate_results")
    if not isinstance(results, list) or len(results) != 12:
        raise EqualSeriesRunnerError("equal-series candidate inventory changed")
    expected_ids = {
        item["candidate_id"] for item in parent_protocol._candidate_payloads()
    }
    actual_ids = {
        (item.get("candidate") or {}).get("candidate_id")
        for item in results
        if isinstance(item, Mapping)
    }
    if actual_ids != expected_ids:
        raise EqualSeriesRunnerError("equal-series candidate identities changed")
    eligible = sorted(
        item["candidate"]["candidate_id"]
        for item in results
        if (item.get("selection_diagnostics") or {}).get("eligible") is True
    )
    selection = value.get("selection") or {}
    if sorted(selection.get("eligible_candidate_ids") or []) != eligible:
        raise EqualSeriesRunnerError("equal-series eligibility inventory changed")
    selected_id = selection.get("selected_candidate_id")
    if selected_id is not None and selected_id not in eligible:
        raise EqualSeriesRunnerError("selected candidate is not eligible")
    if (selected_id is None) != (value.get("adaptive_posterior") is None):
        raise EqualSeriesRunnerError("selection and posterior disagree")
    if (selected_id is None) != (value.get("result_state") == RESULT_NO_ELIGIBLE):
        raise EqualSeriesRunnerError("selection and result state disagree")
    if selected_id is not None:
        posterior = value.get("adaptive_posterior") or {}
        if posterior.get("candidate_id") != selected_id:
            raise EqualSeriesRunnerError("posterior candidate binding changed")
        for team in posterior.get("teams") or []:
            components = (team or {}).get("components") or {}
            for name in ("lineup_synergy", "team_policy"):
                component = components.get(name) or {}
                if (
                    component.get("status") != "UNAVAILABLE"
                    or component.get("posterior_mean_logit") is not None
                    or component.get("posterior_sd_logit") is not None
                ):
                    raise EqualSeriesRunnerError(
                        "unavailable team component became numeric"
                    )

    records = value.get("source_locks")
    if not isinstance(records, list) or len(records) != 8:
        raise EqualSeriesRunnerError("source-lock inventory changed")
    repo_root = Path(root)
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise EqualSeriesRunnerError("source-lock record is malformed")
        locator = record.get("locator")
        if not isinstance(locator, str) or not locator or locator in seen:
            raise EqualSeriesRunnerError("source-lock locator is invalid")
        seen.add(locator)
        expected = _require_sha256(record.get("raw_sha256"), f"{locator} raw_sha256")
        try:
            raw = (repo_root / locator).read_bytes()
        except OSError as error:
            raise EqualSeriesRunnerError(f"bound source is unavailable: {locator}") from error
        if len(raw) != record.get("bytes") or _sha256(raw) != expected:
            raise EqualSeriesRunnerError(f"bound source drifted: {locator}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--built-at", required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    payload = build_equal_series_adaptive_artifact(built_at=args.built_at)
    raw_sha256 = parent_runner.write_adaptive_artifact_no_clobber(
        args.out,
        payload,
    )
    print(
        json.dumps(
            {
                "output": str(args.out),
                "raw_sha256": raw_sha256,
                "artifact_sha256": payload["artifact_sha256"],
                "result_state": payload["result_state"],
                "eligible_candidate_ids": payload["selection"][
                    "eligible_candidate_ids"
                ],
                "selected_candidate_id": payload["selection"][
                    "selected_candidate_id"
                ],
                "sealed_final_opened": payload["sealed_final"]["opened"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT",
    "EqualSeriesRunnerError",
    "RESULT_NO_ELIGIBLE",
    "RESULT_SELECTED",
    "SCHEMA_VERSION",
    "build_equal_series_adaptive_artifact",
    "validate_equal_series_adaptive_artifact",
]
