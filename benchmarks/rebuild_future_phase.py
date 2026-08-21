"""Rebuild the research-only OE phase artifacts in an isolated output root.

The command verifies the frozen source receipt, source-file bytes, accepted
census, and Leaguepedia crosswalk before it reads the phase inputs.  It writes
only to an explicit directory below ``/private/tmp``.  The result remains
development-only and keeps the proxy-series authority blocker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from lol_kills.research.future_phase_curve import (
    FuturePhaseCurveError,
    _phase_partition_map_frame,
    build_strict_prior_team_features,
    evaluate_phase_curve,
    fit_phase_curve,
    phase_series_assignment_sha256,
    prepare_phase_frame,
    verify_source_receipt_artifact,
)
from lol_kills.v2.tierlists.accepted_census import canonical_game_ids, identity_sha256


FEATURE_COLUMNS = (
    "prior_form_earnedgold_diff",
    "prior_form_dpm_diff",
    "prior_form_visionscore_diff",
)
DEFAULT_OUTPUT_ROOT = Path("/private/tmp/scryglass-future-phase-rebuild-v5")
OUTPUT_FILENAMES = (
    "future-phase-candidate.json",
    "future-phase-evaluation.json",
    "run-receipt.json",
)


class PhaseRebuildError(FuturePhaseCurveError):
    """The isolated phase rebuild cannot be bound safely."""


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PhaseRebuildError(f"{label} is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PhaseRebuildError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise PhaseRebuildError(f"{label} must be a JSON object")
    return value


def _safe_relative_path(base: Path, locator: str, label: str) -> Path:
    relative = Path(locator)
    if relative.is_absolute() or ".." in relative.parts:
        raise PhaseRebuildError(f"{label} locator is unsafe")
    path = (base.resolve() / relative).resolve()
    try:
        path.relative_to(base.resolve())
    except ValueError as error:
        raise PhaseRebuildError(f"{label} locator escapes its root") from error
    if path.is_symlink() or not path.is_file():
        raise PhaseRebuildError(f"{label} is missing or unsafe: {path}")
    return path


def _locate_source_file(
    locator: str,
    *,
    freeze_root: Path,
    source_root: Path,
    label: str,
) -> Path:
    """Resolve a receipt locator against the frozen root, then its source root."""

    for base in (freeze_root, source_root):
        try:
            return _safe_relative_path(base, locator, label)
        except PhaseRebuildError:
            continue
    raise PhaseRebuildError(f"{label} cannot be found from the frozen source roots")


def verify_source_bundle(
    receipt_path: Path,
    *,
    freeze_root: Path,
    source_root: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Verify the durable source receipt and every referenced source file."""

    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise PhaseRebuildError("source receipt is missing or unsafe")
    source_receipt_file_sha256 = _sha256_path(receipt_path)
    receipt = _load_json(receipt_path, "source receipt")
    verify_source_receipt_artifact(
        {
            "locator": str(receipt_path),
            "bytes": receipt_path.stat().st_size,
            "sha256": source_receipt_file_sha256,
            "source_receipt_sha256": receipt.get("receipt_sha256"),
        },
        expected_source_game_count=int(receipt.get("source_game_count") or -1),
        expected_source_identity_sha256=str(receipt.get("source_identity_sha256") or ""),
        expected_source_as_of=str(receipt.get("source_as_of") or ""),
    )
    source_files = receipt.get("source_files")
    if not isinstance(source_files, Mapping):
        raise PhaseRebuildError("source receipt source_files are missing")
    resolved: dict[str, Path] = {}
    for label, record in source_files.items():
        if not isinstance(record, Mapping):
            raise PhaseRebuildError(f"source receipt file binding is invalid: {label}")
        locator = record.get("locator") or record.get("path")
        if not isinstance(locator, str) or not locator.strip():
            raise PhaseRebuildError(f"source receipt file locator is missing: {label}")
        path = _locate_source_file(
            locator,
            freeze_root=freeze_root,
            source_root=source_root,
            label=f"source file {label}",
        )
        if int(record.get("bytes") or -1) != path.stat().st_size:
            raise PhaseRebuildError(f"source file bytes changed: {label}")
        if str(record.get("sha256") or "").lower() != _sha256_path(path):
            raise PhaseRebuildError(f"source file hash changed: {label}")
        resolved[str(label)] = path
    for label in ("maps", "teams"):
        if label not in resolved:
            raise PhaseRebuildError(f"source receipt has no {label} frame binding")
    census_path = resolved.get("accepted_census")
    if census_path is not None:
        census = _load_json(census_path, "accepted census")
        accepted = tuple(str(value) for value in census.get("game_ids", ()))
        if tuple(canonical_game_ids(accepted)) != accepted:
            raise PhaseRebuildError("accepted census IDs are not canonical")
        if int(census.get("game_count") or -1) != len(accepted):
            raise PhaseRebuildError("accepted census count changed")
        if str(census.get("source_identity_sha256") or "") != identity_sha256(accepted):
            raise PhaseRebuildError("accepted census identity changed")
        if list(accepted) != list(receipt.get("accepted_game_ids") or ()):
            raise PhaseRebuildError("accepted census IDs differ from source receipt")
    return receipt, resolved


def _frame_game_ids(frame: pd.DataFrame, label: str) -> pd.Series:
    for column in ("game_uid", "gameid", "game_id"):
        if column in frame.columns:
            values = frame[column].astype("string")
            if values.isna().any() or values.str.strip().eq("").any():
                raise PhaseRebuildError(f"{label} has an empty game ID")
            return values.astype(str)
    raise PhaseRebuildError(f"{label} has no game ID column")


def select_accepted_rows(
    frame: pd.DataFrame,
    *,
    accepted_ids: Sequence[str],
    declared_extra_ids: Sequence[str],
    label: str,
) -> pd.DataFrame:
    """Filter declared source extras and require complete accepted coverage."""

    values = _frame_game_ids(frame, label)
    accepted = set(str(value) for value in accepted_ids)
    extras = set(str(value) for value in declared_extra_ids)
    observed = set(values)
    unknown = sorted(observed - accepted - extras)
    if unknown:
        raise PhaseRebuildError(f"{label} contains undeclared game IDs: {unknown[:3]}")
    selected = frame.loc[values.isin(accepted)].copy()
    selected_ids = set(_frame_game_ids(selected, label))
    missing = sorted(accepted - selected_ids)
    if missing:
        raise PhaseRebuildError(f"{label} is missing accepted games: {missing[:3]}")
    return selected.reset_index(drop=True)


def _complete_team_map_ids(teams: pd.DataFrame, accepted_ids: Sequence[str]) -> set[str]:
    team_id_column = next(
        (name for name in ("teamid", "team_id") if name in teams.columns), None
    )
    if team_id_column is None:
        raise PhaseRebuildError("teams has no stable team identity column")
    ids = _frame_game_ids(teams, "teams")
    work = teams.copy()
    work["_game_id"] = ids
    work["_team_id_present"] = work[team_id_column].notna() & work[team_id_column].astype(str).str.strip().ne("")
    accepted = set(str(value) for value in accepted_ids)
    counts = work.loc[work["_game_id"].isin(accepted)].groupby("_game_id", sort=False)[
        "_team_id_present"
    ].sum()
    return {str(game_id) for game_id, count in counts.items() if int(count) == 2}


def build_phase_frame(
    maps: pd.DataFrame,
    teams: pd.DataFrame,
    *,
    accepted_ids: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Prepare checkpoint targets and strict-prior team form features."""

    complete_ids = _complete_team_map_ids(teams, accepted_ids)
    accepted = set(str(value) for value in accepted_ids)
    team_ids = _frame_game_ids(teams, "teams")
    complete_teams = teams.loc[team_ids.isin(complete_ids)].copy()
    history = build_strict_prior_team_features(
        complete_teams,
        metric_columns=("earnedgold", "dpm", "visionscore"),
    )
    prepared = prepare_phase_frame(maps, teams)
    result = prepared.merge(
        history,
        on=["game_uid", "date"],
        how="left",
        validate="one_to_one",
    )
    if set(result["game_uid"].astype(str)) != accepted:
        raise PhaseRebuildError("prepared phase frame changed the accepted census")
    return result, {
        "accepted_game_count": len(accepted),
        "complete_team_maps": len(complete_ids),
        "incomplete_team_identity_maps": len(accepted - complete_ids),
        "team_form_feature_rows": int(history["game_uid"].nunique()),
    }


def _receipt_reference(path: Path, receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "locator": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_path(path),
        "source_receipt_sha256": str(receipt["receipt_sha256"]),
    }


def _partition_payload(artifact: Mapping[str, Any]) -> dict[str, Any]:
    partition = artifact.get("series_partition")
    if not isinstance(partition, Mapping):
        raise PhaseRebuildError("phase artifact has no verified series partition")
    required = (
        "source",
        "mapping_sha256",
        "crosswalk_sha256",
        "artifact_sha256",
        "receipt_sha256",
        "receipt_file_sha256",
        "eligible_game_count",
        "eligible_identity_sha256",
        "eligible_game_ids",
        "eligible_assignment_sha256",
        "reference_game_count",
        "reference_assignment_sha256",
        "reference_assignment_match",
        "authoritative",
        "proxy_authority_blocker",
    )
    missing = [name for name in required if name not in partition]
    if missing:
        raise PhaseRebuildError("phase series partition is incomplete: " + ", ".join(missing))
    if artifact.get("cross_model_series_partition", {}).get("status") != "comparable":
        raise PhaseRebuildError("phase artifact did not bind a comparable shared partition")
    if not partition.get("reference_assignment_match"):
        raise PhaseRebuildError("phase artifact did not match the reference assignments")
    return dict(partition)


def _build_rating_reference_partition_frame(
    maps: pd.DataFrame,
    *,
    source_receipt: Mapping[str, Any],
    crosswalk_path: Path,
    crosswalk_receipt_path: Path,
    crosswalk_receipt_file_sha256: str,
) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    """Build the shared partition on the complete source map frame.

    The rating mapper is the reference implementation.  The phase model
    receives its verified full-frame assignments and only selects eligible
    rows by game ID.
    """

    from lol_kills.research.future_value_rating import (
        _map_model_frame,
        bind_verified_leaguepedia_series_crosswalk,
    )

    raw_maps = _phase_partition_map_frame(maps)
    bound_maps = bind_verified_leaguepedia_series_crosswalk(
        raw_maps,
        crosswalk_path=crosswalk_path,
        receipt_path=crosswalk_receipt_path,
        source_receipt=source_receipt,
        expected_receipt_file_sha256=crosswalk_receipt_file_sha256,
    )
    reference = _map_model_frame(
        bound_maps,
        verified_source_receipt_sha256=str(source_receipt["receipt_sha256"]),
        verified_source_receipt=source_receipt,
        verified_crosswalk_receipt_file_sha256=crosswalk_receipt_file_sha256,
    )
    reference_ids = reference["game_id"].astype(str)
    if reference_ids.duplicated().any() or set(reference_ids) != set(raw_maps["game_id"].astype(str)):
        raise PhaseRebuildError("rating reference partition changed source map IDs")
    eligible_ids = set(str(value) for value in source_receipt["model_eligible_game_ids"])
    eligible = reference.loc[reference_ids.isin(eligible_ids)].copy()
    if set(eligible["game_id"].astype(str)) != eligible_ids:
        raise PhaseRebuildError("rating reference partition is missing eligible maps")
    digest = phase_series_assignment_sha256(eligible, game_column="game_id")
    audit = dict(reference.attrs.get("series_cluster_audit") or {})
    return reference, digest, {
        "reference_game_count": int(len(reference)),
        "reference_promoted_game_count": int(
            reference["series_id"].astype(str).str.startswith("leaguepedia:").sum()
        ),
        "reference_eligible_promoted_game_count": int(
            eligible["series_id"].astype(str).str.startswith("leaguepedia:").sum()
        ),
        "reference_assignment_sha256": digest,
        "reference_audit": audit,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    raw = json.dumps(value, indent=2, ensure_ascii=True, sort_keys=True, allow_nan=False).encode("utf-8")
    path.write_bytes(raw)
    return {"locator": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def rebuild_phase_artifacts(
    *,
    receipt_path: Path,
    freeze_root: Path,
    source_root: Path,
    crosswalk_path: Path,
    crosswalk_receipt_path: Path,
    crosswalk_receipt_file_sha256: str,
    output_root: Path,
    max_transfer_groups: int = 50,
) -> dict[str, Any]:
    """Build one candidate and one eligible-census evaluation."""

    tmp_root = Path("/private/tmp").resolve()
    output_root = output_root.resolve()
    try:
        output_root.relative_to(tmp_root)
    except ValueError as error:
        raise PhaseRebuildError("output root must be below /private/tmp") from error
    output_root.mkdir(parents=True, exist_ok=True)
    receipt, source_files = verify_source_bundle(
        receipt_path,
        freeze_root=freeze_root,
        source_root=source_root,
    )
    accepted_ids = tuple(str(value) for value in receipt["accepted_game_ids"])
    extras = receipt.get("source_extra_game_ids")
    if not isinstance(extras, Mapping):
        extras = {}
    maps_source = pd.read_parquet(source_files["maps"])
    teams_source = pd.read_parquet(source_files["teams"])
    receipt_file_sha256 = _sha256_path(receipt_path)
    crosswalk_receipt_file_sha256 = str(crosswalk_receipt_file_sha256).lower()
    if _sha256_path(crosswalk_receipt_path) != crosswalk_receipt_file_sha256:
        raise PhaseRebuildError("crosswalk receipt file hash changed")
    reference_started = time.perf_counter()
    reference_partition_frame, reference_assignment_sha256, reference_stats = (
        _build_rating_reference_partition_frame(
            maps_source,
            source_receipt=receipt,
            crosswalk_path=crosswalk_path,
            crosswalk_receipt_path=crosswalk_receipt_path,
            crosswalk_receipt_file_sha256=crosswalk_receipt_file_sha256,
        )
    )
    reference_elapsed = time.perf_counter() - reference_started
    maps = select_accepted_rows(
        maps_source,
        accepted_ids=accepted_ids,
        declared_extra_ids=extras.get("maps", ()),
        label="maps",
    )
    teams = select_accepted_rows(
        teams_source,
        accepted_ids=accepted_ids,
        declared_extra_ids=extras.get("teams", ()),
        label="teams",
    )
    frame, frame_stats = build_phase_frame(maps, teams, accepted_ids=accepted_ids)
    eligible_ids = set(str(value) for value in receipt["model_eligible_game_ids"])
    eligible_frame = frame.loc[
        frame["game_uid"].astype(str).isin(eligible_ids)
    ].reset_index(drop=True)
    if set(eligible_frame["game_uid"].astype(str)) != eligible_ids:
        raise PhaseRebuildError("eligible phase frame does not match source receipt")
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always", RuntimeWarning)
        fit = fit_phase_curve(
            eligible_frame,
            source_receipt=receipt,
            source_receipt_path=receipt_path,
            source_receipt_file_sha256=receipt_file_sha256,
            feature_columns=FEATURE_COLUMNS,
            crosswalk_path=crosswalk_path,
            crosswalk_receipt_path=crosswalk_receipt_path,
            crosswalk_receipt_file_sha256=crosswalk_receipt_file_sha256,
            series_partition_bound_reference_frame=reference_partition_frame,
            series_partition_assignment_sha256=reference_assignment_sha256,
        )
        evaluation = evaluate_phase_curve(
            eligible_frame,
            source_receipt=receipt,
            source_receipt_path=receipt_path,
            source_receipt_file_sha256=receipt_file_sha256,
            feature_columns=FEATURE_COLUMNS,
            n_splits=3,
            cluster_column=None,
            max_transfer_groups=max_transfer_groups,
            crosswalk_path=crosswalk_path,
            crosswalk_receipt_path=crosswalk_receipt_path,
            crosswalk_receipt_file_sha256=crosswalk_receipt_file_sha256,
            series_partition_bound_reference_frame=reference_partition_frame,
            series_partition_assignment_sha256=reference_assignment_sha256,
        )
    elapsed = time.perf_counter() - started
    numeric_warnings = sorted(
        {
            str(item.message)
            for item in captured_warnings
            if any(
                token in str(item.message).casefold()
                for token in ("overflow", "divide by zero", "invalid value")
            )
        }
    )
    fit_partition = _partition_payload(fit)
    evaluation_partition = _partition_payload(evaluation)
    if {
        key: fit_partition[key]
        for key in (
            "source",
            "mapping_sha256",
            "crosswalk_sha256",
            "artifact_sha256",
            "receipt_sha256",
            "receipt_file_sha256",
            "eligible_game_count",
            "eligible_identity_sha256",
            "eligible_assignment_sha256",
            "reference_game_count",
            "reference_assignment_sha256",
            "reference_assignment_match",
        )
    } != {
        key: evaluation_partition[key]
        for key in (
            "source",
            "mapping_sha256",
            "crosswalk_sha256",
            "artifact_sha256",
            "receipt_sha256",
            "receipt_file_sha256",
            "eligible_game_count",
            "eligible_identity_sha256",
            "eligible_assignment_sha256",
            "reference_game_count",
            "reference_assignment_sha256",
            "reference_assignment_match",
        )
    }:
        raise PhaseRebuildError("fit and evaluation series partitions differ")
    receipt_reference = _receipt_reference(receipt_path, receipt)
    source_payload = {
        "source_as_of": receipt["source_as_of"],
        "source_game_count": int(receipt["source_game_count"]),
        "source_identity_sha256": str(receipt["source_identity_sha256"]),
        "accepted_game_count": len(accepted_ids),
        "accepted_game_ids": list(accepted_ids),
        "accepted_identity_sha256": identity_sha256(accepted_ids),
        "model_eligible_game_count": int(receipt["model_eligible_game_count"]),
        "model_eligible_game_ids": list(receipt["model_eligible_game_ids"]),
        "model_eligible_identity_sha256": str(receipt["model_eligible_identity_sha256"]),
        "source_receipt_sha256": str(receipt["receipt_sha256"]),
        "source_receipt_file_sha256": receipt_file_sha256,
        "source_receipt_artifact": receipt_reference,
        "transport": fit["source_transport"],
        "source_lineage": fit["source_lineage"],
        "series_partition": fit_partition,
        "series_partition_reference": {
            **reference_stats,
            "assignment_difference_count": 0,
            "binding_wall_seconds": reference_elapsed,
        },
    }
    candidate_blockers = [
        "fold-internal fitted weights have no independent promotion receipt",
        "authoritative whole-series identity is unavailable for retained proxy clusters",
        "current output is not wired into public ratings or public phase forecasts",
    ]
    if fit_partition["proxy_authority_blocker"]:
        candidate_blockers.append("authoritative_series_id_missing_proxy_cluster_used")
    if int(evaluation.get("fold_count") or 0) < 3:
        candidate_blockers.extend(
            [
                "complete_chronological_evaluation_missing",
                "requested three chronological folds produced fewer than three valid folds",
            ]
        )
    transfer_group_counts = {
        column: int(eligible_frame[column].astype("string").nunique(dropna=False))
        for column in ("region", "patch")
        if column in eligible_frame.columns
    }
    if any(count > int(max_transfer_groups) for count in transfer_group_counts.values()):
        candidate_blockers.append("regional_and_patch_transfer_evaluation_bounded")
    if numeric_warnings:
        candidate_blockers.append("numerical_overflow_warning_during_fit_or_side_swap")
    candidate = {
        "schema_version": fit["schema_version"],
        "model_version": fit["model_version"],
        "authority": "development_only",
        "status": "research_candidate_only",
        "source": source_payload,
        "source_receipt_sha256": receipt["receipt_sha256"],
        "source_receipt_artifact": receipt_reference,
        "accepted_game_count": len(accepted_ids),
        "model_eligible_game_count": int(receipt["model_eligible_game_count"]),
        "series_identity": fit["series_identity"],
        "cross_model_series_partition": fit["cross_model_series_partition"],
        "series_partition_source": fit_partition["source"],
        "series_partition_mapping_sha256": fit_partition["mapping_sha256"],
        "series_partition_crosswalk_sha256": fit_partition["crosswalk_sha256"],
        "series_partition_artifact_sha256": fit_partition["artifact_sha256"],
        "series_partition_receipt_sha256": fit_partition["receipt_sha256"],
        "series_partition_receipt_file_sha256": fit_partition["receipt_file_sha256"],
        "series_partition_eligible_game_count": fit_partition["eligible_game_count"],
        "series_partition_eligible_identity_sha256": fit_partition["eligible_identity_sha256"],
        "series_partition_eligible_assignment_sha256": fit_partition[
            "eligible_assignment_sha256"
        ],
        "series_partition_reference_game_count": fit_partition[
            "reference_game_count"
        ],
        "series_partition_reference_assignment_sha256": fit_partition[
            "reference_assignment_sha256"
        ],
        "series_partition_proxy_authority_blocker": fit_partition["proxy_authority_blocker"],
        "feature_columns": list(FEATURE_COLUMNS),
        "feature_family": fit["feature_family"],
        "curve_definitions": fit["curve_definitions"],
        "evaluation_scope": {
            "accepted_game_count": len(accepted_ids),
            "model_eligible_game_count": len(eligible_ids),
            "date_start": str(frame["date"].min()),
            "date_end": str(frame["date"].max()),
            "complete_team_maps": frame_stats["complete_team_maps"],
            "incomplete_team_identity_maps": frame_stats["incomplete_team_identity_maps"],
            "team_form_feature_rows": frame_stats["team_form_feature_rows"],
            "cross_model_partition": fit["cross_model_series_partition"],
        },
        "fit": {
            **fit,
            "model_rows": int(len(eligible_frame)),
            "complete_team_maps": int(len(eligible_frame)),
            "incomplete_team_identity_maps": frame_stats["incomplete_team_identity_maps"],
            "accepted_complete_team_maps": frame_stats["complete_team_maps"],
        },
        "blockers": sorted(set(candidate_blockers)),
        "numerical_warnings": numeric_warnings,
    }
    evaluation_output = {
        **evaluation,
        "status": "research_candidate_only",
        "authority": "development_only",
        "source_receipt_artifact": receipt_reference,
        "accepted_game_count": len(accepted_ids),
        "model_eligible_game_count": len(eligible_ids),
        "transfer_group_limit": int(max_transfer_groups),
        "transfer_group_counts": transfer_group_counts,
        "cross_model_series_partition": evaluation["cross_model_series_partition"],
        "series_partition_source": evaluation_partition["source"],
        "series_partition_mapping_sha256": evaluation_partition["mapping_sha256"],
        "series_partition_crosswalk_sha256": evaluation_partition["crosswalk_sha256"],
        "series_partition_artifact_sha256": evaluation_partition["artifact_sha256"],
        "series_partition_receipt_sha256": evaluation_partition["receipt_sha256"],
        "series_partition_receipt_file_sha256": evaluation_partition["receipt_file_sha256"],
        "series_partition_eligible_game_count": evaluation_partition["eligible_game_count"],
        "series_partition_eligible_identity_sha256": evaluation_partition["eligible_identity_sha256"],
        "series_partition_eligible_assignment_sha256": evaluation_partition[
            "eligible_assignment_sha256"
        ],
        "series_partition_reference_game_count": evaluation_partition[
            "reference_game_count"
        ],
        "series_partition_reference_assignment_sha256": evaluation_partition[
            "reference_assignment_sha256"
        ],
        "series_partition_proxy_authority_blocker": evaluation_partition["proxy_authority_blocker"],
        "blockers": sorted(set(candidate_blockers)),
        "numerical_warnings": numeric_warnings,
    }
    candidate_reference = _write_json(output_root / OUTPUT_FILENAMES[0], candidate)
    evaluation_reference = _write_json(output_root / OUTPUT_FILENAMES[1], evaluation_output)
    run_receipt = {
        "schema_version": "scryglass:future-phase-rebuild-receipt:v1",
        "status": "research_only",
        "authority": {
            "development_only": True,
            "public": False,
            "promotion": False,
            "deployment": False,
        },
        "source": source_payload,
        "crosswalk": {
            "artifact": {
                "locator": str(crosswalk_path),
                "bytes": crosswalk_path.stat().st_size,
                "sha256": _sha256_path(crosswalk_path),
            },
            "receipt_file_sha256": crosswalk_receipt_file_sha256,
            "receipt_locator": str(crosswalk_receipt_path),
        },
        "inputs": {
            "maps_rows_read": int(len(maps_source)),
            "teams_rows_read": int(len(teams_source)),
            "accepted_map_rows": int(len(maps)),
            "accepted_team_rows": int(len(teams)),
            **frame_stats,
            "series_partition_reference": {
                **reference_stats,
                "assignment_difference_count": 0,
            },
            "evaluation_rows": int(len(eligible_frame)),
            "transfer_group_limit": int(max_transfer_groups),
            "transfer_group_counts": transfer_group_counts,
        },
        "partition": fit_partition,
        "numerical_warnings": numeric_warnings,
        "outputs": {
            "candidate": candidate_reference,
            "evaluation": evaluation_reference,
        },
        "timing": {
            "reference_partition_wall_seconds": reference_elapsed,
            "fit_and_evaluation_wall_seconds": elapsed,
            "total_wall_seconds": reference_elapsed + elapsed,
        },
    }
    _write_json(output_root / OUTPUT_FILENAMES[2], run_receipt)
    return {
        "output_root": str(output_root),
        "candidate": candidate,
        "evaluation": evaluation_output,
        "run_receipt": run_receipt,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--freeze-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--crosswalk", type=Path, required=True)
    parser.add_argument("--crosswalk-receipt", type=Path, required=True)
    parser.add_argument("--crosswalk-receipt-file-sha256", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-transfer-groups", type=int, default=50)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_transfer_groups < 0:
        raise SystemExit("--max-transfer-groups must be non-negative")
    result = rebuild_phase_artifacts(
        receipt_path=args.receipt,
        freeze_root=args.freeze_root,
        source_root=args.source_root,
        crosswalk_path=args.crosswalk,
        crosswalk_receipt_path=args.crosswalk_receipt,
        crosswalk_receipt_file_sha256=args.crosswalk_receipt_file_sha256,
        output_root=args.output_root,
        max_transfer_groups=args.max_transfer_groups,
    )
    run = result["run_receipt"]
    print(json.dumps({
        "output_root": result["output_root"],
        "source_game_count": run["source"]["source_game_count"],
        "model_eligible_game_count": run["source"]["model_eligible_game_count"],
        "evaluation_rows": run["inputs"]["evaluation_rows"],
        "cross_model_series_partition": run["partition"],
        "timing": run["timing"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
