"""Verify the frozen future-value source on an isolated research runner.

This entry point prepares source receipts only. It does not fit, promote, or
publish a player or team model. A later training stage must consume the
verified source receipt and satisfy the frozen evaluation protocol.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from lol_kills.etl.source_keys import canonical_source_game_key
from lol_kills.research.future_value_rating import (
    FutureValueSourceError,
    RATING_VARIANT_ORDER,
    RatingVariant,
    bind_accepted_future_value_source,
    bind_verified_leaguepedia_series_crosswalk,
    evaluate_future_value,
    _map_model_frame,
    _phase_partition_evidence,
    rating_variant_config_receipt,
    validate_future_value_source_receipt_payload,
    write_source_receipt,
)
from lol_kills.v2.tierlists.accepted_census import (
    canonical_game_ids,
    census_payload,
    identity_sha256,
)


SCHEMA_VERSION = "scryglass:future-value-research-run:v1"
MODEL_RUNTIME_SCHEMA_VERSION = "scryglass:future-value-model-runtime:v1"
FREEZE_SCHEMA_VERSION = "scryglass:future-value-source-freeze:v1"
DEFAULT_FREEZE = Path(
    "data/lol/v2/evaluation/future-value-source-freeze-20260820.json"
)


class FutureValueTrainingError(RuntimeError):
    """The cloud research source does not match the frozen contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise FutureValueTrainingError("research receipt is not canonical JSON") from error


def _load_freeze(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FutureValueTrainingError("future-value source freeze is missing or unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FutureValueTrainingError("future-value source freeze cannot be read") from error
    if not isinstance(value, dict) or value.get("schema_version") != FREEZE_SCHEMA_VERSION:
        raise FutureValueTrainingError("future-value source freeze schema is invalid")
    if value.get("source_mode") != "oe_only":
        raise FutureValueTrainingError("future-value source freeze is not OE-only")
    if not isinstance(value.get("unfiltered_source_game_count"), int) or not isinstance(
        value.get("unfiltered_source_identity_sha256"), str
    ) or re.fullmatch(r"[0-9a-f]{64}", value["unfiltered_source_identity_sha256"], re.I) is None:
        raise FutureValueTrainingError("future-value source freeze raw identity is invalid")
    authority = value.get("authority")
    if not isinstance(authority, Mapping) or authority.get("research_only") is not True:
        raise FutureValueTrainingError("future-value source freeze authority is invalid")
    if any(bool(flag) for name, flag in authority.items() if name != "research_only"):
        raise FutureValueTrainingError("future-value source freeze grants public authority")
    accepted = value.get("accepted_census")
    if not isinstance(accepted, Mapping):
        raise FutureValueTrainingError("future-value source freeze accepted census is invalid")
    if not isinstance(accepted.get("source_game_count"), int) or not isinstance(
        accepted.get("source_identity_sha256"), str
    ) or re.fullmatch(r"[0-9a-f]{64}", accepted["source_identity_sha256"], re.I) is None:
        raise FutureValueTrainingError("future-value source freeze accepted identity is invalid")
    eligible = value.get("model_eligible_census")
    if not isinstance(eligible, Mapping):
        raise FutureValueTrainingError("future-value source freeze model census is missing")
    if not isinstance(eligible.get("game_count"), int) or not isinstance(
        eligible.get("source_identity_sha256"), str
    ) or re.fullmatch(r"[0-9a-f]{64}", eligible["source_identity_sha256"], re.I) is None:
        raise FutureValueTrainingError("future-value source freeze model identity is invalid")
    bridge_sources = value.get("oe_bridge_sources")
    if not isinstance(bridge_sources, list) or not bridge_sources:
        raise FutureValueTrainingError("future-value source freeze bridge sources are missing")
    reference_receipt = value.get("reference_source_receipt_sha256")
    if not isinstance(reference_receipt, str) or re.fullmatch(
        r"[0-9a-f]{64}", reference_receipt, re.I
    ) is None:
        raise FutureValueTrainingError("future-value source freeze receipt reference is invalid")
    receipt_path = value.get("source_receipt_path")
    receipt_file_hash = value.get("source_receipt_file_sha256")
    if (
        not isinstance(receipt_path, str)
        or not receipt_path.strip()
        or not isinstance(receipt_file_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", receipt_file_hash, re.I) is None
    ):
        raise FutureValueTrainingError("future-value durable source receipt binding is invalid")
    return value


def verify_annual_sources(
    annual_root: Path,
    freeze: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Require the exact annual OE bytes named by the freeze."""

    raw_sources = freeze.get("oe_annual_sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise FutureValueTrainingError("source freeze has no annual OE sources")
    records: dict[str, dict[str, Any]] = {}
    for item in raw_sources:
        if not isinstance(item, Mapping):
            raise FutureValueTrainingError("annual OE source record is invalid")
        name = str(item.get("name") or "")
        year = str(item.get("year") or "")
        expected_hash = str(item.get("raw_sha256") or "")
        expected_bytes = item.get("bytes")
        if not name or not year or len(expected_hash) != 64 or not isinstance(expected_bytes, int):
            raise FutureValueTrainingError("annual OE source binding is incomplete")
        path = annual_root / name
        if not path.is_file() or path.is_symlink():
            raise FutureValueTrainingError(f"annual OE source is missing or unsafe: {name}")
        actual_bytes = path.stat().st_size
        actual_hash = _sha256(path)
        if actual_bytes != expected_bytes or actual_hash != expected_hash:
            raise FutureValueTrainingError(f"annual OE source changed: {name}")
        records[year] = {
            "year": int(year),
            "locator": name,
            "bytes": actual_bytes,
            "sha256": actual_hash,
        }
    return dict(sorted(records.items()))


def verify_bridge_sources(
    bridge_root: Path,
    freeze: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Require the exact cached OE API bridge bytes named by the freeze."""

    raw_sources = freeze.get("oe_bridge_sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise FutureValueTrainingError("source freeze bridge records are invalid")
    records: dict[str, dict[str, Any]] = {}
    for item in raw_sources:
        if not isinstance(item, Mapping):
            raise FutureValueTrainingError("bridge source record is invalid")
        name = str(item.get("name") or "")
        expected_hash = str(item.get("raw_sha256") or "")
        expected_bytes = item.get("bytes")
        if not name or Path(name).name != name or len(expected_hash) != 64 or not isinstance(expected_bytes, int):
            raise FutureValueTrainingError("bridge source binding is incomplete")
        path = bridge_root / name
        if not path.is_file() or path.is_symlink():
            raise FutureValueTrainingError(f"bridge source is missing or unsafe: {name}")
        actual_bytes = path.stat().st_size
        actual_hash = _sha256(path)
        if actual_bytes != expected_bytes or actual_hash != expected_hash:
            raise FutureValueTrainingError(f"bridge source changed: {name}")
        records[name] = {
            "locator": name,
            "bytes": actual_bytes,
            "sha256": actual_hash,
        }
    return dict(sorted(records.items()))


def _game_ids(frame: pd.DataFrame) -> tuple[str, ...]:
    if "game_uid" in frame.columns:
        fallback = frame["gameid"] if "gameid" in frame.columns else None
        values = [
            canonical_source_game_key(
                value,
                fallback.loc[index] if fallback is not None else None,
            )
            for index, value in frame["game_uid"].items()
        ]
    elif "gameid" in frame.columns:
        values = [canonical_source_game_key(value) for value in frame["gameid"]]
    else:
        raise FutureValueTrainingError("OE map source has no game identity")
    return canonical_game_ids(values)


def frozen_census(
    maps: pd.DataFrame,
    freeze: Mapping[str, Any],
) -> dict[str, Any]:
    """Build and verify the exact accepted census from frozen source rules."""

    contract = freeze.get("accepted_census")
    if not isinstance(contract, Mapping):
        raise FutureValueTrainingError("source freeze has no accepted census")
    expected_count = contract.get("source_game_count")
    expected_identity = contract.get("source_identity_sha256")
    unfiltered_count = freeze.get("unfiltered_source_game_count")
    if not isinstance(unfiltered_count, int) or unfiltered_count < int(expected_count or 0):
        raise FutureValueTrainingError("frozen unfiltered source count is invalid")
    raw_ids = _game_ids(maps)
    if len(raw_ids) != unfiltered_count:
        raise FutureValueTrainingError("unfiltered source census count changed")
    raw_identity = freeze.get("unfiltered_source_identity_sha256")
    try:
        raw_census = census_payload(raw_ids)
    except ValueError as error:
        raise FutureValueTrainingError("frozen accepted census is empty") from error
    if raw_census["source_identity_sha256"] != raw_identity:
        raise FutureValueTrainingError("unfiltered source census identity changed")
    excluded = set(canonical_game_ids(contract.get("excluded_game_ids") or ()))
    if not excluded or not excluded.issubset(set(raw_ids)):
        raise FutureValueTrainingError("frozen source exclusions are missing from the raw census")
    accepted = tuple(game_id for game_id in raw_ids if game_id not in excluded)
    try:
        filtered_census = census_payload(accepted)
    except ValueError as error:
        raise FutureValueTrainingError("frozen accepted census is empty") from error
    if (
        filtered_census["game_count"] != expected_count
        or filtered_census["source_identity_sha256"] != expected_identity
    ):
        raise FutureValueTrainingError("frozen accepted census identity changed")
    return filtered_census


def _source_file_records(
    oe_root: Path,
    annual_records: Mapping[str, Mapping[str, Any]],
    bridge_records: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    records = {
        f"annual_{year}": dict(record)
        for year, record in sorted(annual_records.items())
    }
    for name, record in sorted((bridge_records or {}).items()):
        records[f"bridge_{name}"] = dict(record)
    for label, name in (
        ("maps", "maps.parquet"),
        ("players", "oe_player_games.parquet"),
        ("teams", "oe_team_games.parquet"),
    ):
        path = oe_root / name
        if not path.is_file() or path.is_symlink():
            raise FutureValueTrainingError(f"normalized OE source is missing or unsafe: {name}")
        records[label] = {
            "locator": f"warehouse/parquet/{name}",
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    return records


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def verify_research_source(
    *,
    annual_root: Path,
    oe_root: Path,
    freeze_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Verify annual bytes, normalized rows, census, and model eligibility."""

    freeze = _load_freeze(freeze_path)
    annual_records = verify_annual_sources(annual_root, freeze)
    paths = {
        "maps": oe_root / "maps.parquet",
        "players": oe_root / "oe_player_games.parquet",
        "teams": oe_root / "oe_team_games.parquet",
    }
    bridge_records = verify_bridge_sources(oe_root.parent, freeze)
    source_files = _source_file_records(oe_root, annual_records, bridge_records)
    maps = pd.read_parquet(paths["maps"])
    census = frozen_census(maps, freeze)
    try:
        source = bind_accepted_future_value_source(
            maps,
            pd.read_parquet(paths["players"]),
            pd.read_parquet(paths["teams"]),
            census=census,
            source_as_of=freeze["source_as_of"],
            source_files=source_files,
        )
    except FutureValueSourceError as error:
        raise FutureValueTrainingError(str(error)) from error

    reference_receipt = str(freeze["reference_source_receipt_sha256"])
    if source.receipt.get("receipt_sha256") != reference_receipt:
        raise FutureValueTrainingError("source receipt identity changed")

    eligible = freeze.get("model_eligible_census")
    if isinstance(eligible, Mapping) and (
        source.receipt.get("model_eligible_game_count") != eligible.get("game_count")
        or source.receipt.get("model_eligible_identity_sha256")
        != eligible.get("source_identity_sha256")
    ):
        raise FutureValueTrainingError("model-eligible census identity changed")

    source_receipt_path = output_root / "future-value-source-receipt.json"
    write_source_receipt(source_receipt_path, source)
    run: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "source_verified_model_unfitted",
        "source_as_of": source.receipt["source_as_of"],
        "source_game_count": source.receipt["source_game_count"],
        "source_identity_sha256": source.receipt["source_identity_sha256"],
        "accepted_game_ids": source.receipt["accepted_game_ids"],
        "model_eligible_game_count": source.receipt["model_eligible_game_count"],
        "model_eligible_identity_sha256": source.receipt[
            "model_eligible_identity_sha256"
        ],
        "source_receipt_sha256": source.receipt["receipt_sha256"],
        "freeze": {
            "locator": str(freeze_path),
            "bytes": freeze_path.stat().st_size,
            "sha256": _sha256(freeze_path),
        },
        "annual_sources": annual_records,
        "bridge_sources": bridge_records,
        "artifacts": {
            "source_receipt": {
                "locator": source_receipt_path.name,
                "bytes": source_receipt_path.stat().st_size,
                "sha256": _sha256(source_receipt_path),
            }
        },
        "blockers": [
            "fitted_metric_weights_missing",
            "fold_internal_rank_3_atoms_missing",
            "complete_chronological_evaluation_missing",
            "current_rating_comparison_missing",
            "downstream_integration_missing",
            "independent_promotion_receipt_missing",
        ],
        "authority": {
            "research_only": True,
            "public_player_rating": False,
            "public_team_rating": False,
            "public_probability": False,
            "odds": False,
            "expected_value": False,
            "recommendation": False,
            "betting": False,
            "promotion": False,
            "deployment": False,
        },
    }
    run["receipt_sha256"] = hashlib.sha256(_canonical_bytes(run)).hexdigest()
    _write_json(output_root / "future-value-research-run.json", run)
    return run


def verify_annual_only(
    *,
    annual_root: Path,
    freeze_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    freeze = _load_freeze(freeze_path)
    records = verify_annual_sources(annual_root, freeze)
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "annual_sources_verified",
        "source_as_of": freeze["source_as_of"],
        "annual_sources": records,
        "authority": {"research_only": True, "deployment": False},
    }
    receipt["receipt_sha256"] = hashlib.sha256(_canonical_bytes(receipt)).hexdigest()
    _write_json(output_root / "future-value-annual-verification.json", receipt)
    return receipt


def _load_json_mapping(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FutureValueTrainingError(f"{label} is missing or unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FutureValueTrainingError(f"{label} cannot be read") from error
    if not isinstance(value, dict):
        raise FutureValueTrainingError(f"{label} is not a JSON object")
    return value


def _phase_hash(value: Any, field: str) -> str:
    result = str(value or "").lower()
    if re.fullmatch(r"[0-9a-f]{64}", result) is None:
        raise FutureValueTrainingError(f"{field} hash is invalid")
    return result


def _phase_partition_fields(
    payload: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    try:
        return _phase_partition_evidence(payload, label=label)
    except FutureValueSourceError as error:
        raise FutureValueTrainingError(str(error)) from error


def _phase_file_record(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> dict[str, Any]:
    if (
        not path.is_absolute()
        or ".." in path.parts
        or path.is_symlink()
        or not path.is_file()
    ):
        raise FutureValueTrainingError(f"{label} is missing or unsafe")
    expected = _phase_hash(expected_sha256, f"expected {label}")
    actual = _sha256(path)
    if actual != expected:
        raise FutureValueTrainingError(f"{label} hash changed")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": actual}


def _phase_reference_game_ids(source_receipt: Mapping[str, Any]) -> tuple[str, ...]:
    accepted = tuple(canonical_game_ids(source_receipt["accepted_game_ids"]))
    extras_value = source_receipt.get("source_extra_game_ids")
    if not isinstance(extras_value, Mapping):
        extras_value = {}
    raw_extra_ids = extras_value.get("maps") or ()
    if not isinstance(raw_extra_ids, (list, tuple)):
        raise FutureValueTrainingError("phase reference extra game IDs are invalid")
    extras = tuple(canonical_game_ids(raw_extra_ids))
    reference_ids = tuple(canonical_game_ids((*accepted, *extras)))
    if len(reference_ids) != len(accepted) + len(extras):
        raise FutureValueTrainingError("phase reference census contains duplicate IDs")
    return reference_ids


def _verify_phase_source_receipt_file(
    source_receipt: Mapping[str, Any],
    *,
    source_receipt_path: Path,
    source_receipt_file_sha256: str,
) -> dict[str, Any]:
    record = _phase_file_record(
        source_receipt_path,
        expected_sha256=source_receipt_file_sha256,
        label="phase source receipt",
    )
    payload = _load_json_mapping(source_receipt_path, "phase source receipt")
    _validate_source_receipt_mapping(payload)
    if dict(payload) != dict(source_receipt):
        raise FutureValueTrainingError("phase source receipt payload changed")
    return record


def _build_phase_partition_binding(
    artifact_path: Path,
    receipt_path: Path,
    *,
    artifact_sha256: str,
    receipt_file_sha256: str,
    source_receipt: Mapping[str, Any],
    source_receipt_path: Path,
    source_receipt_file_sha256: str,
    artifact_kind: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify phase files before constructing the rating evaluator binding."""

    artifact_file = _phase_file_record(
        artifact_path,
        expected_sha256=artifact_sha256,
        label="phase artifact",
    )
    receipt_file = _phase_file_record(
        receipt_path,
        expected_sha256=receipt_file_sha256,
        label="phase run receipt",
    )
    source_receipt_file = _verify_phase_source_receipt_file(
        source_receipt,
        source_receipt_path=source_receipt_path,
        source_receipt_file_sha256=source_receipt_file_sha256,
    )
    artifact = _load_json_mapping(artifact_path, "phase artifact")
    receipt = _load_json_mapping(receipt_path, "phase run receipt")
    artifact_fields = _phase_partition_fields(artifact, label="phase artifact")
    receipt_fields = _phase_partition_fields(receipt, label="phase run receipt")
    for field in (
        "source_receipt_sha256",
        "eligible_game_count",
        "eligible_identity_sha256",
        "eligible_assignment_sha256",
        "reference_assignment_sha256",
        "reference_assignment_match",
        "status",
    ):
        if artifact_fields[field] != receipt_fields[field]:
            raise FutureValueTrainingError(
                f"phase artifact and run receipt differ: {field}"
            )
    source_hash = _phase_hash(source_receipt.get("receipt_sha256"), "source receipt")
    if artifact_fields["source_receipt_sha256"] != source_hash:
        raise FutureValueTrainingError("phase source receipt differs")
    expected_count = int(source_receipt.get("model_eligible_game_count") or -1)
    expected_identity = _phase_hash(
        source_receipt.get("model_eligible_identity_sha256"),
        "source model-eligible identity",
    )
    if artifact_fields["eligible_game_count"] != expected_count:
        raise FutureValueTrainingError("phase eligible count differs from source")
    if artifact_fields["eligible_identity_sha256"] != expected_identity:
        raise FutureValueTrainingError("phase eligible identity differs from source")
    reference_ids = _phase_reference_game_ids(source_receipt)
    expected_reference_count = len(reference_ids)
    expected_reference_identity = identity_sha256(reference_ids)
    if artifact_fields["reference_game_count"] != expected_reference_count:
        raise FutureValueTrainingError("phase reference count differs from source")
    if artifact_fields["reference_identity_sha256"] != expected_reference_identity:
        raise FutureValueTrainingError("phase reference identity differs from source")
    if (
        expected_reference_count != artifact_fields["eligible_game_count"]
        and artifact_fields["reference_assignment_sha256"]
        == artifact_fields["eligible_assignment_sha256"]
    ):
        raise FutureValueTrainingError(
            "phase reference assignment is not full-census bound"
        )
    outputs = receipt.get("outputs")
    if not isinstance(outputs, Mapping) or not isinstance(outputs.get(artifact_kind), Mapping):
        raise FutureValueTrainingError("phase run receipt selected output is missing")
    output = outputs[artifact_kind]
    output_path = output.get("path", output.get("locator"))
    output_path_obj = Path(output_path) if isinstance(output_path, str) else None
    if (
        not isinstance(output_path, str)
        or output_path_obj is None
        or not output_path_obj.is_absolute()
        or output_path_obj.is_symlink()
        or output_path_obj.resolve() != artifact_path.resolve()
    ):
        raise FutureValueTrainingError("phase run receipt output path changed")
    if output.get("bytes") != artifact_file["bytes"] or str(
        output.get("sha256") or ""
    ).lower() != artifact_file["sha256"]:
        raise FutureValueTrainingError("phase run receipt output bytes changed")
    binding = {
        "phase_artifact": artifact_file,
        "phase_receipt": receipt_file,
        "phase_artifact_sha256": artifact_file["sha256"],
        "phase_receipt_file_sha256": receipt_file["sha256"],
        "phase_artifact_kind": artifact_kind,
        "eligible_game_count": artifact_fields["eligible_game_count"],
        "eligible_identity_sha256": artifact_fields["eligible_identity_sha256"],
        "eligible_assignment_sha256": artifact_fields["eligible_assignment_sha256"],
        "reference_game_count": artifact_fields["reference_game_count"],
        "reference_identity_sha256": artifact_fields["reference_identity_sha256"],
        "reference_assignment_sha256": artifact_fields["reference_assignment_sha256"],
        "source_receipt_sha256": source_hash,
        "source_receipt_file": source_receipt_file,
    }
    runtime_binding = {
        "artifact": artifact_file,
        "receipt": receipt_file,
        "artifact_kind": artifact_kind,
        "expected_artifact_sha256": artifact_file["sha256"],
        "expected_receipt_file_sha256": receipt_file["sha256"],
        "eligible_game_count": artifact_fields["eligible_game_count"],
        "eligible_identity_sha256": artifact_fields["eligible_identity_sha256"],
        "eligible_assignment_sha256": artifact_fields["eligible_assignment_sha256"],
        "reference_game_count": artifact_fields["reference_game_count"],
        "reference_identity_sha256": artifact_fields["reference_identity_sha256"],
        "reference_assignment_sha256": artifact_fields["reference_assignment_sha256"],
        "source_receipt_sha256": source_hash,
        "source_receipt": source_receipt_file,
    }
    return binding, runtime_binding


def _validate_source_receipt_mapping(receipt: Mapping[str, Any]) -> None:
    """Verify the source receipt payload before any training binding."""

    try:
        validate_future_value_source_receipt_payload(receipt)
    except FutureValueSourceError as error:
        raise FutureValueTrainingError(str(error)) from error


def verify_variant_input_binding(
    binding_path: Path,
    *,
    source_receipt: Mapping[str, Any],
    expected_game_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Verify the frozen four-variant producer inputs and file receipts."""

    binding = _load_json_mapping(binding_path, "variant input binding")
    _validate_source_receipt_mapping(source_receipt)
    expected_fields = {
        "schema_version",
        "status",
        "source_as_of",
        "source_game_count",
        "source_identity_sha256",
        "source_receipt_sha256",
        "evaluation_game_count",
        "evaluation_game_identity_sha256",
        "evaluation_game_ids",
        "producer_contract",
        "files",
        "authority",
        "receipt_sha256",
    }
    if set(binding) != expected_fields:
        raise FutureValueTrainingError("variant input binding schema is not canonical")
    if binding.get("schema_version") != "scryglass:future-value-variant-input-binding:v1" or binding.get(
        "status"
    ) != "frozen_research_input":
        raise FutureValueTrainingError("variant input binding status is invalid")
    expected_authority = {
        "research_only": True,
        "public": False,
        "probability": False,
        "odds": False,
        "ev": False,
        "recommendation": False,
        "betting": False,
        "promotion": False,
        "deployment": False,
    }
    if dict(binding.get("authority") or {}) != expected_authority:
        raise FutureValueTrainingError("variant input binding authority is invalid")
    claimed_binding_hash = binding.get("receipt_sha256")
    if not isinstance(claimed_binding_hash, str) or hashlib.sha256(
        _canonical_bytes({key: value for key, value in binding.items() if key != "receipt_sha256"})
    ).hexdigest() != claimed_binding_hash:
        raise FutureValueTrainingError("variant input binding receipt changed")
    if binding.get("source_receipt_sha256") != source_receipt.get("receipt_sha256"):
        raise FutureValueTrainingError("variant input binding source receipt changed")
    if binding.get("source_identity_sha256") != source_receipt.get("source_identity_sha256"):
        raise FutureValueTrainingError("variant input binding source identity changed")
    if (
        binding.get("source_as_of") != source_receipt.get("source_as_of")
        or binding.get("source_game_count") != source_receipt.get("source_game_count")
    ):
        raise FutureValueTrainingError("variant input source census changed")
    raw_bound_ids = binding.get("evaluation_game_ids")
    if not isinstance(raw_bound_ids, list):
        raise FutureValueTrainingError("variant input evaluation IDs are invalid")
    bound_ids = tuple(str(value) for value in raw_bound_ids)
    if (
        not bound_ids
        or tuple(canonical_game_ids(bound_ids)) != bound_ids
        or int(binding.get("evaluation_game_count") or -1) != len(bound_ids)
        or binding.get("evaluation_game_identity_sha256") != identity_sha256(bound_ids)
        or not set(bound_ids).issubset(set(map(str, source_receipt["model_eligible_game_ids"])))
    ):
        raise FutureValueTrainingError("variant input evaluation census changed")
    expected_ids = tuple(canonical_game_ids(str(value) for value in (expected_game_ids or ())))
    if expected_ids and bound_ids != expected_ids:
        raise FutureValueTrainingError("variant input evaluation IDs do not match runtime")
    producer_contract = binding.get("producer_contract")
    if not isinstance(producer_contract, Mapping):
        raise FutureValueTrainingError("variant input producer contract is missing")
    expected_contract = {
        "current_rating": "sequential strict-prior Dual Elo and exact-roster player Elo; same timestamp batch uses the prior timestamp state",
        "same_timestamp_policy": "features for all maps at one timestamp are emitted before any map at that timestamp updates history",
        "scaling_curve": "strict-prior player-champion checkpoint histories with champion fallback; current map checkpoint values update history after scoring",
    }
    if dict(producer_contract) != expected_contract:
        raise FutureValueTrainingError("variant input producer contract changed")
    files = binding.get("files")
    required_files = {
        "source_receipt",
        "current_rating_base",
        "atomized_matrix",
        "atomized_manifest",
    }
    if not isinstance(files, Mapping) or set(files) != required_files:
        raise FutureValueTrainingError("variant input file receipts are missing")
    verified_files: dict[str, dict[str, Any]] = {}
    for label, record in files.items():
        if not isinstance(record, Mapping):
            raise FutureValueTrainingError(f"variant input file record is invalid: {label}")
        path_value = record.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise FutureValueTrainingError(f"variant input file path is missing: {label}")
        path = Path(path_value)
        if not path.is_file() or path.is_symlink():
            raise FutureValueTrainingError(f"variant input file is missing or unsafe: {label}")
        expected_bytes = record.get("bytes")
        expected_hash = str(record.get("sha256") or "")
        if not isinstance(expected_bytes, int) or len(expected_hash) != 64:
            raise FutureValueTrainingError(f"variant input file receipt is incomplete: {label}")
        actual_bytes = path.stat().st_size
        actual_hash = _sha256(path)
        if actual_bytes != expected_bytes or actual_hash != expected_hash:
            raise FutureValueTrainingError(f"variant input file changed: {label}")
        verified_files[str(label)] = {
            "path": str(path),
            "bytes": actual_bytes,
            "sha256": actual_hash,
        }
    source_file = verified_files.get("source_receipt")
    if source_file is None:
        raise FutureValueTrainingError("variant input source receipt file is missing")
    source_path = Path(source_file["path"])
    source_payload = _load_json_mapping(source_path, "variant input source receipt")
    if source_payload != dict(source_receipt):
        raise FutureValueTrainingError("variant input source receipt payload changed")
    rating_file = verified_files.get("current_rating_base")
    matrix_file = verified_files.get("atomized_matrix")
    manifest_file = verified_files.get("atomized_manifest")
    if rating_file is None or matrix_file is None or manifest_file is None:
        raise FutureValueTrainingError("atomized producer files are incomplete")
    manifest = _load_json_mapping(Path(manifest_file["path"]), "atomized producer manifest")
    if manifest.get("matrix_sha256") != matrix_file["sha256"]:
        raise FutureValueTrainingError("atomized matrix manifest hash changed")
    if int(manifest.get("rows") or -1) != int(binding.get("evaluation_game_count") or -2):
        raise FutureValueTrainingError("atomized matrix row count changed")
    for label, record in (("current rating", rating_file), ("atomized matrix", matrix_file)):
        path = Path(record["path"])
        try:
            columns = set(pd.read_parquet(path, columns=None).columns)
            game_column = next(
                name for name in ("game_id", "game_uid", "gameid") if name in columns
            )
            frame_ids = tuple(
                canonical_game_ids(
                    pd.read_parquet(path, columns=[game_column])[game_column].astype(str)
                )
            )
        except (OSError, ValueError, StopIteration) as error:
            raise FutureValueTrainingError(f"{label} identity cannot be read") from error
        if frame_ids != bound_ids:
            raise FutureValueTrainingError(f"{label} game IDs changed")
    return {
        "schema_version": binding.get("schema_version"),
        "receipt_sha256": claimed_binding_hash,
        "source_receipt_sha256": source_receipt.get("receipt_sha256"),
        "evaluation_game_count": binding.get("evaluation_game_count"),
        "evaluation_game_identity_sha256": binding.get("evaluation_game_identity_sha256"),
        "files": verified_files,
        "atomized_manifest_sha256": manifest_file["sha256"],
        "atomized_matrix_sha256": matrix_file["sha256"],
        "authority": {"research_only": True, "promotion": False, "deployment": False},
    }


def _row_game_ids(frame: pd.DataFrame, label: str) -> pd.Series:
    if "game_uid" in frame.columns:
        fallback = frame["gameid"] if "gameid" in frame.columns else None
        values = [
            canonical_source_game_key(
                value,
                fallback.loc[index] if fallback is not None else None,
            )
            for index, value in frame["game_uid"].items()
        ]
    elif "gameid" in frame.columns:
        values = [canonical_source_game_key(value) for value in frame["gameid"]]
    else:
        raise FutureValueTrainingError(f"{label} has no game identity")
    ids = pd.Series(values, index=frame.index, dtype="string")
    if ids.isna().any() or ids.str.strip().eq("").any():
        raise FutureValueTrainingError(f"{label} has an empty game identity")
    return ids


def _code_hashes(repo_root: Path) -> dict[str, str]:
    """Bind every producer module used by the four-way research run."""

    paths = (
        "lol_kills/research/future_value_rating.py",
        "lol_kills/research/future_value_training.py",
        "lol_kills/research/future_phase_curve.py",
        "lol_kills/research/atomized_rf_composite.py",
        "lol_kills/research/future_value_draft_score.py",
    )
    output: dict[str, str] = {}
    for relative in paths:
        path = repo_root / relative
        if not path.is_file() or path.is_symlink():
            raise FutureValueTrainingError(f"model producer source is missing: {relative}")
        output[relative] = _sha256(path)
    return output


def _load_feature_ledger_bundle(path: Path | None) -> Mapping[str, Any] | None:
    """Load JSON ledger bundles with explicit per-variant fold bindings.

    JSON is used because a parquet round trip drops ``DataFrame.attrs``, which
    carry the producer receipt.  Every fold record must contain ``rows`` and
    ``attrs``.  The caller still validates the complete binding in the rating
    module before fitting.
    """

    if path is None:
        return None
    value = _load_json_mapping(path, "feature ledger bundle")
    variants = value.get("variants")
    if not isinstance(variants, Mapping):
        raise FutureValueTrainingError("feature ledger bundle variants are missing")
    bundle: dict[str, Any] = {}
    for variant_name, variant_value in variants.items():
        if not isinstance(variant_value, Mapping):
            raise FutureValueTrainingError("feature ledger variant binding is invalid")
        folds = variant_value.get("folds")
        if not isinstance(folds, Mapping):
            raise FutureValueTrainingError("feature ledger fold bindings are missing")
        def read_folds(raw_folds: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
            fold_bundle: dict[str, pd.DataFrame] = {}
            for fold, record in raw_folds.items():
                if not isinstance(record, Mapping) or not isinstance(record.get("rows"), list):
                    raise FutureValueTrainingError("feature ledger fold record is invalid")
                attrs = record.get("attrs")
                if not isinstance(attrs, Mapping):
                    raise FutureValueTrainingError("feature ledger fold attrs are missing")
                frame = pd.DataFrame(record["rows"])
                frame.attrs = dict(attrs)
                fold_bundle[str(fold)] = frame
            return fold_bundle

        outer_bundle = read_folds(folds)
        raw_inner = variant_value.get("inner_folds")
        if raw_inner is None:
            raw_inner = variant_value.get("inner")
        if raw_inner is not None:
            if not isinstance(raw_inner, Mapping):
                raise FutureValueTrainingError("feature ledger inner fold bindings are invalid")
            bundle[str(variant_name)] = {
                "outer": outer_bundle,
                "inner": read_folds(raw_inner),
            }
        else:
            bundle[str(variant_name)] = outer_bundle
    return bundle


def _resolve_variant_names(value: str | None) -> tuple[RatingVariant, ...] | None:
    """Resolve one or all explicit rating variants for the CLI."""

    if value is None or value.strip().casefold() in {"legacy", "current_ratings"}:
        return None
    if value.strip().casefold() == "all":
        return tuple(RATING_VARIANT_ORDER)
    try:
        return (RatingVariant(value.strip()),)
    except ValueError as error:
        raise FutureValueTrainingError(f"unknown rating variant: {value}") from error


def _git_output(repo_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise FutureValueTrainingError("model runtime cannot bind the git source state")
    return result.stdout.strip()


def run_model_evaluation(
    *,
    oe_root: Path,
    freeze_path: Path,
    source_receipt_path: Path,
    model_output_path: Path,
    runtime_receipt_path: Path,
    n_folds: int = 3,
    command: list[str] | None = None,
    rating_variant: str | None = None,
    feature_ledger_path: Path | None = None,
    input_binding_path: Path | None = None,
    crosswalk_path: Path | None = None,
    crosswalk_receipt_path: Path | None = None,
    crosswalk_receipt_file_sha256: str | None = None,
    phase_artifact_path: Path | None = None,
    phase_receipt_path: Path | None = None,
    phase_artifact_sha256: str | None = None,
    phase_receipt_file_sha256: str | None = None,
    phase_artifact_kind: str = "candidate",
) -> dict[str, Any]:
    """Run the frozen research model and emit a gate-grade runtime receipt."""

    freeze = _load_freeze(freeze_path)
    source_receipt = _load_json_mapping(source_receipt_path, "source receipt")
    _validate_source_receipt_mapping(source_receipt)
    crosswalk_values = (
        crosswalk_path,
        crosswalk_receipt_path,
        crosswalk_receipt_file_sha256,
    )
    if any(value is not None for value in crosswalk_values) and not all(
        value is not None for value in crosswalk_values
    ):
        raise FutureValueTrainingError("crosswalk inputs must be supplied together")
    if crosswalk_receipt_file_sha256 is not None and re.fullmatch(
        r"[0-9a-f]{64}", str(crosswalk_receipt_file_sha256), re.I
    ) is None:
        raise FutureValueTrainingError("crosswalk receipt file hash is invalid")
    phase_values = (
        phase_artifact_path,
        phase_receipt_path,
        phase_artifact_sha256,
        phase_receipt_file_sha256,
    )
    if any(value is not None for value in phase_values) and not all(
        value is not None for value in phase_values
    ):
        raise FutureValueTrainingError("phase partition inputs must be supplied together")
    expected_receipt_hash = str(freeze["reference_source_receipt_sha256"])
    expected_receipt_file_hash = str(freeze.get("source_receipt_file_sha256") or "")
    expected_receipt_path = str(freeze.get("source_receipt_path") or "")
    if source_receipt.get("receipt_sha256") != expected_receipt_hash:
        raise FutureValueTrainingError("source receipt identity changed")
    if not expected_receipt_path or Path(expected_receipt_path) != source_receipt_path:
        raise FutureValueTrainingError("source receipt path does not match the freeze")
    receipt_file_hash = _sha256(source_receipt_path)
    if receipt_file_hash != expected_receipt_file_hash:
        raise FutureValueTrainingError("source receipt file hash changed")
    phase_partition_binding = None
    phase_runtime_binding = None
    if phase_artifact_path is not None and phase_receipt_path is not None:
        if phase_artifact_kind not in {"candidate", "evaluation"}:
            raise FutureValueTrainingError("phase artifact kind is invalid")
        phase_partition_binding, phase_runtime_binding = _build_phase_partition_binding(
            phase_artifact_path,
            phase_receipt_path,
            artifact_sha256=str(phase_artifact_sha256),
            receipt_file_sha256=str(phase_receipt_file_sha256),
            source_receipt=source_receipt,
            source_receipt_path=source_receipt_path,
            source_receipt_file_sha256=expected_receipt_file_hash,
            artifact_kind=phase_artifact_kind,
        )
    input_binding = None
    if input_binding_path is not None:
        input_binding = verify_variant_input_binding(
            input_binding_path,
            source_receipt=source_receipt,
            expected_game_ids=source_receipt["model_eligible_game_ids"],
        )

    paths = {
        "maps": oe_root / "maps.parquet",
        "players": oe_root / "oe_player_games.parquet",
        "teams": oe_root / "oe_team_games.parquet",
    }
    normalized_contract = freeze.get("normalized_source_files")
    if not isinstance(normalized_contract, Mapping):
        raise FutureValueTrainingError("normalized source file contract is missing")
    frames: dict[str, pd.DataFrame] = {}
    verified_series_model_frame: pd.DataFrame | None = None
    eligible_ids = set(str(value) for value in source_receipt["model_eligible_game_ids"])
    for label, path in paths.items():
        if not path.is_file() or path.is_symlink():
            raise FutureValueTrainingError(f"model source is missing or unsafe: {label}")
        contract = normalized_contract.get(label)
        if (
            not isinstance(contract, Mapping)
            or contract.get("bytes") != path.stat().st_size
            or contract.get("sha256") != _sha256(path)
        ):
            raise FutureValueTrainingError(f"normalized model source changed: {label}")
        frame = pd.read_parquet(path)
        if label == "maps" and crosswalk_path is not None and crosswalk_receipt_path is not None:
            bound_full_maps = bind_verified_leaguepedia_series_crosswalk(
                frame,
                crosswalk_path=crosswalk_path,
                receipt_path=crosswalk_receipt_path,
                source_receipt=source_receipt,
                expected_receipt_file_sha256=str(crosswalk_receipt_file_sha256),
            )
            full_series_frame = _map_model_frame(
                bound_full_maps,
                verified_source_receipt=source_receipt,
                verified_source_receipt_sha256=str(source_receipt["receipt_sha256"]),
                verified_crosswalk_receipt_file_sha256=str(
                    crosswalk_receipt_file_sha256
                ),
            )
            verified_series_model_frame = full_series_frame[
                full_series_frame["game_id"].astype(str).isin(eligible_ids)
            ].copy()
            verified_series_model_frame.attrs["crosswalk_receipt_file_sha256"] = str(
                crosswalk_receipt_file_sha256
            )
        ids = _row_game_ids(frame, label)
        selected = frame.loc[ids.isin(eligible_ids)].copy()
        selected["game_uid"] = ids.loc[selected.index].to_numpy()
        frames[label] = selected.reset_index(drop=True)
    if frames["maps"]["game_uid"].nunique() != len(eligible_ids):
        raise FutureValueTrainingError("model map frame does not match the eligible census")
    crosswalk_runtime_binding = None
    if crosswalk_path is not None and crosswalk_receipt_path is not None:
        if feature_ledger_path is None:
            raise FutureValueTrainingError(
                "verified crosswalk evaluation requires a feature ledger bundle"
            )
        feature_payload = _load_json_mapping(
            feature_ledger_path, "feature ledger bundle"
        )
        feature_source = feature_payload.get("source")
        if not isinstance(feature_source, Mapping) or feature_source.get(
            "series_partition_receipt_file_sha256"
        ) != str(crosswalk_receipt_file_sha256):
            raise FutureValueTrainingError(
                "crosswalk receipt hash does not match the feature bundle"
            )
        if verified_series_model_frame is None or set(
            verified_series_model_frame["game_id"].astype(str)
        ) != eligible_ids:
            raise FutureValueTrainingError(
                "verified crosswalk model frame does not match the eligible census"
            )
        crosswalk_runtime_binding = {
            "artifact": {
                "path": str(crosswalk_path),
                "bytes": crosswalk_path.stat().st_size,
                "sha256": _sha256(crosswalk_path),
            },
            "receipt": {
                "path": str(crosswalk_receipt_path),
                "bytes": crosswalk_receipt_path.stat().st_size,
                "sha256": _sha256(crosswalk_receipt_path),
            },
            "expected_receipt_file_sha256": str(
                crosswalk_receipt_file_sha256
            ),
        }

    repo_root = Path(__file__).resolve().parents[2]
    code_paths = [
        "lol_kills/research/future_value_rating.py",
        "lol_kills/research/future_value_training.py",
        "lol_kills/ratings/player_elo.py",
        "lol_kills/ratings/hierarchical_bt.py",
        "lol_kills/research/future_phase_curve.py",
        "lol_kills/research/future_value_draft_score.py",
    ]
    dirty_code = _git_output(repo_root, "status", "--porcelain", "--", *code_paths)
    if dirty_code:
        raise FutureValueTrainingError("model code has uncommitted changes")
    code_commit = _git_output(repo_root, "rev-parse", "HEAD")
    producer_code_hashes = _code_hashes(repo_root)
    selected_variants = _resolve_variant_names(rating_variant)
    ledger_bundle = _load_feature_ledger_bundle(feature_ledger_path)
    if selected_variants is not None and ledger_bundle is None:
        raise FutureValueTrainingError(
            "explicit rating variants require a per-variant feature ledger bundle"
        )
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    try:
        from threadpoolctl import threadpool_info

        threadpools = threadpool_info()
    except (ImportError, RuntimeError):
        threadpools = []
    try:
        if selected_variants is None:
            result: dict[str, Any] = evaluate_future_value(
                frames["maps"],
                frames["players"],
                n_folds=int(n_folds),
                source_receipt=source_receipt,
                source_receipt_path=str(source_receipt_path),
                source_receipt_file_sha256=receipt_file_hash,
                runtime_receipt_path=str(runtime_receipt_path),
                crosswalk_receipt_file_sha256=crosswalk_receipt_file_sha256,
                verified_model_frame=verified_series_model_frame,
                phase_partition_binding=phase_partition_binding,
                expected_phase_artifact_sha256=phase_artifact_sha256,
                expected_phase_receipt_file_sha256=phase_receipt_file_sha256,
            )
        else:
            variant_results: dict[str, Any] = {}
            for variant in selected_variants:
                variant_key = variant.value
                variant_ledger = ledger_bundle.get(variant_key)
                if not isinstance(variant_ledger, Mapping):
                    raise FutureValueTrainingError(
                        f"feature ledger bundle is missing variant: {variant_key}"
                    )
                inner_variant_ledger = None
                if "outer" in variant_ledger:
                    outer_variant_ledger = variant_ledger.get("outer")
                    inner_variant_ledger = variant_ledger.get("inner")
                    if not isinstance(outer_variant_ledger, Mapping) or not isinstance(
                        inner_variant_ledger, Mapping
                    ):
                        raise FutureValueTrainingError(
                            f"feature ledger nested bindings are invalid: {variant_key}"
                        )
                    variant_ledger = outer_variant_ledger
                variant_results[variant_key] = evaluate_future_value(
                    frames["maps"],
                    frames["players"],
                    n_folds=int(n_folds),
                    source_receipt=source_receipt,
                    source_receipt_path=str(source_receipt_path),
                    source_receipt_file_sha256=receipt_file_hash,
                    runtime_receipt_path=str(runtime_receipt_path),
                    variant=variant,
                    feature_ledger=variant_ledger,
                    inner_feature_ledger=inner_variant_ledger,
                    crosswalk_receipt_file_sha256=crosswalk_receipt_file_sha256,
                    verified_model_frame=verified_series_model_frame,
                    phase_partition_binding=phase_partition_binding,
                    expected_phase_artifact_sha256=phase_artifact_sha256,
                    expected_phase_receipt_file_sha256=phase_receipt_file_sha256,
                )
            result = {
                "schema_version": "scryglass:future-value-four-variant-evaluation:v1",
                "variants": variant_results,
                "variant_configs": {
                    variant.value: rating_variant_config_receipt(variant)
                    for variant in selected_variants
                },
                "source": {
                    "source_as_of": source_receipt["source_as_of"],
                    "source_game_count": source_receipt["source_game_count"],
                    "source_identity_sha256": source_receipt["source_identity_sha256"],
                    "source_receipt_sha256": source_receipt["receipt_sha256"],
                },
                "authority": {
                    "research_only": True,
                    "public_player_rating": False,
                    "public_team_rating": False,
                    "public_probability": False,
                    "deployment": False,
                    "promotion": False,
                },
            }
    except FutureValueSourceError as error:
        raise FutureValueTrainingError(str(error)) from error
    result["source"]["normalized_source_files"] = {
        str(label): dict(record)
        for label, record in sorted(normalized_contract.items())
    }
    elapsed = time.perf_counter() - started
    completed_at = datetime.now(timezone.utc)
    _write_json(model_output_path, result)
    output_hash = _sha256(model_output_path)
    runtime: dict[str, Any] = {
        "schema_version": MODEL_RUNTIME_SCHEMA_VERSION,
        "status": "research_evaluation_complete",
        "entrypoint": "lol_kills.research.future_value_training.run_model_evaluation",
        "command": list(command or []),
        "code_commit": code_commit,
        "producer_code_hashes": producer_code_hashes,
        "rating_variant": rating_variant or "legacy",
        "feature_ledger_path": str(feature_ledger_path) if feature_ledger_path else None,
        "input_binding": input_binding,
        "series_partition": crosswalk_runtime_binding,
        "phase_partition": phase_runtime_binding,
        "started_at": started_at.isoformat().replace("+00:00", "Z"),
        "completed_at": completed_at.isoformat().replace("+00:00", "Z"),
        "elapsed_seconds": float(elapsed),
        "environment": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "pid": os.getpid(),
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
            },
            "threadpools": threadpools,
        },
        "source": {
            "source_as_of": source_receipt["source_as_of"],
            "source_game_count": source_receipt["source_game_count"],
            "source_identity_sha256": source_receipt["source_identity_sha256"],
            "model_eligible_game_count": source_receipt["model_eligible_game_count"],
            "model_eligible_identity_sha256": source_receipt[
                "model_eligible_identity_sha256"
            ],
            "source_receipt_path": str(source_receipt_path),
            "source_receipt_sha256": source_receipt["receipt_sha256"],
            "source_receipt_file_sha256": receipt_file_hash,
        },
        "input_rows": {
            "maps": int(len(frames["maps"])),
            "players": int(len(frames["players"])),
            "teams": int(len(frames["teams"])),
        },
        "output": {
            "path": str(model_output_path),
            "bytes": model_output_path.stat().st_size,
            "sha256": output_hash,
            "prediction_ledger_sha256": (
                result.get("prediction_ledger", {}).get("sha256")
                if selected_variants is None
                else {
                    key: value.get("prediction_ledger", {}).get("sha256")
                    for key, value in result.get("variants", {}).items()
                }
            ),
            "prediction_ledger_rows": (
                result.get("prediction_ledger", {}).get("row_count")
                if selected_variants is None
                else {
                    key: value.get("prediction_ledger", {}).get("row_count")
                    for key, value in result.get("variants", {}).items()
                }
            ),
        },
        "authority": {
            "research_only": True,
            "public_player_rating": False,
            "public_team_rating": False,
            "public_probability": False,
            "odds": False,
            "expected_value": False,
            "recommendation": False,
            "betting": False,
            "promotion": False,
            "deployment": False,
        },
    }
    runtime["receipt_sha256"] = hashlib.sha256(_canonical_bytes(runtime)).hexdigest()
    _write_json(runtime_receipt_path, runtime)
    return runtime


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annual-root", type=Path)
    parser.add_argument("--oe-root", type=Path)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--annual-only", action="store_true")
    parser.add_argument("--fit-model", action="store_true")
    parser.add_argument("--source-receipt", type=Path)
    parser.add_argument("--model-output", type=Path)
    parser.add_argument("--runtime-receipt", type=Path)
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument(
        "--rating-variant",
        "--variants",
        default=None,
        choices=("legacy", "current_only", "future_player_form", "scaling_curve", "both", "all"),
        help="run the legacy model, one registered variant, or all four variants",
    )
    parser.add_argument(
        "--feature-ledger-bundle",
        type=Path,
        help="JSON bundle with independently bound per-variant fold ledgers",
    )
    parser.add_argument(
        "--input-binding",
        type=Path,
        help="frozen current-rating and atomized producer input binding",
    )
    parser.add_argument("--crosswalk", type=Path)
    parser.add_argument("--crosswalk-receipt", type=Path)
    parser.add_argument("--crosswalk-receipt-file-sha256")
    parser.add_argument("--phase-artifact", type=Path)
    parser.add_argument("--phase-receipt", type=Path)
    parser.add_argument("--phase-artifact-sha256")
    parser.add_argument("--phase-receipt-file-sha256")
    parser.add_argument(
        "--phase-artifact-kind",
        default="candidate",
        choices=("candidate", "evaluation"),
    )
    args = parser.parse_args(argv)
    try:
        if args.fit_model:
            required = {
                "--oe-root": args.oe_root,
                "--source-receipt": args.source_receipt,
                "--model-output": args.model_output,
                "--runtime-receipt": args.runtime_receipt,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                parser.error("model fit requires " + ", ".join(missing))
            result = run_model_evaluation(
                oe_root=args.oe_root,
                freeze_path=args.freeze,
                source_receipt_path=args.source_receipt,
                model_output_path=args.model_output,
                runtime_receipt_path=args.runtime_receipt,
                n_folds=args.n_folds,
                rating_variant=args.rating_variant,
                feature_ledger_path=args.feature_ledger_bundle,
                input_binding_path=args.input_binding,
                crosswalk_path=(
                    None if args.crosswalk is None else args.crosswalk.resolve()
                ),
                crosswalk_receipt_path=(
                    None
                    if args.crosswalk_receipt is None
                    else args.crosswalk_receipt.resolve()
                ),
                crosswalk_receipt_file_sha256=args.crosswalk_receipt_file_sha256,
                phase_artifact_path=(
                    None if args.phase_artifact is None else args.phase_artifact.resolve()
                ),
                phase_receipt_path=(
                    None if args.phase_receipt is None else args.phase_receipt.resolve()
                ),
                phase_artifact_sha256=args.phase_artifact_sha256,
                phase_receipt_file_sha256=args.phase_receipt_file_sha256,
                phase_artifact_kind=args.phase_artifact_kind,
                command=[
                    sys.executable,
                    "-m",
                    "lol_kills.research.future_value_training",
                    *(argv or sys.argv[1:]),
                ],
            )
        elif args.annual_only:
            if args.annual_root is None or args.output_root is None:
                parser.error("--annual-root and --output-root are required")
            result = verify_annual_only(
                annual_root=args.annual_root,
                freeze_path=args.freeze,
                output_root=args.output_root,
            )
        else:
            if args.annual_root is None or args.output_root is None:
                parser.error("--annual-root and --output-root are required")
            if args.oe_root is None:
                parser.error("--oe-root is required unless --annual-only is set")
            result = verify_research_source(
                annual_root=args.annual_root,
                oe_root=args.oe_root,
                freeze_path=args.freeze,
                output_root=args.output_root,
            )
    except FutureValueTrainingError as error:
        parser.exit(1, f"future-value research verification failed: {error}\n")
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_FREEZE",
    "FutureValueTrainingError",
    "frozen_census",
    "verify_annual_only",
    "verify_annual_sources",
    "verify_bridge_sources",
    "verify_research_source",
    "verify_variant_input_binding",
    "run_model_evaluation",
]
