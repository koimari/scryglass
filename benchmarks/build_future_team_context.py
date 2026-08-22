"""Build a source-bound research artifact for future team context.

The producer consumes one verified final V2 model and the frozen source rows.
It records exact-five current rosters, strictly-prior team state, support, and
uncertainty.  A player-form model without a declared team-context component
emits an explicit neutral-zero context and a blocker.  It never grants public
team-rating authority.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from lol_kills.v2.tierlists.accepted_census import identity_sha256
from lol_kills.research.future_value_rating import (
    _canonical_json_bytes,
    validate_future_value_source_receipt_payload,
)
from lol_kills.research.future_value_snapshots import (
    TEAM_CONTEXT_BINDING_SCHEMA_VERSION,
    FutureValueSnapshotError,
    _latest_player_form,
    _latest_team_roster,
    _normalise_source_frames,
    _sha256_file,
    _source_binding,
    _validated_team_context_binding,
    build_future_value_snapshots,
    load_final_fit_model,
)


TEAM_CONTEXT_ARTIFACT_SCHEMA_VERSION = "scryglass:future-team-context-artifact:v1"
TEAM_CONTEXT_RECEIPT_SCHEMA_VERSION = "scryglass:future-team-context-receipt:v1"
TEAM_CONTEXT_AUTHORITY = {
    "research_only": True,
    "public_team_rating": False,
    "public_player_rating": False,
    "public_probability": False,
    "promotion": False,
    "merge": False,
    "deployment": False,
    "odds": False,
    "expected_value": False,
    "recommendation": False,
    "betting": False,
}


class FutureTeamContextError(FutureValueSnapshotError):
    """The team-context producer cannot bind its inputs safely."""


@dataclass(frozen=True)
class TeamContextBuild:
    """Rows and source/model component evidence before file binding."""

    status: str
    blockers: tuple[str, ...]
    rows: tuple[Mapping[str, Any], ...]
    source: Mapping[str, Any]
    model: Mapping[str, Any]
    component: Mapping[str, Any]
    snapshot_receipt_sha256: str | None


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _verified_file_record(path: Path, label: str) -> dict[str, Any]:
    """Return a byte-bound record for one existing regular file."""

    candidate = Path(path)
    if (
        not candidate.is_absolute()
        or ".." in candidate.parts
        or candidate.is_symlink()
        or not candidate.is_file()
    ):
        raise FutureTeamContextError(f"{label} file is missing or unsafe")
    return {
        "path": str(candidate),
        "bytes": candidate.stat().st_size,
        "sha256": _sha256_file(candidate),
    }


def _bind_file_record(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FutureTeamContextError(f"{label} file binding is invalid")
    path = Path(str(value.get("path") or ""))
    actual = _verified_file_record(path, label)
    try:
        declared_bytes = int(value.get("bytes"))
    except (TypeError, ValueError) as error:
        raise FutureTeamContextError(f"{label} file byte binding is invalid") from error
    declared_hash = str(value.get("sha256") or "").lower()
    if declared_bytes != actual["bytes"] or declared_hash != actual["sha256"]:
        raise FutureTeamContextError(f"{label} file binding changed")
    return actual


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FutureTeamContextError(f"{label} is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FutureTeamContextError(f"{label} cannot be read: {path}") from error
    if not isinstance(value, dict):
        raise FutureTeamContextError(f"{label} must be an object")
    return value


def _json_bytes(value: object) -> bytes:
    return _canonical_json_bytes(value)


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _source_file_paths(
    source_root: Path,
    source_receipt_path: Path,
    source_receipt: Mapping[str, Any],
    *,
    expected_source_receipt_sha256: str,
) -> dict[str, Path]:
    """Verify the source receipt and bind its three frozen parquet files."""

    expected = str(expected_source_receipt_sha256 or "").lower()
    if len(expected) != 64 or _sha256_file(source_receipt_path) != expected:
        raise FutureTeamContextError("source receipt file hash changed")
    try:
        validate_future_value_source_receipt_payload(source_receipt)
    except Exception as error:
        raise FutureTeamContextError("source receipt failed validation") from error
    records = source_receipt.get("source_files")
    if not isinstance(records, Mapping):
        raise FutureTeamContextError("source receipt file bindings are missing")
    expected_names = {
        "maps": "maps.parquet",
        "players": "oe_player_games.parquet",
        "teams": "oe_team_games.parquet",
    }
    if source_root.is_symlink() or not source_root.is_dir():
        raise FutureTeamContextError("source root is missing or unsafe")
    if source_receipt_path.is_symlink():
        raise FutureTeamContextError("source receipt is a symlink")
    root = source_root.resolve()
    receipt_root = source_receipt_path.parent.resolve()
    output: dict[str, Path] = {}
    for label, name in expected_names.items():
        record = records.get(label)
        if not isinstance(record, Mapping):
            raise FutureTeamContextError(f"source {label} file binding is missing")
        locator = Path(str(record.get("locator") or ""))
        if locator.is_absolute() or not locator.parts or ".." in locator.parts:
            raise FutureTeamContextError(f"source {label} file locator is unsafe")
        bound = (receipt_root / locator).resolve()
        path = (root / name).resolve()
        if bound != path or path.is_symlink() or not path.is_file():
            raise FutureTeamContextError(f"source {label} file path changed")
        if int(record.get("bytes") or -1) != path.stat().st_size:
            raise FutureTeamContextError(f"source {label} file bytes changed")
        if str(record.get("sha256") or "").lower() != _sha256_file(path):
            raise FutureTeamContextError(f"source {label} file hash changed")
        output[label] = path
    return output


def _parameter_component(
    model: Any,
    binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Bind the exact model coefficient and scale for each context feature."""

    if binding is None:
        payload: dict[str, Any] = {
            "status": "neutral_zero",
            "feature_names": [],
            "parameters": [],
        }
        parameter_receipt = getattr(model, "parameter_receipt", None)
        if callable(parameter_receipt):
            model_parameters = parameter_receipt()
            if not isinstance(model_parameters, Mapping):
                raise FutureTeamContextError("final model parameter receipt is invalid")
            parameter_hash = str(model_parameters.get("parameter_sha256") or "").lower()
            if len(parameter_hash) != 64:
                raise FutureTeamContextError("final model parameter hash is missing")
            payload["model_parameter_sha256"] = parameter_hash
        payload["parameter_component_sha256"] = _sha256_bytes(_json_bytes(payload))
        return payload
    names = tuple(str(value) for value in binding.get("feature_names") or ())
    model_names = tuple(str(value) for value in getattr(model, "feature_names", ()))
    coefficients = list(getattr(model, "coefficients", ()))
    scales = list(getattr(model, "scales", ()))
    imputation = list(getattr(model, "imputation_values", ()))
    parameters: list[dict[str, Any]] = []
    for name in names:
        if name not in model_names:
            raise FutureTeamContextError(
                f"team-context model feature is missing: {name}"
            )
        index = model_names.index(name)
        coefficient = _finite(coefficients[index]) if index < len(coefficients) else None
        scale = _finite(scales[index]) if index < len(scales) else None
        imputed = _finite(imputation[index]) if index < len(imputation) else None
        if coefficient is None or scale is None or scale == 0 or imputed is None:
            raise FutureTeamContextError(
                f"team-context model parameter is invalid: {name}"
            )
        parameters.append(
            {
                "feature": name,
                "coefficient": coefficient,
                "scale": scale,
                "imputation": imputed,
            }
        )
    payload = {
        "status": "available",
        "feature_names": list(names),
        "parameters": parameters,
    }
    parameter_receipt = getattr(model, "parameter_receipt", None)
    if callable(parameter_receipt):
        model_parameters = parameter_receipt()
        if not isinstance(model_parameters, Mapping):
            raise FutureTeamContextError("final model parameter receipt is invalid")
        parameter_hash = str(model_parameters.get("parameter_sha256") or "").lower()
        if len(parameter_hash) != 64:
            raise FutureTeamContextError("final model parameter hash is missing")
        payload["model_parameter_sha256"] = parameter_hash
    payload["parameter_component_sha256"] = _sha256_bytes(_json_bytes(payload))
    return payload


def _roster_rows(
    maps: pd.DataFrame,
    players: pd.DataFrame,
    teams: pd.DataFrame,
    source_receipt: Mapping[str, Any],
    *,
    model_receipt: Mapping[str, Any],
    model: Any | None = None,
) -> pd.DataFrame:
    """Return the latest exact-five roster with stable role identities."""

    try:
        map_frame, player_frame, team_frame, cutoff = _normalise_source_frames(
            maps, players, teams, source_receipt, None
        )
    except FutureValueSnapshotError as error:
        raise FutureTeamContextError(str(error)) from error
    binding = None
    if model is not None:
        try:
            binding = _validated_team_context_binding(
                model,
                model_receipt,
                _source_binding(source_receipt),
            )
        except FutureValueSnapshotError:
            binding = None
    if binding is None and model is None:
        try:
            candidate = model_receipt.get("team_context_binding")
            binding = candidate if isinstance(candidate, Mapping) else None
        except AttributeError:
            binding = None
    context_columns: tuple[str, ...] = ()
    if isinstance(binding, Mapping):
        regional = binding.get("regional_transfer_source")
        if isinstance(regional, Mapping):
            value = str(regional.get("player_region_column") or "").strip()
            if value:
                context_columns = (value,)
    form = _latest_player_form(
        map_frame,
        player_frame,
        context_columns=context_columns,
    )
    form = form[form["date"].le(cutoff)].copy()
    latest = _latest_team_roster(form)
    required = {"team_id", "player_id", "role", "game_id", "date", "side"}
    if not required.issubset(latest.columns):
        raise FutureTeamContextError("latest roster identity is incomplete")
    records: list[dict[str, Any]] = []
    for team_id, group in latest.groupby("team_id", sort=True):
        group = group.sort_values("role", kind="stable")
        if len(group) != 5 or group["player_id"].nunique() != 5:
            raise FutureTeamContextError("latest team roster is not exact five")
        roles = {str(row.role): str(row.player_id) for row in group.itertuples()}
        if len(roles) != 5:
            raise FutureTeamContextError("latest team roster roles are ambiguous")
        records.append(
            {
                "team_id": str(team_id),
                "side": str(group["side"].iloc[0]),
                "last_game_id": str(group["game_id"].iloc[0]),
                "last_game_date": group["date"].iloc[0],
                "roster_player_count": 5,
                "roster_player_ids": [str(value) for value in group["player_id"]],
                "roster_roles": roles,
            }
        )
    return pd.DataFrame(records)


def build_team_context_artifact(
    maps: pd.DataFrame,
    players: pd.DataFrame,
    teams: pd.DataFrame,
    *,
    source_receipt: Mapping[str, Any],
    model: Any,
    model_receipt: Mapping[str, Any],
    source_receipt_file: Mapping[str, Any] | None = None,
    model_receipt_file: Mapping[str, Any] | None = None,
    model_artifact_file: Mapping[str, Any] | None = None,
) -> TeamContextBuild:
    """Build team-context rows from one verified final model object."""

    source = _source_binding(source_receipt)
    if source_receipt_file is not None:
        source = dict(source)
        source["source_receipt_file"] = _bind_file_record(
            source_receipt_file,
            "source receipt",
        )
    try:
        binding = _validated_team_context_binding(model, model_receipt, source)
        binding_error: str | None = None
    except FutureValueSnapshotError as error:
        binding = None
        binding_error = str(error)
    try:
        component = _parameter_component(model, binding)
    except FutureTeamContextError as error:
        binding = None
        binding_error = str(error)
        component = _parameter_component(model, None)
    snapshot = build_future_value_snapshots(
        maps,
        players,
        teams,
        source_receipt=source_receipt,
        model=model,
        model_receipt=model_receipt,
    )
    roster = _roster_rows(
        maps,
        players,
        teams,
        source_receipt,
        model_receipt=model_receipt,
        model=model,
    )
    roster_by_id = {
        str(row["team_id"]): row
        for row in roster.to_dict(orient="records")
    }
    blockers = set(str(value) for value in snapshot.blockers)
    if binding is None:
        blockers.add(
            "team_context_contract_invalid"
            if binding_error
            else "team_context_not_in_final_model"
        )
    output_rows: list[dict[str, Any]] = []
    for source_row in snapshot.team_rows:
        row = dict(source_row)
        team_id = str(row.get("team_id") or "")
        roster_row = roster_by_id.get(team_id)
        if roster_row is None:
            raise FutureTeamContextError("snapshot team is missing exact-five roster")
        row.update(
            {
                "roster_player_count": 5,
                "roster_player_ids": list(roster_row["roster_player_ids"]),
                "roster_roles": dict(roster_row["roster_roles"]),
                "expected_starters": True,
                "team_context_component_receipt_sha256": (
                    binding.get("receipt_sha256") if binding is not None else None
                ),
            }
        )
        player_value = _finite(row.get("role_normalized_player_value_logit"))
        context_value = _finite(row.get("team_context_logit"))
        if binding is None:
            # A player-form-only final fit has no fitted team context.  Keep
            # the additive term explicit and neutral, with a blocker.
            context_value = 0.0
            row["team_context_logit"] = 0.0
            row["team_context_status"] = "neutral_zero_blocked"
            row["team_context_applied"] = False
            row["team_context_blocker"] = (
                "team_context_contract_invalid"
                if binding_error
                else "team_context_not_in_final_model"
            )
            row["future_team_value_logit"] = (
                player_value if player_value is not None else None
            )
        else:
            row["team_context_applied"] = context_value is not None
            row["team_context_blocker"] = (
                None if context_value is not None else "missing_model_feature_value"
            )
            row["team_context_uncertainty_proxy"] = (
                1.0
                / math.sqrt(
                    1.0
                    + max(
                        0.0,
                        min(
                            float(row.get("prior_team_support") or 0.0),
                            float(row.get("roster_continuity_support") or 0.0),
                        ),
                    )
                )
                if context_value is not None
                else None
            )
        row["team_context_support"] = int(
            min(
                int(row.get("prior_team_support") or 0),
                int(row.get("roster_continuity_support") or 0),
            )
        )
        row["team_context_missing"] = binding is None or context_value is None
        if binding is None:
            row["team_context_uncertainty_proxy"] = None
            row["team_context_uncertainty_status"] = "not_available"
        else:
            row["team_context_uncertainty_status"] = (
                "support_proxy" if context_value is not None else "not_available"
            )
        output_rows.append(row)
    output_rows.sort(key=lambda row: str(row.get("team_id") or ""))
    row_payload = [dict(row) for row in output_rows]
    rows_hash = _sha256_bytes(_json_bytes(row_payload))
    component = dict(component)
    component["rows_sha256"] = rows_hash
    component["binding_schema_version"] = TEAM_CONTEXT_BINDING_SCHEMA_VERSION
    component["strict_prior_timing"] = "fit_rows_strictly_before_cutoff"
    component["exact_five_roster"] = True
    component["source_receipt_sha256"] = source["source_receipt_sha256"]
    component["source_identity_sha256"] = source["source_identity_sha256"]
    if binding_error:
        component["binding_error"] = binding_error
    status = "research_only" if not blockers else "research_only_partial"
    model_info = {
        "receipt_sha256": _sha256_bytes(
            _json_bytes({key: value for key, value in model_receipt.items() if key != "receipt_sha256"})
        ),
        "declared_receipt_sha256": str(model_receipt.get("receipt_sha256") or ""),
        "variant": str(model_receipt.get("variant") or ""),
        "fit_window_end": str(model_receipt.get("fit_window_end") or ""),
        "fit_game_identity_sha256": identity_sha256(
            model_receipt.get("fit_game_ids") or ()
        ),
        "model_parameter_sha256": component.get("model_parameter_sha256"),
    }
    if model_receipt_file is not None:
        model_info["receipt_file"] = _bind_file_record(
            model_receipt_file,
            "model receipt",
        )
    if model_artifact_file is not None:
        model_info["artifact_file"] = _bind_file_record(
            model_artifact_file,
            "model artifact",
        )
    return TeamContextBuild(
        status=status,
        blockers=tuple(sorted(blockers)),
        rows=tuple(row_payload),
        source=source,
        model=model_info,
        component=component,
        snapshot_receipt_sha256=str(snapshot.receipt.get("receipt_sha256") or ""),
    )


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def write_team_context_artifact(
    output_root: Path,
    build: TeamContextBuild,
) -> dict[str, Any]:
    """Write the immutable JSON artifact, receipt, and manifest."""

    raw_root = Path(output_root)
    if raw_root.is_symlink():
        raise FutureTeamContextError("team-context output directory is a symlink")
    root = raw_root.resolve()
    if root.exists() and (root.is_symlink() or any(root.iterdir())):
        raise FutureTeamContextError("team-context output directory is not empty")
    root.mkdir(parents=True, exist_ok=True)
    source = dict(build.source)
    if isinstance(source.get("source_receipt_file"), Mapping):
        source["source_receipt_file"] = _bind_file_record(
            source["source_receipt_file"],
            "source receipt",
        )
    model = dict(build.model)
    for key, label in (
        ("receipt_file", "model receipt"),
        ("artifact_file", "model artifact"),
    ):
        if isinstance(model.get(key), Mapping):
            model[key] = _bind_file_record(model[key], label)
    artifact_path = root / "future-team-context.json"
    receipt_path = root / "future-team-context-receipt.json"
    manifest_path = root / "manifest.json"
    artifact_payload = {
        "schema_version": TEAM_CONTEXT_ARTIFACT_SCHEMA_VERSION,
        "status": build.status,
        "authority": dict(TEAM_CONTEXT_AUTHORITY),
        "source": source,
        "model": model,
        "component": dict(build.component),
        "rows": [dict(row) for row in build.rows],
        "blockers": list(build.blockers),
    }
    artifact_path.write_bytes(_json_bytes(artifact_payload))
    receipt_payload: dict[str, Any] = {
        "schema_version": TEAM_CONTEXT_RECEIPT_SCHEMA_VERSION,
        "status": build.status,
        "authority": dict(TEAM_CONTEXT_AUTHORITY),
        "source": source,
        "model": model,
        "component": dict(build.component),
        "artifact": _file_record(artifact_path),
        "snapshot_receipt_sha256": build.snapshot_receipt_sha256,
        "row_count": len(build.rows),
        "blockers": list(build.blockers),
    }
    receipt_payload["receipt_sha256"] = _sha256_bytes(_json_bytes(receipt_payload))
    receipt_path.write_bytes(_json_bytes(receipt_payload))
    manifest: dict[str, Any] = {
        "schema_version": TEAM_CONTEXT_ARTIFACT_SCHEMA_VERSION,
        "status": build.status,
        "authority": dict(TEAM_CONTEXT_AUTHORITY),
        "source_receipt_sha256": source["source_receipt_sha256"],
        "receipt": _file_record(receipt_path),
        "artifact": _file_record(artifact_path),
        "blockers": list(build.blockers),
    }
    manifest["manifest_sha256"] = _sha256_bytes(_json_bytes(manifest))
    manifest_path.write_bytes(_json_bytes(manifest))
    return {
        "artifact_path": str(artifact_path),
        "receipt_path": str(receipt_path),
        "manifest_path": str(manifest_path),
        "receipt": receipt_payload,
        "manifest": manifest,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-receipt", required=True, type=Path)
    parser.add_argument("--source-receipt-sha256", required=True)
    parser.add_argument("--model-receipt", required=True, type=Path)
    parser.add_argument("--model-receipt-sha256", required=True)
    parser.add_argument("--model-artifact", required=True, type=Path)
    parser.add_argument("--model-artifact-sha256", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    source_root = args.source_root.absolute()
    source_receipt_path = args.source_receipt.absolute()
    source_receipt = _load_json(source_receipt_path, "source receipt")
    source_receipt_file = _verified_file_record(source_receipt_path, "source receipt")
    source_paths = _source_file_paths(
        source_root,
        source_receipt_path,
        source_receipt,
        expected_source_receipt_sha256=args.source_receipt_sha256,
    )
    model_receipt_path = args.model_receipt.absolute()
    model_artifact_path = args.model_artifact.absolute()
    model_receipt_file = _verified_file_record(model_receipt_path, "model receipt")
    model_artifact_file = _verified_file_record(model_artifact_path, "model artifact")
    if model_receipt_file["sha256"] != str(args.model_receipt_sha256).lower():
        raise FutureTeamContextError("model receipt file hash changed")
    if model_artifact_file["sha256"] != str(args.model_artifact_sha256).lower():
        raise FutureTeamContextError("model artifact file hash changed")
    model_receipt = _load_json(model_receipt_path, "model receipt")
    model, loaded_receipt = load_final_fit_model(
        model_artifact_path,
        model_receipt_path,
        source_receipt=source_receipt,
    )
    if loaded_receipt.get("receipt_sha256") != model_receipt.get("receipt_sha256"):
        raise FutureTeamContextError("model receipt changed while loading")
    maps = pd.read_parquet(source_paths["maps"])
    players = pd.read_parquet(source_paths["players"])
    teams = pd.read_parquet(source_paths["teams"])
    build = build_team_context_artifact(
        maps,
        players,
        teams,
        source_receipt=source_receipt,
        model=model,
        model_receipt=model_receipt,
        source_receipt_file=source_receipt_file,
        model_receipt_file=model_receipt_file,
        model_artifact_file=model_artifact_file,
    )
    written = write_team_context_artifact(args.output_root, build)
    print(
        json.dumps(
            {
                "status": build.status,
                "blockers": list(build.blockers),
                "rows": len(build.rows),
                "receipt_sha256": written["receipt"]["receipt_sha256"],
                "output_root": str(Path(args.output_root).resolve()),
                "team_context_component": build.component.get("status"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "TEAM_CONTEXT_ARTIFACT_SCHEMA_VERSION",
    "TEAM_CONTEXT_AUTHORITY",
    "TEAM_CONTEXT_RECEIPT_SCHEMA_VERSION",
    "FutureTeamContextError",
    "TeamContextBuild",
    "build_team_context_artifact",
    "write_team_context_artifact",
]
