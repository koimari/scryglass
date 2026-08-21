"""Trusted adapters for public Draft Score artifacts.

The future-value scorer consumes a narrow atom frame.  Public pack JSON and
promotion outputs use wider publication schemas, so this module performs the
conversion only after it verifies the release manifest, source receipt, and
artifact bytes.  Every generated receipt is written beside the research
output and can be replayed by a later process.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from lol_kills.research import future_value_draft_score as core
from lol_kills.v2.tierlists.accepted_census import identity_sha256


ADAPTER_SCHEMA = "scryglass:future-value-draft-score-adapter:v1"
PUBLIC_DRAFT_RECORDS_SCHEMA = "scryglass:draft-records:v1"
ARTIFACT_RECEIPT_SCHEMA = "scryglass:public-draft-artifact-receipt:v1"
CROSSFIT_RECEIPT_SCHEMA = "scryglass:public-crossfit-draft-receipt:v1"
CANONICAL_EDGE_COMPONENTS = core.STATIC_COMPOSITION_COMPONENTS
_CROSSFIT_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "source_receipt_sha256",
        "source_identity_sha256",
        "accepted_game_ids",
        "fold_id",
        "model_id",
        "fit_game_ids",
        "fit_game_identity_sha256",
        "producer_timing",
        "artifact_locator",
        "artifact_bytes",
        "artifact_sha256",
        "receipt_sha256",
    }
)


class DraftScoreAdapterError(ValueError):
    """A public Draft artifact cannot be safely adapted."""


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise DraftScoreAdapterError("adapter value is not canonical JSON") from error


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _file(path: Path | str, label: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    if candidate.is_symlink() or not candidate.is_file():
        raise DraftScoreAdapterError(f"{label} is missing or unsafe")
    current = candidate
    system_aliases = {Path("/tmp"), Path("/var"), Path("/private/tmp"), Path("/private/var")}
    while True:
        if current.is_symlink() and current not in system_aliases:
            raise DraftScoreAdapterError(f"{label} uses a symlink path")
        parent = current.parent
        if parent == current:
            break
        if current in system_aliases:
            break
        current = parent
    return candidate.resolve()


def _json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DraftScoreAdapterError(f"{label} cannot be read") from error
    if not isinstance(value, dict):
        raise DraftScoreAdapterError(f"{label} must be a JSON object")
    return value, raw


def _write_json(path: Path, value: Mapping[str, Any]) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _canonical_json(value) + b"\n"
    path.write_bytes(raw)
    return raw


def _require_sha(value: object, field: str) -> str:
    text = str(value or "").lower()
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise DraftScoreAdapterError(f"{field} must be a SHA-256")
    return text


def _verify_file(path: Path, *, expected_bytes: object, expected_sha256: object, label: str) -> str:
    if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int) or expected_bytes < 0:
        raise DraftScoreAdapterError(f"{label} byte count is invalid")
    raw = path.read_bytes()
    digest = _sha_bytes(raw)
    if len(raw) != expected_bytes or digest != _require_sha(expected_sha256, f"{label} sha256"):
        raise DraftScoreAdapterError(f"{label} bytes or SHA-256 changed")
    return digest


def _manifest_entries(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    entries: list[Mapping[str, Any]] = []
    for key in ("files", "assets", "artifacts"):
        value = manifest.get(key)
        if isinstance(value, list):
            entries.extend(item for item in value if isinstance(item, Mapping))
        elif isinstance(value, Mapping):
            entries.extend(item for item in value.values() if isinstance(item, Mapping))
    return entries


def _manifest_record(manifest: Mapping[str, Any], draft_path: Path, manifest_path: Path) -> Mapping[str, Any]:
    relative = draft_path.relative_to(manifest_path.parent).as_posix() if draft_path.is_relative_to(manifest_path.parent) else draft_path.name
    candidates = []
    for entry in _manifest_entries(manifest):
        path_value = entry.get("path", entry.get("locator"))
        if str(path_value or "") in {relative, draft_path.name, f"features/{draft_path.name}"}:
            candidates.append(entry)
    if not candidates:
        raise DraftScoreAdapterError("public manifest has no draft_records.json entry")
    record = candidates[0]
    if not isinstance(record.get("bytes"), int) or not str(record.get("sha256") or ""):
        raise DraftScoreAdapterError("public manifest draft artifact record is incomplete")
    return record


def _find_source_identity(value: object) -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in {"source_identity_sha256", "source_identity"} and isinstance(item, str):
                try:
                    return _require_sha(item, "source_identity_sha256")
                except DraftScoreAdapterError:
                    continue
            found = _find_source_identity(item)
            if found:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_source_identity(item)
            if found:
                return found
    return None


def _game_rows(payload: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    games = payload.get("games")
    if isinstance(games, Mapping):
        rows = [(str(game_id), value) for game_id, value in games.items() if isinstance(value, Mapping)]
    elif isinstance(games, list):
        rows = []
        for value in games:
            if not isinstance(value, Mapping):
                continue
            game_id = value.get("game_uid", value.get("game_id", value.get("id")))
            if game_id is not None:
                rows.append((str(game_id), value))
    else:
        raise DraftScoreAdapterError("public draft records games are missing")
    if not rows or len({game_id for game_id, _ in rows}) != len(rows):
        raise DraftScoreAdapterError("public draft records game IDs are not unique")
    return sorted(rows, key=lambda item: item[0])


def _finite(value: object, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise DraftScoreAdapterError(f"{field} is not finite") from error
    if not np.isfinite(parsed):
        raise DraftScoreAdapterError(f"{field} is not finite")
    return parsed


def _component_rows(
    payload: Mapping[str, Any],
    *,
    source_receipt: Mapping[str, Any],
    producer_receipt_sha256: str,
    producer_name: str,
    producer_family: str,
    producer_timing: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for game_id, game in _game_rows(payload):
        if game.get("status") in {"unavailable", "missing"}:
            raise DraftScoreAdapterError(f"public draft record is unavailable: {game_id}")
        edge = game.get("edge_components")
        if not isinstance(edge, Mapping):
            raise DraftScoreAdapterError(f"public draft record has no edge components: {game_id}")
        values = {
            f"composition_{component}_logit": _finite(edge.get(component), f"{game_id} {component}")
            for component in CANONICAL_EDGE_COMPONENTS
        }
        total = _finite(edge.get("total"), f"{game_id} total")
        if abs(total - sum(values.values())) > 1e-5:
            raise DraftScoreAdapterError(f"public draft record total changed: {game_id}")
        date = str(game.get("date") or "").strip()
        if not date:
            raise DraftScoreAdapterError(f"public draft record date is missing: {game_id}")
        rows.append(
            {
                "game_id": game_id,
                "date": date,
                **values,
                "source_receipt_sha256": str(source_receipt["receipt_sha256"]).lower(),
                "source_identity_sha256": str(source_receipt["source_identity_sha256"]).lower(),
                "producer_receipt_sha256": producer_receipt_sha256,
                "producer_name": producer_name,
                "producer_family": producer_family,
                "producer_timing": producer_timing,
            }
        )
    return pd.DataFrame(rows)


def _component_digest(frame: pd.DataFrame) -> str:
    rows = []
    for row in frame.sort_values("game_id", kind="stable").to_dict(orient="records"):
        rows.append(
            {
                "game_id": str(row["game_id"]),
                **{
                    feature: float(row[feature])
                    for feature in core.STATIC_COMPOSITION_FEATURES
                },
            }
        )
    return _sha(rows)


@dataclass(frozen=True)
class PublicDraftAtomAdapterResult:
    frame: pd.DataFrame
    source_receipt: Mapping[str, Any]
    source_receipt_path: Path
    manifest_path: Path
    artifact_path: Path
    artifact_receipt_path: Path
    static_atom_receipt_path: Path
    static_atom_receipt: Mapping[str, Any]
    producer_receipt_path: Path
    producer_receipt: Mapping[str, Any]
    chronological_evaluation_suitable: bool


def adapt_public_descriptive_draft_records(
    draft_records_path: Path | str,
    manifest_path: Path | str,
    source_receipt_path: Path | str,
    *,
    output_dir: Path | str | None = None,
    source_root: Path | str | None = None,
) -> PublicDraftAtomAdapterResult:
    """Convert a verified public descriptive pack asset to five atom columns."""

    draft_path = _file(draft_records_path, "public draft records")
    pack_manifest_path = _file(manifest_path, "public pack manifest")
    source_path = _file(source_receipt_path, "canonical source receipt")
    manifest, _manifest_raw = _json(pack_manifest_path, "public pack manifest")
    draft, draft_raw = _json(draft_path, "public draft records")
    record = _manifest_record(manifest, draft_path, pack_manifest_path)
    _verify_file(draft_path, expected_bytes=record.get("bytes"), expected_sha256=record.get("sha256"), label="public draft records")
    try:
        source_receipt = core._validate_source_receipt_payload(
            _json(source_path, "canonical source receipt")[0],
            source_receipt_path=source_path,
            source_root=source_root or source_path.parent,
        )
    except core.FutureValueDraftScoreError as error:
        raise DraftScoreAdapterError(str(error)) from error
    if draft.get("schema_version") != PUBLIC_DRAFT_RECORDS_SCHEMA or draft.get("authority") != "descriptive" or draft.get("estimand") != "composition_only":
        raise DraftScoreAdapterError("public draft records contract is invalid")
    game_rows = _game_rows(draft)
    draft_ids = tuple(game_id for game_id, _ in game_rows)
    if draft_ids != tuple(source_receipt["accepted_game_ids"]):
        raise DraftScoreAdapterError("public draft records do not match accepted census")
    manifest_identity = _find_source_identity(manifest)
    if manifest_identity is not None and manifest_identity != source_receipt["source_identity_sha256"]:
        raise DraftScoreAdapterError("public manifest source identity changed")
    if str(draft.get("source_identity_sha256") or "").lower() != source_receipt["source_identity_sha256"]:
        raise DraftScoreAdapterError("public draft records source identity changed")
    manifest_release = manifest.get("release_id", manifest.get("pack_id"))
    draft_release = draft.get("release_id", draft.get("pack_id"))
    if manifest_release is not None and draft_release is not None and str(manifest_release) != str(draft_release):
        raise DraftScoreAdapterError("public draft records release identity changed")

    target_dir = Path(output_dir).expanduser() if output_dir is not None else draft_path.parent / "future-value-adapter"
    target_dir.mkdir(parents=True, exist_ok=True)
    artifact_receipt_path = target_dir / "public-draft-records-artifact-receipt.json"
    atom_receipt_path = target_dir / "public-descriptive-atom-receipt.json"
    producer_receipt_path = target_dir / "public-descriptive-producer-receipt.json"
    artifact_locator = str(draft_path)
    artifact_receipt_payload = {
        "schema_version": ARTIFACT_RECEIPT_SCHEMA,
        "artifact_locator": artifact_locator,
        "artifact_bytes": len(draft_raw),
        "artifact_sha256": _sha_bytes(draft_raw),
        "source_receipt_sha256": source_receipt["receipt_sha256"],
        "source_identity_sha256": source_receipt["source_identity_sha256"],
        "release_id": draft.get("release_id", manifest.get("release_id", manifest.get("pack_id"))),
    }
    artifact_receipt_raw = _write_json(artifact_receipt_path, artifact_receipt_payload)
    artifact_receipt_sha256 = _sha_bytes(artifact_receipt_raw)
    fit_through = draft.get("fit_through") or draft.get("artifact_as_of")
    dates = [pd.Timestamp(game.get("date"), tz="UTC") for _, game in game_rows]
    if fit_through:
        fit_cutoff = pd.Timestamp(fit_through)
        if fit_cutoff.tzinfo is None:
            fit_cutoff = fit_cutoff.tz_localize("UTC")
        else:
            fit_cutoff = fit_cutoff.tz_convert("UTC")
    else:
        fit_cutoff = None
    suitable = fit_cutoff is None or all(date > fit_cutoff for date in dates)
    static_receipt_payload: dict[str, Any] = {
        "schema_version": core.STATIC_ATOM_RECEIPT_SCHEMA,
        "producer_name": "public_descriptive_draft_records",
        "producer_family": "static_composition",
        "artifact_locator": artifact_locator,
        "artifact_bytes": len(draft_raw),
        "artifact_sha256": _sha_bytes(draft_raw),
        "artifact_receipt_locator": str(artifact_receipt_path),
        "artifact_receipt_bytes": len(artifact_receipt_raw),
        "artifact_receipt_sha256": artifact_receipt_sha256,
        "source_receipt_sha256": source_receipt["receipt_sha256"],
        "source_identity_sha256": source_receipt["source_identity_sha256"],
        "feature_names": list(core.STATIC_COMPOSITION_FEATURES),
        "component_values_sha256": "",
        "release_id": draft.get("release_id", manifest.get("release_id", manifest.get("pack_id"))),
        "fit_through": fit_through,
        "chronological_evaluation_suitable": suitable,
        "chronological_evaluation_reason": None if suitable else "static artifact fit cutoff is later than scored rows",
    }
    frame = _component_rows(
        draft,
        source_receipt=source_receipt,
        producer_receipt_sha256="",
        producer_name="public_descriptive_draft_records",
        producer_family="static_composition",
        producer_timing="pregame_strict_prior",
    )
    static_receipt_payload["component_values_sha256"] = _component_digest(frame)
    static_receipt_payload["receipt_sha256"] = _sha(static_receipt_payload)
    _write_json(atom_receipt_path, static_receipt_payload)
    producer_payload = {
        "schema_version": core.SCHEMA_VERSION,
        "source_receipt_sha256": source_receipt["receipt_sha256"],
        "source_identity_sha256": source_receipt["source_identity_sha256"],
        "accepted_game_count": len(draft_ids),
        "accepted_game_ids": list(draft_ids),
        "producer_name": "public_descriptive_draft_records",
        "producer_family": "static_composition",
        "fit_game_count": 1,
        "fit_game_ids": [draft_ids[0]],
        "fit_game_identity_sha256": core.identity_sha256((draft_ids[0],)),
        "fit_window_start": str(pd.Timestamp(dates[0] - pd.Timedelta(days=1)).isoformat()).replace("+00:00", "Z"),
        "fit_window_end": str(pd.Timestamp(dates[0]).isoformat()).replace("+00:00", "Z"),
        "fit_game_dates": {draft_ids[0]: str(pd.Timestamp(dates[0] - pd.Timedelta(days=1)).isoformat()).replace("+00:00", "Z")},
        "fold_id": "public-pack-descriptive",
        "series_safe_evidence": {
            "series_safe": True,
            "fit_validation_disjoint": True,
            "source_type": "public_pack",
            "series_column": "game_id",
            "cluster_identity_sha256": source_receipt["source_identity_sha256"],
        },
        "producer_timing": "pregame_strict_prior",
        "artifact_locator": artifact_locator,
        "artifact_bytes": len(draft_raw),
        "artifact_sha256": _sha_bytes(draft_raw),
        "artifact_receipt_locator": str(artifact_receipt_path),
        "artifact_receipt_bytes": len(artifact_receipt_raw),
        "artifact_receipt_sha256": artifact_receipt_sha256,
    }
    producer_payload["receipt_sha256"] = _sha(producer_payload)
    _write_json(producer_receipt_path, producer_payload)
    producer_hash = str(producer_payload["receipt_sha256"])
    frame["producer_receipt_sha256"] = producer_hash
    return PublicDraftAtomAdapterResult(
        frame=frame,
        source_receipt=source_receipt,
        source_receipt_path=source_path,
        manifest_path=pack_manifest_path,
        artifact_path=draft_path,
        artifact_receipt_path=artifact_receipt_path,
        static_atom_receipt_path=atom_receipt_path,
        static_atom_receipt=static_receipt_payload,
        producer_receipt_path=producer_receipt_path,
        producer_receipt=producer_payload,
        chronological_evaluation_suitable=suitable,
    )


def load_public_descriptive_draft_atoms(*args: Any, **kwargs: Any) -> PublicDraftAtomAdapterResult:
    """Alias with an explicit loader name for research callers."""

    return adapt_public_descriptive_draft_records(*args, **kwargs)


@dataclass(frozen=True)
class PublicCrossfitAtomAdapterResult:
    frame: pd.DataFrame
    receipt: Mapping[str, Any]
    rows_path: Path
    receipt_path: Path


def adapt_public_crossfit_draft_rows(
    rows_path: Path | str,
    receipt_path: Path | str,
    source_receipt_path: Path | str,
    *,
    source_root: Path | str | None = None,
) -> PublicCrossfitAtomAdapterResult:
    """Convert receipt-bound public cross-fit component rows.

    The adapter accepts JSON object rows and JSON arrays.  A parquet source is
    intentionally outside this adapter because its sidecar receipt must name
    the exact serialization bytes first.
    """

    rows_file = _file(rows_path, "crossfit Draft rows")
    promotion_receipt_file = _file(receipt_path, "crossfit Draft receipt")
    source_file = _file(source_receipt_path, "canonical source receipt")
    rows_payload, rows_raw = _json(rows_file, "crossfit Draft rows")
    receipt, _receipt_raw = _json(promotion_receipt_file, "crossfit Draft receipt")
    source, _source_raw = _json(source_file, "canonical source receipt")
    try:
        verified_source = core._validate_source_receipt_payload(
            source,
            source_receipt_path=source_file,
            source_root=source_root or source_file.parent,
        )
    except core.FutureValueDraftScoreError as error:
        raise DraftScoreAdapterError(str(error)) from error
    required_receipt_fields = {
        "schema_version",
        "source_receipt_sha256",
        "source_identity_sha256",
        "accepted_game_ids",
        "fold_id",
        "model_id",
        "fit_game_ids",
        "fit_game_identity_sha256",
        "producer_timing",
        "artifact_locator",
        "artifact_bytes",
        "artifact_sha256",
        "receipt_sha256",
    }
    if not required_receipt_fields.issubset(receipt):
        raise DraftScoreAdapterError("crossfit Draft receipt is incomplete")
    unknown_receipt_fields = sorted(set(receipt) - _CROSSFIT_RECEIPT_FIELDS)
    if unknown_receipt_fields:
        raise DraftScoreAdapterError(
            "crossfit Draft receipt has unknown fields: " + ", ".join(unknown_receipt_fields)
        )
    if receipt.get("schema_version") != CROSSFIT_RECEIPT_SCHEMA:
        raise DraftScoreAdapterError("crossfit Draft receipt schema is invalid")
    claimed_receipt_hash = _require_sha(receipt["receipt_sha256"], "crossfit receipt_sha256")
    if _sha({key: value for key, value in receipt.items() if key != "receipt_sha256"}) != claimed_receipt_hash:
        raise DraftScoreAdapterError("crossfit Draft receipt hash changed")
    if receipt["source_receipt_sha256"] != verified_source["receipt_sha256"] or receipt["source_identity_sha256"] != verified_source["source_identity_sha256"]:
        raise DraftScoreAdapterError("crossfit Draft receipt source binding changed")
    if list(receipt["accepted_game_ids"]) != list(verified_source["accepted_game_ids"]):
        raise DraftScoreAdapterError("crossfit Draft receipt census changed")
    if not str(receipt.get("fold_id") or "").strip() or not str(receipt.get("model_id") or "").strip():
        raise DraftScoreAdapterError("crossfit Draft receipt model and fold IDs are required")
    fit_ids = receipt["fit_game_ids"]
    if not isinstance(fit_ids, list) or tuple(str(value) for value in fit_ids) != core._normalise_ids(fit_ids, "crossfit fit game IDs"):
        raise DraftScoreAdapterError("crossfit Draft receipt fit IDs are invalid")
    if receipt["fit_game_identity_sha256"] != identity_sha256(tuple(str(value) for value in fit_ids)):
        raise DraftScoreAdapterError("crossfit Draft receipt fit identity changed")
    if receipt["producer_timing"] not in core._ALLOWED_PRODUCER_TIMINGS:
        raise DraftScoreAdapterError("crossfit Draft receipt timing is not pregame")
    claimed_artifact = receipt.get("artifact_sha256", receipt.get("candidate_artifact_sha256"))
    if claimed_artifact is not None:
        if str(receipt["artifact_locator"]) not in {rows_file.name, str(rows_file)}:
            raise DraftScoreAdapterError("crossfit Draft artifact locator changed")
        _verify_file(rows_file, expected_bytes=receipt.get("artifact_bytes", len(rows_raw)), expected_sha256=claimed_artifact, label="crossfit Draft rows")
    raw_rows = rows_payload.get("rows", rows_payload.get("predictions", rows_payload.get("results")))
    if isinstance(raw_rows, Mapping):
        raw_rows = [dict(value, game_id=key) for key, value in raw_rows.items() if isinstance(value, Mapping)]
    if not isinstance(raw_rows, list) or not raw_rows:
        raise DraftScoreAdapterError("crossfit Draft rows are missing")
    converted: list[dict[str, Any]] = []
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise DraftScoreAdapterError("crossfit Draft row is invalid")
        game_id = str(raw.get("game_id", raw.get("game_uid", raw.get("id", ""))))
        if not game_id:
            raise DraftScoreAdapterError("crossfit Draft row game ID is missing")
        date = str(raw.get("date") or raw.get("game_date") or "").strip()
        if not date:
            raise DraftScoreAdapterError(f"crossfit Draft row date is missing: {game_id}")
        values = {
            "composition_base_logit": _finite(raw.get("crossfit_champion_main", 0.0), "crossfit champion component")
            + _finite(raw.get("crossfit_role_champion", 0.0), "crossfit role component"),
            "composition_ally_synergy_logit": _finite(raw.get("crossfit_ally_synergy", 0.0), "crossfit ally component"),
            "composition_enemy_counter_logit": _finite(raw.get("crossfit_enemy_counter", 0.0), "crossfit enemy component"),
            "composition_same_role_logit": _finite(raw.get("crossfit_same_role", 0.0), "crossfit role-pair component"),
            "composition_archetype_interactions_logit": _finite(raw.get("crossfit_archetype_synergy", 0.0), "crossfit archetype synergy")
            + _finite(raw.get("crossfit_archetype_counter", 0.0), "crossfit archetype counter"),
        }
        converted.append(
            {
                "game_id": game_id,
                "date": date,
                **values,
                "source_receipt_sha256": verified_source["receipt_sha256"],
                "source_identity_sha256": verified_source["source_identity_sha256"],
                "producer_receipt_sha256": str(receipt.get("receipt_sha256") or "").lower(),
                "producer_name": "public_crossfit_draft_score",
                "producer_family": "crossfit_composition",
                "producer_timing": str(receipt["producer_timing"]),
            }
        )
    frame = pd.DataFrame(converted).sort_values("game_id", kind="stable").reset_index(drop=True)
    if tuple(frame["game_id"]) != tuple(verified_source["accepted_game_ids"]):
        raise DraftScoreAdapterError("crossfit rows do not match accepted census")
    return PublicCrossfitAtomAdapterResult(
        frame=frame,
        receipt=receipt,
        rows_path=rows_file,
        receipt_path=promotion_receipt_file,
    )


def load_public_crossfit_draft_atoms(*args: Any, **kwargs: Any) -> PublicCrossfitAtomAdapterResult:
    return adapt_public_crossfit_draft_rows(*args, **kwargs)


__all__ = [
    "ADAPTER_SCHEMA",
    "ARTIFACT_RECEIPT_SCHEMA",
    "CANONICAL_EDGE_COMPONENTS",
    "CROSSFIT_RECEIPT_SCHEMA",
    "DraftScoreAdapterError",
    "PublicCrossfitAtomAdapterResult",
    "PublicDraftAtomAdapterResult",
    "adapt_public_crossfit_draft_rows",
    "adapt_public_descriptive_draft_records",
    "load_public_crossfit_draft_atoms",
    "load_public_descriptive_draft_atoms",
]
