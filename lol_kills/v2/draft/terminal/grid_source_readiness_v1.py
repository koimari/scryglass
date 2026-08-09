"""Freeze the observed GRID pre-start draft-event capability boundary.

This is a private retrospective capability assay over five already-local,
hash-pinned archives.  It proves only that the observed feed shape carries a
complete pick/ban sequence before the map-start event.  It does not establish
prospective latency, pre-start role assignments, model validity, or betting
authority, and none of the completed archives may count as future evidence.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import zipfile

from lol_kills.etl.grid_series_events import (
    RECEIPT_ENVELOPE_SCHEMA,
    _transaction_from_message,
    transaction_sequence,
)
from lol_kills.v2.data.common import sha256_canonical_object
from lol_kills.v2.ratings.player.multileague_v3_future_protocol import (
    FUTURE_SEALED_START,
)

from .future_prediction_ledger import AUTHORITY_KEYS


ROOT = Path(__file__).resolve().parents[4]
SCHEMA_VERSION = "scryglass:grid-terminal-draft-source-readiness:v1"
RESULT_STATE = "RETROSPECTIVE_GRID_PRESTART_PICK_BAN_CAPABILITY_OBSERVED"
SOURCE_LOCATOR = "lol_kills/v2/draft/terminal/grid_source_readiness_v1.py"
ADAPTER_LOCATOR = "lol_kills/v2/draft/terminal/grid_future_source_v1.py"
TRANSPORT_LOCATOR = "lol_kills/etl/grid_series_events.py"
DEFAULT_OUTPUT = Path(
    "data/lol/v2/models/draft-terminal/grid-source-readiness-v1.json"
)
DEFAULT_CATALOG = (
    Path.home()
    / ".codex"
    / "skills"
    / "query-grid-research"
    / "assets"
    / "grid-capability-catalog.v1.json"
)
CATALOG_SCHEMA_VERSION = "scryglass.grid.capability-catalog.v1"
CATALOG_VERSION = "1.0.0"
CATALOG_RAW_SHA256 = (
    "dbf09c1f7727fb61e981ace5a1b635f96d16ddc5410d1d687b60f23589f8db8d"
)
CATALOG_CANONICAL_SHA256 = (
    "94fb8703d8bcdaab416c1b5f8ce727d5f486789267ae66e5b06f784766d127ed"
)
ARCHIVE_BINDINGS = {
    "events_2974293_grid.jsonl.zip": (
        624470,
        "6fd44641cf87fc3546a0836f447d9bb21f1e904bb2d2573d572ee5cef861da91",
    ),
    "events_2974294_grid.jsonl.zip": (
        601460,
        "4d008a30cca8b317855f12ef17ceac61dbffb237f372fd1806480e8e2e677ffa",
    ),
    "events_2974295_grid.jsonl.zip": (
        734394,
        "3c3310bea2a7e19823c3c3ba6a6844a2d25a52c783ab1e9880f483a0c0493efd",
    ),
    "events_2974296_grid.jsonl.zip": (
        587598,
        "096b06562d8a92054d2cd3632ab6b311cfe94c1a672a008789768a7e23471e06",
    ),
    "events_2974297_grid.jsonl.zip": (
        720487,
        "81cc43fe1f0b149721a43315e11b75a9b684769077f2a24bb5bf0b3175e35f75",
    ),
}
SOURCE_LOCKS = (SOURCE_LOCATOR, ADAPTER_LOCATOR, TRANSPORT_LOCATOR)
CLAIM_CEILING = (
    "Private retrospective source-capability evidence only. The five completed "
    "archives do not qualify as future predictions, GRID did not expose "
    "pre-start role assignment in the observed events, and this receipt grants "
    "no model, probability, odds, recommendation, or betting authority."
)


class GridSourceReadinessError(RuntimeError):
    """The GRID source capability receipt or its pinned inputs drifted."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise GridSourceReadinessError("GRID readiness value is not canonical") from exc


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp(value: Any, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise GridSourceReadinessError(f"{field} must be RFC-3339") from exc
    if parsed.tzinfo is None:
        raise GridSourceReadinessError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _source_record(root: Path, locator: str) -> dict[str, Any]:
    path = root / locator
    if not path.is_file():
        raise GridSourceReadinessError(f"GRID source implementation missing: {locator}")
    return {
        "locator": locator,
        "bytes": path.stat().st_size,
        "raw_sha256": _sha256_path(path),
    }


def _load_catalog(path: Path) -> tuple[bytes, dict[str, Any]]:
    try:
        raw = path.expanduser().read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GridSourceReadinessError("pinned GRID capability catalog unavailable") from exc
    if not isinstance(value, dict):
        raise GridSourceReadinessError("GRID capability catalog must be an object")
    if _sha256_bytes(raw) != CATALOG_RAW_SHA256:
        raise GridSourceReadinessError("GRID capability catalog raw hash drifted")
    unsigned = {key: item for key, item in value.items() if key != "catalog_sha256"}
    catalog_canonical_sha256 = hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if (
        value.get("schema_version") != CATALOG_SCHEMA_VERSION
        or value.get("catalog_version") != CATALOG_VERSION
        or value.get("catalog_sha256") != CATALOG_CANONICAL_SHA256
        or catalog_canonical_sha256 != CATALOG_CANONICAL_SHA256
    ):
        raise GridSourceReadinessError("GRID capability catalog identity drifted")
    capabilities = {
        row.get("capability"): row.get("status")
        for row in value.get("capabilities") or []
        if isinstance(row, Mapping)
    }
    if capabilities.get("series_events_websocket") != "locally_observed_and_configured":
        raise GridSourceReadinessError("GRID Series Events capability is not observed")
    return raw, value


def _catalog_archive_receipts(catalog: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    observations = catalog.get("local_series_events_observations")
    if not isinstance(observations, Mapping):
        raise GridSourceReadinessError("GRID catalog lacks local event observations")
    rows = observations.get("archive_receipts")
    if not isinstance(rows, list):
        raise GridSourceReadinessError("GRID catalog archive receipts are missing")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        name = Path(str(row.get("path") or "")).name
        if name in ARCHIVE_BINDINGS:
            result[name] = dict(row)
    if set(result) != set(ARCHIVE_BINDINGS):
        raise GridSourceReadinessError("GRID catalog pinned archive set changed")
    return result


def _assay_archive(
    *, root: Path, name: str, catalog_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    expected_bytes, expected_hash = ARCHIVE_BINDINGS[name]
    path = root / "data/lol/warehouse/raw_grid" / name
    if (
        not path.is_file()
        or path.stat().st_size != expected_bytes
        or _sha256_path(path) != expected_hash
    ):
        raise GridSourceReadinessError(f"pinned GRID archive drifted: {name}")
    if (
        catalog_receipt.get("bytes") != expected_bytes
        or catalog_receipt.get("sha256") != expected_hash
        or Path(str(catalog_receipt.get("path") or "")).name != name
    ):
        raise GridSourceReadinessError(f"GRID catalog archive receipt drifted: {name}")
    expected_series = name.split("_", 2)[1]
    draft_rows: list[dict[str, Any]] = []
    first_start: dict[str, Any] | None = None
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
        if len(members) != 1 or not members[0].endswith(".jsonl"):
            raise GridSourceReadinessError(f"GRID archive member set changed: {name}")
        with archive.open(members[0]) as stream:
            for raw_line in stream:
                transaction = _transaction_from_message(raw_line.rstrip(b"\r\n"))
                if str(transaction.get("seriesId") or "") != expected_series:
                    raise GridSourceReadinessError(
                        f"GRID archive series identity changed: {name}"
                    )
                tx_sequence = transaction_sequence(transaction)
                if tx_sequence is None:
                    raise GridSourceReadinessError(
                        f"GRID archive transaction sequence missing: {name}"
                    )
                occurred_at = _timestamp(
                    transaction.get("occurredAt"), "GRID transaction occurredAt"
                )
                events = transaction.get("events")
                if not isinstance(events, list):
                    raise GridSourceReadinessError("GRID transaction events changed")
                for event in events:
                    if not isinstance(event, Mapping):
                        raise GridSourceReadinessError("GRID event is not an object")
                    event_type = event.get("type")
                    if event_type in {
                        "team-picked-character",
                        "team-banned-character",
                    }:
                        actor = event.get("actor")
                        delta = event.get("seriesStateDelta")
                        if not isinstance(actor, Mapping) or not isinstance(delta, Mapping):
                            raise GridSourceReadinessError("GRID draft event shape changed")
                        actor_state = actor.get("state")
                        games = delta.get("games")
                        if (
                            actor.get("type") != "team"
                            or not isinstance(actor_state, Mapping)
                            or actor_state.get("side") not in {"blue", "red"}
                            or not isinstance(games, list)
                            or len(games) != 1
                            or not isinstance(games[0], Mapping)
                        ):
                            raise GridSourceReadinessError("GRID draft side/game shape changed")
                        actions = games[0].get("draftActions")
                        if (
                            not isinstance(actions, list)
                            or len(actions) != 1
                            or not isinstance(actions[0], Mapping)
                        ):
                            raise GridSourceReadinessError("GRID draft action delta changed")
                        action = actions[0]
                        kind = "pick" if event_type == "team-picked-character" else "ban"
                        if action.get("type") != kind:
                            raise GridSourceReadinessError("GRID draft event/action mismatch")
                        draftable = action.get("draftable")
                        if not isinstance(draftable, Mapping):
                            raise GridSourceReadinessError("GRID draftable shape changed")
                        draft_rows.append(
                            {
                                "slot": int(action.get("sequenceNumber")),
                                "kind": kind,
                                "side": actor_state.get("side"),
                                "transaction_sequence": tx_sequence,
                                "occurred_at": occurred_at,
                                "game_id": str(games[0].get("id") or ""),
                                "action_id": str(action.get("id") or ""),
                                "character_id": str(draftable.get("id") or ""),
                                "champion_name": str(draftable.get("name") or ""),
                            }
                        )
                    elif event_type == "series-started-game":
                        target = event.get("target")
                        if not isinstance(target, Mapping) or target.get("type") != "game":
                            raise GridSourceReadinessError("GRID start target shape changed")
                        first_start = {
                            "transaction_sequence": tx_sequence,
                            "occurred_at": occurred_at,
                            "game_id": str(target.get("id") or ""),
                        }
                        break
                if first_start is not None:
                    break
    if first_start is None:
        raise GridSourceReadinessError(f"GRID archive has no map start: {name}")
    if len(draft_rows) != 20 or [row["slot"] for row in draft_rows] != list(range(1, 21)):
        raise GridSourceReadinessError(f"GRID archive draft is incomplete: {name}")
    if [row["transaction_sequence"] for row in draft_rows] != list(range(3, 23)):
        raise GridSourceReadinessError(f"GRID archive draft transaction shape changed: {name}")
    if first_start["transaction_sequence"] != 23:
        raise GridSourceReadinessError(f"GRID archive start sequence changed: {name}")
    if len({row["game_id"] for row in draft_rows}) != 1 or (
        draft_rows[0]["game_id"] != first_start["game_id"]
    ):
        raise GridSourceReadinessError(f"GRID archive draft/start game differs: {name}")
    if any(not row["action_id"] or not row["character_id"] or not row["champion_name"] for row in draft_rows):
        raise GridSourceReadinessError(f"GRID archive draft identifiers missing: {name}")
    if (
        len({row["action_id"] for row in draft_rows}) != 20
        or len({row["character_id"] for row in draft_rows}) != 20
        or len({row["champion_name"] for row in draft_rows}) != 20
    ):
        raise GridSourceReadinessError(f"GRID archive draft identifiers duplicate: {name}")
    counts = {
        f"{side}_{kind}": sum(
            row["side"] == side and row["kind"] == kind for row in draft_rows
        )
        for side in ("blue", "red")
        for kind in ("pick", "ban")
    }
    if any(value != 5 for value in counts.values()):
        raise GridSourceReadinessError(f"GRID archive pick/ban counts changed: {name}")
    if not all(
        current["occurred_at"] >= previous["occurred_at"]
        for previous, current in zip(draft_rows, draft_rows[1:])
    ):
        raise GridSourceReadinessError(f"GRID archive draft time moved backwards: {name}")
    lead = (first_start["occurred_at"] - draft_rows[-1]["occurred_at"]).total_seconds()
    if lead <= 0:
        raise GridSourceReadinessError(f"GRID archive draft did not precede start: {name}")
    return {
        "series_id": expected_series,
        "archive_locator": f"data/lol/warehouse/raw_grid/{name}",
        "archive_bytes": expected_bytes,
        "archive_raw_sha256": expected_hash,
        "draft_action_count": 20,
        "pick_count": 10,
        "ban_count": 10,
        "counts_by_side_and_kind": counts,
        "draft_action_slots": {"first": 1, "last": 20, "contiguous": True},
        "draft_transaction_sequences": {
            "first": 3,
            "last": 22,
            "strictly_increasing": True,
        },
        "first_map_start_transaction_sequence": 23,
        "all_draft_actions_strictly_before_map_start": True,
        "last_draft_to_map_start_seconds": lead,
        "one_exact_game_identity_across_draft_and_start": True,
        "unique_action_character_and_champion_identifiers": True,
        "post_start_transactions_parsed": False,
        "completed_archive_qualifies_future_evidence": False,
    }


def _assays(root: Path, catalog: Mapping[str, Any]) -> list[dict[str, Any]]:
    receipts = _catalog_archive_receipts(catalog)
    return [
        _assay_archive(root=root, name=name, catalog_receipt=receipts[name])
        for name in sorted(ARCHIVE_BINDINGS)
    ]


def _capability_conclusion(assays: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    leads = [float(row["last_draft_to_map_start_seconds"]) for row in assays]
    return {
        "archive_count": len(assays),
        "terminal_pick_ban_prestart_observed_in_all_archives": (
            len(assays) == len(ARCHIVE_BINDINGS)
            and all(
                row.get("all_draft_actions_strictly_before_map_start") is True
                and row.get("draft_action_count") == 20
                for row in assays
            )
        ),
        "observed_last_draft_to_start_seconds_min": min(leads),
        "observed_last_draft_to_start_seconds_max": max(leads),
        "observed_latency_is_not_prospective_authority": True,
        "prestart_role_assignment_available_from_grid": False,
        "reviewed_separate_role_assignment_required": True,
        "prospective_system_receipts_required": True,
        "retrospective_archives_qualify_future_evidence": False,
    }


def build_grid_source_readiness_v1(
    *,
    root: Path = ROOT,
    catalog_path: Path = DEFAULT_CATALOG,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    raw_catalog, catalog = _load_catalog(catalog_path)
    assays = _assays(root, catalog)
    conclusion = _capability_conclusion(assays)
    if not conclusion["terminal_pick_ban_prestart_observed_in_all_archives"]:
        raise GridSourceReadinessError("GRID pre-start draft capability did not pass")
    locked = clock()
    if not isinstance(locked, datetime) or locked.tzinfo is None:
        raise GridSourceReadinessError("GRID readiness clock must be timezone-aware")
    locked = locked.astimezone(timezone.utc)
    if locked >= FUTURE_SEALED_START.replace(tzinfo=timezone.utc):
        raise GridSourceReadinessError(
            "GRID source readiness must be locked before the future boundary"
        )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "result_state": RESULT_STATE,
        "locked_at_utc": locked.isoformat(),
        "clock_attestation": {
            "source": "system_utc_clock_sampled_inside_builder",
            "observed_wall_clock_utc": locked.isoformat(),
            "user_supplied_timestamp_allowed": False,
            "lock_time_not_after_builder_observation": True,
        },
        "grid_catalog": {
            "logical_locator": "query-grid-research/assets/grid-capability-catalog.v1.json",
            "schema_version": CATALOG_SCHEMA_VERSION,
            "catalog_version": CATALOG_VERSION,
            "raw_sha256": _sha256_bytes(raw_catalog),
            "canonical_sha256": catalog["catalog_sha256"],
            "generated_at": catalog["generated_at"],
            "external_to_repository": True,
        },
        "retrospective_archive_assays": assays,
        "capability_conclusion": conclusion,
        "prospective_capture_contract": {
            "receipt_envelope_schema": RECEIPT_ENVELOPE_SCHEMA,
            "exact_websocket_message_bytes_and_hash_required": True,
            "system_receive_time_sampled_inside_receiver_required": True,
            "api_key_persisted": False,
            "action_slots_1_through_20_required": True,
            "five_picks_and_five_bans_per_side_required": True,
            "exact_grid_series_game_team_and_side_binding_required": True,
            "target_map_start_must_not_be_received_at_prediction_time": True,
            "separate_reviewed_role_assignment_required": True,
            "ambiguous_or_incomplete_role_assignment_fails_closed": True,
            "map_start_raw_full_state_must_be_sanitized": True,
            "future_prediction_ledger_remains_required": True,
        },
        "source_locks": [_source_record(root, locator) for locator in SOURCE_LOCKS],
        "authority": {name: False for name in AUTHORITY_KEYS},
        "claim_ceiling": CLAIM_CEILING,
    }
    payload["artifact_sha256"] = sha256_canonical_object(payload)
    return validate_grid_source_readiness_v1(
        payload, root=root, catalog_path=catalog_path
    )


def validate_grid_source_readiness_v1(
    payload: Mapping[str, Any],
    *,
    root: Path = ROOT,
    catalog_path: Path = DEFAULT_CATALOG,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise GridSourceReadinessError("GRID source readiness must be an object")
    value = dict(payload)
    expected = {
        "schema_version",
        "result_state",
        "locked_at_utc",
        "clock_attestation",
        "grid_catalog",
        "retrospective_archive_assays",
        "capability_conclusion",
        "prospective_capture_contract",
        "source_locks",
        "authority",
        "claim_ceiling",
        "artifact_sha256",
    }
    if set(value) != expected or (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("result_state") != RESULT_STATE
    ):
        raise GridSourceReadinessError("GRID source readiness structure changed")
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if value.get("artifact_sha256") != sha256_canonical_object(unsigned):
        raise GridSourceReadinessError("GRID source readiness canonical hash changed")
    locked = _timestamp(value.get("locked_at_utc"), "locked_at_utc")
    if locked >= FUTURE_SEALED_START.replace(tzinfo=timezone.utc):
        raise GridSourceReadinessError("GRID readiness was not locked pre-boundary")
    if value.get("clock_attestation") != {
        "source": "system_utc_clock_sampled_inside_builder",
        "observed_wall_clock_utc": locked.isoformat(),
        "user_supplied_timestamp_allowed": False,
        "lock_time_not_after_builder_observation": True,
    }:
        raise GridSourceReadinessError("GRID readiness clock attestation changed")
    raw_catalog, catalog = _load_catalog(catalog_path)
    if value.get("grid_catalog") != {
        "logical_locator": "query-grid-research/assets/grid-capability-catalog.v1.json",
        "schema_version": CATALOG_SCHEMA_VERSION,
        "catalog_version": CATALOG_VERSION,
        "raw_sha256": _sha256_bytes(raw_catalog),
        "canonical_sha256": catalog["catalog_sha256"],
        "generated_at": catalog["generated_at"],
        "external_to_repository": True,
    }:
        raise GridSourceReadinessError("GRID readiness catalog binding changed")
    assays = _assays(root, catalog)
    if value.get("retrospective_archive_assays") != assays:
        raise GridSourceReadinessError("GRID retrospective archive assays changed")
    conclusion = _capability_conclusion(assays)
    if value.get("capability_conclusion") != conclusion:
        raise GridSourceReadinessError("GRID capability conclusion changed")
    expected_contract = {
        "receipt_envelope_schema": RECEIPT_ENVELOPE_SCHEMA,
        "exact_websocket_message_bytes_and_hash_required": True,
        "system_receive_time_sampled_inside_receiver_required": True,
        "api_key_persisted": False,
        "action_slots_1_through_20_required": True,
        "five_picks_and_five_bans_per_side_required": True,
        "exact_grid_series_game_team_and_side_binding_required": True,
        "target_map_start_must_not_be_received_at_prediction_time": True,
        "separate_reviewed_role_assignment_required": True,
        "ambiguous_or_incomplete_role_assignment_fails_closed": True,
        "map_start_raw_full_state_must_be_sanitized": True,
        "future_prediction_ledger_remains_required": True,
    }
    if value.get("prospective_capture_contract") != expected_contract:
        raise GridSourceReadinessError("GRID prospective capture contract changed")
    records = value.get("source_locks")
    if (
        not isinstance(records, list)
        or [row.get("locator") for row in records if isinstance(row, Mapping)]
        != list(SOURCE_LOCKS)
    ):
        raise GridSourceReadinessError("GRID readiness source inventory changed")
    for record, locator in zip(records, SOURCE_LOCKS):
        if record != _source_record(root, locator):
            raise GridSourceReadinessError(f"GRID readiness source drifted: {locator}")
    authority = value.get("authority")
    if (
        not isinstance(authority, Mapping)
        or set(authority) != set(AUTHORITY_KEYS)
        or any(authority.values())
    ):
        raise GridSourceReadinessError("GRID readiness exceeds authority")
    if value.get("claim_ceiling") != CLAIM_CEILING:
        raise GridSourceReadinessError("GRID readiness claim ceiling changed")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        payload = build_grid_source_readiness_v1(
            root=args.root,
            catalog_path=args.catalog,
        )
        raw_sha256 = ledger_write_no_clobber(args.out, payload)
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
        print(str(exc))
        return 2
    print(
        json.dumps(
            {
                "output": str(args.out),
                "raw_sha256": raw_sha256,
                "artifact_sha256": payload["artifact_sha256"],
                "locked_at_utc": payload["locked_at_utc"],
            },
            sort_keys=True,
        )
    )
    return 0


def ledger_write_no_clobber(path: Path, payload: Mapping[str, Any]) -> str:
    from .future_prediction_ledger import write_no_clobber

    return write_no_clobber(path, payload)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CATALOG_CANONICAL_SHA256",
    "CATALOG_RAW_SHA256",
    "DEFAULT_OUTPUT",
    "GridSourceReadinessError",
    "build_grid_source_readiness_v1",
    "validate_grid_source_readiness_v1",
]
