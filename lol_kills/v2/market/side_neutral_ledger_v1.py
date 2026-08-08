"""Admit independently reviewed side-neutral bundles to an outcome-free ledger.

Only complete bundles whose pre-side capture is strictly after the effective
time of an externally hash-pinned independent review can enter. Duplicate map
identities or side bindings fail the entire build. Admission establishes an
evaluation denominator only; outcomes remain sealed and no rating,
probability, odds, EV, recommendation, or betting authority is granted.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Callable, Mapping, Sequence

from lol_kills.v2.ratings.player.multileague_v3_future_protocol import (
    DOMESTIC_LEAGUES,
    FUTURE_SEALED_START,
)
from lol_kills.v2.ratings.player.multileague_v3_side_neutral_protocol_v2 import (
    INDEPENDENT_REVIEW_ENV,
)
from lol_kills.v2.ratings.player.side_neutral_protocol_review_v1 import (
    REVIEW_LOCATOR,
    load_active_side_neutral_protocol_review,
)

from . import side_neutral_capture_bundle_v1 as bundle_module


ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "scryglass:side-neutral-prospective-evaluation-ledger:v1"
RESULT_STATE = "REVIEWED_SIDE_NEUTRAL_BUNDLES_ADMITTED_OUTCOMES_SEALED"
SOURCE_LOCATOR = "lol_kills/v2/market/side_neutral_ledger_v1.py"
DEFAULT_LEDGER = Path(
    "data/lol/v2/evaluation/multileague-v3/side-neutral-prediction-ledger.json"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
OUTCOME_KEYS = bundle_module.OUTCOME_KEYS
AUTHORITY_KEYS = (
    "prospective_collection_authority",
    "outcome_opening_authority",
    "model_validation_authority",
    "player_rating_authority",
    "team_rating_authority",
    "draft_validation_authority",
    "calibration_authority",
    "probability_authority",
    "odds_authority",
    "expected_value_authority",
    "recommendation_authority",
    "betting_authority",
)
CLAIM_CEILING = (
    "Outcome-free independently reviewed prospective collection ledger only. "
    "Entries count toward a future evaluation denominator; outcomes remain "
    "sealed and the ledger is not an evaluation pass, rating authority, "
    "probability, odds, EV, recommendation, or betting authority."
)


class SideNeutralLedgerError(ValueError):
    """A bundle, review, timing chain, or ledger invariant failed closed."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SideNeutralLedgerError("ledger value is not canonical") from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise SideNeutralLedgerError(f"{field} must be a lowercase SHA-256")
    return value


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SideNeutralLedgerError(f"{field} must be non-empty")
    return value.strip()


def _timestamp(value: Any, field: str) -> datetime:
    text = _nonempty(value, field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SideNeutralLedgerError(f"{field} must be RFC-3339") from exc
    if parsed.tzinfo is None:
        raise SideNeutralLedgerError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _clock_sample(clock: Callable[[], datetime]) -> datetime:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise SideNeutralLedgerError("ledger clock must be timezone-aware")
    return value.astimezone(timezone.utc)


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise SideNeutralLedgerError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _strict_object(raw: bytes, field: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                SideNeutralLedgerError(f"non-finite number in {field}: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SideNeutralLedgerError(f"{field} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise SideNeutralLedgerError(f"{field} must be an object")
    return value


def _assert_no_outcomes(value: Any, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = re.sub(r"[^a-z0-9]+", "", str(key).casefold())
            if normalized in OUTCOME_KEYS:
                raise SideNeutralLedgerError(
                    f"event outcome field is forbidden: {path}.{key}"
                )
            _assert_no_outcomes(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_outcomes(item, f"{path}[{index}]")


def _source_record(root: Path) -> dict[str, Any]:
    path = root / SOURCE_LOCATOR
    if not path.is_file() or path.is_symlink():
        raise SideNeutralLedgerError("ledger implementation is unavailable")
    return {
        "locator": SOURCE_LOCATOR,
        "bytes": path.stat().st_size,
        "raw_sha256": _sha256_path(path),
    }


def _bundle_locator(value: str) -> str:
    path = PurePosixPath(_nonempty(value, "bundle locator"))
    prefix = bundle_module.BUNDLE_PREFIX
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or tuple(path.parts[: len(prefix.parts)]) != prefix.parts
        or path.suffix != ".json"
    ):
        raise SideNeutralLedgerError("bundle locator is outside its immutable root")
    return path.as_posix()


def _load_bundle(root: Path, locator: str) -> tuple[bytes, dict[str, Any]]:
    locator = _bundle_locator(locator)
    path = root / locator
    if not path.is_file() or path.is_symlink():
        raise SideNeutralLedgerError(f"bundle is unavailable: {locator}")
    raw = path.read_bytes()
    payload = _strict_object(raw, locator)
    return raw, bundle_module.validate_side_neutral_capture_bundle(
        payload, root=root
    )


def _entry(
    *,
    locator: str,
    raw: bytes,
    bundle: Mapping[str, Any],
    review_effective: datetime,
) -> dict[str, Any]:
    event = bundle["event"]
    timing = bundle["timing"]
    pre_side = _timestamp(timing["pre_side_captured_at_utc"], "pre-side capture")
    actual_start = _timestamp(timing["actual_map_start_utc"], "actual map start")
    if pre_side <= review_effective:
        raise SideNeutralLedgerError(
            "pre-side capture does not follow independent review effective time"
        )
    if actual_start.replace(tzinfo=None) < FUTURE_SEALED_START:
        raise SideNeutralLedgerError("bundle event predates future holdout boundary")
    draft = bundle["input_receipts"]["side_neutral_draft"]["value"]
    binding = draft["side_binding"]["value"]
    selected_child = draft["terminal_draft_prediction"]["value"]
    rating = selected_child["input_receipts"]["ratings_prediction"]["value"]
    map_start = bundle["input_receipts"]["actual_map_start"]["value"]
    return {
        "event_id": event["event_id"],
        "series_id": event["series_id"],
        "game_number": event["game_number"],
        "league": event["league"],
        "patch": event["patch"],
        "blue_organization_id": event["blue_organization_id"],
        "red_organization_id": event["red_organization_id"],
        "roster_change_stratum": rating["event"]["roster_change_stratum"],
        "pre_side_captured_at_utc": timing["pre_side_captured_at_utc"],
        "side_binding_captured_at_utc": timing["side_binding_captured_at_utc"],
        "terminal_draft_captured_at_utc": timing[
            "terminal_draft_captured_at_utc"
        ],
        "actual_map_start_utc": timing["actual_map_start_utc"],
        "side_source_record_id": binding["public_side_source"]["source_record_id"],
        "selected_rating_receipt_artifact_sha256": draft[
            "selected_rating_binding"
        ]["rating_receipt_artifact_sha256"],
        "terminal_draft_artifact_sha256": selected_child["artifact_sha256"],
        "map_start_artifact_sha256": map_start["artifact_sha256"],
        "bundle_locator": locator,
        "bundle_raw_sha256": _sha256_bytes(raw),
        "bundle_artifact_sha256": bundle["artifact_sha256"],
        "admitted_under_review_effective_at_utc": review_effective.isoformat(),
    }


def _support(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    series = {str(entry["series_id"]) for entry in entries}
    league_series = {
        league: {
            str(entry["series_id"])
            for entry in entries
            if entry["league"] == league
        }
        for league in DOMESTIC_LEAGUES
    }
    changed_series = {
        str(entry["series_id"])
        for entry in entries
        if entry["roster_change_stratum"] == "ONE_OR_BOTH_ROSTERS_CHANGED"
    }
    value = {
        "eligible_maps": len(entries),
        "eligible_series": len(series),
        "series_by_domestic_league": {
            league: len(values) for league, values in league_series.items()
        },
        "one_or_both_rosters_changed_series": len(changed_series),
        "overall_series_minimum": 100,
        "each_domestic_league_series_minimum": 20,
        "one_or_both_rosters_changed_series_minimum": 20,
    }
    value["support_met"] = (
        value["eligible_series"] >= value["overall_series_minimum"]
        and all(
            count >= value["each_domestic_league_series_minimum"]
            for count in value["series_by_domestic_league"].values()
        )
        and value["one_or_both_rosters_changed_series"]
        >= value["one_or_both_rosters_changed_series_minimum"]
    )
    return value


def _authority_false() -> dict[str, bool]:
    return {name: False for name in AUTHORITY_KEYS}


def build_side_neutral_ledger(
    *,
    bundle_locators: Sequence[str],
    environment: Mapping[str, str],
    root: Path = ROOT,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    created = _clock_sample(clock)
    review = load_active_side_neutral_protocol_review(
        root=root, environment=environment, as_of=created
    )
    review_effective = _timestamp(
        review["authorization"]["effective_at_utc"], "review effective time"
    )
    if created <= review_effective:
        raise SideNeutralLedgerError("ledger must follow independent review")
    entries: list[dict[str, Any]] = []
    seen_events: set[tuple[str, int]] = set()
    seen_side_records: set[str] = set()
    seen_locators: set[str] = set()
    for raw_locator in bundle_locators:
        locator = _bundle_locator(raw_locator)
        if locator in seen_locators:
            raise SideNeutralLedgerError("duplicate bundle locator")
        seen_locators.add(locator)
        raw, bundle = _load_bundle(root, locator)
        entry = _entry(
            locator=locator,
            raw=raw,
            bundle=bundle,
            review_effective=review_effective,
        )
        event_key = (entry["event_id"], entry["game_number"])
        if event_key in seen_events:
            raise SideNeutralLedgerError(
                "duplicate or ambiguous side binding for one map"
            )
        seen_events.add(event_key)
        side_record = entry["side_source_record_id"]
        if side_record in seen_side_records:
            raise SideNeutralLedgerError("side source record was reused across maps")
        seen_side_records.add(side_record)
        entries.append(entry)
    entries.sort(
        key=lambda item: (
            item["actual_map_start_utc"],
            item["event_id"],
            item["game_number"],
        )
    )
    support = _support(entries)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "result_state": RESULT_STATE,
        "created_at_utc": created.isoformat(),
        "clock_attestation": {
            "source": "system_utc_clock_sampled_inside_builder",
            "observed_wall_clock_utc": created.isoformat(),
            "user_supplied_timestamp_allowed": False,
        },
        "independent_review": {
            "locator": REVIEW_LOCATOR.as_posix(),
            "external_raw_sha256": _sha(
                environment.get(INDEPENDENT_REVIEW_ENV),
                INDEPENDENT_REVIEW_ENV,
            ),
            "review_id": review["review_id"],
            "reviewed_at_utc": review["reviewed_at_utc"],
            "effective_at_utc": review["authorization"]["effective_at_utc"],
            "prospective_collection_authorized": True,
            "outcome_opening_authorized": False,
        },
        "entries": entries,
        "support": support,
        "qualification": {
            "independent_review_present_and_valid": True,
            "every_pre_side_capture_after_review_effective_time": True,
            "every_bundle_complete_and_outcome_free": True,
            "duplicate_or_ambiguous_side_bindings_absent": True,
            "retrospective_backfill_present": False,
            "eligible_map_count": len(entries),
            "outcomes_present": False,
            "outcomes_accessed": False,
            "independently_pinned_ledger_snapshot": False,
            "opening_authority": False,
        },
        "implementation": _source_record(root),
        "authority": _authority_false(),
        "claim_ceiling": CLAIM_CEILING,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return validate_side_neutral_ledger(
        payload, root=root, environment=environment, as_of=created
    )


def validate_side_neutral_ledger(
    payload: Mapping[str, Any],
    *,
    root: Path = ROOT,
    environment: Mapping[str, str],
    as_of: datetime,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise SideNeutralLedgerError("ledger must be an object")
    value = dict(payload)
    _assert_no_outcomes(value, "side_neutral_ledger")
    expected = {
        "schema_version",
        "result_state",
        "created_at_utc",
        "clock_attestation",
        "independent_review",
        "entries",
        "support",
        "qualification",
        "implementation",
        "authority",
        "claim_ceiling",
        "artifact_sha256",
    }
    if set(value) != expected:
        raise SideNeutralLedgerError("ledger structure changed")
    if value.get("schema_version") != SCHEMA_VERSION or value.get(
        "result_state"
    ) != RESULT_STATE:
        raise SideNeutralLedgerError("ledger identity changed")
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if value.get("artifact_sha256") != _canonical_sha256(unsigned):
        raise SideNeutralLedgerError("ledger hash changed")
    created = _timestamp(value.get("created_at_utc"), "created_at_utc")
    if created > as_of.astimezone(timezone.utc):
        raise SideNeutralLedgerError("ledger timestamp is in the future")
    if value.get("clock_attestation") != {
        "source": "system_utc_clock_sampled_inside_builder",
        "observed_wall_clock_utc": created.isoformat(),
        "user_supplied_timestamp_allowed": False,
    }:
        raise SideNeutralLedgerError("ledger clock changed")
    review = load_active_side_neutral_protocol_review(
        root=root, environment=environment, as_of=as_of
    )
    review_effective = _timestamp(
        review["authorization"]["effective_at_utc"], "review effective time"
    )
    if created <= review_effective:
        raise SideNeutralLedgerError("ledger predates independent review")
    if value.get("independent_review") != {
        "locator": REVIEW_LOCATOR.as_posix(),
        "external_raw_sha256": _sha(
            environment.get(INDEPENDENT_REVIEW_ENV), INDEPENDENT_REVIEW_ENV
        ),
        "review_id": review["review_id"],
        "reviewed_at_utc": review["reviewed_at_utc"],
        "effective_at_utc": review["authorization"]["effective_at_utc"],
        "prospective_collection_authorized": True,
        "outcome_opening_authorized": False,
    }:
        raise SideNeutralLedgerError("independent review binding changed")
    entries = value.get("entries")
    if not isinstance(entries, list):
        raise SideNeutralLedgerError("ledger entries must be a list")
    rebuilt: list[dict[str, Any]] = []
    seen_events: set[tuple[str, int]] = set()
    seen_side_records: set[str] = set()
    seen_locators: set[str] = set()
    for item in entries:
        if not isinstance(item, Mapping):
            raise SideNeutralLedgerError("ledger entry is malformed")
        locator = _bundle_locator(item.get("bundle_locator"))
        if locator in seen_locators:
            raise SideNeutralLedgerError("duplicate bundle locator")
        seen_locators.add(locator)
        raw, bundle = _load_bundle(root, locator)
        expected_entry = _entry(
            locator=locator,
            raw=raw,
            bundle=bundle,
            review_effective=review_effective,
        )
        if dict(item) != expected_entry:
            raise SideNeutralLedgerError("ledger entry binding changed")
        event_key = (expected_entry["event_id"], expected_entry["game_number"])
        if event_key in seen_events:
            raise SideNeutralLedgerError(
                "duplicate or ambiguous side binding for one map"
            )
        seen_events.add(event_key)
        side_record = expected_entry["side_source_record_id"]
        if side_record in seen_side_records:
            raise SideNeutralLedgerError("side source record was reused across maps")
        seen_side_records.add(side_record)
        rebuilt.append(expected_entry)
    expected_order = sorted(
        rebuilt,
        key=lambda item: (
            item["actual_map_start_utc"],
            item["event_id"],
            item["game_number"],
        ),
    )
    if entries != expected_order:
        raise SideNeutralLedgerError("ledger entry order changed")
    if value.get("support") != _support(expected_order):
        raise SideNeutralLedgerError("ledger support changed")
    expected_qualification = {
        "independent_review_present_and_valid": True,
        "every_pre_side_capture_after_review_effective_time": True,
        "every_bundle_complete_and_outcome_free": True,
        "duplicate_or_ambiguous_side_bindings_absent": True,
        "retrospective_backfill_present": False,
        "eligible_map_count": len(entries),
        "outcomes_present": False,
        "outcomes_accessed": False,
        "independently_pinned_ledger_snapshot": False,
        "opening_authority": False,
    }
    if value.get("qualification") != expected_qualification:
        raise SideNeutralLedgerError("ledger qualification changed")
    if value.get("implementation") != _source_record(root):
        raise SideNeutralLedgerError("ledger implementation changed")
    if value.get("authority") != _authority_false() or value.get(
        "claim_ceiling"
    ) != CLAIM_CEILING:
        raise SideNeutralLedgerError("ledger authority boundary changed")
    return value


def write_no_clobber(path: Path, payload: Mapping[str, Any]) -> str:
    raw = (
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise SideNeutralLedgerError(f"refusing to overwrite ledger: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, path)
        except FileExistsError as exc:
            raise SideNeutralLedgerError(
                f"refusing to overwrite ledger: {path}"
            ) from exc
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return _sha256_bytes(raw)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--bundle", action="append", default=[])
    parser.add_argument("--out", type=Path, default=DEFAULT_LEDGER)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        payload = build_side_neutral_ledger(
            bundle_locators=args.bundle,
            environment=os.environ,
            root=root,
        )
        if args.out.resolve(strict=False) != (root / DEFAULT_LEDGER).resolve(
            strict=False
        ):
            raise SideNeutralLedgerError(
                f"output must be the immutable ledger locator: {DEFAULT_LEDGER}"
            )
        raw_sha256 = write_no_clobber(args.out, payload)
    except (OSError, ValueError, RuntimeError) as exc:
        print(str(exc))
        return 2
    print(
        json.dumps(
            {
                "output": str(args.out),
                "raw_sha256": raw_sha256,
                "artifact_sha256": payload["artifact_sha256"],
                "eligible_map_count": payload["qualification"][
                    "eligible_map_count"
                ],
                "outcomes_accessed": False,
                "opening_authority": False,
                "betting_authority": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_LEDGER",
    "RESULT_STATE",
    "SCHEMA_VERSION",
    "SideNeutralLedgerError",
    "build_side_neutral_ledger",
    "validate_side_neutral_ledger",
    "write_no_clobber",
]
