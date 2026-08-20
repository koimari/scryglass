"""Verify the frozen future-value source on an isolated research runner.

This entry point prepares source receipts only. It does not fit, promote, or
publish a player or team model. A later training stage must consume the
verified source receipt and satisfy the frozen evaluation protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from lol_kills.etl.source_keys import canonical_source_game_key
from lol_kills.research.future_value_rating import (
    FutureValueSourceError,
    bind_accepted_future_value_source,
    write_source_receipt,
)
from lol_kills.v2.tierlists.accepted_census import (
    canonical_game_ids,
    census_payload,
)


SCHEMA_VERSION = "scryglass:future-value-research-run:v1"
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annual-root", type=Path, required=True)
    parser.add_argument("--oe-root", type=Path)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--annual-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.annual_only:
            result = verify_annual_only(
                annual_root=args.annual_root,
                freeze_path=args.freeze,
                output_root=args.output_root,
            )
        else:
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
]
