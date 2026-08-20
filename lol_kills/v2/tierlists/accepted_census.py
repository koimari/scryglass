"""Read and write the exact game census accepted for one public release."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from lol_kills.etl.source_keys import canonical_source_game_key


SCHEMA_VERSION = "scryglass:accepted-game-census:v1"


class AcceptedCensusError(ValueError):
    """Raised when an accepted release census is missing or inconsistent."""


def canonical_game_ids(values: Iterable[object]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                game_id
                for value in values
                if (game_id := canonical_source_game_key(value))
            }
        )
    )


def identity_sha256(values: Iterable[object]) -> str:
    game_ids = canonical_game_ids(values)
    return hashlib.sha256(("\n".join(game_ids) + "\n").encode("utf-8")).hexdigest()


def census_payload(values: Iterable[object]) -> dict[str, Any]:
    game_ids = canonical_game_ids(values)
    if not game_ids:
        raise AcceptedCensusError("accepted release census is empty")
    return {
        "schema_version": SCHEMA_VERSION,
        "game_count": len(game_ids),
        "source_identity_sha256": identity_sha256(game_ids),
        "game_ids": list(game_ids),
    }


def write_census(path: Path, values: Iterable[object]) -> dict[str, Any]:
    payload = census_payload(values)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def load_census(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise AcceptedCensusError("accepted release census is missing or is not a regular file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AcceptedCensusError("accepted release census cannot be read") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise AcceptedCensusError("accepted release census schema is invalid")
    raw_ids = payload.get("game_ids")
    if not isinstance(raw_ids, list) or not all(isinstance(value, str) for value in raw_ids):
        raise AcceptedCensusError("accepted release census game IDs are invalid")
    game_ids = canonical_game_ids(raw_ids)
    if list(game_ids) != raw_ids:
        raise AcceptedCensusError("accepted release census game IDs are not canonical and unique")
    expected = census_payload(game_ids)
    if payload != expected:
        raise AcceptedCensusError("accepted release census binding is invalid")
    return expected
