#!/usr/bin/env python3
"""Private, deterministic GRID review for objective-crossmap sequences.

The command turns one verified Riot LiveStats file into a compact evidence
packet for questions such as:

* which objectives and plates were taken, and at what exact game times;
* which player resources changed inside declared windows;
* how much Touch-of-the-Void-compatible true damage appeared in a turret siege;
* how much siege time that damage conditionally saved at the observed
  non-Touch champion damage rate; and
* how an explicit wave counterfactual changes the resource ledger.

It is private personal-research tooling. Raw GRID data and signed download URLs
never appear in the report. Counterfactual output is labelled as an assumption,
not as an observed fact or a causal estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from lol_kills.etl.grid_ingest import (
    _api_key,
    _download,
    _file_list,
    _series_games,
)
from lol_kills.etl.grid_series_events import assert_pro_series
from lol_kills.etl.paths import WAREHOUSE_DIR


SCHEMA_VERSION = "scryglass:grid-sequence-review:v2"
RETRIEVAL_SCHEMA_VERSION = "scryglass:grid-sequence-retrieval:v1"
REQUEST_SCHEMA_VERSION = "scryglass:grid-sequence-request:v1"
MECHANICS_PROFILE_ID = "summoners-rift-26.15-v1"
DEFAULT_ROOT = WAREHOUSE_DIR / "private_grid" / "sequence_review" / "v1"
DEFAULT_CATALOG = (
    Path.home()
    / ".codex"
    / "skills"
    / "query-grid-research"
    / "assets"
    / "grid-capability-catalog.v1.json"
)

MAX_FRAME_SKEW_MS = 1_500
XP_RADIUS = 2_000.0
TOUCH_BASE_QUANTUM = 8.0  # three-stack ranged tick on patch 26.15
TOUCH_TOLERANCE = 1e-4

MELEE_GOLD = 20.0
MELEE_XP = 62.0
CASTER_GOLD = 14.0
CASTER_XP = 31.0
CANNON_XP = 75.0
DUO_XP_SHARE = 0.65
REGULAR_WAVE_GOLD = 3 * MELEE_GOLD + 3 * CASTER_GOLD
REGULAR_WAVE_XP = 3 * MELEE_XP + 3 * CASTER_XP
CANNON_WAVE_XP = REGULAR_WAVE_XP + CANNON_XP

OUTER_TURRET_HP = 9_000.0
OUTER_PLATE_REMAINING_HP_THRESHOLDS = (8_100.0, 6_750.0, 4_950.0, 2_700.0, 0.0)
CAMP_EVENT_PADDING_MS = 3_000
CAMP_LOOKAHEAD_MS = 5 * 60_000

MECHANICS_SOURCES = [
    {
        "title": "Melee minion",
        "page_id": 3312,
        "revision_id": 4013291,
        "revision_timestamp": "2026-04-28T22:24:48Z",
        "source_url": "https://wiki.leagueoflegends.com/en-us/Melee_minion",
        "supports": {"gold": MELEE_GOLD, "xp": MELEE_XP},
    },
    {
        "title": "Caster minion",
        "page_id": 10965,
        "revision_id": 4015019,
        "revision_timestamp": "2026-05-02T12:47:03Z",
        "source_url": "https://wiki.leagueoflegends.com/en-us/Caster_minion",
        "supports": {"gold": CASTER_GOLD, "xp": CASTER_XP},
    },
    {
        "title": "Siege minion",
        "page_id": 3872,
        "revision_id": 4013294,
        "revision_timestamp": "2026-04-28T22:27:15Z",
        "source_url": "https://wiki.leagueoflegends.com/en-us/Siege_minion",
        "supports": {"gold": "50 + 1 every upgrade", "xp": CANNON_XP},
    },
    {
        "title": "Experience (champion)",
        "page_id": 3277,
        "revision_id": 4037005,
        "revision_timestamp": "2026-06-27T16:47:11Z",
        "source_url": "https://wiki.leagueoflegends.com/en-us/Experience_(champion)",
        "supports": {"two_champions_each": DUO_XP_SHARE},
    },
    {
        "title": "Touch of the Void",
        "page_id": 1620205,
        "revision_id": 4036112,
        "revision_timestamp": "2026-06-26T22:37:18Z",
        "source_url": "https://wiki.leagueoflegends.com/en-us/Touch_of_the_Void",
        "supports": {
            "three_stack_melee_tick": 16,
            "three_stack_ranged_tick": 8,
            "tick_interval_seconds": 0.5,
        },
    },
    {
        "title": "Voidgrub camp",
        "page_id": 1620201,
        "revision_id": 4015021,
        "revision_timestamp": "2026-05-02T12:51:17Z",
        "source_url": "https://wiki.leagueoflegends.com/en-us/Voidgrub_camp",
        "supports": {"gold_each": 30, "xp_each": 65, "xp_radius": XP_RADIUS},
    },
    {
        "title": "Turret",
        "page_id": 2467,
        "revision_id": 4019795,
        "revision_timestamp": "2026-05-19T14:01:14Z",
        "source_url": "https://wiki.leagueoflegends.com/en-us/Turret",
        "supports": {
            "outer_health": OUTER_TURRET_HP,
            "outer_plate_remaining_hp_thresholds": list(
                OUTER_PLATE_REMAINING_HP_THRESHOLDS
            ),
        },
    },
]

GOLD_STAT_FIELDS = (
    "ambient",
    "killMinion",
    "supportItemMinion",
    "killNeutralMinion",
    "killPalisade",
    "assist",
    "killChampion",
)
PLAYER_STAT_FIELDS = (
    "MINIONS_KILLED",
    "NEUTRAL_MINIONS_KILLED",
    "TRUE_DAMAGE_DEALT_PLAYER",
    "TOTAL_DAMAGE_DEALT_TO_BUILDINGS",
    "TOTAL_DAMAGE_DEALT_TO_TURRETS",
)
EVENT_SCHEMAS = {
    "epic_monster_kill",
    "turret_plate_destroyed",
    "turret_plate_gold_earned",
    "building_gold_grant",
    "building_destroyed",
}


class GridSequenceReviewError(RuntimeError):
    """Credential-free failure in sequence retrieval or analysis."""


@dataclass(frozen=True)
class Frame:
    time_ms: int
    sequence_index: int
    players: dict[int, dict[str, Any]]


@dataclass(frozen=True)
class GameData:
    identity: dict[str, Any]
    roster: tuple[dict[str, Any], ...]
    frames: tuple[Frame, ...]
    events: tuple[dict[str, Any], ...]
    completeness: dict[str, Any]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_file_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in sorted(row.items())
        if not any(
            marker in str(key).lower()
            for marker in ("url", "token", "key", "secret", "signature")
        )
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_clock(value: str | int | float) -> int:
    """Parse milliseconds or an ``MM:SS.mmm`` game clock."""
    if isinstance(value, (int, float)):
        if value < 0:
            raise GridSequenceReviewError("game clock cannot be negative")
        return int(round(float(value)))
    text = str(value).strip()
    if not text:
        raise GridSequenceReviewError("game clock is empty")
    if ":" not in text:
        try:
            milliseconds = float(text)
        except ValueError as exc:
            raise GridSequenceReviewError(f"invalid game clock {value!r}") from exc
        if milliseconds < 0:
            raise GridSequenceReviewError("game clock cannot be negative")
        return int(round(milliseconds))
    minutes_text, seconds_text = text.split(":", 1)
    try:
        minutes = int(minutes_text)
        seconds = float(seconds_text)
    except ValueError as exc:
        raise GridSequenceReviewError(f"invalid game clock {value!r}") from exc
    if minutes < 0 or not 0 <= seconds < 60:
        raise GridSequenceReviewError(f"invalid game clock {value!r}")
    return int(round((minutes * 60 + seconds) * 1000))


def format_clock(milliseconds: int) -> str:
    minutes, remainder = divmod(int(milliseconds), 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{minutes}:{seconds:02d}.{millis:03d}"


def catalog_provenance(path: Path = DEFAULT_CATALOG) -> dict[str, Any]:
    if not path.is_file():
        raise GridSequenceReviewError(f"GRID capability catalog is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = str(payload.get("catalog_sha256") or "")
    actual = _hash({key: value for key, value in payload.items() if key != "catalog_sha256"})
    if not expected or expected != actual:
        raise GridSequenceReviewError("GRID capability catalog hash does not verify")
    capabilities = {
        str(row.get("capability")): str(row.get("status"))
        for row in payload.get("capabilities") or []
        if isinstance(row, Mapping)
    }
    if capabilities.get("historical_file_listing") != "confirmed":
        raise GridSequenceReviewError("GRID historical file listing is unavailable")
    endpoints = {
        str(row.get("endpoint_id")): str(row.get("schema_sha256"))
        for row in payload.get("endpoints") or []
        if isinstance(row, Mapping)
    }
    return {
        "catalog_version": payload.get("catalog_version"),
        "catalog_sha256": expected,
        "generated_at": payload.get("generated_at"),
        "endpoint_schema_sha256": endpoints,
        "historical_file_download_catalog_status": capabilities.get(
            "historical_file_download"
        ),
        "download_authority": (
            "explicit --download flag for one verified series/game; catalog discovery "
            "did not itself test payload download"
        ),
    }


def load_review_request(path: Path, *, root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    """Load a closed, cached-source review request and validate its identity."""

    if not path.is_file():
        raise GridSequenceReviewError(f"review request is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GridSequenceReviewError("review request is not valid JSON") from exc
    if not isinstance(payload, Mapping) or payload.get("schema_version") != REQUEST_SCHEMA_VERSION:
        raise GridSequenceReviewError("review request schema is unsupported")
    identity = payload.get("identity")
    source = payload.get("source")
    windows = payload.get("windows")
    if not all(isinstance(value, Mapping) for value in (identity, source, windows)):
        raise GridSequenceReviewError("review request identity/source/windows are required")
    series_id = str(identity.get("series_id") or "")
    game_index = identity.get("game_index")
    provider_game_id = str(identity.get("provider_game_id") or "")
    raw_sha256 = str(source.get("raw_sha256") or "")
    if (
        not series_id.isdigit()
        or type(game_index) is not int
        or game_index < 1
        or not provider_game_id
        or len(raw_sha256) != 64
        or any(character not in "0123456789abcdef" for character in raw_sha256)
    ):
        raise GridSequenceReviewError("review request identity or source hash is invalid")
    if source.get("mode") != "cached_content_hash":
        raise GridSequenceReviewError(
            "review requests use cached_content_hash; first retrieval still requires explicit --download"
        )
    raw_path = root / "raw" / f"events_{series_id}_{game_index}_riot_{raw_sha256}.jsonl"
    if not raw_path.is_file() or _sha256_file(raw_path) != raw_sha256:
        raise GridSequenceReviewError("cached request source is missing or has the wrong hash")
    required_windows = ("sequence", "resource_wave", "turret_siege")
    for name in required_windows:
        row = windows.get(name)
        if not isinstance(row, Mapping) or "start" not in row or "end" not in row:
            raise GridSequenceReviewError(f"review request window {name!r} is incomplete")
        if parse_clock(row["end"]) <= parse_clock(row["start"]):
            raise GridSequenceReviewError(f"review request window {name!r} is reversed")
    request = dict(payload)
    request["resolved_raw_path"] = str(raw_path)
    request["request_path"] = str(path)
    request["request_sha256"] = _sha256_file(path)
    return request


def verify_request_acceptance(
    report: Mapping[str, Any], expected: Mapping[str, Any] | None
) -> dict[str, Any]:
    if not expected:
        return {"status": "not_declared", "checks": []}

    def resolve(path: str) -> Any:
        value: Any = report
        for component in path.split("."):
            if isinstance(value, Mapping) and component in value:
                value = value[component]
            elif isinstance(value, Mapping) and component.isdigit() and int(component) in value:
                value = value[int(component)]
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and component.isdigit():
                value = value[int(component)]
            else:
                raise GridSequenceReviewError(
                    f"acceptance path {path!r} does not resolve"
                )
        return value

    checks: list[dict[str, Any]] = []
    for path, expected_value in expected.items():
        actual = resolve(str(path))
        if isinstance(expected_value, (int, float)) and not isinstance(expected_value, bool):
            passed = isinstance(actual, (int, float)) and math.isclose(
                float(actual), float(expected_value), rel_tol=0.0, abs_tol=1e-6
            )
        else:
            passed = actual == expected_value
        checks.append(
            {
                "path": str(path),
                "expected": expected_value,
                "actual": actual,
                "passed": passed,
            }
        )
    failed = [row["path"] for row in checks if not row["passed"]]
    if failed:
        raise GridSequenceReviewError(
            "request acceptance checks failed: " + ", ".join(failed)
        )
    return {"status": "passed", "checks": checks}


def build_review_request_from_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Create a replayable cached-source request and its observed acceptance set."""

    identity = report["identity"]
    windows = report["windows"]
    observed = report["observed"]
    resource_views = observed["resource_views"]
    selected = resource_views["selected"]
    full = resource_views["full_teams"]
    expected: dict[str, Any] = {
        "identity.game_version": identity["game_version"],
        "observed.siege.plate_gold": observed["siege"]["plate_gold"],
        "observed.siege.touch_compatible_true_damage": observed["siege"][
            "touch_compatible_true_damage"
        ],
        "observed.siege.champion_building_damage": observed["siege"][
            "champion_building_damage"
        ],
    }
    action_graph = report.get("action_graph") or {}
    if action_graph.get("graph_sha256"):
        expected["action_graph.graph_sha256"] = action_graph["graph_sha256"]
    for objective in ("dragon", "grubs"):
        for index, event in enumerate(observed[objective]["events"]):
            expected[f"observed.{objective}.events.{index}.game_time"] = event[
                "game_time"
            ]
    for index, event in enumerate(observed["siege"]["plates"]):
        expected[f"observed.siege.plates.{index}.game_time"] = event["game_time"]
    metrics = ("cs", "gold_excluding_plates", "plate_gold", "total_gold", "xp")
    for view_name, view in (("selected", selected), ("full_teams", full)):
        for team_id, totals in view["team_totals"].items():
            for metric in metrics:
                expected[
                    f"observed.resource_views.{view_name}.team_totals.{team_id}.{metric}"
                ] = totals[metric]
        for metric in metrics:
            expected[
                f"observed.resource_views.{view_name}.comparison.reference_minus_opponent.{metric}"
            ] = view["comparison"]["reference_minus_opponent"][metric]
    camps = observed.get("delayed_camps") or {}
    if camps.get("status") == "verified_later_same_game_clears":
        for ledger_name in ("later_camp_resources", "camps_minus_grubs"):
            for metric in ("gold", "xp"):
                expected[f"observed.delayed_camps.{ledger_name}.{metric}"] = camps[
                    ledger_name
                ][metric]
    named = report["counterfactual"].get(
        "named_farm_alternative_including_delayed_camps"
    )
    if isinstance(named, Mapping):
        for ledger_name in (
            "received_less_than_named_farm",
            "after_actual_plates_and_defender_denial",
        ):
            for metric in ("gold", "xp"):
                expected[
                    f"counterfactual.named_farm_alternative_including_delayed_camps.{ledger_name}.{metric}"
                ] = named[ledger_name][metric]
    turret = observed.get("turret_health") or {}
    turret_observation = None
    if turret.get("status") == "observer_estimate_available":
        source_observation = turret["observation"]
        turret_observation = {
            key: source_observation.get(key)
            for key in (
                "game_time",
                "health_estimate",
                "health_low",
                "health_high",
                "method",
                "source",
            )
        }
        fixed = turret["fixed_state_remove_touch_only"]
        expected[
            "observed.turret_health.fixed_state_remove_touch_only.health_estimate"
        ] = fixed["health_estimate"]
        expected[
            "observed.turret_health.fixed_state_remove_touch_only.plate_thresholds_reached"
        ] = fixed["plate_thresholds_reached"]
    team_labels = {
        str(team_id): values["label"]
        for team_id, values in resource_views["team_labels"].items()
    }
    request: dict[str, Any] = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "identity": {
            "series_id": str(identity["provider_series_id"]),
            "game_index": int(identity["game_index"]),
            "provider_game_id": str(identity["provider_game_id"]),
        },
        "source": {
            "mode": "cached_content_hash",
            "raw_sha256": report["source"]["raw_sha256"],
        },
        "windows": {
            "sequence": {
                "start": windows["sequence"]["requested_start"],
                "end": windows["sequence"]["requested_end"],
            },
            "resource_wave": {
                "start": windows["resource"]["requested_start"],
                "end": windows["resource"]["requested_end"],
            },
            "turret_siege": {
                "start": windows["siege"]["requested_start"],
                "end": windows["siege"]["requested_end"],
            },
        },
        "lane": observed["siege"]["lane"],
        "involved_champions": [row["champion"] for row in selected["rows"]],
        "team_labels": team_labels,
        "delayed_camps": list(camps.get("requested_camps") or []),
        "expected_observed": expected,
    }
    if turret_observation is not None:
        request["turret_observation"] = turret_observation
    cannon = report["counterfactual"].get("cannon_gold") or {}
    if cannon.get("status") == "explicit_override":
        request["cannon_gold_override"] = cannon["value"]
    return request


def retrieve_riot_game(
    *,
    series_id: str,
    game_index: int,
    root: Path = DEFAULT_ROOT,
    key: str | None = None,
    catalog_path: Path = DEFAULT_CATALOG,
) -> dict[str, Any]:
    """Retrieve one exact Riot event file to a private content-addressed path."""
    sid = str(series_id).strip()
    if not sid.isdigit():
        raise GridSequenceReviewError("series ID must be a verified numeric GRID ID")
    if game_index < 1:
        raise GridSequenceReviewError("game index must be positive")
    catalog = catalog_provenance(catalog_path)
    secret = key or _api_key()
    series = assert_pro_series(sid, secret)
    games = _series_games(secret, sid)
    if game_index > len(games):
        raise GridSequenceReviewError(
            f"series {sid} has {len(games)} games, not game {game_index}"
        )
    game = games[game_index - 1]
    file_id = f"events-riot-game-{game_index}"
    matches = [row for row in _file_list(secret, sid) if str(row.get("id")) == file_id]
    if len(matches) != 1:
        raise GridSequenceReviewError(f"GRID has no unique ready file {file_id}")
    file_row = matches[0]
    if str(file_row.get("status") or "") != "ready":
        raise GridSequenceReviewError(f"GRID file {file_id} is not ready")
    signed_url = str(file_row.get("fullURL") or "")
    if not signed_url:
        raise GridSequenceReviewError(f"GRID file {file_id} has no download capability")

    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".events_{sid}_{game_index}_riot.", suffix=".jsonl", dir=raw_dir
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        if not _download(signed_url, secret, temporary):
            raise GridSequenceReviewError("GRID event download was rate-limited")
        source_sha = _sha256_file(temporary)
        destination = (
            raw_dir / f"events_{sid}_{game_index}_riot_{source_sha}.jsonl"
        )
        if destination.exists():
            if _sha256_file(destination) != source_sha:
                raise GridSequenceReviewError("content-addressed GRID path conflict")
            temporary.unlink()
        else:
            os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    receipt = {
        "schema_version": RETRIEVAL_SCHEMA_VERSION,
        "retrieved_at": _utc_now(),
        "scope": "private_personal_research_only",
        "provider_series_id": sid,
        "provider_game_id": str(game.get("id") or ""),
        "game_index": game_index,
        "series": series,
        "file_id": file_id,
        "file_metadata": _safe_file_metadata(file_row),
        "raw_path": str(destination),
        "raw_sha256": source_sha,
        "raw_bytes": destination.stat().st_size,
        "catalog": catalog,
        "credentials_serialized": False,
        "signed_url_retained": False,
        "mutations_used": False,
    }
    receipt["receipt_sha256"] = _hash(receipt)
    receipt_path = (
        root
        / "receipts"
        / f"retrieval_{sid}_{game_index}_{receipt['receipt_sha256']}.json"
    )
    _write_json(receipt_path, receipt)
    receipt["receipt_path"] = str(receipt_path)
    return receipt


def adopt_local_source(
    *,
    source: Path,
    series_id: str,
    game_index: int,
    provider_game_id: str,
    root: Path = DEFAULT_ROOT,
    catalog_path: Path = DEFAULT_CATALOG,
) -> dict[str, Any]:
    """Content-address an already verified local source without network I/O."""
    if not source.is_file():
        raise GridSequenceReviewError(f"local Riot event file is missing: {source}")
    catalog = catalog_provenance(catalog_path)
    source_sha = _sha256_file(source)
    destination = (
        root
        / "raw"
        / f"events_{series_id}_{game_index}_riot_{source_sha}.jsonl"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _sha256_file(destination) != source_sha:
            raise GridSequenceReviewError("content-addressed local path conflict")
    else:
        shutil.copy2(source, destination)
    receipt = {
        "schema_version": RETRIEVAL_SCHEMA_VERSION,
        "retrieved_at": _utc_now(),
        "scope": "private_personal_research_only",
        "provider_series_id": str(series_id),
        "provider_game_id": str(provider_game_id),
        "game_index": int(game_index),
        "file_id": f"events-riot-game-{game_index}",
        "source_mode": "already_local_explicit_source",
        "raw_path": str(destination),
        "raw_sha256": source_sha,
        "raw_bytes": destination.stat().st_size,
        "catalog": catalog,
        "credentials_serialized": False,
        "signed_url_retained": False,
        "mutations_used": False,
    }
    receipt["receipt_sha256"] = _hash(receipt)
    receipt_path = (
        root
        / "receipts"
        / f"retrieval_{series_id}_{game_index}_{receipt['receipt_sha256']}.json"
    )
    _write_json(receipt_path, receipt)
    receipt["receipt_path"] = str(receipt_path)
    return receipt


def _number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _safe_player_state(row: Mapping[str, Any]) -> dict[str, Any]:
    stats = {
        str(item.get("name")): _number(item.get("value"))
        for item in row.get("stats") or []
        if isinstance(item, Mapping) and str(item.get("name")) in PLAYER_STAT_FIELDS
    }
    gold_stats = row.get("goldStats") or {}
    if not isinstance(gold_stats, Mapping):
        gold_stats = {}
    position = row.get("position") or {}
    if not isinstance(position, Mapping):
        position = {}
    return {
        "participant_id": int(row.get("participantID") or 0),
        "team_id": int(row.get("teamID") or 0),
        "champion": str(row.get("championName") or ""),
        "role": str(row.get("role") or ""),
        "xp": _number(row.get("XP")),
        "level": int(row.get("level") or 0),
        "total_gold": _number(row.get("totalGold")),
        "current_gold": _number(row.get("currentGold")),
        "x": _number(position.get("x")),
        "z": _number(position.get("z")),
        "gold": {field: _number(gold_stats.get(field)) for field in GOLD_STAT_FIELDS},
        "stats": {field: _number(stats.get(field)) for field in PLAYER_STAT_FIELDS},
    }


def load_game(path: Path) -> GameData:
    """Stream one completed Riot LiveStats JSONL into a safe compact state."""
    if not path.is_file():
        raise GridSequenceReviewError(f"Riot event file is missing: {path}")
    roster: tuple[dict[str, Any], ...] | None = None
    identity: dict[str, Any] | None = None
    frames: list[Frame] = []
    events: list[dict[str, Any]] = []
    game_end_count = 0
    schema_counts: Counter[str] = Counter()
    roots: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise GridSequenceReviewError(
                    f"invalid JSONL at line {line_number}"
                ) from exc
            if not isinstance(row, Mapping):
                raise GridSequenceReviewError(
                    f"non-object JSONL row at line {line_number}"
                )
            schema = str(row.get("rfc461Schema") or "")
            schema_counts[schema] += 1
            root = str(row.get("rootGameID") or row.get("gameID") or "")
            if root:
                roots.add(root)
            if schema == "game_info":
                if roster is not None:
                    raise GridSequenceReviewError("Riot file has multiple game_info rows")
                participants = row.get("participants") or []
                safe_roster = []
                for participant in participants:
                    if not isinstance(participant, Mapping):
                        continue
                    riot_id = participant.get("riotId") or {}
                    display_name = (
                        riot_id.get("displayName")
                        if isinstance(riot_id, Mapping)
                        else None
                    )
                    safe_roster.append(
                        {
                            "participant_id": int(participant.get("participantID") or 0),
                            "team_id": int(participant.get("teamID") or 0),
                            "role": str(participant.get("role") or ""),
                            "champion": str(participant.get("championName") or ""),
                            "player": str(
                                display_name
                                or participant.get("summonerName")
                                or ""
                            ),
                        }
                    )
                roster = tuple(sorted(safe_roster, key=lambda item: item["participant_id"]))
                identity = {
                    "riot_game_id": str(row.get("gameID") or ""),
                    "riot_root_game_id": str(row.get("rootGameID") or ""),
                    "platform_id": str(row.get("platformID") or ""),
                    "game_name": str(row.get("gameName") or ""),
                    "game_version": str(row.get("gameVersion") or ""),
                    "stats_update_interval_ms": int(row.get("statsUpdateInterval") or 0),
                }
            elif schema == "stats_update":
                game_time = row.get("gameTime")
                if not isinstance(game_time, (int, float)):
                    continue
                players = {
                    int(participant.get("participantID") or 0): _safe_player_state(
                        participant
                    )
                    for participant in row.get("participants") or []
                    if isinstance(participant, Mapping)
                }
                frames.append(
                    Frame(
                        time_ms=int(game_time),
                        sequence_index=int(row.get("sequenceIndex") or 0),
                        players=players,
                    )
                )
            elif schema == "game_end":
                game_end_count += 1
            elif schema in EVENT_SCHEMAS:
                event = {
                    "schema": schema,
                    "game_time_ms": int(row.get("gameTime") or 0),
                    "sequence_index": int(row.get("sequenceIndex") or 0),
                }
                for key in (
                    "monsterType",
                    "dragonType",
                    "killer",
                    "killerTeamID",
                    "assistants",
                    "killerGold",
                    "localGold",
                    "globalGold",
                    "lane",
                    "lastHitter",
                    "teamID",
                    "participantID",
                    "bounty",
                    "amount",
                    "source",
                    "buildingType",
                    "turretTier",
                    "position",
                ):
                    if key in row:
                        event[key] = row[key]
                events.append(event)

    if roster is None or identity is None:
        raise GridSequenceReviewError("Riot file has no unique game_info row")
    if len(roster) != 10 or len({row["participant_id"] for row in roster}) != 10:
        raise GridSequenceReviewError("Riot roster is not exactly ten unique players")
    if game_end_count != 1:
        raise GridSequenceReviewError("Riot file is not a uniquely completed game")
    if len(roots) != 1:
        raise GridSequenceReviewError("Riot file contains ambiguous game roots")
    frames.sort(key=lambda frame: (frame.time_ms, frame.sequence_index))
    if not frames:
        raise GridSequenceReviewError("Riot file has no stats_update frames")
    duplicate_times = len(frames) - len({frame.time_ms for frame in frames})
    gaps = [
        right.time_ms - left.time_ms
        for left, right in zip(frames, frames[1:])
        if right.time_ms >= left.time_ms
    ]
    completeness = {
        "status": "verified" if duplicate_times == 0 else "review",
        "stats_frames": len(frames),
        "first_stats_ms": frames[0].time_ms,
        "last_stats_ms": frames[-1].time_ms,
        "maximum_stats_gap_ms": max(gaps, default=0),
        "duplicate_stats_times": duplicate_times,
        "game_info_rows": 1,
        "game_end_rows": game_end_count,
        "schema_counts": dict(schema_counts.most_common()),
    }
    return GameData(
        identity=identity,
        roster=roster,
        frames=tuple(frames),
        events=tuple(sorted(events, key=lambda row: (row["game_time_ms"], row["sequence_index"]))),
        completeness=completeness,
    )


def _frame_after(frames: Sequence[Frame], target_ms: int) -> Frame:
    match = next((frame for frame in frames if frame.time_ms >= target_ms), None)
    if match is None or match.time_ms - target_ms > MAX_FRAME_SKEW_MS:
        raise GridSequenceReviewError(
            f"no stats frame within {MAX_FRAME_SKEW_MS}ms after {format_clock(target_ms)}"
        )
    return match


def _frame_before(frames: Sequence[Frame], target_ms: int) -> Frame:
    match = next((frame for frame in reversed(frames) if frame.time_ms <= target_ms), None)
    if match is None or target_ms - match.time_ms > MAX_FRAME_SKEW_MS:
        raise GridSequenceReviewError(
            f"no stats frame within {MAX_FRAME_SKEW_MS}ms before {format_clock(target_ms)}"
        )
    return match


def _player_delta(start: Frame, end: Frame, participant_id: int) -> dict[str, float]:
    left = start.players.get(participant_id)
    right = end.players.get(participant_id)
    if left is None or right is None:
        raise GridSequenceReviewError(
            f"participant {participant_id} is absent from a requested frame"
        )
    return {
        "xp": right["xp"] - left["xp"],
        "cs": right["stats"]["MINIONS_KILLED"]
        - left["stats"]["MINIONS_KILLED"],
        "neutral_cs": right["stats"]["NEUTRAL_MINIONS_KILLED"]
        - left["stats"]["NEUTRAL_MINIONS_KILLED"],
        "minion_gold": right["gold"]["killMinion"]
        - left["gold"]["killMinion"],
        "support_minion_gold": right["gold"]["supportItemMinion"]
        - left["gold"]["supportItemMinion"],
        "neutral_gold": right["gold"]["killNeutralMinion"]
        - left["gold"]["killNeutralMinion"],
        "champion_kill_gold": right["gold"]["killChampion"]
        - left["gold"]["killChampion"],
        "assist_gold": right["gold"]["assist"] - left["gold"]["assist"],
        "building_gold": right["gold"]["killPalisade"]
        - left["gold"]["killPalisade"],
        "ambient_gold": right["gold"]["ambient"] - left["gold"]["ambient"],
        "total_gold": right["total_gold"] - left["total_gold"],
        "building_damage": right["stats"]["TOTAL_DAMAGE_DEALT_TO_BUILDINGS"]
        - left["stats"]["TOTAL_DAMAGE_DEALT_TO_BUILDINGS"],
        "turret_damage": right["stats"]["TOTAL_DAMAGE_DEALT_TO_TURRETS"]
        - left["stats"]["TOTAL_DAMAGE_DEALT_TO_TURRETS"],
        "true_damage": right["stats"]["TRUE_DAMAGE_DEALT_PLAYER"]
        - left["stats"]["TRUE_DAMAGE_DEALT_PLAYER"],
    }


def _window_deltas(
    game: GameData, start_ms: int, end_ms: int
) -> tuple[dict[str, Any], dict[int, dict[str, float]]]:
    start = _frame_after(game.frames, start_ms)
    end = _frame_after(game.frames, end_ms)
    if end.time_ms <= start.time_ms:
        raise GridSequenceReviewError("window end must follow window start")
    return (
        {
            "requested_start_ms": start_ms,
            "requested_end_ms": end_ms,
            "requested_start": format_clock(start_ms),
            "requested_end": format_clock(end_ms),
            "start_frame_ms": start.time_ms,
            "end_frame_ms": end.time_ms,
            "start_frame": format_clock(start.time_ms),
            "end_frame": format_clock(end.time_ms),
            "duration_seconds": (end.time_ms - start.time_ms) / 1000.0,
            "start_skew_ms": start.time_ms - start_ms,
            "end_skew_ms": end.time_ms - end_ms,
        },
        {
            row["participant_id"]: _player_delta(
                start, end, row["participant_id"]
            )
            for row in game.roster
        },
    )


def _clean_count(value: float) -> int | float:
    rounded = round(value)
    return int(rounded) if abs(value - rounded) <= 1e-4 else value


def _team_label(
    roster: Sequence[Mapping[str, Any]],
    team_id: int,
    explicit: Mapping[int, str] | None = None,
) -> tuple[str, str]:
    if explicit and team_id in explicit and str(explicit[team_id]).strip():
        return str(explicit[team_id]).strip(), "explicit"
    prefixes = {
        str(row.get("player") or "").strip().split()[0]
        for row in roster
        if int(row.get("team_id") or 0) == team_id
        and str(row.get("player") or "").strip()
    }
    if len(prefixes) == 1:
        return next(iter(prefixes)), "common_player_tag"
    return f"Team {team_id}", "team_id_fallback"


def _plate_gold_by_participant(
    game: GameData, *, start_frame_ms: int, end_frame_ms: int
) -> dict[int, float]:
    awards: Counter[int] = Counter()
    for event in game.events:
        if event.get("schema") != "turret_plate_gold_earned":
            continue
        event_ms = int(event.get("game_time_ms") or 0)
        # A state delta is end minus start. Matching the event interval to
        # (start, end] prevents an award already present in the start frame
        # from being subtracted twice.
        if not start_frame_ms < event_ms <= end_frame_ms:
            continue
        participant_id = int(event.get("participantID") or 0)
        if participant_id:
            awards[participant_id] += _number(event.get("bounty"))
    return dict(awards)


def _sum_resource_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    values = list(rows)
    return {
        "cs": sum(_number(row.get("cs")) for row in values),
        "gold_excluding_plates": sum(
            _number(row.get("gold_excluding_plates")) for row in values
        ),
        "plate_gold": sum(_number(row.get("plate_gold")) for row in values),
        "total_gold": sum(_number(row.get("total_gold")) for row in values),
        "xp": sum(_number(row.get("xp")) for row in values),
    }


def _resource_comparison(
    totals: Mapping[int, Mapping[str, Any]], reference_team_id: int
) -> dict[str, Any]:
    opponents = [team_id for team_id in totals if team_id != reference_team_id]
    if reference_team_id not in totals or len(opponents) != 1:
        raise GridSequenceReviewError(
            "resource comparison requires exactly two represented teams"
        )
    opponent_team_id = opponents[0]
    keys = ("cs", "gold_excluding_plates", "plate_gold", "total_gold", "xp")
    return {
        "reference_team_id": reference_team_id,
        "opponent_team_id": opponent_team_id,
        "reference_minus_opponent": {
            key: _number(totals[reference_team_id].get(key))
            - _number(totals[opponent_team_id].get(key))
            for key in keys
        },
    }


def build_resource_views(
    game: GameData,
    *,
    start_ms: int,
    end_ms: int,
    reference_team_id: int,
    involved_champions: Sequence[str] | None = None,
    team_labels: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    """Build additive actual-resource tables with explicit row scope.

    ``total_gold`` already includes plate awards in LiveStats.  The two gold
    columns below are therefore a partition, not two independent gains.
    Every subtotal is recomputed only from its visible rows.
    """

    window, deltas = _window_deltas(game, start_ms, end_ms)
    plate_gold = _plate_gold_by_participant(
        game,
        start_frame_ms=int(window["start_frame_ms"]),
        end_frame_ms=int(window["end_frame_ms"]),
    )
    roster_by_id = {int(row["participant_id"]): row for row in game.roster}
    rows: list[dict[str, Any]] = []
    for participant_id in sorted(roster_by_id):
        roster_row = roster_by_id[participant_id]
        delta = deltas[participant_id]
        plates = _number(plate_gold.get(participant_id))
        total_gold = _number(delta.get("total_gold"))
        excluding = total_gold - plates
        if excluding < -1e-6:
            raise GridSequenceReviewError(
                f"plate gold exceeds total gold for participant {participant_id}"
            )
        rows.append(
            {
                "participant_id": participant_id,
                "team_id": int(roster_row["team_id"]),
                "champion": str(roster_row["champion"]),
                "player": str(roster_row["player"]),
                "role": str(roster_row["role"]),
                "lane_cs": _clean_count(_number(delta.get("cs"))),
                "neutral_cs": _clean_count(_number(delta.get("neutral_cs"))),
                "cs": _clean_count(
                    _number(delta.get("cs")) + _number(delta.get("neutral_cs"))
                ),
                "gold_excluding_plates": excluding,
                "plate_gold": plates,
                "total_gold": total_gold,
                "xp": _number(delta.get("xp")),
            }
        )

    normalized_roster: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = "".join(character for character in row["champion"].casefold() if character.isalnum())
        normalized_roster.setdefault(key, []).append(row)

    if involved_champions:
        requested: list[str] = []
        seen: set[str] = set()
        for champion in involved_champions:
            key = "".join(
                character for character in str(champion).casefold() if character.isalnum()
            )
            if not key or key in seen:
                raise GridSequenceReviewError(
                    "involved champion allowlist is empty or contains duplicates"
                )
            seen.add(key)
            matches = normalized_roster.get(key) or []
            if len(matches) != 1:
                raise GridSequenceReviewError(
                    f"involved champion {champion!r} does not resolve uniquely"
                )
            requested.append(key)
        selected_rows = [
            row
            for row in rows
            if "".join(
                character for character in row["champion"].casefold() if character.isalnum()
            )
            in seen
        ]
        selection_mode = "explicit_champion_allowlist"
    else:
        requested = [row["champion"] for row in rows]
        selected_rows = list(rows)
        selection_mode = "all_players_default"

    team_ids = sorted({int(row["team_id"]) for row in rows})
    if len(team_ids) != 2 or reference_team_id not in team_ids:
        raise GridSequenceReviewError("resource view does not resolve two teams")
    labels: dict[int, dict[str, str]] = {}
    for team_id in team_ids:
        label, method = _team_label(game.roster, team_id, team_labels)
        labels[team_id] = {"label": label, "method": method}

    def view(view_rows: Sequence[Mapping[str, Any]], *, scope: str) -> dict[str, Any]:
        totals: dict[int, dict[str, Any]] = {}
        for team_id in team_ids:
            visible = [row for row in view_rows if int(row["team_id"]) == team_id]
            summed = _sum_resource_rows(visible)
            if abs(
                summed["gold_excluding_plates"]
                + summed["plate_gold"]
                - summed["total_gold"]
            ) > 1e-6:
                raise GridSequenceReviewError("resource gold partition does not add up")
            totals[team_id] = {
                "team_id": team_id,
                "team": labels[team_id]["label"],
                "players_included": len(visible),
                **summed,
            }
        return {
            "scope": scope,
            "rows": [dict(row) for row in view_rows],
            "team_totals": totals,
            "comparison": _resource_comparison(totals, reference_team_id),
        }

    return {
        "window": window,
        "gold_identity": "gold_excluding_plates + plate_gold = total_gold",
        "plate_event_interval": "(start_frame_ms, end_frame_ms]",
        "team_labels": labels,
        "selected": {
            "selection_mode": selection_mode,
            "requested": requested,
            **view(selected_rows, scope="visible_rows_only"),
        },
        "full_teams": view(rows, scope="all_five_players_per_team"),
    }


def _distance(player: Mapping[str, Any], position: Mapping[str, Any]) -> float:
    return math.hypot(
        _number(player.get("x")) - _number(position.get("x")),
        _number(player.get("z")) - _number(position.get("z")),
    )


def _objective_reward(
    game: GameData, events: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if not events:
        return {
            "event_count": 0,
            "direct_killer_gold": 0.0,
            "event_adjacent_xp": 0.0,
            "xp_by_participant": {},
        }
    xp_by_pid: Counter[int] = Counter()
    eligible_by_event = []
    for event in events:
        event_time = int(event["game_time_ms"])
        before = _frame_before(game.frames, event_time)
        after = _frame_after(game.frames, event_time)
        position = event.get("position") or {}
        eligible = []
        for participant_id, state in before.players.items():
            if not isinstance(position, Mapping):
                continue
            distance = _distance(state, position)
            if distance > XP_RADIUS:
                continue
            delta = _player_delta(before, after, participant_id)["xp"]
            if delta <= 0:
                continue
            xp_by_pid[participant_id] += delta
            eligible.append(
                {
                    "participant_id": participant_id,
                    "distance": round(distance, 1),
                    "xp_delta": delta,
                }
            )
        eligible_by_event.append(
            {
                "game_time_ms": event_time,
                "game_time": format_clock(event_time),
                "eligible": eligible,
            }
        )
    roster = {row["participant_id"]: row for row in game.roster}
    return {
        "event_count": len(events),
        "direct_killer_gold": sum(_number(event.get("killerGold")) for event in events),
        "event_adjacent_xp": sum(xp_by_pid.values()),
        "xp_by_participant": {
            roster[participant_id]["champion"]: value
            for participant_id, value in sorted(xp_by_pid.items())
            if participant_id in roster
        },
        "eligibility_by_event": eligible_by_event,
        "method": (
            "positive XP delta between the last stats frame before and first stats "
            "frame after each kill, restricted to players within 2000 units"
        ),
    }


def _event_clock(row: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result["game_time"] = format_clock(int(row["game_time_ms"]))
    return result


def analyze_delayed_camps(
    game: GameData,
    *,
    sequence_end_ms: int,
    taking_team_id: int,
    requested_camps: Sequence[str] | None,
    grub_reward: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit explicitly named camps that the Grub jungler cleared later.

    The caller supplies the counterfactual claim (which camps were available).
    This function only verifies the later same-game clear and its state delta.
    """

    if not requested_camps:
        return {"status": "not_requested"}
    aliases = {
        "gromp": "gromp",
        "wolf": "wolf",
        "wolves": "wolf",
        "murkwolf": "wolf",
        "murkwolves": "wolf",
    }
    normalized: list[str] = []
    for raw in requested_camps:
        key = "".join(character for character in str(raw).casefold() if character.isalnum())
        monster_type = aliases.get(key)
        if monster_type is None or monster_type in normalized:
            raise GridSequenceReviewError(
                f"delayed camp {raw!r} is unsupported or duplicated"
            )
        normalized.append(monster_type)

    junglers = [
        row
        for row in game.roster
        if int(row["team_id"]) == taking_team_id
        and str(row["role"]).casefold() == "jungle"
    ]
    if len(junglers) != 1:
        raise GridSequenceReviewError("taking team does not have one unique jungler")
    jungler = junglers[0]
    participant_id = int(jungler["participant_id"])
    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    for monster_type in normalized:
        candidates = [
            event
            for event in game.events
            if event.get("schema") == "epic_monster_kill"
            and str(event.get("monsterType") or "") == monster_type
            and int(event.get("killer") or 0) == participant_id
            and sequence_end_ms < int(event["game_time_ms"])
            <= sequence_end_ms + CAMP_LOOKAHEAD_MS
        ]
        if not candidates:
            blockers.append(f"no_later_same_jungler_clear:{monster_type}")
            continue
        event = min(candidates, key=lambda row: int(row["game_time_ms"]))
        event_ms = int(event["game_time_ms"])
        nearby_other_camps = [
            other
            for other in game.events
            if other.get("schema") == "epic_monster_kill"
            and int(other.get("killer") or 0) == participant_id
            and other is not event
            and event_ms - CAMP_EVENT_PADDING_MS
            <= int(other["game_time_ms"])
            <= event_ms + CAMP_EVENT_PADDING_MS
        ]
        if nearby_other_camps:
            blockers.append(f"overlapping_later_camp_clear:{monster_type}")
            continue
        window, deltas = _window_deltas(
            game,
            event_ms - CAMP_EVENT_PADDING_MS,
            event_ms + CAMP_EVENT_PADDING_MS,
        )
        delta = deltas[participant_id]
        contamination = {
            key: _number(delta.get(key))
            for key in (
                "cs",
                "minion_gold",
                "support_minion_gold",
                "champion_kill_gold",
                "assist_gold",
                "building_gold",
            )
            if abs(_number(delta.get(key))) > 1e-6
        }
        if contamination:
            blockers.append(f"contaminated_later_camp_clear:{monster_type}")
            continue
        rows.append(
            {
                "monster_type": monster_type,
                "event": _event_clock(event),
                "measurement_window": window,
                "gold": _number(delta.get("neutral_gold")),
                "xp": _number(delta.get("xp")),
                "neutral_cs": _clean_count(_number(delta.get("neutral_cs"))),
                "non_neutral_resource_contamination": contamination,
            }
        )
    if blockers:
        return {
            "status": "unavailable",
            "blockers": blockers,
            "requested_camps": normalized,
            "resolved": rows,
        }

    camp_total = _sum_ledger(rows)
    pit = (grub_reward.get("pit_resources_by_champion") or {}).get(
        str(jungler["champion"]), {}
    )
    grub_jungler = {
        "gold": _number(pit.get("neutral_gold")),
        "xp": _number(pit.get("objective_xp")),
    }
    return {
        "status": "verified_later_same_game_clears",
        "claim_source": "analyst-supplied camp availability; not inferred from GRID",
        "jungler": str(jungler["champion"]),
        "requested_camps": normalized,
        "later_clears": rows,
        "later_camp_resources": camp_total,
        "grub_pit_resources_for_jungler": grub_jungler,
        "camps_minus_grubs": {
            "gold": camp_total["gold"] - grub_jungler["gold"],
            "xp": camp_total["xp"] - grub_jungler["xp"],
        },
        "interpretation": (
            "These resources were delayed, not permanently lost. Use this only "
            "as the named farm-instead counterfactual; actual team totals already "
            "contain the observed route and must not subtract it again."
        ),
    }


def _is_touch_compatible(true_delta: float, building_delta: float) -> bool:
    if true_delta <= 0 or building_delta <= 0 or true_delta > building_delta + 1e-3:
        return False
    units = round(true_delta / TOUCH_BASE_QUANTUM)
    return 1 <= units <= 6 and abs(true_delta - units * TOUCH_BASE_QUANTUM) <= TOUCH_TOLERANCE


def analyze_siege(
    game: GameData,
    *,
    start_ms: int,
    end_ms: int,
    attacking_team_id: int,
    lane: str,
) -> dict[str, Any]:
    window, _ = _window_deltas(game, start_ms, end_ms)
    start_frame = _frame_after(game.frames, start_ms)
    end_frame = _frame_after(game.frames, end_ms)
    roster = {row["participant_id"]: row for row in game.roster}
    attacking_ids = {
        row["participant_id"]
        for row in game.roster
        if row["team_id"] == attacking_team_id
    }
    selected = [
        frame
        for frame in game.frames
        if start_frame.time_ms <= frame.time_ms <= end_frame.time_ms
    ]
    touch_by_pid: Counter[int] = Counter()
    other_true_by_pid: Counter[int] = Counter()
    for left, right in zip(selected, selected[1:]):
        for participant_id in attacking_ids:
            if participant_id not in left.players or participant_id not in right.players:
                continue
            delta = _player_delta(left, right, participant_id)
            true_delta = delta["true_damage"]
            building_delta = delta["building_damage"]
            if _is_touch_compatible(true_delta, building_delta):
                touch_by_pid[participant_id] += true_delta
            elif true_delta > 0 and building_delta > 0:
                other_true_by_pid[participant_id] += true_delta

    building_by_pid = {
        participant_id: max(
            0.0,
            _player_delta(start_frame, end_frame, participant_id)[
                "building_damage"
            ],
        )
        for participant_id in attacking_ids
    }
    champion_building_damage = sum(building_by_pid.values())
    touch_damage = sum(touch_by_pid.values())
    non_touch_damage = champion_building_damage - touch_damage
    duration = window["duration_seconds"]
    non_touch_dps = non_touch_damage / duration if duration > 0 else 0.0
    time_saved = touch_damage / non_touch_dps if non_touch_dps > 0 else None
    counterfactual_duration = duration + time_saved if time_saved is not None else None

    destroyed_team_id = next(
        (
            row["team_id"]
            for row in game.roster
            if row["team_id"] != attacking_team_id
        ),
        None,
    )
    plates = [
        event
        for event in game.events
        if event["schema"] == "turret_plate_destroyed"
        and start_ms <= int(event["game_time_ms"]) <= end_ms
        and str(event.get("lane") or "").lower() == lane.lower()
        and int(event.get("teamID") or 0) == destroyed_team_id
    ]
    plate_times = {int(event["game_time_ms"]) for event in plates}
    plate_gold_rows = [
        event
        for event in game.events
        if event["schema"] == "turret_plate_gold_earned"
        and int(event["game_time_ms"]) in plate_times
        and int(event.get("teamID") or 0) == attacking_team_id
    ]
    plate_gold_by_pid: Counter[int] = Counter()
    for event in plate_gold_rows:
        plate_gold_by_pid[int(event.get("participantID") or 0)] += _number(
            event.get("bounty")
        )

    return {
        "window": window,
        "lane": lane,
        "attacking_team_id": attacking_team_id,
        "plates": [_event_clock(event) for event in plates],
        "plate_gold": sum(plate_gold_by_pid.values()),
        "plate_gold_by_champion": {
            roster[participant_id]["champion"]: value
            for participant_id, value in sorted(plate_gold_by_pid.items())
            if participant_id in roster
        },
        "champion_building_damage": champion_building_damage,
        "building_damage_by_champion": {
            roster[participant_id]["champion"]: value
            for participant_id, value in sorted(building_by_pid.items())
            if value > 0 and participant_id in roster
        },
        "touch_compatible_true_damage": touch_damage,
        "touch_by_champion": {
            roster[participant_id]["champion"]: value
            for participant_id, value in sorted(touch_by_pid.items())
            if participant_id in roster
        },
        "other_building_window_true_damage": sum(other_true_by_pid.values()),
        "other_true_by_champion": {
            roster[participant_id]["champion"]: value
            for participant_id, value in sorted(other_true_by_pid.items())
            if participant_id in roster
        },
        "touch_share_of_champion_building_damage": (
            touch_damage / champion_building_damage
            if champion_building_damage > 0
            else None
        ),
        "non_touch_champion_building_damage": non_touch_damage,
        "non_touch_champion_dps": non_touch_dps,
        "conditional_time_saved_seconds": time_saved,
        "conditional_no_touch_duration_seconds": counterfactual_duration,
        "classification": (
            "true-damage increments that coincide with building damage and are "
            "exact multiples of the patch-26.15 three-stack ranged tick (8), "
            "bounded to at most six tick units per ~1s stats interval"
        ),
        "limits": [
            "Champion-attributed building damage is not total turret damage; direct minion and Voidmite damage is absent.",
            "Voidmite attacks can apply or refresh the summoner's Touch, so their Touch ticks may remain champion-attributed even though direct Voidmite attack damage is absent.",
            "The time counterfactual holds observed non-Touch champion DPS constant and does not replay plate armor, minion targeting, Voidmite attacks, or turret health.",
        ],
    }


def _plates_reached_at_health(health: float) -> int:
    if not 0 <= health <= OUTER_TURRET_HP:
        raise GridSequenceReviewError("outer turret health is outside [0, 9000]")
    return sum(
        health <= threshold + 1e-6
        for threshold in OUTER_PLATE_REMAINING_HP_THRESHOLDS
    )


def analyze_turret_health(
    siege: Mapping[str, Any],
    observation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    unavailable = {
        "status": "unavailable",
        "reason": "Riot LiveStats does not expose structure health",
        "forbidden_shortcut": (
            "Do not subtract champion-attributed building damage from 9000: "
            "that omits minion and direct Voidmite damage and may start after prior damage."
        ),
    }
    if observation is None:
        return unavailable
    try:
        estimate = float(observation["health_estimate"])
        low = float(observation.get("health_low", estimate))
        high = float(observation.get("health_high", estimate))
        observed_ms = parse_clock(observation["game_time"])
    except (KeyError, TypeError, ValueError) as exc:
        raise GridSequenceReviewError("turret observation is malformed") from exc
    if not 0 <= low <= estimate <= high <= OUTER_TURRET_HP:
        raise GridSequenceReviewError("turret observation bounds are invalid")
    touch = _number(siege.get("touch_compatible_true_damage"))
    no_touch = {
        "health_estimate": min(OUTER_TURRET_HP, estimate + touch),
        "health_low": min(OUTER_TURRET_HP, low + touch),
        "health_high": min(OUTER_TURRET_HP, high + touch),
    }
    plate_counts = {
        _plates_reached_at_health(no_touch[key])
        for key in ("health_low", "health_estimate", "health_high")
    }
    no_touch["plate_thresholds_reached"] = (
        next(iter(plate_counts)) if len(plate_counts) == 1 else sorted(plate_counts)
    )
    return {
        "status": "observer_estimate_available",
        "exact_live_stats_health": unavailable,
        "observation": {
            "game_time_ms": observed_ms,
            "game_time": format_clock(observed_ms),
            "health_estimate": estimate,
            "health_low": low,
            "health_high": high,
            "percent_estimate": estimate / OUTER_TURRET_HP,
            "source": observation.get("source"),
            "method": str(observation.get("method") or "observer_health_bar"),
        },
        "outer_turret_health": OUTER_TURRET_HP,
        "plate_remaining_hp_thresholds": list(
            OUTER_PLATE_REMAINING_HP_THRESHOLDS
        ),
        "fixed_state_remove_touch_only": {
            **no_touch,
            "health_percent_estimate": no_touch["health_estimate"]
            / OUTER_TURRET_HP,
            "touch_removed": touch,
            "is_lower_bound_on_health_without_all_grub_effects": True,
        },
        "limits": [
            "The observer bar is an estimate, not an exact LiveStats field.",
            "Adding Touch back holds every other hit and defense decision fixed; it is not a replay.",
            "Direct Voidmite attack damage is unavailable, so removing all Grub effects would leave the turret at least this healthy.",
        ],
    }


def _role_map(roster: Sequence[Mapping[str, Any]], team_id: int) -> dict[str, dict[str, Any]]:
    roles: dict[str, dict[str, Any]] = {}
    for row in roster:
        if int(row["team_id"]) != team_id:
            continue
        key = str(row["role"]).strip().lower()
        if key in roles:
            raise GridSequenceReviewError(f"team {team_id} has duplicate role {key}")
        roles[key] = dict(row)
    required = {"top", "jungle", "middle", "bottom", "support"}
    if set(roles) != required:
        raise GridSequenceReviewError(f"team {team_id} roles are incomplete")
    return roles


def infer_cannon_gold(
    game: GameData, *, resource_start_ms: int, resource_end_ms: int
) -> dict[str, Any]:
    selected = [
        frame
        for frame in game.frames
        if resource_start_ms - 15_000 <= frame.time_ms <= resource_end_ms + 5_000
    ]
    candidates: list[dict[str, Any]] = []
    roster = {row["participant_id"]: row for row in game.roster}
    for left, right in zip(selected, selected[1:]):
        for participant_id in left.players:
            if participant_id not in right.players:
                continue
            delta = _player_delta(left, right, participant_id)
            gold = delta["minion_gold"]
            if delta["cs"] != 1 or not 50 <= gold <= 70:
                continue
            candidates.append(
                {
                    "game_time_ms": right.time_ms,
                    "game_time": format_clock(right.time_ms),
                    "champion": roster.get(participant_id, {}).get("champion"),
                    "gold": gold,
                    "xp_delta": delta["xp"],
                }
            )
    values = [int(round(row["gold"])) for row in candidates]
    counts = Counter(values)
    if not counts:
        return {
            "status": "unavailable",
            "value": None,
            "blockers": ["no_same-wave_observed_cannon_last_hit"],
            "candidates": [],
        }
    top = counts.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return {
            "status": "unavailable",
            "value": None,
            "blockers": ["ambiguous_same-wave_cannon_gold"],
            "candidates": candidates,
        }
    return {
        "status": "verified_from_same_game_stats",
        "value": float(top[0][0]),
        "candidates": candidates,
    }


def _sum_ledger(rows: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    values = list(rows)
    return {
        "gold": sum(_number(row.get("gold")) for row in values),
        "xp": sum(_number(row.get("xp")) for row in values),
    }


def build_crossmap_counterfactual(
    game: GameData,
    *,
    sequence_start_ms: int,
    resource_start_ms: int,
    resource_end_ms: int,
    taking_team_id: int,
    grub_reward: Mapping[str, Any],
    siege: Mapping[str, Any],
    cannon_gold_override: float | None = None,
) -> dict[str, Any]:
    opponent_team_id = next(
        row["team_id"]
        for row in game.roster
        if row["team_id"] != taking_team_id
    )
    taking_roles = _role_map(game.roster, taking_team_id)
    opponent_roles = _role_map(game.roster, opponent_team_id)
    resource_window, resource_deltas = _window_deltas(
        game, resource_start_ms, resource_end_ms
    )
    cannon = infer_cannon_gold(
        game,
        resource_start_ms=resource_start_ms,
        resource_end_ms=resource_end_ms,
    )
    if cannon_gold_override is not None:
        cannon = {
            "status": "explicit_override",
            "value": float(cannon_gold_override),
            "candidates": cannon.get("candidates", []),
        }
    if cannon.get("value") is None:
        return {
            "status": "unavailable",
            "blockers": list(cannon.get("blockers") or []),
            "resource_window": resource_window,
            "cannon_gold": cannon,
        }
    cannon_gold = float(cannon["value"])
    cannon_wave_gold = REGULAR_WAVE_GOLD + cannon_gold
    duo_cannon_xp = CANNON_WAVE_XP * DUO_XP_SHARE

    bottom = taking_roles["bottom"]
    support = taking_roles["support"]
    top = taking_roles["top"]
    opponent_support = opponent_roles["support"]
    defender = opponent_roles["top"]

    bottom_delta = resource_deltas[bottom["participant_id"]]
    support_delta = resource_deltas[support["participant_id"]]
    top_delta = resource_deltas[top["participant_id"]]
    support_control = resource_deltas[opponent_support["participant_id"]][
        "support_minion_gold"
    ]
    support_grub_xp = _number(
        (grub_reward.get("xp_by_participant") or {}).get(support["champion"])
    )
    support_xp_inside_resource = min(support_delta["xp"], support_grub_xp)
    support_lane_xp_inside_resource = (
        support_delta["xp"] - support_xp_inside_resource
    )

    observed_wave_only = {
        bottom["champion"]: {
            "gold": bottom_delta["minion_gold"],
            "xp": bottom_delta["xp"],
            "cs": bottom_delta["cs"],
            "components": ["resource-window lane minion stats"],
        },
        support["champion"]: {
            "gold": support_delta["support_minion_gold"],
            "xp": support_lane_xp_inside_resource + support_grub_xp,
            "cs": support_delta["cs"],
            "components": [
                "resource-window lane XP after removing event-adjacent Grub XP",
                "all event-adjacent Grub XP for this player",
            ],
        },
        top["champion"]: {
            "gold": top_delta["minion_gold"],
            "xp": top_delta["xp"],
            "cs": top_delta["cs"],
            "components": ["resource-window lane minion stats"],
        },
    }
    expected = {
        bottom["champion"]: {
            "gold": cannon_wave_gold,
            "xp": duo_cannon_xp,
            "cs": 7.0,
            "assumption": "full bot cannon wave shared with support",
        },
        support["champion"]: {
            "gold": support_control,
            "xp": duo_cannon_xp,
            "cs": 0.0,
            "assumption": (
                "full bot cannon wave shared with bottom; opponent support's "
                "same-window support-minion gold is the control"
            ),
        },
        top["champion"]: {
            "gold": cannon_wave_gold,
            "xp": CANNON_WAVE_XP,
            "cs": 7.0,
            "assumption": "full top cannon wave solo",
        },
    }

    def comparison(observed: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        rows = {}
        for champion in expected:
            rows[champion] = {
                "observed": dict(observed[champion]),
                "counterfactual": dict(expected[champion]),
                "delta_observed_minus_counterfactual": {
                    "gold": _number(observed[champion].get("gold"))
                    - _number(expected[champion].get("gold")),
                    "xp": _number(observed[champion].get("xp"))
                    - _number(expected[champion].get("xp")),
                    "cs": _number(observed[champion].get("cs"))
                    - _number(expected[champion].get("cs")),
                },
            }
        total = _sum_ledger(
            row["delta_observed_minus_counterfactual"] for row in rows.values()
        )
        return {"players": rows, "total_delta": total}

    wave_only = comparison(observed_wave_only)

    last_grub_ms = max(
        int(event["game_time_ms"])
        for event in game.events
        if event["schema"] == "epic_monster_kill"
        and str(event.get("monsterType") or "") == "VoidGrub"
        and int(event.get("killerTeamID") or 0) == taking_team_id
    )
    support_route_end = _frame_after(game.frames, last_grub_ms).time_ms
    route_window, route_deltas = _window_deltas(
        game, sequence_start_ms, support_route_end
    )
    support_route_delta = route_deltas[support["participant_id"]]
    observed_decision_complete = dict(observed_wave_only)
    observed_decision_complete[support["champion"]] = {
        "gold": support_route_delta["support_minion_gold"]
        + support_route_delta["neutral_gold"],
        "xp": support_route_delta["xp"],
        "cs": support_route_delta["cs"],
        "components": [
            "all support XP, support-minion gold, and neutral-minion gold from "
            "the declared decision start through the first stats frame after the final Grub"
        ],
    }
    decision_complete = comparison(observed_decision_complete)

    siege_window = siege["window"]
    siege_start = _frame_after(game.frames, int(siege_window["requested_start_ms"]))
    siege_end = _frame_after(game.frames, int(siege_window["requested_end_ms"]))
    defender_delta = _player_delta(
        siege_start, siege_end, defender["participant_id"]
    )
    collector_delta = _player_delta(
        siege_start, siege_end, bottom["participant_id"]
    )
    if (
        abs(collector_delta["cs"] - 6.0) <= 1e-6
        and abs(collector_delta["minion_gold"] - REGULAR_WAVE_GOLD) <= 1e-6
    ):
        denied_wave = {
            "status": "verified_from_attacking_collector",
            "kind": "regular",
            "expected_gold": REGULAR_WAVE_GOLD,
            "expected_xp": REGULAR_WAVE_XP,
            "expected_cs": 6.0,
        }
    else:
        denied_wave = {
            "status": "unavailable",
            "kind": None,
            "blockers": ["attacking_collector_does_not_resolve_one_full_wave"],
            "collector_observed": collector_delta,
        }
    if denied_wave["status"] != "unavailable":
        denied_wave.update(
            {
                "defender": defender["champion"],
                "defender_observed": {
                    "gold": defender_delta["minion_gold"],
                    "xp": defender_delta["xp"],
                    "cs": defender_delta["cs"],
                    "ambient_or_other_total_gold": defender_delta["total_gold"],
                },
                "denied": {
                    "gold": denied_wave["expected_gold"]
                    - defender_delta["minion_gold"],
                    "xp": denied_wave["expected_xp"] - defender_delta["xp"],
                    "cs": denied_wave["expected_cs"] - defender_delta["cs"],
                },
            }
        )

    def ledger(comparison_row: Mapping[str, Any]) -> dict[str, Any]:
        own = {
            "gold": _number(comparison_row["total_delta"]["gold"])
            + _number(siege.get("plate_gold")),
            "xp": _number(comparison_row["total_delta"]["xp"]),
        }
        denied = denied_wave.get("denied") or {"gold": 0.0, "xp": 0.0}
        relative = {
            "gold": own["gold"] + _number(denied.get("gold")),
            "xp": own["xp"] + _number(denied.get("xp")),
        }
        return {
            "own_selected_roles_after_plates": own,
            "relative_selected_roles_plus_defender_denial": relative,
            "not_a_team_total": True,
        }

    return {
        "status": "assumption_sensitive",
        "mechanics_profile_id": MECHANICS_PROFILE_ID,
        "resource_window": resource_window,
        "route_window_for_support": route_window,
        "cannon_gold": cannon,
        "wave_values": {
            "regular": {"gold": REGULAR_WAVE_GOLD, "xp": REGULAR_WAVE_XP},
            "cannon": {
                "gold": cannon_wave_gold,
                "solo_xp": CANNON_WAVE_XP,
                "duo_xp_each": duo_cannon_xp,
            },
        },
        "support_gold_control": {
            "champion": opponent_support["champion"],
            "gold": support_control,
        },
        "wave_only_selected_roles": wave_only,
        "decision_complete_support_route": decision_complete,
        "defender_denial": denied_wave,
        "ledgers": {
            "wave_only": ledger(wave_only),
            "decision_complete_support_route": ledger(decision_complete),
        },
        "limits": [
            "The wave-only ledger reproduces the narrow 8:30-8:45 attribution and can omit resources earned earlier in the declared decision path.",
            "The decision-complete support route includes off-lane support resources but does not estimate the XP displaced from the allied mid laner, so it is still not a full-team counterfactual.",
            "Objective cash/XP for other players and the first-drake buff are reported separately to prevent double counting.",
        ],
    }


def _mechanics_profile(game_version: str) -> dict[str, Any]:
    compatible = game_version.startswith("16.15.") or game_version == "16.15"
    return {
        "profile_id": MECHANICS_PROFILE_ID,
        "game_version": game_version,
        "status": "verified_for_counterfactual" if compatible else "unavailable",
        "blockers": [] if compatible else ["game_version_outside_profile_16.15"],
        "sources": MECHANICS_SOURCES,
    }


def _finalize_report_hashes(report: dict[str, Any]) -> None:
    analysis_payload = {
        key: value
        for key, value in report.items()
        if key not in {"runtime_seconds", "report_sha256", "analysis_sha256"}
    }
    # Stable across re-downloads of identical bytes and machine runtime. The
    # retrieval receipt remains in the full report, while the analysis hash is
    # intentionally bound to its content hash rather than its volatile path.
    analysis_payload["source"] = {
        "raw_sha256": report["source"]["raw_sha256"],
        "credentials_serialized": False,
        "signed_url_retained": False,
    }
    report["analysis_sha256"] = _hash(analysis_payload)
    report["report_sha256"] = _hash(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )


def _analyze_sequence_monolith(
    *,
    source_path: Path,
    receipt: Mapping[str, Any],
    sequence_start_ms: int,
    sequence_end_ms: int,
    resource_start_ms: int,
    resource_end_ms: int,
    siege_start_ms: int,
    siege_end_ms: int,
    lane: str = "top",
    cannon_gold_override: float | None = None,
    involved_champions: Sequence[str] | None = None,
    team_labels: Mapping[int, str] | None = None,
    delayed_camps: Sequence[str] | None = None,
    turret_observation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    game = load_game(source_path)
    mechanics = _mechanics_profile(str(game.identity.get("game_version") or ""))
    sequence_events = [
        event
        for event in game.events
        if sequence_start_ms <= int(event["game_time_ms"]) <= sequence_end_ms
    ]
    grubs = [
        event
        for event in sequence_events
        if event["schema"] == "epic_monster_kill"
        and str(event.get("monsterType") or "") == "VoidGrub"
    ]
    taking_teams = {int(event.get("killerTeamID") or 0) for event in grubs}
    if len(grubs) != 3 or len(taking_teams) != 1:
        raise GridSequenceReviewError(
            "declared sequence does not contain one three-Grub sweep"
        )
    taking_team_id = next(iter(taking_teams))
    dragon_events = [
        event
        for event in sequence_events
        if event["schema"] == "epic_monster_kill"
        and str(event.get("monsterType") or "") == "dragon"
    ]
    grub_reward = _objective_reward(game, grubs)
    dragon_reward = _objective_reward(game, dragon_events)

    first_grub_before = _frame_before(game.frames, min(event["game_time_ms"] for event in grubs))
    last_grub_after = _frame_after(game.frames, max(event["game_time_ms"] for event in grubs))
    taking_ids = {
        row["participant_id"]
        for row in game.roster
        if row["team_id"] == taking_team_id
    }
    pit_neutral_gold = sum(
        _player_delta(first_grub_before, last_grub_after, participant_id)[
            "neutral_gold"
        ]
        for participant_id in taking_ids
    )
    grub_reward["observed_pit_neutral_gold"] = pit_neutral_gold
    grub_reward["incidental_neutral_gold_beyond_killer_gold"] = (
        pit_neutral_gold - grub_reward["direct_killer_gold"]
    )
    roster_by_id = {row["participant_id"]: row for row in game.roster}
    pit_resources: dict[str, dict[str, Any]] = {}
    event_xp = grub_reward.get("xp_by_participant") or {}
    for participant_id in sorted(taking_ids):
        champion = roster_by_id[participant_id]["champion"]
        delta = _player_delta(first_grub_before, last_grub_after, participant_id)
        neutral_gold = _number(delta["neutral_gold"])
        objective_xp = _number(event_xp.get(champion))
        if neutral_gold <= 0 and objective_xp <= 0:
            continue
        pit_resources[champion] = {
            "neutral_gold": neutral_gold,
            "objective_xp": objective_xp,
            "neutral_cs": _clean_count(_number(delta["neutral_cs"])),
        }
    grub_reward["pit_resources_by_champion"] = pit_resources

    siege = analyze_siege(
        game,
        start_ms=siege_start_ms,
        end_ms=siege_end_ms,
        attacking_team_id=taking_team_id,
        lane=lane,
    )
    resource_window, resource_deltas = _window_deltas(
        game, resource_start_ms, resource_end_ms
    )
    sequence_window, sequence_deltas = _window_deltas(
        game, sequence_start_ms, sequence_end_ms
    )
    resource_views = build_resource_views(
        game,
        start_ms=sequence_start_ms,
        end_ms=sequence_end_ms,
        reference_team_id=taking_team_id,
        involved_champions=involved_champions,
        team_labels=team_labels,
    )
    delayed_camp_review = analyze_delayed_camps(
        game,
        sequence_end_ms=sequence_end_ms,
        taking_team_id=taking_team_id,
        requested_camps=delayed_camps,
        grub_reward=grub_reward,
    )
    turret_health = analyze_turret_health(siege, turret_observation)

    counterfactual: dict[str, Any]
    if mechanics["status"] == "verified_for_counterfactual":
        counterfactual = build_crossmap_counterfactual(
            game,
            sequence_start_ms=sequence_start_ms,
            resource_start_ms=resource_start_ms,
            resource_end_ms=resource_end_ms,
            taking_team_id=taking_team_id,
            grub_reward=grub_reward,
            siege=siege,
            cannon_gold_override=cannon_gold_override,
        )
    else:
        counterfactual = {
            "status": "unavailable",
            "blockers": mechanics["blockers"],
        }
    if (
        counterfactual.get("status") != "unavailable"
        and delayed_camp_review.get("status")
        == "verified_later_same_game_clears"
    ):
        route = counterfactual["decision_complete_support_route"]["total_delta"]
        camps = delayed_camp_review["camps_minus_grubs"]
        defender = counterfactual["defender_denial"].get("denied") or {}
        actual_minus_named_farm = {
            "gold": _number(route.get("gold")) - _number(camps.get("gold")),
            "xp": _number(route.get("xp")) - _number(camps.get("xp")),
        }
        counterfactual["named_farm_alternative_including_delayed_camps"] = {
            "actual_minus_named_farm": actual_minus_named_farm,
            "received_less_than_named_farm": {
                "gold": -actual_minus_named_farm["gold"],
                "xp": -actual_minus_named_farm["xp"],
            },
            "after_actual_plates_and_defender_denial": {
                "gold": actual_minus_named_farm["gold"]
                + _number(siege.get("plate_gold"))
                + _number(defender.get("gold")),
                "xp": actual_minus_named_farm["xp"]
                + _number(defender.get("xp")),
            },
            "not_a_team_total": True,
            "recipe": (
                "decision-start wave allocation plus analyst-named later camps; "
                "then actual plates and the verified defender wave denial"
            ),
        }

    report = {
        "schema_version": SCHEMA_VERSION,
        "scope": "private_personal_research_only",
        "status": "complete",
        "source": {
            "raw_path": str(source_path),
            "raw_sha256": _sha256_file(source_path),
            "retrieval_receipt_path": receipt.get("receipt_path"),
            "retrieval_receipt_sha256": receipt.get("receipt_sha256"),
            "credentials_serialized": False,
            "signed_url_retained": False,
        },
        "identity": {
            **game.identity,
            "provider_series_id": receipt.get("provider_series_id"),
            "provider_game_id": receipt.get("provider_game_id"),
            "game_index": receipt.get("game_index"),
        },
        "roster": list(game.roster),
        "completeness": game.completeness,
        "mechanics": mechanics,
        "windows": {
            "sequence": sequence_window,
            "resource": resource_window,
            "siege": siege["window"],
        },
        "observed": {
            "taking_team_id": taking_team_id,
            "timeline": [_event_clock(event) for event in sequence_events],
            "grubs": {
                "events": [_event_clock(event) for event in grubs],
                "reward": grub_reward,
            },
            "dragon": {
                "events": [_event_clock(event) for event in dragon_events],
                "reward": dragon_reward,
            },
            "resource_window_by_champion": {
                roster_by_id[participant_id]["champion"]: delta
                for participant_id, delta in sorted(resource_deltas.items())
            },
            "sequence_window_by_champion": {
                roster_by_id[participant_id]["champion"]: delta
                for participant_id, delta in sorted(sequence_deltas.items())
            },
            "resource_views": resource_views,
            "siege": siege,
            "delayed_camps": delayed_camp_review,
            "turret_health": turret_health,
        },
        "counterfactual": counterfactual,
        "interpretation_boundary": {
            "observed": (
                "timestamps, state deltas, event rewards, plate gold, and "
                "champion-attributed damage from the private Riot LiveStats file"
            ),
            "conditional": (
                "wave allocation and turret-time calculations under the declared "
                "counterfactual assumptions"
            ),
            "unavailable": (
                "intrinsic first-drake buff value, a causal policy effect, total turret "
                "damage including minions/Voidmites, exact structure health from LiveStats, "
                "and a universal Grubs-versus-drake verdict"
            ),
        },
    }
    report["runtime_seconds"] = time.perf_counter() - started
    _finalize_report_hashes(report)
    return report


def analyze_sequence(
    *,
    source_path: Path,
    receipt: Mapping[str, Any],
    sequence_start_ms: int,
    sequence_end_ms: int,
    resource_start_ms: int,
    resource_end_ms: int,
    siege_start_ms: int,
    siege_end_ms: int,
    lane: str = "top",
    cannon_gold_override: float | None = None,
    involved_champions: Sequence[str] | None = None,
    team_labels: Mapping[int, str] | None = None,
    delayed_camps: Sequence[str] | None = None,
    turret_observation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the mandatory deterministic action graph through report assembly."""

    # Lazy import avoids a module cycle: the action implementations reuse this
    # module's narrow extraction and mechanics primitives.
    from lol_kills.research.grid_sequence_actions import run_analysis_action_graph

    return run_analysis_action_graph(
        source_path=source_path,
        receipt=receipt,
        sequence_start_ms=sequence_start_ms,
        sequence_end_ms=sequence_end_ms,
        resource_start_ms=resource_start_ms,
        resource_end_ms=resource_end_ms,
        siege_start_ms=siege_start_ms,
        siege_end_ms=siege_end_ms,
        lane=lane,
        cannon_gold_override=cannon_gold_override,
        involved_champions=involved_champions,
        team_labels=team_labels,
        delayed_camps=delayed_camps,
        turret_observation=turret_observation,
    )


def _display_number(value: Any, *, suffix: str = "") -> str:
    number = _number(value)
    if abs(number - round(number)) <= 1e-4:
        return f"{int(round(number)):,}{suffix}"
    return f"{number:,.1f}{suffix}"


def _more_or_less(value: Any, *, suffix: str = "") -> str:
    number = _number(value)
    if abs(number) <= 1e-6:
        return f"same{suffix}"
    return f"{_display_number(abs(number), suffix=suffix)} {'more' if number > 0 else 'less'}"


def _render_resource_view(view: Mapping[str, Any]) -> list[str]:
    rows = list(view.get("rows") or [])
    totals = view.get("team_totals") or {}
    comparison = view.get("comparison") or {}
    reference = int(comparison.get("reference_team_id") or 0)
    opponent = int(comparison.get("opponent_team_id") or 0)
    lines = [
        "| Team | Champion | CS | Gold without plates | Plate gold | Total gold | XP |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        team = totals[int(row["team_id"])]["team"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(team),
                    str(row["champion"]),
                    _display_number(row["cs"]),
                    _display_number(row["gold_excluding_plates"], suffix="g"),
                    _display_number(row["plate_gold"], suffix="g"),
                    _display_number(row["total_gold"], suffix="g"),
                    _display_number(row["xp"]),
                ]
            )
            + " |"
        )
    for team_id in (reference, opponent):
        row = totals[team_id]
        scope_label = (
            "Shown players"
            if str(view.get("scope")) == "visible_rows_only"
            else "Full team"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    f"**{scope_label}: {row['team']}**",
                    "",
                    f"**{_display_number(row['cs'])}**",
                    f"**{_display_number(row['gold_excluding_plates'], suffix='g')}**",
                    f"**{_display_number(row['plate_gold'], suffix='g')}**",
                    f"**{_display_number(row['total_gold'], suffix='g')}**",
                    f"**{_display_number(row['xp'])}**",
                ]
            )
            + " |"
        )
    difference = comparison.get("reference_minus_opponent") or {}
    lines.append(
        "| "
        + " | ".join(
            [
                "**Difference**",
                f"**{totals[reference]['team']} compared with {totals[opponent]['team']}**",
                f"**{_more_or_less(difference.get('cs'))}**",
                f"**{_more_or_less(difference.get('gold_excluding_plates'), suffix='g')}**",
                f"**{_more_or_less(difference.get('plate_gold'), suffix='g')}**",
                f"**{_more_or_less(difference.get('total_gold'), suffix='g')}**",
                f"**{_more_or_less(difference.get('xp'))}**",
            ]
        )
        + " |"
    )
    return lines


def render_summary(report: Mapping[str, Any]) -> str:
    identity = report["identity"]
    observed = report["observed"]
    siege = observed["siege"]
    conditional_time_saved = siege.get("conditional_time_saved_seconds")
    conditional_time_saved_text = (
        f"{conditional_time_saved:.1f}s"
        if isinstance(conditional_time_saved, (int, float))
        else "unavailable"
    )
    counterfactual = report["counterfactual"]
    lines = [
        f"# GRID sequence review: series {identity['provider_series_id']} game {identity['game_index']}",
        "",
        f"Source: `{report['source']['raw_sha256']}` on game version `{identity['game_version']}`.",
        "",
        "## Exact observed sequence",
        "",
    ]
    dragon = observed["dragon"]["events"]
    grubs = observed["grubs"]["events"]
    if dragon:
        lines.append(f"- Dragon: **{dragon[0]['game_time']}**")
    if grubs:
        lines.append(
            "- Grubs: " + " / ".join(f"**{row['game_time']}**" for row in grubs)
        )
    dragon_reward = observed["dragon"]["reward"]
    grub_reward = observed["grubs"]["reward"]
    lines.extend(
        [
            f"- Immediate objective resources: Grub pit **{grub_reward.get('observed_pit_neutral_gold', grub_reward.get('direct_killer_gold', 0)):.0f}g / {grub_reward.get('event_adjacent_xp', 0):.0f} XP**; dragon **{dragon_reward.get('direct_killer_gold', 0):.0f}g / {dragon_reward.get('event_adjacent_xp', 0):.0f} XP**",
            "- Top plates: "
            + " / ".join(
                f"**{row['game_time']}**" for row in siege.get("plates") or []
            ),
            f"- Plate gold: **{siege['plate_gold']:.0f}g**",
            f"- Touch-compatible true damage: **{siege['touch_compatible_true_damage']:.0f}** of **{siege['champion_building_damage']:.1f}** champion-attributed building damage",
            f"- Conditional tower time saved: **{conditional_time_saved_text}**",
            "",
        ]
    )
    resource_views = observed.get("resource_views") or {}
    if resource_views:
        lines.extend(
            [
                "## Actual resources: involved players only",
                "",
                *_render_resource_view(resource_views["selected"]),
                "",
                "## Actual resources: complete teams",
                "",
                *_render_resource_view(resource_views["full_teams"]),
                "",
                "The total-gold column already includes plate gold; the two gold-source columns add up to it.",
                "",
            ]
        )
    delayed_camps = observed.get("delayed_camps") or {}
    if delayed_camps.get("status") == "verified_later_same_game_clears":
        camp_resources = delayed_camps["later_camp_resources"]
        camp_difference = delayed_camps["camps_minus_grubs"]
        lines.extend(
            [
                "## Named farm-instead check",
                "",
                f"- The later same-game camp clears paid **{camp_resources['gold']:.0f}g / {camp_resources['xp']:.0f} XP**.",
                f"- Compared with the jungler's Grub-pit resources, that is **{camp_difference['gold']:.0f}g / {camp_difference['xp']:.0f} XP more** from the camps.",
                "- This was delayed farm, not permanently lost farm, and it is already reflected in the actual-resource tables.",
                "",
            ]
        )
    turret_health = observed.get("turret_health") or {}
    if turret_health.get("status") == "observer_estimate_available":
        observation = turret_health["observation"]
        no_touch = turret_health["fixed_state_remove_touch_only"]
        lines.extend(
            [
                "## Turret-health check",
                "",
                f"- Observer estimate at **{observation['game_time']}**: **{observation['health_estimate']:.0f} HP** ({observation['percent_estimate']:.1%}).",
                f"- Holding every other hit fixed and removing Touch only: at least **{no_touch['health_estimate']:.0f} HP** ({no_touch['health_percent_estimate']:.1%}).",
                f"- That HP crosses **{no_touch['plate_thresholds_reached']}** outer-plate thresholds; it does not imply only two plates.",
                "",
            ]
        )
    if counterfactual.get("status") != "unavailable":
        narrow = counterfactual["wave_only_selected_roles"]["total_delta"]
        complete = counterfactual["decision_complete_support_route"]["total_delta"]
        denied = counterfactual["defender_denial"].get("denied") or {}
        lines.extend(
            [
                "## Counterfactual audit",
                "",
                f"- Narrow wave comparison: the selected MKOI players received **{_more_or_less(narrow['gold'], suffix='g')} / {_more_or_less(narrow['xp'])} XP** than the farm-only recipe.",
                f"- From the declared decision start, counting the support's route: **{_more_or_less(complete['gold'], suffix='g')} / {_more_or_less(complete['xp'])} XP** than the farm-only recipe.",
                f"- Jax was denied **{denied.get('gold', 0):.0f}g / {denied.get('xp', 0):.0f} XP / {denied.get('cs', 0):.0f} CS**.",
                "",
                "These are selected-role ledgers, not a full-team causal total. The second version exposes off-lane support resources but still does not price allied-mid XP displacement or the drake buff.",
                "",
            ]
        )
    lines.extend(
        [
            "## Five-minute rule",
            "",
            "Use the exact series/game identity and declare the sequence, resource, and siege windows. Accept observed extraction automatically. Treat every counterfactual as a named recipe and withhold a single worth-it verdict whenever allied resource displacement or objective-buff value is not modeled.",
            "",
        ]
    )
    return "\n".join(lines)


def render_public_digest(report: Mapping[str, Any]) -> str:
    """Render one screenshot-sized table without mixing observed and estimated rows."""

    observed = report["observed"]
    grubs = observed["grubs"]["reward"]
    dragon = observed["dragon"]["reward"]
    siege = observed["siege"]
    conditional_time_saved = siege.get("conditional_time_saved_seconds")
    conditional_time_saved_text = (
        f"≈{conditional_time_saved:.1f}s faster"
        if isinstance(conditional_time_saved, (int, float))
        else "time unavailable"
    )
    counterfactual = report["counterfactual"]
    resource = observed["resource_views"]["full_teams"]
    comparison = resource["comparison"]
    difference = comparison["reference_minus_opponent"]
    totals = resource["team_totals"]
    reference = int(comparison["reference_team_id"])
    opponent = int(comparison["opponent_team_id"])
    reference_name = totals[reference]["team"]
    opponent_name = totals[opponent]["team"]
    camp_names = " and ".join(
        {"wolf": "Wolves", "gromp": "Gromp"}.get(
            str(value), str(value).title()
        )
        for value in (observed.get("delayed_camps") or {}).get(
            "requested_camps", []
        )
    ) or "named camps"
    objective_gold = _number(grubs.get("observed_pit_neutral_gold")) - _number(
        dragon.get("direct_killer_gold")
    )
    objective_xp = _number(grubs.get("event_adjacent_xp")) - _number(
        dragon.get("event_adjacent_xp")
    )
    denial = counterfactual.get("defender_denial", {}).get("denied") or {}
    named = counterfactual.get("named_farm_alternative_including_delayed_camps")
    if not isinstance(named, Mapping):
        farm_gold = farm_xp = "not calculated"
        combined_gold = combined_xp = "not calculated"
    else:
        less = named["received_less_than_named_farm"]
        combined = named["after_actual_plates_and_defender_denial"]
        farm_gold = f"{_display_number(less['gold'], suffix='g')} less"
        farm_xp = f"{_display_number(less['xp'])} less"
        combined_gold = _more_or_less(combined["gold"], suffix="g")
        combined_xp = _more_or_less(combined["xp"])
    lines = [
        f"### {reference_name}–{opponent_name} crossmap, {report['windows']['sequence']['requested_start'][:-4]}–{report['windows']['sequence']['requested_end'][:-4]}",
        "",
        "| Part of the play | Gold | XP | Plain meaning |",
        "|---|---:|---:|---|",
        f"| **Grubs vs Drake** | {reference_name} {_more_or_less(objective_gold, suffix='g')} | {reference_name} {_more_or_less(objective_xp)} | Immediate objective resources only; buffs are not priced. |",
        f"| **Named farm alternative** | {reference_name} **{farm_gold}** | {reference_name} **{farm_xp}** | Waves plus {camp_names}; selected {reference_name} route, not a team total. |",
        f"| **Top-lane result** | **{siege['plate_gold']:.0f}g** in plates; Jax denied **{denial.get('gold', 0):.0f}g** | Jax denied **{denial.get('xp', 0):.0f}** | Four plates and one denied regular wave. |",
        f"| **Selected-play comparison** | {reference_name} **{combined_gold}** | {reference_name} **{combined_xp}** | Named farm recipe after plates and Jax denial; still not a full-team result. |",
        f"| **Turret speed** | — | — | **{siege['touch_compatible_true_damage']:.0f} Touch damage; {conditional_time_saved_text}** under constant non-Touch champion DPS. |",
        f"| **Actual full-team result** | {reference_name} **{_more_or_less(difference['total_gold'], suffix='g')}** | {reference_name} **{_more_or_less(difference['xp'])}** | Exact scoreboard change across all ten players. |",
        "",
        "Actual totals and farm estimates are separate; direct minion/Voidmite turret damage and objective-buff value are unavailable.",
        "",
    ]
    return "\n".join(lines)


def _default_report_path(root: Path, series_id: str, game_index: int) -> Path:
    return root / "reports" / f"series_{series_id}_game_{game_index}.json"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--series")
    parser.add_argument("--game", type=int)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--provider-game-id")
    parser.add_argument("--grid-env-file", type=Path)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--sequence-start")
    parser.add_argument("--sequence-end")
    parser.add_argument("--resource-start")
    parser.add_argument("--resource-end")
    parser.add_argument("--siege-start")
    parser.add_argument("--siege-end")
    parser.add_argument("--lane")
    parser.add_argument("--cannon-gold", type=float)
    parser.add_argument(
        "--involved",
        help="comma-separated champion allowlist for the compact actual-resource table",
    )
    parser.add_argument(
        "--team-label",
        action="append",
        default=[],
        help="repeatable TEAM_ID=LABEL override for public tables",
    )
    parser.add_argument(
        "--delayed-camps",
        help="comma-separated analyst-supplied farm alternative, currently gromp/wolves",
    )
    parser.add_argument("--turret-observed-at")
    parser.add_argument("--turret-health", type=float)
    parser.add_argument("--turret-health-low", type=float)
    parser.add_argument("--turret-health-high", type=float)
    parser.add_argument("--turret-observer-source")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--public-summary", type=Path)
    parser.add_argument(
        "--save-request",
        type=Path,
        help="write a closed cached-source request after a successful long-form run",
    )
    args = parser.parse_args(argv)

    turret_fields = (
        args.turret_observed_at,
        args.turret_health,
        args.turret_health_low,
        args.turret_health_high,
        args.turret_observer_source,
    )
    request: dict[str, Any] | None = None
    if args.request is not None:
        conflicting = (
            args.series,
            args.game,
            args.source,
            args.provider_game_id,
            args.sequence_start,
            args.sequence_end,
            args.resource_start,
            args.resource_end,
            args.siege_start,
            args.siege_end,
            args.lane,
            args.cannon_gold,
            args.involved,
            args.delayed_camps,
            args.save_request,
            *args.team_label,
            *turret_fields,
        )
        if args.download or any(value is not None for value in conflicting):
            raise GridSequenceReviewError(
                "--request is a closed recipe; do not combine it with source or analysis flags"
            )
        request = load_review_request(args.request, root=args.root)
        identity = request["identity"]
        windows = request["windows"]
        series_id = str(identity["series_id"])
        game_index = int(identity["game_index"])
        provider_game_id = str(identity["provider_game_id"])
        receipt = adopt_local_source(
            source=Path(request["resolved_raw_path"]),
            series_id=series_id,
            game_index=game_index,
            provider_game_id=provider_game_id,
            root=args.root,
            catalog_path=args.catalog,
        )
        sequence_start = windows["sequence"]["start"]
        sequence_end = windows["sequence"]["end"]
        resource_start = windows["resource_wave"]["start"]
        resource_end = windows["resource_wave"]["end"]
        siege_start = windows["turret_siege"]["start"]
        siege_end = windows["turret_siege"]["end"]
        lane = str(request.get("lane") or "top")
        cannon_gold = request.get("cannon_gold_override")
        involved_champions = request.get("involved_champions")
        delayed_camps = request.get("delayed_camps")
        parsed_team_labels = {
            int(team_id): str(label)
            for team_id, label in (request.get("team_labels") or {}).items()
        }
        turret_observation = request.get("turret_observation")
    else:
        required = {
            "--series": args.series,
            "--game": args.game,
            "--sequence-start": args.sequence_start,
            "--sequence-end": args.sequence_end,
            "--resource-start": args.resource_start,
            "--resource-end": args.resource_end,
            "--siege-start": args.siege_start,
            "--siege-end": args.siege_end,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise GridSequenceReviewError(
                "missing required arguments: " + ", ".join(missing)
            )
        if args.download == (args.source is not None):
            raise GridSequenceReviewError(
                "choose exactly one source mode: --download or --source"
            )
        series_id = str(args.series)
        game_index = int(args.game)
        if args.download:
            receipt = retrieve_riot_game(
                series_id=series_id,
                game_index=game_index,
                root=args.root,
                key=_api_key(args.grid_env_file),
                catalog_path=args.catalog,
            )
        else:
            if not args.provider_game_id:
                raise GridSequenceReviewError(
                    "--provider-game-id is required with an already-local --source"
                )
            receipt = adopt_local_source(
                source=args.source,
                series_id=series_id,
                game_index=game_index,
                provider_game_id=args.provider_game_id,
                root=args.root,
                catalog_path=args.catalog,
            )
        sequence_start = args.sequence_start
        sequence_end = args.sequence_end
        resource_start = args.resource_start
        resource_end = args.resource_end
        siege_start = args.siege_start
        siege_end = args.siege_end
        lane = args.lane or "top"
        cannon_gold = args.cannon_gold
        involved_champions = (
            [value.strip() for value in args.involved.split(",") if value.strip()]
            if args.involved
            else None
        )
        delayed_camps = (
            [value.strip() for value in args.delayed_camps.split(",") if value.strip()]
            if args.delayed_camps
            else None
        )
        parsed_team_labels: dict[int, str] = {}
        for raw_label in args.team_label:
            if "=" not in raw_label:
                raise GridSequenceReviewError("--team-label must use TEAM_ID=LABEL")
            raw_team_id, label = raw_label.split("=", 1)
            if not raw_team_id.isdigit() or not label.strip():
                raise GridSequenceReviewError("--team-label must use TEAM_ID=LABEL")
            team_id = int(raw_team_id)
            if team_id in parsed_team_labels:
                raise GridSequenceReviewError(f"duplicate team label for {team_id}")
            parsed_team_labels[team_id] = label.strip()
        if any(value is not None for value in turret_fields) and (
            args.turret_observed_at is None or args.turret_health is None
        ):
            raise GridSequenceReviewError(
                "turret observation requires --turret-observed-at and --turret-health"
            )
        turret_observation = None
        if args.turret_observed_at is not None:
            turret_observation = {
                "game_time": args.turret_observed_at,
                "health_estimate": args.turret_health,
                "health_low": (
                    args.turret_health
                    if args.turret_health_low is None
                    else args.turret_health_low
                ),
                "health_high": (
                    args.turret_health
                    if args.turret_health_high is None
                    else args.turret_health_high
                ),
                "source": args.turret_observer_source,
                "method": "observer_health_bar",
            }

    source_path = Path(str(receipt["raw_path"]))
    report = analyze_sequence(
        source_path=source_path,
        receipt=receipt,
        sequence_start_ms=parse_clock(sequence_start),
        sequence_end_ms=parse_clock(sequence_end),
        resource_start_ms=parse_clock(resource_start),
        resource_end_ms=parse_clock(resource_end),
        siege_start_ms=parse_clock(siege_start),
        siege_end_ms=parse_clock(siege_end),
        lane=lane,
        cannon_gold_override=cannon_gold,
        involved_champions=involved_champions,
        team_labels=parsed_team_labels or None,
        delayed_camps=delayed_camps,
        turret_observation=turret_observation,
    )
    if request is not None:
        report["request"] = {
            "schema_version": request["schema_version"],
            "request_sha256": request["request_sha256"],
        }
        report["request_acceptance"] = verify_request_acceptance(
            report, request.get("expected_observed")
        )
        _finalize_report_hashes(report)
    saved_request = None
    if args.save_request is not None:
        saved_payload = build_review_request_from_report(report)
        _write_json(args.save_request, saved_payload)
        saved_request = {
            "path": str(args.save_request),
            "sha256": _sha256_file(args.save_request),
            "acceptance_checks": len(saved_payload["expected_observed"]),
        }
    output = args.output or _default_report_path(args.root, series_id, game_index)
    summary = args.summary or output.with_suffix(".md")
    public_summary = args.public_summary or output.with_suffix(".public.md")
    _write_json(output, report)
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(render_summary(report), encoding="utf-8")
    public_summary.parent.mkdir(parents=True, exist_ok=True)
    public_summary.write_text(render_public_digest(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(output),
                "summary": str(summary),
                "public_summary": str(public_summary),
                "analysis_sha256": report["analysis_sha256"],
                "report_sha256": report["report_sha256"],
                "runtime_seconds": report["runtime_seconds"],
                "request_acceptance": (report.get("request_acceptance") or {}).get(
                    "status"
                ),
                "saved_request": saved_request,
                "raw_sha256": report["source"]["raw_sha256"],
                "credentials_serialized": False,
                "signed_url_retained": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
