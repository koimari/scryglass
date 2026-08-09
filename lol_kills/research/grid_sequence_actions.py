#!/usr/bin/env python3
"""Atomic, hash-bound tools for one closed GRID sequence review.

Each analytical decision is a named action with declared dependencies,
parameters, a version, and deterministic input/output hashes.  The graph can
run to one requested action or through the complete report.  This keeps future
reviews on an executable path instead of asking an analyst or agent to
reconstruct the path from memory.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from lol_kills.etl.paths import WAREHOUSE_DIR
from lol_kills.research import grid_sequence_review as core


ACTION_GRAPH_SCHEMA = "scryglass:grid-sequence-action-graph:v1"
ACTION_ENVELOPE_SCHEMA = "scryglass:grid-sequence-action-result:v1"
ACTION_GRAPH_VERSION = "mkoifnc-sequence-v1"
DEFAULT_WIKI_DATABASE = WAREHOUSE_DIR.parent / "knowledge" / "league-wiki.sqlite3"


@dataclass(frozen=True)
class ActionSpec:
    name: str
    version: str
    description: str
    dependencies: tuple[str, ...]
    parameters: Callable[["ActionContext"], Mapping[str, Any]]
    execute: Callable[["ActionContext"], Any]


@dataclass
class ActionContext:
    source_path: Path
    receipt: Mapping[str, Any]
    sequence_start_ms: int
    sequence_end_ms: int
    resource_start_ms: int
    resource_end_ms: int
    siege_start_ms: int
    siege_end_ms: int
    lane: str
    cannon_gold_override: float | None
    involved_champions: tuple[str, ...]
    team_labels: dict[int, str]
    delayed_camps: tuple[str, ...]
    turret_observation: Mapping[str, Any] | None
    expected_observed: Mapping[str, Any] | None
    catalog_path: Path
    wiki_database: Path
    source_sha256: str
    game: core.GameData | None = None
    outputs: dict[str, Any] = field(default_factory=dict)
    receipts: list[dict[str, Any]] = field(default_factory=list)


def _json_clone(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def _normalize_delayed_camps(values: Sequence[str] | None) -> tuple[str, ...]:
    aliases = {
        "gromp": "gromp",
        "wolf": "wolf",
        "wolves": "wolf",
        "murkwolf": "wolf",
        "murkwolves": "wolf",
    }
    normalized: list[str] = []
    for raw in values or ():
        key = "".join(
            character for character in str(raw).casefold() if character.isalnum()
        )
        monster_type = aliases.get(key)
        if monster_type is None or monster_type in normalized:
            raise core.GridSequenceReviewError(
                f"delayed camp {raw!r} is unsupported or duplicated"
            )
        normalized.append(monster_type)
    return tuple(normalized)


def _normalize_turret_observation(
    observation: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if observation is None:
        return None
    try:
        estimate = float(observation["health_estimate"])
        normalized = {
            "game_time": core.format_clock(core.parse_clock(observation["game_time"])),
            "health_estimate": estimate,
            "health_low": float(observation.get("health_low", estimate)),
            "health_high": float(observation.get("health_high", estimate)),
            "method": str(observation.get("method") or "observer_health_bar"),
            "source": observation.get("source"),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise core.GridSequenceReviewError("turret observation is malformed") from exc
    return normalized


def _game(context: ActionContext) -> core.GameData:
    if context.game is None:
        raise core.GridSequenceReviewError("parse_game has not produced game state")
    return context.game


def _dependency_hashes(context: ActionContext, spec: ActionSpec) -> dict[str, str]:
    by_name = {row["action"]: row for row in context.receipts}
    missing = [name for name in spec.dependencies if name not in by_name]
    if missing:
        raise core.GridSequenceReviewError(
            f"action {spec.name} cannot run before: {', '.join(missing)}"
        )
    return {name: str(by_name[name]["output_sha256"]) for name in spec.dependencies}


def _hashable_output(action: str, output: Any) -> Any:
    if action != "assemble_report":
        return output
    report = _json_clone(output["report"])
    report["source"] = {
        "raw_sha256": report["source"]["raw_sha256"],
        "credentials_serialized": False,
        "signed_url_retained": False,
    }
    return {"report": report}


def _run_action(context: ActionContext, spec: ActionSpec) -> None:
    dependency_hashes = _dependency_hashes(context, spec)
    declared_parameters = dict(spec.parameters(context))
    action_input = {
        "action": spec.name,
        "version": spec.version,
        "dependencies": dependency_hashes,
        "parameters": declared_parameters,
    }
    output = spec.execute(context)
    # Reject accidental non-JSON state before it can enter a receipt.
    canonical_output = _json_clone(output)
    receipt = {
        "action": spec.name,
        "version": spec.version,
        "dependencies": list(spec.dependencies),
        "input_sha256": core._hash(action_input),
        "output_sha256": core._hash(_hashable_output(spec.name, canonical_output)),
    }
    # Keep the native in-process mapping types (notably integer team IDs) for
    # downstream actions. The cloned form exists only to validate and hash the
    # JSON contract; JSON serialization would otherwise coerce those keys.
    context.outputs[spec.name] = output
    context.receipts.append(receipt)


def _graph_receipt(receipts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    graph = {
        "schema_version": ACTION_GRAPH_SCHEMA,
        "graph_version": ACTION_GRAPH_VERSION,
        "actions": [dict(row) for row in receipts],
    }
    graph["graph_sha256"] = core._hash(graph)
    return graph


def _verify_source(context: ActionContext) -> dict[str, Any]:
    receipt_hash = str(context.receipt.get("raw_sha256") or "")
    if receipt_hash and receipt_hash != context.source_sha256:
        raise core.GridSequenceReviewError("receipt raw hash does not match source bytes")
    receipt_bytes = context.receipt.get("raw_bytes")
    actual_bytes = context.source_path.stat().st_size
    if receipt_bytes is not None and int(receipt_bytes) != actual_bytes:
        raise core.GridSequenceReviewError("receipt byte count does not match source bytes")
    forbidden_true = [
        field
        for field in ("credentials_serialized", "signed_url_retained", "mutations_used")
        if context.receipt.get(field) is True
    ]
    if forbidden_true:
        raise core.GridSequenceReviewError(
            "unsafe GRID receipt fields: " + ", ".join(forbidden_true)
        )
    return {
        "status": "verified",
        "raw_sha256": context.source_sha256,
        "raw_bytes": actual_bytes,
        "provider_series_id": context.receipt.get("provider_series_id"),
        "provider_game_id": context.receipt.get("provider_game_id"),
        "game_index": context.receipt.get("game_index"),
        "credentials_serialized": False,
        "signed_url_retained": False,
        "mutations_used": False,
    }


def _verify_catalog(context: ActionContext) -> dict[str, Any]:
    current = core.catalog_provenance(context.catalog_path)
    recorded = context.receipt.get("catalog")
    if isinstance(recorded, Mapping):
        for field in ("catalog_version", "catalog_sha256", "endpoint_schema_sha256"):
            if recorded.get(field) != current.get(field):
                raise core.GridSequenceReviewError(
                    f"GRID catalog receipt drift for {field}"
                )
    return {"status": "verified", **current}


def _parse_game(context: ActionContext) -> dict[str, Any]:
    game = core.load_game(context.source_path)
    context.game = game
    return {
        "source": {
            "raw_sha256": context.source_sha256,
            "size_bytes": context.source_path.stat().st_size,
        },
        "identity": game.identity,
        "roster": list(game.roster),
        "completeness": game.completeness,
    }


def _wiki_meta(connection: sqlite3.Connection) -> dict[str, Any]:
    keys = (
        "schema_version",
        "latest_jsonl_sha256",
        "snapshot_manifest_sha256",
        "snapshot_complete",
    )
    placeholders = ",".join("?" for _ in keys)
    rows = connection.execute(
        f"SELECT key, value FROM meta WHERE key IN ({placeholders})", keys
    ).fetchall()
    values = {str(key): json.loads(value) for key, value in rows}
    missing = [key for key in keys if key not in values]
    if missing:
        raise core.GridSequenceReviewError(
            "Wiki index provenance is incomplete: " + ", ".join(missing)
        )
    if values["schema_version"] != "scryglass:league-wiki-query-db:v1":
        raise core.GridSequenceReviewError("unsupported local Wiki index schema")
    if values["snapshot_complete"] is not True:
        raise core.GridSequenceReviewError("local Wiki snapshot is incomplete")
    return values


def _verify_mechanics(context: ActionContext) -> dict[str, Any]:
    mechanics = core._mechanics_profile(
        str(_game(context).identity.get("game_version") or "")
    )
    if mechanics["status"] != "verified_for_counterfactual":
        return mechanics
    database = context.wiki_database.resolve()
    if not database.is_file():
        raise core.GridSequenceReviewError(f"local Wiki index is missing: {database}")
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        meta = _wiki_meta(connection)
        verified_sources = []
        for expected in core.MECHANICS_SOURCES:
            row = connection.execute(
                """
                SELECT title, revision_id, revision_timestamp, source_url,
                       content_sha256, document_sha256
                FROM pages WHERE page_id = ?
                """,
                (int(expected["page_id"]),),
            ).fetchone()
            if row is None:
                raise core.GridSequenceReviewError(
                    f"Wiki mechanics page is missing: {expected['title']}"
                )
            title, revision_id, revision_timestamp, source_url, content_sha, document_sha = row
            mismatches = []
            for label, actual, wanted in (
                ("title", title, expected["title"]),
                ("revision_id", revision_id, expected["revision_id"]),
                (
                    "revision_timestamp",
                    revision_timestamp,
                    expected["revision_timestamp"],
                ),
                ("source_url", source_url, expected["source_url"]),
            ):
                if actual != wanted:
                    mismatches.append(f"{label}={actual!r}, expected {wanted!r}")
            if mismatches:
                raise core.GridSequenceReviewError(
                    f"Wiki receipt drift for {expected['title']}: " + "; ".join(mismatches)
                )
            verified_sources.append(
                {
                    **expected,
                    "content_sha256": content_sha,
                    "document_sha256": document_sha,
                }
            )
    finally:
        connection.close()
    return {
        **mechanics,
        "sources": verified_sources,
        "wiki_index": meta,
        "receipt_status": "verified",
    }


def _locate_objectives(context: ActionContext) -> dict[str, Any]:
    game = _game(context)
    sequence_events = [
        event
        for event in game.events
        if context.sequence_start_ms
        <= int(event["game_time_ms"])
        <= context.sequence_end_ms
    ]
    grubs = [
        event
        for event in sequence_events
        if event["schema"] == "epic_monster_kill"
        and str(event.get("monsterType") or "") == "VoidGrub"
    ]
    taking_teams = {int(event.get("killerTeamID") or 0) for event in grubs}
    if len(grubs) != 3 or len(taking_teams) != 1:
        raise core.GridSequenceReviewError(
            "declared sequence does not contain one three-Grub sweep"
        )
    dragon_events = [
        event
        for event in sequence_events
        if event["schema"] == "epic_monster_kill"
        and str(event.get("monsterType") or "") == "dragon"
    ]
    if len(dragon_events) != 1:
        raise core.GridSequenceReviewError(
            "declared sequence does not contain exactly one dragon event"
        )
    return {
        "taking_team_id": next(iter(taking_teams)),
        "sequence_events": sequence_events,
        "grub_events": grubs,
        "dragon_events": dragon_events,
        "timeline": [core._event_clock(event) for event in sequence_events],
    }


def _objective_resources(context: ActionContext) -> dict[str, Any]:
    game = _game(context)
    located = context.outputs["locate_objectives"]
    grubs = located["grub_events"]
    dragon_events = located["dragon_events"]
    taking_team_id = int(located["taking_team_id"])
    grub_reward = core._objective_reward(game, grubs)
    dragon_reward = core._objective_reward(game, dragon_events)
    first_before = core._frame_before(
        game.frames, min(int(event["game_time_ms"]) for event in grubs)
    )
    last_after = core._frame_after(
        game.frames, max(int(event["game_time_ms"]) for event in grubs)
    )
    taking_ids = {
        row["participant_id"]
        for row in game.roster
        if row["team_id"] == taking_team_id
    }
    pit_neutral_gold = sum(
        core._player_delta(first_before, last_after, participant_id)["neutral_gold"]
        for participant_id in taking_ids
    )
    grub_reward["observed_pit_neutral_gold"] = pit_neutral_gold
    grub_reward["incidental_neutral_gold_beyond_killer_gold"] = (
        pit_neutral_gold - grub_reward["direct_killer_gold"]
    )
    roster_by_id = {row["participant_id"]: row for row in game.roster}
    event_xp = grub_reward.get("xp_by_participant") or {}
    pit_resources: dict[str, dict[str, Any]] = {}
    for participant_id in sorted(taking_ids):
        champion = roster_by_id[participant_id]["champion"]
        delta = core._player_delta(first_before, last_after, participant_id)
        neutral_gold = core._number(delta["neutral_gold"])
        objective_xp = core._number(event_xp.get(champion))
        if neutral_gold <= 0 and objective_xp <= 0:
            continue
        pit_resources[champion] = {
            "neutral_gold": neutral_gold,
            "objective_xp": objective_xp,
            "neutral_cs": core._clean_count(core._number(delta["neutral_cs"])),
        }
    grub_reward["pit_resources_by_champion"] = pit_resources
    return {"grubs": grub_reward, "dragon": dragon_reward}


def _resource_ledgers(context: ActionContext) -> dict[str, Any]:
    game = _game(context)
    taking_team_id = int(context.outputs["locate_objectives"]["taking_team_id"])
    resource_window, resource_deltas = core._window_deltas(
        game, context.resource_start_ms, context.resource_end_ms
    )
    sequence_window, sequence_deltas = core._window_deltas(
        game, context.sequence_start_ms, context.sequence_end_ms
    )
    roster_by_id = {row["participant_id"]: row for row in game.roster}
    return {
        "resource_window": resource_window,
        "sequence_window": sequence_window,
        "resource_window_by_champion": {
            roster_by_id[participant_id]["champion"]: delta
            for participant_id, delta in sorted(resource_deltas.items())
        },
        "sequence_window_by_champion": {
            roster_by_id[participant_id]["champion"]: delta
            for participant_id, delta in sorted(sequence_deltas.items())
        },
        "resource_views": core.build_resource_views(
            game,
            start_ms=context.sequence_start_ms,
            end_ms=context.sequence_end_ms,
            reference_team_id=taking_team_id,
            involved_champions=context.involved_champions or None,
            team_labels=context.team_labels or None,
        ),
    }


def _siege_damage(context: ActionContext) -> dict[str, Any]:
    return core.analyze_siege(
        _game(context),
        start_ms=context.siege_start_ms,
        end_ms=context.siege_end_ms,
        attacking_team_id=int(
            context.outputs["locate_objectives"]["taking_team_id"]
        ),
        lane=context.lane,
    )


def _delayed_camps(context: ActionContext) -> dict[str, Any]:
    return core.analyze_delayed_camps(
        _game(context),
        sequence_end_ms=context.sequence_end_ms,
        taking_team_id=int(
            context.outputs["locate_objectives"]["taking_team_id"]
        ),
        requested_camps=context.delayed_camps or None,
        grub_reward=context.outputs["objective_resources"]["grubs"],
    )


def _turret_health(context: ActionContext) -> dict[str, Any]:
    return core.analyze_turret_health(
        context.outputs["siege_damage"], context.turret_observation
    )


def _wave_counterfactual(context: ActionContext) -> dict[str, Any]:
    mechanics = context.outputs["verify_mechanics"]
    if mechanics["status"] != "verified_for_counterfactual":
        return {"status": "unavailable", "blockers": mechanics["blockers"]}
    return core.build_crossmap_counterfactual(
        _game(context),
        sequence_start_ms=context.sequence_start_ms,
        resource_start_ms=context.resource_start_ms,
        resource_end_ms=context.resource_end_ms,
        taking_team_id=int(
            context.outputs["locate_objectives"]["taking_team_id"]
        ),
        grub_reward=context.outputs["objective_resources"]["grubs"],
        siege=context.outputs["siege_damage"],
        cannon_gold_override=context.cannon_gold_override,
    )


def _named_farm_comparison(context: ActionContext) -> dict[str, Any]:
    counterfactual = _json_clone(context.outputs["wave_counterfactual"])
    camps = context.outputs["delayed_camps"]
    siege = context.outputs["siege_damage"]
    if (
        counterfactual.get("status") != "unavailable"
        and camps.get("status") == "verified_later_same_game_clears"
    ):
        route = counterfactual["decision_complete_support_route"]["total_delta"]
        camp_difference = camps["camps_minus_grubs"]
        defender = counterfactual["defender_denial"].get("denied") or {}
        actual_minus_named_farm = {
            "gold": core._number(route.get("gold"))
            - core._number(camp_difference.get("gold")),
            "xp": core._number(route.get("xp"))
            - core._number(camp_difference.get("xp")),
        }
        counterfactual["named_farm_alternative_including_delayed_camps"] = {
            "actual_minus_named_farm": actual_minus_named_farm,
            "received_less_than_named_farm": {
                "gold": -actual_minus_named_farm["gold"],
                "xp": -actual_minus_named_farm["xp"],
            },
            "after_actual_plates_and_defender_denial": {
                "gold": actual_minus_named_farm["gold"]
                + core._number(siege.get("plate_gold"))
                + core._number(defender.get("gold")),
                "xp": actual_minus_named_farm["xp"]
                + core._number(defender.get("xp")),
            },
            "not_a_team_total": True,
            "recipe": (
                "decision-start wave allocation plus analyst-named later camps; "
                "then actual plates and the verified defender wave denial"
            ),
        }
    return counterfactual


def _assemble_report(context: ActionContext) -> dict[str, Any]:
    game = _game(context)
    located = context.outputs["locate_objectives"]
    objectives = context.outputs["objective_resources"]
    resources = context.outputs["resource_ledgers"]
    siege = context.outputs["siege_damage"]
    report = {
        "schema_version": core.SCHEMA_VERSION,
        "scope": "private_personal_research_only",
        "status": "complete",
        "source": {
            "raw_path": str(context.source_path),
            "raw_sha256": context.source_sha256,
            "retrieval_receipt_path": context.receipt.get("receipt_path"),
            "retrieval_receipt_sha256": context.receipt.get("receipt_sha256"),
            "credentials_serialized": False,
            "signed_url_retained": False,
        },
        "identity": {
            **game.identity,
            "provider_series_id": context.receipt.get("provider_series_id"),
            "provider_game_id": context.receipt.get("provider_game_id"),
            "game_index": context.receipt.get("game_index"),
        },
        "roster": list(game.roster),
        "completeness": game.completeness,
        "mechanics": context.outputs["verify_mechanics"],
        "windows": {
            "sequence": resources["sequence_window"],
            "resource": resources["resource_window"],
            "siege": siege["window"],
        },
        "observed": {
            "taking_team_id": located["taking_team_id"],
            "timeline": located["timeline"],
            "grubs": {
                "events": [core._event_clock(event) for event in located["grub_events"]],
                "reward": objectives["grubs"],
            },
            "dragon": {
                "events": [
                    core._event_clock(event) for event in located["dragon_events"]
                ],
                "reward": objectives["dragon"],
            },
            "resource_window_by_champion": resources[
                "resource_window_by_champion"
            ],
            "sequence_window_by_champion": resources[
                "sequence_window_by_champion"
            ],
            "resource_views": resources["resource_views"],
            "siege": siege,
            "delayed_camps": context.outputs["delayed_camps"],
            "turret_health": context.outputs["turret_health"],
        },
        "counterfactual": context.outputs["named_farm_comparison"],
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
    return {"report": report}


def _verify_expected_results(context: ActionContext) -> dict[str, Any]:
    if context.expected_observed is None:
        return {"status": "not_requested", "checks": 0}
    report = _json_clone(context.outputs["assemble_report"]["report"])
    report["action_graph"] = _graph_receipt(context.receipts)
    return core.verify_request_acceptance(report, context.expected_observed)


def _render_evidence(context: ActionContext) -> dict[str, Any]:
    content = core.render_summary(context.outputs["assemble_report"]["report"])
    return {"content": content, "content_sha256": core._hash(content)}


def _render_public(context: ActionContext) -> dict[str, Any]:
    content = core.render_public_digest(
        context.outputs["assemble_report"]["report"]
    )
    return {"content": content, "content_sha256": core._hash(content)}


def _static_parameters(**values: Any) -> Callable[[ActionContext], Mapping[str, Any]]:
    return lambda _context: values


ACTION_SPECS: tuple[ActionSpec, ...] = (
    ActionSpec(
        "verify_source",
        "1",
        "Verify raw bytes, provider identity fields and private read-only boundaries.",
        (),
        lambda c: {"raw_sha256": c.source_sha256},
        _verify_source,
    ),
    ActionSpec(
        "verify_catalog",
        "1",
        "Verify the GRID capability catalog and endpoint schema receipts.",
        ("verify_source",),
        _static_parameters(catalog_contract="query-grid-research/v1"),
        _verify_catalog,
    ),
    ActionSpec(
        "parse_game",
        "1",
        "Verify the raw hash and parse the completed ten-player LiveStats file.",
        ("verify_source", "verify_catalog"),
        lambda c: {
            "raw_sha256": c.source_sha256,
            "parser_schema": core.SCHEMA_VERSION,
        },
        _parse_game,
    ),
    ActionSpec(
        "verify_mechanics",
        "1",
        "Verify the patch profile and every pinned Wiki page receipt.",
        ("parse_game",),
        lambda c: {"game_version": _game(c).identity.get("game_version")},
        _verify_mechanics,
    ),
    ActionSpec(
        "locate_objectives",
        "1",
        "Select the declared window and require one dragon plus one three-Grub sweep.",
        ("parse_game",),
        lambda c: {
            "sequence_start_ms": c.sequence_start_ms,
            "sequence_end_ms": c.sequence_end_ms,
        },
        _locate_objectives,
    ),
    ActionSpec(
        "objective_resources",
        "1",
        "Calculate event-adjacent objective gold, XP, neutral CS and pit resources.",
        ("parse_game", "locate_objectives"),
        _static_parameters(frame_skew_ms=core.MAX_FRAME_SKEW_MS),
        _objective_resources,
    ),
    ActionSpec(
        "resource_ledgers",
        "1",
        "Calculate actual involved-player and complete-team resource tables.",
        ("parse_game", "locate_objectives"),
        lambda c: {
            "sequence_start_ms": c.sequence_start_ms,
            "sequence_end_ms": c.sequence_end_ms,
            "resource_start_ms": c.resource_start_ms,
            "resource_end_ms": c.resource_end_ms,
            "involved_champions": list(c.involved_champions),
            "team_labels": c.team_labels,
        },
        _resource_ledgers,
    ),
    ActionSpec(
        "siege_damage",
        "1",
        "Calculate plates, recipients, champion building damage and Touch timing.",
        ("parse_game", "locate_objectives"),
        lambda c: {
            "siege_start_ms": c.siege_start_ms,
            "siege_end_ms": c.siege_end_ms,
            "lane": c.lane,
        },
        _siege_damage,
    ),
    ActionSpec(
        "delayed_camps",
        "1",
        "Verify later same-game clears and exact rewards for analyst-named camps.",
        ("parse_game", "locate_objectives", "objective_resources"),
        lambda c: {
            "sequence_end_ms": c.sequence_end_ms,
            "requested_camps": list(c.delayed_camps),
            "lookahead_ms": core.CAMP_LOOKAHEAD_MS,
        },
        _delayed_camps,
    ),
    ActionSpec(
        "turret_health",
        "1",
        "Keep exact LiveStats health unavailable and apply only declared observer estimates.",
        ("siege_damage", "verify_mechanics"),
        lambda c: {"observation": c.turret_observation},
        _turret_health,
    ),
    ActionSpec(
        "wave_counterfactual",
        "1",
        "Apply the named wave allocation recipe with patch-pinned minion mechanics.",
        (
            "parse_game",
            "verify_mechanics",
            "locate_objectives",
            "objective_resources",
            "siege_damage",
        ),
        lambda c: {
            "sequence_start_ms": c.sequence_start_ms,
            "resource_start_ms": c.resource_start_ms,
            "resource_end_ms": c.resource_end_ms,
            "cannon_gold_override": c.cannon_gold_override,
        },
        _wave_counterfactual,
    ),
    ActionSpec(
        "named_farm_comparison",
        "1",
        "Combine the wave recipe, delayed camps, actual plates and defender denial once.",
        ("wave_counterfactual", "delayed_camps", "siege_damage"),
        _static_parameters(double_count_policy="actual tables are not adjusted"),
        _named_farm_comparison,
    ),
    ActionSpec(
        "assemble_report",
        "1",
        "Assemble observed and conditional sections without changing their estimands.",
        (
            "parse_game",
            "verify_mechanics",
            "locate_objectives",
            "objective_resources",
            "resource_ledgers",
            "siege_damage",
            "delayed_camps",
            "turret_health",
            "named_farm_comparison",
        ),
        _static_parameters(report_schema=core.SCHEMA_VERSION),
        _assemble_report,
    ),
    ActionSpec(
        "verify_expected_results",
        "1",
        "Compare every closed-request acceptance value and fail on any drift.",
        ("assemble_report",),
        lambda c: {
            "expected_sha256": (
                None
                if c.expected_observed is None
                else core._hash(c.expected_observed)
            )
        },
        _verify_expected_results,
    ),
    ActionSpec(
        "render_evidence",
        "1",
        "Render the detailed private evidence table from the assembled report.",
        ("assemble_report", "verify_expected_results"),
        _static_parameters(format="markdown_evidence"),
        _render_evidence,
    ),
    ActionSpec(
        "render_public",
        "1",
        "Render the screenshot-sized public table without exposing private rows.",
        ("assemble_report", "verify_expected_results"),
        _static_parameters(format="markdown_public_digest"),
        _render_public,
    ),
)

ACTION_BY_NAME = {spec.name: spec for spec in ACTION_SPECS}

ACTION_IO_CONTRACTS: dict[str, dict[str, tuple[str, ...]]] = {
    "verify_source": {
        "inputs": ("raw_sha256", "retrieval_receipt"),
        "outputs": ("status", "raw_sha256", "raw_bytes", "provider_identity"),
    },
    "verify_catalog": {
        "inputs": ("verify_source.output_sha256", "catalog_contract"),
        "outputs": ("status", "catalog_sha256", "endpoint_schema_sha256"),
    },
    "parse_game": {
        "inputs": ("raw_sha256", "parser_schema"),
        "outputs": ("source", "identity", "roster", "completeness"),
    },
    "verify_mechanics": {
        "inputs": ("game_version", "wiki_index"),
        "outputs": ("status", "profile_id", "sources", "wiki_index"),
    },
    "locate_objectives": {
        "inputs": ("sequence_start_ms", "sequence_end_ms"),
        "outputs": (
            "taking_team_id",
            "sequence_events",
            "grub_events",
            "dragon_events",
            "timeline",
        ),
    },
    "objective_resources": {
        "inputs": ("parse_game", "locate_objectives", "frame_skew_ms"),
        "outputs": ("grubs", "dragon"),
    },
    "resource_ledgers": {
        "inputs": (
            "sequence_window",
            "resource_window",
            "involved_champions",
            "team_labels",
        ),
        "outputs": (
            "resource_window",
            "sequence_window",
            "resource_window_by_champion",
            "sequence_window_by_champion",
            "resource_views",
        ),
    },
    "siege_damage": {
        "inputs": ("siege_window", "lane", "taking_team_id"),
        "outputs": (
            "plates",
            "plate_gold",
            "touch_compatible_true_damage",
            "champion_building_damage",
            "conditional_time_saved_seconds",
        ),
    },
    "delayed_camps": {
        "inputs": ("sequence_end_ms", "requested_camps", "lookahead_ms"),
        "outputs": ("status", "later_camp_resources", "camps_minus_grubs"),
    },
    "turret_health": {
        "inputs": ("siege_damage", "observer_estimate"),
        "outputs": ("status", "observation", "fixed_state_remove_touch_only"),
    },
    "wave_counterfactual": {
        "inputs": ("wave_windows", "cannon_gold_override", "mechanics_receipts"),
        "outputs": (
            "wave_only_selected_roles",
            "decision_complete_support_route",
            "defender_denial",
        ),
    },
    "named_farm_comparison": {
        "inputs": ("wave_counterfactual", "delayed_camps", "siege_damage"),
        "outputs": ("named_farm_alternative_including_delayed_camps",),
    },
    "assemble_report": {
        "inputs": ("all_analysis_action_hashes",),
        "outputs": ("report",),
    },
    "verify_expected_results": {
        "inputs": ("report", "expected_observed"),
        "outputs": ("status", "checks"),
    },
    "render_evidence": {
        "inputs": ("verified_report",),
        "outputs": ("content", "content_sha256"),
    },
    "render_public": {
        "inputs": ("verified_report",),
        "outputs": ("content", "content_sha256"),
    },
}


def action_contracts() -> list[dict[str, Any]]:
    return [
        {
            "order": index,
            "action": spec.name,
            "version": spec.version,
            "dependencies": list(spec.dependencies),
            "description": spec.description,
            "inputs": list(ACTION_IO_CONTRACTS[spec.name]["inputs"]),
            "outputs": list(ACTION_IO_CONTRACTS[spec.name]["outputs"]),
        }
        for index, spec in enumerate(ACTION_SPECS, start=1)
    ]


def run_analysis_action_graph(
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
    expected_observed: Mapping[str, Any] | None = None,
    catalog_path: Path = core.DEFAULT_CATALOG,
    wiki_database: Path = DEFAULT_WIKI_DATABASE,
    stop_after: str = "assemble_report",
) -> dict[str, Any]:
    if stop_after not in ACTION_BY_NAME:
        raise core.GridSequenceReviewError(f"unknown action: {stop_after}")
    started = time.perf_counter()
    source_path = source_path.resolve()
    context = ActionContext(
        source_path=source_path,
        receipt=dict(receipt),
        sequence_start_ms=sequence_start_ms,
        sequence_end_ms=sequence_end_ms,
        resource_start_ms=resource_start_ms,
        resource_end_ms=resource_end_ms,
        siege_start_ms=siege_start_ms,
        siege_end_ms=siege_end_ms,
        lane=lane,
        cannon_gold_override=cannon_gold_override,
        involved_champions=tuple(involved_champions or ()),
        team_labels={int(key): str(value) for key, value in (team_labels or {}).items()},
        delayed_camps=_normalize_delayed_camps(delayed_camps),
        turret_observation=_normalize_turret_observation(turret_observation),
        expected_observed=(
            None if expected_observed is None else dict(expected_observed)
        ),
        catalog_path=catalog_path,
        wiki_database=wiki_database,
        source_sha256=core._sha256_file(source_path),
    )
    for spec in ACTION_SPECS:
        _run_action(context, spec)
        if spec.name == stop_after:
            break
    graph = _graph_receipt(context.receipts)
    if stop_after != "assemble_report":
        return {
            "schema_version": ACTION_ENVELOPE_SCHEMA,
            "status": "complete",
            "requested_action": stop_after,
            "action": context.receipts[-1],
            "output": context.outputs[stop_after],
            "graph": graph,
        }
    report = context.outputs["assemble_report"]["report"]
    report["action_graph"] = graph
    report["runtime_seconds"] = time.perf_counter() - started
    core._finalize_report_hashes(report)
    return report


def _closed_request_inputs(
    request: Mapping[str, Any], *, root: Path, catalog: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = request["identity"]
    windows = request["windows"]
    receipt = core.adopt_local_source(
        source=Path(str(request["resolved_raw_path"])),
        series_id=str(identity["series_id"]),
        game_index=int(identity["game_index"]),
        provider_game_id=str(identity["provider_game_id"]),
        root=root,
        catalog_path=catalog,
    )
    arguments = {
        "source_path": Path(str(receipt["raw_path"])),
        "receipt": receipt,
        "sequence_start_ms": core.parse_clock(windows["sequence"]["start"]),
        "sequence_end_ms": core.parse_clock(windows["sequence"]["end"]),
        "resource_start_ms": core.parse_clock(windows["resource_wave"]["start"]),
        "resource_end_ms": core.parse_clock(windows["resource_wave"]["end"]),
        "siege_start_ms": core.parse_clock(windows["turret_siege"]["start"]),
        "siege_end_ms": core.parse_clock(windows["turret_siege"]["end"]),
        "lane": str(request.get("lane") or "top"),
        "cannon_gold_override": request.get("cannon_gold_override"),
        "involved_champions": request.get("involved_champions"),
        "team_labels": {
            int(key): str(value)
            for key, value in (request.get("team_labels") or {}).items()
        },
        "delayed_camps": request.get("delayed_camps"),
        "turret_observation": request.get("turret_observation"),
        "expected_observed": request.get("expected_observed"),
    }
    return arguments, receipt


def _write_json(path: Path, value: Any) -> None:
    core._write_json(path, value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="print the ordered action contracts")
    run = subparsers.add_parser("run", help="run through one named action")
    run.add_argument("--request", type=Path, required=True)
    run.add_argument("--action", choices=tuple(ACTION_BY_NAME), required=True)
    run.add_argument("--root", type=Path, default=core.DEFAULT_ROOT)
    run.add_argument("--catalog", type=Path, default=core.DEFAULT_CATALOG)
    run.add_argument("--wiki-database", type=Path, default=DEFAULT_WIKI_DATABASE)
    run.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "list":
        print(json.dumps({"actions": action_contracts()}, indent=2, sort_keys=True))
        return 0
    request = core.load_review_request(args.request, root=args.root)
    arguments, _receipt = _closed_request_inputs(
        request, root=args.root, catalog=args.catalog
    )
    result = run_analysis_action_graph(
        **arguments,
        catalog_path=args.catalog,
        wiki_database=args.wiki_database,
        stop_after=args.action,
    )
    if args.action == "assemble_report":
        result["request"] = {
            "schema_version": request["schema_version"],
            "request_sha256": request["request_sha256"],
        }
        result["request_acceptance"] = core.verify_request_acceptance(
            result, request.get("expected_observed")
        )
        core._finalize_report_hashes(result)
        envelope = {
            "schema_version": ACTION_ENVELOPE_SCHEMA,
            "status": "complete",
            "requested_action": args.action,
            "action": result["action_graph"]["actions"][-1],
            "analysis_sha256": result["analysis_sha256"],
            "request_acceptance": result["request_acceptance"],
            "output": result,
            "graph": result["action_graph"],
        }
    else:
        envelope = result
    if args.output is not None:
        _write_json(args.output, envelope)
        print(
            json.dumps(
                {
                    "status": "complete",
                    "action": args.action,
                    "output": str(args.output),
                    "output_sha256": core._sha256_file(args.output),
                    "action_output_sha256": envelope["action"]["output_sha256"],
                },
                sort_keys=True,
            )
        )
    else:
        print(json.dumps(envelope, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
