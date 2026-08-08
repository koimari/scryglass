"""Run the internal mechanics-first composite against a frozen map ledger.

This runner is deliberately conservative: the current Leaguepedia batch now
has independently captured patch packets and partial pre-event roster
receipts, plus a result-free retrospective patch crosswalk, but the frozen
rows still do not carry a pre-event-authorized game-to-patch binding and 387
lineups remain unavailable. Those dependencies keep rows closed until their
exact authority is attached.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from lol_kills.knowledge.league_wiki_vault import validate as validate_wiki_vault
from lol_kills.knowledge.patch_authority import load_patch_packet
from lol_kills.research.mechanics_composite import (
    CompositePrediction,
    InteractionLedger,
    LineupSnapshot,
    compose_prediction,
    evaluate_winner_gate,
)


DEFAULT_RUN = Path("data/lol/v2/experiments/leaguepedia/manual-run-2026-07-31")
DEFAULT_OUTPUT = DEFAULT_RUN / "mechanics-engine-v1"
DEFAULT_VAULT = Path("data/lol/knowledge/obsidian/league-wiki")
DEFAULT_PATCH_MATRIX = Path("data/lol/knowledge/patch-packets/wiki/2026/matrix-manifest.json")
DEFAULT_CLIENT_PATCH_MATRIX = Path("data/lol/knowledge/patch-packets/cdragon/2026/matrix-manifest.json")
DEFAULT_ROSTER_RECEIPT_MANIFEST = DEFAULT_RUN / "roster-receipts-v1" / "receipt-manifest.json"
DEFAULT_GRID_PATCH_RECEIPT_MANIFEST = DEFAULT_RUN / "grid-patch-receipts-v1" / "receipt-manifest.json"
DEFAULT_LEAGUEPEDIA_PATCH_RECEIPT_MANIFEST = DEFAULT_RUN / "leaguepedia-patch-receipts-v1" / "receipt-manifest.json"
DEFAULT_LEAGUEPEDIA_PATCH_REVISION_MANIFEST = DEFAULT_RUN / "leaguepedia-patch-revisions-v1" / "receipt-manifest.json"
GRID_OUTCOME_FIELDS = frozenset({"winner_team_id", "complete", "won"})
LEAGUEPEDIA_OUTCOME_FIELDS = frozenset(
    {
        "WinTeam",
        "LossTeam",
        "Team1Kills",
        "Team2Kills",
        "Gamelength_Number",
        "Winner",
        "Team1Score",
        "Team2Score",
    }
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _canonical_hash(value: Mapping[str, Any]) -> str:
    unsigned = dict(value)
    unsigned.pop("manifest_sha256", None)
    return hashlib.sha256(
        json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _resolve_artifact_path(manifest_path: Path, value: Any) -> Path:
    candidate = Path(str(value))
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return manifest_path.parent / candidate


def _outcomes(run_dir: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in _load_jsonl(run_dir / "normalized-outcome-rows.jsonl"):
        game_id = str(row.get("GameId") or "")
        team1 = str(row.get("Team1") or "")
        team2 = str(row.get("Team2") or "")
        winner = str(row.get("WinTeam") or "")
        if not game_id or winner not in {team1, team2}:
            continue
        result[game_id] = "blue" if winner == team1 else "red"
    return result


def _lineup_from_receipt(receipt: Mapping[str, Any]) -> LineupSnapshot:
    role_map = {"top": "top", "jungle": "jng", "mid": "mid", "bot": "bot", "support": "sup"}
    players: list[tuple[str, str]] = []
    raw_players = receipt.get("players", [])
    if isinstance(raw_players, list):
        for row in raw_players:
            if not isinstance(row, Mapping):
                continue
            role = role_map.get(str(row.get("role", "")).casefold())
            player = str(row.get("player", "")).strip()
            if role and player:
                players.append((role, player))
    return LineupSnapshot(
        fixture_id=str(receipt.get("fixture_id", "")),
        as_of=str(receipt.get("as_of", "")),
        event_start=str(receipt.get("event_start", "")),
        team_id=str(receipt.get("team", "")),
        players=tuple(players),
        evidence_hash=receipt.get("evidence_hash"),
        authority_status=str(receipt.get("authority_status", "unavailable")),
        blockers=tuple(str(item) for item in receipt.get("blockers", []) if str(item)),
    )


def _prediction(
    row: Mapping[str, Any],
    roster_receipts: Mapping[str, Mapping[str, Any]] | None = None,
    roster_manifest_sha256: str | None = None,
    patch_receipts: Mapping[str, Mapping[str, Any]] | None = None,
    patch_manifest_sha256: str | None = None,
    bound_patch_labels: set[str] | None = None,
) -> CompositePrediction:
    pregame = row.get("pregame")
    if not isinstance(pregame, Mapping):
        raise ValueError("frozen row has no pregame object")
    fixture_id = str(pregame.get("fixture_id") or "")
    cutoff = str(pregame.get("as_of") or "")
    ledger = InteractionLedger(
        terms=(),
        total_edge=None,
        blockers=("interaction_evidence_unavailable",),
    )
    receipt = (roster_receipts or {}).get(fixture_id)
    lineups: list[LineupSnapshot] = []
    roster_blockers: list[str] = []
    if isinstance(receipt, Mapping):
        teams = receipt.get("teams", {})
        if isinstance(teams, Mapping):
            for side in ("blue", "red"):
                team_receipt = teams.get(side)
                if isinstance(team_receipt, Mapping):
                    lineups.append(_lineup_from_receipt(team_receipt))
        if receipt.get("authority_status") != "confirmed":
            roster_blockers.append("pre_event_roster_receipt_unavailable")
    else:
        roster_blockers.append("pre_event_roster_receipt_missing")

    patch_receipt = (patch_receipts or {}).get(fixture_id)
    patch_authorized = isinstance(patch_receipt, Mapping) and (
        patch_receipt.get("authority_status") == "pre_event_revision"
        and bool(patch_receipt.get("pregame_authorized"))
    )
    patch_label = str(patch_receipt.get("patch") or "") if patch_authorized else ""
    blockers: list[str] = []
    if not patch_authorized:
        blockers.extend(("patch_identity_missing_from_frozen_row", "patch_packet_not_bound"))
    elif bound_patch_labels is None or patch_label not in bound_patch_labels:
        blockers.append("patch_packet_not_bound")
    blockers.append("player_context_not_authorized")
    blockers.extend(roster_blockers)
    input_manifest = {
        "fixture_id": fixture_id,
        "pregame_sha256": row.get("pregame_sha256"),
        "authority_mode": "strict_pre_event",
        "roster_receipt_manifest_sha256": roster_manifest_sha256,
        "roster_receipt_evidence_hash": receipt.get("evidence_hash") if isinstance(receipt, Mapping) else None,
        "patch_receipt_manifest_sha256": patch_manifest_sha256,
        "patch_receipt_evidence_hash": patch_receipt.get("evidence_hash") if patch_authorized else None,
        "patch": patch_label or None,
        "client_patch": patch_receipt.get("client_patch") if patch_authorized else None,
    }
    prediction = compose_prediction(
        fixture_id=fixture_id,
        pre_event_cutoff=cutoff,
        mechanics_score=None,
        player_context_score=None,
        ledger=ledger,
        lineups=lineups,
        input_manifest=input_manifest,
    )
    return CompositePrediction(
        fixture_id=prediction.fixture_id,
        pre_event_cutoff=prediction.pre_event_cutoff,
        predicted_winner=None,
        p_blue=None,
        mechanics_score=None,
        player_context_score=None,
        synergy_counter_ledger=prediction.synergy_counter_ledger,
        availability="unavailable",
        blockers=tuple(sorted(set(prediction.blockers + tuple(blockers)))),
        model_version=prediction.model_version,
        input_manifest_hash=prediction.input_manifest_hash,
    )


def _client_patch_readiness(matrix_path: Path) -> dict[str, Any]:
    blockers: list[str] = []
    status_counts: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    captured_packet_count = 0
    authority_packet_count = 0
    executable_cell_count = 0
    if not matrix_path.exists():
        return {
            "status": "missing",
            "path": str(matrix_path),
            "patch_count": 0,
            "exact_source_patch_count": 0,
            "captured_patch_count": 0,
            "status_counts": {},
            "patches": [],
            "manifest_sha256": None,
            "blockers": ["client_patch_matrix_missing"],
        }
    try:
        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
        if not isinstance(matrix, dict):
            raise ValueError("client patch matrix is not an object")
        expected_hash = matrix.get("manifest_sha256")
        actual_hash = _canonical_hash(matrix)
        if expected_hash != actual_hash:
            blockers.append("client_patch_matrix_hash_mismatch")
        raw_rows = matrix.get("patches", [])
        if not isinstance(raw_rows, list):
            raise ValueError("client patch matrix patches is not a list")
        for raw in raw_rows:
            if not isinstance(raw, Mapping):
                blockers.append("client_patch_matrix_row_invalid")
                continue
            patch = str(raw.get("patch", ""))
            status = str(raw.get("status", "missing"))
            status_counts[status] = status_counts.get(status, 0) + 1
            row = {
                "patch": patch,
                "status": status,
                "exact_patch_source": bool(raw.get("exact_patch_source")),
                "row_sha256": raw.get("row_sha256"),
                "client_patch": raw.get("client_patch"),
                "probe_urls": {
                    key: value.get("url")
                    for key, value in (raw.get("probes", {}) or {}).items()
                    if isinstance(value, Mapping) and value.get("url")
                },
            }
            if status in {"captured", "captured_authority_blocked"}:
                captured_packet_count += 1
                packet_ref = raw.get("authority_packet_path")
                if not packet_ref:
                    blockers.append(f"client_authority_packet_missing:{patch}")
                else:
                    packet_path = _resolve_artifact_path(matrix_path, packet_ref)
                    if not packet_path.exists():
                        blockers.append(f"client_authority_packet_missing:{patch}")
                    else:
                        try:
                            packet = load_patch_packet(packet_path)
                            packet_payload = json.loads(packet_path.read_text(encoding="utf-8"))
                            packet_file_hash = packet_payload.get("packet_sha256") if isinstance(packet_payload, Mapping) else None
                            authority_packet_count += 1
                            executable_cell_count += len(packet.executable_cells)
                            if packet_file_hash != raw.get("authority_packet_sha256"):
                                blockers.append(f"client_authority_packet_hash_mismatch:{patch}")
                            row["authority_packet_sha256"] = packet_file_hash
                            row["executable_cell_count"] = len(packet.executable_cells)
                        except Exception as exc:
                            blockers.append(f"client_authority_packet_invalid:{patch}:{type(exc).__name__}")
            rows.append(row)
    except Exception as exc:
        blockers.append(f"client_patch_matrix_validation_failed:{type(exc).__name__}")
        matrix = {}
        expected_hash = None
    exact_count = sum(bool(row.get("exact_patch_source")) for row in rows)
    captured_count = sum(row.get("status") == "captured" for row in rows)
    if not exact_count:
        blockers.append("client_patch_source_unavailable")
    elif captured_packet_count == 0:
        blockers.append("client_patch_packets_not_captured")
    elif captured_packet_count < exact_count:
        blockers.append("client_patch_packets_incomplete")
    elif captured_packet_count and authority_packet_count < captured_packet_count:
        blockers.append("client_authority_packets_incomplete")
    status = "exact_available" if exact_count and captured_packet_count == exact_count and authority_packet_count == captured_packet_count and not blockers else "blocked"
    return {
        "status": status,
        "path": str(matrix_path),
        "manifest_sha256": expected_hash,
        "patch_count": len(rows),
        "exact_source_patch_count": exact_count,
        "captured_patch_count": captured_count,
        "captured_packet_count": captured_packet_count,
        "authority_packet_count": authority_packet_count,
        "executable_cell_count": executable_cell_count,
        "status_counts": dict(sorted(status_counts.items())),
        "patches": rows,
        "blockers": sorted(set(blockers)),
    }


def _load_roster_receipts(manifest_path: Path) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    if not manifest_path.exists():
        return (
            {
                "status": "missing",
                "path": str(manifest_path),
                "fixture_count": 0,
                "confirmed_fixture_count": 0,
                "unavailable_fixture_count": 0,
                "manifest_sha256": None,
                "receipt_file_sha256": None,
                "blockers": ["roster_receipt_manifest_missing"],
            },
            {},
        )
    blockers: list[str] = []
    index: dict[str, Mapping[str, Any]] = {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("roster receipt manifest is not an object")
        expected_manifest_hash = manifest.get("manifest_sha256")
        if expected_manifest_hash != _canonical_hash(manifest):
            blockers.append("roster_receipt_manifest_hash_mismatch")
        receipt_path = _resolve_artifact_path(manifest_path, manifest.get("receipt_file", ""))
        if not receipt_path.exists():
            blockers.append("roster_receipt_file_missing")
        else:
            raw_receipts = receipt_path.read_bytes()
            expected_file_hash = manifest.get("receipt_file_sha256")
            actual_file_hash = hashlib.sha256(raw_receipts).hexdigest()
            if expected_file_hash != actual_file_hash:
                blockers.append("roster_receipt_file_hash_mismatch")
            for line in raw_receipts.decode("utf-8").splitlines():
                if not line.strip():
                    continue
                receipt = json.loads(line)
                if not isinstance(receipt, Mapping) or not receipt.get("fixture_id"):
                    blockers.append("roster_receipt_row_invalid")
                    continue
                index[str(receipt["fixture_id"])] = receipt
        fixture_count = int(manifest.get("fixture_count", -1))
        confirmed_count = int(manifest.get("confirmed_fixture_count", -1))
        if fixture_count != len(index):
            blockers.append("roster_receipt_fixture_count_mismatch")
        observed_confirmed = sum(row.get("authority_status") == "confirmed" for row in index.values())
        if confirmed_count != observed_confirmed:
            blockers.append("roster_receipt_confirmed_count_mismatch")
        status = "complete" if fixture_count >= 0 and confirmed_count == fixture_count and not blockers else "partial" if confirmed_count > 0 and not blockers else "blocked"
        readiness = {
            "status": status,
            "path": str(manifest_path),
            "manifest_sha256": expected_manifest_hash,
            "receipt_file": str(receipt_path),
            "receipt_file_sha256": manifest.get("receipt_file_sha256"),
            "team_count": int(manifest.get("team_count", 0)),
            "fixture_count": fixture_count,
            "confirmed_fixture_count": confirmed_count,
            "unavailable_fixture_count": int(manifest.get("unavailable_fixture_count", 0)),
            "blockers": sorted(set(blockers)),
        }
        return readiness, index
    except Exception as exc:
        return (
            {
                "status": "blocked",
                "path": str(manifest_path),
                "fixture_count": 0,
                "confirmed_fixture_count": 0,
                "unavailable_fixture_count": 0,
                "manifest_sha256": None,
                "receipt_file_sha256": None,
                "blockers": sorted(set(blockers + [f"roster_receipt_validation_failed:{type(exc).__name__}" ])),
            },
            {},
        )


def _load_grid_patch_receipts(manifest_path: Path) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    """Load the retrospective GRID crosswalk without making it predictive input."""

    if not manifest_path.exists():
        return (
            {
                "status": "missing",
                "path": str(manifest_path),
                "fixture_count": 0,
                "exact_identity_fixture_count": 0,
                "pregame_authorized_fixture_count": 0,
                "unavailable_fixture_count": 0,
                "manifest_sha256": None,
                "receipt_file_sha256": None,
                "blockers": ["grid_patch_receipt_manifest_missing"],
            },
            {},
        )
    blockers: list[str] = []
    index: dict[str, Mapping[str, Any]] = {}
    blocker_counts: dict[str, int] = {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("GRID patch receipt manifest is not an object")
        expected_manifest_hash = manifest.get("manifest_sha256")
        if expected_manifest_hash != _canonical_hash(manifest):
            blockers.append("grid_patch_receipt_manifest_hash_mismatch")
        receipt_path = _resolve_artifact_path(manifest_path, manifest.get("receipt_file", ""))
        if not receipt_path.exists():
            blockers.append("grid_patch_receipt_file_missing")
        else:
            receipt_bytes = receipt_path.read_bytes()
            expected_file_hash = manifest.get("receipt_file_sha256")
            actual_file_hash = hashlib.sha256(receipt_bytes).hexdigest()
            if expected_file_hash != actual_file_hash:
                blockers.append("grid_patch_receipt_file_hash_mismatch")
            for line in receipt_bytes.decode("utf-8").splitlines():
                if not line.strip():
                    continue
                receipt = json.loads(line)
                if not isinstance(receipt, Mapping) or not receipt.get("fixture_id"):
                    blockers.append("grid_patch_receipt_row_invalid")
                    continue
                if GRID_OUTCOME_FIELDS.intersection(receipt):
                    blockers.append("grid_patch_receipt_outcome_field_emitted")
                evidence = receipt.get("evidence")
                if isinstance(evidence, Mapping) and GRID_OUTCOME_FIELDS.intersection(evidence):
                    blockers.append("grid_patch_receipt_outcome_field_emitted")
                index[str(receipt["fixture_id"])] = receipt
                for item in receipt.get("blockers", []):
                    blocker = str(item)
                    if blocker:
                        blocker_counts[blocker] = blocker_counts.get(blocker, 0) + 1
        fixture_count = int(manifest.get("fixture_count", -1))
        if fixture_count != len(index):
            blockers.append("grid_patch_receipt_fixture_count_mismatch")
        exact_count = sum(row.get("authority_status") == "confirmed_metadata" for row in index.values())
        authorized_count = sum(bool(row.get("pregame_authorized")) for row in index.values())
        manifest_exact = int(manifest.get("confirmed_metadata_fixture_count", -1))
        manifest_authorized = int(manifest.get("pregame_authorized_fixture_count", -1))
        if manifest_exact != exact_count:
            blockers.append("grid_patch_receipt_exact_count_mismatch")
        if manifest_authorized != authorized_count:
            blockers.append("grid_patch_receipt_authorized_count_mismatch")
        if exact_count and authorized_count < exact_count:
            blockers.append("grid_patch_source_captured_after_cutoff")
        if authorized_count == fixture_count and fixture_count >= 0 and not blockers:
            status = "pregame_authorized"
        elif exact_count and not any(item.endswith("_missing") or "hash_mismatch" in item for item in blockers):
            status = "retrospective_only"
        else:
            status = "blocked"
        readiness = {
            "status": status,
            "path": str(manifest_path),
            "manifest_sha256": expected_manifest_hash,
            "receipt_file": str(receipt_path),
            "receipt_file_sha256": manifest.get("receipt_file_sha256"),
            "source_captured_at": manifest.get("source_captured_at"),
            "fixture_count": fixture_count,
            "exact_identity_fixture_count": exact_count,
            "pregame_authorized_fixture_count": authorized_count,
            "unavailable_fixture_count": int(manifest.get("unavailable_fixture_count", 0)),
            "outcome_fields_emitted": bool(manifest.get("outcome_fields_emitted", True)),
            "blocker_counts": dict(sorted(blocker_counts.items())),
            "blockers": sorted(set(blockers)),
        }
        return readiness, index
    except Exception as exc:
        return (
            {
                "status": "blocked",
                "path": str(manifest_path),
                "fixture_count": 0,
                "exact_identity_fixture_count": 0,
                "pregame_authorized_fixture_count": 0,
                "unavailable_fixture_count": 0,
                "manifest_sha256": None,
                "receipt_file_sha256": None,
                "blocker_counts": {},
                "blockers": sorted(set(blockers + [f"grid_patch_receipt_validation_failed:{type(exc).__name__}" ])),
            },
            {},
        )


def _load_leaguepedia_patch_receipts(
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    """Load result-free Leaguepedia patch rows and retain their time status."""

    if not manifest_path.exists():
        return (
            {
                "status": "missing",
                "path": str(manifest_path),
                "fixture_count": 0,
                "exact_patch_fixture_count": 0,
                "pregame_authorized_fixture_count": 0,
                "unavailable_fixture_count": 0,
                "manifest_sha256": None,
                "receipt_file_sha256": None,
                "blockers": ["leaguepedia_patch_receipt_manifest_missing"],
            },
            {},
        )
    blockers: list[str] = []
    index: dict[str, Mapping[str, Any]] = {}
    blocker_counts: dict[str, int] = {}
    patch_counts: dict[str, int] = {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("Leaguepedia patch receipt manifest is not an object")
        expected_manifest_hash = manifest.get("manifest_sha256")
        if expected_manifest_hash != _canonical_hash(manifest):
            blockers.append("leaguepedia_patch_receipt_manifest_hash_mismatch")
        receipt_path = _resolve_artifact_path(manifest_path, manifest.get("receipt_file", ""))
        if not receipt_path.exists():
            blockers.append("leaguepedia_patch_receipt_file_missing")
        else:
            receipt_bytes = receipt_path.read_bytes()
            expected_file_hash = manifest.get("receipt_file_sha256")
            if expected_file_hash != hashlib.sha256(receipt_bytes).hexdigest():
                blockers.append("leaguepedia_patch_receipt_file_hash_mismatch")
            for line in receipt_bytes.decode("utf-8").splitlines():
                if not line.strip():
                    continue
                receipt = json.loads(line)
                if not isinstance(receipt, Mapping) or not receipt.get("fixture_id"):
                    blockers.append("leaguepedia_patch_receipt_row_invalid")
                    continue
                if LEAGUEPEDIA_OUTCOME_FIELDS.intersection(receipt):
                    blockers.append("leaguepedia_patch_receipt_outcome_field_emitted")
                evidence = receipt.get("evidence")
                if isinstance(evidence, Mapping) and LEAGUEPEDIA_OUTCOME_FIELDS.intersection(evidence):
                    blockers.append("leaguepedia_patch_receipt_outcome_field_emitted")
                fixture_id = str(receipt["fixture_id"])
                index[fixture_id] = receipt
                patch = str(receipt.get("patch") or "")
                if patch:
                    patch_counts[patch] = patch_counts.get(patch, 0) + 1
                for item in receipt.get("blockers", []):
                    blocker = str(item)
                    if blocker:
                        blocker_counts[blocker] = blocker_counts.get(blocker, 0) + 1
        fixture_count = int(manifest.get("fixture_count", -1))
        if fixture_count != len(index):
            blockers.append("leaguepedia_patch_receipt_fixture_count_mismatch")
        exact_count = sum(row.get("authority_status") == "confirmed_metadata" for row in index.values())
        authorized_count = sum(bool(row.get("pregame_authorized")) for row in index.values())
        if exact_count != int(manifest.get("confirmed_metadata_fixture_count", -1)):
            blockers.append("leaguepedia_patch_receipt_exact_count_mismatch")
        if authorized_count != int(manifest.get("pregame_authorized_fixture_count", -1)):
            blockers.append("leaguepedia_patch_receipt_authorized_count_mismatch")
        if exact_count and authorized_count < exact_count:
            blockers.append("leaguepedia_source_captured_after_cutoff")
        if authorized_count == fixture_count and fixture_count >= 0 and not blockers:
            status = "pregame_authorized"
        elif exact_count and not any("missing" in item or "hash_mismatch" in item for item in blockers):
            status = "retrospective_only"
        else:
            status = "blocked"
        return (
            {
                "status": status,
                "path": str(manifest_path),
                "manifest_sha256": expected_manifest_hash,
                "receipt_file": str(receipt_path),
                "receipt_file_sha256": manifest.get("receipt_file_sha256"),
                "source_captured_at": manifest.get("captured_at"),
                "fixture_count": fixture_count,
                "exact_patch_fixture_count": exact_count,
                "pregame_authorized_fixture_count": authorized_count,
                "unavailable_fixture_count": int(manifest.get("unavailable_fixture_count", 0)),
                "patch_counts": dict(sorted(patch_counts.items())),
                "outcome_fields_requested": manifest.get("outcome_fields_requested", []),
                "outcome_fields_emitted": bool(manifest.get("outcome_fields_emitted", True)),
                "blocker_counts": dict(sorted(blocker_counts.items())),
                "blockers": sorted(set(blockers)),
            },
            index,
        )
    except Exception as exc:
        return (
            {
                "status": "blocked",
                "path": str(manifest_path),
                "fixture_count": 0,
                "exact_patch_fixture_count": 0,
                "pregame_authorized_fixture_count": 0,
                "unavailable_fixture_count": 0,
                "manifest_sha256": None,
                "receipt_file_sha256": None,
                "blocker_counts": {},
                "blockers": sorted(
                    set(blockers + [f"leaguepedia_patch_receipt_validation_failed:{type(exc).__name__}"])
                ),
            },
            {},
        )


def _load_leaguepedia_patch_revision_receipts(
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    """Load revision-backed patch authority, including partial strict coverage."""

    if not manifest_path.exists():
        return (
            {
                "status": "missing",
                "path": str(manifest_path),
                "fixture_count": 0,
                "exact_patch_fixture_count": 0,
                "pregame_authorized_fixture_count": 0,
                "unavailable_fixture_count": 0,
                "manifest_sha256": None,
                "receipt_file_sha256": None,
                "blockers": ["leaguepedia_patch_revision_manifest_missing"],
            },
            {},
        )
    blockers: list[str] = []
    index: dict[str, Mapping[str, Any]] = {}
    blocker_counts: dict[str, int] = {}
    patch_counts: dict[str, int] = {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("Leaguepedia patch revision manifest is not an object")
        expected_manifest_hash = manifest.get("manifest_sha256")
        if expected_manifest_hash != _canonical_hash(manifest):
            blockers.append("leaguepedia_patch_revision_manifest_hash_mismatch")
        if bool(manifest.get("outcome_fields_emitted", True)):
            blockers.append("leaguepedia_patch_revision_outcome_field_emitted")
        receipt_path = _resolve_artifact_path(manifest_path, manifest.get("receipt_file", ""))
        if not receipt_path.exists():
            blockers.append("leaguepedia_patch_revision_receipt_file_missing")
        else:
            receipt_bytes = receipt_path.read_bytes()
            if manifest.get("receipt_file_sha256") != hashlib.sha256(receipt_bytes).hexdigest():
                blockers.append("leaguepedia_patch_revision_receipt_file_hash_mismatch")
            for line in receipt_bytes.decode("utf-8").splitlines():
                if not line.strip():
                    continue
                receipt = json.loads(line)
                if not isinstance(receipt, Mapping) or not receipt.get("fixture_id"):
                    blockers.append("leaguepedia_patch_revision_receipt_row_invalid")
                    continue
                if LEAGUEPEDIA_OUTCOME_FIELDS.intersection(receipt):
                    blockers.append("leaguepedia_patch_revision_outcome_field_emitted")
                evidence = receipt.get("evidence")
                if isinstance(evidence, Mapping):
                    if LEAGUEPEDIA_OUTCOME_FIELDS.intersection(evidence):
                        blockers.append("leaguepedia_patch_revision_outcome_field_emitted")
                    schedule_row = evidence.get("schedule_row")
                    if isinstance(schedule_row, Mapping) and LEAGUEPEDIA_OUTCOME_FIELDS.intersection(schedule_row):
                        blockers.append("leaguepedia_patch_revision_outcome_field_emitted")
                fixture_id = str(receipt["fixture_id"])
                index[fixture_id] = receipt
                patch = str(receipt.get("patch") or "")
                if patch:
                    patch_counts[patch] = patch_counts.get(patch, 0) + 1
                for item in receipt.get("blockers", []):
                    blocker = str(item)
                    if blocker:
                        blocker_counts[blocker] = blocker_counts.get(blocker, 0) + 1
        fixture_count = int(manifest.get("fixture_count", -1))
        if fixture_count != len(index):
            blockers.append("leaguepedia_patch_revision_fixture_count_mismatch")
        exact_count = sum(row.get("authority_status") == "pre_event_revision" for row in index.values())
        authorized_count = sum(bool(row.get("pregame_authorized")) for row in index.values())
        expected_exact = int(manifest.get("pre_event_revision_fixture_count", -1))
        expected_authorized = int(manifest.get("pregame_authorized_fixture_count", -1))
        if expected_exact != exact_count:
            blockers.append("leaguepedia_patch_revision_exact_count_mismatch")
        if expected_authorized != authorized_count:
            blockers.append("leaguepedia_patch_revision_authorized_count_mismatch")
        if authorized_count == fixture_count and fixture_count >= 0 and not blockers:
            status = "pregame_authorized"
        elif authorized_count > 0 and not any("missing" in item or "hash_mismatch" in item for item in blockers):
            status = "partial_pre_event"
        elif exact_count:
            status = "retrospective_only"
        else:
            status = "blocked"
        return (
            {
                "status": status,
                "path": str(manifest_path),
                "manifest_sha256": expected_manifest_hash,
                "receipt_file": str(receipt_path),
                "receipt_file_sha256": manifest.get("receipt_file_sha256"),
                "source_captured_at": manifest.get("captured_at"),
                "fixture_count": fixture_count,
                "exact_patch_fixture_count": exact_count,
                "pregame_authorized_fixture_count": authorized_count,
                "unavailable_fixture_count": int(manifest.get("unavailable_fixture_count", 0)),
                "patch_counts": dict(sorted(patch_counts.items())),
                "outcome_fields_requested": manifest.get("outcome_fields_requested", []),
                "outcome_fields_emitted": bool(manifest.get("outcome_fields_emitted", True)),
                "blocker_counts": dict(sorted(blocker_counts.items())),
                "blockers": sorted(set(blockers)),
            },
            index,
        )
    except Exception as exc:
        return (
            {
                "status": "blocked",
                "path": str(manifest_path),
                "fixture_count": 0,
                "exact_patch_fixture_count": 0,
                "pregame_authorized_fixture_count": 0,
                "unavailable_fixture_count": 0,
                "manifest_sha256": None,
                "receipt_file_sha256": None,
                "blocker_counts": {},
                "blockers": sorted(
                    set(blockers + [f"leaguepedia_patch_revision_validation_failed:{type(exc).__name__}"])
                ),
            },
            {},
        )


def _source_readiness(
    vault_path: Path,
    matrix_path: Path,
    client_matrix_path: Path = DEFAULT_CLIENT_PATCH_MATRIX,
) -> dict[str, Any]:
    blockers: list[str] = []
    try:
        vault = validate_wiki_vault(vault_path)
    except Exception as exc:
        vault = {"complete": False}
        blockers.append(f"wiki_vault_validation_failed:{type(exc).__name__}")
    if not vault.get("complete"):
        blockers.append("wiki_vault_incomplete")

    matrix: dict[str, Any] = {}
    packets: list[dict[str, Any]] = []
    exact_cells = 0
    blocked_cells = 0
    semantic_only_cells = 0
    try:
        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
        unsigned = dict(matrix)
        expected_hash = unsigned.pop("manifest_sha256", None)
        actual_hash = hashlib.sha256(json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        if expected_hash != actual_hash:
            blockers.append("patch_matrix_hash_mismatch")
        for row in matrix.get("patches", []):
            if not isinstance(row, Mapping):
                blockers.append("patch_matrix_row_invalid")
                continue
            packet_path = Path(str(row.get("packet_path", "")))
            if not packet_path.exists():
                blockers.append(f"patch_packet_missing:{row.get('patch', 'unknown')}")
                continue
            packet = load_patch_packet(packet_path)
            if packet.payload_sha256 != row.get("packet_sha256"):
                blockers.append(f"patch_packet_hash_mismatch:{packet.patch}")
            exact_cells += len(packet.executable_cells)
            semantic_only_cells += len(packet.semantic_only_cells)
            blocked_cells += len(packet.blocked_cells)
            packets.append({"patch": packet.patch, "packet_sha256": packet.payload_sha256, "executable_cells": len(packet.executable_cells), "semantic_only_cells": len(packet.semantic_only_cells), "blocked_cells": len(packet.blocked_cells)})
    except Exception as exc:
        blockers.append(f"patch_matrix_validation_failed:{type(exc).__name__}")

    client_matrix = _client_patch_readiness(client_matrix_path)
    blockers.extend(f"client:{blocker}" for blocker in client_matrix.get("blockers", []))
    status = "exact_available" if packets and exact_cells and client_matrix.get("status") == "exact_available" and not blockers else "semantic_only_and_blocked"
    return {
        "status": status,
        "vault": vault,
        "patch_matrix": {
            "path": str(matrix_path),
            "patch_count": len(packets),
            "executable_cell_count": exact_cells,
            "semantic_only_cell_count": semantic_only_cells,
            "blocked_cell_count": blocked_cells,
            "packets": packets,
        },
        "client_patch_matrix": client_matrix,
        "blockers": sorted(set(blockers)),
    }


def _autoresearch_gate(
    source_readiness: Mapping[str, Any],
    report: Mapping[str, Any],
    roster_readiness: Mapping[str, Any] | None = None,
    grid_patch_readiness: Mapping[str, Any] | None = None,
    leaguepedia_patch_readiness: Mapping[str, Any] | None = None,
    leaguepedia_patch_revision_readiness: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Record whether the bounded search preconditions are met."""

    blockers = list(source_readiness.get("blockers", []))
    if source_readiness.get("status") != "exact_available":
        blockers.append("exact_patch_mechanics_not_available")
    if int(report.get("available_predictions", 0)) <= 0:
        blockers.append("no_available_predictions")
    if (roster_readiness or {}).get("status") != "complete":
        blockers.append("pre_event_roster_authority_not_complete")
    if leaguepedia_patch_revision_readiness is not None:
        if leaguepedia_patch_revision_readiness.get("status") != "pregame_authorized":
            blockers.append("pre_event_patch_authority_not_complete")
    else:
        patch_receipts = (grid_patch_readiness, leaguepedia_patch_readiness)
        if not any((receipt or {}).get("status") == "pregame_authorized" for receipt in patch_receipts):
            blockers.append("pregame_patch_identity_not_authorized")
    return {
        "status": "ready" if not blockers else "not_run",
        "blockers": sorted(set(blockers)),
        "partition": {
            "ordering": "event_start_then_fixture_id",
            "train_fraction": 0.60,
            "validation_fraction": 0.20,
            "final_test_fraction": 0.20,
            "final_test_used_for_selection": False,
        },
        "stop_rule": {
            "success": "primary accuracy >= 0.80 and Wilson 95% half-width <= 0.05",
            "stagnation": "three consecutive valid candidates without >= 0.005 chronological validation accuracy improvement or coverage/parity improvement",
        },
    }


def run(
    run_dir: Path,
    output_dir: Path,
    *,
    vault_path: Path = DEFAULT_VAULT,
    patch_matrix_path: Path = DEFAULT_PATCH_MATRIX,
    client_patch_matrix_path: Path = DEFAULT_CLIENT_PATCH_MATRIX,
    roster_receipt_manifest_path: Path = DEFAULT_ROSTER_RECEIPT_MANIFEST,
    grid_patch_receipt_manifest_path: Path = DEFAULT_GRID_PATCH_RECEIPT_MANIFEST,
    leaguepedia_patch_receipt_manifest_path: Path = DEFAULT_LEAGUEPEDIA_PATCH_RECEIPT_MANIFEST,
    leaguepedia_patch_revision_manifest_path: Path = DEFAULT_LEAGUEPEDIA_PATCH_REVISION_MANIFEST,
) -> dict[str, Any]:
    rows = _load_jsonl(run_dir / "frozen-ledger.jsonl")
    roster_readiness, roster_receipts = _load_roster_receipts(roster_receipt_manifest_path)
    grid_patch_readiness, _grid_patch_receipts = _load_grid_patch_receipts(grid_patch_receipt_manifest_path)
    leaguepedia_patch_readiness, _leaguepedia_patch_receipts = _load_leaguepedia_patch_receipts(
        leaguepedia_patch_receipt_manifest_path
    )
    leaguepedia_patch_revision_readiness, leaguepedia_patch_revision_receipts = _load_leaguepedia_patch_revision_receipts(
        leaguepedia_patch_revision_manifest_path
    )
    source_readiness = _source_readiness(vault_path, patch_matrix_path, client_patch_matrix_path)
    bound_patch_labels = {
        str(item.get("patch"))
        for item in source_readiness.get("client_patch_matrix", {}).get("patches", [])
        if isinstance(item, Mapping)
        and item.get("status") == "captured"
        and item.get("authority_packet_sha256")
    }
    predictions = [
        _prediction(
            row,
            roster_receipts,
            roster_readiness.get("manifest_sha256"),
            leaguepedia_patch_revision_receipts,
            leaguepedia_patch_revision_readiness.get("manifest_sha256"),
            bound_patch_labels,
        )
        for row in rows
    ]
    outcomes = _outcomes(run_dir)
    report = evaluate_winner_gate(predictions, outcomes)
    blocker_counts: dict[str, int] = {}
    for prediction in predictions:
        for blocker in prediction.blockers:
            blocker_counts[blocker] = blocker_counts.get(blocker, 0) + 1
    report.update(
        {
            "run_kind": "mechanics_first_readiness",
            "source_run": str(run_dir),
            "prediction_count": len(predictions),
            "blocker_counts": dict(sorted(blocker_counts.items())),
            "source_readiness": source_readiness,
            "roster_readiness": roster_readiness,
            "grid_patch_readiness": grid_patch_readiness,
            "leaguepedia_patch_readiness": leaguepedia_patch_readiness,
            "leaguepedia_patch_revision_readiness": leaguepedia_patch_revision_readiness,
            "prediction_status": "unavailable_due_to_input_authority",
            "claim_ceiling": {
                "source_authority": False,
                "mechanics_execution": False,
                "prediction": False,
                "publication": False,
                "promotion": False,
            },
        }
    )
    report["autoresearch"] = _autoresearch_gate(
        source_readiness,
        report,
        roster_readiness,
        grid_patch_readiness,
        leaguepedia_patch_readiness,
        leaguepedia_patch_revision_readiness,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction.to_mapping(), ensure_ascii=False, sort_keys=True) + "\n")
    (output_dir / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "autoresearch-gate.json").write_text(
        json.dumps(report["autoresearch"], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--vault", type=Path, default=DEFAULT_VAULT)
    parser.add_argument("--patch-matrix", type=Path, default=DEFAULT_PATCH_MATRIX)
    parser.add_argument("--client-patch-matrix", type=Path, default=DEFAULT_CLIENT_PATCH_MATRIX)
    parser.add_argument("--roster-receipt-manifest", type=Path, default=DEFAULT_ROSTER_RECEIPT_MANIFEST)
    parser.add_argument("--grid-patch-receipt-manifest", type=Path, default=DEFAULT_GRID_PATCH_RECEIPT_MANIFEST)
    parser.add_argument("--leaguepedia-patch-receipt-manifest", type=Path, default=DEFAULT_LEAGUEPEDIA_PATCH_RECEIPT_MANIFEST)
    parser.add_argument("--leaguepedia-patch-revision-manifest", type=Path, default=DEFAULT_LEAGUEPEDIA_PATCH_REVISION_MANIFEST)
    args = parser.parse_args(argv)
    result = run(
        args.run,
        args.output,
        vault_path=args.vault,
        patch_matrix_path=args.patch_matrix,
        client_patch_matrix_path=args.client_patch_matrix,
        roster_receipt_manifest_path=args.roster_receipt_manifest,
        grid_patch_receipt_manifest_path=args.grid_patch_receipt_manifest,
        leaguepedia_patch_receipt_manifest_path=args.leaguepedia_patch_receipt_manifest,
        leaguepedia_patch_revision_manifest_path=args.leaguepedia_patch_revision_manifest,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
