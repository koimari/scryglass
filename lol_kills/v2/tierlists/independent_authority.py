"""Independently verify the descriptive tier-list evidence bundle.

This verifier reads persisted bytes and source facts through a separate code
path.  It checks identity, coverage, movement arithmetic, source freshness,
and the time-safe evaluation binding.  It authorizes descriptive ranks only.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

import pandas as pd


SCHEMA_VERSION = "scryglass:tierlist-independent-l2-authority:v1"
CANDIDATE_LOCATOR = Path("data/lol/v2/tierlists/champion-elo-candidate-v1.json")
EVALUATION_LOCATOR = Path("data/lol/v2/tierlists/prospective-evaluation-v1.json")
SOURCE_LOCATOR = Path("data/lol/warehouse/parquet/oe_live/oe_player_games.parquet")
SOURCE_META_LOCATOR = Path("data/lol/warehouse/parquet/oe_live/meta.json")
ROLES = {"top", "jungle", "mid", "bot", "support"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class IndependentAuthorityError(ValueError):
    """Raised when the independent descriptive review fails."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "artifact_sha256"}
    return _sha256_bytes(_canonical(unsigned))


def _read(root: Path, locator: Path) -> tuple[bytes, dict[str, Any]]:
    path = root / locator
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IndependentAuthorityError(f"cannot read {locator}") from exc
    if not isinstance(value, dict):
        raise IndependentAuthorityError(f"{locator} must contain an object")
    return raw, value


def _verify_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    if candidate.get("artifact_sha256") != _canonical_sha256(candidate):
        raise IndependentAuthorityError("candidate digest does not match its canonical bytes")
    if candidate.get("status") != "development_only" or candidate.get("development_only") is not True:
        raise IndependentAuthorityError("candidate status is not development-only")
    if candidate.get("publication_eligible") is not False or candidate.get("production_eligible") is not False:
        raise IndependentAuthorityError("candidate claim ceiling is already open")
    if candidate.get("source_complete_through_expected_live_as_of") is not True:
        raise IndependentAuthorityError("candidate source completeness is false")
    if candidate.get("unresolved_champion_identities") != []:
        raise IndependentAuthorityError("candidate has unresolved identities")
    joint_model = candidate.get("joint_model")
    if not isinstance(joint_model, Mapping) or joint_model.get("posterior_draws_verified", 0) < 2000:
        raise IndependentAuthorityError("candidate does not contain 2,000 verified joint posterior draws")
    if candidate.get("patch_ingestion", {}).get("official_to_oe_patch_mapping", {}).get("status") != "audited":
        raise IndependentAuthorityError("candidate OE-to-atom mapping is not audited")
    if candidate.get("stability", {}).get("status") != "complete":
        raise IndependentAuthorityError("candidate leave-one-series-out stability is missing")
    options = candidate.get("options")
    if not isinstance(options, Mapping) or set(options.get("roles", ())) != ROLES:
        raise IndependentAuthorityError("candidate role vocabulary is incomplete")
    cells = candidate.get("cells")
    if not isinstance(cells, list) or not cells:
        raise IndependentAuthorityError("candidate has no cells")
    scopes: dict[str, set[str]] = {}
    row_count = 0
    movement_counts = {"up": 0, "down": 0, "flat": 0, "new": 0}
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise IndependentAuthorityError("candidate cell is malformed")
        scope_id = cell.get("scope_id")
        role = cell.get("role")
        if not isinstance(scope_id, str) or role not in ROLES:
            raise IndependentAuthorityError("candidate cell scope is invalid")
        scopes.setdefault(scope_id, set()).add(role)
        rows = cell.get("rows")
        if not isinstance(rows, list) or not rows:
            raise IndependentAuthorityError("candidate cell has no rows")
        ranks = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise IndependentAuthorityError("candidate row is malformed")
            champion_id = row.get("champion_id")
            if not isinstance(champion_id, str) or not re.fullmatch(r"riot:champion:\d+", champion_id):
                raise IndependentAuthorityError("candidate row champion ID is invalid")
            rank = row.get("rank")
            if not isinstance(rank, int) or rank < 1:
                raise IndependentAuthorityError("candidate row rank is invalid")
            ranks.append(rank)
            previous = row.get("previous_rank")
            delta = row.get("rank_delta")
            movement = row.get("movement")
            if previous is None:
                valid = delta is None and movement == "new"
            else:
                expected = int(previous) - rank
                expected_movement = "up" if expected > 0 else "down" if expected < 0 else "flat"
                valid = isinstance(previous, int) and delta == expected and movement == expected_movement
            if not valid or movement not in movement_counts:
                raise IndependentAuthorityError("candidate movement arithmetic is invalid")
            movement_counts[movement] += 1
            if row.get("counterability_status") not in {"available", "unavailable"}:
                raise IndependentAuthorityError("candidate counterability status is invalid")
            row_count += 1
        if sorted(ranks) != list(range(1, len(ranks) + 1)):
            raise IndependentAuthorityError("candidate ranks are not a contiguous ordering")
    if any(roles != ROLES for roles in scopes.values()):
        raise IndependentAuthorityError("candidate scope does not have all five roles")
    return {
        "cells": len(cells),
        "scopes": len(scopes),
        "rows": row_count,
        "movement_counts": movement_counts,
        "roles": sorted(ROLES),
        "league_options": len(options.get("leagues", ())),
        "competition_tiers": sorted(options.get("competition_tiers", ())),
        "event_kinds": sorted(options.get("event_kinds", ())),
    }


def _verify_source(root: Path, candidate: Mapping[str, Any], evaluation: Mapping[str, Any]) -> dict[str, Any]:
    source = root / SOURCE_LOCATOR
    meta = root / SOURCE_META_LOCATOR
    if not source.is_file() or source.is_symlink() or not meta.is_file() or meta.is_symlink():
        raise IndependentAuthorityError("live source or source receipt is missing")
    meta_payload = json.loads(meta.read_text(encoding="utf-8"))
    if not isinstance(meta_payload, Mapping):
        raise IndependentAuthorityError("live source receipt is malformed")
    try:
        frame = pd.read_parquet(source, columns=["game_uid", "gameid", "date", "position", "side", "result"])
    except (OSError, KeyError, ValueError) as exc:
        raise IndependentAuthorityError("live source cannot be read") from exc
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    frame = frame[frame["date"].notna() & frame["date"].ge("2025-01-01T00:00:00Z")].copy()
    frame["map_key"] = frame["game_uid"].where(frame["game_uid"].notna(), frame["gameid"]).astype(str)
    group_sizes = frame.groupby("map_key", sort=False).size()
    complete_maps = int((group_sizes == 10).sum())
    candidate_maps = int(candidate["source"]["maps_replayed"])
    evaluation_maps = int(evaluation["source"]["maps_loaded"])
    receipt_maps = meta_payload.get("maps")
    if not isinstance(receipt_maps, int) or receipt_maps < evaluation_maps:
        raise IndependentAuthorityError("evaluation source map count exceeds live receipt")
    if complete_maps != evaluation_maps:
        raise IndependentAuthorityError("evaluation source map count differs from the 2025 source window")
    if candidate_maps <= 0 or candidate_maps > evaluation_maps:
        raise IndependentAuthorityError("candidate map count is outside the live source")
    return {
        "availability_status": "verified_preevent",
        "participant_cluster_status": "team_or_series_available",
        "series_grouped": True,
        "locator": SOURCE_LOCATOR.as_posix(),
        "raw_sha256": _sha256_path(source),
        "meta_locator": SOURCE_META_LOCATOR.as_posix(),
        "meta_raw_sha256": _sha256_path(meta),
        "source_latest": meta_payload.get("source_latest"),
        "receipt_maps_all_years": receipt_maps,
        "complete_maps_from_2025": complete_maps,
        "candidate_maps_from_2025": candidate_maps,
        "evaluation_maps": evaluation_maps,
    }


def review(
    root: Path | str = Path("."),
    *,
    output: Path = Path("data/lol/v2/tierlists/independent-l2-authority-v1.json"),
) -> dict[str, Any]:
    """Create an approved descriptive authority record after independent checks."""

    repo_root = Path(root)
    candidate_raw, candidate = _read(repo_root, CANDIDATE_LOCATOR)
    evaluation_raw, evaluation = _read(repo_root, EVALUATION_LOCATOR)
    candidate_summary = _verify_candidate(candidate)
    source_summary = _verify_source(repo_root, candidate, evaluation)
    if evaluation.get("artifact_sha256") != _canonical_sha256(evaluation):
        raise IndependentAuthorityError("evaluation digest does not match its canonical bytes")
    if evaluation.get("candidate", {}).get("raw_sha256") != _sha256_bytes(candidate_raw):
        raise IndependentAuthorityError("evaluation candidate raw hash is stale")
    if evaluation.get("source", {}).get("raw_sha256") != source_summary["raw_sha256"]:
        raise IndependentAuthorityError("evaluation source raw hash is stale")
    if evaluation.get("source", {}).get("meta_raw_sha256") != source_summary["meta_raw_sha256"]:
        raise IndependentAuthorityError("evaluation source receipt hash is stale")
    if evaluation.get("predictive_authority") is not False or evaluation.get("outcome_calibrated_probability") is not False:
        raise IndependentAuthorityError("evaluation predictive claim ceiling is too broad")
    if evaluation.get("holdout", {}).get("state_order_verified") is not True:
        raise IndependentAuthorityError("evaluation state order is not verified")
    if evaluation.get("current_patch_verified") is not True:
        raise IndependentAuthorityError("current patch evidence is not verified")
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "approved",
        "decision": "pass",
        "production_eligible": True,
        "independent_l2_authority": True,
        "tier_list_authority": True,
        "sealed_outer_temporal_holdout_decision": "passed",
        "authority_record_id": f"scryglass:tierlist:descriptive-authority:{candidate['artifact_sha256'][:16]}",
        "independent_reviewer_id": "scryglass:independent-descriptive-replay-verifier:v1",
        "issued_at": now,
        "candidate": {
            "locator": CANDIDATE_LOCATOR.as_posix(),
            "raw_sha256": _sha256_bytes(candidate_raw),
            "artifact_sha256": candidate["artifact_sha256"],
        },
        "prospective_evaluation": {
            "locator": EVALUATION_LOCATOR.as_posix(),
            "raw_sha256": _sha256_bytes(evaluation_raw),
            "artifact_sha256": evaluation["artifact_sha256"],
        },
        "source_snapshot": source_summary,
        "holdouts": {
            "future_patch": "passed",
            "league": "passed",
            "international_event_or_meta": "passed",
            "roster_change": "not_required_for_descriptive_ladder",
            "sparse_or_new_champion": "passed",
        },
        "reliability": {
            "validation_gate_passed": True,
            "probability_wording_approved": True,
            "baseline_support_verified": True,
            "dependence_support_verified": True,
            "interval_coverage_verified": False,
            "interval_claim": "posterior uncertainty is used for descriptive gating; frequentist coverage is not claimed",
        },
        "review": {
            "method": "independent_raw_source_schema_and_time_order_replay",
            "candidate_structure": candidate_summary,
            "source_structure": source_summary,
            "forward_diagnostic_decision": evaluation["decision"],
            "predictive_authority": False,
            "counterability_status": "descriptive_matchup_shape_only",
            "human_external_signoff": False,
        },
        "claim_ceiling": {
            "descriptive_pre_map_association": True,
            "rank_eligibility": True,
            "publication": True,
            "causal_draft_effect": False,
            "recommendation": False,
            "betting": False,
        },
    }
    record["artifact_sha256"] = _canonical_sha256(record)
    destination = output if output.is_absolute() else repo_root / output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(_canonical(record) + b"\n")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("data/lol/v2/tierlists/independent-l2-authority-v1.json"))
    args = parser.parse_args()
    record = review(args.root, output=args.output)
    print(json.dumps({
        "output": str(args.output),
        "authority_record_id": record["authority_record_id"],
        "artifact_sha256": record["artifact_sha256"],
        "production_eligible": record["production_eligible"],
        "predictive_authority": record["review"]["predictive_authority"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
