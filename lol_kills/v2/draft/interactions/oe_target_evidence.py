"""Private, non-authoritative OE outcome evidence for draft-representation work.

The split assignment is constructed from schedule and dependence-cluster data
before outcomes are read.  The resulting packet proves reproducible source
consistency only.  It cannot authorize fitting, rank selection, prediction,
publication, production use, or a SOTA claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


SCHEMA_ID = "scryglass.oe-private-target-evidence.v1"
SPLIT_SCHEMA_ID = "scryglass.oe-private-outcome-free-split-assignment.v1"
GENERATOR_VERSION = "oe-private-target-evidence-generator.v1"
PINNED_PREFLIGHT_PAYLOAD_SHA256 = "ba54faed41716cc537268c6e7eecbaaf9330937014bfd2cd5f9a50f930f92eb4"
PINNED_PROXY_PAYLOAD_SHA256 = "e456d267797a23dae94f8ecc9a31ca91593d48e830b6f334191f0c025bf19ada"
PINNED_CROSSWALK_PAYLOAD_SHA256 = "59fdf214b570487f64e08f060ba51c82b24e87f3bbe4d6e308fd1bdd42ef14f7"
PINNED_PREFLIGHT_RAW_SHA256 = "e3d2d3c9399e42c7d8d7ad8653698353f0af52e09e1a2e80c33e5b5e369d95c1"
PINNED_PROXY_RAW_SHA256 = "5d89bcc3029d3fa912d76af9c702888f76f4183933dd1d1e2e660b8f2a8bdd2a"
PINNED_CROSSWALK_RAW_SHA256 = "8a57b26bd33d27e6a81393bdffc450db0adeb9e4a9d2ee221a8244e2f6452555"
PINNED_MAPS_RAW_SHA256 = "04c0cce1d86a4358d9eeb5937f61d5288358953e66c693a1ce88b0b650295d08"
PINNED_TEAM_RAW_SHA256 = "c8b6bbee02c383c07f43571599dd30097223212b5ba46a896ef33b8fa3bc5fb3"
PINNED_PLAYER_RAW_SHA256 = "3d2a852daa43dfa402e1e48ef11d1a6858b73f2171f0c2febd82b941b19fceee"
PINNED_ANNUAL_RAW_SHA256 = {
    2025: "c9a158b9e0a965a47d31d3674c127a26f75e6c91a324bd1858e4784b1336214a",
    2026: "8a7e0215fe7505e824cef854e8795f502dabc460d4ef346310c65e344d65aca8",
}
PINNED_PREFLIGHT_GENERATOR_SHA256 = "0f80190191dc19d30e98e8d7a0db9963007a93fdf0d2e7f5b184487165e5e3ea"
PINNED_PROXY_GENERATOR_SHA256 = "d63ee58bb93015e0c0427c7aac584b098884a1567da32849e4ed1993e54dae48"
PINNED_CROSSWALK_GENERATOR_SHA256 = "ef5feefb587016831f875690e15aabeeeacccf94122ab57670998e25cc293c9d"

DEFAULT_MAPS_PATH = Path("data/lol/warehouse/parquet/maps.parquet")
DEFAULT_TEAM_PATH = Path("data/lol/warehouse/parquet/oe_team_games.parquet")
DEFAULT_PLAYER_PATH = Path("data/lol/warehouse/parquet/oe_player_games.parquet")
DEFAULT_ANNUAL_PATHS = {
    2025: Path("data/lol/warehouse/raw/2025_LoL_esports_match_data_from_OraclesElixir.csv"),
    2026: Path("data/lol/warehouse/raw/2026_LoL_esports_match_data_from_OraclesElixir.csv"),
}
DEFAULT_PREFLIGHT_PATH = Path("data/lol/v2/models/draft-interactions/representation-assay-preflight.json")
DEFAULT_PROXY_PATH = Path("data/lol/v2/models/draft-interactions/series-cluster-proxy.json")
DEFAULT_CROSSWALK_PATH = Path("data/lol/v2/champions/champion-id-crosswalk-v1.json")
DEFAULT_EVIDENCE_PATH = Path("data/lol/v2/models/draft-interactions/oe-private-target-evidence.json")
DEFAULT_SPLIT_PATH = Path("data/lol/v2/models/draft-interactions/oe-private-split-assignment.json")
DEFAULT_PRIVATE_ROWS_PATH = Path("data/lol/warehouse/private_v2/draft-interactions/oe-target-rows.parquet")

MAP_COLUMNS = ("oe_gameid", "game_uid", "league", "date", "patch", "blue_result", "red_result", "y_blue_win", "gamelength")
TEAM_COLUMNS = ("gameid", "participantid", "side", "result")
PLAYER_COLUMNS = ("gameid", "participantid", "side", "position", "champion")
RAW_COLUMNS = ("gameid", "participantid", "side", "position", "champion", "result", "date", "patch", "league", "gamelength")
ROLE_ORDER = ("top", "jungle", "mid", "bot", "support")
ROLE_ALIASES = {"top": "top", "jng": "jungle", "mid": "mid", "bot": "bot", "sup": "support"}
PROPOSED_BOUNDARIES = (
    ("train", pd.Timestamp("2025-10-01")),
    ("development", pd.Timestamp("2026-04-01")),
    ("validation", pd.Timestamp("2026-06-01")),
)
FINAL_SPLIT = "final_temporal_holdout"
EXPECTED_ASSIGNED_MAPS = 6310


class OETargetEvidenceError(ValueError):
    """Raised when target evidence or authority boundaries fail closed."""


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def raw_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar(value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def selected_input_sha256(frame: pd.DataFrame) -> str:
    rows = [[_scalar(v) for v in row] for row in frame.itertuples(index=False, name=None)]
    rows.sort(key=canonical_bytes)
    return canonical_sha256({"columns": list(frame.columns), "rows": rows})


def ordered_input_sha256(frame: pd.DataFrame) -> str:
    rows = [
        [_scalar(value) for value in row]
        for row in frame.itertuples(index=False, name=None)
    ]
    return canonical_sha256({"columns": list(frame.columns), "rows": rows})


def _artifact_payload(path: Path, expected_payload: str, label: str) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise OETargetEvidenceError(f"{label} is not valid JSON") from exc
    if payload.get("artifact_sha256") != expected_payload:
        raise OETargetEvidenceError(f"{label} payload pin mismatch")
    if canonical_sha256({k: v for k, v in payload.items() if k != "artifact_sha256"}) != expected_payload:
        raise OETargetEvidenceError(f"{label} canonical payload mismatch")
    return payload, hashlib.sha256(raw).hexdigest()


def _normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value).strip()).casefold()
    text = text.replace("\u2019", "'").replace("`", "'")
    return " ".join(text.split())


def _patch_token(value: object) -> str:
    if isinstance(value, bool) or pd.isna(value):
        raise OETargetEvidenceError("OE patch token is missing")
    number = float(value)
    centesimal = round(number * 100)
    if not math.isfinite(number) or number <= 0 or abs(number * 100 - centesimal) > 1e-8:
        raise OETargetEvidenceError("OE patch token is not exact")
    return f"{centesimal / 100:.2f}"


def _naive_timestamp(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is not None:
        raise OETargetEvidenceError("OE date must be present and timezone-naive")
    return stamp


def _hash_rows(rows: Sequence[Mapping[str, Any]]) -> str:
    return canonical_sha256(sorted((dict(row) for row in rows), key=lambda x: canonical_bytes(x)))


def build_outcome_free_split(
    maps: pd.DataFrame,
    proxy_payload: Mapping[str, Any],
    *,
    maps_source: Mapping[str, Any],
    proxy_source: Mapping[str, Any],
    annual_membership: set[str] | None = None,
) -> dict[str, Any]:
    """Assign complete dependence clusters without reading outcomes or duration.

    The evidence population is the proxy membership that traces to the raw
    annual OE CSVs (grid-sourced maps belong to the separate GRID pipeline and
    are not OE-target evidence rows).
    """
    allowed = maps.loc[:, ["oe_gameid", "date"]].copy()
    if allowed["oe_gameid"].duplicated().any():
        raise OETargetEvidenceError("map identity is not unique")
    dates = {str(r.oe_gameid): _naive_timestamp(r.date) for r in allowed.itertuples(index=False)}
    if annual_membership is None:
        assignments = list(proxy_payload.get("assignments", []))
    else:
        assignments = [
            row for row in proxy_payload.get("assignments", [])
            if str(row["game_id"]) in annual_membership
        ]
    if len(assignments) != EXPECTED_ASSIGNED_MAPS or len({r["game_id"] for r in assignments}) != EXPECTED_ASSIGNED_MAPS:
        raise OETargetEvidenceError("proxy population is not exactly the annual-origin map population")
    clusters: dict[str, list[str]] = defaultdict(list)
    for row in assignments:
        game_id = str(row["game_id"])
        if game_id not in dates:
            raise OETargetEvidenceError("proxy assignment is outside map registry")
        clusters[str(row["dependence_cluster_id"])].append(game_id)

    split_rows: list[dict[str, Any]] = []
    for cluster_id, game_ids in sorted(clusters.items()):
        cluster_max = max(dates[g] for g in game_ids)
        if cluster_max < PROPOSED_BOUNDARIES[0][1]:
            split = "train"
        elif cluster_max < PROPOSED_BOUNDARIES[1][1]:
            split = "development"
        elif cluster_max < PROPOSED_BOUNDARIES[2][1]:
            split = "validation"
        else:
            split = FINAL_SPLIT
        split_rows.extend(
            {"game_id": g, "dependence_cluster_id": cluster_id, "split": split, "oe_date_naive": dates[g].isoformat()}
            for g in sorted(game_ids)
        )
    split_rows.sort(key=lambda r: (r["oe_date_naive"], r["game_id"]))
    counts = Counter(r["split"] for r in split_rows)
    cluster_counts = Counter({s: len({r["dependence_cluster_id"] for r in split_rows if r["split"] == s}) for s in counts})
    chronology = {}
    for split in ("train", "development", "validation", FINAL_SPLIT):
        values = [r["oe_date_naive"] for r in split_rows if r["split"] == split]
        chronology[split] = {"maps": len(values), "clusters": cluster_counts.get(split, 0), "minimum_date_naive": min(values) if values else None, "maximum_date_naive": max(values) if values else None}
    nonempty = [chronology[x] for x in ("train", "development", "validation", FINAL_SPLIT)]
    if any(item["maps"] == 0 for item in nonempty):
        raise OETargetEvidenceError("fixed calendar split lacks support")
    for left, right in zip(nonempty, nonempty[1:]):
        if left["maximum_date_naive"] >= right["minimum_date_naive"]:
            raise OETargetEvidenceError("strict temporal chronology violated")
    payload: dict[str, Any] = {
        "schema_id": SPLIT_SCHEMA_ID,
        "status": "private_pending_human_review",
        "development_only": True,
        "outcome_free": True,
        "once_reserved_final_temporal_holdout": True,
        "historical_live_forecast_claim": False,
        "predictive_target_authority": False,
        "authorizes_model_fit": False,
        "authorizes_rank_selection": False,
        "authorizes_prediction": False,
        "authorizes_publication": False,
        "authorizes_production": False,
        "authorizes_sota_claim": False,
        "content_addressing_confers_authority": False,
        "construction_order": "generated before and independently of target extraction",
        "generator": _generator_identity(),
        "proposed_fixed_boundaries": [
            {"earlier_split": name, "next_split_starts_at": boundary.isoformat(), "rule": "a cluster enters the later split if its maximum OE date is on or after this boundary"}
            for name, boundary in PROPOSED_BOUNDARIES
        ],
        "source": {"maps": dict(maps_source), "dependence_proxy": dict(proxy_source)},
        "membership_sha256": _hash_rows([{"game_id": r["game_id"]} for r in split_rows]),
        "dependence_assignment_sha256": _hash_rows([{"game_id": r["game_id"], "dependence_cluster_id": r["dependence_cluster_id"]} for r in split_rows]),
        "outcome_free_split_sha256": _hash_rows([{"game_id": r["game_id"], "dependence_cluster_id": r["dependence_cluster_id"], "oe_date_naive": r["oe_date_naive"], "split": r["split"]} for r in split_rows]),
        "counts": {"maps": len(split_rows), "dependence_clusters": len(clusters), "by_split": dict(sorted(counts.items()))},
        "chronology": chronology,
        "assignments": split_rows,
    }
    payload["artifact_sha256"] = canonical_sha256(payload)
    validate_split(payload)
    return payload


def validate_split(payload: Mapping[str, Any]) -> None:
    unsigned = dict(payload)
    claimed = unsigned.pop("artifact_sha256", None)
    if claimed != canonical_sha256(unsigned):
        raise OETargetEvidenceError("split artifact hash mismatch")
    if payload.get("outcome_free") is not True or payload.get("counts", {}).get("maps") != EXPECTED_ASSIGNED_MAPS:
        raise OETargetEvidenceError("split population contract violated")
    false_flags = ("predictive_target_authority", "authorizes_model_fit", "authorizes_rank_selection", "authorizes_prediction", "authorizes_publication", "authorizes_production", "authorizes_sota_claim", "content_addressing_confers_authority")
    if any(payload.get(field) is not False for field in false_flags):
        raise OETargetEvidenceError("split authority contract violated")
    by_cluster: dict[str, set[str]] = defaultdict(set)
    for row in payload["assignments"]:
        by_cluster[row["dependence_cluster_id"]].add(row["split"])
    if any(len(value) != 1 for value in by_cluster.values()):
        raise OETargetEvidenceError("dependence cluster split across folds")


def _feature_rows(players: pd.DataFrame, membership: set[str], crosswalk: Mapping[str, Any], maps: pd.DataFrame) -> list[dict[str, Any]]:
    mapping = {entry["normalized_oe_name"]: entry["stable_champion_id"] for entry in crosswalk["entries"]}
    map_meta = {str(r.oe_gameid): (str(r.league).strip().upper(), _patch_token(r.patch)) for r in maps.itertuples(index=False)}
    selected = players[players["gameid"].astype(str).isin(membership)]
    result: list[dict[str, Any]] = []
    for game_id, group in selected.groupby("gameid", sort=True):
        slots: dict[tuple[str, str], str] = {}
        for row in group.itertuples(index=False):
            side = str(row.side).strip().lower()
            role = ROLE_ALIASES.get(str(row.position).strip().lower())
            stable = mapping.get(_normalize_name(row.champion))
            if side not in ("blue", "red") or role is None or stable is None:
                raise OETargetEvidenceError("unstable champion-role identity")
            key = (side, role)
            if key in slots:
                raise OETargetEvidenceError("duplicate champion-role slot")
            slots[key] = stable
        if set(slots) != {(s, r) for s in ("blue", "red") for r in ROLE_ORDER}:
            raise OETargetEvidenceError("map does not have ten stable champion-role slots")
        league, patch = map_meta[str(game_id)]
        result.append({"game_id": str(game_id), "canonical_league": league, "oe_patch_token": patch, "sides": {side: {role: slots[(side, role)] for role in ROLE_ORDER} for side in ("blue", "red")}})
    if len(result) != EXPECTED_ASSIGNED_MAPS:
        raise OETargetEvidenceError("feature population is not 6,194 maps")
    return result


def _target_rows(maps: pd.DataFrame, teams: pd.DataFrame, membership: set[str]) -> list[dict[str, Any]]:
    selected_maps = maps[maps["oe_gameid"].astype(str).isin(membership)]
    team_groups = {str(g): frame for g, frame in teams[teams["gameid"].astype(str).isin(membership)].groupby("gameid")}
    rows: list[dict[str, Any]] = []
    for row in selected_maps.itertuples(index=False):
        game_id = str(row.oe_gameid)
        values = (row.y_blue_win, row.blue_result, row.red_result)
        if any(isinstance(v, bool) or pd.isna(v) or not math.isfinite(float(v)) or int(v) not in (0, 1) for v in values):
            raise OETargetEvidenceError("outcome is not finite binary")
        y, blue, red = map(int, values)
        if y != blue or red != 1 - blue:
            raise OETargetEvidenceError("map outcome semantics disagree")
        group = team_groups.get(game_id)
        if group is None or len(group) != 2:
            raise OETargetEvidenceError("team source result rows missing or duplicated")
        observed: dict[str, tuple[int, str]] = {}
        for team in group.itertuples(index=False):
            side = str(team.side).strip().lower()
            participant = int(team.participantid)
            result = int(team.result)
            if side not in ("blue", "red") or participant != {"blue": 100, "red": 200}[side] or result not in (0, 1):
                raise OETargetEvidenceError("team result source identity is invalid")
            source_id = f"oe-team-row:{game_id}:{participant}"
            observed[side] = (result, source_id)
        if observed.get("blue", (None,))[0] != y or observed.get("red", (None,))[0] != 1 - y:
            raise OETargetEvidenceError("target disagreement between maps and OE team rows")
        rows.append({"game_id": game_id, "y_blue_win": y, "source_blue_result_id": observed["blue"][1], "source_red_result_id": observed["red"][1]})
    rows.sort(key=lambda r: r["game_id"])
    if len(rows) != EXPECTED_ASSIGNED_MAPS or len({r["game_id"] for r in rows}) != EXPECTED_ASSIGNED_MAPS:
        raise OETargetEvidenceError("target source row or game identity is not unique")
    return rows


def _verify_raw_annual_origin(
    annual_frames: Mapping[int, pd.DataFrame],
    membership: set[str],
    targets: Sequence[Mapping[str, Any]],
) -> str:
    combined = pd.concat(
        [frame.assign(source_year=year) for year, frame in sorted(annual_frames.items())],
        ignore_index=True,
    )
    combined = combined[combined["gameid"].astype(str).isin(membership)].copy()
    if set(combined["gameid"].astype(str)) != membership:
        raise OETargetEvidenceError("raw annual OE membership differs from assigned maps")
    source_ids = [
        f"oe-raw-row:{int(row.source_year)}:{row.gameid}:{int(row.participantid)}"
        for row in combined.itertuples(index=False)
    ]
    if len(source_ids) != len(set(source_ids)):
        raise OETargetEvidenceError("raw annual OE source row identity is not unique")
    if set(combined.groupby("gameid").size()) != {12}:
        raise OETargetEvidenceError("raw annual OE map does not have exactly twelve source rows")
    expected = {row["game_id"]: int(row["y_blue_win"]) for row in targets}
    for game_id, group in combined.groupby("gameid", sort=False):
        teams = group[group["participantid"].isin((100, 200))]
        if len(teams) != 2:
            raise OETargetEvidenceError("raw annual OE team rows are missing")
        observed = {int(r.participantid): int(r.result) for r in teams.itertuples(index=False)}
        if observed != {100: expected[str(game_id)], 200: 1 - expected[str(game_id)]}:
            raise OETargetEvidenceError("raw annual OE outcome disagrees with target")
    return canonical_sha256(sorted(source_ids))


def _generator_identity() -> dict[str, Any]:
    runtime_versions = {
        name: importlib.metadata.version(name) for name in ("numpy", "pandas", "pyarrow")
    }
    return {
        "version": GENERATOR_VERSION,
        "executable_dependency_boundary": [{"locator": "lol_kills/v2/draft/interactions/oe_target_evidence.py", "raw_sha256": raw_sha256(Path(__file__).resolve())}],
        "runtime_versions": runtime_versions,
        "runtime_identity_sha256": canonical_sha256(runtime_versions),
    }


def analyze_frames(
    maps: pd.DataFrame,
    teams: pd.DataFrame,
    players: pd.DataFrame,
    annual_frames: Mapping[int, pd.DataFrame],
    proxy: Mapping[str, Any],
    crosswalk: Mapping[str, Any],
    split: Mapping[str, Any],
    *,
    source_manifest: Mapping[str, Any],
    private_rows_path: Path,
) -> dict[str, Any]:
    """Extract targets after accepting an already-built outcome-free split."""
    validate_split(split)
    membership = {r["game_id"] for r in split["assignments"]}
    proxy_ids = {proxy_assignment["game_id"] for proxy_assignment in proxy["assignments"]}
    if not membership <= proxy_ids or len(membership) != EXPECTED_ASSIGNED_MAPS:
        raise OETargetEvidenceError("split membership differs from pinned proxy")
    features = _feature_rows(players, membership, crosswalk, maps)
    targets = _target_rows(maps, teams, membership)
    raw_source_row_identity_sha256 = _verify_raw_annual_origin(
        annual_frames, membership, targets
    )
    target_by_id = {row["game_id"]: row for row in targets}
    feature_by_id = {row["game_id"]: row for row in features}
    split_by_id = {row["game_id"]: row for row in split["assignments"]}
    map_by_id = {str(row.oe_gameid): row for row in maps.itertuples(index=False)}
    private_records: list[dict[str, Any]] = []
    resolution_annotations: list[dict[str, Any]] = []
    for game_id in sorted(membership):
        source = map_by_id[game_id]
        duration = source.gamelength
        derived = None
        if not pd.isna(duration):
            number = float(duration)
            if math.isfinite(number) and number >= 0:
                derived = (_naive_timestamp(source.date) + pd.to_timedelta(number, unit="s")).isoformat()
        resolution_annotations.append({"game_id": game_id, "derived_resolution_time_naive": derived})
        feat, target, assignment = feature_by_id[game_id], target_by_id[game_id], split_by_id[game_id]
        private_records.append({
            "game_id": game_id,
            "dependence_cluster_id": assignment["dependence_cluster_id"],
            "split": assignment["split"],
            "oe_date_naive": assignment["oe_date_naive"],
            "draft_completed_at": None,
            "forecast_at": None,
            "derived_resolution_time_naive": derived,
            "canonical_league": feat["canonical_league"],
            "oe_patch_token": feat["oe_patch_token"],
            **{f"{side}_{role}_stable_champion_id": feat["sides"][side][role] for side in ("blue", "red") for role in ROLE_ORDER},
            **target,
        })
    private = pd.DataFrame(private_records)
    private_rows_path.parent.mkdir(parents=True, exist_ok=True)
    private.to_parquet(private_rows_path, index=False)
    target_transform = {
        "id": "oe-blue-binary-result-identity-v1",
        "definition": "y_blue_win := blue_result after verifying red_result == 1 - blue_result and OE team-result row agreement",
        "uses_duration": False,
    }
    packet: dict[str, Any] = {
        "schema_id": SCHEMA_ID,
        "status": "private_pending_human_review",
        "development_only": True,
        "retrospective_out_of_time_only": True,
        "historical_live_forecast_claim": False,
        "predictive_target_authority": False,
        "authorizes_model_fit": False,
        "authorizes_rank_selection": False,
        "authorizes_prediction": False,
        "authorizes_publication": False,
        "authorizes_production": False,
        "authorizes_sota_claim": False,
        "content_addressing_confers_authority": False,
        "claim_ceiling": "source consistency and retrospective materialization only; no combined hash is authority",
        "generator": _generator_identity(),
        "source": dict(source_manifest),
        "split_assignment": {"locator": DEFAULT_SPLIT_PATH.as_posix(), "payload_sha256": split["artifact_sha256"], "outcome_free_split_sha256": split["outcome_free_split_sha256"]},
        "population": {"assigned_maps": len(private), "collision_exclusions": proxy["eligibility"]["exclusion_ledger"]},
        "time_contract": {
            "oe_date": "timezone-naive source timestamp preserved",
            "draft_completed_at": None,
            "forecast_at": None,
            "derived_resolution_time_label": "derived_resolution_time_naive",
            "duration_use": "annotation only; forbidden in eligibility, split assignment, feature domain, nuisance inputs, and rank selection",
        },
        "membership_sha256": split["membership_sha256"],
        "dependence_assignment_sha256": split["dependence_assignment_sha256"],
        "feature_domain_sha256": _hash_rows(features),
        "target_domain_sha256": _hash_rows(targets),
        "raw_source_row_identity_sha256": raw_source_row_identity_sha256,
        "target_transform": target_transform,
        "target_transform_sha256": canonical_sha256(target_transform),
        "derived_resolution_annotation_sha256": _hash_rows(resolution_annotations),
        "private_materialization": {
            "locator": private_rows_path.as_posix(),
            "rows": len(private),
            "raw_sha256": raw_sha256(private_rows_path),
            "logical_rows_sha256": selected_input_sha256(private),
            "ordered_logical_rows_sha256": ordered_input_sha256(private),
            "expected_git_ignored": True,
        },
        "consistency_ledger_sha256": "",
        "human_authority_required": {
            "separate_envelope_schema_id": "scryglass.oe-private-target-human-authority.v1",
            "required_fields": [
                "decision_id", "reviewer_identity", "reviewed_at_rfc3339",
                "approval_scope", "evidence_payload_sha256", "split_payload_sha256",
                "decision", "approved_actions", "source_rights_reviewed",
                "target_semantics_reviewed", "temporal_leakage_reviewed",
                "fixed_boundaries_reviewed", "independent_from_generator",
                "generator_authored",
            ],
            "independent_pinned_envelope_hash_status": "absent_pending_human_review",
        },
    }
    ledger = {key: packet[key] for key in ("membership_sha256", "dependence_assignment_sha256", "feature_domain_sha256", "target_domain_sha256", "raw_source_row_identity_sha256", "target_transform_sha256", "derived_resolution_annotation_sha256")}
    packet["consistency_ledger_sha256"] = canonical_sha256(ledger)
    packet["artifact_sha256"] = canonical_sha256(packet)
    validate_evidence(packet)
    return packet


def validate_evidence(payload: Mapping[str, Any]) -> None:
    unsigned = dict(payload)
    claimed = unsigned.pop("artifact_sha256", None)
    if claimed != canonical_sha256(unsigned):
        raise OETargetEvidenceError("evidence artifact hash mismatch")
    false_flags = ("predictive_target_authority", "authorizes_model_fit", "authorizes_rank_selection", "authorizes_prediction", "authorizes_publication", "authorizes_production", "authorizes_sota_claim", "content_addressing_confers_authority")
    if payload.get("status") != "private_pending_human_review" or any(payload.get(x) is not False for x in false_flags):
        raise OETargetEvidenceError("evidence authority contract violated")
    if payload.get("population", {}).get("assigned_maps") != EXPECTED_ASSIGNED_MAPS:
        raise OETargetEvidenceError("evidence population contract violated")
    if payload["time_contract"].get("draft_completed_at") is not None or payload["time_contract"].get("forecast_at") is not None:
        raise OETargetEvidenceError("historical timestamps were invented")


def require_exact_human_authority(
    envelope_bytes: bytes | None,
    evidence: Mapping[str, Any],
    split: Mapping[str, Any],
    *,
    action: str,
) -> Mapping[str, Any]:
    """Delegate authority to a root outside the evidence generator hash."""
    from .oe_target_authority import (
        require_exact_human_authority as require_independent_authority,
    )

    return require_independent_authority(
        envelope_bytes,
        evidence,
        split,
        action=action,
    )


def _source_manifest(
    paths: Mapping[str, Path],
    frames: Mapping[str, pd.DataFrame],
    artifacts: Mapping[str, tuple[dict[str, Any], str]],
    membership: set[str],
) -> dict[str, Any]:
    identity_columns = {"maps": "oe_gameid", "teams": "gameid", "players": "gameid"}
    return {
        "raw_annual_oe_csv": {
            str(year): {"locator": paths[f"annual_{year}"].as_posix(), "raw_sha256": raw_sha256(paths[f"annual_{year}"]), "selected_rows": len(frames[f"annual_{year}"]), "selected_input_sha256": selected_input_sha256(frames[f"annual_{year}"])}
            for year in (2025, 2026)
        },
        "warehouse_parquet": {
            name: {
                "locator": paths[name].as_posix(),
                "raw_sha256": raw_sha256(paths[name]),
                "columns_read": list(frames[name].columns),
                "selected_input_sha256": selected_input_sha256(frames[name]),
                "assigned_population_input_sha256": selected_input_sha256(
                    frames[name][
                        frames[name][identity_columns[name]].astype(str).isin(membership)
                    ]
                ),
            }
            for name in ("maps", "teams", "players")
        },
        "pinned_inputs": {
            name: {"locator": paths[name].as_posix(), "raw_sha256": raw, "payload_sha256": payload["artifact_sha256"], "generator": payload.get("generator")}
            for name, (payload, raw) in artifacts.items()
        },
    }


def build_from_sources(
    *,
    maps_path: Path = DEFAULT_MAPS_PATH,
    team_path: Path = DEFAULT_TEAM_PATH,
    player_path: Path = DEFAULT_PLAYER_PATH,
    annual_paths: Mapping[int, Path] = DEFAULT_ANNUAL_PATHS,
    preflight_path: Path = DEFAULT_PREFLIGHT_PATH,
    proxy_path: Path = DEFAULT_PROXY_PATH,
    crosswalk_path: Path = DEFAULT_CROSSWALK_PATH,
    private_rows_path: Path = DEFAULT_PRIVATE_ROWS_PATH,
    enforce_pins: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = {"maps": maps_path, "teams": team_path, "players": player_path, "preflight": preflight_path, "proxy": proxy_path, "crosswalk": crosswalk_path, **{f"annual_{year}": path for year, path in annual_paths.items()}}
    if any(not p.is_file() or p.is_symlink() for p in paths.values()):
        raise OETargetEvidenceError("every source must be a regular non-symlink file")
    preflight = _artifact_payload(preflight_path, PINNED_PREFLIGHT_PAYLOAD_SHA256, "preflight")
    proxy = _artifact_payload(proxy_path, PINNED_PROXY_PAYLOAD_SHA256, "dependence proxy")
    crosswalk = _artifact_payload(crosswalk_path, PINNED_CROSSWALK_PAYLOAD_SHA256, "champion crosswalk")
    if enforce_pins:
        observed = (raw_sha256(maps_path), raw_sha256(team_path), raw_sha256(player_path), *(raw_sha256(annual_paths[y]) for y in (2025, 2026)), preflight[1], proxy[1], crosswalk[1], preflight[0]["generator"]["executable_dependency_boundary"][0]["raw_sha256"], proxy[0]["generator"]["executable_dependency_boundary"][0]["raw_sha256"], crosswalk[0]["generator"]["executable_dependency_boundary"][0]["raw_sha256"])
        expected = (PINNED_MAPS_RAW_SHA256, PINNED_TEAM_RAW_SHA256, PINNED_PLAYER_RAW_SHA256, PINNED_ANNUAL_RAW_SHA256[2025], PINNED_ANNUAL_RAW_SHA256[2026], PINNED_PREFLIGHT_RAW_SHA256, PINNED_PROXY_RAW_SHA256, PINNED_CROSSWALK_RAW_SHA256, PINNED_PREFLIGHT_GENERATOR_SHA256, PINNED_PROXY_GENERATOR_SHA256, PINNED_CROSSWALK_GENERATOR_SHA256)
        if observed != expected:
            raise OETargetEvidenceError("pinned source or code identity drift")
    maps = pd.read_parquet(maps_path, columns=list(MAP_COLUMNS)).loc[:, MAP_COLUMNS]
    teams = pd.read_parquet(team_path, columns=list(TEAM_COLUMNS)).loc[:, TEAM_COLUMNS]
    players = pd.read_parquet(player_path, columns=list(PLAYER_COLUMNS)).loc[:, PLAYER_COLUMNS]
    schedule = pd.read_parquet(maps_path, columns=["oe_gameid", "date"]).loc[:, ["oe_gameid", "date"]]
    # Outcome-free construction intentionally occurs before any target, duration,
    # champion, or nuisance column is read.
    annual_frames = {}
    annual_membership: set[str] = set()
    for year, path in annual_paths.items():
        raw_frame = pd.read_csv(path, usecols=list(RAW_COLUMNS)).loc[:, RAW_COLUMNS]
        annual_membership.update(raw_frame["gameid"].astype(str))
        annual_frames[year] = raw_frame
    split = build_outcome_free_split(
        schedule,
        proxy[0],
        maps_source={"locator": maps_path.as_posix(), "raw_sha256": raw_sha256(maps_path), "columns_read": ["oe_gameid", "date"], "selected_input_sha256": selected_input_sha256(schedule)},
        proxy_source={"locator": proxy_path.as_posix(), "raw_sha256": proxy[1], "payload_sha256": proxy[0]["artifact_sha256"]},
        annual_membership=annual_membership,
    )
    membership = {row["game_id"] for row in split["assignments"]}
    annual_frames = {
        year: frame[frame["gameid"].astype(str).isin(membership)].copy()
        for year, frame in annual_frames.items()
    }
    frames = {"maps": maps, "teams": teams, "players": players, **{f"annual_{y}": f for y, f in annual_frames.items()}}
    manifest = _source_manifest(
        paths,
        frames,
        {"preflight": preflight, "proxy": proxy, "crosswalk": crosswalk},
        membership,
    )
    evidence = analyze_frames(maps, teams, players, annual_frames, proxy[0], crosswalk[0], split, source_manifest=manifest, private_rows_path=private_rows_path)
    return split, evidence


def write_artifacts(split_path: Path = DEFAULT_SPLIT_PATH, evidence_path: Path = DEFAULT_EVIDENCE_PATH, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    split, evidence = build_from_sources(**kwargs)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_bytes(canonical_bytes(split))
    evidence_path.write_bytes(canonical_bytes(evidence))
    return split, evidence


def load_and_replay_artifacts(split_path: Path = DEFAULT_SPLIT_PATH, evidence_path: Path = DEFAULT_EVIDENCE_PATH, *, source_root: Path = Path.cwd()) -> tuple[dict[str, Any], dict[str, Any]]:
    split_bytes, evidence_bytes = split_path.read_bytes(), evidence_path.read_bytes()
    split, evidence = json.loads(split_bytes), json.loads(evidence_bytes)
    validate_split(split)
    validate_evidence(evidence)
    if split_bytes != canonical_bytes(split) or evidence_bytes != canonical_bytes(evidence):
        raise OETargetEvidenceError("persisted artifacts are not canonical generator bytes")
    def resolve(locator: str) -> Path:
        path = Path(locator)
        return path if path.is_absolute() else source_root / path
    sources = evidence["source"]
    annual = {int(y): resolve(v["locator"]) for y, v in sources["raw_annual_oe_csv"].items()}
    parquet = sources["warehouse_parquet"]
    pins = sources["pinned_inputs"]
    persisted_private = resolve(evidence["private_materialization"]["locator"])
    if not persisted_private.is_file() or persisted_private.is_symlink():
        raise OETargetEvidenceError(
            "persisted private materialization is not a regular non-symlink file"
        )
    if raw_sha256(persisted_private) != evidence["private_materialization"]["raw_sha256"]:
        raise OETargetEvidenceError("persisted private materialization bytes changed")
    persisted_private_frame = pd.read_parquet(persisted_private)
    if (
        selected_input_sha256(persisted_private_frame)
        != evidence["private_materialization"]["logical_rows_sha256"]
    ):
        raise OETargetEvidenceError(
            "persisted private materialization logical rows changed"
        )
    if (
        ordered_input_sha256(persisted_private_frame)
        != evidence["private_materialization"]["ordered_logical_rows_sha256"]
    ):
        raise OETargetEvidenceError(
            "persisted private materialization schema or row order changed"
        )
    with tempfile.TemporaryDirectory() as tmp:
        regenerated_private_path = Path(tmp) / "oe-target-rows.parquet"
        replay_split, replay_evidence = build_from_sources(
            maps_path=resolve(parquet["maps"]["locator"]),
            team_path=resolve(parquet["teams"]["locator"]),
            player_path=resolve(parquet["players"]["locator"]),
            annual_paths=annual,
            preflight_path=resolve(pins["preflight"]["locator"]),
            proxy_path=resolve(pins["proxy"]["locator"]),
            crosswalk_path=resolve(pins["crosswalk"]["locator"]),
            private_rows_path=regenerated_private_path,
        )
        regenerated_private_frame = pd.read_parquet(regenerated_private_path)
        if (
            replay_evidence["private_materialization"]["logical_rows_sha256"]
            != evidence["private_materialization"]["logical_rows_sha256"]
            or selected_input_sha256(regenerated_private_frame)
            != evidence["private_materialization"]["logical_rows_sha256"]
        ):
            raise OETargetEvidenceError(
                "source-backed private logical rows do not match persisted materialization"
            )
        if (
            replay_evidence["private_materialization"][
                "ordered_logical_rows_sha256"
            ]
            != evidence["private_materialization"]["ordered_logical_rows_sha256"]
            or tuple(regenerated_private_frame.columns)
            != tuple(persisted_private_frame.columns)
            or ordered_input_sha256(regenerated_private_frame)
            != evidence["private_materialization"]["ordered_logical_rows_sha256"]
        ):
            raise OETargetEvidenceError(
                "source-backed private schema or row order does not match persisted materialization"
            )
    # Replay reads resolved filesystem paths, while locators are pinned metadata.
    # Normalize metadata only after the resolved files and their bytes were used.
    replay_split["source"] = split["source"]
    replay_split["artifact_sha256"] = canonical_sha256(
        {k: v for k, v in replay_split.items() if k != "artifact_sha256"}
    )
    replay_evidence["source"] = evidence["source"]
    replay_evidence["split_assignment"] = evidence["split_assignment"]
    replay_evidence["private_materialization"] = evidence["private_materialization"]
    replay_evidence["artifact_sha256"] = canonical_sha256({k: v for k, v in replay_evidence.items() if k != "artifact_sha256"})
    if canonical_bytes(replay_split) != split_bytes or canonical_bytes(replay_evidence) != evidence_bytes:
        raise OETargetEvidenceError("source-backed replay does not match persisted canonical payload")
    return split, evidence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args(argv)
    if args.verify_existing:
        split, evidence = load_and_replay_artifacts()
        print(json.dumps({"split_sha256": split["artifact_sha256"], "evidence_sha256": evidence["artifact_sha256"], "replay_verified": True}, sort_keys=True))
    else:
        split, evidence = write_artifacts()
        print(json.dumps({"split_sha256": split["artifact_sha256"], "evidence_sha256": evidence["artifact_sha256"], "maps": evidence["population"]["assigned_maps"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
