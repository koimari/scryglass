"""Outcome-free dependence clusters for temporally adjacent pro maps.

The clusters are a conservative resampling/fold-blocking proxy.  They are not
series identities: no source field in the warehouse authoritatively identifies
every best-of series.  The proxy uses only schedule, competition, patch, source
game counter, and warehouse team identity fields; outcomes are never read.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .representation_assay_preflight import (
    load_and_replay_artifact as load_and_replay_preflight_artifact,
)


SCHEMA_ID = "scryglass.draft-interaction-dependence-cluster-proxy.v1"
GENERATOR_VERSION = "dependence-cluster-proxy-generator.v1"
PINNED_SOURCE_MODE = "pinned_development_source"
NONPROMOTABLE_FIXTURE_SOURCE_MODE = "nonpromotable_synthetic_fixture"
DEFAULT_MAPS_PATH = Path("data/lol/warehouse/parquet/maps.parquet")
DEFAULT_PREFLIGHT_PATH = Path(
    "data/lol/v2/models/draft-interactions/representation-assay-preflight.json"
)
DEFAULT_ARTIFACT_PATH = Path(
    "data/lol/v2/models/draft-interactions/series-cluster-proxy.json"
)
PINNED_PREFLIGHT_PAYLOAD_SHA256 = (
    "ba54faed41716cc537268c6e7eecbaaf9330937014bfd2cd5f9a50f930f92eb4"
)
PINNED_PREFLIGHT_RAW_SHA256 = (
    "e3d2d3c9399e42c7d8d7ad8653698353f0af52e09e1a2e80c33e5b5e369d95c1"
)
PINNED_MAPS_RAW_SHA256 = (
    "04c0cce1d86a4358d9eeb5937f61d5288358953e66c693a1ce88b0b650295d08"
)
PINNED_PLAYER_GAMES_RAW_SHA256 = (
    "3d2a852daa43dfa402e1e48ef11d1a6858b73f2171f0c2febd82b941b19fceee"
)
PINNED_PREFLIGHT_GENERATOR_RAW_SHA256 = (
    "0f80190191dc19d30e98e8d7a0db9963007a93fdf0d2e7f5b184487165e5e3ea"
)
MISSING_SPLIT = "__MISSING_SOURCE_SPLIT__"
MAP_COLUMNS = (
    "oe_gameid",
    "game_uid",
    "url",
    "league",
    "oe_year",
    "split",
    "playoffs",
    "date",
    "game",
    "patch",
    "competition_scope",
    "event_kind",
    "is_international",
    "blue_team_key",
    "red_team_key",
    "source_lp",
    "lp_matched",
    "lp_game_id",
)
SENSITIVITIES = (
    ("gap_6h", 6.0, False),
    ("gap_12h", 12.0, False),
    ("gap_36h", 36.0, False),
    ("calendar_day", 36.0, True),
)


class DependenceClusterProxyError(ValueError):
    """Raised when source rows or artifacts violate the fail-closed contract."""


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def raw_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _present(value: object) -> bool:
    return not pd.isna(value) and str(value).strip() != ""


def _text(value: object) -> str:
    return str(value).strip()


def _canonical_bool(value: object, field: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    raise DependenceClusterProxyError(f"{field} must be boolean")


def _canonical_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not _present(value):
        raise DependenceClusterProxyError(f"{field} is missing")
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise DependenceClusterProxyError(f"{field} is not integral")
    return int(number)


def _patch_token(value: object) -> str:
    if isinstance(value, bool) or not _present(value):
        raise DependenceClusterProxyError("patch is missing")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise DependenceClusterProxyError("patch is invalid")
    centesimal = round(number * 100)
    if abs(number * 100 - centesimal) > 1e-8:
        raise DependenceClusterProxyError("patch is not an exact source token")
    return f"{centesimal / 100:.2f}"


def _timestamp(value: object) -> pd.Timestamp:
    if not _present(value):
        raise DependenceClusterProxyError("timestamp is missing")
    result = pd.Timestamp(value)
    if pd.isna(result):
        raise DependenceClusterProxyError("timestamp is missing")
    if result.tzinfo is not None:
        result = result.tz_convert("UTC").tz_localize(None)
    return result


def _record(row: Mapping[str, object]) -> dict[str, Any]:
    game_id = _text(row["oe_gameid"]) if _present(row["oe_gameid"]) else ""
    game_uid = _text(row["game_uid"]) if _present(row["game_uid"]) else ""
    if not game_id or not game_uid or game_id != game_uid:
        raise DependenceClusterProxyError("ambiguous map identity")
    league = _text(row["league"]).upper() if _present(row["league"]) else ""
    scope = _text(row["competition_scope"]).lower() if _present(row["competition_scope"]) else ""
    event = _text(row["event_kind"]).lower() if _present(row["event_kind"]) else ""
    if not league or not scope or not event:
        raise DependenceClusterProxyError("competition identity is missing")
    left = _text(row["blue_team_key"]) if _present(row["blue_team_key"]) else ""
    right = _text(row["red_team_key"]) if _present(row["red_team_key"]) else ""
    if not left or not right:
        raise DependenceClusterProxyError("team identity is missing")
    if left == right:
        raise DependenceClusterProxyError("self team pair")
    team_pair = sorted((left, right))
    split = _text(row["split"]) if _present(row["split"]) else MISSING_SPLIT
    timestamp = _timestamp(row["date"])
    counter = _canonical_int(row["game"], "source game counter")
    if counter < 1 or counter > 5:
        raise DependenceClusterProxyError("source game counter outside 1..5")
    return {
        "game_id": game_id,
        "context": [
            league,
            _canonical_int(row["oe_year"], "source year"),
            split,
            _canonical_bool(row["playoffs"], "playoffs"),
            scope,
            event,
            _canonical_bool(row["is_international"], "is_international"),
            team_pair[0],
            team_pair[1],
        ],
        "timestamp": timestamp,
        "source_game_counter": counter,
        "source_patch_token": _patch_token(row["patch"]),
        "url": _text(row["url"]) if _present(row["url"]) else None,
        "source_lp": _canonical_bool(row["source_lp"], "source_lp"),
        "lp_matched": _canonical_bool(row["lp_matched"], "lp_matched"),
        "lp_game_id": _text(row["lp_game_id"]) if _present(row["lp_game_id"]) else None,
    }


def selected_input_sha256(frame: pd.DataFrame) -> str:
    values: list[list[object]] = []
    for row in frame.itertuples(index=False, name=None):
        canonical: list[object] = []
        for value in row:
            if pd.isna(value):
                canonical.append(None)
            elif isinstance(value, pd.Timestamp):
                canonical.append(value.isoformat())
            elif isinstance(value, np.generic):
                canonical.append(value.item())
            else:
                canonical.append(value)
        values.append(canonical)
    values.sort(key=canonical_bytes)
    return canonical_sha256({"columns": list(frame.columns), "rows": values})


def _prepare(frame: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if tuple(frame.columns) != MAP_COLUMNS:
        raise DependenceClusterProxyError("map input must use the exact outcome-free allowlist")
    seen_ids: Counter[str] = Counter(
        _text(value) for value in frame["oe_gameid"] if _present(value)
    )
    records: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []
    for row in frame.to_dict("records"):
        candidate = _text(row["oe_gameid"]) if _present(row["oe_gameid"]) else "<missing>"
        if candidate != "<missing>" and seen_ids[candidate] != 1:
            exclusions.append({"game_id": candidate, "reason": "ambiguous_map_identity"})
            continue
        try:
            records.append(_record(row))
        except (DependenceClusterProxyError, TypeError, ValueError):
            exclusions.append({"game_id": candidate, "reason": "invalid_or_ambiguous_identity"})

    collision_groups: dict[tuple[object, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        collision_groups[
            tuple(record["context"])
            + (record["timestamp"].isoformat(), record["source_game_counter"])
        ].append(record)
    collisions = {
        record["game_id"]
        for group in collision_groups.values()
        if len(group) > 1
        for record in group
    }
    if collisions:
        records = [record for record in records if record["game_id"] not in collisions]
        exclusions.extend(
            {"game_id": game_id, "reason": "exact_context_time_game_collision"}
            for game_id in sorted(collisions)
        )
    exclusions.sort(key=lambda item: (item["game_id"], item["reason"]))
    return records, exclusions


def _cluster(
    records: Sequence[dict[str, Any]],
    *,
    gap_hours: float,
    calendar_day: bool = False,
    exact_counter_step: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[object, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[tuple(record["context"])].append(record)

    assignments: list[dict[str, Any]] = []
    cluster_members: dict[str, list[dict[str, Any]]] = {}
    observed_counter_gaps: Counter[int] = Counter()
    for context in sorted(grouped, key=canonical_bytes):
        rows = sorted(
            grouped[context],
            key=lambda record: (record["timestamp"], record["game_id"]),
        )
        previous: dict[str, Any] | None = None
        cluster_id = ""
        for record in rows:
            gap = (
                math.inf
                if previous is None
                else (record["timestamp"] - previous["timestamp"]).total_seconds() / 3600
            )
            continues = (
                previous is not None
                and record["source_game_counter"] > previous["source_game_counter"]
                and (
                    not exact_counter_step
                    or record["source_game_counter"] == previous["source_game_counter"] + 1
                )
                and record["source_patch_token"] == previous["source_patch_token"]
                and 0 < gap <= gap_hours
                and (
                    not calendar_day
                    or record["timestamp"].date() == previous["timestamp"].date()
                )
            )
            # Any non-increasing counter starts a new cluster; no earlier row is
            # searched for a more convenient continuation.
            if not continues:
                cluster_id = "dependence-cluster:" + canonical_sha256(
                    {
                        "context": list(context),
                        "first_game_id": record["game_id"],
                        "first_timestamp": record["timestamp"].isoformat(),
                    }
                )[:24]
                cluster_members[cluster_id] = []
            elif record["source_game_counter"] > previous["source_game_counter"] + 1:
                observed_counter_gaps[
                    record["source_game_counter"] - previous["source_game_counter"]
                ] += 1
            cluster_members[cluster_id].append(record)
            assignments.append(
                {
                    "game_id": record["game_id"],
                    "dependence_cluster_id": cluster_id,
                }
            )
            previous = record

    sizes = Counter(len(members) for members in cluster_members.values())
    spans = [
        (
            max(item["timestamp"] for item in members)
            - min(item["timestamp"] for item in members)
        ).total_seconds()
        / 3600
        for members in cluster_members.values()
    ]
    cross_midnight = sum(
        min(item["timestamp"] for item in members).date()
        != max(item["timestamp"] for item in members).date()
        for members in cluster_members.values()
    )
    diagnostics = {
        "assigned_maps": len(assignments),
        "dependence_clusters": len(cluster_members),
        "cluster_size_distribution": [
            {"cluster_size": size, "clusters": count}
            for size, count in sorted(sizes.items())
        ],
        "maximum_cluster_size": max(sizes, default=0),
        "cross_midnight_clusters": cross_midnight,
        "maximum_span_hours": max(spans, default=0.0),
        "gap_hours": gap_hours,
        "calendar_day_required": calendar_day,
        "exact_counter_step_required": exact_counter_step,
        "continued_counter_gaps": [
            {"counter_step": step, "continuations": count}
            for step, count in sorted(observed_counter_gaps.items())
        ],
    }
    assignments.sort(key=lambda item: item["game_id"])
    diagnostics["partition_sha256"] = partition_sha256(assignments)
    return assignments, diagnostics


def _partition_members(
    assignments: Sequence[Mapping[str, str]],
) -> tuple[tuple[str, ...], ...]:
    members: dict[str, list[str]] = defaultdict(list)
    for assignment in assignments:
        members[assignment["dependence_cluster_id"]].append(assignment["game_id"])
    return tuple(sorted(tuple(sorted(game_ids)) for game_ids in members.values()))


def partition_sha256(assignments: Sequence[Mapping[str, str]]) -> str:
    """Return a cluster-label- and row-order-invariant partition digest."""
    return canonical_sha256({"clusters": _partition_members(assignments)})


def _partition_comparison(
    main_assignments: Sequence[Mapping[str, str]],
    candidate_assignments: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    main_by_game = {
        assignment["game_id"]: assignment["dependence_cluster_id"]
        for assignment in main_assignments
    }
    candidate_by_game = {
        assignment["game_id"]: assignment["dependence_cluster_id"]
        for assignment in candidate_assignments
    }
    if set(main_by_game) != set(candidate_by_game):
        raise DependenceClusterProxyError("sensitivity map population differs from main")

    def member_lookup(
        assignments: Sequence[Mapping[str, str]],
    ) -> dict[str, frozenset[str]]:
        groups: dict[str, set[str]] = defaultdict(set)
        for assignment in assignments:
            groups[assignment["dependence_cluster_id"]].add(assignment["game_id"])
        return {
            game_id: frozenset(groups[assignment["dependence_cluster_id"]])
            for assignment in assignments
            for game_id in (assignment["game_id"],)
        }

    main_members = member_lookup(main_assignments)
    candidate_members = member_lookup(candidate_assignments)
    changed_maps = sorted(
        game_id
        for game_id in main_by_game
        if main_members[game_id] != candidate_members[game_id]
    )
    main_to_candidate: dict[str, set[str]] = defaultdict(set)
    candidate_to_main: dict[str, set[str]] = defaultdict(set)
    for game_id in main_by_game:
        main_to_candidate[main_by_game[game_id]].add(candidate_by_game[game_id])
        candidate_to_main[candidate_by_game[game_id]].add(main_by_game[game_id])

    def pairs(partition: tuple[tuple[str, ...], ...]) -> set[tuple[str, str]]:
        return {
            (members[left], members[right])
            for members in partition
            for left in range(len(members))
            for right in range(left + 1, len(members))
        }

    main_pairs = pairs(_partition_members(main_assignments))
    candidate_pairs = pairs(_partition_members(candidate_assignments))
    main_hash = partition_sha256(main_assignments)
    candidate_hash = partition_sha256(candidate_assignments)
    return {
        "main_partition_sha256": main_hash,
        "candidate_partition_sha256": candidate_hash,
        "partition_equal": main_hash == candidate_hash,
        "main_clusters_split": sum(len(ids) > 1 for ids in main_to_candidate.values()),
        "candidate_clusters_merging_main": sum(
            len(ids) > 1 for ids in candidate_to_main.values()
        ),
        "copartition_pairs_split": len(main_pairs - candidate_pairs),
        "copartition_pairs_joined": len(candidate_pairs - main_pairs),
        "changed_maps": changed_maps,
    }


def _oracle_audit(
    records: Sequence[dict[str, Any]], assignments: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    assigned = {item["game_id"]: item["dependence_cluster_id"] for item in assignments}

    def audit(rows: list[tuple[str, str]]) -> dict[str, Any]:
        oracle_to_proxy: dict[str, set[str]] = defaultdict(set)
        proxy_to_oracle: dict[str, set[str]] = defaultdict(set)
        for game_id, oracle_id in rows:
            oracle_to_proxy[oracle_id].add(assigned[game_id])
            proxy_to_oracle[assigned[game_id]].add(oracle_id)
        return {
            "maps": len(rows),
            "oracle_groups": len(oracle_to_proxy),
            "split_oracle_groups": sum(len(ids) > 1 for ids in oracle_to_proxy.values()),
            "merged_oracle_groups": sum(len(ids) > 1 for ids in proxy_to_oracle.values()),
        }

    lpl: list[tuple[str, str]] = []
    for record in records:
        if record["context"][0] != "LPL":
            continue
        url_match = re.search(r"(?:[?&])bmid=(\d+)(?:&|$)", record["url"] or "")
        prefix_match = re.fullmatch(r"(\d+)-\1_game_[1-5]", record["game_id"])
        if url_match is None or prefix_match is None or url_match.group(1) != prefix_match.group(1):
            raise DependenceClusterProxyError("LPL oracle identity disagreement")
        lpl.append((record["game_id"], url_match.group(1)))

    leaguepedia: list[tuple[str, str]] = []
    for record in records:
        if not record["lp_matched"]:
            continue
        value = record["lp_game_id"]
        match = re.fullmatch(r"(.+)_([1-5])", value or "")
        if match is None:
            raise DependenceClusterProxyError("Leaguepedia oracle identity disagreement")
        leaguepedia.append((record["game_id"], match.group(1)))
    return {
        "lpl_gameid_prefix_and_url_bmid": audit(lpl),
        "leaguepedia_game_id": audit(leaguepedia),
        "interpretation": (
            "zero split/merge disagreement supports this proxy on audited rows only; "
            "it does not establish authoritative series identity elsewhere"
        ),
    }


def _generator_identity() -> dict[str, Any]:
    path = Path(__file__).resolve()
    return {
        "version": GENERATOR_VERSION,
        "executable_dependency_boundary": [
            {
                "locator": "lol_kills/v2/draft/interactions/series_cluster_proxy.py",
                "raw_sha256": raw_sha256(path),
            }
        ],
        "runtime_versions": {
            "numpy": importlib.metadata.version("numpy"),
            "pandas": importlib.metadata.version("pandas"),
            "pyarrow": importlib.metadata.version("pyarrow"),
        },
    }


def analyze_frame(
    frame: pd.DataFrame,
    *,
    maps_locator: str,
    maps_raw_sha256: str,
    preflight_payload: Mapping[str, Any],
    preflight_locator: str,
    preflight_raw_sha256: str,
    source_mode: str = NONPROMOTABLE_FIXTURE_SOURCE_MODE,
) -> dict[str, Any]:
    if source_mode not in {
        PINNED_SOURCE_MODE,
        NONPROMOTABLE_FIXTURE_SOURCE_MODE,
    }:
        raise DependenceClusterProxyError("unknown source mode")
    if preflight_payload.get("artifact_sha256") != PINNED_PREFLIGHT_PAYLOAD_SHA256:
        raise DependenceClusterProxyError("preflight payload pin mismatch")
    if source_mode == PINNED_SOURCE_MODE:
        observed_pins = (
            preflight_raw_sha256,
            maps_raw_sha256,
            preflight_payload["source"]["maps"]["raw_sha256"],
            preflight_payload["source"]["player_games"]["raw_sha256"],
            preflight_payload["generator"]["executable_dependency_boundary"][0][
                "raw_sha256"
            ],
        )
        required_pins = (
            PINNED_PREFLIGHT_RAW_SHA256,
            PINNED_MAPS_RAW_SHA256,
            PINNED_MAPS_RAW_SHA256,
            PINNED_PLAYER_GAMES_RAW_SHA256,
            PINNED_PREFLIGHT_GENERATOR_RAW_SHA256,
        )
        if observed_pins != required_pins:
            raise DependenceClusterProxyError("pinned source or generator identity mismatch")
    records, exclusions = _prepare(frame)
    assignments, main = _cluster(records, gap_hours=36.0)
    main["comparison_to_main"] = _partition_comparison(assignments, assignments)
    sensitivities: dict[str, Any] = {}
    for name, gap, calendar_day in SENSITIVITIES:
        sensitivity_assignments, diagnostic = _cluster(
            records, gap_hours=gap, calendar_day=calendar_day
        )
        diagnostic["comparison_to_main"] = _partition_comparison(
            assignments, sensitivity_assignments
        )
        diagnostic["assignments"] = sensitivity_assignments
        sensitivities[name] = diagnostic
    exact_assignments, exact_step = _cluster(
        records, gap_hours=36.0, exact_counter_step=True
    )
    exact_step["comparison_to_main"] = _partition_comparison(
        assignments, exact_assignments
    )
    exact_step["assignments"] = exact_assignments
    sensitivities["exact_counter_step"] = exact_step
    payload: dict[str, Any] = {
        "schema_id": SCHEMA_ID,
        "development_only": True,
        "outcome_free": True,
        "predictive_authority": False,
        "authoritative_series_identity": False,
        "representation_rank_selected": False,
        "authorizes_model_selection": False,
        "authorizes_publication": False,
        "content_addressing_confers_authority": False,
        "source_mode": source_mode,
        "claim_ceiling": (
            "dependence_cluster_id is an outcome-free resampling and fold-blocking "
            "proxy, never a source-observed series_id"
        ),
        "generator": _generator_identity(),
        "source": {
            "maps": {
                "locator": maps_locator,
                "raw_sha256": maps_raw_sha256,
                "columns_read": list(MAP_COLUMNS),
                "rows": len(frame),
                "selected_input_sha256": selected_input_sha256(frame),
            },
            "pinned_preflight": {
                "locator": preflight_locator,
                "raw_sha256": preflight_raw_sha256,
                "payload_sha256": PINNED_PREFLIGHT_PAYLOAD_SHA256,
                "maps_source": preflight_payload["source"]["maps"],
                "player_games_source": preflight_payload["source"]["player_games"],
                "generator": preflight_payload["generator"],
            },
        },
        "rule": {
            "context_fields": [
                "canonical league",
                "source year",
                "split with explicit missing marker",
                "playoffs",
                "competition_scope",
                "event_kind",
                "is_international",
                "unordered warehouse team_key pair",
            ],
            "sort": ["timestamp", "game_id"],
            "continuation": (
                "strictly increasing source game counter, exact source patch "
                "token, positive elapsed time no greater than 36 hours"
            ),
            "non_increasing_counter": "reset; never search backward for a continuation",
            "identifier_name": "dependence_cluster_id",
        },
        "eligibility": {
            "registry_maps": len(frame),
            "assigned_maps": len(assignments),
            "excluded_maps": len(exclusions),
            "exclusion_ledger": exclusions,
        },
        "assignments": assignments,
        "cluster_arithmetic": main,
        "oracle_audit": _oracle_audit(records, assignments),
        "sensitivities": sensitivities,
        "downstream_contract": {
            "rolling_folds_must_not_split_dependence_cluster": True,
            "bootstrap_sampling_law": {
                "observed_cluster_count_symbol": "K",
                "draws_per_replicate": "K",
                "draw_distribution": (
                    "uniform over the K observed dependence_cluster_id values"
                ),
                "replacement": True,
                "carried_values": [
                    "draw_multiplicity",
                    "cluster_delta_total",
                    "cluster_map_count",
                ],
                "replicate": (
                    "sum(draw_multiplicity * cluster_delta_total) / "
                    "sum(draw_multiplicity * cluster_map_count)"
                ),
                "forbidden": [
                    "equal-weight mean of cluster means",
                    "probability-proportional-to-size cluster draws",
                    "map-count weighting applied a second time to cluster totals",
                ],
            },
        },
    }
    payload["artifact_sha256"] = canonical_sha256(payload)
    validate_artifact(payload)
    if source_mode == PINNED_SOURCE_MODE:
        if (
            sensitivities["gap_36h"]["partition_sha256"]
            != main["partition_sha256"]
            or not sensitivities["gap_36h"]["comparison_to_main"]["partition_equal"]
        ):
            raise DependenceClusterProxyError("36-hour sensitivity differs from main")
        expected = {
            "assigned_maps": 12708,
            "dependence_clusters": 6143,
            "cluster_size_distribution": [
                {"cluster_size": 1, "clusters": 2323},
                {"cluster_size": 2, "clusters": 1873},
                {"cluster_size": 3, "clusters": 1395},
                {"cluster_size": 4, "clusters": 306},
                {"cluster_size": 5, "clusters": 246},
            ],
            "cross_midnight_clusters": 29,
        }
        observed = {key: main[key] for key in expected}
        if observed != expected or abs(main["maximum_span_hours"] - 24.030277777777776) > 1e-12:
            raise DependenceClusterProxyError("pinned canonical cluster audit mismatch")
        expected_oracles = {
            "lpl_gameid_prefix_and_url_bmid": (2730, 1059, 0, 0),
            "leaguepedia_game_id": (1074, 469, 0, 0),
        }
        for name, values in expected_oracles.items():
            audit = payload["oracle_audit"][name]
            observed_oracle = (
                audit["maps"],
                audit["oracle_groups"],
                audit["split_oracle_groups"],
                audit["merged_oracle_groups"],
            )
            if observed_oracle != values:
                raise DependenceClusterProxyError("pinned oracle audit mismatch")
    return payload


def build_from_parquet(
    maps_path: Path = DEFAULT_MAPS_PATH,
    preflight_path: Path = DEFAULT_PREFLIGHT_PATH,
    *,
    maps_locator: str | None = None,
    preflight_locator: str | None = None,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Build only from the exact pinned empirical sources."""
    root = Path.cwd() if source_root is None else source_root
    preflight = load_and_replay_preflight_artifact(
        preflight_path, source_root=root
    )
    frame = pd.read_parquet(maps_path, columns=list(MAP_COLUMNS))
    return analyze_frame(
        frame,
        maps_locator=maps_locator if maps_locator is not None else str(maps_path),
        maps_raw_sha256=raw_sha256(maps_path),
        preflight_payload=preflight,
        preflight_locator=(
            preflight_locator if preflight_locator is not None else str(preflight_path)
        ),
        preflight_raw_sha256=raw_sha256(preflight_path),
        source_mode=PINNED_SOURCE_MODE,
    )


def build_nonpromotable_fixture_from_parquet(
    maps_path: Path,
    preflight_path: Path,
    *,
    maps_locator: str | None = None,
    preflight_locator: str | None = None,
) -> dict[str, Any]:
    """Build a synthetic fixture that production replay will always reject."""
    frame = pd.read_parquet(maps_path, columns=list(MAP_COLUMNS))
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    return analyze_frame(
        frame,
        maps_locator=maps_locator if maps_locator is not None else str(maps_path),
        maps_raw_sha256=raw_sha256(maps_path),
        preflight_payload=preflight,
        preflight_locator=(
            preflight_locator if preflight_locator is not None else str(preflight_path)
        ),
        preflight_raw_sha256=raw_sha256(preflight_path),
        source_mode=NONPROMOTABLE_FIXTURE_SOURCE_MODE,
    )


def validate_artifact(payload: Mapping[str, Any]) -> None:
    unsigned = dict(payload)
    claimed = unsigned.pop("artifact_sha256", None)
    if claimed != canonical_sha256(unsigned):
        raise DependenceClusterProxyError("artifact hash does not match canonical payload")
    for field in (
        "development_only",
        "outcome_free",
    ):
        if payload.get(field) is not True:
            raise DependenceClusterProxyError(f"{field} contract violated")
    for field in (
        "predictive_authority",
        "authoritative_series_identity",
        "representation_rank_selected",
        "authorizes_model_selection",
        "authorizes_publication",
        "content_addressing_confers_authority",
    ):
        if payload.get(field) is not False:
            raise DependenceClusterProxyError("authority contract violated")
    if payload.get("source_mode") not in {
        PINNED_SOURCE_MODE,
        NONPROMOTABLE_FIXTURE_SOURCE_MODE,
    }:
        raise DependenceClusterProxyError("source mode contract violated")

    def validate_partition(
        value: object, *, label: str
    ) -> tuple[list[Mapping[str, str]], Counter[str]]:
        if not isinstance(value, list):
            raise DependenceClusterProxyError(f"{label} assignments must be a list")
        checked: list[Mapping[str, str]] = []
        for item in value:
            if (
                not isinstance(item, Mapping)
                or set(item) != {"game_id", "dependence_cluster_id"}
                or not isinstance(item["game_id"], str)
                or not item["game_id"]
                or item["game_id"] != item["game_id"].strip()
                or any(character.isspace() for character in item["game_id"])
                or not isinstance(item["dependence_cluster_id"], str)
                or not item["dependence_cluster_id"]
            ):
                raise DependenceClusterProxyError(
                    f"{label} assignment shape is invalid"
                )
            checked.append(item)
        if len({item["game_id"] for item in checked}) != len(checked):
            raise DependenceClusterProxyError(
                f"{label} assignment game IDs are not unique"
            )
        return checked, Counter(
            item["dependence_cluster_id"] for item in checked
        )

    def require_structural_arithmetic(
        diagnostic: Mapping[str, Any],
        partition: Sequence[Mapping[str, str]],
        counts: Counter[str],
        *,
        label: str,
    ) -> None:
        expected_distribution = [
            {"cluster_size": size, "clusters": cluster_count}
            for size, cluster_count in sorted(Counter(counts.values()).items())
        ]
        expected = {
            "assigned_maps": len(partition),
            "dependence_clusters": len(counts),
            "cluster_size_distribution": expected_distribution,
            "maximum_cluster_size": max(counts.values(), default=0),
            "partition_sha256": partition_sha256(partition),
        }
        if diagnostic.get("assigned_maps") != expected["assigned_maps"]:
            raise DependenceClusterProxyError(
                f"{label} assignment arithmetic mismatch"
            )
        if diagnostic.get("partition_sha256") != expected["partition_sha256"]:
            raise DependenceClusterProxyError(
                f"{label} partition structural cluster arithmetic mismatch"
            )
        if any(
            diagnostic.get(key) != value
            for key, value in expected.items()
            if key not in {"assigned_maps", "partition_sha256"}
        ):
            raise DependenceClusterProxyError(
                f"{label} structural cluster arithmetic mismatch"
            )
        if expected["maximum_cluster_size"] > 5:
            raise DependenceClusterProxyError(
                f"{label} cluster exceeds best-of-five ceiling"
            )

    assignments = payload["assignments"]
    assignments, counts = validate_partition(assignments, label="main")
    arithmetic = payload["cluster_arithmetic"]
    if not isinstance(arithmetic, Mapping):
        raise DependenceClusterProxyError("cluster arithmetic must be an object")
    require_structural_arithmetic(
        arithmetic, assignments, counts, label="main"
    )
    expected_main_comparison = _partition_comparison(assignments, assignments)
    if arithmetic.get("comparison_to_main") != expected_main_comparison:
        raise DependenceClusterProxyError("main partition comparison mismatch")

    eligibility = payload["eligibility"]
    source_rows = payload.get("source", {}).get("maps", {}).get("rows")
    if (
        not isinstance(source_rows, int)
        or isinstance(source_rows, bool)
        or source_rows < 0
        or eligibility.get("registry_maps") != source_rows
    ):
        raise DependenceClusterProxyError(
            "eligibility arithmetic mismatch: registry does not match source map rows"
        )
    if (
        eligibility.get("assigned_maps") + eligibility.get("excluded_maps")
        != eligibility.get("registry_maps")
        or eligibility.get("assigned_maps") != len(assignments)
    ):
        raise DependenceClusterProxyError("eligibility arithmetic mismatch")
    ledger = eligibility.get("exclusion_ledger")
    if not isinstance(ledger, list) or len(ledger) != eligibility.get("excluded_maps"):
        raise DependenceClusterProxyError("exclusion ledger arithmetic mismatch")
    allowed_exclusion_reasons = {
        "ambiguous_map_identity",
        "invalid_or_ambiguous_identity",
        "exact_context_time_game_collision",
    }
    ledger_pairs: list[tuple[str, str]] = []
    for item in ledger:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"game_id", "reason"}
            or not isinstance(item["game_id"], str)
            or not item["game_id"]
            or item["game_id"] != item["game_id"].strip()
            or any(character.isspace() for character in item["game_id"])
            or item["reason"] not in allowed_exclusion_reasons
        ):
            raise DependenceClusterProxyError("exclusion ledger entry is invalid")
        ledger_pairs.append((item["game_id"], item["reason"]))
    ledger_ids = [game_id for game_id, _ in ledger_pairs]
    assigned_ids = {item["game_id"] for item in assignments}
    if (
        len(set(ledger_ids)) != len(ledger_ids)
        or assigned_ids.intersection(ledger_ids)
        or ledger_pairs != sorted(ledger_pairs)
    ):
        raise DependenceClusterProxyError(
            "exclusion ledger must be unique, disjoint, and canonical"
        )

    main_hash = arithmetic["partition_sha256"]
    sensitivities = payload.get("sensitivities")
    expected_sensitivity_rules = {
        name: {
            "gap_hours": gap,
            "calendar_day_required": calendar_day,
            "exact_counter_step_required": False,
        }
        for name, gap, calendar_day in SENSITIVITIES
    }
    expected_sensitivity_rules["exact_counter_step"] = {
        "gap_hours": 36.0,
        "calendar_day_required": False,
        "exact_counter_step_required": True,
    }
    if (
        not isinstance(sensitivities, Mapping)
        or set(sensitivities) != set(expected_sensitivity_rules)
    ):
        raise DependenceClusterProxyError("sensitivity manifest mismatch")
    for name, expected_rule in expected_sensitivity_rules.items():
        diagnostic = sensitivities[name]
        if not isinstance(diagnostic, Mapping):
            raise DependenceClusterProxyError(f"{name} diagnostic is invalid")
        candidate, candidate_counts = validate_partition(
            diagnostic.get("assignments"), label=name
        )
        if {item["game_id"] for item in candidate} != assigned_ids:
            raise DependenceClusterProxyError(
                f"{name} sensitivity map population differs from main"
            )
        require_structural_arithmetic(
            diagnostic,
            candidate,
            candidate_counts,
            label=name,
        )
        if any(diagnostic.get(key) != value for key, value in expected_rule.items()):
            raise DependenceClusterProxyError(f"{name} rule contract mismatch")
        expected_comparison = _partition_comparison(assignments, candidate)
        if diagnostic.get("comparison_to_main") != expected_comparison:
            raise DependenceClusterProxyError(
                f"{name} partition comparison mismatch"
            )
    if sensitivities["gap_36h"]["partition_sha256"] != main_hash:
        raise DependenceClusterProxyError("gap_36h partition must equal main")


def write_artifact(
    artifact_path: Path = DEFAULT_ARTIFACT_PATH,
    *,
    maps_path: Path = DEFAULT_MAPS_PATH,
    preflight_path: Path = DEFAULT_PREFLIGHT_PATH,
) -> dict[str, Any]:
    payload = build_from_parquet(maps_path, preflight_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(canonical_bytes(payload))
    return payload


def write_nonpromotable_fixture_artifact(
    artifact_path: Path,
    *,
    maps_path: Path,
    preflight_path: Path,
) -> dict[str, Any]:
    payload = build_nonpromotable_fixture_from_parquet(maps_path, preflight_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(canonical_bytes(payload))
    return payload


def _load_canonical_artifact(path: Path) -> tuple[dict[str, Any], bytes]:
    persisted = path.read_bytes()
    stored = json.loads(persisted)
    if not isinstance(stored, dict):
        raise DependenceClusterProxyError("persisted artifact must be an object")
    validate_artifact(stored)
    if persisted != canonical_bytes(stored):
        raise DependenceClusterProxyError("persisted artifact bytes are not canonical")
    if stored["generator"] != _generator_identity():
        raise DependenceClusterProxyError("proxy generator identity mismatch")
    return stored, persisted


def load_and_replay_artifact(
    path: Path, *, source_root: Path | None = None
) -> dict[str, Any]:
    """Strictly replay the exact pinned empirical artifact.

    This production loader has no downgrade path.  Synthetic fixtures must use
    ``load_and_replay_nonpromotable_fixture_artifact``.
    """
    stored, persisted = _load_canonical_artifact(path)
    if stored["source_mode"] != PINNED_SOURCE_MODE:
        raise DependenceClusterProxyError("production replay rejects nonpinned source mode")
    source = stored["source"]
    preflight_source = source["pinned_preflight"]
    exact_embedded_pins = (
        source["maps"]["raw_sha256"],
        preflight_source["raw_sha256"],
        preflight_source["payload_sha256"],
        preflight_source["maps_source"]["raw_sha256"],
        preflight_source["player_games_source"]["raw_sha256"],
        preflight_source["generator"]["executable_dependency_boundary"][0][
            "raw_sha256"
        ],
    )
    required_embedded_pins = (
        PINNED_MAPS_RAW_SHA256,
        PINNED_PREFLIGHT_RAW_SHA256,
        PINNED_PREFLIGHT_PAYLOAD_SHA256,
        PINNED_MAPS_RAW_SHA256,
        PINNED_PLAYER_GAMES_RAW_SHA256,
        PINNED_PREFLIGHT_GENERATOR_RAW_SHA256,
    )
    if exact_embedded_pins != required_embedded_pins:
        raise DependenceClusterProxyError("embedded production pins mismatch")
    root = Path.cwd() if source_root is None else source_root

    def resolve(locator: object, label: str) -> Path:
        if not isinstance(locator, str) or not locator:
            raise DependenceClusterProxyError(f"{label} locator is invalid")
        candidate = Path(locator)
        resolved = candidate if candidate.is_absolute() else root / candidate
        if not resolved.is_file() or resolved.is_symlink():
            raise DependenceClusterProxyError(f"{label} is not a regular source file")
        return resolved

    maps_path = resolve(source["maps"]["locator"], "maps")
    preflight_path = resolve(preflight_source["locator"], "preflight")
    if raw_sha256(maps_path) != PINNED_MAPS_RAW_SHA256:
        raise DependenceClusterProxyError("pinned maps bytes changed")
    if raw_sha256(preflight_path) != PINNED_PREFLIGHT_RAW_SHA256:
        raise DependenceClusterProxyError("pinned preflight bytes changed")
    replayed_preflight = load_and_replay_preflight_artifact(
        preflight_path, source_root=root
    )
    if (
        replayed_preflight["artifact_sha256"] != PINNED_PREFLIGHT_PAYLOAD_SHA256
        or replayed_preflight["source"]["maps"] != preflight_source["maps_source"]
        or replayed_preflight["source"]["player_games"]
        != preflight_source["player_games_source"]
        or replayed_preflight["generator"] != preflight_source["generator"]
    ):
        raise DependenceClusterProxyError("strict preflight replay differs from embedded pins")
    replayed = build_from_parquet(
        maps_path,
        preflight_path,
        maps_locator=source["maps"]["locator"],
        preflight_locator=preflight_source["locator"],
        source_root=root,
    )
    if canonical_bytes(replayed) != persisted:
        raise DependenceClusterProxyError("source-backed replay does not match artifact")
    return stored


def load_and_replay_nonpromotable_fixture_artifact(path: Path) -> dict[str, Any]:
    """Replay a clearly nonpromotable synthetic fixture."""
    stored, persisted = _load_canonical_artifact(path)
    if stored["source_mode"] != NONPROMOTABLE_FIXTURE_SOURCE_MODE:
        raise DependenceClusterProxyError("fixture loader requires nonpromotable mode")
    replayed = build_nonpromotable_fixture_from_parquet(
        Path(stored["source"]["maps"]["locator"]),
        Path(stored["source"]["pinned_preflight"]["locator"]),
    )
    if canonical_bytes(replayed) != persisted:
        raise DependenceClusterProxyError("fixture source-backed replay does not match")
    return stored


def assert_rolling_folds_do_not_split_cluster(
    assignments: Sequence[Mapping[str, str]],
    fold_by_game_id: Mapping[str, object],
) -> None:
    folds: dict[str, set[object]] = defaultdict(set)
    for assignment in assignments:
        game_id = assignment["game_id"]
        if game_id not in fold_by_game_id:
            raise DependenceClusterProxyError("fold assignment is missing a map")
        folds[assignment["dependence_cluster_id"]].add(fold_by_game_id[game_id])
    if any(len(values) != 1 for values in folds.values()):
        raise DependenceClusterProxyError("rolling folds split a dependence cluster")


def map_weighted_cluster_bootstrap_replicate(
    cluster_summaries: Mapping[str, Mapping[str, object]],
    *,
    seed: int,
) -> dict[str, Any]:
    """Draw one exact map-weighted cluster-bootstrap replicate.

    Exactly K cluster IDs are sampled uniformly with replacement from the K
    observed IDs.  Cluster totals and counts are carried through the draw;
    cluster means are never averaged and cluster size never affects sampling
    probability.
    """
    if not cluster_summaries:
        raise DependenceClusterProxyError("bootstrap requires observed clusters")
    cluster_ids = sorted(cluster_summaries)
    parsed: dict[str, tuple[float, int]] = {}
    for cluster_id in cluster_ids:
        summary = cluster_summaries[cluster_id]
        total = summary.get("cluster_delta_total")
        count = summary.get("cluster_map_count")
        if (
            isinstance(total, bool)
            or not isinstance(total, (int, float, np.integer, np.floating))
            or not math.isfinite(float(total))
        ):
            raise DependenceClusterProxyError("cluster delta total must be finite")
        if (
            isinstance(count, bool)
            or not isinstance(count, (int, np.integer))
            or int(count) <= 0
        ):
            raise DependenceClusterProxyError("cluster map count must be positive")
        parsed[cluster_id] = (float(total), int(count))

    k = len(cluster_ids)
    rng = np.random.default_rng(seed)
    drawn_indices = rng.integers(0, k, size=k)
    drawn_ids = [cluster_ids[int(index)] for index in drawn_indices]
    multiplicities = Counter(drawn_ids)
    sampled_total = sum(
        multiplicity * parsed[cluster_id][0]
        for cluster_id, multiplicity in multiplicities.items()
    )
    sampled_count = sum(
        multiplicity * parsed[cluster_id][1]
        for cluster_id, multiplicity in multiplicities.items()
    )
    observed_total = sum(total for total, _ in parsed.values())
    observed_count = sum(count for _, count in parsed.values())
    return {
        "sampling_law": (
            "draw K observed dependence cluster IDs uniformly with replacement"
        ),
        "observed_cluster_count": k,
        "draw_count": k,
        "drawn_cluster_ids": drawn_ids,
        "draw_multiplicities": [
            {
                "dependence_cluster_id": cluster_id,
                "draw_multiplicity": multiplicities[cluster_id],
                "cluster_delta_total": parsed[cluster_id][0],
                "cluster_map_count": parsed[cluster_id][1],
            }
            for cluster_id in sorted(multiplicities)
        ],
        "observed_map_weighted_point_estimate": observed_total / observed_count,
        "sampled_delta_total": sampled_total,
        "sampled_map_count": sampled_count,
        "replicate": sampled_total / sampled_count,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps", type=Path, default=DEFAULT_MAPS_PATH)
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args(argv)
    if args.replay:
        load_and_replay_artifact(args.output)
    else:
        write_artifact(args.output, maps_path=args.maps, preflight_path=args.preflight)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
