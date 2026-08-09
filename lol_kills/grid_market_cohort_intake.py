"""Resume a compact, event/state-only GRID map intake.

This is a data-foundation collector. It does not estimate fight odds, map-win
probabilities, or any causal objective effect. It retains only the immutable
GRID event archives and a manifest of maps whose terminal and first-Grub
checkpoint states are present.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from lol_kills.grid_market_cohort import (
    LEAGUE_TOURNAMENTS,
    MANIFEST_SCHEMA_VERSION,
    GridIngestError,
    GridMarketCohortError,
    _CentralDataClient,
    _api_key,
    _catalog_provenance,
    _download_file,
    _existing_download,
    _file_list,
    _hash,
    _iter_grid_events,
    _utc_now,
    discover_series,
)


DEFAULT_START = "2026-01-14T00:00:00Z"
DEFAULT_END = "2026-07-31T23:59:59Z"
GRUB_TYPES = {"player-completed-slayVoidGrub", "team-completed-slayVoidGrub"}


def _game(event: Mapping[str, Any], game_id: str) -> Mapping[str, Any] | None:
    for game in (event.get("seriesState") or {}).get("games") or []:
        if isinstance(game, Mapping) and str(game.get("id") or "") == game_id:
            return game
    return None


def _clock(event: Mapping[str, Any], game_id: str) -> int | None:
    game = _game(event, game_id)
    value = (game or {}).get("clock") or {}
    seconds = value.get("currentSeconds")
    return seconds if isinstance(seconds, int) and not isinstance(seconds, bool) else None


def _event_order(event: Mapping[str, Any], game_id: str) -> tuple[int, int, str]:
    sequence = event.get("_provider_sequence")
    sequence_value = sequence if isinstance(sequence, int) else 2**63 - 1
    event_clock = _clock(event, game_id)
    clock_value = event_clock if event_clock is not None else 2**63 - 1
    return (sequence_value, clock_value, str(event.get("id") or ""))


def _team_snapshot(game: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(game, Mapping):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for team in game.get("teams") or []:
        if not isinstance(team, Mapping) or team.get("id") is None:
            continue
        money = team.get("money")
        net_worth = team.get("netWorth")
        if (
            isinstance(money, bool)
            or not isinstance(money, (int, float))
            or isinstance(net_worth, bool)
            or not isinstance(net_worth, (int, float))
        ):
            continue
        result[str(team["id"])] = {
            "money": money,
            "netWorth": net_worth,
            "side": team.get("side"),
        }
    return result if len(result) == 2 else {}


def _terminal(game: Mapping[str, Any] | None) -> bool:
    if not isinstance(game, Mapping) or game.get("finished") is not True:
        return False
    teams = game.get("teams") or []
    if len(teams) != 2 or {str(row.get("side") or "") for row in teams} != {"blue", "red"}:
        return False
    if any(len(row.get("players") or []) != 5 for row in teams if isinstance(row, Mapping)):
        return False
    winners = [row for row in teams if isinstance(row, Mapping) and row.get("won") is True]
    return len(winners) == 1


def _sequence_receipt(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    sequences = sorted(
        {
            int(event["_provider_sequence"])
            for event in events
            if isinstance(event.get("_provider_sequence"), int)
            and not isinstance(event.get("_provider_sequence"), bool)
        }
    )
    gaps = [
        [left + 1, right - 1]
        for left, right in zip(sequences, sequences[1:])
        if right > left + 1
    ]
    return {
        "provider_sequence_min": sequences[0] if sequences else None,
        "provider_sequence_max": sequences[-1] if sequences else None,
        "provider_sequence_gap_count": sum(right - left + 1 for left, right in gaps),
        "provider_sequence_gap_intervals": gaps,
        "sequence_gap_interpretation": "descriptive event feed; no missing events imputed",
    }


def _event_path(key: str, series_id: str, root: Path) -> tuple[Path, dict[str, Any]]:
    existing = _existing_download(series_id=series_id, file_id="events-grid", root=root)
    if existing is not None:
        return Path(str(existing["raw_path"])), existing
    rows = [
        row
        for row in _file_list(key, series_id)
        if str(row.get("id") or "") == "events-grid"
        and str(row.get("status") or "") == "ready"
    ]
    if len(rows) != 1:
        raise GridMarketCohortError("events-grid is not uniquely ready")
    receipt = _download_file(key=key, series_id=series_id, file_row=rows[0], root=root)
    return Path(str(receipt["raw_path"])), receipt


def _series_maps(
    *,
    path: Path,
    series: Mapping[str, Any],
    league: str,
) -> list[dict[str, Any]]:
    events = list(_iter_grid_events(path))
    by_game: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        for game in (event.get("seriesState") or {}).get("games") or []:
            if isinstance(game, Mapping) and game.get("id"):
                by_game.setdefault(str(game["id"]), []).append(event)
    sequence = _sequence_receipt(events)
    rows: list[dict[str, Any]] = []
    for game_id, game_events in sorted(by_game.items()):
        ordered = sorted(game_events, key=lambda event: _event_order(event, game_id))
        grub_events = [event for event in ordered if event.get("type") in GRUB_TYPES]
        if not grub_events:
            continue
        terminal_game = next(
            (
                _game(event, game_id)
                for event in reversed(ordered)
                if _terminal(_game(event, game_id))
            ),
            None,
        )
        if terminal_game is None:
            continue
        first_grub = min(grub_events, key=lambda event: _event_order(event, game_id))
        first_index = ordered.index(first_grub)
        grub_clock = _clock(first_grub, game_id)
        at = _team_snapshot(_game(first_grub, game_id))
        pre = None
        post = None
        pre_event = None
        post_event = None
        for event in reversed(ordered[:first_index]):
            snapshot = _team_snapshot(_game(event, game_id))
            if snapshot and _clock(event, game_id) is not None and (
                grub_clock is None or _clock(event, game_id) < grub_clock
            ):
                pre, pre_event = snapshot, event
                break
        for event in ordered[first_index + 1 :]:
            snapshot = _team_snapshot(_game(event, game_id))
            if snapshot and _clock(event, game_id) is not None and (
                grub_clock is None or _clock(event, game_id) >= grub_clock
            ):
                post, post_event = snapshot, event
                break
        if not pre or not at or not post:
            continue
        rows.append(
            {
                "league": league,
                "provider_series_id": str(series.get("id") or ""),
                "provider_game_id": game_id,
                "series_start_time_scheduled": series.get("startTimeScheduled"),
                "provider_tournament": series.get("tournament") or {},
                "first_grub_clock_seconds": grub_clock,
                "grub_event_count": len(grub_events),
                "checkpoint_state_coverage": {
                    "pre_first_grub": True,
                    "at_first_grub": True,
                    "post_first_grub": True,
                    "pre_clock_seconds": _clock(pre_event, game_id),
                    "at_clock_seconds": grub_clock,
                    "post_clock_seconds": _clock(post_event, game_id),
                },
                "terminal": {
                    "finished": terminal_game.get("finished"),
                    "teams": [
                        {
                            "id": team.get("id"),
                            "side": team.get("side"),
                            "won": team.get("won"),
                            "player_count": len(team.get("players") or []),
                        }
                        for team in terminal_game.get("teams") or []
                    ],
                },
                "event_sequence": sequence,
            }
        )
    return rows


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(
    *,
    root: Path,
    target: int,
    soft_quota: int,
    minimum_per_league: int,
    maximum_series_per_league: int,
    start_time: str,
    end_time: str,
    leagues: Sequence[str],
) -> dict[str, Any]:
    key = _api_key()
    client = _CentralDataClient(key)
    catalog = _catalog_provenance()
    all_rows: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    discovery_receipts: list[dict[str, Any]] = []
    progress_path = root / "manifests" / "intake-progress.json"
    for league in leagues:
        candidates, discovery = discover_series(
            client,
            league=league,
            tournament_ids=LEAGUE_TOURNAMENTS[league],
            maximum_rows=maximum_series_per_league,
            start_time=start_time,
            end_time=end_time,
        )
        discovery_receipts.append(discovery)
        print(f"discovered league={league} series={len(candidates)}", flush=True)
        available_for_league = 0
        for index, series in enumerate(candidates, 1):
            if available_for_league >= soft_quota:
                break
            series_id = str(series.get("id") or "")
            try:
                events_path, receipt = _event_path(key, series_id, root)
                rows = _series_maps(path=events_path, series=series, league=league)
                all_rows.extend(rows)
                available_for_league += len(rows)
            except (GridMarketCohortError, GridIngestError, OSError, ValueError) as exc:
                quarantined.append(
                    {
                        "league": league,
                        "provider_series_id": series_id,
                        "blockers": [f"processing.{type(exc).__name__}"],
                    }
                )
            if index % 5 == 0 or available_for_league >= soft_quota:
                selected_so_far = min(len(all_rows), target)
                remaining = max(0, target - selected_so_far)
                print(
                    f"progress league={league} series={index}/{len(candidates)} "
                    f"league_maps={available_for_league} all_maps={len(all_rows)} "
                    f"remaining_target={remaining}",
                    flush=True,
                )
                _write_json(
                    progress_path,
                    {
                        "generated_at": _utc_now(),
                        "league": league,
                        "processed_series": index,
                        "available_maps": len(all_rows),
                        "available_maps_by_league": dict(Counter(row["league"] for row in all_rows)),
                        "quarantined_records": len(quarantined),
                    },
                )
    counts = Counter(row["league"] for row in all_rows)
    if any(counts.get(league, 0) < minimum_per_league for league in leagues):
        missing = {league: counts.get(league, 0) for league in leagues if counts.get(league, 0) < minimum_per_league}
        raise GridMarketCohortError(f"league minimum not met: {missing}")
    # Deterministic quota selection: reserve the minimum per league, then fill
    # the exact target chronologically across the remaining eligible maps.
    all_rows.sort(key=lambda row: (str(row.get("series_start_time_scheduled") or ""), row["provider_game_id"]))
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for league in leagues:
        league_rows = [row for row in all_rows if row["league"] == league]
        for row in league_rows[:minimum_per_league]:
            if row["provider_game_id"] not in selected_ids:
                selected.append(row)
                selected_ids.add(row["provider_game_id"])
    for row in all_rows:
        if len(selected) >= target:
            break
        if row["provider_game_id"] not in selected_ids:
            selected.append(row)
            selected_ids.add(row["provider_game_id"])
    if len(selected) < target:
        raise GridMarketCohortError(f"eligible maps below target: {len(selected)} < {target}")
    selected = selected[:target]
    coverage = Counter(row["league"] for row in selected)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "scope": {
            "privacy": "private_personal_research_only",
            "data_foundation_only": True,
            "models_trained": False,
            "probabilities_authorized": False,
            "causal_claims_authorized": False,
            "publication_authorized": False,
        },
        "configuration": {
            "registry_version": "scryglass.grid-major-league-season-registry.v3",
            "target_maps": target,
            "soft_quota_per_league": soft_quota,
            "minimum_per_league": minimum_per_league,
            "maximum_series_per_league": maximum_series_per_league,
            "leagues": list(leagues),
            "start_time": start_time,
            "end_time": end_time,
            "exact_tournament_registry": {league: list(LEAGUE_TOURNAMENTS[league]) for league in leagues},
            "checkpoint_rule": "pre and post snapshots are event-feed states around the first Grub event; no values are imputed",
        },
        "catalog_provenance": catalog,
        "discovery_receipts": discovery_receipts,
        "coverage": {
            "selected_maps_total": len(selected),
            "selected_maps_by_league": dict(sorted(coverage.items())),
            "eligible_maps_before_selection": len(all_rows),
            "eligible_maps_by_league_before_selection": dict(sorted(counts.items())),
            "quarantined_records_total": len(quarantined),
        },
        "selected_games": selected,
        "quarantine": quarantined,
        "authority": {
            "cohort_eligible_for_checkpoint_audit": True,
            "fight_probability_estimation": "not performed",
            "causal_effect_estimation": "not authorized",
            "model_or_betting_authority": "unavailable",
        },
    }
    manifest["manifest_sha256"] = _hash(manifest)
    manifest_path = root / "manifests" / f"map-intake-{manifest['manifest_sha256']}.json"
    _write_json(manifest_path, manifest)
    print(json.dumps({"manifest_path": str(manifest_path), "manifest_sha256": manifest["manifest_sha256"], "coverage": manifest["coverage"]}, sort_keys=True), flush=True)
    return {**manifest, "manifest_path": str(manifest_path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--target", type=int, default=1400)
    parser.add_argument("--soft-quota", type=int, default=280)
    parser.add_argument("--minimum-per-league", type=int, default=200)
    parser.add_argument("--maximum-series-per-league", type=int, default=500)
    parser.add_argument("--start-time", default=DEFAULT_START)
    parser.add_argument("--end-time", default=DEFAULT_END)
    args = parser.parse_args()
    run(
        root=args.root,
        target=args.target,
        soft_quota=args.soft_quota,
        minimum_per_league=args.minimum_per_league,
        maximum_series_per_league=args.maximum_series_per_league,
        start_time=args.start_time,
        end_time=args.end_time,
        leagues=tuple(LEAGUE_TOURNAMENTS),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
