"""Export the small public ratings payload (2025–2026 default).

Usage:
  python3 -m lol_kills.export.public_pack
  python3 -m lol_kills.export.public_pack --years 2025,2026 --out output/public_pack
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from lol_kills.export import pack_spec as spec
from lol_kills.export.leaderboards import build_leaderboards
from lol_kills.export.pack_records import (
    build_maps_frame_from_team_games,
    build_player_champion_records,
    build_profile_records,
    build_player_records,
    build_team_records,
    filter_public_team_rating_maps,
    merge_accepted_profile_games,
    public_team_affiliation,
    summarize_player_affiliations,
)
from lol_kills.export.player_metadata import build_player_metadata
from lol_kills.export.public_schedule import (
    PublicScheduleError,
    build_public_schedule,
    validate_public_schedule,
)
from lol_kills.etl.competition import TAXONOMY_VERSION, canonicalize_competition_frame, competition_tier
from lol_kills.etl.source_keys import canonical_source_game_key
from lol_kills.refresh_ledger import worker_commit as resolve_worker_commit
from lol_kills.ratings.dual_elo import build_dual_ratings, lineup_hashes_from_players
from lol_kills.ratings.evidence import attach_player_evidence, attach_team_evidence
from lol_kills.ratings.hierarchical_bt import build_team_weekly_ranks, fit_hierarchical_bt
from lol_kills.ratings.player_elo import (
    build_maps_frame_from_players,
    build_player_ratings,
    build_player_weekly_ranks,
)
from lol_kills.research.composition_signal import (
    MODEL_VERSION,
    CompositionSignalError,
    build_composition_games,
    evaluate_composition_signal,
    score_games_temporally,
    write_evaluation_report,
    validate_public_signal,
)

ROOT = Path(__file__).resolve().parents[2]
WAREHOUSE = ROOT / "data" / "lol" / "warehouse" / "parquet"
LIVE_WAREHOUSE = WAREHOUSE / "oe_live"
FEATURES = ROOT / "data" / "lol" / "features"
MODELS = ROOT / "data" / "lol" / "models"
TEAMS_JSON = ROOT / "web" / "composer" / "teams.json"
DEFAULT_OUT = ROOT / "output" / "public_pack"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _accepted_profile_games(project: Path) -> dict[str, dict[str, Any]]:
    pointer = project / "apps" / "scryglass" / "public" / "packs" / "manifest.json"
    try:
        manifest = json.loads(pointer.read_text(encoding="utf-8"))
        pack_id = str(manifest.get("pack_id") or "")
        profile_path = pointer.parent / pack_id / "features" / "profile_records.json"
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
        games = payload.get("games")
        return games if isinstance(games, dict) else {}
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}


def _accepted_public_schedule(project: Path) -> dict[str, Any] | None:
    """Read the last published optional schedule for source-outage continuity."""

    pointer = project / "apps" / "scryglass" / "public" / "packs" / "manifest.json"
    try:
        manifest = json.loads(pointer.read_text(encoding="utf-8"))
        pack_id = str(manifest.get("pack_id") or "")
        schedule_path = pointer.parent / pack_id / "features" / "schedule.json"
        payload = json.loads(schedule_path.read_text(encoding="utf-8"))
        validate_public_schedule(payload)
        return payload
    except (OSError, ValueError, TypeError, json.JSONDecodeError, PublicScheduleError):
        return None


def source_identity_sha256(game_ids: Iterable[str]) -> str:
    """Bind a pack to its sorted, canonical source game identities."""

    canonical = sorted(
        {
            game_id
            for value in game_ids
            if (game_id := canonical_source_game_key(value))
        }
    )
    raw = ("\n".join(canonical) + "\n").encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _champion_image_urls(project: Path) -> dict[str, str]:
    """Reuse the accepted tier-list champion identity map for profile art."""

    path = project / "apps" / "scryglass" / "public" / "rankings" / "tierlists.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return {}
    return {
        str(row["champion"]): str(row["champion_image_url"])
        for row in rows
        if isinstance(row, dict)
        and isinstance(row.get("champion"), str)
        and isinstance(row.get("champion_image_url"), str)
    }


def _present(cols: Sequence[str], available: Iterable[str]) -> list[str]:
    avail = set(available)
    return [c for c in cols if c in avail]


def _public_player_rating_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Exclude players whose league graph cannot support a public rank."""

    return [
        row
        for row in rows
        if row.get("evidence_disconnected") != 1
        and str(row.get("evidence_state") or "").lower() != "disconnected"
    ]


def _attach_public_team_evidence(
    ratings: pd.DataFrame,
    *,
    source_as_of: pd.Timestamp,
    weekly_ranks: Mapping[str, Any],
) -> pd.DataFrame:
    """Attach the public evidence contract to every team rating row."""

    stability: dict[str, float] = {}
    by_team = weekly_ranks.get("by_team", {})
    if isinstance(by_team, Mapping):
        for team, row in by_team.items():
            if not isinstance(row, Mapping):
                continue
            value = pd.to_numeric(row.get("mu_delta"), errors="coerce")
            if pd.notna(value):
                stability[str(team)] = abs(float(value))
    return attach_team_evidence(
        ratings,
        source_as_of=source_as_of,
        weekly_stability=stability,
    )


def _complete_player_game_ids(frame: pd.DataFrame) -> set[str]:
    """Return game IDs with two complete, uniquely identified five-player sides."""

    required = {"game_uid", "playername", "side", "position"}
    if frame.empty or not required.issubset(frame.columns):
        return set()
    rows = frame.dropna(subset=["game_uid", "playername", "side", "position"]).copy()
    if rows.empty:
        return set()
    rows["game_uid"] = rows["game_uid"].astype(str)
    rows["side"] = rows["side"].astype(str).str.title()
    rows = rows[rows["side"].isin({"Blue", "Red"})]
    games = rows.groupby("game_uid", sort=False).agg(
        rows=("playername", "size"),
        players=("playername", "nunique"),
        sides=("side", "nunique"),
    )
    sides = rows.groupby(["game_uid", "side"], sort=False).agg(
        rows=("playername", "size"),
        roles=("position", "nunique"),
    )
    complete_games = set(
        games.index[
            games["rows"].eq(10)
            & games["players"].eq(10)
            & games["sides"].eq(2)
        ].astype(str)
    )
    complete_sides = sides["rows"].eq(5) & sides["roles"].eq(5)
    side_counts = complete_sides.groupby(level="game_uid").agg(["size", "sum"])
    complete_side_games = set(
        side_counts.index[
            side_counts["size"].eq(2) & side_counts["sum"].eq(2)
        ].astype(str)
    )
    return complete_games.intersection(complete_side_games)


def _filter_years(table: pa.Table, years: Sequence[int], year_cols: Sequence[str]) -> pa.Table:
    years_list = list(years)
    # The live OE overlay can carry both the original ``year`` and the
    # normalized ``oe_year``.  They differ for a small API overlay.  Prefer
    # the normalized source year instead of widening the map set with OR.
    available = set(table.column_names)
    col = "oe_year" if "oe_year" in available else next(
        (value for value in year_cols if value in available),
        None,
    )
    if col is None:
        return table
    arr = table[col]
    try:
        as_int = pc.cast(arr, pa.int64(), safe=False)
    except Exception:
        as_int = pc.cast(pc.utf8_to_int(pc.cast(arr, pa.string())), pa.int64(), safe=False)
    mask = pc.is_in(as_int, value_set=pa.array(years_list, type=pa.int64()))
    return table.filter(mask)


def _filter_year_frame(
    frame: pd.DataFrame,
    years: Sequence[int],
    year_cols: Sequence[str],
) -> pd.DataFrame:
    column = "oe_year" if "oe_year" in frame.columns else next(
        (value for value in year_cols if value in frame.columns),
        None,
    )
    if column is None:
        return frame
    values = pd.to_numeric(frame[column], errors="coerce")
    return frame[values.isin(years)].copy()


def _ensure_year_column(table: pa.Table) -> pa.Table:
    """Add a UTC-derived year when a live map overlay has no source year."""
    if "year" in table.column_names or "oe_year" in table.column_names:
        return table
    if "date" not in table.column_names:
        return table
    frame = table.to_pandas()
    dates = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame["year"] = dates.dt.year.astype("Int64")
    return pa.Table.from_pandas(frame, preserve_index=False)


def _canonical_pack_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Recheck competition labels, including older cached canonical columns."""

    return canonicalize_competition_frame(frame)


def _validate_public_record_tiers(records: dict[str, dict[str, Any]], *, label: str) -> None:
    invalid = {"ORACLE_ELIXIR_API", "OE_API", "PUBLIC_DATALISK_API"}
    for identity, record in records.items():
        leagues = {str(value).upper() for value in record.get("leagues", [])}
        if leagues.intersection(invalid):
            raise RuntimeError(f"{label} {identity} exposes a transport label as a league")
        league = record.get("current_league")
        tier = record.get("current_tier")
        expected = competition_tier(league)
        if tier is not None and expected != tier:
            raise RuntimeError(
                f"{label} {identity} has inconsistent league tier: "
                f"league={league} tier={tier} expected={expected}"
            )


def _number(value: Any) -> float | None:
    """Coerce a numeric value, tolerating None and non-numeric payloads."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_draft_records_payload(
    composition_result: Any,
    composition_games: Sequence[Mapping[str, Any]],
    composition_evaluation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Compact whole-archive draft evidence: per-game draft edge on the
    model's logit scale (the coefficient-sum difference between sides)."""
    payload: dict[str, Any] = {
        "schema_version": "scryglass:draft-records:v1",
        "model_version": str((composition_evaluation or {}).get("model_version") or ""),
        "fit_through": (composition_evaluation or {}).get("fit_through"),
        "games": {},
    }
    draft_game_index = {str(game["game_uid"]): game for game in composition_games}
    signals = getattr(composition_result, "signals", None)
    if signals is None and isinstance(composition_result, Mapping):
        signals = composition_result
    signals = signals or {}
    for game_id, signal in signals.items():
        if not isinstance(signal, Mapping):
            continue
        game = draft_game_index.get(str(game_id))
        if not isinstance(game, Mapping) or signal.get("status") not in ("available", "limited"):
            continue
        blue_signal = _number(signal.get("blue", {}).get("signal"))
        red_signal = _number(signal.get("red", {}).get("signal"))
        draft_edge = (
            round(blue_signal - red_signal, 4)
            if blue_signal is not None and red_signal is not None
            else None
        )
        payload["games"][str(game_id)] = {
            "date": str(game.get("date") or ""),
            "league": str(game.get("league") or ""),
            "blue_team": str(game.get("blue_team") or ""),
            "red_team": str(game.get("red_team") or ""),
            "blue_signal": blue_signal,
            "red_signal": red_signal,
            # Descriptive draft advantage on the model's logit scale (the
            # coefficient-sum difference). NOT a win probability: the public
            # signal omits the model's control terms, so it is a ranked edge,
            # not a calibrated probability.
            "draft_edge": draft_edge,
        }
    return payload


def _draft_players_from_signals(
    signals: Mapping[str, Any], games: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Per-player draft contribution and highest-pick rate across the archive.

    Player identity comes from the composition game roster (the public signal
    carries side/role/champion only).
    """
    rosters: dict[str, Mapping[str, Any]] = {}
    for game in games:
        if isinstance(game, Mapping):
            rosters[str(game.get("game_uid"))] = game
    scores: dict[str, list[float]] = {}
    best_picks: dict[str, int] = {}
    roles: dict[str, str] = {}
    teams: dict[str, str] = {}
    for game_id, signal in signals.items():
        if not isinstance(signal, Mapping):
            continue
        game = rosters.get(str(game_id))
        picks = [pick for pick in signal.get("picks") or [] if isinstance(pick, Mapping)]
        best_by_side: dict[str, float] = {}
        for side in ("blue", "red"):
            values = [
                float(pick["contribution"])
                for pick in picks
                if str(pick.get("side") or "").strip().casefold() == side
                and _number(pick.get("contribution")) is not None
            ]
            if values:
                best_by_side[side] = max(values)
        for pick in picks:
            side = str(pick.get("side") or "").strip().casefold()
            role = str(pick.get("role") or "").strip()
            contribution = _number(pick.get("contribution"))
            if contribution is None or not side or not role:
                continue
            name = ""
            team = ""
            if isinstance(game, Mapping):
                side_roster = game.get(side)
                if isinstance(side_roster, Mapping):
                    slot = side_roster.get(role)
                    if isinstance(slot, Mapping):
                        name = str(slot.get("player") or "").strip()
                        team = str(slot.get("team") or "").strip()
            if not name:
                continue
            scores.setdefault(name, []).append(float(contribution))
            if side in best_by_side and abs(float(contribution) - best_by_side[side]) <= 1e-9:
                best_picks[name] = best_picks.get(name, 0) + 1
            if not roles.get(name):
                roles[name] = role
            if not teams.get(name):
                teams[name] = team
    rows = []
    for name, values in scores.items():
        if len(values) < 5:
            continue
        rows.append({
            "player": name,
            "games": len(values),
            "draft_score": sum(values) / len(values),
            "best_pick_rate": best_picks.get(name, 0) / len(values),
            "role": roles.get(name),
            "team": teams.get(name),
        })
    return rows


def _validate_public_composition_records(
    profile_records: Mapping[str, Any],
) -> dict[str, int]:
    """Validate composition evidence against each published ten-player game."""

    games = profile_records.get("games") if isinstance(profile_records, Mapping) else None
    if not isinstance(games, Mapping):
        raise CompositionSignalError("profile records have no game collection")
    counts = {"games": 0, "available": 0, "limited": 0, "unavailable": 0}
    for game_id, game in games.items():
        if not isinstance(game, Mapping):
            raise CompositionSignalError(f"profile game {game_id} is malformed")
        signal = game.get("draft_contribution")
        if signal is None:
            continue
        validate_public_signal(signal, game)
        status = str(signal.get("status"))
        counts["games"] += 1
        counts[status] += 1
    return counts


def _normalized_game_uid(frame: pd.DataFrame) -> pd.Series:
    """Use the source game UID and fall back to the OE game ID per row."""
    if not {"game_uid", "gameid", "oe_gameid"}.intersection(frame.columns):
        raise ValueError("rating source has no game identity column")
    fallback = frame["gameid"] if "gameid" in frame.columns else None
    values: list[str] = []
    if "game_uid" in frame.columns:
        source = frame["game_uid"]
    elif "gameid" in frame.columns:
        source = frame["gameid"]
    else:
        source = frame["oe_gameid"]
    for index, value in source.items():
        fallback_value = fallback.loc[index] if fallback is not None else None
        values.append(canonical_source_game_key(value, fallback_value))
    return pd.Series(values, index=frame.index, dtype="string").replace("", pd.NA)


def _canonicalize_game_ids(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply one canonical source key to every game identity column."""

    if not {"game_uid", "gameid", "oe_gameid"}.intersection(frame.columns):
        return frame
    normalized = _normalized_game_uid(frame)
    for column in ("game_uid", "gameid", "oe_gameid"):
        if column in frame.columns:
            frame[column] = normalized
    return frame


def export_public_pack(
    *,
    years: Sequence[int] | None = None,
    out_root: Path | None = None,
    pack_id: str | None = None,
    warehouse_root: Path | None = None,
    project_root: Path | None = None,
    runtime_root: Path | None = None,
    allowed_game_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    years = tuple(years or spec.DEFAULT_YEARS)
    project = Path(project_root or ROOT).resolve()
    runtime = Path(runtime_root or project).resolve()
    features_root = runtime / "data" / "lol" / "features"
    warehouse = Path(
        warehouse_root
        if warehouse_root is not None
        else runtime / "data" / "lol" / "warehouse" / "parquet" / "oe_live"
        if (runtime / "data" / "lol" / "warehouse" / "parquet" / "oe_live" / "meta.json").exists()
        else runtime / "data" / "lol" / "warehouse" / "parquet"
    )
    # Include UTC time so the 15-minute freshness workflow can publish more
    # than one immutable pack per day without colliding in Blob storage.
    stamp = datetime.now(timezone.utc).strftime("%Y.%m.%d.%H%M")
    pack_id = pack_id or f"v{stamp}"
    out_root = Path(out_root or runtime / "output" / "public_pack")
    pack_dir = out_root / pack_id
    if pack_dir.exists():
        shutil.rmtree(pack_dir)
    pack_dir.mkdir(parents=True)

    files_meta: list[dict[str, Any]] = []

    def progress(message: str) -> None:
        print(f"[public-pack] {message}", flush=True)

    def register(meta: dict[str, Any], rel: str) -> None:
        meta = dict(meta)
        meta["relative"] = rel
        meta["path"] = rel
        files_meta.append(meta)

    # Read full local sources to calculate ratings. Raw rows stay local.
    progress("reading compact team source")
    team_path = warehouse / "oe_team_games.parquet"
    team_available = pq.ParquetFile(team_path).schema_arrow.names
    team_columns = _present(
        (
            "gameid", "game_uid", "oe_gameid", "date", "year", "oe_year",
            "league", "league_source", "tournament", "result", "side",
            "position", "teamname", "grid_series_id",
        ),
        team_available,
    )
    team_source = _canonicalize_game_ids(
        _canonical_pack_frame(pq.read_table(team_path, columns=team_columns).to_pandas())
    )
    team_rating_frame = _filter_year_frame(team_source, years, ("year", "oe_year"))
    team_maps_for_ratings = build_maps_frame_from_team_games(team_rating_frame)
    del team_rating_frame, team_source

    player_path = warehouse / "oe_player_games.parquet"
    player_available = pq.ParquetFile(player_path).schema_arrow.names

    # --- maps ---
    progress("reading canonical maps")
    maps_path = warehouse / "maps.parquet"
    map_available = pq.ParquetFile(maps_path).schema_arrow.names
    maps = pq.read_table(maps_path, columns=spec.maps_columns(map_available))
    maps = pa.Table.from_pandas(canonicalize_competition_frame(maps.to_pandas()), preserve_index=False)
    maps = _ensure_year_column(maps)
    maps = _filter_years(maps, years, ("year", "oe_year"))
    maps_for_records = _canonicalize_game_ids(maps.to_pandas())
    maps = pa.Table.from_pandas(maps_for_records, preserve_index=False)
    live_source = (warehouse / "meta.json").exists()
    source_completeness_audit: dict[str, Any] = {
        "policy": "publish only maps with two complete, uniquely identified five-player sides",
        "rejected_incomplete_player_maps": 0,
    }
    if live_source:
        identity_columns = _present(
            (
                "gameid", "game_uid", "oe_gameid", "year", "oe_year",
                "playername", "side", "position",
            ),
            player_available,
        )
        player_identity = _filter_year_frame(
            _canonicalize_game_ids(
                pq.read_table(player_path, columns=identity_columns).to_pandas()
            ),
            years,
            ("year", "oe_year"),
        )
        player_identity["game_uid"] = _normalized_game_uid(player_identity)
        complete_ids = _complete_player_game_ids(player_identity)
        original_ids = set(_normalized_game_uid(maps_for_records).dropna().astype(str))
        accepted_ids = original_ids.intersection(complete_ids)
        if allowed_game_ids is not None:
            allowed = {canonical_source_game_key(value) for value in allowed_game_ids}
            accepted_ids.intersection_update(value for value in allowed if value)
        rejected_ids = original_ids.difference(accepted_ids)
        if not accepted_ids:
            raise RuntimeError("public pack source has no complete player maps")
        maps_for_records = maps_for_records[
            _normalized_game_uid(maps_for_records).isin(accepted_ids)
        ].copy()
        team_maps_for_ratings = team_maps_for_ratings[
            _normalized_game_uid(team_maps_for_ratings).isin(accepted_ids)
        ].copy()
        source_completeness_audit.update(
            {
                "candidate_maps": len(original_ids),
                "accepted_maps": len(accepted_ids),
                "rejected_incomplete_player_maps": len(rejected_ids),
                "rejected_identity_sha256": source_identity_sha256(rejected_ids),
            }
        )
        del player_identity, complete_ids, original_ids, accepted_ids, rejected_ids
        maps = pa.Table.from_pandas(maps_for_records, preserve_index=False)
    source_as_of = pd.to_datetime(maps_for_records["date"], utc=True, errors="coerce").max()
    if pd.isna(source_as_of):
        raise RuntimeError("public pack source has no usable map dates")
    source_game_ids = sorted(set(_normalized_game_uid(maps_for_records).dropna().astype(str)))
    if len(source_game_ids) != len(maps_for_records):
        raise RuntimeError("public pack source is not one row per canonical game identity")
    del maps
    progress("validated canonical maps")
    # The feature-oriented maps table intentionally covers the major/public
    # event slice.  Team ladders need the full OE team-game population so
    # Tier 2 and Tier 3 organizations receive both records and estimates.
    rating_input = (
        maps_for_records
        if live_source
        else team_maps_for_ratings if not team_maps_for_ratings.empty else maps_for_records
    )
    rating_input = filter_public_team_rating_maps(rating_input)
    if rating_input.empty:
        raise RuntimeError("public pack team rating source has no eligible team maps")
    progress("checking source identity alignment")
    if (warehouse / "meta.json").exists():
        map_ids = set(_normalized_game_uid(maps_for_records).dropna().astype(str))
        team_ids = set(_normalized_game_uid(team_maps_for_ratings).dropna().astype(str))
        if team_ids != map_ids:
            raise RuntimeError(
                "OE live public pack team inputs do not share the deduplicated map set; "
                f"maps={len(map_ids)} team={len(team_ids)}"
            )
        del map_ids, team_ids
    del team_maps_for_ratings
    progress("source identity alignment passed")
    progress("building records and ratings")
    team_records_payload = build_team_records(rating_input)

    progress("reading player affiliations")
    player_record_columns = _present(
        (
            "gameid", "game_uid", "oe_gameid", "year", "oe_year", "league",
            "league_source", "competition_scope", "event_kind", "is_international",
            "is_interregional", "competition_tier", "date", "position", "side",
            "playername", "teamname", "result", "tournament",
        ),
        player_available,
    )
    player_records_frame = _filter_year_frame(
        _canonicalize_game_ids(
            pq.read_table(player_path, columns=player_record_columns).to_pandas()
        ),
        years,
        ("year", "oe_year"),
    )
    # Live OE rows can carry empty derived competition fields after source
    # reconciliation. Rebuild them from the source league before affiliation
    # records are created so missing values cannot become public "nan" labels.
    player_records_frame = canonicalize_competition_frame(player_records_frame)
    player_records_frame["game_uid"] = _normalized_game_uid(player_records_frame)
    if player_records_frame["game_uid"].isna().any():
        raise RuntimeError("public pack rating source has rows without a game identity")
    if live_source:
        map_ids = set(source_game_ids)
        player_records_frame = player_records_frame[
            player_records_frame["game_uid"].astype(str).isin(map_ids)
        ].copy()
        player_ids = set(player_records_frame["game_uid"].dropna().astype(str))
        if player_ids != map_ids:
            raise RuntimeError(
                "OE live public pack player inputs do not share the deduplicated map set; "
                f"maps={len(map_ids)} player={len(player_ids)}"
            )
        player_rows = player_records_frame.groupby("game_uid", sort=False).agg(
            rows=("playername", "size"),
            players=("playername", "nunique"),
            sides=("side", "nunique"),
        )
        side_rows = player_records_frame.groupby(["game_uid", "side"], sort=False).agg(
            rows=("playername", "size"),
            roles=("position", "nunique"),
        )
        if (
            not player_rows["rows"].eq(10).all()
            or not player_rows["players"].eq(10).all()
            or not player_rows["sides"].eq(2).all()
            or not side_rows["rows"].eq(5).all()
            or not side_rows["roles"].eq(5).all()
        ):
            raise RuntimeError("public pack rating source has incomplete player maps")
        del map_ids, player_ids, player_rows, side_rows
    player_rating_row_count = len(player_records_frame)

    player_records_frame.drop(
        columns=[
            column
            for column in ("gameid", "game_uid", "oe_gameid", "year", "oe_year")
            if column in player_records_frame.columns
        ],
        inplace=True,
    )

    progress("building player affiliations")
    player_records_payload = build_player_records(
        player_records_frame,
        team_records=team_records_payload,
        canonicalized=True,
    )
    del player_records_frame
    progress("checking player affiliations")
    affiliation_audit = summarize_player_affiliations(
        player_records_payload,
        team_records_payload,
    )
    progress("reading player rating lineups")
    player_rating_columns = _present(
        (
            "gameid", "game_uid", "date", "year", "oe_year", "league", "result",
            "side", "position", "teamname", "playername",
        ),
        player_available,
    )
    player_rating_input = _filter_year_frame(
        _canonicalize_game_ids(
            pq.read_table(player_path, columns=player_rating_columns).to_pandas()
        ),
        years,
        ("year", "oe_year"),
    )
    player_rating_input = canonicalize_competition_frame(player_rating_input)
    if live_source:
        player_rating_input["game_uid"] = _normalized_game_uid(player_rating_input)
        player_rating_input = player_rating_input[
            player_rating_input["game_uid"].astype(str).isin(source_game_ids)
        ].copy()
    player_maps_for_ratings = (
        maps_for_records
        if live_source
        else build_maps_frame_from_players(player_rating_input)
    )
    if player_maps_for_ratings.empty:
        raise RuntimeError("public pack rating source has no complete player maps")
    progress("building sequential team ratings")
    dual_rating_features = build_dual_ratings(
        rating_input,
        lineup_by_game=lineup_hashes_from_players(player_rating_input),
        output_dir=features_root,
    )
    progress("building sequential player ratings")
    build_player_ratings(
        player_maps_for_ratings,
        player_rating_input,
        output_dir=features_root,
        player_records=player_records_payload,
    )
    player_snapshot_path = features_root / "player_ratings_snapshot.parquet"
    if player_snapshot_path.exists():
        progress("attaching player evidence")
        player_snapshot = pd.read_parquet(player_snapshot_path)
        player_snapshot = attach_player_evidence(
            player_snapshot,
            source_as_of=source_as_of,
        )
        player_snapshot.to_parquet(player_snapshot_path, index=False)
    progress("fitting team ladder")
    public_ratings, public_ratings_meta = fit_hierarchical_bt(
        rating_input,
        write=True,
        output_dir=features_root,
    )
    public_ratings_meta["pack_years"] = list(years)
    public_ratings_meta["rating_window"] = "full canonical OE team-game window as this pack"
    public_ratings_meta["source_as_of"] = source_as_of.isoformat().replace("+00:00", "Z")
    public_ratings_meta["source_mode"] = "oe_live" if live_source else "warehouse"
    public_ratings_meta["evidence_contract"] = "2026-08-09.1"
    features_root.mkdir(parents=True, exist_ok=True)
    (features_root / "ratings_meta.json").write_text(
        json.dumps(public_ratings_meta, indent=2),
        encoding="utf-8",
    )
    (features_root / "ratings_hierarchical_meta.json").write_text(
        json.dumps(public_ratings_meta, indent=2),
        encoding="utf-8",
    )
    # Write only the display JSON used by ratings pages.
    feat_dir = pack_dir / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    _validate_public_record_tiers(team_records_payload, label="team")
    _validate_public_record_tiers(player_records_payload, label="player")

    progress("building weekly movement")
    team_weekly_ranks = build_team_weekly_ranks(
        rating_input,
        as_of=source_as_of,
        min_series=5,
    )
    public_ratings = _attach_public_team_evidence(
        public_ratings,
        source_as_of=source_as_of,
        weekly_ranks=team_weekly_ranks,
    )
    team_weekly_dest = feat_dir / "team_weekly_ranks.json"
    team_weekly_dest.write_text(json.dumps(team_weekly_ranks, indent=2), encoding="utf-8")
    register(
        {
            "rows": len(team_weekly_ranks.get("by_team", {})),
            "cols": None,
            "bytes": team_weekly_dest.stat().st_size,
            "sha256": _sha256(team_weekly_dest),
            "columns": None,
        },
        "features/team_weekly_ranks.json",
    )

    weekly_ranks = build_player_weekly_ranks(
        player_maps_for_ratings,
        player_rating_input,
        as_of=pd.to_datetime(maps_for_records["date"], utc=True, errors="coerce").max(),
        min_games=20,
        player_records=player_records_payload,
    )
    player_meta_path = features_root / "player_ratings_meta.json"
    player_model_manifest: dict[str, Any] = {}
    if player_meta_path.exists():
        player_meta = json.loads(player_meta_path.read_text(encoding="utf-8"))
        player_meta["source_as_of"] = source_as_of.isoformat().replace("+00:00", "Z")
        player_meta["source_mode"] = "oe_live" if live_source else "warehouse"
        player_meta["window_years"] = list(years)
        player_meta["evidence_contract"] = "2026-08-09.1"
        player_meta_path.write_text(json.dumps(player_meta, indent=2), encoding="utf-8")
        global_rating = player_meta.get("global_rating") or {}
        player_model_manifest = {
            key: global_rating.get(key)
            for key in (
                "model",
                "n_maps",
                "n_players",
                "n_components",
                "largest_component_players",
                "connected_share",
                "holdout",
                "tier_adjustments",
                "player_statistics_used",
            )
        }
    weekly_dest = feat_dir / "player_weekly_ranks.json"
    weekly_dest.write_text(json.dumps(weekly_ranks, indent=2), encoding="utf-8")
    register(
        {
            "rows": len(weekly_ranks.get("by_player", {})),
            "cols": None,
            "bytes": weekly_dest.stat().st_size,
            "sha256": _sha256(weekly_dest),
            "columns": None,
        },
        "features/player_weekly_ranks.json",
    )

    progress("building profile artifacts")
    player_metadata = build_player_metadata(
        player_records_payload.keys(),
        player_context={
            player: record.get("current_team")
            for player, record in player_records_payload.items()
        },
    )
    metadata_dest = feat_dir / "player_metadata.json"
    metadata_dest.write_text(json.dumps(player_metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    register(
        {
            "rows": len(player_metadata),
            "cols": None,
            "bytes": metadata_dest.stat().st_size,
            "sha256": _sha256(metadata_dest),
            "columns": None,
        },
        "features/player_metadata.json",
    )

    profile_source_columns = _present(
        (
            "gameid", "game_uid", "date", "year", "oe_year", "league",
            "league_source", "tournament", "result", "side", "position",
            "teamname", "playername", "champion", "kills", "deaths", "assists",
            "patch", "grid_series_id",
            "teamkills", "gamelength", "dpm", "damageshare", "totalgold",
            "total cs", "minionkills", "monsterkills", "cspm", "visionscore",
            "wardsplaced", "wpm", "wcpm", "golddiffat10", "dragons",
            "heralds", "void_grubs", "barons", "atakhans", "towers", "inhibitors",
            "ban1", "ban2", "ban3", "ban4", "ban5",
        ),
        player_available,
    )
    player_profile_frame = _filter_year_frame(
        _canonicalize_game_ids(
            pq.read_table(player_path, columns=profile_source_columns).to_pandas()
        ),
        years,
        ("year", "oe_year"),
    )
    if live_source:
        player_profile_frame["game_uid"] = _normalized_game_uid(player_profile_frame)
        player_profile_frame = player_profile_frame[
            player_profile_frame["game_uid"].astype(str).isin(source_game_ids)
        ].copy()
    champion_image_urls = _champion_image_urls(project)
    player_champions_payload = build_player_champion_records(player_profile_frame)
    profile_records_payload = build_profile_records(
        player_profile_frame,
        champion_image_urls=champion_image_urls,
        include_archive=True,
    )
    progress("building composition evidence")
    composition_source_digest = source_identity_sha256(source_game_ids)
    composition_worker_commit = resolve_worker_commit(project)
    composition_games = build_composition_games(
        player_profile_frame,
        strength_features=dual_rating_features,
    )
    composition_model_dir = runtime / "data" / "lol" / "models" / "composition_signal"
    composition_evaluation_path = composition_model_dir / "evaluation.json"
    composition_evaluation: dict[str, Any] | None = None
    if composition_evaluation_path.exists():
        try:
            candidate_evaluation = json.loads(
                composition_evaluation_path.read_text(encoding="utf-8")
            )
            if (
                candidate_evaluation.get("model_version") == MODEL_VERSION
                and candidate_evaluation.get("source_hash") == composition_source_digest
                and candidate_evaluation.get("canonical_game_identity_sha256") == composition_source_digest
                and candidate_evaluation.get("worker_commit") == composition_worker_commit
            ):
                composition_evaluation = candidate_evaluation
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            composition_evaluation = None
    if composition_evaluation is None:
        progress("evaluating composition evidence")
        composition_evaluation = evaluate_composition_signal(
            composition_games,
            source_hash=composition_source_digest,
            canonical_game_identity_sha256=composition_source_digest,
            worker_commit=composition_worker_commit,
        )
        write_evaluation_report(composition_evaluation, composition_evaluation_path)
    promotion_gate = composition_evaluation.get("promotion_gate") or {}
    if promotion_gate.get("composition_candidate_passes") is not True:
        raise CompositionSignalError(
            "composition signal promotion gate did not pass; public release remains on the previous pack"
        )
    composition_result = score_games_temporally(
        composition_games,
        target_game_ids=None,
        cache_dir=composition_model_dir,
        source_digest=composition_source_digest,
        worker_commit=composition_worker_commit,
    )
    composition_audit = dict(composition_result.audit)
    composition_audit["source_as_of"] = source_as_of.isoformat().replace("+00:00", "Z")
    composition_audit["canonical_game_identity_sha256"] = composition_source_digest
    composition_audit["evaluation"] = {
        "status": "passed",
        "source_hash": composition_evaluation.get("source_hash"),
        "canonical_game_identity_sha256": composition_evaluation.get(
            "canonical_game_identity_sha256"
        ),
        "fit_through": composition_evaluation.get("fit_through"),
        "worker_commit": composition_evaluation.get("worker_commit"),
        "promotion_gate_passed": True,
    }
    draft_records_payload = build_draft_records_payload(
        composition_result, composition_games, composition_evaluation
    )
    for game_id, signal in composition_result.signals.items():
        if game_id in profile_records_payload.get("games", {}):
            profile_records_payload["games"][game_id]["draft_contribution"] = signal
        archive_candidate = profile_records_payload.get("_archive_games", {}).get(game_id)
        if isinstance(archive_candidate, dict):
            archive_candidate["draft_contribution"] = signal
<<<<<<< HEAD
=======
        game = draft_game_index.get(str(game_id))
        if not isinstance(game, Mapping) or signal.get("status") not in ("available", "limited"):
            continue
        blue_signal = _number(signal.get("blue", {}).get("signal"))
        red_signal = _number(signal.get("red", {}).get("signal"))
        draft_edge = (
            round(blue_signal - red_signal, 4)
            if blue_signal is not None and red_signal is not None
            else None
        )
        draft_records_payload["games"][str(game_id)] = {
            "date": str(game.get("date") or ""),
            "league": str(game.get("league") or ""),
            "competition_tier": str(game.get("competition_tier") or "") or None,
            "blue_team": str(game.get("blue_team") or ""),
            "red_team": str(game.get("red_team") or ""),
            "blue_signal": blue_signal,
            "red_signal": red_signal,
            # Descriptive draft advantage on the model's logit scale (the
            # coefficient-sum difference). NOT a win probability: the public
            # signal omits the model's control terms, so it is a ranked edge,
            # not a calibrated probability.
            "draft_edge": draft_edge,
        }
>>>>>>> origin/main
    draft_records_dest = feat_dir / "draft_records.json"
    draft_records_dest.write_text(
        json.dumps(draft_records_payload, separators=(",", ":"), ensure_ascii=False),
        encoding="utf-8",
    )
    register(
        {
            "rows": len(draft_records_payload["games"]),
            "cols": None,
            "bytes": draft_records_dest.stat().st_size,
            "sha256": _sha256(draft_records_dest),
            "columns": None,
        },
        "features/draft_records.json",
    )
    archive_games = merge_accepted_profile_games(
        profile_records_payload.pop("_archive_games", {}),
        _accepted_profile_games(project),
    )
    profile_game_ids = set(profile_records_payload.get("games", {})).intersection(archive_games)
    profile_records_payload["games"] = {
        game_id: archive_games[game_id]
        for game_id in sorted(profile_game_ids)
    }
    for index_name in ("players", "teams"):
        profile_records_payload[index_name] = {
            identity: [game_id for game_id in game_ids if game_id in profile_game_ids]
            for identity, game_ids in profile_records_payload.get(index_name, {}).items()
            if any(game_id in profile_game_ids for game_id in game_ids)
        }
    published_composition = _validate_public_composition_records(profile_records_payload)
    composition_audit.update(
        {
            "published_games": published_composition["games"],
            "published_available_games": published_composition["available"],
            "published_limited_games": published_composition["limited"],
            "published_unavailable_games": published_composition["unavailable"],
            "published_status": (
                "available"
                if published_composition["available"]
                else "limited"
                if published_composition["limited"]
                else "unavailable"
            ),
        }
    )
    del player_profile_frame

    # These records are intentionally built from the same year-filtered
    # canonical rows that are exported above.  This avoids mixing a full
    # history snapshot with a different pack-window win rate.
    for filename, payload in (
        ("team_records.json", team_records_payload),
        ("player_records.json", player_records_payload),
    ):
        dest = feat_dir / filename
        dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        register(
            {
                "rows": len(payload),
                "cols": None,
                "bytes": dest.stat().st_size,
                "sha256": _sha256(dest),
                "columns": None,
            },
            f"features/{filename}",
        )

    player_champions_dest = feat_dir / "player_champion_records.json"
    player_champions_dest.write_text(
        json.dumps(player_champions_payload, separators=(",", ":"), ensure_ascii=False),
        encoding="utf-8",
    )
    register(
        {
            "rows": sum(len(rows) for rows in player_champions_payload.values()),
            "cols": 8,
            "bytes": player_champions_dest.stat().st_size,
            "sha256": _sha256(player_champions_dest),
            "columns": [
                "champion",
                "games",
                "wins",
                "losses",
                "wr",
                "kills",
                "deaths",
                "assists",
            ],
        },
        "features/player_champion_records.json",
    )

    profile_records_dest = feat_dir / "profile_records.json"
    profile_records_dest.write_text(
        json.dumps(profile_records_payload, separators=(",", ":"), ensure_ascii=False),
        encoding="utf-8",
    )
    register(
        {
            "rows": len(profile_records_payload["games"]),
            "cols": None,
            "bytes": profile_records_dest.stat().st_size,
            "sha256": _sha256(profile_records_dest),
            "columns": None,
        },
        "features/profile_records.json",
    )

    match_index_payload = {
        "schema_version": "scryglass:match-index:v1",
        "years": [2025, 2026],
        "games": sorted(
            [
                {
                    "game_id": game_id,
                    "date": game["date"],
                    "league": game["league"],
                    "competition_tier": game.get("competition_tier"),
                    "blue_team": game["blue_team"],
                    "red_team": game["red_team"],
                    "blue_win": game["blue_win"],
                    "champions": [
                        player.get("champion")
                        for player in game.get("players", [])
                        if player.get("champion")
                    ],
                    "grades_available": sum(
                        1
                        for player in game.get("players", [])
                        if (player.get("grade") or {}).get("status") == "available"
                    ),
                }
                for game_id, game in archive_games.items()
            ],
            key=lambda game: (game["date"], game["game_id"]),
            reverse=True,
        ),
    }
    match_index_dest = feat_dir / "match_index.json"
    match_index_dest.write_text(
        json.dumps(match_index_payload, separators=(",", ":"), ensure_ascii=False),
        encoding="utf-8",
    )
    register(
        {
            "rows": len(match_index_payload["games"]),
            "cols": None,
            "bytes": match_index_dest.stat().st_size,
            "sha256": _sha256(match_index_dest),
            "columns": None,
        },
        "features/match_index.json",
    )

    # Leaguepedia supplies future fixtures that do not exist in Oracle's
    # Elixir. This artifact is optional and display-only. A failed fetch keeps
    # the previous valid schedule when one is available.
    progress("refreshing optional public schedule")
    schedule_payload: dict[str, Any] | None = None
    try:
        schedule_payload = build_public_schedule()
    except Exception as error:  # noqa: BLE001 - this lane must stay non-blocking
        schedule_payload = _accepted_public_schedule(project)
        if schedule_payload is not None:
            schedule_payload = dict(schedule_payload)
            schedule_payload["refresh_status"] = "cached"
        progress(f"optional schedule fetch unavailable ({type(error).__name__})")
    if schedule_payload is not None:
        schedule_dest = feat_dir / "schedule.json"
        schedule_dest.write_text(
            json.dumps(schedule_payload, separators=(",", ":"), ensure_ascii=False),
            encoding="utf-8",
        )
        register(
            {
                "rows": len(schedule_payload.get("upcoming", [])),
                "cols": None,
                "bytes": schedule_dest.stat().st_size,
                "sha256": _sha256(schedule_dest),
                "columns": None,
            },
            "features/schedule.json",
        )

    for archive_year in (2025, 2026):
        year_games = {
            game_id: game
            for game_id, game in archive_games.items()
            if str(game.get("date") or "").startswith(f"{archive_year}-")
        }
        archive_payload = {
            "schema_version": "scryglass:match-records:v1",
            "year": archive_year,
            "games": year_games,
        }
        archive_dest = feat_dir / f"match_records_{archive_year}.json"
        archive_dest.write_text(
            json.dumps(archive_payload, separators=(",", ":"), ensure_ascii=False),
            encoding="utf-8",
        )
        register(
            {
                "rows": len(year_games),
                "cols": None,
                "bytes": archive_dest.stat().st_size,
                "sha256": _sha256(archive_dest),
                "columns": None,
            },
            f"features/match_records_{archive_year}.json",
        )
    del archive_games

    for src_name, cols, out_name in (
        ("ratings_snapshot.parquet", spec.RATINGS_SNAPSHOT_COLS, "ratings_snapshot.json"),
        (
            "player_ratings_snapshot.parquet",
            spec.PLAYER_RATINGS_SNAPSHOT_COLS,
            "player_ratings_snapshot.json",
        ),
    ):
        src = features_root / src_name
        if src_name == "ratings_snapshot.parquet":
            t = pa.Table.from_pandas(public_ratings, preserve_index=False)
        elif not src.exists():
            continue
        else:
            t = pq.read_table(src)
        t = t.select(_present(cols, t.column_names))
        dest = feat_dir / out_name
        rows = t.to_pylist()
        if out_name == "player_ratings_snapshot.json":
            rows = _public_player_rating_rows(rows)
            for row in rows:
                row["last_team"] = public_team_affiliation(row.get("last_team"))
        dest.write_text(json.dumps(rows), encoding="utf-8")
        register(
            {
                "rows": len(rows),
                "cols": t.num_columns,
                "bytes": dest.stat().st_size,
                "sha256": _sha256(dest),
                "columns": t.column_names,
            },
            f"features/{dest.name}",
        )

    # Support-chat leaderboards: per-player aggregates + top-N indexes over the
    # already-public payloads. Optional display artifact; never part of the gate.
    try:
        player_rating_rows = json.loads(
            (feat_dir / "player_ratings_snapshot.json").read_text(encoding="utf-8")
        )
        team_rating_rows = json.loads(
            (feat_dir / "ratings_snapshot.json").read_text(encoding="utf-8")
        )
        team_records_payload_raw = dict(team_records_payload)
        player_champion_records_raw = dict(player_champions_payload)
        match_index_raw = dict(match_index_payload)
        draft_players_rows = _draft_players_from_signals(
            composition_result.signals, composition_games
        )
        leaderboards = build_leaderboards(
            player_records_payload,
            profile_records_payload,
            player_rating_rows,
            team_rating_rows,
            team_records=team_records_payload_raw,
            player_champion_records=player_champion_records_raw,
            match_index=match_index_raw,
            draft_records=draft_records_payload,
            draft_players=draft_players_rows,
        )
        leaderboards_dest = feat_dir / "leaderboards.json"
        leaderboards_dest.write_text(
            json.dumps(leaderboards, separators=(",", ":"), ensure_ascii=False),
            encoding="utf-8",
        )
        register(
            {
                "rows": len(leaderboards.get("players", {})),
                "cols": None,
                "bytes": leaderboards_dest.stat().st_size,
                "sha256": _sha256(leaderboards_dest),
                "columns": None,
            },
            "features/leaderboards.json",
        )
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise RuntimeError("support-chat leaderboards could not be built") from error

    present_paths = {str(item["path"]) for item in files_meta}
    missing_public_files = sorted(
        set(spec.PUBLIC_RATING_REQUIRED_FILES) - present_paths
    )
    if missing_public_files:
        raise RuntimeError(
            "public ratings contract incomplete; missing: "
            + ", ".join(missing_public_files)
        )
    leaked_public_files = sorted(
        present_paths.intersection(spec.WITHHELD_PUBLIC_FILES)
    )
    if leaked_public_files:
        raise RuntimeError(
            "withheld public artifacts present: " + ", ".join(leaked_public_files)
        )

    live_meta = warehouse / "meta.json"
    rating_artifact_paths = {
        item["path"]: {
            key: item[key]
            for key in ("sha256", "bytes", "rows")
            if key in item
        }
        for item in files_meta
        if item["path"]
        in {
            "features/ratings_snapshot.json",
            "features/player_ratings_snapshot.json",
            "features/team_weekly_ranks.json",
            "features/player_weekly_ranks.json",
        }
    }

    progress("finalizing manifest")
    total_bytes = sum(f["bytes"] for f in files_meta)
    manifest: dict[str, Any] = {
        "pack_id": pack_id,
        "schema_version": spec.SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "filters": {
            "years": list(years),
            "leagues": "all_in_year_window",
            "leagues_note": spec.DEFAULT_LEAGUES_NOTE,
        },
        "identity": {
            "taxonomy_version": TAXONOMY_VERSION,
            "team_key": "one canonical organization identity across regional and international events",
            "team_affiliation": "current league membership excludes cup and cross-league event labels; players inherit league and tier from their current team",
            "league_source": "raw source label retained on rows for auditability",
            "competition_tier": "tier1/tier2/tier3 assigned from the canonical league taxonomy; international and interregional are separate scopes",
            "deprecated_leagues": {
                "LTA": "AMERICAS",
                "LTA N": "LCS",
                "LTA NORTH": "LCS",
                "LTA S": "CBLOL",
                "LTA SOUTH": "CBLOL",
            },
        },
        "attribution": spec.ATTRIBUTION,
        "composition_signal": composition_audit,
        "excluded": [
            "raw game rows",
            "research studies",
            "training and prediction artifacts",
        ],
        "ratings": {
            "source_mode": "oe_live" if live_meta.exists() else "warehouse",
            "source_as_of": source_as_of.isoformat().replace("+00:00", "Z"),
            "window_years": list(years),
            "map_rows": int(len(maps_for_records)),
            "source_game_count": len(source_game_ids),
            "source_identity_sha256": source_identity_sha256(source_game_ids),
            "source_completeness": source_completeness_audit,
            "team_rating_rows": int(len(rating_input)),
            "player_rating_rows": int(player_rating_row_count),
            "player_model": player_model_manifest,
            "affiliation_audit": affiliation_audit,
            "artifacts": rating_artifact_paths,
            "claim_ceiling": "Source-bound descriptive ratings and historical rank movement only.",
        },
        "base_url": None,  # filled by upload / atlas config
        "total_bytes": total_bytes,
        "total_files": len(files_meta),
        "files": files_meta,
    }
    man_path = pack_dir / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # also write latest pointer
    latest = out_root / "latest.json"
    latest.write_text(
        json.dumps({"pack_id": pack_id, "path": pack_id, "created_utc": manifest["created_utc"]}, indent=2),
        encoding="utf-8",
    )

    # atlas-facing copy of manifest (for static deploy before Blob upload)
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Export the public LoL ratings payload")
    ap.add_argument("--years", default="2025,2026", help="Comma-separated years (default 2025,2026)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output root directory")
    ap.add_argument("--pack-id", default=None, help="Override pack id (default vYYYY.MM.DD)")
    ap.add_argument("--warehouse-root", type=Path, default=None, help="Use a source-root overlay for live refreshes")
    args = ap.parse_args(argv)
    years = tuple(int(x.strip()) for x in args.years.split(",") if x.strip())
    man = export_public_pack(
        years=years,
        out_root=args.out,
        pack_id=args.pack_id,
        warehouse_root=args.warehouse_root,
    )
    mb = man["total_bytes"] / (1024 * 1024)
    print(f"Wrote pack {man['pack_id']} → {args.out / man['pack_id']}")
    print(f"Files: {man['total_files']}  Size: {mb:.1f} MB  schema={man['schema_version']}")
    for f in man["files"]:
        print(f"  {f['path']}: {f['bytes']/1024:.0f} KB  rows={f.get('rows')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
