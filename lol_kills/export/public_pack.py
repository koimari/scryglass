"""Export the small public ratings payload (2025–2026 default).

Usage:
  python3 -m lol_kills.export.public_pack
  python3 -m lol_kills.export.public_pack --years 2025,2026 --out output/public_pack
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from lol_kills.etl.aliases import normalize_champ
from lol_kills.export import pack_spec as spec
from lol_kills.export.leaderboards import build_leaderboards
from lol_kills.export.pack_records import (
    _draft_text,
    build_maps_frame_from_team_games,
    build_player_champion_records,
    build_profile_records,
    build_player_records,
    build_team_records,
    filter_public_team_rating_maps,
    merge_accepted_profile_games,
    public_patch_for_source,
    public_team_affiliation,
    summarize_player_affiliations,
)
from lol_kills.export.player_metadata import build_player_metadata
from lol_kills.export.public_schedule import (
    PublicScheduleError,
    build_public_schedule,
    validate_public_schedule,
)
from lol_kills.export.public_query_projection import (
    build_public_query_projection,
    write_public_query_projection,
)
from lol_kills.export.promoted_draft_authority import (
    PromotedDraftAuthorityError,
    load_promoted_draft_authority,
    validate_promoted_results_payload,
)
from lol_kills.etl.competition import TAXONOMY_VERSION, canonicalize_competition_frame, competition_tier
from lol_kills.etl.source_keys import canonical_source_game_key
from lol_kills.refresh_ledger import worker_commit as resolve_worker_commit
from lol_kills.ratings.dual_elo import (
    DualEloConfig,
    apply_team_momentum_snapshot,
    build_dual_ratings,
    lineup_hashes_from_players,
)
from lol_kills.ratings.evidence import attach_player_evidence, attach_team_evidence
from lol_kills.ratings.hierarchical_bt import build_team_weekly_ranks, fit_hierarchical_bt
from lol_kills.ratings.momentum_config import (
    DEFAULT_MOMENTUM_SCALE,
    DEFAULT_MOMENTUM_WINDOW_GAMES,
    momentum_manifest_metadata,
    require_public_momentum_disabled,
)
from lol_kills.ratings.player_elo import (
    PlayerEloConfig,
    build_maps_frame_from_players,
    build_player_ratings,
    build_player_weekly_ranks,
)
from lol_kills.research.composition_signal import (
    CompositionSignalError,
    build_composition_games,
    validate_public_signal,
)
from lol_kills.research.descriptive_draft_score import (
    DescriptiveDraftScoreError,
    EXCLUDED_TERMS as DESCRIPTIVE_SCORE_EXCLUDED_TERMS,
    INCLUDED_TERMS as DESCRIPTIVE_SCORE_INCLUDED_TERMS,
    MODEL_VERSION as DESCRIPTIVE_SCORE_MODEL_VERSION,
    SCHEMA_VERSION as DESCRIPTIVE_SIGNAL_SCHEMA_VERSION,
    load_model as load_descriptive_score_model,
    score_game as score_descriptive_game,
)
ROOT = Path(__file__).resolve().parents[2]
WAREHOUSE = ROOT / "data" / "lol" / "warehouse" / "parquet"
LIVE_WAREHOUSE = WAREHOUSE / "oe_live"
FEATURES = ROOT / "data" / "lol" / "features"
MODELS = ROOT / "data" / "lol" / "models"
TEAMS_JSON = ROOT / "web" / "composer" / "teams.json"
DEFAULT_OUT = ROOT / "output" / "public_pack"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DRAFT_ISSUED_UTC_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$"
)
DESCRIPTIVE_AUTHORITY_PATH = (
    ROOT / "data" / "lol" / "v2" / "evaluation" / "composition-descriptive-authority.json"
)
DESCRIPTIVE_RECIPE_PATH = (
    ROOT / "data" / "lol" / "v2" / "evaluation" / "composition-descriptive-recipe.json"
)


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
    """Return approved CommunityDragon art for every known champion.

    Tier output can be empty during a fresh patch window. Profile and match
    art must not depend on a tier list row, so the pinned champion crosswalk is
    the fallback identity source.
    """

    def approved(value: object) -> str | None:
        url = str(value or "").strip()
        if not url:
            return None
        parsed = urllib.parse.urlsplit(url)
        if (
            parsed.scheme != "https"
            or parsed.hostname != "cdn.communitydragon.org"
            or parsed.username
            or parsed.password
            or parsed.port not in {None, 443}
            or parsed.fragment
        ):
            return None
        return url

    def champion_key(value: object) -> str:
        return normalize_champ(str(value or "")).strip().casefold()

    urls: dict[str, str] = {}

    path = project / "apps" / "scryglass" / "public" / "rankings" / "tierlists.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        payload = {}
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = str(row.get("champion") or "").strip()
            url = approved(row.get("champion_image_url"))
            if name and url:
                urls[normalize_champ(name)] = url

    crosswalk_path = (
        project
        / "data"
        / "lol"
        / "v2"
        / "champions"
        / "champion-id-crosswalk-v1.json"
    )
    try:
        crosswalk = json.loads(crosswalk_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        crosswalk = {}
    entries = crosswalk.get("entries") if isinstance(crosswalk, dict) else None
    existing_keys = {champion_key(key) for key in urls}
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            numeric_id = entry.get("riot_numeric_id")
            try:
                numeric_id = int(numeric_id)
            except (TypeError, ValueError):
                continue
            if numeric_id <= 0:
                continue
            name = normalize_champ(
                str(entry.get("riot_display_name") or entry.get("oe_name") or "").strip()
            )
            if not name or champion_key(name) in existing_keys:
                continue
            urls[name] = f"https://cdn.communitydragon.org/latest/champion/{numeric_id}/square"
            existing_keys.add(champion_key(name))

    # The older OE crosswalk has 171 entries.  The current canonical
    # 26.16 ontology contains the two newer standard champions as well.
    ontology_path = (
        project
        / "data"
        / "lol"
        / "v2"
        / "champions"
        / "champion-ontology-seed-26.16.json"
    )
    try:
        ontology = json.loads(ontology_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        ontology = {}
    champions = ontology.get("champions") if isinstance(ontology, dict) else None
    if isinstance(champions, list):
        for champion in champions:
            if not isinstance(champion, dict):
                continue
            name = normalize_champ(str(champion.get("display_name") or "").strip())
            champion_id = str(champion.get("champion_id") or "").strip()
            match = re.fullmatch(r"riot:champion:(\d+)", champion_id)
            if not name or match is None or champion_key(name) in existing_keys:
                continue
            urls[name] = f"https://cdn.communitydragon.org/latest/champion/{match.group(1)}/square"
            existing_keys.add(champion_key(name))
    return dict(sorted(urls.items()))


def _present(cols: Sequence[str], available: Iterable[str]) -> list[str]:
    avail = set(available)
    return [c for c in cols if c in avail]


def serialize_rating_snapshot_rows(
    table: pa.Table,
    columns: Sequence[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Select the public rating contract while retaining enabled momentum."""

    selected = _present(columns, table.column_names)
    if not selected:
        raise ValueError("rating snapshot has no public columns")
    selected_table = table.select(selected)
    return selected_table.to_pylist(), selected


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


def _profile_archive_frame(
    table: pa.Table,
    years: tuple[int, ...],
) -> pd.DataFrame:
    """Build the one canonical player archive used by public match surfaces."""

    frame = _filter_year_frame(
        _canonicalize_game_ids(table.to_pandas()),
        years,
        ("year", "oe_year"),
    )
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


def _load_descriptive_authority(project: Path) -> tuple[dict[str, Any], str]:
    """Load and bind the checked-in descriptive Draft Score receipt.

    This receipt authorizes descriptive composition evidence only. It does not
    authorize a probability, recommendation, odds, or betting output.
    """

    authority_path = project / DESCRIPTIVE_AUTHORITY_PATH.relative_to(ROOT)
    recipe_path = project / DESCRIPTIVE_RECIPE_PATH.relative_to(ROOT)
    scorer_path = project / "lol_kills" / "research" / "descriptive_draft_score.py"
    try:
        authority_raw = authority_path.read_bytes()
        authority = json.loads(authority_raw.decode("utf-8"))
        recipe_raw = recipe_path.read_bytes()
        scorer_raw = scorer_path.read_bytes()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, DescriptiveDraftScoreError) as error:
        raise CompositionSignalError(
            "descriptive Draft Score authority receipt is unavailable"
        ) from error
    if not isinstance(authority, dict) or not isinstance(recipe_raw, bytes):
        raise CompositionSignalError("descriptive Draft Score authority receipt is malformed")
    issued_utc = authority.get("issued_utc")
    if (
        not isinstance(issued_utc, str)
        or not DRAFT_ISSUED_UTC_RE.fullmatch(issued_utc)
        or _parse_issued_utc(issued_utc) is None
    ):
        raise CompositionSignalError("descriptive Draft Score authority timestamp is invalid")
    if (
        authority.get("schema_version") != "scryglass:draft-authority:v1"
        or authority.get("status") != "descriptive"
        or authority.get("estimand") != "composition_only"
        or authority.get("model_version") != DESCRIPTIVE_SCORE_MODEL_VERSION
        or authority.get("recipe_sha256") != hashlib.sha256(recipe_raw).hexdigest()
        or authority.get("artifact_sha256") != load_descriptive_score_model()[1]
        or authority.get("scorer_code_sha256") != hashlib.sha256(scorer_raw).hexdigest()
        or authority.get("probability_authority") is not False
        or authority.get("recommendation_authority") is not False
        or authority.get("betting_authority") is not False
        or authority.get("included_terms") != list(DESCRIPTIVE_SCORE_INCLUDED_TERMS)
        or authority.get("excluded_terms") != list(DESCRIPTIVE_SCORE_EXCLUDED_TERMS)
    ):
        raise CompositionSignalError(
            "descriptive Draft Score authority receipt does not bind the active recipe"
        )
    return authority, hashlib.sha256(authority_raw).hexdigest()


def _parse_issued_utc(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None and parsed.utcoffset() == timezone.utc.utcoffset(parsed) else None


def build_draft_records_payload(
    composition_result: Any,
    composition_games: Sequence[Mapping[str, Any]],
    composition_evaluation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Compact whole-archive descriptive composition evidence.

    The edge stays in model units. The payload contains no probability field.
    """
    result_audit = getattr(composition_result, "audit", None)
    if result_audit is None and isinstance(composition_result, Mapping):
        result_audit = composition_result.get("audit")
    if not isinstance(result_audit, Mapping):
        result_audit = {}
    metadata = composition_evaluation if isinstance(composition_evaluation, Mapping) else {}
    payload: dict[str, Any] = {
        "schema_version": "scryglass:draft-records:v1",
        "authority": "descriptive",
        "estimand": "composition_only",
        "model_version": str(
            result_audit.get("model_version")
            or metadata.get("model_version")
            or DESCRIPTIVE_SCORE_MODEL_VERSION
        ),
        "fit_through": result_audit.get("fit_through") or metadata.get("fit_through"),
        "source_identity_sha256": result_audit.get("source_identity_sha256")
        or metadata.get("source_identity_sha256"),
        "artifact_sha256": result_audit.get("artifact_sha256") or metadata.get("artifact_sha256"),
        "authority_receipt_sha256": result_audit.get("receipt_sha256")
        or metadata.get("receipt_sha256"),
        "archetype_interaction_source": result_audit.get("archetype_interaction_source")
        or metadata.get("archetype_interaction_source"),
        "source_as_of": result_audit.get("source_as_of") or metadata.get("source_as_of"),
        "source_patch_binding": result_audit.get("source_patch_binding")
        or metadata.get("source_patch_binding"),
        "evaluation": result_audit.get("evaluation") or metadata.get("evaluation"),
        "sample_window": {
            "target_games": metadata.get("target_games") or result_audit.get("target_games"),
            "source_as_of": result_audit.get("source_as_of") or metadata.get("source_as_of"),
        },
        "player_comfort": metadata.get("player_comfort") or {
            "status": "unavailable",
            "contribution": None,
            "source": None,
            "sha256": None,
            "reason": "No release-bound player familiarity source is available.",
        },
        "games": {},
    }
    draft_game_index = {str(game["game_uid"]): game for game in composition_games}
    signals = getattr(composition_result, "signals", None)
    if signals is None and isinstance(composition_result, Mapping):
        signals = composition_result.get("signals") or composition_result
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
            "competition_tier": str(game.get("competition_tier") or "") or None,
            "blue_team": str(game.get("blue_team") or ""),
            "red_team": str(game.get("red_team") or ""),
            "blue_signal": blue_signal,
            "red_signal": red_signal,
            "blue_components": signal.get("blue", {}).get("components"),
            "red_components": signal.get("red", {}).get("components"),
            "edge_components": signal.get("edge_components"),
            # Descriptive draft advantage on the model's logit scale (the
            # coefficient-sum difference). NOT a win probability: the public
            # signal omits the model's control terms, so it is a ranked edge,
            # not a calibrated probability.
            "draft_edge": draft_edge,
        }
    return payload


def _draft_publication_decision(
    composition_evaluation: Mapping[str, Any] | None,
    *,
    descriptive_authority: Mapping[str, Any] | None = None,
    receipt_sha256: str | None = None,
    release_id: str | None = None,
) -> dict[str, Any]:
    """Return the public Draft decision.

    A descriptive receipt may authorize composition evidence. The predictive
    promotion path remains separate and closed.
    """

    if descriptive_authority is not None:
        if (
            descriptive_authority.get("status") != "descriptive"
            or descriptive_authority.get("estimand") != "composition_only"
            or descriptive_authority.get("model_version") != DESCRIPTIVE_SCORE_MODEL_VERSION
            or not isinstance(receipt_sha256, str)
            or not SHA256_RE.fullmatch(receipt_sha256)
        ):
            raise CompositionSignalError("descriptive Draft Score authority is malformed")
        return {
            "schema_version": "scryglass:draft-authority:v1",
            "status": "descriptive",
            "authority": "descriptive",
            "release_id": release_id,
            "model_version": DESCRIPTIVE_SCORE_MODEL_VERSION,
            "artifact_sha256": descriptive_authority.get("artifact_sha256"),
            "receipt_sha256": receipt_sha256,
            "issued_utc": descriptive_authority.get("issued_utc"),
            "estimand": "composition_only",
            "probability_authority": False,
            "recommendation_authority": False,
            "betting_authority": False,
            "reason": None,
        }

    if composition_evaluation is None:
        return {
            "status": "unavailable",
            "reason": "model_not_promoted",
        }

    if not isinstance(composition_evaluation, Mapping):
        raise CompositionSignalError("composition evaluation is malformed")
    promotion_gate = composition_evaluation.get("promotion_gate")
    if not isinstance(promotion_gate, Mapping):
        raise CompositionSignalError("composition evaluation promotion gate is malformed")
    candidate_passes = promotion_gate.get("composition_candidate_passes")
    if type(candidate_passes) is not bool:
        raise CompositionSignalError(
            "composition evaluation promotion gate is contradictory"
        )

    return {
        "status": "unavailable",
        # Candidate evaluation details stay private.  The public contract
        # reports only that no promotion authority is available.
        "reason": "model_not_promoted",
    }


def _withhold_unpromoted_draft_fields(payload: Mapping[str, Any]) -> None:
    """Remove model-derived draft fields from one public profile payload."""

    if isinstance(payload, dict):
        payload.pop("draft_pool_audit", None)
    games = payload.get("games")
    if not isinstance(games, Mapping):
        return
    for game in games.values():
        if not isinstance(game, dict):
            continue
        game.pop("draft_contribution", None)
        game.pop("draft_pool", None)


def _draft_players_from_signals(
    signals: Mapping[str, Any], games: Mapping[str, Any] | Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Build player rows from the same enriched profile records as profiles.

    ``best_available`` is attached to each public composition pick after the
    ban, pick-order, patch, and tier evidence checks. Missing quality facts do
    not enter the denominator.
    """
    profile_games = games.get("games") if isinstance(games, Mapping) else None
    if isinstance(profile_games, Mapping):
        iterable = [game for game in profile_games.values() if isinstance(game, Mapping)]
    else:
        iterable = [game for game in games if isinstance(game, Mapping)]
    scores: dict[str, list[float]] = {}
    best_picks: dict[str, int] = {}
    evaluated_picks: dict[str, int] = {}
    roles: dict[str, str] = {}
    teams: dict[str, str] = {}
    for game in iterable:
        game_id = str(game.get("game_id") or game.get("game_uid") or "")
        signal = game.get("draft_contribution") if isinstance(profile_games, Mapping) else signals.get(game_id)
        signal_picks = [pick for pick in (signal.get("picks") if isinstance(signal, Mapping) else []) or [] if isinstance(pick, Mapping)]
        pool = game.get("draft_pool") if isinstance(game.get("draft_pool"), Mapping) else {}
        pool_picks = [pick for pick in pool.get("picked", []) if isinstance(pick, Mapping)]
        picks = [pick for pick in pool_picks if pick.get("best_available") in (True, False)] or [
            pick for pick in signal_picks if pick.get("best_available") in (True, False)
        ]
        if not picks:
            continue
        profile_roster = {
            (str(row.get("side") or "").title(), _draft_role(row.get("role"))): row
            for row in game.get("players", [])
            if isinstance(row, Mapping)
        }
        for pick in picks:
            side = str(pick.get("side") or "").strip().title()
            role = _draft_role(pick.get("role"))
            contribution = _number(pick.get("contribution"))
            if contribution is None:
                matching_signal = next(
                    (
                        candidate
                        for candidate in signal_picks
                        if str(candidate.get("side") or "").strip().title() == side
                        and _draft_role(candidate.get("role")) == role
                        and _draft_key(candidate.get("champion")) == _draft_key(pick.get("champion"))
                    ),
                    None,
                )
                contribution = _number(matching_signal.get("contribution")) if isinstance(matching_signal, Mapping) else None
            quality = pick.get("best_available")
            if quality not in (True, False) or not side or not role:
                continue
            name = ""
            team = ""
            slot = profile_roster.get((side, role))
            if isinstance(slot, Mapping):
                name = str(slot.get("player") or "").strip()
                team = str((game.get("blue_team") if side == "Blue" else game.get("red_team")) or "").strip()
            else:
                side_roster = game.get(side.casefold())
                if isinstance(side_roster, Mapping):
                    source_role = {"jungle": "jng", "support": "sup"}.get(role, role)
                    source = side_roster.get(source_role)
                    if isinstance(source, Mapping):
                        name = str(source.get("player") or "").strip()
                        team = str(source.get("team") or "").strip()
            if not name:
                continue
            evaluated_picks[name] = evaluated_picks.get(name, 0) + 1
            if quality:
                best_picks[name] = best_picks.get(name, 0) + 1
            if contribution is not None:
                scores.setdefault(name, []).append(float(contribution))
            if not roles.get(name):
                roles[name] = role
            if not teams.get(name):
                teams[name] = team
    rows = []
    for name, evaluated in evaluated_picks.items():
        if evaluated < 5:
            continue
        values = scores.get(name, [])
        if not values:
            continue
        rows.append({
            "player": name,
            "games": evaluated,
            "pick_contribution": sum(values) / len(values),
            "best_available_rate": best_picks.get(name, 0) / evaluated,
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


def _gate_published_draft_contributions(
    profile_records: Mapping[str, Any],
    draft_records_payload: Mapping[str, Any] | None = None,
) -> dict[str, int]:
    """Keep valid composition scores and retain only complete pool evidence.

    A ten-pick composition score does not need ban and pick-order evidence.
    Best-available player metrics keep the narrower complete-pool boundary.
    """

    games = profile_records.get("games") if isinstance(profile_records, Mapping) else None
    if not isinstance(games, Mapping):
        return {"score_games": 0, "pool_games": 0, "removed_games": 0}
    score_games: set[str] = set()
    pool_games: set[str] = set()
    removed = 0
    for game_id, game in games.items():
        if not isinstance(game, dict):
            continue
        signal = game.get("draft_contribution")
        if not isinstance(signal, Mapping):
            continue
        pool = game.get("draft_pool")
        pool_picks = pool.get("picked") if isinstance(pool, Mapping) else None
        try:
            evaluated_picks = int(pool.get("evaluated_picks")) if isinstance(pool, Mapping) else 0
        except (TypeError, ValueError):
            evaluated_picks = 0
        if signal.get("status") != "available":
            game.pop("draft_contribution", None)
            game.pop("draft_pool", None)
            removed += 1
            continue
        score_games.add(str(game_id))
        complete = (
            isinstance(pool, Mapping)
            and pool.get("status") == "complete"
            and isinstance(pool_picks, list)
            and len(pool_picks) == 10
            and evaluated_picks == 10
        )
        if complete:
            pool_games.add(str(game_id))
        else:
            game.pop("draft_pool", None)

    if isinstance(draft_records_payload, dict):
        records = draft_records_payload.get("games")
        if isinstance(records, dict):
            for game_id in list(records):
                if str(game_id) not in score_games:
                    records.pop(game_id, None)
                    continue
                if str(game_id) not in pool_games and isinstance(records[game_id], dict):
                    records[game_id].pop("draft_pool", None)
    return {
        "score_games": len(score_games),
        "pool_games": len(pool_games),
        "removed_games": removed,
    }


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


def _draft_metadata_from_maps(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Extract source pick order and patch identity without publishing raw maps."""

    if frame is None or frame.empty:
        return {}
    def clean(value: Any) -> str:
        if value is None:
            return ""
        try:
            if bool(pd.isna(value)):
                return ""
        except (TypeError, ValueError):
            pass
        return str(value).strip()

    output: dict[str, dict[str, Any]] = {}
    for _, row in frame.iterrows():
        game_id = canonical_source_game_key(
            clean(row.get("game_uid")) or clean(row.get("oe_gameid")) or clean(row.get("gameid"))
        )
        if not game_id:
            continue
        source_patch = clean(row.get("oe_patch_token")) or clean(row.get("patch"))
        event_time = row.get("date")
        realm_patch = (
            clean(row.get("server_patch"))
            or clean(row.get("game_patch"))
            or clean(row.get("realm_patch"))
            or clean(row.get("authoritative_patch"))
        )
        realm_kind = (
            clean(row.get("patch_realm"))
            or clean(row.get("realm_kind"))
            or clean(row.get("server_kind"))
        )
        value: dict[str, Any] = {
            # OE stores the source token in the client namespace (16.x). The
            # public pack uses Riot's 26.x label.
            "patch": public_patch_for_source(
                source_patch,
                event_time=event_time,
                realm_patch=realm_patch or None,
                realm_kind=realm_kind or None,
            ) or None,
            "oe_patch_token": source_patch or None,
            "realm_patch": realm_patch or None,
            "realm_kind": realm_kind or None,
            "blue_bans": [
                clean(row.get(f"blue_ban{slot}"))
                for slot in range(1, 6)
                if clean(row.get(f"blue_ban{slot}"))
            ],
            "red_bans": [
                clean(row.get(f"red_ban{slot}"))
                for slot in range(1, 6)
                if clean(row.get(f"red_ban{slot}"))
            ],
            "blue_picks": [
                clean(row.get(f"blue_pick{slot}"))
                for slot in range(1, 6)
                if clean(row.get(f"blue_pick{slot}"))
            ],
            "red_picks": [
                clean(row.get(f"red_pick{slot}"))
                for slot in range(1, 6)
                if clean(row.get(f"red_pick{slot}"))
            ],
            "blue_first_pick": row.get("blue_firstPick"),
        }
        output[game_id] = value
    return output


def _draft_key(value: Any) -> str:
    return normalize_champ(str(value or "").strip()).casefold()


def _draft_patch(
    value: Any,
    *,
    event_time: Any | None = None,
    realm_patch: Any | None = None,
    realm_kind: Any | None = None,
) -> str:
    """Return Riot's public patch label for an OE or client patch token."""

    return public_patch_for_source(
        value,
        event_time=event_time,
        realm_patch=realm_patch,
        realm_kind=realm_kind,
    )


def _draft_role(value: Any) -> str:
    return {
        "jng": "jungle",
        "jungle": "jungle",
        "sup": "support",
        "support": "support",
        "adc": "bot",
        "bot": "bot",
        "top": "top",
        "mid": "mid",
    }.get(str(value or "").strip().casefold(), str(value or "").strip().casefold())


def _tier_payload_candidates(
    project: Path,
    runtime: Path,
    explicit_path: Path | None = None,
) -> list[Path]:
    """Return the generated tier artifacts that this pack may consume.

    The tier board is a runtime publication input. A clean source checkout
    does not contain a checked-in copy of the public board. An explicit path
    is useful for local pack builds and remains subject to the same payload
    validation as the worker-generated path.
    """

    values: list[Path] = []
    if explicit_path is not None:
        candidate = explicit_path.expanduser()
        if not candidate.is_absolute():
            candidate = project / candidate
        values.append(candidate)
    else:
        values.extend(
            (
                runtime / "apps" / "scryglass" / "public" / "rankings" / "tierlists.json",
                project / "apps" / "scryglass" / "public" / "rankings" / "tierlists.json",
            )
        )
    unique: list[Path] = []
    seen: set[Path] = set()
    for value in values:
        resolved = value.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def _load_tier_payload(
    project: Path,
    runtime: Path,
    explicit_path: Path | None = None,
) -> tuple[Mapping[str, Any] | None, str | None, str | None]:
    """Load one available generated tier board and bind its raw digest.

    The returned digest identifies the exact bytes used to evaluate the
    published best-available pools. Invalid, staged, and empty boards are
    ignored. The caller decides whether the release can continue without a
    valid board.
    """

    candidates = _tier_payload_candidates(project, runtime, explicit_path)
    for path in candidates:
        try:
            raw = path.read_bytes()
            payload = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, Mapping):
            continue
        if payload.get("schema_version") != "rankings-tierlists-v2":
            continue
        if payload.get("status") != "available":
            continue
        rows = payload.get("rows")
        if not isinstance(rows, list) or not rows:
            continue
        return payload, hashlib.sha256(raw).hexdigest(), str(path)
    return None, None, None


def _attach_published_draft_pools(
    profile_records: dict[str, Any],
    tier_payload: Mapping[str, Any] | None,
    *,
    tier_payload_sha256: str | None = None,
    tier_receipt_sha256: str | None = None,
) -> dict[str, int | float]:
    """Attach ban/unpicked pools and best-available pick facts to each game.

    The published tier board is the champion-quality source. A pick is
    evaluable only when the game has an exact patch, complete bans, and a
    source pick order. Unknown rows stay null instead of becoming a loss.
    """

    rows = tier_payload.get("rows") if isinstance(tier_payload, Mapping) else None
    rows_by_scope: dict[tuple[str, str], list[dict[str, Any]]] = {}
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            patch = _draft_patch(row.get("patch"))
            role = _draft_role(row.get("role"))
            champion = str(row.get("champion") or "").strip()
            rank = row.get("rank")
            if not patch or role not in {"top", "jungle", "mid", "bot", "support"} or not champion:
                continue
            try:
                rank_value = int(rank)
            except (TypeError, ValueError):
                continue
            rows_by_scope.setdefault((patch, role), []).append(
                {"champion": champion, "key": _draft_key(champion), "rank": rank_value}
            )
    for values in rows_by_scope.values():
        values.sort(key=lambda row: (row["rank"], row["champion"].casefold()))

    games = profile_records.get("games") if isinstance(profile_records, Mapping) else None
    if not isinstance(games, Mapping):
        return {"games": 0, "complete_bans": 0, "complete_pick_order": 0, "quality_games": 0, "quality_picks": 0, "coverage": 0.0}

    total = complete_bans = complete_order = quality_games = quality_picks = 0
    for game in games.values():
        if not isinstance(game, dict):
            continue
        total += 1
        pool = game.get("draft_pool")
        if not isinstance(pool, dict):
            pool = {
                "schema_version": "scryglass:draft-pool:v1",
                "status": "unavailable",
                "source": "oracle-elixir",
                "patch": _draft_patch(
                    game.get("patch"),
                    event_time=game.get("date"),
                    realm_patch=game.get("realm_patch"),
                    realm_kind=game.get("realm_kind"),
                ) or None,
                "bans": {"Blue": [], "Red": []},
                "picked": [],
                "unpicked": [],
            }
            game["draft_pool"] = pool
        bans = pool.get("bans") if isinstance(pool.get("bans"), Mapping) else {}
        blue_bans = [str(value).strip() for value in bans.get("Blue", []) if str(value).strip()]
        red_bans = [str(value).strip() for value in bans.get("Red", []) if str(value).strip()]
        phase_one_banned = {_draft_key(value) for value in (*blue_bans[:3], *red_bans[:3])}
        banned = {_draft_key(value) for value in (*blue_bans, *red_bans)}
        bans_complete = len(blue_bans) == 5 and len(red_bans) == 5 and len(banned) == 10
        complete_bans += int(bans_complete)
        picked = pool.get("picked") if isinstance(pool.get("picked"), list) else []
        picked = [dict(item) for item in picked if isinstance(item, Mapping)]
        orders: list[int] = []
        try:
            orders = [int(item.get("order")) for item in picked]
        except (TypeError, ValueError):
            orders = []
        order_complete = (
            len(picked) == 10
            and len({_draft_key(item.get("champion")) for item in picked}) == 10
            and sorted(orders) == list(range(1, 11))
        )
        complete_order += int(order_complete)
        patch = _draft_patch(
            pool.get("patch") or game.get("patch"),
            event_time=game.get("date"),
            realm_patch=game.get("realm_patch"),
            realm_kind=game.get("realm_kind"),
        )
        picked_keys = {_draft_key(item.get("champion")) for item in picked if item.get("champion")}
        universe = {
            row["key"]: row["champion"]
            for (row_patch, _role), scoped in rows_by_scope.items()
            if row_patch == patch
            for row in scoped
        }
        pool["bans"] = {"Blue": blue_bans, "Red": red_bans}
        pool["patch"] = patch or None
        pool["picked"] = picked
        pool["unpicked"] = sorted(
            (champion for key, champion in universe.items() if key not in banned and key not in picked_keys),
            key=str.casefold,
        )
        evaluated = 0
        for item in picked:
            role = _draft_role(item.get("role"))
            champion_key = _draft_key(item.get("champion"))
            item["best_available"] = None
            item["tier_rank"] = None
            item["available_count"] = None
            if not (bans_complete and order_complete and patch and role and champion_key):
                continue
            scoped = rows_by_scope.get((patch, role), [])
            order = int(item["order"])
            bans_before_pick = phase_one_banned if order <= 6 else banned
            prior = {
                _draft_key(previous.get("champion"))
                for previous in picked
                if isinstance(previous, Mapping)
                and previous.get("order") is not None
                and int(previous.get("order")) < order
            }
            available = [
                row
                for row in scoped
                if row["key"] not in bans_before_pick and row["key"] not in prior
            ]
            chosen = next((row for row in scoped if row["key"] == champion_key), None)
            if not chosen or not available:
                continue
            item["tier_rank"] = chosen["rank"]
            item["available_count"] = len(available)
            item["best_available"] = bool(chosen["rank"] == min(row["rank"] for row in available))
            evaluated += 1
        quality_picks += evaluated
        quality_games += int(evaluated == 10)
        pool["status"] = "complete" if evaluated == 10 else "limited" if bans_complete or picked else "unavailable"
        pool["source"] = "published-tier-list"
        if tier_payload_sha256 and SHA256_RE.fullmatch(tier_payload_sha256):
            pool["tier_payload_sha256"] = tier_payload_sha256
        if tier_receipt_sha256 and SHA256_RE.fullmatch(tier_receipt_sha256):
            pool["tier_receipt_sha256"] = tier_receipt_sha256
        pool["basis"] = "lowest published role rank among champions not banned or picked earlier"
        pool["evaluated_picks"] = evaluated
        pool["reason"] = None if evaluated == 10 else "Best-available rate excludes picks without complete ban, order, patch, or tier evidence."

        signal = game.get("draft_contribution")
        if isinstance(signal, dict) and isinstance(signal.get("picks"), list):
            by_identity = {
                (_draft_role(pick.get("role")), str(pick.get("side") or "").title(), _draft_key(pick.get("champion"))): pick
                for pick in picked
            }
            for pick in signal["picks"]:
                if not isinstance(pick, dict):
                    continue
                source = by_identity.get(
                    (_draft_role(pick.get("role")), str(pick.get("side") or "").title(), _draft_key(pick.get("champion")))
                )
                pick["best_available"] = source.get("best_available") if source else None
                pick["tier_rank"] = source.get("tier_rank") if source else None
                pick["available_count"] = source.get("available_count") if source else None

    return {
        "games": total,
        "complete_bans": complete_bans,
        "complete_pick_order": complete_order,
        "quality_games": quality_games,
        "quality_picks": quality_picks,
        "coverage": round(quality_games / total, 4) if total else 0.0,
    }


def export_public_pack(
    *,
    years: Sequence[int] | None = None,
    out_root: Path | None = None,
    pack_id: str | None = None,
    warehouse_root: Path | None = None,
    project_root: Path | None = None,
    runtime_root: Path | None = None,
    tier_payload_path: Path | None = None,
    tier_publication: Mapping[str, Any] | None = None,
    allowed_game_ids: Sequence[str] | None = None,
    promoted_draft_receipt_path: Path | None = None,
    promoted_draft_receipt_sha256: str | None = None,
    promoted_draft_results_path: Path | None = None,
    promoted_draft_results_sha256: str | None = None,
    momentum_window_games: int = DEFAULT_MOMENTUM_WINDOW_GAMES,
    momentum_scale: float = DEFAULT_MOMENTUM_SCALE,
) -> dict[str, Any]:
    require_public_momentum_disabled(
        window_games=momentum_window_games,
        scale=momentum_scale,
        entrypoint="export_public_pack",
    )
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
    stamp = datetime.now(timezone.utc).strftime("%Y.%m.%d.%H%M%S")
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
            "firstPick",
            *(f"ban{slot}" for slot in range(1, 6)),
            *(f"pick{slot}" for slot in range(1, 6)),
            "patch", "blue_firstPick", "red_firstPick",
            *(f"blue_ban{slot}" for slot in range(1, 6)),
            *(f"red_ban{slot}" for slot in range(1, 6)),
            *(f"blue_pick{slot}" for slot in range(1, 6)),
            *(f"red_pick{slot}" for slot in range(1, 6)),
            "dragons", "heralds", "void_grubs", "barons", "atakhans",
            "towers", "inhibitors", "oe_patch_token", "server_patch",
            "game_patch", "realm_patch", "authoritative_patch", "patch_realm",
            "realm_kind", "server_kind",
        ),
        team_available,
    )
    team_source = _canonicalize_game_ids(
        _canonical_pack_frame(pq.read_table(team_path, columns=team_columns).to_pandas())
    )
    team_rating_frame = _filter_year_frame(team_source, years, ("year", "oe_year"))
    team_maps_for_ratings = build_maps_frame_from_team_games(team_rating_frame)
    team_profile_source = _filter_year_frame(team_source, years, ("year", "oe_year"))
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
    # The compact player feed keeps bans but drops team-level pick slots. The
    # team feed retains the complete draft metadata. Overlay those fields onto
    # the canonical map frame before profile records are built.
    draft_metadata_columns = [
        "patch",
        "oe_patch_token", "server_patch", "game_patch", "realm_patch",
        "authoritative_patch", "patch_realm", "realm_kind", "server_kind",
        "blue_firstPick", "red_firstPick",
        *(f"blue_ban{slot}" for slot in range(1, 6)),
        *(f"red_ban{slot}" for slot in range(1, 6)),
        *(f"blue_pick{slot}" for slot in range(1, 6)),
        *(f"red_pick{slot}" for slot in range(1, 6)),
    ]
    metadata_source_columns = [
        column for column in ("game_uid", *draft_metadata_columns)
        if column in team_maps_for_ratings.columns
    ]
    if len(metadata_source_columns) > 1 and "game_uid" in maps_for_records.columns:
        metadata_source = (
            team_maps_for_ratings[metadata_source_columns]
            .drop_duplicates("game_uid")
            .copy()
        )
        maps_for_records = maps_for_records.merge(
            metadata_source,
            on="game_uid",
            how="left",
            suffixes=("", "_team_source"),
        )
        for column in draft_metadata_columns:
            source_column = f"{column}_team_source"
            if source_column not in maps_for_records.columns:
                continue
            source_values = maps_for_records[source_column]
            source_present = source_values.map(
                lambda value: bool(_draft_text(value)),
            )
            if column not in maps_for_records.columns:
                maps_for_records[column] = source_values
            else:
                existing_missing = maps_for_records[column].isna() | ~maps_for_records[column].map(
                    lambda value: bool(_draft_text(value)),
                )
                maps_for_records[column] = maps_for_records[column].where(
                    ~existing_missing | ~source_present,
                    source_values,
                )
            maps_for_records.drop(columns=[source_column], inplace=True)
    source_as_of = pd.to_datetime(maps_for_records["date"], utc=True, errors="coerce").max()
    if pd.isna(source_as_of):
        raise RuntimeError("public pack source has no usable map dates")
    source_game_ids = sorted(set(_normalized_game_uid(maps_for_records).dropna().astype(str)))
    if len(source_game_ids) != len(maps_for_records):
        raise RuntimeError("public pack source is not one row per canonical game identity")
    if (
        not team_profile_source.empty
        and {"game_uid", "gameid", "oe_gameid"}.intersection(team_profile_source.columns)
    ):
        team_profile_source = team_profile_source[
            _normalized_game_uid(team_profile_source).astype(str).isin(source_game_ids)
        ].copy()
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
    team_rating_cfg = DualEloConfig(
        momentum_window_games=momentum_window_games,
        momentum_scale=momentum_scale,
    )
    player_rating_cfg = PlayerEloConfig(
        momentum_window_games=momentum_window_games,
        momentum_scale=momentum_scale,
    )
    progress("building sequential team ratings")
    build_dual_ratings(
        rating_input,
        cfg=team_rating_cfg,
        lineup_by_game=lineup_hashes_from_players(player_rating_input),
        output_dir=features_root,
    )
    progress("building sequential player ratings")
    build_player_ratings(
        player_maps_for_ratings,
        player_rating_input,
        cfg=player_rating_cfg,
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
    sequential_team_snapshot = pd.read_parquet(features_root / "ratings_dual_snapshot.parquet")
    public_ratings = apply_team_momentum_snapshot(
        public_ratings,
        sequential_team_snapshot,
        team_rating_cfg,
    )
    public_ratings.to_parquet(features_root / "ratings_snapshot.parquet", index=False)
    public_ratings_meta["pack_years"] = list(years)
    public_ratings_meta["rating_window"] = "full canonical OE team-game window as this pack"
    public_ratings_meta["source_as_of"] = source_as_of.isoformat().replace("+00:00", "Z")
    public_ratings_meta["source_mode"] = "oe_live" if live_source else "warehouse"
    public_ratings_meta["evidence_contract"] = "2026-08-09.1"
    public_ratings_meta["momentum"] = momentum_manifest_metadata(
        window_games=momentum_window_games,
        scale=momentum_scale,
    )
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
        current=public_ratings,
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
        cfg=player_rating_cfg,
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
            "patch", "oe_patch_token", "server_patch", "game_patch", "realm_patch",
            "authoritative_patch", "patch_realm", "realm_kind", "server_kind",
            "grid_series_id",
            "teamkills", "gamelength", "dpm", "damageshare", "totalgold",
            "total cs", "minionkills", "monsterkills", "cspm", "visionscore",
            "wardsplaced", "wpm", "wcpm", "golddiffat10", "dragons",
            "heralds", "void_grubs", "barons", "atakhans", "towers", "inhibitors",
            "ban1", "ban2", "ban3", "ban4", "ban5",
            "pick1", "pick2", "pick3", "pick4", "pick5", "firstPick",
            "blue_firstPick",
        ),
        player_available,
    )
    player_profile_frame = _profile_archive_frame(
        pq.read_table(player_path, columns=profile_source_columns),
        years,
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
        team_games=team_profile_source,
        draft_metadata=_draft_metadata_from_maps(maps_for_records),
        include_archive=True,
    )
    progress("checking composition publication authority")
    composition_source_digest = source_identity_sha256(source_game_ids)
    composition_worker_commit = resolve_worker_commit(project)
    composition_model_dir = runtime / "data" / "lol" / "models" / "composition_signal"
    descriptive_authority: dict[str, Any] | None = None
    descriptive_receipt_sha256: str | None = None
    composition_error: str | None = None
    try:
        descriptive_authority, descriptive_receipt_sha256 = _load_descriptive_authority(project)
    except CompositionSignalError as error:
        composition_error = str(error)

    composition_games: list[dict[str, Any]] = []
    composition_result: dict[str, Any] | None = None
    draft_records_payload: dict[str, Any] | None = None
    team_draft_records_payload: dict[str, Any] | None = None
    promoted_results_payload: dict[str, Any] | None = None
    descriptive_publication: dict[str, Any] | None = None
    draft_players: list[dict[str, Any]] = []
    composition_audit: dict[str, Any] = {
        "schema_version": "scryglass:composition-signal-descriptive:v1",
        "model_version": DESCRIPTIVE_SCORE_MODEL_VERSION,
        "estimand": "composition_only",
        "status": "unavailable",
        "source_identity_sha256": composition_source_digest,
        "worker_commit": composition_worker_commit,
        "authority": "unavailable",
        "reason": composition_error or "descriptive Draft Score authority is unavailable",
        "probability_authority": False,
    }
    if descriptive_authority is not None and descriptive_receipt_sha256 is not None:
        composition_games = build_composition_games(player_profile_frame)
        descriptive_model, descriptive_artifact_sha256 = load_descriptive_score_model()
        artifact_fit_through: str | None = None
        try:
            parsed_artifact_as_of = pd.to_datetime(
                descriptive_model.get("as_of"), utc=True, errors="raise"
            )
            artifact_fit_through = parsed_artifact_as_of.isoformat().replace("+00:00", "Z")
        except (TypeError, ValueError, OverflowError):
            artifact_fit_through = None
        signals: dict[str, dict[str, Any]] = {}
        for game in composition_games:
            try:
                signals[str(game["game_uid"])] = score_descriptive_game(
                    game,
                    model=descriptive_model,
                    artifact_sha256=descriptive_artifact_sha256,
                )
            except (KeyError, TypeError, ValueError, DescriptiveDraftScoreError):
                continue
        composition_audit = {
            "schema_version": DESCRIPTIVE_SIGNAL_SCHEMA_VERSION,
            "model_version": DESCRIPTIVE_SCORE_MODEL_VERSION,
            "estimand": "composition_only",
            "included_terms": list(DESCRIPTIVE_SCORE_INCLUDED_TERMS),
            "excluded_terms": list(DESCRIPTIVE_SCORE_EXCLUDED_TERMS),
            "training_order": "frozen static artifact; descriptive historical evidence",
            "status": "available" if signals else "unavailable",
            "target_games": len(composition_games),
            "available_games": sum(signal.get("status") == "available" for signal in signals.values()),
            "limited_games": sum(signal.get("status") == "limited" for signal in signals.values()),
            "unavailable_games": len(composition_games) - len(signals),
            "fit_through": artifact_fit_through,
            "artifact_sha256": descriptive_artifact_sha256,
            "source_identity_sha256": composition_source_digest,
            "worker_commit": composition_worker_commit,
            "authority": "descriptive",
            "receipt_sha256": descriptive_receipt_sha256,
            "probability_authority": False,
            "recommendation_authority": False,
            "betting_authority": False,
        }
        composition_audit.update(
            {
                "source_as_of": source_as_of.isoformat().replace("+00:00", "Z"),
                "canonical_game_identity_sha256": composition_source_digest,
                "authority_id": descriptive_authority.get("authority_id"),
                "artifact_sha256": descriptive_artifact_sha256,
                "artifact_commit": descriptive_authority.get("artifact_commit"),
                "source_patch_binding": descriptive_authority.get("source_patch_binding"),
                "evaluation": descriptive_authority.get("evaluation"),
            "player_comfort": descriptive_authority.get("component_contract", {}).get("player_comfort"),
            "archetype_interaction_source": descriptive_authority.get("archetype_interaction_source"),
        }
        )
        composition_result = {"signals": signals, "audit": composition_audit}
        draft_records_payload = build_draft_records_payload(
            composition_result,
            composition_games,
            composition_audit,
        )
        # Team Draft Score needs only a complete composition. Best-available
        # player metrics need the narrower pick-pool gate applied below.
        team_draft_records_payload = copy.deepcopy(draft_records_payload)
        draft_publication = _draft_publication_decision(
            None,
            descriptive_authority=descriptive_authority,
            receipt_sha256=descriptive_receipt_sha256,
            release_id=pack_id,
        )
        descriptive_publication = dict(draft_publication)
        draft_records_payload.update(
            {
                "release_id": pack_id,
                "artifact_sha256": draft_publication.get("artifact_sha256"),
                "authority_receipt_sha256": draft_publication.get("receipt_sha256"),
            }
        )
        for game_id, signal in signals.items():
            if game_id in profile_records_payload.get("games", {}):
                profile_records_payload["games"][game_id]["draft_contribution"] = signal
            archive_candidate = profile_records_payload.get("_archive_games", {}).get(game_id)
            if isinstance(archive_candidate, dict):
                archive_candidate["draft_contribution"] = signal
    else:
        draft_publication = _draft_publication_decision(None)

    promoted_inputs = (
        promoted_draft_receipt_path,
        promoted_draft_receipt_sha256,
        promoted_draft_results_path,
        promoted_draft_results_sha256,
    )
    if any(value is not None for value in promoted_inputs):
        if any(value is None for value in promoted_inputs):
            raise RuntimeError("promoted Draft inputs are incomplete")
        if descriptive_publication is None or draft_records_payload is None:
            raise RuntimeError("promoted Draft release requires descriptive Draft evidence")
        assert promoted_draft_receipt_path is not None
        assert promoted_draft_receipt_sha256 is not None
        assert promoted_draft_results_path is not None
        assert promoted_draft_results_sha256 is not None
        try:
            draft_publication, _promotion_receipt = load_promoted_draft_authority(
                receipt_path=Path(promoted_draft_receipt_path),
                expected_file_sha256=promoted_draft_receipt_sha256,
                release_id=pack_id,
            )
            if (
                not SHA256_RE.fullmatch(promoted_draft_results_sha256)
                or _sha256(Path(promoted_draft_results_path))
                != promoted_draft_results_sha256
            ):
                raise PromotedDraftAuthorityError("promoted Draft result file changed")
            promoted_results_payload = json.loads(
                Path(promoted_draft_results_path).read_text(encoding="utf-8")
            )
            validate_promoted_results_payload(
                promoted_results_payload,
                authority=draft_publication,
            )
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            PromotedDraftAuthorityError,
        ) as error:
            raise RuntimeError("promoted Draft release inputs are invalid") from error
        result_ids = set(map(str, promoted_results_payload["results"]))
        descriptive_ids = set(map(str, draft_records_payload.get("games", {})))
        if not result_ids.issubset(descriptive_ids):
            raise RuntimeError("promoted Draft results are outside descriptive evidence")

    draft_published = draft_publication["status"] in {"descriptive", "promoted"}

    if draft_published and tier_publication is None:
        if any(value is not None for value in promoted_inputs):
            raise RuntimeError(
                "promoted Draft releases require a release-bound tier publication"
            )
        draft_publication = _draft_publication_decision(None)
        draft_published = False
        draft_records_payload = None
        team_draft_records_payload = None
        descriptive_publication = None
        composition_audit.update(
            {
                "status": "unavailable",
                "authority": "unavailable",
                "reason": "release-bound tier publication is unavailable",
                "probability_authority": False,
                "recommendation_authority": False,
                "betting_authority": False,
            }
        )

    tier_receipt_sha256: str | None = None
    if tier_publication is not None:
        if not isinstance(tier_publication, Mapping):
            raise RuntimeError("tier publication binding is malformed")
        publication_path = tier_publication.get("payload_path")
        if not isinstance(publication_path, str) or not publication_path.strip():
            raise RuntimeError("tier publication has no runtime payload path")
        tier_payload_path = Path(publication_path)
        tier_receipt_sha256 = str(tier_publication.get("receipt_sha256") or "") or None
        if tier_publication.get("status") != "available":
            raise RuntimeError("tier publication is not available")
        if tier_publication.get("production_status") not in {"production_built", "production_promoted"}:
            raise RuntimeError("tier publication is not production-bound")
        if not SHA256_RE.fullmatch(str(tier_publication.get("payload_sha256") or "")):
            raise RuntimeError("tier publication payload digest is missing")
        if not SHA256_RE.fullmatch(str(tier_publication.get("receipt_sha256") or "")):
            raise RuntimeError("tier publication receipt digest is missing")
    tier_payload, tier_payload_sha256, tier_payload_source = _load_tier_payload(
        project,
        runtime,
        tier_payload_path,
    )
    if tier_publication is not None and (
        tier_payload is None
        or tier_payload_sha256 != tier_publication.get("payload_sha256")
    ):
        raise RuntimeError("tier publication payload digest does not match the loaded bytes")
    if tier_publication is not None:
        expected_source_identity = str(tier_publication.get("source_identity_sha256") or "")
        if expected_source_identity and expected_source_identity != composition_source_digest:
            raise RuntimeError("tier publication source identity does not match the pack source")
    if draft_published:
        _attach_published_draft_pools(
            profile_records_payload,
            tier_payload,
            tier_payload_sha256=tier_payload_sha256,
            tier_receipt_sha256=tier_receipt_sha256,
        )
        archive_payload = profile_records_payload.get("_archive_games")
        if isinstance(archive_payload, dict):
            _attach_published_draft_pools(
                {"games": archive_payload},
                tier_payload,
                tier_payload_sha256=tier_payload_sha256,
                tier_receipt_sha256=tier_receipt_sha256,
            )

    archive_games = merge_accepted_profile_games(
        profile_records_payload.pop("_archive_games", {}),
        _accepted_profile_games(project),
    )
    if draft_published:
        archive_view = {"games": archive_games}
        _attach_published_draft_pools(
            archive_view,
            tier_payload,
            tier_payload_sha256=tier_payload_sha256,
            tier_receipt_sha256=tier_receipt_sha256,
        )
        _gate_published_draft_contributions(archive_view, draft_records_payload)
        archive_games = archive_view["games"]
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
    if draft_published:
        draft_pool_audit = _attach_published_draft_pools(
            {"games": profile_records_payload["games"]},
            tier_payload,
            tier_payload_sha256=tier_payload_sha256,
            tier_receipt_sha256=tier_receipt_sha256,
        )
        if draft_records_payload is not None:
            for game_id, entry in draft_records_payload.get("games", {}).items():
                profile_game = profile_records_payload.get("games", {}).get(game_id)
                if isinstance(profile_game, Mapping) and isinstance(profile_game.get("draft_pool"), Mapping):
                    entry["draft_pool"] = profile_game["draft_pool"]
        _gate_published_draft_contributions(
            profile_records_payload,
        )
        # Player best-available evidence needs complete ban, order and pool
        # evidence, a narrower denominator than team Draft coverage, which
        # keeps every scoreable complete draft in team_draft_records_payload.
        # Restrict the published Draft records to games whose pool survived the
        # archive gate above. This keeps pool presence identical on both sides,
        # which the bounded query projection requires, and leaves every
        # published record pool-complete, which the Supabase publisher
        # requires. Records are dropped only when their pool evidence is
        # incomplete; the whole-archive team payload is untouched.
        if draft_records_payload is not None:
            gated_games = profile_records_payload.get("games", {})
            records = draft_records_payload.get("games")
            if isinstance(records, dict):
                for record_game_id in list(records):
                    archive_game = gated_games.get(record_game_id)
                    archive_pool = (
                        archive_game.get("draft_pool")
                        if isinstance(archive_game, Mapping)
                        else None
                    )
                    if (
                        not isinstance(archive_pool, Mapping)
                        or archive_pool.get("status") != "complete"
                    ):
                        records.pop(record_game_id, None)
        # The bounded query API accepts a descriptive Draft pick only when its
        # evidence_status is exactly 'available'
        # (supabase/migrations/20260815060001_descriptive_draft_query_api.sql:135).
        # The projection sanitizer also admits 'role_estimate'
        # (lol_kills/export/public_query_projection.py:427), so a game whose
        # role was estimated rather than observed passes every client check and
        # is then refused by the server, which reports the refusal as an
        # invalid canonical row digest. Drop the Draft evidence for those games
        # here, on both sides together, so the archive and the records stay
        # symmetric for the projection and no unpublishable signal is staged.
        # The signal is withheld rather than downgraded: an estimated role is
        # weaker evidence than the descriptive lane asserts.
        archive_games = profile_records_payload.get("games")
        if isinstance(archive_games, dict):
            record_games = (
                draft_records_payload.get("games")
                if isinstance(draft_records_payload, Mapping)
                else None
            )
            for archive_game_id in list(archive_games):
                archive_game = archive_games.get(archive_game_id)
                if not isinstance(archive_game, Mapping):
                    continue
                signal = archive_game.get("draft_contribution")
                if not isinstance(signal, Mapping):
                    continue
                picks = signal.get("picks")
                if not isinstance(picks, list):
                    continue
                if any(
                    not isinstance(pick, Mapping)
                    or pick.get("evidence_status") != "available"
                    for pick in picks
                ):
                    archive_game.pop("draft_contribution", None)
                    archive_game.pop("draft_pool", None)
                    if isinstance(record_games, dict):
                        record_games.pop(archive_game_id, None)
        if not tier_payload_source:
            raise RuntimeError(
                "descriptive Draft release requires an available generated tier payload"
            )
        if not draft_records_payload or not draft_records_payload.get("games"):
            raise RuntimeError("descriptive Draft release has no usable games")
        published_draft_ids = {
            str(game_id) for game_id in draft_records_payload.get("games", {})
        }
        for game_id, game in archive_games.items():
            if str(game_id) not in published_draft_ids and isinstance(game, dict):
                game.pop("draft_contribution", None)
                game.pop("draft_pool", None)
        draft_players = _draft_players_from_signals({}, profile_records_payload)
        profile_records_payload["draft_pool_audit"] = {
            "schema_version": "scryglass:draft-pool-audit:v1",
            "source": "Oracle's Elixir bans and pick order plus published patch tier list",
                "scope": "published profile window after accepted-profile bridge",
            "tier_payload_sha256": tier_payload_sha256,
            "tier_payload_source": tier_payload_source,
            **draft_pool_audit,
        }
        published_composition = _validate_public_composition_records(profile_records_payload)
        composition_audit.update(
            {
                "published_games": published_composition["games"],
                "published_available_games": published_composition["available"],
                "published_limited_games": published_composition["limited"],
                "published_unavailable_games": published_composition["unavailable"],
            }
        )
    else:
        _withhold_unpromoted_draft_fields(profile_records_payload)
        _withhold_unpromoted_draft_fields({"games": archive_games})
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

    if draft_records_payload is not None and draft_published:
        draft_records_dest = feat_dir / "draft_records.json"
        draft_records_dest.write_text(
            json.dumps(draft_records_payload, separators=(",", ":"), ensure_ascii=False),
            encoding="utf-8",
        )
        register(
            {
                "rows": len(draft_records_payload.get("games", {})),
                "cols": None,
                "bytes": draft_records_dest.stat().st_size,
                "sha256": _sha256(draft_records_dest),
                "columns": None,
            },
            "features/draft_records.json",
        )

    if promoted_results_payload is not None:
        promoted_results_dest = feat_dir / "promoted_draft_results.json"
        promoted_results_dest.write_text(
            json.dumps(
                promoted_results_payload,
                separators=(",", ":"),
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        register(
            {
                "rows": len(promoted_results_payload.get("results", {})),
                "cols": None,
                "bytes": promoted_results_dest.stat().st_size,
                "sha256": _sha256(promoted_results_dest),
                "columns": None,
            },
            "features/promoted_draft_results.json",
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
        quarters: dict[int, dict[str, Any]] = {}
        for game_id, game in year_games.items():
            month = int(str(game.get("date") or "")[5:7] or 0)
            quarter = ((month - 1) // 3) + 1
            bucket = quarters.setdefault(
                quarter,
                {"schema_version": archive_payload["schema_version"], "year": archive_year, "games": {}},
            )
            bucket["games"][game_id] = game
        for quarter in (1, 2, 3, 4):
            bucket = quarters.setdefault(
                quarter,
                {"schema_version": archive_payload["schema_version"], "year": archive_year, "games": {}},
            )
            archive_dest = feat_dir / f"match_records_{archive_year}_q{quarter}.json"
            archive_dest.write_text(
                json.dumps(bucket, separators=(",", ":"), ensure_ascii=False),
                encoding="utf-8",
            )
            register(
                {
                    "rows": len(bucket["games"]),
                    "cols": None,
                    "bytes": archive_dest.stat().st_size,
                    "sha256": _sha256(archive_dest),
                    "columns": None,
                },
                f"features/match_records_{archive_year}_q{quarter}.json",
            )
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
        rows, serialized_columns = serialize_rating_snapshot_rows(t, cols)
        dest = feat_dir / out_name
        if out_name == "player_ratings_snapshot.json":
            rows = _public_player_rating_rows(rows)
            for row in rows:
                row["last_team"] = public_team_affiliation(row.get("last_team"))
        dest.write_text(json.dumps(rows), encoding="utf-8")
        register(
            {
                "rows": len(rows),
                "cols": len(serialized_columns),
                "bytes": dest.stat().st_size,
                "sha256": _sha256(dest),
                "columns": serialized_columns,
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
        leaderboards = build_leaderboards(
            player_records_payload,
            profile_records_payload,
            player_rating_rows,
            team_rating_rows,
            team_records=team_records_payload_raw,
            player_champion_records=player_champion_records_raw,
            match_index=match_index_raw,
            draft_records=team_draft_records_payload,
            draft_players=draft_players,
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

    progress("building bounded public query projection")
    query_archive_games = archive_games
    if draft_published and draft_records_payload is not None:
        published_draft_ids = {
            str(game_id) for game_id in draft_records_payload.get("games", {})
        }
        query_archive_games = {}
        for game_id, game in archive_games.items():
            if not isinstance(game, Mapping):
                continue
            query_game = dict(game)
            if str(game_id) not in published_draft_ids:
                query_game.pop("draft_contribution", None)
                query_game.pop("draft_pool", None)
            query_archive_games[str(game_id)] = query_game
    query_projection = build_public_query_projection(
        release_id=pack_id,
        player_ratings=player_rating_rows,
        team_ratings=team_rating_rows,
        player_records=player_records_payload,
        team_records=team_records_payload,
        player_champion_records=player_champions_payload,
        profile_records=profile_records_payload,
        archive_games=query_archive_games,
        player_weekly_ranks=weekly_ranks,
        team_weekly_ranks=team_weekly_ranks,
        player_metadata=player_metadata,
        leaderboards=leaderboards,
        draft_authority=descriptive_publication if draft_published else None,
        draft_records=draft_records_payload,
    )
    if draft_published and draft_records_payload is not None:
        expected_draft_ids = {
            str(game_id) for game_id in draft_records_payload.get("games", {})
        }
        actual_draft_ids = {
            str(row.get("game_id"))
            for row in query_projection.get("datasets", {}).get("games", [])
            if isinstance(row, Mapping)
            and isinstance(row.get("payload"), Mapping)
            and "draft_contribution" in row["payload"]
        }
        if actual_draft_ids != expected_draft_ids:
            raise RuntimeError(
                "public query Draft IDs do not match draft_records: "
                f"expected={len(expected_draft_ids)} actual={len(actual_draft_ids)}"
            )
    query_api_manifest = write_public_query_projection(query_projection, pack_dir)
    del archive_games, query_projection

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
    draft_manifest = {
        "schema_version": "scryglass:draft-authority:v1",
        "status": draft_publication["status"],
        "release_id": pack_id,
        "model_version": draft_publication.get("model_version"),
        "artifact_sha256": draft_publication.get("artifact_sha256"),
        "receipt_sha256": draft_publication.get("receipt_sha256"),
        "issued_utc": draft_publication.get("issued_utc"),
        "reason": draft_publication["reason"],
    }
    if draft_publication["status"] in {"descriptive", "promoted"}:
        draft_manifest.update(
            {
                "authority": draft_publication["authority"],
                "estimand": draft_publication["estimand"],
                "probability_authority": draft_publication["probability_authority"],
                "recommendation_authority": draft_publication["recommendation_authority"],
                "betting_authority": False,
            }
        )
    if draft_publication["status"] == "promoted" and descriptive_publication:
        draft_manifest["descriptive_authority"] = descriptive_publication
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
        "draft_authority": draft_manifest,
        "query_api": query_api_manifest,
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
            "momentum": momentum_manifest_metadata(
                window_games=momentum_window_games,
                scale=momentum_scale,
            ),
            "claim_ceiling": "Source-bound descriptive ratings and historical rank movement only.",
        },
        "draft": composition_audit,
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
    ap.add_argument(
        "--pack-id",
        default=None,
        help="Override pack id (default vYYYY.MM.DD.HHMMSS)",
    )
    ap.add_argument("--warehouse-root", type=Path, default=None, help="Use a source-root overlay for live refreshes")
    ap.add_argument(
        "--tier-payload",
        type=Path,
        default=None,
        help="Use this generated rankings-tierlists-v2 payload for Draft pool evidence",
    )
    ap.add_argument(
        "--momentum-window-games",
        type=int,
        default=DEFAULT_MOMENTUM_WINDOW_GAMES,
        help="Explicit research momentum window in prior maps; default is zero",
    )
    ap.add_argument(
        "--momentum-scale",
        type=float,
        default=DEFAULT_MOMENTUM_SCALE,
        help="Explicit research momentum scale in rating points; default is zero",
    )
    ap.add_argument("--promoted-draft-receipt", type=Path, default=None)
    ap.add_argument("--promoted-draft-receipt-sha256", default=None)
    ap.add_argument("--promoted-draft-results", type=Path, default=None)
    ap.add_argument("--promoted-draft-results-sha256", default=None)
    args = ap.parse_args(argv)
    years = tuple(int(x.strip()) for x in args.years.split(",") if x.strip())
    man = export_public_pack(
        years=years,
        out_root=args.out,
        pack_id=args.pack_id,
        warehouse_root=args.warehouse_root,
        tier_payload_path=args.tier_payload,
        momentum_window_games=args.momentum_window_games,
        momentum_scale=args.momentum_scale,
        promoted_draft_receipt_path=args.promoted_draft_receipt,
        promoted_draft_receipt_sha256=args.promoted_draft_receipt_sha256,
        promoted_draft_results_path=args.promoted_draft_results,
        promoted_draft_results_sha256=args.promoted_draft_results_sha256,
    )
    mb = man["total_bytes"] / (1024 * 1024)
    print(f"Wrote pack {man['pack_id']} → {args.out / man['pack_id']}")
    print(f"Files: {man['total_files']}  Size: {mb:.1f} MB  schema={man['schema_version']}")
    for f in man["files"]:
        print(f"  {f['path']}: {f['bytes']/1024:.0f} KB  rows={f.get('rows')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
