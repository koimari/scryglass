"""Private, outcome-free completed-draft features for the accepted G1 maps.

This is deliberately a materialization boundary, not a draft model.  It reads
only the six participant columns from the already-approved Oracle's Elixir
snapshot and only the non-target fields from the accepted G1 rows.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from typing import Any
from datetime import datetime

import pandas as pd

from lol_kills.v2.champions import id_crosswalk


ROOT = Path(__file__).resolve().parents[3]
BASE_ROWS = ROOT / "data/lol/v2/snapshots/real-v1/lpl-private-development-rows.jsonl"
BASE_MANIFEST = ROOT / "data/lol/v2/snapshots/real-v1/lpl-private-development-manifest.json"
PLAYER_GAMES = ROOT / "data/lol/warehouse/parquet/oe_player_games.parquet"
CROSSWALK = ROOT / "data/lol/v2/champions/champion-id-crosswalk-v1.json"
OUTPUT_ROWS = ROOT / "data/lol/v2/snapshots/real-v1/lpl-private-draft-features-rows.jsonl"
OUTPUT_MANIFEST = ROOT / "data/lol/v2/snapshots/real-v1/lpl-private-draft-features-manifest.json"

BASE_ROWS_SHA256 = "4ed79abb0b2471a666ab5643b91edf33c2fdde19e361c456aa589d2e9a4df846"
BASE_MANIFEST_SHA256 = "3af87fffb2b32fd95aeb920409abe0254fa158b3dc7f079650b3472731d4ff72"
BASE_MANIFEST_RAW_SHA256 = "dca3c7b8fb5c6bc6bc4ebf8779448bbbb0728a592d85461bee479c5f39d608e1"
PLAYER_GAMES_SHA256 = "c19b364ccafa8f149f8365885f1ae8ae68242e62ead73ecc1d290f5c212edfd5"
SELECTED_TARGET_SHA256 = "4c332fa4e6cb155341bcffd83bd0ee1be2e04f3b5950b8a7745931253dd8bd2d"
SPLIT_PAYLOAD_SHA256 = "469c8d2c568a6a4480db277bf41f7eacf72964e33997f0a4e1f53f60285cd3e4"
TARGET_AUTHORITY_RAW_SHA256 = "b1d0a6e37abb9a74dee8689dc19ab54d30fd15516bd4ee454906a075d8f20788"
TARGET_EVIDENCE_SHA256 = "6697ed142324f86e9b233c4a2b36dd501584e7e64449bb6cd9404f6a367d74f9"
CROSSWALK_RAW_SHA256 = "8a57b26bd33d27e6a81393bdffc450db0adeb9e4a9d2ee221a8244e2f6452555"
CROSSWALK_CANONICAL_SHA256 = "59fdf214b570487f64e08f060ba51c82b24e87f3bbe4d6e308fd1bdd42ef14f7"
G2_ARTIFACT = ROOT / "data/lol/v2/models/player/real-v1/private-development-artifact.json"
G2_ARTIFACT_RAW_SHA256 = "b0d8276fd164735db0abd9b2353c7e10168c599e5607e4db3d15cd12bd9d7b50"
G2_ARTIFACT_CANONICAL_SHA256 = "35e8831fb4d39fd60ec7f8f59b934ff5571f788ec8dc1151c78661b67ab6d4fd"
G2_FOLD_MAP_DIGESTS = {"DEVELOPMENT": "975981c6df01c45b2734a3d0eeebb35153e1b313e550c18ed636ec514cc3b0e4", "TRAIN": "1ec97562fd182ce5f5e806b2e6f21aebe24b7702bf3a25685a4d7864605243dc", "VALIDATION": "1c631e4963fc1bb7d2e862a5271a8aec065f413fde170bb204d379a0e087b75f"}
G2_FOLD_ORIGIN_DIGESTS = {"DEVELOPMENT": "f3cb56e7efdf9068c8374c3e1bc57e33b36a24ddd9e93fa1c5cde88a45e6795e", "TRAIN": "aae8c05cebffce28f7a01d8f3ae9fe4987b143f9fca091612b7ca23b7556548d", "VALIDATION": "348ecd6ba6c0bdc47d17396496764357d254fcb6e39e832b44ebead2947650f5"}
SCHEMA = "scryglass:g1-lpl-completed-draft-features:v1"
ROLES = ("top", "jungle", "mid", "bot", "support")
ROLE_ALIASES = {"top": "top", "jng": "jungle", "jungle": "jungle", "mid": "mid", "bot": "bot", "sup": "support", "support": "support"}
SIDE_ALIASES = {"blue": "blue", "red": "red"}
SOURCE_COLUMNS = ("gameid", "side", "position", "playerid", "teamid", "champion")
TRANSFORM_CONFIG = {"source_columns": list(SOURCE_COLUMNS), "source_projection_sort": list(SOURCE_COLUMNS), "roles": list(ROLES), "role_aliases": dict(sorted(ROLE_ALIASES.items())), "side_aliases": dict(sorted(SIDE_ALIASES.items())), "map_sort": ["source_local_event_start", "source_game_id"], "pick_sort": ["source_side", "role_order"], "unknown_champion_policy": "typed_map_unavailability_no_alias_borrowing", "availability": "completed_draft_identity_known_by_game_start_but_retrospective_source_has_no_historical_ingest_current_live_or_forecast_authority"}


class G1DraftFeatureError(ValueError):
    """A source or identity invariant cannot be safely materialized."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def raw_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_file(path: Path, *, root: Path = ROOT, label: str = "file") -> Path:
    if ".." in path.parts:
        raise G1DraftFeatureError(f"{label} traversal rejected")
    absolute = path.absolute()
    try:
        relative = absolute.relative_to(root.absolute())
    except ValueError as error:
        raise G1DraftFeatureError(f"{label} outside repository") from error
    current = root.absolute()
    for index, part in enumerate(relative.parts):
        current = current / part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError as error:
            raise G1DraftFeatureError(f"{label} missing") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise G1DraftFeatureError(f"{label} symlink rejected")
        if index < len(relative.parts) - 1 and not stat.S_ISDIR(metadata.st_mode):
            raise G1DraftFeatureError(f"{label} unsafe parent")
    metadata = os.lstat(absolute)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise G1DraftFeatureError(f"{label} must be unaliased regular file")
    return absolute


def _expect_raw(path: Path, expected: str, *, label: str) -> Path:
    path = _safe_file(path, label=label)
    if raw_sha256(path) != expected:
        raise G1DraftFeatureError(f"{label} raw sha256 mismatch")
    return path


def _base_rows(path: Path = BASE_ROWS, manifest_path: Path = BASE_MANIFEST) -> list[dict[str, Any]]:
    """Read only the G1 map/lineup surface; `target` is never accessed."""

    rows_path = _expect_raw(path, BASE_ROWS_SHA256, label="base rows")
    manifest_path = _expect_raw(manifest_path, BASE_MANIFEST_RAW_SHA256, label="base manifest")
    manifest = json.loads(manifest_path.read_bytes())
    unsigned = dict(manifest)
    claimed = unsigned.pop("manifest_sha256", None)
    if claimed != BASE_MANIFEST_SHA256 or sha256(unsigned) != claimed or manifest_path.read_bytes() != canonical_bytes(manifest) + b"\n":
        raise G1DraftFeatureError("base manifest identity mismatch")
    if manifest.get("rows_sha256") != BASE_ROWS_SHA256 or manifest.get("canonical_selected_target_rows_sha256") != SELECTED_TARGET_SHA256:
        raise G1DraftFeatureError("base manifest row/target pin mismatch")
    if manifest.get("target_authority", {}).get("split_payload_sha256") != SPLIT_PAYLOAD_SHA256:
        raise G1DraftFeatureError("base manifest split pin mismatch")
    if manifest.get("target_authority", {}).get("authority_raw_sha256") != TARGET_AUTHORITY_RAW_SHA256 or manifest.get("target_authority", {}).get("evidence_payload_sha256") != TARGET_EVIDENCE_SHA256:
        raise G1DraftFeatureError("base target authority pin mismatch")
    if manifest.get("final_holdout", {}).get("status") != "SEALED_UNREAD" or manifest.get("final_holdout", {}).get("accessed") is not False:
        raise G1DraftFeatureError("base final holdout boundary mismatch")

    selected: list[dict[str, Any]] = []
    allowed = {"source_game_id", "partition", "source_local_event_start", "observed_lineups", "game_side"}
    for line in rows_path.read_bytes().splitlines():
        raw = json.loads(line)
        # This whitelist is the outcome boundary: target/result keys are not
        # read, emitted, or represented in any logical projection.
        row = {key: raw[key] for key in allowed}
        if row["partition"] not in {"TRAIN", "DEVELOPMENT", "VALIDATION"}:
            raise G1DraftFeatureError("final or unknown G1 membership")
        selected.append(row)
    selected.sort(key=lambda value: (value["source_local_event_start"], value["source_game_id"]))
    if len(selected) != 1226 or len({row["source_game_id"] for row in selected}) != 1226:
        raise G1DraftFeatureError("accepted G1 map membership drift")
    return selected


def _crosswalk_table(vocabulary: Iterable[str], path: Path = CROSSWALK) -> tuple[dict[str, str], dict[str, str]]:
    path = _safe_file(path, label="champion crosswalk")
    raw = path.read_bytes()
    payload = json.loads(raw)
    if raw_sha256(path) != CROSSWALK_RAW_SHA256 or payload.get("artifact_sha256") != CROSSWALK_CANONICAL_SHA256:
        raise G1DraftFeatureError("champion crosswalk immutable pin mismatch")
    try:
        id_crosswalk.validate_artifact(payload)
    except id_crosswalk.ChampionIdCrosswalkError as error:
        raise G1DraftFeatureError("champion crosswalk authority invalid") from error
    if raw != canonical_bytes(payload):
        raise G1DraftFeatureError("champion crosswalk noncanonical bytes")
    try:
        verified = id_crosswalk.load_and_replay_artifact(path)
    except id_crosswalk.ChampionIdCrosswalkError as error:
        raise G1DraftFeatureError("champion crosswalk source-backed replay failed") from error
    # This is an exact source-vocabulary boundary, not a fallback matcher.
    # A future/unknown raw string is retained and typed unavailable downstream.
    recognized = {str(entry["oe_name"]) for entry in payload["entries"]}
    table: dict[str, str] = {}
    for name in sorted(set(vocabulary)):
        if name not in recognized:
            continue
        try:
            table[name] = id_crosswalk.resolve_champion_id(verified, name)
        except id_crosswalk.ChampionIdCrosswalkError as error:
            # The name is authenticated as recognized, so a resolver failure is
            # integrity/staleness, never an invitation to borrow an alias.
            raise G1DraftFeatureError("champion crosswalk recognized-name resolution failed") from error
    generator = payload["generator"]["executable_dependency_boundary"][0]
    return table, {"locator": str(path.relative_to(ROOT)), "raw_sha256": CROSSWALK_RAW_SHA256, "canonical_sha256": CROSSWALK_CANONICAL_SHA256, "generator_locator": generator["locator"], "generator_raw_sha256": generator["raw_sha256"], "metadata_raw_sha256": payload["metadata"]["raw_sha256"], "player_games_raw_sha256": payload["warehouse_sources"]["player_games"]["raw_sha256"]}


def _g2_origin_identity() -> dict[str, Any]:
    path = _expect_raw(G2_ARTIFACT, G2_ARTIFACT_RAW_SHA256, label="accepted G2 origin")
    payload = json.loads(path.read_bytes())
    unsigned = dict(payload)
    if unsigned.pop("artifact_sha256", None) != G2_ARTIFACT_CANONICAL_SHA256 or sha256(unsigned) != G2_ARTIFACT_CANONICAL_SHA256:
        raise G1DraftFeatureError("accepted G2 origin canonical identity mismatch")
    pins = payload.get("adapter_input_pins")
    if not isinstance(pins, Mapping) or pins.get("fold_map_digests") != G2_FOLD_MAP_DIGESTS or pins.get("fold_origin_digests") != G2_FOLD_ORIGIN_DIGESTS:
        raise G1DraftFeatureError("accepted G2 origin fold identity mismatch")
    return {"g2_artifact_locator": str(path.relative_to(ROOT)), "g2_artifact_raw_sha256": G2_ARTIFACT_RAW_SHA256, "g2_artifact_canonical_sha256": G2_ARTIFACT_CANONICAL_SHA256, "fold_map_digests": G2_FOLD_MAP_DIGESTS, "fold_origin_digests": G2_FOLD_ORIGIN_DIGESTS}


def _projection(source_rows: Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
    projected: list[dict[str, str]] = []
    for item in source_rows:
        if set(item) != set(SOURCE_COLUMNS):
            raise G1DraftFeatureError("selected source projection column mismatch")
        row = {key: str(item[key]) for key in SOURCE_COLUMNS}
        if any(not value.strip() or value.lower() == "nan" for value in row.values()):
            raise G1DraftFeatureError("selected source projection missing value")
        projected.append(row)
    projected.sort(key=lambda row: tuple(row[key] for key in SOURCE_COLUMNS))
    return projected


def _normalize_side(value: str) -> str:
    try:
        return SIDE_ALIASES[value.strip().casefold()]
    except KeyError as error:
        raise G1DraftFeatureError("illegal source side") from error


def _normalize_role(value: str) -> str:
    try:
        return ROLE_ALIASES[value.strip().casefold()]
    except KeyError as error:
        raise G1DraftFeatureError("illegal source role") from error


def _expected_participants(base: Mapping[str, Any]) -> set[tuple[str, str, str, str]]:
    expected: set[tuple[str, str, str, str]] = set()
    lineups = base["observed_lineups"]
    if not isinstance(lineups, list) or len(lineups) != 2:
        raise G1DraftFeatureError("base lineup side count mismatch")
    for lineup in lineups:
        side = _normalize_side(str(lineup["observed_game_side"]))
        team = str(lineup["team_id"])
        players = lineup["player_ids_by_role"]
        if set(players) != set(ROLES):
            raise G1DraftFeatureError("base lineup legal role mismatch")
        for role in ROLES:
            expected.add((side, role, str(players[role]), team))
    if len(expected) != 10:
        raise G1DraftFeatureError("base lineup duplicate participant")
    return expected


def materialize_from_projection(
    *, base_rows: Sequence[Mapping[str, Any]], source_rows: Iterable[Mapping[str, Any]], champion_table: Mapping[str, str]
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Pure no-label transformation used by both real and hostile tests."""

    projection = _projection(source_rows)
    seen_base_ids: set[str] = set()
    for base in base_rows:
        game_id = str(base.get("source_game_id", ""))
        if not game_id or game_id in seen_base_ids:
            raise G1DraftFeatureError("duplicate or missing base source game id")
        seen_base_ids.add(game_id)
        if base.get("partition") not in {"TRAIN", "DEVELOPMENT", "VALIDATION"}:
            raise G1DraftFeatureError("final or unknown G1 membership")
        try:
            parsed_time = datetime.fromisoformat(str(base.get("source_local_event_start")))
        except ValueError as error:
            raise G1DraftFeatureError("malformed source-local event time") from error
        if parsed_time.tzinfo is not None:
            raise G1DraftFeatureError("source-local event time must remain naive")
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in projection:
        grouped.setdefault(row["gameid"], []).append(row)
    output: list[dict[str, Any]] = []
    for base in base_rows:
        game_id = str(base["source_game_id"])
        entries = grouped.pop(game_id, None)
        if entries is None or len(entries) != 10:
            raise G1DraftFeatureError("accepted map does not have exactly ten source picks")
        expected = _expected_participants(base)
        observed: set[tuple[str, str, str, str]] = set()
        picks: list[dict[str, Any]] = []
        unavailable: list[str] = []
        for source in entries:
            side, role = _normalize_side(source["side"]), _normalize_role(source["position"])
            participant = (side, role, source["playerid"], source["teamid"])
            observed.add(participant)
            stable = champion_table.get(source["champion"])
            if stable is None:
                unavailable.append(source["champion"])
            picks.append({"source_side": side, "role": role, "source_player_id": source["playerid"], "source_team_id": source["teamid"], "source_champion": source["champion"], "stable_champion_id": stable})
        if observed != expected or len(observed) != 10:
            raise G1DraftFeatureError("source participant differs from accepted G1 observed lineup")
        if {pick["source_side"] for pick in picks} != {"blue", "red"} or any(sum(pick["source_side"] == side for pick in picks) != 5 for side in ("blue", "red")):
            raise G1DraftFeatureError("source side completion mismatch")
        if any({pick["role"] for pick in picks if pick["source_side"] == side} != set(ROLES) for side in ("blue", "red")):
            raise G1DraftFeatureError("source role completion mismatch")
        if len({pick["source_player_id"] for pick in picks}) != 10 or len({pick["source_team_id"] for pick in picks}) != 2:
            raise G1DraftFeatureError("source player/team completion mismatch")
        if len({pick["source_champion"] for pick in picks}) != 10:
            raise G1DraftFeatureError("global champion duplication")
        resolved = [pick["stable_champion_id"] for pick in picks if pick["stable_champion_id"] is not None]
        if len(resolved) != len(set(resolved)):
            raise G1DraftFeatureError("global champion duplication")
        picks.sort(key=lambda pick: (pick["source_side"], ROLES.index(pick["role"])))
        output.append({"schema_version": SCHEMA, "source_game_id": game_id, "partition": base["partition"], "source_local_event_start": base["source_local_event_start"], "availability": "COMPLETED_DRAFT_AVAILABLE_AT_OR_BEFORE_EVENT_START" if not unavailable else "TYPED_UNAVAILABLE_CHAMPION_IDENTITY", "unavailability": None if not unavailable else {"code": "UNKNOWN_OR_AMBIGUOUS_SOURCE_CHAMPION", "source_champions": sorted(set(unavailable))}, "picks": picks})
    if grouped:
        # Remaining source maps do not change membership, but accepted-map
        # identity is exact and projection is separately hashed.
        pass
    output.sort(key=lambda row: (row["source_local_event_start"], row["source_game_id"]))
    return output, projection


def _safe_write_many(items: Sequence[tuple[Path, bytes]]) -> None:
    if len({path.absolute() for path, _ in items}) != len(items):
        raise G1DraftFeatureError("duplicate output paths")
    prepared: list[tuple[Path, bytes]] = []
    existing: list[bool] = []
    for path, data in items:
        parent = path.parent
        _safe_file(parent / ".keep", label="impossible") if False else None
        if ".." in path.parts:
            raise G1DraftFeatureError("output traversal rejected")
        absolute = path.absolute()
        try:
            relative = absolute.relative_to(ROOT.absolute())
        except ValueError as error:
            raise G1DraftFeatureError("output outside repository") from error
        current = ROOT.absolute()
        for part in relative.parts[:-1]:
            current = current / part
            metadata = os.lstat(current)
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise G1DraftFeatureError("unsafe output parent")
        try:
            leaf = os.lstat(path)
        except FileNotFoundError:
            leaf = None
        if leaf is not None and (stat.S_ISLNK(leaf.st_mode) or not stat.S_ISREG(leaf.st_mode) or leaf.st_nlink != 1):
            raise G1DraftFeatureError("unsafe output leaf")
        existing.append(leaf is not None)
        prepared.append((path, data))
    if any(existing):
        if not all(existing):
            raise G1DraftFeatureError("immutable output pair is incomplete")
        if all(path.read_bytes() == data for path, data in prepared):
            return
        raise G1DraftFeatureError("immutable output already exists with a different generation")
    staged: list[tuple[Path, str]] = []
    committed: list[Path] = []
    try:
        for path, data in prepared:
            fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
            with os.fdopen(fd, "wb") as handle:
                handle.write(data); handle.flush(); os.fsync(handle.fileno())
            staged.append((path, temporary))
        for path, temporary in staged:
            os.replace(temporary, path)
            committed.append(path)
    except BaseException:
        for path in reversed(committed):
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
        raise
    finally:
        for _, temporary in staged:
            if os.path.exists(temporary):
                os.unlink(temporary)


def build(
    *, rows_path: Path = OUTPUT_ROWS, manifest_path: Path = OUTPUT_MANIFEST,
    base_rows_path: Path = BASE_ROWS, base_manifest_path: Path = BASE_MANIFEST,
    player_games_path: Path = PLAYER_GAMES, crosswalk_path: Path = CROSSWALK,
) -> dict[str, Any]:
    base = _base_rows(base_rows_path, base_manifest_path)
    player_games_path = _expect_raw(player_games_path, PLAYER_GAMES_SHA256, label="OE player-games")
    frame = pd.read_parquet(player_games_path, columns=list(SOURCE_COLUMNS))
    selected = frame.loc[frame["gameid"].astype(str).isin({row["source_game_id"] for row in base}), list(SOURCE_COLUMNS)]
    source_rows = selected.to_dict("records")
    if len(source_rows) != 12260:
        raise G1DraftFeatureError("accepted source participant row count drift")
    table, crosswalk_identity = _crosswalk_table((str(row["champion"]) for row in source_rows), crosswalk_path)
    rows, projection = materialize_from_projection(base_rows=base, source_rows=source_rows, champion_table=table)
    if len(rows) != 1226 or sum(len(row["picks"]) for row in rows) != 12260:
        raise G1DraftFeatureError("completed-draft row/pick count drift")
    row_bytes = b"".join(canonical_bytes(row) + b"\n" for row in rows)
    rows_raw = hashlib.sha256(row_bytes).hexdigest()
    rows_canonical = sha256(rows)
    source_projection_hash = sha256(projection)
    manifest = {
        "schema_version": SCHEMA,
        "rows_locator": str(rows_path.relative_to(ROOT)), "rows_raw_sha256": rows_raw, "rows_canonical_sha256": rows_canonical,
        "coverage": {"accepted_map_count": 1226, "feature_row_count": len(rows), "pick_count": 12260, "identity_unavailable_map_count": sum(row["availability"] != "COMPLETED_DRAFT_AVAILABLE_AT_OR_BEFORE_EVENT_START" for row in rows)},
        "source": {"locator": str(player_games_path.relative_to(ROOT)), "raw_sha256": PLAYER_GAMES_SHA256, "rights_status": "PRIVATE_REVIEWED", "selected_columns": list(SOURCE_COLUMNS), "selected_projection_row_count": len(projection), "selected_projection_canonical_sha256": source_projection_hash, "selected_projection_sort": list(SOURCE_COLUMNS)},
        "base_g1": {"rows_locator": str(base_rows_path.relative_to(ROOT)), "rows_raw_sha256": BASE_ROWS_SHA256, "manifest_locator": str(base_manifest_path.relative_to(ROOT)), "manifest_raw_sha256": BASE_MANIFEST_RAW_SHA256, "manifest_canonical_sha256": BASE_MANIFEST_SHA256, "selected_target_sha256": SELECTED_TARGET_SHA256, "split_payload_sha256": SPLIT_PAYLOAD_SHA256, "target_authority_raw_sha256": TARGET_AUTHORITY_RAW_SHA256, "target_evidence_canonical_sha256": TARGET_EVIDENCE_SHA256},
        "accepted_membership_origin": _g2_origin_identity(),
        "champion_crosswalk": crosswalk_identity,
        "transform": {"locator": "lol_kills/v2/data/g1_draft_features.py", "raw_sha256": raw_sha256(Path(__file__)), "config": TRANSFORM_CONFIG, "config_sha256": sha256(TRANSFORM_CONFIG)},
        "availability": {"kind": "COMPLETED_DRAFT_AVAILABLE_AT_OR_BEFORE_EVENT_START", "event_time": "source_local_event_start_from_accepted_G1_only", "limitation": "mechanically known by game start; the retrospective snapshot supplies no historical-ingest receipt or current/live/forecast authority", "not_authorized": ["historical_ingest", "current", "live", "forecast", "prediction"]},
        "final_holdout": {"status": "SEALED_UNREAD", "accessed": False, "included": False},
        "claim_ceiling": {"private_model_fit": True, "private_rank_selection": True, "prediction": False, "publication": False, "promotion": False, "sota": False, "final_holdout": False, "public_pack": False},
    }
    manifest["manifest_sha256"] = sha256(manifest)
    _safe_write_many(((rows_path, row_bytes), (manifest_path, canonical_bytes(manifest) + b"\n")))
    return manifest


def verify(*, rows_path: Path = OUTPUT_ROWS, manifest_path: Path = OUTPUT_MANIFEST, expected_manifest_sha256: str) -> dict[str, Any]:
    rows_path = _safe_file(rows_path, label="draft feature rows")
    manifest_path = _safe_file(manifest_path, label="draft feature manifest")
    manifest = json.loads(manifest_path.read_bytes())
    unsigned = dict(manifest); claimed = unsigned.pop("manifest_sha256", None)
    if not isinstance(expected_manifest_sha256, str) or claimed != sha256(unsigned) or claimed != expected_manifest_sha256:
        raise G1DraftFeatureError("draft feature manifest identity mismatch")
    if manifest.get("schema_version") != SCHEMA or manifest.get("rows_raw_sha256") != raw_sha256(rows_path):
        raise G1DraftFeatureError("draft feature persisted row identity mismatch")
    rows = [json.loads(line) for line in rows_path.read_bytes().splitlines()]
    if len(rows) != 1226 or sha256(rows) != manifest.get("rows_canonical_sha256"):
        raise G1DraftFeatureError("draft feature canonical row identity mismatch")
    if any("target" in row or len(row.get("picks", [])) != 10 for row in rows):
        raise G1DraftFeatureError("draft feature outcome or completion boundary mismatch")
    if manifest.get("transform") != {"locator": "lol_kills/v2/data/g1_draft_features.py", "raw_sha256": raw_sha256(Path(__file__)), "config": TRANSFORM_CONFIG, "config_sha256": sha256(TRANSFORM_CONFIG)}:
        raise G1DraftFeatureError("draft feature transform bytes drift")
    base = _base_rows()
    player_games = _expect_raw(PLAYER_GAMES, PLAYER_GAMES_SHA256, label="OE player-games")
    frame = pd.read_parquet(player_games, columns=list(SOURCE_COLUMNS))
    selected = frame.loc[frame["gameid"].astype(str).isin({row["source_game_id"] for row in base}), list(SOURCE_COLUMNS)]
    table, crosswalk_identity = _crosswalk_table((str(row["champion"]) for row in selected.to_dict("records")))
    if manifest.get("champion_crosswalk") != crosswalk_identity:
        raise G1DraftFeatureError("champion crosswalk identity drift")
    replayed_rows, projection = materialize_from_projection(base_rows=base, source_rows=selected.to_dict("records"), champion_table=table)
    source = manifest.get("source", {})
    expected_source = {"locator": str(PLAYER_GAMES.relative_to(ROOT)), "raw_sha256": PLAYER_GAMES_SHA256, "rights_status": "PRIVATE_REVIEWED", "selected_columns": list(SOURCE_COLUMNS), "selected_projection_row_count": len(projection), "selected_projection_canonical_sha256": sha256(projection), "selected_projection_sort": list(SOURCE_COLUMNS)}
    if source != expected_source:
        raise G1DraftFeatureError("selected source projection identity mismatch")
    if replayed_rows != rows:
        raise G1DraftFeatureError("draft feature replay mismatch")
    if manifest.get("coverage") != {"accepted_map_count": 1226, "feature_row_count": 1226, "pick_count": 12260, "identity_unavailable_map_count": 0}:
        raise G1DraftFeatureError("draft feature coverage mismatch")
    if manifest.get("base_g1") != {"rows_locator": str(BASE_ROWS.relative_to(ROOT)), "rows_raw_sha256": BASE_ROWS_SHA256, "manifest_locator": str(BASE_MANIFEST.relative_to(ROOT)), "manifest_raw_sha256": BASE_MANIFEST_RAW_SHA256, "manifest_canonical_sha256": BASE_MANIFEST_SHA256, "selected_target_sha256": SELECTED_TARGET_SHA256, "split_payload_sha256": SPLIT_PAYLOAD_SHA256, "target_authority_raw_sha256": TARGET_AUTHORITY_RAW_SHA256, "target_evidence_canonical_sha256": TARGET_EVIDENCE_SHA256}:
        raise G1DraftFeatureError("base G1 authority identity mismatch")
    if manifest.get("accepted_membership_origin") != _g2_origin_identity():
        raise G1DraftFeatureError("accepted membership origin identity mismatch")
    if manifest.get("availability") != {"kind": "COMPLETED_DRAFT_AVAILABLE_AT_OR_BEFORE_EVENT_START", "event_time": "source_local_event_start_from_accepted_G1_only", "limitation": "mechanically known by game start; the retrospective snapshot supplies no historical-ingest receipt or current/live/forecast authority", "not_authorized": ["historical_ingest", "current", "live", "forecast", "prediction"]}:
        raise G1DraftFeatureError("availability authority mismatch")
    if manifest.get("final_holdout") != {"status": "SEALED_UNREAD", "accessed": False, "included": False} or manifest.get("claim_ceiling") != {"private_model_fit": True, "private_rank_selection": True, "prediction": False, "publication": False, "promotion": False, "sota": False, "final_holdout": False, "public_pack": False}:
        raise G1DraftFeatureError("claim or final boundary mismatch")
    return manifest
