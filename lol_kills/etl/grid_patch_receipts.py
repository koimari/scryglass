"""Build exact GRID game-to-patch crosswalk receipts without target leakage.

The local GRID games table contains a useful exact client patch field, but it
also contains final-state and outcome fields.  This module uses the table only
to resolve a frozen Leaguepedia fixture when team, time, all ten champions,
and all ten player names match exactly.  The emitted receipt contains no
winner or final-state value and remains retrospective when the local source
snapshot was captured after the fixture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from lol_kills.etl.aliases import normalize_team


SCHEMA_VERSION = "scryglass:grid-patch-receipts:v1"
DEFAULT_RUN = Path("data/lol/v2/experiments/leaguepedia/manual-run-2026-07-31")
DEFAULT_SOURCE_GAMES = Path("data/lol/warehouse/grid_drakes/games.parquet")
DEFAULT_SOURCE_CATALOG = Path("data/lol/warehouse/grid_drakes/series_catalog.json")
DEFAULT_OUTPUT = DEFAULT_RUN / "grid-patch-receipts-v1"
MATCH_WINDOW_SECONDS = 7200
OUTCOME_FIELDS = frozenset({"winner_team_id", "complete", "won"})


class GridPatchReceiptError(ValueError):
    """Raised when the crosswalk source or frozen ledger is malformed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_object(value: Any) -> str:
    return _sha_bytes(_canonical(value))


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_json(path: Path, value: Any) -> None:
    _write_atomic(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


def _json_list(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return ()
    else:
        decoded = value
    if not isinstance(decoded, list):
        return ()
    return tuple(str(item).strip() for item in decoded if str(item).strip())


def _public_patch(client_patch: Any) -> str | None:
    text = str(client_patch or "").strip()
    if not text.startswith("16."):
        return None
    try:
        minor = int(text.split(".", 1)[1])
    except (IndexError, ValueError):
        return None
    return f"26.{minor:02d}"


def _source_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _jsonable(value) for key, value in row.items()}


def _teams(row: Mapping[str, Any]) -> frozenset[str]:
    return frozenset(
        {
            normalize_team(str(row.get("team_1_name", ""))),
            normalize_team(str(row.get("team_2_name", ""))),
        }
    )


def _frozen_champions(pregame: Mapping[str, Any]) -> frozenset[str]:
    return frozenset(
        str(champion)
        for side in ("blue", "red")
        for champion in pregame.get(side, {}).get("picks", [])
    )


def _frozen_players(pregame: Mapping[str, Any]) -> frozenset[str]:
    return frozenset(
        str(player.get("player", ""))
        for side in ("blue", "red")
        for player in pregame.get(side, {}).get("players", [])
        if isinstance(player, Mapping) and player.get("player")
    )


def _candidate_score(
    *,
    pregame: Mapping[str, Any],
    grid_row: Mapping[str, Any],
    event_time: pd.Timestamp,
) -> tuple[int, int, float] | None:
    if _teams(grid_row) != frozenset(
        {
            normalize_team(str(pregame.get("blue", {}).get("team", ""))),
            normalize_team(str(pregame.get("red", {}).get("team", ""))),
        }
    ):
        return None
    try:
        grid_time = pd.Timestamp(grid_row.get("date"))
        if grid_time.tzinfo is None:
            grid_time = grid_time.tz_localize("UTC")
        else:
            grid_time = grid_time.tz_convert("UTC")
    except Exception:
        return None
    delta = abs(float((grid_time - event_time).total_seconds()))
    if delta > MATCH_WINDOW_SECONDS:
        return None
    champions_exact = int(
        (
            frozenset(_json_list(grid_row.get("team_1_champions")))
            | frozenset(_json_list(grid_row.get("team_2_champions")))
        )
        == _frozen_champions(pregame)
    )
    players_exact = int(
        (
            frozenset(_json_list(grid_row.get("team_1_players")))
            | frozenset(_json_list(grid_row.get("team_2_players")))
        )
        == _frozen_players(pregame)
    )
    if not champions_exact or not players_exact:
        return None
    return (players_exact, champions_exact, -delta)


def resolve_fixture(
    *,
    fixture_id: str,
    pregame: Mapping[str, Any],
    candidates: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Resolve one fixture, failing closed on absent or duplicate identity."""

    try:
        event_time = pd.Timestamp(str(pregame.get("event_start")))
        if event_time.tzinfo is None:
            event_time = event_time.tz_localize("UTC")
        else:
            event_time = event_time.tz_convert("UTC")
    except Exception:
        event_time = None
    scored: list[tuple[tuple[int, int, float], Mapping[str, Any]]] = []
    if event_time is not None:
        for candidate in candidates:
            score = _candidate_score(pregame=pregame, grid_row=candidate, event_time=event_time)
            if score is not None:
                scored.append((score, candidate))
    blockers: list[str] = []
    selected: Mapping[str, Any] | None = None
    if event_time is None:
        blockers.append("event_start_invalid")
    elif not scored:
        blockers.append("no_exact_grid_game_identity")
    else:
        scored.sort(key=lambda item: item[0], reverse=True)
        best_score = scored[0][0]
        best = [item for item in scored if item[0] == best_score]
        if len(best) != 1:
            blockers.append("ambiguous_grid_game_identity")
        else:
            selected = best[0][1]

    evidence: dict[str, Any] = {
        "source_kind": "grid_games_parquet",
        "candidate_count": len(scored),
        "outcome_fields_excluded": sorted(OUTCOME_FIELDS),
    }
    if selected is not None:
        client_patch = str(selected.get("patch", ""))
        patch = _public_patch(client_patch)
        if patch is None:
            blockers.append("grid_patch_field_invalid")
        evidence.update(
            {
                "grid_series_id": str(selected.get("series_id", "")),
                "grid_game_id": str(selected.get("game_id", "")),
                "grid_tournament_id": str(selected.get("tournament_id", "")),
                "grid_patch": client_patch,
                "public_patch": patch,
                "grid_date": str(selected.get("date", "")),
                "source_row_sha256": _sha_object(_source_row(selected)),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": fixture_id,
        "event_start": str(pregame.get("event_start", "")),
        "as_of": str(pregame.get("as_of", "")),
        "patch": evidence.get("public_patch"),
        "client_patch": evidence.get("grid_patch"),
        "authority_status": "confirmed_metadata" if selected is not None and not blockers else "unavailable",
        "pregame_authorized": False,
        "blockers": sorted(set(blockers)),
        "evidence": evidence,
        "evidence_hash": _sha_object({"fixture_id": fixture_id, "evidence": evidence}),
    }


def _load_frozen(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) and isinstance(row.get("pregame"), Mapping) for row in rows):
        raise GridPatchReceiptError("frozen ledger contains an invalid pregame row")
    return rows


def build_receipts(
    run_dir: Path,
    output_dir: Path,
    *,
    source_games: Path = DEFAULT_SOURCE_GAMES,
    source_catalog: Path = DEFAULT_SOURCE_CATALOG,
) -> dict[str, Any]:
    rows = _load_frozen(run_dir / "frozen-ledger.jsonl")
    if not source_games.exists() or not source_catalog.exists():
        raise GridPatchReceiptError("GRID source games or catalog file is missing")
    games_bytes = source_games.read_bytes()
    catalog_bytes = source_catalog.read_bytes()
    catalog = json.loads(catalog_bytes)
    if not isinstance(catalog, Mapping):
        raise GridPatchReceiptError("GRID series catalog is not an object")
    source_captured_at = str(catalog.get("generatedAt", ""))
    games = pd.read_parquet(source_games)
    games["_grid_time"] = pd.to_datetime(games["date"], utc=True, errors="coerce")
    index: dict[tuple[Any, frozenset[str]], list[dict[str, Any]]] = defaultdict(list)
    for raw in games.to_dict(orient="records"):
        grid_time = raw.get("_grid_time")
        if pd.isna(grid_time):
            continue
        index[(grid_time.date(), _teams(raw))].append(raw)

    receipts: list[dict[str, Any]] = []
    for row in rows:
        pregame = row["pregame"]
        event_start = pd.Timestamp(str(pregame.get("event_start")))
        teams = frozenset(
            {
                normalize_team(str(pregame.get("blue", {}).get("team", ""))),
                normalize_team(str(pregame.get("red", {}).get("team", ""))),
            }
        )
        candidates: list[Mapping[str, Any]] = []
        for day_delta in (-1, 0, 1):
            day = (event_start + pd.Timedelta(days=day_delta)).date()
            candidates.extend(index.get((day, teams), []))
        receipt = resolve_fixture(
            fixture_id=str(pregame.get("fixture_id", "")),
            pregame=pregame,
            candidates=candidates,
        )
        if receipt["authority_status"] == "confirmed_metadata":
            try:
                captured = pd.Timestamp(source_captured_at)
                cutoff = pd.Timestamp(str(pregame.get("as_of")))
                if captured.tzinfo is None:
                    captured = captured.tz_localize("UTC")
                if cutoff.tzinfo is None:
                    cutoff = cutoff.tz_localize("UTC")
                if captured <= cutoff:
                    receipt["pregame_authorized"] = True
                else:
                    receipt["blockers"].append("grid_source_captured_after_cutoff")
            except Exception:
                receipt["blockers"].append("grid_source_capture_time_invalid")
            receipt["blockers"] = sorted(set(receipt["blockers"]))
        receipts.append(receipt)

    receipts.sort(key=lambda row: (row["event_start"], row["fixture_id"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = output_dir / "patch-receipts.jsonl"
    _write_atomic(
        receipt_path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in receipts).encode("utf-8"),
    )
    confirmed = sum(row["authority_status"] == "confirmed_metadata" for row in receipts)
    pregame_authorized = sum(row["pregame_authorized"] for row in receipts)
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "run_dir": str(run_dir),
        "captured_at": _utc_now(),
        "source_games": str(source_games),
        "source_games_sha256": _sha_bytes(games_bytes),
        "source_catalog": str(source_catalog),
        "source_catalog_sha256": _sha_bytes(catalog_bytes),
        "source_captured_at": source_captured_at,
        "fixture_count": len(receipts),
        "confirmed_metadata_fixture_count": confirmed,
        "pregame_authorized_fixture_count": pregame_authorized,
        "unavailable_fixture_count": len(receipts) - confirmed,
        "receipt_file": str(receipt_path),
        "receipt_file_sha256": _sha_bytes(receipt_path.read_bytes()),
        "outcome_fields_emitted": False,
        "claim_ceiling": {
            "exact_patch_identity": confirmed > 0,
            "pregame_patch_authority": pregame_authorized == len(receipts),
            "winner_prediction": False,
            "publication": False,
        },
    }
    manifest = {**unsigned, "manifest_sha256": _sha_object(unsigned)}
    _write_json(output_dir / "receipt-manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-games", type=Path, default=DEFAULT_SOURCE_GAMES)
    parser.add_argument("--source-catalog", type=Path, default=DEFAULT_SOURCE_CATALOG)
    args = parser.parse_args(argv)
    try:
        result = build_receipts(
            args.run_dir,
            args.output_dir,
            source_games=args.source_games,
            source_catalog=args.source_catalog,
        )
    except (OSError, ValueError, GridPatchReceiptError) as exc:
        print(str(exc))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
