"""Capture result-free, hash-bound Leaguepedia patch receipts.

The frozen ledger already has Leaguepedia game identifiers.  This capture asks
Cargo only for the game identity, teams, timestamp, and patch; winner, kills,
length, and other outcome fields are never requested or emitted.  A receipt is
still retrospective when the source is fetched after the map cutoff.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from lol_kills.etl.aliases import normalize_team
from lol_kills.net import require_https_url


SCHEMA_VERSION = "scryglass:leaguepedia-patch-receipts:v1"
DEFAULT_RUN = Path("data/lol/v2/experiments/leaguepedia/manual-run-2026-07-31")
DEFAULT_OUTPUT = DEFAULT_RUN / "leaguepedia-patch-receipts-v1"
API_URL = "https://lol.fandom.com/wiki/Special:CargoExport"
USER_AGENT = "Scryglass-leaguepedia-patch-receipts/1.0"
FIELDS = (
    "ScoreboardGames.GameId",
    "ScoreboardGames.Team1",
    "ScoreboardGames.Team2",
    "ScoreboardGames.Patch",
    "ScoreboardGames.DateTime_UTC",
)
OUTCOME_FIELDS = frozenset(
    {
        "WinTeam",
        "LossTeam",
        "Team1Kills",
        "Team2Kills",
        "Gamelength_Number",
        "winner_team_id",
        "complete",
        "won",
    }
)


class LeaguepediaPatchReceiptError(ValueError):
    """Raised when a patch receipt input is malformed."""


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_object(value: Any) -> str:
    return _sha_bytes(_canonical(value))


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_json(path: Path, value: Any) -> None:
    _write_atomic(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _load_ledger(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, Mapping) and isinstance(row.get("pregame"), Mapping) for row in rows):
        raise LeaguepediaPatchReceiptError("frozen ledger contains an invalid pregame row")
    return [dict(row) for row in rows]


def _batches(values: list[str], size: int = 35) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _cargo_url(game_ids: list[str]) -> str:
    where = "(" + " OR ".join(
        f'ScoreboardGames.GameId="{game_id.replace(chr(34), chr(92) + chr(34))}"'
        for game_id in game_ids
    ) + ")"
    params = {
        "tables": "ScoreboardGames",
        "fields": ",".join(FIELDS),
        "where": where,
        "order_by": "ScoreboardGames.DateTime_UTC ASC",
        "limit": "500",
        "format": "json",
    }
    return API_URL + "?" + urllib.parse.urlencode(params)


def _fetch(url: str, timeout: int) -> bytes:
    url = require_https_url(url, hosts={"lol.fandom.com"})
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _patch_label(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if number < 26:
        return None
    return f"{number:.2f}"


def _client_patch(value: str | None) -> str | None:
    if not value or not value.startswith("26."):
        return None
    try:
        return f"16.{int(value.split('.', 1)[1])}"
    except (IndexError, ValueError):
        return None


def _row_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "fixture_id": str(row.get("GameId", "")),
        "team_1": str(row.get("Team1", "")),
        "team_2": str(row.get("Team2", "")),
        "patch": _patch_label(row.get("Patch")),
        "event_start": str(row.get("DateTime UTC", "")),
    }


def build_receipts(
    run_dir: Path,
    output_dir: Path,
    *,
    batch_size: int = 35,
    delay: float = 0.35,
    timeout: int = 120,
) -> dict[str, Any]:
    ledger = _load_ledger(run_dir / "frozen-ledger.jsonl")
    fixture_ids = [str(row["pregame"].get("fixture_id", "")) for row in ledger]
    if not all(fixture_ids):
        raise LeaguepediaPatchReceiptError("frozen ledger contains a fixture without an ID")
    captured_at = _now()
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    pages: list[dict[str, Any]] = []
    source_rows: dict[str, list[dict[str, Any]]] = {}
    for page_index, batch in enumerate(_batches(fixture_ids, batch_size)):
        url = _cargo_url(batch)
        raw = _fetch(url, timeout)
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise LeaguepediaPatchReceiptError(f"patch page {page_index} did not return an array")
        page_name = f"patches-page-{page_index:03d}.json"
        page_path = raw_dir / page_name
        _write_atomic(page_path, raw)
        pages.append(
            {
                "page": page_name,
                "fixture_ids": batch,
                "row_count": len(parsed),
                "sha256": _sha_bytes(raw),
                "captured_at": captured_at,
                "api_url": url,
                "requested_fields": list(FIELDS),
            }
        )
        for row in parsed:
            if not isinstance(row, Mapping):
                continue
            identity = _row_identity(row)
            source_rows.setdefault(identity["fixture_id"], []).append(
                {**identity, "raw_row_sha256": _sha_object(identity), "page_sha256": _sha_bytes(raw)}
            )
        if delay:
            time.sleep(delay)

    receipts: list[dict[str, Any]] = []
    for ledger_row in ledger:
        pregame = ledger_row["pregame"]
        fixture_id = str(pregame.get("fixture_id", ""))
        candidates = source_rows.get(fixture_id, [])
        blockers: list[str] = []
        selected: dict[str, Any] | None = None
        if len(candidates) != 1:
            blockers.append("leaguepedia_patch_identity_ambiguous" if len(candidates) > 1 else "leaguepedia_patch_identity_missing")
        else:
            selected = candidates[0]
            if not selected.get("patch"):
                blockers.append("leaguepedia_patch_field_missing")
            blockers.append("leaguepedia_source_captured_after_cutoff")
        frozen_teams = {
            normalize_team(str(pregame.get("blue", {}).get("team", ""))),
            normalize_team(str(pregame.get("red", {}).get("team", ""))),
        }
        source_teams = (
            {
                normalize_team(str(selected.get("team_1", ""))),
                normalize_team(str(selected.get("team_2", ""))),
            }
            if selected is not None
            else set()
        )
        evidence = {
            "source_kind": "leaguepedia_scoreboardgames_patch",
            "source_captured_at": captured_at,
            "candidate_count": len(candidates),
            "team_set_matches": selected is not None and source_teams == frozen_teams,
            "outcome_fields_requested": [],
            "outcome_fields_emitted": sorted(OUTCOME_FIELDS.intersection(set().union(*(set(row) for row in candidates)))) if candidates else [],
        }
        if selected is not None:
            evidence.update(
                {
                    "source_row_sha256": selected["raw_row_sha256"],
                    "source_page_sha256": selected["page_sha256"],
                    "source_event_start": selected["event_start"],
                    "source_team_1": selected["team_1"],
                    "source_team_2": selected["team_2"],
                }
            )
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "fixture_id": fixture_id,
            "event_start": str(pregame.get("event_start", "")),
            "as_of": str(pregame.get("as_of", "")),
            "patch": selected.get("patch") if selected else None,
            "client_patch": _client_patch(selected.get("patch")) if selected else None,
            "authority_status": "confirmed_metadata" if selected is not None and not any(
                item.endswith("missing") or item.endswith("mismatch") for item in blockers
            ) else "unavailable",
            "pregame_authorized": False,
            "blockers": sorted(set(blockers)),
            "evidence": evidence,
        }
        receipt["evidence_hash"] = _sha_object({"fixture_id": fixture_id, "evidence": evidence})
        receipts.append(receipt)

    receipts.sort(key=lambda row: (row["event_start"], row["fixture_id"]))
    receipt_path = output_dir / "patch-receipts.jsonl"
    receipt_bytes = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in receipts).encode("utf-8")
    _write_atomic(receipt_path, receipt_bytes)
    confirmed = sum(row["authority_status"] == "confirmed_metadata" for row in receipts)
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "run_dir": str(run_dir),
        "captured_at": captured_at,
        "fixture_count": len(receipts),
        "confirmed_metadata_fixture_count": confirmed,
        "pregame_authorized_fixture_count": 0,
        "unavailable_fixture_count": len(receipts) - confirmed,
        "raw_pages": pages,
        "receipt_file": str(receipt_path),
        "receipt_file_sha256": _sha_bytes(receipt_bytes),
        "requested_fields": list(FIELDS),
        "outcome_fields_requested": [],
        "outcome_fields_emitted": False,
        "claim_ceiling": {
            "exact_patch_identity": confirmed == len(receipts),
            "pregame_patch_authority": False,
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
    parser.add_argument("--batch-size", type=int, default=35)
    parser.add_argument("--delay", type=float, default=0.35)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args(argv)
    try:
        result = build_receipts(
            args.run_dir,
            args.output_dir,
            batch_size=args.batch_size,
            delay=args.delay,
            timeout=args.timeout,
        )
    except (OSError, ValueError, LeaguepediaPatchReceiptError) as exc:
        print(str(exc))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
