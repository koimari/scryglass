"""Capture pre-period Leaguepedia player/draft rows for temporal snapshots.

The existing manual run captures July drafts and outcomes separately.  This
companion captures the player rows for the already-sealed January--June game
catalog so a rolling scorer can fit only on information available before each
July map.  Raw pages are retained and hashed; no outcome fields are requested.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lol_kills.etl.manual_leaguepedia_batch import _batches, _cargo_url, _fetch, _json_rows, _safe


SCHEMA_VERSION = "scryglass:leaguepedia-autoresearch-prior-drafts:v1"
PLAYER_FIELDS = (
    "ScoreboardPlayers.GameId",
    "ScoreboardPlayers.Team",
    "ScoreboardPlayers.Name",
    "ScoreboardPlayers.Champion",
    "ScoreboardPlayers.IngameRole",
    "ScoreboardPlayers.Side",
    "ScoreboardPlayers.DateTime_UTC",
)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _rfc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _catalog_rows(catalog_dir: Path) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for page in sorted(catalog_dir.glob("prior-page-*.json")):
        for row in _json_rows(page):
            game_id = _safe(row.get("GameId"))
            if game_id:
                rows.setdefault(game_id, row)
    output = list(rows.values())
    output.sort(key=lambda row: (_safe(row.get("DateTime UTC")), _safe(row.get("GameId"))))
    return output


def capture_prior_drafts(run_dir: Path, *, sleep_s: float = 0.35, batch_size: int = 35) -> dict[str, Any]:
    catalog_dir = run_dir / "raw" / "prior-games"
    output_dir = run_dir / "raw" / "prior-drafts"
    output_dir.mkdir(parents=True, exist_ok=True)
    games = _catalog_rows(catalog_dir)
    if not games:
        raise ValueError(f"no prior catalog rows found in {catalog_dir}")
    observed_at = _rfc_now()
    fields = ",".join(PLAYER_FIELDS)
    pages: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    game_ids = [str(row["GameId"]) for row in games]

    for page_index, batch in enumerate(_batches(game_ids, size=batch_size)):
        where = "(" + " OR ".join(
            f'ScoreboardPlayers.GameId="{game_id.replace(chr(34), chr(92) + chr(34))}"'
            for game_id in batch
        ) + ")"
        url = _cargo_url("ScoreboardPlayers", fields, where, limit=500)
        raw = _fetch(url)
        filename = f"prior-draft-page-{page_index:03d}.json"
        (output_dir / filename).write_bytes(raw)
        rows = _json_rows(output_dir / filename)
        pages.append(
            {
                "page": filename,
                "api_url": url,
                "available_at": observed_at,
                "game_ids": batch,
                "row_count": len(rows),
                "sha256": _sha(raw),
            }
        )
        all_rows.extend(rows)
        if sleep_s > 0 and page_index + 1 < (len(game_ids) + batch_size - 1) // batch_size:
            time.sleep(sleep_s)

    by_game = {game_id: 0 for game_id in game_ids}
    normalized: list[dict[str, Any]] = []
    for row in all_rows:
        game_id = _safe(row.get("GameId"))
        if game_id not in by_game:
            continue
        by_game[game_id] += 1
        normalized.append(
            {
                "game_id": game_id,
                "team": _safe(row.get("Team")),
                "player": _safe(row.get("Name")),
                "champion": _safe(row.get("Champion")),
                "role": _safe(row.get("IngameRole")),
                "side": _safe(row.get("Side")),
                "date": _safe(row.get("DateTime UTC")),
            }
        )
    normalized.sort(key=lambda row: (row["date"], row["game_id"], row["side"], row["role"], row["player"]))
    _write_jsonl(output_dir / "normalized-prior-draft-rows.jsonl", normalized)
    missing = sorted(game_id for game_id, count in by_game.items() if count == 0)
    incomplete = sorted(game_id for game_id, count in by_game.items() if count not in {0, 10})
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "pre-event rolling Draft Score runtime seed",
        "observed_at": observed_at,
        "catalog_dir": str(catalog_dir),
        "fields": list(PLAYER_FIELDS),
        "game_count": len(game_ids),
        "unique_game_ids": len(by_game),
        "raw_row_count": len(all_rows),
        "normalized_row_count": len(normalized),
        "pages": pages,
        "missing_game_ids": missing,
        "incomplete_game_ids": incomplete,
        "draft_rows_contain_outcome_fields": any(
            any(key in row for key in ("WinTeam", "Winner", "Team1Kills", "Team2Kills", "PlayerWin"))
            for row in all_rows
        ),
    }
    _write_json(output_dir / "prior-draft-capture-manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sleep", type=float, default=0.35)
    parser.add_argument("--batch-size", type=int, default=35)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 50:
        raise SystemExit("--batch-size must be between 1 and 50")
    print(json.dumps(capture_prior_drafts(args.run_dir, sleep_s=args.sleep, batch_size=args.batch_size), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
