#!/usr/bin/env python3
"""Verify the public Scryglass release through anonymous HTTP responses."""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from datetime import datetime, timezone
from typing import Any


def _read(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"Cache-Control": "no-cache", "User-Agent": "scryglass-release-verifier/1"},
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        if response.status != 200:
            raise RuntimeError(f"{url} returned HTTP {response.status}")
        return response.read()


def _json(url: str) -> dict[str, Any]:
    value = json.loads(_read(url))
    if not isinstance(value, dict):
        raise RuntimeError(f"{url} did not return one JSON object")
    return value


def _count(pattern: str, text: str, label: str) -> int:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match is None:
        raise RuntimeError(f"could not read {label} from production HTML")
    return int(match.group(1))


def _timestamp(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise RuntimeError(f"{label} is missing")
    try:
        # Manifest timestamps arrive in two shapes: writers that normalize to
        # a trailing Z, and receipt passthroughs that keep +00:00. Appending a
        # suffix blindly turned the second shape into "...+00:00+00:00" and
        # failed the smoke against a perfectly valid release.
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise RuntimeError(f"{label} is invalid") from error
    if parsed.tzinfo is None:
        raise RuntimeError(f"{label} has no timezone")
    return parsed.astimezone(timezone.utc)


def _asset_json(site: str, manifest: dict[str, Any], path: str) -> dict[str, Any]:
    files = manifest.get("files")
    if not isinstance(files, list):
        raise RuntimeError("public manifest has no file inventory")
    matches = [row for row in files if isinstance(row, dict) and row.get("path") == path]
    if len(matches) != 1:
        raise RuntimeError(f"public manifest does not contain exactly one {path}")
    url = matches[0].get("url")
    release_id = str(manifest.get("release_id") or "")
    if not isinstance(url, str) or not url.startswith(f"/api/assets/{release_id}/"):
        raise RuntimeError(f"public manifest has an invalid {path} URL")
    return _json(f"{site}{url}")


def _verify_census(
    manifest: dict[str, Any],
    draft: dict[str, Any],
    matches: dict[str, Any],
    tiers: dict[str, Any],
) -> dict[str, int]:
    release_id = str(manifest.get("release_id") or "")
    source_as_of = _timestamp(manifest.get("source_as_of"), "manifest source_as_of")
    source_game_count = manifest.get("source_game_count")
    source_identity = manifest.get("source_identity_sha256")
    if not isinstance(source_game_count, int) or source_game_count < 1:
        raise RuntimeError("manifest source_game_count is invalid")
    if not isinstance(source_identity, str) or not re.fullmatch(r"[0-9a-f]{64}", source_identity):
        raise RuntimeError("manifest source_identity_sha256 is invalid")

    if draft.get("release_id") != release_id:
        raise RuntimeError("Draft records have a different release")
    if _timestamp(draft.get("source_as_of"), "Draft source_as_of") != source_as_of:
        raise RuntimeError("Draft records have a different source timestamp")
    if draft.get("source_identity_sha256") != source_identity:
        raise RuntimeError("Draft records have a different source identity")
    sample_window = draft.get("sample_window")
    if not isinstance(sample_window, dict) or sample_window.get("target_games") != source_game_count:
        raise RuntimeError("Draft records have a different source game count")
    if _timestamp(tiers.get("as_of"), "Tier Lists as_of") != source_as_of:
        raise RuntimeError("Tier Lists have a different source timestamp")

    match_rows = matches.get("games")
    draft_rows = draft.get("games")
    if not isinstance(match_rows, list) or not isinstance(draft_rows, dict):
        raise RuntimeError("match or Draft census is malformed")
    match_ids: list[str] = []
    complete_ids: set[str] = set()
    for row in match_rows:
        if not isinstance(row, dict) or not isinstance(row.get("game_id"), str):
            raise RuntimeError("match census contains an invalid game")
        game_id = row["game_id"]
        match_ids.append(game_id)
        champions = row.get("champions")
        if (
            isinstance(champions, list)
            and len(champions) == 10
            and all(isinstance(champion, str) and champion.strip() for champion in champions)
            and len(set(champions)) == 10
        ):
            complete_ids.add(game_id)
    if len(match_ids) != len(set(match_ids)):
        raise RuntimeError("match census contains duplicate game IDs")

    draft_ids = set(map(str, draft_rows))
    missing = complete_ids - draft_ids
    extra = draft_ids - complete_ids
    if missing or extra:
        raise RuntimeError(
            "Draft Score coverage differs from the complete ten-pick match census: "
            f"missing={len(missing)}, extra={len(extra)}"
        )

    return {
        "source_games": source_game_count,
        "public_matches": len(match_ids),
        "draft_scores": len(draft_ids),
    }


def _current_results_expected(matches: dict[str, Any], now: datetime | None = None) -> bool:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    rows = matches.get("games")
    if not isinstance(rows, list):
        return False
    for row in rows:
        if not isinstance(row, dict) or row.get("competition_tier") != "tier1":
            continue
        try:
            played = _timestamp(row.get("date"), "match date")
        except RuntimeError:
            continue
        if (played.year, played.month) == (current.year, current.month):
            return True
    return False


def _verify_once(site: str, expected_release: str | None) -> tuple[dict[str, Any], str]:
    health = _json(f"{site}/api/health")
    manifest = _json(f"{site}/packs/manifest.json")
    release_id = str(manifest.get("release_id") or manifest.get("pack_id") or "")
    if not re.fullmatch(r"v\d{4}\.\d{2}\.\d{2}\.\d{6}", release_id):
        raise RuntimeError("public manifest has no canonical release ID")
    if expected_release and release_id != expected_release:
        raise RuntimeError(f"served release {release_id} differs from {expected_release}")
    if health.get("status") != "ok" or health.get("stale") is not False:
        raise RuntimeError("public health is not ready")

    draft_records = _asset_json(site, manifest, "features/draft_records.json")
    match_index = _asset_json(site, manifest, "features/match_index.json")
    tier_records = _asset_json(site, manifest, "rankings/tierlists-latest.json")
    census = _verify_census(manifest, draft_records, match_index, tier_records)

    draft = _read(f"{site}/elo?tab=draft").decode("utf-8", "replace")
    tiers = _read(f"{site}/tiers").decode("utf-8", "replace")
    matches = _read(f"{site}/matches?section=results").decode("utf-8", "replace")
    for name, page in (("Draft", draft), ("Tier Lists", tiers), ("Matches", matches)):
        if release_id not in page:
            raise RuntimeError(f"{name} page has a different release")

    team_count = _count(r"(\d+)\s+teams", draft, "team Draft count")
    patch_games = _count(r"(\d+)\s+patch games", tiers, "patch game count")
    if team_count < 1:
        raise RuntimeError("team Draft leaderboard is empty")
    if patch_games < 1:
        raise RuntimeError("latest patch census is empty")
    if _current_results_expected(match_index) and "No accepted games match these filters." in matches:
        raise RuntimeError("default completed-match view is empty")

    return ({
        "release_id": release_id,
        "status": "ok",
        "team_draft_rows": team_count,
        "latest_patch_games": patch_games,
        **census,
    }, release_id)


def verify(site: str, expected_release: str | None) -> dict[str, Any]:
    site = site.rstrip("/")
    for attempt in range(2):
        result, release_id = _verify_once(site, expected_release)
        final_manifest = _json(f"{site}/packs/manifest.json")
        final_release = str(final_manifest.get("release_id") or final_manifest.get("pack_id") or "")
        if final_release == release_id:
            return result
        if expected_release or attempt == 1:
            raise RuntimeError("active release changed during verification")
    raise RuntimeError("active release changed during verification")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", default="https://scryglass.xyz")
    parser.add_argument("--release-id")
    args = parser.parse_args()
    print(json.dumps(verify(args.site, args.release_id), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
