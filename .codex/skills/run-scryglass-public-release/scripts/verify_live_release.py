#!/usr/bin/env python3
"""Verify the public Scryglass release through anonymous HTTP responses."""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
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


def verify(site: str, expected_release: str | None) -> dict[str, Any]:
    site = site.rstrip("/")
    health = _json(f"{site}/api/health")
    manifest = _json(f"{site}/packs/manifest.json")
    release_id = str(manifest.get("release_id") or manifest.get("pack_id") or "")
    if not re.fullmatch(r"v\d{4}\.\d{2}\.\d{2}\.\d{6}", release_id):
        raise RuntimeError("public manifest has no canonical release ID")
    if expected_release and release_id != expected_release:
        raise RuntimeError(f"served release {release_id} differs from {expected_release}")
    if health.get("status") != "ok" or health.get("stale") is not False:
        raise RuntimeError("public health is not ready")

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
    if "No accepted games match these filters." in matches:
        raise RuntimeError("default completed-match view is empty")

    return {
        "release_id": release_id,
        "status": "ok",
        "team_draft_rows": team_count,
        "latest_patch_games": patch_games,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", default="https://scryglass.xyz")
    parser.add_argument("--release-id")
    args = parser.parse_args()
    print(json.dumps(verify(args.site, args.release_id), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
