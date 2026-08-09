"""Generate the L9 development tier-list artifact (one league x patch x role cell).

Development-only: the artifact carries rank_eligibility=false and an
all-false claim ceiling; it is never promoted by this module.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from lol_kills.v2.data.source_tree import canonical_source_tree_sha256

from .appearances import AppearanceScope, AppearanceTable
from .artifact import (
    DEFAULT_ARTIFACT_PATH,
    build_tier_list_artifact,
    load_frozen_terminal_model,
    write_tier_list_artifact,
)
from .model import SOURCE_TREE_ALLOWLIST, load_crosswalk_vocabulary


def _default_created_at() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z").replace(".000000", "")


def build_cell(
    root: Path,
    *,
    scope_id: str,
    role: str,
    scope_kind: str = "league",
    competition_tier: str | None = None,
    patch_id: str | None = None,
    as_of: str | None = None,
    created_at: str | None = None,
) -> dict:
    """Build one cell: resolve current patch + as_of from the appearance source."""
    appearances = AppearanceTable.from_oe_player_games(root)
    if patch_id is None:
        patch_id = appearances.latest_patch(
            scope_id, scope_kind=scope_kind, competition_tier=competition_tier, as_of=as_of
        )
    if scope_kind == "league" and competition_tier is None:
        competition_tier = "tier1"
    scope = AppearanceScope(
        scope_kind=scope_kind,
        scope_id=scope_id,
        role=role,
        patch_id=patch_id,
        competition_tier=competition_tier,
    )
    if as_of is None:
        cell = appearances.filter(scope, as_of="9999-12-31T23:59:59Z")
        latest = cell.window()["latest_event_end"]
        if latest is None:
            raise SystemExit(f"no appearances in cell {scope.scope_id} {scope.patch_id} {scope.role}")
        as_of = latest
    cell = appearances.filter(scope, as_of=as_of)
    terminal_model = load_frozen_terminal_model(root)
    crosswalk = load_crosswalk_vocabulary(root)
    source_tree_sha256 = canonical_source_tree_sha256(root, list(SOURCE_TREE_ALLOWLIST))
    payload = build_tier_list_artifact(
        scope=scope,
        as_of=as_of,
        terminal_model=terminal_model,
        crosswalk=crosswalk,
        appearances=cell,
        appearance_source_sha256=appearances.raw_sha256,
        appearance_source_locator=appearances.source_locator,
        created_at=created_at or _default_created_at(),
        source_tree_sha256=source_tree_sha256,
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--league", default="LEC", help="league scope id (e.g. LEC, LCS, LCK, LPL)")
    parser.add_argument("--international", choices=("MSI", "EWC", "WORLDS"), default=None)
    parser.add_argument("--role", default="mid", choices=("top", "jungle", "mid", "bot", "support"))
    parser.add_argument("--competition-tier", default=None, help="tier1 or tier2 (league scope; default tier1)")
    parser.add_argument("--patch", default=None, help="explicit patch id; default current patch")
    parser.add_argument("--as-of", default=None, help="explicit as_of; default latest cell map end")
    parser.add_argument("--out", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    scope_kind = "international" if args.international else "league"
    scope_id = (args.international or args.league).upper()
    payload = build_cell(
        args.root,
        scope_id=scope_id,
        role=args.role,
        scope_kind=scope_kind,
        competition_tier=args.competition_tier,
        patch_id=args.patch,
        as_of=args.as_of,
    )
    if payload["status"] == "unavailable":
        print(f"cell unavailable: {payload['error']['reason']}")
        return 2
    artifact_sha256 = write_tier_list_artifact(args.out, payload, force=args.force)
    print(f"wrote {args.out}")
    print(f"artifact_sha256 = {artifact_sha256}")
    print(f"scope = {payload['scope']['scope_id']} {payload['patch_id']} {payload['role']}")
    print(f"rows = {len(payload['rows'])}, fail_closed_status = {payload['fail_closed_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
