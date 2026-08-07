"""Build the development tier-list scope index: all default cells + options.

Contract cell artifacts (one league x patch x role) are built with the
canonical builder (lol_kills.v2.tierlists.artifact) and stored under
data/lol/v2/tierlists/cells/.  The index lists every cell with its locator,
raw sha256, and scope metadata, plus the filter vocabulary (regions, leagues,
international events, competition tiers, roles, patches) that the public
surface uses.  The index is a development convenience layer; the cells are
the canonical artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from lol_kills.v2.data.common import ROLES

from .artifact import write_tier_list_artifact
from .generate_artifacts import build_cell
from .schema import COMPETITION_TIERS, INTERNATIONAL_SCOPES, REGIONS, TIERLIST_SCHEMA_ID

DEFAULT_INDEX_PATH = Path("data/lol/v2/tierlists/index-v1.json")
DEFAULT_CELLS_DIR = Path("data/lol/v2/tierlists/cells")

# Default headline scope set: tier1 domestic leagues + international events.
DEFAULT_LEAGUES = ("LEC", "LCS", "LCK", "LPL")
DEFAULT_EVENTS = ("MSI", "EWC")


def _raw_sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _cell_file_name(scope_id: str, role: str, patch_id: str) -> str:
    safe = scope_id.lower().replace(" ", "-")
    return f"tierlist-{safe}-{patch_id}-{role}-development-v1.json"


def build_index(root: Path, *, index_path: Path = DEFAULT_INDEX_PATH, cells_dir: Path = DEFAULT_CELLS_DIR) -> dict:
    cells_dir_abs = root / cells_dir
    cells_dir_abs.mkdir(parents=True, exist_ok=True)
    cells: list[dict] = []
    for league in DEFAULT_LEAGUES:
        for role in ROLES:
            payload = build_cell(root, scope_id=league, role=role, scope_kind="league", competition_tier="tier1")
            fname = _cell_file_name(league, role, payload["patch_id"])
            path = cells_dir_abs / fname
            write_tier_list_artifact(path, payload, force=True)
            cells.append({
                "artifact_id": payload["artifact_id"],
                "scope_kind": "league",
                "league": league,
                "event_kind": None,
                "competition_tier": "tier1",
                "role": role,
                "patch_id": payload["patch_id"],
                "as_of": payload["as_of"],
                "locator": str(cells_dir / fname),
                "raw_sha256": _raw_sha256(path.read_bytes()),
                "row_count": len(payload["rows"]),
                "status": payload["status"],
                "fail_closed_status": payload["fail_closed_status"],
            })
    for event in DEFAULT_EVENTS:
        for role in ROLES:
            payload = build_cell(root, scope_id=event, role=role, scope_kind="international")
            fname = _cell_file_name(event, role, payload["patch_id"])
            path = cells_dir_abs / fname
            write_tier_list_artifact(path, payload, force=True)
            cells.append({
                "artifact_id": payload["artifact_id"],
                "scope_kind": "international",
                "league": None,
                "event_kind": event.lower(),
                "competition_tier": "international",
                "role": role,
                "patch_id": payload["patch_id"],
                "as_of": payload["as_of"],
                "locator": str(cells_dir / fname),
                "raw_sha256": _raw_sha256(path.read_bytes()),
                "row_count": len(payload["rows"]),
                "status": payload["status"],
                "fail_closed_status": payload["fail_closed_status"],
            })
    payload = {
        "schema_version": TIERLIST_SCHEMA_ID,
        "artifact_kind": "tier_list_scope_index",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "development_only": True,
        "cells": cells,
        "options": {
            "regions": sorted(REGIONS),
            "leagues": DEFAULT_LEAGUES,
            "event_kinds": [e.lower() for e in DEFAULT_EVENTS],
            "competition_tiers": ["tier1", "international"],
            "roles": list(ROLES),
            "patches": sorted({c["patch_id"] for c in cells}),
        },
    }
    payload["artifact_sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in payload.items() if k != "artifact_sha256"},
                   sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (root / index_path).parent.mkdir(parents=True, exist_ok=True)
    (root / index_path).write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def mirror_to_app(root: Path, cells_dir: Path = DEFAULT_CELLS_DIR) -> None:
    """Mirror index + cells into apps/lol-atlas/public/v2/tierlists so the
    Vercel lambda can read them (Turbopack only traces paths inside the app
    root).  The public mirror is a serving copy; the repo data stays canonical.
    """
    import shutil

    dst = root / "apps/lol-atlas/public/v2/tierlists"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "cells").mkdir(parents=True, exist_ok=True)
    for name in ("index-v1.json",):
        shutil.copy(root / DEFAULT_INDEX_PATH, dst / name)
    for f in sorted((root / cells_dir).glob("*.json")):
        shutil.copy(f, dst / "cells" / f.name)
    print(f"mirrored tier-list data to {dst}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build all default tier-list cells + the scope index.")
    ap.add_argument("--root", type=Path, default=Path.cwd())
    ap.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    ap.add_argument("--no-mirror", action="store_true", help="skip the apps/lol-atlas public mirror")
    args = ap.parse_args()
    index = build_index(args.root, index_path=args.index)
    if not args.no_mirror:
        mirror_to_app(args.root)
    print(f"wrote {args.index}")
    print(f"cells = {len(index['cells'])}, generated_at = {index['generated_at']}")
    print("artifact_sha256 =", index["artifact_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
