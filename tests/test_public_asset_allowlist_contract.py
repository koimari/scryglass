from __future__ import annotations

import re
from pathlib import Path

from lol_kills.export.pack_spec import PUBLIC_ASSET_PATHS
from lol_kills.export import supabase_publication


ROOT = Path(__file__).resolve().parents[1]
FINAL_MIGRATION = ROOT / "supabase/migrations/20260814010000_bounded_public_query_api.sql"


def _quoted_paths(source: str, start: str, end: str) -> set[str]:
    section = source.split(start, 1)[1].split(end, 1)[0]
    return set(re.findall(r'["\']((?:features|rankings)/[^"\']+\.json)["\']', section))


def test_public_asset_allowlist_has_cross_language_parity() -> None:
    expected = set(PUBLIC_ASSET_PATHS)
    assert set(supabase_publication.PUBLIC_ASSET_PATHS) == expected

    server_pack = (ROOT / "apps/scryglass/src/lib/serverPack.ts").read_text(encoding="utf-8")
    server_paths = _quoted_paths(
        server_pack,
        "export const PUBLIC_ASSET_PATHS = new Set([",
        "]);",
    )

    boundary = (ROOT / "apps/scryglass/scripts/audit-public-boundary.mjs").read_text(
        encoding="utf-8"
    )
    boundary_paths = _quoted_paths(boundary, "const allowedAssetPaths = new Set([", "]);")

    migration = FINAL_MIGRATION.read_text(encoding="utf-8")
    sql_paths = _quoted_paths(
        migration,
        "-- Final effective asset contract. PUBLIC_ASSET_ALLOWLIST_V1",
        "));",
    )

    assert server_paths == expected
    assert boundary_paths == expected
    assert sql_paths == expected
