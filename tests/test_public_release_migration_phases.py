from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS = ROOT / "supabase/migrations"


def test_rollout_migrations_exist_in_order() -> None:
    names = sorted(path.name for path in MIGRATIONS.glob("20260814*.sql"))
    assert names.index("20260814000001_add_quarter_match_records.sql") < names.index(
        "20260814010000_bounded_public_query_api.sql"
    )
    assert names.index("20260814010000_bounded_public_query_api.sql") < names.index(
        "20260814020000_private_storage_phase.sql"
    )
    assert names.index("20260814020000_private_storage_phase.sql") < names.index(
        "20260814030000_strict_public_cutover.sql"
    )


def test_private_storage_phase_keeps_legacy_table_reads_for_compatibility() -> None:
    source = (MIGRATIONS / "20260814020000_private_storage_phase.sql").read_text(
        encoding="utf-8"
    )
    assert "set public = false" in source
    assert "revoke all on public.scryglass_public_releases" not in source
    assert "get_scryglass_active_inline_asset" not in source


def test_strict_cutover_revokes_legacy_reads_and_inline_rpc() -> None:
    source = (MIGRATIONS / "20260814030000_strict_public_cutover.sql").read_text(
        encoding="utf-8"
    )
    assert "revoke all on public.scryglass_public_releases" in source
    assert "revoke all on public.scryglass_public_assets" in source
    assert "drop function if exists public.get_scryglass_active_inline_asset(text, text)" in source
