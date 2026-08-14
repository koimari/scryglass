from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS = ROOT / "supabase/migrations"


def test_phase_one_contains_only_additive_cutover_migrations() -> None:
    names = sorted(path.name for path in MIGRATIONS.glob("20260814*.sql"))
    assert names == [
        "20260814000001_add_quarter_match_records.sql",
        "20260814010000_bounded_public_query_api.sql",
    ]


def test_later_storage_phases_are_deferred_to_follow_up_commits() -> None:
    assert not (MIGRATIONS / "20260814020000_private_storage_phase.sql").exists()
    assert not (MIGRATIONS / "20260814030000_strict_public_cutover.sql").exists()


def test_phase_one_accepts_legacy_match_rows_during_compatibility() -> None:
    source = (
        MIGRATIONS / "20260813010000_public_release_security_hardening.sql"
    ).read_text(encoding="utf-8")
    assert "features/match_records_2025.json" in source
    assert "features/match_records_2026.json" in source
