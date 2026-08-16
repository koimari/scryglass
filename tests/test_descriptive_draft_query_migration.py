from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = (
    ROOT
    / "supabase"
    / "migrations"
    / "20260815060001_descriptive_draft_query_api.sql"
)


def _sql() -> str:
    return MIGRATION.read_text(encoding="utf-8")


def test_descriptive_query_migration_is_active_release_bound_and_bounded() -> None:
    sql = _sql()

    assert "release.status = 'active'" in sql
    assert "'{draft_authority,artifact_sha256}'" in sql
    assert "'{draft_authority,receipt_sha256}'" in sql
    assert "set statement_timeout = '5s'" in sql
    assert "pg_catalog.octet_length(result::text) > 500000" in sql
    assert "probability_authority" in sql
    assert "recommendation_authority" in sql
    assert "betting_authority" in sql


def test_descriptive_query_migration_exposes_exact_player_profile_metric() -> None:
    sql = _sql()
    start = sql.index("if profile_metric ? 'pick_contribution' then")
    end = sql.index("elsif profile_metric ? 'draft_edge' then", start)
    player_projection = sql[start:end]

    for field in (
        "best_available_rate",
        "games",
        "pick_contribution",
        "pool_definition",
        "ban_coverage",
        "scope",
    ):
        assert f"'{field}'" in player_projection
    for forbidden in (
        "draft_score",
        "probability",
        "r9e",
        "elo",
        "momentum",
        "gold",
        "objectives",
    ):
        assert f"'{forbidden}'" not in player_projection


def test_descriptive_query_migration_keeps_internal_helpers_private() -> None:
    sql = _sql()

    for signature in (
        "public.scryglass_descriptive_signal_is_valid(jsonb)",
        "public.scryglass_descriptive_summary_is_valid(jsonb)",
        "public.scryglass_query_descriptive_authority(text)",
        "public.scryglass_strip_query_draft_fields(jsonb)",
        "public.scryglass_json_has_unbound_descriptive_draft(jsonb,jsonb)",
    ):
        assert f"revoke all on function {signature}" in sql
    assert "grant execute on function public.scryglass_query_descriptive_authority" not in sql
