from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS = ROOT / "supabase/migrations"


def test_migrations_are_split_into_ordered_cutover_phases() -> None:
    names = sorted(path.name for path in MIGRATIONS.glob("20260814*.sql"))
    assert names == [
        "20260814000001_add_quarter_match_records.sql",
        "20260814010000_bounded_public_query_api.sql",
        "20260814010001_query_seal_statement_budget.sql",
        "20260814010002_query_activate_statement_budget.sql",
        "20260814010003_release_retention_cascade_guard.sql",
        "20260814122801_discard_staging_release.sql",
        "20260814135746_supabase_advisor_cleanup.sql",
        "20260814135848_restore_fk_indexes.sql",
        "20260814160000_private_storage_phase.sql",
        "20260814161000_oe_import_patch_receipts.sql",
        "20260814170000_strict_public_cutover.sql",
        "20260814190000_query_stage_batch_budget.sql",
        "20260814200000_ratings_page_budget.sql",
    ]


def test_query_staging_batch_budget_matches_worker() -> None:
    migration = (
        MIGRATIONS / "20260814190000_query_stage_batch_budget.sql"
    ).read_text(encoding="utf-8")
    publisher = (
        ROOT / "lol_kills/export/supabase_publication.py"
    ).read_text(encoding="utf-8")

    assert "jsonb_array_length(p_rows) > 500" in migration
    assert "octet_length(p_rows::text) > 3500000" in migration
    assert "QUERY_STAGE_BATCH_ROWS = 500" in publisher
    assert "QUERY_STAGE_BATCH_BYTES = 3_200_000" in publisher
    assert "discard_stale_staging_releases(limit=1)" in publisher


def test_ratings_page_budget_updates_the_private_rpc() -> None:
    migration = (
        MIGRATIONS / "20260814200000_ratings_page_budget.sql"
    ).read_text(encoding="utf-8")
    assert "scryglass_private" in migration
    assert "p_limit, 20), 1), 100" in migration
    assert "unknown limit guard" in migration


def test_phase_workdir_excludes_later_migrations() -> None:
    script = (ROOT / "tools/prepare_supabase_phase.py").read_text(encoding="utf-8")
    runbook = (ROOT / "docs/public-refresh-runbook.md").read_text(encoding="utf-8")
    assert '"additive": 20260814160000' in script
    assert '"storage": 20260814170000' in script
    assert "prepare_supabase_phase.py --phase additive" in runbook
    assert "prepare_supabase_phase.py --phase storage" in runbook
    assert "prepare_supabase_phase.py --phase strict" in runbook
    bounded = runbook.split("## Bounded query and private Storage cutover", 1)[1]
    assert "npx supabase db push --linked\n" not in bounded


def test_runbook_never_pushes_all_cutover_phases_from_the_repository() -> None:
    runbook = (ROOT / "docs/public-refresh-runbook.md").read_text(encoding="utf-8")
    assert "npx supabase db push --linked\n" not in runbook
    assert runbook.index("--phase additive") < runbook.index("--phase storage")
    assert runbook.index("--phase storage") < runbook.index("--phase strict")


def test_phase_two_keeps_compatibility_reads_and_phase_three_removes_them() -> None:
    phase_two = (
        MIGRATIONS / "20260814160000_private_storage_phase.sql"
    ).read_text(encoding="utf-8")
    phase_three = (
        MIGRATIONS / "20260814170000_strict_public_cutover.sql"
    ).read_text(encoding="utf-8")

    assert "set public = false" in phase_two
    assert 'revoke select on public.scryglass_public_releases from anon, authenticated' in phase_three
    assert 'revoke all on function public.get_scryglass_active_inline_asset(text, text)' in phase_three
    assert "status in ('staging', 'active', 'superseded')" in phase_three
    assert "release_manifest #>> '{query_api,schema_version}'" in phase_three


def test_phase_one_accepts_legacy_match_rows_during_compatibility() -> None:
    source = (
        MIGRATIONS / "20260813010000_public_release_security_hardening.sql"
    ).read_text(encoding="utf-8")
    assert "features/match_records_2025.json" in source
    assert "features/match_records_2026.json" in source


def test_staging_cleanup_is_locked_and_service_only() -> None:
    migration = (
        MIGRATIONS / "20260814122801_discard_staging_release.sql"
    ).read_text(encoding="utf-8")

    assert "pg_advisory_xact_lock" in migration
    assert "discard_scryglass_staging_release(text)" in migration
    assert "drain_scryglass_staging_cleanup(text)" in migration
    assert "discard_stale_scryglass_staging_releases(integer, integer)" in migration
    assert "Only a staging Scryglass release can be discarded" in migration
    assert "grant execute on function public.discard_scryglass_staging_release(text)\n  to service_role" in migration
    assert "from public, anon, authenticated, service_role" in migration
    assert "status = 'active'" not in migration.split(
        "create or replace function public.discard_scryglass_staging_release",
        1,
    )[1]


def test_query_retention_cascade_allows_only_internal_table_owner() -> None:
    migration = (
        MIGRATIONS / "20260815010000_query_retention_cascade_guard.sql"
    ).read_text(encoding="utf-8")

    assert "pg_advisory_xact_lock" in migration
    assert "'scryglass_release_retention_owner'" in migration
    assert "'scryglass_release_transition_owner'" in migration
    assert "'postgres'" in migration
    assert "release.status = 'superseded'" in migration
    assert "from public, anon, authenticated, service_role" in migration


def test_retention_prunes_one_query_release_per_transaction() -> None:
    migration = (
        MIGRATIONS / "20260815020000_bounded_retention_prune.sql"
    ).read_text(encoding="utf-8")

    fast_path = "if tg_op = 'DELETE' and current_user = 'postgres' then"
    assert fast_path in migration
    assert migration.index(fast_path) < migration.index("pg_advisory_xact_lock")
    assert "limit 1" in migration
    assert "'has_more', has_more" in migration
    assert "owner to scryglass_release_retention_owner" in migration
    assert "grant execute on function public.prune_scryglass_public_releases_v2(integer)" in migration


def test_trusted_retention_cascade_skips_the_per_row_lock() -> None:
    migration = (
        MIGRATIONS / "20260815030000_fast_retention_cascade.sql"
    ).read_text(encoding="utf-8")

    fast_path = "and current_user in ("
    assert fast_path in migration
    assert migration.index(fast_path) < migration.index("pg_advisory_xact_lock")
    assert "'scryglass_release_retention_owner'" in migration
    assert "'scryglass_release_transition_owner'" in migration
    assert "'postgres'" in migration
    assert "from public, anon, authenticated, service_role" in migration


def test_live_oe_import_schema_drift_is_forward_compatible() -> None:
    migration = (
        MIGRATIONS / "20260814161000_oe_import_patch_receipts.sql"
    ).read_text(encoding="utf-8")
    assert "add column if not exists riot_patch_receipts integer not null default 0" in migration
    assert "scryglass_oe_imports_riot_patch_receipts_check" in migration
    assert "oe-normalization:v3" in migration


def test_supabase_advisor_cleanup_keeps_public_wrappers_and_private_tables() -> None:
    migration = (
        MIGRATIONS / "20260814135746_supabase_advisor_cleanup.sql"
    ).read_text(encoding="utf-8").lower()

    assert "create schema if not exists scryglass_private" in migration
    assert "security invoker" in migration
    assert "security definer" in migration
    assert "set search_path to public, pg_temp" in migration
    assert 'create policy "deny public api access"' in migration
    assert "using (false) with check (false)" in migration
    for index in (
        "scryglass_oe_game_versions_date_idx",
        "scryglass_oe_games_version_fk_idx",
        "scryglass_oe_games_year_date_idx",
        "scryglass_oe_games_league_date_idx",
        "scryglass_refresh_runs_started_idx",
        "scryglass_refresh_runs_retry_idx",
        "scryglass_refresh_runs_failures_idx",
        "scryglass_public_health_release_idx",
        "scryglass_public_health_run_idx",
    ):
        assert f"drop index if exists public.{index}" in migration
