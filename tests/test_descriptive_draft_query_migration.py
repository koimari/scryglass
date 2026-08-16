from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import tempfile

import pytest


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


def test_descriptive_query_migration_cuts_over_active_release_and_storage() -> None:
    sql = _sql()

    assert "create or replace function public.get_scryglass_active_release(" in sql
    assert "create or replace function public.get_scryglass_active_asset(" in sql
    assert (
        "create or replace function "
        "scryglass_private.is_active_scryglass_storage_object(" in sql
    )
    assert "'draft_authority',case when authority.value is not null then" in sql
    assert "'status','descriptive'" in sql
    assert "'authority','descriptive'" in sql
    assert "'estimand','composition_only'" in sql
    assert sql.count(
        "asset.path<>'features/draft_records.json' or authority.value is not null"
    ) == 4
    assert sql.count(
        "or public.scryglass_query_descriptive_authority(release.release_id) is not null"
    ) == 2
    assert "grant execute on function public.get_scryglass_active_release(text)" in sql
    assert "grant execute on function public.get_scryglass_active_asset(text,text)" in sql


def _postgres_tools() -> tuple[str, str, str]:
    paths = tuple(shutil.which(name) for name in ("initdb", "pg_ctl", "psql"))
    if any(path is None for path in paths):
        pytest.skip("local PostgreSQL runtime is unavailable")
    return paths  # type: ignore[return-value]


def test_descriptive_query_active_release_runtime(tmp_path: Path) -> None:
    initdb, pg_ctl, psql = _postgres_tools()
    data_dir = tmp_path / "pgdata"
    socket_dir = Path(tempfile.mkdtemp(prefix="scryglass-pg-", dir="/tmp"))
    log_path = tmp_path / "postgres.log"
    subprocess.run(
        [initdb, "-D", str(data_dir), "--no-locale", "--encoding=UTF8", "--auth=trust"],
        check=True,
        capture_output=True,
        text=True,
    )
    server_options = (
        f"-c listen_addresses='' -c unix_socket_directories='{socket_dir}' -p 55439"
    )
    subprocess.run(
        [pg_ctl, "-D", str(data_dir), "-o", server_options, "-l", str(log_path), "-w", "start"],
        check=True,
        capture_output=True,
        text=True,
    )
    command = [
        psql,
        "-v",
        "ON_ERROR_STOP=1",
        "-Atq",
        "-h",
        str(socket_dir),
        "-p",
        "55439",
        "-d",
        "postgres",
    ]
    setup = """
      create role anon;
      create role authenticated;
      create role service_role;
      create schema scryglass_private;
      create table public.scryglass_public_releases (
        release_id text primary key,
        status text not null,
        manifest jsonb not null,
        created_at timestamptz not null,
        source_as_of timestamptz
      );
      create table public.scryglass_public_assets (
        release_id text not null,
        path text not null,
        storage_path text,
        body jsonb,
        bytes bigint not null,
        sha256 text not null,
        content_type text not null,
        primary key (release_id,path)
      );
      create table public.scryglass_public_query_receipts (
        release_id text not null,
        dataset text not null,
        row_count integer not null
      );
    """
    try:
        subprocess.run(command, input=setup, check=True, capture_output=True, text=True)
        subprocess.run(
            command,
            input=MIGRATION.read_text(encoding="utf-8"),
            check=True,
            capture_output=True,
            text=True,
        )
        runtime = """
          insert into public.scryglass_public_releases
            (release_id,status,manifest,created_at,source_as_of)
          values
          (
            'v2026.08.15.060001','active',
            jsonb_build_object(
              'filters',jsonb_build_object('years',jsonb_build_array(2026)),
              'draft_authority',jsonb_build_object(
                'schema_version','scryglass:draft-authority:v1',
                'status','descriptive','authority','descriptive',
                'estimand','composition_only','release_id','v2026.08.15.060001',
                'model_version','composition-static-v1',
                'artifact_sha256',repeat('a',64),
                'receipt_sha256',repeat('c',64),
                'issued_utc','2026-08-15T06:00:01Z',
                'probability_authority',false,
                'recommendation_authority',false,
                'betting_authority',false
              ),
              'files',jsonb_build_array(
                jsonb_build_object('path','features/ratings_snapshot.json','bytes',11,'sha256',repeat('b',64)),
                jsonb_build_object('path','features/draft_records.json','bytes',13,'sha256',repeat('d',64))
              ),
              'release',jsonb_build_object(
                'artifact_hashes',jsonb_build_object(
                  'features/ratings_snapshot.json',repeat('b',64),
                  'features/draft_records.json',repeat('d',64)
                )
              )
            ),
            '2026-08-15T06:00:01Z','2026-08-15T05:00:00Z'
          ),
          (
            'v2026.08.15.060002','active',
            jsonb_build_object(
              'filters',jsonb_build_object('years',jsonb_build_array(2026)),
              'draft_authority',jsonb_build_object(
                'schema_version','scryglass:draft-authority:v1',
                'status','unavailable','authority','unavailable',
                'release_id','v2026.08.15.060002'
              ),
              'files',jsonb_build_array(
                jsonb_build_object('path','features/ratings_snapshot.json','bytes',11,'sha256',repeat('e',64)),
                jsonb_build_object('path','features/draft_records.json','bytes',13,'sha256',repeat('f',64))
              ),
              'release',jsonb_build_object(
                'artifact_hashes',jsonb_build_object(
                  'features/ratings_snapshot.json',repeat('e',64),
                  'features/draft_records.json',repeat('f',64)
                )
              )
            ),
            '2026-08-15T06:00:02Z','2026-08-15T05:00:00Z'
          );

          insert into public.scryglass_public_releases
            (release_id,status,manifest,created_at,source_as_of)
          select
            'v2026.08.15.060003','active',
            jsonb_set(
              jsonb_set(
                manifest,
                '{draft_authority,release_id}',
                to_jsonb('v2026.08.15.060003'::text)
              ),
              '{draft_authority,issued_utc}',
              to_jsonb('2026-02-30T06:00:03Z'::text)
            ),
            '2026-08-15T06:00:03Z','2026-08-15T05:00:00Z'
          from public.scryglass_public_releases
          where release_id='v2026.08.15.060001';

          insert into public.scryglass_public_releases
            (release_id,status,manifest,created_at,source_as_of)
          select
            'v2026.08.15.060004','active',
            jsonb_set(
              manifest,
              '{draft_authority,release_id}',
              to_jsonb('v2026.08.15.060004'::text)
            ) #- '{draft_authority,issued_utc}',
            '2026-08-15T06:00:04Z','2026-08-15T05:00:00Z'
          from public.scryglass_public_releases
          where release_id='v2026.08.15.060001';

          insert into public.scryglass_public_assets
            (release_id,path,storage_path,body,bytes,sha256,content_type)
          values
            ('v2026.08.15.060001','features/ratings_snapshot.json',
             'v2026.08.15.060001/features/ratings_snapshot.json',null,11,repeat('b',64),'application/json'),
            ('v2026.08.15.060001','features/draft_records.json',
             'v2026.08.15.060001/features/draft_records.json',null,13,repeat('d',64),'application/json'),
            ('v2026.08.15.060002','features/ratings_snapshot.json',
             'v2026.08.15.060002/features/ratings_snapshot.json',null,11,repeat('e',64),'application/json'),
            ('v2026.08.15.060002','features/draft_records.json',
             'v2026.08.15.060002/features/draft_records.json',null,13,repeat('f',64),'application/json'),
            ('v2026.08.15.060003','features/draft_records.json',
             'v2026.08.15.060003/features/draft_records.json',null,13,repeat('d',64),'application/json'),
            ('v2026.08.15.060004','features/draft_records.json',
             'v2026.08.15.060004/features/draft_records.json',null,13,repeat('d',64),'application/json');

          do $$
          declare
            descriptive jsonb;
            unavailable jsonb;
            invalid_calendar jsonb;
            missing_issued jsonb;
          begin
            select manifest into descriptive
            from public.get_scryglass_active_release('v2026.08.15.060001');
            if descriptive#>>'{draft_authority,status}' <> 'descriptive'
               or descriptive#>>'{draft_authority,authority}' <> 'descriptive'
               or descriptive#>>'{draft_authority,estimand}' <> 'composition_only'
               or descriptive#>>'{draft_authority,issued_utc}' <> '2026-08-15T06:00:01Z'
               or (descriptive->>'total_files')::integer <> 2
               or (descriptive->>'total_bytes')::integer <> 24
               or descriptive#>>'{release,artifact_hashes,features/draft_records.json}' <> repeat('d',64)
               or not exists (
                 select 1 from jsonb_array_elements(descriptive->'files') file
                 where file->>'path'='features/draft_records.json'
               )
            then
              raise exception 'descriptive active manifest did not expose its bound draft asset';
            end if;

            select manifest into unavailable
            from public.get_scryglass_active_release('v2026.08.15.060002');
            if unavailable ? 'draft_authority'
               or (unavailable->>'total_files')::integer <> 1
               or (unavailable->>'total_bytes')::integer <> 11
               or unavailable#>'{release,artifact_hashes}' ? 'features/draft_records.json'
               or exists (
                 select 1 from jsonb_array_elements(unavailable->'files') file
                 where file->>'path'='features/draft_records.json'
               )
            then
              raise exception 'unavailable active manifest exposed draft authority or data';
            end if;

            select manifest into invalid_calendar
            from public.get_scryglass_active_release('v2026.08.15.060003');
            select manifest into missing_issued
            from public.get_scryglass_active_release('v2026.08.15.060004');
            if public.scryglass_query_descriptive_authority('v2026.08.15.060003') is not null
               or public.scryglass_query_descriptive_authority('v2026.08.15.060004') is not null
               or invalid_calendar ? 'draft_authority'
               or missing_issued ? 'draft_authority'
               or exists (
                 select 1 from jsonb_array_elements(invalid_calendar->'files') file
                 where file->>'path'='features/draft_records.json'
               )
               or exists (
                 select 1 from jsonb_array_elements(missing_issued->'files') file
                 where file->>'path'='features/draft_records.json'
               )
            then
              raise exception 'invalid or missing issued UTC exposed descriptive authority';
            end if;

            if (select count(*) from public.get_scryglass_active_asset(
                 'v2026.08.15.060001','features/draft_records.json')) <> 1
               or (select count(*) from public.get_scryglass_active_asset(
                 'v2026.08.15.060002','features/draft_records.json')) <> 0
               or not scryglass_private.is_active_scryglass_storage_object(
                 'v2026.08.15.060001/features/draft_records.json')
               or scryglass_private.is_active_scryglass_storage_object(
                 'v2026.08.15.060002/features/draft_records.json')
               or (select count(*) from public.get_scryglass_active_asset(
                 'v2026.08.15.060003','features/draft_records.json')) <> 0
               or (select count(*) from public.get_scryglass_active_asset(
                 'v2026.08.15.060004','features/draft_records.json')) <> 0
               or scryglass_private.is_active_scryglass_storage_object(
                 'v2026.08.15.060003/features/draft_records.json')
               or scryglass_private.is_active_scryglass_storage_object(
                 'v2026.08.15.060004/features/draft_records.json')
            then
              raise exception 'active draft asset gate did not follow descriptive authority';
            end if;
          end;
          $$;
        """
        subprocess.run(command, input=runtime, check=True, capture_output=True, text=True)
    finally:
        subprocess.run(
            [pg_ctl, "-D", str(data_dir), "-w", "stop", "-m", "fast"],
            check=False,
            capture_output=True,
            text=True,
        )
        shutil.rmtree(socket_dir,ignore_errors=True)
