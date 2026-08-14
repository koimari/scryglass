-- Supabase advisor cleanup for the phase-one public query API.
--
-- Public RPCs remain the intentional Data API surface.  Their implementations
-- move to a schema that is not exposed by PostgREST.  Public SQL wrappers use
-- SECURITY INVOKER and call the pinned private implementations.  This keeps
-- the API contract while removing SECURITY DEFINER functions from public.

begin;

create schema if not exists scryglass_private;
comment on schema scryglass_private is
  'Private Scryglass implementation schema. Keep it out of Supabase Data API schemas.';

revoke all on schema scryglass_private from public;
grant usage on schema scryglass_private to anon, authenticated, service_role;

do $$
declare
  target record;
  call_args text;
  select_sql text;
  body text;
  volatility text;
begin
  for target in
    select
      p.proname as name,
      pg_get_function_identity_arguments(p.oid) as identity_arguments,
      pg_get_function_arguments(p.oid) as arguments,
      pg_get_function_result(p.oid) as result_type,
      p.pronargs,
      case p.provolatile
        when 'i' then 'IMMUTABLE'
        when 'v' then 'VOLATILE'
        else 'STABLE'
      end as volatility
    from pg_catalog.pg_proc p
    join pg_catalog.pg_namespace n on n.oid = p.pronamespace
    where n.nspname = 'public'
      and p.proname = any(array[
        'get_scryglass_active_asset',
        'get_scryglass_active_inline_asset',
        'get_scryglass_active_release',
        'get_scryglass_champions',
        'get_scryglass_match',
        'get_scryglass_match_facets',
        'get_scryglass_matches',
        'get_scryglass_player_champions',
        'get_scryglass_player_profile',
        'get_scryglass_private_health',
        'get_scryglass_public_health',
        'get_scryglass_query_entities',
        'get_scryglass_query_status',
        'get_scryglass_rating_facets',
        'get_scryglass_ratings',
        'get_scryglass_team_profile',
        'get_scryglass_tier_facets',
        'get_scryglass_tier_rows',
        'get_scryglass_tier_scope',
        'is_active_scryglass_storage_object'
      ])
    order by p.oid
  loop
    call_args := coalesce(
      (
        select string_agg('$' || argument_number::text, ', ' order by argument_number)
        from generate_series(1, target.pronargs) as argument_number
      ),
      ''
    );

    execute format(
      'alter function public.%I(%s) set schema scryglass_private',
      target.name,
      target.identity_arguments
    );
    execute format(
      'alter function scryglass_private.%I(%s) security %s',
      target.name,
      target.identity_arguments,
      case when target.volatility = 'VOLATILE' then 'definer' else 'definer' end
    );
    execute format(
      'alter function scryglass_private.%I(%s) set search_path to public, pg_temp',
      target.name,
      target.identity_arguments
    );
    execute format(
      'revoke all on function scryglass_private.%I(%s) from public',
      target.name,
      target.identity_arguments
    );
    execute format(
      'grant execute on function scryglass_private.%I(%s) to anon, authenticated, service_role',
      target.name,
      target.identity_arguments
    );

    select_sql := case
      when target.result_type like 'TABLE(%' then 'select * from '
      else 'select '
    end;
    body := select_sql || format(
      'scryglass_private.%I(%s)',
      target.name,
      call_args
    );
    volatility := target.volatility;

    execute format(
      'create or replace function public.%I(%s) returns %s language sql %s security invoker set search_path to public, pg_temp set statement_timeout = ''5s'' as %L',
      target.name,
      target.arguments,
      target.result_type,
      volatility,
      body
    );
    execute format(
      'revoke all on function public.%I(%s) from public',
      target.name,
      target.identity_arguments
    );
    execute format(
      'grant execute on function public.%I(%s) to anon, authenticated, service_role',
      target.name,
      target.identity_arguments
    );
  end loop;
end;
$$;

-- RLS was already enabled on these tables.  The explicit policies remove the
-- advisor warning and make the intended private access visible in the schema.
do $$
declare
  table_name text;
begin
  foreach table_name in array array[
    'scryglass_diagnostic_credentials',
    'scryglass_oe_game_versions',
    'scryglass_oe_games',
    'scryglass_oe_imports',
    'scryglass_public_query_receipts',
    'scryglass_public_query_rows',
    'scryglass_refresh_runs',
    'scryglass_storage_cleanup'
  ] loop
    execute format(
      'drop policy if exists "deny public api access" on public.%I',
      table_name
    );
    execute format(
      'create policy "deny public api access" on public.%I as restrictive for all to anon, authenticated using (false) with check (false)',
      table_name
    );
  end loop;
end;
$$;

-- These indexes have no scans in the live project and duplicate no active
-- query contract.  They add write and vacuum work to the refresh path.
drop index if exists public.scryglass_oe_game_versions_date_idx;
drop index if exists public.scryglass_oe_games_version_fk_idx;
drop index if exists public.scryglass_oe_games_year_date_idx;
drop index if exists public.scryglass_oe_games_league_date_idx;
drop index if exists public.scryglass_refresh_runs_started_idx;
drop index if exists public.scryglass_refresh_runs_retry_idx;
drop index if exists public.scryglass_refresh_runs_failures_idx;
drop index if exists public.scryglass_public_health_release_idx;
drop index if exists public.scryglass_public_health_run_idx;

commit;
