-- Phase 3: require the bounded query release and remove compatibility reads.
-- Apply only after the strict web build and two verified Storage-only releases
-- are live.

begin;

create or replace function public.assert_scryglass_query_release(p_release_id text)
returns void
language plpgsql
security definer
set search_path = ''
set statement_timeout = '120s'
as $$
declare
  release_manifest jsonb;
  required_datasets constant text[] := array[
    'players', 'teams', 'player_champions', 'games', 'identity_games',
    'champions', 'aliases', 'tier_rows', 'tier_scopes',
    'tier_matrix_rows', 'tier_similarity_champions',
    'tier_similarity_edges'
  ];
  required_dataset text;
  receipt public.scryglass_public_query_receipts%rowtype;
  actual_rows integer;
  actual_bytes bigint;
  actual_storage_digest text;
begin
  select release.manifest
  into release_manifest
  from public.scryglass_public_releases release
  where release.release_id = p_release_id
    and release.status in ('staging', 'active', 'superseded');

  if release_manifest is null
     or release_manifest #>> '{query_api,schema_version}'
        is distinct from 'scryglass:query-api:v1'
     or release_manifest #>> '{query_api,status}' is distinct from 'available'
     or pg_catalog.jsonb_typeof(release_manifest #> '{query_api,datasets}')
        is distinct from 'object'
     or public.scryglass_json_has_draft_fields(release_manifest -> 'query_api')
  then
    raise exception 'Scryglass query API authority is invalid';
  end if;

  if (
    select pg_catalog.array_agg(key order by key)
    from pg_catalog.jsonb_object_keys(release_manifest #> '{query_api,datasets}') key
  ) is distinct from (
    select pg_catalog.array_agg(value order by value)
    from pg_catalog.unnest(required_datasets) value
  ) then
    raise exception 'Scryglass query dataset inventory is not exact';
  end if;

  foreach required_dataset in array required_datasets
  loop
    select *
    into receipt
    from public.scryglass_public_query_receipts candidate
    where candidate.release_id = p_release_id
      and candidate.dataset = required_dataset;

    if not found
       or receipt.row_count is distinct from (
         release_manifest #>> array[
           'query_api', 'datasets', required_dataset, 'rows'
         ]
       )::integer
       or receipt.source_bytes is distinct from (
         release_manifest #>> array[
           'query_api', 'datasets', required_dataset, 'bytes'
         ]
       )::bigint
       or receipt.source_sha256 is distinct from release_manifest #>> array[
         'query_api', 'datasets', required_dataset, 'sha256'
       ]
       or receipt.row_digest_sha256 is distinct from release_manifest #>> array[
         'query_api', 'datasets', required_dataset, 'row_digest_sha256'
       ]
    then
      raise exception 'Scryglass query dataset receipt is invalid: %',
        required_dataset;
    end if;

    select
      pg_catalog.count(*),
      coalesce(pg_catalog.sum(pg_catalog.octet_length(row.source_json)), 0),
      pg_catalog.encode(
        extensions.digest(
          pg_catalog.convert_to(
            coalesce(
              pg_catalog.string_agg(
                row.row_key || ':' || pg_catalog.encode(
                  extensions.digest(
                    pg_catalog.convert_to(row.source_json, 'UTF8'),
                    'sha256'
                  ),
                  'hex'
                ),
                E'\n' order by row.row_key
              ),
              ''
            ),
            'UTF8'
          ),
          'sha256'
        ),
        'hex'
      )
    into actual_rows, actual_bytes, actual_storage_digest
    from public.scryglass_public_query_rows row
    where row.release_id = p_release_id
      and row.dataset = required_dataset;

    if actual_rows <> receipt.row_count
       or actual_bytes <> receipt.storage_bytes
       or actual_storage_digest <> receipt.storage_sha256
    then
      raise exception 'Scryglass query dataset storage receipt is invalid: %',
        required_dataset;
    end if;
  end loop;
end;
$$;

grant create on schema public to scryglass_release_transition_owner;
alter function public.assert_scryglass_query_release(text)
  owner to scryglass_release_transition_owner;
revoke create on schema public from scryglass_release_transition_owner;
alter function public.assert_scryglass_query_release(text)
  security definer;
alter function public.assert_scryglass_query_release(text)
  set search_path to '';
alter function public.assert_scryglass_query_release(text)
  set statement_timeout = '120s';

grant select on public.scryglass_public_query_rows,
  public.scryglass_public_query_receipts
  to scryglass_release_transition_owner;
grant usage on schema extensions to scryglass_release_transition_owner;
grant execute on function public.scryglass_json_has_draft_fields(jsonb)
  to scryglass_release_transition_owner;

-- Only the transition owner can change release status or activation time.
-- The publisher still writes staging metadata through its existing service
-- role permissions. Direct status changes cannot bypass activation checks.
create or replace function public.guard_scryglass_public_release_mutation()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $$
begin
  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtext('scryglass-public-release')
  );

  if tg_op = 'INSERT' then
    if new.status <> 'staging' then
      raise exception 'Scryglass releases must begin in staging';
    end if;
    return new;
  end if;

  if tg_op = 'DELETE' then
    if current_user not in (
         'scryglass_release_transition_owner',
         'scryglass_release_retention_owner',
         'postgres'
       )
       or old.status <> 'superseded'
    then
      raise exception 'Only the release owners can delete superseded releases';
    end if;
    return old;
  end if;

  if new.release_id is distinct from old.release_id
     or new.created_at is distinct from old.created_at
     or (
       old.status <> 'staging'
       and (
         new.manifest is distinct from old.manifest
         or new.source_as_of is distinct from old.source_as_of
       )
     )
  then
    raise exception 'Scryglass release metadata is immutable';
  end if;

  if new.status is distinct from old.status
     or new.activated_at is distinct from old.activated_at
  then
    if current_user not in (
         'scryglass_release_transition_owner',
         'postgres'
       )
    then
      raise exception 'Scryglass release transitions require the activation RPC';
    end if;
    if old.status = 'staging'
       and new.status not in ('active', 'superseded')
    then
      raise exception 'A staging release can only be activated or discarded';
    end if;
    if old.status = 'active' and new.status <> 'superseded' then
      raise exception 'An active release can only become superseded';
    end if;
    if old.status = 'superseded' and new.status <> 'active' then
      raise exception 'A superseded release can only be restored';
    end if;
  end if;

  return new;
end;
$$;

drop trigger if exists guard_scryglass_public_release_mutation
  on public.scryglass_public_releases;
create trigger guard_scryglass_public_release_mutation
before insert or update or delete on public.scryglass_public_releases
for each row execute function public.guard_scryglass_public_release_mutation();

revoke update on public.scryglass_public_releases
  from service_role;
revoke delete on public.scryglass_public_releases from service_role;

-- The strict web uses active-release RPCs and bounded query functions. The
-- base tables and parsed inline compatibility RPC are private after this point.
revoke select on public.scryglass_public_releases from anon, authenticated;
revoke select on public.scryglass_public_assets from anon, authenticated;
revoke select on public.scryglass_public_health from anon, authenticated;
do $$
begin
  if pg_catalog.to_regprocedure(
    'public.get_scryglass_active_inline_asset(text,text)'
  ) is not null then
    execute 'revoke all on function public.get_scryglass_active_inline_asset(text, text) from public, anon, authenticated, service_role';
    execute 'comment on function public.get_scryglass_active_inline_asset(text, text) is ''Retired at strict public cutover. Storage-only releases are required.''';
  end if;
end;
$$;

commit;
