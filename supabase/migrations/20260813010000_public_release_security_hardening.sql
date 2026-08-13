-- Final public release boundary: private immutable Storage, active-only reads,
-- exact asset metadata, and service-role-only release transitions.

alter table public.scryglass_public_assets
  add column if not exists content_type text;

update public.scryglass_public_assets
set content_type = 'application/json'
where content_type is null;

alter table public.scryglass_public_assets
  alter column content_type set default 'application/json',
  alter column content_type set not null;

alter table public.scryglass_public_assets
  drop constraint if exists scryglass_public_assets_content_type_check;
alter table public.scryglass_public_assets
  add constraint scryglass_public_assets_content_type_check
  check (content_type = 'application/json');

alter table public.scryglass_public_assets
  drop constraint if exists scryglass_public_assets_path_check;
alter table public.scryglass_public_assets
  add constraint scryglass_public_assets_path_check
  check (
    path in (
      'features/ratings_snapshot.json',
      'features/player_ratings_snapshot.json',
      'features/team_records.json',
      'features/team_weekly_ranks.json',
      'features/player_records.json',
      'features/player_champion_records.json',
      'features/profile_records.json',
      'features/match_index.json',
      'features/match_records_2025.json',
      'features/match_records_2026.json',
      'features/player_weekly_ranks.json',
      'features/player_metadata.json',
      'features/schedule.json',
      'features/leaderboards.json',
      'features/draft_records.json',
      'rankings/tierlists.json',
      'rankings/tierlists-latest.json'
    )
  );

alter table public.scryglass_public_assets
  drop constraint if exists scryglass_public_assets_storage_binding_check;
alter table public.scryglass_public_assets
  add constraint scryglass_public_assets_storage_binding_check
  check (
    storage_path is null
    or storage_path = release_id || '/' || path
  );

update storage.buckets
set public = false,
    file_size_limit = 125829120,
    allowed_mime_types = array['application/json']
where id = 'scryglass-public';

drop policy if exists "read active Scryglass Storage assets" on storage.objects;
create policy "read active Scryglass Storage assets"
  on storage.objects
  for select
  to anon, authenticated
  using (
    bucket_id = 'scryglass-public'
    and exists (
      select 1
      from public.scryglass_public_assets asset
      join public.scryglass_public_releases release
        on release.release_id = asset.release_id
      where release.status = 'active'
        and asset.storage_path = storage.objects.name
        and asset.release_id = pg_catalog.split_part(storage.objects.name, '/', 1)
        and asset.path = pg_catalog.substr(
          storage.objects.name,
          pg_catalog.strpos(storage.objects.name, '/') + 1
        )
    )
  );

drop policy if exists "service role manages Scryglass Storage assets" on storage.objects;
create policy "service role manages Scryglass Storage assets"
  on storage.objects
  for all
  to service_role
  using (bucket_id = 'scryglass-public')
  with check (bucket_id = 'scryglass-public');

create table if not exists public.scryglass_storage_cleanup (
  storage_path text primary key,
  release_id text not null,
  queued_at timestamptz not null default now(),
  check (
    release_id ~ '^v[0-9]{4}\.[0-9]{2}\.[0-9]{2}\.([0-9]{4}|[0-9]{6})$'
    and storage_path like release_id || '/%'
  )
);
alter table public.scryglass_storage_cleanup enable row level security;

create or replace function public.activate_scryglass_public_release(p_release_id text)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $$
declare
  previous_release_id text;
  release_manifest jsonb;
  required_assets constant text[] := array[
    'features/ratings_snapshot.json',
    'features/player_ratings_snapshot.json',
    'features/team_records.json',
    'features/team_weekly_ranks.json',
    'features/player_records.json',
    'features/player_champion_records.json',
    'features/profile_records.json',
    'features/match_index.json',
    'features/match_records_2025.json',
    'features/match_records_2026.json',
    'features/player_weekly_ranks.json',
    'features/player_metadata.json',
    'rankings/tierlists.json',
    'rankings/tierlists-latest.json'
  ];
  present_assets integer;
  manifest_files integer;
  manifest_paths integer;
  release_assets integer;
  invalid_assets integer;
  draft_asset_count integer;
  draft_authority_status text;
begin
  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtext('scryglass-public-release')
  );

  select manifest
  into release_manifest
  from public.scryglass_public_releases
  where release_id = p_release_id
    and status in ('staging', 'active')
  for update;

  if release_manifest is null then
    raise exception 'Scryglass release is not ready for activation';
  end if;
  if p_release_id !~ '^v[0-9]{4}\.[0-9]{2}\.[0-9]{2}\.[0-9]{6}$'
     or release_manifest ->> 'pack_id' <> p_release_id
     or release_manifest #>> '{release,release_id}' <> p_release_id
     or pg_catalog.jsonb_typeof(release_manifest -> 'files') <> 'array'
     or pg_catalog.jsonb_typeof(release_manifest #> '{release,artifact_hashes}') <> 'object'
  then
    raise exception 'Scryglass release manifest binding is invalid';
  end if;

  draft_authority_status := release_manifest #>> '{draft_authority,status}';
  if release_manifest #>> '{draft_authority,schema_version}'
       is distinct from 'scryglass:draft-authority:v1'
     or release_manifest #>> '{draft_authority,release_id}' is distinct from p_release_id
     or pg_catalog.coalesce(draft_authority_status, '')
       not in ('unavailable', 'promoted')
  then
    raise exception 'Scryglass draft authority binding is invalid';
  end if;
  select count(*)
  into draft_asset_count
  from public.scryglass_public_assets
  where release_id = p_release_id
    and path = 'features/draft_records.json';
  if (
    draft_authority_status = 'unavailable'
    and draft_asset_count <> 0
  ) or (
    draft_authority_status = 'promoted'
    and (
      draft_asset_count <> 1
      or pg_catalog.coalesce(
        release_manifest #>> '{draft_authority,model_version}', ''
      ) = ''
      or pg_catalog.coalesce(
        release_manifest #>> '{draft_authority,receipt_sha256}', ''
      )
        !~ '^[0-9a-f]{64}$'
    )
  ) then
    raise exception 'Scryglass draft asset does not match its authority';
  end if;

  select count(*)
  into present_assets
  from public.scryglass_public_assets
  where release_id = p_release_id
    and path = any(required_assets);
  if present_assets <> pg_catalog.cardinality(required_assets) then
    raise exception 'Scryglass release has % of % required assets',
      present_assets, pg_catalog.cardinality(required_assets);
  end if;

  select pg_catalog.jsonb_array_length(release_manifest -> 'files'),
         count(distinct file ->> 'path')
  into manifest_files, manifest_paths
  from pg_catalog.jsonb_array_elements(release_manifest -> 'files') file;
  select count(*) into release_assets
  from public.scryglass_public_assets
  where release_id = p_release_id;
  if manifest_files <> manifest_paths or manifest_files <> release_assets then
    raise exception 'Scryglass release manifest inventory is not exact';
  end if;

  select count(*)
  into invalid_assets
  from public.scryglass_public_assets asset
  where asset.release_id = p_release_id
    and (
      asset.body is not null
      or asset.storage_path is distinct from asset.release_id || '/' || asset.path
      or asset.content_type is distinct from 'application/json'
      or release_manifest #>> array['release', 'artifact_hashes', asset.path]
        is distinct from asset.sha256
      or not exists (
        select 1
        from storage.objects object
        where object.bucket_id = 'scryglass-public'
          and object.name = asset.storage_path
          and object.user_metadata ->> 'sha256' = asset.sha256
          and (object.user_metadata ->> 'bytes')::bigint = asset.bytes
          and object.user_metadata ->> 'content_type' = asset.content_type
      )
      or not exists (
        select 1
        from pg_catalog.jsonb_array_elements(release_manifest -> 'files') file
        where file ->> 'path' = asset.path
          and (file ->> 'bytes')::bigint = asset.bytes
          and file ->> 'sha256' = asset.sha256
      )
    );
  if invalid_assets <> 0 then
    raise exception 'Scryglass release has invalid asset metadata';
  end if;

  select release_id
  into previous_release_id
  from public.scryglass_public_releases
  where status = 'active'
    and release_id <> p_release_id
  limit 1;

  update public.scryglass_public_releases
  set status = 'superseded'
  where status = 'active'
    and release_id <> p_release_id;

  update public.scryglass_public_releases
  set status = 'active', activated_at = now()
  where release_id = p_release_id;

  return pg_catalog.jsonb_build_object(
    'status', 'active',
    'release_id', p_release_id,
    'previous_release_id', previous_release_id
  );
end;
$$;

create or replace function public.restore_scryglass_public_release(p_release_id text)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $$
declare
  replaced_release_id text;
begin
  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtext('scryglass-public-release')
  );
  if not exists (
    select 1
    from public.scryglass_public_releases
    where release_id = p_release_id
      and status in ('active', 'superseded')
  ) then
    raise exception 'Scryglass rollback release is unavailable';
  end if;
  select release_id
  into replaced_release_id
  from public.scryglass_public_releases
  where status = 'active'
    and release_id <> p_release_id
  limit 1;
  update public.scryglass_public_releases
  set status = 'superseded'
  where status = 'active'
    and release_id <> p_release_id;
  update public.scryglass_public_releases
  set status = 'active', activated_at = now()
  where release_id = p_release_id;
  return pg_catalog.jsonb_build_object(
    'status', 'restored',
    'release_id', p_release_id,
    'replaced_release_id', replaced_release_id
  );
end;
$$;

create or replace function public.prune_scryglass_public_releases_v2(
  p_keep integer default 3
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $$
declare
  removed_releases text[];
  queued_storage_paths text[];
begin
  if p_keep < 1 or p_keep > 10 then
    raise exception 'Scryglass retention must be between 1 and 10 releases';
  end if;
  select pg_catalog.coalesce(pg_catalog.array_agg(release_id), array[]::text[])
  into removed_releases
  from public.scryglass_public_releases
  where status = 'superseded'
    and release_id not in (
      select release_id
      from public.scryglass_public_releases
      where status = 'superseded'
      order by activated_at desc nulls last, created_at desc
      limit pg_catalog.greatest(p_keep - 1, 0)
    );
  insert into public.scryglass_storage_cleanup (storage_path, release_id)
  select storage_path, release_id
  from public.scryglass_public_assets
  where release_id = any(removed_releases)
    and storage_path is not null
  on conflict (storage_path) do nothing;
  delete from public.scryglass_public_releases
  where release_id = any(removed_releases);
  select pg_catalog.coalesce(pg_catalog.array_agg(storage_path), array[]::text[])
  into queued_storage_paths
  from public.scryglass_storage_cleanup;
  return pg_catalog.jsonb_build_object(
    'deleted_count', pg_catalog.cardinality(removed_releases),
    'release_ids', pg_catalog.to_jsonb(removed_releases),
    'storage_paths', pg_catalog.to_jsonb(queued_storage_paths)
  );
end;
$$;

create or replace function public.ack_scryglass_storage_cleanup(
  p_storage_paths text[]
)
returns integer
language plpgsql
security invoker
set search_path = ''
as $$
declare
  deleted_count integer;
begin
  if p_storage_paths is null
     or pg_catalog.cardinality(p_storage_paths) > 500
     or exists (
       select 1
       from pg_catalog.unnest(p_storage_paths) as item(storage_path)
       where item.storage_path is null
          or item.storage_path !~ '^v[0-9]{4}\.[0-9]{2}\.[0-9]{2}\.([0-9]{4}|[0-9]{6})/'
     )
  then
    raise exception 'Scryglass Storage cleanup acknowledgement is invalid';
  end if;
  delete from public.scryglass_storage_cleanup
  where storage_path = any(p_storage_paths);
  get diagnostics deleted_count = row_count;
  return deleted_count;
end;
$$;

revoke all on public.scryglass_public_releases
  from public, anon, authenticated, service_role;
revoke all on public.scryglass_public_assets
  from public, anon, authenticated, service_role;
revoke all on public.scryglass_storage_cleanup
  from public, anon, authenticated, service_role;
grant select on public.scryglass_public_releases to anon, authenticated;
grant select on public.scryglass_public_assets to anon, authenticated;
grant select, insert, update, delete on public.scryglass_public_releases to service_role;
grant select, insert, update on public.scryglass_public_assets to service_role;
grant select, insert, delete on public.scryglass_storage_cleanup to service_role;

revoke all on function public.activate_scryglass_public_release(text)
  from public, anon, authenticated, service_role;
revoke all on function public.restore_scryglass_public_release(text)
  from public, anon, authenticated, service_role;
revoke all on function public.prune_scryglass_public_releases(integer)
  from public, anon, authenticated, service_role;
revoke all on function public.prune_scryglass_public_releases_v2(integer)
  from public, anon, authenticated, service_role;
revoke all on function public.ack_scryglass_storage_cleanup(text[])
  from public, anon, authenticated, service_role;
grant execute on function public.activate_scryglass_public_release(text) to service_role;
grant execute on function public.restore_scryglass_public_release(text) to service_role;
grant execute on function public.prune_scryglass_public_releases_v2(integer) to service_role;
grant execute on function public.ack_scryglass_storage_cleanup(text[]) to service_role;
