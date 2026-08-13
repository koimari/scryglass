-- Phase 1 hardening keeps the legacy read path available while compatible
-- active-release RPC clients deploy. The later bounded-query cutover migration
-- makes Storage private and removes direct browser table reads.

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
      'features/match_records_2025_q1.json',
      'features/match_records_2025_q2.json',
      'features/match_records_2025_q3.json',
      'features/match_records_2025_q4.json',
      'features/match_records_2026_q1.json',
      'features/match_records_2026_q2.json',
      'features/match_records_2026_q3.json',
      'features/match_records_2026_q4.json',
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
set public = true,
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

create extension if not exists pgcrypto with schema extensions;

create table if not exists public.scryglass_diagnostic_credentials (
  credential_id text primary key check (credential_id = 'public-health'),
  token_sha256 text not null check (token_sha256 ~ '^[0-9a-f]{64}$'),
  updated_at timestamptz not null default now()
);
alter table public.scryglass_diagnostic_credentials enable row level security;

-- Draft output stays private until an independently verifiable promotion
-- receipt exists. Queue any legacy object for deletion, remove its asset row,
-- and sanitize every stored manifest before the public RPCs become available.
insert into public.scryglass_storage_cleanup (storage_path, release_id)
select storage_path, release_id
from public.scryglass_public_assets
where path = 'features/draft_records.json'
  and storage_path is not null
on conflict (storage_path) do nothing;

delete from public.scryglass_public_assets
where path = 'features/draft_records.json';

with sanitized as (
  select
    release.release_id,
    coalesce(
      (
        select pg_catalog.jsonb_agg(file order by file ->> 'path')
        from pg_catalog.jsonb_array_elements(
          case
            when pg_catalog.jsonb_typeof(release.manifest -> 'files') = 'array'
              then release.manifest -> 'files'
            else '[]'::jsonb
          end
        ) as file
        where file ->> 'path' <> 'features/draft_records.json'
      ),
      '[]'::jsonb
    ) as files
  from public.scryglass_public_releases as release
), rebuilt as (
  select
    sanitized.release_id,
    sanitized.files,
    pg_catalog.jsonb_array_length(sanitized.files) as total_files,
    coalesce(
      (
        select pg_catalog.sum((file ->> 'bytes')::bigint)
        from pg_catalog.jsonb_array_elements(sanitized.files) as file
      ),
      0
    ) as total_bytes
  from sanitized
)
update public.scryglass_public_releases as release
set manifest = pg_catalog.jsonb_set(
  pg_catalog.jsonb_set(
    pg_catalog.jsonb_set(
      pg_catalog.jsonb_set(
        pg_catalog.jsonb_set(
          release.manifest,
          '{files}',
          rebuilt.files,
          true
        ),
        '{total_files}',
        pg_catalog.to_jsonb(rebuilt.total_files),
        true
      ),
      '{total_bytes}',
      pg_catalog.to_jsonb(rebuilt.total_bytes),
      true
    ),
    '{release,artifact_hashes}',
    coalesce(
      release.manifest #> '{release,artifact_hashes}',
      '{}'::jsonb
    ) - 'features/draft_records.json',
    true
  ),
  '{draft_authority}',
  pg_catalog.jsonb_build_object(
    'schema_version', 'scryglass:draft-authority:v1',
    'status', 'unavailable',
    'release_id', release.release_id,
    'model_version', null,
    'receipt_sha256', null,
    'issued_utc', null,
    'reason', 'model_not_promoted'
  ),
  true
)
from rebuilt
where rebuilt.release_id = release.release_id;

create or replace function public.assert_scryglass_public_release_integrity(
  p_release_id text
)
returns void
language plpgsql
security invoker
set search_path = ''
as $$
declare
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
    'features/match_records_2025_q1.json',
    'features/match_records_2025_q2.json',
    'features/match_records_2025_q3.json',
    'features/match_records_2025_q4.json',
    'features/match_records_2026_q1.json',
    'features/match_records_2026_q2.json',
    'features/match_records_2026_q3.json',
    'features/match_records_2026_q4.json',
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
begin
  select manifest
  into release_manifest
  from public.scryglass_public_releases
  where release_id = p_release_id
    and status in ('staging', 'active', 'superseded');

  if release_manifest is null then
    raise exception 'Scryglass release is unavailable';
  end if;
  if p_release_id !~ '^v[0-9]{4}\.[0-9]{2}\.[0-9]{2}\.[0-9]{6}$'
     or release_manifest ->> 'pack_id' <> p_release_id
     or release_manifest #>> '{release,release_id}' <> p_release_id
     or pg_catalog.jsonb_typeof(release_manifest -> 'files') <> 'array'
     or pg_catalog.jsonb_typeof(release_manifest #> '{release,artifact_hashes}') <> 'object'
     or release_manifest #>> '{draft_authority,schema_version}'
       is distinct from 'scryglass:draft-authority:v1'
     or release_manifest #>> '{draft_authority,status}' is distinct from 'unavailable'
     or release_manifest #>> '{draft_authority,release_id}' is distinct from p_release_id
  then
    raise exception 'Scryglass release binding is invalid';
  end if;

  select count(*)
  into draft_asset_count
  from public.scryglass_public_assets
  where release_id = p_release_id
    and path = 'features/draft_records.json';
  if draft_asset_count <> 0 then
    raise exception 'Scryglass release contains unavailable draft data';
  end if;

  select count(*)
  into present_assets
  from public.scryglass_public_assets
  where release_id = p_release_id
    and path = any(required_assets);
  if present_assets <> pg_catalog.cardinality(required_assets) then
    raise exception 'Scryglass release required asset inventory is incomplete';
  end if;

  select pg_catalog.jsonb_array_length(release_manifest -> 'files'),
         count(distinct file ->> 'path')
  into manifest_files, manifest_paths
  from pg_catalog.jsonb_array_elements(release_manifest -> 'files') file;
  select count(*)
  into release_assets
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
    raise exception 'Scryglass release asset integrity is invalid';
  end if;
end;
$$;

create or replace function public.guard_scryglass_public_asset_mutation()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $$
declare
  target_release_id text := coalesce(new.release_id, old.release_id);
begin
  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtext('scryglass-public-release')
  );
  if not exists (
    select 1
    from public.scryglass_public_releases
    where release_id = target_release_id
      and status = 'staging'
  ) then
    raise exception 'Scryglass assets are immutable outside staging';
  end if;
  return new;
end;
$$;

drop trigger if exists guard_scryglass_public_asset_mutation
  on public.scryglass_public_assets;
create trigger guard_scryglass_public_asset_mutation
before insert or update on public.scryglass_public_assets
for each row execute function public.guard_scryglass_public_asset_mutation();

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
    if old.status <> 'superseded' then
      raise exception 'Only superseded Scryglass releases can be deleted';
    end if;
    return old;
  end if;
  if old.status <> 'staging'
     and (
       new.release_id is distinct from old.release_id
       or new.manifest is distinct from old.manifest
       or new.source_as_of is distinct from old.source_as_of
     )
  then
    raise exception 'Scryglass release metadata is immutable after staging';
  end if;
  if (old.status = 'active' and new.status not in ('active', 'superseded'))
     or (old.status = 'superseded' and new.status not in ('superseded', 'active'))
  then
    raise exception 'Scryglass release status transition is invalid';
  end if;
  return new;
end;
$$;

drop trigger if exists guard_scryglass_public_release_mutation
  on public.scryglass_public_releases;
create trigger guard_scryglass_public_release_mutation
before insert or update or delete on public.scryglass_public_releases
for each row execute function public.guard_scryglass_public_release_mutation();

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
    'features/match_records_2025_q1.json',
    'features/match_records_2025_q2.json',
    'features/match_records_2025_q3.json',
    'features/match_records_2025_q4.json',
    'features/match_records_2026_q1.json',
    'features/match_records_2026_q2.json',
    'features/match_records_2026_q3.json',
    'features/match_records_2026_q4.json',
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
  perform public.assert_scryglass_public_release_integrity(p_release_id);

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
     or draft_authority_status is distinct from 'unavailable'
  then
    raise exception 'Scryglass draft authority binding is invalid';
  end if;
  select count(*)
  into draft_asset_count
  from public.scryglass_public_assets
  where release_id = p_release_id
    and path = 'features/draft_records.json';
  if draft_asset_count <> 0 then
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
  release_manifest jsonb;
  draft_asset_count integer;
begin
  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtext('scryglass-public-release')
  );
  perform public.assert_scryglass_public_release_integrity(p_release_id);
  select manifest
  into release_manifest
    from public.scryglass_public_releases
    where release_id = p_release_id
      and status in ('active', 'superseded')
    for update;
  if release_manifest is null then
    raise exception 'Scryglass rollback release is unavailable';
  end if;
  if release_manifest #>> '{draft_authority,schema_version}'
       is distinct from 'scryglass:draft-authority:v1'
     or release_manifest #>> '{draft_authority,status}' is distinct from 'unavailable'
     or release_manifest #>> '{draft_authority,release_id}' is distinct from p_release_id
  then
    raise exception 'Scryglass rollback draft authority is invalid';
  end if;
  select count(*)
  into draft_asset_count
  from public.scryglass_public_assets
  where release_id = p_release_id
    and path = 'features/draft_records.json';
  if draft_asset_count <> 0 then
    raise exception 'Scryglass rollback draft asset is unavailable';
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
  select coalesce(pg_catalog.array_agg(release_id), array[]::text[])
  into removed_releases
  from public.scryglass_public_releases
  where status = 'superseded'
    and release_id not in (
      select release_id
      from public.scryglass_public_releases
      where status = 'superseded'
      order by activated_at desc nulls last, created_at desc
      limit greatest(p_keep - 1, 0)
    );
  insert into public.scryglass_storage_cleanup (storage_path, release_id)
  select storage_path, release_id
  from public.scryglass_public_assets
  where release_id = any(removed_releases)
    and storage_path is not null
  on conflict (storage_path) do nothing;
  delete from public.scryglass_public_releases
  where release_id = any(removed_releases);
  select coalesce(pg_catalog.array_agg(storage_path), array[]::text[])
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

create or replace function public.get_scryglass_active_release(
  p_release_id text default null
)
returns table (release_id text, status text, manifest jsonb)
language sql
stable
security definer
set search_path = ''
as $$
  select
    candidate.release_id,
    candidate.status,
    pg_catalog.jsonb_strip_nulls(
      pg_catalog.jsonb_build_object(
        'pack_id', candidate.release_id,
        'schema_version', '2.0.0',
        'created_utc', pg_catalog.to_jsonb(candidate.created_at),
        'filters', pg_catalog.jsonb_build_object(
          'years', coalesce(
            (
              select pg_catalog.jsonb_agg(item.value order by item.ordinality)
              from pg_catalog.jsonb_array_elements(
                case
                  when pg_catalog.jsonb_typeof(candidate.manifest #> '{filters,years}') = 'array'
                    then candidate.manifest #> '{filters,years}'
                  else '[]'::jsonb
                end
              ) with ordinality as item(value, ordinality)
              where pg_catalog.jsonb_typeof(item.value) = 'number'
                and item.ordinality <= 10
            ),
            '[]'::jsonb
          ),
          'leagues', 'all_in_year_window'
        ),
        'attribution', 'Completed professional match data from approved public sources.',
        'excluded', pg_catalog.jsonb_build_array(
          'raw game rows',
          'research studies',
          'predictive artifacts'
        ),
        'base_url', null,
        'data_backend', 'supabase',
        'tier', pg_catalog.jsonb_build_object(
          'status', 'available',
          'as_of', pg_catalog.to_jsonb(candidate.source_as_of)
        ),
        'draft_authority', pg_catalog.jsonb_build_object(
          'schema_version', 'scryglass:draft-authority:v1',
          'status', 'unavailable',
          'release_id', candidate.release_id,
          'model_version', null,
          'receipt_sha256', null,
          'issued_utc', null,
          'reason', coalesce(
            candidate.manifest #> '{draft_authority,reason}',
            '"model_not_promoted"'::jsonb
          )
        ),
        'ratings', pg_catalog.jsonb_build_object(
          'source_as_of', pg_catalog.to_jsonb(candidate.source_as_of),
          'claim_ceiling', 'descriptive'
        ),
        'total_bytes', pg_catalog.to_jsonb(
          coalesce(
            (
              select pg_catalog.sum(asset.bytes)
              from public.scryglass_public_assets as asset
              where asset.release_id = candidate.release_id
                and asset.path <> 'features/draft_records.json'
            ),
            0
          )
        ),
        'total_files', pg_catalog.to_jsonb(
          (
            select pg_catalog.count(*)
            from public.scryglass_public_assets as asset
            where asset.release_id = candidate.release_id
              and asset.path <> 'features/draft_records.json'
          )
        ),
        'files', coalesce(
          (
            select pg_catalog.jsonb_agg(
              pg_catalog.jsonb_build_object(
                'path', asset.path,
                'bytes', asset.bytes,
                'sha256', asset.sha256
              )
              order by asset.path
            )
            from public.scryglass_public_assets as asset
            where asset.release_id = candidate.release_id
              and asset.path <> 'features/draft_records.json'
          ),
          '[]'::jsonb
        ),
        'release', pg_catalog.jsonb_build_object(
          'release_id', candidate.release_id,
          'tier_list_version', candidate.manifest #> '{release,tier_list_version}',
          'artifact_hashes', coalesce(
            (
              select pg_catalog.jsonb_object_agg(asset.path, asset.sha256 order by asset.path)
              from public.scryglass_public_assets as asset
              where asset.release_id = candidate.release_id
                and asset.path <> 'features/draft_records.json'
            ),
            '{}'::jsonb
          )
        )
      )
    ) as manifest
  from public.scryglass_public_releases as candidate
  where candidate.status = 'active'
    and (p_release_id is null or candidate.release_id = p_release_id)
$$;

create or replace function public.get_scryglass_active_asset(
  p_release_id text,
  p_path text
)
returns table (
  release_id text,
  path text,
  storage_path text,
  bytes bigint,
  sha256 text,
  content_type text
)
language sql
stable
security definer
set search_path = ''
as $$
  select
    asset.release_id,
    asset.path,
    asset.storage_path,
    asset.bytes,
    asset.sha256,
    asset.content_type
  from public.scryglass_public_assets as asset
  join public.scryglass_public_releases as release
    on release.release_id = asset.release_id
  where release.status = 'active'
    and asset.release_id = p_release_id
    and asset.path = p_path
    and asset.path <> 'features/draft_records.json'
    and asset.body is null
    and asset.storage_path = asset.release_id || '/' || asset.path
$$;

create or replace function public.get_scryglass_public_health()
returns table (
  status text,
  refresh_status text,
  checked_at timestamptz,
  last_success_at timestamptz,
  source_as_of timestamptz,
  active_release_id text,
  stale boolean
)
language sql
stable
security definer
set search_path = ''
as $$
  select
    health.status,
    health.refresh_status,
    health.checked_at,
    health.last_success_at,
    health.source_as_of,
    health.active_release_id,
    health.stale
  from public.scryglass_public_health as health
  where health.health_id = 'public-refresh'
$$;

create or replace function public.get_scryglass_private_health(p_token text)
returns table (
  status text,
  refresh_status text,
  checked_at timestamptz,
  last_success_at timestamptz,
  source_as_of timestamptz,
  active_release_id text,
  last_run_id text,
  worker_commit text,
  stale boolean
)
language sql
stable
security definer
set search_path = ''
as $$
  select
    health.status,
    health.refresh_status,
    health.checked_at,
    health.last_success_at,
    health.source_as_of,
    health.active_release_id,
    health.last_run_id,
    health.worker_commit,
    health.stale
  from public.scryglass_public_health as health
  where health.health_id = 'public-refresh'
    and pg_catalog.length(p_token) between 32 and 512
    and exists (
      select 1
      from public.scryglass_diagnostic_credentials as credential
      where credential.credential_id = 'public-health'
        and credential.token_sha256 = pg_catalog.encode(
          extensions.digest(pg_catalog.convert_to(p_token, 'UTF8'), 'sha256'),
          'hex'
        )
    )
$$;

create or replace function public.is_active_scryglass_storage_object(
  p_storage_path text
)
returns boolean
language sql
stable
security definer
set search_path = ''
as $$
  select coalesce(
    pg_catalog.length(p_storage_path) <= 220
    and exists (
      select 1
      from public.scryglass_public_assets as asset
      join public.scryglass_public_releases as release
        on release.release_id = asset.release_id
      where release.status = 'active'
        and asset.body is null
        and asset.path <> 'features/draft_records.json'
        and asset.storage_path = p_storage_path
        and asset.storage_path = asset.release_id || '/' || asset.path
    ),
    false
  )
$$;

drop policy if exists "read active Scryglass Storage assets" on storage.objects;
create policy "read active Scryglass Storage assets"
  on storage.objects
  for select
  to anon, authenticated
  using (
    bucket_id = 'scryglass-public'
    and public.is_active_scryglass_storage_object(storage.objects.name)
  );

revoke all on public.scryglass_public_releases
  from public, anon, authenticated, service_role;
revoke all on public.scryglass_public_assets
  from public, anon, authenticated, service_role;
revoke all on public.scryglass_public_health
  from public, anon, authenticated, service_role;
revoke all on public.scryglass_storage_cleanup
  from public, anon, authenticated, service_role;
revoke all on public.scryglass_diagnostic_credentials
  from public, anon, authenticated, service_role;
grant select, insert, update, delete on public.scryglass_public_releases to service_role;
grant select, insert, update on public.scryglass_public_assets to service_role;
grant select, insert, update on public.scryglass_public_health to service_role;
grant select, insert, delete on public.scryglass_storage_cleanup to service_role;
grant select, insert, update on public.scryglass_diagnostic_credentials to service_role;

-- Temporary rollout compatibility. RLS still limits these reads. The final
-- cutover migration revokes them after the web app uses the bounded RPCs.
grant select on public.scryglass_public_releases to anon, authenticated;
grant select on public.scryglass_public_assets to anon, authenticated;
grant select on public.scryglass_public_health to anon, authenticated;

revoke all on function public.get_scryglass_active_release(text)
  from public, anon, authenticated, service_role;
revoke all on function public.get_scryglass_active_asset(text, text)
  from public, anon, authenticated, service_role;
revoke all on function public.get_scryglass_public_health()
  from public, anon, authenticated, service_role;
revoke all on function public.get_scryglass_private_health(text)
  from public, anon, authenticated, service_role;
revoke all on function public.is_active_scryglass_storage_object(text)
  from public, anon, authenticated, service_role;
grant execute on function public.get_scryglass_active_release(text) to anon, authenticated;
grant execute on function public.get_scryglass_active_asset(text, text) to anon, authenticated;
grant execute on function public.get_scryglass_public_health() to anon, authenticated;
grant execute on function public.get_scryglass_private_health(text) to anon, authenticated;
grant execute on function public.is_active_scryglass_storage_object(text) to anon, authenticated;

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
revoke all on function public.assert_scryglass_public_release_integrity(text)
  from public, anon, authenticated, service_role;
revoke all on function public.guard_scryglass_public_asset_mutation()
  from public, anon, authenticated, service_role;
revoke all on function public.guard_scryglass_public_release_mutation()
  from public, anon, authenticated, service_role;
grant execute on function public.activate_scryglass_public_release(text) to service_role;
grant execute on function public.restore_scryglass_public_release(text) to service_role;
grant execute on function public.prune_scryglass_public_releases_v2(integer) to service_role;
grant execute on function public.ack_scryglass_storage_cleanup(text[]) to service_role;
grant execute on function public.assert_scryglass_public_release_integrity(text) to service_role;
