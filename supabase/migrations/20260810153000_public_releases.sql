create table if not exists public.scryglass_public_releases (
  release_id text primary key,
  status text not null default 'staging'
    check (status in ('staging', 'active', 'superseded')),
  manifest jsonb not null,
  source_as_of timestamptz,
  created_at timestamptz not null default now(),
  activated_at timestamptz,
  check (release_id ~ '^v[0-9]{4}\.[0-9]{2}\.[0-9]{2}\.[0-9]{6}$'),
  check (manifest ->> 'pack_id' = release_id)
);

create unique index if not exists scryglass_one_active_release
  on public.scryglass_public_releases ((status))
  where status = 'active';

create table if not exists public.scryglass_public_assets (
  release_id text not null references public.scryglass_public_releases(release_id)
    on delete cascade,
  path text not null,
  body jsonb,
  storage_path text,
  bytes bigint not null check (bytes >= 0),
  sha256 text not null check (sha256 ~ '^[0-9a-f]{64}$'),
  created_at timestamptz not null default now(),
  primary key (release_id, path),
  constraint scryglass_public_assets_payload_check check (
    (body is not null and storage_path is null)
    or (body is null and storage_path is not null)
  ),
  check (
    path in (
      'features/ratings_snapshot.json',
      'features/player_ratings_snapshot.json',
      'features/team_records.json',
      'features/team_weekly_ranks.json',
      'features/player_records.json',
      'features/player_champion_records.json',
      'features/profile_records.json',
      'features/player_weekly_ranks.json',
      'features/player_metadata.json',
      'rankings/tierlists.json'
    )
  )
);

insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values (
  'scryglass-public',
  'scryglass-public',
  true,
  52428800,
  array['application/json']
)
on conflict (id) do update
set public = excluded.public,
    file_size_limit = excluded.file_size_limit,
    allowed_mime_types = excluded.allowed_mime_types;

alter table public.scryglass_public_releases enable row level security;
alter table public.scryglass_public_assets enable row level security;

drop policy if exists "read active Scryglass release" on public.scryglass_public_releases;
create policy "read active Scryglass release"
  on public.scryglass_public_releases
  for select
  to anon, authenticated
  using (status = 'active');

drop policy if exists "read active Scryglass assets" on public.scryglass_public_assets;
create policy "read active Scryglass assets"
  on public.scryglass_public_assets
  for select
  to anon, authenticated
  using (
    exists (
      select 1
      from public.scryglass_public_releases release
      where release.release_id = scryglass_public_assets.release_id
        and release.status = 'active'
    )
  );

create or replace function public.activate_scryglass_public_release(p_release_id text)
returns jsonb
language plpgsql
security definer
set search_path = ''
as $$
declare
  previous_release_id text;
  required_assets constant text[] := array[
    'features/ratings_snapshot.json',
    'features/player_ratings_snapshot.json',
    'features/team_records.json',
    'features/team_weekly_ranks.json',
    'features/player_records.json',
    'features/player_champion_records.json',
    'features/profile_records.json',
    'features/player_weekly_ranks.json',
    'features/player_metadata.json',
    'rankings/tierlists.json'
  ];
  present_assets integer;
begin
  perform pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtext('scryglass-public-release'));

  if not exists (
    select 1
    from public.scryglass_public_releases
    where release_id = p_release_id
      and status in ('staging', 'active')
  ) then
    raise exception 'Scryglass release is not ready for activation';
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

  return jsonb_build_object(
    'status', 'active',
    'release_id', p_release_id,
    'previous_release_id', previous_release_id
  );
end;
$$;

create or replace function public.restore_scryglass_public_release(p_release_id text)
returns jsonb
language plpgsql
security definer
set search_path = ''
as $$
declare
  replaced_release_id text;
begin
  perform pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtext('scryglass-public-release'));

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

  return jsonb_build_object(
    'status', 'restored',
    'release_id', p_release_id,
    'replaced_release_id', replaced_release_id
  );
end;
$$;

create or replace function public.prune_scryglass_public_releases(p_keep integer default 3)
returns integer
language plpgsql
security definer
set search_path = ''
as $$
declare
  deleted_count integer;
begin
  if p_keep < 1 or p_keep > 10 then
    raise exception 'Scryglass retention must be between 1 and 10 releases';
  end if;

  with deleted as (
    delete from public.scryglass_public_releases
    where status = 'superseded'
      and release_id not in (
        select release_id
        from public.scryglass_public_releases
        where status = 'superseded'
        order by activated_at desc nulls last, created_at desc
        limit greatest(p_keep - 1, 0)
      )
    returning release_id
  )
  select count(*) into deleted_count from deleted;

  return deleted_count;
end;
$$;

revoke all on public.scryglass_public_releases from anon, authenticated;
revoke all on public.scryglass_public_assets from anon, authenticated;
grant select on public.scryglass_public_releases to anon, authenticated;
grant select on public.scryglass_public_assets to anon, authenticated;

revoke all on function public.activate_scryglass_public_release(text) from public, anon, authenticated;
revoke all on function public.restore_scryglass_public_release(text) from public, anon, authenticated;
revoke all on function public.prune_scryglass_public_releases(integer) from public, anon, authenticated;
grant execute on function public.activate_scryglass_public_release(text) to service_role;
grant execute on function public.restore_scryglass_public_release(text) to service_role;
grant execute on function public.prune_scryglass_public_releases(integer) to service_role;
