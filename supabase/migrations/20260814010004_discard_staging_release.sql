-- Staging releases are disposable publication work.  Give the publisher an
-- audited, locked cleanup path so failed refreshes cannot accumulate query
-- rows and Storage objects indefinitely.

grant select, delete on public.scryglass_public_query_rows
  to scryglass_release_transition_owner;
grant select, delete on public.scryglass_public_query_receipts
  to scryglass_release_transition_owner;
grant select, delete on public.scryglass_public_assets
  to scryglass_release_transition_owner;
grant delete on public.scryglass_public_releases
  to scryglass_release_transition_owner;
grant select, insert, delete on public.scryglass_storage_cleanup
  to scryglass_release_transition_owner;

create or replace function public.guard_scryglass_query_mutation()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $$
declare
  target_release text := coalesce(new.release_id, old.release_id);
  target_dataset text := coalesce(new.dataset, old.dataset);
begin
  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtext('scryglass-public-release')
  );
  if tg_op = 'DELETE' then
    if current_user in (
         'scryglass_release_retention_owner',
         'scryglass_release_transition_owner'
       )
       or exists (
         select 1
         from public.scryglass_public_releases release
         where release.release_id = target_release
           and release.status = 'superseded'
       ) then
      return old;
    end if;
    raise exception 'Scryglass query data can only leave with a superseded release';
  end if;
  if tg_op <> 'INSERT' then
    raise exception 'Scryglass query rows and receipts are immutable';
  end if;
  if not exists (
    select 1
    from public.scryglass_public_releases release
    where release.release_id = target_release
      and release.status = 'staging'
  ) then
    raise exception 'Scryglass query data can only enter a staging release';
  end if;
  if tg_table_name = 'scryglass_public_query_rows'
     and exists (
       select 1
       from public.scryglass_public_query_receipts receipt
       where receipt.release_id = target_release
         and receipt.dataset = target_dataset
     ) then
    raise exception 'A sealed Scryglass query dataset is immutable';
  end if;
  return new;
end;
$$;

create or replace function public.discard_scryglass_staging_release(
  p_release_id text
)
returns jsonb
language plpgsql
security definer
set search_path = ''
as $$
declare
  storage_paths text[];
  discarded text;
begin
  if p_release_id is null
     or p_release_id !~ '^v[0-9]{4}\.[0-9]{2}\.[0-9]{2}\.[0-9]{6}$'
  then
    raise exception 'Scryglass staging release ID is invalid';
  end if;

  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtext('scryglass-public-release')
  );

  if exists (
    select 1
    from public.scryglass_public_releases release
    where release.release_id = p_release_id
      and release.status <> 'staging'
  ) then
    raise exception 'Only a staging Scryglass release can be discarded';
  end if;

  if not exists (
    select 1
    from public.scryglass_public_releases release
    where release.release_id = p_release_id
      and release.status = 'staging'
  ) then
    select coalesce(
      pg_catalog.array_agg(cleanup.storage_path order by cleanup.storage_path),
      array[]::text[]
    )
    into storage_paths
    from public.scryglass_storage_cleanup cleanup
    where cleanup.release_id = p_release_id;
    return pg_catalog.jsonb_build_object(
      'status', 'absent',
      'release_id', p_release_id,
      'storage_paths', pg_catalog.to_jsonb(storage_paths)
    );
  end if;

  select coalesce(
    pg_catalog.array_agg(asset.storage_path order by asset.storage_path)
      filter (where asset.storage_path is not null),
    array[]::text[]
  )
  into storage_paths
  from public.scryglass_public_assets asset
  where asset.release_id = p_release_id;

  insert into public.scryglass_storage_cleanup (storage_path, release_id)
  select path, p_release_id
  from pg_catalog.unnest(storage_paths) as item(path)
  on conflict (storage_path) do nothing;

  update public.scryglass_public_releases
  set status = 'superseded'
  where release_id = p_release_id
    and status = 'staging';

  delete from public.scryglass_public_query_rows
  where release_id = p_release_id;
  delete from public.scryglass_public_query_receipts
  where release_id = p_release_id;
  delete from public.scryglass_public_assets
  where release_id = p_release_id;
  delete from public.scryglass_public_releases
  where release_id = p_release_id
    and status = 'superseded'
  returning release_id into discarded;

  if discarded is null then
    raise exception 'Scryglass staging release discard did not remove the release';
  end if;

  return pg_catalog.jsonb_build_object(
    'status', 'discarded',
    'release_id', discarded,
    'storage_paths', pg_catalog.to_jsonb(storage_paths)
  );
end;
$$;

grant create on schema public to scryglass_release_transition_owner;
alter function public.discard_scryglass_staging_release(text)
  owner to scryglass_release_transition_owner;
revoke all on function public.discard_scryglass_staging_release(text)
  from public, anon, authenticated, service_role;
grant execute on function public.discard_scryglass_staging_release(text)
  to service_role;

create or replace function public.drain_scryglass_staging_cleanup(
  p_release_id text
)
returns jsonb
language plpgsql
security definer
set search_path = ''
as $$
declare
  storage_paths text[];
begin
  if p_release_id is null
     or p_release_id !~ '^v[0-9]{4}\.[0-9]{2}\.[0-9]{2}\.[0-9]{6}$'
  then
    raise exception 'Scryglass staging release ID is invalid';
  end if;

  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtext('scryglass-public-release')
  );

  if exists (
    select 1
    from public.scryglass_public_releases release
    where release.release_id = p_release_id
      and release.status = 'staging'
  ) then
    return pg_catalog.jsonb_build_object(
      'status', 'staging',
      'release_id', p_release_id,
      'storage_paths', '[]'::jsonb
    );
  end if;

  select coalesce(
    pg_catalog.array_agg(cleanup.storage_path order by cleanup.storage_path),
    array[]::text[]
  )
  into storage_paths
  from public.scryglass_storage_cleanup cleanup
  where cleanup.release_id = p_release_id;

  return pg_catalog.jsonb_build_object(
    'status', 'ready',
    'release_id', p_release_id,
    'storage_paths', pg_catalog.to_jsonb(storage_paths)
  );
end;
$$;

grant create on schema public to scryglass_release_transition_owner;
alter function public.drain_scryglass_staging_cleanup(text)
  owner to scryglass_release_transition_owner;
revoke all on function public.drain_scryglass_staging_cleanup(text)
  from public, anon, authenticated, service_role;
grant execute on function public.drain_scryglass_staging_cleanup(text)
  to service_role;

create or replace function public.discard_stale_scryglass_staging_releases(
  p_min_age_minutes integer default 360,
  p_limit integer default 10
)
returns jsonb
language plpgsql
security definer
set search_path = ''
as $$
declare
  candidate record;
  discarded jsonb;
  release_ids text[] := array[]::text[];
  storage_paths text[] := array[]::text[];
begin
  if p_min_age_minutes is null
     or p_min_age_minutes < 30
     or p_min_age_minutes > 10080
     or p_limit is null
     or p_limit < 1
     or p_limit > 20
  then
    raise exception 'Scryglass staging retention bounds are invalid';
  end if;

  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtext('scryglass-public-release')
  );

  for candidate in
    select release.release_id
    from public.scryglass_public_releases release
    where release.status = 'staging'
      and release.created_at < now() - make_interval(mins => p_min_age_minutes)
    order by release.created_at
    limit p_limit
  loop
    discarded := public.discard_scryglass_staging_release(candidate.release_id);
    if discarded ->> 'status' = 'discarded' then
      release_ids := release_ids || candidate.release_id;
      storage_paths := storage_paths || coalesce(
        array(
          select pg_catalog.jsonb_array_elements_text(
            discarded -> 'storage_paths'
          )
        ),
        array[]::text[]
      );
    end if;
  end loop;

  return pg_catalog.jsonb_build_object(
    'status', 'complete',
    'release_ids', pg_catalog.to_jsonb(release_ids),
    'storage_paths', pg_catalog.to_jsonb(storage_paths)
  );
end;
$$;

alter function public.discard_stale_scryglass_staging_releases(integer, integer)
  owner to scryglass_release_transition_owner;
revoke create on schema public from scryglass_release_transition_owner;
revoke all on function public.discard_stale_scryglass_staging_releases(integer, integer)
  from public, anon, authenticated, service_role;
grant execute on function public.discard_stale_scryglass_staging_releases(integer, integer)
  to service_role;
