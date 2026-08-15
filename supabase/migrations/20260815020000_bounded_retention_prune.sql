-- Keep each retention transaction inside the Data API statement budget.
-- Query releases can contain more than 160,000 indexed rows, so prune one
-- superseded release per RPC call and let the publisher repeat bounded calls.

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
  -- Foreign-key cascades execute as the table owner. The parent release guard
  -- has already restricted deletion to a superseded release. Avoid one lock
  -- and parent lookup for every query row in that internal cascade.
  if tg_op = 'DELETE' and current_user = 'postgres' then
    return old;
  end if;

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

create or replace function public.prune_scryglass_public_releases_v2(
  p_keep integer default 3
)
returns jsonb
language plpgsql
security definer
set search_path = ''
as $$
declare
  removed_releases text[];
  queued_storage_paths text[];
begin
  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtext('scryglass-public-release')
  );
  if p_keep is null or p_keep < 1 or p_keep > 10 then
    raise exception 'Scryglass retention must be between 1 and 10 releases';
  end if;

  select coalesce(
    pg_catalog.array_agg(candidate.release_id),
    array[]::text[]
  )
  into removed_releases
  from (
    select release.release_id
    from public.scryglass_public_releases release
    where release.status = 'superseded'
      and release.release_id not in (
        select retained.release_id
        from public.scryglass_public_releases retained
        where retained.status = 'superseded'
        order by retained.activated_at desc nulls last, retained.created_at desc
        limit greatest(p_keep - 1, 0)
      )
    order by release.activated_at asc nulls first, release.created_at asc
    limit 1
  ) candidate;

  insert into public.scryglass_storage_cleanup (storage_path, release_id)
  select asset.storage_path, asset.release_id
  from public.scryglass_public_assets asset
  where asset.release_id = any(removed_releases)
    and asset.storage_path is not null
  on conflict (storage_path) do nothing;

  delete from public.scryglass_public_releases release
  where release.release_id = any(removed_releases);

  select coalesce(
    pg_catalog.array_agg(cleanup.storage_path order by cleanup.storage_path),
    array[]::text[]
  )
  into queued_storage_paths
  from public.scryglass_storage_cleanup cleanup;

  return pg_catalog.jsonb_build_object(
    'deleted_count', pg_catalog.cardinality(removed_releases),
    'release_ids', pg_catalog.to_jsonb(removed_releases),
    'storage_paths', pg_catalog.to_jsonb(queued_storage_paths)
  );
end;
$$;

alter function public.prune_scryglass_public_releases_v2(integer)
  owner to scryglass_release_retention_owner;

revoke all on function public.guard_scryglass_query_mutation()
  from public, anon, authenticated, service_role;
revoke all on function public.prune_scryglass_public_releases_v2(integer)
  from public, anon, authenticated, service_role;
grant execute on function public.prune_scryglass_public_releases_v2(integer)
  to service_role;
