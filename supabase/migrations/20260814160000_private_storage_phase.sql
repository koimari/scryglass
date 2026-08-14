-- Phase 2: make the public release bucket private while the compatible web
-- build still supports the active inline compatibility RPC.
--
-- This migration keeps direct table reads available for the old web build.
-- Phase 3 removes those reads after the strict web build is live.

begin;

create schema if not exists scryglass_private;
revoke all on schema scryglass_private from public;
grant usage on schema scryglass_private
  to anon, authenticated, service_role, scryglass_release_transition_owner;

update storage.buckets
set public = false,
    file_size_limit = 125829120,
    allowed_mime_types = array['application/json']
where id = 'scryglass-public';

create or replace function scryglass_private.is_active_scryglass_storage_object(
  p_storage_path text
)
returns boolean
language sql
stable
security definer
set search_path = public, pg_temp
set statement_timeout = '5s'
as $$
  select coalesce(
    pg_catalog.octet_length(p_storage_path) between 1 and 220
    and exists (
      select 1
      from public.scryglass_public_assets asset
      join public.scryglass_public_releases release
        on release.release_id = asset.release_id
      where release.status = 'active'
        and asset.body is null
        and asset.path <> 'features/draft_records.json'
        and asset.storage_path = p_storage_path
        and asset.storage_path = asset.release_id || '/' || asset.path
        and asset.content_type = 'application/json'
        and release.manifest #>> array['release', 'artifact_hashes', asset.path]
          = asset.sha256
        and exists (
          select 1
          from pg_catalog.jsonb_array_elements(
            case
              when pg_catalog.jsonb_typeof(release.manifest -> 'files') = 'array'
                then release.manifest -> 'files'
              else '[]'::jsonb
            end
          ) file
          where file ->> 'path' = asset.path
            and (file ->> 'bytes')::bigint = asset.bytes
            and file ->> 'sha256' = asset.sha256
        )
    ),
    false
  )
$$;

create or replace function public.is_active_scryglass_storage_object(
  p_storage_path text
)
returns boolean
language sql
stable
security invoker
set search_path = public, pg_temp
set statement_timeout = '5s'
as $$
  select scryglass_private.is_active_scryglass_storage_object($1)
$$;

revoke all on function public.is_active_scryglass_storage_object(text)
  from public, anon, authenticated, service_role;
grant execute on function public.is_active_scryglass_storage_object(text)
  to anon, authenticated;

drop policy if exists "read active Scryglass Storage assets" on storage.objects;
create policy "read active Scryglass Storage assets"
  on storage.objects
  for select
  to anon, authenticated
  using (
    bucket_id = 'scryglass-public'
    and public.is_active_scryglass_storage_object(storage.objects.name)
  );

drop policy if exists "service role manages Scryglass Storage assets" on storage.objects;
create policy "service role manages Scryglass Storage assets"
  on storage.objects
  for all
  to service_role
  using (bucket_id = 'scryglass-public')
  with check (bucket_id = 'scryglass-public');

create or replace function public.guard_active_scryglass_storage_object()
returns trigger
language plpgsql
security definer
set search_path = public, pg_temp
as $$
declare
  bucket_is_public boolean := old.bucket_id = 'scryglass-public'
    or new.bucket_id = 'scryglass-public';
begin
  if bucket_is_public and exists (
    select 1
    from public.scryglass_public_assets asset
    join public.scryglass_public_releases release
      on release.release_id = asset.release_id
    where release.status in ('active', 'superseded')
      and asset.storage_path in (old.name, new.name)
  ) then
    raise exception 'Published Scryglass Storage objects are immutable';
  end if;
  return case when tg_op = 'DELETE' then old else new end;
end;
$$;

-- The bounded-query migration revokes CREATE after its helper ownership setup.
-- Re-open that privilege only for this ownership transfer, then close it.
grant create on schema public to scryglass_release_transition_owner;
alter function public.guard_active_scryglass_storage_object()
  owner to scryglass_release_transition_owner;
revoke create on schema public from scryglass_release_transition_owner;
revoke all on function public.guard_active_scryglass_storage_object()
  from public, anon, authenticated, service_role;

drop trigger if exists guard_active_scryglass_storage_object on storage.objects;
create trigger guard_active_scryglass_storage_object
before update or delete on storage.objects
for each row execute function public.guard_active_scryglass_storage_object();

commit;
