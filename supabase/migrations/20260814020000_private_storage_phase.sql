-- Phase 2: make the release-bound Storage objects private.
-- The compatible web build is deployed before this migration. Direct table
-- reads and the parsed inline compatibility RPC remain available until the
-- strict cutover in 20260814030000.

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
    and public.is_active_scryglass_storage_object(storage.objects.name)
  );

-- A service-role publisher may write a staging object. It must not replace or
-- remove an object that belongs to an active release. The same transaction
-- lock is used by activation and restore, so a checked object cannot change
-- between integrity verification and the status transition.
create or replace function public.guard_scryglass_storage_object_mutation()
returns trigger
language plpgsql
security definer
set search_path = ''
as $$
declare
  old_active boolean := false;
  new_active boolean := false;
begin
  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtext('scryglass-public-release')
  );

  if tg_op in ('UPDATE', 'DELETE') then
    old_active := old.bucket_id = 'scryglass-public'
      and public.is_active_scryglass_storage_object(old.name);
  end if;
  if tg_op in ('INSERT', 'UPDATE') then
    new_active := new.bucket_id = 'scryglass-public'
      and public.is_active_scryglass_storage_object(new.name);
  end if;

  if old_active or (tg_op = 'INSERT' and new_active) then
    raise exception 'Active Scryglass Storage objects are immutable';
  end if;
  if tg_op = 'DELETE' then
    return old;
  end if;
  return new;
end;
$$;

drop trigger if exists guard_scryglass_storage_object_mutation on storage.objects;
create trigger guard_scryglass_storage_object_mutation
before insert or update or delete on storage.objects
for each row execute function public.guard_scryglass_storage_object_mutation();

revoke all on function public.guard_scryglass_storage_object_mutation()
  from public, anon, authenticated, service_role;
