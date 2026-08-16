-- PostgreSQL runs foreign-key cascade triggers as the table owner. Permit
-- that internal context to remove query rows for a superseded release while
-- direct Data API table deletes remain revoked.
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
         'scryglass_release_transition_owner',
         'postgres'
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

revoke all on function public.guard_scryglass_query_mutation()
  from public, anon, authenticated, service_role;
