-- Cascaded asset deletes run under the table owner in PostgreSQL. Keep the
-- retention RPC able to remove assets for a superseded release while direct
-- service-role table deletes remain revoked.
create or replace function public.guard_scryglass_public_asset_mutation()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $$
begin
  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtext('scryglass-public-release')
  );
  if tg_op = 'UPDATE' and (
    new.release_id is distinct from old.release_id
    or new.path is distinct from old.path
  ) then
    raise exception 'Scryglass asset identity is immutable';
  end if;
  if tg_op = 'DELETE' then
    if current_user in ('scryglass_release_retention_owner', 'postgres')
       or exists (
         select 1
         from public.scryglass_public_releases release
         where release.release_id = old.release_id
           and release.status = 'superseded'
       ) then
      return old;
    end if;
    raise exception 'Scryglass assets can only leave with a superseded release';
  end if;
  if not exists (
       select 1
       from public.scryglass_public_releases release
       where release.release_id = new.release_id
         and release.status = 'staging'
     )
     or (
       tg_op = 'UPDATE'
       and not exists (
         select 1
         from public.scryglass_public_releases release
         where release.release_id = old.release_id
           and release.status = 'staging'
       )
     ) then
    raise exception 'Scryglass assets are immutable outside staging';
  end if;
  return new;
end;
$$;
