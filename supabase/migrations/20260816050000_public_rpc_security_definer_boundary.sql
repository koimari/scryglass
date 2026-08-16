-- Keep the public RPC names stable while moving the current privileged
-- implementations out of the Data API schema. Older private copies from the
-- earlier hardening migration are replaced by the current implementations.

drop function if exists scryglass_private.get_scryglass_active_asset(text, text);
drop function if exists scryglass_private.get_scryglass_active_release(text);

alter function public.get_scryglass_active_asset(text, text)
  set schema scryglass_private;
alter function public.get_scryglass_active_release(text)
  set schema scryglass_private;

revoke all on function scryglass_private.get_scryglass_active_asset(text, text)
  from public, service_role;
revoke all on function scryglass_private.get_scryglass_active_release(text)
  from public, service_role;
grant execute on function scryglass_private.get_scryglass_active_asset(text, text)
  to anon, authenticated;
grant execute on function scryglass_private.get_scryglass_active_release(text)
  to anon, authenticated;

create function public.get_scryglass_active_asset(
  p_release_id text,
  p_path text
)
returns table(
  release_id text,
  path text,
  storage_path text,
  bytes bigint,
  sha256 text,
  content_type text
)
language sql
stable
security invoker
set search_path = ''
set statement_timeout = '5s'
as $$
  select * from scryglass_private.get_scryglass_active_asset(p_release_id, p_path);
$$;

create function public.get_scryglass_active_release(
  p_release_id text default null
)
returns table(release_id text, status text, manifest jsonb)
language sql
stable
security invoker
set search_path = ''
set statement_timeout = '5s'
as $$
  select * from scryglass_private.get_scryglass_active_release(p_release_id);
$$;

revoke all on function public.get_scryglass_active_asset(text, text)
  from public, service_role;
revoke all on function public.get_scryglass_active_release(text)
  from public, service_role;
grant execute on function public.get_scryglass_active_asset(text, text)
  to anon, authenticated;
grant execute on function public.get_scryglass_active_release(text)
  to anon, authenticated;
