-- Keep the public RPC names stable while ensuring the Data API entry points
-- are invoker functions. Privileged validation remains in scryglass_private.

alter function public.get_scryglass_active_asset(text, text)
  security invoker;
alter function public.get_scryglass_active_release(text)
  security invoker;

create or replace function public.get_scryglass_active_asset(
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
set statement_timeout = '5s'
as $$
  select * from scryglass_private.get_scryglass_active_asset(p_release_id, p_path);
$$;

create or replace function public.get_scryglass_active_release(
  p_release_id text default null
)
returns table(release_id text, status text, manifest jsonb)
language sql
stable
security invoker
set statement_timeout = '5s'
as $$
  select * from scryglass_private.get_scryglass_active_release(p_release_id);
$$;

revoke all on function scryglass_private.get_scryglass_active_asset(text, text)
  from service_role;
revoke all on function scryglass_private.get_scryglass_active_release(text)
  from service_role;
