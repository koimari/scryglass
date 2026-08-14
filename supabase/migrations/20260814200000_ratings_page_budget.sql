-- Allow the bounded ratings page to return the full 100-row public page.
-- Chat and exact-name callers keep their smaller application-side limits.
do $$
declare
  function_definition text;
  updated_definition text;
begin
  select pg_catalog.pg_get_functiondef(p.oid)
    into function_definition
  from pg_catalog.pg_proc p
  join pg_catalog.pg_namespace n on n.oid = p.pronamespace
  where n.nspname = 'scryglass_private'
    and p.proname = 'get_scryglass_ratings'
    and p.pronargs = 12
  limit 1;

  if function_definition is null then
    raise exception 'Scryglass private ratings function is missing';
  end if;

  if pg_catalog.position(
    'least(greatest(coalesce(p_limit, 20), 1), 100)' in function_definition
  ) > 0 then
    return;
  end if;

  updated_definition := pg_catalog.replace(
    function_definition,
    'least(greatest(coalesce(p_limit, 20), 1), 20)',
    'least(greatest(coalesce(p_limit, 20), 1), 100)'
  );
  if updated_definition = function_definition then
    raise exception 'Scryglass ratings function has an unknown limit guard';
  end if;
  execute updated_definition;
end;
$$;
