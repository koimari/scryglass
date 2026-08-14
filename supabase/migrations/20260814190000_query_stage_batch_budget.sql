-- Keep service-only query staging within the publisher's request budget.
-- The original function accepted 100 rows per call. A current projection can
-- contain more than 150 MB, so that limit creates thousands of REST calls.
-- The larger bound remains limited to the service role and to 3.5 MB of JSON.

do $migration$
declare
  definition text;
begin
  select pg_catalog.pg_get_functiondef(
    'public.stage_scryglass_query_rows(text,text,jsonb)'::pg_catalog.regprocedure
  )
  into definition;
  if definition like '%jsonb_array_length(p_rows) > 500%'
     and definition like '%octet_length(p_rows::text) > 3500000%' then
    return;
  end if;
  if definition not like '%jsonb_array_length(p_rows) > 100%'
     or definition not like '%octet_length(p_rows::text) > 450000%' then
    raise exception 'Scryglass query staging function has an unknown batch budget';
  end if;
  definition := pg_catalog.replace(definition, 'jsonb_array_length(p_rows) > 100', 'jsonb_array_length(p_rows) > 500');
  definition := pg_catalog.replace(definition, 'octet_length(p_rows::text) > 450000', 'octet_length(p_rows::text) > 3500000');
  execute definition;
end;
$migration$;
