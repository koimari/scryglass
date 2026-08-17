-- Use local names that cannot collide with release and asset columns when the
-- Storage policy evaluates a private active-object lookup.

create or replace function scryglass_private.is_active_scryglass_storage_object(
  p_storage_path text
)
returns boolean
language plpgsql
stable
security definer
set search_path = ''
set statement_timeout = '5s'
as $$
declare
  v_release_id text := pg_catalog.split_part(p_storage_path, '/', 1);
  v_asset_path text := pg_catalog.substr(
    p_storage_path,
    pg_catalog.length(v_release_id) + 2
  );
  v_authority_status text;
begin
  select release.manifest #>> '{draft_authority,status}'
    into v_authority_status
  from public.scryglass_public_releases as release
  where release.release_id = v_release_id
    and release.status = 'active';

  if v_asset_path = 'features/promoted_draft_results.json'
    and v_authority_status <> 'promoted'
  then
    return false;
  end if;
  if v_asset_path = 'features/draft_records.json'
    and public.scryglass_query_descriptive_authority(v_release_id) is null
  then
    return false;
  end if;
  return scryglass_private.is_active_scryglass_storage_object_before_promotion(
    p_storage_path
  );
end;
$$;

revoke all on function scryglass_private.is_active_scryglass_storage_object(text)
  from public, service_role;
grant execute on function scryglass_private.is_active_scryglass_storage_object(text)
  to anon, authenticated, supabase_storage_admin;
