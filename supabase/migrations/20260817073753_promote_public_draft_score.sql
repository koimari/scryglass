-- Promote the fixed public Draft Score while retaining the separate
-- descriptive composition evidence used by bounded ranking queries.

alter function public.assert_scryglass_public_release_integrity(text)
  rename to assert_scryglass_public_release_integrity_before_promotion;

revoke all on function public.assert_scryglass_public_release_integrity_before_promotion(text)
  from public, anon, authenticated, service_role;
grant execute on function public.assert_scryglass_public_release_integrity_before_promotion(text)
  to scryglass_release_transition_owner;

create function public.assert_scryglass_public_release_integrity(
  p_release_id text
)
returns void
language plpgsql
security definer
set search_path = ''
set statement_timeout = '30s'
as $$
declare
  release_manifest jsonb;
  authority jsonb;
  descriptive jsonb;
  required_assets constant text[] := array[
    'features/ratings_snapshot.json','features/player_ratings_snapshot.json',
    'features/team_records.json','features/team_weekly_ranks.json',
    'features/player_records.json','features/player_champion_records.json',
    'features/profile_records.json','features/match_index.json',
    'features/match_records_2025_q1.json','features/match_records_2025_q2.json',
    'features/match_records_2025_q3.json','features/match_records_2025_q4.json',
    'features/match_records_2026_q1.json','features/match_records_2026_q2.json',
    'features/match_records_2026_q3.json','features/match_records_2026_q4.json',
    'features/player_weekly_ranks.json','features/player_metadata.json',
    'rankings/tierlists.json','rankings/tierlists-latest.json'
  ];
  present_assets integer;
  manifest_files integer;
  manifest_paths integer;
  manifest_hashes integer;
  release_assets integer;
  invalid_assets integer;
begin
  select release.manifest into release_manifest
  from public.scryglass_public_releases release
  where release.release_id=p_release_id
    and release.status in ('staging','active','superseded');
  authority := release_manifest->'draft_authority';
  if authority->>'status' is distinct from 'promoted' then
    perform public.assert_scryglass_public_release_integrity_before_promotion(p_release_id);
    return;
  end if;
  descriptive := authority->'descriptive_authority';
  if release_manifest is null
    or release_manifest->>'pack_id' is distinct from p_release_id
    or release_manifest#>>'{release,release_id}' is distinct from p_release_id
    or authority->>'schema_version' is distinct from 'scryglass:draft-authority:v1'
    or authority->>'authority' is distinct from 'promoted'
    or authority->>'release_id' is distinct from p_release_id
    or authority->>'estimand' is distinct from 'prematch_map_win_probability_with_controlled_draft_intervention'
    or pg_catalog.octet_length(authority->>'model_version') not between 1 and 100
    or authority->>'artifact_sha256' !~ '^[0-9a-f]{64}$'
    or authority->>'receipt_sha256' !~ '^[0-9a-f]{64}$'
    or authority->>'issued_utc' !~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}([.][0-9]{1,6})?Z$'
    or coalesce((authority->>'probability_authority')::boolean,false) is not true
    or coalesce((authority->>'recommendation_authority')::boolean,false) is not true
    or coalesce((authority->>'betting_authority')::boolean,true) is not false
    or authority->'reason' is distinct from 'null'::jsonb
    or descriptive->>'schema_version' is distinct from 'scryglass:draft-authority:v1'
    or descriptive->>'status' is distinct from 'descriptive'
    or descriptive->>'authority' is distinct from 'descriptive'
    or descriptive->>'estimand' is distinct from 'composition_only'
    or descriptive->>'release_id' is distinct from p_release_id
    or pg_catalog.octet_length(descriptive->>'model_version') not between 1 and 100
    or descriptive->>'artifact_sha256' !~ '^[0-9a-f]{64}$'
    or descriptive->>'receipt_sha256' !~ '^[0-9a-f]{64}$'
    or descriptive->>'issued_utc' !~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}([.][0-9]{1,6})?Z$'
    or coalesce((descriptive->>'probability_authority')::boolean,true)
    or coalesce((descriptive->>'recommendation_authority')::boolean,true)
    or coalesce((descriptive->>'betting_authority')::boolean,true)
  then raise exception 'Scryglass promoted Draft Score authority is invalid'; end if;
  begin
    perform (authority->>'issued_utc')::timestamptz;
    perform (descriptive->>'issued_utc')::timestamptz;
  exception when others then
    raise exception 'Scryglass promoted Draft Score timestamp is invalid';
  end;
  if (select pg_catalog.count(*) from public.scryglass_public_assets asset
      where asset.release_id=p_release_id and asset.path='features/draft_records.json')<>1
    or (select pg_catalog.count(*) from public.scryglass_public_assets asset
      where asset.release_id=p_release_id and asset.path='features/promoted_draft_results.json')<>1
  then raise exception 'Scryglass promoted Draft Score assets are missing'; end if;
  select pg_catalog.count(*) into present_assets
  from public.scryglass_public_assets asset
  where asset.release_id=p_release_id and asset.path=any(required_assets);
  if present_assets<>pg_catalog.cardinality(required_assets) then
    raise exception 'Scryglass release required asset inventory is incomplete';
  end if;
  select pg_catalog.jsonb_array_length(release_manifest->'files'),
    pg_catalog.count(distinct file->>'path')
  into manifest_files,manifest_paths
  from pg_catalog.jsonb_array_elements(release_manifest->'files') file;
  select pg_catalog.count(*) into manifest_hashes
  from pg_catalog.jsonb_object_keys(release_manifest#>'{release,artifact_hashes}');
  select pg_catalog.count(*) into release_assets
  from public.scryglass_public_assets asset where asset.release_id=p_release_id;
  if manifest_files<>manifest_paths or manifest_files<>manifest_hashes
    or manifest_files<>release_assets
  then raise exception 'Scryglass release manifest inventory is not exact'; end if;
  select pg_catalog.count(*) into invalid_assets
  from public.scryglass_public_assets asset
  where asset.release_id=p_release_id and (
    asset.body is not null
    or asset.storage_path is distinct from asset.release_id||'/'||asset.path
    or asset.content_type is distinct from 'application/json'
    or release_manifest#>>array['release','artifact_hashes',asset.path]
      is distinct from asset.sha256
    or not exists(select 1 from storage.objects object
      where object.bucket_id='scryglass-public' and object.name=asset.storage_path
        and object.user_metadata->>'sha256'=asset.sha256
        and (object.user_metadata->>'bytes')::bigint=asset.bytes
        and object.user_metadata->>'content_type'=asset.content_type)
    or not exists(select 1 from pg_catalog.jsonb_array_elements(release_manifest->'files') file
      where file->>'path'=asset.path and (file->>'bytes')::bigint=asset.bytes
        and file->>'sha256'=asset.sha256));
  if invalid_assets<>0 then
    raise exception 'Scryglass release asset integrity is invalid';
  end if;
end;
$$;

revoke all on function public.assert_scryglass_public_release_integrity(text)
  from public, anon, authenticated, service_role;
grant execute on function public.assert_scryglass_public_release_integrity(text)
  to scryglass_release_transition_owner;

create or replace function public.scryglass_query_descriptive_authority(p_release_id text)
returns jsonb
language plpgsql
stable
security definer
set search_path = ''
set statement_timeout = '5s'
as $$
declare
  manifest_authority jsonb;
  authority jsonb;
begin
  select release.manifest->'draft_authority' into manifest_authority
  from public.scryglass_public_releases release
  where release.release_id=p_release_id and release.status='active';
  authority := case
    when manifest_authority->>'status'='descriptive' then manifest_authority
    when manifest_authority->>'status'='promoted'
      and manifest_authority->>'authority'='promoted'
      and manifest_authority->>'estimand'='prematch_map_win_probability_with_controlled_draft_intervention'
      and coalesce((manifest_authority->>'probability_authority')::boolean,false)
      and coalesce((manifest_authority->>'recommendation_authority')::boolean,false)
      and coalesce((manifest_authority->>'betting_authority')::boolean,true) is false
      then manifest_authority->'descriptive_authority'
    else null end;
  if authority->>'schema_version'='scryglass:draft-authority:v1'
    and authority->>'status'='descriptive' and authority->>'authority'='descriptive'
    and authority->>'estimand'='composition_only' and authority->>'release_id'=p_release_id
    and pg_catalog.octet_length(authority->>'model_version') between 1 and 100
    and authority->>'artifact_sha256'~'^[0-9a-f]{64}$'
    and authority->>'receipt_sha256'~'^[0-9a-f]{64}$'
    and authority->>'issued_utc'~'^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}([.][0-9]{1,6})?Z$'
    and (authority->>'issued_utc')::timestamptz is not null
    and coalesce((authority->>'probability_authority')::boolean,true) is false
    and coalesce((authority->>'recommendation_authority')::boolean,true) is false
    and coalesce((authority->>'betting_authority')::boolean,true) is false
  then return authority; end if;
  return null;
exception when invalid_datetime_format or datetime_field_overflow then return null;
end;
$$;

alter function scryglass_private.get_scryglass_active_release(text)
  rename to get_scryglass_active_release_before_promotion;

create function scryglass_private.get_scryglass_active_release(p_release_id text default null)
returns table(release_id text,status text,manifest jsonb)
language plpgsql
stable
security definer
set search_path = ''
set statement_timeout = '5s'
as $$
declare
  row_value record;
  promoted jsonb;
begin
  select * into row_value
  from scryglass_private.get_scryglass_active_release_before_promotion(p_release_id);
  if not found then return; end if;
  select release.manifest->'draft_authority' into promoted
  from public.scryglass_public_releases release
  where release.release_id=row_value.release_id and release.status='active'
    and release.manifest#>>'{draft_authority,status}'='promoted';
  if promoted is not null then
    row_value.manifest:=pg_catalog.jsonb_set(row_value.manifest,'{draft_authority}',
      pg_catalog.jsonb_build_object(
        'schema_version','scryglass:draft-authority:v1','status','promoted',
        'authority','promoted','release_id',row_value.release_id,
        'model_version',promoted->>'model_version','artifact_sha256',promoted->>'artifact_sha256',
        'receipt_sha256',promoted->>'receipt_sha256','issued_utc',promoted->>'issued_utc',
        'estimand','prematch_map_win_probability_with_controlled_draft_intervention',
        'probability_authority',true,'recommendation_authority',true,
        'betting_authority',false,'reason',null));
  end if;
  return query select row_value.release_id,row_value.status,row_value.manifest;
end;
$$;

alter function scryglass_private.get_scryglass_active_asset(text,text)
  rename to get_scryglass_active_asset_before_promotion;

create function scryglass_private.get_scryglass_active_asset(p_release_id text,p_path text)
returns table(release_id text,path text,storage_path text,bytes bigint,sha256 text,content_type text)
language plpgsql
stable
security definer
set search_path = ''
set statement_timeout = '5s'
as $$
declare authority_status text;
begin
  select release.manifest#>>'{draft_authority,status}' into authority_status
  from public.scryglass_public_releases release
  where release.release_id=p_release_id and release.status='active';
  if p_path='features/promoted_draft_results.json' and authority_status<>'promoted' then return; end if;
  if p_path='features/draft_records.json'
    and public.scryglass_query_descriptive_authority(p_release_id) is null then return; end if;
  return query select asset.release_id,asset.path,asset.storage_path,asset.bytes,
    asset.sha256,asset.content_type
  from scryglass_private.get_scryglass_active_asset_before_promotion(p_release_id,p_path) asset;
end;
$$;

alter function scryglass_private.is_active_scryglass_storage_object(text)
  rename to is_active_scryglass_storage_object_before_promotion;

create function scryglass_private.is_active_scryglass_storage_object(p_storage_path text)
returns boolean
language plpgsql
stable
security definer
set search_path = ''
set statement_timeout = '5s'
as $$
declare release_id text:=pg_catalog.split_part(p_storage_path,'/',1);
declare asset_path text:=pg_catalog.substr(p_storage_path,pg_catalog.length(release_id)+2);
declare authority_status text;
begin
  select release.manifest#>>'{draft_authority,status}' into authority_status
  from public.scryglass_public_releases release
  where release.release_id=release_id and release.status='active';
  if asset_path='features/promoted_draft_results.json' and authority_status<>'promoted' then return false; end if;
  if asset_path='features/draft_records.json'
    and public.scryglass_query_descriptive_authority(release_id) is null then return false; end if;
  return scryglass_private.is_active_scryglass_storage_object_before_promotion(p_storage_path);
end;
$$;

revoke all on function scryglass_private.get_scryglass_active_release_before_promotion(text) from public,anon,authenticated,service_role;
revoke all on function scryglass_private.get_scryglass_active_asset_before_promotion(text,text) from public,anon,authenticated,service_role;
revoke all on function scryglass_private.is_active_scryglass_storage_object_before_promotion(text) from public,anon,authenticated,service_role;
revoke all on function scryglass_private.get_scryglass_active_release(text) from public,service_role;
revoke all on function scryglass_private.get_scryglass_active_asset(text,text) from public,service_role;
revoke all on function scryglass_private.is_active_scryglass_storage_object(text) from public,anon,authenticated,service_role;
grant execute on function scryglass_private.get_scryglass_active_release(text) to anon,authenticated;
grant execute on function scryglass_private.get_scryglass_active_asset(text,text) to anon,authenticated;
grant execute on function scryglass_private.is_active_scryglass_storage_object(text) to supabase_storage_admin;
