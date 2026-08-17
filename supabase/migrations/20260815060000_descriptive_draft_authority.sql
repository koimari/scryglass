-- Allow the non-predictive, composition-only Draft Score receipt through the
-- final release integrity gate. Predictive draft authority remains closed.

create or replace function public.scryglass_json_has_draft_fields(p_value jsonb)
returns boolean
language plpgsql
immutable
security invoker
set search_path = ''
as $$
declare
  item record;
  child jsonb;
begin
  if p_value is null then
    return false;
  end if;
  if pg_catalog.jsonb_typeof(p_value) = 'object' then
    for item in
      select entry.key,entry.value
      from pg_catalog.jsonb_each(p_value) as entry(key,value)
    loop
      if pg_catalog.lower(item.key) = any(array[
        'authority_receipt_sha256', 'best_available', 'draft_authority',
        'draft_contribution', 'draft_edge', 'draft_pool',
        'draft_probability', 'draft_score', 'draft_win_share',
        'r9e', 'r9e_state_space', 'development_composite',
        'match_probability', 'match_win_expectation', 'team_rating',
        'player_rating', 'elo', 'mu_diff', 'sigma_pair', 'momentum',
        'gold', 'objectives', 'recommendation', 'betting', 'phase_curve',
        'live_state', 'odds', 'ev'
      ]) or public.scryglass_json_has_draft_fields(item.value) then
        return true;
      end if;
    end loop;
  elsif pg_catalog.jsonb_typeof(p_value) = 'array' then
    for child in
      select entry.element
      from pg_catalog.jsonb_array_elements(p_value) as entry(element)
    loop
      if public.scryglass_json_has_draft_fields(child) then
        return true;
      end if;
    end loop;
  end if;
  return false;
end;
$$;

create or replace function public.assert_scryglass_public_release_integrity(
  p_release_id text
)
returns void
language plpgsql
security definer
set search_path = ''
as $$
declare
  release_manifest jsonb;
  draft_authority_status text;
  draft_issued_at timestamptz;
  required_assets constant text[] := array[
    'features/ratings_snapshot.json',
    'features/player_ratings_snapshot.json',
    'features/team_records.json',
    'features/team_weekly_ranks.json',
    'features/player_records.json',
    'features/player_champion_records.json',
    'features/profile_records.json',
    'features/match_index.json',
    'features/match_records_2025_q1.json',
    'features/match_records_2025_q2.json',
    'features/match_records_2025_q3.json',
    'features/match_records_2025_q4.json',
    'features/match_records_2026_q1.json',
    'features/match_records_2026_q2.json',
    'features/match_records_2026_q3.json',
    'features/match_records_2026_q4.json',
    'features/player_weekly_ranks.json',
    'features/player_metadata.json',
    'rankings/tierlists.json',
    'rankings/tierlists-latest.json'
  ];
  present_assets integer;
  manifest_files integer;
  manifest_paths integer;
  manifest_hashes integer;
  release_assets integer;
  invalid_assets integer;
  draft_asset_count integer;
begin
  if pg_catalog.octet_length(coalesce(p_release_id, '')) not between 1 and 30
     or p_release_id !~ '^v[0-9]{4}\.[0-9]{2}\.[0-9]{2}\.[0-9]{6}$'
  then
    raise exception 'Scryglass release identity is invalid';
  end if;

  select release.manifest
  into release_manifest
  from public.scryglass_public_releases release
  where release.release_id = p_release_id
    and release.status in ('staging', 'active', 'superseded');

  if release_manifest is null then
    raise exception 'Scryglass release is unavailable';
  end if;

  draft_authority_status := release_manifest #>> '{draft_authority,status}';
  if release_manifest ->> 'pack_id' <> p_release_id
     or release_manifest #>> '{release,release_id}' <> p_release_id
     or pg_catalog.jsonb_typeof(release_manifest -> 'files') <> 'array'
     or pg_catalog.jsonb_typeof(
       release_manifest #> '{release,artifact_hashes}'
     ) <> 'object'
     or release_manifest #>> '{draft_authority,schema_version}'
       is distinct from 'scryglass:draft-authority:v1'
     or draft_authority_status is null
     or draft_authority_status not in ('unavailable', 'descriptive')
     or release_manifest #>> '{draft_authority,release_id}'
       is distinct from p_release_id
  then
    raise exception 'Scryglass release binding is invalid';
  end if;

  if draft_authority_status = 'descriptive'
     and (
       release_manifest #>> '{draft_authority,authority}'
         is distinct from 'descriptive'
       or release_manifest #>> '{draft_authority,estimand}'
         is distinct from 'composition_only'
       or pg_catalog.octet_length(
         release_manifest #>> '{draft_authority,model_version}'
       ) not between 1 and 100
       or release_manifest #>> '{draft_authority,artifact_sha256}' is null
       or release_manifest #>> '{draft_authority,artifact_sha256}'
         !~ '^[0-9a-f]{64}$'
       or release_manifest #>> '{draft_authority,receipt_sha256}' is null
       or release_manifest #>> '{draft_authority,receipt_sha256}'
         !~ '^[0-9a-f]{64}$'
       or release_manifest #>> '{draft_authority,issued_utc}' is null
       or release_manifest #>> '{draft_authority,issued_utc}'
         !~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}([.][0-9]{1,6})?Z$'
       or coalesce((release_manifest #>> '{draft_authority,probability_authority}')::boolean, true)
       or coalesce((release_manifest #>> '{draft_authority,recommendation_authority}')::boolean, true)
       or coalesce((release_manifest #>> '{draft_authority,betting_authority}')::boolean, true)
     )
  then
    raise exception 'Scryglass descriptive Draft Score authority is invalid';
  end if;

  if draft_authority_status = 'descriptive' then
    begin
      draft_issued_at := (
        release_manifest #>> '{draft_authority,issued_utc}'
      )::timestamptz;
    exception when others then
      raise exception 'Scryglass descriptive Draft Score authority timestamp is invalid';
    end;
  end if;

  select pg_catalog.count(*)
  into draft_asset_count
  from public.scryglass_public_assets asset
  where asset.release_id = p_release_id
    and asset.path = 'features/draft_records.json';
  if draft_authority_status = 'unavailable' and draft_asset_count <> 0 then
    raise exception 'Scryglass release contains unavailable draft data';
  end if;
  if draft_authority_status = 'descriptive' and draft_asset_count <> 1 then
    raise exception 'Scryglass descriptive Draft Score asset is missing';
  end if;

  select pg_catalog.count(*)
  into present_assets
  from public.scryglass_public_assets asset
  where asset.release_id = p_release_id
    and asset.path = any(required_assets);
  if present_assets <> pg_catalog.cardinality(required_assets) then
    raise exception 'Scryglass release required asset inventory is incomplete';
  end if;

  select pg_catalog.jsonb_array_length(release_manifest -> 'files'),
         pg_catalog.count(distinct file ->> 'path')
  into manifest_files, manifest_paths
  from pg_catalog.jsonb_array_elements(release_manifest -> 'files') file;
  select pg_catalog.count(*)
  into manifest_hashes
  from pg_catalog.jsonb_object_keys(
    release_manifest #> '{release,artifact_hashes}'
  );
  select pg_catalog.count(*)
  into release_assets
  from public.scryglass_public_assets asset
  where asset.release_id = p_release_id;
  if manifest_files <> manifest_paths
     or manifest_files <> manifest_hashes
     or manifest_files <> release_assets
  then
    raise exception 'Scryglass release manifest inventory is not exact';
  end if;

  select pg_catalog.count(*)
  into invalid_assets
  from public.scryglass_public_assets asset
  where asset.release_id = p_release_id
    and (
      asset.body is not null
      or asset.storage_path is distinct from asset.release_id || '/' || asset.path
      or asset.content_type is distinct from 'application/json'
      or release_manifest #>> array[
        'release', 'artifact_hashes', asset.path
      ] is distinct from asset.sha256
      or not exists (
        select 1
        from storage.objects object
        where object.bucket_id = 'scryglass-public'
          and object.name = asset.storage_path
          and object.user_metadata ->> 'sha256' = asset.sha256
          and (object.user_metadata ->> 'bytes')::bigint = asset.bytes
          and object.user_metadata ->> 'content_type' = asset.content_type
      )
      or not exists (
        select 1
        from pg_catalog.jsonb_array_elements(release_manifest -> 'files') file
        where file ->> 'path' = asset.path
          and (file ->> 'bytes')::bigint = asset.bytes
          and file ->> 'sha256' = asset.sha256
      )
    );
  if invalid_assets <> 0 then
    raise exception 'Scryglass release asset integrity is invalid';
  end if;
end;
$$;

alter function public.assert_scryglass_public_release_integrity(text)
  security definer;
grant execute on function public.assert_scryglass_public_release_integrity(text)
  to scryglass_release_transition_owner;
