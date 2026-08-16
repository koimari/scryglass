-- Permit one strict descriptive Draft Score subset through bounded query RPCs.
-- All values remain bound to the active release authority. Probability and
-- match-strength fields remain outside this contract.

create or replace function public.scryglass_descriptive_signal_is_valid(
  p_value jsonb
)
returns boolean
language plpgsql
immutable
security invoker
set search_path = ''
as $$
declare
  side_name text;
  side_value jsonb;
  component_name text;
  component_sum numeric;
  pick_value jsonb;
begin
  if pg_catalog.jsonb_typeof(p_value) <> 'object'
     or (select pg_catalog.array_agg(key order by key)
         from pg_catalog.jsonb_object_keys(p_value) key) is distinct from array[
       'artifact_sha256','authority','authority_receipt_sha256','blue',
       'edge_components','estimand','model_version','note','picks','red',
       'release_id','schema_version','status'
     ]::text[]
     or p_value ->> 'schema_version'
       is distinct from 'scryglass:draft-descriptive-signal:v1'
     or p_value ->> 'status' is distinct from 'available'
     or p_value ->> 'authority' is distinct from 'descriptive'
     or p_value ->> 'estimand' is distinct from 'composition_only'
     or coalesce(p_value ->> 'release_id','')
       !~ '^v[0-9]{4}\.[0-9]{2}\.[0-9]{2}\.[0-9]{6}$'
     or pg_catalog.octet_length(coalesce(p_value ->> 'model_version',''))
       not between 1 and 100
     or coalesce(p_value ->> 'artifact_sha256','') !~ '^[0-9a-f]{64}$'
     or coalesce(p_value ->> 'authority_receipt_sha256','')
       !~ '^[0-9a-f]{64}$'
     or p_value ->> 'note'
       is distinct from 'Descriptive composition score in model units.'
     or pg_catalog.jsonb_typeof(p_value -> 'picks') <> 'array'
     or pg_catalog.jsonb_array_length(p_value -> 'picks') <> 10
  then
    return false;
  end if;

  foreach side_name in array array['blue','red']
  loop
    side_value := p_value -> side_name;
    if pg_catalog.jsonb_typeof(side_value) <> 'object'
       or (select pg_catalog.array_agg(key order by key)
           from pg_catalog.jsonb_object_keys(side_value) key)
         is distinct from array['components','prior_role_games','signal']::text[]
       or pg_catalog.jsonb_typeof(side_value -> 'signal') <> 'number'
       or pg_catalog.jsonb_typeof(side_value -> 'prior_role_games') <> 'number'
       or coalesce(side_value ->> 'prior_role_games','') !~ '^[0-9]+$'
       or pg_catalog.jsonb_typeof(side_value -> 'components') <> 'object'
       or (select pg_catalog.array_agg(key order by key)
           from pg_catalog.jsonb_object_keys(side_value -> 'components') key)
         is distinct from array[
           'ally_synergy','archetype_interactions','base','enemy_counter','same_role'
         ]::text[]
       or exists (
         select 1
         from pg_catalog.jsonb_each(side_value -> 'components') component
         where pg_catalog.jsonb_typeof(component.value) <> 'number'
       )
    then
      return false;
    end if;
    select pg_catalog.sum(component.value::text::numeric)
    into component_sum
    from pg_catalog.jsonb_each(side_value -> 'components') component;
    if pg_catalog.abs(component_sum - (side_value ->> 'signal')::numeric) > 0.00001
    then
      return false;
    end if;
  end loop;

  if pg_catalog.jsonb_typeof(p_value -> 'edge_components') <> 'object'
     or (select pg_catalog.array_agg(key order by key)
         from pg_catalog.jsonb_object_keys(p_value -> 'edge_components') key)
       is distinct from array[
         'ally_synergy','archetype_interactions','base','enemy_counter','same_role','total'
       ]::text[]
     or exists (
       select 1
       from pg_catalog.jsonb_each(p_value -> 'edge_components') component
       where pg_catalog.jsonb_typeof(component.value) <> 'number'
     )
  then
    return false;
  end if;
  foreach component_name in array array[
    'base','archetype_interactions','ally_synergy','enemy_counter','same_role'
  ]
  loop
    if pg_catalog.abs(
      (p_value #>> array['edge_components',component_name])::numeric
      - (
        (p_value #>> array['blue','components',component_name])::numeric
        - (p_value #>> array['red','components',component_name])::numeric
      )
    ) > 0.00001 then
      return false;
    end if;
  end loop;
  select pg_catalog.sum(component.value::text::numeric)
  into component_sum
  from pg_catalog.jsonb_each(p_value -> 'edge_components') component
  where component.key <> 'total';
  if pg_catalog.abs(
    component_sum - (p_value #>> '{edge_components,total}')::numeric
  ) > 0.00001 then
    return false;
  end if;

  for pick_value in
    select item.value from pg_catalog.jsonb_array_elements(p_value -> 'picks') item(value)
  loop
    if pg_catalog.jsonb_typeof(pick_value) <> 'object'
       or (select pg_catalog.array_agg(key order by key)
           from pg_catalog.jsonb_object_keys(pick_value) key)
         is distinct from array[
           'champion','contribution','evidence_status','prior_role_games','role','side'
         ]::text[]
       or coalesce(pick_value ->> 'side','') not in ('Blue','Red')
       or coalesce(pick_value ->> 'role','') not in ('top','jng','mid','bot','sup')
       or pg_catalog.octet_length(coalesce(pick_value ->> 'champion',''))
         not between 1 and 100
       or pg_catalog.jsonb_typeof(pick_value -> 'contribution') <> 'number'
       or pg_catalog.jsonb_typeof(pick_value -> 'prior_role_games') <> 'number'
       or coalesce(pick_value ->> 'prior_role_games','') !~ '^[0-9]+$'
       or coalesce(pick_value ->> 'evidence_status','')
         <> 'available'
    then
      return false;
    end if;
  end loop;
  if (select pg_catalog.count(distinct concat_ws(':',item.value ->> 'side',item.value ->> 'role'))
      from pg_catalog.jsonb_array_elements(p_value -> 'picks') item(value)) <> 10
     or (select pg_catalog.count(distinct pg_catalog.lower(item.value ->> 'champion'))
         from pg_catalog.jsonb_array_elements(p_value -> 'picks') item(value)) <> 10
  then
    return false;
  end if;
  return true;
exception when others then
  return false;
end;
$$;

create or replace function public.scryglass_descriptive_summary_is_valid(
  p_value jsonb
)
returns boolean
language plpgsql
immutable
security invoker
set search_path = ''
as $$
declare
  keys text[];
begin
  if pg_catalog.jsonb_typeof(p_value) <> 'object' then
    return false;
  end if;
  select pg_catalog.array_agg(key order by key)
  into keys
  from pg_catalog.jsonb_object_keys(p_value) key;
  if keys is distinct from array[
       'artifact_sha256','authority','authority_receipt_sha256','ban_coverage',
       'best_available_rate','estimand','games','model_version','pick_contribution',
       'pool_definition','release_id','schema_version','scope'
     ]::text[]
     and keys is distinct from array[
       'artifact_sha256','authority','authority_receipt_sha256','draft_edge',
       'estimand','games','model_version','positive_edge_rate','release_id',
       'schema_version','scope'
     ]::text[]
  then
    return false;
  end if;
  if p_value ->> 'schema_version'
       is distinct from 'scryglass:draft-descriptive-summary:v1'
     or p_value ->> 'authority' is distinct from 'descriptive'
     or p_value ->> 'estimand' is distinct from 'composition_only'
     or p_value ->> 'scope' is distinct from 'whole_archive'
     or coalesce(p_value ->> 'release_id','')
       !~ '^v[0-9]{4}\.[0-9]{2}\.[0-9]{2}\.[0-9]{6}$'
     or pg_catalog.octet_length(coalesce(p_value ->> 'model_version',''))
       not between 1 and 100
     or coalesce(p_value ->> 'artifact_sha256','') !~ '^[0-9a-f]{64}$'
     or coalesce(p_value ->> 'authority_receipt_sha256','')
       !~ '^[0-9a-f]{64}$'
     or pg_catalog.jsonb_typeof(p_value -> 'games') <> 'number'
     or coalesce(p_value ->> 'games','') !~ '^[1-9][0-9]*$'
  then
    return false;
  end if;
  if p_value ? 'pick_contribution' then
    return pg_catalog.jsonb_typeof(p_value -> 'pick_contribution') = 'number'
      and pg_catalog.jsonb_typeof(p_value -> 'best_available_rate') = 'number'
      and (p_value ->> 'best_available_rate')::numeric between 0 and 1
      and pg_catalog.jsonb_typeof(p_value -> 'ban_coverage') = 'number'
      and (p_value ->> 'ban_coverage')::numeric between 0 and 1
      and coalesce(p_value ->> 'pool_definition','')
        = 'Best available champion in the published unbanned role pool';
  end if;
  return pg_catalog.jsonb_typeof(p_value -> 'draft_edge') = 'number'
    and pg_catalog.jsonb_typeof(p_value -> 'positive_edge_rate') = 'number'
    and (p_value ->> 'positive_edge_rate')::numeric between 0 and 1;
exception when others then
  return false;
end;
$$;

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
  normalized_key text;
begin
  if p_value is null then
    return false;
  end if;
  if pg_catalog.jsonb_typeof(p_value) = 'object' then
    for item in
      select entry.key,entry.value
      from pg_catalog.jsonb_each(p_value) entry(key,value)
    loop
      normalized_key := pg_catalog.lower(item.key);
      if normalized_key = 'draft_contribution' then
        if public.scryglass_descriptive_signal_is_valid(item.value) is not true then
          return true;
        end if;
      elsif normalized_key = 'draft_metric' then
        if public.scryglass_descriptive_summary_is_valid(item.value) is not true then
          return true;
        end if;
      elsif normalized_key = any(array[
        'average_win_share','best_available','betting','development_composite',
        'draft_authority','draft_edge','draft_pool','draft_probability','draft_score',
        'draft_win_share','elo','ev','expected_value','fair_odds','gold','live_state',
        'match_probability','match_win_expectation','momentum','mu_diff','objectives',
        'odds','p_blue','p_red','phase_curve','player_elo','player_rating',
        'probability','r9e','r9e_state_space','rating_uncertainty','recommendation',
        'sigma_pair','strength','team_elo','team_rating','win_probability'
      ]) or normalized_key like 'r9e\_%' escape '\'
         or normalized_key like 'control\_%' escape '\'
         or normalized_key like 'strength\_%' escape '\'
         or normalized_key like 'phase\_%' escape '\'
         or normalized_key like 'live\_%' escape '\'
         or public.scryglass_json_has_draft_fields(item.value)
      then
        return true;
      end if;
    end loop;
  elsif pg_catalog.jsonb_typeof(p_value) = 'array' then
    for child in
      select entry.element
      from pg_catalog.jsonb_array_elements(p_value) entry(element)
    loop
      if public.scryglass_json_has_draft_fields(child) then
        return true;
      end if;
    end loop;
  end if;
  return false;
end;
$$;

create or replace function public.scryglass_query_descriptive_authority(
  p_release_id text
)
returns jsonb
language plpgsql
stable
security definer
set search_path = ''
set statement_timeout = '5s'
as $$
declare
  authority jsonb;
begin
  select release.manifest -> 'draft_authority'
  into authority
  from public.scryglass_public_releases release
  where release.release_id = p_release_id
    and release.status = 'active'
    and release.manifest #>> '{draft_authority,schema_version}'
      = 'scryglass:draft-authority:v1'
    and release.manifest #>> '{draft_authority,status}' = 'descriptive'
    and release.manifest #>> '{draft_authority,authority}' = 'descriptive'
    and release.manifest #>> '{draft_authority,estimand}' = 'composition_only'
    and release.manifest #>> '{draft_authority,release_id}' = p_release_id
    and pg_catalog.octet_length(
      release.manifest #>> '{draft_authority,model_version}'
    ) between 1 and 100
    and release.manifest #>> '{draft_authority,artifact_sha256}'
      ~ '^[0-9a-f]{64}$'
    and release.manifest #>> '{draft_authority,receipt_sha256}'
      ~ '^[0-9a-f]{64}$'
    and release.manifest #>> '{draft_authority,issued_utc}'
      ~ '^[0-9]{4}-(0[1-9]|1[0-2])-([0-2][0-9]|3[01])T([01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]([.][0-9]{1,6})?Z$'
    and (release.manifest #>> '{draft_authority,issued_utc}')::pg_catalog.timestamptz
      is not null
    and coalesce(
      (release.manifest #>> '{draft_authority,probability_authority}')::boolean,
      true
    ) is false
    and coalesce(
      (release.manifest #>> '{draft_authority,recommendation_authority}')::boolean,
      true
    ) is false
    and coalesce(
      (release.manifest #>> '{draft_authority,betting_authority}')::boolean,
      true
    ) is false
  limit 1;
  return authority;
exception
  when invalid_datetime_format or datetime_field_overflow then
    return null;
end;
$$;

create or replace function public.scryglass_strip_query_draft_fields(
  p_value jsonb
)
returns jsonb
language plpgsql
immutable
security invoker
set search_path = ''
as $$
declare
  item record;
  result jsonb := case pg_catalog.jsonb_typeof(p_value)
    when 'object' then '{}'::jsonb
    when 'array' then '[]'::jsonb
    else p_value
  end;
  child jsonb;
begin
  if p_value is null then
    return null;
  end if;
  if pg_catalog.jsonb_typeof(p_value) = 'object' then
    for item in select entry.key,entry.value from pg_catalog.jsonb_each(p_value) entry(key,value)
    loop
      if pg_catalog.lower(item.key) = any(array[
        'authority','authority_receipt_sha256','average_win_share','best_available',
        'betting','development_composite','draft_authority','draft_contribution',
        'draft_edge','draft_metric','draft_pool','draft_probability','draft_score',
        'draft_win_share','elo','ev','expected_value','fair_odds','live_state',
        'match_probability','match_win_expectation','momentum','mu_diff','odds',
        'p_blue','p_red','phase_curve','player_elo','player_rating','probability',
        'r9e','r9e_state_space','rating_uncertainty','recommendation','sigma_pair',
        'strength','team_elo','team_rating','win_probability'
      ]) then
        continue;
      end if;
      result := result || pg_catalog.jsonb_build_object(
        item.key,
        public.scryglass_strip_query_draft_fields(item.value)
      );
    end loop;
    return result;
  end if;
  if pg_catalog.jsonb_typeof(p_value) = 'array' then
    for child in select entry.element from pg_catalog.jsonb_array_elements(p_value) entry(element)
    loop
      result := result || pg_catalog.jsonb_build_array(
        public.scryglass_strip_query_draft_fields(child)
      );
    end loop;
  end if;
  return result;
end;
$$;

create or replace function public.scryglass_json_has_unbound_descriptive_draft(
  p_value jsonb,
  p_authority jsonb
)
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
    for item in select entry.key,entry.value from pg_catalog.jsonb_each(p_value) entry(key,value)
    loop
      if pg_catalog.lower(item.key) in ('draft_contribution','draft_metric') then
        if item.value ->> 'authority' is distinct from 'descriptive'
           or item.value ->> 'estimand' is distinct from 'composition_only'
           or item.value ->> 'release_id' is distinct from p_authority ->> 'release_id'
           or item.value ->> 'model_version' is distinct from p_authority ->> 'model_version'
           or item.value ->> 'artifact_sha256' is distinct from p_authority ->> 'artifact_sha256'
           or item.value ->> 'authority_receipt_sha256'
             is distinct from p_authority ->> 'receipt_sha256'
        then
          return true;
        end if;
      elsif public.scryglass_json_has_unbound_descriptive_draft(item.value,p_authority) then
        return true;
      end if;
    end loop;
  elsif pg_catalog.jsonb_typeof(p_value) = 'array' then
    for child in select entry.element from pg_catalog.jsonb_array_elements(p_value) entry(element)
    loop
      if public.scryglass_json_has_unbound_descriptive_draft(child,p_authority) then
        return true;
      end if;
    end loop;
  end if;
  return false;
end;
$$;

create or replace function public.scryglass_bounded_query_result(value jsonb)
returns jsonb
language plpgsql
stable
security definer
set search_path = ''
set statement_timeout = '5s'
as $$
declare
  release_id text := value ->> 'release_id';
  authority jsonb := public.scryglass_query_descriptive_authority(release_id);
  result jsonb := value;
  profile_metric jsonb;
begin
  if result is null then
    return null;
  end if;
  if authority is null then
    result := public.scryglass_strip_query_draft_fields(result);
  else
    if public.scryglass_json_has_unbound_descriptive_draft(result,authority) then
      raise exception 'Scryglass public query Draft data is not bound to the active release';
    end if;
    if result #> '{row,payload,draft_metric}' is not null
       and not (result ? 'draft_metric')
    then
      profile_metric := result #> '{row,payload,draft_metric}';
    end if;
    if public.scryglass_json_has_draft_fields(result) then
      raise exception 'Scryglass public query response exceeds its authority or byte budget';
    end if;
    if profile_metric ? 'pick_contribution' then
      result := pg_catalog.jsonb_set(
        result,
        '{draft_metric}',
        pg_catalog.jsonb_build_object(
          'best_available_rate',profile_metric -> 'best_available_rate',
          'games',profile_metric -> 'games',
          'pick_contribution',profile_metric -> 'pick_contribution',
          'pool_definition',profile_metric -> 'pool_definition',
          'ban_coverage',profile_metric -> 'ban_coverage',
          'scope',profile_metric -> 'scope'
        ),
        true
      );
    elsif profile_metric ? 'draft_edge' then
      result := pg_catalog.jsonb_set(
        result,
        '{draft_metric}',
        pg_catalog.jsonb_build_object(
          'draft_edge',profile_metric -> 'draft_edge',
          'games',profile_metric -> 'games',
          'positive_edge_rate',profile_metric -> 'positive_edge_rate',
          'scope',profile_metric -> 'scope'
        ),
        true
      );
    end if;
    if result::text like '%"draft_contribution"%'
       or result::text like '%"draft_metric"%'
    then
      result := pg_catalog.jsonb_set(result,'{authority}','"descriptive"'::jsonb,true);
    end if;
  end if;
  if (authority is null and public.scryglass_json_has_draft_fields(result))
     or pg_catalog.octet_length(result::text) > 500000
  then
    raise exception 'Scryglass public query response exceeds its authority or byte budget';
  end if;
  return result;
end;
$$;

-- The compatibility manifest exposes the descriptive receipt and its backing
-- asset only when both are bound to the active release. All other manifest
-- fields remain the fixed public projection from the bounded query migration.
create or replace function public.get_scryglass_active_release(
  p_release_id text default null
)
returns table (release_id text, status text, manifest jsonb)
language sql
stable
security definer
set search_path = ''
set statement_timeout = '5s'
as $$
  select
    candidate.release_id,
    candidate.status,
    pg_catalog.jsonb_strip_nulls(
      pg_catalog.jsonb_build_object(
        'pack_id',candidate.release_id,
        'schema_version','2.0.0',
        'created_utc',pg_catalog.to_jsonb(candidate.created_at),
        'filters',pg_catalog.jsonb_build_object(
          'years',coalesce((select pg_catalog.jsonb_agg(item.value order by item.ordinality)
            from pg_catalog.jsonb_array_elements(case
              when pg_catalog.jsonb_typeof(candidate.manifest#>'{filters,years}')='array'
                then candidate.manifest#>'{filters,years}' else '[]'::jsonb end)
              with ordinality item(value,ordinality)
            where pg_catalog.jsonb_typeof(item.value)='number' and item.ordinality<=10),'[]'::jsonb),
          'leagues','all_in_year_window'
        ),
        'attribution','Completed professional match data from approved public sources.',
        'excluded',pg_catalog.jsonb_build_array('raw game rows','research studies','predictive artifacts'),
        'base_url',null,
        'data_backend','supabase',
        'tier',pg_catalog.jsonb_build_object(
          'status','available',
          'as_of',pg_catalog.to_jsonb(candidate.source_as_of)
        ),
        'draft_authority',case when authority.value is not null then
          pg_catalog.jsonb_build_object(
            'schema_version','scryglass:draft-authority:v1',
            'status','descriptive',
            'authority','descriptive',
            'estimand','composition_only',
            'release_id',candidate.release_id,
            'model_version',authority.value->>'model_version',
            'artifact_sha256',authority.value->>'artifact_sha256',
            'receipt_sha256',authority.value->>'receipt_sha256',
            'issued_utc',authority.value->>'issued_utc',
            'probability_authority',false,
            'recommendation_authority',false,
            'betting_authority',false
          ) else null end,
        'query_api',case when
          candidate.manifest#>>'{query_api,schema_version}'='scryglass:query-api:v1'
          and candidate.manifest#>>'{query_api,status}'='available'
          and pg_catalog.jsonb_typeof(candidate.manifest#>'{query_api,datasets}')='object'
          and (select pg_catalog.count(*) from public.scryglass_public_query_receipts receipt
            where receipt.release_id=candidate.release_id)=12
          and (select pg_catalog.count(*) from pg_catalog.jsonb_object_keys(
            candidate.manifest#>'{query_api,datasets}'
          ))=12
        then pg_catalog.jsonb_build_object(
          'schema_version','scryglass:query-api:v1','status','available',
          'datasets',(select pg_catalog.jsonb_object_agg(receipt.dataset,
            pg_catalog.jsonb_build_object('rows',receipt.row_count)
            order by receipt.dataset)
            from public.scryglass_public_query_receipts receipt
            where receipt.release_id=candidate.release_id)
        ) else null end,
        'ratings',pg_catalog.jsonb_build_object(
          'source_as_of',pg_catalog.to_jsonb(candidate.source_as_of),
          'claim_ceiling','descriptive'
        ),
        'total_bytes',(select pg_catalog.to_jsonb(coalesce(pg_catalog.sum(asset.bytes),0))
          from public.scryglass_public_assets asset
          where asset.release_id=candidate.release_id
            and (asset.path<>'features/draft_records.json' or authority.value is not null)),
        'total_files',(select pg_catalog.to_jsonb(pg_catalog.count(*))
          from public.scryglass_public_assets asset
          where asset.release_id=candidate.release_id
            and (asset.path<>'features/draft_records.json' or authority.value is not null)),
        'files',coalesce((select pg_catalog.jsonb_agg(pg_catalog.jsonb_build_object(
          'path',asset.path,'bytes',asset.bytes,'sha256',asset.sha256) order by asset.path)
          from public.scryglass_public_assets asset
          where asset.release_id=candidate.release_id
            and (asset.path<>'features/draft_records.json' or authority.value is not null)),'[]'::jsonb),
        'release',pg_catalog.jsonb_build_object(
          'release_id',candidate.release_id,
          'tier_list_version',candidate.manifest#>'{release,tier_list_version}',
          'artifact_hashes',coalesce((select pg_catalog.jsonb_object_agg(
            asset.path,asset.sha256 order by asset.path)
            from public.scryglass_public_assets asset
            where asset.release_id=candidate.release_id
              and (asset.path<>'features/draft_records.json' or authority.value is not null)),'{}'::jsonb)
        )
      )
    ) manifest
  from public.scryglass_public_releases candidate
  left join lateral (
    select public.scryglass_query_descriptive_authority(candidate.release_id) value
  ) authority on true
  where candidate.status='active'
    and (p_release_id is null or (
      pg_catalog.octet_length(p_release_id) between 1 and 30
      and p_release_id~'^v[0-9]{4}\.[0-9]{2}\.[0-9]{2}\.[0-9]{6}$'
      and candidate.release_id=p_release_id
    ))
$$;

create or replace function public.get_scryglass_active_asset(
  p_release_id text,
  p_path text
)
returns table (
  release_id text,
  path text,
  storage_path text,
  bytes bigint,
  sha256 text,
  content_type text
)
language sql
stable
security definer
set search_path = ''
set statement_timeout = '5s'
as $$
  select asset.release_id,asset.path,asset.storage_path,asset.bytes,
    asset.sha256,asset.content_type
  from public.scryglass_public_assets asset
  join public.scryglass_public_releases release
    on release.release_id=asset.release_id
  where release.status='active'
    and pg_catalog.octet_length(p_release_id) between 1 and 30
    and p_release_id~'^v[0-9]{4}\.[0-9]{2}\.[0-9]{2}\.[0-9]{6}$'
    and pg_catalog.octet_length(p_path) between 1 and 100
    and asset.release_id=p_release_id and asset.path=p_path
    and (
      asset.path<>'features/draft_records.json'
      or public.scryglass_query_descriptive_authority(release.release_id) is not null
    )
    and asset.body is null
    and asset.storage_path=asset.release_id||'/'||asset.path
    and asset.content_type='application/json'
    and release.manifest#>>array['release','artifact_hashes',asset.path]=asset.sha256
    and exists(
      select 1 from pg_catalog.jsonb_array_elements(case
        when pg_catalog.jsonb_typeof(release.manifest->'files')='array'
          then release.manifest->'files' else '[]'::jsonb end) file
      where file->>'path'=asset.path
        and (file->>'bytes')::bigint=asset.bytes
        and file->>'sha256'=asset.sha256
    )
$$;

create or replace function scryglass_private.is_active_scryglass_storage_object(
  p_storage_path text
)
returns boolean
language sql
stable
security definer
set search_path = public, pg_temp
set statement_timeout = '5s'
as $$
  select coalesce(
    pg_catalog.octet_length(p_storage_path) between 1 and 220
    and exists (
      select 1
      from public.scryglass_public_assets asset
      join public.scryglass_public_releases release
        on release.release_id = asset.release_id
      where release.status = 'active'
        and asset.body is null
        and (
          asset.path <> 'features/draft_records.json'
          or public.scryglass_query_descriptive_authority(release.release_id) is not null
        )
        and asset.storage_path = p_storage_path
        and asset.storage_path = asset.release_id || '/' || asset.path
        and asset.content_type = 'application/json'
        and release.manifest #>> array['release', 'artifact_hashes', asset.path]
          = asset.sha256
        and exists (
          select 1
          from pg_catalog.jsonb_array_elements(
            case
              when pg_catalog.jsonb_typeof(release.manifest -> 'files') = 'array'
                then release.manifest -> 'files'
              else '[]'::jsonb
            end
          ) file
          where file ->> 'path' = asset.path
            and (file ->> 'bytes')::bigint = asset.bytes
            and file ->> 'sha256' = asset.sha256
        )
    ),
    false
  )
$$;

revoke all on function public.scryglass_descriptive_signal_is_valid(jsonb)
  from public,anon,authenticated,service_role;
revoke all on function public.scryglass_descriptive_summary_is_valid(jsonb)
  from public,anon,authenticated,service_role;
revoke all on function public.scryglass_query_descriptive_authority(text)
  from public,anon,authenticated,service_role;
revoke all on function public.scryglass_strip_query_draft_fields(jsonb)
  from public,anon,authenticated,service_role;
revoke all on function public.scryglass_json_has_unbound_descriptive_draft(jsonb,jsonb)
  from public,anon,authenticated,service_role;
revoke all on function public.get_scryglass_active_release(text)
  from public,anon,authenticated,service_role;
revoke all on function public.get_scryglass_active_asset(text,text)
  from public,anon,authenticated,service_role;
grant execute on function public.get_scryglass_active_release(text)
  to anon,authenticated;
grant execute on function public.get_scryglass_active_asset(text,text)
  to anon,authenticated;

comment on function public.scryglass_bounded_query_result(jsonb) is
  'Caps public RPC responses and binds descriptive Draft rows to the active receipt.';
