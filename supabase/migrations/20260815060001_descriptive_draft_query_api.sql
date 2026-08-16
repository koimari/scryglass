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
language sql
stable
security definer
set search_path = ''
set statement_timeout = '5s'
as $$
  select release.manifest -> 'draft_authority'
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
  limit 1
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

comment on function public.scryglass_bounded_query_result(jsonb) is
  'Caps public RPC responses and binds descriptive Draft rows to the active receipt.';
