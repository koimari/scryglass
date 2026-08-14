-- Release-bound bounded query API. The large compatibility assets remain for
-- one dual-write window. Browser routes read these active-only functions.

alter table public.scryglass_refresh_runs
  add column if not exists requirements_lock_sha256 text;

alter table public.scryglass_refresh_runs
  drop constraint if exists scryglass_refresh_runs_requirements_lock_sha256_check;
alter table public.scryglass_refresh_runs
  add constraint scryglass_refresh_runs_requirements_lock_sha256_check
  check (
    requirements_lock_sha256 is null
    or requirements_lock_sha256 ~ '^[0-9a-f]{64}$'
  );

create or replace function public.require_scryglass_requirements_lock()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $$
begin
  if new.requirements_lock_sha256 is null then
    raise exception 'Scryglass refresh run requires a release lock digest';
  end if;
  return new;
end;
$$;

drop trigger if exists require_scryglass_requirements_lock
  on public.scryglass_refresh_runs;
create trigger require_scryglass_requirements_lock
before insert on public.scryglass_refresh_runs
for each row execute function public.require_scryglass_requirements_lock();

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
        'draft_probability', 'draft_score', 'draft_win_share'
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

create table public.scryglass_public_query_rows (
  release_id text not null references public.scryglass_public_releases(release_id)
    on delete cascade,
  dataset text not null check (dataset in (
    'players', 'teams', 'player_champions', 'games', 'identity_games',
    'champions', 'aliases', 'tier_rows', 'tier_scopes',
    'tier_matrix_rows', 'tier_similarity_champions',
    'tier_similarity_edges'
  )),
  row_key text not null check (pg_catalog.length(row_key) between 1 and 200),
  player_id text,
  team_id text,
  champion_id text,
  identity_id text,
  reference_id text,
  scope_id text,
  name text check (name is null or pg_catalog.length(name) between 1 and 100),
  search_key text check (search_key is null or pg_catalog.length(search_key) between 1 and 100),
  kind text,
  alias_key text,
  role text,
  team text,
  league text,
  tier text,
  active boolean,
  rating double precision,
  adjusted_rating double precision,
  movement double precision,
  games integer,
  wins integer,
  win_rate double precision,
  grade_a_games integer,
  grade_games integer,
  champion text,
  champion_key text,
  score double precision,
  game_id text,
  played_at text,
  year integer,
  blue_team text,
  red_team text,
  blue_team_id text,
  red_team_id text,
  blue_win integer,
  champions text[],
  ordinal integer,
  image_url text check (
    image_url is null
    or image_url ~ '^https://cdn\.communitydragon\.org/'
  ),
  patch text,
  region text,
  rank integer,
  played_maps integer,
  payload jsonb not null,
  source_bytes integer not null check (source_bytes between 2 and 65536),
  source_sha256 text not null check (source_sha256 ~ '^[0-9a-f]{64}$'),
  row_sha256 text not null check (row_sha256 ~ '^[0-9a-f]{64}$'),
  source_json text not null check (pg_catalog.octet_length(source_json) <= 70000),
  primary key (release_id, dataset, row_key),
  check (pg_catalog.octet_length(payload::text) <= 65536),
  check (not public.scryglass_json_has_draft_fields(payload)),
  check (rating is null or (rating > '-Infinity'::double precision and rating < 'Infinity'::double precision)),
  check (adjusted_rating is null or (adjusted_rating > '-Infinity'::double precision and adjusted_rating < 'Infinity'::double precision)),
  check (movement is null or (movement > '-Infinity'::double precision and movement < 'Infinity'::double precision)),
  check (win_rate is null or (win_rate > '-Infinity'::double precision and win_rate < 'Infinity'::double precision)),
  check (score is null or (score > '-Infinity'::double precision and score < 'Infinity'::double precision))
);

create table public.scryglass_public_query_receipts (
  release_id text not null references public.scryglass_public_releases(release_id)
    on delete cascade,
  dataset text not null,
  schema_version text not null check (schema_version = 'scryglass:query-api:v1'),
  row_count integer not null check (row_count >= 0),
  source_bytes bigint not null check (source_bytes >= 2),
  source_sha256 text not null check (source_sha256 ~ '^[0-9a-f]{64}$'),
  row_digest_sha256 text not null check (row_digest_sha256 ~ '^[0-9a-f]{64}$'),
  storage_bytes bigint not null check (storage_bytes >= 0),
  storage_sha256 text not null check (storage_sha256 ~ '^[0-9a-f]{64}$'),
  sealed_at timestamptz not null default now(),
  primary key (release_id, dataset)
);

create index scryglass_query_ratings_idx
  on public.scryglass_public_query_rows
  (release_id, dataset, active, tier, league, role, adjusted_rating desc nulls last);
create index scryglass_query_identity_idx
  on public.scryglass_public_query_rows
  (release_id, dataset, kind, alias_key);
create index scryglass_query_games_idx
  on public.scryglass_public_query_rows
  (release_id, dataset, played_at desc, game_id);
create index scryglass_query_profile_games_idx
  on public.scryglass_public_query_rows
  (release_id, dataset, kind, identity_id, ordinal);
create index scryglass_query_champions_idx
  on public.scryglass_public_query_rows
  (release_id, dataset, champion_key, games desc);
create index scryglass_query_tier_idx
  on public.scryglass_public_query_rows
  (release_id, dataset, patch, role, region, league, tier, rank);
create index scryglass_query_scope_idx
  on public.scryglass_public_query_rows
  (release_id, dataset, scope_id, ordinal);

alter table public.scryglass_public_query_rows enable row level security;
alter table public.scryglass_public_query_receipts enable row level security;

create or replace function public.guard_scryglass_query_mutation()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $$
declare
  target_release text := coalesce(new.release_id, old.release_id);
  target_dataset text := coalesce(new.dataset, old.dataset);
begin
  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtext('scryglass-public-release')
  );
  if tg_op = 'DELETE' then
    if current_user = 'scryglass_release_retention_owner'
       or exists (
      select 1 from public.scryglass_public_releases release
      where release.release_id = target_release and release.status = 'superseded'
    ) then
      return old;
    end if;
    raise exception 'Scryglass query data can only leave with a superseded release';
  end if;
  if tg_op <> 'INSERT' then
    raise exception 'Scryglass query rows and receipts are immutable';
  end if;
  if not exists (
    select 1 from public.scryglass_public_releases release
    where release.release_id = target_release and release.status = 'staging'
  ) then
    raise exception 'Scryglass query data can only enter a staging release';
  end if;
  if tg_table_name = 'scryglass_public_query_rows' and exists (
    select 1 from public.scryglass_public_query_receipts receipt
    where receipt.release_id = target_release and receipt.dataset = target_dataset
  ) then
    raise exception 'A sealed Scryglass query dataset is immutable';
  end if;
  return new;
end;
$$;

create trigger guard_scryglass_query_rows
before insert or update or delete on public.scryglass_public_query_rows
for each row execute function public.guard_scryglass_query_mutation();
create trigger guard_scryglass_query_receipts
before insert or update or delete on public.scryglass_public_query_receipts
for each row execute function public.guard_scryglass_query_mutation();

create or replace function public.stage_scryglass_query_rows(
  p_release_id text,
  p_dataset text,
  p_rows jsonb
)
returns integer
language plpgsql
security definer
set search_path = ''
as $$
declare
  item jsonb;
  source jsonb;
  source_json_text text;
  payload_json_text text;
  expected_row_sha256 text;
  inserted_count integer := 0;
  row_delta integer;
begin
  if p_dataset not in (
    'players', 'teams', 'player_champions', 'games', 'identity_games',
    'champions', 'aliases', 'tier_rows', 'tier_scopes',
    'tier_matrix_rows', 'tier_similarity_champions', 'tier_similarity_edges'
  ) or pg_catalog.jsonb_typeof(p_rows) <> 'array'
     or pg_catalog.jsonb_array_length(p_rows) > 100
     or pg_catalog.octet_length(p_rows::text) > 450000
  then
    raise exception 'Scryglass query staging batch is invalid';
  end if;
  for item in
    select element.value
    from pg_catalog.jsonb_array_elements(p_rows) as element(value)
  loop
    source_json_text := item ->> 'source_json';
    payload_json_text := item ->> 'payload_json';
    expected_row_sha256 := item ->> 'row_sha256';
    if item ->> 'release_id' <> p_release_id
       or item ->> 'dataset' <> p_dataset
       or (item - 'release_id' - 'dataset' - 'source_json' - 'payload_json' - 'row_sha256') <> '{}'::jsonb
       or pg_catalog.octet_length(coalesce(source_json_text,'')) > 70000
       or pg_catalog.octet_length(coalesce(payload_json_text,'')) > 65536
       or expected_row_sha256 !~ '^[0-9a-f]{64}$'
    then
      raise exception 'Scryglass query row binding is invalid';
    end if;
    begin
      source := source_json_text::jsonb;
    exception when others then
      raise exception 'Scryglass query source row is invalid JSON';
    end;
    if pg_catalog.encode(extensions.digest(pg_catalog.convert_to(source_json_text,'UTF8'),'sha256'),'hex')
         <> expected_row_sha256
       or pg_catalog.jsonb_typeof(source -> 'payload') <> 'object'
       or payload_json_text::jsonb <> source -> 'payload'
       or pg_catalog.encode(extensions.digest(pg_catalog.convert_to(payload_json_text,'UTF8'),'sha256'),'hex')
         <> source ->> 'source_sha256'
       or pg_catalog.length(source ->> 'row_key') not between 1 and 200
       or public.scryglass_json_has_draft_fields(source)
    then
      raise exception 'Scryglass canonical query row digest is invalid';
    end if;
    item := source;
    insert into public.scryglass_public_query_rows (
      release_id, dataset, row_key, player_id, team_id, champion_id,
      identity_id, reference_id, scope_id, name, search_key, kind,
      alias_key, role, team, league, tier, active, rating,
      adjusted_rating, movement, games, wins, win_rate, grade_a_games,
      grade_games, champion, champion_key, score, game_id, played_at,
      year, blue_team, red_team, blue_team_id, red_team_id, blue_win,
      champions, ordinal, image_url, patch, region, rank, played_maps,
      payload, source_bytes, source_sha256, row_sha256, source_json
    ) values (
      p_release_id, p_dataset, item ->> 'row_key', item ->> 'player_id',
      item ->> 'team_id', item ->> 'champion_id', item ->> 'identity_id',
      item ->> 'reference_id', item ->> 'scope_id', item ->> 'name',
      item ->> 'search_key', item ->> 'kind', item ->> 'alias_key',
      item ->> 'role', item ->> 'team', item ->> 'league', item ->> 'tier',
      case when item ? 'active' then (item ->> 'active')::boolean end,
      nullif(item ->> 'rating', '')::double precision,
      nullif(item ->> 'adjusted_rating', '')::double precision,
      nullif(item ->> 'movement', '')::double precision,
      nullif(item ->> 'games', '')::integer,
      nullif(item ->> 'wins', '')::integer,
      nullif(item ->> 'win_rate', '')::double precision,
      nullif(item ->> 'grade_a_games', '')::integer,
      nullif(item ->> 'grade_games', '')::integer,
      item ->> 'champion', item ->> 'champion_key',
      nullif(item ->> 'score', '')::double precision,
      item ->> 'game_id', item ->> 'played_at',
      nullif(item ->> 'year', '')::integer,
      item ->> 'blue_team', item ->> 'red_team',
      item ->> 'blue_team_id', item ->> 'red_team_id',
      nullif(item ->> 'blue_win', '')::integer,
      case when pg_catalog.jsonb_typeof(item -> 'champions') = 'array' then
        array(select pg_catalog.jsonb_array_elements_text(item -> 'champions'))
      end,
      nullif(item ->> 'ordinal', '')::integer,
      item ->> 'image_url', item ->> 'patch', item ->> 'region',
      nullif(item ->> 'rank', '')::integer,
      nullif(item ->> 'played_maps', '')::integer, item -> 'payload',
      (item ->> 'source_bytes')::integer, item ->> 'source_sha256',
      expected_row_sha256, source_json_text
    ) on conflict (release_id, dataset, row_key) do nothing;
    get diagnostics row_delta = row_count;
    inserted_count := inserted_count + row_delta;
  end loop;
  return inserted_count;
end;
$$;

create or replace function public.seal_scryglass_query_dataset(
  p_release_id text,
  p_dataset text,
  p_source_rows integer,
  p_source_bytes bigint,
  p_source_sha256 text,
  p_row_digest_sha256 text
)
returns jsonb
language plpgsql
security definer
set search_path = ''
as $$
declare
  release_manifest jsonb;
  actual_rows integer;
  actual_bytes bigint;
  actual_row_digest text;
  actual_storage_digest text;
  prior public.scryglass_public_query_receipts%rowtype;
begin
  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtext('scryglass-public-release')
  );
  select release.manifest into release_manifest
  from public.scryglass_public_releases release
  where release.release_id = p_release_id and release.status = 'staging';
  if release_manifest is null
     or release_manifest #>> '{query_api,schema_version}'
       is distinct from 'scryglass:query-api:v1'
     or release_manifest #>> '{query_api,status}' is distinct from 'available'
     or (release_manifest #>> array['query_api','datasets',p_dataset,'rows'])::integer
       is distinct from p_source_rows
     or (release_manifest #>> array['query_api','datasets',p_dataset,'bytes'])::bigint
       is distinct from p_source_bytes
     or release_manifest #>> array['query_api','datasets',p_dataset,'sha256']
       is distinct from p_source_sha256
     or release_manifest #>> array['query_api','datasets',p_dataset,'row_digest_sha256']
       is distinct from p_row_digest_sha256
     or p_source_sha256 !~ '^[0-9a-f]{64}$'
     or p_row_digest_sha256 !~ '^[0-9a-f]{64}$'
  then
    raise exception 'Scryglass query receipt does not match its release manifest';
  end if;

  select
    pg_catalog.count(*),
    coalesce(pg_catalog.sum(pg_catalog.octet_length(row.source_json)), 0),
    pg_catalog.encode(extensions.digest(pg_catalog.convert_to(coalesce(
      pg_catalog.string_agg(row.row_key || ':' || row.row_sha256, E'\n' order by row.row_key),
      ''
    ), 'UTF8'), 'sha256'), 'hex'),
    pg_catalog.encode(extensions.digest(pg_catalog.convert_to(coalesce(
      pg_catalog.string_agg(
        row.row_key || ':' || pg_catalog.encode(
          extensions.digest(pg_catalog.convert_to(row.source_json, 'UTF8'), 'sha256'), 'hex'
        ),
        E'\n' order by row.row_key
      ),
      ''
    ), 'UTF8'), 'sha256'), 'hex')
  into actual_rows, actual_bytes, actual_row_digest, actual_storage_digest
  from public.scryglass_public_query_rows row
  where row.release_id = p_release_id and row.dataset = p_dataset;

  if actual_rows <> p_source_rows
     or actual_row_digest <> p_row_digest_sha256
  then
    raise exception 'Scryglass staged query rows do not match their receipt';
  end if;

  select * into prior
  from public.scryglass_public_query_receipts receipt
  where receipt.release_id = p_release_id and receipt.dataset = p_dataset;
  if found then
    if prior.row_count <> p_source_rows
       or prior.source_bytes <> p_source_bytes
       or prior.source_sha256 <> p_source_sha256
       or prior.row_digest_sha256 <> p_row_digest_sha256
       or prior.storage_bytes <> actual_bytes
       or prior.storage_sha256 <> actual_storage_digest
    then
      raise exception 'Scryglass query receipt is immutable';
    end if;
  else
    insert into public.scryglass_public_query_receipts (
      release_id, dataset, schema_version, row_count, source_bytes,
      source_sha256, row_digest_sha256, storage_bytes, storage_sha256
    ) values (
      p_release_id, p_dataset, 'scryglass:query-api:v1', actual_rows,
      p_source_bytes, p_source_sha256, p_row_digest_sha256,
      actual_bytes, actual_storage_digest
    );
  end if;
  return pg_catalog.jsonb_build_object(
    'release_id', p_release_id,
    'dataset', p_dataset,
    'rows', actual_rows,
    'storage_bytes', actual_bytes,
    'storage_sha256', actual_storage_digest
  );
end;
$$;

create or replace function public.assert_scryglass_query_release(p_release_id text)
returns void
language plpgsql
security invoker
set search_path = ''
as $$
declare
  release_manifest jsonb;
  required_datasets constant text[] := array[
    'players', 'teams', 'player_champions', 'games', 'identity_games',
    'champions', 'aliases', 'tier_rows', 'tier_scopes',
    'tier_matrix_rows', 'tier_similarity_champions',
    'tier_similarity_edges'
  ];
  required_dataset text;
  receipt public.scryglass_public_query_receipts%rowtype;
  actual_rows integer;
  actual_bytes bigint;
  actual_storage_digest text;
begin
  select release.manifest into release_manifest
  from public.scryglass_public_releases release
  where release.release_id = p_release_id
    and release.status in ('staging', 'active', 'superseded');
  if release_manifest is null then
    raise exception 'Scryglass query release is unavailable';
  end if;
  -- Legacy releases remain restorable during the dual-write window.
  if not (release_manifest ? 'query_api') then
    return;
  end if;
  if release_manifest #>> '{query_api,schema_version}'
       is distinct from 'scryglass:query-api:v1'
     or release_manifest #>> '{query_api,status}' is distinct from 'available'
     or pg_catalog.jsonb_typeof(release_manifest #> '{query_api,datasets}') <> 'object'
     or public.scryglass_json_has_draft_fields(release_manifest -> 'query_api')
  then
    raise exception 'Scryglass query API authority is invalid';
  end if;
  if (
    select pg_catalog.array_agg(key order by key)
    from pg_catalog.jsonb_object_keys(release_manifest #> '{query_api,datasets}') key
  ) is distinct from (
    select pg_catalog.array_agg(value order by value)
    from pg_catalog.unnest(required_datasets) value
  ) then
    raise exception 'Scryglass query dataset inventory is not exact';
  end if;
  foreach required_dataset in array required_datasets
  loop
    select * into receipt
    from public.scryglass_public_query_receipts candidate
    where candidate.release_id = p_release_id
      and candidate.dataset = required_dataset;
    if not found
       or receipt.row_count is distinct from
         (release_manifest #>> array[
           'query_api','datasets',required_dataset,'rows'
         ])::integer
       or receipt.source_bytes is distinct from
         (release_manifest #>> array[
           'query_api','datasets',required_dataset,'bytes'
         ])::bigint
       or receipt.source_sha256 is distinct from
         release_manifest #>> array[
           'query_api','datasets',required_dataset,'sha256'
         ]
       or receipt.row_digest_sha256 is distinct from
         release_manifest #>> array[
           'query_api','datasets',required_dataset,'row_digest_sha256'
         ]
    then
      raise exception 'Scryglass query dataset receipt is invalid: %',
        required_dataset;
    end if;
    select
      pg_catalog.count(*),
      coalesce(pg_catalog.sum(pg_catalog.octet_length(row.source_json)), 0),
      pg_catalog.encode(extensions.digest(pg_catalog.convert_to(coalesce(
        pg_catalog.string_agg(
          row.row_key || ':' || pg_catalog.encode(
            extensions.digest(pg_catalog.convert_to(row.source_json, 'UTF8'), 'sha256'), 'hex'
          ),
          E'\n' order by row.row_key
        ),
        ''
      ), 'UTF8'), 'sha256'), 'hex')
    into actual_rows, actual_bytes, actual_storage_digest
    from public.scryglass_public_query_rows row
    where row.release_id = p_release_id
      and row.dataset = required_dataset;
    if actual_rows <> receipt.row_count
       or actual_bytes <> receipt.storage_bytes
       or actual_storage_digest <> receipt.storage_sha256
    then
      raise exception 'Scryglass query dataset storage receipt is invalid: %',
        required_dataset;
    end if;
  end loop;
end;
$$;

create or replace function public.scryglass_active_query_release_id()
returns text
language sql
stable
security definer
set search_path = ''
as $$
  select release.release_id
  from public.scryglass_public_releases release
  where release.status = 'active'
    and release.manifest #>> '{query_api,schema_version}' = 'scryglass:query-api:v1'
    and release.manifest #>> '{query_api,status}' = 'available'
  limit 1
$$;

create or replace function public.scryglass_bounded_query_result(value jsonb)
returns jsonb
language plpgsql
stable
security definer
set search_path = ''
as $$
begin
  if value is null then
    return null;
  end if;
  if public.scryglass_json_has_draft_fields(value)
     or pg_catalog.octet_length(value::text) > 500000
  then
    raise exception 'Scryglass public query response exceeds its authority or byte budget';
  end if;
  return value;
end;
$$;

create or replace function public.scryglass_text_array_is_bounded(
  value text[],
  max_items integer,
  max_element_bytes integer default 100
)
returns boolean
language sql
immutable
security invoker
set search_path = ''
as $$
  select max_items between 0 and 100
    and max_element_bytes between 1 and 500
    and coalesce(pg_catalog.cardinality(value), 0) <= max_items
    and not exists (
      select 1
      from pg_catalog.unnest(value) item
      where item is null
        or pg_catalog.octet_length(pg_catalog.btrim(item)) not between 1 and max_element_bytes
    )
$$;

create or replace function public.get_scryglass_ratings(
  p_kind text,
  p_leagues text[] default null,
  p_tiers text[] default null,
  p_roles text[] default null,
  p_teams text[] default null,
  p_names text[] default null,
  p_active boolean default null,
  p_search text default null,
  p_order text default 'rating_desc',
  p_min_games integer default 0,
  p_limit integer default 20,
  p_offset integer default 0
)
returns jsonb
language plpgsql
stable
security definer
set search_path = ''
set statement_timeout = '5s'
as $$
declare
  active_release text := public.scryglass_active_query_release_id();
  dataset_name text;
  result jsonb;
  bounded_limit integer := least(greatest(coalesce(p_limit, 20), 1), 20);
begin
  dataset_name := case pg_catalog.lower(p_kind)
    when 'player' then 'players' when 'players' then 'players'
    when 'team' then 'teams' when 'teams' then 'teams' else null end;
  if active_release is null or dataset_name is null
     or pg_catalog.octet_length(coalesce(p_kind, '')) not between 1 and 20
     or p_order is null or pg_catalog.octet_length(p_order) not between 1 and 30
     or p_order not in (
       'rating_desc', 'rating_asc', 'movement_desc', 'games_desc', 'games_asc',
       'wins_desc', 'wins_asc', 'name_asc', 'win_rate_desc', 'win_rate_asc',
       'grade_a_desc', 'grade_a_asc'
     )
     or p_min_games is null or p_min_games not between 0 and 10000
     or p_limit is null
     or p_offset is null or p_offset not between 0 and 10000
     or pg_catalog.octet_length(coalesce(p_search, '')) > 100
     or not public.scryglass_text_array_is_bounded(p_leagues, 50)
     or not public.scryglass_text_array_is_bounded(p_tiers, 20)
     or not public.scryglass_text_array_is_bounded(p_roles, 10)
     or not public.scryglass_text_array_is_bounded(p_teams, 50)
     or not public.scryglass_text_array_is_bounded(p_names, 20)
  then
    raise exception 'Scryglass rating query is invalid';
  end if;
  with filtered as (
    select row.*
    from public.scryglass_public_query_rows row
    where row.release_id = active_release
      and row.dataset = dataset_name
      and coalesce(row.games, 0) >= p_min_games
      and (p_leagues is null or row.league = any(p_leagues))
      and (p_tiers is null or row.tier = any(p_tiers))
      and (p_roles is null or row.role = any(p_roles))
      and (p_teams is null or exists (
        select 1 from pg_catalog.unnest(p_teams) requested(name)
        where row.team_id=coalesce(
          (select exact_team.team_id
            from public.scryglass_public_query_rows exact_team
            where exact_team.release_id=active_release
              and exact_team.dataset='teams'
              and exact_team.name=requested.name limit 1),
          (select alias.identity_id from public.scryglass_public_query_rows alias
            where alias.release_id=active_release and alias.dataset='aliases'
              and alias.kind='team'
              and alias.alias_key=pg_catalog.lower(pg_catalog.btrim(requested.name)) limit 1),
          (select team_row.team_id from public.scryglass_public_query_rows team_row
            where team_row.release_id=active_release and team_row.dataset='teams'
              and team_row.search_key=pg_catalog.lower(pg_catalog.btrim(requested.name)) limit 1)
        )
      ))
      and (p_names is null or exists (
        select 1 from pg_catalog.unnest(p_names) requested(name)
        where coalesce(row.player_id,row.team_id)=coalesce(
          (select coalesce(exact_row.player_id,exact_row.team_id)
            from public.scryglass_public_query_rows exact_row
            where exact_row.release_id=active_release
              and exact_row.dataset=dataset_name
              and exact_row.name=requested.name limit 1),
          (select alias.identity_id from public.scryglass_public_query_rows alias
            where alias.release_id=active_release and alias.dataset='aliases'
              and alias.kind=case dataset_name when 'players' then 'player' else 'team' end
              and alias.alias_key=pg_catalog.lower(pg_catalog.btrim(requested.name)) limit 1),
          (select direct.identity from (
            select coalesce(candidate.player_id,candidate.team_id) identity
            from public.scryglass_public_query_rows candidate
            where candidate.release_id=active_release and candidate.dataset=dataset_name
              and candidate.search_key=pg_catalog.lower(pg_catalog.btrim(requested.name)) limit 1
          ) direct)
        )
      ))
      and (p_active is null or row.active = p_active)
      and (
        p_search is null or pg_catalog.btrim(p_search) = ''
        or row.search_key like '%' || pg_catalog.lower(pg_catalog.btrim(p_search)) || '%'
      )
  ), page as (
    select * from filtered
    order by
      case when p_order = 'rating_desc' then adjusted_rating end desc nulls last,
      case when p_order = 'rating_asc' then adjusted_rating end asc nulls last,
      case when p_order = 'movement_desc' then movement end desc nulls last,
      case when p_order = 'games_desc' then games end desc nulls last,
      case when p_order = 'games_asc' then games end asc nulls last,
      case when p_order = 'wins_desc' then wins end desc nulls last,
      case when p_order = 'wins_asc' then wins end asc nulls last,
      case when p_order = 'win_rate_desc' then win_rate end desc nulls last,
      case when p_order = 'win_rate_asc' then win_rate end asc nulls last,
      case when p_order = 'grade_a_desc' then grade_a_games end desc nulls last,
      case when p_order = 'grade_a_asc' then grade_a_games end asc nulls last,
      case when p_order = 'name_asc' then name end asc,
      name asc, row_key asc
    limit bounded_limit offset p_offset
  )
  select pg_catalog.jsonb_build_object(
    'schema_version', 'scryglass:query-api:v1',
    'release_id', active_release,
    'rows', coalesce((
      select pg_catalog.jsonb_agg(pg_catalog.jsonb_strip_nulls(
        pg_catalog.jsonb_build_object(
          'row_key', row_key, 'name', name, 'role', role, 'team', team,
          'league', league, 'tier', tier, 'active', active, 'rating', rating,
          'adjusted_rating', adjusted_rating, 'movement', movement,
          'games', games, 'wins', wins, 'win_rate', win_rate,
          'grade_a_games', grade_a_games, 'grade_games', grade_games,
          'payload', payload
        )
      )) from page
    ), '[]'::jsonb),
    'limit', bounded_limit,
    'offset', p_offset,
    'total', (select pg_catalog.count(*) from filtered)
  ) into result;
  return public.scryglass_bounded_query_result(result);
end;
$$;

create or replace function public.get_scryglass_rating_facets(
  p_kind text,
  p_tiers text[] default null
)
returns jsonb
language plpgsql
stable
security definer
set search_path = ''
set statement_timeout = '5s'
as $$
declare
  active_release text := public.scryglass_active_query_release_id();
  dataset_name text;
  result jsonb;
begin
  dataset_name := case pg_catalog.lower(p_kind)
    when 'player' then 'players' when 'players' then 'players'
    when 'team' then 'teams' when 'teams' then 'teams' else null end;
  if active_release is null or dataset_name is null
     or pg_catalog.octet_length(coalesce(p_kind, '')) not between 1 and 20
     or not public.scryglass_text_array_is_bounded(p_tiers, 20)
  then raise exception 'Scryglass rating facet query is invalid'; end if;
  with rows as (
    select * from public.scryglass_public_query_rows row
    where row.release_id = active_release and row.dataset = dataset_name
      and (p_tiers is null or row.tier = any(p_tiers))
  )
  select pg_catalog.jsonb_build_object(
    'schema_version', 'scryglass:query-api:v1', 'release_id', active_release,
    'leagues', coalesce((select pg_catalog.jsonb_agg(value order by value)
      from (select distinct league value from rows where league is not null limit 250) item), '[]'::jsonb),
    'tiers', coalesce((select pg_catalog.jsonb_agg(value order by value)
      from (select distinct tier value from rows where tier is not null limit 50) item), '[]'::jsonb),
    'roles', coalesce((select pg_catalog.jsonb_agg(value order by value)
      from (select distinct role value from rows where role is not null limit 20) item), '[]'::jsonb),
    'min_games', coalesce((select pg_catalog.min(games) from rows), 0),
    'max_games', coalesce((select pg_catalog.max(games) from rows), 0)
  ) into result;
  return public.scryglass_bounded_query_result(result);
end;
$$;

create or replace function public.get_scryglass_player_champions(
  p_player text default null,
  p_champion text default null,
  p_leagues text[] default null,
  p_tiers text[] default null,
  p_roles text[] default null,
  p_teams text[] default null,
  p_active boolean default null,
  p_min_games integer default 5,
  p_order text default 'best',
  p_limit integer default 20,
  p_offset integer default 0
)
returns jsonb
language plpgsql
stable
security definer
set search_path = ''
set statement_timeout = '5s'
as $$
declare
  active_release text := public.scryglass_active_query_release_id();
  result jsonb;
  bounded_limit integer := least(greatest(coalesce(p_limit, 20), 1), 20);
begin
  if active_release is null
     or p_order is null or pg_catalog.octet_length(p_order) not between 1 and 30
     or p_order not in (
       'best', 'worst', 'games_desc', 'games_asc', 'median', 'mean',
       'rating_desc', 'rating_asc', 'win_rate_desc', 'win_rate_asc'
     )
     or p_min_games is null or p_min_games not between 0 and 10000
     or p_limit is null
     or p_offset is null or p_offset not between 0 and 10000
     or pg_catalog.octet_length(coalesce(p_player, '')) > 100
     or pg_catalog.octet_length(coalesce(p_champion, '')) > 100
     or not public.scryglass_text_array_is_bounded(p_leagues, 50)
     or not public.scryglass_text_array_is_bounded(p_tiers, 20)
     or not public.scryglass_text_array_is_bounded(p_roles, 10)
     or not public.scryglass_text_array_is_bounded(p_teams, 50)
  then raise exception 'Scryglass player champion query is invalid'; end if;
  with filtered_base as (
    select
      pc.*, player.name player_name, player.role player_role,
      player.team player_team, player.league player_league,
      player.tier player_tier, player.active player_active,
      player.rating player_rating, player.adjusted_rating player_adjusted_rating,
      tier_row.rank tier_rank, tier_row.score tier_score
    from public.scryglass_public_query_rows pc
    join public.scryglass_public_query_rows player
      on player.release_id = pc.release_id and player.dataset = 'players'
      and player.player_id = pc.player_id
    left join lateral (
      select tier.rank, tier.score
      from public.scryglass_public_query_rows tier
      where tier.release_id = pc.release_id and tier.dataset = 'tier_rows'
        and tier.search_key = pc.champion_key
        and (player.role is null or tier.role = player.role)
      order by tier.patch desc, tier.rank asc nulls last
      limit 1
    ) tier_row on true
    where pc.release_id = active_release and pc.dataset = 'player_champions'
      and coalesce(pc.games, 0) >= p_min_games
      and (p_player is null or player.name = p_player
        or player.search_key = pg_catalog.lower(pg_catalog.btrim(p_player))
        or player.player_id in (
          select alias.identity_id from public.scryglass_public_query_rows alias
          where alias.release_id = active_release and alias.dataset = 'aliases'
            and alias.kind = 'player'
            and alias.alias_key = pg_catalog.lower(pg_catalog.btrim(p_player))
        ))
      and (p_champion is null or pc.champion = p_champion
        or pc.champion_key = pg_catalog.lower(pg_catalog.btrim(p_champion))
        or pc.champion_id in (
          select alias.identity_id from public.scryglass_public_query_rows alias
          where alias.release_id = active_release and alias.dataset = 'aliases'
            and alias.kind = 'champion'
            and alias.alias_key = pg_catalog.lower(pg_catalog.btrim(p_champion))
        ))
      and (p_leagues is null or player.league = any(p_leagues))
      and (p_tiers is null or player.tier = any(p_tiers))
      and (p_roles is null or player.role = any(p_roles))
      and (p_teams is null or exists (
        select 1 from pg_catalog.unnest(p_teams) requested(name)
        where player.team_id=coalesce(
          (select exact_team.team_id
            from public.scryglass_public_query_rows exact_team
            where exact_team.release_id=active_release
              and exact_team.dataset='teams'
              and exact_team.name=requested.name limit 1),
          (select alias.identity_id from public.scryglass_public_query_rows alias
            where alias.release_id=active_release and alias.dataset='aliases'
              and alias.kind='team'
              and alias.alias_key=pg_catalog.lower(pg_catalog.btrim(requested.name)) limit 1),
          (select team_row.team_id from public.scryglass_public_query_rows team_row
            where team_row.release_id=active_release and team_row.dataset='teams'
              and team_row.search_key=pg_catalog.lower(pg_catalog.btrim(requested.name)) limit 1)
        )
      ))
      and (p_active is null or player.active = p_active)
  ), statistics as (
    select
      pg_catalog.percentile_cont(0.5) within group (order by win_rate) median_reference,
      pg_catalog.avg(win_rate) mean_reference
    from filtered_base where win_rate is not null
  ), enriched as (
    select filtered_base.*,
      pg_catalog.abs(filtered_base.win_rate - statistics.median_reference) median_distance,
      pg_catalog.abs(filtered_base.win_rate - statistics.mean_reference) mean_distance,
      statistics.median_reference, statistics.mean_reference
    from filtered_base cross join statistics
  ), page as (
    select * from enriched
    order by
      case when p_order = 'best' then tier_rank end asc nulls last,
      case when p_order = 'best' then score end desc nulls last,
      case when p_order = 'worst' then tier_rank end desc nulls last,
      case when p_order = 'worst' then score end asc nulls last,
      case when p_order = 'games_desc' then games end desc nulls last,
      case when p_order = 'games_asc' then games end asc nulls last,
      case when p_order = 'median' then median_distance end asc nulls last,
      case when p_order = 'mean' then mean_distance end asc nulls last,
      case when p_order = 'rating_desc' then player_adjusted_rating end desc nulls last,
      case when p_order = 'rating_asc' then player_adjusted_rating end asc nulls last,
      case when p_order = 'win_rate_desc' then win_rate end desc nulls last,
      case when p_order = 'win_rate_asc' then win_rate end asc nulls last,
      player_name asc, champion asc, row_key asc
    limit bounded_limit offset p_offset
  )
  select pg_catalog.jsonb_build_object(
    'schema_version', 'scryglass:query-api:v1', 'release_id', active_release,
    'rows', coalesce((select pg_catalog.jsonb_agg(
      pg_catalog.jsonb_strip_nulls(pg_catalog.jsonb_build_object(
        'row_key', row_key, 'player_id', player_id, 'player', player_name,
        'champion_id', champion_id, 'champion', champion, 'role', player_role,
        'team', player_team, 'league', player_league, 'tier', player_tier,
        'active', player_active, 'rating', player_rating,
        'adjusted_rating', player_adjusted_rating, 'games', games, 'wins', wins,
        'win_rate', win_rate, 'score', score, 'tier_rank', tier_rank,
        'tier_score', tier_score, 'median_distance', median_distance,
        'mean_distance', mean_distance, 'payload', payload
      ))
    ) from page), '[]'::jsonb),
    'median_reference', (select median_reference from statistics),
    'mean_reference', (select mean_reference from statistics),
    'limit', bounded_limit, 'offset', p_offset,
    'total', (select pg_catalog.count(*) from filtered_base)
  ) into result;
  return public.scryglass_bounded_query_result(result);
end;
$$;

create or replace function public.get_scryglass_champions(
  p_leagues text[] default null,
  p_tiers text[] default null,
  p_roles text[] default null,
  p_active boolean default null,
  p_min_games integer default 5,
  p_order text default 'games_desc',
  p_limit integer default 20,
  p_offset integer default 0
)
returns jsonb
language plpgsql
stable
security definer
set search_path = ''
set statement_timeout = '5s'
as $$
declare
  active_release text := public.scryglass_active_query_release_id();
  result jsonb;
  bounded_limit integer := least(greatest(coalesce(p_limit, 20), 1), 20);
begin
  if active_release is null
     or p_order is null or pg_catalog.octet_length(p_order) not between 1 and 30
     or p_order not in ('games_desc','games_asc','win_rate_desc','win_rate_asc','name_asc')
     or p_min_games is null or p_min_games not between 0 and 10000
     or p_limit is null
     or p_offset is null or p_offset not between 0 and 10000
     or not public.scryglass_text_array_is_bounded(p_leagues, 50)
     or not public.scryglass_text_array_is_bounded(p_tiers, 20)
     or not public.scryglass_text_array_is_bounded(p_roles, 10)
  then raise exception 'Scryglass champion query is invalid'; end if;
  with filtered as (
    select pc.*
    from public.scryglass_public_query_rows pc
    join public.scryglass_public_query_rows player
      on player.release_id = pc.release_id and player.dataset = 'players'
      and player.player_id = pc.player_id
    where pc.release_id = active_release and pc.dataset = 'player_champions'
      and (p_leagues is null or player.league = any(p_leagues))
      and (p_tiers is null or player.tier = any(p_tiers))
      and (p_roles is null or player.role = any(p_roles))
      and (p_active is null or player.active = p_active)
  ), aggregated as (
    select champion_id, pg_catalog.min(champion) champion,
      pg_catalog.sum(games)::integer games, pg_catalog.sum(wins)::integer wins,
      case when pg_catalog.sum(games) > 0
        then pg_catalog.sum(wins)::double precision / pg_catalog.sum(games) end win_rate,
      pg_catalog.count(distinct player_id)::integer players
    from filtered group by champion_id
    having pg_catalog.sum(games) >= p_min_games
  ), page as (
    select * from aggregated order by
      case when p_order='games_desc' then games end desc nulls last,
      case when p_order='games_asc' then games end asc nulls last,
      case when p_order='win_rate_desc' then win_rate end desc nulls last,
      case when p_order='win_rate_asc' then win_rate end asc nulls last,
      champion asc
    limit bounded_limit offset p_offset
  )
  select pg_catalog.jsonb_build_object(
    'schema_version','scryglass:query-api:v1','release_id',active_release,
    'rows',coalesce((select pg_catalog.jsonb_agg(pg_catalog.to_jsonb(page)) from page),'[]'::jsonb),
    'limit',bounded_limit,'offset',p_offset,'total',(select pg_catalog.count(*) from aggregated)
  ) into result;
  return public.scryglass_bounded_query_result(result);
end;
$$;

create or replace function public.get_scryglass_tier_rows(
  p_kind text,
  p_patches text[] default null,
  p_regions text[] default null,
  p_leagues text[] default null,
  p_tiers text[] default null,
  p_roles text[] default null,
  p_search text default null,
  p_min_games integer default 0,
  p_order text default 'rank_asc',
  p_limit integer default 20,
  p_offset integer default 0
)
returns jsonb
language plpgsql
stable
security definer
set search_path = ''
set statement_timeout = '5s'
as $$
declare
  active_release text := public.scryglass_active_query_release_id();
  normalized_kind text;
  result jsonb;
  bounded_limit integer := least(greatest(coalesce(p_limit, 20), 1), 20);
begin
  normalized_kind := case pg_catalog.lower(p_kind)
    when 'champion' then 'champion' when 'champions' then 'champion'
    when 'player' then 'player' when 'players' then 'player'
    when 'team' then 'team' when 'teams' then 'team' else null end;
  if active_release is null or normalized_kind is null
     or pg_catalog.octet_length(coalesce(p_kind, '')) not between 1 and 20
     or p_order is null or pg_catalog.octet_length(p_order) not between 1 and 30
     or p_order not in ('rank_asc','rank_desc','score_desc','score_asc','name_asc')
     or p_min_games is null or p_min_games not between 0 and 10000
     or p_limit is null
     or p_offset is null or p_offset not between 0 and 10000
     or pg_catalog.octet_length(coalesce(p_search,'')) > 100
     or not public.scryglass_text_array_is_bounded(p_patches, 20, 20)
     or not public.scryglass_text_array_is_bounded(p_regions, 20, 50)
     or not public.scryglass_text_array_is_bounded(p_leagues, 50)
     or not public.scryglass_text_array_is_bounded(p_tiers, 20)
     or not public.scryglass_text_array_is_bounded(p_roles, 10)
  then raise exception 'Scryglass tier row query is invalid'; end if;
  with eligible as (
    select * from public.scryglass_public_query_rows row
    where row.release_id=active_release and row.dataset='tier_rows'
      and row.kind=normalized_kind and coalesce(row.played_maps,0)>=p_min_games
      and (p_roles is null or row.role=any(p_roles))
      and (p_search is null or pg_catalog.btrim(p_search)=''
        or row.search_key like '%'||pg_catalog.lower(pg_catalog.btrim(p_search))||'%')
      and (
        (p_regions is not null
          and row.region=any(p_regions)
          and (p_leagues is null or row.league=any(p_leagues))
          and (p_tiers is null or row.tier=any(p_tiers)))
        or (p_regions is null and p_leagues is not null
          and row.league=any(p_leagues)
          and (p_tiers is null or row.tier=any(p_tiers)))
        or (p_regions is null and p_leagues is null and p_tiers is not null
          and row.region is null and row.league is null and row.tier=any(p_tiers))
        or (p_regions is null and p_leagues is null and p_tiers is null
          and row.region is null and row.league is null and row.tier is null)
      )
  ), filtered as (
    select * from eligible row
    where p_patches is null or row.patch=any(p_patches)
  ), latest_filtered as (
    select * from filtered
    where p_patches is not null or patch=(
      select candidate.patch from eligible candidate
      where candidate.patch ~ '^[0-9]{1,2}\.[0-9]{1,2}$'
      order by pg_catalog.split_part(candidate.patch,'.',1)::integer desc,
        pg_catalog.split_part(candidate.patch,'.',2)::integer desc
      limit 1
    )
  ), page as (
    select * from latest_filtered order by
      case when p_order='rank_asc' then rank end asc nulls last,
      case when p_order='rank_desc' then rank end desc nulls last,
      case when p_order='score_desc' then score end desc nulls last,
      case when p_order='score_asc' then score end asc nulls last,
      name asc,row_key asc limit bounded_limit offset p_offset
  )
  select pg_catalog.jsonb_build_object(
    'schema_version','scryglass:query-api:v1','release_id',active_release,
    'rows',coalesce((select pg_catalog.jsonb_agg(
      pg_catalog.jsonb_strip_nulls(pg_catalog.jsonb_build_object(
        'row_key',row_key,'kind',kind,'name',name,'patch',patch,'region',region,
        'league',league,'tier',tier,'role',role,'rank',rank,'score',score,
        'played_maps',played_maps,'payload',payload
      ))) from page),'[]'::jsonb),
    'limit',bounded_limit,'offset',p_offset,'total',(select pg_catalog.count(*) from latest_filtered)
  ) into result;
  return public.scryglass_bounded_query_result(result);
end;
$$;

create or replace function public.get_scryglass_tier_facets()
returns jsonb
language plpgsql
stable
security definer
set search_path = ''
set statement_timeout = '5s'
as $$
declare
  active_release text := public.scryglass_active_query_release_id();
  result jsonb;
begin
  if active_release is null then raise exception 'Scryglass query release is unavailable'; end if;
  with rows as (
    select * from public.scryglass_public_query_rows row
    where row.release_id=active_release and row.dataset='tier_rows'
  ), scopes as (
    select * from public.scryglass_public_query_rows row
    where row.release_id=active_release and row.dataset='tier_scopes'
  )
  select pg_catalog.jsonb_build_object(
    'schema_version','scryglass:query-api:v1','release_id',active_release,
    'options',pg_catalog.jsonb_build_object(
      'patches',coalesce((select pg_catalog.jsonb_agg(value order by value)
        from (select distinct patch value from rows where patch is not null limit 100) item),'[]'::jsonb),
      'roles',coalesce((select pg_catalog.jsonb_agg(value order by value)
        from (select distinct role value from rows where role is not null limit 10) item),'[]'::jsonb),
      'regions',coalesce((select pg_catalog.jsonb_agg(value order by value)
        from (select distinct region value from rows where region is not null limit 100) item),'[]'::jsonb),
      'leagues',coalesce((select pg_catalog.jsonb_agg(value order by value)
        from (select distinct league value from rows where league is not null limit 250) item),'[]'::jsonb),
      'tiers',coalesce((select pg_catalog.jsonb_agg(value order by value)
        from (select distinct tier value from rows where tier is not null limit 50) item),'[]'::jsonb),
      'tier_buckets',coalesce((select pg_catalog.jsonb_agg(value order by value)
        from (select distinct payload->>'tier_bucket' value from rows
          where payload->>'tier_bucket' is not null limit 50) item),'[]'::jsonb)
    ),
    'scopes',coalesce((select pg_catalog.jsonb_agg(
      pg_catalog.jsonb_strip_nulls(pg_catalog.jsonb_build_object(
        'scope_id',scope_id,'patch',patch,'role',role,'row_count',rank,
        'regions',payload->'regional_views',
        'response_matrix_available',exists(
          select 1 from public.scryglass_public_query_rows matrix
          where matrix.release_id=active_release and matrix.dataset='tier_matrix_rows'
            and matrix.scope_id=scopes.scope_id
        )
      )) order by patch,role
    ) from (select * from scopes order by patch desc,role limit 500) scopes),'[]'::jsonb)
  ) into result;
  return public.scryglass_bounded_query_result(result);
end;
$$;

create or replace function public.get_scryglass_tier_scope(
  p_patch text,
  p_role text default null,
  p_region text default null,
  p_league text default null,
  p_tier text default null,
  p_similarity_limit integer default 100
)
returns jsonb
language plpgsql
stable
security definer
set search_path = ''
set statement_timeout = '5s'
as $$
declare
  active_release text := public.scryglass_active_query_release_id();
  selected_scope public.scryglass_public_query_rows%rowtype;
  played_ids text[];
  selected_ids text[];
  result jsonb;
begin
  if active_release is null
     or pg_catalog.octet_length(coalesce(p_patch,'')) not between 1 and 20
     or pg_catalog.octet_length(coalesce(p_role,''))>20
     or pg_catalog.octet_length(coalesce(p_region,''))>50
     or pg_catalog.octet_length(coalesce(p_league,''))>100
     or pg_catalog.octet_length(coalesce(p_tier,''))>100
     or p_similarity_limit is null
     or p_similarity_limit not between 0 and 100
  then raise exception 'Scryglass tier scope query is invalid'; end if;
  if p_role is not null then
    select * into selected_scope from public.scryglass_public_query_rows scope
    where scope.release_id=active_release and scope.dataset='tier_scopes'
      and scope.patch=p_patch and scope.role=p_role limit 1;
    if not found then
      return public.scryglass_bounded_query_result(pg_catalog.jsonb_build_object(
        'schema_version','scryglass:query-api:v1','release_id',active_release,
        'scope',null,'rows','[]'::jsonb,'structural_similarity',null,
        'champion_images','{}'::jsonb
      ));
    end if;
  end if;
  select pg_catalog.array_agg(distinct row.champion_id)
  into played_ids
  from public.scryglass_public_query_rows row
  where row.release_id=active_release and row.dataset='tier_rows'
    and row.patch=p_patch and (p_role is null or row.role=p_role)
    and (
      (p_region is not null and row.region=p_region
        and (p_league is null or row.league=p_league)
        and (p_tier is null or row.tier=p_tier))
      or (p_region is null and p_league is not null and row.league=p_league
        and (p_tier is null or row.tier=p_tier))
      or (p_region is null and p_league is null and p_tier is not null
        and row.region is null and row.league is null and row.tier=p_tier)
      or (p_region is null and p_league is null and p_tier is null
        and row.region is null and row.league is null and row.tier is null)
    )
    and row.champion_id is not null;
  with candidates as (
    select champion.champion_id,
      pg_catalog.max(edge.score) best_similarity,
      case when champion.champion_id=any(coalesce(played_ids,array[]::text[])) then 0 else 1 end unpicked
    from public.scryglass_public_query_rows champion
    left join public.scryglass_public_query_rows edge
      on edge.release_id=champion.release_id and edge.dataset='tier_similarity_edges'
      and edge.champion_id=champion.champion_id
      and edge.reference_id=any(coalesce(played_ids,array[]::text[]))
    where p_role is not null
      and champion.release_id=active_release and champion.dataset='tier_similarity_champions'
    group by champion.champion_id
  ), chosen as (
    select champion_id from candidates
    order by unpicked asc, best_similarity desc nulls last, champion_id
    limit coalesce(pg_catalog.cardinality(played_ids),0)+p_similarity_limit
  ) select pg_catalog.array_agg(champion_id) into selected_ids from chosen;
  with tier_rows as (
    select * from public.scryglass_public_query_rows row
    where row.release_id=active_release and row.dataset='tier_rows'
      and row.patch=p_patch and (p_role is null or row.role=p_role)
      and (
        (p_region is not null and row.region=p_region
          and (p_league is null or row.league=p_league)
          and (p_tier is null or row.tier=p_tier))
        or (p_region is null and p_league is not null and row.league=p_league
          and (p_tier is null or row.tier=p_tier))
        or (p_region is null and p_league is null and p_tier is not null
          and row.region is null and row.league is null and row.tier=p_tier)
        or (p_region is null and p_league is null and p_tier is null
          and row.region is null and row.league is null and row.tier is null)
      )
    order by row.role,row.rank limit 800
  ), similarity_champions as (
    select * from public.scryglass_public_query_rows row
    where row.release_id=active_release and row.dataset='tier_similarity_champions'
      and row.champion_id=any(coalesce(selected_ids,array[]::text[]))
    order by row.ordinal
  )
  select pg_catalog.jsonb_build_object(
    'schema_version','scryglass:query-api:v1','release_id',active_release,
    'scope',case when p_role is null then null else selected_scope.payload end,
    'rows',coalesce((select pg_catalog.jsonb_agg(payload) from tier_rows),'[]'::jsonb),
    'structural_similarity',case when pg_catalog.cardinality(selected_ids)>0 then
      pg_catalog.jsonb_build_object(
        'schema_version',coalesce((select payload#>>'{library,schema_version}' from public.scryglass_public_query_rows
          where release_id=active_release and dataset='tier_similarity_champions' and payload?'library' limit 1),'scryglass:champion-structural-similarity:v1'),
        'source_atom_bridge_sha256',(select payload#>>'{library,source_atom_bridge_sha256}' from public.scryglass_public_query_rows
          where release_id=active_release and dataset='tier_similarity_champions' and payload?'library' limit 1),
        'minimum_similarity',(select payload#>'{library,minimum_similarity}' from public.scryglass_public_query_rows
          where release_id=active_release and dataset='tier_similarity_champions' and payload?'library' limit 1),
        'weights',(select payload#>'{library,weights}' from public.scryglass_public_query_rows
          where release_id=active_release and dataset='tier_similarity_champions' and payload?'library' limit 1),
        'champions',coalesce((select pg_catalog.jsonb_agg(payload-'library' order by ordinal) from similarity_champions),'[]'::jsonb),
        'similarity',coalesce((select pg_catalog.jsonb_agg(matrix order by left_ordinal) from (
          select left_champion.ordinal left_ordinal,
            pg_catalog.jsonb_agg(edge.score order by right_champion.ordinal) matrix
          from similarity_champions left_champion cross join similarity_champions right_champion
          join public.scryglass_public_query_rows edge
            on edge.release_id=active_release and edge.dataset='tier_similarity_edges'
            and edge.champion_id=left_champion.champion_id
            and edge.reference_id=right_champion.champion_id
          group by left_champion.ordinal
        ) matrices),'[]'::jsonb)
      ) else null end,
    'champion_images',coalesce((select pg_catalog.jsonb_object_agg(name,image_url)
      from public.scryglass_public_query_rows champion
      where champion.release_id=active_release and champion.dataset='champions'
        and champion.search_key in (
          select distinct tier.search_key
          from public.scryglass_public_query_rows tier
          where tier.release_id=active_release and tier.dataset='tier_rows'
            and tier.patch=p_patch and (p_role is null or tier.role=p_role)
            and (
              (p_region is not null and tier.region=p_region
                and (p_league is null or tier.league=p_league)
                and (p_tier is null or tier.tier=p_tier))
              or (p_region is null and p_league is not null and tier.league=p_league
                and (p_tier is null or tier.tier=p_tier))
              or (p_region is null and p_league is null and p_tier is not null
                and tier.region is null and tier.league is null and tier.tier=p_tier)
              or (p_region is null and p_league is null and p_tier is null
                and tier.region is null and tier.league is null and tier.tier is null)
            )
        )
        and champion.image_url is not null limit 250),'{}'::jsonb)
  ) into result;
  if p_role is not null then
    result:=pg_catalog.jsonb_set(result,'{scope,response_matrix}',coalesce((
      select pg_catalog.jsonb_build_object(
        'champions',pg_catalog.jsonb_agg(payload->'champion' order by ordinal),
        'edge_pp',pg_catalog.jsonb_agg(payload->'edge_pp' order by ordinal),
        'interval_low_pp',pg_catalog.jsonb_agg(payload->'interval_low_pp' order by ordinal),
        'interval_high_pp',pg_catalog.jsonb_agg(payload->'interval_high_pp' order by ordinal),
        'evidence',pg_catalog.jsonb_agg(payload->'evidence' order by ordinal),
        'effective_maps',pg_catalog.jsonb_agg(payload->'effective_maps' order by ordinal),
        'basis',pg_catalog.jsonb_agg(payload->'basis' order by ordinal),
        'grade_thresholds_pp',(select threshold.payload->'grade_thresholds_pp'
          from public.scryglass_public_query_rows threshold
          where threshold.release_id=active_release
            and threshold.dataset='tier_matrix_rows'
            and threshold.scope_id=selected_scope.scope_id
            and threshold.payload?'grade_thresholds_pp'
          order by threshold.ordinal limit 1)
      ) from public.scryglass_public_query_rows matrix
      where matrix.release_id=active_release and matrix.dataset='tier_matrix_rows'
        and matrix.scope_id=selected_scope.scope_id
    ),'{}'::jsonb),true);
  end if;
  return public.scryglass_bounded_query_result(result);
end;
$$;

create or replace function public.get_scryglass_player_profile(p_name text)
returns jsonb
language plpgsql
stable
security definer
set search_path = ''
set statement_timeout = '5s'
as $$
declare
  active_release text := public.scryglass_active_query_release_id();
  resolved_id text;
  player public.scryglass_public_query_rows%rowtype;
  result jsonb;
begin
  if active_release is null or pg_catalog.octet_length(coalesce(p_name,'')) not between 1 and 100
  then raise exception 'Scryglass player profile query is invalid'; end if;
  select coalesce(
    (select exact_player.player_id
      from public.scryglass_public_query_rows exact_player
      where exact_player.release_id=active_release
        and exact_player.dataset='players' and exact_player.name=p_name limit 1),
    (select alias.identity_id
      from public.scryglass_public_query_rows alias
      where alias.release_id=active_release and alias.dataset='aliases'
        and alias.kind='player'
        and alias.alias_key=pg_catalog.lower(pg_catalog.btrim(p_name)) limit 1),
    (select direct.player_id
      from public.scryglass_public_query_rows direct
      where direct.release_id=active_release and direct.dataset='players'
        and direct.search_key=pg_catalog.lower(pg_catalog.btrim(p_name)) limit 1)
  ) into resolved_id;
  select * into player from public.scryglass_public_query_rows row
  where row.release_id=active_release and row.dataset='players'
    and row.player_id=resolved_id limit 1;
  if not found then
    return public.scryglass_bounded_query_result(pg_catalog.jsonb_build_object(
      'schema_version','scryglass:query-api:v1','release_id',active_release,
      'row',null,'team_row',null,'standing',null,'champions','[]'::jsonb,
      'recent_games','[]'::jsonb,'champion_images','{}'::jsonb
    ));
  end if;
  with recent as (
    select game.* from public.scryglass_public_query_rows link
    join public.scryglass_public_query_rows game
      on game.release_id=link.release_id and game.dataset='games'
      and game.game_id=link.game_id
    where link.release_id=active_release and link.dataset='identity_games'
      and link.kind='player' and link.identity_id=player.player_id
    order by link.ordinal limit 10
  ), standing as (
    select
      1+pg_catalog.count(*) filter (where row.adjusted_rating>player.adjusted_rating
        and row.tier is not distinct from player.tier) tier_rank,
      pg_catalog.count(*) filter (where row.tier is not distinct from player.tier) tier_total,
      1+pg_catalog.count(*) filter (where row.adjusted_rating>player.adjusted_rating
        and row.role is not distinct from player.role) role_rank,
      pg_catalog.count(*) filter (where row.role is not distinct from player.role) role_total
    from public.scryglass_public_query_rows row
    where row.release_id=active_release and row.dataset='players'
      and row.adjusted_rating is not null
  )
  select pg_catalog.jsonb_build_object(
    'schema_version','scryglass:query-api:v1','release_id',active_release,
    'row',pg_catalog.jsonb_strip_nulls(pg_catalog.jsonb_build_object(
      'row_key',player.row_key,'player_id',player.player_id,'name',player.name,
      'role',player.role,'team',player.team,'league',player.league,'tier',player.tier,
      'active',player.active,'rating',player.rating,'adjusted_rating',player.adjusted_rating,
      'movement',player.movement,'games',player.games,'wins',player.wins,
      'win_rate',player.win_rate,'grade_a_games',player.grade_a_games,
      'grade_games',player.grade_games,'payload',player.payload
    )),
    'team_row',(select pg_catalog.jsonb_strip_nulls(pg_catalog.jsonb_build_object(
      'team_id',team_id,'name',name,'league',league,'tier',tier,'rating',rating,
      'adjusted_rating',adjusted_rating,'games',games,'wins',wins,'win_rate',win_rate,
      'payload',payload
    )) from public.scryglass_public_query_rows team
      where team.release_id=active_release and team.dataset='teams'
        and team.team_id=player.team_id limit 1),
    'standing',(select pg_catalog.to_jsonb(standing) from standing),
    'champions',coalesce((select pg_catalog.jsonb_agg(
      pg_catalog.jsonb_build_object('champion_id',champion_id,'champion',champion,
        'games',games,'wins',wins,'win_rate',win_rate,'score',score,'payload',payload)
      order by games desc,champion
    ) from (select * from public.scryglass_public_query_rows pc
      where pc.release_id=active_release and pc.dataset='player_champions'
        and pc.player_id=player.player_id order by games desc limit 20) pc),'[]'::jsonb),
    'recent_games',coalesce((select pg_catalog.jsonb_agg(
      pg_catalog.jsonb_strip_nulls(pg_catalog.jsonb_build_object(
        'game_id',game_id,'played_at',played_at,'year',year,'league',league,
        'tier',tier,'blue_team',blue_team,'red_team',red_team,'blue_win',blue_win,
        'champions',champions,'payload',payload
      )) order by played_at desc,game_id desc
    ) from recent),'[]'::jsonb),
    'champion_images',coalesce((select pg_catalog.jsonb_object_agg(champion.name,champion.image_url)
      from public.scryglass_public_query_rows champion
      where champion.release_id=active_release and champion.dataset='champions'
        and champion.image_url is not null and champion.name in (
          select pg_catalog.unnest(game.champions) from recent game
        ) limit 50),'{}'::jsonb)
  ) into result;
  return public.scryglass_bounded_query_result(result);
end;
$$;

create or replace function public.get_scryglass_team_profile(p_name text)
returns jsonb
language plpgsql
stable
security definer
set search_path = ''
set statement_timeout = '5s'
as $$
declare
  active_release text := public.scryglass_active_query_release_id();
  resolved_id text;
  team public.scryglass_public_query_rows%rowtype;
  result jsonb;
begin
  if active_release is null or pg_catalog.octet_length(coalesce(p_name,'')) not between 1 and 100
  then raise exception 'Scryglass team profile query is invalid'; end if;
  select coalesce(
    (select exact_team.team_id
      from public.scryglass_public_query_rows exact_team
      where exact_team.release_id=active_release
        and exact_team.dataset='teams' and exact_team.name=p_name limit 1),
    (select alias.identity_id
      from public.scryglass_public_query_rows alias
      where alias.release_id=active_release and alias.dataset='aliases'
        and alias.kind='team'
        and alias.alias_key=pg_catalog.lower(pg_catalog.btrim(p_name)) limit 1),
    (select direct.team_id
      from public.scryglass_public_query_rows direct
      where direct.release_id=active_release and direct.dataset='teams'
        and direct.search_key=pg_catalog.lower(pg_catalog.btrim(p_name)) limit 1)
  ) into resolved_id;
  select * into team from public.scryglass_public_query_rows row
  where row.release_id=active_release and row.dataset='teams' and row.team_id=resolved_id limit 1;
  if not found then
    return public.scryglass_bounded_query_result(pg_catalog.jsonb_build_object(
      'schema_version','scryglass:query-api:v1','release_id',active_release,
      'row',null,'standing',null,'roster','[]'::jsonb,
      'recent_games','[]'::jsonb,'champion_images','{}'::jsonb
    ));
  end if;
  with recent as (
    select game.* from public.scryglass_public_query_rows link
    join public.scryglass_public_query_rows game
      on game.release_id=link.release_id and game.dataset='games' and game.game_id=link.game_id
    where link.release_id=active_release and link.dataset='identity_games'
      and link.kind='team' and link.identity_id=team.team_id order by link.ordinal limit 10
  ), roster as (
    select player.*,
      1+(select pg_catalog.count(*) from public.scryglass_public_query_rows peer
        where peer.release_id=active_release and peer.dataset='players'
          and peer.role is not distinct from player.role
          and peer.adjusted_rating>player.adjusted_rating) role_rank,
      (select pg_catalog.count(*) from public.scryglass_public_query_rows peer
        where peer.release_id=active_release and peer.dataset='players'
          and peer.role is not distinct from player.role) role_total
    from public.scryglass_public_query_rows player
    where player.release_id=active_release and player.dataset='players'
      and player.team_id=team.team_id and player.active is true
    order by player.adjusted_rating desc nulls last limit 5
  )
  select pg_catalog.jsonb_build_object(
    'schema_version','scryglass:query-api:v1','release_id',active_release,
    'row',pg_catalog.jsonb_strip_nulls(pg_catalog.jsonb_build_object(
      'row_key',team.row_key,'team_id',team.team_id,'name',team.name,
      'league',team.league,'tier',team.tier,'active',team.active,'rating',team.rating,
      'adjusted_rating',team.adjusted_rating,'movement',team.movement,
      'games',team.games,'wins',team.wins,'win_rate',team.win_rate,'payload',team.payload
    )),
    'standing',pg_catalog.jsonb_build_object(
      'tier_rank',1+(select pg_catalog.count(*) from public.scryglass_public_query_rows peer
        where peer.release_id=active_release and peer.dataset='teams'
          and peer.tier is not distinct from team.tier
          and peer.adjusted_rating>team.adjusted_rating),
      'tier_total',(select pg_catalog.count(*) from public.scryglass_public_query_rows peer
        where peer.release_id=active_release and peer.dataset='teams'
          and peer.tier is not distinct from team.tier)
    ),
    'roster',coalesce((select pg_catalog.jsonb_agg(pg_catalog.jsonb_strip_nulls(
      pg_catalog.jsonb_build_object('player_id',player_id,'name',name,'role',role,
        'rating',rating,'adjusted_rating',adjusted_rating,'games',games,
        'win_rate',win_rate,'role_rank',role_rank,'role_total',role_total,'payload',payload)
    ) order by adjusted_rating desc nulls last) from roster),'[]'::jsonb),
    'recent_games',coalesce((select pg_catalog.jsonb_agg(
      pg_catalog.jsonb_strip_nulls(pg_catalog.jsonb_build_object(
        'game_id',game_id,'played_at',played_at,'year',year,'league',league,
        'tier',tier,'blue_team',blue_team,'red_team',red_team,'blue_win',blue_win,
        'champions',champions,'payload',payload
      )) order by played_at desc,game_id desc
    ) from recent),'[]'::jsonb),
    'champion_images',coalesce((select pg_catalog.jsonb_object_agg(champion.name,champion.image_url)
      from public.scryglass_public_query_rows champion
      where champion.release_id=active_release and champion.dataset='champions'
        and champion.image_url is not null and champion.name in (
          select pg_catalog.unnest(game.champions) from recent game
        ) limit 50),'{}'::jsonb)
  ) into result;
  return public.scryglass_bounded_query_result(result);
end;
$$;

create or replace function public.get_scryglass_matches(
  p_leagues text[] default null,
  p_tiers text[] default null,
  p_team text default null,
  p_champion text default null,
  p_years integer[] default null,
  p_from text default null,
  p_to text default null,
  p_before text default null,
  p_limit integer default 20,
  p_offset integer default 0
)
returns jsonb
language plpgsql
stable
security definer
set search_path = ''
set statement_timeout = '5s'
as $$
declare
  active_release text := public.scryglass_active_query_release_id();
  resolved_team text;
  resolved_champion text;
  result jsonb;
  bounded_limit integer := least(greatest(coalesce(p_limit,20),1),20);
begin
  if active_release is null
     or p_limit is null
     or p_offset is null or p_offset not between 0 and 10000
     or pg_catalog.octet_length(coalesce(p_team,''))>100
     or pg_catalog.octet_length(coalesce(p_champion,''))>100
     or not public.scryglass_text_array_is_bounded(p_leagues, 50)
     or not public.scryglass_text_array_is_bounded(p_tiers, 20)
     or coalesce(pg_catalog.cardinality(p_years),0)>20
     or exists(select 1 from pg_catalog.unnest(p_years) year where year not between 1900 and 2100)
     or (p_from is not null and (
       pg_catalog.octet_length(p_from) not between 10 and 40
       or p_from !~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}([T ][0-9:.+Z-]{1,30})?$'
     ))
     or (p_to is not null and (
       pg_catalog.octet_length(p_to) not between 10 and 40
       or p_to !~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}([T ][0-9:.+Z-]{1,30})?$'
     ))
     or (p_before is not null and (
       pg_catalog.octet_length(p_before) not between 10 and 40
       or p_before !~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}([T ][0-9:.+Z-]{1,30})?$'
     ))
  then raise exception 'Scryglass match query is invalid'; end if;
  if p_team is not null then
    select coalesce(
      (select exact_team.team_id
        from public.scryglass_public_query_rows exact_team
        where exact_team.release_id=active_release
          and exact_team.dataset='teams' and exact_team.name=p_team limit 1),
      (select alias.identity_id
        from public.scryglass_public_query_rows alias
        where alias.release_id=active_release and alias.dataset='aliases'
          and alias.kind='team'
          and alias.alias_key=pg_catalog.lower(pg_catalog.btrim(p_team)) limit 1),
      (select direct.team_id
        from public.scryglass_public_query_rows direct
        where direct.release_id=active_release and direct.dataset='teams'
          and direct.search_key=pg_catalog.lower(pg_catalog.btrim(p_team)) limit 1)
    ) into resolved_team;
  end if;
  if p_champion is not null then
    select coalesce(
      (select exact_champion.champion_id
        from public.scryglass_public_query_rows exact_champion
        where exact_champion.release_id=active_release
          and exact_champion.dataset='champions'
          and exact_champion.name=p_champion limit 1),
      (select alias.identity_id
        from public.scryglass_public_query_rows alias
        where alias.release_id=active_release and alias.dataset='aliases'
          and alias.kind='champion'
          and alias.alias_key=pg_catalog.lower(pg_catalog.btrim(p_champion)) limit 1),
      (select direct.champion_id
        from public.scryglass_public_query_rows direct
        where direct.release_id=active_release and direct.dataset='champions'
          and direct.search_key=pg_catalog.lower(pg_catalog.btrim(p_champion)) limit 1)
    ) into resolved_champion;
  end if;
  with filtered as (
    select row.* from public.scryglass_public_query_rows row
    where row.release_id=active_release and row.dataset='games'
      and (p_leagues is null or row.league=any(p_leagues))
      and (p_tiers is null or row.tier=any(p_tiers))
      and (p_years is null or row.year=any(p_years))
      and (p_team is null or row.blue_team_id=resolved_team or row.red_team_id=resolved_team)
      and (p_champion is null or exists(
        select 1 from public.scryglass_public_query_rows champion
        where champion.release_id=active_release and champion.dataset='champions'
          and champion.champion_id=resolved_champion and champion.name=any(row.champions)
      ))
      and (p_from is null or row.played_at>=p_from)
      and (p_to is null or row.played_at<p_to)
      and (p_before is null or row.played_at<p_before)
  ), page as (
    select * from filtered order by played_at desc,game_id desc
    limit bounded_limit offset p_offset
  )
  select pg_catalog.jsonb_build_object(
    'schema_version','scryglass:query-api:v1','release_id',active_release,
    'rows',coalesce((select pg_catalog.jsonb_agg(pg_catalog.jsonb_strip_nulls(
      pg_catalog.jsonb_build_object('game_id',game_id,'played_at',played_at,'year',year,
        'league',league,'tier',tier,'blue_team',blue_team,'red_team',red_team,
        'blue_win',blue_win,'champions',champions,'payload',payload)
    ) order by played_at desc,game_id desc) from page),'[]'::jsonb),
    'champion_images',coalesce((select pg_catalog.jsonb_object_agg(champion.name,champion.image_url)
      from (
        select image.name,image.image_url
        from public.scryglass_public_query_rows image
        where image.release_id=active_release and image.dataset='champions'
          and image.image_url is not null and image.name in (
            select pg_catalog.unnest(game.champions) from page game
          )
        order by image.name limit 200
      ) champion),'{}'::jsonb),
    'limit',bounded_limit,'offset',p_offset,'total',(select pg_catalog.count(*) from filtered)
  ) into result;
  return public.scryglass_bounded_query_result(result);
end;
$$;

create or replace function public.get_scryglass_match(p_game_id text)
returns jsonb
language plpgsql
stable
security definer
set search_path = ''
set statement_timeout = '5s'
as $$
declare
  active_release text := public.scryglass_active_query_release_id();
  game public.scryglass_public_query_rows%rowtype;
  result jsonb;
begin
  if active_release is null or pg_catalog.octet_length(coalesce(p_game_id,'')) not between 1 and 200
  then raise exception 'Scryglass match detail query is invalid'; end if;
  select * into game from public.scryglass_public_query_rows row
  where row.release_id=active_release and row.dataset='games' and row.game_id=p_game_id limit 1;
  if not found then
    return public.scryglass_bounded_query_result(pg_catalog.jsonb_build_object(
      'schema_version','scryglass:query-api:v1','release_id',active_release,
      'row',null,'champion_images','{}'::jsonb
    ));
  end if;
  select pg_catalog.jsonb_build_object(
    'schema_version','scryglass:query-api:v1','release_id',active_release,
    'row',pg_catalog.jsonb_strip_nulls(pg_catalog.jsonb_build_object(
      'game_id',game.game_id,'played_at',game.played_at,'year',game.year,
      'league',game.league,'tier',game.tier,'blue_team',game.blue_team,
      'red_team',game.red_team,'blue_win',game.blue_win,'champions',game.champions,
      'payload',game.payload
    )),
    'champion_images',coalesce((select pg_catalog.jsonb_object_agg(champion.name,champion.image_url)
      from public.scryglass_public_query_rows champion
      where champion.release_id=active_release and champion.dataset='champions'
        and champion.image_url is not null and champion.name=any(game.champions) limit 50),'{}'::jsonb)
  ) into result;
  return public.scryglass_bounded_query_result(result);
end;
$$;

create or replace function public.get_scryglass_match_facets(
  p_tiers text[] default null,
  p_years integer[] default null,
  p_from text default null,
  p_to text default null,
  p_team text default null
)
returns jsonb
language plpgsql
stable
security definer
set search_path = ''
set statement_timeout = '5s'
as $$
declare
  active_release text := public.scryglass_active_query_release_id();
  resolved_team text;
  result jsonb;
begin
  if active_release is null
    or (p_from is not null and (
      pg_catalog.octet_length(p_from) not between 10 and 40
      or p_from!~'^[0-9]{4}-[0-9]{2}-[0-9]{2}([T ][0-9:.+Z-]{1,30})?$'
    ))
    or (p_to is not null and (
      pg_catalog.octet_length(p_to) not between 10 and 40
      or p_to!~'^[0-9]{4}-[0-9]{2}-[0-9]{2}([T ][0-9:.+Z-]{1,30})?$'
    ))
    or pg_catalog.octet_length(coalesce(p_team,''))>100
    or not public.scryglass_text_array_is_bounded(p_tiers,20)
    or coalesce(pg_catalog.cardinality(p_years),0)>20
    or exists(select 1 from pg_catalog.unnest(p_years) year where year not between 1900 and 2100)
  then raise exception 'Scryglass match facet query is invalid'; end if;
  if p_team is not null then
    select coalesce(
      (select exact_team.team_id
        from public.scryglass_public_query_rows exact_team
        where exact_team.release_id=active_release
          and exact_team.dataset='teams' and exact_team.name=p_team limit 1),
      (select alias.identity_id
        from public.scryglass_public_query_rows alias
        where alias.release_id=active_release and alias.dataset='aliases'
          and alias.kind='team'
          and alias.alias_key=pg_catalog.lower(pg_catalog.btrim(p_team)) limit 1),
      (select direct.team_id
        from public.scryglass_public_query_rows direct
        where direct.release_id=active_release and direct.dataset='teams'
          and direct.search_key=pg_catalog.lower(pg_catalog.btrim(p_team)) limit 1)
    ) into resolved_team;
  end if;
  with rows as (
    select * from public.scryglass_public_query_rows row
    where row.release_id=active_release and row.dataset='games'
      and (p_tiers is null or row.tier=any(p_tiers))
      and (p_years is null or row.year=any(p_years))
      and (p_from is null or row.played_at>=p_from) and (p_to is null or row.played_at<p_to)
      and (p_team is null or row.blue_team_id=resolved_team or row.red_team_id=resolved_team)
  )
  select pg_catalog.jsonb_build_object(
    'schema_version','scryglass:query-api:v1','release_id',active_release,
    'tiers',coalesce((select pg_catalog.jsonb_agg(value order by value) from
      (select distinct tier value from rows where tier is not null limit 50)item),'[]'::jsonb),
    'years',coalesce((select pg_catalog.jsonb_agg(value order by value) from
      (select distinct year value from rows where year is not null limit 20)item),'[]'::jsonb),
    'months',coalesce((select pg_catalog.jsonb_agg(value order by value) from
      (select distinct pg_catalog.substr(played_at,1,7) value from rows limit 120)item),'[]'::jsonb),
    'teams',coalesce((select pg_catalog.jsonb_agg(value order by value) from
      (select distinct value from (select blue_team value from rows union select red_team value from rows)all_teams
        where value is not null limit 1000)item),'[]'::jsonb),
    'leagues',coalesce((select pg_catalog.jsonb_agg(value order by value) from
      (select distinct league value from rows where league is not null limit 250)item),'[]'::jsonb)
  ) into result;
  return public.scryglass_bounded_query_result(result);
end;
$$;

create or replace function public.get_scryglass_query_status()
returns jsonb
language sql
stable
security definer
set search_path = ''
set statement_timeout = '5s'
as $$
  select public.scryglass_bounded_query_result(pg_catalog.jsonb_build_object(
    'schema_version','scryglass:query-api:v1','release_id',release.release_id,
    'datasets',coalesce((select pg_catalog.jsonb_object_agg(
      receipt.dataset,pg_catalog.jsonb_build_object('rows',receipt.row_count)
      order by receipt.dataset
    ) from public.scryglass_public_query_receipts receipt
      where receipt.release_id=release.release_id),'{}'::jsonb)
  )) from public.scryglass_public_releases release where release.status='active' limit 1
$$;

create or replace function public.get_scryglass_query_entities()
returns jsonb
language plpgsql
stable
security definer
set search_path = ''
set statement_timeout = '5s'
as $$
declare
  active_release text := public.scryglass_active_query_release_id();
  result jsonb;
begin
  if active_release is null then raise exception 'Scryglass query release is unavailable'; end if;
  select pg_catalog.jsonb_build_object(
    'schema_version','scryglass:query-api:v1','release_id',active_release,
    'players',coalesce((select pg_catalog.jsonb_agg(name order by name)
      from (select name from public.scryglass_public_query_rows where release_id=active_release and dataset='players' order by name limit 5000) item),'[]'::jsonb),
    'teams',coalesce((select pg_catalog.jsonb_agg(name order by name)
      from (select name from public.scryglass_public_query_rows where release_id=active_release and dataset='teams' order by name limit 1000) item),'[]'::jsonb),
    'champions',coalesce((select pg_catalog.jsonb_agg(name order by name)
      from (select name from public.scryglass_public_query_rows where release_id=active_release and dataset='champions' order by name limit 500) item),'[]'::jsonb),
    'leagues',coalesce((select pg_catalog.jsonb_agg(value order by value) from
      (select distinct league value from public.scryglass_public_query_rows where release_id=active_release and league is not null order by league limit 250)item),'[]'::jsonb),
    'aliases',coalesce((select pg_catalog.jsonb_agg(pg_catalog.jsonb_build_object(
      'kind',alias.kind,'alias',alias.alias_key,'name',canonical.name
    ) order by alias.kind,alias.alias_key) from public.scryglass_public_query_rows alias
      join public.scryglass_public_query_rows canonical
        on canonical.release_id=alias.release_id
        and canonical.dataset=case alias.kind when 'player' then 'players'
          when 'team' then 'teams' when 'champion' then 'champions' end
        and coalesce(canonical.player_id,canonical.team_id,canonical.champion_id)=alias.identity_id
      where alias.release_id=active_release and alias.dataset='aliases'
        and alias.alias_key<>canonical.search_key
      limit 5000),'[]'::jsonb)
  ) into result;
  return public.scryglass_bounded_query_result(result);
end;
$$;

-- Phase 1 exposes a fixed manifest projection with a minimal query marker.
-- Legacy JSONB assets remain available only through the parsed-data RPC below.
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
        'tier',pg_catalog.jsonb_build_object('status','available','as_of',pg_catalog.to_jsonb(candidate.source_as_of)),
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
          from public.scryglass_public_assets asset where asset.release_id=candidate.release_id
            and asset.path<>'features/draft_records.json'),
        'total_files',(select pg_catalog.to_jsonb(pg_catalog.count(*))
          from public.scryglass_public_assets asset where asset.release_id=candidate.release_id
            and asset.path<>'features/draft_records.json'),
        'files',coalesce((select pg_catalog.jsonb_agg(pg_catalog.jsonb_build_object(
          'path',asset.path,'bytes',asset.bytes,'sha256',asset.sha256) order by asset.path)
          from public.scryglass_public_assets asset where asset.release_id=candidate.release_id
            and asset.path<>'features/draft_records.json'),'[]'::jsonb),
        'release',pg_catalog.jsonb_build_object(
          'release_id',candidate.release_id,
          'tier_list_version',candidate.manifest#>'{release,tier_list_version}',
          'artifact_hashes',coalesce((select pg_catalog.jsonb_object_agg(
            asset.path,asset.sha256 order by asset.path)
            from public.scryglass_public_assets asset where asset.release_id=candidate.release_id
              and asset.path<>'features/draft_records.json'),'{}'::jsonb)
        )
      )
    ) manifest
  from public.scryglass_public_releases candidate
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
    and asset.path<>'features/draft_records.json'
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

create or replace function public.get_scryglass_active_inline_asset(
  p_release_id text,
  p_path text
)
returns table (
  release_id text,
  path text,
  body jsonb,
  source_bytes bigint,
  source_sha256 text,
  content_type text
)
language sql
stable
security definer
set search_path = ''
set statement_timeout = '5s'
as $$
  select asset.release_id,asset.path,asset.body,asset.bytes,
    asset.sha256,asset.content_type
  from public.scryglass_public_assets asset
  join public.scryglass_public_releases release
    on release.release_id=asset.release_id
  where release.status='active'
    and pg_catalog.octet_length(p_release_id) between 1 and 30
    and p_release_id~'^v[0-9]{4}\.[0-9]{2}\.[0-9]{2}\.[0-9]{6}$'
    and pg_catalog.octet_length(p_path) between 1 and 100
    and asset.release_id=p_release_id and asset.path=p_path
    and asset.path<>'features/draft_records.json'
    and asset.body is not null and asset.storage_path is null
    and pg_catalog.jsonb_typeof(asset.body) in ('object','array')
    and not public.scryglass_json_has_draft_fields(asset.body)
    and asset.bytes between 2 and 125829120
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
    and pg_catalog.octet_length(pg_catalog.jsonb_build_object(
      'release_id',asset.release_id,'path',asset.path,'body',asset.body,
      'source_bytes',asset.bytes,'source_sha256',asset.sha256,
      'content_type',asset.content_type
    )::text)<=500000
$$;

-- Final effective asset contract. PUBLIC_ASSET_ALLOWLIST_V1
alter table public.scryglass_public_assets
  drop constraint if exists scryglass_public_assets_path_check;
alter table public.scryglass_public_assets
  add constraint scryglass_public_assets_path_check check (path in (
    'features/ratings_snapshot.json','features/player_ratings_snapshot.json',
    'features/team_records.json','features/team_weekly_ranks.json',
    'features/player_records.json','features/player_champion_records.json',
    'features/profile_records.json','features/match_index.json',
    'features/match_records_2025.json','features/match_records_2025_q1.json',
    'features/match_records_2025_q2.json','features/match_records_2025_q3.json',
    'features/match_records_2025_q4.json','features/match_records_2026.json',
    'features/match_records_2026_q1.json','features/match_records_2026_q2.json',
    'features/match_records_2026_q3.json','features/match_records_2026_q4.json',
    'features/player_weekly_ranks.json','features/player_metadata.json',
    'features/schedule.json','features/leaderboards.json',
    'features/draft_records.json','rankings/tierlists.json',
    'rankings/tierlists-latest.json'
  ));

-- Reapply the complete integrity contract here. Production has already run
-- the earlier hardening migration, so editing that file cannot upgrade it.
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
  if release_manifest ->> 'pack_id' <> p_release_id
     or release_manifest #>> '{release,release_id}' <> p_release_id
     or pg_catalog.jsonb_typeof(release_manifest -> 'files') <> 'array'
     or pg_catalog.jsonb_typeof(
       release_manifest #> '{release,artifact_hashes}'
     ) <> 'object'
     or release_manifest #>> '{draft_authority,schema_version}'
       is distinct from 'scryglass:draft-authority:v1'
     or release_manifest #>> '{draft_authority,status}'
       is distinct from 'unavailable'
     or release_manifest #>> '{draft_authority,release_id}'
       is distinct from p_release_id
  then
    raise exception 'Scryglass release binding is invalid';
  end if;

  select pg_catalog.count(*)
  into draft_asset_count
  from public.scryglass_public_assets asset
  where asset.release_id = p_release_id
    and asset.path = 'features/draft_records.json';
  if draft_asset_count <> 0 then
    raise exception 'Scryglass release contains unavailable draft data';
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

create or replace function public.guard_scryglass_public_asset_mutation()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $$
begin
  perform pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtext('scryglass-public-release'));
  if tg_op='UPDATE' and (
    new.release_id is distinct from old.release_id or new.path is distinct from old.path
  ) then raise exception 'Scryglass asset identity is immutable'; end if;
  if tg_op='DELETE' then
    if current_user='scryglass_release_retention_owner'
       or exists(select 1 from public.scryglass_public_releases release
      where release.release_id=old.release_id and release.status='superseded') then return old; end if;
    raise exception 'Scryglass assets can only leave with a superseded release';
  end if;
  if not exists(select 1 from public.scryglass_public_releases release
    where release.release_id=new.release_id and release.status='staging')
     or (tg_op='UPDATE' and not exists(select 1 from public.scryglass_public_releases release
       where release.release_id=old.release_id and release.status='staging'))
  then raise exception 'Scryglass assets are immutable outside staging'; end if;
  return new;
end;
$$;

create or replace function public.guard_scryglass_public_release_mutation()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $$
declare
  transition_authorized boolean := current_user='scryglass_release_transition_owner';
begin
  perform pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtext('scryglass-public-release'));
  if tg_op='INSERT' then
    if new.status<>'staging' then raise exception 'Scryglass releases must begin in staging'; end if;
    return new;
  end if;
  if tg_op='DELETE' then
    if old.status<>'superseded' then raise exception 'Only superseded Scryglass releases can be deleted'; end if;
    return old;
  end if;
  if new.release_id is distinct from old.release_id
     or new.created_at is distinct from old.created_at
     or (new.activated_at is distinct from old.activated_at and not transition_authorized)
     or (old.status<>'staging' and (
       new.manifest is distinct from old.manifest or new.source_as_of is distinct from old.source_as_of
     ))
  then raise exception 'Scryglass release metadata is immutable'; end if;
  if new.status is distinct from old.status and not transition_authorized
  then raise exception 'Scryglass release status changes require the transition RPC'; end if;
  return new;
end;
$$;

do $role$
begin
  if not exists (
    select 1 from pg_catalog.pg_roles
    where rolname='scryglass_release_transition_owner'
  ) then
    create role scryglass_release_transition_owner
      nologin noinherit nosuperuser nocreatedb nocreaterole bypassrls;
  end if;
  if not exists (
    select 1 from pg_catalog.pg_roles
    where rolname='scryglass_release_retention_owner'
  ) then
    create role scryglass_release_retention_owner
      nologin noinherit nosuperuser nocreatedb nocreaterole bypassrls;
  end if;
end
$role$;

-- The migration runner owns these helper roles during the transition. This
-- grant lets it change function ownership in a clean Supabase replay. The
-- roles stay non-login and cannot be assumed by public callers.
do $grant$
begin
  execute format(
    'grant %I to %I',
    'scryglass_release_transition_owner',
    current_user
  );
  execute format(
    'grant %I to %I',
    'scryglass_release_retention_owner',
    current_user
  );
end
$grant$;

grant usage on schema public to scryglass_release_transition_owner;
grant create on schema public to scryglass_release_transition_owner;
grant select on public.scryglass_public_releases
  to scryglass_release_transition_owner;
grant select on public.scryglass_public_assets
  to scryglass_release_transition_owner;
grant update(status,activated_at) on public.scryglass_public_releases
  to scryglass_release_transition_owner;
alter function public.assert_scryglass_public_release_integrity(text)
  security definer;
alter function public.assert_scryglass_query_release(text)
  security definer;
grant execute on function public.assert_scryglass_public_release_integrity(text)
  to scryglass_release_transition_owner;
grant execute on function public.assert_scryglass_query_release(text)
  to scryglass_release_transition_owner;

grant usage on schema public to scryglass_release_retention_owner;
grant create on schema public to scryglass_release_retention_owner;
grant select,delete on public.scryglass_public_releases
  to scryglass_release_retention_owner;
grant select on public.scryglass_public_assets
  to scryglass_release_retention_owner;
grant select,insert,delete on public.scryglass_storage_cleanup
  to scryglass_release_retention_owner;

create or replace function public.prune_scryglass_public_releases_v2(
  p_keep integer default 3
)
returns jsonb
language plpgsql
security definer
set search_path = ''
as $$
declare
  removed_releases text[];
  queued_storage_paths text[];
begin
  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtext('scryglass-public-release')
  );
  if p_keep is null or p_keep < 1 or p_keep > 10 then
    raise exception 'Scryglass retention must be between 1 and 10 releases';
  end if;
  select coalesce(pg_catalog.array_agg(release.release_id), array[]::text[])
  into removed_releases
  from public.scryglass_public_releases release
  where release.status = 'superseded'
    and release.release_id not in (
      select retained.release_id
      from public.scryglass_public_releases retained
      where retained.status = 'superseded'
      order by retained.activated_at desc nulls last, retained.created_at desc
      limit greatest(p_keep - 1, 0)
    );
  insert into public.scryglass_storage_cleanup (storage_path, release_id)
  select asset.storage_path, asset.release_id
  from public.scryglass_public_assets asset
  where asset.release_id = any(removed_releases)
    and asset.storage_path is not null
  on conflict (storage_path) do nothing;
  delete from public.scryglass_public_releases release
  where release.release_id = any(removed_releases);
  select coalesce(
    pg_catalog.array_agg(cleanup.storage_path order by cleanup.storage_path),
    array[]::text[]
  )
  into queued_storage_paths
  from public.scryglass_storage_cleanup cleanup;
  return pg_catalog.jsonb_build_object(
    'deleted_count', pg_catalog.cardinality(removed_releases),
    'release_ids', pg_catalog.to_jsonb(removed_releases),
    'storage_paths', pg_catalog.to_jsonb(queued_storage_paths)
  );
end;
$$;

create or replace function public.ack_scryglass_storage_cleanup(
  p_storage_paths text[]
)
returns integer
language plpgsql
security definer
set search_path = ''
as $$
declare
  deleted_count integer;
begin
  if p_storage_paths is null
     or pg_catalog.cardinality(p_storage_paths) > 500
     or exists (
       select 1
       from pg_catalog.unnest(p_storage_paths) item(storage_path)
       where item.storage_path is null
          or pg_catalog.octet_length(item.storage_path) not between 1 and 220
          or item.storage_path !~
            '^v[0-9]{4}\.[0-9]{2}\.[0-9]{2}\.[0-9]{6}/'
     )
  then
    raise exception 'Scryglass Storage cleanup acknowledgement is invalid';
  end if;
  delete from public.scryglass_storage_cleanup cleanup
  where cleanup.storage_path = any(p_storage_paths);
  get diagnostics deleted_count = row_count;
  return deleted_count;
end;
$$;

alter function public.prune_scryglass_public_releases_v2(integer)
  owner to scryglass_release_retention_owner;
alter function public.ack_scryglass_storage_cleanup(text[])
  owner to scryglass_release_retention_owner;

create or replace function public.activate_scryglass_public_release(p_release_id text)
returns jsonb
language plpgsql
security definer
set search_path = ''
as $$
declare previous_release_id text;
begin
  perform pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtext('scryglass-public-release'));
  perform public.assert_scryglass_public_release_integrity(p_release_id);
  perform public.assert_scryglass_query_release(p_release_id);
  if not exists(select 1 from public.scryglass_public_releases where release_id=p_release_id and status in ('staging','active'))
  then raise exception 'Scryglass release is not ready for activation'; end if;
  select release_id into previous_release_id from public.scryglass_public_releases
    where status='active' and release_id<>p_release_id limit 1;
  update public.scryglass_public_releases set status='superseded'
    where status='active' and release_id<>p_release_id;
  update public.scryglass_public_releases set status='active',activated_at=now()
    where release_id=p_release_id;
  return pg_catalog.jsonb_build_object('status','active','release_id',p_release_id,
    'previous_release_id',previous_release_id);
end;
$$;

create or replace function public.restore_scryglass_public_release(p_release_id text)
returns jsonb
language plpgsql
security definer
set search_path = ''
as $$
declare replaced_release_id text;
begin
  perform pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtext('scryglass-public-release'));
  perform public.assert_scryglass_public_release_integrity(p_release_id);
  perform public.assert_scryglass_query_release(p_release_id);
  if not exists(select 1 from public.scryglass_public_releases where release_id=p_release_id and status in ('active','superseded'))
  then raise exception 'Scryglass rollback release is unavailable'; end if;
  select release_id into replaced_release_id from public.scryglass_public_releases
    where status='active' and release_id<>p_release_id limit 1;
  update public.scryglass_public_releases set status='superseded'
    where status='active' and release_id<>p_release_id;
  update public.scryglass_public_releases set status='active',activated_at=now()
    where release_id=p_release_id;
  return pg_catalog.jsonb_build_object('status','restored','release_id',p_release_id,
    'replaced_release_id',replaced_release_id);
end;
$$;

alter function public.activate_scryglass_public_release(text)
  owner to scryglass_release_transition_owner;
alter function public.restore_scryglass_public_release(text)
  owner to scryglass_release_transition_owner;

revoke create on schema public
  from scryglass_release_transition_owner, scryglass_release_retention_owner;

drop trigger if exists guard_scryglass_public_asset_mutation on public.scryglass_public_assets;
create trigger guard_scryglass_public_asset_mutation
before insert or update or delete on public.scryglass_public_assets
for each row execute function public.guard_scryglass_public_asset_mutation();
drop trigger if exists guard_scryglass_public_release_mutation on public.scryglass_public_releases;
create trigger guard_scryglass_public_release_mutation
before insert or update or delete on public.scryglass_public_releases
for each row execute function public.guard_scryglass_public_release_mutation();

revoke all on public.scryglass_public_query_rows
  from public,anon,authenticated,service_role;
revoke all on public.scryglass_public_query_receipts
  from public,anon,authenticated,service_role;
revoke delete on public.scryglass_public_releases from service_role;
revoke all on public.scryglass_storage_cleanup
  from public,anon,authenticated,service_role;
grant select on public.scryglass_public_query_rows to service_role;
grant select on public.scryglass_public_query_receipts to service_role;

revoke all on function public.require_scryglass_requirements_lock()
  from public,anon,authenticated,service_role;
revoke all on function public.scryglass_json_has_draft_fields(jsonb)
  from public,anon,authenticated,service_role;
revoke all on function public.guard_scryglass_query_mutation()
  from public,anon,authenticated,service_role;
revoke all on function public.stage_scryglass_query_rows(text,text,jsonb)
  from public,anon,authenticated,service_role;
revoke all on function public.seal_scryglass_query_dataset(text,text,integer,bigint,text,text)
  from public,anon,authenticated,service_role;
revoke all on function public.assert_scryglass_query_release(text)
  from public,anon,authenticated,service_role;
revoke all on function public.scryglass_active_query_release_id()
  from public,anon,authenticated,service_role;
revoke all on function public.scryglass_bounded_query_result(jsonb)
  from public,anon,authenticated,service_role;
revoke all on function public.scryglass_text_array_is_bounded(text[],integer,integer)
  from public,anon,authenticated,service_role;
revoke all on function public.guard_scryglass_public_asset_mutation()
  from public,anon,authenticated,service_role;
revoke all on function public.guard_scryglass_public_release_mutation()
  from public,anon,authenticated,service_role;
revoke all on function public.activate_scryglass_public_release(text)
  from public,anon,authenticated,service_role;
revoke all on function public.restore_scryglass_public_release(text)
  from public,anon,authenticated,service_role;
revoke all on function public.prune_scryglass_public_releases_v2(integer)
  from public,anon,authenticated,service_role;
revoke all on function public.ack_scryglass_storage_cleanup(text[])
  from public,anon,authenticated,service_role;
grant execute on function public.stage_scryglass_query_rows(text,text,jsonb) to service_role;
grant execute on function public.seal_scryglass_query_dataset(text,text,integer,bigint,text,text) to service_role;
grant execute on function public.assert_scryglass_query_release(text) to service_role;
grant execute on function public.activate_scryglass_public_release(text) to service_role;
grant execute on function public.restore_scryglass_public_release(text) to service_role;
grant execute on function public.prune_scryglass_public_releases_v2(integer)
  to service_role;
grant execute on function public.ack_scryglass_storage_cleanup(text[])
  to service_role;

revoke all on function public.get_scryglass_ratings(text,text[],text[],text[],text[],text[],boolean,text,text,integer,integer,integer)
  from public,anon,authenticated,service_role;
revoke all on function public.get_scryglass_rating_facets(text,text[])
  from public,anon,authenticated,service_role;
revoke all on function public.get_scryglass_player_champions(text,text,text[],text[],text[],text[],boolean,integer,text,integer,integer)
  from public,anon,authenticated,service_role;
revoke all on function public.get_scryglass_champions(text[],text[],text[],boolean,integer,text,integer,integer)
  from public,anon,authenticated,service_role;
revoke all on function public.get_scryglass_tier_rows(text,text[],text[],text[],text[],text[],text,integer,text,integer,integer)
  from public,anon,authenticated,service_role;
revoke all on function public.get_scryglass_tier_facets()
  from public,anon,authenticated,service_role;
revoke all on function public.get_scryglass_tier_scope(text,text,text,text,text,integer)
  from public,anon,authenticated,service_role;
revoke all on function public.get_scryglass_player_profile(text)
  from public,anon,authenticated,service_role;
revoke all on function public.get_scryglass_team_profile(text)
  from public,anon,authenticated,service_role;
revoke all on function public.get_scryglass_matches(text[],text[],text,text,integer[],text,text,text,integer,integer)
  from public,anon,authenticated,service_role;
revoke all on function public.get_scryglass_match(text)
  from public,anon,authenticated,service_role;
revoke all on function public.get_scryglass_match_facets(text[],integer[],text,text,text)
  from public,anon,authenticated,service_role;
revoke all on function public.get_scryglass_query_status()
  from public,anon,authenticated,service_role;
revoke all on function public.get_scryglass_query_entities()
  from public,anon,authenticated,service_role;
revoke all on function public.get_scryglass_active_release(text)
  from public,anon,authenticated,service_role;
revoke all on function public.get_scryglass_active_asset(text,text)
  from public,anon,authenticated,service_role;
revoke all on function public.get_scryglass_active_inline_asset(text,text)
  from public,anon,authenticated,service_role;

grant execute on function public.get_scryglass_ratings(text,text[],text[],text[],text[],text[],boolean,text,text,integer,integer,integer) to anon,authenticated;
grant execute on function public.get_scryglass_rating_facets(text,text[]) to anon,authenticated;
grant execute on function public.get_scryglass_player_champions(text,text,text[],text[],text[],text[],boolean,integer,text,integer,integer) to anon,authenticated;
grant execute on function public.get_scryglass_champions(text[],text[],text[],boolean,integer,text,integer,integer) to anon,authenticated;
grant execute on function public.get_scryglass_tier_rows(text,text[],text[],text[],text[],text[],text,integer,text,integer,integer) to anon,authenticated;
grant execute on function public.get_scryglass_tier_facets() to anon,authenticated;
grant execute on function public.get_scryglass_tier_scope(text,text,text,text,text,integer) to anon,authenticated;
grant execute on function public.get_scryglass_player_profile(text) to anon,authenticated;
grant execute on function public.get_scryglass_team_profile(text) to anon,authenticated;
grant execute on function public.get_scryglass_matches(text[],text[],text,text,integer[],text,text,text,integer,integer) to anon,authenticated;
grant execute on function public.get_scryglass_match(text) to anon,authenticated;
grant execute on function public.get_scryglass_match_facets(text[],integer[],text,text,text) to anon,authenticated;
grant execute on function public.get_scryglass_query_status() to anon,authenticated;
grant execute on function public.get_scryglass_query_entities() to anon,authenticated;
grant execute on function public.get_scryglass_active_release(text) to anon,authenticated;
grant execute on function public.get_scryglass_active_asset(text,text) to anon,authenticated;
grant execute on function public.get_scryglass_active_inline_asset(text,text) to anon,authenticated;

comment on table public.scryglass_public_query_rows is
  'Release-bound bounded rows. Public clients use active-only RPCs.';
comment on table public.scryglass_public_query_receipts is
  'Source and database digests sealed before release activation.';
