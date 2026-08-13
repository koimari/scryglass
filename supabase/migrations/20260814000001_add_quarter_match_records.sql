-- Add the quarter-split match records to the publication allowlist.

alter table public.scryglass_public_assets
  drop constraint if exists scryglass_public_assets_path_check;

alter table public.scryglass_public_assets
  add constraint scryglass_public_assets_path_check check (
    path in (
      'features/ratings_snapshot.json',
      'features/player_ratings_snapshot.json',
      'features/team_records.json',
      'features/team_weekly_ranks.json',
      'features/player_records.json',
      'features/player_champion_records.json',
      'features/profile_records.json',
      'features/match_index.json',
      'features/match_records_2025.json',
      'features/match_records_2025_q1.json',
      'features/match_records_2025_q2.json',
      'features/match_records_2025_q3.json',
      'features/match_records_2025_q4.json',
      'features/match_records_2026.json',
      'features/match_records_2026_q1.json',
      'features/match_records_2026_q2.json',
      'features/match_records_2026_q3.json',
      'features/match_records_2026_q4.json',
      'features/player_weekly_ranks.json',
      'features/player_metadata.json',
      'features/schedule.json',
      'features/leaderboards.json',
      'features/draft_records.json',
      'rankings/tierlists.json',
      'rankings/tierlists-latest.json'
    )
  );



create or replace function public.activate_scryglass_public_release(p_release_id text)
returns jsonb
language plpgsql
security definer
set search_path = ''
as $$
declare
  previous_release_id text;
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
    'rankings/tierlists.json'
  ];
  present_assets integer;
begin
  perform pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtext('scryglass-public-release'));

  if not exists (
    select 1
    from public.scryglass_public_releases
    where release_id = p_release_id
      and status in ('staging', 'active')
  ) then
    raise exception 'Scryglass release is not ready for activation';
  end if;

  select count(*)
  into present_assets
  from public.scryglass_public_assets
  where release_id = p_release_id
    and path = any(required_assets);

  if present_assets <> pg_catalog.cardinality(required_assets) then
    raise exception 'Scryglass release has % of % required assets',
      present_assets, pg_catalog.cardinality(required_assets);
  end if;

  update public.scryglass_public_releases
  set status = 'active'
  where release_id = p_release_id
  returning null into previous_release_id;

  update public.scryglass_public_releases
  set status = 'superseded'
  where release_id <> p_release_id
    and status = 'active';

  return jsonb_build_object('release_id', p_release_id, 'status', 'active');
end;
$$;
