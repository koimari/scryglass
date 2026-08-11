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
      'features/match_records_2026.json',
      'features/player_weekly_ranks.json',
      'features/player_metadata.json',
      'features/schedule.json',
      'rankings/tierlists.json',
      'rankings/tierlists-latest.json'
    )
  );
