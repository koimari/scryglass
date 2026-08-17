-- Add the separately validated promoted Draft result asset.
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
    'features/draft_records.json','features/promoted_draft_results.json',
    'rankings/tierlists.json','rankings/tierlists-latest.json'
  ));
