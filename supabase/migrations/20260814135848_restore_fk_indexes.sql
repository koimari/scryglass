-- Foreign-key covering indexes are required even when their scan count is
-- currently low.  They protect parent updates and deletes on private tables.

begin;

create index if not exists scryglass_oe_games_version_fk_idx
  on public.scryglass_oe_games (canonical_game_id, payload_sha256);

create index if not exists scryglass_public_health_release_idx
  on public.scryglass_public_health (active_release_id);

create index if not exists scryglass_public_health_run_idx
  on public.scryglass_public_health (last_run_id);

create index if not exists scryglass_refresh_runs_retry_idx
  on public.scryglass_refresh_runs (retry_of);

commit;
