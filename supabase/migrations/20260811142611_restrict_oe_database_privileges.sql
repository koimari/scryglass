revoke all on public.scryglass_oe_game_versions from service_role;
revoke all on public.scryglass_oe_games from service_role;
revoke all on public.scryglass_oe_imports from service_role;

grant select, insert on public.scryglass_oe_game_versions to service_role;
grant select, insert, update on public.scryglass_oe_games to service_role;
grant select, insert, update on public.scryglass_oe_imports to service_role;

create index scryglass_oe_games_version_fk_idx
  on public.scryglass_oe_games (canonical_game_id, payload_sha256);
