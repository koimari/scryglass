create table public.scryglass_oe_game_versions (
  canonical_game_id text not null,
  payload_sha256 text not null,
  source_year smallint not null,
  game_date timestamptz not null,
  league text not null,
  patch text,
  statistics_complete boolean not null,
  source_file_sha256 text not null,
  payload jsonb not null,
  imported_at timestamptz not null default now(),
  primary key (canonical_game_id, payload_sha256),
  check (canonical_game_id <> ''),
  check (payload_sha256 ~ '^[0-9a-f]{64}$'),
  check (source_file_sha256 ~ '^[0-9a-f]{64}$'),
  check (source_year between 2014 and 2100),
  check (league <> ''),
  check (jsonb_typeof(payload) = 'object'),
  check (payload ? 'canonical_game_id' and payload ? 'schema_version'),
  check (payload ->> 'canonical_game_id' = canonical_game_id),
  check (payload ->> 'schema_version' = 'scryglass:oe-game:v1'),
  check (jsonb_typeof(payload -> 'team_rows') = 'array'),
  check (jsonb_typeof(payload -> 'player_rows') = 'array'),
  check (jsonb_array_length(payload -> 'team_rows') = 2),
  check (jsonb_array_length(payload -> 'player_rows') = 10)
);

create index scryglass_oe_game_versions_date_idx
  on public.scryglass_oe_game_versions (game_date desc);

create table public.scryglass_oe_games (
  canonical_game_id text primary key,
  payload_sha256 text not null,
  source_year smallint not null,
  game_date timestamptz not null,
  league text not null,
  patch text,
  statistics_complete boolean not null,
  source_file_sha256 text not null,
  updated_at timestamptz not null default now(),
  foreign key (canonical_game_id, payload_sha256)
    references public.scryglass_oe_game_versions(canonical_game_id, payload_sha256),
  check (canonical_game_id <> ''),
  check (payload_sha256 ~ '^[0-9a-f]{64}$'),
  check (source_file_sha256 ~ '^[0-9a-f]{64}$'),
  check (source_year between 2014 and 2100),
  check (league <> '')
);

create index scryglass_oe_games_year_date_idx
  on public.scryglass_oe_games (source_year, game_date desc);

create index scryglass_oe_games_league_date_idx
  on public.scryglass_oe_games (league, game_date desc);

create table public.scryglass_oe_imports (
  source_year smallint not null,
  source_file_sha256 text not null,
  source_bytes bigint not null,
  source_rows integer not null,
  source_games integer not null,
  accepted_games integer not null,
  new_games integer not null,
  corrected_games integer not null,
  unchanged_games integer not null,
  statistics_complete_games integer not null,
  quarantined_game_ids jsonb not null default '[]'::jsonb,
  source_observed_through timestamptz not null,
  completed_at timestamptz not null default now(),
  primary key (source_year, source_file_sha256),
  check (source_file_sha256 ~ '^[0-9a-f]{64}$'),
  check (source_year between 2014 and 2100),
  check (source_bytes >= 10000),
  check (source_rows > 0),
  check (source_games > 0),
  check (accepted_games >= 0 and accepted_games <= source_games),
  check (new_games >= 0 and corrected_games >= 0 and unchanged_games >= 0),
  check (new_games + corrected_games + unchanged_games = accepted_games),
  check (statistics_complete_games >= 0 and statistics_complete_games <= accepted_games),
  check (jsonb_typeof(quarantined_game_ids) = 'array')
);

alter table public.scryglass_oe_game_versions enable row level security;
alter table public.scryglass_oe_games enable row level security;
alter table public.scryglass_oe_imports enable row level security;

revoke all on public.scryglass_oe_game_versions from public, anon, authenticated;
revoke all on public.scryglass_oe_games from public, anon, authenticated;
revoke all on public.scryglass_oe_imports from public, anon, authenticated;

grant select, insert on public.scryglass_oe_game_versions to service_role;
grant select, insert, update on public.scryglass_oe_games to service_role;
grant select, insert, update on public.scryglass_oe_imports to service_role;
