create table public.scryglass_refresh_runs (
  run_id text primary key,
  scheduled_for timestamptz not null,
  retry_of text references public.scryglass_refresh_runs(run_id)
    on delete set null,
  status text not null
    check (status in ('checking', 'success', 'no_change', 'partial', 'error')),
  stage text not null
    check (
      stage in (
        'acquire',
        'validate_source',
        'ingest',
        'reconcile',
        'derive',
        'validate_artifacts',
        'stage_release',
        'activate_release',
        'invalidate_cache',
        'smoke',
        'complete'
      )
    ),
  input_fingerprint text,
  worker_commit text not null,
  source_file_sha256 text,
  source_observed_through timestamptz,
  release_id text references public.scryglass_public_releases(release_id)
    on delete set null,
  accepted_games integer not null default 0 check (accepted_games >= 0),
  new_games integer not null default 0 check (new_games >= 0),
  corrected_games integer not null default 0 check (corrected_games >= 0),
  unchanged_games integer not null default 0 check (unchanged_games >= 0),
  quarantined_games integer not null default 0 check (quarantined_games >= 0),
  stage_durations jsonb not null default '{}'::jsonb
    check (jsonb_typeof(stage_durations) = 'object'),
  failure_code text,
  failure_detail text,
  started_at timestamptz not null default now(),
  completed_at timestamptz,
  updated_at timestamptz not null default now(),
  check (run_id ~ '^refresh-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{12}$'),
  check (worker_commit ~ '^[0-9a-f]{40}$'),
  check (input_fingerprint is null or input_fingerprint ~ '^[0-9a-f]{64}$'),
  check (source_file_sha256 is null or source_file_sha256 ~ '^[0-9a-f]{64}$'),
  check (failure_code is null or char_length(failure_code) between 1 and 80),
  check (failure_detail is null or char_length(failure_detail) <= 2000),
  check (
    (status = 'checking' and completed_at is null)
    or (status <> 'checking' and completed_at is not null)
  )
);

create index scryglass_refresh_runs_started_idx
  on public.scryglass_refresh_runs (started_at desc);

create index scryglass_refresh_runs_retry_idx
  on public.scryglass_refresh_runs (retry_of)
  where retry_of is not null;

create index scryglass_refresh_runs_failures_idx
  on public.scryglass_refresh_runs (completed_at desc)
  where status in ('partial', 'error');

create index scryglass_refresh_runs_release_idx
  on public.scryglass_refresh_runs (release_id)
  where release_id is not null;

create table public.scryglass_public_health (
  health_id text primary key default 'public-refresh'
    check (health_id = 'public-refresh'),
  status text not null
    check (status in ('ok', 'partial', 'error')),
  refresh_status text not null
    check (refresh_status in ('idle', 'running', 'failed', 'stale')),
  checked_at timestamptz not null,
  last_success_at timestamptz,
  source_as_of timestamptz,
  active_release_id text references public.scryglass_public_releases(release_id)
    on delete set null,
  last_run_id text references public.scryglass_refresh_runs(run_id)
    on delete set null,
  worker_commit text,
  stale boolean not null default false,
  updated_at timestamptz not null default now(),
  check (worker_commit is null or worker_commit ~ '^[0-9a-f]{40}$')
);

create index scryglass_public_health_release_idx
  on public.scryglass_public_health (active_release_id)
  where active_release_id is not null;

create index scryglass_public_health_run_idx
  on public.scryglass_public_health (last_run_id)
  where last_run_id is not null;

alter table public.scryglass_refresh_runs enable row level security;
alter table public.scryglass_public_health enable row level security;

revoke all on public.scryglass_refresh_runs from public, anon, authenticated, service_role;
revoke all on public.scryglass_public_health from public, anon, authenticated, service_role;

grant select, insert, update on public.scryglass_refresh_runs to service_role;
grant select, insert, update on public.scryglass_public_health to service_role;
grant select on public.scryglass_public_health to anon, authenticated;

create policy "read Scryglass public refresh health"
  on public.scryglass_public_health
  for select
  to anon, authenticated
  using (health_id = 'public-refresh');

revoke all on public.scryglass_public_releases from service_role;
revoke all on public.scryglass_public_assets from service_role;
grant select, insert, update, delete on public.scryglass_public_releases to service_role;
grant select, insert, update on public.scryglass_public_assets to service_role;

alter function public.activate_scryglass_public_release(text) security invoker;
alter function public.restore_scryglass_public_release(text) security invoker;
alter function public.prune_scryglass_public_releases(integer) security invoker;

revoke all on function public.activate_scryglass_public_release(text)
  from public, anon, authenticated;
revoke all on function public.restore_scryglass_public_release(text)
  from public, anon, authenticated;
revoke all on function public.prune_scryglass_public_releases(integer)
  from public, anon, authenticated;
grant execute on function public.activate_scryglass_public_release(text) to service_role;
grant execute on function public.restore_scryglass_public_release(text) to service_role;
grant execute on function public.prune_scryglass_public_releases(integer) to service_role;

comment on table public.scryglass_refresh_runs is
  'Private audit ledger for the six-hour Oracle''s Elixir public refresh.';
comment on table public.scryglass_public_health is
  'Sanitized public state for the last Scryglass refresh and active release.';
