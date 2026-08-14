begin;
select plan(44);

select is(
  (select public from storage.buckets where id = 'scryglass-public'),
  true,
  'phase 1 keeps the public asset bucket during the compatibility window'
);
select is(
  (select file_size_limit from storage.buckets where id = 'scryglass-public'),
  125829120::bigint,
  'public asset bucket has a bounded file size'
);

select ok(
  has_table_privilege('anon', 'public.scryglass_public_releases', 'select'),
  'anon can select the active release through RLS'
);
select ok(
  has_table_privilege('authenticated', 'public.scryglass_public_releases', 'select'),
  'authenticated can select the active release through RLS'
);
select ok(
  has_table_privilege('anon', 'public.scryglass_public_assets', 'select'),
  'anon can select active asset metadata through RLS'
);
select ok(
  has_table_privilege('authenticated', 'public.scryglass_public_assets', 'select'),
  'authenticated can select active asset metadata through RLS'
);
select ok(
  not has_table_privilege('anon', 'public.scryglass_public_releases', 'insert'),
  'anon cannot stage a release'
);
select ok(
  not has_table_privilege('authenticated', 'public.scryglass_public_assets', 'update'),
  'authenticated cannot change asset metadata'
);
select ok(
  not has_table_privilege('anon', 'public.scryglass_storage_cleanup', 'select'),
  'anon cannot inspect retained Storage paths'
);
select ok(
  has_table_privilege('service_role', 'public.scryglass_public_releases', 'insert'),
  'service role can stage a release'
);
select ok(
  has_table_privilege('service_role', 'public.scryglass_public_assets', 'update'),
  'service role can stage asset metadata'
);
select ok(
  not has_table_privilege('service_role', 'public.scryglass_storage_cleanup', 'delete'),
  'service role cannot delete the Storage cleanup queue directly'
);

select ok(
  not has_function_privilege(
    'anon',
    'public.activate_scryglass_public_release(text)',
    'execute'
  ),
  'anon cannot activate releases'
);
select ok(
  not has_function_privilege(
    'authenticated',
    'public.restore_scryglass_public_release(text)',
    'execute'
  ),
  'authenticated cannot restore releases'
);
select ok(
  not has_function_privilege(
    'anon',
    'public.prune_scryglass_public_releases_v2(integer)',
    'execute'
  ),
  'anon cannot prune releases'
);
select ok(
  has_function_privilege(
    'service_role',
    'public.activate_scryglass_public_release(text)',
    'execute'
  ),
  'service role can activate releases'
);
select ok(
  has_function_privilege(
    'service_role',
    'public.restore_scryglass_public_release(text)',
    'execute'
  ),
  'service role can restore releases'
);
select ok(
  has_function_privilege(
    'service_role',
    'public.prune_scryglass_public_releases_v2(integer)',
    'execute'
  ),
  'service role can prune releases'
);
select ok(
  has_function_privilege(
    'service_role',
    'public.ack_scryglass_storage_cleanup(text[])',
    'execute'
  ),
  'service role can acknowledge completed Storage cleanup'
);

select is(
  (
    select prosecdef
    from pg_proc
    where oid = 'public.activate_scryglass_public_release(text)'::regprocedure
  ),
  true,
  'activation is security definer'
);
select is(
  (
    select prosecdef
    from pg_proc
    where oid = 'public.restore_scryglass_public_release(text)'::regprocedure
  ),
  true,
  'restore is security definer'
);
select is(
  (
    select prosecdef
    from pg_proc
    where oid = 'public.prune_scryglass_public_releases_v2(integer)'::regprocedure
  ),
  true,
  'retention is security definer'
);
select is(
  (
    select prosecdef
    from pg_proc
    where oid = 'public.ack_scryglass_storage_cleanup(text[])'::regprocedure
  ),
  true,
  'Storage cleanup acknowledgement is security definer'
);

select policies_are(
  'storage',
  'objects',
  array[
    'read active Scryglass Storage assets',
    'service role manages Scryglass Storage assets'
  ],
  'Storage has only the release-bound public read and service-role policies'
);

set local role service_role;

select lives_ok(
  $$
    insert into public.scryglass_public_releases (
      release_id, status, manifest, source_as_of
    ) values (
      'v2026.08.13.120000',
      'staging',
      '{"pack_id":"v2026.08.13.120000","files":[],"release":{"release_id":"v2026.08.13.120000","artifact_hashes":{}}}'::jsonb,
      now()
    )
  $$,
  'service role can stage a canonical release row'
);

select throws_ok(
  $$
    insert into public.scryglass_public_assets (
      release_id, path, body, storage_path, bytes, sha256, content_type
    ) values (
      'v2026.08.13.120000',
      'features/ratings_snapshot.json',
      null,
      'v2026.08.13.120000/../ratings_snapshot.json',
      2,
      repeat('a', 64),
      'application/json'
    )
  $$,
  '23514',
  null,
  'Storage path must equal release ID and asset path'
);

select throws_ok(
  $$
    insert into public.scryglass_public_assets (
      release_id, path, body, storage_path, bytes, sha256, content_type
    ) values (
      'v2026.08.13.120000',
      'features/ratings_snapshot.json',
      null,
      'v2026.08.13.120000/features/ratings_snapshot.json',
      2,
      repeat('a', 64),
      'text/plain'
    )
  $$,
  '23514',
  null,
  'asset MIME type must be application/json'
);

select throws_ok(
  $$
    select public.activate_scryglass_public_release('v2026.08.13.120000')
  $$,
  null,
  'activation rejects an incomplete manifest and asset set'
);

with paths(path) as (
  select unnest(array[
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
  ]::text[])
), metadata as (
  select
    jsonb_agg(
      jsonb_build_object('path', path, 'bytes', 2, 'sha256', repeat('a', 64))
      order by path
    ) as files,
    jsonb_object_agg(path, repeat('a', 64)) as hashes
  from paths
)
update public.scryglass_public_releases
set manifest = jsonb_build_object(
  'pack_id', release_id,
  'files', metadata.files,
  'draft_authority', jsonb_build_object(
    'schema_version', 'scryglass:draft-authority:v1',
    'status', 'unavailable',
    'release_id', release_id,
    'model_version', null,
    'receipt_sha256', null,
    'reason', 'model_not_promoted'
  ),
  'release', jsonb_build_object(
    'release_id', release_id,
    'artifact_hashes', metadata.hashes
  )
)
from metadata
where release_id = 'v2026.08.13.120000';

insert into public.scryglass_public_assets (
  release_id, path, body, storage_path, bytes, sha256, content_type
)
select
  'v2026.08.13.120000',
  path,
  null,
  'v2026.08.13.120000/' || path,
  2,
  repeat('a', 64),
  'application/json'
from unnest(array[
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
]::text[]) path;

insert into storage.objects (
  bucket_id, name, metadata, user_metadata
)
select
  'scryglass-public',
  storage_path,
  jsonb_build_object('size', bytes, 'mimetype', content_type),
  jsonb_build_object(
    'sha256', sha256,
    'bytes', bytes,
    'content_type', content_type
  )
from public.scryglass_public_assets
where release_id = 'v2026.08.13.120000';

update public.scryglass_public_releases
set manifest = pg_catalog.jsonb_set(
  manifest,
  array['release', 'artifact_hashes', 'features/ratings_snapshot.json'],
  pg_catalog.to_jsonb(repeat('b', 64))
)
where release_id = 'v2026.08.13.120000';
select throws_ok(
  $$select public.activate_scryglass_public_release('v2026.08.13.120000')$$,
  null,
  'activation rejects a manifest digest that differs from asset metadata'
);
update public.scryglass_public_releases
set manifest = pg_catalog.jsonb_set(
  manifest,
  array['release', 'artifact_hashes', 'features/ratings_snapshot.json'],
  pg_catalog.to_jsonb(repeat('a', 64))
)
where release_id = 'v2026.08.13.120000';

update storage.objects
set user_metadata = pg_catalog.jsonb_set(
  user_metadata,
  '{sha256}',
  pg_catalog.to_jsonb(repeat('b', 64))
)
where bucket_id = 'scryglass-public'
  and name = 'v2026.08.13.120000/features/ratings_snapshot.json';
select throws_ok(
  $$select public.activate_scryglass_public_release('v2026.08.13.120000')$$,
  null,
  'activation rejects corrupted Storage object metadata'
);
update storage.objects
set user_metadata = pg_catalog.jsonb_set(
  user_metadata,
  '{sha256}',
  pg_catalog.to_jsonb(repeat('a', 64))
)
where bucket_id = 'scryglass-public'
  and name = 'v2026.08.13.120000/features/ratings_snapshot.json';

select is(
  public.activate_scryglass_public_release('v2026.08.13.120000') ->> 'status',
  'active',
  'activation accepts an exact immutable asset set'
);
select is(
  (
    select status
    from public.scryglass_public_releases
    where release_id = 'v2026.08.13.120000'
  ),
  'active',
  'activation leaves the new release active'
);

select lives_ok(
  $$
    insert into public.scryglass_public_releases (
      release_id, status, manifest, source_as_of
    ) select
      'v2026.08.12.120000',
      'staging',
      pg_catalog.jsonb_set(
        pg_catalog.jsonb_set(
          pg_catalog.jsonb_set(
            manifest,
            '{pack_id}',
            pg_catalog.to_jsonb('v2026.08.12.120000'::text)
          ),
          '{release,release_id}',
          pg_catalog.to_jsonb('v2026.08.12.120000'::text)
        ),
        '{draft_authority,release_id}',
        pg_catalog.to_jsonb('v2026.08.12.120000'::text)
      ),
      source_as_of
    from public.scryglass_public_releases
    where release_id = 'v2026.08.13.120000'
  $$,
  'a complete rollback target can be staged for the test'
);

insert into public.scryglass_public_assets (
  release_id, path, body, storage_path, bytes, sha256, content_type
)
select
  'v2026.08.12.120000',
  path,
  null,
  'v2026.08.12.120000/' || path,
  bytes,
  sha256,
  content_type
from public.scryglass_public_assets
where release_id = 'v2026.08.13.120000';

insert into storage.objects (
  bucket_id, name, metadata, user_metadata
)
select
  'scryglass-public',
  storage_path,
  jsonb_build_object('size', bytes, 'mimetype', content_type),
  jsonb_build_object(
    'sha256', sha256,
    'bytes', bytes,
    'content_type', content_type
  )
from public.scryglass_public_assets
where release_id = 'v2026.08.12.120000';

select is(
  public.activate_scryglass_public_release('v2026.08.12.120000') ->> 'status',
  'active',
  'a complete rollback target can be activated through the transition function'
);

select is(
  public.restore_scryglass_public_release('v2026.08.13.120000') ->> 'status',
  'restored',
  'restore promotes the selected prior release'
);
select is(
  (select status from public.scryglass_public_releases where release_id = 'v2026.08.13.120000'),
  'active',
  'restore leaves the prior release active'
);

insert into public.scryglass_public_releases (
  release_id, status, manifest, source_as_of
) values (
  'v2026.08.11.120000',
  'staging',
  '{"pack_id":"v2026.08.11.120000","files":[],"release":{"release_id":"v2026.08.11.120000","artifact_hashes":{}}}'::jsonb,
  now()
);

select throws_ok(
  $$select public.prune_scryglass_public_releases_v2(0)$$,
  null,
  'retention rejects zero releases'
);
select throws_ok(
  $$select public.prune_scryglass_public_releases_v2(11)$$,
  null,
  'retention rejects more than ten releases'
);
select is(
  (public.prune_scryglass_public_releases_v2(3) ->> 'deleted_count')::integer,
  0,
  'retention returns a structured zero result when nothing expires'
);
select is(
  pg_catalog.jsonb_typeof(
    public.prune_scryglass_public_releases_v2(3) -> 'storage_paths'
  ),
  'array',
  'retention returns the Storage deletion inventory'
);

reset role;
set local role anon;
select is(
  (
    select count(*)::integer
    from public.scryglass_public_releases
    where release_id = 'v2026.08.11.120000'
  ),
  0,
  'anon cannot read a staged release'
);
select throws_ok(
  $$select public.activate_scryglass_public_release('v2026.08.13.120000')$$,
  '42501',
  null,
  'anon activation is denied'
);
reset role;

set local role authenticated;
select is(
  (
    select count(*)::integer
    from public.scryglass_public_releases
    where release_id = 'v2026.08.11.120000'
  ),
  0,
  'authenticated cannot read a staged release'
);
select throws_ok(
  $$select public.restore_scryglass_public_release('v2026.08.12.120000')$$,
  '42501',
  null,
  'authenticated restore is denied'
);
reset role;

select * from finish();
rollback;
