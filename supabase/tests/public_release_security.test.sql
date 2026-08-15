begin;
select plan(43);

select is(
  (select public from storage.buckets where id = 'scryglass-public'),
  false,
  'the public release bucket is private after the Storage cutover'
);
select is(
  (select file_size_limit from storage.buckets where id = 'scryglass-public'),
  125829120::bigint,
  'the private release bucket keeps its bounded file size'
);

select ok(
  not has_table_privilege('anon', 'public.scryglass_public_releases', 'select'),
  'anon cannot read the release base table'
);
select ok(
  not has_table_privilege('authenticated', 'public.scryglass_public_releases', 'select'),
  'authenticated cannot read the release base table'
);
select ok(
  not has_table_privilege('anon', 'public.scryglass_public_assets', 'select'),
  'anon cannot read the asset base table'
);
select ok(
  not has_table_privilege('authenticated', 'public.scryglass_public_assets', 'select'),
  'authenticated cannot read the asset base table'
);
select ok(
  not has_table_privilege('anon', 'public.scryglass_public_health', 'select'),
  'anon cannot read the health base table'
);
select ok(
  not has_table_privilege('authenticated', 'public.scryglass_public_health', 'select'),
  'authenticated cannot read the health base table'
);
select ok(
  has_table_privilege('service_role', 'public.scryglass_public_releases', 'insert'),
  'service role can stage a release'
);
select ok(
  has_table_privilege('service_role', 'public.scryglass_public_assets', 'update'),
  'service role can write staging asset metadata'
);
select ok(
  not has_column_privilege('service_role', 'public.scryglass_public_releases', 'status', 'update'),
  'service role cannot change release status directly'
);
select ok(
  not has_function_privilege(
    'anon',
    'public.get_scryglass_active_inline_asset(text,text)',
    'execute'
  ),
  'anon cannot call the retired inline asset RPC'
);
select ok(
  has_function_privilege(
    'anon',
    'public.get_scryglass_active_release(text)',
    'execute'
  ),
  'anon can call the sanitized active-release RPC'
);
select ok(
  has_function_privilege(
    'anon',
    'public.get_scryglass_ratings(text,text[],text[],text[],text[],text[],boolean,text,text,integer,integer,integer)',
    'execute'
  ),
  'anon can call the bounded ratings RPC'
);
select ok(
  has_function_privilege(
    'service_role',
    'public.activate_scryglass_public_release(text)',
    'execute'
  ),
  'service role can call the activation RPC'
);
select ok(
  has_function_privilege(
    'service_role',
    'public.restore_scryglass_public_release(text)',
    'execute'
  ),
  'service role can call the restore RPC'
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
    where oid = 'public.get_scryglass_active_release(text)'::regprocedure
  ),
  false,
  'the public active-release wrapper is security invoker'
);
select is(
  (
    select prosecdef
    from pg_proc
    where oid = 'scryglass_private.get_scryglass_active_release(text)'::regprocedure
  ),
  true,
  'the private active-release implementation is security definer'
);
select policies_are(
  'storage',
  'objects',
  array[
    'read active Scryglass Storage assets',
    'service role manages Scryglass Storage assets'
  ],
  'Storage has only the release-bound read and service-role policies'
);

-- Build one complete query-authorized Storage-only release. The fixture uses
-- tiny rows and files so it tests the final contract without private data.
do $fixture$
declare
  rid constant text := 'v2026.08.21.120000';
  asset_paths constant text[] := array[
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
  datasets constant text[] := array[
    'players', 'teams', 'player_champions', 'games', 'identity_games',
    'champions', 'aliases', 'tier_rows', 'tier_scopes',
    'tier_matrix_rows', 'tier_similarity_champions',
    'tier_similarity_edges'
  ];
  files jsonb := '[]'::jsonb;
  hashes jsonb := '{}'::jsonb;
  dataset_manifest jsonb := '{}'::jsonb;
  dataset text;
  path text;
  source_json text;
  payload_json constant text := '{}';
  source_sha text;
  row_sha text;
  row_digest text;
  rows jsonb;
begin
  foreach path in array asset_paths loop
    files := files || jsonb_build_array(
      jsonb_build_object('path', path, 'bytes', 2, 'sha256', repeat('a', 64))
    );
    hashes := hashes || jsonb_build_object(path, repeat('a', 64));
  end loop;

  foreach dataset in array datasets loop
    source_sha := encode(extensions.digest(convert_to(payload_json, 'UTF8'), 'sha256'), 'hex');
    source_json := jsonb_build_object(
      'row_key', dataset,
      'payload', payload_json::jsonb,
      'source_bytes', 2,
      'source_sha256', source_sha
    )::text;
    row_sha := encode(extensions.digest(convert_to(source_json, 'UTF8'), 'sha256'), 'hex');
    row_digest := encode(
      extensions.digest(convert_to(dataset || ':' || row_sha, 'UTF8'), 'sha256'),
      'hex'
    );
    dataset_manifest := dataset_manifest || jsonb_build_object(
      dataset,
      jsonb_build_object(
        'rows', 1,
        'bytes', 2,
        'sha256', repeat('a', 64),
        'row_digest_sha256', row_digest
      )
    );
  end loop;

  insert into public.scryglass_public_releases (
    release_id, status, manifest, source_as_of
  ) values (
    rid,
    'staging',
    jsonb_build_object(
      'pack_id', rid,
      'files', files,
      'draft_authority', jsonb_build_object(
        'schema_version', 'scryglass:draft-authority:v1',
        'status', 'unavailable',
        'release_id', rid,
        'model_version', null,
        'receipt_sha256', null,
        'reason', 'model_not_promoted'
      ),
      'query_api', jsonb_build_object(
        'schema_version', 'scryglass:query-api:v1',
        'status', 'available',
        'datasets', dataset_manifest
      ),
      'release', jsonb_build_object(
        'release_id', rid,
        'artifact_hashes', hashes
      )
    ),
    now()
  );

  insert into public.scryglass_public_assets (
    release_id, path, body, storage_path, bytes, sha256, content_type
  )
  select rid, item.path, null, rid || '/' || item.path,
    2, repeat('a', 64), 'application/json'
  from unnest(asset_paths) as item(path);

  insert into storage.objects (bucket_id, name, user_metadata)
  select 'scryglass-public', rid || '/' || item.path,
    jsonb_build_object('sha256', repeat('a', 64), 'bytes', 2, 'content_type', 'application/json')
  from unnest(asset_paths) as item(path);

  foreach dataset in array datasets loop
    source_json := jsonb_build_object(
      'row_key', dataset,
      'payload', payload_json::jsonb,
      'source_bytes', 2,
      'source_sha256', encode(extensions.digest(convert_to(payload_json, 'UTF8'), 'sha256'), 'hex')
    )::text;
    row_sha := encode(extensions.digest(convert_to(source_json, 'UTF8'), 'sha256'), 'hex');
    rows := jsonb_build_array(jsonb_build_object(
      'release_id', rid,
      'dataset', dataset,
      'source_json', source_json,
      'payload_json', payload_json,
      'row_sha256', row_sha
    ));
    perform public.stage_scryglass_query_rows(rid, dataset, rows);
    row_digest := encode(
      extensions.digest(convert_to(dataset || ':' || row_sha, 'UTF8'), 'sha256'),
      'hex'
    );
    perform public.seal_scryglass_query_dataset(
      rid, dataset, 1, 2, repeat('a', 64), row_digest
    );
  end loop;

  perform public.activate_scryglass_public_release(rid);
end;
$fixture$;

select is(
  (select count(*)::integer from public.get_scryglass_active_release('v2026.08.21.120000')),
  1,
  'active-release RPC returns one active release'
);
select is(
  (
    select manifest #>> '{query_api,status}'
    from public.get_scryglass_active_release('v2026.08.21.120000')
  ),
  'available',
  'active-release RPC exposes the bounded query marker'
);
select ok(
  not public.scryglass_json_has_draft_fields(
    (select manifest from public.get_scryglass_active_release('v2026.08.21.120000'))
  ),
  'active-release projection contains no Draft fields'
);
select is(
  (
    select count(*)::integer
    from public.get_scryglass_active_asset(
      'v2026.08.21.120000',
      'features/ratings_snapshot.json'
    )
  ),
  1,
  'active-asset RPC returns one bound Storage asset'
);
select is(
  (
    select storage_path
    from public.get_scryglass_active_asset(
      'v2026.08.21.120000',
      'features/ratings_snapshot.json'
    )
  ),
  'v2026.08.21.120000/features/ratings_snapshot.json',
  'active-asset RPC returns the canonical Storage path'
);
select is(
  (public.get_scryglass_query_status() ->> 'schema_version'),
  'scryglass:query-api:v1',
  'bounded query status has the versioned contract'
);
select ok(
  not public.scryglass_json_has_draft_fields(public.get_scryglass_query_status()),
  'bounded query status contains no Draft fields'
);

set local role anon;
select throws_ok(
  $$select * from public.scryglass_public_releases$$,
  '42501',
  null,
  'anon base-table release reads are denied'
);
select throws_ok(
  $$select * from public.scryglass_public_assets$$,
  '42501',
  null,
  'anon base-table asset reads are denied'
);
select throws_ok(
  $$select public.get_scryglass_active_inline_asset('v2026.08.21.120000','features/ratings_snapshot.json')$$,
  '42501',
  null,
  'anon inline compatibility reads are denied'
);
reset role;

set local role authenticated;
select throws_ok(
  $$select * from public.scryglass_public_health$$,
  '42501',
  null,
  'authenticated base-table health reads are denied'
);
reset role;

set local role service_role;
select throws_ok(
  $$delete from public.scryglass_public_query_rows where release_id='v2026.08.21.120000'$$,
  '42501',
  null,
  'service role cannot delete published query rows directly'
);
select throws_ok(
  $$update public.scryglass_public_releases set status='superseded' where release_id='v2026.08.21.120000'$$,
  '42501',
  null,
  'service role cannot bypass the activation RPC'
);
select lives_ok(
  $$select set_config('scryglass.release_transition_authorized','1',false)$$,
  'transition spoof setup remains a normal function call'
);
select throws_ok(
  $$update public.scryglass_public_releases set status='superseded' where release_id='v2026.08.21.120000'$$,
  '42501',
  null,
  'a caller-set transition flag cannot authorize a status change'
);
select throws_ok(
  $$update storage.objects set user_metadata='{"bytes":3}' where name='v2026.08.21.120000/features/ratings_snapshot.json'$$,
  null,
  null,
  'published Storage metadata cannot be overwritten'
);
select throws_ok(
  $$delete from storage.objects where name='v2026.08.21.120000/features/ratings_snapshot.json'$$,
  null,
  null,
  'published Storage objects cannot be deleted'
);
reset role;

insert into public.scryglass_public_releases (
  release_id, status, manifest, source_as_of
) values (
  'v2026.08.22.120000',
  'staging',
  '{"pack_id":"v2026.08.22.120000","files":[],"release":{"release_id":"v2026.08.22.120000","artifact_hashes":{}}}'::jsonb,
  now()
);
select throws_ok(
  $$select public.assert_scryglass_query_release('v2026.08.22.120000')$$,
  null,
  null,
  'strict query assertion rejects a legacy release without query_api'
);
select is(
  (
    select count(*)::integer
    from public.get_scryglass_active_release('v2026.08.22.120000')
  ),
  0,
  'inactive release IDs are absent from the public projection'
);

set local role scryglass_release_transition_owner;
update public.scryglass_public_releases
set status = 'superseded'
where release_id = 'v2026.08.21.120000';
reset role;

set local role service_role;
select lives_ok(
  $$select public.prune_scryglass_public_releases_v2(1)$$,
  'retention pruning removes a superseded release with sealed query rows'
);
reset role;

select is(
  (
    select count(*)::integer
    from public.scryglass_public_releases
    where release_id = 'v2026.08.21.120000'
  ),
  0,
  'retention pruning removes the superseded release row'
);
select is(
  (
    select count(*)::integer
    from public.scryglass_public_query_rows
    where release_id = 'v2026.08.21.120000'
  ),
  0,
  'retention pruning cascades through sealed query rows'
);

select * from finish();
rollback;
