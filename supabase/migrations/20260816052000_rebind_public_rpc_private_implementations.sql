create or replace function scryglass_private.get_scryglass_active_release(
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
        'tier',pg_catalog.jsonb_build_object(
          'status','available',
          'as_of',pg_catalog.to_jsonb(candidate.source_as_of)
        ),
        'draft_authority',case when authority.value is not null then
          pg_catalog.jsonb_build_object(
            'schema_version','scryglass:draft-authority:v1',
            'status','descriptive',
            'authority','descriptive',
            'estimand','composition_only',
            'release_id',candidate.release_id,
            'model_version',authority.value->>'model_version',
            'artifact_sha256',authority.value->>'artifact_sha256',
            'receipt_sha256',authority.value->>'receipt_sha256',
            'issued_utc',authority.value->>'issued_utc',
            'probability_authority',false,
            'recommendation_authority',false,
            'betting_authority',false
          ) else null end,
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
          from public.scryglass_public_assets asset
          where asset.release_id=candidate.release_id
            and (asset.path<>'features/draft_records.json' or authority.value is not null)),
        'total_files',(select pg_catalog.to_jsonb(pg_catalog.count(*))
          from public.scryglass_public_assets asset
          where asset.release_id=candidate.release_id
            and (asset.path<>'features/draft_records.json' or authority.value is not null)),
        'files',coalesce((select pg_catalog.jsonb_agg(pg_catalog.jsonb_build_object(
          'path',asset.path,'bytes',asset.bytes,'sha256',asset.sha256) order by asset.path)
          from public.scryglass_public_assets asset
          where asset.release_id=candidate.release_id
            and (asset.path<>'features/draft_records.json' or authority.value is not null)),'[]'::jsonb),
        'release',pg_catalog.jsonb_build_object(
          'release_id',candidate.release_id,
          'tier_list_version',candidate.manifest#>'{release,tier_list_version}',
          'artifact_hashes',coalesce((select pg_catalog.jsonb_object_agg(
            asset.path,asset.sha256 order by asset.path)
            from public.scryglass_public_assets asset
            where asset.release_id=candidate.release_id
              and (asset.path<>'features/draft_records.json' or authority.value is not null)),'{}'::jsonb)
        )
      )
    ) manifest
  from public.scryglass_public_releases candidate
  left join lateral (
    select public.scryglass_query_descriptive_authority(candidate.release_id) value
  ) authority on true
  where candidate.status='active'
    and (p_release_id is null or (
      pg_catalog.octet_length(p_release_id) between 1 and 30
      and p_release_id~'^v[0-9]{4}\.[0-9]{2}\.[0-9]{2}\.[0-9]{6}$'
      and candidate.release_id=p_release_id
    ))
$$;

create or replace function scryglass_private.get_scryglass_active_asset(
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
    and (
      asset.path<>'features/draft_records.json'
      or public.scryglass_query_descriptive_authority(release.release_id) is not null
    )
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

revoke all on function scryglass_private.get_scryglass_active_asset(text, text) from service_role;
revoke all on function scryglass_private.get_scryglass_active_release(text) from service_role;
grant execute on function scryglass_private.get_scryglass_active_asset(text, text) to anon, authenticated;
grant execute on function scryglass_private.get_scryglass_active_release(text) to anon, authenticated;
