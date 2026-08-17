-- The app validates descriptive match and profile fields against the nested
-- composition receipt. Preserve that fixed safe projection under promotion.

create or replace function scryglass_private.get_scryglass_active_release(
  p_release_id text default null
)
returns table(release_id text, status text, manifest jsonb)
language plpgsql
stable
security definer
set search_path = ''
set statement_timeout = '5s'
as $$
declare
  row_value record;
  promoted jsonb;
  descriptive jsonb;
begin
  select * into row_value
  from scryglass_private.get_scryglass_active_release_before_promotion(
    p_release_id
  );
  if not found then
    return;
  end if;

  select release.manifest -> 'draft_authority'
    into promoted
  from public.scryglass_public_releases as release
  where release.release_id = row_value.release_id
    and release.status = 'active'
    and release.manifest #>> '{draft_authority,status}' = 'promoted';

  if promoted is not null then
    descriptive := public.scryglass_query_descriptive_authority(
      row_value.release_id
    );
    if descriptive is null then
      raise exception 'Promoted release has no bound descriptive authority';
    end if;
    row_value.manifest := pg_catalog.jsonb_set(
      row_value.manifest,
      '{draft_authority}',
      pg_catalog.jsonb_build_object(
        'schema_version', 'scryglass:draft-authority:v1',
        'status', 'promoted',
        'authority', 'promoted',
        'release_id', row_value.release_id,
        'model_version', promoted ->> 'model_version',
        'artifact_sha256', promoted ->> 'artifact_sha256',
        'receipt_sha256', promoted ->> 'receipt_sha256',
        'issued_utc', promoted ->> 'issued_utc',
        'estimand', 'prematch_map_win_probability_with_controlled_draft_intervention',
        'probability_authority', true,
        'recommendation_authority', true,
        'betting_authority', false,
        'reason', null,
        'descriptive_authority', descriptive
      )
    );
  end if;
  return query
  select row_value.release_id, row_value.status, row_value.manifest;
end;
$$;

revoke all on function scryglass_private.get_scryglass_active_release(text)
  from public, service_role;
grant execute on function scryglass_private.get_scryglass_active_release(text)
  to anon, authenticated;
