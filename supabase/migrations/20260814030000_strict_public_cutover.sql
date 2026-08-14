-- Phase 3: close the compatibility window after the strict web build and two
-- query-complete, Storage-only releases have passed their probes.

revoke all on public.scryglass_public_releases
  from public, anon, authenticated;
revoke all on public.scryglass_public_assets
  from public, anon, authenticated;
revoke all on public.scryglass_public_health
  from public, anon, authenticated;
revoke all on public.scryglass_storage_cleanup
  from public, anon, authenticated;
revoke all on public.scryglass_diagnostic_credentials
  from public, anon, authenticated;

-- The strict application reads fixed active-release projections and bounded
-- query RPCs. It no longer needs the parsed JSONB compatibility path.
drop function if exists public.get_scryglass_active_inline_asset(text, text);

comment on table public.scryglass_public_releases is
  'Private release metadata. Public clients use fixed active-release RPCs.';
comment on table public.scryglass_public_assets is
  'Private release asset metadata. Public clients use active asset RPCs and private Storage.';

