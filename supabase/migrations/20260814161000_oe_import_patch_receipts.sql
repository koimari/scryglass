-- The bounded migration was applied before the importer receipt fields were
-- added to its source file. This forward migration brings production to the
-- schema expected by the attested worker without rewriting migration history.

begin;

alter table public.scryglass_oe_imports
  add column if not exists riot_patch_receipts integer not null default 0;

alter table public.scryglass_oe_imports
  drop constraint if exists scryglass_oe_imports_riot_patch_receipts_check;

alter table public.scryglass_oe_imports
  add constraint scryglass_oe_imports_riot_patch_receipts_check
  check (riot_patch_receipts >= 0);

alter table public.scryglass_oe_imports
  alter column transform_version set default 'oe-normalization:v3';

commit;
