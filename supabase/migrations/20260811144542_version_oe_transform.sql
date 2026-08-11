alter table public.scryglass_oe_imports
  add column transform_version text not null default 'oe-normalization:v1',
  add check (transform_version <> '');
