alter table public.scryglass_oe_imports
  add column if not exists quarantined_games jsonb not null default '{}'::jsonb;

alter table public.scryglass_oe_imports
  add constraint scryglass_oe_imports_quarantined_games_object
  check (jsonb_typeof(quarantined_games) = 'object');

comment on column public.scryglass_oe_imports.quarantined_games is
  'Bounded reason by canonical game ID. A later source cycle evaluates each game again.';
