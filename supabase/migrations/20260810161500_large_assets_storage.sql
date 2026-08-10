alter table public.scryglass_public_assets
  alter column body drop not null;

alter table public.scryglass_public_assets
  add column if not exists storage_path text;

alter table public.scryglass_public_assets
  drop constraint if exists scryglass_public_assets_payload_check;

alter table public.scryglass_public_assets
  add constraint scryglass_public_assets_payload_check check (
    (body is not null and storage_path is null)
    or (body is null and storage_path is not null)
  );

insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values (
  'scryglass-public',
  'scryglass-public',
  true,
  52428800,
  array['application/json']
)
on conflict (id) do update
set public = excluded.public,
    file_size_limit = excluded.file_size_limit,
    allowed_mime_types = excluded.allowed_mime_types;
