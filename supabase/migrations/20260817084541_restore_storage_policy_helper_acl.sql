-- The public Storage policy runs through the SECURITY INVOKER wrapper as the
-- requesting anon or authenticated role. Keep the private helper callable by
-- those two roles so the wrapper can evaluate active-release membership.

revoke all on function scryglass_private.is_active_scryglass_storage_object(text)
  from public, service_role;
grant execute on function scryglass_private.is_active_scryglass_storage_object(text)
  to anon, authenticated, supabase_storage_admin;
