revoke all on function public.rls_auto_enable()
  from public, anon, authenticated, service_role;

comment on function public.rls_auto_enable() is
  'Internal event-trigger function. Direct API execution is forbidden.';
