-- Close the mutable search_path warning on the public invoker wrappers.
alter function public.get_scryglass_active_asset(text, text)
  set search_path = '';
alter function public.get_scryglass_active_release(text)
  set search_path = '';
