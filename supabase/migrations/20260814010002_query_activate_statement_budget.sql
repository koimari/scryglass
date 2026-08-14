-- Activation verifies the complete private asset and query receipt inventory.
-- Keep the release transition inside the same bounded publication budget.
alter function public.activate_scryglass_public_release(text)
  set statement_timeout = '120s';
