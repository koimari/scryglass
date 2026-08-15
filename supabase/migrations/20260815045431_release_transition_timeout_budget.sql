-- Release verification scans all sealed query rows before the status change.
-- Keep the complete transition inside one bounded administrative budget.

alter function public.activate_scryglass_public_release(text)
  set statement_timeout to '120s';

alter function public.restore_scryglass_public_release(text)
  set statement_timeout to '120s';
