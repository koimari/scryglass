-- The publication worker seals complete query datasets in one transaction.
-- Games and player-champion rows are larger than the web request budget, so
-- their canonical digest needs a publication-only database statement budget.
alter function public.seal_scryglass_query_dataset(text, text, integer, bigint, text, text)
  set statement_timeout = '120s';
