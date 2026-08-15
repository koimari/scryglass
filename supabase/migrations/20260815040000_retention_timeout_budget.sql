-- The Data API applies an eight-second role timeout. One bounded release
-- cascade can contain about 161,000 indexed query rows and needs more time.
-- Keep the exception on this administrative RPC and within the API ceiling.

alter function public.prune_scryglass_public_releases_v2(integer)
  set statement_timeout to '60s';
