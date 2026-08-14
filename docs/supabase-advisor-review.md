# Supabase advisor review

Review date: 2026-08-14

Project: `uytblwbtkwuukbbrugdi`

The live project is on the Pro plan. The database is PostgreSQL 17.6.1.155.
The public refresh is `ok`, `idle`, and `stale: false`.

## Advisor result

The security advisor returns zero lints after
`20260814135746_supabase_advisor_cleanup`.

The performance advisor returns zero lints after
`20260814135848_restore_fk_indexes` and the indexed foreign-key probes.

The cleanup migration keeps public invoker wrappers. Private implementation
functions live in `scryglass_private` with a fixed search path. Eight private
tables have deny policies for the public roles. Nine non-foreign-key indexes
were removed. Four foreign-key indexes remain because they protect referential
checks.

## Auth connection allocation

The Auth database pool is set to `10` with unit `percent`. The project has 60
database connections, so the current Auth allocation is proportional to the
database size. This setting remains suitable when the compute size changes.

## Verification

The live checks used Supabase MCP advisor calls and a Management API readback.
The next refresh must run the same advisor checks after migrations and data
publication. A non-empty advisor result blocks release review.

The pool setting is documented by Supabase in
[connection management](https://supabase.com/docs/guides/database/connection-management).
The setting is read and changed through the documented
[Auth configuration API](https://supabase.com/docs/reference/api/v1-update-a-projects-auth-settings).
