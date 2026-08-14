# Public release readiness

Status: **HOLD**

Review date: 2026-08-13

Release branch: `release/public-readiness-audit`

Base commit: `8023372`

Current upstream integration target: `7396d219`

Preserved visual work: `ced2f63`

## Release rule

Public release requires every confirmed finding in this register to be closed and verified. A scanner result can close as a false positive only when the register contains the evidence. Draft probabilities also require a separate, hash-bound promotion record.

## Closure register

`Closed in branch` means that the candidate contains a fix and a regression test. The item stays outside the production release proof until the ordered rollout and production check pass.

| ID | Severity | Exploit or failure path | Component and evidence | Owner | Fix | Regression test | Verification status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| REL-001 | Critical | A predictable staged or retired release path can bypass active-release selection. | Supabase Storage bucket and `/api/assets/*`; the bucket was public. | Data platform | Make the bucket private. Authorize one active allowlisted asset before Storage access. | Anonymous staged, retired, unknown, Draft, and traversal reads return no row or `404`. | Closed in branch; clean Supabase replay and production probe pending. |
| REL-002 | Critical | Old public Vercel Blob URLs expose packs, rankings, a 50 MB warehouse archive, and Draft rows. | Retired Blob store `97gks2fobqkgppwx`; direct range requests returned data. | Operations | Disable all pack and warehouse Blob writes. Delete the four approved prefixes with a digest-bound workflow. | Inventory drift, pagination, ETag delete, preserved-object digest, and empty-prefix proof tests. | Closed externally. Run `31749119639` deleted 2,335 objects and 659,691,167 bytes. The redacted receipt proves all four prefixes empty and 15 unrelated objects unchanged. Five independent URL probes returned `404`. |
| REL-003 | High | Static and active manifests can name different releases and hashes. | Checked-in manifest, database release row, and compatibility URL. | Web and publisher | Serve a fixed projection of the active database release. Remove historical static pointers. | Pack ID, release ID, asset row, manifest file, and artifact digest parity. | Closed in branch; new active release pending. |
| REL-004 | High | Equal-size Storage corruption can pass a metadata-only readback. | Supabase publisher and asset proxy. | Data platform | Hash downloaded bytes before activation. Check MIME type, size, path, and stored metadata. | Equal-size changed bytes fail publication and serving. | Closed in branch; linked-project smoke pending. |
| REL-005 | High | A failed health check after activation can leave a failed release active. | `public_refresh.py` publication transaction. | Publisher | Mark success after strict health alignment. Restore the prior verified release on later failure. | Health-alignment failure restores the prior release and records failure. | Closed in branch; production rollback drill pending. |
| REL-006 | High | A supplied commit value or dirty checkout can stamp false provenance. | Worker launch paths and refresh ledger. | Operations | Resolve Git HEAD, compare the expected SHA, require a clean tree, and bind the hashed lock. | Dirty, mismatched HEAD, missing lock marker, and changed lock fail closed. | Closed in branch; provisioned worker proof pending. |
| REL-007 | High | Unsafe pack IDs can escape a target path or replace an existing release. | Manual pack and upload tools. | Publisher | Enforce the release ID grammar, resolved containment, immutable staging, and no-clobber writes. | Traversal, encoded traversal, invalid ID, and existing-target cases fail. | Closed in branch. |
| REL-008 | High | A superseded release can be restored after its objects changed or disappeared. | Supabase restore RPC and publisher restore path. | Data platform | Reuse the activation inventory checks and hash every object before restore. | Missing, changed, Draft-bearing, valid prior release, and post-transition mutation cases. | Closed in branch; restore now rechecks Storage bytes after activation and recovers the replaced release when that readback fails. Focused publication suite: 28 passed. Production rollback drill pending. |
| REL-009 | High | Active release rows and assets can be changed through service writes. | Supabase tables and repeated publisher calls. | Data platform | Reject manifest and asset mutation after staging. Verify an existing active release without restaging. | Active and superseded update, insert, and delete attempts fail. | Closed in branch; clean replay pending. |
| DEP-001 | High | Vulnerable transitive packages can process crafted build or image input. | Scryglass and Elemental Drakes npm graphs; GitHub alerts for `sharp`, `postcss`, and `nanoid`. | Web | Upgrade Next.js and refresh both lockfiles. | Full and production `npm audit` return zero findings in both apps. | Local audits clean; GitHub alerts must close after merge. |
| DEP-002 | High | Mutable Python dependency ranges can change worker code without review. | Worker and CI Python environments. | Operations | Use hashed release and CI locks. Install with `--require-hashes` and record the lock digest. | Install, marker, digest, and `pip check` gates. | Closed in branch. |
| CI-001 | High | Built output can escape a source-only public-boundary scan. | GitHub validation workflow. | CI | Run the boundary scan before and after the production build. | A forbidden built fixture makes the post-build job fail. | Closed in branch. |
| CI-002 | High | Main can merge without browser, Supabase, security, or second Python-suite gates. | GitHub Actions and ruleset `20711858`. | CI and repository owner | Add stable jobs, then require them on protected `main`. | Clean-checkout workflow run with every job green. | Ruleset closed. Active ruleset `20711858` now requires app, rankings-data, Vercel, browser, Supabase clean replay, workflow and shell security, Elemental Drakes, both CodeQL scans, secret scan, and dependency review. The full rankings-data run remains pending before merge. |
| API-001 | High | Serverless instances can each reset a process-local chat request budget. | `/api/chat/*` and `/api/data-published`. | Web operations | Keep local budgets and enforce the reviewed Vercel Firewall rules. | Boundary, burst, method, body, and multi-request probes. | Closed externally. The production WAF has active 60-per-minute chat and 6-per-minute publication rules. A 65-request production probe returned 60 responses with `200` and 5 edge denials. |
| API-002 | Medium | Oversize, malformed, or slow public questions can consume parser and function time. | Chat handlers. | Web | Enforce 500-character questions, 100-character names, 8 KB bodies, content types, bounded results, and five-second handlers. | Stable `413`, `415`, `422`, `429`, and timeout cases. | Closed in branch; preview fuzz pending. |
| API-003 | High | A buffered asset response exceeds the Vercel function response limit. | `/api/assets/*`; active files include multi-megabyte match and profile assets. | Web | Stream the verified Storage body and preserve exact length, MIME type, and ETag. | A lazy 5 MB response streams without calling `arrayBuffer()`. | Closed in branch; deployed large-object probe pending. |
| API-004 | Medium | HTML and error pages can miss browser controls or block their own scripts. | Next.js CSP proxy and response headers. | Web | Add request nonces, strict path matching, HSTS, frame, MIME, referrer, permissions, and opener policies. | Normal, error, prefetch, file-like 404, and production-header browser tests. | Closed in branch; preview and production checks pending. |
| API-005 | Medium | Old chat and profile data can remain cached after activation. | Next.js cache invalidation and module-derived caches. | Web and publisher | Key derived data by release and invalidate each route family with valid Next.js calls. | Two-release warm-process test plus release-header probes. | Closed in branch. Dynamic API and asset paths now use `route` revalidation targets. Deployment probes remain open. |
| DATA-001 | High | Draft fields can appear without an independent promotion authority. | Pack, publisher, Storage policy, query API, UI, and chat. | Research governance | Force authority to `unavailable` and publish zero Draft assets until receipt verification exists. | Promotion-shaped data, old promoted releases, nested Draft keys, restore, and direct Storage cases fail. | Fail-closed branch in progress; promotion remains open. |
| DATA-002 | Medium | Anonymous PostgREST calls can expose worker commits and run IDs. | Public health and release tables. | Data platform | Revoke base-table reads. Expose fixed public and token-bound diagnostic RPC projections. | Anonymous privilege, field projection, invalid token, and valid token tests. | Closed in branch; clean replay pending. |
| DATA-003 | High | Pages download and parse assets up to about 68 MB and keep them in an unbounded process cache. | Ratings, profiles, matches, tiers, and chat readers. | Data platform and web | Publish release-bound query tables and bounded RPCs. Cap every encoded response at 500 KB. | Pagination, filter, active-release, release-change, payload-size, and Draft-field tests. | In progress. |
| DATA-004 | Medium | Copied asset allowlists can drift across Python, TypeScript, audit code, and SQL. | Pack specification and publication consumers. | Data platform | Generate or compare every consumer with the canonical specification in CI. | Cross-language exact-set parity test. | In progress. |
| UI-001 | High | The default Tier 1 join can return an empty page when Tier 1 data exists. | `/elo` loader and ratings filters. | Web | Use the release-bound ratings query and surface required-data errors. | First load contains Tier 1 teams; required read failure reaches the error boundary. | Closed in branch; browser run pending. |
| UI-002 | High | The ratings page can serialize about 7.5 MB of HTML. | `/elo` server payload. | Web | Send one bounded page with facets and server pagination. | Raw initial HTML stays at or below 500 KB. | Closed in branch; production build measurement pending. |
| UI-003 | Medium | Important routes lack consistent loading, error, offline, empty, and missing states. | App Router pages and profiles. | Web | Add route-level boundaries and explicit states. | Desktop, tablet, 390 px, keyboard, reduced-motion, offline, empty, and error checks. | In progress. |
| OPS-001 | High | Production reports a failed refresh and inconsistent lineage. | `/api/health`, active release, and worker. | Operations | Complete the ordered cutover with a clean worker and one verified release. | Final receipt aligns app, worker, release, manifest, hashes, health, and freshness. | Open. |
| OPS-002 | High | Rollback can invalidate and probe the wrong release ID or omit route families. | Publisher rollback and smoke suite. | Operations | Use the restored database release ID and probe manifest, asset digest, profiles, matches, tiers, schedule, and chat. | Rollback fixture with changed lineage and all-family release headers. | Closed in branch; production drill pending. |
| SEC-001 | High | Shell construction can execute an external repository path. | Research helper subprocess. | Python | Pass an argument list with `shell=False`. | Metacharacter path test and high-severity Bandit scan. | Closed in branch. |
| SEC-002 | Medium | URL, archive, redirect, and dynamic SQL inputs can escape intended hosts or paths. | Python fetch, archive, schedule, and publication modules. | Python | Use shared HTTPS allowlists, containment checks, safe extraction, and fixed SQL identifiers. | Unsafe scheme, host, redirect, archive member, and identifier cases. | Closed in branch; full scanner triage pending. |
| SEC-003 | High | Live GRID state can be written below Next.js `public/`. | `live_snapshots.py` and build boundary. | Research worker | Require an explicit private runtime root and ban `public/live`. | Default path and boundary-scan fixtures fail closed. | Closed in branch. |
| LEGAL-001 | High | Oracle's Elixir reuse authority is absent. | Published aggregates derived from OE data. | Repository owner | Record a license or written permission. Keep raw source files private. | Rights register review. | Open external gate. |
| LEGAL-002 | High | Riot product registration evidence is absent. | Public Scryglass product. | Repository owner | Complete or record Riot Developer Portal registration and audit status. | Registration receipt review. | Open external gate. |
| LEGAL-003 | Medium | Public privacy, source, legal, and disclosure notices are incomplete. | Public information pages, `security.txt`, sitemap, and robots. | Web and repository owner | Publish fixed routes, attribution, contact, and policy text. | Route, link, metadata, and browser checks. | Closed in branch; owner review and production check pending. |

## Required production receipt

The final receipt must bind these values:

- application commit;
- worker commit and clean tree digest;
- active release ID;
- manifest release ID;
- every public asset SHA-256;
- source identity and observation time;
- post-activation cache probes;
- public health status `ok`, refresh status `idle`, and `stale: false`.

The release stays on HOLD when any value is missing or differs.
