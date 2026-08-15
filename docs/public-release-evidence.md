# Public release evidence

Status: **HOLD**

This file records repeatable evidence for the public-release review. Secrets and private source rows stay outside this record.

## Candidate origin

| Field | Value |
| --- | --- |
| Review date | 2026-08-15 |
| Base branch | `origin/main` |
| Base commit | `098347f92261cc0c2d040a2fcf47a1587aa9e419` |
| Current production deployment | `dpl_H7TVPvGiakcn1pG75ZZnBASn7475` (`READY`, production) |
| Active release | `v2026.08.15.170332` |
| Ledger branch | `codex/release-ledger-refresh-20260815` |
| Preserved visual commit | `ced2f63` |
| Preserved source commit | `73b7dfa` on `preserve/public-readiness-ui` |
| Initial worktree state | This ledger refresh started from clean `origin/main` at `098347f92261cc0c2d040a2fcf47a1587aa9e419`. The visual and chat paths are already part of the main-line candidate history. |

## Toolchain captured on 2026-08-13

| Tool | Version |
| --- | --- |
| Node.js | `v26.6.0` |
| npm | `11.18.0` |
| uv | `0.11.7` |
| System Python | `3.9.6` |
| Release Python target | `3.12` |
| Supabase CLI in the current worktree | `2.114.0` |
| GitHub CLI | `2.89.0` |
| Vercel CLI | `50.28.0` |

CI installs the versions in the repository lockfiles. A local version is evidence about this review host only.

## Public route inventory

Pages:

- `/`, `/elo`, `/elo/player/[player]`, `/elo/team/[team]`;
- `/matches`, `/matches/[game]`, `/tiers`, `/methodology`, `/chat`;
- `/privacy`, `/sources`, `/legal`, `/security`;
- `/support` as a compatibility redirect.

Public data and service routes:

- `/api/health`;
- `/api/assets/[...path]`;
- `/api/public-data/tierlists` as a retired HTTP 410 compatibility route;
- `/api/chat/*`;
- `/packs/manifest.json`;
- `/robots.txt`, `/sitemap.xml`, `/.well-known/security.txt`.

The publication callback `/api/data-published` requires its bearer secret. Detailed diagnostics require separate authentication.

## Public asset inventory at review start

The repository contained six historical pack directories, one conflicting static manifest, and one 2.8 MB static tier-list file. Historical pack directories and the static manifest are removed in this candidate. Compatibility URLs resolve through the active release after the ordered Storage cutover.

Allowed publication paths come from one code-owned specification. Each active row must bind the release ID, path, byte count, SHA-256, content type, and immutable private Storage path. Raw source rows, training inputs, private receipts, service identifiers, and internal run details are excluded.

## Static scan record

Semgrep OSS 1.173.0 scanned 1,921 tracked targets with the `security-audit`,
`python`, and `typescript` configurations on 2026-08-15. It used a 300-second
per-file timeout and completed with zero errors or timeouts. The scan reported
19 pattern matches. The written triage is in
[`security/semgrep-triage-20260815.md`](security/semgrep-triage-20260815.md).
Its JSON receipt SHA-256 is
`fd55a60f19a007d1f5a43c8008d18fa9d60620cd915c3d83dfeb56e70aa8e3e9`.

Bandit reports no high-severity findings after the subprocess fix. The Python release lock reports no known vulnerabilities through `pip-audit`.

## Security and dependency closure

The current main release checks completed successfully in Validate run
[`31907363092`](https://github.com/koimari/scryglass/actions/runs/31907363092)
and Security run
[`31907363120`](https://github.com/koimari/scryglass/actions/runs/31907363120),
both at commit `098347f92261cc0c2d040a2fcf47a1587aa9e419`.
The PR checks also passed CodeQL for Python and JavaScript, Secret scan,
Dependency review, workflow and shell security, and the Elemental Drakes
dependency, type, lint, build, and SBOM job. The GitHub Dependabot alert API
returned zero open alerts for `koimari/scryglass` on 2026-08-14.

The Supabase security and performance advisors returned empty lint lists for
project `uytblwbtkwuukbbrugdi` when checked on 2026-08-15. The advisor result
is checked again after each schema or publication change.

The current CodeQL API snapshot for `refs/heads/main` at
`098347f92261cc0c2d040a2fcf47a1587aa9e419` has 341 open note-level alerts,
0 warnings, 0 errors, and 0 security-severity alerts. The note-level backlog
is recorded in [`security/codeql-notes-triage-20260815.md`](security/codeql-notes-triage-20260815.md).
The release security gate is clear. The note backlog remains a maintenance
follow-up.

## Retired public Blob cleanup

GitHub Actions run [`31749119639`](https://github.com/koimari/scryglass/actions/runs/31749119639) retired the approved `packs/`, `rankings/`, `state/`, and `tierlists/` prefixes on 2026-08-13.

The reviewed inventory digest was `132950895d1232dcb47b1bbe97f023980f3a554f208720d9625fad3207c47e53`. The workflow deleted 2,335 objects and 659,691,167 bytes. Its post-delete inventory matched the preserved-object digest `9d6a020aece53636b8df586e1c30ba86bead2235f44f524cfefffb5501eec036`, which covers 15 unrelated objects and 146,394,627 bytes. The receipt reports all four retired prefixes empty. Its GitHub artifact digest is `sha256:b22375e50d912b2b66b1023ed4b28606520ed9ea1358a018478f1a8a32085330`.

Independent requests after the workflow returned `404` for the known Draft record, warehouse snapshot, tier-list index, rankings file, and historical profile-record URLs.

## Production HOLD control

Production currently serves active release `v2026.08.15.170332` from Vercel
deployment `dpl_H7TVPvGiakcn1pG75ZZnBASn7475`, URL
`scryglass-f4f9xfbqa-koidevelopments.vercel.app`, with application commit
`098347f92261cc0c2d040a2fcf47a1587aa9e419`. The public health response checked
at `2026-08-15T20:58:17.438Z` reported `status: ok`,
`refresh_status: idle`, and `stale: false`. Authenticated diagnostics bind the
same release to run `refresh-20260815T170332Z-4792cd8e3df6`, source as-of
`2026-08-15T11:02:09Z`, and worker commit
`0b89411416491c0f7ebaccf4de699d4384303dde`.

The Vercel production deployment serving application commit
`098347f92261cc0c2d040a2fcf47a1587aa9e419` is READY. The active database
release and compatibility manifest both name `v2026.08.15.170332`.
Supabase reports 22 active Storage assets, 144396917 bytes, zero inline assets,
zero Draft assets, 161446 bounded query rows, and 12 sealed dataset receipts.
A byte-for-byte audit of all 22 deployed `/api/assets` responses passed. A
representative retired-release asset returned `404`.

The clean worker receipt records requirements lock SHA-256
`54c223c88fada349f883b2ed79064b96495a6de181132b723b8ba78cb4a5cc3d`,
input fingerprint
`55286da3f632527cbb7cb6de9182f2f85376545c5521cd598c9d4bffd718b0ea`,
and source SHA-256
`f00aa5aa4aa4fa595a81eec954cc4ac71e99e3582b2cc80826f20491b48b8e27`.
The current deployment has no error or fatal runtime logs in the deployment-scoped
production query.

PR #252 added exact release markers to the ratings, match, tier, player, team,
and match-detail pages. A production rollback drill moved all page and API
families to `v2026.08.15.025550`, verified the smallest manifest asset by byte
count and SHA-256, then returned every family to `v2026.08.15.033633`. Both
cache callbacks confirmed the requested and served release IDs. PR #253 fixed
the Data API timeout found during the first return attempt. Both activation and
restore now have a 120-second function budget. The repeated drill passed in
both directions.

The Production and Development Vercel values for `SCRYGLASS_SUPABASE_URL` and `SCRYGLASS_SUPABASE_PUBLISHABLE_KEY` contained a literal `\\n` suffix. Both environments now contain canonical values. The Preview values were already canonical. The web project keeps the publication service key outside Vercel and holds only the public key plus the separate diagnostic credential.

## Regional refresh evidence

The regional refresh is active in release `v2026.08.15.170332`. It records
17,544 replayed maps, a 1,223-map live window, 200 cells, 7,589 rows, and 1,105
regional views across 40 scopes. The views cover CBLOL, INTERNATIONAL, LCK,
LCP, LCS, LEC, LJL, LPL, PCS, TCL, and VCS. The candidate artifact SHA-256 is
`fc598f7c4aeba73b91cc69671f8563dd6b35e20b7963d06d1237cb1bd0e4970a`.
The production index raw SHA-256 is
`c921357a6a686cb3ad2f91df2eaf39a9588b45826ed4f97a8a01bdd31d2095b3`.

The latest public patch is `26.16`. The accepted OE source keeps its exact
`16.16` token. The public projection maps that token to season patch `26.16`.
Five accepted LEC maps from 2026-08-14 carry that patch. The production browser
shows `latest (26.16)`, preserves 26.15 as a separate option, and exposes the
LEC regional view. The default five-game evidence rule yields no ranked
champion because no champion has five appearances. The 1+ evidence view shows
41 champions across the five roles.

## Active Vercel firewall rules

The production WAF has two active path-scoped rules:

- `Enforce chat request budget`, rule `rule_enforce_chat_request_budget_vLN2R1`, limits `/api/chat/*` to 60 requests per minute per IP;
- `Enforce publication callback budget`, rule `rule_enforce_publication_callback_budget_KHvJXI`, limits POST `/api/data-published` to six requests per minute per IP.

A production probe sent 65 concurrent requests to `/api/chat/navigation`. It
returned 60 `200` responses and 5 edge denials. The edge action is `deny`, so
the WAF response is `403`; requests that reach the application limiter keep
the application `429` contract.

## Protected main-branch checks

Ruleset `20711858` is active on `refs/heads/main`. It requires these contexts:
`app`, `rankings-data`, `Vercel`, `browser`, `Supabase clean replay`,
`Workflow and shell security`, `elemental-drakes`, `CodeQL
(javascript-typescript)`, `CodeQL (python)`, `Secret scan`, and `Dependency
review`. The required checks for this evidence update passed.

Ruleset `20892261`, named `Scryglass CodeQL merge protection`, is also active
on `refs/heads/main`. It requires CodeQL `errors_and_warnings` and security
alerts `medium_or_higher`. Ruleset `20711858` remains unchanged. The current
CodeQL API snapshot meets both required thresholds. The 341 note-level alerts
remain documented follow-up work.

## Ordered query and Storage cutover

1. Merge and apply additive migrations `20260813010000` and `20260814010000`.
2. Merge the compatible web build. Verify the active legacy inline release and the bounded RPC path.
3. Merge and apply private Storage migration `20260814160000`.
4. Install the tested publisher. Create and verify two Storage-only, query-complete releases.
5. Merge the strict web build. Merge and apply strict migration `20260814170000`, then repeat every route, asset, cache, and rollback probe.

Each migration phase uses a separate merge commit and an exact Supabase dry run.
The operator stops when a checkout contains a later phase. The temporary inline
RPC returns parsed server JSON only. It never serves bytes or reuses the source
digest as a response ETag. Phase 3 removes this RPC and every large-asset page
fallback.

## Private research-suite receipt

The required two-pass private research suite completed successfully. The
receipt is `/private/tmp/scryglass-private-evidence/private-suite-final-497f0ec-two-pass.json`.

| Field | Value |
| --- | --- |
| Status | `passed` |
| Current code commit | `497f0ecc33a67e40df7de366ff470ced521fc4ab` |
| Passes | `2` |
| Current shards | `6` |
| Parallel workers | `8` |
| Test files | `240` |
| Requirements CI lock SHA-256 | `bc1af12da8b1adb9b2e456dad20a100619ae32d17be13e2ffeeb1360510412bf` |
| Current test inventory SHA-256 | `aecd8f4ab4ef57bb046c8869a5e65ba09bbb4903e62123f59a0378e1c0d19db0` |
| Receipt SHA-256 | `2d5b6af9d6abe21683166d267e8455f4bd0325a6d608946d6ad2eb61a463f29f` |
| Completed at (UTC) | `2026-08-15T21:22:47.085978+00:00` |

## Evidence still required

- an independent, hash-bound Draft Score promotion record;
- a separately evaluated and promoted phase-curve record;
- an independent momentum promotion record. Momentum stays research-only with
  the active default at zero;

The merged application, zero-state Supabase replay, browser, security, asset,
HTML budget, public-boundary, CodeQL severity, and Supabase advisor checks are
recorded in the protected checks.
PRs #249, #250, #252, and #253 closed resumable large-asset uploads, the
retired tier probe, release-bound page checks, and the complete rollback probe.
The current active release has passed the full 22-asset deployed hash audit.
Supabase security and performance advisors return zero findings. The release
remains on HOLD until the outstanding evidence above is complete.

## Optional follow-up notes

Riot product registration and formal owner review of public legal and policy
text are governance follow-ups. They are not practical release blockers for
the current noncommercial product. Public Draft probability, betting, odds, and
recommendation outputs remain unavailable until the required authority and
promotion records pass.
