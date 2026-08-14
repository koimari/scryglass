# Public release evidence

Status: **HOLD**

This file records repeatable evidence for the public-release review. Secrets and private source rows stay outside this record.

## Candidate origin

| Field | Value |
| --- | --- |
| Review date | 2026-08-14 |
| Base branch | `origin/main` |
| Base commit | `6fa61f1d987a93adbe1e4ab6be75adea1895f211` |
| Candidate branch | `main` after PR #238 |
| Preserved visual commit | `ced2f63` |
| Preserved source commit | `73b7dfa` on `preserve/public-readiness-ui` |
| Initial worktree state | The 12 visual and chat paths were committed before the candidate branch was made. The candidate started from a clean `origin/main` and cherry-picked that commit. |

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
- `/api/public-data/tierlists`;
- `/api/chat/*`;
- `/packs/manifest.json`;
- `/robots.txt`, `/sitemap.xml`, `/.well-known/security.txt`.

The publication callback `/api/data-published` requires its bearer secret. Detailed diagnostics require separate authentication.

## Public asset inventory at review start

The repository contained six historical pack directories, one conflicting static manifest, and one 2.8 MB static tier-list file. Historical pack directories and the static manifest are removed in this candidate. Compatibility URLs resolve through the active release after the ordered Storage cutover.

Allowed publication paths come from one code-owned specification. Each active row must bind the release ID, path, byte count, SHA-256, content type, and immutable private Storage path. Raw source rows, training inputs, private receipts, service identifiers, and internal run details are excluded.

## Static scan record

Semgrep OSS scanned 978 tracked files with the `security-audit`, `python`, and `typescript` rules on 2026-08-13. It reported 22 pattern matches. Twenty-one matches are guarded HTTPS requests or an argument-list subprocess with `shell=False`; the exact guards have focused tests. One Oracle's Elixir metadata request lacked the shared host guard and was fixed. Semgrep timed out on selected rules in 13 large research files, so CodeQL and Bandit remain required independent gates.

Bandit reports no high-severity findings after the subprocess fix. The Python release lock reports no known vulnerabilities through `pip-audit`.

## Retired public Blob cleanup

GitHub Actions run [`31749119639`](https://github.com/koimari/scryglass/actions/runs/31749119639) retired the approved `packs/`, `rankings/`, `state/`, and `tierlists/` prefixes on 2026-08-13.

The reviewed inventory digest was `132950895d1232dcb47b1bbe97f023980f3a554f208720d9625fad3207c47e53`. The workflow deleted 2,335 objects and 659,691,167 bytes. Its post-delete inventory matched the preserved-object digest `9d6a020aece53636b8df586e1c30ba86bead2235f44f524cfefffb5501eec036`, which covers 15 unrelated objects and 146,394,627 bytes. The receipt reports all four retired prefixes empty. Its GitHub artifact digest is `sha256:b22375e50d912b2b66b1023ed4b28606520ed9ea1358a018478f1a8a32085330`.

Independent requests after the workflow returned `404` for the known Draft record, warehouse snapshot, tier-list index, rankings file, and historical profile-record URLs.

## Production HOLD control

Production currently serves active release `v2026.08.14.181106`. The public health response at
2026-08-14T20:43:37.530Z reported `status: ok`, `refresh_status: idle`, and
`stale: false`. Authenticated diagnostics bound the same release to run
`refresh-20260814T193826Z-1a385de262e2`, source observation
`2026-08-14T09:09:44+00:00`, and worker commit
`dace1bb60d5b9f49144d7602457a6e10c822b5ad`.

The active database release and manifest both name `v2026.08.14.181106`.
Supabase reports 22 active Storage assets, 145,019,319 bytes, zero inline
assets, zero Draft assets, and zero invalid metadata rows. A byte-for-byte
audit of all 22 deployed `/api/assets` responses passed.

The worker checkout currently has a later clean commit than the commit recorded
by the active receipt. The worker proof stays open until a clean, lock-bound
refresh records the same worker commit required by the final receipt.

The Production and Development Vercel values for `SCRYGLASS_SUPABASE_URL` and `SCRYGLASS_SUPABASE_PUBLISHABLE_KEY` contained a literal `\\n` suffix. Both environments now contain canonical values. The Preview values were already canonical. The web project keeps the publication service key outside Vercel and holds only the public key plus the separate diagnostic credential.

## Regional refresh evidence

The regional refresh completed as a deferred production build in the worker
runtime. The current receipt is
`tierlist-live-refresh-20260814T052013765685Z-7427f9957fe5fc9b.json`. It
records 17,503 replayed maps, a 1,182-map live window, 195 cells, 7,546 rows,
and 1,100 regional views across 39 scopes. The views cover CBLOL,
INTERNATIONAL, LCK, LCP, LCS, LEC, LJL, LPL, PCS, TCL, and VCS. The production
index raw digest is
`e8c5aab7bd365ec440c531b44b4a18c7b500218c3c69a132abc1f2ea2f7d8954`; its
embedded artifact digest is
`db4dc5b457d0f81a8e15ca003c105df69e1170c53e83ade8ca24fa3fba008592`. The
latest accepted source is public patch `26.15`; the output contains no
client-only `16.xx` labels. Publication stays deferred until the ordered
release checks pass.

The 26.16 atom bridge is a development input. The current accepted OE source
has no 16.16 game rows, so the regional model continues to use the audited
26.15 bridge. This keeps patch identity and atom provenance aligned.

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
review`. The candidate PR remains blocked from merge while the full
`rankings-data` run is pending.

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

## Evidence still required

- two public release Python-suite passes from the hashed Python 3.12 environment;
- two private research-suite passes from the hashed environment with the approved artifact bundle mounted;
- source-rights records and Riot product registration;
- an independent, hash-bound Draft Score promotion record;
- a clean worker checkout and lock digest bound to the active receipt;
- a production rollback drill with all route-family and release-ID probes;
- the final aligned production receipt.

The merged application, zero-state Supabase replay, browser, security, asset,
HTML budget, and public-boundary checks are recorded in PR #238. The current
active release has passed the full 22-asset deployed hash audit. The release
remains on HOLD until the outstanding evidence above is complete.
