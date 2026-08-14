# Public release evidence

Status: **HOLD**

This file records repeatable evidence for the public-release review. Secrets and private source rows stay outside this record.

## Candidate origin

| Field | Value |
| --- | --- |
| Review date | 2026-08-13 |
| Base branch | `origin/main` |
| Base commit | `8023372` |
| Candidate branch | `release/public-readiness-audit` |
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

The legacy `xyz.scryglass.public-refresh` launchd job is unloaded and persistently disabled. A submitted `scryglass.manual-cycle-7` refresh was also stopped during derive, before publication. No `lol_kills.public_refresh` process remained after the stop. The last production release stayed `v2026.08.13.183000`.

Run `refresh-20260813T223244Z-803f0d6b7322` is recorded as `error` with no release ID. Production health now reports `status: partial` and `refresh_status: failed`. The worker stays disabled until the ordered migration, application, publisher, and clean-refresh cutover reaches its health-alignment gate.

The Production and Development Vercel values for `SCRYGLASS_SUPABASE_URL` and `SCRYGLASS_SUPABASE_PUBLISHABLE_KEY` contained a literal `\\n` suffix. Both environments now contain canonical values. The Preview values were already canonical. The web project keeps the publication service key outside Vercel and holds only the public key plus the separate diagnostic credential.

## Regional refresh evidence

The regional refresh completed as a deferred production build in the worker
runtime. Receipt
`tierlist-live-refresh-20260814T032716887375Z-898104ca0bcf88e7.json` records
17,483 replayed maps, a 1,162-map live window, 195 cells, 7,544 rows, and 970
regional views across 39 scopes. Its production index digest is
`2a9ebe414cafa089fbd2572149b127276db5458e3422b9327b27c8be64d3a555`. The
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
3. Merge and apply private Storage migration `20260814020000`.
4. Install the tested publisher. Create and verify two Storage-only, query-complete releases.
5. Merge the strict web build. Merge and apply strict migration `20260814030000`, then repeat every route, asset, cache, and rollback probe.

Each migration phase uses a separate merge commit and an exact Supabase dry run.
The operator stops when a checkout contains a later phase. The temporary inline
RPC returns parsed server JSON only. It never serves bytes or reuses the source
digest as a response ETag. Phase 3 removes this RPC and every large-asset page
fallback.

## Evidence still required

- clean CI from a zero-state Supabase replay;
- two full Python-suite passes from the hashed Python 3.12 environment;
- complete app tests, type check, lint, build, browser, accessibility, and performance checks;
- preview security-header and abuse-budget probes;
- source-rights records and Riot product registration;
- an independent, hash-bound Draft Score promotion record;
- the final aligned production receipt.
