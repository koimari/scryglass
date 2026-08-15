# Semgrep triage, 2026-08-15

## Scan receipt

- Semgrep OSS: `1.173.0`
- Configurations: `p/security-audit`, `p/python`, `p/typescript`
- Tracked targets: 1,921
- Rules run: 295
- Findings: 19
- Scan errors or timeouts: 0
- JSON receipt SHA-256: `fd55a60f19a007d1f5a43c8008d18fa9d60620cd915c3d83dfeb56e70aa8e3e9`

The scan used a 300-second per-file timeout and a 4,096 MB memory limit. It
completed without the 13 research-file timeouts from the earlier run.

Semgrep skipped 25 tracked files above its 1 MB default. Every skipped file is
JSON, JSONL, or PNG data. No executable source file was skipped.

## Finding triage

### Dynamic `urllib` calls

Eighteen findings report a dynamic URL passed to `urllib`. Seventeen call sites
pass the URL through `require_https_url` before the request. The guard requires
HTTPS, rejects credentials and non-standard ports, and accepts only the caller's
fixed host set. The covered hosts are Leaguepedia, GRID, Google Drive,
CommunityDragon, and the League Wiki.

The remaining finding is the internal `_http_bytes` transport in
`lol_kills/public_refresh.py`. It has no public request input. Production
preflight requires the Supabase backend and fixes the site origin to
`https://scryglass.xyz`. The legacy remote-manifest branch cannot run in the
production configuration. Alert webhook targets come only from the private
worker environment. Public requests cannot set that value.

These 18 findings are scanner pattern matches with bounded call paths. The
network guard tests reject unsafe schemes, credentials, ports, hosts, and
subdomain confusion. The public refresh tests cover the fixed production
origin and callback paths.

### Subprocess argument flow

One finding reports caller data in the GRID catalog helper subprocess. The
helper passes a Python argument list to `subprocess.run`. It uses
`sys.executable`, a resolved working directory, and fixed module and option
tokens. Python keeps `shell=False` by default. No shell string or shell
interpolation exists. A metacharacter in an argument stays one argument.

This finding is a scanner pattern match. It has no command-injection path.

## Result

All 19 findings have a recorded disposition. The complete scan found no new
security defect. Bandit, both CodeQL language scans, the secret scan,
dependency review, workflow and shell security, and the focused network tests
remain independent required gates.
