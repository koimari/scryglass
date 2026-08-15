# Private research CI boundary

The public repository does not contain the frozen warehouse, source snapshots,
mechanics packets, or model authority files used by private research tests.
Those files stay outside the public checkout because they contain source rows,
internal receipts, or restricted research material.

The protected `rankings-data` check runs the public release suite twice. The
suite covers patch identity, the 26.16 atom receipt, tier-list construction,
regional refresh, public query projection, Supabase publication, network
guards, and retired-asset controls. The target list is explicit in
`.github/workflows/validate.yml`.

The complete private research suite remains a separate gate. It must run twice
from the hashed Python environment after the approved artifact bundle is
mounted at the repository paths declared by each receipt. The pinned LCC
bridge rebuild skips with an explicit unavailable result when that bundle is
not mounted. The private job must run it after the bundle is mounted. A
missing bundle is not replaced with a synthetic file and it does not grant
authority to any public route.

Production remains on HOLD until the private suite has a recorded receipt and
the public release receipt has aligned source, worker, release, manifest, and
health values.

## 2026-08-15 diagnostic run

The hashed Python 3.12 environment completed one full test run on clean
Scryglass commit `d9584a2`. The exact 181-file LCC level-1 bundle was mounted.
Its rebuild test passed. The full result was 2,320 passed, 372 failed, 326
setup errors, and 14 skipped.

Most setup errors came from private warehouse and authority inputs that were
absent from the clean checkout. A copy-on-write diagnostic mount of the local
private store supplied the missing paths. It then found source-byte drift in
older receipts. For example, the champion crosswalk requires maps hash
`04c0cce1d86a4358d9eeb5937f61d5288358953e66c693a1ce88b0b650295d08`.
That map file exists in the frozen multileague snapshot. Its paired player
file hash `3d2a852daa43dfa402e1e48ef11d1a6858b73f2171f0c2febd82b941b19fceee`
is absent from the current private store.

This result is a failed gate. It grants no authority. The next private bundle
must list every receipt-bound source path and hash before the two required
runs start.
