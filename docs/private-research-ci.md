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
