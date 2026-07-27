from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.scientific_audit_inventory import inventory


class ScientificAuditInventoryTest(unittest.TestCase):
    def test_inventory_is_deterministic_and_hash_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "lol_kills" / "example.py"
            source.parent.mkdir(parents=True)
            source.write_text("import math\n\ndef score(x):\n    return math.exp(x)\n")

            first = inventory(root, include_tests=False)
            second = inventory(root, include_tests=False)
            self.assertEqual(first["inventory_sha256"], second["inventory_sha256"])
            self.assertEqual(first["files"], 1)
            self.assertEqual(first["symbols"], 1)
            self.assertEqual(first["parse_errors"], 0)

            source.write_text("def score(x):\n    return x\n")
            changed = inventory(root, include_tests=False)
            self.assertNotEqual(first["inventory_sha256"], changed["inventory_sha256"])

    def test_public_entry_is_classified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            route = root / "apps" / "lol-atlas" / "src" / "app" / "api" / "x" / "route.ts"
            route.parent.mkdir(parents=True)
            route.write_text('export function GET() { return Response.json({ ok: true }); }\n')

            payload = inventory(root, include_tests=False)
            self.assertEqual(payload["sources"][0]["reachability_hint"], "public_entry")
            self.assertIn("GET", payload["sources"][0]["symbols"])


if __name__ == "__main__":
    unittest.main()
