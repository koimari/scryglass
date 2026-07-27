from __future__ import annotations

import inspect
import unittest

from lol_kills.etl.join import build_map_warehouse


class JoinScopeContractTests(unittest.TestCase):
    def test_production_warehouse_defaults_to_all_canonical_leagues(self) -> None:
        signature = inspect.signature(build_map_warehouse)
        self.assertIs(signature.parameters["majors_only"].default, False)


if __name__ == "__main__":
    unittest.main()
