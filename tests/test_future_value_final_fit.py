from __future__ import annotations

import json

from benchmarks.build_future_value_final_fit import _evaluation_blockers
from tests.test_future_value_snapshots import _source_receipt


def test_final_fit_imports_source_bound_evaluation_blockers(tmp_path) -> None:
    source = _source_receipt(["g1", "g2"])
    evaluation = {
        "source": {
            "source_as_of": source["source_as_of"],
            "source_game_count": source["source_game_count"],
            "source_identity_sha256": source["source_identity_sha256"],
            "source_receipt_sha256": source["receipt_sha256"],
        },
        "variants": {
            "future_player_form": {
                "blockers": [
                    "nested_inner_feature_ledger_missing_fixed_c_used",
                    "authoritative_series_id_missing_proxy_cluster_used",
                ]
            }
        },
    }
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps(evaluation), encoding="utf-8")
    assert _evaluation_blockers(path, source) == (
        "authoritative_series_id_missing_proxy_cluster_used",
        "nested_inner_feature_ledger_missing_fixed_c_used",
    )
