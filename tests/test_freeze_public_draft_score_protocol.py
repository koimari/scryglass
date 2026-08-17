from __future__ import annotations

import json

import pytest

from lol_kills.research.atomized_rf_composite import (
    CATEGORICAL_CONTEXT_COLUMNS,
    FEATURE_SCHEMA_VERSION,
    MODEL_COLUMNS,
)
from lol_kills.research.freeze_public_draft_score_protocol import (
    ProtocolFreezeError,
    freeze_protocol,
)
from lol_kills.research.public_draft_score_promotion import sha256_path


def test_protocol_freeze_binds_matrix_and_refuses_overwrite(tmp_path: Path) -> None:
    base = tmp_path / "protocol-v25.json"
    matrix = tmp_path / "matrix.parquet"
    manifest = tmp_path / "matrix.manifest.json"
    output = tmp_path / "protocol-v29.json"
    base.write_text("{}\n", encoding="utf-8")
    matrix.write_bytes(b"matrix-bytes")
    manifest.write_text(
        json.dumps(
            {
                "schema_version": FEATURE_SCHEMA_VERSION,
                "matrix_sha256": sha256_path(matrix),
                "columns": [*MODEL_COLUMNS, *CATEGORICAL_CONTEXT_COLUMNS],
                "model_columns": list(MODEL_COLUMNS),
                "categorical_columns": list(CATEGORICAL_CONTEXT_COLUMNS),
            }
        ),
        encoding="utf-8",
    )

    document = freeze_protocol(
        base_protocol_path=base,
        matrix_path=matrix,
        manifest_path=manifest,
        output_path=output,
        iteration_id="v29",
        previous_receipt="a" * 64,
        frozen_utc="2026-08-16T22:00:00Z",
        single_change="Use the receipt-bound rating context and v10 matrix.",
    )

    assert document["inherits"] == "protocol-v25.json"
    assert document["iteration"]["matrix_sha256"] == sha256_path(matrix)
    assert document["iteration"]["matrix_manifest_sha256"] == sha256_path(
        manifest
    )
    assert document["feature_contract"]["schema_version"] == (
        FEATURE_SCHEMA_VERSION
    )
    with pytest.raises(ProtocolFreezeError, match="already exists"):
        freeze_protocol(
            base_protocol_path=base,
            matrix_path=matrix,
            manifest_path=manifest,
            output_path=output,
            iteration_id="v29",
            previous_receipt="a" * 64,
            frozen_utc="2026-08-16T22:00:00Z",
            single_change="Changed after the first freeze.",
        )
