from __future__ import annotations

import json

import pandas as pd
import pytest

from lol_kills.export.public_pack import _strict_json, export_public_pack


@pytest.mark.parametrize("pack_id", ("../escape", "nested/pack", ".", "..", ""))
def test_public_pack_rejects_unsafe_pack_id_before_filesystem_use(pack_id: str) -> None:
    with pytest.raises(ValueError, match="Unsafe pack_id"):
        export_public_pack(pack_id=pack_id)


def test_public_snapshot_json_serializes_clocks_and_rejects_nan() -> None:
    payload = [
        {
            "last_series_at": pd.Timestamp(
                "2026-07-26T21:25:07Z"
            ),
            "rating": 1600.0,
        }
    ]
    encoded = _strict_json(payload)
    assert json.loads(encoded) == [
        {
            "last_series_at": "2026-07-26T21:25:07+00:00",
            "rating": 1600.0,
        }
    ]
    with pytest.raises(ValueError, match="Out of range float"):
        _strict_json({"rating": float("nan")})
