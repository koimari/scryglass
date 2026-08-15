from __future__ import annotations

import pytest

from lol_kills.v2.draft.interactions import representation_rank_private_result as result


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_optional_finite_rejects_nonfinite_values(value: float) -> None:
    assert result._optional_finite(value) is False
