"""Independent root for enabling the reviewed private rank-assay runner.

This tiny module is deliberately outside the runner review subject.  A future
review may pin one permit's raw bytes here without changing the already
reviewed runner, tests, or contract-core subject that the permit approves.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PINNED_RUNNER_REVIEW_PERMIT_RAW_SHA256: str | None = (
    "c9ea237c63bdfff7f5c1cd4bade0a1e373d92fd5fe6057c206e8ac62acd56401"
)


class RunnerReviewAuthorityError(ValueError):
    """Raised when no independently pinned runner-review permit is valid."""


def require_independent_runner_review_permit(
    source: Mapping[str, Any],
    *,
    review_core_sha256: str,
    root: Path,
) -> None:
    pinned = PINNED_RUNNER_REVIEW_PERMIT_RAW_SHA256
    if pinned is None:
        raise RunnerReviewAuthorityError(
            "independent runner-review permit is not pinned"
        )
    if (
        set(source) != {"locator", "raw_sha256"}
        or source.get("raw_sha256") != pinned
    ):
        raise RunnerReviewAuthorityError("runner-review permit identity invalid")
    path = Path(str(source["locator"]))
    path = path if path.is_absolute() else root / path
    if not path.is_file() or path.is_symlink():
        raise RunnerReviewAuthorityError("runner-review permit is not a regular file")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != pinned:
        raise RunnerReviewAuthorityError("runner-review permit bytes changed")
    payload = json.loads(raw)
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    if raw != canonical:
        raise RunnerReviewAuthorityError("runner-review permit is not canonical JSON")
    if payload != {
        "approved_action": "private_target_m0_load_and_rank_assay",
        "decision": "PASS",
        "final_temporal_holdout_sealed": True,
        "independent_from_runner_and_generator": True,
        "review_core_sha256": review_core_sha256,
        "schema_id": "scryglass.representation-rank-runner-review-permit.v1",
    }:
        raise RunnerReviewAuthorityError("independent runner-review permit invalid")
