"""Wave-2 L6 draft-interactions kernel modules."""

from .artifacts import (
    build_authority,
    build_development_report,
    build_fixture_payload,
    build_interactions_config,
)
from .fixtures import load_synthetic_rows, reveal_synthetic_seed
from .model import (
    DraftInteractionCandidate,
    DraftInteractionFit,
    DraftInteractionFamily,
    DraftInteractionModel,
    run_candidate_selection,
    score_row_pair_swap,
)
from .types import (
    CANONICAL_ROLES,
    DraftCompositionRow,
    DraftInteractionError,
    DraftInteractionFitDiagnostics,
    DraftInteractionPrediction,
    normalize_side,
    validate_side_roles,
)

__all__ = [
    "CANONICAL_ROLES",
    "DraftCompositionRow",
    "DraftInteractionError",
    "DraftInteractionFitDiagnostics",
    "DraftInteractionPrediction",
    "DraftInteractionCandidate",
    "DraftInteractionFit",
    "DraftInteractionFamily",
    "DraftInteractionModel",
    "run_candidate_selection",
    "score_row_pair_swap",
    "normalize_side",
    "validate_side_roles",
    "load_synthetic_rows",
    "reveal_synthetic_seed",
    "build_interactions_config",
    "build_fixture_payload",
    "build_development_report",
    "build_authority",
]
