"""Provenance and publication primitives for model-v2 foundations."""

from .allowlist import ArtifactAllowlist, CandidateArtifact, enforce_candidate_publication, generate_artifact_allowlist
from .publication import (
    PublicationMatrix,
    PublicationMatrixDecision,
    PublicationMatrixError,
    PublicationMatrixRow,
    validate_publication_matrix,
    make_default_publication_matrix,
)
from .snapshots import (
    LineageReport,
    SourceSnapshot,
    SourceSnapshotManifest,
    SourceSnapshotRow,
    SourceSnapshotSnapshotError,
    SourceTreeMismatchError,
    SourceSnapshotRowSummary,
    TrainingSnapshot,
    TrainingSnapshotError,
    make_default_freshness_report,
)

__all__ = [
    "ArtifactAllowlist",
    "CandidateArtifact",
    "PublicationMatrix",
    "PublicationMatrixDecision",
    "PublicationMatrixError",
    "PublicationMatrixRow",
    "TrainingSnapshot",
    "TrainingSnapshotError",
    "LineageReport",
    "SourceSnapshot",
    "SourceSnapshotManifest",
    "SourceSnapshotRow",
    "SourceSnapshotRowSummary",
    "SourceSnapshotSnapshotError",
    "SourceTreeMismatchError",
    "enforce_candidate_publication",
    "generate_artifact_allowlist",
    "make_default_freshness_report",
    "make_default_publication_matrix",
    "validate_publication_matrix",
]
