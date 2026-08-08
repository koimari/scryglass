"""L7 terminal Draft Score mechanics."""

from .g1_roster import G1_SCHEMA_VERSION, G1RosterError, G1RosterEvidence
from .grid_promotion_gate import (
    SCHEMA_VERSION as GRID_PROMOTION_GATE_SCHEMA_VERSION,
    GridPromotionGateError,
    evaluate_grid_promotion_gate,
)
from .l2_authority import (
    L2_AUTHORITY_SCHEMA_VERSION,
    L2AuthorityRecordError,
    authority_record_payload_sha256,
    load_l2_authority_record,
    validate_l2_authority_record,
)
from .l2_readiness import inspect_l2_readiness
from .model import (
    TerminalDraft,
    TerminalDraftError,
    TerminalModel,
    TerminalScore,
    render_terminal_contract,
    score_terminal_draft,
    validate_terminal_actions,
)
from .promotion import (
    PROMOTION_SCHEMA_VERSION,
    PromotionReceiptError,
    TerminalPromotionBindings,
    load_promotion_receipt,
    promotion_receipt_authorizes,
    receipt_payload_sha256,
    validate_promotion_receipt,
)

__all__ = [
    "TerminalDraft",
    "TerminalDraftError",
    "TerminalModel",
    "TerminalScore",
    "render_terminal_contract",
    "score_terminal_draft",
    "validate_terminal_actions",
    "G1_SCHEMA_VERSION",
    "G1RosterError",
    "G1RosterEvidence",
    "GRID_PROMOTION_GATE_SCHEMA_VERSION",
    "GridPromotionGateError",
    "evaluate_grid_promotion_gate",
    "L2_AUTHORITY_SCHEMA_VERSION",
    "L2AuthorityRecordError",
    "authority_record_payload_sha256",
    "load_l2_authority_record",
    "validate_l2_authority_record",
    "inspect_l2_readiness",
    "PROMOTION_SCHEMA_VERSION",
    "PromotionReceiptError",
    "TerminalPromotionBindings",
    "load_promotion_receipt",
    "promotion_receipt_authorizes",
    "receipt_payload_sha256",
    "validate_promotion_receipt",
]
