"""L7 terminal Draft Score mechanics.

The public names stay available from this package. Heavy audit modules load
only when a caller asks for them. The tier worker needs the neutral model and
must not import the private L2 readiness graph during cold start.
"""

from importlib import import_module


_EXPORTS = {
    "G1_SCHEMA_VERSION": (".g1_roster", "G1_SCHEMA_VERSION"),
    "G1RosterError": (".g1_roster", "G1RosterError"),
    "G1RosterEvidence": (".g1_roster", "G1RosterEvidence"),
    "GRID_PROMOTION_GATE_SCHEMA_VERSION": (".grid_promotion_gate", "SCHEMA_VERSION"),
    "GridPromotionGateError": (".grid_promotion_gate", "GridPromotionGateError"),
    "evaluate_grid_promotion_gate": (".grid_promotion_gate", "evaluate_grid_promotion_gate"),
    "L2_AUTHORITY_SCHEMA_VERSION": (".l2_authority", "L2_AUTHORITY_SCHEMA_VERSION"),
    "L2AuthorityRecordError": (".l2_authority", "L2AuthorityRecordError"),
    "authority_record_payload_sha256": (".l2_authority", "authority_record_payload_sha256"),
    "load_l2_authority_record": (".l2_authority", "load_l2_authority_record"),
    "validate_l2_authority_record": (".l2_authority", "validate_l2_authority_record"),
    "inspect_l2_readiness": (".l2_readiness", "inspect_l2_readiness"),
    "TerminalDraft": (".model", "TerminalDraft"),
    "TerminalDraftError": (".model", "TerminalDraftError"),
    "TerminalModel": (".model", "TerminalModel"),
    "TerminalScore": (".model", "TerminalScore"),
    "render_terminal_contract": (".model", "render_terminal_contract"),
    "score_terminal_draft": (".model", "score_terminal_draft"),
    "validate_terminal_actions": (".model", "validate_terminal_actions"),
    "PROMOTION_SCHEMA_VERSION": (".promotion", "PROMOTION_SCHEMA_VERSION"),
    "PromotionReceiptError": (".promotion", "PromotionReceiptError"),
    "TerminalPromotionBindings": (".promotion", "TerminalPromotionBindings"),
    "load_promotion_receipt": (".promotion", "load_promotion_receipt"),
    "promotion_receipt_authorizes": (".promotion", "promotion_receipt_authorizes"),
    "receipt_payload_sha256": (".promotion", "receipt_payload_sha256"),
    "validate_promotion_receipt": (".promotion", "validate_promotion_receipt"),
}


def __getattr__(name: str):
    spec = _EXPORTS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = spec
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value

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
