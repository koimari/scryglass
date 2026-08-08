"""Deterministic routing boundary for the resident League oracle.

The model should call one tool.  This router decides locally, in a stable
order, whether a request is an exact packet calculation or a semantic
slot/state request.  It adds route metadata without changing the underlying
answer contract, so a caller can audit which resident path produced a result.
"""

from __future__ import annotations

from typing import Any, Mapping

from .lol_oracle import LeagueOracleEngine
from .semantic_engine import SemanticOracleEngine


ROUTER_VERSION = "oracle-router-v1.1.0"


class LeagueOracleRouter:
    """Route exact questions first and semantic state questions second."""

    def __init__(
        self,
        exact_engine: LeagueOracleEngine,
        semantic_engine: SemanticOracleEngine | None = None,
    ) -> None:
        self.exact_engine = exact_engine
        self.semantic_engine = semantic_engine

    @staticmethod
    def _decorate(answer: Mapping[str, Any], *, route: str) -> dict[str, Any]:
        result = dict(answer)
        result["router"] = ROUTER_VERSION
        result["route"] = route
        provenance = dict(result.get("provenance") or {})
        provenance["router"] = ROUTER_VERSION
        provenance["route"] = route
        result["provenance"] = provenance
        return result

    def answer(
        self,
        question: str,
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return one answer using the fixed resident routing order.

        A supplied context is an explicit semantic request and is therefore
        evaluated by the semantic engine first.  Natural-language questions
        use the exact packet first; only an unavailable exact result is sent
        to semantic classification.  An unrecognized semantic request never
        replaces a more informative exact-engine result.
        """

        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        if context is not None and not isinstance(context, Mapping):
            raise ValueError("context must be an object")

        semantic = self.semantic_engine
        if context is not None and semantic is not None:
            semantic_answer = semantic.answer(question, context)
            if semantic_answer.get("intent") != "unknown":
                route = (
                    "semantic_execution"
                    if semantic_answer.get("status") == "available"
                    else "semantic_contract"
                    if semantic_answer.get("status") == "needs_input"
                    else "semantic_validation"
                    if semantic_answer.get("status") == "invalid_scenario"
                    else "semantic_unsupported"
                )
                return self._decorate(semantic_answer, route=route)

        exact_answer = self.exact_engine.answer(question)
        if exact_answer.get("status") == "available":
            return self._decorate(exact_answer, route="exact_packet")

        if semantic is not None:
            semantic_answer = semantic.answer(question, context)
            if semantic_answer.get("intent") != "unknown":
                route = (
                    "semantic_execution"
                    if semantic_answer.get("status") == "available"
                    else "semantic_contract"
                    if semantic_answer.get("status") == "needs_input"
                    else "semantic_validation"
                    if semantic_answer.get("status") == "invalid_scenario"
                    else "semantic_unsupported"
                )
                return self._decorate(semantic_answer, route=route)

        return self._decorate(exact_answer, route="exact_unsupported")


__all__ = ["LeagueOracleRouter", "ROUTER_VERSION"]
